#!/bin/bash
# Multi-GPU auto evaluation for SimplerEnv bridge tasks with visual prompting
# and VLM-based task decomposition.
#
# Usage:
#   EVAL_NAME=eval_vp_vlm_decompose CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/SimplerEnv/eval_files/auto_eval_scripts/run_eval.sh /path/to/checkpoint.pt
#
# All outputs (server logs, task logs, overlay videos, eval results) are saved under:
#   <ckpt_dir>/<EVAL_NAME>/

echo "$(which python)"

###########################################################################################
# === Please modify the following paths according to your environment ===
# cd /path/to/VP-VLA
export star_vla_python=/path/to/conda_envs/starVLA/bin/python
export sim_python=/path/to/conda_envs/simpler_env/bin/python
export sam3_python=/path/to/conda_envs/sam3/bin/python
export SimplerEnv_PATH=/path/to/SimplerEnv
export PYTHONPATH=$(pwd):${PYTHONPATH}
# === End of environment variable configuration ===
###########################################################################################

# Port bases (each GPU gets policy_base+N, sam3_base+N, vlm_base+N)
policy_base_port=4350
sam3_base_port=4450
vlm_base_port=4550

MODEL_PATH=$1
EVAL_NAME=${2:-${EVAL_NAME:-"eval_vp_vla_simpler_env"}}
TSET_NUM=1
run_count=0

if [ -z "$MODEL_PATH" ]; then
  echo "Error: MODEL_PATH not provided as the first argument"
  echo "Usage: bash run_eval.sh /path/to/checkpoint.pt [eval_name]"
  exit 1
fi

ckpt_path=${MODEL_PATH}
ckpt_dir_base=$(dirname "${ckpt_path}")
EVAL_DIR="${ckpt_dir_base}/${EVAL_NAME}"
mkdir -p "${EVAL_DIR}"

# Visual prompting config (must match training)
TARGET_OBJECT_PROMPT_MODE="crosshair"
TARGET_LOCATION_PROMPT_MODE="box"
VLM_QUERY_ON_GRIPPER_CHANGE=true
SAVE_OVERLAY_VIDEO=true
VLM_DECOMPOSE_SUBTASKS=true

# PID tracking
policyserver_pids=()
sam3server_pids=()
vlmserver_pids=()
eval_pids=()

# ============================================================
# Helper Functions
# ============================================================

start_policy_service() {
  local gpu_id=$1
  local ckpt_path=$2
  local port=$3
  local server_log_dir="${EVAL_DIR}/server_logs"
  local svc_log="${server_log_dir}/$(basename "${ckpt_path%.*}")_policy_server_${port}.log"
  mkdir -p "${server_log_dir}"

  if lsof -iTCP:"${port}" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Port ${port} is occupied; attempting to free it..."
    lsof -iTCP:"${port}" -sTCP:LISTEN -t | xargs kill -9 2>/dev/null
    sleep 2
  fi
  echo "Starting policy server on GPU ${gpu_id}, port ${port}"
  CUDA_VISIBLE_DEVICES=${gpu_id} ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${ckpt_path} \
    --port ${port} \
    --use_bf16 \
    > "${svc_log}" 2>&1 &

  local pid=$!
  policyserver_pids+=($pid)
  sleep 30
}

start_sam3_service() {
  local gpu_id=$1
  local port=$2
  local server_log_dir="${EVAL_DIR}/server_logs"
  local svc_log="${server_log_dir}/sam3_server_gpu${gpu_id}_${port}.log"
  mkdir -p "${server_log_dir}"

  if lsof -iTCP:"${port}" -sTCP:LISTEN -t >/dev/null 2>&1; then
    lsof -iTCP:"${port}" -sTCP:LISTEN -t | xargs kill -9 2>/dev/null
    sleep 2
  fi
  echo "Starting SAM3 server on GPU ${gpu_id}, port ${port}"
  CUDA_VISIBLE_DEVICES=${gpu_id} ${sam3_python} \
    examples/Robocasa_tabletop/visual_prompt_utility/sam3_server.py \
    --port ${port} \
    > "${svc_log}" 2>&1 &

  local pid=$!
  sam3server_pids+=($pid)
}

start_vlm_service() {
  local gpu_id=$1
  local port=$2
  local server_log_dir="${EVAL_DIR}/server_logs"
  local svc_log="${server_log_dir}/vlm_server_gpu${gpu_id}_${port}.log"
  mkdir -p "${server_log_dir}"

  if lsof -iTCP:"${port}" -sTCP:LISTEN -t >/dev/null 2>&1; then
    lsof -iTCP:"${port}" -sTCP:LISTEN -t | xargs kill -9 2>/dev/null
    sleep 2
  fi
  echo "Starting VLM server (SimplerEnv single-gripper) on GPU ${gpu_id}, port ${port}"
  CUDA_VISIBLE_DEVICES=${gpu_id} ${sam3_python} \
    examples/SimplerEnv/eval_files/vlm_server_subtask_simpler.py \
    --port ${port} \
    > "${svc_log}" 2>&1 &

  local pid=$!
  vlmserver_pids+=($pid)
}

stop_all_services() {
  if [ "${#eval_pids[@]}" -gt 0 ]; then
    echo "Waiting for evaluation jobs to finish..."
    for pid in "${eval_pids[@]}"; do
      if ps -p "$pid" > /dev/null 2>&1; then
        wait "$pid"
        status=$?
        if [ $status -ne 0 ]; then
          echo "Warning: evaluation job $pid exited abnormally (status: $status)"
        fi
      fi
    done
  fi

  echo "Stopping all servers..."
  for pid in "${policyserver_pids[@]}"; do
    if ps -p "$pid" > /dev/null 2>&1; then
      kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    fi
  done
  for pid in "${sam3server_pids[@]}"; do
    if ps -p "$pid" > /dev/null 2>&1; then
      kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    fi
  done
  for pid in "${vlmserver_pids[@]}"; do
    if ps -p "$pid" > /dev/null 2>&1; then
      kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    fi
  done

  eval_pids=()
  policyserver_pids=()
  sam3server_pids=()
  vlmserver_pids=()
  echo "All services stopped"
}

# ============================================================
# GPU Setup
# ============================================================
IFS=',' read -r -a CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#CUDA_DEVICES[@]}

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "NUM_GPUS: $NUM_GPUS"
echo "Visual Prompting: SAM3=${TARGET_OBJECT_PROMPT_MODE}/${TARGET_LOCATION_PROMPT_MODE}"
echo "VLM Decompose Tasks: ${VLM_DECOMPOSE_SUBTASKS}"
echo "Eval Name: ${EVAL_NAME}"
echo "Eval Dir: ${EVAL_DIR}"

# Build VP args string
VP_ARGS="--use-sam3 --target-object-prompt-mode ${TARGET_OBJECT_PROMPT_MODE} --target-location-prompt-mode ${TARGET_LOCATION_PROMPT_MODE}"
VP_ARGS="${VP_ARGS} --use-vlm"
if [ "${VLM_QUERY_ON_GRIPPER_CHANGE}" = true ]; then
    VP_ARGS="${VP_ARGS} --vlm-query-on-gripper-change"
else
    VP_ARGS="${VP_ARGS} --no-vlm-query-on-gripper-change"
fi
if [ "${VLM_DECOMPOSE_SUBTASKS}" = true ]; then
    VP_ARGS="${VP_ARGS} --vlm-decompose-subtasks"
fi

# ============================================================
# Bridge V1 Tasks
# ============================================================
scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
robot_init_x=0.147
robot_init_y=0.028

declare -a ENV_NAMES=(
  StackGreenCubeOnYellowCubeBakedTexInScene-v0
  PutCarrotOnPlateInScene-v0
  PutSpoonOnTableClothInScene-v0
)

for i in "${!ENV_NAMES[@]}"; do
  env="${ENV_NAMES[i]}"
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}

    ckpt_base=$(basename "${ckpt_path}")
    ckpt_name="${ckpt_base%.*}"

    tag="run${run_idx}"
    task_log="${EVAL_DIR}/logs/${ckpt_name}_${env}.log.${tag}"
    mkdir -p "${EVAL_DIR}/logs"

    policy_port=$((policy_base_port + run_count))
    sam3_port=$((sam3_base_port + run_count))
    vlm_port=$((vlm_base_port + run_count))

    overlay_dir="${EVAL_DIR}/overlay_videos/${env}/${tag}"
    mkdir -p "${overlay_dir}"

    OVERLAY_ARGS=""
    if [ "${SAVE_OVERLAY_VIDEO}" = true ]; then
        OVERLAY_ARGS="--save-overlay-video --overlay-video-dir ${overlay_dir}"
    fi

    echo "Launching VP+VLM-decompose task [${env}] run#${run_idx} on GPU ${gpu_id}"

    start_policy_service ${gpu_id} ${ckpt_path} ${policy_port}
    start_sam3_service ${gpu_id} ${sam3_port}
    start_vlm_service ${gpu_id} ${vlm_port}
    sleep 60

    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} \
      examples/SimplerEnv/eval_files/start_simpler_env_visual_prompt.py \
      --port ${policy_port} \
      --ckpt-path ${ckpt_path} \
      --robot ${robot} \
      --policy-setup widowx_bridge \
      --control-freq 5 \
      --sim-freq 500 \
      --max-episode-steps 120 \
      --env-name "${env}" \
      --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
      --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 24 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --logging-dir ${EVAL_DIR}/results/${ckpt_name}_${env}_run${run_idx} \
      --sam3-port ${sam3_port} \
      --vlm-port ${vlm_port} \
      ${VP_ARGS} \
      ${OVERLAY_ARGS} \
      > "${task_log}" 2>&1 &

    eval_pids+=($!)
    run_count=$((run_count + 1))
  done
done

# ============================================================
# Bridge V2 Tasks
# ============================================================
declare -a ENV_NAMES_V2=(
  PutEggplantInBasketScene-v0
)

scene_name=bridge_table_1_v2
robot=widowx_sink_camera_setup
rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
robot_init_x=0.127
robot_init_y=0.06

for i in "${!ENV_NAMES_V2[@]}"; do
  env="${ENV_NAMES_V2[i]}"
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}

    ckpt_base=$(basename "${ckpt_path}")
    ckpt_name="${ckpt_base%.*}"

    tag="run${run_idx}"
    task_log="${EVAL_DIR}/logs/${ckpt_name}_${env}.log.${tag}"
    mkdir -p "${EVAL_DIR}/logs"

    policy_port=$((policy_base_port + run_count))
    sam3_port=$((sam3_base_port + run_count))
    vlm_port=$((vlm_base_port + run_count))

    overlay_dir="${EVAL_DIR}/overlay_videos/${env}/${tag}"
    mkdir -p "${overlay_dir}"

    OVERLAY_ARGS=""
    if [ "${SAVE_OVERLAY_VIDEO}" = true ]; then
        OVERLAY_ARGS="--save-overlay-video --overlay-video-dir ${overlay_dir}"
    fi

    echo "Launching VP+VLM-decompose V2 task [${env}] run#${run_idx} on GPU ${gpu_id}"

    start_policy_service ${gpu_id} ${ckpt_path} ${policy_port}
    start_sam3_service ${gpu_id} ${sam3_port}
    start_vlm_service ${gpu_id} ${vlm_port}
    sleep 60

    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env_visual_prompt.py \
      --ckpt-path ${ckpt_path} \
      --port ${policy_port} \
      --robot ${robot} \
      --policy-setup widowx_bridge \
      --control-freq 5 \
      --sim-freq 500 \
      --max-episode-steps 120 \
      --env-name "${env}" \
      --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
      --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 24 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --logging-dir ${EVAL_DIR}/results/${ckpt_name}_${env}_run${run_idx} \
      --sam3-port ${sam3_port} \
      --vlm-port ${vlm_port} \
      ${VP_ARGS} \
      ${OVERLAY_ARGS} \
      2>&1 | tee "${task_log}" &

    eval_pids+=($!)
    run_count=$((run_count + 1))
  done
done

stop_all_services
wait
echo "All VP+VLM-decompose evaluations finished"

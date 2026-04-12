#!/bin/bash

# ============================================================
# Multi-GPU evaluation for Robocasa tabletop tasks with visual prompting
# and VLM-based task decomposition.
#
# Usage:
#   bash examples/Robocasa_tabletop/eval_files/run_eval.sh /path/to/checkpoint.pt [n_envs] [max_episode_steps] [n_action_steps]
# ============================================================

###########################################################################################
# === Please modify the paths to Python executables in conda environments ===
starVLA_PYTHON=/gpfs/wangzixuan/conda_envs/starVLA/bin/python
ROBOCASA_PYTHON=/gpfs/wangzixuan/conda_envs/robocasa/bin/python
SAM3_PYTHON=/gpfs/wangzixuan/conda_envs/sam3/bin/python

export PYTHONPATH=$(pwd):${PYTHONPATH}
CKPT_DEFAULT="/path/to/your/checkpoint.pt"
id_name=${id_name:-"eval_vp_vla_robocasa_tabletop"}
# Port bases: can be overridden via env for parallel runs (e.g. PORT_OFFSET=8)
PORT_OFFSET=${PORT_OFFSET:-0}
BASE_PORT=$((52398 + PORT_OFFSET))
SAM3_PORT_BASE=$((52194 + PORT_OFFSET))
VLM_PORT_BASE=$((52295 + PORT_OFFSET))
# === End of environment variable configuration ===
###########################################################################################

N_ENVS_DEFAULT=1
MAX_EPISODE_STEPS_DEFAULT=720
N_ACTION_STEPS_DEFAULT=12

NUM_GPUS=8

# Visual prompting configuration
USE_SAM3=${USE_SAM3:-true}
USE_VLM=${USE_VLM:-true}
VLM_QUERY_ON_GRIPPER_CHANGE=${VLM_QUERY_ON_GRIPPER_CHANGE:-true}

TARGET_OBJECT_PROMPT_MODE=${TARGET_OBJECT_PROMPT_MODE:-"crosshair"}
TARGET_LOCATION_PROMPT_MODE=${TARGET_LOCATION_PROMPT_MODE:-"box"}
SAVE_OVERLAY_VIDEO=${SAVE_OVERLAY_VIDEO:-true}
OVERLAY_VIDEO_DIR=${OVERLAY_VIDEO_DIR:-"./results/overlay_videos_${id_name}"}

VLM_DECOMPOSE_SUBTASKS=${VLM_DECOMPOSE_SUBTASKS:-true}

# Parse command-line arguments
CKPT_PATH=${1:-$CKPT_DEFAULT}
N_ENVS=${2:-$N_ENVS_DEFAULT}
MAX_EPISODE_STEPS=${3:-$MAX_EPISODE_STEPS_DEFAULT}
N_ACTION_STEPS=${4:-$N_ACTION_STEPS_DEFAULT}


echo "=== Evaluation Configuration ==="
echo "Checkpoint Path      : ${CKPT_PATH}"
echo "Number of Envs       : ${N_ENVS}"
echo "Max Episode Steps    : ${MAX_EPISODE_STEPS}"
echo "Action Chunk Length  : ${N_ACTION_STEPS}"
echo "Use SAM3             : ${USE_SAM3}"
echo "Use VLM              : ${USE_VLM}"
if [ "$USE_SAM3" = true ]; then
    echo "Target Object Mode   : ${TARGET_OBJECT_PROMPT_MODE}"
    echo "Target Location Mode : ${TARGET_LOCATION_PROMPT_MODE}"
fi
if [ "$USE_VLM" = true ]; then
    echo "VLM Gripper Change   : ${VLM_QUERY_ON_GRIPPER_CHANGE}"
fi
echo "Save Overlay Video   : ${SAVE_OVERLAY_VIDEO}"
if [ "$SAVE_OVERLAY_VIDEO" = true ]; then
    echo "Overlay Video Dir    : ${OVERLAY_VIDEO_DIR}"
fi
echo "VLM Decompose Tasks  : ${VLM_DECOMPOSE_SUBTASKS}"
echo "================================"

# ============================================================
# Evaluation Function
# ============================================================

GetVideoOutPath() {
    local CKPT_PATH=$1
    local N_ACTION_STEPS=$2
    local MAX_EPISODE_STEPS=$3
    local N_ENVS=$4
    local ENV_NAME=$5
    local SAVE_ROOT=$(dirname "$(dirname "$CKPT_PATH")")
    local ckpt_name=$(basename "$CKPT_PATH" .pt)
    echo "${SAVE_ROOT}/videos_${id_name}/${ckpt_name}/n_action_steps_${N_ACTION_STEPS}_max_episode_steps_${MAX_EPISODE_STEPS}_n_envs_${N_ENVS}_${ENV_NAME}"
}

CountVideos() {
    local VIDEO_PATH=$1
    if [[ -d "$VIDEO_PATH" ]]; then
        find "$VIDEO_PATH" -maxdepth 1 -name "*.mp4" 2>/dev/null | wc -l
    else
        echo 0
    fi
}

CheckEvalSuccess() {
    local VIDEO_PATH=$1
    local EXPECTED_COUNT=${2:-50}
    local video_count=$(CountVideos "$VIDEO_PATH")
    if [[ "$video_count" -eq "$EXPECTED_COUNT" ]]; then
        return 0
    else
        return 1
    fi
}

EvalEnv() {
    local GPU_ID=$1
    local PORT=$2
    local ENV_NAME=$3
    local CKPT_PATH=$4
    local LOG_DIR=$5
    local ROBOCASA_PYTHON=$6
    local N_ENVS=$7
    local MAX_EPISODE_STEPS=$8
    local N_ACTION_STEPS=$9
    local USE_SAM3=${10}
    local SAM3_PORT=${11}
    local USE_VLM=${12}
    local VLM_PORT=${13}
    local SAVE_ROOT=$(dirname "$(dirname "$CKPT_PATH")")
    local ckpt_name=$(basename "$CKPT_PATH" .pt)
    local VIDEO_OUT_PATH="${SAVE_ROOT}/videos_${id_name}/${ckpt_name}/n_action_steps_${N_ACTION_STEPS}_max_episode_steps_${MAX_EPISODE_STEPS}_n_envs_${N_ENVS}_${ENV_NAME}"
    mkdir -p "${VIDEO_OUT_PATH}"

    echo "Launching evaluation | GPU ${GPU_ID} | Port ${PORT} | Env ${ENV_NAME}"

    SAM3_ARGS=""
    if [ "$USE_SAM3" = true ]; then
        SAM3_ARGS="--args.use_sam3 --args.sam3_port ${SAM3_PORT} --args.target_object_prompt_mode ${TARGET_OBJECT_PROMPT_MODE} --args.target_location_prompt_mode ${TARGET_LOCATION_PROMPT_MODE}"
    fi
    
    VLM_ARGS=""
    if [ "$USE_VLM" = true ]; then
        if [ "$VLM_QUERY_ON_GRIPPER_CHANGE" = true ]; then
            VLM_ARGS="--args.use_vlm --args.vlm_port ${VLM_PORT} --args.vlm_query_on_gripper_change"
        else
            VLM_ARGS="--args.use_vlm --args.vlm_port ${VLM_PORT} --args.no-vlm_query_on_gripper_change"
        fi
    fi
    
    OVERLAY_ARGS=""
    if [ "$SAVE_OVERLAY_VIDEO" = true ]; then
        local OVERLAY_DIR="${OVERLAY_VIDEO_DIR:-${VIDEO_OUT_PATH}/overlay_videos}"
        mkdir -p "${OVERLAY_DIR}"
        mkdir -p "${OVERLAY_DIR}/${ENV_NAME}"
        OVERLAY_ARGS="--args.save_overlay_video --args.overlay_video_dir ${OVERLAY_DIR}/${ENV_NAME}"
    fi

    EXTRA_ARGS=""
    if [ "$VLM_DECOMPOSE_SUBTASKS" = true ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --args.vlm_decompose_subtasks"
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    ${ROBOCASA_PYTHON} examples/Robocasa_tabletop/eval_files/simulation_env.py \
        --args.env_name "${ENV_NAME}" \
        --args.port "${PORT}" \
        --args.n_episodes 50 \
        --args.n_envs "${N_ENVS}" \
        --args.max_episode_steps "${MAX_EPISODE_STEPS}" \
        --args.n_action_steps "${N_ACTION_STEPS}" \
        --args.video_out_path "${VIDEO_OUT_PATH}" \
        --args.pretrained_path "${CKPT_PATH}" \
        ${SAM3_ARGS} \
        ${VLM_ARGS} \
        ${OVERLAY_ARGS} \
        ${EXTRA_ARGS} \
        > "${LOG_DIR}/eval_env_${ENV_NAME//\//_}_gpu${GPU_ID}.log" 2>&1
}

# ============================================================
# Environment List
# ============================================================

ENV_NAMES=(
  gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
)

# ============================================================
# Runtime Configuration
# ============================================================

LOG_DIR="${CKPT_PATH}.log/eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "=== Launching Multi-GPU Evaluation ==="
echo "GPUs            : ${NUM_GPUS}"
echo "Num Environments: ${#ENV_NAMES[@]}"
echo "Log Directory   : ${LOG_DIR}"

# ============================================================
# Step 1: Launch Policy Servers
# ============================================================

SERVER_PIDS=()

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + GPU_ID))
    echo "Starting policy server | GPU ${GPU_ID} | Port ${PORT}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    ${starVLA_PYTHON} deployment/model_server/server_policy.py \
        --ckpt_path "${CKPT_PATH}" \
        --port "${PORT}" \
        --use_bf16 \
        > "${LOG_DIR}/server_gpu${GPU_ID}_port${PORT}.log" 2>&1 &

    SERVER_PIDS[$GPU_ID]=$!

    sleep 10
done

sleep 30

# ============================================================
# Step 1.5: Launch SAM3/VLM Servers (if enabled)
# ============================================================

SAM3_PIDS=()
if [ "$USE_SAM3" = true ]; then
    for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
        SAM3_PORT=$((SAM3_PORT_BASE + GPU_ID))
        echo "Starting SAM3 server | GPU ${GPU_ID} | Port ${SAM3_PORT}"
        CUDA_VISIBLE_DEVICES=${GPU_ID} \
        ${SAM3_PYTHON} examples/Robocasa_tabletop/visual_prompt_utility/sam3_server.py \
            --port "${SAM3_PORT}" \
            > "${LOG_DIR}/sam3_gpu${GPU_ID}_port${SAM3_PORT}.log" 2>&1 &
        SAM3_PIDS[$GPU_ID]=$!
    done
    sleep 30
fi

VLM_PIDS=()
if [ "$USE_VLM" = true ]; then
    for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
        VLM_PORT=$((VLM_PORT_BASE + GPU_ID))
        echo "Starting VLM server | GPU ${GPU_ID} | Port ${VLM_PORT}"
        CUDA_VISIBLE_DEVICES=${GPU_ID} \
        ${SAM3_PYTHON} examples/Robocasa_tabletop/visual_prompt_utility/vlm_server_subtask.py \
            --port "${VLM_PORT}" \
            > "${LOG_DIR}/vlm_gpu${GPU_ID}_port${VLM_PORT}.log" 2>&1 &
        VLM_PIDS[$GPU_ID]=$!
    done
    sleep 30
fi

# ============================================================
# Step 2: Dispatch Environments to GPUs (with retry logic)
# ============================================================

MAX_RETRIES=5
EXPECTED_VIDEO_COUNT=50
MONITOR_INTERVAL=10

declare -A PID_TO_ENV
declare -A PID_TO_GPU
declare -A ENV_RETRY_COUNT

LaunchEvaluation() {
    local ENV_NAME=$1
    local GPU_ID=$2
    local PORT=$3
    
    local SAVE_ROOT=$(dirname "$(dirname "$CKPT_PATH")")
    local ckpt_name=$(basename "$CKPT_PATH" .pt)
    local VIDEO_OUT_PATH="${SAVE_ROOT}/videos_${id_name}/${ckpt_name}/n_action_steps_${N_ACTION_STEPS}_max_episode_steps_${MAX_EPISODE_STEPS}_n_envs_${N_ENVS}_${ENV_NAME}"
    mkdir -p "${VIDEO_OUT_PATH}"

    echo "Launching evaluation | GPU ${GPU_ID} | Port ${PORT} | Env ${ENV_NAME}"

    local SAM3_PORT=$((SAM3_PORT_BASE + GPU_ID))
    local VLM_PORT=$((VLM_PORT_BASE + GPU_ID))

    local SAM3_ARGS=""
    if [ "$USE_SAM3" = true ]; then
        SAM3_ARGS="--args.use_sam3 --args.sam3_port ${SAM3_PORT} --args.target_object_prompt_mode ${TARGET_OBJECT_PROMPT_MODE} --args.target_location_prompt_mode ${TARGET_LOCATION_PROMPT_MODE}"
    fi
    
    local VLM_ARGS=""
    if [ "$USE_VLM" = true ]; then
        if [ "$VLM_QUERY_ON_GRIPPER_CHANGE" = true ]; then
            VLM_ARGS="--args.use_vlm --args.vlm_port ${VLM_PORT} --args.vlm_query_on_gripper_change"
        else
            VLM_ARGS="--args.use_vlm --args.vlm_port ${VLM_PORT} --args.no-vlm_query_on_gripper_change"
        fi
    fi
    
    local OVERLAY_ARGS=""
    if [ "$SAVE_OVERLAY_VIDEO" = true ]; then
        local OVERLAY_DIR="${OVERLAY_VIDEO_DIR:-${VIDEO_OUT_PATH}/overlay_videos}"
        mkdir -p "${OVERLAY_DIR}"
        mkdir -p "${OVERLAY_DIR}/${ENV_NAME}"
        OVERLAY_ARGS="--args.save_overlay_video --args.overlay_video_dir ${OVERLAY_DIR}/${ENV_NAME}"
    fi

    local EXTRA_ARGS=""
    if [ "$VLM_DECOMPOSE_SUBTASKS" = true ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --args.vlm_decompose_subtasks"
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    ${ROBOCASA_PYTHON} examples/Robocasa_tabletop/eval_files/simulation_env.py \
        --args.env_name "${ENV_NAME}" \
        --args.port "${PORT}" \
        --args.n_episodes 50 \
        --args.n_envs "${N_ENVS}" \
        --args.max_episode_steps "${MAX_EPISODE_STEPS}" \
        --args.n_action_steps "${N_ACTION_STEPS}" \
        --args.video_out_path "${VIDEO_OUT_PATH}" \
        --args.pretrained_path "${CKPT_PATH}" \
        ${SAM3_ARGS} \
        ${VLM_ARGS} \
        ${OVERLAY_ARGS} \
        ${EXTRA_ARGS} \
        > "${LOG_DIR}/eval_env_${ENV_NAME//\//_}_gpu${GPU_ID}.log" 2>&1 &
    
    local PID=$!
    PID_TO_ENV[$PID]="$ENV_NAME"
    PID_TO_GPU[$PID]=$GPU_ID
    echo "  Started with PID ${PID}"
}

CheckProcessStatus() {
    local PID=$1
    if ! kill -0 "$PID" 2>/dev/null; then
        wait "$PID" 2>/dev/null
        return $?
    fi
    return 255
}

CheckAndCleanFailedEval() {
    local ENV_NAME=$1
    local CRASHED_FLAG=${2:-""}
    local VIDEO_PATH=$(GetVideoOutPath "${CKPT_PATH}" "${N_ACTION_STEPS}" "${MAX_EPISODE_STEPS}" "${N_ENVS}" "${ENV_NAME}")
    local VIDEO_COUNT=$(CountVideos "$VIDEO_PATH")
    
    if [[ "$VIDEO_COUNT" -eq "$EXPECTED_VIDEO_COUNT" ]]; then
        echo "[SUCCESS] ${ENV_NAME}: ${VIDEO_COUNT}/${EXPECTED_VIDEO_COUNT} videos"
        return 0
    else
        if [[ "$CRASHED_FLAG" != "crashed" ]]; then
            echo "[FAILED]  ${ENV_NAME}: ${VIDEO_COUNT}/${EXPECTED_VIDEO_COUNT} videos"
        fi
        if [[ -d "$VIDEO_PATH" ]]; then
            echo "          Deleting failed videos in: ${VIDEO_PATH}"
            rm -rf "${VIDEO_PATH}"/*
        fi
        return 1
    fi
}

RunEvaluationsWithMonitoring() {
    local -n env_list=$1
    local -n failed_list=$2
    
    PID_TO_ENV=()
    PID_TO_GPU=()
    
    declare -A ENV_RETRIES
    declare -A GPU_BUSY
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        GPU_BUSY[$i]=0
    done
    
    local QUEUE=("${env_list[@]}")
    local QUEUE_IDX=0
    
    for ENV_NAME in "${env_list[@]}"; do
        ENV_RETRIES["$ENV_NAME"]=0
    done
    
    local COUNT=0
    while [[ $QUEUE_IDX -lt ${#QUEUE[@]} && $COUNT -lt $NUM_GPUS ]]; do
        local ENV_NAME="${QUEUE[$QUEUE_IDX]}"
        local GPU_ID=$((COUNT % NUM_GPUS))
        local PORT=$((BASE_PORT + GPU_ID))
        
        LaunchEvaluation "$ENV_NAME" "$GPU_ID" "$PORT"
        GPU_BUSY[$GPU_ID]=1
        
        QUEUE_IDX=$((QUEUE_IDX + 1))
        COUNT=$((COUNT + 1))
        sleep 2
    done
    
    while [[ ${#PID_TO_ENV[@]} -gt 0 ]]; do
        sleep $MONITOR_INTERVAL
        
        local PIDS_TO_REMOVE=()
        for PID in "${!PID_TO_ENV[@]}"; do
            local ENV_NAME="${PID_TO_ENV[$PID]}"
            local GPU_ID="${PID_TO_GPU[$PID]}"
            
            if ! kill -0 "$PID" 2>/dev/null; then
                wait "$PID" 2>/dev/null
                local EXIT_CODE=$?
                
                PIDS_TO_REMOVE+=("$PID")
                GPU_BUSY[$GPU_ID]=0
                
                local CRASHED_FLAG=""
                if [[ $EXIT_CODE -ne 0 ]]; then
                    CRASHED_FLAG="crashed"
                    local VIDEO_PATH=$(GetVideoOutPath "${CKPT_PATH}" "${N_ACTION_STEPS}" "${MAX_EPISODE_STEPS}" "${N_ENVS}" "${ENV_NAME}")
                    local VIDEO_COUNT=$(CountVideos "$VIDEO_PATH")
                    echo "[CRASHED] ${ENV_NAME} (PID ${PID}) exited with code ${EXIT_CODE} - ${VIDEO_COUNT}/${EXPECTED_VIDEO_COUNT} videos"
                fi
                
                if ! CheckAndCleanFailedEval "$ENV_NAME" "$CRASHED_FLAG"; then
                    local CURRENT_RETRIES=${ENV_RETRIES["$ENV_NAME"]:-0}
                    CURRENT_RETRIES=$((CURRENT_RETRIES + 1))
                    ENV_RETRIES["$ENV_NAME"]=$CURRENT_RETRIES
                    
                    if [[ $CURRENT_RETRIES -lt $MAX_RETRIES ]]; then
                        QUEUE+=("$ENV_NAME")
                        echo "          Requeued for retry (attempt ${CURRENT_RETRIES}/${MAX_RETRIES})"
                    else
                        failed_list+=("$ENV_NAME")
                        echo "          Max retries (${MAX_RETRIES}) reached, giving up"
                    fi
                fi
                
                if [[ $QUEUE_IDX -lt ${#QUEUE[@]} ]]; then
                    local NEXT_ENV="${QUEUE[$QUEUE_IDX]}"
                    local PORT=$((BASE_PORT + GPU_ID))
                    
                    sleep 2
                    LaunchEvaluation "$NEXT_ENV" "$GPU_ID" "$PORT"
                    GPU_BUSY[$GPU_ID]=1
                    QUEUE_IDX=$((QUEUE_IDX + 1))
                fi
            fi
        done
        
        for PID in "${PIDS_TO_REMOVE[@]}"; do
            unset PID_TO_ENV[$PID]
            unset PID_TO_GPU[$PID]
        done
        
        if [[ ${#PID_TO_ENV[@]} -gt 0 ]]; then
            local REMAINING=$((${#QUEUE[@]} - QUEUE_IDX + ${#PID_TO_ENV[@]}))
            echo "Monitoring: ${#PID_TO_ENV[@]} running, ${REMAINING} remaining (queue: $((${#QUEUE[@]} - QUEUE_IDX)))"
        fi
    done
}

# ============================================================
# Step 2.5: Filter out already-successful evaluations
# ============================================================

echo ""
echo "=== Checking for previously successful evaluations ==="

ENVS_TO_EVALUATE=()
SKIPPED_ENVS=()

for ENV_NAME in "${ENV_NAMES[@]}"; do
    VIDEO_PATH=$(GetVideoOutPath "${CKPT_PATH}" "${N_ACTION_STEPS}" "${MAX_EPISODE_STEPS}" "${N_ENVS}" "${ENV_NAME}")
    if CheckEvalSuccess "$VIDEO_PATH" "$EXPECTED_VIDEO_COUNT"; then
        SKIPPED_ENVS+=("$ENV_NAME")
        echo "[SKIP] ${ENV_NAME}: Already has ${EXPECTED_VIDEO_COUNT} videos"
    else
        ENVS_TO_EVALUATE+=("$ENV_NAME")
        VIDEO_COUNT=$(CountVideos "$VIDEO_PATH")
        echo "[QUEUE] ${ENV_NAME}: ${VIDEO_COUNT}/${EXPECTED_VIDEO_COUNT} videos"
    fi
done

echo ""
echo "Skipped: ${#SKIPPED_ENVS[@]} environments (already successful)"
echo "To evaluate: ${#ENVS_TO_EVALUATE[@]} environments"

FINAL_FAILED_ENVS=()

if [[ ${#ENVS_TO_EVALUATE[@]} -eq 0 ]]; then
    echo ""
    echo "=== All evaluations already completed! Nothing to do. ==="
else
    echo ""
    echo "=== Running Evaluations (with up to ${MAX_RETRIES} retries per environment) ==="
    RunEvaluationsWithMonitoring ENVS_TO_EVALUATE FINAL_FAILED_ENVS
fi

# Final status report
echo ""
echo "=== Final Evaluation Status ==="
if [[ ${#FINAL_FAILED_ENVS[@]} -eq 0 ]]; then
    echo "All evaluations completed successfully!"
else
    echo "WARNING: ${#FINAL_FAILED_ENVS[@]} evaluation(s) still failed after ${MAX_RETRIES} retries:"
    for ENV_NAME in "${FINAL_FAILED_ENVS[@]}"; do
        VIDEO_PATH=$(GetVideoOutPath "${CKPT_PATH}" "${N_ACTION_STEPS}" "${MAX_EPISODE_STEPS}" "${N_ENVS}" "${ENV_NAME}")
        VIDEO_COUNT=$(CountVideos "$VIDEO_PATH")
        echo "  - ${ENV_NAME}: ${VIDEO_COUNT}/${EXPECTED_VIDEO_COUNT} videos"
    done
fi

# ============================================================
# Step 3: Cleanup
# ============================================================

echo ""
echo "Shutting down policy servers..."

for PID in "${SERVER_PIDS[@]}"; do
    kill "${PID}" 2>/dev/null && echo "Killed policy server PID ${PID}"
done

if [ "$USE_SAM3" = true ]; then
    echo "Shutting down SAM3 servers..."
    for PID in "${SAM3_PIDS[@]}"; do
        kill "${PID}" 2>/dev/null && echo "Killed SAM3 server PID ${PID}"
    done
fi

if [ "$USE_VLM" = true ]; then
    echo "Shutting down VLM servers..."
    for PID in "${VLM_PIDS[@]}"; do
        kill "${PID}" 2>/dev/null && echo "Killed VLM server PID ${PID}"
    done
fi

echo "=== Evaluation Finished ==="

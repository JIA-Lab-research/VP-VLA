#!/bin/bash
# Parallel Processing Launcher Script (FORWARD ORDER) for OXE Datasets
#
# Processes bridge_orig_lerobot and fractal20220817_data_lerobot datasets
# using VLM+SAM servers to generate visual prompt segmentation masks.
#
# Usage:
#   ./run_parallel_processing.sh [NUM_GPUS] [SAM_SERVERS_PER_GPU] [VLM_SERVERS_PER_GPU]
#
# Example:
#   ./run_parallel_processing.sh 8 6 6

set -e

# ===============================
# CONFIGURATION
# ===============================

NUM_GPUS=${1:-8}
SAM_SERVERS_PER_GPU=${2:-6}
NUM_SAM_SERVERS=$((NUM_GPUS * SAM_SERVERS_PER_GPU))
VLM_SERVERS_PER_GPU=${3:-6}
NUM_VLM_SERVERS=$((NUM_GPUS * VLM_SERVERS_PER_GPU))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTILITY_DIR="${SCRIPT_DIR}/../../examples/Robocasa_tabletop/visual_prompt_utility"
DATASETS_ROOT="/path/to/oxe/datasets"                  # TODO: set your OXE datasets root
OUTPUT_DIR="/path/to/visual_prompt_output"              # TODO: set your output directory

SAM_MODEL_PATH="/path/to/sam3"                          # TODO: set your SAM3 model path
VLM_MODEL_PATH="/path/to/Qwen3-VL-4B-Instruct"         # TODO: set your VLM model path

SAM_BASE_PORT=20094
VLM_BASE_PORT=24200

DATASETS=(
    "bridge_orig_lerobot"
    "fractal20220817_data_lerobot"
)

# ===============================
# LOGGING
# ===============================

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ===============================
# SERVER MANAGEMENT
# ===============================

start_sam_servers() {
    log "Starting ${NUM_SAM_SERVERS} SAM3 servers (${SAM_SERVERS_PER_GPU} per GPU, ${NUM_GPUS} GPUs)..."
    local server_idx=0
    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        for s in $(seq 0 $((SAM_SERVERS_PER_GPU - 1))); do
            port=$((SAM_BASE_PORT + server_idx))
            log "  Starting SAM3 server on GPU ${gpu}, port ${port}"
            CUDA_VISIBLE_DEVICES=${gpu} python "${UTILITY_DIR}/sam3_server.py" \
                --model-path "${SAM_MODEL_PATH}" \
                --port ${port} \
                --idle-timeout -1 \
                > "${LOG_DIR}/sam_server_${port}.log" 2>&1 &
            echo $! > "${LOG_DIR}/sam_server_${port}.pid"
            server_idx=$((server_idx + 1))
        done
    done
}

start_vlm_servers() {
    log "Starting ${NUM_VLM_SERVERS} VLM servers (${VLM_SERVERS_PER_GPU} per GPU, ${NUM_GPUS} GPUs)..."
    local server_idx=0
    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        for s in $(seq 0 $((VLM_SERVERS_PER_GPU - 1))); do
            port=$((VLM_BASE_PORT + server_idx))
            log "  Starting VLM server on GPU ${gpu}, port ${port}"
            CUDA_VISIBLE_DEVICES=${gpu} python "${SCRIPT_DIR}/vlm_server_subtask_oxe.py" \
                --model-path "${VLM_MODEL_PATH}" \
                --port ${port} \
                --idle-timeout -1 \
                > "${LOG_DIR}/vlm_server_${port}.log" 2>&1 &
            echo $! > "${LOG_DIR}/vlm_server_${port}.pid"
            server_idx=$((server_idx + 1))
        done
    done
}

wait_for_servers() {
    log "Waiting for servers to be ready..."
    sleep 30

    for i in $(seq 0 $((NUM_SAM_SERVERS - 1))); do
        port=$((SAM_BASE_PORT + i))
        if [ -f "${LOG_DIR}/sam_server_${port}.pid" ]; then
            pid=$(cat "${LOG_DIR}/sam_server_${port}.pid")
            if ! ps -p $pid > /dev/null 2>&1; then
                log "  WARNING: SAM3 server on port ${port} is not running!"
            fi
        fi
    done

    for i in $(seq 0 $((NUM_VLM_SERVERS - 1))); do
        port=$((VLM_BASE_PORT + i))
        if [ -f "${LOG_DIR}/vlm_server_${port}.pid" ]; then
            pid=$(cat "${LOG_DIR}/vlm_server_${port}.pid")
            if ! ps -p $pid > /dev/null 2>&1; then
                log "  WARNING: VLM server on port ${port} is not running!"
            fi
        fi
    done

    log "Servers should be ready."
}

stop_servers() {
    log "Stopping servers..."
    for pidfile in "${LOG_DIR}"/*.pid; do
        if [ -f "$pidfile" ]; then
            pid=$(cat "$pidfile")
            if ps -p $pid > /dev/null 2>&1; then
                log "  Stopping process $pid"
                kill $pid 2>/dev/null || true
            fi
            rm -f "$pidfile"
        fi
    done
}

# ===============================
# WORKER MANAGEMENT
# ===============================

wait_for_workers() {
    log "Waiting for all workers to complete..."
    while true; do
        local running=0
        for pidfile in "${LOG_DIR}"/worker_*.pid; do
            if [ -f "$pidfile" ]; then
                pid=$(cat "$pidfile")
                if ps -p $pid > /dev/null 2>&1; then
                    running=$((running + 1))
                else
                    rm -f "$pidfile"
                fi
            fi
        done
        if [ $running -eq 0 ]; then
            break
        fi
        log "  ${running} workers still running..."
        sleep 60
    done
    log "All workers completed."
}

get_total_episodes() {
    local dataset_path=$1
    python3 -c "
import json
with open('${dataset_path}/meta/info.json') as f:
    print(json.load(f)['total_episodes'])
"
}

# ===============================
# MAIN PROCESSING LOOP
# ===============================

process_dataset() {
    local dataset_name=$1
    local dataset_path="${DATASETS_ROOT}/${dataset_name}"

    if [ ! -d "${dataset_path}" ]; then
        log "ERROR: Dataset not found: ${dataset_path}"
        return 1
    fi

    local total_episodes
    total_episodes=$(get_total_episodes "${dataset_path}")
    log "Dataset: ${dataset_name} (${total_episodes} episodes)"

    local episodes_per_worker=$(( (total_episodes + NUM_SAM_SERVERS - 1) / NUM_SAM_SERVERS ))

    local worker_idx=0
    local ep_start=0

    while [ $ep_start -lt $total_episodes ]; do
        local batch_start=$ep_start
        local workers_this_batch=0

        log "=============================================="
        log "Starting batch from episode ${ep_start} for ${dataset_name}"
        log "=============================================="

        for w in $(seq 0 $((NUM_SAM_SERVERS - 1))); do
            if [ $ep_start -ge $total_episodes ]; then
                break
            fi

            local ep_end=$((ep_start + episodes_per_worker - 1))
            if [ $ep_end -ge $total_episodes ]; then
                ep_end=$((total_episodes - 1))
            fi

            local sam_port=$((SAM_BASE_PORT + w))
            local vlm_port=$((VLM_BASE_PORT + w))
            local worker_id="${dataset_name}_ep${ep_start}-${ep_end}"

            log "  Launching worker: ${worker_id} (SAM:${sam_port}, VLM:${vlm_port})"

            python "${SCRIPT_DIR}/process_dataset_parallel_oxe.py" \
                --dataset-root "${dataset_path}" \
                --episode-start ${ep_start} \
                --episode-end ${ep_end} \
                --sam-port ${sam_port} \
                --vlm-port ${vlm_port} \
                --output-dir "${OUTPUT_DIR}" \
                > "${LOG_DIR}/worker_${worker_id}.log" 2>&1 &

            echo $! > "${LOG_DIR}/worker_${worker_id}.pid"
            workers_this_batch=$((workers_this_batch + 1))
            ep_start=$((ep_end + 1))
        done

        wait_for_workers
    done
}

# ===============================
# MAIN
# ===============================

main() {
    log "=============================================="
    log "OXE Parallel Processing Pipeline (FORWARD ORDER)"
    log "=============================================="
    log "Configuration:"
    log "  NUM_GPUS: ${NUM_GPUS}"
    log "  SAM_SERVERS_PER_GPU: ${SAM_SERVERS_PER_GPU}"
    log "  NUM_SAM_SERVERS: ${NUM_SAM_SERVERS}"
    log "  VLM_SERVERS_PER_GPU: ${VLM_SERVERS_PER_GPU}"
    log "  NUM_VLM_SERVERS: ${NUM_VLM_SERVERS}"
    log "  DATASETS: ${DATASETS[*]}"
    log "  OUTPUT_DIR: ${OUTPUT_DIR}"
    log "=============================================="

    mkdir -p "${OUTPUT_DIR}"
    trap stop_servers EXIT

    start_sam_servers
    start_vlm_servers
    wait_for_servers

    for dataset_name in "${DATASETS[@]}"; do
        process_dataset "${dataset_name}"
    done

    log "=============================================="
    log "All datasets completed!"
    log "=============================================="
}

if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi

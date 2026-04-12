#!/bin/bash
# Parallel Processing Launcher Script (Pre-extracted Frames Version)
# 
# This script launches:
# 1. Multiple SAM3 servers (multiple per GPU for higher throughput)
# 2. Multiple VLM servers (multiple per GPU for higher throughput)
# 3. Worker processes to process episodes in parallel
#
# Uses pre-extracted JPEG frames instead of video decoding to reduce CPU usage.
# Frames should be pre-extracted to FRAMES_ROOT directory.
#
# Usage:
#   ./run_parallel_processing.sh [NUM_GPUS] [SAM_SERVERS_PER_GPU] [VLM_SERVERS_PER_GPU] [WORKERS_PER_TASK]
#
# Example:
#   ./run_parallel_processing.sh 8 2 2 1   # 8 GPUs, 6 SAM/GPU (48 total), 6 VLM/GPU (48 total), 2 workers/task
#                                          # 24 tasks × 2 workers = 48 workers, each with dedicated SAM+VLM pair

set -e

# ===============================
# CONFIGURATION
# ===============================

# Number of GPUs to use
NUM_GPUS=${1:-8}

# Number of SAM servers per GPU
SAM_SERVERS_PER_GPU=${2:-6}

# Total SAM servers
NUM_SAM_SERVERS=$((NUM_GPUS * SAM_SERVERS_PER_GPU))

# Number of VLM servers per GPU (should match SAM for 1:1 pairing)
VLM_SERVERS_PER_GPU=${3:-6}

# Total VLM servers
NUM_VLM_SERVERS=$((NUM_GPUS * VLM_SERVERS_PER_GPU))

# Number of workers per task (episodes are split among workers)
# With 24 tasks and 2 workers/task = 48 workers, matching 48 SAM+VLM pairs
WORKERS_PER_TASK=${4:-2}

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTILITY_DIR="${SCRIPT_DIR}/../../examples/Robocasa_tabletop/visual_prompt_utility"
DATASET_ROOT="/path/to/robocasa/dataset"             # TODO: set your dataset root
FRAMES_ROOT="/path/to/extracted_frames_robocasa"      # TODO: set your pre-extracted frames root
OUTPUT_DIR="/path/to/visual_prompt_output"             # TODO: set your output directory

# Model paths
SAM_MODEL_PATH="/path/to/sam3"                         # TODO: set your SAM3 model path
VLM_MODEL_PATH="/path/to/Qwen3-VL-4B-Instruct"        # TODO: set your VLM model path

# Port ranges
# With 8 GPUs and 6 servers/GPU: 48 SAM ports (13094-13141), 48 VLM ports (17200-17247)
# Each worker gets a dedicated SAM+VLM pair for 1:1 mapping
SAM_BASE_PORT=13094
VLM_BASE_PORT=17200

# All task folders
TASKS=(
    "gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000"
    "gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000"
)

# Episodes per task
EPISODES_PER_TASK=1000

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
            
            CUDA_VISIBLE_DEVICES=${gpu} python "${SCRIPT_DIR}/vlm_server_subtask.py" \
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
    sleep 30  # Give servers time to load models
    
    # Check if servers are responsive
    for i in $(seq 0 $((NUM_SAM_SERVERS - 1))); do
        port=$((SAM_BASE_PORT + i))
        log "  Checking SAM3 server on port ${port}..."
        # Simple check: see if process is still running
        if [ -f "${LOG_DIR}/sam_server_${port}.pid" ]; then
            pid=$(cat "${LOG_DIR}/sam_server_${port}.pid")
            if ! ps -p $pid > /dev/null 2>&1; then
                log "  WARNING: SAM3 server on port ${port} is not running!"
            fi
        fi
    done
    
    for i in $(seq 0 $((NUM_VLM_SERVERS - 1))); do
        port=$((VLM_BASE_PORT + i))
        log "  Checking VLM server on port ${port}..."
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

launch_workers_for_task() {
    local task=$1
    local sam_port=$2
    local vlm_port=$3
    local worker_offset=$4
    
    local episodes_per_worker=$((EPISODES_PER_TASK / WORKERS_PER_TASK))
    
    log "Launching ${WORKERS_PER_TASK} workers for task: ${task}"
    log "  SAM port: ${sam_port}, VLM port: ${vlm_port}"
    
    for w in $(seq 0 $((WORKERS_PER_TASK - 1))); do
        local start=$((w * episodes_per_worker))
        local end=$(((w + 1) * episodes_per_worker - 1))
        
        # Handle remainder for last worker
        if [ $w -eq $((WORKERS_PER_TASK - 1)) ]; then
            end=$((EPISODES_PER_TASK - 1))
        fi
        
        local worker_id="${task}_w${w}"
        log "  Worker ${w}: episodes ${start}-${end}"
        
        python "${SCRIPT_DIR}/process_dataset_parallel.py" \
            --dataset-root "${DATASET_ROOT}" \
            --task "${task}" \
            --episode-start ${start} \
            --episode-end ${end} \
            --sam-port ${sam_port} \
            --vlm-port ${vlm_port} \
            --output-dir "${OUTPUT_DIR}" \
            --frames-root "${FRAMES_ROOT}" \
            > "${LOG_DIR}/worker_${worker_id}.log" 2>&1 &
        
        echo $! > "${LOG_DIR}/worker_${worker_id}.pid"
    done
}

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

# ===============================
# MAIN PROCESSING LOOP
# ===============================

process_all_tasks() {
    local num_tasks=${#TASKS[@]}
    local tasks_per_batch=$((NUM_SAM_SERVERS / WORKERS_PER_TASK))
    
    if [ $tasks_per_batch -lt 1 ]; then
        tasks_per_batch=1
    fi
    
    log "Processing ${num_tasks} tasks in batches of ${tasks_per_batch}"
    
    local task_idx=0
    while [ $task_idx -lt $num_tasks ]; do
        log "=============================================="
        log "Starting batch: tasks ${task_idx} to $((task_idx + tasks_per_batch - 1))"
        log "=============================================="
        
        # Launch workers for this batch
        for b in $(seq 0 $((tasks_per_batch - 1))); do
            local current_task_idx=$((task_idx + b))
            if [ $current_task_idx -ge $num_tasks ]; then
                break
            fi
            
            local task="${TASKS[$current_task_idx]}"
            
            # Each worker gets its own dedicated SAM+VLM pair
            # Worker index across all tasks in this batch: (b * WORKERS_PER_TASK) + w
            for w in $(seq 0 $((WORKERS_PER_TASK - 1))); do
                local worker_global_idx=$(((b * WORKERS_PER_TASK) + w))
                local worker_sam_port=$((SAM_BASE_PORT + (worker_global_idx % NUM_SAM_SERVERS)))
                local worker_vlm_port=$((VLM_BASE_PORT + (worker_global_idx % NUM_VLM_SERVERS)))
                
                local episodes_per_worker=$((EPISODES_PER_TASK / WORKERS_PER_TASK))
                local start=$((w * episodes_per_worker))
                local end=$(((w + 1) * episodes_per_worker - 1))
                
                if [ $w -eq $((WORKERS_PER_TASK - 1)) ]; then
                    end=$((EPISODES_PER_TASK - 1))
                fi
                
                local worker_id="${task}_w${w}"
                log "  Launching worker: ${worker_id} (SAM:${worker_sam_port}, VLM:${worker_vlm_port}, ep:${start}-${end})"
                
                python "${SCRIPT_DIR}/process_dataset_parallel.py" \
                    --dataset-root "${DATASET_ROOT}" \
                    --task "${task}" \
                    --episode-start ${start} \
                    --episode-end ${end} \
                    --sam-port ${worker_sam_port} \
                    --vlm-port ${worker_vlm_port} \
                    --output-dir "${OUTPUT_DIR}" \
                    --frames-root "${FRAMES_ROOT}" \
                    > "${LOG_DIR}/worker_${worker_id}.log" 2>&1 &
                
                echo $! > "${LOG_DIR}/worker_${worker_id}.pid"
            done
        done
        
        # Wait for this batch to complete
        wait_for_workers
        
        task_idx=$((task_idx + tasks_per_batch))
    done
}

# ===============================
# MAIN
# ===============================

main() {
    log "=============================================="
    log "Parallel Processing Pipeline (Pre-extracted Frames)"
    log "=============================================="
    log "Configuration:"
    log "  NUM_GPUS: ${NUM_GPUS}"
    log "  SAM_SERVERS_PER_GPU: ${SAM_SERVERS_PER_GPU}"
    log "  NUM_SAM_SERVERS: ${NUM_SAM_SERVERS} (total)"
    log "  VLM_SERVERS_PER_GPU: ${VLM_SERVERS_PER_GPU}"
    log "  NUM_VLM_SERVERS: ${NUM_VLM_SERVERS} (total)"
    log "  WORKERS_PER_TASK: ${WORKERS_PER_TASK}"
    log "  TASKS: ${#TASKS[@]}"
    log "  FRAMES_ROOT: ${FRAMES_ROOT}"
    log "  OUTPUT_DIR: ${OUTPUT_DIR}"
    log "=============================================="
    
    # Create output directory
    mkdir -p "${OUTPUT_DIR}"
    
    # Trap to cleanup on exit
    trap stop_servers EXIT
    
    # Start servers
    start_sam_servers
    start_vlm_servers
    wait_for_servers
    
    # Process all tasks
    process_all_tasks
    
    log "=============================================="
    log "All tasks completed!"
    log "=============================================="
}

# Run if executed directly
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi

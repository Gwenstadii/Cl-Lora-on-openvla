#!/bin/bash
set -euo pipefail

CONDA_ENV="/root/autodl-tmp/conda_envs/RoboTwin"
REPO_DIR="/root/autodl-tmp/RoboTwin/policy/pi05"
TRAIN_CONFIG="pi0_robotwin_stack_bowls_two_cl_lora_proto_replay_v2"
MODEL_NAME="robotwin_stack_bowls_two_pi05_lang1"
GPU_IDS="0,1,2,3"
CHECKPOINTS="10000,20000"
TASKS="taskA,taskB,taskC"
EXPERIMENT_TAG="pi0_cl_lora_proto_replay_after_taskC_v2_free"
TASK_CONFIG="pi05_clean_lang1"
SEED="0"
EPISODES="50"

LOG_DIR="${REPO_DIR}/run_logs"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${TRAIN_CONFIG}_${RUN_STAMP}.log"

mkdir -p "${LOG_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

activate_conda() {
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
        # shellcheck source=/dev/null
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
    elif [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck source=/dev/null
        source "/root/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        # shellcheck source=/dev/null
        source "/opt/conda/etc/profile.d/conda.sh"
    else
        echo "Cannot find conda initialization script." >&2
        exit 1
    fi
    conda activate "${CONDA_ENV}"
}

export_common_cache_env() {
    export TMPDIR=/root/autodl-tmp/robotwin_cache/tmp
    export PIP_CACHE_DIR=/root/autodl-tmp/robotwin_cache/pip
    export XDG_CACHE_HOME=/root/autodl-tmp/robotwin_cache/xdg
    export HF_HOME=/root/autodl-tmp/robotwin_cache/hf
    export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/robotwin_cache/hf/hub
    export UV_CACHE_DIR=/root/autodl-tmp/robotwin_cache/uv
    export WANDB_MODE=offline
    mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${UV_CACHE_DIR}"
}

export_eval_cuda_env() {
    export CUDA_HOME=/root/autodl-tmp/cuda_toolkits/cuda124
    export CUDA_PATH="${CUDA_HOME}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
    export CUDACXX="${CUDA_HOME}/bin/nvcc"
    export TORCH_CUDA_ARCH_LIST="8.0"
}

preflight_check() {
    local norm_stats="${REPO_DIR}/assets/${TRAIN_CONFIG}/${MODEL_NAME}/norm_stats.json"
    log "Preflight: checking norm stats: ${norm_stats}"
    if [ ! -f "${norm_stats}" ]; then
        echo "Missing norm stats: ${norm_stats}" >&2
        echo "Run this before launching training:" >&2
        echo "  uv run scripts/compute_norm_stats.py --config-name ${TRAIN_CONFIG}" >&2
        exit 1
    fi
}

run_train() {
    log "Starting training: ${TRAIN_CONFIG}, model=${MODEL_NAME}, gpus=${GPU_IDS}"
    activate_conda
    cd "${REPO_DIR}"
    export_common_cache_env
    preflight_check
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; training with --overwrite"
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_cl_lora.py \
        "${TRAIN_CONFIG}" \
        --exp-name="${MODEL_NAME}" \
        --overwrite
    log "Training finished successfully."
}

run_eval() {
    log "Starting evaluation: checkpoints=${CHECKPOINTS}, tasks=${TASKS}"
    activate_conda
    cd "${REPO_DIR}"
    # shellcheck source=/dev/null
    source .venv/bin/activate
    export_common_cache_env
    export_eval_cuda_env
    bash eval_checkpoints_tasks.sh \
        --train-config-name "${TRAIN_CONFIG}" \
        --model-name "${MODEL_NAME}" \
        --checkpoints "${CHECKPOINTS}" \
        --tasks "${TASKS}" \
        --experiment-tag "${EXPERIMENT_TAG}" \
        --task-config "${TASK_CONFIG}" \
        --seed "${SEED}" \
        --gpu-ids "${GPU_IDS}" \
        --episodes "${EPISODES}"
    log "Evaluation finished successfully."
}

main() {
    log "Run log: ${LOG_FILE}"
    run_train
    run_eval
    log "All done."
}

main "$@" 2>&1 | tee -a "${LOG_FILE}"

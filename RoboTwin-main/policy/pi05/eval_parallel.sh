#!/bin/bash
set -euo pipefail

policy_name=pi05
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_ids=${6}
checkpoint_id=${7:-10000}
fixed_instruction="${8:-}"
cl_task_id="${9:-}"
eval_repo_id="${10:-}"
total_episodes="${11:-50}"
result_tag="${12:-parallel_$(date +%Y%m%d_%H%M%S)}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

IFS=',' read -ra gpus <<< "${gpu_ids}"
num_workers=${#gpus[@]}
if [ "${num_workers}" -lt 1 ]; then
    echo "No GPU ids provided." >&2
    exit 1
fi

base_episodes=$((total_episodes / num_workers))
extra_episodes=$((total_episodes % num_workers))

run_dir="${repo_root}/eval_result/${task_name}/${policy_name}/${task_config}/${model_name}/${result_tag}"
log_dir="${repo_root}/eval_parallel_logs/${task_name}/${task_config}/${model_name}/${result_tag}"
mkdir -p "${run_dir}" "${log_dir}"

echo "Parallel eval tag: ${result_tag}"
echo "GPUs: ${gpu_ids}; workers: ${num_workers}; total episodes: ${total_episodes}"
echo "Result dir: ${run_dir}"
echo "Log dir: ${log_dir}"

pids=()
worker_ids=()

for worker_id in "${!gpus[@]}"; do
    gpu_id="${gpus[$worker_id]}"
    gpu_id="${gpu_id//[[:space:]]/}"
    worker_episodes="${base_episodes}"
    if [ "${worker_id}" -lt "${extra_episodes}" ]; then
        worker_episodes=$((worker_episodes + 1))
    fi
    if [ "${worker_episodes}" -le 0 ]; then
        continue
    fi

    log_file="${log_dir}/worker_${worker_id}_gpu_${gpu_id}.log"
    echo "Launching worker ${worker_id}/${num_workers} on GPU ${gpu_id}, episodes=${worker_episodes}"

    bash "${script_dir}/eval.sh" \
        "${task_name}" \
        "${task_config}" \
        "${train_config_name}" \
        "${model_name}" \
        "${seed}" \
        "${gpu_id}" \
        "${checkpoint_id}" \
        "${fixed_instruction}" \
        "${cl_task_id}" \
        "${eval_repo_id}" \
        "${worker_episodes}" \
        "${worker_id}" \
        "${num_workers}" \
        "${result_tag}" \
        > "${log_file}" 2>&1 &

    pids+=("$!")
    worker_ids+=("${worker_id}")
done

failed=0
for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"
    worker_id="${worker_ids[$idx]}"
    if wait "${pid}"; then
        echo "Worker ${worker_id} finished."
    else
        echo "Worker ${worker_id} failed. See ${log_dir}/worker_${worker_id}_*.log" >&2
        failed=1
    fi
done

if [ "${failed}" -ne 0 ]; then
    exit 1
fi

cd "${repo_root}"
python script/merge_eval_results.py --run-dir "${run_dir}"

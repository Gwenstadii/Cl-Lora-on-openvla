#!/bin/bash
set -uo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash eval_checkpoints_tasks.sh \
    --train-config-name CONFIG \
    --model-name EXPERIMENT \
    --checkpoints STEP[,STEP...] \
    --tasks TASK[,TASK...] \
    [--experiment-tag TAG] \
    [--task-config CONFIG] \
    [--seed SEED] \
    [--gpu-ids GPU[,GPU...]] \
    [--episodes COUNT]

Task aliases:
  taskA, A, handover_mic
  taskB, B, grab_roller
  taskC, C, stack_bowls_two
  taskD, D, open_laptop

Example:
  bash eval_checkpoints_tasks.sh \
    --train-config-name pi05_robotwin_grab_roller_cl_lora_uniform_replay \
    --model-name robotwin_grab_roller_pi05_lang1 \
    --checkpoints 2500,5000,7500,10000 \
    --tasks taskA,taskB \
    --experiment-tag uniform_replay_after_taskB_v1
EOF
}

die() {
    echo "Error: $*" >&2
    echo >&2
    usage >&2
    exit 2
}

require_value() {
    local option="$1"
    local value="${2-}"
    [[ -n "${value}" ]] || die "${option} requires a value."
}

train_config_name=""
model_name=""
checkpoints_csv=""
tasks_csv=""
experiment_tag="batch_eval"
task_config="pi05_clean_lang1"
seed=0
gpu_ids="0,1,2,3"
total_episodes=50

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-config-name|--train-config)
            require_value "$1" "${2-}"
            train_config_name="$2"
            shift 2
            ;;
        --model-name|--exp-name)
            require_value "$1" "${2-}"
            model_name="$2"
            shift 2
            ;;
        --checkpoints)
            require_value "$1" "${2-}"
            checkpoints_csv="$2"
            shift 2
            ;;
        --tasks)
            require_value "$1" "${2-}"
            tasks_csv="$2"
            shift 2
            ;;
        --experiment-tag|--tag)
            require_value "$1" "${2-}"
            experiment_tag="$2"
            shift 2
            ;;
        --task-config)
            require_value "$1" "${2-}"
            task_config="$2"
            shift 2
            ;;
        --seed)
            require_value "$1" "${2-}"
            seed="$2"
            shift 2
            ;;
        --gpu-ids)
            require_value "$1" "${2-}"
            gpu_ids="$2"
            shift 2
            ;;
        --episodes)
            require_value "$1" "${2-}"
            total_episodes="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
done

[[ -n "${train_config_name}" ]] || die "--train-config-name is required."
[[ -n "${model_name}" ]] || die "--model-name is required."
[[ -n "${checkpoints_csv}" ]] || die "--checkpoints is required."
[[ -n "${tasks_csv}" ]] || die "--tasks is required."
[[ "${seed}" =~ ^[0-9]+$ ]] || die "--seed must be a non-negative integer."
[[ "${total_episodes}" =~ ^[1-9][0-9]*$ ]] || die "--episodes must be a positive integer."
gpu_ids="${gpu_ids//[[:space:]]/}"
[[ "${gpu_ids}" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--gpu-ids must look like 0 or 0,1,2,3."
[[ "${experiment_tag}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || die "--experiment-tag may contain only letters, digits, dot, underscore, and hyphen."

IFS=',' read -ra checkpoint_tokens <<< "${checkpoints_csv}"
checkpoints=()
seen_checkpoints="|"
for token in "${checkpoint_tokens[@]}"; do
    checkpoint_id="${token//[[:space:]]/}"
    [[ "${checkpoint_id}" =~ ^[1-9][0-9]*$ ]] || die "Invalid checkpoint step: ${token}"
    [[ "${seen_checkpoints}" != *"|${checkpoint_id}|"* ]] || die "Duplicate checkpoint: ${checkpoint_id}"
    checkpoints+=("${checkpoint_id}")
    seen_checkpoints+="${checkpoint_id}|"
done
[[ ${#checkpoints[@]} -gt 0 ]] || die "No checkpoint steps were provided."

resolve_task() {
    local task_token="${1,,}"
    case "${task_token}" in
        taska|a|handover_mic)
            printf '%s|%s|%s|%s|%s\n' \
                "taskA" "handover_mic" "0" "robotwin_handover_mic_pi05_lang1" \
                "Pick up the handheld microphone and hand it over"
            ;;
        taskb|b|grab_roller)
            printf '%s|%s|%s|%s|%s\n' \
                "taskB" "grab_roller" "1" "robotwin_grab_roller_pi05_lang1" \
                "Grab the smooth wooden roller with both arms"
            ;;
        taskc|c|stack_bowls_two)
            printf '%s|%s|%s|%s|%s\n' \
                "taskC" "stack_bowls_two" "2" "robotwin_stack_bowls_two_pi05_lang1" \
                "Stack the small smooth brown-rimmed bowl directly over the smooth bowl with glossy finish"
            ;;
        taskd|d|open_laptop)
            printf '%s|%s|%s|%s|%s\n' \
                "taskD" "open_laptop" "3" "robotwin_open_laptop_pi05_lang1" \
                "Raise the lid of the rectangular laptop with hinge"
            ;;
        *)
            return 1
            ;;
    esac
}

IFS=',' read -ra task_tokens <<< "${tasks_csv}"
task_labels=()
task_names=()
cl_task_ids=()
eval_repo_ids=()
instructions=()
seen_tasks="|"
for token in "${task_tokens[@]}"; do
    compact_token="${token//[[:space:]]/}"
    task_spec="$(resolve_task "${compact_token}")" || die "Unknown task: ${token}"
    IFS='|' read -r task_label task_name cl_task_id eval_repo_id instruction <<< "${task_spec}"
    [[ "${seen_tasks}" != *"|${task_label}|"* ]] || die "Duplicate task: ${task_label}"
    task_labels+=("${task_label}")
    task_names+=("${task_name}")
    cl_task_ids+=("${cl_task_id}")
    eval_repo_ids+=("${eval_repo_id}")
    instructions+=("${instruction}")
    seen_tasks+="${task_label}|"
done
[[ ${#task_names[@]} -gt 0 ]] || die "No evaluation tasks were provided."

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
policy_name="pi05"

[[ -f "${script_dir}/eval_parallel.sh" ]] || die "Cannot find ${script_dir}/eval_parallel.sh"
for checkpoint_id in "${checkpoints[@]}"; do
    checkpoint_dir="${script_dir}/checkpoints/${train_config_name}/${model_name}/${checkpoint_id}"
    [[ -d "${checkpoint_dir}/params" ]] || die "Missing checkpoint params: ${checkpoint_dir}/params"
done

run_stamp="$(date +%Y%m%d_%H%M%S)"
checkpoint_tag="$(IFS=_; echo "${checkpoints[*]}")"
task_tag="$(IFS=_; echo "${task_labels[*]}")"
batch_log_dir="${repo_root}/eval_batch_logs"
batch_log="${batch_log_dir}/${experiment_tag}_${checkpoint_tag}_${task_tag}_${run_stamp}.log"
mkdir -p "${batch_log_dir}"

summary_rows=()
failed=0
run_index=0
total_runs=$((${#checkpoints[@]} * ${#task_names[@]}))

echo "Batch evaluation started: ${run_stamp}" | tee -a "${batch_log}"
echo "Experiment tag: ${experiment_tag}" | tee -a "${batch_log}"
echo "Train config: ${train_config_name}" | tee -a "${batch_log}"
echo "Model/experiment: ${model_name}" | tee -a "${batch_log}"
echo "Checkpoints: ${checkpoints[*]}" | tee -a "${batch_log}"
echo "Tasks: ${task_labels[*]}" | tee -a "${batch_log}"
echo "GPUs: ${gpu_ids}; episodes per evaluation: ${total_episodes}" | tee -a "${batch_log}"
echo "Combined terminal log: ${batch_log}" | tee -a "${batch_log}"

for checkpoint_id in "${checkpoints[@]}"; do
    for task_index in "${!task_names[@]}"; do
        run_index=$((run_index + 1))
        task_label="${task_labels[$task_index]}"
        task_name="${task_names[$task_index]}"
        cl_task_id="${cl_task_ids[$task_index]}"
        eval_repo_id="${eval_repo_ids[$task_index]}"
        instruction="${instructions[$task_index]}"
        result_tag="${task_label}_${experiment_tag}_ckpt${checkpoint_id}_seed${seed}_${total_episodes}_${run_stamp}"
        result_dir="${repo_root}/eval_result/${task_name}/${policy_name}/${task_config}/${model_name}/${result_tag}"

        echo | tee -a "${batch_log}"
        echo "================================================================" | tee -a "${batch_log}"
        echo "[${run_index}/${total_runs}] checkpoint=${checkpoint_id}, ${task_label} (${task_name})" \
            | tee -a "${batch_log}"
        echo "result_tag=${result_tag}" | tee -a "${batch_log}"
        printf 'Command:' | tee -a "${batch_log}"
        printf ' %q' \
            bash "${script_dir}/eval_parallel.sh" \
            "${task_name}" "${task_config}" "${train_config_name}" "${model_name}" \
            "${seed}" "${gpu_ids}" "${checkpoint_id}" "${instruction}" \
            "${cl_task_id}" "${eval_repo_id}" "${total_episodes}" "${result_tag}" \
            | tee -a "${batch_log}"
        echo | tee -a "${batch_log}"
        echo "----------------------------------------------------------------" | tee -a "${batch_log}"

        bash "${script_dir}/eval_parallel.sh" \
            "${task_name}" \
            "${task_config}" \
            "${train_config_name}" \
            "${model_name}" \
            "${seed}" \
            "${gpu_ids}" \
            "${checkpoint_id}" \
            "${instruction}" \
            "${cl_task_id}" \
            "${eval_repo_id}" \
            "${total_episodes}" \
            "${result_tag}" \
            2>&1 | tee -a "${batch_log}"
        status=${PIPESTATUS[0]}

        if [[ ${status} -eq 0 && -f "${result_dir}/_merged_result.txt" ]]; then
            successes="$(awk '/^Successes:/ {print $2}' "${result_dir}/_merged_result.txt")"
            success_rate="$(awk '/^Success rate:/ {print $3}' "${result_dir}/_merged_result.txt")"
            summary_rows+=("${checkpoint_id}|${task_label}|${task_name}|${successes}|${success_rate}|${result_dir}")
            echo "Completed: checkpoint=${checkpoint_id}, ${task_label}, ${successes}, rate=${success_rate}" \
                | tee -a "${batch_log}"
        else
            failed=1
            summary_rows+=("${checkpoint_id}|${task_label}|${task_name}|FAILED|FAILED|${result_dir}")
            echo "FAILED: checkpoint=${checkpoint_id}, ${task_label}, exit_status=${status}" \
                | tee -a "${batch_log}" >&2
        fi
    done
done

echo | tee -a "${batch_log}"
echo "================ Batch Evaluation Summary ================" | tee -a "${batch_log}"
printf '%-10s %-8s %-22s %-12s %-12s\n' "Checkpoint" "Task" "Task name" "Successes" "Rate" \
    | tee -a "${batch_log}"
for row in "${summary_rows[@]}"; do
    IFS='|' read -r checkpoint_id task_label task_name successes success_rate result_dir <<< "${row}"
    printf '%-10s %-8s %-22s %-12s %-12s\n' \
        "${checkpoint_id}" "${task_label}" "${task_name}" "${successes}" "${success_rate}" \
        | tee -a "${batch_log}"
    echo "  ${result_dir}" >> "${batch_log}"
done
echo "Combined terminal log: ${batch_log}" | tee -a "${batch_log}"

exit "${failed}"

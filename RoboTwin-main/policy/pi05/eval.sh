#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
policy_name=pi05
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}
checkpoint_id=${7:-10000}
fixed_instruction="${8:-}"
cl_task_id="${9:-}"
eval_repo_id="${10:-}"
eval_episodes="${11:-}"
eval_worker_id="${12:-}"
eval_num_workers="${13:-}"
eval_result_tag="${14:-}"
eval_seed_start="${15:-}"
eval_seed_stride="${16:-}"
extra_args=()
if [ -n "${cl_task_id}" ]; then
    extra_args+=(--cl_task_id "${cl_task_id}")
fi
if [ -n "${eval_repo_id}" ]; then
    extra_args+=(--eval_repo_id "${eval_repo_id}")
fi
if [ -n "${eval_episodes}" ]; then
    extra_args+=(--eval_episodes "${eval_episodes}")
fi
if [ -n "${eval_worker_id}" ]; then
    extra_args+=(--eval_worker_id "${eval_worker_id}")
fi
if [ -n "${eval_num_workers}" ]; then
    extra_args+=(--eval_num_workers "${eval_num_workers}")
fi
if [ -n "${eval_result_tag}" ]; then
    extra_args+=(--eval_result_tag "${eval_result_tag}")
fi
if [ -n "${eval_seed_start}" ]; then
    extra_args+=(--eval_seed_start "${eval_seed_start}")
fi
if [ -n "${eval_seed_stride}" ]; then
    extra_args+=(--eval_seed_stride "${eval_seed_stride}")
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

if [ -f "${script_dir}/.venv/bin/activate" ]; then
    source "${script_dir}/.venv/bin/activate"
fi
cd "${script_dir}/../.." # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --checkpoint_id "${checkpoint_id}" \
    --fixed_instruction "${fixed_instruction}" \
    "${extra_args[@]}"

#!/bin/bash
# =============================================================================
# eval_multi_gpu.sh — 多卡并行评估（按卡分配 episode，师兄建议的 batch 按卡分）
#
# 用法:
#   bash eval_multi_gpu.sh <task_name> <task_config> <checkpoint_path> <seed> \
#                          <gpus> <unnorm_key> [eval_task_id] [episodes] [tag]
#
# 参数:
#   task_name        任务名, e.g. handover_mic
#   task_config      task_config/*.yml 名, e.g. demo_clean
#   checkpoint_path  checkpoint 目录
#   seed             seed 参数 (base_seed = 100000 * (1+seed))
#   gpus             物理卡列表, 逗号分隔, 每个条目起一个 worker（不要带空格）:
#                    "4,5"     -> 2 个 worker, 各占一张卡 (推荐, 结果直接 2x)
#                    "4,4,5,5" -> 4 个 worker, 每张卡 2 个 (CPU 核够时可再翻倍)
#   unnorm_key       dataset_statistics 里的 key, e.g. aloha_handover_mic_clean
#   eval_task_id     0=当前权重, 1=A bank, 2=B bank, 3=C, 4=D
#   episodes         总评估 episode 数 (默认 50, 与单卡一致), 自动均分到各 worker
#   tag              结果目录 tag (默认时间戳; 建议传固定 tag 如 v39_taskA_g2)
#
# 说明:
#   - 各 worker 的 seed 自动错开 (worker i 从 base_seed+i 开始, 步长=worker 数),
#     与单卡评估覆盖完全相同的 seed 集合, 成功率可直接对比。
#   - 每个 worker 独立进程、独立模型副本, 日志写在
#       eval_result/<task>/<policy>/<config>/<ckpt>/<tag>/worker_XX/eval.log
#     实时进度: tail -f 该日志。
#   - 全部结束后自动合并各 worker 结果, 生成 _merged_result.json / .txt。
#   - 评估前记得 export OPENVLA_BASE_PATH=... (deploy_policy.py 需要)。
# =============================================================================

set -u

if [ $# -lt 6 ]; then
    echo "用法: bash eval_multi_gpu.sh <task_name> <task_config> <checkpoint_path> <seed> <gpus> <unnorm_key> [eval_task_id] [episodes] [tag]"
    echo "示例: bash policy/openvla-oft/eval_multi_gpu.sh handover_mic demo_clean \$CKPT 0 4,5 aloha_handover_mic_clean 1 50 v39_taskA_g2"
    exit 1
fi

policy_name=openvla-oft
task_name=${1}
task_config=${2}
checkpoint_path=${3}
seed=${4}
gpus=${5// /}  # 去掉空格, 防手滑
unnorm_key=${6}
eval_task_id=${7:-0}
episodes=${8:-50}
result_tag=${9:-$(date +%Y%m%d_%H%M%S)}

if [ ! -d "$checkpoint_path" ]; then
    echo "ERROR: checkpoint 目录不存在: $checkpoint_path"
    exit 1
fi

IFS=',' read -ra GPU_ARR <<< "$gpus"
num_workers=${#GPU_ARR[@]}
per_worker=$(( (episodes + num_workers - 1) / num_workers ))  # 向上取整, 保证总 episode 数 >= episodes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."  # 到 RoboTwin-main 根目录

# 快速环境自检: prismatic 必须在当前 python 环境里 (openvla conda env)
if ! python -c "import prismatic" 2>/dev/null; then
    echo "[FAIL] 当前 python 环境里找不到 prismatic —— 请先: cd /mnt/data/pengshengdi && source server_env.sh"
    echo "       (确认输出里有 [OK] conda env 和 prismatic OK 两行)"
    exit 1
fi

unset CUDA_VISIBLE_DEVICES  # 每个 worker 各自指定, 不继承外层

echo "======================================================================"
echo "== 多卡并行评估: $num_workers 个 worker on GPUs [$gpus]"
echo "== 总 episode: $episodes (每 worker $per_worker), eval_task_id=$eval_task_id"
echo "== task=$task_name, config=$task_config, ckpt=$checkpoint_path"
echo "== tag=$result_tag"
echo "======================================================================"

pids=()
trap 'echo "Interrupted! killing workers..."; for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null; done; exit 130' INT TERM

for i in $(seq 0 $((num_workers - 1))); do
    gpu=${GPU_ARR[$i]}
    worker_dir="eval_result/${task_name}/${policy_name}/${task_config}/${checkpoint_path}/${result_tag}/worker_$(printf %02d "$i")"
    mkdir -p "$worker_dir"
    log_file="$worker_dir/eval.log"
    echo ">> worker $i on GPU $gpu, log: $log_file"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning \
    python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
        --overrides \
        --task_name ${task_name} \
        --task_config ${task_config} \
        --checkpoint_path ${checkpoint_path} \
        --ckpt_setting ${checkpoint_path} \
        --seed ${seed} \
        --policy_name ${policy_name} \
        --unnorm_key ${unnorm_key} \
        --eval_task_id ${eval_task_id} \
        --eval_video_log False \
        --clear_cache_freq 50 \
        --eval_num_workers ${num_workers} \
        --eval_worker_id ${i} \
        --eval_episodes ${per_worker} \
        --eval_result_tag ${result_tag} \
        > "$log_file" 2>&1 &
    pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done

run_dir="eval_result/${task_name}/${policy_name}/${task_config}/${checkpoint_path}/${result_tag}"
echo ""
echo "== 全部 worker 结束 (fail=$fail) =="
for i in $(seq 0 $((num_workers - 1))); do
    log_file="$run_dir/worker_$(printf %02d "$i")/eval.log"
    echo "--- worker $i (GPU ${GPU_ARR[$i]}) 末尾 ---"
    grep -E "Success rate|Error|Traceback" "$log_file" | tail -3 || true
done

if [ "$fail" -eq 0 ]; then
    echo ""
    echo "== 合并结果 =="
    python script/merge_eval_results.py --run-dir "$run_dir"
    echo ""
    echo "结果目录: $run_dir"
    echo "合并结果: $run_dir/_merged_result.json (+ .txt)"
else
    echo ""
    echo "!! 有 worker 失败 (fail=$fail), 跳过合并。请查看上面的日志。"
    exit 1
fi

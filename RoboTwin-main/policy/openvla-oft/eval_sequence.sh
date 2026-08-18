#!/bin/bash
# =============================================================================
# eval_sequence.sh — 同一个 checkpoint 顺序评估多个任务
# （串行连续执行, 每任务实时进度, 全部结束后打印成功率汇总）
#
# 用法:
#   bash eval_sequence.sh <checkpoint_path> <gpus> [episodes] [tag] [tasks...]
#
#   参数:
#     checkpoint_path  checkpoint 目录 (如 $LOGS_ROOT/rt_v39_taskC--40000_chkpt)
#     gpus             物理卡列表, 逗号分隔 (如 4,5; 每个条目一个 worker)
#     episodes         每个任务的评估 episode 数, 默认 50
#     tag              结果目录 tag 前缀, 默认 eval_seq_<时间戳>; 每任务自动加 _taskX 后缀
#     tasks...         任务子集, 默认 "A B C D"; 可任意组合/顺序
#                      A=handover_mic  B=grab_roller  C=stack_bowls_two  D=open_laptop
#
#   示例:
#     bash eval_sequence.sh $LOGS_ROOT/rt_v39_taskC--40000_chkpt 4,5 50 v39C A B C
#     bash eval_sequence.sh $LOGS_ROOT/rt_v39_taskD--40000_chkpt 4,5 50 v39D A B C D
#
#   说明:
#     - 任务间串行执行 (一个 checkpoint 的所有任务连在一起不断), 每任务内部双卡并行
#     - 每任务实时显示: [wX|GPU Y] Success! / Fail! / Success rate: X/25 => Z%
#     - 某任务失败不中断, 汇总里标 FAILED, 继续下一个任务
#     - 全部结束后打印汇总, 并存到 eval_result/<tag>_summary.txt
# =============================================================================

set -u -o pipefail

if [ $# -lt 2 ]; then
    echo "用法: bash eval_sequence.sh <checkpoint_path> <gpus> [episodes] [tag] [tasks...]"
    echo "示例: bash eval_sequence.sh \$LOGS_ROOT/rt_v39_taskC--40000_chkpt 4,5 50 v39C A B C"
    exit 1
fi

checkpoint_path=${1}
gpus=${2// /}
episodes=${3:-50}
tag=${4:-eval_seq_$(date +%Y%m%d_%H%M%S)}
shift 4
if [ $# -ge 1 ]; then
    TASKS=("$@")
else
    TASKS=(A B C D)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."   # 到 RoboTwin-main 根

[ -d "$checkpoint_path" ] || { echo "[FAIL] checkpoint 目录不存在: $checkpoint_path"; exit 1; }
mkdir -p eval_result

# 任务表: 名称 -> "task_name unnorm_key eval_task_id"
task_info() {
    case "$1" in
        A) echo "handover_mic aloha_handover_mic_clean 1" ;;
        B) echo "grab_roller aloha_grab_roller_clean 2" ;;
        C) echo "stack_bowls_two aloha_stack_bowls_two_clean 3" ;;
        D) echo "open_laptop aloha_open_laptop_clean 4" ;;
        *) echo "" ;;
    esac
}

echo "======================================================================"
echo "== 顺序评估: ckpt=$checkpoint_path"
echo "== 任务: ${TASKS[*]} | 每任务 $episodes episode | GPUs [$gpus]"
echo "======================================================================"

results=()
for t in "${TASKS[@]}"; do
    info=$(task_info "$t")
    if [ -z "$info" ]; then
        echo "[WARN] 未知任务 '$t', 跳过"
        continue
    fi
    read -r tname unnorm tid <<< "$info"
    tag_full="${tag}_task${t}"
    eval_log="eval_result/${tag}_task${t}.log"
    echo ""
    echo "==================== Task $t : $tname (eval_task_id=$tid) ===================="
    bash policy/openvla-oft/eval_multi_gpu.sh "$tname" demo_clean "$checkpoint_path" 0 \
        "$gpus" "$unnorm" "$tid" "$episodes" "$tag_full" 2>&1 | tee "$eval_log"
    rc=${PIPESTATUS[0]}
    rate=$(grep -E "Merged success rate" "$eval_log" | tail -1)
    if [ $rc -ne 0 ] || [ -z "$rate" ]; then
        results+=("Task $t ($tname): FAILED (rc=$rc)")
    else
        results+=("Task $t ($tname): $rate")
    fi
done

summary="eval_result/${tag}_summary.txt"
echo ""
echo "==================== 成功率汇总: $checkpoint_path ===================="
{
    echo "checkpoint: $checkpoint_path"
    echo "tasks: ${TASKS[*]} | episodes: $episodes | gpus: $gpus"
    echo ""
    for line in "${results[@]}"; do
        echo "$line"
    done
} | tee "$summary"
echo ""
echo "汇总已存: $summary"

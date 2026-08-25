#!/bin/bash
# =============================================================================
# scan_film_scope.sh — FiLM 部分恢复 scope 扫描（4 卡并行加速）
#
# 背景: γ 插值实证 C 对 FiLM 是"全或无"(0.1~0.8 全 0)。scope 扫描换一种粒度:
#       整层替换(选中层完全恢复任务FiLM, 未选中层保持当前漂移FiLM),
#       看能否找到 A/C 存活的"中间态"(残留 0.2-0.5)。
#
# 用法:
#   bash scan_film_scope.sh <ckpt> <task_name> <unnorm_key> <eval_task_id> [gpus] [episodes]
# 示例:
#   bash scan_film_scope.sh $LOGS_ROOT/rt_v39b2_taskD--40000_chkpt \
#       stack_bowls_two aloha_stack_bowls_two_clean 3 4,5,6,7 10
#
# 扫描范围: siglip / dinov2 / k5 / k10 / k20
#   (all 的 γ=1 高保留结果已有, 不重复扫)
# 并行策略: 前 4 个 scope 各占一张卡同时跑, 第 5 个(k20)收尾
# 输出: 每个 scope 一行 Merged success rate; 完整日志在 /tmp/scope_scan/
# =============================================================================

set -u

if [ $# -lt 4 ]; then
    echo "用法: bash scan_film_scope.sh <ckpt> <task_name> <unnorm_key> <eval_task_id> [gpus] [episodes]"
    exit 1
fi

CKPT=${1}
TASK=${2}
UNNORM=${3}
EID=${4}
GPUS=${5:-4,5,6,7}
EPISODES=${6:-10}

[ -d "$CKPT" ] || { echo "[FAIL] checkpoint 不存在: $CKPT"; exit 1; }
mkdir -p /tmp/scope_scan

IFS=',' read -ra GPU_ARR <<< "$GPUS"
if [ ${#GPU_ARR[@]} -lt 4 ]; then
    echo "[WARN] 只有 ${#GPU_ARR[@]} 张卡, 并行度下降, 但会跑完"
fi

SCOPES=(siglip dinov2 k5 k10)
EXTRA=(k20)
pids=()

run_scope() {  # $1=scope  $2=gpu
    local s=$1 g=$2
    echo ">> [$(date +%H:%M:%S)] scope=$s on GPU $g (log: /tmp/scope_scan/$s.log)"
    bash RoboTwin-main/policy/openvla-oft/eval_multi_gpu.sh \
        "$TASK" demo_clean "$CKPT" 0 "$g" \
        "$UNNORM" "$EID" "$EPISODES" "scope_${s}" 1.0 "$s" \
        > "/tmp/scope_scan/$s.log" 2>&1
}

# 第一批: 4 个 scope 并行, 各占一张卡
for i in "${!SCOPES[@]}"; do
    run_scope "${SCOPES[$i]}" "${GPU_ARR[$i]}" &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

# 第二批: k20 收尾 (用第一张卡)
run_scope "${EXTRA[0]}" "${GPU_ARR[0]}"

# 汇总
echo ""
echo "================ FiLM scope 扫描结果 ($TASK, eval_task_id=$EID, episodes=$EPISODES) ================"
for s in "${SCOPES[@]}" "${EXTRA[@]}"; do
    rate=$(grep "Merged success rate" "/tmp/scope_scan/$s.log" | tail -1)
    echo "scope=$s: ${rate:-<未完成/失败>}"
done
echo ""
echo "完整日志: /tmp/scope_scan/*.log"

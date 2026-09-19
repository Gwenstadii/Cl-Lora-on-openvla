#!/usr/bin/env bash
# =============================================================================
# run_v39_bank_scope_sweep.sh — bank "部分恢复"扫描（纯评估，零训练）
#
# 目的: 量化 "恢复多少任务记忆 → 保留多少成功率"，并区分两种语义:
#   ① 不加 @zero: 未恢复的参数**保持当前 ckpt 的值**（= 新任务的适配器）
#      ⇒ 语义是"半个网络在做新任务、半个在做旧任务" → 实测直接崩（见 §10.10）
#   ② 加 @zero  : 未恢复的部分置零（B=0 ⇒ 该分支无贡献）= 干净的"记忆缺失"语义
#      ⇒ 这才是能画出连续曲线的家族
#
# 用法:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   # 在某个 ckpt 上扫一组 scope，评 A(+B/C)，32ep，4卡8worker
#   CKPT=$LOGS_ROOT/rt_v39b9_n6_taskB--40000_chkpt TAGP=n6 \
#     bash run_v39_bank_scope_sweep.sh
#   # 自定义 scope 列表（引号包住，空格分隔）
#   CKPT=... TAGP=n6 SCOPES="action_head@zero action_head+layers:28-31@zero all" \
#     bash run_v39_bank_scope_sweep.sh
#
# 环境变量:
#   CKPT        必填，被评估的 ckpt
#   TAGP        结果 tag 前缀（默认 scope_sweep）
#   SCOPES      空格分隔的 scope 列表（默认见下）
#   TASKS       评估任务（默认 "A B"）
#   EPISODES    每任务 episode 数（默认 32）
#   GPUS_EVAL   评估 worker 卡列表（默认 4,4,5,5,6,6,7,7）
#   FILM_GAMMA_G 评估端 γ（默认 0，与主表口径一致）
# =============================================================================

set -u

: "${LOGS_ROOT:?请先 source server_env.sh（LOGS_ROOT 未设置）}"

CKPT="${CKPT:?用法: CKPT=<ckpt路径> [TAGP=xx] bash run_v39_bank_scope_sweep.sh}"
TAGP="${TAGP:-scope_sweep}"
SCOPES="${SCOPES:-all action_head@zero action_head+layers:28-31@zero action_head+layers:26-31@zero action_head+layers:24-31@zero action_head}"
TASKS="${TASKS:-A B}"
EPISODES="${EPISODES:-32}"
GPUS_EVAL="${GPUS_EVAL:-4,4,5,5,6,6,7,7}"
FILM_GAMMA_G="${FILM_GAMMA_G:-0}"
ROBOTWIN="/mnt/data/pengshengdi/RoboTwin-main"
EVAL_OUT="$ROBOTWIN/eval_result"

[ -d "$CKPT" ] || { echo "[FAIL] ckpt 不存在: $CKPT"; exit 1; }
cd "$ROBOTWIN" || exit 1

if command -v nvidia-smi >/dev/null 2>&1; then
    used=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    echo "[INFO] 当前 GPU 上已有 $used 个计算进程（>0 时注意显存；不够就设 GPUS_EVAL=4,4,5,5）"
fi

sanitize() { echo "$1" | tr ':+,' '___' | tr -cd 'a-zA-Z0-9_'; }

echo "============ bank 部分恢复扫描 ============"
echo "ckpt   : $CKPT"
echo "scopes : $SCOPES"
echo "tasks  : $TASKS | episodes: $EPISODES | workers: $GPUS_EVAL | γ=$FILM_GAMMA_G"
echo ""

for sc in $SCOPES; do
    tag="${TAGP}_$(sanitize "$sc")"
    echo "############################################################"
    echo "#### scope = $sc   →   tag = $tag"
    echo "############################################################"
    BANK_RESTORE_SCOPE="$sc" FILM_GAMMA="$FILM_GAMMA_G" \
        bash policy/openvla-oft/eval_sequence.sh "$CKPT" "$GPUS_EVAL" "$EPISODES" "$tag" $TASKS
    echo "-- 生效核对（应含 restore_scope='$sc'）--"
    for t in $TASKS; do
        lf="$EVAL_OUT/${tag}_task${t}.log"
        [ -f "$lf" ] && grep -m1 -E "部分恢复|restore_scope=" "$lf" | sed 's/^/   /'
    done
done

echo ""
echo "======================== 扫描汇总 ========================"
printf "%-46s %-12s %-12s\n" "scope" "TaskA" "TaskB"
for sc in $SCOPES; do
    tag="${TAGP}_$(sanitize "$sc")"
    a=$(grep "Merged success rate" "$EVAL_OUT/${tag}_taskA.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" || true)
    b=$(grep "Merged success rate" "$EVAL_OUT/${tag}_taskB.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" || true)
    printf "%-46s %-12s %-12s\n" "$sc" "${a:-无数据}" "${b:-无数据}"
done
echo ""
echo "判读:"
echo "  · 不加 @zero 的一族 = 未恢复部分仍是新任务的 B ⇒ 半A半B 身份冲突 → 预期全崩"
echo "  · @zero 一族 = 记忆缺失语义 ⇒ 若出现 0.3~0.7 中间值, 即拿到连续的记忆-保留率曲线"
echo "  · all 应复现已知满恢复结果（n6 = 0.9643）"

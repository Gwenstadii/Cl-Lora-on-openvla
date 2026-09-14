#!/usr/bin/env bash
# =============================================================================
# run_v39_prototype_2x2.sh — 原型回放 2×2 补齐（两条线串行跑完）
#
#   ① v39v  : 冻结 FiLM + specific-A 冻结 + 原型回放 + 无 KD  ← 方法本体定稿配置（此前从未跑过）
#   ② v39r3 : 冻结 FiLM + specific-A 漂移 + 原型回放 + 无 KD  ← 回放能否替代 A 冻结（此前从未跑过）
#
# 补齐后的完整格表（冻结 FiLM + 无 KD）:
#                     | specific-A 冻结        | specific-A 漂移
#     无回放          | v39b5（未跑, 可选）    | v39b4 = 0-0.57-0-0.85 ✅已知
#     原型回放        | **v39v ← 本脚本①**    | **v39r3 ← 本脚本②**（vs b4 = 干净单变量）
#     uniform 回放    | v39u（在跑）           | v39w（可选，未跑）
#
# 干净单变量对照:
#   v39r3 vs v39b4 : 唯一差异 = 加原型回放
#   v39r3 vs v39v  : 唯一差异 = freeze_specific_a
#   v39v  vs v39u  : 唯一差异 = 回放选帧规则（prototype vs uniform）
#
# 用法（tmux 前台, 预计两条线共 6 段 40k steps 训练）:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   tmux new -s train2x2
#   bash run_v39_prototype_2x2.sh 2>&1 | tee train_v39_2x2.log
#
#   只跑其中一条: ARMS="v39r3" bash run_v39_prototype_2x2.sh
#   中途 Ctrl-C 后重跑: 已完成的 ckpt 会自动 SKIP（脚本幂等）
#
# 单条脚本: bash run_v39v_prototype_replay.sh [FREEZE_SPECIFIC_A=False]
# =============================================================================

set -u

: "${LOGS_ROOT:?请先 source server_env.sh（LOGS_ROOT 未设置）}"
: "${VLA_PATH:?请先 source server_env.sh（VLA_PATH 未设置）}"

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARMS="${ARMS:-v39v v39r3}"
LOGDIR="${LOGDIR:-/mnt/data/pengshengdi}"
SUM_DIR="/mnt/data/pengshengdi/RoboTwin-main/eval_result"

[ -n "${LOGS_ROOT:-}" ] || { echo "[FAIL] LOGS_ROOT 未设置 —— 先 source server_env.sh"; exit 1; }
[ -n "${VLA_PATH:-}" ]  || { echo "[FAIL] VLA_PATH 未设置 —— 先 source server_env.sh"; exit 1; }

echo "============ 原型回放 2×2 补齐: [$ARMS] ============"
echo "共享: A ckpt = $LOGS_ROOT/rt_v39_taskA--30000_chkpt（复用）"
echo "共享: 原型 buffers = $LOGS_ROOT/replay_buffers（复用, 与 v39r2d 同源）"

for arm in $ARMS; do
    case "$arm" in
        v39v)  FSA=True ;;
        v39r3) FSA=False ;;
        *) echo "[WARN] 未知支线 '$arm'（仅支持 v39v / v39r3）, 跳过"; continue ;;
    esac
    echo ""
    echo "############################################################"
    echo "######## 开始 $arm (FREEZE_SPECIFIC_A=$FSA) ########"
    echo "############################################################"
    FREEZE_SPECIFIC_A="$FSA" bash "$SELF_DIR/run_v39v_prototype_replay.sh" \
        2>&1 | tee "$LOGDIR/train_${arm}.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 2 ]; then
        echo "[STOP] $arm 停在 B 自评门禁（rc=2）—— 后续支线不再继续。"
        echo "       排查后重跑: ARMS=\"$arm\" bash run_v39_prototype_2x2.sh"
        exit 2
    elif [ "$rc" -ne 0 ]; then
        echo "[FAIL] $arm 执行失败 (rc=$rc), 见 $LOGDIR/train_${arm}.log"
        exit "$rc"
    fi
    echo "[OK] $arm 完成"
done

echo ""
echo "================ 2×2 结果汇总 ================"
for arm in $ARMS; do
    f="$SUM_DIR/${arm}D_summary.txt"
    echo "---- $arm ----"
    [ -f "$f" ] && cat "$f" || echo "  [WARN] 缺汇总: $f"
done
echo ""
echo "参照: v39b4 (A漂移+无回放) = 0-0.57-0-0.85 | v39r2d (脏版原型回放+KD) = 0.62-0.88-0.88-0.80"

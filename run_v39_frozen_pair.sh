#!/usr/bin/env bash
# =============================================================================
# run_v39_frozen_pair.sh — 叙事②"冻结 FiLM"下的回放单变量对（一条命令跑完两臂）
#
#   ① b5  : 全程冻结 FiLM + specific-A 冻结 + **无回放** + proprio 修复   （run_v39b5_baseline_BCD.sh）
#   ② v39v: 全程冻结 FiLM + specific-A 冻结 + **原型回放**（无 KD）        （run_v39v_prototype_replay.sh）
#
#   ⇒ 两臂**唯一差异 = 回放开关**（冻结 FiLM / A 冻结 / 步数 / 起点 / buffer 全同）
#   ⇒ 直接量化"回放的真实边际价值"（叙事②：结构隔离为主 + 回放增强 X）
#
# 与叙事①的分工:
#   叙事① 回放卖点 → 遗忘源 = FiLM 漂移 0.2×，配对为 b3 ↔ v39r5（v39r5 优先跑）
#   叙事② 结构卖点 → 无遗忘源（全冻结），配对为 b5 ↔ v39v（本脚本）
#
# 预期与判读:
#   · b5 已高（预测 A/B/C ≥0.6，参照 v39f 的 A/B≈0.7）⇒ CL-LoRA 结构隔离本身很强
#   · v39v - b5 = 回放的边际价值；若差距小 ⇒ 如实报告"回放为补充而非必需"
#
# 用法（tmux 前台, 两条线共 6 段 40k 训练）:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   tmux new -s trainFrz
#   bash run_v39_frozen_pair.sh 2>&1 | tee train_v39_frozen_pair.log
#
#   只跑一臂: 直接调各自脚本（b5: bash run_v39b5_baseline_BCD.sh；v39v: bash run_v39v_prototype_replay.sh）
# =============================================================================

set -u

: "${LOGS_ROOT:?请先 source server_env.sh（LOGS_ROOT 未设置）}"
: "${VLA_PATH:?请先 source server_env.sh（VLA_PATH 未设置）}"

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="${LOGDIR:-/mnt/data/pengshengdi}"
SUM_DIR="/mnt/data/pengshengdi/RoboTwin-main/eval_result"
ARMS="${ARMS:-b5 v39v}"

echo "============ 冻结 FiLM 单变量对: [$ARMS] ============"
echo "共享: A ckpt = $LOGS_ROOT/rt_v39_taskA--30000_chkpt | buffers = $LOGS_ROOT/replay_buffers"

for arm in $ARMS; do
    echo ""
    echo "############################################################"
    echo "######## 开始 $arm ########"
    echo "############################################################"
    case "$arm" in
        b5)   bash "$SELF_DIR/run_v39b5_baseline_BCD.sh"        2>&1 | tee "$LOGDIR/train_v39b5.log" ;;
        v39v) FREEZE_SPECIFIC_A=True bash "$SELF_DIR/run_v39v_prototype_replay.sh" \
                                                                2>&1 | tee "$LOGDIR/train_v39v.log" ;;
        v39r3) FREEZE_SPECIFIC_A=False bash "$SELF_DIR/run_v39v_prototype_replay.sh" \
                                                                2>&1 | tee "$LOGDIR/train_v39r3.log" ;;
        *) echo "[WARN] 未知支线 '$arm'（支持 b5 / v39v / v39r3），跳过"; continue ;;
    esac
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 2 ]; then
        echo "[STOP] $arm 停在 B 自评门禁 (rc=2)；排查后重跑本脚本（已完成 ckpt 会 SKIP）"
        exit 2
    elif [ "$rc" -ne 0 ]; then
        echo "[FAIL] $arm 失败 (rc=$rc)，见 $LOGDIR/train_${arm}.log"
        exit "$rc"
    fi
    echo "[OK] $arm 完成"
done

echo ""
echo "================ 结果汇总 ================"
for tag in v39b5D v39vD; do
    f="$SUM_DIR/${tag}_summary.txt"
    echo "---- $tag ----"
    [ -f "$f" ] && cat "$f" || echo "  [WARN] 缺汇总: $f（可跑: bash show_eval_rates.sh $tag）"
done
echo ""
echo "判读: v39v - b5 = 回放的边际价值；两臂唯一差异 = 回放开关。"
echo "参照: v39r2d（冻结FiLM+回放+KD）= 0.62-0.88-0.88-0.80 | b3（漂移FiLM 0.2×+无回放）= 0.26-0.88-0.72-0.82"

#!/usr/bin/env bash
# =============================================================================
# show_eval_rates.sh — 打印某条支线的各任务评估成功率（从 eval_result/ 日志）
#
# 用法:
#   bash show_eval_rates.sh <tag> [训练日志]
#     tag     评估时 eval_sequence.sh 的 tag, 例如 v39r3D / v39vD / v39b3D / v39uD
#     训练日志 可选, 例如 /mnt/data/pengshengdi/train_v39r3.log（评估被中断时兜底）
#
# 例:
#   bash show_eval_rates.sh v39r3D
#   bash show_eval_rates.sh v39r3D /mnt/data/pengshengdi/train_v39r3.log
#   bash show_eval_rates.sh v39b3D_g1          # γ=1 重评那次的 tag
#
# 说明: eval_result/<tag>_task<X>.log 由 eval_sequence.sh 逐任务 tee 生成；
#       <tag>_summary.txt 只在四个任务全部跑完后才写。
# =============================================================================

set -u

TAG="${1:-}"
[ -n "$TAG" ] || { echo "用法: bash show_eval_rates.sh <tag> [训练日志]"; exit 1; }
TRAIN_LOG="${2:-}"
ROOT="/mnt/data/pengshengdi/RoboTwin-main"

cd "$ROOT" 2>/dev/null || { echo "[FAIL] 目录不存在: $ROOT"; exit 1; }

echo "======================================================================"
echo "== $TAG 各任务成功率   ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "======================================================================"

found=0
for t in A B C D; do
    f="eval_result/${TAG}_task${t}.log"
    if [ -f "$f" ]; then
        rate=$(grep "Merged success rate" "$f" | tail -1 || true)
        if [ -n "$rate" ]; then
            printf "  Task %-2s : %s\n" "$t" "$rate"
            found=1
        else
            # 未跑完: 打印各 worker 最新进度
            printf "  Task %-2s : （未完成，最新进度 ↓）\n" "$t"
            grep -E "Success rate:" "$f" | tail -3 | sed 's/^/           /'
        fi
    else
        printf "  Task %-2s : （无日志: %s）\n" "$t" "$f"
    fi
done

if [ -f "eval_result/${TAG}_summary.txt" ]; then
    echo ""
    echo "---- summary 文件: eval_result/${TAG}_summary.txt ----"
    cat "eval_result/${TAG}_summary.txt"
fi

if [ -n "$TRAIN_LOG" ] && [ -f "$TRAIN_LOG" ]; then
    echo ""
    echo "---- 从训练日志兜底提取: $TRAIN_LOG ----"
    grep -E "Task [ABCD] :|Merged success rate" "$TRAIN_LOG" | tail -12
fi

echo ""
echo "提示: 任务顺序 A→B→C→D 串行；A/B 已出、C/D 未出属正常（中断即停）。"
[ "$found" = "1" ] || echo "[WARN] 一个成功率都没解析到：确认 tag 拼写，或 ls eval_result/ 看实际文件名。"

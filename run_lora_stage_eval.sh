#!/usr/bin/env bash
# =============================================================================
# run_lora_stage_eval.sh — 普通 LoRA 基线的"逐阶段累积评估"（填 v41.md §5 的 LoRA 表）
#
# 目的: 普通 LoRA 没有 bank，旧任务成绩就是"当前权重直接评"（eval_task_id=0，灾难性遗忘口径）。
#   要在 4 个阶段 ckpt 上分别评估"到目前为止见过的所有任务"：
#     A  ckpt (30000) → 评 A
#     B  ckpt (40000) → 评 A B
#     C  ckpt (40000) → 评 A B C
#     D  ckpt (40000) → 评 A B C D
#   ⇒ 得到一条完整的遗忘曲线（每一列在 4 个阶段的读数），与 v41/v41_r 同口径（50ep、γ=0）。
#
# 用法:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   tmux new -s evalLoRA
#   bash run_lora_stage_eval.sh 2>&1 | tee train_lora_stage_eval.log
#
# 环境变量:
#   GPUS_EVAL   评估 worker 卡列表（默认 4,4,5,5,6,6,7,7；卡紧就 2,2,3,3）
#   EPISODES    每任务 episode（默认 50，与 v41 口径一致）
#   STAGES      要评的阶段（默认 "A B C D"）；只跑某几个就改这里
# =============================================================================

set -u

: "${LOGS_ROOT:?请先 source server_env.sh（LOGS_ROOT 未设置）}"

EVAL_SH="/mnt/data/pengshengdi/RoboTwin-main/policy/openvla-oft/eval_multi_gpu.sh"
GPUS_EVAL="${GPUS_EVAL:-4,4,5,5,6,6,7,7}"
EPISODES="${EPISODES:-50}"
STAGES="${STAGES:-A B C D}"
OUT_DIR="${OUT_DIR:-/mnt/data/pengshengdi}"

[ -f "$EVAL_SH" ] || { echo "[FAIL] 找不到 $EVAL_SH"; exit 1; }

# 阶段 → ckpt 目录
ckpt_of() {
    case "$1" in
        A) echo "$LOGS_ROOT/rt_lora_taskA--30000_chkpt" ;;
        B) echo "$LOGS_ROOT/rt_lora_taskB--40000_chkpt" ;;
        C) echo "$LOGS_ROOT/rt_lora_taskC--40000_chkpt" ;;
        D) echo "$LOGS_ROOT/rt_lora_taskD--40000_chkpt" ;;
        *) echo "" ;;
    esac
}

# 任务 → "task_name unnorm_key"
task_info() {
    case "$1" in
        A) echo "handover_mic aloha_handover_mic_clean" ;;
        B) echo "grab_roller aloha_grab_roller_clean" ;;
        C) echo "stack_bowls_two aloha_stack_bowls_two_clean" ;;
        D) echo "open_laptop aloha_open_laptop_clean" ;;
        *) echo "" ;;
    esac
}

parse_rate() {  # $1=log → 0-1 成功率（合并行缺失时用各 worker 均值），或空
    local merged
    merged=$(grep "Merged success rate" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" || true)
    if [ -n "$merged" ]; then echo "$merged"; return; fi
    python - "$1" <<'PY'
import re, sys
vals = []
try:
    for ln in open(sys.argv[1], errors="ignore"):
        m = re.search(r"Success rate:\s*(\d+)/(\d+)\s*=>\s*([\d.]+)%", ln)
        if m:
            vals.append(float(m.group(3)) / 100.0)
except FileNotFoundError:
    pass
print(f"{sum(vals)/len(vals):.4f}" if vals else "")
PY
}

echo "======================================================================"
echo "== 普通 LoRA 逐阶段累积评估（灾难性遗忘口径: eval_task_id=0）"
echo "== 阶段: $STAGES | episodes: $EPISODES | workers: $GPUS_EVAL"
echo "======================================================================"

for st in $STAGES; do
    CKPT="$(ckpt_of "$st")"
    echo ""
    echo "################ 阶段 $st : $(basename "$CKPT") ################"
    if [ ! -d "$CKPT" ]; then
        echo "[SKIP] ckpt 不存在: $CKPT"
        continue
    fi
    # 该阶段之前（含自己）的所有任务
    TASKS=""
    for t in A B C D; do
        TASKS="$TASKS $t"
        [ "$t" = "$st" ] && break
    done
    echo "    评估任务:$TASKS"
    for t in $TASKS; do
        info=$(task_info "$t")
        read -r tname unnorm <<< "$info"
        tag="lora${st}_eval_${t}"
        log="$OUT_DIR/lora_eval_${st}_${t}.log"
        echo ""
        echo "---- [$st ckpt] Task $t : $tname (eval_task_id=0, ${EPISODES}ep) ----"
        bash "$EVAL_SH" "$tname" demo_clean "$CKPT" 0 "$GPUS_EVAL" "$unnorm" 0 "$EPISODES" "$tag" 0 \
            2>&1 | tee "$log" | grep -v "svulkan2.*error"
        rc=${PIPESTATUS[0]}
        rate=$(parse_rate "$log")
        if [ $rc -ne 0 ] || [ -z "$rate" ]; then
            echo "[WARN] $st/$t 未取到成功率 (rc=$rc) → 看 $log"
        else
            echo "[OK] $st ckpt / Task $t = $rate"
        fi
    done
done

echo ""
echo "==================== 汇总表（行=阶段 ckpt, 列=任务；可直接填进 v41.md §5.2）===================="
printf "%-10s %-12s %-12s %-12s %-12s\n" "ckpt" "Task A" "Task B" "Task C" "Task D"
for st in A B C D; do
    line="$st"
    for t in A B C D; do
        r=$(parse_rate "$OUT_DIR/lora_eval_${st}_${t}.log")
        line="$line|${r:--}"
    done
    printf "%-10s %-12s %-12s %-12s %-12s\n" "$st" \
        "$(echo "$line" | cut -d'|' -f2)" "$(echo "$line" | cut -d'|' -f3)" \
        "$(echo "$line" | cut -d'|' -f4)" "$(echo "$line" | cut -d'|' -f5)"
done
echo ""
echo "说明: 单元格 '-' = 该阶段还没见过该任务（不评）。"
echo "      与历史记录的差异（早期普通 LoRA D 记录为 0/0/0/0.82、A 自评 0.9167）可能来自 episode 数不同，"
echo "      统一以本次 ${EPISODES}ep 口径为准；若要与其他臂比较请保持同 episode 数。"

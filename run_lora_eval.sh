#!/bin/bash
# =============================================================================
# run_lora_eval.sh — 普通 LoRA 支线评估（deploy_policy.py 修复版 40c112a 之后用）
#
# 评估对象: rt_lora_taskD--40000_chkpt (merged 全模型, 无 CL bank)
# 评估方式: eval_task_id=0 (当前权重即旧任务成绩 = 灾难性遗忘基线)
# 并行策略: 串行逐任务 × 8 worker (4,4,5,5,6,6,7,7) —— 不要多任务并行,
#           4 个 × 8 worker 会挤爆 4 卡显存秒退。
#
# 用法（前台跑, 实时看进度）:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   tmux new -s evalLoRA
#   bash run_lora_eval.sh 2>&1 | tee train_lora_eval.log
#
# 关键输出: 每个任务的 [wX|GPU Y] Success!/Merged success rate + 末尾汇总
# =============================================================================

set -u

CKPT_D="$LOGS_ROOT/rt_lora_taskD--40000_chkpt"
GPUS="${EVAL_GPUS:-4,4,5,5,6,6,7,7}"     # 8 worker
EPISODES="${EVAL_EPISODES:-50}"

# ---------- 前置检查 ----------
[ -n "${LOGS_ROOT:-}" ] || { echo "[FAIL] LOGS_ROOT 未设置 —— 请先: source server_env.sh"; exit 1; }
[ -d "$CKPT_D" ] || { echo "[FAIL] checkpoint 不存在: $CKPT_D"; exit 1; }
ls "$CKPT_D"/config.json >/dev/null 2>&1 || { echo "[FAIL] checkpoint 缺 config.json (merged 产物不完整?)"; exit 1; }

echo "======================================================================"
echo "== 普通 LoRA 支线评估 (merged ckpt, eval_task_id=0, 8 worker)"
echo "== ckpt: $CKPT_D"
echo "======================================================================"

# 任务映射: A/B/C/D -> task_name unnorm_key
task_info() {
    case "$1" in
        A) echo "handover_mic aloha_handover_mic_clean" ;;
        B) echo "grab_roller aloha_grab_roller_clean" ;;
        C) echo "stack_bowls_two aloha_stack_bowls_two_clean" ;;
        D) echo "open_laptop aloha_open_laptop_clean" ;;
        *) echo "" ;;
    esac
}

results=()
for t in A B C D; do
    info=$(task_info "$t")
    read -r tname unnorm <<< "$info"
    tag="loraD_${t}"
    log="/tmp/lora_eval_${t}.log"
    echo ""
    echo "==================== Task $t : $tname (eval_task_id=0) ===================="
    bash /mnt/data/pengshengdi/RoboTwin-main/policy/openvla-oft/eval_multi_gpu.sh \
        "$tname" demo_clean "$CKPT_D" 0 "$GPUS" "$unnorm" 0 "$EPISODES" "$tag" 0 \
        2>&1 | tee "$log" | grep -v "svulkan2.*error"
    rc=${PIPESTATUS[0]}
    rate=$(grep "Merged success rate" "$log" | tail -1)
    if [ $rc -ne 0 ] || [ -z "$rate" ]; then
        echo "[FAIL] Task $t 评估失败 (rc=$rc), 日志: $log"
        results+=("Task $t ($tname): FAILED")
    else
        echo "[OK] Task $t 完成: $rate"
        results+=("Task $t ($tname): $rate")
    fi
done

echo ""
echo "==================== 普通 LoRA 支线评估汇总 ===================="
for line in "${results[@]}"; do
    echo "  $line"
done
echo ""
echo "对照: CL-LoRA 无回放 v39b4 = 0-0.57-0-0.85; 方法 v39r2d = 0.62-0.88-0.88-0.80"

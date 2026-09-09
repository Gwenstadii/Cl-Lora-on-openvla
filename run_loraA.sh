#!/bin/bash
# =============================================================================
# run_loraA.sh — 普通 LoRA 支线 Stage 1 (A) 训练 + 训完自动自评
#
# 公平化配置: --lora_scope cl (注入范围与 CL-LoRA 对齐: 仅 L16-31 attn+ffn,
#   visual/投影器冻结, proprio 冻结); FiLM 训练 (stage1 与 CL 一致, 公平)。
#
# 用法（tmux 里前台跑）:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   tmux new -s loraA
#   bash run_loraA.sh 2>&1 | tee train_loraA.log
#
# 产物: $LOGS_ROOT/rt_lora_taskA--30000_chkpt (merged 全模型)
# 自评: 20 episode × 8 worker, 打印成功率 + PASS/WARN 判定
# =============================================================================

set -u

TRAIN_DIR="/mnt/data/pengshengdi/openvla-oft"
CKPT_A="$LOGS_ROOT/rt_lora_taskA--30000_chkpt"
GPUS="${GPUS:-4,5,6,7}"
EVAL_GPUS="${EVAL_GPUS:-4,4,5,5,6,6,7,7}"
PASS_THRESHOLD="${PASS_THRESHOLD:-0.85}"   # A 单任务达标线 (CL-LoRA A=0.98)

check_env() {
    [ -n "${VLA_PATH:-}" ]  || { echo "[FAIL] VLA_PATH 未设置 —— 请先: source server_env.sh"; exit 1; }
    [ -n "${LOGS_ROOT:-}" ] || { echo "[FAIL] LOGS_ROOT 未设置 —— 请先: source server_env.sh"; exit 1; }
    [ -f "$VLA_PATH/config.json" ] || { echo "[FAIL] 基座模型不存在: $VLA_PATH/config.json"; exit 1; }
}

echo "================ 前置检查 ================"
check_env
[ -d "$CKPT_A" ] && { echo "[SKIP] A checkpoint 已存在: $CKPT_A (想重训请先删除或改名)"; }
echo "[OK] VLA_PATH  = $VLA_PATH"
echo "[OK] LOGS_ROOT = $LOGS_ROOT"
echo "[OK] 配置: 普通 PEFT LoRA (lora_scope=cl, 对齐 CL-LoRA 注入范围), rank16, FiLM+proprio, 无冻结无 bank"
echo "============ 开始 Stage 1 (A) 训练 30000 步 ============"

cd "$TRAIN_DIR" || { echo "[FAIL] 目录不存在: $TRAIN_DIR"; exit 1; }

# 若已存在则跳过训练（支持续跑/重评）
if [ ! -d "$CKPT_A" ]; then
    env CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline \
    torchrun --standalone --nproc_per_node 4 vla-scripts/finetune.py \
        --vla_path "$VLA_PATH" \
        --data_root_dir datasets/rlds \
        --dataset_name aloha_handover_mic_clean \
        --run_root_dir "$LOGS_ROOT" --run_id_override rt_lora_taskA \
        --max_steps 30000 --save_freq 10000 \
        --batch_size 2 --learning_rate 5e-4 \
        --lr_warmup_steps 200 --num_steps_before_decay 100000 \
        --use_l1_regression True --use_diffusion False \
        --use_film True --use_proprio True --num_images_in_input 3 \
        --use_lora True --lora_rank 16 --lora_scope cl --lora_dropout 0.0 \
        --merge_lora_during_training True --image_aug True
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[FAIL] A 训练失败 (exit=$rc)"
        exit $rc
    fi
    echo "[OK] A 训练完成 -> $CKPT_A"
else
    echo "[INFO] 复用已有 A checkpoint, 直接评估"
fi

# ---------- 训完自动自评 (20 episode) ----------
echo ""
echo "==== 开始 A 自评 (20 episode × 8 worker) ===="
bash /mnt/data/pengshengdi/RoboTwin-main/policy/openvla-oft/eval_multi_gpu.sh \
    handover_mic demo_clean "$CKPT_A" 0 "$EVAL_GPUS" \
    aloha_handover_mic_clean 0 20 loraA_v2 0 \
    2>&1 | tee /tmp/loraA_self.log | grep -v "svulkan2.*error"
rc=${PIPESTATUS[0]}
rate=$(grep "Merged success rate" /tmp/loraA_self.log | tail -1 | grep -oE "[0-9]+\.[0-9]+")
if [ $rc -ne 0 ] || [ -z "$rate" ]; then
    echo "[FAIL] A 自评异常 (rc=$rc), 日志: /tmp/loraA_self.log"
    exit 1
fi

echo ""
echo "================ A 单任务自评结果 ================"
echo "  A: $rate"
ok=$(python -c "print('PASS' if $rate >= $PASS_THRESHOLD else 'WARN')")
echo "  判定: $ok (达标线 $PASS_THRESHOLD, CL-LoRA A=0.98 参考)"
if [ "$ok" = "PASS" ]; then
    echo ""
    echo ">>> A 达标, 下一步: bash run_lora_BCD_afterA.sh (训 B/C/D + 全任务评估)"
else
    echo ""
    echo ">>> A 未达标(<$PASS_THRESHOLD): 先别放 BCD, 把 /tmp/loraA_self.log 和训练日志贴给分析"
fi

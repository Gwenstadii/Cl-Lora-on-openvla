#!/bin/bash
# =============================================================================
# run_v39_anchor_BCD.sh — 无回放基线 v3 (漂移版 + FiLM 锚定正则)
#
# 叙事定位: 无回放基线支线的"可控残留"形态。
#   v1 (漂移版) / v2 (film_lr_scale=0.2) 的教训:
#     A/C 对任何 FiLM 移动都零容忍 (v2 把 FiLM lr 降到 0.2×, A/C 依然归零)。
#   v3 用锚定正则 λ 把 FiLM 拉向 Stage1(A) 的值 —— 低 lr 是"走得慢但持续走",
#   锚定是"走远了被拉回"。λ 连续可调 → 旧任务残留可落在 0.2 附近。
#
# 配置: 漂移版 (FiLM 训练, 无冻结, 无回放, 无 KD) + --film_anchor_reg λ
#   λ 越大 FiLM 越贴住 A (残留越高); 推荐从 1e-3 量级开始试, 看 loss_anchor 走势调。
#
# 用法（tmux 里前台跑）:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   tmux new -s trainB3
#   bash run_v39_anchor_BCD.sh 2>&1 | tee train_v39_anchor.log
#
# 产物: $LOGS_ROOT/rt_v39b3_taskB/C/D--40000_chkpt
# 验证点: 日志 "[FiLM] 锚定正则开启: λ=..., anchor=..." 且 tqdm 出现 loss_anchor
# =============================================================================

set -u

TRAIN_DIR="/mnt/data/pengshengdi/openvla-oft"
DATA_DIR="/mnt/data/pengshengdi/RoboTwin-main/data"
CKPT_A="$LOGS_ROOT/rt_v39_taskA--30000_chkpt"     # Stage 1 最终 checkpoint (Stage 2 的起点 + 锚定目标)

GPUS="${GPUS:-6,7}"
ANCHOR_REG="${ANCHOR_REG:-1e-3}"                  # FiLM 锚定正则权重 λ (漂移拉回力度)

# ---------- 前置检查 ----------
check_env() {
    [ -n "${VLA_PATH:-}" ]  || { echo "[FAIL] VLA_PATH 未设置 —— 请先: source server_env.sh"; exit 1; }
    [ -n "${LOGS_ROOT:-}" ] || { echo "[FAIL] LOGS_ROOT 未设置 —— 请先: source server_env.sh"; exit 1; }
    [ -f "$VLA_PATH/config.json" ] || { echo "[FAIL] 基座模型不存在: $VLA_PATH/config.json"; exit 1; }
}

echo "================ 前置检查 ================"
check_env
[ -d "$CKPT_A" ] || { echo "[FAIL] Task A checkpoint 不存在: $CKPT_A"; exit 1; }
ls "$CKPT_A"/vision_backbone--30000_checkpoint.pt >/dev/null 2>&1 \
    || { echo "[FAIL] Task A checkpoint 缺少 vision_backbone--30000_checkpoint.pt (锚定目标)"; exit 1; }
echo "[OK] VLA_PATH  = $VLA_PATH"
echo "[OK] LOGS_ROOT = $LOGS_ROOT"
echo "[OK] Stage1 ckpt = $CKPT_A (起点 + 锚定目标)"
echo "[OK] GPUS = $GPUS | ANCHOR_REG(λ) = $ANCHOR_REG"
echo "============ 开始 无回放基线v3(锚定正则) Stage 2 -> 3 -> 4 连续训练 ============"

cd "$TRAIN_DIR" || { echo "[FAIL] 目录不存在: $TRAIN_DIR"; exit 1; }

COMMON_ARGS=(--batch_size 1 --grad_accumulation_steps 4 --learning_rate 5e-4
  --lr_warmup_steps 200 --num_steps_before_decay 100000
  --use_cl_lora True --lora_rank 16 --shared_depth 8 --first_lora_layer 16
  --orthogonal_init True --freeze_a True --use_block_scale True --freeze_specific_a True
  --use_kd False --use_replay False --image_aug True
  --use_proprio True --use_film True --num_images_in_input 3
  --freeze_film_stage2 False --film_anchor_reg "$ANCHOR_REG" \
  --film_anchor_dir "$CKPT_A" --film_anchor_step 30000)

run_stage() {  # $1=stage  $2=dataset  $3=run_id  $4=prev_checkpoint_dir  $5=prev_step
    local stage=$1 ds=$2 rid=$3 prev_dir=$4 prev_step=$5
    echo ""
    echo "############ Stage $stage : $ds (from $prev_dir) ############"
    env CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline \
    torchrun --standalone --nproc_per_node 2 vla-scripts/train_cl_lora.py \
        --run_root_dir "$LOGS_ROOT" --run_id_override "$rid" \
        --max_steps 40000 --save_freq 10000 \
        --vla_path "$VLA_PATH" \
        --dataset_name "$ds" \
        --stage "$stage" \
        --previous_checkpoint_dir "$prev_dir" \
        --previous_checkpoint_step "$prev_step" \
        "${COMMON_ARGS[@]}"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[FAIL] Stage $stage ($ds) 训练失败 (exit=$rc), 终止后续 Stage"
        exit $rc
    fi
    echo "[OK] Stage $stage ($ds) 完成 -> $LOGS_ROOT/$rid--40000_chkpt"
}

# Stage 2: Task B (漂移版 + FiLM 锚定 A)
run_stage 2 aloha_grab_roller_clean rt_v39b3_taskB "$CKPT_A" 30000

# Stage 3: Task C
run_stage 3 aloha_stack_bowls_two_clean rt_v39b3_taskC "$LOGS_ROOT/rt_v39b3_taskB--40000_chkpt" 40000

# Stage 4: Task D
run_stage 4 aloha_open_laptop_clean rt_v39b3_taskD "$LOGS_ROOT/rt_v39b3_taskC--40000_chkpt" 40000

echo ""
echo "==== 无回放基线v3(锚定正则) 全部完成 ===="
echo "    B: $LOGS_ROOT/rt_v39b3_taskB--40000_chkpt"
echo "    C: $LOGS_ROOT/rt_v39b3_taskC--40000_chkpt"
echo "    D: $LOGS_ROOT/rt_v39b3_taskD--40000_chkpt"
echo "    之后评估: bash policy/openvla-oft/eval_sequence.sh \$LOGS_ROOT/rt_v39b3_taskD--40000_chkpt 6,7 50 v39b3D A B C D"

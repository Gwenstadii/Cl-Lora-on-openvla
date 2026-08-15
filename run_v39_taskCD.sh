#!/bin/bash
# =============================================================================
# run_v39_taskCD.sh — V39 Task C (stack_bowls_two) + Task D (open_laptop) 连续训练
# Stage 3 -> Stage 4 自动衔接: Stage 3 训练成功后才启动 Stage 4, 不用人守着
#
# 用法（tmux 里前台跑, 实时看输出）:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   tmux new -s trainCD
#   bash run_v39_taskCD.sh 2>&1 | tee train_v39_taskCD.log
#
# 配置: 与 Task A/B 完全一致 (first_lora_layer=16, shared_depth=8, rank=16)
#       Stage 3/4 各 40000 步, 每 10000 步保存; 从各自上一阶段 checkpoint 续训
#       结束产物: $LOGS_ROOT/rt_v39_taskC--40000_chkpt 和 rt_v39_taskD--40000_chkpt
# =============================================================================

set -u

TRAIN_DIR="/mnt/data/pengshengdi/openvla-oft"
DATA_DIR="/mnt/data/pengshengdi/RoboTwin-main/data"
CKPT_B="$LOGS_ROOT/rt_v39_taskB--40000_chkpt"     # Stage 2 最终 checkpoint (Stage 3 的起点)

# ---------- 前置检查 ----------
check_env() {
    [ -n "${VLA_PATH:-}" ]  || { echo "[FAIL] VLA_PATH 未设置 —— 请先: source server_env.sh"; exit 1; }
    [ -n "${LOGS_ROOT:-}" ] || { echo "[FAIL] LOGS_ROOT 未设置 —— 请先: source server_env.sh"; exit 1; }
    [ -f "$VLA_PATH/config.json" ] || { echo "[FAIL] 基座模型不存在: $VLA_PATH/config.json"; exit 1; }
}

echo "================ 前置检查 ================"
check_env
[ -d "$CKPT_B" ] || { echo "[FAIL] Task B checkpoint 不存在: $CKPT_B"; exit 1; }
ls "$CKPT_B"/cl_lora_adapter.pt "$CKPT_B"/cl_lora_config.json >/dev/null 2>&1 \
    || { echo "[FAIL] Task B checkpoint 缺少 cl_lora_adapter.pt / cl_lora_config.json"; exit 1; }
for ds in stack_bowls_two open_laptop; do
    ls -d "$DATA_DIR"/*"${ds}"* >/dev/null 2>&1 \
        || echo "[WARN] 未找到 $ds 数据目录: $DATA_DIR/*${ds}* (训练可能报数据集找不到)"
done
echo "[OK] VLA_PATH    = $VLA_PATH"
echo "[OK] LOGS_ROOT   = $LOGS_ROOT"
echo "[OK] Stage2 ckpt = $CKPT_B"
echo "============ 开始 Stage 3 -> Stage 4 连续训练 (共 80000 步) ============"

cd "$TRAIN_DIR" || { echo "[FAIL] 目录不存在: $TRAIN_DIR"; exit 1; }

# 与 Task A/B 相同的公共超参
COMMON_ARGS=(--batch_size 1 --grad_accumulation_steps 4 --learning_rate 5e-4
  --lr_warmup_steps 200 --num_steps_before_decay 100000
  --use_cl_lora True --lora_rank 16 --shared_depth 8 --first_lora_layer 16
  --orthogonal_init True --freeze_a True --use_block_scale True --freeze_specific_a True
  --use_kd False --use_replay False --image_aug True
  --use_proprio True --use_film True --num_images_in_input 3)

run_stage() {  # $1=stage  $2=dataset  $3=run_id  $4=prev_checkpoint_dir  $5=prev_step
    local stage=$1 ds=$2 rid=$3 prev_dir=$4 prev_step=$5
    echo ""
    echo "############ Stage $stage : $ds (from $prev_dir) ############"
    env CUDA_VISIBLE_DEVICES=4,5 PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline \
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

# Stage 3: Task C = stack_bowls_two, 从 Task B checkpoint 续训
run_stage 3 aloha_stack_bowls_two_clean rt_v39_taskC "$CKPT_B" 40000

# Stage 4: Task D = open_laptop, 从 Stage 3 产物自动衔接
run_stage 4 aloha_open_laptop_clean rt_v39_taskD "$LOGS_ROOT/rt_v39_taskC--40000_chkpt" 40000

echo ""
echo "==== Stage 3 + Stage 4 全部完成 ===="
echo "    C: $LOGS_ROOT/rt_v39_taskC--40000_chkpt"
echo "    D: $LOGS_ROOT/rt_v39_taskD--40000_chkpt"
echo "    之后评估: eval_task_id 1=A 2=B 3=C 4=D"

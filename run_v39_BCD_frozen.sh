#!/bin/bash
# =============================================================================
# run_v39_BCD_frozen.sh — V39 冻结 FiLM 版: Task B → C → D 连续重训 (Stage 2-4)
#
# 背景: 无回放基线 B/C/D 训练里 FiLM (~450M) 持续漂移导致旧任务灾难性遗忘
#       (A 0.98→0.04→0, C 0.86→0.02)。本脚本用「Stage>1 冻结 FiLM」(方案 A)
#       重跑 B/C/D, 验证旧任务保留率是否回升。train_cl_lora.py 已内置冻结逻辑。
#
# 用法（tmux 里前台跑, 实时看输出）:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   tmux new -s trainF
#   bash run_v39_BCD_frozen.sh 2>&1 | tee train_v39_BCD_frozen.log
#
# 产物（独立 run_id, 不覆盖漂移版 rt_v39_taskX）:
#   $LOGS_ROOT/rt_v39f_taskB--40000_chkpt
#   $LOGS_ROOT/rt_v39f_taskC--40000_chkpt
#   $LOGS_ROOT/rt_v39f_taskD--40000_chkpt
# 验证点: Stage 2 日志里 # total trainable params 应从 ~4.5亿 降到 ~300万
# =============================================================================

set -u

TRAIN_DIR="/mnt/data/pengshengdi/openvla-oft"
DATA_DIR="/mnt/data/pengshengdi/RoboTwin-main/data"
CKPT_A="$LOGS_ROOT/rt_v39_taskA--30000_chkpt"     # Stage 1 最终 checkpoint (Stage 2 的起点, FiLM 冻结源)

# 使用的 GPU（物理卡号, 逗号分隔; 环境变量可覆盖, 默认 6,7）
GPUS="${GPUS:-6,7}"

# ---------- 前置检查 ----------
check_env() {
    [ -n "${VLA_PATH:-}" ]  || { echo "[FAIL] VLA_PATH 未设置 —— 请先: source server_env.sh"; exit 1; }
    [ -n "${LOGS_ROOT:-}" ] || { echo "[FAIL] LOGS_ROOT 未设置 —— 请先: source server_env.sh"; exit 1; }
    [ -f "$VLA_PATH/config.json" ] || { echo "[FAIL] 基座模型不存在: $VLA_PATH/config.json"; exit 1; }
}

echo "================ 前置检查 ================"
check_env
[ -d "$CKPT_A" ] || { echo "[FAIL] Task A checkpoint 不存在: $CKPT_A"; exit 1; }
ls "$CKPT_A"/cl_lora_adapter.pt "$CKPT_A"/cl_lora_config.json >/dev/null 2>&1 \
    || { echo "[FAIL] Task A checkpoint 缺少 cl_lora_adapter.pt / cl_lora_config.json"; exit 1; }
for ds in grab_roller stack_bowls_two open_laptop; do
    ls -d "$DATA_DIR"/*"${ds}"* >/dev/null 2>&1 \
        || echo "[WARN] 未找到 $ds 数据目录: $DATA_DIR/*${ds}* (训练可能报数据集找不到)"
done
echo "[OK] VLA_PATH    = $VLA_PATH"
echo "[OK] LOGS_ROOT   = $LOGS_ROOT"
echo "[OK] Stage1 ckpt = $CKPT_A (FiLM 冻结源)"
echo "[OK] GPUS        = $GPUS"
echo "============ 开始 冻结FiLM 版 Stage 2 -> 3 -> 4 连续训练 ============"

cd "$TRAIN_DIR" || { echo "[FAIL] 目录不存在: $TRAIN_DIR"; exit 1; }

# 与漂移版完全相同的公共超参（只多/改了 FiLM 冻结逻辑, 在代码里）
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

# Stage 2: Task B = grab_roller, 从 Task A checkpoint 续训（FiLM 自此冻结）
run_stage 2 aloha_grab_roller_clean rt_v39f_taskB "$CKPT_A" 30000

# Stage 3: Task C = stack_bowls_two
run_stage 3 aloha_stack_bowls_two_clean rt_v39f_taskC "$LOGS_ROOT/rt_v39f_taskB--40000_chkpt" 40000

# Stage 4: Task D = open_laptop
run_stage 4 aloha_open_laptop_clean rt_v39f_taskD "$LOGS_ROOT/rt_v39f_taskC--40000_chkpt" 40000

echo ""
echo "==== 冻结FiLM 版 Stage 2 + 3 + 4 全部完成 ===="
echo "    B: $LOGS_ROOT/rt_v39f_taskB--40000_chkpt"
echo "    C: $LOGS_ROOT/rt_v39f_taskC--40000_chkpt"
echo "    D: $LOGS_ROOT/rt_v39f_taskD--40000_chkpt"
echo "    之后评估: bash policy/openvla-oft/eval_sequence.sh \$LOGS_ROOT/rt_v39f_taskD--40000_chkpt 6,7 50 v39fD A B C D"

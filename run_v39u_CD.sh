#!/bin/bash
# =============================================================================
# run_v39u_CD.sh — CL-LoRA + 普通回放(uniform) 支线 · 第二步
#   C → D 连续训练 (uniform 回放, D 阶段用与 v39r2d 相同的 buffer 加权) + 全任务评估
#
# 前提: run_v39u_B.sh 已跑完且 B 自评达标
# 配置: 与 v39r2d 对齐 (冻结 FiLM + A 冻结 + proprio 修复 + every4/w0.5 + KD0.2),
#       D 阶段 buffers 加权 A×2+B×1+C×3 (与原型回放支线一致的采样预算分配)
#
# 用法:
#   tmux new -s trainU_CD
#   bash run_v39u_CD.sh 2>&1 | tee train_v39u_CD.log
# =============================================================================

set -u

TRAIN_DIR="/mnt/data/pengshengdi/openvla-oft"
CKPT_B="$LOGS_ROOT/rt_v39u_taskB--40000_chkpt"
BUF_U="$LOGS_ROOT/replay_buffers_uniform"
GPUS="${GPUS:-4,5,6,7}"
EVAL_GPUS="${EVAL_GPUS:-4,4,5,5,6,6,7,7}"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NPROC=${#GPU_ARR[@]}
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM=$((8 / (BATCH_SIZE * NPROC)))

[ -n "${LOGS_ROOT:-}" ] || { echo "[FAIL] LOGS_ROOT 未设置"; exit 1; }
[ -d "$CKPT_B" ] || { echo "[FAIL] B ckpt 不存在: $CKPT_B —— 先跑 run_v39u_B.sh"; exit 1; }
for t in taskA taskB taskC; do
    [ -f "$BUF_U/$t/manifest.jsonl" ] || { echo "[FAIL] uniform buffer 缺失: $BUF_U/$t"; exit 1; }
done

echo "[OK] B ckpt = $CKPT_B | GPUS=$GPUS | 有效batch=8 (batch=$BATCH_SIZE × accum=$GRAD_ACCUM × nproc=$NPROC)"
echo "============ uniform 回放支线: C -> D ============"

cd "$TRAIN_DIR" || { echo "[FAIL] 目录不存在: $TRAIN_DIR"; exit 1; }

run_stage() {  # $1=stage  $2=dataset  $3=run_id  $4=prev_dir  $5=prev_step  $6=teacher_dir  $7=teacher_step  $8..=buffers
    local stage=$1 ds=$2 rid=$3 prev_dir=$4 prev_step=$5 tdir=$6 tstep=$7
    shift 7
    local buffers_csv
    buffers_csv=$(IFS=,; echo "$*")
    if [ -d "$LOGS_ROOT/$rid--40000_chkpt" ]; then
        echo "[SKIP] $rid--40000_chkpt 已存在"
        return 0
    fi
    echo ""
    echo "############ Stage $stage : $ds ############"
    echo "    buffers: $buffers_csv"
    env CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline \
    torchrun --standalone --nproc_per_node $NPROC vla-scripts/train_cl_lora.py \
        --run_root_dir "$LOGS_ROOT" --run_id_override "$rid" \
        --max_steps 40000 --save_freq 10000 \
        --vla_path "$VLA_PATH" \
        --dataset_name "$ds" \
        --stage "$stage" \
        --previous_checkpoint_dir "$prev_dir" --previous_checkpoint_step "$prev_step" \
        --teacher_checkpoint_dir "$tdir" --teacher_checkpoint_step "$tstep" \
        --replay_buffer_dirs "$buffers_csv" \
        --batch_size "$BATCH_SIZE" --grad_accumulation_steps "$GRAD_ACCUM" --learning_rate 5e-4 \
        --lr_warmup_steps 200 --num_steps_before_decay 100000 \
        --use_cl_lora True --lora_rank 16 --shared_depth 8 --first_lora_layer 16 \
        --orthogonal_init True --freeze_a True --use_block_scale True --freeze_specific_a True \
        --use_kd True --use_replay True --freeze_film_stage2 True \
        --replay_every_n_steps 4 --replay_loss_weight 0.5 --lambda_kd 0.2 \
        --image_aug True --use_proprio True --use_film True --num_images_in_input 3
    [ $? -ne 0 ] && { echo "[FAIL] Stage $stage 训练失败"; exit 1; }
    echo "[OK] Stage $stage 完成 -> $LOGS_ROOT/$rid--40000_chkpt"
}

# Stage 3 (C): buffers A+B
run_stage 3 aloha_stack_bowls_two_clean rt_v39u_taskC \
    "$CKPT_B" 40000 "$CKPT_B" 40000 \
    "$BUF_U/taskA" "$BUF_U/taskB"

# Stage 4 (D): buffers 加权 A×2+B×1+C×3 (与原型回放支线一致)
run_stage 4 aloha_open_laptop_clean rt_v39u_taskD \
    "$LOGS_ROOT/rt_v39u_taskC--40000_chkpt" 40000 \
    "$LOGS_ROOT/rt_v39u_taskC--40000_chkpt" 40000 \
    "$BUF_U/taskA" "$BUF_U/taskA" "$BUF_U/taskB" "$BUF_U/taskC" "$BUF_U/taskC" "$BUF_U/taskC"

echo ""
echo "==== uniform 回放支线 C/D 完成 ===="

# ---------- 全任务评估 ----------
if [ "${EVAL_AFTER_TRAIN:-1}" = "1" ]; then
    CKPT_D="$LOGS_ROOT/rt_v39u_taskD--40000_chkpt"
    EVAL_LOG="$(dirname "$TRAIN_DIR")/train_v39u_eval.log"
    echo "==== 自动全任务评估 (γ=0, 8 worker) ===="
    FILM_GAMMA=0 bash /mnt/data/pengshengdi/RoboTwin-main/policy/openvla-oft/eval_sequence.sh \
        "$CKPT_D" "$EVAL_GPUS" 50 v39uD A B C D 2>&1 | tee "$EVAL_LOG" | grep -v "svulkan2.*error"
    echo ""
    echo "================ v39u (uniform 回放) 评估汇总 ================"
    grep "Merged success rate" "$EVAL_LOG"
    echo ""
    echo "对照 v39r2d (原型回放) = 0.62 / 0.88 / 0.88 / 0.80"
else
    echo "[SKIP] EVAL_AFTER_TRAIN=0"
fi

#!/bin/bash
# =============================================================================
# run_v39u_B.sh — CL-LoRA + 普通回放(uniform) 支线 · 第一步
#   A 验证(复用现有 ckpt + 自评) → 建 uniform buffers(A/B/C) → 训 B → 自评 B
#
# 定位: "回放形式消融" —— 与 v39r2d (原型回放) 唯一差异 = buffer 选帧规则
#   (uniform 均匀时间采样 vs 物理分段+原型Top-K), 预算/格式/训练配置完全一致。
#
# 配置(与 v39r2d 对齐): 冻结 FiLM(stage≥2) + A 冻结(True) + proprio 修复 +
#   回放 every4/weight0.5 + KD λ0.2 + buffers 顺序同 v39r2d
#
# 用法（tmux 里前台跑）:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   tmux new -s trainU_B
#   bash run_v39u_B.sh 2>&1 | tee train_v39u_B.log
#
# 产物: $LOGS_ROOT/replay_buffers_uniform/task{A,B,C}
#       $LOGS_ROOT/rt_v39u_taskB--40000_chkpt
# 判定: B 自评 PASS(≥0.8) 后再跑 run_v39u_CD.sh
# =============================================================================

set -u

TRAIN_DIR="/mnt/data/pengshengdi/openvla-oft"
CKPT_A="$LOGS_ROOT/rt_v39_taskA--30000_chkpt"
BUF_P="$LOGS_ROOT/replay_buffers"              # 原型 buffer (用于预算匹配)
BUF_U="$LOGS_ROOT/replay_buffers_uniform"      # uniform buffer 输出
GPUS="${GPUS:-4,5,6,7}"
EVAL_GPUS="${EVAL_GPUS:-4,4,5,5,6,6,7,7}"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NPROC=${#GPU_ARR[@]}
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM=$((8 / (BATCH_SIZE * NPROC)))
PASS_THRESHOLD="${PASS_THRESHOLD:-0.80}"

check_env() {
    [ -n "${VLA_PATH:-}" ]  || { echo "[FAIL] VLA_PATH 未设置 —— 先 source server_env.sh"; exit 1; }
    [ -n "${LOGS_ROOT:-}" ] || { echo "[FAIL] LOGS_ROOT 未设置 —— 先 source server_env.sh"; exit 1; }
    [ -f "$VLA_PATH/config.json" ] || { echo "[FAIL] 基座模型不存在"; exit 1; }
}

echo "================ 前置检查 ================"
check_env
[ -d "$CKPT_A" ] || { echo "[FAIL] A ckpt 不存在: $CKPT_A (需先跑 CL-LoRA Stage 1)"; exit 1; }
for t in taskA taskB taskC; do
    [ -f "$BUF_P/$t/manifest.jsonl" ] || { echo "[FAIL] 原型 buffer 缺失(预算匹配用): $BUF_P/$t"; exit 1; }
done
echo "[OK] A ckpt = $CKPT_A (复用, 无需重训——uniform 消融的差异只在 stage2+)"
echo "[OK] GPUS=$GPUS | NPROC=$NPROC | batch_size=$BATCH_SIZE | grad_accum=$GRAD_ACCUM (有效batch=8)"

cd "$TRAIN_DIR" || { echo "[FAIL] 目录不存在: $TRAIN_DIR"; exit 1; }

# ---------- 1) A 自评（确认 A 正常） ----------
echo ""
echo "==== [1/4] A 自评 (20ep, 8 worker) ===="
bash /mnt/data/pengshengdi/RoboTwin-main/policy/openvla-oft/eval_multi_gpu.sh \
    handover_mic demo_clean "$CKPT_A" 0 "$EVAL_GPUS" \
    aloha_handover_mic_clean 0 20 v39u_A_check 0 \
    2>&1 | tee /tmp/v39u_A.log | grep -v "svulkan2.*error"
A_RATE=$(grep "Merged success rate" /tmp/v39u_A.log | tail -1 | grep -oE "[0-9]+\.[0-9]+")
echo "A 自评: ${A_RATE:-异常}"
python -c "import sys; sys.exit(0 if float('${A_RATE:-0}') >= $PASS_THRESHOLD else 1)" \
    || { echo "[WARN] A 自评 < $PASS_THRESHOLD, 建议先排查 A 再继续 (仍继续执行, 可 Ctrl-C)"; sleep 5; }

# ---------- 2) 建 uniform buffers（预算匹配原型） ----------
echo ""
echo "==== [2/4] 构建 uniform replay buffers (预算匹配原型 buffer) ===="
mkdir -p "$BUF_U"
build_one() {  # $1=tag  $2=dataset  $3=stats 来源 ckpt
    local tag=$1 ds=$2 stats_ckpt=$3
    if [ -f "$BUF_U/$tag/manifest.jsonl" ] && [ -s "$BUF_U/$tag/manifest.jsonl" ]; then
        echo "[SKIP] $BUF_U/$tag 已存在 ($(wc -l < "$BUF_U/$tag/manifest.jsonl") samples)"
        return 0
    fi
    echo "---- 构建 $tag (uniform) ----"
    CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python vla-scripts/build_uniform_replay_buffer_robotwin.py \
        --data-root-dir datasets/rlds \
        --dataset-name "$ds" \
        --output-dir "$BUF_U/$tag" \
        --stats-path "$stats_ckpt/dataset_statistics.json" \
        --match-budget-buffer-dir "$BUF_P/$tag" \
        --num-episodes 10 --overwrite || { echo "[FAIL] $tag uniform buffer 构建失败"; exit 1; }
    echo "[OK] $tag uniform buffer: $(wc -l < "$BUF_U/$tag/manifest.jsonl") samples"
}

build_one taskA aloha_handover_mic_clean "$CKPT_A"
build_one taskB aloha_grab_roller_clean "$LOGS_ROOT/rt_v39_taskB--40000_chkpt"
build_one taskC aloha_stack_bowls_two_clean "$LOGS_ROOT/rt_v39_taskC--40000_chkpt"

# ---------- 3) Stage 2 (B): uniform 回放 ----------
echo ""
echo "==== [3/4] Stage 2 (B) 训练: uniform 回放 + KD ===="
if [ -d "$LOGS_ROOT/rt_v39u_taskB--40000_chkpt" ]; then
    echo "[SKIP] rt_v39u_taskB--40000_chkpt 已存在, 跳过训练"
else
    env CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline \
    torchrun --standalone --nproc_per_node $NPROC vla-scripts/train_cl_lora.py \
        --run_root_dir "$LOGS_ROOT" --run_id_override rt_v39u_taskB \
        --max_steps 40000 --save_freq 10000 \
        --vla_path "$VLA_PATH" \
        --dataset_name aloha_grab_roller_clean \
        --stage 2 \
        --previous_checkpoint_dir "$CKPT_A" --previous_checkpoint_step 30000 \
        --teacher_checkpoint_dir "$CKPT_A" --teacher_checkpoint_step 30000 \
        --replay_buffer_dirs "$BUF_U/taskA" \
        --batch_size "$BATCH_SIZE" --grad_accumulation_steps "$GRAD_ACCUM" --learning_rate 5e-4 \
        --lr_warmup_steps 200 --num_steps_before_decay 100000 \
        --use_cl_lora True --lora_rank 16 --shared_depth 8 --first_lora_layer 16 \
        --orthogonal_init True --freeze_a True --use_block_scale True --freeze_specific_a True \
        --use_kd True --use_replay True --freeze_film_stage2 True \
        --replay_every_n_steps 4 --replay_loss_weight 0.5 --lambda_kd 0.2 \
        --image_aug True --use_proprio True --use_film True --num_images_in_input 3
    [ $? -ne 0 ] && { echo "[FAIL] Stage 2 (B) 训练失败"; exit 1; }
    echo "[OK] Stage 2 (B) 完成 -> $LOGS_ROOT/rt_v39u_taskB--40000_chkpt"
fi

# ---------- 4) B 自评（新任务是否正常学习） ----------
echo ""
echo "==== [4/4] B 自评 (20ep, 8 worker) ===="
bash /mnt/data/pengshengdi/RoboTwin-main/policy/openvla-oft/eval_multi_gpu.sh \
    grab_roller demo_clean "$LOGS_ROOT/rt_v39u_taskB--40000_chkpt" 0 "$EVAL_GPUS" \
    aloha_grab_roller_clean 2 20 v39u_B_self 0 \
    2>&1 | tee /tmp/v39u_B.log | grep -v "svulkan2.*error"
B_RATE=$(grep "Merged success rate" /tmp/v39u_B.log | tail -1 | grep -oE "[0-9]+\.[0-9]+")
echo ""
echo "================ v39u 第一步结果 ================"
echo "  A 自评: ${A_RATE:-异常}   (参考: CL-LoRA A=0.98)"
echo "  B 自评: ${B_RATE:-异常}   (参考: v39r2d 的 B=0.88)"
if python -c "import sys; sys.exit(0 if float('${B_RATE:-0}') >= $PASS_THRESHOLD else 1)"; then
    echo "  >>> B 达标, 继续: bash run_v39u_CD.sh (训 C/D + 全任务评估)"
else
    echo "  >>> B 未达标(<$PASS_THRESHOLD): 把 /tmp/v39u_B.log 与训练日志贴给分析"
fi

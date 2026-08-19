#!/bin/bash
# =============================================================================
# run_v39_replay_BCD.sh — V39 原型回放版: Task B → C → D 连续训练 (Stage 2-4)
#
# 目标: 在无回放基线 (A 0.98→0.04→0, C 0.86→0.02 灾难性遗忘) 之上,
#       用原型回放 + KD 验证旧任务保留率 (预期: 新任务正常学习, 旧任务遗忘率极小)
#
# 流程 (全程自动):
#   1. 为 Task A/B/C 构建原型回放 buffer (复用训练同款 RLDS 管线, 归一化一致)
#   2. Stage 2 (B): 从 rt_v39_taskA--30000_chkpt 续训 + A 的 buffer + KD(A teacher)
#   3. Stage 3 (C): 从回放版 B 续训 + A+B 的 buffer + KD(B teacher)
#   4. Stage 4 (D): 从回放版 C 续训 + A+B+C 的 buffer + KD(C teacher)
#
# 用法（tmux 里前台跑, 实时看输出）:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   tmux new -s trainR
#   bash run_v39_replay_BCD.sh 2>&1 | tee train_v39_replay_BCD.log
#
# 产物 (独立 run_id, 与漂移版 rt_v39_taskX / 冻结版 rt_v39f_taskX 区分):
#   $LOGS_ROOT/replay_buffers/taskA|taskB|taskC   (原型回放 buffer)
#   $LOGS_ROOT/rt_v39r_taskB/C/D--40000_chkpt     (回放版 checkpoint)
# =============================================================================

set -u

TRAIN_DIR="/mnt/data/pengshengdi/openvla-oft"
DATA_DIR="/mnt/data/pengshengdi/RoboTwin-main/data"
CKPT_A="$LOGS_ROOT/rt_v39_taskA--30000_chkpt"           # Stage 1 最终 checkpoint (回放起点)
BUFFER_ROOT="$LOGS_ROOT/replay_buffers"                 # buffer 输出目录

# 使用的 GPU（物理卡号, 逗号分隔; 环境变量可覆盖, 默认 4,5 —— 67 卡留给冻结版训练）
GPUS="${GPUS:-4,5}"
BUILD_GPU="${BUILD_GPU:-4}"                              # 建 buffer 用的单卡 (取 GPUS 第一张也可)
BUILD_GPU="${GPUS%%,*}"                                  # 自动取 GPUS 第一张

# 回放超参 (LIBERO 验证过的默认值)
NUM_EPISODES="${NUM_EPISODES:-10}"                       # 每个任务用几条轨迹建 buffer
TOP_K="${TOP_K:-3}"                                      # 每 segment 选几帧
REPLAY_EVERY="${REPLAY_EVERY:-1}"                        # 每 N 步算一次 replay loss
REPLAY_WEIGHT="${REPLAY_WEIGHT:-1.0}"
KD_WEIGHT="${KD_WEIGHT:-1.0}"

# ---------- 前置检查 ----------
check_env() {
    [ -n "${VLA_PATH:-}" ]  || { echo "[FAIL] VLA_PATH 未设置 —— 请先: source server_env.sh"; exit 1; }
    [ -n "${LOGS_ROOT:-}" ] || { echo "[FAIL] LOGS_ROOT 未设置 —— 请先: source server_env.sh"; exit 1; }
    [ -f "$VLA_PATH/config.json" ] || { echo "[FAIL] 基座模型不存在: $VLA_PATH/config.json"; exit 1; }
}

echo "================ 前置检查 ================"
check_env
[ -d "$CKPT_A" ] || { echo "[FAIL] Task A checkpoint 不存在: $CKPT_A"; exit 1; }
ls "$CKPT_A"/cl_lora_adapter.pt "$CKPT_A"/cl_lora_config.json "$CKPT_A"/teacher_snapshot--30000.pt >/dev/null 2>&1 \
    || { echo "[FAIL] Task A checkpoint 缺少 cl_lora_adapter.pt / cl_lora_config.json / teacher_snapshot--30000.pt"; exit 1; }
echo "[OK] VLA_PATH    = $VLA_PATH"
echo "[OK] LOGS_ROOT   = $LOGS_ROOT"
echo "[OK] Stage1 ckpt = $CKPT_A"
echo "[OK] GPUS        = $GPUS (build 用 $BUILD_GPU)"
echo "[OK] 回放超参: episodes=$NUM_EPISODES top_k=$TOP_K replay_every=$REPLAY_EVERY replay_w=$REPLAY_WEIGHT kd_w=$KD_WEIGHT"
echo "============ 开始: 建 buffer -> Stage 2 -> 3 -> 4 (原型回放版) ============"

cd "$TRAIN_DIR" || { echo "[FAIL] 目录不存在: $TRAIN_DIR"; exit 1; }

# ---------- 1) 构建原型回放 buffer ----------
# $1=task_short(A/B/C)  $2=dataset_name  $3=输出目录
build_buffer() {
    local tag=$1 ds=$2 out=$3
    if [ -f "$out/manifest.jsonl" ] && [ -s "$out/manifest.jsonl" ] && [ "${FORCE_REBUILD:-0}" != "1" ]; then
        echo "[SKIP] buffer 已存在: $out ($(wc -l < "$out/manifest.jsonl") samples)"
        return 0
    fi
    echo "############ 构建 $tag buffer: $ds ############"
    CUDA_VISIBLE_DEVICES=$BUILD_GPU python vla-scripts/build_replay_buffer_robotwin.py \
        --vla-path "$VLA_PATH" \
        --data-root-dir datasets/rlds \
        --dataset-name "$ds" \
        --output-dir "$out" \
        --num-episodes "$NUM_EPISODES" \
        --top-k "$TOP_K" \
        --overwrite
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[FAIL] $tag buffer 构建失败 (exit=$rc)"
        exit $rc
    fi
    echo "[OK] $tag buffer 完成: $out ($(wc -l < "$out/manifest.jsonl") samples)"
}

mkdir -p "$BUFFER_ROOT"
build_buffer A aloha_handover_mic_clean "$BUFFER_ROOT/taskA"
build_buffer B aloha_grab_roller_clean "$BUFFER_ROOT/taskB"
build_buffer C aloha_stack_bowls_two_clean "$BUFFER_ROOT/taskC"

# ---------- 2-4) 回放训练 ----------
COMMON_ARGS=(--batch_size 1 --grad_accumulation_steps 4 --learning_rate 5e-4
  --lr_warmup_steps 200 --num_steps_before_decay 100000
  --use_cl_lora True --lora_rank 16 --shared_depth 8 --first_lora_layer 16
  --orthogonal_init True --freeze_a True --use_block_scale True --freeze_specific_a True
  --image_aug True --use_proprio True --use_film True --num_images_in_input 3
  --use_kd True --use_replay True
  --replay_every_n_steps "$REPLAY_EVERY" --replay_loss_weight "$REPLAY_WEIGHT" --lambda_kd "$KD_WEIGHT")

run_replay_stage() {  # $1=stage  $2=dataset  $3=run_id  $4=prev_dir  $5=prev_step  $6=teacher_dir  $7=teacher_step  $8..=buffer_dirs
    local stage=$1 ds=$2 rid=$3 prev_dir=$4 prev_step=$5 tdir=$6 tstep=$7
    shift 7
    local buffers="$*"
    echo ""
    echo "############ Stage $stage : $ds ############"
    echo "    replay buffers: $buffers"
    echo "    teacher: $tdir/teacher_snapshot--$tstep.pt"
    env CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline \
    torchrun --standalone --nproc_per_node 2 vla-scripts/train_cl_lora.py \
        --run_root_dir "$LOGS_ROOT" --run_id_override "$rid" \
        --max_steps 40000 --save_freq 10000 \
        --vla_path "$VLA_PATH" \
        --dataset_name "$ds" \
        --stage "$stage" \
        --previous_checkpoint_dir "$prev_dir" \
        --previous_checkpoint_step "$prev_step" \
        --teacher_checkpoint_dir "$tdir" --teacher_checkpoint_step "$tstep" \
        --replay_buffer_dirs $buffers \
        "${COMMON_ARGS[@]}"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[FAIL] Stage $stage ($ds) 训练失败 (exit=$rc), 终止后续 Stage"
        exit $rc
    fi
    echo "[OK] Stage $stage ($ds) 完成 -> $LOGS_ROOT/$rid--40000_chkpt"
}

# Stage 2: Task B, replay=A buffer, teacher=A
run_replay_stage 2 aloha_grab_roller_clean rt_v39r_taskB \
    "$CKPT_A" 30000 \
    "$CKPT_A" 30000 \
    "$BUFFER_ROOT/taskA"

# Stage 3: Task C, replay=A+B buffers, teacher=回放版B
run_replay_stage 3 aloha_stack_bowls_two_clean rt_v39r_taskC \
    "$LOGS_ROOT/rt_v39r_taskB--40000_chkpt" 40000 \
    "$LOGS_ROOT/rt_v39r_taskB--40000_chkpt" 40000 \
    "$BUFFER_ROOT/taskA" "$BUFFER_ROOT/taskB"

# Stage 4: Task D, replay=A+B+C buffers, teacher=回放版C
run_replay_stage 4 aloha_open_laptop_clean rt_v39r_taskD \
    "$LOGS_ROOT/rt_v39r_taskC--40000_chkpt" 40000 \
    "$LOGS_ROOT/rt_v39r_taskC--40000_chkpt" 40000 \
    "$BUFFER_ROOT/taskA" "$BUFFER_ROOT/taskB" "$BUFFER_ROOT/taskC"

echo ""
echo "==== 原型回放版 Stage 2 + 3 + 4 全部完成 ===="
echo "    B: $LOGS_ROOT/rt_v39r_taskB--40000_chkpt"
echo "    C: $LOGS_ROOT/rt_v39r_taskC--40000_chkpt"
echo "    D: $LOGS_ROOT/rt_v39r_taskD--40000_chkpt"
echo "    之后评估: bash policy/openvla-oft/eval_sequence.sh \$LOGS_ROOT/rt_v39r_taskD--40000_chkpt 4,5 50 v39rD A B C D"

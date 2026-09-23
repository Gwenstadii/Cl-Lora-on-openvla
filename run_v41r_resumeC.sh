#!/usr/bin/env bash
# =============================================================================
# run_v41r_resumeC.sh — 从 v41_r 的 C 段中间 ckpt（10000 步）续训到满 40000 步
#
# 背景: v41_r 的 C 段训到 10000 步中断/需要补齐。训练脚本没有 "从第 N 步接着跑到 M 步"
#   的原生开关，所以这里用项目里既有的做法（v39r2 用过）：
#     ① 以 **同 stage 的中间 ckpt** 作为 previous_checkpoint，并加 `--skip_reinit True`
#        （否则 stage>1 的冻结/清零逻辑会把当前 stage 的 specific-B 重新清零，等于白训）
#     ② 用**新的 run_id** 训练剩余步数（30000 步），避免覆盖正在续训的源目录
#     ③ 训练完把产物**对齐命名**成 `rt_v41_r_taskC--40000_chkpt`（连内部文件名里的步号一起改），
#        这样后续 D 段脚本按 `--previous_checkpoint_step 40000` 能正确找到所有模块文件
#
# 用法:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   bash run_v41r_resumeC.sh 2>&1 | tee train_v41r_resumeC.log
#
#   ⚠️ 运行前务必确认：**没有别的进程在往 rt_v41_r_taskC* 写**（同一 run 两个进程会互相踩）
#      pkill -f "run_id_override rt_v41_r_taskC"    # 仅在你确认原进程已死/要杀掉时执行
#
# 环境变量: GPUS(默认 4,5,6,7) BATCH_SIZE(默认 2) STEPS_REMAIN(默认 30000) NO_RENAME(1=不改名)
# =============================================================================

set -u

: "${LOGS_ROOT:?请先 source server_env.sh（LOGS_ROOT 未设置）}"
: "${VLA_PATH:?请先 source server_env.sh（VLA_PATH 未设置）}"

TRAIN_DIR="/mnt/data/pengshengdi/openvla-oft"
BUF_P="$LOGS_ROOT/replay_buffers"
FROM_DIR="${FROM_DIR:-$LOGS_ROOT/rt_v41_r_taskC--10000_chkpt}"
FROM_STEP="${FROM_STEP:-10000}"
TMP_RID="${TMP_RID:-rt_v41_r_taskC_resume}"
TARGET_DIR="${TARGET_DIR:-$LOGS_ROOT/rt_v41_r_taskC--40000_chkpt}"
STEPS_REMAIN="${STEPS_REMAIN:-30000}"
DATASET="${DATASET:-aloha_stack_bowls_two_clean}"
STAGE="${STAGE:-3}"
NO_RENAME="${NO_RENAME:-0}"

GPUS="${GPUS:-4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-2}"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NPROC=${#GPU_ARR[@]}
GRAD_ACCUM=$((8 / (BATCH_SIZE * NPROC)))

echo "================ 前置检查 ================"
[ -d "$FROM_DIR" ] || { echo "[FAIL] 源 ckpt 不存在: $FROM_DIR"; exit 1; }
[ -d "$BUF_P/taskA" ] && [ -d "$BUF_P/taskB" ] || { echo "[FAIL] 缺 replay buffer: $BUF_P/task{A,B}"; exit 1; }

echo "[INFO] 源 ckpt 内容检查（必须齐）:"
for f in cl_lora_adapter.pt "action_head--${FROM_STEP}_checkpoint.pt" \
         "vision_backbone--${FROM_STEP}_checkpoint.pt" "proprio_projector--${FROM_STEP}_checkpoint.pt" \
         dataset_statistics.json; do
    if [ -e "$FROM_DIR/$f" ]; then echo "   [OK] $f"; else echo "   [MISSING] $f"; MISS=1; fi
done
[ "${MISS:-0}" = "1" ] && { echo "[FAIL] 源 ckpt 文件不全 —— 换一个可用的中间 ckpt（如 --20000_chkpt）或改用 FROM_DIR/FROM_STEP"; exit 1; }
echo "[OK] 续训配置: stage=$STAGE dataset=$DATASET | 剩余 $STEPS_REMAIN 步 | 有效 batch=8 (batch=$BATCH_SIZE × $NPROC 卡)"
echo "[OK] previous= $FROM_DIR @$FROM_STEP | skip_reinit=True（不清零当前 stage 的 specific-B）"
echo "[OK] 输出 run_id= $TMP_RID，训完对齐为 $TARGET_DIR"
[ -e "$LOGS_ROOT/${TMP_RID}--${STEPS_REMAIN}_chkpt" ] && echo "[WARN] $LOGS_ROOT/${TMP_RID}--${STEPS_REMAIN}_chkpt 已存在（会被本次同名 step 目录覆盖）"

cd "$TRAIN_DIR" || { echo "[FAIL] 目录不存在: $TRAIN_DIR"; exit 1; }

env CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline \
torchrun --standalone --nproc_per_node $NPROC vla-scripts/train_cl_lora.py \
    --run_root_dir "$LOGS_ROOT" --run_id_override "$TMP_RID" \
    --max_steps "$STEPS_REMAIN" --save_freq 10000 \
    --vla_path "$VLA_PATH" --dataset_name "$DATASET" --stage "$STAGE" \
    --previous_checkpoint_dir "$FROM_DIR" --previous_checkpoint_step "$FROM_STEP" \
    --skip_reinit True \
    --batch_size "$BATCH_SIZE" --grad_accumulation_steps "$GRAD_ACCUM" --learning_rate 5e-4 \
    --lr_warmup_steps 200 --num_steps_before_decay 100000 \
    --use_cl_lora True --lora_rank 16 --shared_depth 8 --first_lora_layer 16 \
    --orthogonal_init True --freeze_a True --use_block_scale True --freeze_specific_a False \
    --specific_a_trainable_layers "24-31" --specific_a_freeze_action_head True \
    --bank_film_mode film \
    --use_kd False --freeze_film_stage2 False --film_lr_scale 0.2 --lambda_kd 0.2 \
    --use_replay True --replay_buffer_dirs "$BUF_P/taskA,$BUF_P/taskB" \
    --replay_every_n_steps 4 --replay_loss_weight 0.5 \
    --image_aug True --use_proprio True --use_film True --num_images_in_input 3
rc=$?
[ $rc -ne 0 ] && { echo "[FAIL] 续训失败 (rc=$rc)"; exit 1; }
echo "[OK] 续训完成 -> $LOGS_ROOT/${TMP_RID}--${STEPS_REMAIN}_chkpt"

# ---------- 对齐命名（关键：内部文件名里的步号也要改） ----------
SRC="$LOGS_ROOT/${TMP_RID}--${STEPS_REMAIN}_chkpt"
if [ "$NO_RENAME" = "1" ]; then
    echo "[SKIP] NO_RENAME=1，未改名。后续 D 段请手动把 CKPT_C 指到: $SRC @${STEPS_REMAIN}"
else
    [ -d "$SRC" ] || { echo "[FAIL] 找不到续训产物: $SRC"; exit 1; }
    if [ -e "$TARGET_DIR" ]; then
        echo "[WARN] $TARGET_DIR 已存在 —— 不覆盖，产物保留为 $SRC（@${STEPS_REMAIN}）"; exit 0
    fi
    cp -r "$SRC" "$TARGET_DIR"
    ( cd "$TARGET_DIR" && for f in *"${STEPS_REMAIN}"*; do [ -e "$f" ] && mv "$f" "${f//${STEPS_REMAIN}/40000}"; done )
    echo "[OK] 已对齐: $TARGET_DIR"
    echo "     内容:"
    ls -1 "$TARGET_DIR" | sed 's/^/       /'
fi

echo ""
echo "下一步（在 C ckpt 上评估 A/B/C，与无回放臂的 v41ConC 同口径对照）:"
echo "  cd /mnt/data/pengshengdi/RoboTwin-main"
echo "  FILM_GAMMA=0 bash policy/openvla-oft/eval_sequence.sh $TARGET_DIR 4,4,5,5,6,6,7,7 50 v41_rConC A B C"
echo "  bash /mnt/data/pengshengdi/show_eval_rates.sh v41_rConC"

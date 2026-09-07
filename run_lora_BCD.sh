#!/bin/bash
# =============================================================================
# run_lora_BCD.sh — 普通 LoRA 支线（标准 PEFT LoRA 顺序微调，RoboTwin）
#
# 定位: CL 最朴素基线 —— 无共享/特定层隔离、无冻结、无 bank、无回放。
#   A 微调 → B 从 A 的 merged ckpt 续 → C → D；旧任务权重被新任务覆盖，
#   评估 D ckpt 的 A/B/C = 灾难性遗忘基线的直接测法。
#   与 CL-LoRA 支线 (v39b4) / 方法 (v39r2d) 形成对比层级:
#     普通 LoRA (全漂移) < CL-LoRA 无回放 < CL-LoRA + 回放
#
# 实现: vla-scripts/finetune.py (主仓库版, 与当前 prismatic 同代)
#   use_cl_lora=False 默认 → PEFT LoraConfig(target_modules="all-linear")
#   merge_lora_during_training=True → 每 checkpoint 存 merged 全模型
#   stage 衔接 = 下一阶段 --vla_path 指向上阶段 merged ckpt 目录
#
# 用法（tmux 里前台跑, 训完自动评估）:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   tmux new -s trainLoRA
#   bash run_lora_BCD.sh 2>&1 | tee train_lora_BCD.log
#
# 产物: $LOGS_ROOT/rt_lora_taskA/B/C/D--{step}_chkpt (merged 全模型)
# 结果: eval_result/loraD_task*.log 的 Merged success rate
# =============================================================================

set -u

TRAIN_DIR="/mnt/data/pengshengdi/openvla-oft"
GPUS="${GPUS:-4,5,6,7}"
NPROC=0
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NPROC=${#GPU_ARR[@]}
BATCH_SIZE="${BATCH_SIZE:-2}"          # 总 batch = batch_size × nproc = 8 (与 CL 有效 batch 对齐)

# ---------- 前置检查 ----------
check_env() {
    [ -n "${VLA_PATH:-}" ]  || { echo "[FAIL] VLA_PATH 未设置 —— 请先: source server_env.sh"; exit 1; }
    [ -n "${LOGS_ROOT:-}" ] || { echo "[FAIL] LOGS_ROOT 未设置 —— 请先: source server_env.sh"; exit 1; }
    [ -f "$VLA_PATH/config.json" ] || { echo "[FAIL] 基座模型不存在: $VLA_PATH/config.json"; exit 1; }
}

echo "================ 前置检查 ================"
check_env
echo "[OK] VLA_PATH  = $VLA_PATH (Stage 1 起点)"
echo "[OK] LOGS_ROOT = $LOGS_ROOT"
echo "[OK] GPUS=$GPUS | NPROC=$NPROC | batch_size=$BATCH_SIZE (总 batch=$((BATCH_SIZE*NPROC)))"
echo "[OK] 配置: 普通 PEFT LoRA (all-linear, rank16) + FiLM + proprio, 无 CL 机制"
echo "============ 开始 普通 LoRA 顺序微调 A -> B -> C -> D ============"

cd "$TRAIN_DIR" || { echo "[FAIL] 目录不存在: $TRAIN_DIR"; exit 1; }

run_finetune() {  # $1=stage  $2=dataset  $3=run_id  $4=max_steps  $5=vla_path
    local stage=$1 ds=$2 rid=$3 max_steps=$4 vla=$5
    echo ""
    echo "############ Stage $stage : $ds (from $vla) ############"
    env CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline \
    torchrun --standalone --nproc_per_node $NPROC vla-scripts/finetune.py \
        --vla_path "$vla" \
        --data_root_dir datasets/rlds \
        --dataset_name "$ds" \
        --run_root_dir "$LOGS_ROOT" \
        --run_id_override "$rid" \
        --max_steps "$max_steps" --save_freq 10000 \
        --batch_size "$BATCH_SIZE" --learning_rate 5e-4 \
        --lr_warmup_steps 200 --num_steps_before_decay 100000 \
        --use_l1_regression True --use_diffusion False \
        --use_film True --use_proprio True --num_images_in_input 3 \
        --use_lora True --lora_rank 16 --lora_dropout 0.0 \
        --merge_lora_during_training True --image_aug True
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[FAIL] Stage $stage ($ds) 训练失败 (exit=$rc), 终止后续 Stage"
        exit $rc
    fi
    echo "[OK] Stage $stage ($ds) 完成 -> $LOGS_ROOT/$rid--${max_steps}_chkpt"
}

# Stage 1: A (从基座模型)
run_finetune 1 aloha_handover_mic_clean rt_lora_taskA 30000 "$VLA_PATH"

# Stage 2: B (从 A 的 merged ckpt 续训)
run_finetune 2 aloha_grab_roller_clean rt_lora_taskB 40000 "$LOGS_ROOT/rt_lora_taskA--30000_chkpt"

# Stage 3: C
run_finetune 3 aloha_stack_bowls_two_clean rt_lora_taskC 40000 "$LOGS_ROOT/rt_lora_taskB--40000_chkpt"

# Stage 4: D
run_finetune 4 aloha_open_laptop_clean rt_lora_taskD 40000 "$LOGS_ROOT/rt_lora_taskC--40000_chkpt"

echo ""
echo "==== 普通 LoRA 顺序微调全部完成 ===="

# ---------- 合并历史任务 dataset_statistics (评估旧任务需要各任务 unnorm key) ----------
CKPT_D="$LOGS_ROOT/rt_lora_taskD--40000_chkpt"
python - <<'PYEOF'
import json, os
log = os.environ.get("LOGS_ROOT", "")
target = os.path.join(log, "rt_lora_taskD--40000_chkpt", "dataset_statistics.json")
merged = {}
if os.path.isfile(target):
    with open(target) as f:
        merged = json.load(f)
for tag, step in [("rt_lora_taskA", 30000), ("rt_lora_taskB", 40000), ("rt_lora_taskC", 40000)]:
    p = os.path.join(log, f"{tag}--{step}_chkpt", "dataset_statistics.json")
    if os.path.isfile(p):
        with open(p) as f:
            for k, v in json.load(f).items():
                merged.setdefault(k, v)
with open(target, "w") as f:
    json.dump(merged, f, indent=2)
print(f"[OK] dataset_statistics 合并完成: {list(merged.keys())}")
PYEOF

# ---------- 自动评估 (eval_task_id=0: 普通 LoRA 无 bank, 当前权重即旧任务成绩) ----------
if [ "${EVAL_AFTER_TRAIN:-1}" = "1" ]; then
    echo ""
    echo "==== 自动开始全任务评估 (eval_task_id=0, ${EVAL_GPUS:-4,4,5,5,6,6,7,7}) ===="
    declare -A TASK_NAME=( [A]=handover_mic [B]=grab_roller [C]=stack_bowls_two [D]=open_laptop )
    declare -A UNNORM=( [A]=aloha_handover_mic_clean [B]=aloha_grab_roller_clean [C]=aloha_stack_bowls_two_clean [D]=aloha_open_laptop_clean )
    for t in A B C D; do
        bash /mnt/data/pengshengdi/RoboTwin-main/policy/openvla-oft/eval_multi_gpu.sh \
            "${TASK_NAME[$t]}" demo_clean "$CKPT_D" 0 "${EVAL_GPUS:-4,4,5,5,6,6,7,7}" \
            "${UNNORM[$t]}" 0 50 "loraD_$t" 0 \
            > "/mnt/data/pengshengdi/train_lora_eval_${t}.log" 2>&1
        echo "Task $t: $(grep 'Merged success rate' /mnt/data/pengshengdi/train_lora_eval_${t}.log)"
    done
    echo ""
    echo "==== 普通 LoRA 支线评估汇总 ===="
    for t in A B C D; do
        echo "Task $t: $(grep 'Merged success rate' /mnt/data/pengshengdi/train_lora_eval_${t}.log)"
    done
else
    echo "[SKIP] EVAL_AFTER_TRAIN=0, 跳过自动评估"
fi

#!/bin/bash
# =============================================================================
# run_lora_BCD_afterA.sh — 普通 LoRA 支线 Stage 2-4 (B/C/D) + 训完自动全任务评估
#
# 前提: A 已达标 (rt_lora_taskA--30000_chkpt 存在, 建议 run_loraA.sh 判定 PASS)
# 流程: B (从 A merged 续, 40000) → C → D → 合并 stats → 全任务评估 (eval_task_id=0)
# 配置: 与 A 相同 (lora_scope=cl, rank16, FiLM 训练——普通 LoRA 无冻结机制,
#       旧任务权重被新任务覆盖 = 灾难性遗忘基线)
#
# 用法（tmux 里前台跑）:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   tmux new -s loraBCD
#   bash run_lora_BCD_afterA.sh 2>&1 | tee train_lora_BCD2.log
#
# 产物: $LOGS_ROOT/rt_lora_taskB/C/D--40000_chkpt
# 结果: 末尾自动打印 A/B/C/D 全任务评估汇总
# =============================================================================

set -u

TRAIN_DIR="/mnt/data/pengshengdi/openvla-oft"
CKPT_A="$LOGS_ROOT/rt_lora_taskA--30000_chkpt"
GPUS="${GPUS:-4,5,6,7}"
EVAL_GPUS="${EVAL_GPUS:-4,4,5,5,6,6,7,7}"

check_env() {
    [ -n "${LOGS_ROOT:-}" ] || { echo "[FAIL] LOGS_ROOT 未设置 —— 请先: source server_env.sh"; exit 1; }
}

echo "================ 前置检查 ================"
check_env
[ -d "$CKPT_A" ] || { echo "[FAIL] A checkpoint 不存在: $CKPT_A —— 先跑 run_loraA.sh"; exit 1; }
ls "$CKPT_A"/config.json >/dev/null 2>&1 || { echo "[FAIL] A merged ckpt 不完整"; exit 1; }
echo "[OK] A ckpt = $CKPT_A (B/C/D 的起点)"
echo "[OK] GPUS=$GPUS | EVAL_GPUS=$EVAL_GPUS"
echo "============ 开始 普通 LoRA B -> C -> D 顺序微调 ============"

cd "$TRAIN_DIR" || { echo "[FAIL] 目录不存在: $TRAIN_DIR"; exit 1; }

run_finetune() {  # $1=stage  $2=dataset  $3=run_id  $4=max_steps  $5=vla_path
    local stage=$1 ds=$2 rid=$3 max_steps=$4 vla=$5
    if [ -d "$LOGS_ROOT/$rid--${max_steps}_chkpt" ]; then
        echo "[SKIP] $rid--${max_steps}_chkpt 已存在, 跳过训练"
        return 0
    fi
    echo ""
    echo "############ Stage $stage : $ds (from $vla) ############"
    env CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline \
    torchrun --standalone --nproc_per_node 4 vla-scripts/finetune.py \
        --vla_path "$vla" \
        --data_root_dir datasets/rlds \
        --dataset_name "$ds" \
        --run_root_dir "$LOGS_ROOT" --run_id_override "$rid" \
        --max_steps "$max_steps" --save_freq 10000 \
        --batch_size 2 --learning_rate 5e-4 \
        --lr_warmup_steps 200 --num_steps_before_decay 100000 \
        --use_l1_regression True --use_diffusion False \
        --use_film True --use_proprio True --num_images_in_input 3 \
        --use_lora True --lora_rank 16 --lora_scope cl --lora_dropout 0.0 \
        --merge_lora_during_training True --image_aug True
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[FAIL] Stage $stage ($ds) 训练失败 (exit=$rc), 终止后续 Stage"
        exit $rc
    fi
    echo "[OK] Stage $stage ($ds) 完成 -> $LOGS_ROOT/$rid--${max_steps}_chkpt"
}

# Stage 2: B (从 A merged 续训)
run_finetune 2 aloha_grab_roller_clean rt_lora_taskB 40000 "$CKPT_A"

# Stage 3: C
run_finetune 3 aloha_stack_bowls_two_clean rt_lora_taskC 40000 "$LOGS_ROOT/rt_lora_taskB--40000_chkpt"

# Stage 4: D
run_finetune 4 aloha_open_laptop_clean rt_lora_taskD 40000 "$LOGS_ROOT/rt_lora_taskC--40000_chkpt"

echo ""
echo "==== 普通 LoRA B/C/D 顺序微调完成 ===="

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

# ---------- 自动全任务评估 (eval_task_id=0) ----------
if [ "${EVAL_AFTER_TRAIN:-1}" = "1" ]; then
    declare -A TASK_NAME=( [A]=handover_mic [B]=grab_roller [C]=stack_bowls_two [D]=open_laptop )
    declare -A UNNORM=( [A]=aloha_handover_mic_clean [B]=aloha_grab_roller_clean [C]=aloha_stack_bowls_two_clean [D]=aloha_open_laptop_clean )
    echo ""
    echo "==== 自动开始全任务评估 (eval_task_id=0, 8 worker) ===="
    results=()
    for t in A B C D; do
        bash /mnt/data/pengshengdi/RoboTwin-main/policy/openvla-oft/eval_multi_gpu.sh \
            "${TASK_NAME[$t]}" demo_clean "$CKPT_D" 0 "$EVAL_GPUS" \
            "${UNNORM[$t]}" 0 50 "loraD_${t}" 0 \
            > "/tmp/lora_eval_${t}.log" 2>&1
        rate=$(grep "Merged success rate" "/tmp/lora_eval_${t}.log" | tail -1)
        if [ -z "$rate" ]; then
            results+=("Task $t: FAILED")
        else
            results+=("Task $t: $rate")
        fi
    done
    echo ""
    echo "==================== 普通 LoRA 支线评估汇总 ===================="
    for line in "${results[@]}"; do
        echo "  $line"
    done
    echo ""
    echo "对照: CL-LoRA 无回放 v39b4 = 0-0.57-0-0.85; 方法 v39r2d = 0.62-0.88-0.88-0.80"
else
    echo "[SKIP] EVAL_AFTER_TRAIN=0, 跳过自动评估"
fi

#!/bin/bash
# =============================================================================
# server_env.sh — RoboTwin CL-LoRA 训练/评估前置环境准备（用 source 执行）
#
# 用法（每次新开终端, 在服务器上执行）:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#
# 作用:
#   1) 激活 conda 环境 (openvla, Python 3.10)
#   2) 自动探测基座模型路径, export VLA_PATH / OPENVLA_BASE_PATH
#   3) export LOGS_ROOT / PYTORCH_ALLOC_CONF / WANDB_MODE
#   4) 打印环境确认: 模型 / 代码版本 / torch / GPU 4,5 号卡状态
#
# 规则:
#   - 若你已经手动 export 过某个变量, 脚本保留你的值(不覆盖)。
#   - 若探测不到基座模型, 脚本会明确警告, 按提示手动 export 后重跑。
#   - 必须 source (不能 bash server_env.sh), 否则 export 不生效。
# =============================================================================

REPO_ROOT="/mnt/data/pengshengdi"
CONDA_ENV="$REPO_ROOT/openvla"

echo "================ RoboTwin CL-LoRA 环境准备 ================"

# ---------- 1) conda 环境 ----------
if command -v conda >/dev/null 2>&1; then
    conda activate "$CONDA_ENV" 2>/dev/null \
        && echo "[OK] conda env: $CONDA_ENV" \
        || echo "[WARN] conda activate 失败, 请手动执行: conda activate $CONDA_ENV"
else
    echo "[WARN] 未找到 conda 命令, 请手动执行: conda activate $CONDA_ENV"
fi

# ---------- 2) 基座模型自动探测 ----------
CANDIDATES=(
    "$REPO_ROOT/models/openvla-7b"
    "$REPO_ROOT/openvla-7b"
    "/root/autodl-tmp/models/openvla-7b"
    "$HOME/models/openvla-7b"
)
if [ -z "${VLA_PATH:-}" ]; then
    for c in "${CANDIDATES[@]}"; do
        if [ -f "$c/config.json" ]; then
            export VLA_PATH="$c"
            echo "[OK] 探测到基座模型: $c"
            break
        fi
    done
    if [ -z "${VLA_PATH:-}" ]; then
        echo "[WARN] 未探测到基座模型! 请先找到 openvla-7b/config.json 所在目录, 然后手动执行:"
        echo "       export VLA_PATH=/path/to/openvla-7b"
        echo "       export OPENVLA_BASE_PATH=/path/to/openvla-7b"
        echo "       再重新 source server_env.sh"
    fi
fi
export OPENVLA_BASE_PATH="${OPENVLA_BASE_PATH:-${VLA_PATH:-}}"

# ---------- 3) 其余必需环境变量 ----------
export LOGS_ROOT="${LOGS_ROOT:-$REPO_ROOT/LOGS-RT}"        # 训练 checkpoint 输出目录
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# ---------- 4) 确认信息 ----------
echo ""
echo "---- 环境变量 ----"
echo "VLA_PATH           = ${VLA_PATH:-<未设置!>}"
echo "OPENVLA_BASE_PATH  = ${OPENVLA_BASE_PATH:-<未设置!>}"
echo "LOGS_ROOT          = $LOGS_ROOT"
echo "PYTORCH_ALLOC_CONF  = $PYTORCH_ALLOC_CONF"
echo "WANDB_MODE         = $WANDB_MODE"

echo ""
echo "---- 基座模型 ----"
if [ -n "${VLA_PATH:-}" ] && [ -f "$VLA_PATH/config.json" ]; then
    ls -la "$VLA_PATH/config.json"
else
    echo "[FAIL] 基座模型 config.json 不存在 —— 训练(--vla_path)和评估(OPENVLA_BASE_PATH)都会报错!"
fi

echo ""
echo "---- 代码版本 (服务器) ----"
git -C "$REPO_ROOT" log --oneline -1

echo ""
echo "---- torch / CUDA ----"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu_avail', torch.cuda.is_available())"

echo ""
echo "---- GPU 状态 (关注 4/5 号卡是否空闲) ----"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

echo ""
echo "==== 环境准备完成, 可以开始训练/评估 ===="

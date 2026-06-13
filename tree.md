# CL-LoRA on OpenVLA

基于 OpenVLA-7B 的**持续学习 LoRA（CL-LoRA）**微调项目，支持 LIBERO 仿真基准和 ALOHA 真实机器人场景。

## 项目概述

本项目实现了一套用于视觉-语言-动作（VLA）模型的持续学习微调管线：

- **基础模型**：OpenVLA-7B（SigLIP + DINOv2 融合视觉编码器 → 投影 MLP → LLaMa-2 7B LLM）
- **微调方法**：标准 LoRA / CL-LoRA（持续学习 LoRA，含正交初始化、冻结 A 矩阵、分块缩放门控）
- **持续学习策略**：教师模型快照 + 知识蒸馏损失 + 基于原型的关键帧回放
- **动作预测头**：L1 回归头（MLP → 连续动作块）或扩散动作头（DDIM）
- **评估场景**：LIBERO 模拟基准（Spatial / Object / Goal / 10 / 90）+ ALOHA 双臂真实机器人

## 目录结构

```
.
├── openvla-oft/                          # ★ 主代码目录
│   ├── vla-scripts/                      # 训练、部署、数据处理脚本
│   │   ├── finetune.py                   #   主训练脚本（DDP/FSDP，~50 个可配参数）
│   │   ├── cl_lora.py                    #   CL-LoRA 核心实现（CLLoRALinear + 注入函数）
│   │   ├── deploy.py                     #   FastAPI 模型服务器（/act 接口，用于 ALOHA 部署）
│   │   ├── run_libero_eval.py            #   LIBERO 评估入口
│   │   ├── build_replay_buffer_openvla.py #  回放缓冲构建（物理分割 + 原型选择 + 关键帧保存）
│   │   ├── replay_dataset.py             #   回放数据 PyTorch Dataset 加载器
│   │   ├── merge_lora_weights_and_save.py #  LoRA 权重合并与保存
│   │   └── check_rlds_keys.py            #   RLDS 数据集结构诊断工具
│   │
│   ├── cllora代码/                       # CL-LoRA 实验代码
│   │   ├── cl_lora.py                    #   CL-LoRA 模块（同 vla-scripts/cl_lora.py）
│   │   ├── dataset.py                    #   RLDS 数据集加载与标准化（29KB 核心模块）
│   │   ├── finetune.py                   #   训练脚本（同 vla-scripts/finetune.py）
│   │   ├── openvla_utils.py              #   模型加载/推理/工具函数（31KB）
│   │   ├── run_libero_eval.py            #   LIBERO 评估脚本
│   │   └── traineval_txt                 #   训练+评估参考命令
│   │
│   ├── 90成功率代码/                     # 标准 LoRA 基线（LIBERO-Spatial 达 90%+ 成功率）
│   │   ├── dataset.py                    #   同上
│   │   ├── finetune.py                   #   同上
│   │   ├── run_libero_eval.py            #   同上
│   │   └── traineval.txt                 #   过拟合基线参考命令
│   │
│   ├── pi0.5代码/                        # PI0.5 参考实现（JAX/Flax，CL-LoRA + 原型回放方法论来源）
│   │   ├── pi0.py                        #   PI0 顶层模型（双专家：PaliGemma 2B + Gemma 300M）
│   │   ├── gemma.py                      #   Gemma Transformer + CL-LoRA 支持
│   │   ├── lora.py                       #   LoRA 模块（正交初始化 / freeze_A / block_scale）
│   │   ├── config.py                     #   训练配置系统（Assets/Data/Rehearsal/Train）
│   │   ├── train_cl_lora.py              #   CL-LoRA 训练脚本（教师模型 + KD + 回放损失）
│   │   └── build_replay_buffer.py        #   离线回放缓冲构建（物理分割 + 原型 + 关键帧）
│   │
│   ├── prismatic/                        # ★ OpenVLA 模型架构核心库
│   │   ├── conf/
│   │   │   ├── models.py                 #   模型配置（视觉/语言 backbone、架构规格）
│   │   │   └── vla.py                    #   VLA 策略配置（数据混合、冻结策略、混合精度）
│   │   ├── models/
│   │   │   ├── load.py                   #   预训练模型加载入口（load / load_vla）
│   │   │   ├── materialize.py            #   backbone 注册表工厂
│   │   │   ├── action_heads.py           #   动作预测头（L1Regression / Diffusion + 正弦时间嵌入）
│   │   │   ├── film_vit_wrapper.py       #   FiLM 条件 ViT（语言嵌入调制视觉特征）
│   │   │   ├── projectors.py             #   投影器（ProprioProjector / NoisyActionProjector）
│   │   │   ├── backbones/
│   │   │   │   ├── vision/               #   视觉 backbone（CLIP/SigLIP/DINOv2/DinoSigLIP/DinoCLIP）
│   │   │   │   └── llm/                  #   语言 backbone（LLaMa-2 / Mistral / Phi + 提示构建器）
│   │   │   ├── vlas/
│   │   │   │   └── openvla.py            #   OpenVLA 类（动作分词 + 预测 + 归一化）
│   │   │   └── vlms/
│   │   │       └── prismatic.py          #   PrismaticVLM（视觉 → 投影 → LLM 架构）
│   │   ├── extern/hf/
│   │   │   ├── configuration_prismatic.py #  HuggingFace PretrainedConfig
│   │   │   ├── modeling_prismatic.py     #   HuggingFace PreTrainedModel（完整前向逻辑）
│   │   │   └── processing_prismatic.py   #   HuggingFace Processor（图像/文本处理）
│   │   ├── vla/
│   │   │   ├── action_tokenizer.py       #   动作分词器（离散化 → 最少使用的 token）
│   │   │   ├── constants.py              #   平台常量（LIBERO/ALOHA/BRIDGE 动作维度等）
│   │   │   └── datasets/                 #   RLDS 数据集管线（OXE 配置/变换/混合物）
│   │   ├── training/                     #   训练基础设施（DDP/FSDP 策略、指标、损失计算）
│   │   └── util/                         #   工具函数（批处理、投影器、数据工具）
│   │
│   ├── experiments/                      # 评估实验
│   │   ├── logs/                         #   ~90 个评估日志（libero_spatial/object/goal）
│   │   └── robot/
│   │       ├── openvla_utils.py          #   评估核心工具（模型加载/推理管线）
│   │       ├── libero/                   #   LIBERO 仿真评估（环境/视频/数据再生）
│   │       └── aloha/                    #   ALOHA 真机评估（客户端-服务器架构）
│   │
│   ├── 任务指引/                          # 中文任务指导文档
│   │   ├── openvla_smolvla_quick_checklist.md  # 迁移清单（模型/训练/回放/实验顺序）
│   │   └── pi05_cl_lora_replay_README.md       # CL-LoRA + 回放完整参考（1036 行，含行号引用）
│   │
│   ├── LIBERO/                           # LIBERO 基准套件（第三方，含 assets/ 被 gitignore）
│   ├── pyproject.toml                    # 项目配置
│   ├── LICENSE                           # MIT License
│   ├── README.md                         # 上游项目 README（英文）
│   ├── SETUP.md                          # 环境安装指南
│   ├── LIBERO.md                         # LIBERO 使用文档
│   └── ALOHA.md                          # ALOHA 使用文档
│
├── modified_libero_rlds/                 # 修改后的 LIBERO 数据集（RLDS 格式，tfrecord 被 gitignore）
│   ├── libero_spatial_no_noops/          #   过滤 no-op 后的 Spatial 任务
│   ├── libero_object_no_noops/           #   过滤 no-op 后的 Object 任务
│   ├── libero_goal_no_noops/             #   过滤 no-op 后的 Goal 任务
│   ├── libero_10_no_noops/               #   过滤 no-op 后的 Long 任务
│   ├── libero_spatial/                   #   原始 Spatial 数据集
│   └── README.md                         #   数据集说明
│
├── LOGS/                                 # 训练日志与检查点（被 gitignore）
├── models/                               # 预训练模型权重（被 gitignore）
├── hf_cache/                             # HuggingFace 缓存（被 gitignore）
├── conda_envs/                           # Conda 环境（被 gitignore）
└── .gitignore
```

## 核心模块说明

### 1. 模型架构（`prismatic/`）

```
视觉编码器                     投影器                   语言模型
┌──────────────────┐     ┌──────────┐     ┌─────────────────────┐
│ DinoSigLIP (ViT) │ ──► │ MLP/FC   │ ──► │ LLaMa-2 7B (32 层)  │
│ ┌──────────────┐ │     └──────────┘     │                     │
│ │  SigLIP ViT  │ │                      │  ┌───────────────┐  │
│ │  DINOv2 ViT  │ │    ┌──────────┐      │  │ LoRA / CL-LoRA│  │
│ │  (融合特征)   │ │    │ FiLM 调制 │◄─────┼──│ (适配层)      │  │
│ └──────────────┘ │    └──────────┘      │  └───────────────┘  │
└──────────────────┘                      └─────────┬───────────┘
                                                    │
                                          ┌─────────▼───────────┐
                                          │   动作预测头          │
                                          │   L1Regression /     │
                                          │   Diffusion DDIM      │
                                          └───────────────────────┘
```

- **`prismatic/models/backbones/vision/`** — 视觉编码器实现，支持多种 ViT 变体，OpenVLA 使用 DinoSigLIP 融合骨干
- **`prismatic/models/backbones/llm/`** — LLM 骨干封装（LLaMa-2 / Mistral / Phi），含对话模板提示构建
- **`prismatic/models/vlms/prismatic.py`** — PrismaticVLM 主体，视觉→投影→语言的完整前向逻辑
- **`prismatic/models/vlas/openvla.py`** — OpenVLA 类，继承 PrismaticVLM，增加动作分词与预测
- **`prismatic/models/action_heads.py`** — 动作头：L1 回归头（直接输出连续动作）和扩散头（DDIM 去噪）
- **`prismatic/models/film_vit_wrapper.py`** — FiLM 条件化：用语言嵌入的线性投影调制视觉特征 `x' = (1+γ)x + β`
- **`prismatic/extern/hf/modeling_prismatic.py`** — HuggingFace 格式模型，支持 `.generate()` 并行解码

### 2. CL-LoRA 核心实现（`openvla-oft/vla-scripts/cl_lora.py`）

```
CL-LoRA 架构
┌──────────────────────────────────────────────┐
│  共享浅层 (shared shallow layers)              │
│  ┌──────────────────────────────────────┐    │
│  │  LoRA-A: 正交初始化 → 冻结 (防遗忘)    │    │
│  │  LoRA-B: 零初始化 → 可训练             │    │
│  └──────────────────────────────────────┘    │
│                                              │
│  特定深层 (specific deep layers)              │
│  ┌──────────────────────────────────────┐    │
│  │  LoRA-A: 可训练                        │    │
│  │  LoRA-B: 可训练                        │    │
│  │  block_scale: 1.0 + 0.5*tanh(w) 门控  │    │
│  └──────────────────────────────────────┘    │
└──────────────────────────────────────────────┘
```

- **`CLLoRALinear`** — 替代 `nn.Linear` 的 CL-LoRA 线性层，区分共享/特定层，支持正交初始化和分块缩放
- **`inject_cl_lora_into_model()`** — 遍历 LlamaDecoderLayer，自动注入 CL-LoRA 到所有 Attention/QKV/FFN 线性层
- **共享层**：浅层（第 0~shared_depth-1 层）的 LoRA-A 正交初始化并冻结（保护旧知识）
- **特定层**：深层（第 shared_depth 层及以后）的 LoRA-A/B 均可训练，含可学习的 `block_scale` 门控参数

### 3. 训练管线（`openvla-oft/vla-scripts/finetune.py` + `cllora代码/dataset.py`）

```
RLDS 数据 → make_dataset_from_rlds() → 标准化/增强 → RLDSBatchTransform
    ↓
CL-LoRA 注入 → DDP/FSDP 训练 → 损失: L_task + λ_KD * L_KD + λ_replay * L_replay
    ↓
检查点保存 → evaluation
```

- **`finetune.py`** — 主训练脚本，`FinetuneConfig` 数据类控制所有超参数：
  - `lora_rank`：LoRA 秩（默认 32）
  - `use_cl_lora`：是否启用 CL-LoRA
  - `shared_depth` / `specific_depth`：共享/特定层深度分配
  - `orthogonal_init`：是否正交初始化共享层 LoRA-A
  - `freeze_a`：是否冻结共享层 LoRA-A
  - `use_block_scale`：是否使用分块缩放门控
  - `clip_weight`：正交初始化后的裁剪缩放
  - `action_head_type`：`l1_regression` 或 `ddim_diffusion`
- **`dataset.py`** — RLDS 数据加载管线：多数据集混合、图像/状态标准化、Goal Relabeling、任务增强

### 4. 持续学习策略

```
训练任务 A （CL-LoRA）
    ↓
构建回放缓冲 → 物理轨迹分割 → 原型计算 → Top-K 关键帧选择
    ↓
训练任务 B
    ↓
L_total = L_current_task + λ_KD * L_KD + λ_replay * L_replay
    ↑                       ↑                   ↑
    当前任务损失         知识蒸馏损失        回放损失
                    (教师=仅可训参数的快照)  (从回放缓冲采样)
```

- **知识蒸馏**：教师模型只保存 CL-LoRA 可训练参数的快照（非完整模型副本），KD 损失作用于新旧参数输出之间
- **原型回放**：物理规则分割轨迹（平动/转动/夹爪变化） → 视觉特征提取（仅视觉 backbone，绕过大 LLM） → 余弦相似度计算每段原型 → 选择最接近原型的 Top-K 帧

### 5. 评估系统

- **`experiments/robot/libero/run_libero_eval.py`** — LIBERO 仿真评估，支持 5 个任务套件，记录成功率和回放视频
- **`experiments/robot/aloha/run_aloha_eval.py`** — ALOHA 真机评估，客户端-服务器架构（模型运行在 GPU 服务器，客户端通过 `/act` 接口请求动作）
- **`experiments/logs/`** — ~90 个评估日志文件

## LoRA 底层原理

### 标准 LoRA

对于预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 使用低秩分解：

$$h = W_0 x + \frac{\alpha}{r} \cdot B A x$$

其中 $A \in \mathbb{R}^{r \times k}$（随机高斯初始化），$B \in \mathbb{R}^{d \times r}$（零初始化），$r \ll \min(d, k)$。

### CL-LoRA 改进

在标准 LoRA 基础上的三个关键改进：

| 机制 | 作用 | 实现位置 |
|------|------|----------|
| **正交初始化** | 共享层的 A 矩阵正交初始化，使不同任务的 LoRA 子空间互相正交，减少任务间干扰 | `lora.py:orthogonal_init()` |
| **冻结 A 矩阵** | 共享层冻结 A（`jax.lax.stop_gradient` / `requires_grad=False`），迫使新任务在正交子空间中学习，保护旧知识 | `cl_lora.py:freeze_a` |
| **分块缩放** | 每层学习独立的 `block_scale = 1.0 + 0.5·tanh(w)`，自动调节新旧任务间的平衡，避免手动调参 | `cl_lora.py:block_scale` |

### 层分配策略

```
LlamaDecoderLayer (32 层)
├── 第 0 ~ shared_depth-1 层  →  共享层（防遗忘，A 冻结）
└── 第 shared_depth ~ 31 层   →  特定层（可塑性，A/B 全可训练）
```

## 快速开始

### 环境安装

参见 `openvla-oft/SETUP.md`

```bash
conda create -n openvla-oft python=3.10
conda activate openvla-oft
pip install -e openvla-oft/
pip install flash-attn==2.7.3 --no-build-isolation
```

### 标准 LoRA 微调

```bash
cd openvla-oft
torchrun --standalone --nproc_per_node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir ../modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir ../LOGS/standard_lora \
  --lora_rank 16 \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --use_l1_regression \
  --num_steps 4000
```

### CL-LoRA 微调

```bash
torchrun --standalone --nproc_per_node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir ../modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir ../LOGS/cl_lora \
  --use_cl_lora \
  --shared_depth 16 \
  --orthogonal_init \
  --freeze_a \
  --use_block_scale \
  --clip_weight 1.0 \
  --lora_rank 32 \
  --num_steps 4000
```

### 评估

```bash
# LIBERO 仿真评估
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint ../LOGS/cl_lora/checkpoint-4000 \
  --task_suite_name libero_spatial \
  --use_cl_lora

# ALOHA 真机（先启动模型服务器）
python vla-scripts/deploy.py --pretrained_checkpoint ../LOGS/cl_lora/checkpoint-4000
# 然后在机器人端运行
python experiments/robot/aloha/run_aloha_eval.py --server_url http://<gpu-server>:8000
```

## 关键文件速查

| 你想了解的 | 去看这些文件 |
|-----------|-------------|
| 模型怎么前向传播的 | `prismatic/extern/hf/modeling_prismatic.py` |
| LoRA 怎么注入到模型里的 | `vla-scripts/cl_lora.py` |
| CL-LoRA 的训练损失怎么算的 | `pi0.5代码/train_cl_lora.py` |
| 数据怎么加载和处理的 | `cllora代码/dataset.py` |
| 动作怎么从离散 token 变回连续值的 | `prismatic/vla/action_tokenizer.py` |
| 回放缓冲怎么构建的 | `pi0.5代码/build_replay_buffer.py`（参考实现）或 `vla-scripts/build_replay_buffer_openvla.py`（OpenVLA 版） |
| LIBERO 评估怎么跑的 | `experiments/robot/libero/run_libero_eval.py` |
| 训练超参数有哪些 | `vla-scripts/finetune.py` 中的 `FinetuneConfig` |
| CL-LoRA 完整原理 | `任务指引/pi05_cl_lora_replay_README.md` |
| 迁移到 OpenVLA 怎么做 | `任务指引/openvla_smolvla_quick_checklist.md` |

## 相关论文

- OpenVLA: Kim et al. "OpenVLA: An Open-Source Vision-Language-Action Model." arXiv:2406.09246, 2024.
- OpenVLA-OFT: Kim, Finn, Liang. "Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success." arXiv:2502.19645, 2025.
- LoRA: Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR, 2022.

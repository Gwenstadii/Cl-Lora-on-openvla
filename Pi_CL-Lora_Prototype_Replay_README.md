# Pi 系列 VLA 持续学习：CL-Lora + Prototype Replay 机制说明

本文档用于说明当前仓库中面向 `VLA + Continual Learning` 的 `CL-Lora + Prototype Replay` 抗遗忘机制。文档重点是方法原理、代码实现入口、实验分支设定和复现实验时需要注意的流程，不包含具体实验结果数值。

适用范围：

| 项目 | 说明 |
| --- | --- |
| 主模型 | `pi0.5` / `pi0` 系列 VLA |
| 主实验集 | LIBERO 顺序任务学习 |
| 核心问题 | 新任务持续学习时，缓解旧任务灾难性遗忘，同时保持新任务可塑性 |
| 核心方法 | CL-Lora 结构稳定性 + Prototype Replay 少样本旧任务复习 |
| 对照支线 | 普通 LoRA、CL-Lora no replay、CL-Lora + uniform replay |

## 一、原理解析

### 1. 总体训练逻辑

当前方案把增量学习拆成连续阶段。每个阶段只训练当前任务，但从第二阶段开始可以选择是否加入旧任务回放。

| 机制 | 实现位置 | 作用 |
| --- | --- | --- |
| 普通训练入口 | `scripts/train.py` | 用于普通 LoRA baseline，不包含 CL-Lora KD / replay 逻辑 |
| CL-Lora 训练入口 | `scripts/train_cl_lora.py:337` | 支持任务 loss、KD loss、replay loss 的联合训练 |
| 总 loss 组合 | `scripts/train_cl_lora.py:437` | `loss_task + lambda_kd * loss_kd + replay_loss_weight * loss_rehearsal` |
| replay runtime 初始化 | `scripts/train_cl_lora.py:144` | 按配置创建旧任务回放数据流 |
| replay batch 注入 | `scripts/train_cl_lora.py:540` | 按 `replay_every_n_steps` 控制是否在当前 step 加入回放 |

注意：普通 LoRA 支线应严格使用 `scripts/train.py`。如果普通 LoRA 误用 `scripts/train_cl_lora.py`，默认 KD / replay 相关逻辑可能改变训练目标，导致 baseline 不再干净。

### 2. CL-Lora 的结构稳定性

CL-Lora 的目标不是为每个任务保存一套独立 adapter，而是在同一套可持续更新的 LoRA 参数里增强结构稳定性。这样更贴近持续学习设定：模型在顺序接收新任务时仍保持单一策略参数集合。

| 机制 | 实现位置 | 作用 |
| --- | --- | --- |
| LoRA 配置扩展 | `src/openpi/models/lora.py:12` | 定义 rank、alpha、dropout、共享层、特定层、block 权重和正交初始化选项 |
| 正交初始化 | `src/openpi/models/lora.py:34` | 为 CL-Lora 的低秩分支提供更稳定的初始方向 |
| Einsum LoRA 初始化 | `src/openpi/models/lora.py:52` | 对 attention einsum 层注入 LoRA 参数 |
| CL-Lora 正交 A / 零 B | `src/openpi/models/lora.py:61` | `lora_a` 使用正交初始化，`lora_b` 初始化为 0，减少一开始对原模型输出的扰动 |
| `freeze_a` | `src/openpi/models/lora.py:75` | 通过 `stop_gradient` 固定 A 分支，使更新更集中在 B 分支 |
| `block_scale` | `src/openpi/models/lora.py:84` | 使用 block 权重调制 LoRA 输出强度 |
| FeedForward LoRA | `src/openpi/models/lora.py:133` | 对 MLP / FFN 层也支持同类 CL-Lora 注入 |

CL-Lora 的核心直觉是：让 LoRA 子空间具有更稳定的方向约束，避免每个新任务都完全改写已有 adapter 表示。它能提供一定结构抗遗忘能力，但实验上仍需要 replay 才能在多阶段持续学习中获得稳定旧任务保持。

### 3. CL-Lora 在 pi 模型里的接入方式

CL-Lora 通过 Gemma variant 接入 pi 系列模型。配置中把前半部分层设为共享区域，把后半部分层设为更偏任务适配的区域，并启用 block 权重。

| 机制 | 实现位置 | 作用 |
| --- | --- | --- |
| Gemma variant 注册 | `src/openpi/models/gemma.py:57` | 支持 `gemma_2b_cl_lora` 和 `gemma_300m_cl_lora` |
| pi0.5 侧 CL-Lora | `src/openpi/models/gemma.py:111` | `gemma_2b_cl_lora`，rank / alpha 适配 pi0.5 |
| pi0 侧 CL-Lora | `src/openpi/models/gemma.py:127` | `gemma_300m_cl_lora`，rank / alpha 适配 pi0 |
| 共享层和特定层 | `src/openpi/models/gemma.py:121` | `shared_pos` 和 `specific_pos` 划分持续学习中的稳定区与适配区 |
| block 权重 | `src/openpi/models/gemma.py:123` | 允许不同 block 的 LoRA 更新贡献被调制 |

这部分决定了“CL-Lora”不是普通 LoRA 的命名变化，而是模型内部 LoRA 注入方式、初始化方式和层级划分共同形成的结构改造。

### 4. Balanced 冻结策略

普通 LoRA 只训练 LoRA 参数时，模型可能可塑性不足；如果放开太多主干参数，又容易破坏旧任务能力。因此仓库中使用 balanced freezing：冻结重主干，开放 LoRA 和少量动作侧轻量头。

| 机制 | 实现位置 | 作用 |
| --- | --- | --- |
| 严格 LoRA 冻结 | `src/openpi/models/pi0_config.py:120` | 只放开 LoRA 和可选 block 权重 |
| balanced 冻结 | `src/openpi/models/pi0_config.py:139` | 放开 LoRA、block 权重和 action-side 轻量投影层 |
| 动作侧头 | `src/openpi/models/pi0_config.py:154` | 包含 `action_in_proj`、`action_out_proj`、`time_mlp`、`state_proj` 等 |

balanced 冻结是当前四大实验分支的共同控制变量。它帮助模型在 LIBERO 任务上获得足够的新任务学习能力，同时尽量不破坏视觉语言主干。

### 5. 任务过滤与数据加载

LIBERO 数据是多任务集合。每个训练 config 通过 `target_task_name` 指定当前阶段任务，数据加载时会筛出对应 task 的 episode 和 frame。

| 机制 | 实现位置 | 作用 |
| --- | --- | --- |
| 数据配置读取 | `src/openpi/training/config.py:523` | 每个训练 config 绑定对应 data config |
| target task 过滤 | `src/openpi/training/data_loader.py:250` | 对 LeRobot 格式 LIBERO 数据执行任务级过滤 |
| episode 筛选 | `src/openpi/training/data_loader.py:269` | 扫描 episode 元信息，保留目标任务 episode |
| frame 子集构建 | `src/openpi/training/data_loader.py:307` | 输出目标任务对应的 frame 数和 episode 数 |
| 通用 data loader | `src/openpi/training/data_loader.py:391` | 创建训练用 Torch data loader |

因此每个阶段训练只看当前任务的完整数据；旧任务信息只通过 replay buffer 回来。

### 6. Prototype Replay 的构建原理

Prototype Replay 的目标是在少量旧任务 episode 中，选出最能代表旧任务关键物理过程的 replay 样本。它不是均匀抽帧，而是先按物理轨迹进行隐式 skill 分段，再在每段中选出最接近段原型的 top-K 帧。

构建脚本入口：

| 机制 | 实现位置 | 作用 |
| --- | --- | --- |
| prototype buffer 脚本 | `scripts/build_replay_buffer.py:1` | 离线构建 prototype replay buffer |
| episode budget | `scripts/build_replay_buffer.py:636` | `max_episodes` 控制用于构建 buffer 的旧任务 episode 数 |
| segment 生成 | `scripts/build_replay_buffer.py:310` | 根据边界构造隐式物理 segment |
| 运动统计 | `scripts/build_replay_buffer.py:327` | 计算 translation / rotation / gripper 变化 |
| 短段合并 | `scripts/build_replay_buffer.py:388` | 避免过碎 segment，同时保留关键 gripper 段 |
| 视觉特征抽取 | `scripts/build_replay_buffer.py:689` | 为每帧提取视觉特征 |
| segment prototype | `scripts/build_replay_buffer.py:695` | 对段内视觉特征做归一化均值，形成 prototype |
| top-K 选择 | `scripts/build_replay_buffer.py:700` | 每段选择与 prototype 余弦相似度最高的帧 |
| replay sample 保存 | `scripts/build_replay_buffer.py:725` | 保存与训练 data loader 兼容的 `.npz` 样本 |
| meta 信息保存 | `scripts/build_replay_buffer.py:774` | 保存分段参数、选择参数、样本统计和训练来源 |

默认科研叙事中的 few-shot 性主要是 episode 层面的 few-shot，即只用少量旧任务 episode 构建 buffer；每段仍可保持 top-K 不变，以免同时引入太多变量。

### 7. Prototype Replay 的训练读取方式

Prototype Replay 最终保存为离线 buffer，训练时不再访问旧任务完整数据，而是读取 `.npz` replay samples。

| 机制 | 实现位置 | 作用 |
| --- | --- | --- |
| RehearsalConfig | `src/openpi/training/config.py:460` | 定义 replay 是否开启、loss 权重、采样策略、buffer 路径 |
| replay sample 读取 | `src/openpi/training/data_loader.py:180` | 读取 image、wrist image、state、action chunk、prompt |
| replay data loader | `src/openpi/training/data_loader.py:207` | 把离线 buffer 转成训练 batch |
| offline buffer runtime | `scripts/train_cl_lora.py:156` | 为每个 buffer 目录创建 replay loader |
| replay loss | `scripts/train_cl_lora.py:418` | 对旧任务 replay batch 计算 flow matching loss |

Replay loss 和当前任务 loss 是联合优化关系。`loss_weight` 越大，旧任务约束越强；但过强也可能抑制新任务可塑性。因此多阶段时需要根据旧任务数量、新任务难度和任务冲突程度适配 replay loss。

### 8. 普通均匀 Replay Baseline

Uniform Replay 是为了公平比较 Prototype Replay 的样本选择价值。它严格复用相同的旧任务 episode budget 和样本预算，只把“按物理分段选原型帧”替换为“按时间均匀抽帧”。

| 机制 | 实现位置 | 作用 |
| --- | --- | --- |
| uniform buffer 脚本 | `scripts/build_uniform_replay_buffer.py:3` | 构建普通均匀 replay buffer |
| 与 prototype 同预算 | `scripts/build_uniform_replay_buffer.py:7` | 从 reference prototype buffer 匹配样本数 |
| 均匀帧索引 | `scripts/build_uniform_replay_buffer.py:189` | 在每个 episode 时间轴上均匀选择 frame |
| `.npz` 保存 | `scripts/build_uniform_replay_buffer.py:342` | 保存为与 prototype replay 同格式的训练样本 |
| meta 信息 | `scripts/build_uniform_replay_buffer.py:390` | 记录 uniform_time、budget source、episode allocation |

该 baseline 用于回答：在相同 replay 样本预算下，prototype replay 的物理分段与原型筛选是否比普通均匀抽样更有效。

### 9. 评估与可解释性工具

当前仓库提供了若干与 replay buffer 相关的分析脚本，用于支撑方法的可解释性和科研叙事。

| 指标 / 工具 | 实现位置 | 作用 |
| --- | --- | --- |
| 帧压缩率 | `scripts/buffer_compress_rate_compute.py:23` | 计算 replay 样本数占原任务完整帧数比例 |
| compression ratio | `scripts/buffer_compress_rate_compute.py:159` | 输出原帧数 / replay 帧数 |
| 原型覆盖误差 PCE | `scripts/prototype_coverage_error.py:1` | 计算完整旧任务帧到最近 prototype 的覆盖误差 |
| replay sample 覆盖误差 | `scripts/replay_sample_coverage_error.py:1` | 计算完整旧任务帧到最近 replay sample 的覆盖误差 |
| replay 样本冗余率 | `scripts/replay_sample_redundancy.py:1` | 检查 replay sample 间的最近邻相似度 |
| 隐式分段可视化 | `scripts/visualize_prototype_segments.py:1` | 展示物理运动曲线、segment 时间轴和被选中的 prototype 帧 |

建议主文档优先报告与论文叙事最直接相关的指标：原始性能矩阵、BWT、单位 replay 样本保持收益、压缩率、PCE 和隐式分段可视化。冗余率和 replay sample 覆盖误差可以作为补充分析，尤其在视觉分布本身很紧密的任务上不必强行作为主指标。

## 二、实验设定说明

### 1. LIBERO 顺序学习总设定

当前 pi 系列 LIBERO 实验采用阶段式顺序微调。每一阶段只学习一个新任务，并在阶段结束后评估当前模型对所有已学任务的性能。

| 阶段 | 训练内容 | 评估对象 |
| --- | --- | --- |
| Stage 1 | 学习 taskA | taskA |
| Stage 2 | 从 taskA checkpoint 继续学习 taskB | taskA、taskB |
| Stage 3 | 从 taskB checkpoint 继续学习 taskC | taskA、taskB、taskC |
| Stage 4 | 从 taskC checkpoint 继续学习 taskD | taskA、taskB、taskC、taskD |

推荐保存原始性能矩阵。它是后续计算 BWT、保持收益和对比各分支遗忘程度的基础。

### 2. 分支一：普通 LoRA + no replay + balanced 冻结

| 项目 | 设定 |
| --- | --- |
| 训练入口 | `scripts/train.py` |
| replay | 不使用 |
| LoRA 类型 | 普通 LoRA |
| 冻结策略 | balanced freezing |
| 代表配置 | `src/openpi/training/config.py:1897` |
| pi0 代表配置 | `src/openpi/training/config.py:842` |

该分支是最基础的顺序微调 baseline，用于展示普通 LoRA 在连续任务学习中容易发生灾难性遗忘。它回答的问题是：如果只靠普通参数高效微调，不做结构稳定性和旧任务回放，旧任务能力会保留到什么程度。

注意：该分支不应启用 CL-Lora 专用训练脚本，也不应注入 replay buffer。

### 3. 分支二：CL-Lora + no replay + balanced 冻结

| 项目 | 设定 |
| --- | --- |
| 训练入口 | `scripts/train_cl_lora.py` |
| replay | 不使用 |
| LoRA 类型 | CL-Lora |
| 冻结策略 | balanced freezing |
| KD | 通常显式设为 `lambda_kd=0.0`，除非单独研究 KD |
| 代表配置 | `src/openpi/training/config.py:2202` |
| pi0 代表配置 | `src/openpi/training/config.py:1264` |

该分支用于验证 CL-Lora 结构本身是否能提供一定抗遗忘能力。它不是最终方法，而是把“结构稳定性”和“旧任务回放”拆开，单独观察 CL-Lora 的贡献。

若该分支在更多阶段后仍出现明显遗忘，不能直接否定 CL-Lora 的价值；更合理的解释是：CL-Lora 提供结构稳定性，但多阶段持续学习仍需要 prototype replay 提供旧任务数据约束。

### 4. 分支三：CL-Lora + Prototype Replay + balanced 冻结

| 项目 | 设定 |
| --- | --- |
| 训练入口 | `scripts/train_cl_lora.py` |
| replay | 开启 offline prototype buffer |
| buffer 构建 | `scripts/build_replay_buffer.py` |
| 旧任务样本 | 少量 episode，经隐式物理分段和 prototype top-K 选择 |
| 冻结策略 | balanced freezing |
| 代表配置 | `src/openpi/training/config.py:2314` |
| pi0 代表配置 | `src/openpi/training/config.py:1092` |

这是主方法分支。它用于验证：CL-Lora 的结构稳定性与 prototype replay 的少样本旧任务约束结合后，能否同时保持旧任务稳定性和新任务可塑性。

Prototype replay 的标准流程：

| 步骤 | 脚本 / 配置 | 说明 |
| --- | --- | --- |
| 训练当前任务 | `scripts/train_cl_lora.py` | 得到当前阶段最优 checkpoint |
| 构建当前任务 buffer | `scripts/build_replay_buffer.py` | 用该任务 checkpoint 提取视觉特征并构建 prototype buffer |
| 下一阶段训练 | `scripts/train_cl_lora.py` | 在新任务训练中加载所有旧任务 buffer |
| 压缩率统计 | `scripts/buffer_compress_rate_compute.py` | 记录 replay 样本预算和压缩倍数 |

当迁移到其他模型或 benchmark 时，优先保持这几个变量一致：任务顺序、旧任务 episode budget、每段 top-K、replay loss、训练步数、checkpoint 选择规则和评估次数。

### 5. 分支四：CL-Lora + 普通均匀 Replay + balanced 冻结

| 项目 | 设定 |
| --- | --- |
| 训练入口 | `scripts/train_cl_lora.py` |
| replay | 开启 offline uniform buffer |
| buffer 构建 | `scripts/build_uniform_replay_buffer.py` |
| 旧任务样本 | 与 prototype replay 同 episode budget、同 replay 样本数 |
| 冻结策略 | balanced freezing |
| 代表配置 | `src/openpi/training/config.py:2373` |
| pi0 代表配置 | `src/openpi/training/config.py:1579` |

该分支是 prototype replay 的核心公平对照。它不改变 replay 数据规模，只改变 replay 样本选择机制。

它回答的问题是：性能提升来自“有 replay”本身，还是来自“按隐式物理 skill 分段并选择 prototype 样本”的 replay 机制。

### 6. 四分支对照逻辑

| 对照 | 目的 |
| --- | --- |
| 普通 LoRA vs CL-Lora no replay | 验证结构稳定性是否缓解纯顺序微调遗忘 |
| CL-Lora no replay vs CL-Lora + Prototype Replay | 验证旧任务少样本 replay 是否是稳定多阶段性能的关键 |
| CL-Lora + Uniform Replay vs CL-Lora + Prototype Replay | 验证 prototype 筛选机制是否优于同预算普通回放 |
| 普通 LoRA vs 主方法 | 展示完整方法相对基础微调的增量学习优势 |

这四个分支共同形成论文叙事闭环：普通微调会遗忘，CL-Lora 能提供一定结构稳定性，但要实现多阶段稳定持续学习，需要 prototype replay 以少样本方式恢复旧任务约束；相同预算下，prototype replay 比 uniform replay 更能保留关键旧任务能力。

### 7. 迁移到其他模型或 benchmark 的建议

如果要在其他 VLA 模型或新 benchmark 上复用该实验设计，建议按以下顺序迁移。

| 优先级 | 内容 | 说明 |
| --- | --- | --- |
| 1 | 打通单任务普通 LoRA | 先确认训练、评估、动作归一化和数据读取链路正确 |
| 2 | 加入 CL-Lora 结构 | 对齐 LoRA 注入层、rank、alpha、冻结策略 |
| 3 | 构建 prototype buffer | 确认状态向量中的 xyz、rotation、gripper 索引正确 |
| 4 | 跑 CL-Lora + Prototype Replay | 先复现两阶段，再扩展三阶段和四阶段 |
| 5 | 加入 uniform replay baseline | 严格匹配 prototype buffer 的 episode budget 和样本数 |
| 6 | 计算指标和可视化 | BWT、单位 replay 样本保持收益、压缩率、PCE、隐式分段图 |

迁移时最容易出问题的位置通常是动作归一化、状态维度索引、gripper 语义、训练脚本是否选错、checkpoint 是否来自正确阶段，以及 replay buffer 路径是否和训练配置一致。

### 8. 推荐记录内容

每条实验支线建议固定记录以下信息。

| 类别 | 建议记录 |
| --- | --- |
| 训练设置 | config 名称、训练脚本、起始 checkpoint、训练步数、学习率、batch size、冻结策略 |
| replay 设置 | buffer 路径、episode budget、样本数、replay loss、采样策略 |
| 评估设置 | task suite、target task、评估次数、checkpoint step |
| 性能矩阵 | 每阶段对所有已学任务的成功率 |
| 持续学习指标 | BWT、单位 replay 样本保持收益 |
| replay 指标 | 压缩率、PCE、可视化、必要时补充 RSCE / 冗余率 |

这些记录能保证后续在 pi0、pi0.5、RoboTwin、CALVIN 或真机数据上扩展时，实验逻辑仍然可追溯、可复现、可横向比较。


# PI0.5_cl_lora_replay_README

## 1. 文档目的

这份文档用于给负责 `OpenVLA` 和 `SmolVLA` 改造的同学提供一个可参考的实现模板，说明我们在 `PI0.5` 分支中是如何把两类持续学习机制引入到 VLA 模型里的：

1. `CL-LoRA`（其中已包含教师学生模型蒸馏机制）
2. `隐式技能引导的原型回放（latent-skill-guided prototype replay）`

这份文档的目标不是让其他分支逐行照抄 `PI0.5` 的实现，而是帮助大家快速看清楚：

- 我们的核心科研想法到底落在了哪些代码层面
- 每一层分别承担什么作用
- 哪些地方是 `PI` 特有实现，哪些地方是其他 VLA 架构也应该迁移的共性思想

## 2. 统一科研目标

核心科研 idea 是：

- 不再采用“普通 LoRA 顺序微调 taskA -> taskB -> taskC”的方式直接做连续学习。
- 而是把 `CL-LoRA` 与 `隐式技能引导的原型回放` 一起引入 VLA 模型。
- 目标是在新任务学习过程中，同时缓解：
  - 对旧任务的灾难性遗忘
  - 对新任务的可塑性不足

目前我们主要聚焦于三个基座：

- `PI` 分支：我负责
- `OpenVLA` 分支：詠图负责
- `SmolVLA` 分支：涛志负责

因此，这份文档的用途是：

- 先把 `PI0.5` 中已经实现好的持续学习链路讲清楚
- 再提炼出对 `OpenVLA` / `SmolVLA` 可迁移的最小实现单元

## 3. PI0.5 分支目前形成的完整闭环

当前 `PI0.5` 分支中的持续学习闭环是：

1. 先用 `CL-LoRA` 训练 `taskA`。
2. 训练完 `taskA` 后，运行 `scripts/build_replay_buffer.py`，从 `taskA` 轨迹中离线构建 replay buffer。
3. buffer 的构建方式不是随机抽帧，而是：
   - 先按物理运动规则对连续轨迹进行隐式 skill 切分
   - 再对每一段计算视觉特征原型
   - 最后从每段中只保留最接近原型的关键帧
4. 训练 `taskB` 时，同时读取：
   - `taskB` 当前训练数据
   - 上一阶段旧任务的 teacher 快照
   - `taskA` 的 replay buffer
5. 优化目标由三部分组成：
   - 当前任务损失
   - KD 蒸馏损失
   - replay 回放损失
6. 训练 `taskC` 时重复同样过程。

对应到代码层面，这个闭环主要由以下脚本共同完成：

- 模型结构层：
  - `src/openpi/models/lora.py`
  - `src/openpi/models/gemma.py`
  - `src/openpi/models/pi0.py`
  - `src/openpi/models/pi0_config.py`
- 训练流程层：
  - `scripts/train_cl_lora.py`
  - `src/openpi/training/config.py`
  - `src/openpi/training/data_loader.py`
- 离线回放构建层：
  - `scripts/build_replay_buffer.py`

## 4. 整体改动总览：每层分别做了什么

从系统设计上看，这次改造不是“只改了一个冻结函数”或者“只加了一个 replay 脚本”，而是同时改了四层：

1. 模型参数化层
   - 让 LoRA 从普通 LoRA 变成了带 shared/specific 结构和 block-scale 的 `CL-LoRA`
2. 顶层模型集成层
   - 让 `PI0.5` 在不改原始任务定义的前提下，接入新的 CL-LoRA backbone 行为
3. 训练优化层
   - 把 teacher、KD、replay 都注入到原有 flow-matching 训练目标中
4. 数据与记忆层
   - 把旧任务轨迹压缩成“按隐式技能切分后的原型关键帧记忆”

下面按这四层分别展开。

## 5. 模型结构层改动：CL-LoRA 是如何落到代码里的

### 5.1 `src/openpi/models/lora.py`：把普通 LoRA 变成可控的 CL-LoRA

这个文件是最底层的核心改造点。它决定 LoRA 本身的参数如何初始化、如何前向传播、以及在持续学习场景下哪些分支应该被冻结、哪些分支应该被放大。

关键位置如下：

- `src/openpi/models/lora.py:12`
  - 定义 `LoRAConfig`
- `src/openpi/models/lora.py:27`
- `src/openpi/models/lora.py:29`
  - 在 `LoRAConfig` 中新增 `use_orthogonal_init`
- `src/openpi/models/lora.py:34`
- `src/openpi/models/lora.py:36`
  - 新增正交初始化函数 `orthogonal_init`
- `src/openpi/models/lora.py:61`
- `src/openpi/models/lora.py:63`
- `src/openpi/models/lora.py:64`
- `src/openpi/models/lora.py:66`
- `src/openpi/models/lora.py:67`
  - 对 attention 内部 LoRA 参数初始化逻辑做区分：
    - 普通 LoRA：按原始初始化方式
    - CL-LoRA：`lora_a` 正交初始化，`lora_b` 零初始化
- `src/openpi/models/lora.py:70`
- `src/openpi/models/lora.py:78`
- `src/openpi/models/lora.py:85`
- `src/openpi/models/lora.py:89`
  - attention 路径新增两个关键控制变量：
    - `freeze_a`
    - `block_scale`
- `src/openpi/models/lora.py:133`
- `src/openpi/models/lora.py:147`
- `src/openpi/models/lora.py:155`
- `src/openpi/models/lora.py:160`
- `src/openpi/models/lora.py:180`
- `src/openpi/models/lora.py:189`
- `src/openpi/models/lora.py:193`
  - FFN 的 LoRA 路径同步支持同样的 CL-LoRA 逻辑

这一层的本质含义是：

- 普通 LoRA 只是“增加低秩适配器”
- 我们的 CL-LoRA 则进一步决定：
  - 哪些 adapter basis 要固定
  - 哪些 adapter 要更偏任务共享
  - 哪些 adapter 要更偏任务特化
  - 某些深层 adapter 的影响力是否需要通过 block-scale 单独调节

换句话说，`lora.py` 负责把 LoRA 从“单纯的低秩微调”升级成“可做持续学习控制的低秩微调”。

### 5.2 `src/openpi/models/gemma.py`：把 shared/specific 结构写进 backbone 配置

如果说 `lora.py` 负责“LoRA 这个模块怎么变”，那么 `gemma.py` 负责“这种 CL-LoRA 应该加到哪些层、以什么结构加”。

关键位置如下：

- `src/openpi/models/gemma.py:42`
  - 定义 Gemma backbone 的总配置 `Config`
- `src/openpi/models/gemma.py:51`
- `src/openpi/models/gemma.py:52`
- `src/openpi/models/gemma.py:53`
- `src/openpi/models/gemma.py:54`
  - 在 `Config` 中新增：
    - `shared_pos`
    - `specific_pos`
    - `use_block_weight`
- `src/openpi/models/gemma.py:57`
  - 在 variant 类型中新增 `gemma_2b_cl_lora`、`gemma_300m_cl_lora`
- `src/openpi/models/gemma.py:111`
- `src/openpi/models/gemma.py:127`
  - 分别定义 `gemma_2b_cl_lora` 与 `gemma_300m_cl_lora`
- `src/openpi/models/gemma.py:122`
- `src/openpi/models/gemma.py:123`
- `src/openpi/models/gemma.py:124`
- `src/openpi/models/gemma.py:125`
- `src/openpi/models/gemma.py:137`
- `src/openpi/models/gemma.py:138`
- `src/openpi/models/gemma.py:139`
- `src/openpi/models/gemma.py:140`
  - 明确设定：
    - LoRA 使用正交初始化
    - 浅层是 shared
    - 深层是 specific
    - 开启 `use_block_weight`

当前配置里，`CL-LoRA` 的分层策略是：

- 总层数：18 层
- 前 9 层：`shared`
- 后 9 层：`specific`

这一步非常关键，因为它把持续学习思想从“训练技巧”变成了“模型结构先验”。

### 5.3 `src/openpi/models/gemma.py`：让 attention / FFN 真正接收 freeze 与 scale 控制

仅仅在配置里写 `shared_pos` 和 `specific_pos` 还不够，必须把这些控制量真正送进 transformer block 中。

关键位置如下：

- `src/openpi/models/gemma.py:203`
  - attention 逻辑开始接收不同 expert 的输入
- `src/openpi/models/gemma.py:210`
- `src/openpi/models/gemma.py:211`
  - 取出当前 expert 的：
    - `freeze_as`
    - `block_scales`
- `src/openpi/models/gemma.py:218`
- `src/openpi/models/gemma.py:228`
- `src/openpi/models/gemma.py:234`
  - 在 Q/K/V 投影里把 `freeze_a`、`block_scale` 真正传给 LoRA 路径
- `src/openpi/models/gemma.py:279`
- `src/openpi/models/gemma.py:288`
  - attention 输出投影也同样使用这两个控制量

也就是说，到了这一层，`shared/specific` 已经不只是概念，而是真正开始影响每一层的 LoRA 前向行为。
### 5.4 `src/openpi/models/gemma.py`：为 specific 层引入可学习 block weights

这是 `CL-LoRA` 在架构层最有辨识度的部分之一。

关键位置如下：

- `src/openpi/models/gemma.py:372`
  - `Module` 是整个 Gemma transformer 的主体
- `src/openpi/models/gemma.py:393`
- `src/openpi/models/gemma.py:397`
- `src/openpi/models/gemma.py:399`
- `src/openpi/models/gemma.py:400`
- `src/openpi/models/gemma.py:402`
- `src/openpi/models/gemma.py:409`
  - 给每个 expert 按需创建 `cl_block_weights_i`
- `src/openpi/models/gemma.py:418`
- `src/openpi/models/gemma.py:425`
- `src/openpi/models/gemma.py:426`
- `src/openpi/models/gemma.py:427`
  - `nn.scan` 的 block stack 现在会额外接收：
    - `freeze_stack`
    - `scale_stack`
- `src/openpi/models/gemma.py:455`
- `src/openpi/models/gemma.py:460`
- `src/openpi/models/gemma.py:464`
- `src/openpi/models/gemma.py:465`
  - 根据 `shared_pos` 构造 `freeze_stack`
- `src/openpi/models/gemma.py:470`
- `src/openpi/models/gemma.py:472`
- `src/openpi/models/gemma.py:474`
- `src/openpi/models/gemma.py:478`
- `src/openpi/models/gemma.py:480`
  - 根据 `specific_pos` 与 `cl_block_weights` 构造 `scale_stack`
- `src/openpi/models/gemma.py:486`
- `src/openpi/models/gemma.py:487`
- `src/openpi/models/gemma.py:489`
  - 把 `freeze_stack` 和 `scale_stack` 送进每一层 block

这里实现的含义是：

- `shared` 层：LoRA A 分支冻结，更偏保持共享知识
- `specific` 层：LoRA 不仅可训练，而且还有额外的 per-block 缩放权重

这样做的持续学习意义是：

- 浅层尽量保持通用视觉/语言/动作语义结构稳定
- 深层允许更强的任务特化，但这种特化不是无约束地“全放开”，而是由 block weight 柔性调控

这正是我们希望在“抗遗忘”和“可塑性”之间做结构化权衡的关键。

### 5.5 `src/openpi/models/pi0.py`：把 CL-LoRA backbone 接进 PI0.5 顶层模型

`PI0.5` 顶层模型并没有被重写任务定义，而是在原有 denoising / flow-matching 框架上接入了新的 CL-LoRA backbone 行为。

关键位置如下：

- `src/openpi/models/pi0.py:71`
  - `Pi0` 顶层模型定义
- `src/openpi/models/pi0.py:77`
- `src/openpi/models/pi0.py:78`
  - 根据 `paligemma_variant` 与 `action_expert_variant` 获取 Gemma 配置
- `src/openpi/models/pi0.py:80`
- `src/openpi/models/pi0.py:82`
- `src/openpi/models/pi0.py:83`
  - 如果检测到 `CL-LoRA` variant 但未正确开启正交初始化，会打印 warning
- `src/openpi/models/pi0.py:90`
- `src/openpi/models/pi0.py:91`
- `src/openpi/models/pi0.py:92`
- `src/openpi/models/pi0.py:97`
  - 用新的 Gemma module 构建双 expert LLM
- `src/openpi/models/pi0.py:108`
- `src/openpi/models/pi0.py:116`
  - 保留原有 action input/output projection 路径
- `src/openpi/models/pi0.py:109`
- `src/openpi/models/pi0.py:110`
- `src/openpi/models/pi0.py:111`
  - 保留 PI0.5 的时间条件输入路径
- `src/openpi/models/pi0.py:198`
- `src/openpi/models/pi0.py:211`
- `src/openpi/models/pi0.py:220`
- `src/openpi/models/pi0.py:227`
  - `forward_denoise` 的整体任务定义仍保持原逻辑

因此，`PI0.5` 分支的思路不是重新发明任务，而是：

- 保持原有 PI0.5 的输入输出接口和去噪任务形式不变
- 只在 backbone 的参数化方式与训练约束上注入持续学习机制

### 5.6 `src/openpi/models/pi0_config.py`：提供三种冻结策略

为了后续实验可控对比，我在 `pi0_config.py` 中保留了三套冻结策略。

关键位置如下：

- `src/openpi/models/pi0_config.py:80`
  - `get_freeze_filter`
  - 对应较接近官方 LoRA 的冻结方式
- `src/openpi/models/pi0_config.py:111`
  - `get_strict_lora_freeze_filter`
  - 仅保留 LoRA 参数与可选 block weights 可训练
- `src/openpi/models/pi0_config.py:139`
  - `get_balanced_lora_freeze_filter`
  - 在保留 LoRA / block weights 的同时，也保留轻量 action-side heads 可训练

当前 `CL-LoRA` 主配置实际采用的是：

- `get_balanced_lora_freeze_filter(allow_block_weights=True)`
- 对应位置：
  - `src/openpi/training/config.py:876`
  - `src/openpi/training/config.py:882`

需要注意：
- 如果冻结过严，模型可能学不动新任务
- 如果冻结过松，旧任务遗忘会更严重
- 当前 `PI` 分支把 `balanced freeze + block weights` 作为主线实验方案，是因为它更有希望同时兼顾可塑性与稳定性，但最终采取什么冻结策略，需要根据你们模型架构和参数量而定，唯一目标就是做出好的性能对比，即能凸显我们CL-Lora+隐式技能原型回放方法的优势就行了，不要让实验过程影响结论（自己品味）。

## 6. 训练流程层改动：持续学习是如何注入优化过程的

### 6.1 `src/openpi/training/config.py`：把 replay、KD、local root 等训练控制项显式配置化

这个文件的作用是：把持续学习需要的关键开关全部纳入训练配置，而不是散落在脚本里写死。

关键位置如下：

- `src/openpi/training/config.py:97`
- `src/openpi/training/config.py:98`
- `src/openpi/training/config.py:100`
  - 在 `DataConfig` 中新增：
    - `root`
    - `target_task_name`
- `src/openpi/training/config.py:283`
- `src/openpi/training/config.py:290`
- `src/openpi/training/config.py:291`
- `src/openpi/training/config.py:292`
- `src/openpi/training/config.py:352`
- `src/openpi/training/config.py:357`
- `src/openpi/training/config.py:358`
  - `LeRobotLiberoDataConfig` 支持：
    - 本地数据集路径
    - 单任务筛选
- `src/openpi/training/config.py:303`
- `src/openpi/training/config.py:307`
- `src/openpi/training/config.py:308`
- `src/openpi/training/config.py:309`
- `src/openpi/training/config.py:310`
- `src/openpi/training/config.py:311`
- `src/openpi/training/config.py:323`
- `src/openpi/training/config.py:324`
- `src/openpi/training/config.py:325`
  - 保证 LIBERO 数据最终会被映射成统一的模型输入格式
- `src/openpi/training/config.py:461`
- `src/openpi/training/config.py:462`
- `src/openpi/training/config.py:463`
- `src/openpi/training/config.py:465`
- `src/openpi/training/config.py:467`
- `src/openpi/training/config.py:469`
- `src/openpi/training/config.py:472`
- `src/openpi/training/config.py:474`
- `src/openpi/training/config.py:476`
  - 新增 `RehearsalConfig`
  - 定义 replay 的：
    - 是否启用
    - 来源（旧任务全量数据 or offline buffer）
    - loss 权重
    - 采样策略
- `src/openpi/training/config.py:515`
- `src/openpi/training/config.py:516`
- `src/openpi/training/config.py:517`
- `src/openpi/training/config.py:519`
- `src/openpi/training/config.py:577`
- `src/openpi/training/config.py:579`
  - 把 `lambda_kd`、`rehearsal`、`kd_lambda_schedule`、`trainable_filter` 等全部纳入 `TrainConfig`

这一层的价值是：

- 持续学习不再是“临时改脚本”的实验 hack
- 而是变成训练配置可控、可复现实验的一部分

### 6.2 `src/openpi/training/config.py`：显式区分普通 LoRA 与 CL-LoRA 训练配置

当前与单任务 / 连续学习直接相关的两个配置入口是：

- `src/openpi/training/config.py:792`
- `src/openpi/training/config.py:793`
  - `pi05_libero_taskA_lora`
- `src/openpi/training/config.py:841`
- `src/openpi/training/config.py:842`
  - `pi05_libero_taskA_cl_lora`

其中：

- 普通 LoRA 配置使用：
  - `gemma_2b_lora`
  - `gemma_300m_lora`
  - 对应位置：
    - `src/openpi/training/config.py:799`
    - `src/openpi/training/config.py:800`
- CL-LoRA 配置使用：
  - `gemma_2b_cl_lora`
  - `gemma_300m_cl_lora`
  - 对应位置：
    - `src/openpi/training/config.py:843`

另外需要注意两点：

- 当前 `CL-LoRA` 主配置已经显式启用了：
  - `balanced freeze + allow_block_weights=True`
  - 对应位置：
    - `src/openpi/training/config.py:876`
    - `src/openpi/training/config.py:882`
- `config.py` 中也已经预留了 offline replay 的示例写法：
  - `src/openpi/training/config.py:865`
  - `src/openpi/training/config.py:873`

这意味着：

- 训练脚本不需要重写，切换 replay 主要靠配置驱动
- 这一点对 `OpenVLA` / `SmolVLA` 的迁移非常有参考价值

### 6.3 `scripts/train_cl_lora.py`：判断何时需要 teacher，何时需要 KD schedule

持续学习训练并不是任何时候都需要 teacher（比如训练第一个任务taskA时就没有教师模型，训练后续任务时才能用前一任务训得的权重作为教师指导），因此脚本里先对 teacher 的必要性做了统一判断。

关键位置如下：

- `scripts/train_cl_lora.py:92`
- `scripts/train_cl_lora.py:94`
- `scripts/train_cl_lora.py:97`
- `scripts/train_cl_lora.py:102`
- `scripts/train_cl_lora.py:111`
  - `_teacher_required`
  - 负责判断：
    - 是否启用 output KD
    - 是否启用 repr KD
- `scripts/train_cl_lora.py:113`
- `scripts/train_cl_lora.py:117`
- `scripts/train_cl_lora.py:124`
- `scripts/train_cl_lora.py:129`
  - `_resolve_lambda_kd`
  - 支持：
    - constant
    - linear warmup
    - cosine 等 KD 权重调度形式

这一步的好处是：

- teacher 与 KD 不会被无脑开启
- 训练脚本可以在纯单任务 / 带 KD / 带 replay 等多种模式之间复用

### 6.4 `scripts/train_cl_lora.py`：统一构建 rehearsal runtime，兼容全量旧数据与 offline buffer

这一层是 replay 机制真正接入训练脚本的关键。

关键位置如下：

- `scripts/train_cl_lora.py:137`
- `scripts/train_cl_lora.py:138`
- `scripts/train_cl_lora.py:142`
  - `_RehearsalRuntime`
  - 管理多个 replay loader 与 iterator
- `scripts/train_cl_lora.py:144`
  - `_build_rehearsal_runtime`
- `scripts/train_cl_lora.py:161`
- `scripts/train_cl_lora.py:165`
- `scripts/train_cl_lora.py:171`
- `scripts/train_cl_lora.py:173`
  - 当 `source == offline_buffer` 时，直接用 replay buffer 构造 loader
- `scripts/train_cl_lora.py:175`
- `scripts/train_cl_lora.py:182`
- `scripts/train_cl_lora.py:184`
- `scripts/train_cl_lora.py:190`
  - 当 `source == task_dataset` 时，使用旧任务完整数据集做 rehearsal
- `scripts/train_cl_lora.py:199`
- `scripts/train_cl_lora.py:200`
- `scripts/train_cl_lora.py:203`
- `scripts/train_cl_lora.py:210`
  - `_next_rehearsal_batch`
  - 支持：
    - `round_robin`
    - `random`

这说明当前 `PI` 分支在设计上并没有把 replay 写死成单一方案，而是保留了：

- 旧任务全量 replay
- 原型压缩 replay

两种路径。
### 6.5 `scripts/train_cl_lora.py`：高效初始化 student / teacher，减少显存浪费

这一块是 `PI` 分支在实际训练中非常关键的工程性改动。

关键位置如下：

- `scripts/train_cl_lora.py:230`
  - `init_train_state`
- `scripts/train_cl_lora.py:240`
- `scripts/train_cl_lora.py:241`
- `scripts/train_cl_lora.py:242`
- `scripts/train_cl_lora.py:247`
  - 构建 CL-LoRA 专用 weight decay mask：
    - `lora_a` 不做 weight decay
    - `block_weights` 不做 weight decay
- `scripts/train_cl_lora.py:255`
- `scripts/train_cl_lora.py:259`
  - trainable 的 CL 参数保留在 fp32
  - frozen 参数转成 bf16
- `scripts/train_cl_lora.py:278`
- `scripts/train_cl_lora.py:282`
- `scripts/train_cl_lora.py:286`
- `scripts/train_cl_lora.py:292`
- `scripts/train_cl_lora.py:295`
  - student 初始化与 base 权重合并都先在 CPU 上完成
- `scripts/train_cl_lora.py:297`
- `scripts/train_cl_lora.py:298`
- `scripts/train_cl_lora.py:300`
- `scripts/train_cl_lora.py:301`
  - teacher 只保留“可训练参数快照”，而不是复制整个 full model

这一设计非常重要，原因是：

- 在大模型 VLA 上，如果 teacher 也按 full model 保留，显存和内存会非常重
- 但对于我们这个 CL-LoRA 设定而言：
  - frozen backbone 的 teacher 与 student 本来就是相同的
  - 真正变化的只有 LoRA / block weights / 小头部参数
- 因此只快照 trainable params 就够了

这一点对 `OpenVLA` / `SmolVLA` 同样非常值得迁移。

### 6.6 `scripts/train_cl_lora.py`：总损失 = 当前任务损失 + KD 损失 + replay 损失

这一段决定了持续学习训练的核心目标函数。

关键位置如下：

- `scripts/train_cl_lora.py:337`
  - `train_step_cl_lora`
- `scripts/train_cl_lora.py:365`
- `scripts/train_cl_lora.py:375`
- `scripts/train_cl_lora.py:389`
  - 当前任务的基础 flow-matching loss
- `scripts/train_cl_lora.py:391`
- `scripts/train_cl_lora.py:395`
- `scripts/train_cl_lora.py:400`
- `scripts/train_cl_lora.py:404`
- `scripts/train_cl_lora.py:408`
- `scripts/train_cl_lora.py:411`
- `scripts/train_cl_lora.py:414`
  - teacher output KD
  - 做法是：
    - 先把当前 student 的 trainable params 快照保存下来
    - 再把 teacher snapshot 替换进去
    - 前向出 teacher 输出
    - 最后恢复 student 当前 trainable params
- `scripts/train_cl_lora.py:418`
- `scripts/train_cl_lora.py:420`
- `scripts/train_cl_lora.py:422`
- `scripts/train_cl_lora.py:424`
- `scripts/train_cl_lora.py:427`
- `scripts/train_cl_lora.py:435`
  - replay loss
  - 对 replay batch 用同样的 flow-matching 任务损失计算
- `scripts/train_cl_lora.py:437`
- `scripts/train_cl_lora.py:438`
  - 总损失为：
    - `loss_task + lambda_kd * loss_kd + replay_weight * loss_rehearsal`


- 当前任务损失负责学新任务
- KD 负责保留旧任务行为约束
- replay 负责提供旧任务样本支持

三者都是显式可控的。

### 6.7 `scripts/train_cl_lora.py`：在主训练循环中以最小侵入方式注入 replay

关键位置如下：

- `scripts/train_cl_lora.py:461`
  - `main`
- `scripts/train_cl_lora.py:488`
- `scripts/train_cl_lora.py:493`
  - 主数据 loader 与 rehearsal runtime 分开构建
- `scripts/train_cl_lora.py:514`
- `scripts/train_cl_lora.py:519`
- `scripts/train_cl_lora.py:521`
- `scripts/train_cl_lora.py:522`
  - `ptrain_step` 编译时显式包含：
    - teacher
    - main batch
    - replay batch
    - replay_on 开关
- `scripts/train_cl_lora.py:540`
- `scripts/train_cl_lora.py:542`
- `scripts/train_cl_lora.py:545`
- `scripts/train_cl_lora.py:548`
- `scripts/train_cl_lora.py:558`
  - 每隔 `replay_every_n_steps` 执行一次 replay
  - 如果当前 step 不需要 replay，则直接用 shape-compatible dummy batch 保证 JIT 结构稳定

这一部分的优点是：

- replay 被加入训练流程，但没有破坏原有训练循环结构
- 对其他 VLA 分支来说，这是一种很值得借鉴的“低侵入式接入方式”

## 7. 数据与回放层改动：隐式技能引导的原型回放是如何落地的

### 7.1 `src/openpi/training/data_loader.py`：支持本地 root 与任务级过滤

为了做单任务连续微调，我们需要把完整 LIBERO 数据中某一个具体任务筛出来训练。

关键位置如下：

- `src/openpi/training/data_loader.py:250`
  - `create_torch_dataset`
- `src/openpi/training/data_loader.py:260`
- `src/openpi/training/data_loader.py:263`
  - 支持把 `root` 显式传给 LeRobot dataset
- `src/openpi/training/data_loader.py:270`
- `src/openpi/training/data_loader.py:281`
- `src/openpi/training/data_loader.py:287`
- `src/openpi/training/data_loader.py:298`
- `src/openpi/training/data_loader.py:311`
- `src/openpi/training/data_loader.py:322`
  - 对 episode 做任务级筛选与重索引

这个逻辑的作用是：

- 在不重新拆分数据集的情况下，直接从完整 LIBERO 数据中取出 `taskA` / `taskB` / `taskC`
- 为顺序连续微调提供数据入口

### 7.2 `src/openpi/training/data_loader.py`：新增 ReplayBufferDataset，让 offline buffer 可直接参与训练

关键位置如下：

- `src/openpi/training/data_loader.py:143`
  - `ReplayBufferDataset`
- `src/openpi/training/data_loader.py:149`
- `src/openpi/training/data_loader.py:153`
- `src/openpi/training/data_loader.py:158`
- `src/openpi/training/data_loader.py:162`
  - 从 `manifest.jsonl` 读取 replay 样本索引
- `src/openpi/training/data_loader.py:181`
- `src/openpi/training/data_loader.py:182`
- `src/openpi/training/data_loader.py:183`
- `src/openpi/training/data_loader.py:184`
- `src/openpi/training/data_loader.py:186`
- `src/openpi/training/data_loader.py:193`
  - 从 `.npz` 读取：
    - `image`
    - `wrist_image`
    - `state`
    - `action_chunk` / `actions`
    - `prompt`
- `src/openpi/training/data_loader.py:197`
- `src/openpi/training/data_loader.py:199`
- `src/openpi/training/data_loader.py:202`
- `src/openpi/training/data_loader.py:203`
  - 统一返回当前 LIBERO transform chain 期望的键
- `src/openpi/training/data_loader.py:207`
- `src/openpi/training/data_loader.py:217`
- `src/openpi/training/data_loader.py:227`
- `src/openpi/training/data_loader.py:232`
- `src/openpi/training/data_loader.py:245`
  - `create_replay_buffer_data_loader`
  - 把 replay buffer 包装成与正常训练数据一致的 dataloader 接口

这一层的关键设计思想是：

- replay 不应该另起一套复杂训练接口
- 最稳妥的方式是让 replay 数据长得和正常训练样本一样
- 这样训练脚本只要多一个 batch 输入口即可

### 7.3 `src/openpi/policies/libero_policy.py` 与 `src/openpi/models/model.py`：保证 replay 样本与正常样本最终映射到相同 Observation 结构

这个细节很重要，因为 replay 机制真正稳不稳，取决于 replay 样本是否真的兼容原模型输入。

关键位置如下：

- `src/openpi/policies/libero_policy.py:43`
  - `LiberoInputs`
- `src/openpi/policies/libero_policy.py:53`
- `src/openpi/policies/libero_policy.py:54`
- `src/openpi/policies/libero_policy.py:57`
- `src/openpi/policies/libero_policy.py:60`
- `src/openpi/policies/libero_policy.py:61`
- `src/openpi/policies/libero_policy.py:63`
- `src/openpi/policies/libero_policy.py:65`
- `src/openpi/policies/libero_policy.py:75`
- `src/openpi/policies/libero_policy.py:81`
  - 定义了 LIBERO 输入到模型前的标准格式
  - 其中右腕相机如果不存在，会自动补零
- `src/openpi/models/model.py:110`
- `src/openpi/models/model.py:117`
- `src/openpi/models/model.py:123`
- `src/openpi/models/model.py:124`
- `src/openpi/models/model.py:125`
- `src/openpi/models/model.py:126`
  - `Observation.from_dict` 最终把这些输入变成结构化的 Observation

这意味着：

- 只要 replay buffer 输出的键名和当前 transform chain 对齐
- replay 样本最终就会和正常训练样本变成完全一致的模型输入类型

这是当前 replay 方案能够无缝接入训练流程的根本原因。
### 7.4 `scripts/build_replay_buffer.py`：离线构建隐式技能引导的原型回放 buffer

这是本次 replay 机制最核心的新脚本。

关键位置如下：

- `scripts/build_replay_buffer.py:1`
  - 顶部 docstring 已明确写出完整 pipeline
- `scripts/build_replay_buffer.py:39`
  - `BuildReplayBufferConfig`
- `scripts/build_replay_buffer.py:42`
- `scripts/build_replay_buffer.py:48`
- `scripts/build_replay_buffer.py:51`
- `scripts/build_replay_buffer.py:53`
- `scripts/build_replay_buffer.py:55`
- `scripts/build_replay_buffer.py:60`
- `scripts/build_replay_buffer.py:63`
- `scripts/build_replay_buffer.py:68`
- `scripts/build_replay_buffer.py:77`
- `scripts/build_replay_buffer.py:80`
  - 定义了 replay buffer 构建所需的核心参数

#### 7.4.1 物理规则驱动的隐式技能切分

对应位置：

- `scripts/build_replay_buffer.py:213`
- `scripts/build_replay_buffer.py:221`
- `scripts/build_replay_buffer.py:237`
- `scripts/build_replay_buffer.py:253`
- `scripts/build_replay_buffer.py:265`
- `scripts/build_replay_buffer.py:268`
  - `_compute_chunk_motions`
  - 根据 state 中的：
    - xyz 位移
    - 姿态变化
    - gripper 变化
    统计每个短时间 chunk 的主导运动模式
- `scripts/build_replay_buffer.py:282`
- `scripts/build_replay_buffer.py:288`
- `scripts/build_replay_buffer.py:290`
- `scripts/build_replay_buffer.py:294`
- `scripts/build_replay_buffer.py:304`
  - `_find_boundaries`
  - 根据：
    - gripper 事件
    - 稳定的主导运动模式切换
    找 segment 边界
- `scripts/build_replay_buffer.py:309`
- `scripts/build_replay_buffer.py:317`
- `scripts/build_replay_buffer.py:318`
- `scripts/build_replay_buffer.py:321`
  - `_build_segments`
  - 先保留所有初始切分段，不提前丢弃短段

这一步对应我们“隐式技能”的定义方式：

- 不需要像AtomicVLA那样给每段打显式语义标签
- 只要求段内帧在物理运动意义上高度一致就够了

#### 7.4.2 短段合并，避免过度碎片化

对应位置：

- `scripts/build_replay_buffer.py:387`
  - `_merge_short_segments`
- `scripts/build_replay_buffer.py:392`
- `scripts/build_replay_buffer.py:394`
- `scripts/build_replay_buffer.py:402`
- `scripts/build_replay_buffer.py:403`
  - 对太短的 segment 进行合并
- `scripts/build_replay_buffer.py:406`
- `scripts/build_replay_buffer.py:407`
- `scripts/build_replay_buffer.py:409`
  - 对于“虽短但 gripper 变化显著”的 segment 选择保留
- `scripts/build_replay_buffer.py:421`
- `scripts/build_replay_buffer.py:424`
- `scripts/build_replay_buffer.py:427`
- `scripts/build_replay_buffer.py:429`
- `scripts/build_replay_buffer.py:434`
- `scripts/build_replay_buffer.py:439`
  - 用邻接段的运动相似度来决定合并到左侧还是右侧

这一步的意义是：

- 防止 LIBERO 轨迹被切得过碎
- 又不会把 gripper 开合这类本来短但有意义的 skill 段误合并掉

#### 7.4.3 用原模型视觉头提取段内特征并计算原型

对应位置：

- `scripts/build_replay_buffer.py:530`
- `scripts/build_replay_buffer.py:532`
- `scripts/build_replay_buffer.py:537`
- `scripts/build_replay_buffer.py:538`
- `scripts/build_replay_buffer.py:539`
- `scripts/build_replay_buffer.py:540`
- `scripts/build_replay_buffer.py:544`
- `scripts/build_replay_buffer.py:546`
  - `_extract_visual_feature`
  - 直接使用当前模型的视觉 tower 提特征
- `scripts/build_replay_buffer.py:551`
- `scripts/build_replay_buffer.py:552`
- `scripts/build_replay_buffer.py:557`
- `scripts/build_replay_buffer.py:558`
- `scripts/build_replay_buffer.py:560`
- `scripts/build_replay_buffer.py:567`
- `scripts/build_replay_buffer.py:568`
  - `_load_feature_model`
  - 支持从显式 checkpoint 路径加载模型参数做特征提取

这里的设计很重要：

- 我们没有额外训练一个新的 prototype encoder
- 而是直接使用当前 PI 模型已有的视觉表征
- 这样 replay prototype 与模型本身的内部表征空间天然一致

#### 7.4.4 每段计算 prototype，并只保留最接近 prototype 的关键帧

对应位置：

- `scripts/build_replay_buffer.py:604`
  - `main`
- `scripts/build_replay_buffer.py:608`
- `scripts/build_replay_buffer.py:609`
- `scripts/build_replay_buffer.py:613`
- `scripts/build_replay_buffer.py:614`
  - 从 train config 中解析数据集与目标任务
- `scripts/build_replay_buffer.py:632`
- `scripts/build_replay_buffer.py:633`
- `scripts/build_replay_buffer.py:634`
  - 读取目标任务 episode
- `scripts/build_replay_buffer.py:668`
- `scripts/build_replay_buffer.py:674`
- `scripts/build_replay_buffer.py:675`
- `scripts/build_replay_buffer.py:676`
- `scripts/build_replay_buffer.py:677`
  - 对每条轨迹执行隐式技能切分
- `scripts/build_replay_buffer.py:688`
- `scripts/build_replay_buffer.py:691`
- `scripts/build_replay_buffer.py:694`
- `scripts/build_replay_buffer.py:695`
- `scripts/build_replay_buffer.py:696`
- `scripts/build_replay_buffer.py:697`
  - 对每段：
    - 提取所有帧特征
    - 做 L2 normalize
    - 取均值作为段原型
    - 计算每帧到原型的 cosine
- `scripts/build_replay_buffer.py:699`
- `scripts/build_replay_buffer.py:702`
- `scripts/build_replay_buffer.py:703`
  - 选出每段最接近 prototype 的 `top_k_per_segment` 帧

这一步对应我们 replay 方案的核心思想：

- 不保留整段旧轨迹
- 也不做随机采样
- 而是只保留最能代表该隐式 skill 的关键帧

#### 7.4.5 把关键帧按后续训练可直接读取的格式写入 buffer

对应位置：

- `scripts/build_replay_buffer.py:705`
- `scripts/build_replay_buffer.py:706`
- `scripts/build_replay_buffer.py:709`
- `scripts/build_replay_buffer.py:721`
- `scripts/build_replay_buffer.py:722`
  - 写段级 prototype 记录
- `scripts/build_replay_buffer.py:724`
- `scripts/build_replay_buffer.py:728`
- `scripts/build_replay_buffer.py:730`
- `scripts/build_replay_buffer.py:732`
- `scripts/build_replay_buffer.py:734`
- `scripts/build_replay_buffer.py:735`
- `scripts/build_replay_buffer.py:736`
- `scripts/build_replay_buffer.py:738`
- `scripts/build_replay_buffer.py:739`
- `scripts/build_replay_buffer.py:741`
- `scripts/build_replay_buffer.py:742`
- `scripts/build_replay_buffer.py:750`
  - 把每个关键帧保存成 `.npz`，包含：
    - RGB 图像
    - wrist 图像
    - state
    - 当前 action
    - `action_chunk`
    - task / prompt
    - task_id
    - segment_id
    - episode/frame 索引元信息
    - 到 prototype 的相似度
- `scripts/build_replay_buffer.py:753`
- `scripts/build_replay_buffer.py:755`
- `scripts/build_replay_buffer.py:762`
- `scripts/build_replay_buffer.py:768`
  - 写 `manifest.jsonl`
- `scripts/build_replay_buffer.py:773`
- `scripts/build_replay_buffer.py:779`
- `scripts/build_replay_buffer.py:795`
- `scripts/build_replay_buffer.py:799`
- `scripts/build_replay_buffer.py:804`
- `scripts/build_replay_buffer.py:808`
  - 写 `meta.json`

至此，replay buffer 不只是“几张图片”，而是：

- 一组可直接被下次训练读取的紧凑记忆样本
- 并且保留了足够的任务/段/帧级元信息，方便后续做分析与可视化

## 8. Replay buffer 如何重新接回训练流程

当前 `PI` 分支采用的是一种非常干净的接法：

1. `build_replay_buffer.py` 负责离线生成：
   - `manifest.jsonl`
   - `samples/*.npz`
   - `prototypes/*.npy`
   - `meta.json`
2. `ReplayBufferDataset` 负责把这些 `.npz` 还原成标准训练样本
   - 对应：`src/openpi/training/data_loader.py:143`
3. `create_replay_buffer_data_loader` 负责把这些样本送过原有 LIBERO transform chain
   - 对应：`src/openpi/training/data_loader.py:207`
4. `_build_rehearsal_runtime` 负责把 replay buffer loader 接进训练脚本
   - 对应：`scripts/train_cl_lora.py:144`
5. `train_step_cl_lora` 负责把 replay batch 纳入总损失
   - 对应：`scripts/train_cl_lora.py:337`

因此，当前 replay 设计的优势是：

- 不需要另写第二套训练目标
- 不需要另造一个完全独立的 replay trainer
- 只要 replay 样本接口对齐，就能复用原任务损失

## 9. 对 OpenVLA / SmolVLA 同学最有价值的可迁移结论

不需要逐行照抄 `PI0.5` 实现，而应重点迁移以下三类思想。

### 9.1 模型结构侧最该迁移的部分

最值得迁移的是：

1. CL 专用 LoRA 初始化
   - 参考：
     - `src/openpi/models/lora.py:29`
     - `src/openpi/models/lora.py:63`
     - `src/openpi/models/lora.py:64`
2. shared / specific 分层
   - 参考：
     - `src/openpi/models/gemma.py:52`
     - `src/openpi/models/gemma.py:53`
     - `src/openpi/models/gemma.py:123`
     - `src/openpi/models/gemma.py:124`
3. 冻结 shared adapter basis
   - 参考：
     - `src/openpi/models/lora.py:78`
     - `src/openpi/models/lora.py:189`
     - `src/openpi/models/gemma.py:460`
     - `src/openpi/models/gemma.py:465`
4. specific 深层 block-scale 调控
   - 参考：
     - `src/openpi/models/gemma.py:399`
     - `src/openpi/models/gemma.py:478`
     - `src/openpi/models/gemma.py:480`

### 9.2 训练流程侧最该迁移的部分

最值得迁移的是：

1. teacher 只快照 trainable CL 参数，而不是复制 full model
   - 参考：
     - `scripts/train_cl_lora.py:297`
     - `scripts/train_cl_lora.py:300`
     - `scripts/train_cl_lora.py:301`
     - `scripts/train_cl_lora.py:404`
2. 总损失 = 当前任务 + KD + replay
   - 参考：
     - `scripts/train_cl_lora.py:411`
     - `scripts/train_cl_lora.py:425`
     - `scripts/train_cl_lora.py:437`
3. replay 作为辅助 datastream 接入，而不是重写训练主循环
   - 参考：
     - `scripts/train_cl_lora.py:161`
     - `scripts/train_cl_lora.py:165`
     - `scripts/train_cl_lora.py:545`

### 9.3 回放机制侧最该迁移的部分

最值得迁移的是：

1. 用物理运动规则做隐式 skill 切分
   - 参考：
     - `scripts/build_replay_buffer.py:213`
     - `scripts/build_replay_buffer.py:282`
2. 短段合并，但保留短 gripper 段
   - 参考：
     - `scripts/build_replay_buffer.py:387`
     - `scripts/build_replay_buffer.py:407`
3. 用当前模型视觉特征求 segment prototype
   - 参考：
     - `scripts/build_replay_buffer.py:530`
     - `scripts/build_replay_buffer.py:696`
4. 每段仅保留 top-k 关键帧
   - 参考：
     - `scripts/build_replay_buffer.py:699`
     - `scripts/build_replay_buffer.py:702`
     - `scripts/build_replay_buffer.py:732`

## 10. 总结

当前 `PI0.5` 分支已经不是“在 LoRA 上加了一个小补丁”，而是形成了一个完整的持续学习实现栈：

- 模型层：`CL-LoRA` 改变了 adapter 在深浅层的分工方式
- 配置层：冻结策略、KD、replay 都配置化
- 训练层：teacher + KD + replay 共同约束新任务学习
- 记忆层：用“隐式技能切分 + 原型关键帧”压缩旧任务经验

因此，这个分支对 `OpenVLA` 和 `SmolVLA` 的价值，不在于 PI 的类名或模块命名，而在于已经把“VLA + 持续学习”拆成了可迁移的四层结构：

1. backbone 级别的 CL-LoRA 参数化
2. 顶层模型级别的无缝接入
3. 训练目标级别的 KD + replay 组合
4. 记忆构建级别的 prototype replay

- 因此Smolvla和Openvla两个分支不需要照抄 `PaliGemma` 或 `PI0.5` 的细节
- 但应尽量在各自模型里复现同样的持续学习分解方式
- 这样三条分支最后才有可能在同一科研叙事下形成统一论文故事

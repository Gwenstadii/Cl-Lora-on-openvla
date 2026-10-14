[English](README.md) | **简体中文**

# CL-LoRA on OpenVLA

**面向 7B 视觉-语言-动作模型的持续学习。**
一套参数高效的框架，让 **OpenVLA-7B** 能够顺序学习一串操作任务，而不会灾难性地遗忘此前已经掌握的本领。

![框架总览](assets/fig1_overview.png)

*图 1 —— 左侧是持续 VLA 的学习设定，右侧是两个机制：冻结主干内的任务路由「共享 / 专属」LoRA 适配，以及为旧任务提供行为锚点的动作条件原型回放。*

---

## 核心结果

在 RoboTwin 2.0 上**顺序**学习四个双臂任务（Handover Mic → Grab Roller → Stack Bowls Two → Open Laptop）。下表是**最后一个阶段训练结束后**的成功率，每任务 50 个 episode，所有需要路由的方法统一给定任务身份：

| 方法 | Task A | Task B | Task C | Task D（最新） | 旧任务均值 |
|:--|:--:|:--:|:--:|:--:|:--:|
| 顺序 LoRA（无任何保护） | 0.00 | 0.00 | 0.00 | 0.82 | **0.00** |
| CL-LoRA，无回放 | 0.00 | 0.78 | 0.54 | 0.88 | **0.44** |
| **CL-LoRA + 原型回放**（本仓库） | **0.78** | 0.69 | 0.62 | **0.87** | **0.70** |
| CL-LoRA + 均匀回放（**同预算**） | 0.00 | 0.67 | 0.61 | 0.75 | **0.43** |

**最老的任务从 0.00 抬到 0.78，旧任务均值从 0.44 抬到 0.70，而新任务一分钱没付（0.88 → 0.87）。** 同样帧预算下的均匀回放什么都没拿回来（0.43，与完全不用回放持平），还**倒贴**了新任务 13 个点 —— 所以增益来自**"存了哪些帧"**，而不是"用了回放"这件事本身。

---

## 问题是什么

像 OpenVLA 这样的视觉-语言-动作模型，通常用参数高效微调适配新任务：冻结 7B 主干，只训练一小块低秩适配器。这个配方对**一个**任务没问题，但和真实部署严重不匹配 —— 机器人是**一个接一个**收到任务的：

* 训练时只能拿到当前任务的示教，把历史数据全部存下来反复重训并不现实；
* 直接在同一套适配器上继续训练，新梯度会**覆盖**旧技能。上表第一行的顺序 LoRA 就把**每一个**早期任务都打到了 0%；
* 在 VLA 里，这种遗忘不是"分类边界丢了"，而是**条件动作分布发生了漂移** —— `π(a₁:ₕ | o, l, s)`。面对旧的观测、指令和机器人状态，策略仍然会输出一个局部看起来合理、但对旧任务在物理上是错的动作块。**没有任何地方会报错，机器人就是做不成。**

要从这种状态里恢复，必须在**两个地方同时**提供稳定性：**参数空间**（让新梯度无法改写每一个旧技能）和**数据空间**（让网络里被多个任务共用的那部分，持续收到"旧任务长什么样"的信息）。

---

## 方法

框架由两个机制耦合而成，与具体主干无关。本仓库是它在 **OpenVLA-7B / OpenVLA-OFT** 上的实例化，并在 LIBERO 与 RoboTwin 2.0 上做了验证。

### 一 · 任务路由的共享 / 专属适配（CL-LoRA）

![任务路由的共享-专属适配](assets/fig2_shared_specific.png)

每一个低秩更新被拆成**角色不同、生命周期也不同**的三部分：

    y_ℓ = W_ℓ x_ℓ + (α/r) · γ_{ℓ,τ} · (x_ℓ A_ℓ) · B^{route(τ)}_ℓ

| 组件 | 角色 | 生命周期 |
|:--|:--|:--|
| **A_ℓ** —— 正交初始化的下投影 | 一个**稳定的低秩坐标系**；不存储任何任务知识 | **终身冻结** |
| **B^sh_ℓ** —— 共享上投影 | 可跨任务复用、随任务流持续演化的适配 | **整条任务流上持续可训练** |
| **B^sp_{ℓ,τ}** —— 任务专属槽 | 每个任务一个紧凑的残差槽，存进任务库 | 单任务持有，**受保护** |
| **γ_{ℓ,τ} = 1 + 0.5·tanh(β)** —— 有界门控 | 任务 τ 的残差注入到第 ℓ 层的强度 | 单任务持有，取值域 (0.5, 1.5) |

层被**划分**为**共享层**（用 `B^sh`）与**专属层**（路由到本任务自己的槽）。推理时，模型使用共享通路 + 被请求任务的槽。

冻结 `A` 的意义在于：一个存下来的槽，**只在"它训练时所处的那个坐标系"下才有意义** —— 把 `A` 冻住，每个槽就永远是可插拔的。

### 二 · 动作条件的原型回放（ACPR）

![动作条件原型回放](assets/fig3_acpr_pipeline.png)

仅靠参数隔离，无法保证**共享通路**继续与旧任务兼容。所以每个任务训练结束后，本仓库会构建一份很小的回放记忆 —— 但它**不是均匀抽帧**：

1. **物理引导的隐式技能分段** —— 用末端平移 / 旋转、夹爪状态转换与运动强度，把一条示教切成若干原子技能段；过短的段被合并，有意义的夹爪转换段被保留。
2. **动作条件的多模态行为描述子** —— 每个候选帧得到融合描述子 `z = Norm([√w_c·c ; √w_a·a ; √w_p·p])`，三个分量分别来自 **context**（VLA prefix 表示）、**action-token**（动作生成路径）与**物理运动**特征，其中 `w_a = 0.50` —— 记忆的首要职责是保存**"任务是怎么做的"**，而不只是视觉多样性。
3. **鲁棒时序 k-center 选帧** —— 逐段进行：从 medoid 起步，用 MAD 规则剔除离群帧，施加最小时间间隔约束，然后贪心地逐个加入能最大幅度降低段内覆盖误差的帧。每段贡献固定配额，因此**记忆规模随"技能阶段的个数"增长，而与示教长度无关**。

由于分段只依赖物理信号（不依赖模型），选帧又是逐段进行的，即使示教很长且大部分是静止时段，buffer 依然小且均衡。

### 两个机制如何咬合

训练任务 `t` 时：当前任务的样本更新**共享通路**与**当前槽**；回放样本则经由**它自己那个旧槽**前向来重构旧行为，但**流向旧槽的梯度被阻断**。因此回放从不是在"继续训练旧适配器"—— 它约束的是共享通路（以及全局动作生成接口），让它们保持与那些旧槽兼容。此外，共享通路在**含回放的 step 上用完整学习率**，在没有旧任务证据进入 batch 的 step 上则用**温和缩放后的学习率**。

---

## 实验结果

**RoboTwin 2.0** —— 四阶段双臂任务流，每任务 50 个 episode，`γ = 0`：

| 评估 checkpoint | 方法 | A | B | C | D |
|:--|:--|:--:|:--:|:--:|:--:|
| B 阶段后 | 无回放 | 0.36 | 0.94 | — | — |
| | 原型回放 | **0.74** | 0.96 | — | — |
| | 均匀回放 | 0.89 | 0.94 | — | — |
| C 阶段后 | 无回放 | 0.00 | 0.84 | 0.92 | — |
| | 原型回放 | **0.76** | 0.78 | 0.94 | — |
| | 均匀回放 | 0.04 | 0.84 | 0.86 | — |
| D 阶段后 | 无回放 | 0.00 | 0.78 | 0.54 | 0.88 |
| | **原型回放** | **0.78** | 0.69 | 0.62 | **0.87** |
| | 均匀回放 | 0.00 | 0.67 | 0.61 | 0.75 |

有回放时，最老的任务在三个阶段的评估点上**保持平线**（0.74 / 0.76 / 0.78）；没有回放则在两个阶段后归零。均匀回放在一个阶段内看起来还行（0.89），随后就塌了（0.04 → 0.00）—— 它把整条轨迹覆盖得很广，却没有钉住真正决定成败的那些阶段。

**LIBERO** —— 四阶段任务流（spatial → spatial → object → goal），specific-`A` 冻结：

| Stage-4 checkpoint | A | B | C | D |
|:--|:--:|:--:|:--:|:--:|
| 无回放 | 0.44 | 0.66 | 0.94 | 1.00 |
| **+ 原型回放** | **0.96** | **0.94** | 0.88 | 1.00 |

---

## 仓库结构

    openvla-oft/                          OpenVLA-OFT 分支（训练 + 评估栈）
      vla-scripts/
        cl_lora.py                        CLLoRALinear、适配器注入、任务槽库
        train_cl_lora.py                  持续学习训练器（阶段管理 / 回放 / 任务库）
        finetune.py                       普通 LoRA 基线 / 单阶段微调
        replay_dataset.py                 回放 buffer 的 Dataset，供训练器消费
        build_replay_buffer_openvla.py    ACPR buffer 构建器 —— LIBERO
        build_replay_buffer_robotwin.py   ACPR buffer 构建器 —— RoboTwin（双臂）
        build_uniform_replay_buffer*.py   均匀回放基线（预算对齐）
        analyze_cl_lora_drift.py          逐阶段量化 A / B / block-scale 的漂移
        analyze_film_drift.py             语言调制（FiLM）漂移分析
        analyze_buffer_coverage.py        对比各套 buffer 到底存了哪些帧
      experiments/robot/libero/           LIBERO 评估（任务库恢复）
      experiments/robot/openvla_utils.py  共享的模型 / 适配器加载

    RoboTwin-main/policy/openvla-oft/     RoboTwin 2.0 策略服务与评估
      deploy_policy.py                    带任务库路由的推理
      eval_sequence.sh / eval_multi_gpu.sh  多阶段、多任务评估驱动脚本

    run_v39*.sh / run_v41*.sh / run_lora*.sh   实验启动脚本（阶段 A -> D）

    REPLAY_BUG_NOTES.md                   完整实验记录（含负结果）
    CLAUDE.md                             架构与代码地图
    modified_libero_rlds/                 RLDS 元数据（.tfrecord 数据未纳入版本控制）

---

## 快速开始

    # 1. 环境
    cd openvla-oft
    pip install -e .
    pip install "flash-attn==2.5.5" --no-build-isolation

    # 2. 用第一个任务构建原型回放 buffer（ACPR）
    python vla-scripts/build_replay_buffer_robotwin.py \
      --vla-path  <openvla-7b> \
      --data-root-dir datasets/rlds --dataset-name aloha_handover_mic_clean \
      --stats-path <taskA-checkpoint>/dataset_statistics.json \
      --output-dir replay_buffers/taskA --num-episodes 10 --top-k 3 --overwrite

    # 3. 用 CL-LoRA + 回放学习下一个任务
    torchrun --standalone --nproc_per_node 4 vla-scripts/train_cl_lora.py \
      --vla_path <openvla-7b> --dataset_name aloha_grab_roller_clean \
      --stage 2 --previous_checkpoint_dir <taskA-checkpoint> --previous_checkpoint_step 30000 \
      --use_cl_lora True --lora_rank 16 --shared_depth 8 --first_lora_layer 16 \
      --orthogonal_init True --freeze_a True --use_block_scale True \
      --use_replay True --replay_buffer_dirs replay_buffers/taskA \
      --replay_every_n_steps 4 --replay_loss_weight 0.5 \
      --batch_size 2 --learning_rate 5e-4 --max_steps 40000

    # 4. 评估到目前为止见过的所有任务
    bash policy/openvla-oft/eval_sequence.sh <checkpoint> 4,4,5,5,6,6,7,7 50 tag A B C D

仓库根目录的 shell 启动脚本（`run_v41r_resumeC.sh`、`run_v39b9_layermask.sh` …）就是把上面这些命令连同层掩码、冻结策略与评估门禁一起包起来的产物 —— 想复现上表某一行，用它们最快。

---

## 文档

| 文档 | 内容 |
|:--|:--|
| `REPLAY_BUG_NOTES.md` | 完整实验记录：每一组配置、每一次评估，**以及负结果**（均匀回放为什么失败、回放的梯度究竟够不到哪些漂移通道、若干早期结论是如何被推翻的） |
| `CLAUDE.md` | OpenVLA-OFT 分支的架构与代码地图 |
| `RoboTwin_CL_handover.md` | RoboTwin 平台说明与评估协议 |
| `Pi_CL-Lora_Prototype_Replay_README.md` | 本移植与 π0.5 参考实现的关系 |

如果只看一件事，请看 `REPLAY_BUG_NOTES.md` 里的 **"回放可达性（replay reachability）"判据**：回放能修复的，只是**梯度确实能作用到评估路径上、且任务对该损伤的容错足够**的那部分。这套框架大量非显然的行为 —— 包括什么会失败、为什么失败 —— 都可以从它推出来。

---

## 引用

方法本身描述在一篇正在审稿的稿件中（工作标题 *PROPEL-VLA: Prototype-Guided Parameter-Efficient Lifelong Adaptation for Vision-Language-Action Models*）。等它公开后，这里会补上正式的引用信息。

## 致谢

建立在 [OpenVLA](https://github.com/openvla/openvla) 与 OpenVLA-OFT 微调栈之上，并在 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 与 [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) 上评估。LoRA 正交初始化 / 冻结 `A` 的设计沿用 π0.5 的 CL-LoRA 参考实现。

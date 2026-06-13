# OpenVLA / SmolVLA 最小迁移清单

## 目标

参考在 `PI0.5` 上改好的`CL-LoRA`+`隐式技能引导的原型回放`的持续学习思路，最小化迁移到 `OpenVLA` 或 `SmolVLA` 上（需自己根据模型架构进行相应调整，并不是拿过来就能用），形成可跑的版本：

不要求逐行复现 `PI` 代码，但需要引入以下核心思想。

## 一、模型结构的改动

1. 把普通 LoRA 改成可控 LoRA
- 需要支持：
  - shared / specific 两类层
  - `freeze_a`
  - `block_scale`
- 参考代码：
  - `src/openpi/models/lora.py:29`
  - `src/openpi/models/lora.py:63`
  - `src/openpi/models/lora.py:78`
  - `src/openpi/models/lora.py:189`

2. 在 backbone 配置中显式写 shared / specific 分层
- 至少要能指定：
  - 哪些层是 shared
  - 哪些层是 specific
- 参考代码：
  - `src/openpi/models/gemma.py:52`
  - `src/openpi/models/gemma.py:53`
  - `src/openpi/models/gemma.py:123`
  - `src/openpi/models/gemma.py:124`

3. 给 specific 层增加可学习 block weights
- 参考代码：
  - `src/openpi/models/gemma.py:399`
  - `src/openpi/models/gemma.py:478`
  - `src/openpi/models/gemma.py:480`

4. 不改原模型任务定义，只改参数化方式
- 原则：
  - 保持原来的输入输出接口和训练主任务
  - 只在 adapter 结构和训练约束上做持续学习改造
- 参考代码：
  - `src/openpi/models/pi0.py:198`
  - `src/openpi/models/pi0.py:220`
  - `src/openpi/models/pi0.py:227`

## 二、训练流程的改动

1. teacher模型不要复制 full model
- 只保存旧阶段 trainable CL 参数快照
- 参考代码：
  - `scripts/train_cl_lora.py:297`
  - `scripts/train_cl_lora.py:300`
  - `scripts/train_cl_lora.py:301`

2. 总损失必须包含三项
- `loss_task`  （原有的训练损失）
- `loss_kd`   （教师模型和学生模型之间的蒸馏损失）
- `loss_rehearsal`    （回放损失）
- 参考代码：
  - `scripts/train_cl_lora.py:411`
  - `scripts/train_cl_lora.py:425`
  - `scripts/train_cl_lora.py:437`

3. replay 作为辅助 batch 接入训练循环
- 不要重写一套独立 trainer
- 参考代码：
  - `scripts/train_cl_lora.py:161`
  - `scripts/train_cl_lora.py:165`
  - `scripts/train_cl_lora.py:545`

4. 把 KD / replay / freeze 都配置化
- 不要把这些开关写死在脚本里，而是作为参数在执行命令时传入
- 参考代码：
  - `src/openpi/training/config.py:461`
  - `src/openpi/training/config.py:515`
  - `src/openpi/training/config.py:519`

## 三、“隐技能引导的原型回放”的改造

1. 对旧任务轨迹做物理规则（是参考AtomicVLA中的原子技能切分规则优化而来）切分
- 用：
  - 位移
  - 姿态变化
  - gripper 变化
- 参考代码：
  - `scripts/build_replay_buffer.py:213`
  - `scripts/build_replay_buffer.py:282`

2. 增加短段合并
- 避免切得过碎
- 但保留短 gripper 事件段
- 参考代码：
  - `scripts/build_replay_buffer.py:387`
  - `scripts/build_replay_buffer.py:407`

3. 用模型自身视觉特征计算 segment prototype
- 不要另外训练 encoder
- 参考代码：
  - `scripts/build_replay_buffer.py:530`
  - `scripts/build_replay_buffer.py:696`

4. 每段只保留 top-k 关键帧进 buffer
- 参考代码：
  - `scripts/build_replay_buffer.py:699`
  - `scripts/build_replay_buffer.py:702`
  - `scripts/build_replay_buffer.py:732`

## 四、最小可跑实验顺序

1. 先做普通 LoRA 二阶段顺序微调 baseline （已完成）
- `taskA -> taskB` （taskA和taskB目前采取同分布模式，即LIBERO-spatial里的两个不同任务）
- 记录 taskA 遗忘程度  
- Openvla目前顺序微调结果：
    1.基于Openvla-7B微调taskA（taskA_eval=0.9）-> 2.基于taskA_checkpoint微调taskB（taskB_eval=0.52, taskA_eval=0） 
- Smolvla目前顺序微调结果：
    1.基于Smolvla_base微调taskA（taskA_eval=0.6）-> 2.基于taskA_checkpoint微调taskB（taskB_eval=0.4, taskA_eval=0）

2. 再做纯CL-LoRA，不开replay
- 先验证仅靠双适配器+块权重结构约束能否减缓遗忘

3. 最后加 prototype replay
- 验证 `CL-LoRA + replay` 是否进一步提升旧任务保持率与新任务学习效果

## 五、迁移时最容易犯的错误

1. 只加 replay，不改 LoRA 结构
- 这样很难体现我们的核心创新点

2. 只改 LoRA 结构，不把 replay 接进训练总损失
- 这样旧任务样本记忆无法真正参与优化

3. 直接复制 PI 模块名
- 正确做法是迁移“机制”，不是迁移“命名”

4. teacher 复制 full model
- 对大模型显存/内存代价太高

## 六、总结

- 在你们自己的负责的VLA backbone里实现“shared 浅层 + specific 深层 + block-scale + 教师学生模型蒸馏”的 `CL-LoRA`
- 再把“隐式技能切分 + prototype 关键帧”构成的旧任务 buffer，以replay batch 形式接进 `task + KD + replay` 总损失


# scripts/train_cl_lora.py
# 双模型架构（教师和学生）
#该脚本目的：用于训练双模型架构（教师和学生）的模型，使用CL-Lora微调。
import dataclasses
import functools
import logging
import platform
import os
from typing import Any
import orbax.checkpoint as ocp
import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb
import tyro
import pathlib
import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

# 初始化日志格式
def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)
    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers[0].setFormatter(formatter)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return
    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    if resuming and (ckpt_dir / "wandb_id.txt").exists():
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

def _snapshot_trainable_params(
    params: nnx.State,
    trainable_filter: nnx.filterlib.Filter,
) -> at.Params:
    """Materialize a pure-dict snapshot of the trainable subset (teacher swap-safe)."""
    return params.filter(trainable_filter).to_pure_dict()

def _cast_frozen_params_to_bf16(
    params: nnx.State,
    freeze_filter: nnx.filterlib.Filter,
) -> nnx.State:
    """Cast only frozen params to bf16; keep trainable params (LoRA/block_weights) in fp32."""
    return nnx_utils.state_map(
        params,
        freeze_filter,
        lambda p: p.replace(p.value.astype(jnp.bfloat16)),
    )

def _teacher_required(config: _config.TrainConfig) -> bool:
    # Output-KD
    kd_needed = getattr(config, "lambda_kd", 0.0) > 0.0

    sched = getattr(config, "kd_lambda_schedule", None)
    if sched is not None and sched.enabled:
        target = config.lambda_kd if sched.end_value is None else sched.end_value
        kd_needed = kd_needed or (target > 0.0)

    # Representation-KD (mechanism ③) may also require teacher even when lambda_kd == 0
    repr_cfg = getattr(config, "repr_kd", None)
    if repr_cfg is not None and repr_cfg.enabled:
        repr_needed = (
            repr_cfg.suffix_hidden_weight > 0.0
            or repr_cfg.prelogit_weight > 0.0
            or repr_cfg.shared_kv_weight > 0.0
        )
        kd_needed = kd_needed or repr_needed

    return kd_needed

def _resolve_lambda_kd(config: _config.TrainConfig, step) -> at.Float[at.Array, ""]:
    sched = getattr(config, "kd_lambda_schedule", None)
    base = jnp.asarray(config.lambda_kd, dtype=jnp.float32)

    if sched is None or not sched.enabled or sched.kind == "constant":
        return base

    start = jnp.asarray(sched.start_value, dtype=jnp.float32)
    end = jnp.asarray(base if sched.end_value is None else sched.end_value, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)

    if sched.kind == "linear_warmup":
        warmup_steps = max(1, int(sched.warmup_steps))
        progress = jnp.clip(step_f / float(warmup_steps), 0.0, 1.0)
        return start + progress * (end - start)

    if sched.kind == "cosine":
        total_steps = max(1, int(config.num_train_steps if sched.total_steps is None else sched.total_steps))
        progress = jnp.clip(step_f / float(total_steps), 0.0, 1.0)
        cosine = 0.5 * (1.0 - jnp.cos(jnp.pi * progress))
        return start + cosine * (end - start)

    raise ValueError(f"Unsupported kd schedule kind: {sched.kind}")

@dataclasses.dataclass
class _RehearsalRuntime:  # 回放类型，包含多个回放任务的加载器和迭代器
    tasks: list[str]
    loaders: list[Any]
    iters: list[Any]
    rr_index: int = 0

def _build_rehearsal_runtime(config: _config.TrainConfig, data_sharding) -> _RehearsalRuntime | None:
    reh = getattr(config, "rehearsal", None)
    if reh is None or not reh.enabled:
        return None
    if reh.loss_weight <= 0:
        raise ValueError("rehearsal.enabled=True but rehearsal.loss_weight <= 0.")

    if not isinstance(config.data, _config.LeRobotLiberoDataConfig):
        raise TypeError(
            "Rehearsal currently supports LeRobotLiberoDataConfig only. "
            f"Got: {type(config.data).__name__}"
        )

    tasks: list[str] = []
    loaders: list[Any] = []
    iters: list[Any] = []

    if reh.source == "offline_buffer":
        if not reh.buffer_dirs:
            raise ValueError("rehearsal.source='offline_buffer' but rehearsal.buffer_dirs is empty.")
        for buffer_dir in reh.buffer_dirs:
            loader = _data_loader.create_replay_buffer_data_loader(
                config,
                replay_buffer_dir=buffer_dir,
                sharding=data_sharding,
                shuffle=True,
            )
            tasks.append(pathlib.Path(buffer_dir).name)
            loaders.append(loader)
            iters.append(iter(loader))
    else:
        if not reh.task_names:
            raise ValueError("rehearsal.enabled=True but rehearsal.task_names is empty.")
        current_task = config.data.target_task_name
        for task_name in reh.task_names:
            if current_task is not None and task_name == current_task:
                logging.warning("Skipping rehearsal task identical to current task: %s", task_name)
                continue
            replay_data_cfg = dataclasses.replace(config.data, target_task_name=task_name)
            replay_train_cfg = dataclasses.replace(config, data=replay_data_cfg)
            loader = _data_loader.create_data_loader(
                replay_train_cfg,
                sharding=data_sharding,
                shuffle=True,
            )
            tasks.append(task_name)
            loaders.append(loader)
            iters.append(iter(loader))

    if not iters:
        raise ValueError("No valid rehearsal loaders were created.")
    logging.info("Initialized rehearsal loaders for: %s", tasks)
    return _RehearsalRuntime(tasks=tasks, loaders=loaders, iters=iters)


def _next_rehearsal_batch(runtime: _RehearsalRuntime, strategy: str):
    if strategy == "random":
        idx = np.random.randint(0, len(runtime.iters))
    else:
        idx = runtime.rr_index % len(runtime.iters)
        runtime.rr_index += 1

    try:
        return next(runtime.iters[idx])
    except StopIteration:
        runtime.iters[idx] = iter(runtime.loaders[idx])
        return next(runtime.iters[idx])


def _make_cl_lora_weight_decay_mask(trainable_params: nnx.State) -> nnx.State:
    """True = apply weight decay, False = no weight decay."""
    no_decay_keys = set(trainable_params.filter(nnx_utils.PathRegex(".*lora_a.*")).flat_state())
    no_decay_keys |= set(trainable_params.filter(nnx_utils.PathRegex(".*block_weights.*")).flat_state())
    return trainable_params.map(lambda k, _: k not in no_decay_keys)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    loaded_params = loader.load(params_shape)
    # 注意：Base Model 可能不包含 LoRA 参数，这里我们允许部分加载
    # 只要 Base 参数对齐即可，新增的 LoRA 参数保持初始化状态（B=0）
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )

# 初始化训练状态（Student + Teacher）
@at.typecheck
def init_train_state(
    config: _config.TrainConfig, 
    init_rng: at.KeyArrayLike, 
    mesh: jax.sharding.Mesh, 
    *, 
    resume: bool
) -> tuple[training_utils.TrainState, Any, Any, Any]:
    
    use_teacher = _teacher_required(config)

    # Build a CL-LoRA-specific weight-decay mask:
    # - no weight decay on lora_a (so shared A stays strictly frozen when freeze_a=True)
    # - no weight decay on block_weights (more stable gate learning)
    with jax.default_device(jax.devices("cpu")[0]):
        mask_model = config.model.create(jax.random.key(0))
        mask_params = nnx.state(mask_model)
        mask_params = _cast_frozen_params_to_bf16(mask_params, config.freeze_filter)
        weight_decay_mask = _make_cl_lora_weight_decay_mask(mask_params.filter(config.trainable_filter))

    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=weight_decay_mask)

    def init(rng: at.KeyArrayLike) -> tuple[training_utils.TrainState, Any]:
        rng, student_rng = jax.random.split(rng)
        
        # 只创建 Student 模型
        student_model = config.model.create(student_rng)
        student_params = nnx.state(student_model)
                # Match the original trainer: only cast frozen params to bfloat16.
        # Keep trainable CL-LoRA params (LoRA / block weights) in fp32 for stability.
        student_params = _cast_frozen_params_to_bf16(student_params, config.freeze_filter)


        train_state = training_utils.TrainState(
            step=0,
            params=student_params,
            model_def=nnx.graphdef(student_model),
            tx=tx,
            opt_state=tx.init(student_params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else student_params,
        )
        return train_state, None

    train_state_shape, _ = jax.eval_shape(init, init_rng)

    # 🔴 关键：使用 FSDP=2，必须在这里指定切片策略
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    logging.info("Initializing random model weights on CPU (to avoid GPU OOM)...")
    with jax.default_device(jax.devices("cpu")[0]):
        train_state, _ = init(init_rng)
        
        logging.info("Loading and merging Base weights on CPU...")
        partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
        
        if partial_params is not None:
            student_model = nnx.merge(train_state.model_def, train_state.params)
            
            _, student_state = nnx.split(student_model)
            merged_student_dict = ocp.transform_utils.intersect_trees(student_state.to_pure_dict(), partial_params)
            student_state.replace_by_pure_dict(merged_student_dict)
            # Re-apply frozen bf16 cast AFTER checkpoint merge, because merge can overwrite dtypes.
            student_state = _cast_frozen_params_to_bf16(student_state, config.freeze_filter)
            train_state = dataclasses.replace(train_state, params=student_state)

            logging.info("Weights merged successfully on CPU.")

        if use_teacher:
            # Only snapshot trainable params (LoRA / block weights / etc.) to reduce GPU memory.
            # Frozen backbone params are identical between teacher and student throughout CL-LoRA training.
            teacher_params = _snapshot_trainable_params(train_state.params, config.trainable_filter)
            teacher_sharding = sharding.fsdp_sharding(teacher_params, mesh, log=False)
        else:
            teacher_params = None
            teacher_sharding = None

    import gc
    del partial_params
    gc.collect()

    logging.info("Pushing model weights and optimizer states to GPUs safely...")
    
    # 🔴 极其关键：带 shape 参考的同步阻塞推送，防止类型膨胀和碎片化
    def put_to_device(x, s, ref):
        if isinstance(x, (jax.Array, np.ndarray)):
            sharding_spec = s.value if hasattr(s, 'value') else s
            ref_dtype = ref.value.dtype if hasattr(ref, 'value') else getattr(ref, 'dtype', x.dtype)
            if x.dtype != ref_dtype:
                x = x.astype(ref_dtype)
            
            # 强制同步阻塞
            out = jax.device_put(x, sharding_spec)
            jax.block_until_ready(out)
            return out
        return x

    train_state = jax.tree_util.tree_map(put_to_device, train_state, state_sharding, train_state_shape)
    
    if use_teacher:
        teacher_params = jax.tree_util.tree_map(put_to_device, teacher_params, teacher_sharding, teacher_params)

    logging.info("GPU placement complete!")

    return train_state, state_sharding, teacher_params, teacher_sharding

# 核心训练步骤 (CL-LoRA 版)
@at.typecheck
def train_step_cl_lora(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    teacher_params: Any,
    batch: tuple[_model.Observation, _model.Actions],
    replay_batch: tuple[_model.Observation, _model.Actions],
    replay_on: at.Bool[at.Array, ""],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:

    
    # 1. 准备 Student 模型
    student_model = nnx.merge(state.model_def, state.params)
    student_model.train() # Student 开启 Dropout
    
    # 判断是否启用了 Teacher
    use_teacher = teacher_params is not None
    

    # 2. 定义 Loss 函数
    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        teacher_trainable_snapshot: Any,  # pure dict snapshot or None
    ):
        def _flow_task_loss(local_model, local_rng, local_observation, local_actions):
            preprocess_rng, step_rng = jax.random.split(local_rng)
            local_observation = _model.preprocess_observation(preprocess_rng, local_observation, train=True)

            noise_rng, time_rng = jax.random.split(step_rng)
            batch_shape = local_actions.shape[:-2]
            noise = jax.random.normal(noise_rng, local_actions.shape)
            time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001

            u_t = noise - local_actions
            v_t = local_model.forward_denoise(local_observation, local_actions, noise, time)
            return jnp.mean(jnp.square(v_t - u_t))

        preprocess_rng, step_rng = jax.random.split(rng)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=True)

        noise_rng, time_rng = jax.random.split(step_rng)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001

        u_t = noise - actions
        v_student = model.forward_denoise(observation, actions, noise, time)

        loss_task = jnp.mean(jnp.square(v_student - u_t))

        lambda_kd = _resolve_lambda_kd(config, state.step)
        # IMPORTANT:
        # lambda_kd is a JAX scalar (tracer under jit), so do NOT use it in a Python `if`.
        # We gate KD contribution by multiplication in total_loss instead.
        if use_teacher:
            assert teacher_trainable_snapshot is not None

            _, student_current_state = nnx.split(model)

            # Snapshot student's current trainable params as a pure dict (avoid alias/reference risk).
            student_trainable_snapshot = student_current_state.filter(config.trainable_filter).to_pure_dict()

            # Swap in teacher's trainable snapshot.
            student_current_state.replace_by_pure_dict(teacher_trainable_snapshot)
            temp_teacher_model = nnx.merge(state.model_def, student_current_state)
            temp_teacher_model.eval()

            v_teacher = temp_teacher_model.forward_denoise(observation, actions, noise, time)
            v_teacher = jax.lax.stop_gradient(v_teacher)

            loss_kd = jnp.mean(jnp.square(v_student - v_teacher))

            # Restore student's trainable params from snapshot.
            student_current_state.replace_by_pure_dict(student_trainable_snapshot)
        else:
            loss_kd = jnp.array(0.0, dtype=loss_task.dtype)
        
        reh_cfg = getattr(config, "rehearsal", None)
        if reh_cfg is not None and reh_cfg.enabled:
            replay_loss_weight = jnp.asarray(reh_cfg.loss_weight, dtype=loss_task.dtype)

            def _compute_replay_loss(_):
                replay_rng = jax.random.fold_in(rng, 99991)  # deterministic but separate stream
                replay_obs, replay_actions = replay_batch
                return _flow_task_loss(model, replay_rng, replay_obs, replay_actions)

            loss_rehearsal = jax.lax.cond(
                replay_on,
                _compute_replay_loss,
                lambda _: jnp.array(0.0, dtype=loss_task.dtype),
                operand=None,
            )
        else:
            replay_loss_weight = jnp.array(0.0, dtype=loss_task.dtype)
            loss_rehearsal = jnp.array(0.0, dtype=loss_task.dtype)

        total_loss = loss_task + lambda_kd * loss_kd + replay_loss_weight * loss_rehearsal
        return total_loss, {"loss_task": loss_task, "loss_kd": loss_kd, "loss_rehearsal": loss_rehearsal, "loss_total": total_loss, "lambda_kd": lambda_kd}

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    # 注意这里传参的改变，为了适配 loss_fn
    (loss, info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        student_model, train_rng, observation, actions, teacher_params
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(student_model, new_params)
    new_params = nnx.state(student_model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    info.update({"grad_norm": optax.global_norm(grads)})
    
    return new_state, info

def main(config: _config.TrainConfig):
    init_logging()
    
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(f"Batch size {config.batch_size} must be divisible by {jax.device_count()}.")

    # 设置缓存
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    # 1. Mesh 和 Sharding
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 2. Checkpoint 管理
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # 3. 数据加载
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    rehearsal_runtime = _build_rehearsal_runtime(config, data_sharding)

    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info("Initialized data loader.")

    # 4. 初始化 Student 和 Teacher
    # teacher_params 在这里被初始化并加载 Base 权重
    train_state, train_state_sharding, teacher_params, teacher_sharding = init_train_state(
        config, init_rng, mesh, resume=resuming
    )
    jax.block_until_ready(train_state)
    jax.block_until_ready(teacher_params)
    logging.info("Initialized CL-LoRA Train State (Student + Teacher).")

    if resuming:
        # 注意：这里只恢复 Student 的状态，Teacher 依然是初始化的 Base 状态（这正是我们想要的）
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # 5. 编译训练步
    # 注意把 teacher_params 也传进去，并指定 sharding
    ptrain_step = jax.jit(
        functools.partial(train_step_cl_lora, config),
        in_shardings=(
            replicated_sharding,   # rng
            train_state_sharding,  # state
            teacher_sharding,      # teacher
            data_sharding,         # main batch
            data_sharding,         # replay batch
            replicated_sharding,   # replay_on
        ),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )


    # 6. 训练循环
    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        replay_enabled = (
            rehearsal_runtime is not None
            and (step % max(1, config.rehearsal.replay_every_n_steps) == 0)
        )

        if replay_enabled:
            replay_batch = _next_rehearsal_batch(rehearsal_runtime, config.rehearsal.sample_strategy)
        else:
            # keep shape-compatible dummy input; branch is masked by replay_on=False
            replay_batch = batch

        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(
                train_rng,
                train_state,
                teacher_params,
                batch,
                replay_batch,
                jnp.asarray(replay_enabled, dtype=jnp.bool_),
            )

        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            # 打印 Loss
            info_str = f"L_total={reduced_info['loss_total']:.4f} (Task={reduced_info['loss_task']:.4f}, KD={reduced_info['loss_kd']:.4f})"
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Training finished.")
    checkpoint_manager.wait_until_finished()

if __name__ == "__main__":
    # 使用 tyro CLI 解析配置
    # 示例调用: python scripts/train_cl_lora.py pi05_libero_low_mem_finetune --exp_name=test01
    main(_config.cli())
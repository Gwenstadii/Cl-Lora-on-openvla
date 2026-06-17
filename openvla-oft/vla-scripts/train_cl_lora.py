"""
train_cl_lora.py — CL-LoRA continual learning training script for OpenVLA.

Ports the PI-series CL-LoRA + prototype replay continual learning pipeline to PyTorch.
Supports three training branches via CLI flags:

  Branch 2: CL-LoRA + no replay       (--no_kd --no_replay)
  Branch 3: CL-LoRA + Prototype Replay (--use_kd --use_replay --replay_buffer_dirs .../prototypes)
  Branch 4: CL-LoRA + Uniform Replay   (--use_kd --use_replay --replay_buffer_dirs .../uniform)

Reference: PI0.5 train_cl_lora.py (pi0.5代码/train_cl_lora.py)
"""

import copy
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from accelerate import PartialState
from huggingface_hub import snapshot_download
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)
from cl_lora import inject_cl_lora_into_model
from replay_dataset import PrototypeReplayDataset

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import NoisyActionProjector, ProprioProjector
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==============================================================================
# Config
# ==============================================================================

@dataclass
class TrainCLConfig:
    # fmt: off
    # ---- Model ----
    vla_path: str = "openvla/openvla-7b"

    # ---- Dataset ----
    data_root_dir: Path = Path("datasets/rlds")
    dataset_name: str = "libero_spatial_no_noops"
    run_root_dir: Path = Path("runs")
    shuffle_buffer_size: int = 100_000

    # ---- Action head ----
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    use_film: bool = False
    num_images_in_input: int = 1
    use_proprio: bool = False

    # ---- Training ----
    batch_size: int = 8
    learning_rate: float = 5e-4
    lr_warmup_steps: int = 0
    num_steps_before_decay: int = 100_000
    grad_accumulation_steps: int = 1
    max_steps: int = 200_000
    image_aug: bool = True
    diffusion_sample_freq: int = 50

    # ---- CL-LoRA ----
    use_cl_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    shared_depth: int = 16
    orthogonal_init: bool = True
    freeze_a: bool = True
    use_block_scale: bool = True
    clip_weight: float = 1.0
    merge_lora_during_training: bool = False

    # ---- Knowledge Distillation ----
    use_kd: bool = True
    lambda_kd: float = 1.0
    teacher_checkpoint_dir: Optional[str] = None
    teacher_checkpoint_step: Optional[int] = None

    # ---- Replay ----
    use_replay: bool = False
    replay_buffer_dirs: List[str] = field(default_factory=list)
    replay_loss_weight: float = 1.0
    replay_every_n_steps: int = 1
    replay_sample_strategy: str = "round_robin"
    replay_batch_size: Optional[int] = None

    # ---- Stage management ----
    stage: int = 1
    previous_checkpoint_dir: Optional[str] = None
    previous_checkpoint_step: Optional[int] = None

    # ---- Checkpoint / Resume ----
    save_freq: int = 10_000
    save_latest_checkpoint_only: bool = False
    resume: bool = False
    resume_step: Optional[int] = None

    # ---- Logging ----
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"
    run_id_note: Optional[str] = None
    run_id_override: Optional[str] = None
    wandb_log_freq: int = 10

    # ---- Validation ----
    use_val_set: bool = False
    val_freq: int = 10_000
    val_time_limit: int = 180
    # fmt: on


# ==============================================================================
# Helpers
# ==============================================================================

def remove_ddp_in_checkpoint(state_dict: dict) -> dict:
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)


def count_parameters(module: nn.Module, name: str) -> None:
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


def get_run_id(cfg: TrainCLConfig) -> str:
    if cfg.run_id_override is not None:
        return cfg.run_id_override
    run_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    run_id += f"+cl-lora-r{cfg.lora_rank}+sd{cfg.shared_depth}"
    if cfg.use_kd:
        run_id += f"+kd{cfg.lambda_kd}"
    if cfg.use_replay:
        run_id += "+replay"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    return run_id


# ==============================================================================
# Teacher snapshot (PI-style: only save trainable CL-LoRA params)
# ==============================================================================

def save_teacher_snapshot(
    model: nn.Module,
    action_head: Optional[nn.Module],
    save_dir: Path,
    step: int,
) -> None:
    """Save snapshot of only the trainable CL-LoRA parameters and action head.

    Mirrors PI _snapshot_trainable_params(): captures a minimal parameter set
    so that teacher consumes negligible extra GPU memory during KD.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    snapshot = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            snapshot[f"model.{name}"] = param.data.cpu().clone()

    if action_head is not None:
        for name, param in action_head.named_parameters():
            if param.requires_grad:
                snapshot[f"action_head.{name}"] = param.data.cpu().clone()

    save_path = save_dir / f"teacher_snapshot--{step}.pt"
    torch.save(snapshot, save_path)
    print(f"[Teacher] Snapshot saved ({len(snapshot)} tensors) → {save_path}")


def load_teacher_snapshot(path: str) -> Dict[str, torch.Tensor]:
    print(f"[Teacher] Loading snapshot from {path}")
    return torch.load(path, weights_only=True, map_location="cpu")


def swap_to_teacher(model: nn.Module, action_head: Optional[nn.Module], teacher: dict, device) -> dict:
    """Swap teacher trainable params into the model. Returns original student params."""
    student_copy = {}
    for name, param in model.named_parameters():
        key = f"model.{name}"
        if key in teacher:
            student_copy[name] = param.data.clone()
            param.data.copy_(teacher[key].to(device))

    if action_head is not None:
        for name, param in action_head.named_parameters():
            key = f"action_head.{name}"
            if key in teacher:
                student_copy[f"ah_{name}"] = param.data.clone()
                param.data.copy_(teacher[key].to(device))

    return student_copy


def restore_student(model: nn.Module, action_head: Optional[nn.Module], student_copy: dict) -> None:
    """Restore student params after teacher KD forward pass."""
    for name, param in model.named_parameters():
        if name in student_copy:
            param.data.copy_(student_copy[name])

    if action_head is not None:
        for name, param in action_head.named_parameters():
            key = f"ah_{name}"
            if key in student_copy:
                param.data.copy_(student_copy[key])


# ==============================================================================
# Forward pass — returns both loss and intermediate predictions for KD/replay
# ==============================================================================

def run_forward_pass_extended(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    action_tokenizer,
    device_id,
    use_l1_regression: bool,
    use_diffusion: bool,
    use_proprio: bool,
    use_film: bool,
    num_patches: int,
    compute_diffusion_l1: bool = False,
    num_diffusion_steps_train: Optional[int] = None,
    return_predictions: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extended forward pass. When return_predictions=True, also returns
    (predicted_actions, actions_hidden_states) for KD loss computation.

    Returns:
        loss, metrics, predicted_actions, actions_hidden_states
    """
    metrics = {}
    predicted_actions = None
    actions_hidden_states_out = None

    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)

    if use_diffusion:
        noisy_dict = action_head.module.sample_noisy_actions(ground_truth_actions)
        noise, noisy_actions, diffusion_timestep_embeddings = (
            noisy_dict["noise"],
            noisy_dict["noisy_actions"],
            noisy_dict["diffusion_timestep_embeddings"],
        )
    else:
        noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=noisy_actions if use_diffusion else None,
            noisy_action_projector=noisy_action_projector if use_diffusion else None,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None,
            use_film=use_film,
        )

    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    if not (use_l1_regression or use_diffusion):
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)
        curr_action_accuracy = compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask=current_action_mask)
        curr_action_l1_loss = compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=current_action_mask)
        next_actions_accuracy = compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask)
        next_actions_l1_loss = compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask)
        metrics.update({
            "loss_value": loss.item(),
            "curr_action_accuracy": curr_action_accuracy.item(),
            "curr_action_l1_loss": curr_action_l1_loss.item(),
            "next_actions_accuracy": next_actions_accuracy.item(),
            "next_actions_l1_loss": next_actions_l1_loss.item(),
        })
    else:
        last_hidden_states = output.hidden_states[-1]
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )

        if use_l1_regression:
            predicted_actions = action_head.module.predict_action(actions_hidden_states)
            loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)

        if use_diffusion:
            noise_pred = action_head.module.predict_noise(actions_hidden_states)
            noise_pred = noise_pred.reshape(noise.shape)
            loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")

            if compute_diffusion_l1:
                with torch.no_grad():
                    predicted_actions = _run_diffusion_sampling(
                        vla=vla, action_head=action_head,
                        noisy_action_projector=noisy_action_projector,
                        proprio_projector=proprio_projector,
                        batch=batch, batch_size=batch_size,
                        num_patches=num_patches,
                        actions_shape=ground_truth_actions.shape,
                        device_id=device_id,
                        current_action_mask=current_action_mask,
                        next_actions_mask=next_actions_mask,
                        use_proprio=use_proprio, use_film=use_film,
                    )

        metrics.update({"loss_value": loss.item()})

        should_log_l1 = not use_diffusion or (use_diffusion and compute_diffusion_l1)
        if should_log_l1:
            ground_truth_curr_action = ground_truth_actions[:, 0]
            predicted_curr_action = predicted_actions[:, 0]
            ground_truth_next_actions = ground_truth_actions[:, 1:]
            predicted_next_actions = predicted_actions[:, 1:]
            curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
            next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
            metrics.update({
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),
            })

        if return_predictions:
            actions_hidden_states_out = actions_hidden_states

    return loss, metrics, predicted_actions, actions_hidden_states_out


def _run_diffusion_sampling(
    vla, action_head, noisy_action_projector, proprio_projector,
    batch, batch_size, num_patches, actions_shape, device_id,
    current_action_mask, next_actions_mask, use_proprio, use_film,
) -> torch.Tensor:
    noise = torch.randn(size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), device=device_id, dtype=torch.bfloat16)
    action_head.module.noise_scheduler.set_timesteps(action_head.module.num_diffusion_steps_train)

    curr_noisy_actions = noise
    for t in action_head.module.noise_scheduler.timesteps:
        timesteps = torch.Tensor([t]).repeat(batch_size).to(device_id)
        diffusion_timestep_embeddings = action_head.module.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device).unsqueeze(1)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                labels=batch["labels"],
                output_hidden_states=True,
                proprio=batch["proprio"] if use_proprio else None,
                proprio_projector=proprio_projector if use_proprio else None,
                noisy_actions=curr_noisy_actions,
                noisy_action_projector=noisy_action_projector,
                diffusion_timestep_embeddings=diffusion_timestep_embeddings,
                use_film=use_film,
            )
            last_hidden_states = output.hidden_states[-1]
            text_hidden_states = last_hidden_states[:, num_patches:-1]
            actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1).to(torch.bfloat16)
            noise_pred = action_head.module.predict_noise(actions_hidden_states)

        curr_noisy_actions = action_head.module.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

    return curr_noisy_actions.reshape(actions_shape)


# ==============================================================================
# Replay helpers
# ==============================================================================

def _build_replay_loaders(
    cfg: TrainCLConfig,
    batch_transform: RLDSBatchTransform,
) -> Tuple[Optional[List[DataLoader]], Optional[List[Any]]]:
    """Build DataLoaders for each replay buffer directory."""
    if not cfg.use_replay:
        return None, None
    if not cfg.replay_buffer_dirs:
        raise ValueError("--use_replay True but --replay_buffer_dirs is empty. "
                         "Provide at least one replay buffer directory.")
    loaders = []
    iters = []
    replay_bs = cfg.replay_batch_size if cfg.replay_batch_size is not None else cfg.batch_size

    for buf_dir in cfg.replay_buffer_dirs:
        dataset = PrototypeReplayDataset(buf_dir, batch_transform)
        if len(dataset) == 0:
            raise RuntimeError(f"Replay buffer at {buf_dir} contains 0 samples. "
                               "Check that the buffer was built correctly.")
        collator = PaddedCollatorForActionPrediction(
            model_max_length=2048, pad_token_id=0, padding_side="right"
        )
        loader = DataLoader(
            dataset,
            batch_size=replay_bs,
            shuffle=True,
            collate_fn=collator,
            num_workers=0,
        )
        loaders.append(loader)
        iters.append(iter(loader))

    print(f"[Replay] Initialized {len(loaders)} replay loaders from: {cfg.replay_buffer_dirs}")
    return loaders, iters


def _next_replay_batch(iters: list, loaders: list, strategy: str) -> dict:
    """Get next replay batch using round-robin or random strategy."""
    if strategy == "random":
        import random
        idx = random.randint(0, len(iters) - 1)
    else:
        idx = 0  # round-robin: external counter
        # We use a simpler approach: cycle through loaders
        _next_replay_batch._rr_counter = getattr(_next_replay_batch, '_rr_counter', 0)
        idx = _next_replay_batch._rr_counter % len(iters)
        _next_replay_batch._rr_counter += 1

    try:
        return next(iters[idx])
    except StopIteration:
        iters[idx] = iter(loaders[idx])
        return next(iters[idx])


def compute_smoothened_metrics(metrics_deques: dict) -> dict:
    smoothened = {}
    for name, dq in metrics_deques.items():
        if dq and len(dq) > 0:
            smoothened[name] = sum(dq) / len(dq)
    return smoothened


def log_metrics_to_wandb(metrics: dict, prefix: str, step: int, wandb_run) -> None:
    log_dict = {}
    for name, value in metrics.items():
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_run.log(log_dict, step=step)


# ==============================================================================
# Main training function
# ==============================================================================

@draccus.wrap()
def train_cl_lora(cfg: TrainCLConfig) -> None:
    assert cfg.use_l1_regression or cfg.use_diffusion, "Must use L1 regression or diffusion action head."
    assert cfg.max_steps > 0, f"max_steps must be > 0, got {cfg.max_steps}"

    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"CL-LoRA Training | Stage {cfg.stage} | Dataset: {cfg.dataset_name}")
    print(f"  KD: {cfg.use_kd} (λ={cfg.lambda_kd}) | Replay: {cfg.use_replay} (λ={cfg.replay_loss_weight})")

    run_id = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # ---- GPU setup ----
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"cl+{run_id}")

    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # ---- Model loading ----
    if model_is_on_hf_hub(cfg.vla_path):
        vla_download_path = snapshot_download(repo_id=cfg.vla_path)
        cfg.vla_path = vla_download_path
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)
    dist.barrier()

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)

    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # ---- CL-LoRA injection ----
    if cfg.use_cl_lora:
        total_layers = 32
        shared_split_ratio = max(1, cfg.shared_depth) / total_layers
        vla = inject_cl_lora_into_model(
            vla,
            rank=cfg.lora_rank,
            alpha=cfg.lora_rank,
            dropout=cfg.lora_dropout,
            shared_split_ratio=shared_split_ratio,
            orthogonal_init=cfg.orthogonal_init,
            freeze_a=cfg.freeze_a,
            use_block_scale=cfg.use_block_scale,
        )
        print(f"[CL-LoRA] Injected with shared_depth={cfg.shared_depth}, rank={cfg.lora_rank}")

    # ---- Load previous stage checkpoint (sequential training) ----
    if cfg.previous_checkpoint_dir is not None:
        prev_dir = cfg.previous_checkpoint_dir
        prev_step = cfg.previous_checkpoint_step or cfg.max_steps
        print(f"[Stage {cfg.stage}] Loading previous checkpoint from {prev_dir} (step {prev_step})")
        # Load action_head
        if cfg.use_l1_regression or cfg.use_diffusion:
            ah_state = load_checkpoint("action_head", prev_dir, prev_step)
            # action_head not yet created; will load after init
            _pending_action_head_state = ah_state
        else:
            _pending_action_head_state = None
        # Load vision_backbone if FiLM
        _pending_vision_state = None
        if cfg.use_film:
            _pending_vision_state = load_checkpoint("vision_backbone", prev_dir, prev_step)
    else:
        _pending_action_head_state = None
        _pending_vision_state = None

    # ---- FiLM ----
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        if _pending_vision_state is not None:
            vla.model.vision_backbone.load_state_dict(_pending_vision_state)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    # ---- DDP ----
    vla = wrap_ddp(vla, device_id, find_unused=True)

    # ---- Proprio projector ----
    proprio_projector = None
    if cfg.use_proprio:
        raise NotImplementedError("Proprio not yet implemented for CL-LoRA. Set --use_proprio=False.")

    # ---- Action head ----
    action_head = None
    if cfg.use_l1_regression:
        action_head = L1RegressionActionHead(
            input_dim=vla.module.llm_dim,
            hidden_dim=vla.module.llm_dim,
            action_dim=ACTION_DIM,
        ).to(torch.bfloat16).to(device_id)
        if _pending_action_head_state is not None:
            action_head.load_state_dict(_pending_action_head_state)
        action_head = wrap_ddp(action_head, device_id)
        count_parameters(action_head, "action_head")

    noisy_action_projector = None
    if cfg.use_diffusion:
        raise NotImplementedError("Diffusion not yet implemented for CL-LoRA. Set --use_l1_regression=True.")

    # ---- Teacher snapshot ----
    teacher_snapshot = None
    if cfg.use_kd:
        if cfg.teacher_checkpoint_dir is not None:
            teacher_path = cfg.teacher_checkpoint_dir
            if cfg.teacher_checkpoint_step is not None:
                teacher_path = os.path.join(teacher_path, f"teacher_snapshot--{cfg.teacher_checkpoint_step}.pt")
            elif os.path.isdir(teacher_path):
                # Auto-discover latest snapshot
                snapshots = sorted([f for f in os.listdir(teacher_path) if f.startswith("teacher_snapshot--")])
                if snapshots:
                    teacher_path = os.path.join(teacher_path, snapshots[-1])
            teacher_snapshot = load_teacher_snapshot(teacher_path)
            print(f"[KD] Loaded teacher snapshot with {len(teacher_snapshot)} tensors")
        else:
            print("[KD] WARNING: use_kd=True but no teacher_checkpoint_dir specified. KD will be skipped.")

    # ---- NUM_PATCHES ----
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    if cfg.use_proprio:
        NUM_PATCHES += 1
    if cfg.use_diffusion:
        NUM_PATCHES += 1

    # ---- Optimizer ----
    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    if action_head is not None:
        trainable_params += [p for p in action_head.parameters() if p.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    original_lr = optimizer.param_groups[0]["lr"]
    scheduler = MultiStepLR(optimizer, milestones=[cfg.num_steps_before_decay], gamma=0.1)

    # ---- Action tokenizer & data ----
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    use_wrist_image = cfg.num_images_in_input > 1

    batch_transform = RLDSBatchTransform(
        action_tokenizer, processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
    )
    train_dataset = RLDSDataset(
        cfg.data_root_dir, cfg.dataset_name, batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )

    # ---- Replay loaders ----
    replay_loaders, replay_iters = _build_replay_loaders(cfg, batch_transform)

    # ---- Metrics tracking ----
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "loss_task": deque(maxlen=cfg.grad_accumulation_steps),
        "loss_kd": deque(maxlen=cfg.grad_accumulation_steps),
        "loss_replay": deque(maxlen=cfg.grad_accumulation_steps),
    }

    # ---- Training loop ----
    with tqdm.tqdm(total=cfg.max_steps, leave=True) as progress:
        vla.train()
        if action_head is not None:
            action_head.train()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            log_step = gradient_step_idx

            # 1. Current task forward pass
            compute_diffusion_l1 = cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0
            loss_task, metrics, student_pred, _ = run_forward_pass_extended(
                vla=vla, action_head=action_head,
                noisy_action_projector=noisy_action_projector,
                proprio_projector=proprio_projector,
                batch=batch, action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=NUM_PATCHES,
                compute_diffusion_l1=compute_diffusion_l1,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
                return_predictions=True,
            )

            # 2. KD loss (student_pred still has gradients; teacher runs with frozen params via swap)
            loss_kd = torch.tensor(0.0, device=device_id)
            if cfg.use_kd and teacher_snapshot is not None and student_pred is not None:
                student_copy = swap_to_teacher(vla.module, action_head.module if action_head is not None else None, teacher_snapshot, device_id)
                with torch.no_grad():
                    _, _, teacher_pred, _ = run_forward_pass_extended(
                        vla=vla, action_head=action_head,
                        noisy_action_projector=noisy_action_projector,
                        proprio_projector=proprio_projector,
                        batch=batch, action_tokenizer=action_tokenizer,
                        device_id=device_id,
                        use_l1_regression=cfg.use_l1_regression,
                        use_diffusion=cfg.use_diffusion,
                        use_proprio=cfg.use_proprio,
                        use_film=cfg.use_film,
                        num_patches=NUM_PATCHES,
                        compute_diffusion_l1=False,
                        num_diffusion_steps_train=None,
                        return_predictions=True,
                    )
                restore_student(vla.module, action_head.module if action_head is not None else None, student_copy)
                if teacher_pred is not None:
                    loss_kd = F.mse_loss(student_pred, teacher_pred.detach())

            # 3. Replay loss
            loss_replay = torch.tensor(0.0, device=device_id)
            metrics["loss_replay"] = 0.0
            if cfg.use_replay and replay_iters is not None:
                use_replay_this_step = (log_step % cfg.replay_every_n_steps == 0)
                if use_replay_this_step:
                    replay_batch = _next_replay_batch(replay_iters, replay_loaders, cfg.replay_sample_strategy)
                    loss_replay, replay_metrics, _, _ = run_forward_pass_extended(
                        vla=vla, action_head=action_head,
                        noisy_action_projector=noisy_action_projector,
                        proprio_projector=proprio_projector,
                        batch=replay_batch, action_tokenizer=action_tokenizer,
                        device_id=device_id,
                        use_l1_regression=cfg.use_l1_regression,
                        use_diffusion=cfg.use_diffusion,
                        use_proprio=cfg.use_proprio,
                        use_film=cfg.use_film,
                        num_patches=NUM_PATCHES,
                        compute_diffusion_l1=False,
                        num_diffusion_steps_train=None,
                        return_predictions=False,
                    )
                    metrics["loss_replay"] = loss_replay.item()

            # 4. Total loss
            loss_total = loss_task + cfg.lambda_kd * loss_kd + cfg.replay_loss_weight * loss_replay

            # Store metrics
            metrics["loss_task"] = loss_task.item()
            metrics["loss_kd"] = loss_kd.item() if isinstance(loss_kd, torch.Tensor) else loss_kd
            for name in recent_metrics:
                if name in metrics:
                    recent_metrics[name].append(metrics[name])

            # Backward
            normalized_loss = loss_total / cfg.grad_accumulation_steps
            normalized_loss.backward()

            # Optimizer step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()
                # Live metrics on progress bar
                smoothened = compute_smoothened_metrics(recent_metrics)
                postfix = {"loss": f"{smoothened.get('loss_value', 0):.4f}"}
                if cfg.use_kd:
                    postfix["kd"] = f"{smoothened.get('loss_kd', 0):.4f}"
                if cfg.use_replay:
                    postfix["rep"] = f"{smoothened.get('loss_replay', 0):.4f}"
                progress.set_postfix(postfix)

            # LR warmup
            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            # Logging
            smoothened = compute_smoothened_metrics(recent_metrics)
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                log_metrics_to_wandb(smoothened, "CL Train", log_step, wandb)
                wandb.log({"CL Train/Learning Rate": scheduler.get_last_lr()[0]}, step=log_step)

            # Checkpoint
            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                if distributed_state.is_main_process:
                    checkpoint_dir = run_dir if cfg.save_latest_checkpoint_only else Path(str(run_dir) + f"--{log_step}_chkpt")
                    checkpoint_dir = Path(checkpoint_dir)
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)

                    processor.save_pretrained(checkpoint_dir)

                    # Save CL-LoRA config (for eval alignment)
                    import json as _json
                    cl_config = {
                        "lora_rank": cfg.lora_rank,
                        "alpha": cfg.lora_rank,
                        "shared_split_ratio": max(1, cfg.shared_depth) / 32,
                        "shared_depth": cfg.shared_depth,
                        "orthogonal_init": cfg.orthogonal_init,
                        "freeze_a": cfg.freeze_a,
                        "use_block_scale": cfg.use_block_scale,
                    }
                    with open(checkpoint_dir / "cl_lora_config.json", "w") as _f:
                        _json.dump(cl_config, _f)

                    # Save CL-LoRA adapter (trainable params only)
                    cl_lora_state = {k: v.cpu() for k, v in vla.module.state_dict().items() if any(x in k for x in ['lora_a', 'lora_b', 'block_scale'])}
                    torch.save(cl_lora_state, checkpoint_dir / "cl_lora_adapter.pt")

                    # Save action head
                    if action_head is not None:
                        torch.save(action_head.module.state_dict(), checkpoint_dir / f"action_head--{log_step}_checkpoint.pt")

                    # Save teacher snapshot (for next stage)
                    save_teacher_snapshot(vla.module, action_head.module if action_head is not None else None, checkpoint_dir, log_step)

                    # Save dataset statistics
                    save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)

                    print(f"Checkpoint saved at step {log_step} → {checkpoint_dir}")

                dist.barrier()

            if gradient_step_idx > 0 and log_step >= cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping...")
                break

    # ---- Final save ----
    if distributed_state.is_main_process:
        final_dir = run_dir if cfg.save_latest_checkpoint_only else Path(str(run_dir) + f"--{cfg.max_steps}_chkpt")
        final_dir = Path(final_dir)
        final_dir.mkdir(parents=True, exist_ok=True)
        # Save CL-LoRA config
        import json as _json
        cl_config = {
            "lora_rank": cfg.lora_rank,
            "alpha": cfg.lora_rank,
            "shared_split_ratio": max(1, cfg.shared_depth) / 32,
            "shared_depth": cfg.shared_depth,
            "orthogonal_init": cfg.orthogonal_init,
            "freeze_a": cfg.freeze_a,
            "use_block_scale": cfg.use_block_scale,
        }
        with open(final_dir / "cl_lora_config.json", "w") as _f:
            _json.dump(cl_config, _f)
        processor.save_pretrained(final_dir)
        cl_lora_state = {k: v.cpu() for k, v in vla.module.state_dict().items() if any(x in k for x in ['lora_a', 'lora_b', 'block_scale'])}
        torch.save(cl_lora_state, final_dir / "cl_lora_adapter.pt")
        if action_head is not None:
            torch.save(action_head.module.state_dict(), final_dir / f"action_head--{cfg.max_steps}_checkpoint.pt")
        save_teacher_snapshot(vla.module, action_head.module if action_head is not None else None, final_dir, cfg.max_steps)
        save_dataset_statistics(train_dataset.dataset_statistics, final_dir)
        print(f"Final checkpoint saved → {final_dir}")

    dist.barrier()
    print("CL-LoRA training complete.")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    train_cl_lora()

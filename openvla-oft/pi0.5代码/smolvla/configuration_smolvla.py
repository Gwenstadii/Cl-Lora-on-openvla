# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import OBS_IMAGES


@PreTrainedConfig.register_subclass("smolvla")
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True
    enable_cl_lora: bool = False
    # Historical "ordinary LoRA" baseline from the 2026-04-17 CL-LoRA experiment notes.
    # This is not PEFT LoRA. It uses the CL-LoRA routing codepath with zero shared layers
    # and disables block-wise scaling so the run behaves like the high-scoring single-task
    # baseline that reached ~90% on Task A when initialized from a pretrained SmolVLA policy.
    cl_lora_ordinary_baseline: bool = False
    # Compatibility switch for the v1/v2/v3 single-task SmolVLA regime. When enabled together with
    # `enable_cl_lora=true`, the action expert remains trainable and the task-head projections are
    # wrapped with CL-LoRA as well. This is intentionally *not* the same baseline as ordinary PEFT.
    legacy_hybrid_lora: bool = False
    lora_r: int = 8
    lora_alpha: float = 16.0
    cl_shared_layers: int = 9
    use_block_scale: bool = True
    orth_loss_weight: float = 0.0
    enable_kd: bool = False
    teacher_path: str | None = None
    kd_weight: float = 1.0
    enable_replay: bool = False
    replay_dir: str | None = None
    replay_weight: float = 1.0
    replay_task_fallback: str | None = None
    replay_require_task: bool = True
    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = False  # Set to False in case of training the expert from scratch. True when init from pretrained SmolVLA weights

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )
        if self.cl_lora_ordinary_baseline:
            if not self.enable_cl_lora:
                raise ValueError(
                    "`cl_lora_ordinary_baseline=true` requires `enable_cl_lora=true` because this baseline "
                    "runs through the CL-LoRA implementation rather than PEFT LoRA."
                )
            if self.legacy_hybrid_lora:
                raise ValueError(
                    "`cl_lora_ordinary_baseline=true` cannot be combined with `legacy_hybrid_lora=true` because "
                    "they describe two different historical regimes."
                )
            self.cl_shared_layers = 0
            self.use_block_scale = False
        if self.enable_cl_lora and not self.train_expert_only:
            raise ValueError(
                "`enable_cl_lora=true` currently requires `train_expert_only=true` to avoid interfering with the "
                "base SmolVLA backbone."
            )
        if self.legacy_hybrid_lora and not self.enable_cl_lora:
            raise ValueError("`legacy_hybrid_lora=true` requires `enable_cl_lora=true`.")
        if self.enable_cl_lora and self.lora_r <= 0:
            raise ValueError(f"`lora_r` must be positive when `enable_cl_lora=true`, got {self.lora_r}.")
        if self.cl_shared_layers < 0:
            raise ValueError(f"`cl_shared_layers` must be >= 0, got {self.cl_shared_layers}.")
        if self.orth_loss_weight < 0:
            raise ValueError(f"`orth_loss_weight` must be >= 0, got {self.orth_loss_weight}.")
        if self.enable_kd and not self.teacher_path:
            raise ValueError("`enable_kd=true` requires `teacher_path` to be set.")
        if self.teacher_path and not self.enable_kd:
            raise ValueError("`teacher_path` is set but `enable_kd` is false. Set `enable_kd=true` or remove it.")
        if self.kd_weight < 0:
            raise ValueError(f"`kd_weight` must be >= 0, got {self.kd_weight}.")
        if self.enable_replay and not self.replay_dir:
            raise ValueError("`enable_replay=true` requires `replay_dir` to be set.")
        if self.replay_dir and not self.enable_replay:
            raise ValueError(
                "`replay_dir` is set but `enable_replay` is false. Set `enable_replay=true` or remove it."
            )
        if self.replay_weight < 0:
            raise ValueError(f"`replay_weight` must be >= 0, got {self.replay_weight}.")

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

# Config_of_building_PI0/05
import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0

# 模型配置类
@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore  代表tokenized_prompt的最大长度，pi0.5版本默认200，pi0版本默认48
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    # CL-LoRA v2 controls. By default these are off, so base pi0/pi0.5
    # and ordinary LoRA configs keep their original behavior.
    cl_task_id: int = 0
    cl_num_tasks: int = 4
    cl_pali_lora_bank: bool = False
    cl_action_lora_bank: bool = False
    cl_pali_block_weight_bank: bool = False
    cl_action_block_weight_bank: bool = False
    cl_freeze_shared_lora: bool = False
    cl_freeze_pali_lora_b: bool = False
    cl_freeze_action_lora_b: bool = False
    cl_freeze_lora_a: bool = False

    # dataclass已自动生成__init__,此处__post_init__用于根据pi0.5条件来设置相关参数
    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)  # 如果是pi0.5版本则离散状态输入为True，否则为False

    # 取得模型类型
    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    # 用于创造模型实例的方法，输入相关随机种子
    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    # 规定输入输出的规格
    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)  # 图像输入[batch_size, 224, 224, 3]，float32类型
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)  # 图像掩码输入[batch_size]，bool类型，代表每个样本是否有对应的图像输入（因为可能存在部分图像缺失的情况）

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),  # 原始状态输入[batch_size, action_dim]，float32类型 (后续才在tokenizer里被离散化成tokenized_prompt的一部分输入到pi0.5模型里)
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),  # tokenized_prompt输入[batch_size, max_token_len]，int32类型，pi0.5此处已包含语言指令和离散化的状态输入，pi0则只有语言指令（因为状态输入是连续的了）
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)  # 动作格式

        return observation_spec, action_spec

    # ①官方冻结 （冻结 llm 内非 LoRA 参数，非 llm 模块如SigLIP、动作头等仍可训练）
    def get_freeze_filter(self, *, allow_block_weights: bool = False) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []  # 存储那些需要被冻结的参数的过滤器列表
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        # 若第一个专家的配置包含lora，则冻结第一个专家的参数，如果第二个专家的配置不包含lora，则不冻结第二个专家的参数
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        # 若第二个专家的配置包含lora，则冻结第二个专家的参数
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True
        # 如果has_lora为true，则把含lora的参数都排除掉
        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if allow_block_weights:
            filters.append(
                nnx.Not(nnx_utils.PathRegex(r".*cl_block_weights.*")),
            )
        # 如果filter列表还为空，说明没有任何参数需要被冻结，直接返回nnx.Nothing，否则返回filter里所有过滤器的交集（即满足所有过滤条件的参数才会被冻结）
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)

    # ②严格LoRA冻结过滤器（冻结所有非LoRA参数，仅训lora矩阵和块权重，CL-LORA专用）
    def get_strict_lora_freeze_filter(
        self,
        *,
        allow_block_weights: bool = False,
    ) -> nnx.filterlib.Filter:
        """Strict adapter-only freeze.
        Trainable:
        - LoRA params
        - optional CL block weights
        Frozen:
        - everything else
        """
        has_lora = ("lora" in self.paligemma_variant) or ("lora" in self.action_expert_variant)
        if not has_lora and not allow_block_weights:
            return nnx.Nothing
        filters = [
            nnx.Param,
            # Keep all LoRA params trainable.
            nnx.Not(nnx_utils.PathRegex(r".*lora.*")),
        ]
        if allow_block_weights:
            # CL-LoRA block weights are named like cl_block_weights_0, cl_block_weights_1, ...
            filters.append(
                nnx.Not(nnx_utils.PathRegex(r".*cl_block_weights.*")),
            )
        # freeze_filter matches params that should be frozen
        return nnx.All(*filters)

    # ③平衡LoRA冻结过滤器（冻结Siglip和LLM大主干，保留lora、块权重、动作投影层、时间投影层等的训练）
    def get_balanced_lora_freeze_filter(
        self,
        *,
        allow_block_weights: bool = False,
        allow_action_block_weights_only: bool = False,
        allow_action_expert_norms: bool = False,
    ) -> nnx.filterlib.Filter:
        """Balanced freeze: adapter + lightweight action-side heads.
        Trainable:
        - LoRA params
        - optional CL block weights
        - optional action-expert-only CL block weights
        - lightweight action-side heads
        - optional action-expert AdaRMSNorm and final norm parameters
        Frozen:
        - SigLIP vision tower
        - all other backbone params
        """
        has_lora = ("lora" in self.paligemma_variant) or ("lora" in self.action_expert_variant)
        if (
            not has_lora
            and not allow_block_weights
            and not allow_action_block_weights_only
            and not allow_action_expert_norms
        ):
            return nnx.Nothing
        small_heads_filter = nnx_utils.PathRegex(
            r"(?:action_in_proj|action_out_proj|time_mlp_in|time_mlp_out|"
            r"state_proj|action_time_mlp_in|action_time_mlp_out)(?:/.*)?"
        )
        filters = [
            nnx.Param,
            # Keep all LoRA params trainable.
            nnx.Not(nnx_utils.PathRegex(r".*lora.*")),
            # Keep lightweight non-LLM heads trainable.
            nnx.Not(small_heads_filter),
        ]
        if allow_action_expert_norms:
            # Expert 1 is the action expert. This admits only its AdaRMSNorm
            # modulation and final norm parameters, not attention/FFN kernels.
            action_expert_norms_filter = nnx_utils.PathRegex(
                r".*(?:pre_attention_norm_1|pre_ffw_norm_1|final_norm_1)(?:/.*)?"
            )
            filters.append(nnx.Not(action_expert_norms_filter))
        if allow_action_block_weights_only:
            # In pi0/pi0.5 expert order, cl_block_weights_0 is PaliGemma and
            # cl_block_weights_1 is the action expert. Later CL stages should
            # keep the visual-language block routing stable while adapting the
            # action expert to the new task.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(r".*cl_block_weights_1.*")),
            )
        elif allow_block_weights:
            filters.append(
                nnx.Not(nnx_utils.PathRegex(r".*cl_block_weights.*")),
            )
        # freeze_filter matches params that should be frozen
        return nnx.All(*filters)

from __future__ import annotations

import re

import torch
import torch.nn as nn


class CLLoRALinear(nn.Module):
    """LoRA wrapper with explicit shared/specific task isolation."""

    _DEFAULT_TASK_ID = "default"

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int,
        lora_alpha: float,
        is_shared: bool = True,
        use_block_scale: bool = False,
    ) -> None:
        super().__init__()

        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"`base_layer` must be an nn.Linear, got {type(base_layer)!r}.")
        if r <= 0:
            raise ValueError(f"`r` must be a positive integer, got {r}.")

        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.is_shared = is_shared
        self.use_block_scale = use_block_scale and not is_shared
        self.active_task_id: str | None = None
        self.allow_new_tasks = True

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        if self.is_shared:
            self.lora_A = nn.Parameter(torch.empty(in_features, r), requires_grad=False)
            self.lora_B = nn.Parameter(torch.empty(r, out_features))
            if use_block_scale:
                self.block_scale = nn.Parameter(torch.ones(1))
            else:
                self.register_parameter("block_scale", None)
            self.reset_parameters()
        else:
            self.lora_A = nn.ParameterDict()
            self.lora_B = nn.ParameterDict()
            self.block_scale = nn.ParameterDict() if self.use_block_scale else None
            self.add_task(self._DEFAULT_TASK_ID)
            self.active_task_id = self._DEFAULT_TASK_ID

    def reset_parameters(self) -> None:
        if not self.is_shared:
            return
        nn.init.orthogonal_(self.lora_A)
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def _normalize_task_id(self, task_id: str | None) -> str:
        task_name = self._DEFAULT_TASK_ID if task_id is None else str(task_id).strip()
        if not task_name:
            task_name = self._DEFAULT_TASK_ID
        task_name = re.sub(r"[^0-9A-Za-z_]+", "_", task_name)
        return task_name or self._DEFAULT_TASK_ID

    def _create_task(self, task_key: str) -> None:
        self.lora_A[task_key] = nn.Parameter(
            torch.empty(self.base_layer.in_features, self.r)
        )
        self.lora_B[task_key] = nn.Parameter(
            torch.empty(self.r, self.base_layer.out_features)
        )
        nn.init.orthogonal_(self.lora_A[task_key])
        nn.init.zeros_(self.lora_B[task_key])
        if self.block_scale is not None:
            self.block_scale[task_key] = nn.Parameter(torch.ones(1))

    def add_task(self, task_id: str | None, *, allow_existing_only: bool = False) -> str:
        task_key = self._normalize_task_id(task_id)
        if self.is_shared:
            return task_key

        if task_key not in self.lora_A:
            if allow_existing_only:
                raise KeyError(
                    f"Task adapter '{task_key}' is not registered. "
                    "Register task ids before creating the optimizer."
                )
            self._create_task(task_key)
        return task_key

    def set_active_task(self, task_id: str | None) -> str:
        task_key = self.add_task(task_id, allow_existing_only=not self.allow_new_tasks)
        if not self.is_shared:
            self.active_task_id = task_key
        return task_key

    def get_active_task(self) -> str | None:
        return self.active_task_id if not self.is_shared else None

    def get_registered_task_ids(self, *, include_default: bool = False) -> list[str]:
        if self.is_shared:
            return []

        task_ids = sorted(self.lora_A.keys())
        if include_default:
            return task_ids
        return [task_id for task_id in task_ids if task_id != self._DEFAULT_TASK_ID]

    def block_scale_orthogonality_loss(self) -> torch.Tensor | None:
        if self.is_shared or self.block_scale is None:
            return None

        task_ids = self.get_registered_task_ids()
        if len(task_ids) <= 1:
            return None

        scales = torch.stack(
            [self.block_scale[task_id].reshape(-1) for task_id in task_ids],
            dim=0,
        )
        scales = torch.nn.functional.normalize(scales, p=2, dim=1, eps=1e-8)
        gram = scales @ scales.transpose(0, 1)
        off_diag = gram - torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        return off_diag.pow(2).mean()

    def get_task_block_scale(self, task_id: str) -> torch.Tensor | None:
        if self.is_shared or self.block_scale is None:
            return None
        task_key = self._normalize_task_id(task_id)
        if task_key not in self.block_scale:
            return None
        return self.block_scale[task_key].reshape(-1)

    def has_task(self, task_id: str | None) -> bool:
        if self.is_shared:
            return True
        return self._normalize_task_id(task_id) in self.lora_A

    def register_task_ids(self, task_ids: list[str] | tuple[str, ...] | set[str]) -> None:
        if self.is_shared:
            return
        for task_id in task_ids:
            self.add_task(task_id)

    def set_allow_new_tasks(self, allow: bool) -> None:
        self.allow_new_tasks = allow

    def _get_specific_parameters(
        self,
        task_id: str | None,
        output_dtype: torch.dtype,
        output_device: torch.device,
    ) -> tuple[nn.Parameter, nn.Parameter, torch.Tensor | float]:
        if task_id is None:
            task_id = self.active_task_id
            if task_id is None and len(self.lora_A) == 1:
                task_id = next(iter(self.lora_A.keys()))
            if task_id is None:
                raise ValueError(
                    "Multiple task-specific adapters exist, but no `task_id` was provided for routing."
                )

        task_key = self.set_active_task(task_id)
        lora_a = self.lora_A[task_key]
        lora_b = self.lora_B[task_key]
        scale: torch.Tensor | float = 1.0
        if self.block_scale is not None:
            scale = self.block_scale[task_key].to(dtype=output_dtype, device=output_device)
        return lora_a, lora_b, scale

    def forward(self, x: torch.Tensor, task_id: str | None = None) -> torch.Tensor:
        base_output = self.base_layer(x)

        if self.is_shared:
            lora_a = self.lora_A.to(dtype=x.dtype, device=x.device)
            lora_b = self.lora_B.to(dtype=x.dtype, device=x.device)
            lora_hidden = x @ lora_a
            lora_output = lora_hidden @ lora_b
            scale = 1.0 if self.block_scale is None else self.block_scale.to(
                dtype=base_output.dtype, device=base_output.device
            )
            return base_output + lora_output * self.scaling * scale

        lora_a, lora_b, scale = self._get_specific_parameters(
            task_id=task_id,
            output_dtype=base_output.dtype,
            output_device=base_output.device,
        )
        lora_hidden = x @ lora_a.to(dtype=x.dtype, device=x.device)
        lora_output = lora_hidden @ lora_b.to(dtype=x.dtype, device=x.device)
        return base_output + lora_output * self.scaling * scale

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        legacy_weight_key = f"{prefix}weight"
        base_weight_key = f"{prefix}base_layer.weight"
        if legacy_weight_key in state_dict and base_weight_key not in state_dict:
            state_dict[base_weight_key] = state_dict.pop(legacy_weight_key)

        legacy_bias_key = f"{prefix}bias"
        base_bias_key = f"{prefix}base_layer.bias"
        if legacy_bias_key in state_dict and base_bias_key not in state_dict:
            state_dict[base_bias_key] = state_dict.pop(legacy_bias_key)

        if not self.is_shared:
            task_keys = set()
            for key in state_dict:
                if key.startswith(f"{prefix}lora_A."):
                    task_keys.add(key[len(f"{prefix}lora_A.") :].split(".", 1)[0])
                elif key.startswith(f"{prefix}lora_B."):
                    task_keys.add(key[len(f"{prefix}lora_B.") :].split(".", 1)[0])
                elif self.block_scale is not None and key.startswith(f"{prefix}block_scale."):
                    task_keys.add(key[len(f"{prefix}block_scale.") :].split(".", 1)[0])

            for task_key in sorted(task_keys):
                self.add_task(task_key)

            preferred_task_key = None
            non_default_task_keys = [task_key for task_key in sorted(task_keys) if task_key != self._DEFAULT_TASK_ID]
            if non_default_task_keys:
                preferred_task_key = non_default_task_keys[0]
            elif self._DEFAULT_TASK_ID in task_keys:
                preferred_task_key = self._DEFAULT_TASK_ID

            if preferred_task_key is not None:
                self.active_task_id = preferred_task_key

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

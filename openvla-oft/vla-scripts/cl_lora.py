"""
CL-LoRA module for continual learning on OpenVLA.

Implements CL-LoRA as described in:
  "CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning"
  He, Duan, Zhu — CVPR 2025

Key design (Section 4):
  1. Task-SHARED adapters in first l blocks: B (down-proj) = fixed random orthogonal,
     A (up-proj) = zero-init + trainable. B frozen, A continuously updated.
  2. Task-SPECIFIC adapters in remaining N-l blocks: standard LoRA (A/B both
     trainable) with learnable block-wise scaling weights mu_t^i (per module).
  3. Orthogonality loss L_orth between block weight vectors of different tasks.
  4. Visual backbone and LLM base weights fully frozen.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLLoRALinear(nn.Module):
    """CL-LoRA linear layer. Replaces nn.Linear in transformer blocks.

    Shared layers (paper Eq.6):
      B(lora_a, down-proj) = fixed random orthogonal, NOT trained
      A(lora_b, up-proj)   = zero-init, TRAINABLE

    Specific layers (paper Eq.5, 11):
      A(lora_a, down-proj) = kaiming init, trainable
      B(lora_b, up-proj)   = zero-init, trainable
      scaled by learnable block_scale mu (per module)

    Forward:  result = Wx + scaling * mu * lora_b @ lora_a @ x
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 32,
        alpha: float = 32.0,
        dropout: float = 0.0,
        is_shared: bool = True,
        orthogonal_init: bool = True,
        freeze_a: bool = True,
        use_block_scale: bool = True,
    ):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.is_shared = is_shared

        # Freeze base weight
        self.weight = base_layer.weight
        self.weight.requires_grad = False
        if base_layer.bias is not None:
            self.bias = base_layer.bias
            self.bias.requires_grad = False
        else:
            self.register_parameter('bias', None)

        # LoRA A (down-proj) and B (up-proj)
        self.lora_a = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Per-module learnable block_scale (specific layers only)
        if not self.is_shared and use_block_scale:
            self.block_scale = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter('block_scale', None)

        self._orthogonal_init = orthogonal_init
        self._freeze_a = freeze_a and is_shared
        self.reset_parameters()

    def reset_parameters(self):
        # Paper Eq.6: shared B(down=our lora_a) = random orthogonal, frozen
        # Paper Eq.5: specific A(down=our lora_a) = standard random init, trainable
        if self.is_shared and self._orthogonal_init:
            nn.init.orthogonal_(self.lora_a)
        else:
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        # Up-projection (our lora_b): always zero-init
        nn.init.zeros_(self.lora_b)

        if self._freeze_a:
            self.lora_a.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)

        lora_out = F.linear(self.dropout(x), self.lora_a)
        lora_out = F.linear(lora_out, self.lora_b)

        scale = self.scaling
        if self.block_scale is not None:
            effective_scale = 1.0 + 0.5 * torch.tanh(self.block_scale)
            scale = scale * effective_scale

        return result + lora_out * scale

    def collect_block_scale(self):
        """Return block_scale value for L_orth computation (paper Eq.12)."""
        if self.block_scale is not None:
            return self.block_scale.detach().clone()
        return None


def inject_cl_lora_into_model(
    model,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    shared_split_ratio: float = 0.5,
    orthogonal_init: bool = True,
    freeze_a: bool = True,
    use_block_scale: bool = True,
    target_modules: list = None,
):
    """Inject CL-LoRA into LlamaDecoderLayer transformer blocks.

    Block weights: per-module learnable scalar inside each CLLoRALinear (mu_t^i).
    L_orth uses the average block_scale per specific layer.
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

    llama_layers = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == "LlamaDecoderLayer":
            llama_layers.append((name, module))

    total_depth = len(llama_layers)
    if total_depth == 0:
        raise RuntimeError("No LlamaDecoderLayer found in model.")

    shared_depth_count = max(1, int(total_depth * shared_split_ratio))
    specific_depth_count = total_depth - shared_depth_count

    print(f"\n--- Injecting CL-LoRA (per-module block_scale, PI scope) ---")
    print(f"LlamaDecoderLayer depth: {total_depth}")
    print(f"Shared layers (fixed B, train A):  0 to {shared_depth_count - 1}")
    print(f"Specific layers (both train, +mu): {shared_depth_count} to {total_depth - 1}")
    print(f"Target modules:                     {target_modules}")
    print(f"Vision backbone:                    UNTOUCHED (frozen)\n")

    replaced_shared = 0
    replaced_specific = 0

    for layer_idx, (layer_name, layer_module) in enumerate(llama_layers):
        is_shared = layer_idx < shared_depth_count

        for name, module in layer_module.named_modules():
            if any(name.endswith(t) for t in target_modules) and isinstance(module, nn.Linear):
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                child_name = name.rsplit('.', 1)[-1]

                parent_module = layer_module
                if parent_name:
                    for part in parent_name.split('.'):
                        parent_module = getattr(parent_module, part)

                cl_lora_layer = CLLoRALinear(
                    base_layer=module,
                    rank=rank, alpha=alpha, dropout=dropout,
                    is_shared=is_shared,
                    orthogonal_init=orthogonal_init,
                    freeze_a=freeze_a,
                    use_block_scale=use_block_scale,
                ).to(module.weight.device).to(module.weight.dtype)

                setattr(parent_module, child_name, cl_lora_layer)
                if is_shared:
                    replaced_shared += 1
                else:
                    replaced_specific += 1

    print(f"Replaced {replaced_shared} shared + {replaced_specific} specific = "
          f"{replaced_shared + replaced_specific} Linear layers with CLLoRALinear.\n")
    return model

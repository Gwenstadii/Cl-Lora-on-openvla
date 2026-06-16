"""
CL-LoRA module for continual learning on OpenVLA.
Implements CLLoRALinear with shared/specific layer split,
orthogonal initialization, frozen LoRA-A, and block-scale gating.

Reference: PI0.5 CL-LoRA (openpi/models/lora.py)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLLoRALinear(nn.Module):
    """Continual-Learning LoRA linear layer.

    Shared layers: LoRA-A orthogonally initialized and frozen (protects old knowledge).
    Specific layers: LoRA-A and LoRA-B both trainable, with learnable block_scale gating.
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

        # LoRA A and B matrices
        self.lora_a = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Block-scale gating (specific layers only)
        if not self.is_shared and use_block_scale:
            self.block_scale = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter('block_scale', None)

        self._orthogonal_init = orthogonal_init
        self._freeze_a = freeze_a and is_shared
        self._use_block_scale = use_block_scale
        self.reset_parameters()

    def reset_parameters(self):
        if self.is_shared and self._orthogonal_init:
            nn.init.orthogonal_(self.lora_a)
        else:
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
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


def _is_llama_decoder_parent(name: str, llama_layer_names: set) -> bool:
    """Check if a module name belongs to a LlamaDecoderLayer subtree."""
    for ll_name in llama_layer_names:
        if name == ll_name or name.startswith(ll_name + "."):
            return True
    return False


def inject_cl_lora_into_model(
    model,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    shared_split_ratio: float = 0.5,
    orthogonal_init: bool = True,
    freeze_a: bool = True,
    use_block_scale: bool = True,
):
    """Replace ALL nn.Linear layers with CLLoRALinear (matching PEFT all-linear scope).

    For layers inside LlamaDecoderLayer: apply shared/specific depth-based split.
    For all other Linear layers (projectors, lm_head, etc.): treat as SPECIFIC (fully trainable).

    This ensures CL-LoRA covers the same set of layers as PEFT's ``target_modules="all-linear"``,
    making standard-LoRA and CL-LoRA directly comparable under the control-variable principle.
    """
    # ---- Phase 1: discover LlamaDecoderLayer depth ----
    llama_layer_names = set()
    for name, module in model.named_modules():
        if module.__class__.__name__ == "LlamaDecoderLayer":
            llama_layer_names.add(name)

    total_llama_depth = len(llama_layer_names)
    if total_llama_depth == 0:
        raise RuntimeError("No LlamaDecoderLayer found in model.")
    shared_depth_count = max(1, int(total_llama_depth * shared_split_ratio))

    # Build ordering: sort decoder layer names by their numeric index
    llama_ordered = []
    for name in llama_layer_names:
        # name is like "language_model.model.layers.0" or "model.layers.0"
        idx = None
        parts = name.split(".")
        for p in parts:
            try:
                idx = int(p)
                break
            except ValueError:
                continue
        llama_ordered.append((idx if idx is not None else 9999, name))
    llama_ordered.sort(key=lambda x: x[0])
    depth_rank = {name: i for i, (_, name) in enumerate(llama_ordered)}

    # ---- Phase 2: replace EVERY nn.Linear in the model ----
    print(f"\n--- Injecting CL-LoRA (all-linear scope) ---")
    print(f"LlamaDecoderLayer depth: {total_llama_depth}")
    print(f"Shared layers (frozen A):  ranks 0 to {shared_depth_count - 1}")
    print(f"Specific layers             ranks {shared_depth_count} to {total_llama_depth - 1}")
    print(f"Non-decoder Linear layers:  ALL treated as specific\n")

    replaced_llama = 0
    replaced_other = 0

    # We must walk the module tree and replace leaves in-place.
    # named_modules() gives full paths; we replace via parent.setattr.
    module_list = list(model.named_modules())

    for full_name, module in module_list:
        if not isinstance(module, nn.Linear):
            continue
        # Skip layers already replaced (CLLoRALinear wraps the original Linear)
        if isinstance(module, CLLoRALinear):
            continue

        # Determine whether this Linear lives inside a LlamaDecoderLayer
        is_in_llama = _is_llama_decoder_parent(full_name, llama_layer_names)

        if is_in_llama:
            # Find which LlamaDecoderLayer this belongs to
            parent_llama_name = full_name
            while parent_llama_name not in llama_layer_names and "." in parent_llama_name:
                parent_llama_name = parent_llama_name.rsplit(".", 1)[0]
            layer_rank = depth_rank.get(parent_llama_name, total_llama_depth)
            is_shared = layer_rank < shared_depth_count
            replaced_llama += 1
        else:
            is_shared = False  # non-decoder layers always specific (no depth concept)
            replaced_other += 1

        # Locate parent to do the replacement
        if "." in full_name:
            parent_name, child_name = full_name.rsplit(".", 1)
        else:
            parent_name, child_name = "", full_name

        parent_module = model
        if parent_name:
            for part in parent_name.split("."):
                parent_module = getattr(parent_module, part)

        cl_lora_layer = CLLoRALinear(
            base_layer=module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            is_shared=is_shared,
            orthogonal_init=orthogonal_init,
            freeze_a=freeze_a,
            use_block_scale=use_block_scale,
        ).to(module.weight.device).to(module.weight.dtype)

        setattr(parent_module, child_name, cl_lora_layer)

    print(f"Replaced {replaced_llama} LlamaDecoderLayer Linear layers + "
          f"{replaced_other} other Linear layers with CLLoRALinear.")
    print(f"Total: {replaced_llama + replaced_other} layers\n")
    return model

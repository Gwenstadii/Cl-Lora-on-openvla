"""
CL-LoRA module for continual learning on OpenVLA — PI-aligned v35.

Architecture (PI-aligned):
  Backbone (LLaMA-7B):    CL-LoRA injected → Stage 1 all trainable → ALL frozen after.
  Action Expert (MLP):    CL-LoRA injected → shared A+B+specific A frozen after Stage 1,
                          specific B + block_scale per-task (bank).

PI mapping:
  PaliGemma ≈ LLaMA-7B (Stage 1 train → freeze all)
  Action Expert ≈ CLLoRAActionHead (shared/specific split + per-task bank)

Reference: PI0.5 CL-LoRA (openpi/models/lora.py, openpi/models/gemma.py)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLLoRALinear(nn.Module):
    """Continual-Learning LoRA linear layer.

    Shared layers: LoRA-A orthogonally initialized and frozen (protects old knowledge).
    Specific layers: LoRA-A and LoRA-B both trainable, with learnable block_scale gating.

    Forward:  result = Wx + scaling * block_scale_gate * B @ A @ x
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

        # Freeze base weight (principle 2: frozen LLM backbone)
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

        # Block-scale gating (specific layers only, principle 4)
        if not self.is_shared and use_block_scale:
            self.block_scale = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter('block_scale', None)

        self._orthogonal_init = orthogonal_init
        self._freeze_a = freeze_a and is_shared  # principle 3: only shared layers freeze A
        self._use_block_scale = use_block_scale
        self.reset_parameters()

    def reset_parameters(self):
        # shared A: orthogonal init; specific A: kaiming init
        if self.is_shared and self._orthogonal_init:
            nn.init.orthogonal_(self.lora_a)
        else:
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
        # PI approach: shared A is trainable in Stage 1, frozen AFTER Stage 1
        # (freeze happens in freeze_stage1_params, not here)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)

        lora_out = F.linear(self.dropout(x), self.lora_a)
        lora_out = F.linear(lora_out, self.lora_b)

        scale = self.scaling
        if self.block_scale is not None:
            effective_scale = 1.0 + 0.5 * torch.tanh(self.block_scale)
            scale = scale * effective_scale

        return result + lora_out * scale


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
    """Inject CL-LoRA into LlamaDecoderLayer attention + FFN linear layers ONLY.

    Matches PI's injection scope exactly:
      - attn → q_proj, k_proj, v_proj, o_proj
      - ffn  → gate_proj, up_proj, down_proj

    Visual backbone, lm_head, projector, and all other Linear layers are LEFT UNTOUCHED.
    (principle 1: frozen visual encoder)
    """
    # Step 1: discover LlamaDecoderLayer depth ordering
    llama_layers = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == "LlamaDecoderLayer":
            llama_layers.append((name, module))

    total_depth = len(llama_layers)
    if total_depth == 0:
        raise RuntimeError("No LlamaDecoderLayer found in model. Check model architecture.")
    shared_depth_count = max(1, int(total_depth * shared_split_ratio))

    print(f"\n--- Injecting CL-LoRA (PI scope: decoder attn + ffn only) ---")
    print(f"LlamaDecoderLayer depth: {total_depth}")
    print(f"Shared layers (frozen A):    0 to {shared_depth_count - 1}")
    print(f"Specific layers (learnable):  {shared_depth_count} to {total_depth - 1}")
    print(f"Vision backbone:              UNTOUCHED (frozen)")
    print(f"lm_head / projector:          UNTOUCHED (frozen)\n")

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
    replaced_count = 0

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
                replaced_count += 1

    print(f"Replaced {replaced_count} Linear layers with CLLoRALinear.")
    print(f"(Expected: {total_depth} layers × 7 modules = {total_depth * 7})\n")
    return model


# ==============================================================================
# PI-Aligned CL-LoRA Action Head (Action Expert)
# ==============================================================================

class CLLoRAActionHead(nn.Module):
    """PI-aligned Action Expert with CL-LoRA.

    Multi-layer MLP where each Linear layer is replaced with CLLoRALinear.
    Shared layers (first shared_depth): frozen after Stage 1.
    Specific layers (remaining): specific A frozen after Stage 1, specific B per-task (bank).

    PI mapping:
      Shared layers ≈ Action Expert shared LoRA (frozen)
      Specific layers ≈ Action Expert specific LoRA (A frozen, B banked)
    """

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dims: list = None,
        action_dim: int = 7,
        rank: int = 16,
        alpha: float = 16.0,
        shared_depth: int = 2,
        orthogonal_init: bool = True,
        use_block_scale: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [4096, 2048, 1024]
        dims = [input_dim] + list(hidden_dims) + [1]  # output 1 scalar per position
        self.num_layers = len(dims) - 1

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            is_shared = i < shared_depth
            linear = nn.Linear(dims[i], dims[i + 1])
            cl_lora = CLLoRALinear(
                base_layer=linear,
                rank=rank,
                alpha=alpha,
                is_shared=is_shared,
                orthogonal_init=orthogonal_init,
                freeze_a=True,
                use_block_scale=(not is_shared and use_block_scale),
            )
            self.layers.append(cl_lora)

        self.act = nn.SiLU()
        self.shared_depth = shared_depth
        self.input_dim = input_dim
        self.action_dim = action_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape  # N = NUM_ACTIONS_CHUNK * ACTION_DIM (e.g. 8*7=56)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = self.act(x)
        return x.reshape(B, -1, self.action_dim)  # [B, 56, 1] → [B, 8, 7]

    def predict_action(self, x: torch.Tensor) -> torch.Tensor:
        """Interface expected by OpenVLA modeling_prismatic.py"""
        return self.forward(x.to(self.layers[0].weight.dtype))


def inject_cl_lora_into_action_head(action_head, rank, alpha, shared_depth,
                                     orthogonal_init, use_block_scale, device):
    """Replace all Linear layers in action head with CLLoRALinear."""
    replaced = 0
    for i, layer in enumerate(list(action_head.layers)):
        if isinstance(layer, CLLoRALinear):
            continue  # already injected
    # CLLoRAActionHead already creates CLLoRALinear layers in __init__
    # This function is a no-op for CLLoRAActionHead (layers are already CL-LoRA)
    print(f"[CL-LoRA] Action Head ready: {action_head.num_layers} layers, "
          f"shared_depth={shared_depth}, rank={rank}")
    return action_head


def create_cl_lora_action_head(
    input_dim: int = 4096,
    action_dim: int = 7,
    rank: int = 16,
    shared_depth: int = 2,
    device=None,
    dtype=torch.bfloat16,
) -> CLLoRAActionHead:
    """Create a PI-aligned CLLoRAActionHead."""
    ah = CLLoRAActionHead(
        input_dim=input_dim,
        hidden_dims=[2048, 2048, 1024],
        action_dim=action_dim,
        rank=rank,
        alpha=float(rank),
        shared_depth=shared_depth,
        orthogonal_init=True,
        use_block_scale=True,
    )
    if device is not None:
        ah = ah.to(device)
    ah = ah.to(dtype)
    n_shared = shared_depth
    n_specific = ah.num_layers - shared_depth
    n_bank = n_specific * 3  # lora_a + lora_b + block_scale per specific layer
    print(f"[CL-LoRA Action Head] {ah.num_layers} layers ({n_shared} shared + {n_specific} specific), "
          f"{n_bank} banked params, rank={rank}")
    return ah


# ==============================================================================
# PI-Style Task Bank (Stage 1 freeze + per-task specific B / block_scale)
# ==============================================================================
# PI-aligned v35:
#   After Stage 1:
#     LLaMA        → ALL LoRA frozen (shared A+B + specific A+B) — like frozen PaliGemma
#     Action Head  → shared A+B + specific A frozen, specific B trainable
#   Stage 2+:
#     LLaMA        → ALL LoRA frozen (unchanged)
#     Action Head  → specific B + block_scale reinit, trained per task → bank
#   Bank: Action Head specific A + B + block_scale (action_head only, LLaMA unchanging)


def _iter_cl_lora(module) -> list:
    """Collect all (name, CLLoRALinear) pairs from a module tree."""
    result = []
    for name, sub in module.named_modules():
        if isinstance(sub, CLLoRALinear):
            result.append((name, sub))
    return result


def freeze_stage1_params(model, freeze_specific_a: bool = True,
                         action_head=None) -> None:
    """PI-aligned v35: freeze ALL LLaMA LoRA + Action Head shared+specific A.

    LLaMA: ALL LoRA frozen (shared A+B + specific A+B) — like frozen PaliGemma.
    Action Head: shared A+B + specific A frozen; specific B stays trainable.
    """
    frozen_llama = 0
    for name, module in model.named_modules():
        if isinstance(module, CLLoRALinear):
            module.lora_a.requires_grad = False
            module.lora_b.requires_grad = False
            frozen_llama += 2

    frozen_ah = 0
    if action_head is not None:
        ah = action_head.module if hasattr(action_head, 'module') else action_head
        for name, module in ah.named_modules():
            if isinstance(module, CLLoRALinear):
                if module.is_shared:
                    module.lora_a.requires_grad = False
                    module.lora_b.requires_grad = False
                    frozen_ah += 2
                else:
                    module.lora_a.requires_grad = False  # specific A frozen
                    # specific B stays trainable
                    frozen_ah += 1

    print(f"[TaskBank] Stage 1 freeze: LLaMA={frozen_llama} locked (all LoRA), "
          f"ActionHead={frozen_ah} (shared+specific A){' + specific A' if freeze_specific_a else ''}")


def reinit_bank_for_new_task(model, action_head=None) -> None:
    """PI-aligned v35: reset Action Head specific LoRA-B + block_scale to zero.

    LLaMA is fully frozen → no reinit needed.
    Action Head specific B starts fresh for new task.
    """
    count = 0
    if action_head is not None:
        ah = action_head.module if hasattr(action_head, 'module') else action_head
        for name, module in ah.named_modules():
            if isinstance(module, CLLoRALinear) and not module.is_shared:
                nn.init.zeros_(module.lora_b)
                if module.block_scale is not None:
                    nn.init.zeros_(module.block_scale)
                count += 1
    print(f"[TaskBank] Reinitialized {count} Action Head specific layers for new task")


def save_task_bank(model, action_head, bank_dir: str, stage: int) -> None:
    """PI-aligned v35: save Action Head specific A+B+block_scale to bank.

    LLaMA is fully frozen → not banked.
    Action Head specific params → per-task bank.
    """
    import os
    os.makedirs(str(bank_dir), exist_ok=True)
    bank = {}
    ah = action_head.module if hasattr(action_head, 'module') else action_head
    if ah is not None:
        for name, module in ah.named_modules():
            if isinstance(module, CLLoRALinear) and not module.is_shared:
                layer_key = name.replace('.', '_')
                bank[f"ah.{layer_key}.lora_a"] = module.lora_a.data.cpu().clone()
                bank[f"ah.{layer_key}.lora_b"] = module.lora_b.data.cpu().clone()
                if module.block_scale is not None:
                    bank[f"ah.{layer_key}.block_scale"] = module.block_scale.data.cpu().clone()
    path = os.path.join(str(bank_dir), f"task_{stage}_bank.pt")
    torch.save(bank, path)
    print(f"[TaskBank] Saved stage {stage} bank ({len(bank)} Action Head tensors) → {path}")


def load_task_bank(model, action_head, bank_path: str) -> None:
    """PI-aligned v35: restore Action Head specific A+B+block_scale from bank."""
    bank = torch.load(bank_path, map_location='cpu', weights_only=True)
    ah = action_head.module if hasattr(action_head, 'module') else action_head
    if ah is not None:
        for name, module in ah.named_modules():
            if isinstance(module, CLLoRALinear) and not module.is_shared:
                layer_key = name.replace('.', '_')
                for suffix in ['lora_a', 'lora_b', 'block_scale']:
                    key = f"ah.{layer_key}.{suffix}"
                    if key in bank:
                        target = getattr(module, suffix, None)
                        if target is not None:
                            target.data.copy_(bank[key].to(target.device))
    print(f"[TaskBank] Loaded Action Head bank ({len(bank)} tensors)")

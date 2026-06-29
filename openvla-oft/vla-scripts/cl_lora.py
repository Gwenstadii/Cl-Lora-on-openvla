"""
CL-LoRA module for continual learning on OpenVLA.

Injects CLLoRALinear ONLY into LlamaDecoderLayer's attention (q/k/v/o_proj)
and FFN (gate/up/down_proj) — matching PI's "attn" + "ffn" scope.

Design principles (from PI):
  1. FREEZE visual encoder (SigLIP) — NO LoRA, stable features across tasks
  2. FREEZE LLM backbone weights — only LoRA adapter params are trainable
  3. Shared layers: LoRA-A orthogonal init + frozen (anti-forgetting)
  4. Specific layers: LoRA-A/B trainable + block-scale gating
  5. Lightweight action head only

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
# PI-Style Task Bank (Stage 1 freeze + per-task specific B / block_scale)
# ==============================================================================
# After Stage 1: shared LoRA-B + specific LoRA-A are FROZEN (shared knowledge).
# Stage 2+: specific LoRA-B + block_scale + action_head are per-task (bank).
# During eval, old tasks load their bank to restore task-specific params.
# This is the PI adaptation of CL-LoRA for VLA models.


def freeze_stage1_params(model, freeze_specific_a: bool = True) -> None:
    """Paper-aligned CL-LoRA: shared B permanently frozen, shared A continues learning.

    Shared LoRA-B (down-proj): permanently frozen (anti-forgetting on input projection).
    Shared LoRA-A (up-proj): NOT frozen, continues to update across all tasks.
    Specific LoRA-A: frozen if freeze_specific_a=True (orthogonal subspace protection).
    """
    frozen = 0
    for name, module in model.named_modules():
        if isinstance(module, CLLoRALinear):
            if module.is_shared:
                module.lora_b.requires_grad = False
                frozen += 1
            elif freeze_specific_a:
                module.lora_a.requires_grad = False
                frozen += 1
    extra = " + specific A" if freeze_specific_a else ""
    print(f"[TaskBank] Stage 1 freeze: {frozen} params locked (shared B{extra})")


def reinit_bank_for_new_task(model) -> None:
    """Before training Stage 2+: reset specific LoRA-B and block_scale to zero.

    Specific B starts fresh so the new task learns its own output mapping.
    Block_scale starts at identity (effective_scale = 1.0 + 0.5*tanh(0) = 1.0).
    """
    for name, module in model.named_modules():
        if isinstance(module, CLLoRALinear) and not module.is_shared:
            nn.init.zeros_(module.lora_b)
            if module.block_scale is not None:
                nn.init.zeros_(module.block_scale)
    print("[TaskBank] Reinitialized specific LoRA-B + block_scale for new task")


def save_task_bank(model, action_head, bank_dir: str, stage: int) -> None:
    """Save per-task bank: specific LoRA-A + LoRA-B + block_scale + action_head."""
    import os
    os.makedirs(str(bank_dir), exist_ok=True)
    bank = {}
    for name, module in model.named_modules():
        if isinstance(module, CLLoRALinear) and not module.is_shared:
            layer_key = name.replace('.', '_')
            bank[f"{layer_key}.lora_a"] = module.lora_a.data.cpu().clone()
            bank[f"{layer_key}.lora_b"] = module.lora_b.data.cpu().clone()
            if module.block_scale is not None:
                bank[f"{layer_key}.block_scale"] = module.block_scale.data.cpu().clone()
    if action_head is not None:
        for k, v in action_head.state_dict().items():
            bank[f"action_head.{k}"] = v.cpu().clone()
    path = os.path.join(str(bank_dir), f"task_{stage}_bank.pt")
    torch.save(bank, path)
    print(f"[TaskBank] Saved stage {stage} bank ({len(bank)} tensors) → {path}")


def load_task_bank(model, action_head, bank_path: str) -> None:
    """Load per-task bank: restore specific LoRA-A + LoRA-B + block_scale + action_head."""
    bank = torch.load(bank_path, map_location='cpu', weights_only=True)
    for name, module in model.named_modules():
        if isinstance(module, CLLoRALinear) and not module.is_shared:
            layer_key = name.replace('.', '_')
            for suffix in ['lora_a', 'lora_b', 'block_scale']:
                key = f"{layer_key}.{suffix}"
                if key in bank:
                    target = getattr(module, suffix, None)
                    if target is not None:
                        target.data.copy_(bank[key].to(target.device))
    if action_head is not None:
        ah_state = {k.replace('action_head.', ''): v
                    for k, v in bank.items() if k.startswith('action_head.')}
        if ah_state:
            current_keys = set(action_head.state_dict().keys())
            if not set(ah_state.keys()).intersection(current_keys):
                ah_state = {k.replace('module.', ''): v for k, v in ah_state.items()}
            action_head.load_state_dict(ah_state, strict=False)
            print(f"[TaskBank] Loaded action_head from bank")

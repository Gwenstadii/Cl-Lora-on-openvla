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
        # principle 3: shared layers use orthogonal init for A
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
# Task-specific parameter isolation (CL-LoRA paper: per-task block_scale + shared B)
# ==============================================================================

def save_task_snapshot(model, action_head, save_path: str, stage: int) -> None:
    """Save ISOLATION parameters only: shared LoRA-B + block_scale + action_head.

    Specific LoRA-A/B are NOT saved — they are shared knowledge and come from
    the latest checkpoint. This ensures old-task retention comes purely from
    structural isolation (not from full model cloning).
    """
    import os

    snapshot = {}
    for name, module in model.named_modules():
        if isinstance(module, CLLoRALinear):
            layer_key = name.replace('.', '_')
            if module.is_shared:
                # Isolation: shared-layer LoRA-B (per-task projection on frozen A)
                snapshot[f"{layer_key}.lora_b"] = module.lora_b.data.cpu().clone()
            else:
                # Isolation: block_scale (per-task gating for specific layers)
                if module.block_scale is not None:
                    snapshot[f"{layer_key}.block_scale"] = module.block_scale.data.cpu().clone()
                # SPECIFIC A/B NOT saved — shared knowledge from latest checkpoint

    if action_head is not None:
        for name, param in action_head.state_dict().items():
            snapshot[f"action_head.{name}"] = param.cpu().clone()

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    task_path = f"{save_path}/task_{stage}_snapshot.pt" if os.path.isdir(save_path) else save_path
    torch.save(snapshot, task_path)
    print(f"[TaskSnapshot] Saved isolation params for stage {stage} "
          f"({len(snapshot)} tensors, shared B + block_scale + action_head only) → {task_path}")


def load_task_snapshot(model, action_head, snapshot_path: str) -> None:
    """Restore ISOLATION parameters for old-task evaluation.

    Only shared LoRA-B + block_scale are restored.
    Specific LoRA-A/B use latest checkpoint values (shared knowledge).
    """
    snapshot = torch.load(snapshot_path, map_location='cpu', weights_only=True)

    for name, module in model.named_modules():
        if isinstance(module, CLLoRALinear):
            layer_key = name.replace('.', '_')
            # Restore shared B
            key = f"{layer_key}.lora_b"
            if key in snapshot and module.is_shared:
                module.lora_b.data.copy_(snapshot[key].to(module.lora_b.device))
            # Restore block_scale
            key = f"{layer_key}.block_scale"
            if key in snapshot and not module.is_shared and module.block_scale is not None:
                module.block_scale.data.copy_(snapshot[key].to(module.block_scale.device))

    if action_head is not None:
        ah_state = {k.replace('action_head.', ''): v
                    for k, v in snapshot.items() if k.startswith('action_head.')}
        if ah_state:
            current_keys = set(action_head.state_dict().keys())
            if not set(ah_state.keys()).intersection(current_keys):
                ah_state = {k.replace('module.', ''): v for k, v in ah_state.items()}
            action_head.load_state_dict(ah_state, strict=False)
            print(f"[TaskSnapshot] Loaded action_head from snapshot")


def reinit_task_specific_for_new_stage(model, shared_depth: int) -> None:
    """Reinitialize task-specific parameters for a new training stage.

    Shared-layer LoRA-B → zero (fresh start for new task)
    Specific-layer LoRA-A → kept (transfer from previous)
    Specific-layer LoRA-B → kept (transfer from previous)
    block_scale (specific layers) → zero (fresh gate for new task)
    """
    for name, module in model.named_modules():
        if isinstance(module, CLLoRALinear):
            if module.is_shared:
                # Shared layers: reinit B to zero, A stays frozen (same across tasks)
                nn.init.zeros_(module.lora_b)
            else:
                # Specific layers: reinit block_scale to zero for new task
                if module.block_scale is not None:
                    nn.init.zeros_(module.block_scale)
                # A and B keep their values from previous stage (transfer learning)

    print(f"[TaskInit] Reinitialized shared LoRA-B + block_scale for new stage "
          f"(shared_depth={shared_depth})")

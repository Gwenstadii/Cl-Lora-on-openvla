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
import re
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
    first_lora_layer: int = 0,
):
    """Inject CL-LoRA into LlamaDecoderLayer attention + FFN linear layers ONLY.

    first_lora_layer (default 0): only inject LoRA starting from this layer index.
      Layers < first_lora_layer are left as bare frozen Linear (no LoRA at all).
      This enables a PI-style "action expert": first_lora_layer=28 → only L28-31 have LoRA.

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

    # Only layers >= first_lora_layer get LoRA
    injected_depth = total_depth - first_lora_layer
    if injected_depth <= 0:
        raise RuntimeError(f"first_lora_layer={first_lora_layer} >= total_depth={total_depth}. No layers to inject.")
    shared_depth_count = first_lora_layer + max(1, int(injected_depth * shared_split_ratio))

    print(f"\n--- Injecting CL-LoRA (PI scope: decoder attn + ffn only) ---")
    print(f"LlamaDecoderLayer depth: {total_depth}")
    print(f"Bare layers (no LoRA):      0 to {first_lora_layer - 1}" if first_lora_layer > 0 else "")
    print(f"Shared layers (frozen A):    {first_lora_layer} to {shared_depth_count - 1}")
    print(f"Specific layers (learnable): {shared_depth_count} to {total_depth - 1}")
    print(f"Vision backbone:             UNTOUCHED (frozen)")
    print(f"lm_head / projector:         UNTOUCHED (frozen)\n")

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
    replaced_count = 0
    skipped_count = 0

    for layer_idx, (layer_name, layer_module) in enumerate(llama_layers):
        if layer_idx < first_lora_layer:
            skipped_count += 7  # 7 modules per layer
            continue  # bare layer, no LoRA

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

    expected = injected_depth * 7
    print(f"Replaced {replaced_count} Linear layers with CLLoRALinear (expected {expected}).")
    if first_lora_layer > 0:
        print(f"Skipped {first_lora_layer * 7} layers in bare backbone (no LoRA).\n")
    else:
        print()
    return model


def inject_cl_lora_into_action_head(
    action_head: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    orthogonal_init: bool = True,
    freeze_a: bool = True,
    use_block_scale: bool = True,
) -> nn.Module:
    """Inject CL-LoRA into action_head's Linear layers (all as specific layers).

    Matches PI's action-expert LoRA injection. The action_head Linear layers
    get CLLoRALinear wrappers with is_shared=False → per-task bank.
    """
    replaced = 0
    for name, module in action_head.named_modules():
        if isinstance(module, nn.Linear):
            parent = action_head
            parts = name.rsplit('.', 1)
            child = parts[-1]
            if len(parts) > 1:
                for p in parts[0].split('.'):
                    parent = getattr(parent, p)
            cl = CLLoRALinear(
                base_layer=module,
                rank=rank, alpha=alpha, dropout=dropout,
                is_shared=False,  # all action_head layers are specific
                orthogonal_init=orthogonal_init,
                freeze_a=freeze_a,
                use_block_scale=use_block_scale,
            ).to(module.weight.device).to(module.weight.dtype)
            setattr(parent, child, cl)
            replaced += 1
    print(f"[CL-LoRA] Injected {replaced} Linear layers in action_head.")
    return action_head


# ==============================================================================
# PI-Style Task Bank (Stage 1 freeze + per-task specific B / block_scale)
# ==============================================================================
# After Stage 1: shared LoRA-B + specific LoRA-A are FROZEN (shared knowledge).
# Stage 2+: specific LoRA-B + block_scale + action_head are per-task (bank).
# During eval, old tasks load their bank to restore task-specific params.
# This is the PI adaptation of CL-LoRA for VLA models.


def _freeze_and_reinit_modules(modules, freeze_specific_a: bool, reinit: bool,
                               freeze_shared: bool = True) -> int:
    """Apply freeze + optional reinit to a collection of CLLoRALinear modules.

    freeze_shared=False 时**不冻结** shared 层 A/B（研究用: 制造"共享通路漂移"遗忘源）。
    注意 shared 参数不进 bank（`save_task_bank` 只存 specific），所以解冻后**没有任何恢复路径**，
    漂移会同时污染所有任务 —— 这是与 specific-A 漂移（只伤旧任务）本质不同的遗忘通道。
    """
    frozen = 0
    for module in modules:
        if not isinstance(module, CLLoRALinear):
            continue
        if module.is_shared:
            if freeze_shared:
                module.lora_a.requires_grad = False
                module.lora_b.requires_grad = False
                if module.block_scale is not None:
                    module.block_scale.requires_grad = False   # 修复: 共享层 block_scale 此前漏冻结, Stage2+ 持续漂移
                frozen += 3
        elif freeze_specific_a:
            module.lora_a.requires_grad = False
            frozen += 1
        if reinit and not module.is_shared:
            nn.init.zeros_(module.lora_b)
            if module.block_scale is not None:
                nn.init.zeros_(module.block_scale)
    return frozen


def shared_param_ids(model) -> set:
    """返回 model 里 shared CLLoRALinear 的 lora_a/lora_b 参数 id 集合（供分层 lr 使用）。"""
    ids = set()
    for m in model.modules():
        if isinstance(m, CLLoRALinear) and m.is_shared:
            ids.add(id(m.lora_a))
            ids.add(id(m.lora_b))
    return ids


def freeze_stage1_params(model, freeze_specific_a: bool = True, action_head=None,
                         freeze_shared: bool = True) -> None:
    """V11-aligned: shared A+B both permanently frozen after Stage 1.

    Shared LoRA-A + LoRA-B: permanently frozen (complete anti-forgetting) —— 除非 freeze_shared=False（实验开关）。
    Specific LoRA-A: frozen if freeze_specific_a=True (orthogonal subspace protection).
    """
    frozen = _freeze_and_reinit_modules(model.modules(), freeze_specific_a, reinit=False,
                                        freeze_shared=freeze_shared)
    if action_head is not None:
        frozen += _freeze_and_reinit_modules(action_head.modules(), freeze_specific_a, reinit=False,
                                             freeze_shared=freeze_shared)
    extra = " + specific A" if freeze_specific_a else ""
    shared_note = "" if freeze_shared else " ⚠️ shared A/B 保持可训练(实验: 共享通路漂移源)"
    print(f"[TaskBank] Stage 1 freeze: {frozen} params locked (shared A + shared B{extra}){shared_note}")


def reinit_bank_for_new_task(model, action_head=None, freeze_shared: bool = True) -> None:
    """Before training Stage 2+: reset specific LoRA-B and block_scale to zero.

    Specific B starts fresh so the new task learns its own output mapping.
    Block_scale starts at identity (effective_scale = 1.0 + 0.5*tanh(0) = 1.0).
    freeze_shared 需与 freeze_stage1_params 保持一致（否则会把刚解冻的 shared A/B 又冻回去）。
    """
    _freeze_and_reinit_modules(model.modules(), freeze_specific_a=False, reinit=True,
                               freeze_shared=freeze_shared)
    if action_head is not None:
        _freeze_and_reinit_modules(action_head.modules(), freeze_specific_a=False, reinit=True,
                                   freeze_shared=freeze_shared)
    print("[TaskBank] Reinitialized specific LoRA-B + block_scale for new task"
          + ("" if freeze_shared else " (shared A/B 保持可训练)"))


def _is_film_key(key: str) -> bool:
    """FiLM 调制参数判定: scale/shift（与 _in_film_scope 的默认口径一致）。"""
    return "scale" in key or "shift" in key


def _bank_size_mb(bank: dict) -> float:
    total = 0
    for v in bank.values():
        if torch.is_tensor(v):
            total += v.numel() * v.element_size()
        elif isinstance(v, dict):
            total += sum(t.numel() * t.element_size() for t in v.values() if torch.is_tensor(t))
    return total / 1e6


def apply_specific_a_layer_mask(model, action_head=None, trainable_layers=None,
                               freeze_action_head_a: bool = True,
                               action_head_keep: str = ""):
    """按层控制 specific-A 是否可训练 —— "保护量"旋钮（比 lr 缩放更可能给出连续曲线）。

    机制（为什么这个旋钮是"分级"的，而 lr 缩放不是）:
      评估任务 K 时 = W + Σ_{frozen 层} s·g_K·B_K·A_1  +  Σ_{unfrozen 层} s·g_K·B_K·A_final
      ⇒ frozen 层的贡献**精确复原**，unfrozen 层**局部损坏** ⇒ 损伤按"坏掉几层"分级；
      而 lr 缩放会让**所有层同时轻微漂移**，每层都配不上 ⇒ 一步跨过阈值（实测 A=0/56）。
      自适应优化器下位移 ≈ lr·√N，40k 步后已到 A 自身尺度 ⇒ 任何非零 lr 都会漂穿。

    trainable_layers: 允许保持可训练的 LLM 层号集合（如 {28,29,30,31}）；None = 全部可训练；set() = 全部冻结。
    freeze_action_head_a / action_head_keep:
      动作头 4 个注入 Linear 没有层号，单独控制。
      action_head_keep: ""=按 freeze_action_head_a；"all"=全解冻；"none"=全冻结；
                        或子串列表（如 "fc2" / "fc1,mlp_resnet_blocks"）匹配模块名。
    返回 (frozen_count, trainable_count)。
    """
    frozen = trainable = 0
    for name, module in model.named_modules():
        if not isinstance(module, CLLoRALinear) or module.is_shared:
            continue
        m = re.search(r"layers\.(\d+)\.", name)
        if m is None:
            continue
        idx = int(m.group(1))
        keep = trainable_layers is None or idx in trainable_layers
        module.lora_a.requires_grad = keep
        if keep:
            trainable += 1
        else:
            frozen += 1
    if action_head is not None:
        keep_spec = (action_head_keep or "").strip().lower()
        subs = [s.strip() for s in keep_spec.split(",") if s.strip()] if keep_spec else []
        for name, module in action_head.named_modules():
            if not isinstance(module, CLLoRALinear):
                continue
            if keep_spec == "all":
                keep = True
            elif keep_spec in ("none", ""):
                keep = not freeze_action_head_a
            else:
                keep = any(s in name for s in subs)
            module.lora_a.requires_grad = keep
            if keep:
                trainable += 1
            else:
                frozen += 1
    return frozen, trainable


def parse_layer_spec(spec: str, lo: int = 24, hi: int = 31):
    """解析层号字符串: "28-31" / "24,25,26" / "none" / "" → set(层号)。
    · ""     → None（= 全部可训练，原行为）
    · "none" → set()（= 全部冻结；用于"只让动作头 A 漂移"这类对照）
    支持组合: "24-27,30"。"""
    spec = (spec or "").strip()
    if not spec:
        return None
    if spec.lower() in ("none", "no", "-", "0"):
        return set()
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return {i for i in out if lo <= i <= hi}


def specific_a_param_ids(model, action_head=None) -> set:
    """返回 specific 层（含 action_head 全部注入层）的 lora_a 参数 id 集合。
    用于 `--specific_a_lr_scale`：给"部分解冻/温和漂移"的 A 单独设 lr。"""
    ids = set()
    for m in model.modules():
        if isinstance(m, CLLoRALinear) and not m.is_shared:
            ids.add(id(m.lora_a))
    if action_head is not None:
        for m in action_head.modules():
            if isinstance(m, CLLoRALinear):
                ids.add(id(m.lora_a))
    return ids


def save_task_bank(model, action_head, bank_dir: str, stage: int, film_mode: str = "film",
                   save_specific_a: bool = False) -> None:
    """Save per-task bank: specific LoRA-B + block_scale + action_head (+ FiLM, + 可选 specific-A).

    film_mode 控制 vision_backbone 存多少（默认 "film"）:
      "none" = 不存 FiLM         → bank 最小, 评估端 γ 无效（等价 γ=0）
      "film" = 只存 scale/shift  → ~0.2MB, γ/scope 能力与 "full" 完全等价（loader 本来就只挑这些 key）
      "full" = 整份 vision_backbone（旧行为）→ ~2.4GB, 其中 99.99% 是冻结的 ViT 权重, 从未被使用

    save_specific_a=True: 额外把 specific 层（含 action_head）的 **lora_a** 一起存入 bank。
      用途: `freeze_specific_a=False`（A 自由漂移）时，评估任务 K 可同时恢复 A_K + B_K
      ⇒ 配对精确复原 ⇒ 保留率≈自评（"每任务 A 快照"范式, 与"冻结 A"并列的另一种防遗忘机制）。
      代价: 每个任务多约 5.6M 参数（bf16 ≈ 11MB），与已有的 B 载荷同量级。
    """
    import os
    os.makedirs(str(bank_dir), exist_ok=True)
    bank = {}
    for name, module in model.named_modules():
        if isinstance(module, CLLoRALinear) and not module.is_shared:
            layer_key = name.replace('.', '_')
            bank[f"{layer_key}.lora_b"] = module.lora_b.data.cpu().clone()
            if save_specific_a:
                bank[f"{layer_key}.lora_a"] = module.lora_a.data.cpu().clone()
            if module.block_scale is not None:
                bank[f"{layer_key}.block_scale"] = module.block_scale.data.cpu().clone()
    if action_head is not None:
        for name, module in action_head.named_modules():
            if isinstance(module, CLLoRALinear):
                ah_key = name.replace('.', '_')
                bank[f"action_head.{ah_key}.lora_b"] = module.lora_b.data.cpu().clone()
                if save_specific_a:
                    bank[f"action_head.{ah_key}.lora_a"] = module.lora_a.data.cpu().clone()
                if module.block_scale is not None:
                    bank[f"action_head.{ah_key}.block_scale"] = module.block_scale.data.cpu().clone()

    # RoboTwin 特异优化: 每任务 FiLM 一并存入 bank (评估端按 film_gamma/scope 恢复)。
    # 历史: 该诊断诞生于 v39b2 时期("评估用漂移后的 FiLM → A/C 归零"), 后来证明
    #   A/C 归零主因是 proprio bug + specific-A 配对错位, FiLM 漂移强度几乎不影响
    #   retention(b3≈b6) ⇒ 主表口径 γ=0 时这份 FiLM 根本不会被读取。
    # 存储量由 film_mode 控制, 默认 "film"（只存 scale/shift, 体积降 ~10000×）。
    vb = getattr(model, "vision_backbone", None)
    if vb is not None and film_mode != "none":
        vb_sd = vb.state_dict()
        if film_mode == "film":
            vb_sd = {k: v for k, v in vb_sd.items() if _is_film_key(k)}
        bank["vision_backbone"] = {k: v.cpu().clone() for k, v in vb_sd.items()}

    path = os.path.join(str(bank_dir), f"task_{stage}_bank.pt")
    torch.save(bank, path)
    print(f"[TaskBank] Saved stage {stage} bank ({len(bank)} tensors, film_mode={film_mode}, "
          f"~{_bank_size_mb(bank):.1f} MB) → {path}")


def _in_film_scope(key: str, scope: str) -> bool:
    """判断 FiLM state_dict key 是否属于指定恢复范围."""
    if not scope or scope == "all":
        return True
    # 只处理 FiLM 参数 (scale/shift), 冻结的 ViT 权重不属于调制层
    if "scale" not in key and "shift" not in key:
        return False
    is_fused = "fused_featurizer" in key
    is_siglip = ("featurizer" in key) and not is_fused
    if scope == "siglip":
        return is_siglip
    if scope == "dinov2":
        return is_fused
    if scope.startswith("k"):
        import re
        m = re.search(r"blocks\.(\d+)\.", key)
        if m and int(m.group(1)) < int(scope[1:]):
            return True
        return False
    return True


def load_task_bank(model, action_head, bank_path: str, film_gamma: float = 1.0, film_scope: str = "all") -> None:
    """Load per-task bank: restore specific LoRA-B + block_scale + action_head (+ FiLM).

    film_gamma: FiLM 恢复程度 (评估端标定旋钮):
      1.0 = 完全恢复该任务 FiLM (高保留, 各任务回到"刚训完"水平)
      0.0 = 不恢复 (用当前 checkpoint 的 FiLM, 即原漂移行为)
      0~1 之间 = 任务 FiLM 与当前 FiLM 线性插值 (残留量连续可调)

    film_scope: FiLM 部分恢复范围 (只恢复选中的层, 其余保持当前 checkpoint 的):
      "all"    = 全部 FiLM (默认)
      "siglip" = 只恢复 SigLIP 主干的 FiLM (featurizer.*)
      "dinov2" = 只恢复 DINOv2 主干的 FiLM (fused_featurizer.*)
      "k<N>"   = 只恢复每个主干前 N 个 block 的 FiLM (如 k10)
    """
    bank = torch.load(bank_path, map_location='cpu', weights_only=True)
    loaded_a = 0
    for name, module in model.named_modules():
        if isinstance(module, CLLoRALinear) and not module.is_shared:
            layer_key = name.replace('.', '_')
            for suffix in ['lora_a', 'lora_b', 'block_scale']:
                key = f"{layer_key}.{suffix}"
                if key in bank:
                    target = getattr(module, suffix, None)
                    if target is not None:
                        target.data.copy_(bank[key].to(target.device))
                        if suffix == 'lora_a':
                            loaded_a += 1
    if action_head is not None:
        for name, module in action_head.named_modules():
            if isinstance(module, CLLoRALinear):
                ah_key = name.replace('.', '_')
                for suffix in ['lora_a', 'lora_b', 'block_scale']:
                    key = f"action_head.{ah_key}.{suffix}"
                    if key in bank:
                        target = getattr(module, suffix, None)
                        if target is not None:
                            target.data.copy_(bank[key].to(target.device))
                            if suffix == 'lora_a':
                                loaded_a += 1
        print(f"[TaskBank] Loaded action_head LoRA from bank")
    if loaded_a:
        print(f"[TaskBank] 恢复 specific-A 快照 {loaded_a} 个张量（bank 内含 A ⇒ 配对精确复原）")

    # FiLM 恢复 (bank 含 vision_backbone 时)
    if "vision_backbone" not in bank or getattr(model, "vision_backbone", None) is None:
        if film_gamma >= 1.0:
            print("[TaskBank] ⚠️ bank 不含 FiLM(vision_backbone) —— 跳过 FiLM 恢复, "
                  "评估使用当前 checkpoint 的 FiLM (等价 γ=0; film_mode=none 时的预期行为)")
        return
    vb = model.vision_backbone
    task_film = bank["vision_backbone"]
    cur = vb.state_dict()
    # 按 scope 过滤要恢复的层: all / siglip / dinov2 / k<N>(前N个block)
    selected = {k for k in task_film if k in cur and _in_film_scope(k, film_scope)}
    if film_gamma >= 1.0:
        mixed = {k: task_film[k] for k in selected}
        vb.load_state_dict(mixed, strict=False)
        print(f"[TaskBank] FiLM 恢复 scope={film_scope} ({len(mixed)}/{len(task_film)} tensors)")
    elif film_gamma > 0.0:
        mixed = {}
        for k in selected:
            mixed[k] = (film_gamma * task_film[k].to(cur[k].device).float()
                        + (1.0 - film_gamma) * cur[k].float()).to(cur[k].dtype)
        vb.load_state_dict(mixed, strict=False)
        print(f"[TaskBank] FiLM 插值恢复 γ={film_gamma} scope={film_scope} ({len(mixed)} tensors)")
    else:
        print(f"[TaskBank] film_gamma=0, 不恢复 FiLM (保持当前 checkpoint 的 FiLM)")

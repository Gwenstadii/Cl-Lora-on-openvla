# vla-scripts/cl_lora.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLLoRALinear(nn.Module):
    def __init__(
        self, 
        base_layer: nn.Linear, 
        rank: int = 32, 
        alpha: float = 32.0, 
        dropout: float = 0.0,
        is_shared: bool = True
    ):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.is_shared = is_shared
        
        # 1. 继承并冻结原模型的 Base 权重
        self.weight = base_layer.weight
        self.weight.requires_grad = False
        if base_layer.bias is not None:
            self.bias = base_layer.bias
            self.bias.requires_grad = False
        else:
            self.register_parameter('bias', None)
            
        # 2. 定义 LoRA A 和 B 矩阵
        self.lora_a = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # 3. 块权重 (仅限 Specific 深层)
        if not self.is_shared:
            self.block_scale = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter('block_scale', None)

        self.reset_parameters()

    def reset_parameters(self):
        """严格对照 PI0.5 的 CL-LoRA 初始化逻辑"""
        # Shared 浅层：A 矩阵正交初始化并严格冻结，B 矩阵全零初始化
        if self.is_shared:
            nn.init.orthogonal_(self.lora_a)
            nn.init.zeros_(self.lora_b)
            self.lora_a.requires_grad = False  # 核心抗遗忘约束
        # Specific 深层：A 矩阵正交初始化（或Kaiming），B 矩阵全零初始化，全部可训练
        else:
            nn.init.orthogonal_(self.lora_a)
            nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 主干网络的冻结前向计算
        result = F.linear(x, self.weight, self.bias)
        
        # LoRA 分支的前向计算
        lora_out = F.linear(self.dropout(x), self.lora_a)
        lora_out = F.linear(lora_out, self.lora_b)
        
        # 影响力缩放计算
        scale = self.scaling
        if self.block_scale is not None:
            # 采用 PI0.5 风格的柔性缩放：1.0 + 0.5 * tanh(w)
            effective_scale = 1.0 + 0.5 * torch.tanh(self.block_scale)
            scale = scale * effective_scale
            
        return result + lora_out * scale

def inject_cl_lora_into_model(model, rank=16, alpha=16.0, dropout=0.0, shared_split_ratio=0.5):
    """
    遍历并替换 LlamaDecoderLayer 中的注意力机制和前馈网络线性层。
    """
    llama_layers = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == "LlamaDecoderLayer":
            llama_layers.append((name, module))
            
    total_depth = len(llama_layers)
    shared_depth = int(total_depth * shared_split_ratio)
    
    print(f"\n--- Injecting CL-LoRA ---")
    print(f"Total Depth: {total_depth}")
    print(f"Shared Layers (Frozen A): 0 to {shared_depth-1}")
    print(f"Specific Layers (Learnable Block Scale): {shared_depth} to {total_depth-1}\n")
    
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    replaced_count = 0
    
    for layer_idx, (layer_name, layer_module) in enumerate(llama_layers):
        is_shared = layer_idx < shared_depth
        
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
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    is_shared=is_shared
                ).to(module.weight.device).to(module.weight.dtype)
                
                setattr(parent_module, child_name, cl_lora_layer)
                replaced_count += 1
                
    print(f"Successfully replaced {replaced_count} Linear layers with CLLoRALinear.\n")
    return model
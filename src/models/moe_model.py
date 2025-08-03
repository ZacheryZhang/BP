"""
混合专家模型 (Mixture of Experts)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from .base_model import BaseModel

class Expert(nn.Module):
    """专家网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class Gate(nn.Module):
    """门控网络"""
    
    def __init__(self, input_dim: int, num_experts: int, dropout: float = 0.1):
        super().__init__()
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_experts),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)

class MoELayer(nn.Module):
    """混合专家层"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_experts: int, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 专家网络
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim, dropout)
            for _ in range(num_experts)
        ])
        
        # 门控网络
        self.gate = Gate(input_dim, num_experts, dropout)
        
        # 噪声用于负载均衡
        self.noise_linear = nn.Linear(input_dim, num_experts)
    
    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        # 计算门控权重
        gate_weights = self.gate(x)  # [batch_size, num_experts]
        
        # 添加噪声（仅在训练时）
        if training:
            noise = torch.randn_like(gate_weights) * 0.1
            gate_weights = gate_weights + noise
            gate_weights = F.softmax(gate_weights, dim=-1)
        
        # 选择top-k专家
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)
        
        # 计算专家输出
        expert_outputs = []
        for i in range(self.num_experts):
            expert_output = self.experts[i](x)
            expert_outputs.append(expert_output)
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, output_dim]
        
        # 加权组合top-k专家的输出
        final_output = torch.zeros(batch_size, expert_outputs.size(-1), device=x.device)
        
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]  # [batch_size]
            expert_weight = top_k_weights[:, i].unsqueeze(-1)  # [batch_size, 1]
            
            # 选择对应专家的输出
            selected_outputs = expert_outputs[torch.arange(batch_size), expert_idx]
            final_output += expert_weight * selected_outputs
        
        # 计算负载均衡损失
        load_balance_loss = self._compute_load_balance_loss(gate_weights)
        
        return final_output, load_balance_loss
    
    def _compute_load_balance_loss(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """计算负载均衡损失"""
        # 计算每个专家的平均使用率
        expert_usage = torch.mean(gate_weights, dim=0)  # [num_experts]
        
        # 理想情况下每个专家的使用率应该相等
        target_usage = 1.0 / self.num_experts
        
        # 计算均方误差
        load_balance_loss = torch.mean((expert_usage - target_usage) ** 2)
        
        return load_balance_loss

class MoEModel(BaseModel):
    """混合专家血压预测模型"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.input_dim = config.get('input_dim', 100)
        self.hidden_dims = config.get('hidden_dims', [128, 64])
        self.output_dim = config.get('output_dim', 2)
        self.num_experts = config.get('num_experts', 4)
        self.top_k = config.get('top_k', 2)
        self.dropout = config.get('dropout', 0.1)
        self.load_balance_weight = config.get('load_balance_weight', 0.01)
        
        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # MoE层
        self.moe_layers = nn.ModuleList()
        prev_dim = self.hidden_dims[0]
        
        for hidden_dim in self.hidden_dims[1:]:
            moe_layer = MoELayer(
                input_dim=prev_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_experts=self.num_experts,
                top_k=self.top_k,
                dropout=self.dropout
            )
            self.moe_layers.append(moe_layer)
            prev_dim = hidden_dim
        
        # 输出层
        self.output_layer = nn.Linear(prev_dim, self.output_dim)
        
        # 存储负载均衡损失
        self.load_balance_losses = []
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 清空之前的负载均衡损失
        self.load_balance_losses = []
        
        # 输入层
        x = self.input_layer(x)
        
        # MoE层
        for moe_layer in self.moe_layers:
            x, load_balance_loss = moe_layer(x, self.training)
            self.load_balance_losses.append(load_balance_loss)
        
        # 输出层
        output = self.output_layer(x)
        
        return output
    
    def get_total_loss(self, prediction_loss: torch.Tensor) -> torch.Tensor:
        """获取包含负载均衡损失的总损失"""
        total_load_balance_loss = sum(self.load_balance_losses)
        total_loss = prediction_loss + self.load_balance_weight * total_load_balance_loss
        return total_loss
    
    def get_expert_usage_stats(self, dataloader) -> Dict[str, torch.Tensor]:
        """获取专家使用统计"""
        self.eval()
        expert_usage_stats = {f'layer_{i}': torch.zeros(self.num_experts) 
                            for i in range(len(self.moe_layers))}
        
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(next(self.parameters()).device)
                batch_size = batch_x.size(0)
                total_samples += batch_size
                
                # 前向传播并收集门控权重
                x = self.input_layer(batch_x)
                
                for i, moe_layer in enumerate(self.moe_layers):
                    gate_weights = moe_layer.gate(x)
                    expert_usage_stats[f'layer_{i}'] += torch.sum(gate_weights, dim=0).cpu()
                    x, _ = moe_layer(x, training=False)
        
        # 归一化统计结果
        for layer_name in expert_usage_stats:
            expert_usage_stats[layer_name] /= total_samples
        
        return expert_usage_stats

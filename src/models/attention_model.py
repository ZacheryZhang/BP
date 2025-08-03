"""
注意力机制模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any
from .base_model import BaseModel

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # 线性变换并重塑为多头
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attention = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 重塑并通过输出层
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(attention)
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, 
                                   V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, V)

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # 多头注意力
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class AttentionModel(BaseModel):
    """基于注意力机制的血压预测模型"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.input_dim = config.get('input_dim', 100)
        self.d_model = config.get('d_model', 128)
        self.n_heads = config.get('n_heads', 8)
        self.n_layers = config.get('n_layers', 4)
        self.d_ff = config.get('d_ff', 512)
        self.dropout = config.get('dropout', 0.1)
        self.output_dim = config.get('output_dim', 2)
        self.max_seq_len = config.get('max_seq_len', 100)
        
        # 输入投影
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(self.d_model, self.max_seq_len)
        
        # Transformer层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 输出层
        self.output_layers = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.output_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 如果输入是1D，需要添加序列维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # 输入投影
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # 位置编码
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Transformer层
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # 全局平均池化
        x = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        x = self.global_pool(x)  # [batch_size, d_model, 1]
        x = x.squeeze(-1)  # [batch_size, d_model]
        
        # 输出层
        output = self.output_layers(x)
        
        return output

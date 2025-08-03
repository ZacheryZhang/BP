"""
基础模型类
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

class BaseModel(nn.Module, ABC):
    """基础模型抽象类"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model_name = self.__class__.__name__
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'config': self.config
        }
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_info': self.get_model_info()
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location='cpu')
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

class MLPModel(BaseModel):
    """多层感知机模型"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.input_dim = config.get('input_dim', 100)
        self.hidden_dims = config.get('hidden_dims', [128, 64, 32])
        self.output_dim = config.get('output_dim', 2)
        self.dropout_rate = config.get('dropout_rate', 0.2)
        self.activation = config.get('activation', 'relu')
        
        # 构建网络层
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation())
            layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _get_activation(self):
        """获取激活函数"""
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'sigmoid':
            return nn.Sigmoid()
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            return nn.ReLU()
    
    def _initialize_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(x)

"""
模型训练器
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model: nn.Module, config: Dict):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            config: 训练配置
        """
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 训练参数
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 100)
        self.patience = config.get('early_stopping_patience', 10)
        
        # 优化器和损失函数
        self.optimizer = self._setup_optimizer()
        self.criterion = self._setup_criterion()
        
        # 训练历史
        self.train_history = {
            'loss': [],
            'val_loss': [],
            'metrics': []
        }
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def _setup_optimizer(self) -> optim.Optimizer:
        """设置优化器"""
        optimizer_name = self.config.get('optimizer', 'adam')
        
        if optimizer_name.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_name.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        else:
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def _setup_criterion(self) -> nn.Module:
        """设置损失函数"""
        loss_name = self.config.get('loss_function', 'mse')
        
        if loss_name.lower() == 'mse':
            return nn.MSELoss()
        elif loss_name.lower() == 'mae':
            return nn.L1Loss()
        elif loss_name.lower() == 'huber':
            return nn.SmoothL1Loss()
        else:
            return nn.MSELoss()
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict:
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            
        Returns:
            训练历史字典
        """
        # 准备数据
        train_loader = self._prepare_dataloader(X_train, y_train, shuffle=True)
        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = self._prepare_dataloader(X_val, y_val, shuffle=False)
        
        print(f"开始训练模型: {self.model.model_name}")
        print(f"设备: {self.device}")
        print(f"训练样本数: {len(X_train)}")
        if val_loader:
            print(f"验证样本数: {len(X_val)}")
        print("-" * 50)
        
        for epoch in range(self.epochs):
            # 训练阶段
            train_loss = self._train_epoch(train_loader)
            self.train_history['loss'].append(train_loss)
            
            # 验证阶段
            if val_loader:
                val_loss, val_metrics = self._validate_epoch(val_loader)
                self.train_history['val_loss'].append(val_loss)
                self.train_history['metrics'].append(val_metrics)
                
                # 早停检查
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    # 保存最佳模型
                    self._save_best_model()
                else:
                    self.patience_counter += 1
                
                # 打印进度
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}] - "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, "
                          f"Val MAE: {val_metrics['mae']:.4f}")
                
                # 早停
                if self.patience_counter >= self.patience:
                    print(f"早停在第 {epoch+1} 轮，验证损失未改善")
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}] - Train Loss: {train_loss:.4f}")
        
        print("训练完成!")
        return self.train_history
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        # 计算指标
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        metrics = self._calculate_metrics(predictions, targets)
        
        return total_loss / len(val_loader), metrics
    
    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """计算评估指标"""
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)
        
        # 计算MAPE
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
    
    def _prepare_dataloader(self, X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> DataLoader:
        """准备数据加载器"""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
    
    def _save_best_model(self):
        """保存最佳模型"""
        self.best_model_state = self.model.state_dict().copy()
    
    def load_best_model(self):
        """加载最佳模型"""
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.train_history['loss'], label='训练损失', color='blue')
        if self.train_history['val_loss']:
            axes[0, 0].plot(self.train_history['val_loss'], label='验证损失', color='red')
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        if self.train_history['metrics']:
            # MAE曲线
            mae_values = [m['mae'] for m in self.train_history['metrics']]
            axes[0, 1].plot(mae_values, label='验证MAE', color='green')
            axes[0, 1].set_title('MAE曲线')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # R²曲线
            r2_values = [m['r2'] for m in self.train_history['metrics']]
            axes[1, 0].plot(r2_values, label='验证R²', color='purple')
            axes[1, 0].set_title('R²曲线')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('R²')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # MAPE曲线
            mape_values = [m['mape'] for m in self.train_history['metrics']]
            axes[1, 1].plot(mape_values, label='验证MAPE', color='orange')
            axes[1, 1].set_title('MAPE曲线')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('MAPE (%)')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

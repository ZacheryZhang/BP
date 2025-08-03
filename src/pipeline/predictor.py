"""
完整的血压预测流水线
"""
import numpy as np
import torch
import joblib
import yaml
from pathlib import Path
from typing import Dict, Tuple, Union

from ..data.preprocessing import PPGPreprocessor
from ..data.feature_extraction import FeatureExtractor
from ..models.attention_model import AttentionModel
from ..models.moe_model import MoEModel
from ..utils.metrics import calculate_metrics

class BPPredictor:
    """血压预测器主类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化预测器
        
        Args:
            config_path: 配置文件路径
        """
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
            
        self.preprocessor = PPGPreprocessor(self.config['data'])
        self.feature_extractor = FeatureExtractor(self.config['features'])
        
        # 模型组件
        self.attention_model = None
        self.moe_model = None
        self.traditional_models = {}
        
        # 标准化器
        self.scaler = None
        self.is_trained = False
        
    def _default_config(self):
        """默认配置"""
        return {
            'data': {'sampling_rate': 125, 'signal_length': 1000},
            'features': {'paa_size': 50},
            'models': {
                'attention': {'input_dim': 1000, 'hidden_dim': 128},
                'moe': {'num_experts': 4, 'hidden_dim': 64}
            }
        }
    
    def preprocess_signal(self, ppg_signal: np.ndarray) -> np.ndarray:
        """
        预处理PPG信号
        
        Args:
            ppg_signal: 原始PPG信号
            
        Returns:
            预处理后的信号
        """
        return self.preprocessor.process_single_signal(ppg_signal)
    
    def extract_features(self, ppg_signal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        提取特征
        
        Args:
            ppg_signal: 预处理后的PPG信号
            
        Returns:
            特征字典
        """
        return self.feature_extractor.extract_all_features(ppg_signal)
    
    def predict(self, ppg_signal: np.ndarray) -> Dict[str, float]:
        """
        预测血压
        
        Args:
            ppg_signal: PPG信号 (shape: [signal_length,])
            
        Returns:
            预测结果字典 {'systolic': float, 'diastolic': float}
        """
        if not self.is_trained:
            raise ValueError("模型未训练，请先加载训练好的模型")
        
        # 1. 预处理
        processed_signal = self.preprocess_signal(ppg_signal)
        
        # 2. 特征提取
        features = self.extract_features(processed_signal)
        
        # 3. 注意力编码
        if self.attention_model:
            attention_features = self._apply_attention(processed_signal)
            features.update(attention_features)
        
        # 4. 特征标准化
        feature_vector = self._combine_features(features)
        if self.scaler:
            feature_vector = self.scaler.transform(feature_vector.reshape(1, -1))
        
        # 5. 模型预测
        predictions = {}
        
        # MoE模型预测
        if self.moe_model:
            moe_pred = self._predict_with_moe(feature_vector)
            predictions['moe'] = moe_pred
        
        # 传统模型预测
        for model_name, model in self.traditional_models.items():
            pred = model.predict(feature_vector)[0]
            predictions[model_name] = pred
        
        # 集成预测结果
        final_prediction = self._ensemble_predictions(predictions)
        
        return {
            'systolic': final_prediction[0],
            'diastolic': final_prediction[1],
            'individual_predictions': predictions
        }
    
    def _apply_attention(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """应用注意力机制"""
        signal_tensor = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            attention_weights, encoded_features = self.attention_model(signal_tensor)
        
        return {
            'attention_weights': attention_weights.numpy(),
            'encoded_features': encoded_features.numpy()
        }
    
    def _combine_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """组合所有特征"""
        feature_list = []
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                if value.ndim > 1:
                    feature_list.extend(value.flatten())
                else:
                    feature_list.extend(value)
            else:
                feature_list.append(value)
        
        return np.array(feature_list)
    
    def _predict_with_moe(self, features: np.ndarray) -> np.ndarray:
        """使用MoE模型预测"""
        features_tensor = torch.FloatTensor(features)
        
        with torch.no_grad():
            prediction = self.moe_model(features_tensor)
        
        return prediction.numpy()
    
    def _ensemble_predictions(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """集成多个模型的预测结果"""
        if not predictions:
            raise ValueError("没有可用的预测结果")
        
        # 简单平均集成
        pred_values = list(predictions.values())
        ensemble_pred = np.mean(pred_values, axis=0)
        
        return ensemble_pred
    
    @classmethod
    def load_model(cls, model_path: str) -> 'BPPredictor':
        """
        加载训练好的模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            加载的预测器实例
        """
        model_data = joblib.load(model_path)
        
        predictor = cls(model_data.get('config_path'))
        predictor.attention_model = model_data.get('attention_model')
        predictor.moe_model = model_data.get('moe_model')
        predictor.traditional_models = model_data.get('traditional_models', {})
        predictor.scaler = model_data.get('scaler')
        predictor.is_trained = True
        
        return predictor
    
    def save_model(self, model_path: str):
        """
        保存模型
        
        Args:
            model_path: 保存路径
        """
        model_data = {
            'attention_model': self.attention_model,
            'moe_model': self.moe_model,
            'traditional_models': self.traditional_models,
            'scaler': self.scaler,
            'config': self.config
        }
        
        joblib.dump(model_data, model_path)
        print(f"模型已保存到: {model_path}")

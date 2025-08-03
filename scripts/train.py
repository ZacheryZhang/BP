#!/usr/bin/env python3
"""
训练脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from src.data.preprocessing import PPGPreprocessor
from src.data.feature_extraction import FeatureExtractor
from src.models.base_model import MLPModel
from src.models.attention_model import AttentionModel
from src.models.moe_model import MoEModel
from src.training.trainer import ModelTrainer
from src.utils.metrics import evaluate_model

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def generate_synthetic_data(n_samples: int = 1000, signal_length: int = 1000) -> tuple:
    """
    生成合成数据用于演示
    
    Args:
        n_samples: 样本数量
        signal_length: 信号长度
        
    Returns:
        (signals, blood_pressures) 元组
    """
    print(f"生成 {n_samples} 个合成PPG信号样本...")
    
    signals = []
    blood_pressures = []
    
    for i in range(n_samples):
        # 随机参数
        heart_rate = np.random.normal(75, 15)  # 心率
        heart_rate = np.clip(heart_rate, 50, 120)
        
        # 生成PPG信号
        t = np.linspace(0, signal_length/125, signal_length)
        f_heart = heart_rate / 60
        
        # 主要PPG波形
        ppg_main = np.sin(2 * np.pi * f_heart * t)
        ppg_harmonic = 0.3 * np.sin(2 * np.pi * 2 * f_heart * t + np.pi/4)
        
        # 呼吸影响
        f_resp = np.random.normal(0.25, 0.05)
        ppg_resp = 0.2 * np.sin(2 * np.pi * f_resp * t)
        
        # 噪声
        noise = 0.1 * np.random.normal(0, 1, signal_length)
        
        # 基线漂移
        baseline = 0.15 * np.sin(2 * np.pi * 0.05 * t)
        
        signal = ppg_main + ppg_harmonic + ppg_resp + noise + baseline
        signals.append(signal)
        
        # 生成对应的血压值（基于一些简单的关系）
        # 这里使用简化的生理关系
        age_factor = np.random.normal(40, 15)  # 模拟年龄因素
        age_factor = np.clip(age_factor, 20, 80)
        
        # 收缩压：基于心率和年龄
        systolic = 90 + (heart_rate - 60) * 0.5 + (age_factor - 40) * 0.3 + np.random.normal(0, 5)
        systolic = np.clip(systolic, 90, 180)
        
        # 舒张压：通常是收缩压的60-80%
        diastolic_ratio = np.random.normal(0.67, 0.05)
        diastolic = systolic * diastolic_ratio + np.random.normal(0, 3)
        diastolic = np.clip(diastolic, 60, 120)
        
        blood_pressures.append([systolic, diastolic])
    
    return np.array(signals), np.array(blood_pressures)

def preprocess_data(signals: np.ndarray, config: dict) -> np.ndarray:
    """预处理数据"""
    print("预处理PPG信号...")
    preprocessor = PPGPreprocessor(config['data'])
    
    processed_signals = []
    for signal in signals:
        processed_signal = preprocessor.process_single_signal(signal)
        processed_signals.append(processed_signal)
    
    return np.array(processed_signals)

def extract_features(signals: np.ndarray, config: dict) -> np.ndarray:
    """提取特征"""
    print("提取特征...")
    feature_extractor = FeatureExtractor(config['features'])
    
    all_features = []
    for signal in signals:
        features = feature_extractor.extract_all_features(signal)
        
        # 组合所有特征
        feature_vector = []
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                if value.ndim > 1:
                    feature_vector.extend(value.flatten())
                else:
                    feature_vector.extend(value)
            else:
                feature_vector.append(value)
        
        all_features.append(feature_vector)
    
    return np.array(all_features)

def train_model(model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray, config: dict):
    """训练指定类型的模型"""
    print(f"训练 {model_type} 模型...")
    
    input_dim = X_train.shape[1]
    
    if model_type == 'mlp':
        model_config = config['models'].get('mlp', {})
        model_config['input_dim'] = input_dim
        model_config['output_dim'] = 2
        model = MLPModel(model_config)
    
    elif model_type == 'attention':
        model_config = config['models'].get('attention', {})
        model_config['input_dim'] = input_dim
        model_config['output_dim'] = 2
        model = AttentionModel(model_config)
    
    elif model_type == 'moe':
        model_config = config['models'].get('moe', {})
        model_config['input_dim'] = input_dim
        model_config['output_dim'] = 2
        model = MoEModel(model_config)
    
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    # 创建训练器
    trainer = ModelTrainer(model, config['training'])
    
    # 训练模型
    history = trainer.train(X_train, y_train, X_val, y_val)
    
    # 加载最佳模型
    trainer.load_best_model()
    
    return trainer, history

def main():
    parser = argparse.ArgumentParser(description='训练PPG血压预测模型')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--model', type=str, default='mlp',
                       choices=['mlp', 'attention', 'moe'],
                       help='模型类型')
    parser.add_argument('--data', type=str, default=None,
                       help='数据文件路径（如果不提供则生成合成数据）')
    parser.add_argument('--output', type=str, default='data/models/',
                       help='模型输出目录')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 加载或生成数据
    if args.data and os.path.exists(args.data):
        print(f"从 {args.data} 加载数据...")
        # 这里应该实现真实数据的加载逻辑
        # data = pd.read_csv(args.data)
        # signals = data['signals'].values
        # blood_pressures = data[['systolic', 'diastolic']].values
        raise NotImplementedError("真实数据加载尚未实现")
    else:
        print("使用合成数据进行演示...")
        signals, blood_pressures = generate_synthetic_data(
            n_samples=config.get('n_samples', 1000),
            signal_length=config['data']['signal_length']
        )
    
    # 预处理数据
    processed_signals = preprocess_data(signals, config)
    
    # 提取特征
    features = extract_features(processed_signals, config)
    
    print(f"特征维度: {features.shape}")
    print(f"血压标签维度: {blood_pressures.shape}")
    
    # 数据标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        features_scaled, blood_pressures, 
        test_size=0.2, random_state=42
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")
    
    # 训练模型
    trainer, history = train_model(args.model, X_train, y_train, X_val, y_val, config)
    
    # 评估模型
    print("\n评估模型性能...")
    train_pred = trainer.predict(X_train)
    val_pred = trainer.predict(X_val)
    
    train_metrics = evaluate_model(y_train, train_pred)
    val_metrics = evaluate_model(y_val, val_pred)
    
    print("\n训练集性能:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n验证集性能:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 保存模型和相关文件
    model_path = os.path.join(args.output, f'{args.model}_model.pth')
    scaler_path = os.path.join(args.output, f'{args.model}_scaler.joblib')
    config_path = os.path.join(args.output, f'{args.model}_config.yaml')
    
    trainer.model.save_model(model_path)
    joblib.dump(scaler, scaler_path)
    
    # 保存配置
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n模型已保存到: {model_path}")
    print(f"标准化器已保存到: {scaler_path}")
    print(f"配置已保存到: {config_path}")
    
    # 绘制训练历史
    trainer.plot_training_history()

if __name__ == '__main__':
    main()

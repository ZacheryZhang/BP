#!/usr/bin/env python3
"""
模型评估脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import numpy as np
import joblib
import torch
from sklearn.model_selection import train_test_split

from src.models.base_model import MLPModel
from src.models.attention_model import AttentionModel
from src.models.moe_model import MoEModel
from src.utils.metrics import (
    evaluate_model, plot_prediction_results, plot_error_distribution,
    plot_bland_altman, generate_evaluation_report
)

def load_model_and_scaler(model_path: str, scaler_path: str, config_path: str):
    """加载模型和标准化器"""
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 确定模型类型
    if 'mlp' in model_path.lower():
        model = MLPModel.load_model(model_path)
    elif 'attention' in model_path.lower():
        model = AttentionModel.load_model(model_path)
    elif 'moe' in model_path.lower():
        model = MoEModel.load_model(model_path)
    else:
        raise ValueError("无法从文件名确定模型类型")
    
    # 加载标准化器
    scaler = joblib.load(scaler_path)
    
    return model, scaler, config

def generate_test_data(config: dict, n_samples: int = 200):
    """生成测试数据"""
    from scripts.train import generate_synthetic_data, preprocess_data, extract_features
    
    # 生成合成数据
    signals, blood_pressures = generate_synthetic_data(
        n_samples=n_samples,
        signal_length=config['data']['signal_length']
    )
    
    # 预处理
    processed_signals = preprocess_data(signals, config)
    
    # 特征提取
    features = extract_features(processed_signals, config)
    
    return features, blood_pressures

def evaluate_single_model(model_path: str, scaler_path: str, config_path: str,
                         test_data: tuple = None, output_dir: str = "results/"):
    """评估单个模型"""
    print(f"评估模型: {model_path}")
    
    # 加载模型
    model, scaler, config = load_model_and_scaler(model_path, scaler_path, config_path)
    model.eval()
    
    # 准备测试数据
    if test_data is None:
        X_test, y_test = generate_test_data(config, n_samples=200)
    else:
        X_test, y_test = test_data
    
    # 标准化特征
    X_test_scaled = scaler.transform(X_test)
    
    # 预测
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy()
    
    # 评估
    metrics = evaluate_model(y_test, y_pred)
    
    # 生成报告
    model_name = os.path.basename(model_path).replace('.pth', '')
    report = generate_evaluation_report(y_test, y_pred, model_name)
    
    print(report)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存报告
    report_path = os.path.join(output_dir, f'{model_name}_evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 绘制图表
    plot_prediction_results(y_test, y_pred, 
                          title=f'{model_name} 预测结果',
                          save_path=os.path.join(output_dir, f'{model_name}_predictions.png'))
    
    plot_error_distribution(y_test, y_pred,
                          save_path=os.path.join(output_dir, f'{model_name}_error_dist.png'))
    
    plot_bland_altman(y_test, y_pred,
                     save_path=os.path.join(output_dir, f'{model_name}_bland_altman.png'))
    
    return metrics, y_test, y_pred

def compare_models(model_configs: list, test_data: tuple = None, output_dir: str = "results/"):
    """比较多个模型"""
    print("开始模型比较...")
    
    results = {}
    all_predictions = {}
    
    for config in model_configs:
        model_name = config['name']
        metrics, y_test, y_pred = evaluate_single_model(
            config['model_path'], 
            config['scaler_path'], 
            config['config_path'],
            test_data, 
            output_dir
        )
        
        results[model_name] = metrics
        all_predictions[model_name] = y_pred
    
    # 生成比较报告
    comparison_report = generate_comparison_report(results)
    print(comparison_report)
    
    # 保存比较报告
    comparison_path = os.path.join(output_dir, 'model_comparison.txt')
    with open(comparison_path, 'w', encoding='utf-8') as f:
        f.write(comparison_report)
    
    # 绘制比较图表
    plot_model_comparison(results, save_path=os.path.join(output_dir, 'model_comparison.png'))
    
    return results

def generate_comparison_report(results: dict) -> str:
    """生成模型比较报告"""
    report = """
模型性能比较报告
{'='*50}

"""
    
    # 创建比较表格
    metrics_to_compare = ['mae', 'rmse', 'r2', 'sbp_mae', 'dbp_mae', 'bp_classification_accuracy']
    
    report += f"{'模型':<15}"
    for metric in metrics_to_compare:
        report += f"{metric:<12}"
    report += "\n" + "-" * 100 + "\n"
    
    for model_name, metrics in results.items():
        report += f"{model_name:<15}"
        for metric in metrics_to_compare:
            value = metrics.get(metric, 0)
            if 'accuracy' in metric:
                report += f"{value*100:<12.2f}"
            else:
                report += f"{value:<12.4f}"
        report += "\n"
    
    # 找出最佳模型
    report += "\n最佳模型:\n"
    
    best_models = {}
    for metric in metrics_to_compare:
        if metric == 'r2' or 'accuracy' in metric:
            # 越大越好
            best_model = max(results.keys(), key=lambda x: results[x].get(metric, 0))
        else:
            # 越小越好
            best_model = min(results.keys(), key=lambda x: results[x].get(metric, float('inf')))
        
        best_models[metric] = best_model
        value = results[best_model][metric]
        if 'accuracy' in metric:
            report += f"  {metric}: {best_model} ({value*100:.2f}%)\n"
        else:
            report += f"  {metric}: {best_model} ({value:.4f})\n"
    
    return report

def plot_model_comparison(results: dict, save_path: str = None):
    """绘制模型比较图表"""
    import matplotlib.pyplot as plt
    
    models = list(results.keys())
    metrics = ['mae', 'rmse', 'r2', 'sbp_mae', 'dbp_mae']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [results[model].get(metric, 0) for model in models]
        
        bars = axes[i].bar(models, values, alpha=0.7)
        axes[i].set_title(f'{metric.upper()}', fontweight='bold')
        axes[i].set_ylabel(metric.upper())
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(values),
                        f'{value:.3f}', ha='center', va='bottom')
    
    # 血压分类准确率
    acc_values = [results[model].get('bp_classification_accuracy', 0) * 100 for model in models]
    bars = axes[5].bar(models, acc_values, alpha=0.7, color='green')
    axes[5].set_title('血压分类准确率 (%)', fontweight='bold')
    axes[5].set_ylabel('准确率 (%)')
    axes[5].tick_params(axis='x', rotation=45)
    axes[5].grid(True, alpha=0.3)
    
    for bar, value in zip(bars, acc_values):
        axes[5].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='评估PPG血压预测模型')
    parser.add_argument('--model', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--scaler', type=str, required=True,
                       help='标准化器文件路径')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--output', type=str, default='results/',
                       help='结果输出目录')
    parser.add_argument('--compare', type=str, nargs='*',
                       help='要比较的其他模型路径（格式：model_path,scaler_path,config_path）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    if args.compare:
        # 模型比较模式
        model_configs = [{
            'name': os.path.basename(args.model).replace('.pth', ''),
            'model_path': args.model,
            'scaler_path': args.scaler,
            'config_path': args.config
        }]
        
        for compare_str in args.compare:
            parts = compare_str.split(',')
            if len(parts) == 3:
                model_configs.append({
                    'name': os.path.basename(parts[0]).replace('.pth', ''),
                    'model_path': parts[0],
                    'scaler_path': parts[1],
                    'config_path': parts[2]
                })
        
        compare_models(model_configs, output_dir=args.output)
    
    else:
        # 单模型评估模式
        evaluate_single_model(args.model, args.scaler, args.config, output_dir=args.output)

if __name__ == '__main__':
    main()

"""
评估指标模块
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    评估模型性能
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        评估指标字典
    """
    metrics = {}
    
    # 基本回归指标
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # MAPE (平均绝对百分比误差)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    metrics['mape'] = mape
    
    # 分别计算收缩压和舒张压的指标
    if y_true.shape[1] == 2:  # 假设第一列是收缩压，第二列是舒张压
        # 收缩压指标
        metrics['sbp_mae'] = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
        metrics['sbp_rmse'] = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
        metrics['sbp_r2'] = r2_score(y_true[:, 0], y_pred[:, 0])
        
        # 舒张压指标
        metrics['dbp_mae'] = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
        metrics['dbp_rmse'] = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
        metrics['dbp_r2'] = r2_score(y_true[:, 1], y_pred[:, 1])
        
        # 血压分类准确率
        metrics.update(calculate_bp_classification_accuracy(y_true, y_pred))
    
    return metrics

def calculate_bp_classification_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算血压分类准确率
    
    Args:
        y_true: 真实血压值 [N, 2] (收缩压, 舒张压)
        y_pred: 预测血压值 [N, 2] (收缩压, 舒张压)
        
    Returns:
        分类准确率指标
    """
    def classify_bp(systolic, diastolic):
        """血压分类"""
        if systolic < 120 and diastolic < 80:
            return 0  # 正常
        elif systolic < 130 and diastolic < 80:
            return 1  # 血压偏高
        elif systolic < 140 or diastolic < 90:
            return 2  # 高血压1期
        elif systolic < 180 or diastolic < 120:
            return 3  # 高血压2期
        else:
            return 4  # 高血压危象
    
    # 分类真实值和预测值
    true_classes = np.array([classify_bp(sbp, dbp) for sbp, dbp in y_true])
    pred_classes = np.array([classify_bp(sbp, dbp) for sbp, dbp in y_pred])
    
    # 计算准确率
    accuracy = np.mean(true_classes == pred_classes)
    
    # 计算每个类别的准确率
    class_names = ['正常', '血压偏高', '高血压1期', '高血压2期', '高血压危象']
    class_accuracies = {}
    
    for i, class_name in enumerate(class_names):
        mask = true_classes == i
        if np.sum(mask) > 0:
            class_acc = np.mean(true_classes[mask] == pred_classes[mask])
            class_accuracies[f'{class_name}_accuracy'] = class_acc
        else:
            class_accuracies[f'{class_name}_accuracy'] = 0.0
    
    return {
        'bp_classification_accuracy': accuracy,
        **class_accuracies
    }

def plot_prediction_results(y_true: np.ndarray, y_pred: np.ndarray, 
                          title: str = "血压预测结果", save_path: str = None):
    """
    绘制预测结果图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        title: 图表标题
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 收缩压散点图
    axes[0, 0].scatter(y_true[:, 0], y_pred[:, 0], alpha=0.6, color='red')
    axes[0, 0].plot([y_true[:, 0].min(), y_true[:, 0].max()], 
                   [y_true[:, 0].min(), y_true[:, 0].max()], 'k--', lw=2)
    axes[0, 0].set_xlabel('真实收缩压 (mmHg)')
    axes[0, 0].set_ylabel('预测收缩压 (mmHg)')
    axes[0, 0].set_title('收缩压预测 vs 真实值')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 舒张压散点图
    axes[0, 1].scatter(y_true[:, 1], y_pred[:, 1], alpha=0.6, color='blue')
    axes[0, 1].plot([y_true[:, 1].min(), y_true[:, 1].max()], 
                   [y_true[:, 1].min(), y_true[:, 1].max()], 'k--', lw=2)
    axes[0, 1].set_xlabel('真实舒张压 (mmHg)')
    axes[0, 1].set_ylabel('预测舒张压 (mmHg)')
    axes[0, 1].set_title('舒张压预测 vs 真实值')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 残差图 - 收缩压
    sbp_residuals = y_true[:, 0] - y_pred[:, 0]
    axes[1, 0].scatter(y_pred[:, 0], sbp_residuals, alpha=0.6, color='red')
    axes[1, 0].axhline(y=0, color='k', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('预测收缩压 (mmHg)')
    axes[1, 0].set_ylabel('残差 (mmHg)')
    axes[1, 0].set_title('收缩压残差图')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 残差图 - 舒张压
    dbp_residuals = y_true[:, 1] - y_pred[:, 1]
    axes[1, 1].scatter(y_pred[:, 1], dbp_residuals, alpha=0.6, color='blue')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('预测舒张压 (mmHg)')
    axes[1, 1].set_ylabel('残差 (mmHg)')
    axes[1, 1].set_title('舒张压残差图')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray, 
                          save_path: str = None):
    """
    绘制误差分布图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 收缩压误差分布
    sbp_errors = y_pred[:, 0] - y_true[:, 0]
    axes[0, 0].hist(sbp_errors, bins=30, alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].axvline(np.mean(sbp_errors), color='darkred', linestyle='--', 
                      label=f'均值: {np.mean(sbp_errors):.2f}')
    axes[0, 0].set_xlabel('预测误差 (mmHg)')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].set_title('收缩压预测误差分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 舒张压误差分布
    dbp_errors = y_pred[:, 1] - y_true[:, 1]
    axes[0, 1].hist(dbp_errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].axvline(np.mean(dbp_errors), color='darkblue', linestyle='--', 
                      label=f'均值: {np.mean(dbp_errors):.2f}')
    axes[0, 1].set_xlabel('预测误差 (mmHg)')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].set_title('舒张压预测误差分布')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 绝对误差分布
    sbp_abs_errors = np.abs(sbp_errors)
    dbp_abs_errors = np.abs(dbp_errors)
    
    axes[1, 0].hist(sbp_abs_errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].axvline(np.mean(sbp_abs_errors), color='darkorange', linestyle='--',
                      label=f'MAE: {np.mean(sbp_abs_errors):.2f}')
    axes[1, 0].set_xlabel('绝对误差 (mmHg)')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].set_title('收缩压绝对误差分布')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(dbp_abs_errors, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].axvline(np.mean(dbp_abs_errors), color='darkgreen', linestyle='--',
                      label=f'MAE: {np.mean(dbp_abs_errors):.2f}')
    axes[1, 1].set_xlabel('绝对误差 (mmHg)')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].set_title('舒张压绝对误差分布')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_bland_altman(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
    """
    绘制Bland-Altman图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, (bp_type, color) in enumerate([('收缩压', 'red'), ('舒张压', 'blue')]):
        mean_values = (y_true[:, i] + y_pred[:, i]) / 2
        diff_values = y_pred[:, i] - y_true[:, i]
        
        mean_diff = np.mean(diff_values)
        std_diff = np.std(diff_values)
        
        axes[i].scatter(mean_values, diff_values, alpha=0.6, color=color)
        axes[i].axhline(mean_diff, color='red', linestyle='-', 
                       label=f'均值差异: {mean_diff:.2f}')
        axes[i].axhline(mean_diff + 1.96*std_diff, color='red', linestyle='--',
                       label=f'+1.96SD: {mean_diff + 1.96*std_diff:.2f}')
        axes[i].axhline(mean_diff - 1.96*std_diff, color='red', linestyle='--',
                       label=f'-1.96SD: {mean_diff - 1.96*std_diff:.2f}')
        
        axes[i].set_xlabel(f'平均{bp_type} (mmHg)')
        axes[i].set_ylabel(f'{bp_type}差异 (预测-真实)')
        axes[i].set_title(f'{bp_type} Bland-Altman图')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def calculate_clinical_accuracy(y_true: np.ndarray, y_pred: np.ndarray, 
                              tolerances: list = [5, 10, 15]) -> Dict[str, Dict[str, float]]:
    """
    计算临床准确性指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        tolerances: 容忍度列表 (mmHg)
        
    Returns:
        临床准确性指标
    """
    results = {}
    
    for tolerance in tolerances:
        sbp_within_tolerance = np.mean(np.abs(y_true[:, 0] - y_pred[:, 0]) <= tolerance)
        dbp_within_tolerance = np.mean(np.abs(y_true[:, 1] - y_pred[:, 1]) <= tolerance)
        both_within_tolerance = np.mean(
            (np.abs(y_true[:, 0] - y_pred[:, 0]) <= tolerance) & 
            (np.abs(y_true[:, 1] - y_pred[:, 1]) <= tolerance)
        )
        
        results[f'tolerance_{tolerance}mmHg'] = {
            'sbp_accuracy': sbp_within_tolerance * 100,
            'dbp_accuracy': dbp_within_tolerance * 100,
            'both_accuracy': both_within_tolerance * 100
        }
    
    return results

def generate_evaluation_report(y_true: np.ndarray, y_pred: np.ndarray, 
                             model_name: str = "模型") -> str:
    """
    生成评估报告
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        model_name: 模型名称
        
    Returns:
        评估报告字符串
    """
    metrics = evaluate_model(y_true, y_pred)
    clinical_acc = calculate_clinical_accuracy(y_true, y_pred)
    
    report = f"""
{model_name} 评估报告
{'='*50}

基本回归指标:
  - 平均绝对误差 (MAE): {metrics['mae']:.4f} mmHg
  - 均方根误差 (RMSE): {metrics['rmse']:.4f} mmHg
  - 决定系数 (R²): {metrics['r2']:.4f}
  - 平均绝对百分比误差 (MAPE): {metrics['mape']:.2f}%

收缩压指标:
  - MAE: {metrics['sbp_mae']:.4f} mmHg
  - RMSE: {metrics['sbp_rmse']:.4f} mmHg
  - R²: {metrics['sbp_r2']:.4f}

舒张压指标:
  - MAE: {metrics['dbp_mae']:.4f} mmHg
  - RMSE: {metrics['dbp_rmse']:.4f} mmHg
  - R²: {metrics['dbp_r2']:.4f}

血压分类准确率: {metrics['bp_classification_accuracy']:.2f}%

临床准确性:
"""
    
    for tolerance_key, acc_dict in clinical_acc.items():
        tolerance = tolerance_key.split('_')[1]
        report += f"  {tolerance} 内准确率:\n"
        report += f"    - 收缩压: {acc_dict['sbp_accuracy']:.1f}%\n"
        report += f"    - 舒张压: {acc_dict['dbp_accuracy']:.1f}%\n"
        report += f"    - 两者都在范围内: {acc_dict['both_accuracy']:.1f}%\n"
    
    return report

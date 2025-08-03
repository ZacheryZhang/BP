"""
可视化工具模块
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PPGVisualizer:
    """PPG数据可视化器"""
    
    def __init__(self, config=None):
        """
        初始化可视化器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.sampling_rate = self.config.get('sampling_rate', 125)
        self.figsize = self.config.get('figsize', (12, 8))
        
    def plot_ppg_signal(self, signal, title="PPG信号", save_path=None):
        """
        绘制PPG信号
        
        Args:
            signal: PPG信号数据
            title: 图标题
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        time_axis = np.arange(len(signal)) / self.sampling_rate
        ax.plot(time_axis, signal, 'b-', linewidth=1.5)
        
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('幅度')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_multiple_signals(self, signals, labels=None, title="多个PPG信号对比", 
                            max_signals=5, save_path=None):
        """
        绘制多个PPG信号对比
        
        Args:
            signals: 信号列表
            labels: 标签列表
            title: 图标题
            max_signals: 最大显示信号数
            save_path: 保存路径
        """
        n_signals = min(len(signals), max_signals)
        
        fig, axes = plt.subplots(n_signals, 1, figsize=(self.figsize[0], self.figsize[1]*n_signals/2))
        if n_signals == 1:
            axes = [axes]
        
        for i in range(n_signals):
            time_axis = np.arange(len(signals[i])) / self.sampling_rate
            axes[i].plot(time_axis, signals[i], linewidth=1.5)
            
            if labels is not None:
                axes[i].set_title(f'信号 {i+1}: {labels[i]}')
            else:
                axes[i].set_title(f'信号 {i+1}')
            
            axes[i].set_xlabel('时间 (秒)')
            axes[i].set_ylabel('幅度')
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_signal_preprocessing_steps(self, original, filtered, normalized, 
                                      title="信号预处理步骤", save_path=None):
        """
        绘制信号预处理各步骤
        
        Args:
            original: 原始信号
            filtered: 滤波后信号
            normalized: 归一化后信号
            title: 图标题
            save_path: 保存路径
        """
        fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1]*1.5))
        
        time_axis = np.arange(len(original)) / self.sampling_rate
        
        # 原始信号
        axes[0].plot(time_axis, original, 'b-', linewidth=1.5)
        axes[0].set_title('原始信号')
        axes[0].set_ylabel('幅度')
        axes[0].grid(True, alpha=0.3)
        
        # 滤波后信号
        axes[1].plot(time_axis, filtered, 'g-', linewidth=1.5)
        axes[1].set_title('滤波后信号')
        axes[1].set_ylabel('幅度')
        axes[1].grid(True, alpha=0.3)
        
        # 归一化后信号
        axes[2].plot(time_axis, normalized, 'r-', linewidth=1.5)
        axes[2].set_title('归一化后信号')
        axes[2].set_xlabel('时间 (秒)')
        axes[2].set_ylabel('幅度')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_frequency_spectrum(self, signal, title="频谱分析", save_path=None):
        """
        绘制频谱分析
        
        Args:
            signal: 信号数据
            title: 图标题
            save_path: 保存路径
        """
        from scipy.fft import fft, fftfreq
        
        # 计算FFT
        fft_values = fft(signal)
        freqs = fftfreq(len(signal), 1/self.sampling_rate)
        
        # 只取正频率部分
        positive_freqs = freqs[:len(freqs)//2]
        magnitude = np.abs(fft_values[:len(fft_values)//2])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        
        # 时域信号
        time_axis = np.arange(len(signal)) / self.sampling_rate
        ax1.plot(time_axis, signal, 'b-', linewidth=1.5)
        ax1.set_title('时域信号')
        ax1.set_xlabel('时间 (秒)')
        ax1.set_ylabel('幅度')
        ax1.grid(True, alpha=0.3)
        
        # 频域信号
        ax2.plot(positive_freqs, magnitude, 'r-', linewidth=1.5)
        ax2.set_title('频域信号')
        ax2.set_xlabel('频率 (Hz)')
        ax2.set_ylabel('幅度')
        ax2.set_xlim(0, 20)  # 只显示0-20Hz
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_distribution(self, features, feature_names, title="特征分布", 
                                save_path=None):
        """
        绘制特征分布
        
        Args:
            features: 特征矩阵
            feature_names: 特征名称列表
            title: 图标题
            save_path: 保存路径
        """
        n_features = min(len(feature_names), 16)  # 最多显示16个特征
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for i in range(n_features):
            axes[i].hist(features[:, i], bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(feature_names[i], fontsize=10)
            axes[i].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_features, 16):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_matrix(self, features, feature_names, title="特征相关性矩阵", 
                              save_path=None):
        """
        绘制特征相关性矩阵
        
        Args:
            features: 特征矩阵
            feature_names: 特征名称列表
            title: 图标题
            save_path: 保存路径
        """
        # 计算相关性矩阵
        corr_matrix = np.corrcoef(features.T)
        
        # 只显示前20个特征（避免图太大）
        n_features = min(len(feature_names), 20)
        corr_subset = corr_matrix[:n_features, :n_features]
        names_subset = feature_names[:n_features]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 绘制热力图
        im = ax.imshow(corr_subset, cmap='coolwarm', vmin=-1, vmax=1)
        
        # 设置标签
        ax.set_xticks(range(n_features))
        ax.set_yticks(range(n_features))
        ax.set_xticklabels(names_subset, rotation=45, ha='right')
        ax.set_yticklabels(names_subset)
        
        # 添加颜色条
        plt.colorbar(im, ax=ax)
        
        # 添加数值标注
        for i in range(n_features):
            for j in range(n_features):
                text = ax.text(j, i, f'{corr_subset[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class ModelVisualizer:
    """模型结果可视化器"""
    
    def __init__(self, config=None):
        """初始化模型可视化器"""
        self.config = config or {}
        self.figsize = self.config.get('figsize', (12, 8))
    
    def plot_training_history(self, train_losses, val_losses, title="训练历史", 
                            save_path=None):
        """
        绘制训练历史
        
        Args:
            train_losses: 训练损失列表
            val_

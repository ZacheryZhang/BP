"""
PPG信号预处理模块
"""
import numpy as np
from scipy import signal
from scipy.signal import savgol_filter
from typing import Optional, Tuple

class PPGPreprocessor:
    """PPG信号预处理器"""
    
    def __init__(self, config: dict = None):
        """
        初始化预处理器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.sampling_rate = self.config.get('sampling_rate', 125)
        self.filter_low = self.config.get('filter_low', 0.5)
        self.filter_high = self.config.get('filter_high', 8.0)
        self.normalize = self.config.get('normalize', True)
        
    def process_single_signal(self, ppg_signal: np.ndarray) -> np.ndarray:
        """
        处理单个PPG信号
        
        Args:
            ppg_signal: 原始PPG信号
            
        Returns:
            预处理后的信号
        """
        # 1. 去除异常值
        processed_signal = self._remove_outliers(ppg_signal)
        
        # 2. 带通滤波
        processed_signal = self._bandpass_filter(processed_signal)
        
        # 3. 平滑滤波
        processed_signal = self._smooth_filter(processed_signal)
        
        # 4. 归一化
        if self.normalize:
            processed_signal = self._normalize_signal(processed_signal)
        
        return processed_signal
    
    def _remove_outliers(self, signal: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """移除异常值"""
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        
        # 使用3σ准则
        mask = np.abs(signal - mean_val) < threshold * std_val
        
        # 对异常值进行插值替换
        if not np.all(mask):
            signal_clean = signal.copy()
            outlier_indices = np.where(~mask)[0]
            
            for idx in outlier_indices:
                # 使用邻近值的平均值替换
                start_idx = max(0, idx - 2)
                end_idx = min(len(signal), idx + 3)
                signal_clean[idx] = np.mean(signal[start_idx:end_idx][mask[start_idx:end_idx]])
            
            return signal_clean
        
        return signal
    
    def _bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """带通滤波"""
        nyquist = self.sampling_rate / 2
        low = self.filter_low / nyquist
        high = self.filter_high / nyquist
        
        # 设计Butterworth滤波器
        b, a = signal.butter(4, [low, high], btype='band')
        
        # 应用零相位滤波
        filtered_signal = signal.filtfilt(b, a, signal)
        
        return filtered_signal
    
    def _smooth_filter(self, signal: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
        """Savitzky-Golay平滑滤波"""
        if len(signal) < window_length:
            window_length = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
        
        return savgol_filter(signal, window_length, polyorder)
    
    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """信号归一化"""
        # Z-score标准化
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        
        if std_val > 0:
            return (signal - mean_val) / std_val
        else:
            return signal - mean_val
    
    def process_batch_signals(self, signals: np.ndarray) -> np.ndarray:
        """
        批量处理信号
        
        Args:
            signals: 信号数组 (shape: [n_samples, signal_length])
            
        Returns:
            预处理后的信号数组
        """
        processed_signals = []
        
        for signal in signals:
            processed_signal = self.process_single_signal(signal)
            processed_signals.append(processed_signal)
        
        return np.array(processed_signals)

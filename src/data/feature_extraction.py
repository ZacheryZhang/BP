"""
PPG信号特征提取模块
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import pywt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureExtractor:
    """PPG信号特征提取器"""
    
    def __init__(self, config):
        """
        初始化特征提取器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.sampling_rate = config.get('sampling_rate', 125)
        self.paa_size = config.get('paa_size', 50)
        self.wavelet = config.get('wavelet', 'db4')
        self.wavelet_levels = config.get('wavelet_levels', 5)
        self.entropy_m = config.get('entropy_m', 2)
        self.entropy_r = config.get('entropy_r', 0.2)
    
    def extract_statistical_features(self, signal_data):
        """
        提取统计特征
        
        Args:
            signal_data: PPG信号数据
            
        Returns:
            dict: 统计特征字典
        """
        if len(signal_data) == 0:
            raise ValueError("信号数据不能为空")
        
        if np.any(~np.isfinite(signal_data)):
            raise ValueError("信号数据包含无效值")
        
        features = {}
        
        # 基本统计量
        features['mean'] = np.mean(signal_data)
        features['std'] = np.std(signal_data)
        features['var'] = np.var(signal_data)
        features['skewness'] = stats.skew(signal_data)
        features['kurtosis'] = stats.kurtosis(signal_data)
        features['min'] = np.min(signal_data)
        features['max'] = np.max(signal_data)
        features['range'] = features['max'] - features['min']
        
        # 高级统计量
        features['rms'] = np.sqrt(np.mean(signal_data**2))
        features['energy'] = np.sum(signal_data**2)
        features['power'] = features['energy'] / len(signal_data)
        
        # 分位数特征
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            features[f'percentile_{p}'] = np.percentile(signal_data, p)
        
        # 变异系数
        if features['mean'] != 0:
            features['coefficient_of_variation'] = features['std'] / abs(features['mean'])
        else:
            features['coefficient_of_variation'] = 0
        
        return features
    
    def extract_frequency_features(self, signal_data):
        """
        提取频域特征
        
        Args:
            signal_data: PPG信号数据
            
        Returns:
            dict: 频域特征字典
        """
        # FFT变换
        fft_values = fft(signal_data)
        freqs = fftfreq(len(signal_data), 1/self.sampling_rate)
        
        # 只取正频率部分
        positive_freqs = freqs[:len(freqs)//2]
        magnitude = np.abs(fft_values[:len(fft_values)//2])
        power_spectrum = magnitude**2
        
        features = {}
        
        # 主导频率
        dominant_freq_idx = np.argmax(magnitude)
        features['dominant_freq'] = positive_freqs[dominant_freq_idx]
        
        # 频谱重心
        features['spectral_centroid'] = np.sum(positive_freqs * magnitude) / np.sum(magnitude)
        
        # 频谱带宽
        centroid = features['spectral_centroid']
        features['spectral_bandwidth'] = np.sqrt(
            np.sum(((positive_freqs - centroid)**2) * magnitude) / np.sum(magnitude)
        )
        
        # 频谱滚降点 (85%能量点)
        cumulative_power = np.cumsum(power_spectrum)
        total_power = cumulative_power[-1]
        rolloff_idx = np.where(cumulative_power >= 0.85 * total_power)[0]
        if len(rolloff_idx) > 0:
            features['spectral_rolloff'] = positive_freqs[rolloff_idx[0]]
        else:
            features['spectral_rolloff'] = positive_freqs[-1]
        
        # 频谱平坦度
        geometric_mean = stats.gmean(magnitude + 1e-10)
        arithmetic_mean = np.mean(magnitude)
        features['spectral_flatness'] = geometric_mean / arithmetic_mean
        
        # 低频/高频功率比
        lf_cutoff = 0.5  # Hz
        hf_cutoff = 15   # Hz
        lf_mask = (positive_freqs >= 0.04) & (positive_freqs <= lf_cutoff)
        hf_mask = (positive_freqs >= lf_cutoff) & (positive_freqs <= hf_cutoff)
        
        lf_power = np.sum(power_spectrum[lf_mask])
        hf_power = np.sum(power_spectrum[hf_mask])
        
        if hf_power > 0:
            features['power_ratio_lf_hf'] = lf_power / hf_power
        else:
            features['power_ratio_lf_hf'] = 0
        
        return features
    
    def extract_time_domain_features(self, signal_data):
        """
        提取时域特征
        
        Args:
            signal_data: PPG信号数据
            
        Returns:
            dict: 时域特征字典
        """
        features = {}
        
        # Hjorth参数
        diff1 = np.diff(signal_data)
        diff2 = np.diff(diff1)
        
        var_signal = np.var(signal_data)
        var_diff1 = np.var(diff1)
        var_diff2 = np.var(diff2)
        
        # Activity (方差)
        features['hjorth_activity'] = var_signal
        
        # Mobility (一阶导数标准差/信号标准差)
        if var_signal > 0:
            features['hjorth_mobility'] = np.sqrt(var_diff1 / var_signal)
        else:
            features['hjorth_mobility'] = 0
        
        # Complexity
        if var_diff1 > 0:
            mobility_diff = np.sqrt(var_diff2 / var_diff1)
            features['hjorth_complexity'] = mobility_diff / features['hjorth_mobility']
        else:
            features['hjorth_complexity'] = 0
        
        # 零交叉率
        zero_crossings = np.where(np.diff(np.signbit(signal_data)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(signal_data)
        
        # 峰值检测
        peaks, _ = signal.find_peaks(signal_data, height=np.mean(signal_data))
        features['peak_count'] = len(peaks)
        
        if len(peaks) > 0:
            peak_amplitudes = signal_data[peaks]
            features['peak_amplitude_mean'] = np.mean(peak_amplitudes)
            features['peak_amplitude_std'] = np.std(peak_amplitudes)
            
            # 峰间间隔
            if len(peaks) > 1:
                peak_intervals = np.diff(peaks) / self.sampling_rate
                features['peak_interval_mean'] = np.mean(peak_intervals)
                features['peak_interval_std'] = np.std(peak_intervals)
            else:
                features['peak_interval_mean'] = 0
                features['peak_interval_std'] = 0
        else:
            features['peak_amplitude_mean'] = 0
            features['peak_amplitude_std'] = 0
            features['peak_interval_mean'] = 0
            features['peak_interval_std'] = 0
        
        return features
    
    def extract_morphological_features(self, signal_data):
        """
        提取形态学特征
        
        Args:
            signal_data: PPG信号数据
            
        Returns:
            dict: 形态学特征字典
        """
        features = {}
        
        # 寻找主要的脉搏波
        peaks, _ = signal.find_peaks(signal_data, height=np.mean(signal_data))
        
        if len(peaks) > 0:
            # 选择最高的峰作为参考
            main_peak_idx = peaks[np.argmax(signal_data[peaks])]
            
            # 脉搏宽度 (半峰宽)
            peak_height = signal_data[main_peak_idx]
            half_height = (peak_height + np.mean(signal_data)) / 2
            
            # 寻找半峰宽点
            left_idx = main_peak_idx
            right_idx = main_peak_idx
            
            while left_idx > 0 and signal_data[left_idx] > half_height:
                left_idx -= 1
            
            while right_idx < len(signal_data) - 1 and signal_data[right_idx] > half_height:
                right_idx += 1
            
            features['pulse_width'] = (right_idx - left_idx) / self.sampling_rate
            
            # 上升时间和下降时间
            # 寻找波谷
            valleys, _ = signal.find_peaks(-signal_data)
            
            if len(valleys) > 0:
                # 找到主峰前后的波谷
                left_valley = valleys[valleys < main_peak_idx]
                right_valley = valleys[valleys > main_peak_idx]
                
                if len(left_valley) > 0:
                    left_valley_idx = left_valley[-1]
                    features['rise_time'] = (main_peak_idx - left_valley_idx) / self.sampling_rate
                else:
                    features['rise_time'] = 0
                
                if len(right_valley) > 0:
                    right_valley_idx = right_valley[0]
                    features['fall_time'] = (right_valley_idx - main_peak_idx) / self.sampling_rate
                else:
                    features['fall_time'] = 0
            else:
                features['rise_time'] = 0
                features['fall_time'] = 0
            
            # 脉搏面积
            if len(valleys) >= 2:
                left_valley_idx = valleys[valleys < main_peak_idx][-1] if len(valleys[valleys < main_peak_idx]) > 0 else 0
                right_valley_idx = valleys[valleys > main_peak_idx][0] if len(valleys[valleys > main_peak_idx]) > 0 else len(signal_data) - 1
                
                pulse_segment = signal_data[left_valley_idx:right_valley_idx+1]
                baseline = np.min(pulse_segment)
                features['pulse_area'] = np.trapz(pulse_segment - baseline)
            else:
                features['pulse_area'] = 0
            
            # 脉搏对称性
            if features['rise_time'] > 0 and features['fall_time'] > 0:
                features['pulse_symmetry'] = (features['rise_time'] - features['fall_time']) / (features['rise_time'] + features['fall_time'])
            else:
                features['pulse_symmetry'] = 0
        
        else:
            features['pulse_width'] = 0
            features['rise_time'] = 0
            features['fall_time'] = 0
            features['pulse_area'] = 0
            features['pulse_symmetry'] = 0
        
        return features
    
    def extract_wavelet_features(self, signal_data):
        """
        提取小波特征
        
        Args:
            signal_data: PPG信号数据
            
        Returns:
            np.ndarray: 小波特征向量
        """
        # 小波分解
        coeffs = pywt.wavedec(signal_data, self.wavelet, level=self.wavelet_levels)
        
        # 计算各层的能量
        energies = []
        for coeff in coeffs:
            energy = np.sum(coeff**2)
            energies.append(energy)
        
        # 归一化能量
        total_energy = sum(energies)
        if total_energy > 0:
            normalized_energies = [e / total_energy for e in energies]
        else:
            normalized_energies = [0] * len(energies)
        
        return np.array(normalized_energies)
    
    def extract_paa_features(self, signal_data):
        """
        提取分段聚合近似(PAA)特征
        
        Args:
            signal_data: PPG信号数据
            
        Returns:
            np.ndarray: PAA特征向量
        """
        n = len(signal_data)
        segment_length = n // self.paa_size
        
        paa_features = []
        for i in range(self.paa_size):
            start_idx = i * segment_length
            end_idx = min((i + 1) * segment_length, n)
            
            if start_idx < end_idx:
                segment_mean = np.mean(signal_data[start_idx:end_idx])
            else:
                segment_mean = 0
            
            paa_features.append(segment_mean)
        
        return np.array(paa_features)
    
    def extract_entropy_features(self, signal_data):
        """
        提取熵特征
        
        Args:
            signal_data: PPG信号数据
            
        Returns:
            dict: 熵特征字典
        """
        features = {}
        
        # 样本熵
        features['sample_entropy'] = self._sample_entropy(signal_data, self.entropy_m, self.entropy_r)
        
        # 近似熵
        features['approximate_entropy'] = self._approximate_entropy(signal_data, self.entropy_m, self.entropy_r)
        
        # 排列熵
        features['permutation_entropy'] = self._permutation_entropy(signal_data)
        
        return features
    
    def extract_nonlinear_features(self, signal_data):
        """
        提取非线性特征
        
        Args:
            signal_data: PPG信号数据
            
        Returns:
            dict: 非线性特征字典
        """
        features = {}
        
        # 关联维数 (简化版本)
        features['correlation_dimension'] = self._correlation_dimension(signal_data)
        
        # 最大Lyapunov指数 (简化估计)
        features['largest_lyapunov_exponent'] = self._lyapunov_exponent(signal_data)
        
        # Hurst指数
        features['hurst_exponent'] = self._hurst_exponent(signal_data)
        
        # 去趋势波动分析
        features['detrended_fluctuation'] = self._dfa(signal_data)
        
        return features
    
    def extract_all_features(self, signal_data):
        """
        提取所有特征
        
        Args:
            signal_data: PPG信号数据
            
        Returns:
            dict: 所有特征字典
        """
        if len(signal_data) == 0:
            raise ValueError("信号数据不能为空")
        
        if np.any(~np.isfinite(signal_data)):
            raise ValueError("信号数据包含无效值")
        
        all_features = {}
        
        # 提取各类特征
        all_features['statistical'] = self.extract_statistical_features(signal_data)
        all_features['frequency'] = self.extract_frequency_features(signal_data)
        all_features['time_domain'] = self.extract_time_domain_features(signal_data)
        all_features['morphological'] = self.extract_morphological_features(signal_data)
        all_features['wavelet'] = self.extract_wavelet_features(signal_data)
        all_features['p

"""
Enhanced I-ASNH Meta-Learning Framework for KDD-Quality Results
Improved feature engineering, meta-learning, and evaluation for publication-quality performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional
import logging
from collections import deque
import random

class ConvolutionalFeatureExtractor(nn.Module):
    """Convolutional feature extractor for temporal patterns"""

    def __init__(self, seq_len: int = 96, feature_dim: int = 128):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract convolutional features from time series"""
        x = x.unsqueeze(1)  # Add channel dimension
        conv_out = self.conv_layers(x)
        conv_out = conv_out.squeeze(-1)  # Remove last dimension
        features = self.fc(conv_out)
        return features

class FeatureFusionNetwork(nn.Module):
    """Enhanced feature fusion network"""

    def __init__(self, feature_dim: int = 128):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim * 2),
            nn.Dropout(0.2),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, conv_features: torch.Tensor, stat_features: torch.Tensor) -> torch.Tensor:
        """Fuse convolutional and statistical features"""
        combined = torch.cat([conv_features, stat_features], dim=1)
        fused = self.fusion(combined)
        return fused

class EnhancedStatisticalFeatureExtractor(nn.Module):
    """Enhanced statistical feature extractor with comprehensive time series characteristics"""
    
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        
        # Expanded feature dimension to accommodate more features
        self.stat_proj = nn.Sequential(
            nn.Linear(32, 128),  # Increased from 20 to 32 features
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )
        
    def extract_enhanced_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract comprehensive statistical features optimized for method selection"""
        batch_size = x.shape[0]
        features = []
        
        for i in range(batch_size):
            series = x[i].cpu().numpy()
            n = len(series)
            
            # Basic statistics
            mean_val = np.mean(series)
            std_val = np.std(series)
            min_val = np.min(series)
            max_val = np.max(series)
            median_val = np.median(series)
            
            # Enhanced trend analysis
            diff = np.diff(series)
            trend_strength = np.std(diff) / (np.std(series) + 1e-8)
            
            # Linear trend coefficient
            x_vals = np.arange(n)
            trend_coef = np.polyfit(x_vals, series, 1)[0] if n > 1 else 0
            
            # Multiple autocorrelation lags for better pattern detection
            autocorr_1 = np.corrcoef(series[:-1], series[1:])[0, 1] if n > 1 else 0
            autocorr_7 = np.corrcoef(series[:-7], series[7:])[0, 1] if n > 7 else 0
            autocorr_12 = np.corrcoef(series[:-12], series[12:])[0, 1] if n > 12 else 0
            autocorr_24 = np.corrcoef(series[:-24], series[24:])[0, 1] if n > 24 else 0
            autocorr_48 = np.corrcoef(series[:-48], series[48:])[0, 1] if n > 48 else 0
            
            # Variability measures
            cv = std_val / (abs(mean_val) + 1e-8)
            iqr = np.percentile(series, 75) - np.percentile(series, 25)
            
            # Enhanced seasonality detection
            seasonal_strength_24 = 0
            seasonal_strength_12 = 0
            if n >= 24:
                seasonal_diff_24 = np.var(series[::24]) if len(series[::24]) > 1 else 0
                seasonal_strength_24 = seasonal_diff_24 / (np.var(series) + 1e-8)
            if n >= 12:
                seasonal_diff_12 = np.var(series[::12]) if len(series[::12]) > 1 else 0
                seasonal_strength_12 = seasonal_diff_12 / (np.var(series) + 1e-8)
            
            # Complexity and entropy measures
            hist, _ = np.histogram(series, bins=10)
            hist = hist / n
            entropy = -np.sum(hist * np.log(hist + 1e-8))
            
            # Stationarity indicator (simplified)
            rolling_std = np.std(series[:n//2]) if n > 10 else std_val
            stationarity = abs(rolling_std - np.std(series[n//2:])) / (std_val + 1e-8) if n > 10 else 0
            
            # Outlier detection
            q1, q3 = np.percentile(series, [25, 75])
            outlier_ratio = np.sum((series < q1 - 1.5*iqr) | (series > q3 + 1.5*iqr)) / n
            
            # Nonlinearity measures
            second_diff = np.diff(diff) if len(diff) > 1 else np.array([0])
            nonlinearity = np.std(second_diff) / (np.std(diff) + 1e-8) if len(diff) > 1 else 0
            
            # Distribution characteristics
            skewness = stats.skew(series)
            kurtosis = stats.kurtosis(series)
            
            # Frequency domain characteristics (simplified)
            if n > 4:
                fft_vals = np.abs(np.fft.fft(series))[:n//2]
                spectral_centroid = np.sum(np.arange(len(fft_vals)) * fft_vals) / (np.sum(fft_vals) + 1e-8)
                spectral_rolloff = np.percentile(fft_vals, 85)
            else:
                spectral_centroid = 0
                spectral_rolloff = 0
            
            # Method-specific indicators
            # These features help distinguish which methods work best
            volatility = np.std(diff) / (abs(mean_val) + 1e-8)  # For GARCH-like methods
            persistence = autocorr_1  # For AR-like methods
            seasonality_score = max(seasonal_strength_24, seasonal_strength_12)  # For seasonal methods
            
            # Combine all features (32 features total)
            stat_feat = [
                mean_val, std_val, min_val, max_val, median_val,
                trend_strength, trend_coef, 
                autocorr_1, autocorr_7, autocorr_12, autocorr_24, autocorr_48,
                cv, iqr, seasonal_strength_24, seasonal_strength_12,
                skewness, kurtosis, entropy, stationarity, outlier_ratio,
                nonlinearity, spectral_centroid, spectral_rolloff,
                volatility, persistence, seasonality_score,
                n, np.sum(np.abs(diff)), np.mean(np.abs(diff)), 
                np.percentile(series, 10), np.percentile(series, 90)
            ]
            
            # Handle NaN/inf values
            stat_feat = [0.0 if np.isnan(f) or np.isinf(f) else float(f) for f in stat_feat]
            features.append(stat_feat)
        
        return torch.tensor(features, dtype=torch.float32, device=x.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through enhanced statistical feature extractor"""
        stat_features = self.extract_enhanced_features(x)
        processed_features = self.stat_proj(stat_features)
        return processed_features

class EnhancedMetaLearningNetwork(nn.Module):
    """Enhanced meta-learning network with improved architecture for better selection accuracy"""
    
    def __init__(self, feature_dim: int = 128, num_methods: int = 12):
        super().__init__()
        
        self.num_methods = num_methods
        
        # Enhanced method selection network with attention mechanism
        self.method_selector = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_methods)
        )
        
        # Confidence estimation network
        self.confidence_estimator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Learnable temperature parameter for better calibration
        self.temperature = nn.Parameter(torch.tensor(2.0))
        
        # Method-specific feature attention
        self.method_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with enhanced method selection and confidence estimation
        """
        batch_size = features.shape[0]
        
        # Apply self-attention to features
        attended_features, _ = self.method_attention(
            features.unsqueeze(1), features.unsqueeze(1), features.unsqueeze(1)
        )
        attended_features = attended_features.squeeze(1)
        
        # Method selection with temperature scaling
        method_logits = self.method_selector(attended_features)
        method_probs = F.softmax(method_logits / torch.clamp(self.temperature, min=0.1, max=10.0), dim=1)
        
        # Confidence estimation
        confidence = self.confidence_estimator(attended_features)
        
        return method_probs, confidence

class EnhancedIASNHFramework(nn.Module):
    """Enhanced I-ASNH framework targeting >80% selection accuracy for KDD-quality results"""
    
    def __init__(self, seq_len: int = 96, feature_dim: int = 128, num_methods: int = 12):
        super().__init__()
        
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.num_methods = num_methods
        
        # Enhanced feature extractors
        self.conv_extractor = ConvolutionalFeatureExtractor(seq_len, feature_dim)
        self.stat_extractor = EnhancedStatisticalFeatureExtractor(feature_dim)
        self.fusion_network = FeatureFusionNetwork(feature_dim)
        self.meta_network = EnhancedMetaLearningNetwork(feature_dim, num_methods)
        
        # Performance tracking for adaptive learning
        self.performance_history = {}
        self.method_names = [
            'Linear', 'ARIMA', 'ETS', 'Prophet', 'LSTM', 'DLinear',
            'Transformer', 'N_BEATS', 'Informer', 'DeepAR', 'FFORMA', 'Rule_based'
        ]
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced forward pass with improved feature fusion"""
        # Extract features
        conv_features = self.conv_extractor(x)
        stat_features = self.stat_extractor(x)
        
        # Fuse features
        fused_features = self.fusion_network(conv_features, stat_features)
        
        # Meta-learning prediction
        method_probs, confidence = self.meta_network(fused_features)
        
        return method_probs, confidence
    
    def select_method(self, x: torch.Tensor, return_confidence: bool = False) -> Tuple[int, Optional[float]]:
        """Select best method with enhanced decision making"""
        self.eval()
        with torch.no_grad():
            method_probs, confidence = self.forward(x)
            
            # Enhanced selection strategy
            selected_method = torch.argmax(method_probs, dim=1).item()
            conf_score = confidence.item() if return_confidence else None
            
            return (selected_method, conf_score) if return_confidence else selected_method

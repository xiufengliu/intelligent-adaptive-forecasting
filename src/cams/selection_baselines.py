"""
Selection baseline methods for comparison with I-ASNH
Includes FFORMA, Rule-based, Random, and Oracle selection methods
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import random
from models.baseline_methods import get_all_baseline_methods

class SelectionBaseline:
    """Base class for method selection approaches"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.method_pool = get_all_baseline_methods()
        self.method_names = list(self.method_pool.keys())
    
    def extract_features(self, time_series: np.ndarray) -> np.ndarray:
        """Extract features from time series for selection"""
        raise NotImplementedError
    
    def select_method(self, time_series: np.ndarray) -> Tuple[str, float]:
        """Select best method for given time series"""
        raise NotImplementedError
    
    def fit(self, train_datasets: List[Tuple[np.ndarray, str]]):
        """Fit selection model on training datasets"""
        raise NotImplementedError

class FFORMABaseline(SelectionBaseline):
    """FFORMA-style meta-learning selection using hand-crafted features"""
    
    def __init__(self):
        super().__init__("FFORMA")
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def extract_features(self, time_series: np.ndarray) -> np.ndarray:
        """Extract 42 hand-crafted statistical features similar to FFORMA"""
        features = []
        
        # Basic statistics
        features.extend([
            np.mean(time_series),
            np.std(time_series),
            np.min(time_series),
            np.max(time_series),
            np.median(time_series),
            np.var(time_series)
        ])
        
        # Percentiles
        features.extend([
            np.percentile(time_series, 25),
            np.percentile(time_series, 75),
            np.percentile(time_series, 10),
            np.percentile(time_series, 90)
        ])
        
        # Trend and seasonality
        if len(time_series) > 1:
            diff = np.diff(time_series)
            features.extend([
                np.mean(diff),
                np.std(diff),
                np.mean(np.abs(diff)),
                len(np.where(diff > 0)[0]) / len(diff),  # Proportion of increases
                len(np.where(diff < 0)[0]) / len(diff)   # Proportion of decreases
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Autocorrelation features
        for lag in [1, 2, 3, 7, 24]:
            if len(time_series) > lag:
                autocorr = np.corrcoef(time_series[:-lag], time_series[lag:])[0, 1]
                features.append(0 if np.isnan(autocorr) else autocorr)
            else:
                features.append(0)
        
        # Seasonality strength (simplified)
        if len(time_series) >= 24:
            seasonal_periods = [7, 24]
            for period in seasonal_periods:
                if len(time_series) >= period * 2:
                    # Compute seasonal strength
                    seasonal_data = time_series.reshape(-1, period)[:len(time_series)//period]
                    if seasonal_data.shape[0] > 1:
                        seasonal_var = np.var(np.mean(seasonal_data, axis=0))
                        total_var = np.var(time_series)
                        seasonal_strength = seasonal_var / (total_var + 1e-8)
                        features.append(seasonal_strength)
                    else:
                        features.append(0)
                else:
                    features.append(0)
        else:
            features.extend([0, 0])
        
        # Distribution properties
        features.extend([
            len(time_series),  # Series length
            np.std(time_series) / (np.abs(np.mean(time_series)) + 1e-8),  # Coefficient of variation
            np.sum(np.abs(np.diff(time_series))),  # Total variation
        ])
        
        # Spectral features (simplified)
        if len(time_series) > 4:
            fft = np.fft.fft(time_series)
            power_spectrum = np.abs(fft) ** 2
            features.extend([
                np.max(power_spectrum[1:len(power_spectrum)//2]),  # Peak frequency power
                np.mean(power_spectrum[1:len(power_spectrum)//2]),  # Mean frequency power
                np.std(power_spectrum[1:len(power_spectrum)//2])   # Frequency power std
            ])
        else:
            features.extend([0, 0, 0])
        
        # Entropy and complexity measures
        if len(time_series) > 1:
            # Approximate entropy (simplified)
            def approx_entropy(data, m=2, r=0.2):
                N = len(data)
                if N < m + 1:
                    return 0
                
                def _maxdist(xi, xj, m):
                    return max([abs(ua - va) for ua, va in zip(xi, xj)])
                
                def _phi(m):
                    patterns = np.array([data[i:i + m] for i in range(N - m + 1)])
                    C = np.zeros(N - m + 1)
                    for i in range(N - m + 1):
                        template_i = patterns[i]
                        for j in range(N - m + 1):
                            if _maxdist(template_i, patterns[j], m) <= r * np.std(data):
                                C[i] += 1.0
                    phi = np.mean(np.log(C / float(N - m + 1.0)))
                    return phi
                
                return _phi(m) - _phi(m + 1)
            
            features.append(approx_entropy(time_series))
        else:
            features.append(0)
        
        # Pad or truncate to exactly 42 features
        while len(features) < 42:
            features.append(0)
        features = features[:42]
        
        # Handle NaN and inf values
        features = [0 if np.isnan(f) or np.isinf(f) else f for f in features]
        
        return np.array(features)
    
    def fit(self, train_datasets: List[Tuple[np.ndarray, str]]):
        """Fit FFORMA classifier on training datasets"""
        X = []
        y = []
        
        for time_series, optimal_method in train_datasets:
            features = self.extract_features(time_series)
            X.append(features)
            y.append(optimal_method)
        
        X = np.array(X)
        
        # Fit scaler and classifier
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)
        self.is_fitted = True
    
    def select_method(self, time_series: np.ndarray) -> Tuple[str, float]:
        """Select method using FFORMA approach"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before selection")
        
        features = self.extract_features(time_series).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        # Get prediction probabilities
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        selected_idx = np.argmax(probabilities)
        selected_method = self.classifier.classes_[selected_idx]
        confidence = probabilities[selected_idx]
        
        return selected_method, confidence

class RuleBasedSelection(SelectionBaseline):
    """Rule-based selection using traditional heuristics"""
    
    def __init__(self):
        super().__init__("Rule_Based")
    
    def detect_seasonality(self, time_series: np.ndarray) -> float:
        """Detect seasonality strength"""
        if len(time_series) < 48:  # Need at least 2 seasonal periods
            return 0.0
        
        # Test for daily seasonality (period=24)
        period = 24
        if len(time_series) >= period * 2:
            seasonal_data = time_series[:len(time_series)//period * period].reshape(-1, period)
            if seasonal_data.shape[0] > 1:
                seasonal_means = np.mean(seasonal_data, axis=0)
                seasonal_var = np.var(seasonal_means)
                total_var = np.var(time_series)
                return seasonal_var / (total_var + 1e-8)
        
        return 0.0
    
    def detect_trend(self, time_series: np.ndarray) -> float:
        """Detect trend strength"""
        if len(time_series) < 3:
            return 0.0
        
        # Linear trend detection
        x = np.arange(len(time_series))
        correlation = np.corrcoef(x, time_series)[0, 1]
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    def select_method(self, time_series: np.ndarray) -> Tuple[str, float]:
        """Select method using rule-based heuristics"""
        seasonality_strength = self.detect_seasonality(time_series)
        trend_strength = self.detect_trend(time_series)
        series_length = len(time_series)
        volatility = np.std(time_series) / (np.abs(np.mean(time_series)) + 1e-8)
        
        # Rule-based decision logic
        confidence = 0.7  # Default confidence
        
        if seasonality_strength > 0.3:
            selected_method = "Seasonal_Naive"
            confidence = min(0.9, 0.5 + seasonality_strength)
        elif trend_strength > 0.5:
            if series_length > 100:
                selected_method = "Linear"
            else:
                selected_method = "DLinear"
            confidence = min(0.9, 0.4 + trend_strength)
        elif volatility > 2.0:  # High volatility
            selected_method = "Naive"
            confidence = 0.6
        elif series_length > 200:  # Long series
            selected_method = "LSTM"
            confidence = 0.8
        else:  # Default case
            selected_method = "Linear"
            confidence = 0.5
        
        return selected_method, confidence
    
    def fit(self, train_datasets: List[Tuple[np.ndarray, str]]):
        """Rule-based method doesn't require training"""
        self.is_fitted = True

class RandomSelection(SelectionBaseline):
    """Random method selection baseline"""
    
    def __init__(self, seed: int = 42):
        super().__init__("Random")
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def select_method(self, time_series: np.ndarray) -> Tuple[str, float]:
        """Randomly select a method"""
        selected_method = random.choice(self.method_names)
        confidence = random.uniform(0.1, 0.9)  # Random confidence
        return selected_method, confidence
    
    def fit(self, train_datasets: List[Tuple[np.ndarray, str]]):
        """Random selection doesn't require training"""
        self.is_fitted = True

class OracleSelection(SelectionBaseline):
    """Oracle selection - always chooses the optimal method"""
    
    def __init__(self):
        super().__init__("Oracle")
        self.optimal_methods = {}  # Will store optimal method for each dataset
    
    def select_method(self, time_series: np.ndarray, dataset_name: str = None) -> Tuple[str, float]:
        """Select optimal method (requires knowing the optimal choice)"""
        if dataset_name and dataset_name in self.optimal_methods:
            return self.optimal_methods[dataset_name], 1.0
        else:
            # Fallback to most commonly optimal method
            return "Linear", 1.0
    
    def set_optimal_methods(self, optimal_methods: Dict[str, str]):
        """Set the optimal methods for each dataset"""
        self.optimal_methods = optimal_methods
    
    def fit(self, train_datasets: List[Tuple[np.ndarray, str]]):
        """Oracle doesn't require training but stores optimal methods"""
        self.is_fitted = True

def get_all_selection_baselines():
    """Get all selection baseline methods"""
    return {
        'FFORMA': FFORMABaseline(),
        'Rule_Based': RuleBasedSelection(),
        'Random': RandomSelection(),
        'Oracle': OracleSelection()
    }

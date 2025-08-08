"""
CaMS (Calibrated Meta-Selection) core implementation.

Refactored from the previous I-ASNH implementation. Provides calibrated
selection confidence and unified policy supporting supervised meta-learning
with optional reinforcement learning refinement.
"""
from __future__ import annotations

import logging
import random
from collections import deque
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats


class MultiScaleConvolutionalExtractor(nn.Module):
    """Multi-scale convolutional feature extraction with attention mechanism"""

    def __init__(self, input_dim: int = 96, feature_dim: int = 128):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 48, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(48, 96, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(96, 128, kernel_size=7, padding=3)

        self.bn1 = nn.BatchNorm1d(48)
        self.bn2 = nn.BatchNorm1d(96)
        self.bn3 = nn.BatchNorm1d(128)

        self.attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=8, dropout=0.1, batch_first=True
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_proj = nn.Linear(128, feature_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract convolutional features from raw series [B, L] -> [B, D]."""
        x = x.unsqueeze(1)
        h1 = F.relu(self.bn1(self.conv1(x)))
        h2 = F.relu(self.bn2(self.conv2(h1)))
        h3 = F.relu(self.bn3(self.conv3(h2)))
        h3t = h3.transpose(1, 2)
        attended, _ = self.attention(h3t, h3t, h3t)
        attended = attended.transpose(1, 2)
        pooled = self.global_pool(attended).squeeze(-1)
        features = self.feature_proj(pooled)
        return self.dropout(features)


class StatisticalFeatureExtractor(nn.Module):
    """Statistical feature extraction for domain knowledge integration"""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.stat_features_dim = 20
        self.stat_proj = nn.Sequential(
            nn.Linear(self.stat_features_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, feature_dim),
        )

    def extract_statistical_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        features = []
        for i in range(batch_size):
            series = x[i].cpu().numpy()
            mean_val = np.mean(series)
            std_val = np.std(series)
            min_val = np.min(series)
            max_val = np.max(series)
            median_val = np.median(series)
            diff = np.diff(series)
            trend_strength = np.std(diff) / (np.std(series) + 1e-8)
            autocorr_1 = np.corrcoef(series[:-1], series[1:])[0, 1] if len(series) > 1 else 0
            autocorr_7 = np.corrcoef(series[:-7], series[7:])[0, 1] if len(series) > 7 else 0
            autocorr_24 = (
                np.corrcoef(series[:-24], series[24:])[0, 1] if len(series) > 24 else 0
            )
            cv = std_val / (abs(mean_val) + 1e-8)
            iqr = np.percentile(series, 75) - np.percentile(series, 25)
            seasonal_strength = 0
            if len(series) >= 24:
                seasonal_diff = np.var(series[::24]) if len(series[::24]) > 1 else 0
                seasonal_strength = seasonal_diff / (np.var(series) + 1e-8)
            stat_feat = [
                mean_val,
                std_val,
                min_val,
                max_val,
                median_val,
                trend_strength,
                autocorr_1,
                autocorr_7,
                autocorr_24,
                cv,
                iqr,
                seasonal_strength,
                stats.skew(series),
                stats.kurtosis(series),
                len(series),
                np.sum(np.abs(diff)),
                np.mean(np.abs(diff)),
                np.std(diff),
                np.percentile(series, 25),
                np.percentile(series, 75),
            ]
            stat_feat = [0.0 if np.isnan(f) or np.isinf(f) else f for f in stat_feat]
            features.append(stat_feat)
        return torch.tensor(features, dtype=torch.float32, device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and project statistical features [B, L] -> [B, D]."""
        return self.stat_proj(self.extract_statistical_features(x))


class FeatureFusionNetwork(nn.Module):
    """Fuse convolutional and statistical features."""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, conv_features: torch.Tensor, stat_features: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([conv_features, stat_features], dim=1)
        return self.fusion(combined)


class MetaLearningNetwork(nn.Module):
    """Meta-learning network for method selection with confidence estimation."""

    def __init__(self, feature_dim: int = 128, num_methods: int = 6):
        super().__init__()
        self.num_methods = num_methods
        self.method_selector = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_methods),
        )
        self.confidence_estimator = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.method_discriminators = nn.ModuleList(
            [nn.Sequential(nn.Linear(feature_dim, 32), nn.ReLU(), nn.Linear(32, 1)) for _ in range(num_methods)]
        )
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        method_logits = self.method_selector(features)
        discriminator_logits = torch.cat([d(features) for d in self.method_discriminators], dim=1)
        combined_logits = method_logits + 0.3 * discriminator_logits
        scaled_logits = combined_logits / torch.clamp(self.temperature, min=0.1, max=10.0)
        method_probs = F.softmax(scaled_logits, dim=1)
        confidence_base = self.confidence_estimator(features)
        entropy = -torch.sum(method_probs * torch.log(method_probs + 1e-8), dim=1, keepdim=True)
        max_entropy = torch.log(torch.tensor(float(self.num_methods)))
        normalized_entropy = entropy / max_entropy
        confidence = confidence_base * (1.0 - 0.5 * normalized_entropy)
        return method_probs, confidence

    def compute_loss(
        self,
        method_probs: torch.Tensor,
        confidence: torch.Tensor,
        optimal_methods: torch.Tensor,
        performance_scores: torch.Tensor,
    ) -> torch.Tensor:
        smoothed_targets = F.one_hot(optimal_methods, num_classes=self.num_methods).float()
        smoothed_targets = smoothed_targets * 0.9 + 0.1 / self.num_methods
        selection_loss = F.kl_div(F.log_softmax(method_probs, dim=1), smoothed_targets, reduction="batchmean")
        batch_mean_probs = method_probs.mean(dim=0)
        diversity_loss = -torch.sum(batch_mean_probs * torch.log(batch_mean_probs + 1e-8))
        diversity_loss = -diversity_loss
        temp_reg = torch.abs(self.temperature - 2.0)
        if performance_scores.max() > performance_scores.min():
            normalized_performance = (performance_scores - performance_scores.min()) / (
                performance_scores.max() - performance_scores.min()
            )
        else:
            normalized_performance = torch.ones_like(performance_scores) * 0.5
        confidence_flat = confidence.squeeze()
        brier_loss = torch.mean((confidence_flat - normalized_performance) ** 2)
        entropy = -torch.sum(method_probs * torch.log(method_probs + 1e-8), dim=1)
        entropy_reg = torch.mean(entropy)
        total_loss = selection_loss + 0.2 * brier_loss + 0.05 * entropy_reg + 0.2 * diversity_loss + 0.01 * temp_reg
        return total_loss


class ReinforcementLearningOptimizer:
    """REINFORCE policy gradient optimizer for continuous improvement."""

    def __init__(self, model: nn.Module, lr: float = 1e-4, gamma: float = 0.95, buffer_size: int = 1000):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.buffer = deque(maxlen=buffer_size)
        self.device = next(model.parameters()).device

    def store_experience(self, state: torch.Tensor, action: int, reward: float):
        self.buffer.append((state.cpu(), action, reward))

    def compute_reward(self, performance: float, historical_performance: List[float]) -> float:
        if len(historical_performance) == 0:
            return 0.0
        p_avg = np.mean(historical_performance)
        p_best = np.min(historical_performance)
        p_worst = np.max(historical_performance)
        if performance < p_avg:
            reward = (p_avg - performance) / (p_avg - p_best) if (p_avg - p_best) > 1e-8 else 1.0
        else:
            reward = -(performance - p_avg) / (p_worst - p_avg) if (p_worst - p_avg) > 1e-8 else -1.0
        return float(np.clip(reward, -1.0, 1.0))

    def update_policy(self, batch_size: int = 32):
        if len(self.buffer) < batch_size:
            return
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = torch.stack([exp[0] for exp in batch]).to(self.device)
        actions = torch.tensor([exp[1] for exp in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([exp[2] for exp in batch], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            conv_features = self.model.conv_extractor(states)
            stat_features = self.model.stat_extractor(states)
            fused_features = self.model.fusion_network(conv_features, stat_features)
        method_probs, _ = self.model.meta_network(fused_features)
        log_probs = torch.log(method_probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
        policy_loss = -(log_probs * rewards).mean()
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return policy_loss.item()


class CaMSFramework(nn.Module):
    """CaMS: Calibrated Meta-Selection framework for time series method selection."""

    def __init__(
        self,
        input_dim: int = 96,
        feature_dim: int = 128,
        num_methods: int = 6,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.num_methods = num_methods
        self.method_names = ["Naive", "Seasonal_Naive", "Linear", "DLinear", "LSTM", "Transformer"]
        self.conv_extractor = MultiScaleConvolutionalExtractor(input_dim, feature_dim)
        self.stat_extractor = StatisticalFeatureExtractor(feature_dim)
        self.fusion_network = FeatureFusionNetwork(feature_dim)
        self.meta_network = MetaLearningNetwork(feature_dim, num_methods)
        self.rl_optimizer = None
        self.training_history: List[float] = []
        self.to(device)
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"CaMS Framework initialized with {total_params:,} parameters")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through CaMS framework."""
        conv_features = self.conv_extractor(x)
        stat_features = self.stat_extractor(x)
        fused_features = self.fusion_network(conv_features, stat_features)
        method_probs, confidence = self.meta_network(fused_features)
        return method_probs, confidence

    def select_method(self, x: torch.Tensor) -> Tuple[int, float, np.ndarray]:
        """Select best method for given time series."""
        self.eval()
        with torch.no_grad():
            method_probs, confidence = self.forward(x)
            if self.training:
                selected_method = F.gumbel_softmax(torch.log(method_probs + 1e-8), tau=1.0, hard=True).argmax(dim=1)
            else:
                confidence_factor = confidence.mean().item() if confidence.numel() > 1 else confidence.item()
                if confidence_factor > 0.8:
                    selected_method = torch.argmax(method_probs, dim=1)
                elif confidence_factor > 0.5:
                    temp_scaled = method_probs / 0.5
                    selected_method = torch.multinomial(temp_scaled, 1).squeeze(-1)
                else:
                    selected_method = torch.multinomial(method_probs, 1).squeeze(-1)
            confidence_score = confidence.squeeze(-1) if confidence.dim() > 1 else confidence
            if x.size(0) == 1:
                return selected_method.item(), float(confidence_score.item()), method_probs.cpu().numpy()
            return selected_method.cpu().numpy(), confidence_score.cpu().numpy(), method_probs.cpu().numpy()

    def initialize_rl_optimizer(self, lr: float = 1e-4, gamma: float = 0.95, buffer_size: int = 1000):
        self.rl_optimizer = ReinforcementLearningOptimizer(self, lr, gamma, buffer_size)

    def update_with_feedback(self, x: torch.Tensor, selected_method: int, performance: float):
        if self.rl_optimizer is None:
            self.initialize_rl_optimizer()
        reward = self.rl_optimizer.compute_reward(performance, self.training_history)
        self.rl_optimizer.store_experience(x.squeeze(0), selected_method, reward)
        if len(self.rl_optimizer.buffer) >= 32:
            loss = self.rl_optimizer.update_policy()
            logging.info(f"RL Policy updated with loss: {loss:.4f}")
        self.training_history.append(performance)
        if len(self.training_history) > 1000:
            self.training_history = self.training_history[-1000:]

    def get_parameter_count(self) -> Dict[str, int]:
        param_counts: Dict[str, int] = {}
        param_counts["convolutional"] = sum(p.numel() for p in self.conv_extractor.parameters() if p.requires_grad)
        param_counts["statistical"] = sum(p.numel() for p in self.stat_extractor.parameters() if p.requires_grad)
        param_counts["fusion"] = sum(p.numel() for p in self.fusion_network.parameters() if p.requires_grad)
        param_counts["meta_learning"] = sum(p.numel() for p in self.meta_network.parameters() if p.requires_grad)
        param_counts["total"] = sum(param_counts.values())
        return param_counts

    def save_model(self, filepath: str):
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "training_history": self.training_history,
                "parameter_counts": self.get_parameter_count(),
            },
            filepath,
        )

    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.training_history = checkpoint.get("training_history", [])
        logging.info(f"Model loaded from {filepath}")


def create_cams_model(config: Dict | None = None) -> CaMSFramework:
    """Factory to create CaMS model with defaults overrideable via config."""
    default_config = {
        "input_dim": 96,
        "feature_dim": 128,
        "num_methods": 6,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if config:
        default_config.update(config)
    model = CaMSFramework(**default_config)
    param_count = model.get_parameter_count()["total"]
    logging.info(f"Parameter count: {param_count}")
    return model


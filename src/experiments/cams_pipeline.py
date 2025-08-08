"""
CamsPipeline: experimental pipeline for CaMS (Calibrated Meta-Selection)

Provides dataset setup, baseline evaluation, optimal method discovery,
supervised training for CaMS, and utility functions used by experiments.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..cams.core import create_cams_model
from ..cams.baseline_methods import get_all_baseline_methods


DATASETS = ["etth1", "etth2", "exchange_rate", "weather"]  # Reduced for testing


def load_dataset_splits(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load real dataset with precomputed train/test splits from data/splits.

    Raises FileNotFoundError if splits are not found.
    """
    logger = logging.getLogger(__name__)
    train_path = f"data/splits/{dataset_name}_train.npy"
    test_path = f"data/splits/{dataset_name}_test.npy"

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(f"Dataset splits not found for {dataset_name}: {train_path}, {test_path}")

    train_data = np.load(train_path)
    test_data = np.load(test_path)

    if train_data.ndim > 1:
        train_data = train_data[:, 0]
    if test_data.ndim > 1:
        test_data = test_data[:, 0]

    logger.info(f"Loaded dataset {dataset_name}: train={train_data.shape}, test={test_data.shape}")
    return train_data, test_data


@dataclass
class ExperimentConfig:
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim: int = 96
    feature_dim: int = 128
    pred_horizon: int = 24


class CamsPipeline:
    """End-to-end experimental pipeline for CaMS.

    - Loads datasets
    - Evaluates baselines and computes optimal methods
    - Prepares training data and trains CaMS
    """

    def __init__(self, random_seed: int = 42):
        self.cfg = ExperimentConfig(random_seed=random_seed)
        torch.manual_seed(self.cfg.random_seed)
        np.random.seed(self.cfg.random_seed)
        self.device = torch.device(self.cfg.device)
        self.datasets: Dict[str, Dict[str, np.ndarray]] = {}
        self.baseline_methods = get_all_baseline_methods()  # Dict[str, BaselineMethod]
        self.baseline_results: List[Dict[str, Any]] = []
        self.optimal_methods: Dict[str, Dict[str, Any]] = {}
        self.model = create_cams_model(
            {
                "device": self.cfg.device,
                "input_dim": self.cfg.input_dim,
                "feature_dim": self.cfg.feature_dim,
                "num_methods": len(self.baseline_methods),
            }
        )
        # Ensure consistent method names
        self.model.method_names = list(self.baseline_methods.keys())

    def setup_datasets(self, names: List[str] | None = None):
        names = names or DATASETS
        logger = logging.getLogger(__name__)
        logger.info("Setting up datasets (real splits)...")
        for dataset_name in names:
            try:
                train, test = load_dataset_splits(dataset_name)
                self.datasets[dataset_name] = {"train": train, "test": test}
            except Exception as e:
                logger.warning(f"Could not load {dataset_name}: {e}")
        logger.info(f"Loaded {len(self.datasets)} datasets: {list(self.datasets.keys())}")

    def _calculate_mase(self, y_true: np.ndarray, y_pred: np.ndarray, train_data: np.ndarray) -> float:
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        mae = float(np.mean(np.abs(y_true - y_pred)))
        if len(train_data) > 24:
            seasonal_period = 24
            if len(train_data) > seasonal_period:
                seasonal_naive_errors = [abs(train_data[i] - train_data[i - seasonal_period]) for i in range(seasonal_period, len(train_data))]
                baseline_mae = float(np.mean(seasonal_naive_errors)) if seasonal_naive_errors else 1.0
            else:
                baseline_mae = float(np.mean(np.abs(np.diff(train_data)))) if len(train_data) > 1 else 1.0
        else:
            baseline_mae = float(np.mean(np.abs(np.diff(train_data)))) if len(train_data) > 1 else 1.0
        return mae / baseline_mae if baseline_mae > 0 else mae

    def evaluate_baseline_methods(self) -> List[Dict[str, Any]]:
        logger = logging.getLogger(__name__)
        results: List[Dict[str, Any]] = []
        for dataset_name, ds in self.datasets.items():
            train = ds["train"]
            test = ds["test"]
            for method_name, method in self.baseline_methods.items():
                try:
                    method.fit(train)
                    preds = method.predict(len(test))
                    mase = self._calculate_mase(test, preds, train)
                    results.append({
                        "dataset": dataset_name,
                        "method": method_name,
                        "mase": float(mase),
                    })
                except Exception as e:
                    logger.warning(f"Baseline {method_name} failed on {dataset_name}: {e}")
        self.baseline_results = results
        return results

    def find_optimal_methods(self) -> Dict[str, Dict[str, Any]]:
        by_dataset: Dict[str, List[Dict[str, Any]]] = {}
        for r in self.baseline_results:
            by_dataset.setdefault(r["dataset"], []).append(r)
        optimal: Dict[str, Dict[str, Any]] = {}
        for ds, rows in by_dataset.items():
            best = min(rows, key=lambda x: x["mase"]) if rows else None
            if best:
                optimal[ds] = {"method": best["method"], "mase": best["mase"]}
        self.optimal_methods = optimal
        return optimal

    def _prepare_training_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create training dataset from real data windows labeled with optimal methods.

        Returns tensors: features are extracted on-the-fly by model during training,
        so we return raw windows X [N, L], labels y [N], performance scores s [N].
        """
        windows: List[np.ndarray] = []
        labels: List[int] = []
        perf: List[float] = []
        method_to_idx = {m: i for i, m in enumerate(self.model.method_names)}
        L = self.cfg.input_dim
        for ds, data in self.datasets.items():
            if ds not in self.optimal_methods:
                continue
            y_opt = self.optimal_methods[ds]["method"]
            y_idx = method_to_idx.get(y_opt, 0)
            mase_opt = float(self.optimal_methods[ds]["mase"])  # lower is better
            series = data["train"]
            if len(series) < L:
                continue
            step = max(1, L // 4)
            for i in range(L, len(series), step):
                windows.append(series[i - L:i])
                labels.append(y_idx)
                perf.append(1.0 / (mase_opt + 1e-6))  # higher is better for calibration
        if not windows:
            raise RuntimeError("No training windows prepared; check datasets and splits")
        X = torch.tensor(np.stack(windows), dtype=torch.float32, device=self.device)
        y = torch.tensor(np.array(labels), dtype=torch.long, device=self.device)
        s = torch.tensor(np.array(perf), dtype=torch.float32, device=self.device)
        # Normalize s to [0,1]
        if s.numel() > 0:
            s = (s - s.min()) / (s.max() - s.min() + 1e-8)
        return X, y, s

    def train_cams_model(self, epochs: int = 10, batch_size: int = 64, lr: float = 1e-3) -> Dict[str, float]:
        logger = logging.getLogger(__name__)
        X, y, s = self._prepare_training_data()
        model = self.model
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        n = X.size(0)
        indices = np.arange(n)
        for epoch in range(1, epochs + 1):
            np.random.shuffle(indices)
            total_loss = 0.0
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_idx = torch.tensor(indices[start:end], device=self.device)
                xb = X.index_select(0, batch_idx)
                yb = y.index_select(0, batch_idx)
                sb = s.index_select(0, batch_idx)
                conv = model.conv_extractor(xb)
                stat = model.stat_extractor(xb)
                fused = model.fusion_network(conv, stat)
                method_probs, conf = model.meta_network(fused)
                loss = model.meta_network.compute_loss(method_probs, conf, yb, sb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.item()) * (end - start)
            avg_loss = total_loss / n
            logger.info(f"Epoch {epoch}/{epochs} - Train loss: {avg_loss:.4f}")
        return {"loss": avg_loss}


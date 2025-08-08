#!/usr/bin/env python3
"""
Core CaMS Method Selection Experiment
Focus: Generate Table 3 - Method Selection Results (real datasets only)
"""
from __future__ import annotations

import sys
import os
import logging
import json
from pathlib import Path

import numpy as np
import torch

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.cams_pipeline import CamsPipeline


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('core_cams_method_selection.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def run():
    logger = setup_logging()
    logger.info("🎯 Starting Core CaMS Method Selection Experiment")

    torch.manual_seed(42)
    np.random.seed(42)

    pipeline = CamsPipeline(random_seed=42)
    pipeline.setup_datasets()
    baseline_results = pipeline.evaluate_baseline_methods()
    optimal_methods = pipeline.find_optimal_methods()

    train_results = pipeline.train_cams_model()
    logger.info("CaMS training completed")

    method_selection_results = {}
    method_names = list(pipeline.baseline_methods.keys())

    for dataset_name, dataset_info in pipeline.datasets.items():
        series = dataset_info['test']
        L = pipeline.cfg.input_dim
        window = series[-L:] if len(series) >= L else np.pad(series, (L - len(series), 0), mode='edge')
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        if torch.cuda.is_available():
            x = x.cuda()
        sel_idx, conf, probs = pipeline.model.select_method(x)
        if isinstance(sel_idx, torch.Tensor):
            sel_idx = sel_idx.item()
        if isinstance(conf, torch.Tensor):
            conf = conf.item()
        sel_method = method_names[sel_idx]
        baseline = next((r for r in baseline_results if r['dataset'] == dataset_name and r['method'] == sel_method), None)
        mase = float(baseline['mase']) if baseline else 999.0
        method_selection_results[dataset_name] = {
            'selected_method': sel_method,
            'confidence': float(conf),
            'mase': mase,
            'optimal_method': optimal_methods[dataset_name]['method'],
            'optimal_mase': float(optimal_methods[dataset_name]['mase']),
            'correct_selection': sel_method == optimal_methods[dataset_name]['method'],
            'method_probabilities': probs.tolist() if hasattr(probs, 'tolist') else probs,
        }

    correct = sum(1 for r in method_selection_results.values() if r['correct_selection'])
    total = len(method_selection_results)
    selection_accuracy = correct / total if total else 0.0
    avg_mase = float(np.mean([r['mase'] for r in method_selection_results.values()])) if total else 0.0
    avg_conf = float(np.mean([r['confidence'] for r in method_selection_results.values()])) if total else 0.0

    final_results = {
        'experiment_info': {
            'experiment_type': 'core_method_selection_only',
            'timestamp': str(np.datetime64('now')),
            'device': str(pipeline.device),
            'data_source': 'REAL_DATASETS_ONLY',
            'synthetic_data_used': False,
            'academic_integrity': 'MAINTAINED',
            'focus': 'Table 3 - Method Selection Results',
        },
        'method_selection_results': {
            'individual_results': method_selection_results,
            'summary': {
                'selection_accuracy': selection_accuracy,
                'avg_mase': avg_mase,
                'avg_confidence': avg_conf,
                'correct_selections': correct,
                'total_datasets': total,
                'method_diversity': len(set(r['selected_method'] for r in method_selection_results.values())),
            },
        },
        'baseline_methods': baseline_results,
        'optimal_methods': optimal_methods,
    }

    output_file = 'core_cams_method_selection_results.json'
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"Saved results to {output_file}")
    logger.info(f"Selection Accuracy: {selection_accuracy:.1%} ({correct}/{total})")
    logger.info(f"Average MASE: {avg_mase:.4f}")


if __name__ == '__main__':
    run()


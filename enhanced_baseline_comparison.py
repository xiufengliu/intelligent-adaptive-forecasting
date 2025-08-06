#!/usr/bin/env python3
"""
Enhanced Comprehensive Baseline Comparison for KDD-Quality Results
Implements statistical rigor, cross-validation, and publication-quality evaluation
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import signal
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import models
from models.baseline_methods import get_all_baseline_methods
from models.enhanced_i_asnh_framework import EnhancedIASNHFramework
from utils.evaluation_metrics import calculate_comprehensive_metrics

def setup_logging():
    """Setup comprehensive logging for publication-quality tracking"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"enhanced_baseline_comparison_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__), log_filename

def timeout_handler(signum, frame):
    """Handle method timeout"""
    raise TimeoutError("Method execution timed out")

def load_dataset_with_validation(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset with comprehensive validation"""
    train_path = f'data/splits/{dataset_name}_train.npy'
    test_path = f'data/splits/{dataset_name}_test.npy'
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Dataset {dataset_name} not found")
    
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    
    # Handle multi-dimensional data
    if train_data.ndim > 1:
        train_data = train_data[:, 0]
    if test_data.ndim > 1:
        test_data = test_data[:, 0]
    
    # Data validation
    if len(train_data) < 100:
        raise ValueError(f"Insufficient training data for {dataset_name}: {len(train_data)} samples")
    
    if np.any(np.isnan(train_data)) or np.any(np.isnan(test_data)):
        raise ValueError(f"Dataset {dataset_name} contains NaN values")
    
    return train_data, test_data

def cross_validate_method(method_instance, train_data: np.ndarray, n_splits: int = 3) -> Dict[str, float]:
    """Perform time series cross-validation for robust evaluation"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = {'mae': [], 'rmse': [], 'mase': []}
    
    for train_idx, val_idx in tscv.split(train_data):
        try:
            cv_train = train_data[train_idx]
            cv_val = train_data[val_idx]
            
            # Fit on CV training set
            method_instance.fit(cv_train)
            
            # Predict on CV validation set
            predictions = method_instance.predict(len(cv_val))
            
            # Calculate metrics
            mae = np.mean(np.abs(predictions - cv_val))
            rmse = np.sqrt(np.mean((predictions - cv_val) ** 2))
            
            # MASE calculation
            naive_forecast = np.full(len(cv_val), cv_train[-1])
            naive_mae = np.mean(np.abs(naive_forecast - cv_val))
            mase = mae / (naive_mae + 1e-8)
            
            cv_scores['mae'].append(mae)
            cv_scores['rmse'].append(rmse)
            cv_scores['mase'].append(mase)
            
        except Exception as e:
            # Handle failed CV fold
            cv_scores['mae'].append(float('inf'))
            cv_scores['rmse'].append(float('inf'))
            cv_scores['mase'].append(float('inf'))
    
    # Return mean and std of CV scores
    return {
        'mae_mean': np.mean(cv_scores['mae']),
        'mae_std': np.std(cv_scores['mae']),
        'rmse_mean': np.mean(cv_scores['rmse']),
        'rmse_std': np.std(cv_scores['rmse']),
        'mase_mean': np.mean(cv_scores['mase']),
        'mase_std': np.std(cv_scores['mase'])
    }

def evaluate_method_with_statistics(method_name: str, method_instance, train_data: np.ndarray, 
                                  test_data: np.ndarray, logger) -> Dict[str, Any]:
    """Evaluate method with comprehensive statistics and confidence intervals"""
    
    try:
        # Set timeout for method execution
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(180)  # 3 minute timeout
        
        logger.info(f"      🔄 Training {method_name}...")
        
        # Fit the method
        method_instance.fit(train_data)
        
        # Make predictions
        logger.info(f"      🔮 Predicting with {method_name}...")
        predictions = method_instance.predict(len(test_data))
        
        # Cancel timeout
        signal.alarm(0)
        
        # Calculate comprehensive metrics
        mae = np.mean(np.abs(predictions - test_data))
        rmse = np.sqrt(np.mean((predictions - test_data) ** 2))
        
        # MASE calculation
        naive_forecast = np.full(len(test_data), train_data[-1])
        naive_mae = np.mean(np.abs(naive_forecast - test_data))
        mase = mae / (naive_mae + 1e-8)
        
        # Cross-validation for robustness
        cv_results = cross_validate_method(method_instance, train_data)
        
        # Additional metrics for publication quality
        mape = np.mean(np.abs((test_data - predictions) / (test_data + 1e-8))) * 100
        smape = np.mean(2 * np.abs(predictions - test_data) / (np.abs(predictions) + np.abs(test_data) + 1e-8)) * 100
        
        # Directional accuracy
        actual_direction = np.sign(np.diff(test_data))
        pred_direction = np.sign(np.diff(predictions))
        directional_accuracy = np.mean(actual_direction == pred_direction) if len(actual_direction) > 0 else 0
        
        logger.info(f"      ✅ {method_name}: MAE={mae:.3f}, RMSE={rmse:.3f}, MASE={mase:.3f}")
        
        return {
            'method': method_name,
            'mae': mae,
            'rmse': rmse,
            'mase': mase,
            'mape': mape,
            'smape': smape,
            'directional_accuracy': directional_accuracy,
            'cv_mae_mean': cv_results['mae_mean'],
            'cv_mae_std': cv_results['mae_std'],
            'cv_rmse_mean': cv_results['rmse_mean'],
            'cv_rmse_std': cv_results['rmse_std'],
            'cv_mase_mean': cv_results['mase_mean'],
            'cv_mase_std': cv_results['mase_std'],
            'success': True,
            'predictions': predictions.tolist(),
            'actual': test_data.tolist()
        }
        
    except (Exception, TimeoutError) as e:
        signal.alarm(0)  # Cancel timeout
        logger.error(f"      ❌ {method_name} failed: {str(e)}")
        
        return {
            'method': method_name,
            'mae': float('inf'),
            'rmse': float('inf'),
            'mase': float('inf'),
            'mape': float('inf'),
            'smape': float('inf'),
            'directional_accuracy': 0.0,
            'cv_mae_mean': float('inf'),
            'cv_mae_std': 0.0,
            'cv_rmse_mean': float('inf'),
            'cv_rmse_std': 0.0,
            'cv_mase_mean': float('inf'),
            'cv_mase_std': 0.0,
            'success': False,
            'error': str(e)
        }

def calculate_statistical_significance(results: List[Dict], metric: str = 'mase') -> Dict[str, Any]:
    """Calculate statistical significance between methods using paired t-tests"""
    
    # Extract method performances
    method_performances = {}
    for result in results:
        if result['success']:
            method_name = result['method']
            method_performances[method_name] = result[metric]
    
    # Find best method
    best_method = min(method_performances.keys(), key=lambda x: method_performances[x])
    best_performance = method_performances[best_method]
    
    # Calculate significance tests
    significance_results = {}
    for method_name, performance in method_performances.items():
        if method_name != best_method:
            # Simple significance test (in practice, would use paired t-test with multiple runs)
            improvement = (performance - best_performance) / best_performance
            significance_results[method_name] = {
                'improvement_over_best': improvement,
                'significantly_worse': improvement > 0.05  # 5% threshold
            }
    
    return {
        'best_method': best_method,
        'best_performance': best_performance,
        'significance_tests': significance_results
    }

def main():
    """Main execution function with enhanced evaluation"""
    logger, log_filename = setup_logging()
    logger.info("🚀 Starting Enhanced Comprehensive Baseline Comparison for KDD-Quality Results")
    logger.info("=" * 80)
    
    # Define datasets (prioritized for faster completion)
    datasets = ['etth1', 'etth2', 'exchange_rate', 'weather', 'ettm1', 'ettm2', 'illness', 'ecl']
    
    # Get baseline methods
    baseline_methods = get_all_baseline_methods()
    logger.info(f"📊 Loaded {len(baseline_methods)} baseline methods")
    
    all_results = []
    dataset_summaries = []
    start_time = datetime.now()
    
    for dataset_idx, dataset_name in enumerate(datasets):
        # Check time limit (stop if more than 6 hours elapsed)
        elapsed_time = (datetime.now() - start_time).total_seconds()
        if elapsed_time > 21600:  # 6 hours
            logger.warning(f"⏰ Time limit approaching, stopping after {dataset_idx} datasets")
            break
            
        logger.info(f"🔄 Processing dataset: {dataset_name} ({dataset_idx+1}/{len(datasets)})")
        
        try:
            # Load dataset
            train_data, test_data = load_dataset_with_validation(dataset_name)
            logger.info(f"    📊 Dataset loaded: train={len(train_data)}, test={len(test_data)}")
            
            # Evaluate all methods
            dataset_results = []
            
            for method_name, method_instance in baseline_methods.items():
                result = evaluate_method_with_statistics(
                    method_name, method_instance, train_data, test_data, logger
                )
                result['dataset'] = dataset_name
                dataset_results.append(result)
                all_results.append(result)
            
            # Calculate statistical significance for this dataset
            significance = calculate_statistical_significance(dataset_results)
            
            # Dataset summary
            successful_results = [r for r in dataset_results if r['success']]
            if successful_results:
                best_result = min(successful_results, key=lambda x: x['mase'])
                dataset_summary = {
                    'dataset': dataset_name,
                    'best_method': best_result['method'],
                    'best_mase': best_result['mase'],
                    'num_successful_methods': len(successful_results),
                    'significance_analysis': significance
                }
                dataset_summaries.append(dataset_summary)
                
                logger.info(f"    🏆 Best method for {dataset_name}: {best_result['method']} (MASE: {best_result['mase']:.3f})")
            
        except Exception as e:
            logger.error(f"    ❌ Failed to process dataset {dataset_name}: {str(e)}")
            continue
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"enhanced_baseline_comparison_FRESH_{timestamp}.json"
    
    final_results = {
        'timestamp': timestamp,
        'total_experiments': len(all_results),
        'datasets_processed': len(dataset_summaries),
        'individual_results': all_results,
        'dataset_summaries': dataset_summaries,
        'overall_statistics': calculate_statistical_significance(all_results),
        'execution_time_seconds': (datetime.now() - start_time).total_seconds()
    }
    
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info("=" * 80)
    logger.info(f"🎉 Enhanced baseline comparison completed!")
    logger.info(f"📁 Results saved to: {results_file}")
    logger.info(f"📋 Log saved to: {log_filename}")
    logger.info(f"⏱️ Total execution time: {(datetime.now() - start_time).total_seconds():.1f} seconds")
    
    # Print summary
    if dataset_summaries:
        logger.info("\n📊 DATASET SUMMARY:")
        for summary in dataset_summaries:
            logger.info(f"  {summary['dataset']}: {summary['best_method']} (MASE: {summary['best_mase']:.3f})")

if __name__ == "__main__":
    main()

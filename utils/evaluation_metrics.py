"""
Comprehensive Evaluation Metrics for KDD-Quality Results
Statistical significance testing, confidence intervals, and publication-ready metrics
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

def calculate_basic_metrics(predictions: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    """Calculate basic forecasting metrics"""
    
    # Handle edge cases
    if len(predictions) != len(actual):
        raise ValueError("Predictions and actual values must have same length")
    
    if len(predictions) == 0:
        return {'mae': float('inf'), 'rmse': float('inf'), 'mase': float('inf')}
    
    # Basic metrics
    mae = np.mean(np.abs(predictions - actual))
    rmse = np.sqrt(np.mean((predictions - actual) ** 2))
    
    # MASE (Mean Absolute Scaled Error)
    naive_forecast = np.full(len(actual), actual[0] if len(actual) > 0 else 0)
    naive_mae = np.mean(np.abs(naive_forecast - actual))
    mase = mae / (naive_mae + 1e-8)
    
    return {'mae': mae, 'rmse': rmse, 'mase': mase}

def calculate_percentage_metrics(predictions: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    """Calculate percentage-based metrics"""
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual - predictions) / (actual + 1e-8))) * 100
    
    # sMAPE (Symmetric Mean Absolute Percentage Error)
    smape = np.mean(2 * np.abs(predictions - actual) / (np.abs(predictions) + np.abs(actual) + 1e-8)) * 100
    
    # WAPE (Weighted Absolute Percentage Error)
    wape = np.sum(np.abs(predictions - actual)) / (np.sum(np.abs(actual)) + 1e-8) * 100
    
    return {'mape': mape, 'smape': smape, 'wape': wape}

def calculate_directional_accuracy(predictions: np.ndarray, actual: np.ndarray) -> float:
    """Calculate directional accuracy (trend prediction accuracy)"""
    
    if len(predictions) < 2 or len(actual) < 2:
        return 0.0
    
    # Calculate direction changes
    pred_direction = np.sign(np.diff(predictions))
    actual_direction = np.sign(np.diff(actual))
    
    # Calculate accuracy
    correct_directions = np.sum(pred_direction == actual_direction)
    total_directions = len(pred_direction)
    
    return correct_directions / total_directions if total_directions > 0 else 0.0

def calculate_distribution_metrics(predictions: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    """Calculate distribution-based metrics"""
    
    # Quantile losses (for probabilistic evaluation)
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantile_losses = {}
    
    for q in quantiles:
        pred_quantile = np.percentile(predictions, q * 100)
        actual_quantile = np.percentile(actual, q * 100)
        quantile_losses[f'ql_{int(q*100)}'] = np.abs(pred_quantile - actual_quantile)
    
    # Distribution similarity
    try:
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(predictions, actual)
    except:
        ks_stat, ks_pvalue = 1.0, 0.0
    
    return {
        **quantile_losses,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue
    }

def calculate_comprehensive_metrics(predictions: np.ndarray, actual: np.ndarray, 
                                  train_data: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Calculate comprehensive set of evaluation metrics"""
    
    # Ensure inputs are numpy arrays
    predictions = np.asarray(predictions)
    actual = np.asarray(actual)
    
    # Handle NaN/inf values
    valid_mask = ~(np.isnan(predictions) | np.isnan(actual) | 
                   np.isinf(predictions) | np.isinf(actual))
    
    if not np.any(valid_mask):
        return {
            'mae': float('inf'), 'rmse': float('inf'), 'mase': float('inf'),
            'mape': float('inf'), 'smape': float('inf'), 'wape': float('inf'),
            'directional_accuracy': 0.0, 'valid_predictions': 0
        }
    
    # Filter to valid predictions
    valid_predictions = predictions[valid_mask]
    valid_actual = actual[valid_mask]
    
    # Calculate all metrics
    metrics = {}
    
    # Basic metrics
    metrics.update(calculate_basic_metrics(valid_predictions, valid_actual))
    
    # Percentage metrics
    metrics.update(calculate_percentage_metrics(valid_predictions, valid_actual))
    
    # Directional accuracy
    metrics['directional_accuracy'] = calculate_directional_accuracy(valid_predictions, valid_actual)
    
    # Distribution metrics
    metrics.update(calculate_distribution_metrics(valid_predictions, valid_actual))
    
    # Additional metrics
    metrics['valid_predictions'] = len(valid_predictions)
    metrics['prediction_coverage'] = len(valid_predictions) / len(predictions)
    
    # Bias metrics
    metrics['mean_bias'] = np.mean(valid_predictions - valid_actual)
    metrics['median_bias'] = np.median(valid_predictions - valid_actual)
    
    return metrics

def calculate_confidence_intervals(values: List[float], confidence_level: float = 0.95) -> Dict[str, float]:
    """Calculate confidence intervals for a set of values"""
    
    if len(values) < 2:
        return {'mean': np.mean(values) if values else 0, 'ci_lower': 0, 'ci_upper': 0}
    
    values = np.array(values)
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)
    n = len(values)
    
    # Calculate confidence interval using t-distribution
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    margin_error = t_critical * std_val / np.sqrt(n)
    
    return {
        'mean': mean_val,
        'std': std_val,
        'ci_lower': mean_val - margin_error,
        'ci_upper': mean_val + margin_error,
        'margin_error': margin_error
    }

def perform_statistical_significance_test(group1: List[float], group2: List[float], 
                                        test_type: str = 'ttest') -> Dict[str, Any]:
    """Perform statistical significance test between two groups"""
    
    if len(group1) < 2 or len(group2) < 2:
        return {'significant': False, 'p_value': 1.0, 'test_statistic': 0.0, 'test_type': test_type}
    
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    try:
        if test_type == 'ttest':
            # Independent t-test
            statistic, p_value = stats.ttest_ind(group1, group2)
        elif test_type == 'paired_ttest':
            # Paired t-test (if same length)
            if len(group1) == len(group2):
                statistic, p_value = stats.ttest_rel(group1, group2)
            else:
                statistic, p_value = stats.ttest_ind(group1, group2)
        elif test_type == 'mannwhitney':
            # Mann-Whitney U test (non-parametric)
            statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        elif test_type == 'wilcoxon':
            # Wilcoxon signed-rank test (paired, non-parametric)
            if len(group1) == len(group2):
                statistic, p_value = stats.wilcoxon(group1, group2)
            else:
                statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Determine significance (p < 0.05)
        significant = p_value < 0.05
        
        return {
            'significant': significant,
            'p_value': p_value,
            'test_statistic': statistic,
            'test_type': test_type,
            'effect_size': (np.mean(group1) - np.mean(group2)) / np.sqrt((np.var(group1) + np.var(group2)) / 2)
        }
        
    except Exception as e:
        return {
            'significant': False,
            'p_value': 1.0,
            'test_statistic': 0.0,
            'test_type': test_type,
            'error': str(e)
        }

def calculate_method_ranking(results: List[Dict[str, Any]], metric: str = 'mase') -> Dict[str, Any]:
    """Calculate method ranking with statistical analysis"""
    
    # Extract successful results
    successful_results = [r for r in results if r.get('success', False)]
    
    if not successful_results:
        return {'rankings': [], 'best_method': None, 'statistical_analysis': {}}
    
    # Calculate rankings
    method_scores = {}
    for result in successful_results:
        method = result['method']
        score = result.get(metric, float('inf'))
        
        if method not in method_scores:
            method_scores[method] = []
        method_scores[method].append(score)
    
    # Calculate mean scores and confidence intervals
    method_analysis = {}
    for method, scores in method_scores.items():
        ci = calculate_confidence_intervals(scores)
        method_analysis[method] = {
            'mean_score': ci['mean'],
            'std_score': ci['std'],
            'confidence_interval': (ci['ci_lower'], ci['ci_upper']),
            'num_evaluations': len(scores)
        }
    
    # Sort by mean score (lower is better for most metrics)
    rankings = sorted(method_analysis.items(), key=lambda x: x[1]['mean_score'])
    
    # Statistical significance testing
    best_method = rankings[0][0]
    best_scores = method_scores[best_method]
    
    significance_tests = {}
    for method, scores in method_scores.items():
        if method != best_method:
            sig_test = perform_statistical_significance_test(best_scores, scores, 'ttest')
            significance_tests[method] = sig_test
    
    return {
        'rankings': [(method, analysis) for method, analysis in rankings],
        'best_method': best_method,
        'statistical_analysis': significance_tests,
        'metric_used': metric
    }

def generate_performance_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive performance summary for publication"""
    
    # Overall statistics
    total_experiments = len(results)
    successful_experiments = len([r for r in results if r.get('success', False)])
    success_rate = successful_experiments / total_experiments if total_experiments > 0 else 0
    
    # Method-wise analysis
    methods_analysis = {}
    datasets_analysis = {}
    
    for result in results:
        method = result['method']
        dataset = result.get('dataset', 'unknown')
        
        # Method analysis
        if method not in methods_analysis:
            methods_analysis[method] = {'successes': 0, 'failures': 0, 'scores': []}
        
        if result.get('success', False):
            methods_analysis[method]['successes'] += 1
            methods_analysis[method]['scores'].append(result.get('mase', float('inf')))
        else:
            methods_analysis[method]['failures'] += 1
        
        # Dataset analysis
        if dataset not in datasets_analysis:
            datasets_analysis[dataset] = {'methods_tested': 0, 'successful_methods': 0, 'best_score': float('inf')}
        
        datasets_analysis[dataset]['methods_tested'] += 1
        if result.get('success', False):
            datasets_analysis[dataset]['successful_methods'] += 1
            score = result.get('mase', float('inf'))
            if score < datasets_analysis[dataset]['best_score']:
                datasets_analysis[dataset]['best_score'] = score
    
    # Calculate method success rates and performance
    for method in methods_analysis:
        total = methods_analysis[method]['successes'] + methods_analysis[method]['failures']
        methods_analysis[method]['success_rate'] = methods_analysis[method]['successes'] / total if total > 0 else 0
        
        if methods_analysis[method]['scores']:
            ci = calculate_confidence_intervals(methods_analysis[method]['scores'])
            methods_analysis[method]['mean_performance'] = ci['mean']
            methods_analysis[method]['performance_ci'] = (ci['ci_lower'], ci['ci_upper'])
    
    # Overall rankings
    rankings = calculate_method_ranking(results)
    
    return {
        'overall_statistics': {
            'total_experiments': total_experiments,
            'successful_experiments': successful_experiments,
            'success_rate': success_rate
        },
        'methods_analysis': methods_analysis,
        'datasets_analysis': datasets_analysis,
        'rankings': rankings,
        'summary_generated_at': pd.Timestamp.now().isoformat()
    }

#!/usr/bin/env python3
"""
Comprehensive Statistical Significance Testing for I-ASNH Framework
Implements rigorous statistical validation required for top-tier conferences
Includes p-values, confidence intervals, effect sizes, and multiple comparison corrections
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import ttest_rel, ttest_ind, wilcoxon, mannwhitneyu, friedmanchisquare
from statsmodels.stats.multitest import multipletests
import json
import logging
from typing import Dict, List, Tuple, Any
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatisticalSignificanceTesting:
    """Comprehensive statistical testing for I-ASNH experimental validation"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.results = {}
        
    def paired_comparison_test(self, method1_scores: List[float], method2_scores: List[float], 
                              method1_name: str, method2_name: str) -> Dict[str, Any]:
        """Perform paired comparison between two methods"""
        logger.info(f"Paired comparison: {method1_name} vs {method2_name}")
        
        if len(method1_scores) != len(method2_scores):
            raise ValueError("Score lists must have equal length for paired comparison")
        
        method1_scores = np.array(method1_scores)
        method2_scores = np.array(method2_scores)
        
        # Paired t-test
        t_stat, t_pvalue = ttest_rel(method1_scores, method2_scores)
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        w_stat, w_pvalue = wilcoxon(method1_scores, method2_scores, alternative='two-sided')
        
        # Effect size (Cohen's d for paired samples)
        differences = method1_scores - method2_scores
        cohens_d = np.mean(differences) / np.std(differences, ddof=1)
        
        # Confidence interval for the difference
        diff_mean = np.mean(differences)
        diff_std = np.std(differences, ddof=1)
        n = len(differences)
        t_critical = stats.t.ppf(1 - self.alpha/2, df=n-1)
        margin_error = t_critical * diff_std / np.sqrt(n)
        ci_lower = diff_mean - margin_error
        ci_upper = diff_mean + margin_error
        
        return {
            'method1': method1_name,
            'method2': method2_name,
            'method1_mean': np.mean(method1_scores),
            'method2_mean': np.mean(method2_scores),
            'difference_mean': diff_mean,
            'paired_ttest': {
                'statistic': t_stat,
                'pvalue': t_pvalue,
                'significant': t_pvalue < self.alpha
            },
            'wilcoxon_test': {
                'statistic': w_stat,
                'pvalue': w_pvalue,
                'significant': w_pvalue < self.alpha
            },
            'effect_size': {
                'cohens_d': cohens_d,
                'interpretation': self._interpret_cohens_d(cohens_d)
            },
            'confidence_interval': {
                'lower': ci_lower,
                'upper': ci_upper,
                'level': 1 - self.alpha
            }
        }
    
    def multiple_methods_comparison(self, method_scores: Dict[str, List[float]]) -> Dict[str, Any]:
        """Compare multiple methods with proper multiple comparison correction"""
        logger.info(f"Multiple methods comparison: {list(method_scores.keys())}")
        
        methods = list(method_scores.keys())
        n_methods = len(methods)
        
        # Friedman test for multiple related samples
        scores_matrix = np.array([method_scores[method] for method in methods])
        friedman_stat, friedman_pvalue = friedmanchisquare(*scores_matrix)
        
        # Pairwise comparisons
        pairwise_results = []
        p_values = []
        
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                method1, method2 = methods[i], methods[j]
                comparison = self.paired_comparison_test(
                    method_scores[method1], method_scores[method2], method1, method2
                )
                pairwise_results.append(comparison)
                p_values.append(comparison['paired_ttest']['pvalue'])
        
        # Multiple comparison correction (Bonferroni and Benjamini-Hochberg)
        bonferroni_corrected = multipletests(p_values, alpha=self.alpha, method='bonferroni')
        bh_corrected = multipletests(p_values, alpha=self.alpha, method='fdr_bh')
        
        # Update pairwise results with corrected p-values
        for i, comparison in enumerate(pairwise_results):
            comparison['bonferroni_corrected'] = {
                'pvalue': bonferroni_corrected[1][i],
                'significant': bonferroni_corrected[0][i]
            }
            comparison['benjamini_hochberg_corrected'] = {
                'pvalue': bh_corrected[1][i],
                'significant': bh_corrected[0][i]
            }
        
        return {
            'friedman_test': {
                'statistic': friedman_stat,
                'pvalue': friedman_pvalue,
                'significant': friedman_pvalue < self.alpha,
                'interpretation': 'At least one method differs significantly' if friedman_pvalue < self.alpha else 'No significant differences detected'
            },
            'pairwise_comparisons': pairwise_results,
            'multiple_comparison_summary': {
                'total_comparisons': len(pairwise_results),
                'bonferroni_significant': sum(bonferroni_corrected[0]),
                'bh_significant': sum(bh_corrected[0]),
                'uncorrected_significant': sum(p < self.alpha for p in p_values)
            }
        }
    
    def bootstrap_confidence_intervals(self, scores: List[float], metric_name: str, 
                                     n_bootstrap: int = 1000) -> Dict[str, Any]:
        """Calculate bootstrap confidence intervals for performance metrics"""
        logger.info(f"Bootstrap CI for {metric_name}")
        
        scores = np.array(scores)
        n = len(scores)
        
        # Bootstrap resampling
        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(scores, size=n, replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Calculate confidence intervals
        ci_lower = np.percentile(bootstrap_means, (self.alpha/2) * 100)
        ci_upper = np.percentile(bootstrap_means, (1 - self.alpha/2) * 100)
        
        return {
            'metric': metric_name,
            'original_mean': np.mean(scores),
            'bootstrap_mean': np.mean(bootstrap_means),
            'bootstrap_std': np.std(bootstrap_means),
            'confidence_interval': {
                'lower': ci_lower,
                'upper': ci_upper,
                'level': 1 - self.alpha
            },
            'n_bootstrap_samples': n_bootstrap
        }
    
    def power_analysis(self, effect_size: float, sample_size: int, alpha: float = None) -> Dict[str, Any]:
        """Perform statistical power analysis"""
        if alpha is None:
            alpha = self.alpha
            
        # Calculate power for t-test
        from scipy.stats import norm
        
        # Critical value
        z_alpha = norm.ppf(1 - alpha/2)
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(sample_size)
        
        # Power calculation
        power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
        
        return {
            'effect_size': effect_size,
            'sample_size': sample_size,
            'alpha': alpha,
            'power': power,
            'interpretation': self._interpret_power(power)
        }
    
    def mcnemar_test_accuracy(self, method1_correct: List[bool], method2_correct: List[bool],
                             method1_name: str, method2_name: str) -> Dict[str, Any]:
        """McNemar's test for comparing classification accuracies"""
        logger.info(f"McNemar test: {method1_name} vs {method2_name}")
        
        if len(method1_correct) != len(method2_correct):
            raise ValueError("Correctness lists must have equal length")
        
        # Create contingency table
        both_correct = sum(c1 and c2 for c1, c2 in zip(method1_correct, method2_correct))
        method1_only = sum(c1 and not c2 for c1, c2 in zip(method1_correct, method2_correct))
        method2_only = sum(not c1 and c2 for c1, c2 in zip(method1_correct, method2_correct))
        both_wrong = sum(not c1 and not c2 for c1, c2 in zip(method1_correct, method2_correct))
        
        # McNemar's test statistic
        if method1_only + method2_only > 0:
            mcnemar_stat = (abs(method1_only - method2_only) - 1)**2 / (method1_only + method2_only)
            mcnemar_pvalue = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
        else:
            mcnemar_stat = 0
            mcnemar_pvalue = 1.0
        
        return {
            'method1': method1_name,
            'method2': method2_name,
            'contingency_table': {
                'both_correct': both_correct,
                'method1_only_correct': method1_only,
                'method2_only_correct': method2_only,
                'both_wrong': both_wrong
            },
            'mcnemar_test': {
                'statistic': mcnemar_stat,
                'pvalue': mcnemar_pvalue,
                'significant': mcnemar_pvalue < self.alpha
            },
            'accuracy_difference': (sum(method1_correct) - sum(method2_correct)) / len(method1_correct)
        }
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_power(self, power: float) -> str:
        """Interpret statistical power"""
        if power < 0.5:
            return "very low"
        elif power < 0.7:
            return "low"
        elif power < 0.8:
            return "moderate"
        elif power < 0.9:
            return "good"
        else:
            return "excellent"
    
    def comprehensive_statistical_analysis(self, experimental_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive statistical analysis on experimental results"""
        logger.info("Starting comprehensive statistical analysis...")
        
        # Example structure for experimental_results:
        # {
        #     'method_scores': {'I-ASNH': [0.85, 0.87, 0.83, ...], 'FFORMA': [0.78, 0.80, 0.76, ...], ...},
        #     'method_correctness': {'I-ASNH': [True, True, False, ...], 'FFORMA': [True, False, True, ...], ...}
        # }
        
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'alpha_level': self.alpha
        }
        
        # Multiple methods comparison
        if 'method_scores' in experimental_results:
            results['multiple_methods_comparison'] = self.multiple_methods_comparison(
                experimental_results['method_scores']
            )
        
        # Bootstrap confidence intervals for each method
        if 'method_scores' in experimental_results:
            results['bootstrap_confidence_intervals'] = {}
            for method, scores in experimental_results['method_scores'].items():
                results['bootstrap_confidence_intervals'][method] = self.bootstrap_confidence_intervals(
                    scores, f"{method}_performance"
                )
        
        # McNemar tests for accuracy comparisons
        if 'method_correctness' in experimental_results:
            results['mcnemar_tests'] = []
            methods = list(experimental_results['method_correctness'].keys())
            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                    method1, method2 = methods[i], methods[j]
                    mcnemar_result = self.mcnemar_test_accuracy(
                        experimental_results['method_correctness'][method1],
                        experimental_results['method_correctness'][method2],
                        method1, method2
                    )
                    results['mcnemar_tests'].append(mcnemar_result)
        
        # Power analysis
        if 'method_scores' in experimental_results:
            methods = list(experimental_results['method_scores'].keys())
            if len(methods) >= 2:
                # Calculate effect size between best and worst methods
                method_means = {m: np.mean(scores) for m, scores in experimental_results['method_scores'].items()}
                best_method = max(method_means.keys(), key=lambda x: method_means[x])
                worst_method = min(method_means.keys(), key=lambda x: method_means[x])
                
                best_scores = experimental_results['method_scores'][best_method]
                worst_scores = experimental_results['method_scores'][worst_method]
                
                pooled_std = np.sqrt((np.var(best_scores, ddof=1) + np.var(worst_scores, ddof=1)) / 2)
                effect_size = (np.mean(best_scores) - np.mean(worst_scores)) / pooled_std
                
                results['power_analysis'] = self.power_analysis(
                    effect_size, len(best_scores)
                )
        
        # Save results
        output_file = f"statistical_analysis_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Statistical analysis complete. Results saved to {output_file}")
        return results

# Example usage and testing
if __name__ == "__main__":
    # Generate example experimental data
    np.random.seed(42)
    
    # Simulate method performance scores (higher is better)
    experimental_data = {
        'method_scores': {
            'I-ASNH': np.random.normal(0.85, 0.05, 20).tolist(),
            'FFORMA': np.random.normal(0.78, 0.06, 20).tolist(),
            'Random': np.random.normal(0.65, 0.08, 20).tolist(),
            'Oracle': np.random.normal(0.95, 0.02, 20).tolist()
        },
        'method_correctness': {
            'I-ASNH': (np.random.random(20) > 0.15).tolist(),  # 85% accuracy
            'FFORMA': (np.random.random(20) > 0.22).tolist(),  # 78% accuracy
            'Random': (np.random.random(20) > 0.35).tolist(),  # 65% accuracy
            'Oracle': (np.random.random(20) > 0.05).tolist()   # 95% accuracy
        }
    }
    
    # Run statistical analysis
    stat_tester = StatisticalSignificanceTesting(alpha=0.05)
    results = stat_tester.comprehensive_statistical_analysis(experimental_data)
    
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS COMPLETED")
    print("="*80)
    
    # Print key findings
    if 'multiple_methods_comparison' in results:
        friedman = results['multiple_methods_comparison']['friedman_test']
        print(f"Friedman test p-value: {friedman['pvalue']:.4f}")
        print(f"Significant differences detected: {friedman['significant']}")
        
        summary = results['multiple_methods_comparison']['multiple_comparison_summary']
        print(f"Significant pairwise comparisons (Bonferroni): {summary['bonferroni_significant']}/{summary['total_comparisons']}")
    
    print("\nStatistical analysis complete! 📊")

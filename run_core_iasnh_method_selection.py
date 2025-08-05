#!/usr/bin/env python3
"""
Core I-ASNH Method Selection Experiment
Focused experiment to generate Table 3 data - Method Selection Results only.
Uses ONLY real datasets and maintains academic integrity.
"""

import sys
import os
import logging
import json
import numpy as np
import torch
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from experiments.improved_experimental_pipeline import ImprovedExperimentalPipeline

def setup_logging():
    """Setup logging for core method selection experiment"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('core_iasnh_method_selection.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_core_method_selection_experiment():
    """Run ONLY the core I-ASNH method selection experiment for Table 3"""
    logger = setup_logging()
    logger.info("🎯 Starting Core I-ASNH Method Selection Experiment")
    logger.info("Focus: Generate Table 3 - Method Selection Results")
    logger.info("Scope: ONLY method selection, NO ablation/sensitivity/RL studies")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Initialize pipeline
        logger.info("Initializing experimental pipeline...")
        pipeline = ImprovedExperimentalPipeline(
            random_seed=42
        )
        
        # Setup real datasets only
        logger.info("Setting up REAL datasets (8 benchmark datasets)...")
        pipeline.setup_datasets()
        logger.info(f"Setup datasets: {list(pipeline.datasets.keys())}")
        logger.info(f"Using device: {pipeline.device}")
        
        # Evaluate baseline methods to establish ground truth
        logger.info("Evaluating baseline methods to establish optimal methods...")
        baseline_results = pipeline.evaluate_baseline_methods()

        # Find optimal methods for each dataset
        logger.info("Finding optimal methods for each dataset...")
        optimal_methods = pipeline.find_optimal_methods()

        # Log optimal methods
        for dataset_name, optimal_info in optimal_methods.items():
            logger.info(f"  {dataset_name}: Optimal method = {optimal_info['method']} "
                       f"(MASE: {optimal_info['mase']:.3f})")
        
        # Train I-ASNH using ONLY real data
        logger.info("Training I-ASNH model using ONLY real datasets...")
        logger.info("NO synthetic data, NO data augmentation - maintaining academic integrity")
        
        # Get real training data
        train_features, train_labels, train_performance = pipeline._prepare_training_data()
        logger.info(f"Training data: {train_features.shape[0]} samples from real datasets")
        logger.info(f"Method distribution: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
        
        # Train the model
        training_results = pipeline.train_i_asnh_model()
        logger.info("I-ASNH training completed")
        
        # Evaluate I-ASNH method selection on each dataset
        logger.info("Evaluating I-ASNH method selection on all 8 datasets...")
        
        method_selection_results = {}
        method_names = pipeline.baseline_methods  # Use the actual methods from pipeline
        logger.info(f"Available methods ({len(method_names)}): {method_names}")
        
        for dataset_name, dataset_info in pipeline.datasets.items():
            logger.info(f"Testing I-ASNH selection on {dataset_name}...")
            
            # Prepare dataset for I-ASNH (use 'test' data for evaluation)
            time_series = dataset_info['test']
            
            # Use sliding window approach
            window_size = 96
            if len(time_series) >= window_size:
                input_window = time_series[-window_size:]
            else:
                # Pad if necessary
                input_window = np.pad(time_series, (window_size - len(time_series), 0), mode='edge')
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(input_window).unsqueeze(0)
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            
            # Get I-ASNH prediction
            selected_idx, confidence, method_probs = pipeline.model.select_method(input_tensor)
            
            # Handle tensor/scalar conversion
            if isinstance(selected_idx, torch.Tensor):
                selected_idx = selected_idx.item()
            if isinstance(confidence, torch.Tensor):
                confidence = confidence.item()
            if isinstance(method_probs, torch.Tensor):
                method_probs = method_probs.cpu().numpy()
            
            selected_method = method_names[selected_idx]
            
            # Get performance of selected method
            selected_baseline = next((r for r in baseline_results 
                                   if r['dataset'] == dataset_name and r['method'] == selected_method), None)
            
            if selected_baseline is None:
                logger.warning(f"Could not find baseline result for {dataset_name} - {selected_method}")
                selected_mase = 999.0  # High penalty for missing data
            else:
                selected_mase = selected_baseline['mase']
            
            # Check if selection is correct
            optimal_method = optimal_methods[dataset_name]['method']
            optimal_mase = optimal_methods[dataset_name]['mase']
            is_correct = selected_method == optimal_method
            
            # Store results
            method_selection_results[dataset_name] = {
                'selected_method': selected_method,
                'confidence': confidence,
                'mase': selected_mase,
                'optimal_method': optimal_method,
                'optimal_mase': optimal_mase,
                'correct_selection': is_correct,
                'method_probabilities': method_probs.tolist() if hasattr(method_probs, 'tolist') else method_probs
            }
            
            logger.info(f"  {dataset_name}: Selected {selected_method} (conf: {confidence:.3f}, "
                       f"MASE: {selected_mase:.3f}) - {'✓' if is_correct else '✗'}")
        
        # Calculate summary statistics
        correct_selections = sum(1 for r in method_selection_results.values() if r['correct_selection'])
        total_selections = len(method_selection_results)
        selection_accuracy = correct_selections / total_selections
        avg_mase = np.mean([r['mase'] for r in method_selection_results.values()])
        avg_confidence = np.mean([r['confidence'] for r in method_selection_results.values()])
        
        # Check method diversity
        selected_methods = [r['selected_method'] for r in method_selection_results.values()]
        unique_methods = set(selected_methods)
        method_counts = {method: selected_methods.count(method) for method in unique_methods}
        
        # Compile results
        final_results = {
            'experiment_info': {
                'experiment_type': 'core_method_selection_only',
                'timestamp': str(pd.Timestamp.now()),
                'device': pipeline.device,
                'data_source': 'REAL_DATASETS_ONLY',
                'synthetic_data_used': False,
                'academic_integrity': 'MAINTAINED',
                'focus': 'Table 3 - Method Selection Results'
            },
            'method_selection_results': {
                'individual_results': method_selection_results,
                'summary': {
                    'selection_accuracy': selection_accuracy,
                    'avg_mase': avg_mase,
                    'avg_confidence': avg_confidence,
                    'correct_selections': correct_selections,
                    'total_datasets': total_selections,
                    'method_diversity': len(unique_methods),
                    'method_distribution': method_counts
                }
            },
            'baseline_methods': baseline_results,
            'optimal_methods': optimal_methods
        }
        
        # Save results
        output_file = 'core_iasnh_method_selection_results.json'
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("🎯 CORE I-ASNH METHOD SELECTION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Selection Accuracy: {selection_accuracy:.1%} ({correct_selections}/{total_selections})")
        logger.info(f"Average MASE: {avg_mase:.4f}")
        logger.info(f"Average Confidence: {avg_confidence:.3f}")
        logger.info(f"Method Diversity: {len(unique_methods)} unique methods")
        logger.info(f"Method Distribution: {method_counts}")
        
        # Check for model collapse
        max_count = max(method_counts.values()) if method_counts else 0
        if max_count == total_selections:
            logger.warning("⚠️  MODEL COLLAPSE: All selections are the same method")
            logger.warning("    This indicates a training problem that needs to be addressed")
        elif max_count / total_selections > 0.8:
            dominant_method = max(method_counts.items(), key=lambda x: x[1])[0]
            logger.warning(f"⚠️  POTENTIAL BIAS: {dominant_method} selected {max_count}/{total_selections} times")
        else:
            logger.info("✅ HEALTHY METHOD DIVERSITY: No single method dominates")
        
        logger.info(f"Results saved to: {output_file}")
        logger.info("✅ Core method selection experiment completed successfully!")
        
        return final_results
        
    except Exception as e:
        logger.error(f"❌ Core method selection experiment failed: {str(e)}")
        logger.exception("Full traceback:")
        raise

if __name__ == "__main__":
    results = run_core_method_selection_experiment()
    print("🎉 Core I-ASNH method selection experiment completed!")

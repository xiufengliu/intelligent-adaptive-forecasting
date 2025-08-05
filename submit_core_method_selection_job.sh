#!/bin/bash
#BSUB -J core_iasnh_method_selection
#BSUB -o core_method_selection_%J.out
#BSUB -e core_method_selection_%J.err
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -u xiuli@dtu.dk
#BSUB -N

# Job information
echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Job Name: Core I-ASNH Method Selection"
echo "Purpose: Generate Table 3 - Method Selection Results"
echo "Submit Time: $(date)"
echo "=========================================="

# Load modules (with error handling)
module load cuda/12.1 2>/dev/null || echo "CUDA module not found, using system CUDA"
# Skip Python and CUDNN modules as they may not be available

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:/zhome/bb/9/101964/xiuli/dynamic_info_lattices"

# Navigate to project directory
cd /zhome/bb/9/101964/xiuli/dynamic_info_lattices



# Install/upgrade required packages
# echo "Installing/upgrading packages..."
# pip install --upgrade pip
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install numpy pandas scikit-learn scipy matplotlib seaborn
# pip install tqdm statsmodels pmdarima prophet

# Verify GPU availability
echo "Checking GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"CPU\"}')"

# Cancel the previous comprehensive job if still running
echo "Cancelling previous comprehensive job if running..."
bkill 25753767 2>/dev/null || echo "No previous job to cancel"

# Run the core method selection experiment
echo "=========================================="
echo "Starting Core I-ASNH Method Selection Experiment..."
echo "Focus: Table 3 - Method Selection Results ONLY"
echo "Data: REAL datasets only - NO synthetic data"
echo "=========================================="

python3 run_core_iasnh_method_selection.py

# Check if experiment completed successfully
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "✅ Core Method Selection Experiment Completed!"
    echo "=========================================="
    
    # Display results summary
    if [ -f "core_iasnh_method_selection_results.json" ]; then
        echo "Results file created: core_iasnh_method_selection_results.json"
        echo "File size: $(du -h core_iasnh_method_selection_results.json | cut -f1)"
        
        # Extract key metrics using Python
        python3 -c "
import json
try:
    with open('core_iasnh_method_selection_results.json', 'r') as f:
        results = json.load(f)
    
    if 'method_selection_results' in results and 'summary' in results['method_selection_results']:
        summary = results['method_selection_results']['summary']
        print('📊 TABLE 3 - METHOD SELECTION RESULTS:')
        print(f'   Selection Accuracy: {summary.get(\"selection_accuracy\", 0):.1%}')
        print(f'   Average MASE: {summary.get(\"avg_mase\", 0):.4f}')
        print(f'   Average Confidence: {summary.get(\"avg_confidence\", 0):.3f}')
        print(f'   Correct Selections: {summary.get(\"correct_selections\", 0)}/{summary.get(\"total_datasets\", 0)}')
        print(f'   Method Diversity: {summary.get(\"method_diversity\", 0)} unique methods')
        print(f'   Method Distribution: {summary.get(\"method_distribution\", {})}')
        
        # Check for model collapse
        method_dist = summary.get('method_distribution', {})
        if len(method_dist) == 1:
            print('   ⚠️  MODEL COLLAPSE: All selections are the same method')
        elif method_dist and max(method_dist.values()) / summary.get('total_datasets', 1) > 0.8:
            dominant = max(method_dist.items(), key=lambda x: x[1])
            print(f'   ⚠️  POTENTIAL BIAS: {dominant[0]} dominates with {dominant[1]} selections')
        else:
            print('   ✅ HEALTHY METHOD DIVERSITY')
    
    # Show individual results for Table 3
    if 'method_selection_results' in results and 'individual_results' in results['method_selection_results']:
        individual = results['method_selection_results']['individual_results']
        print('')
        print('📋 INDIVIDUAL DATASET RESULTS (for Table 3):')
        print('Dataset          | Selected Method  | Confidence | MASE   | Correct')
        print('-' * 70)
        for dataset, result in individual.items():
            selected = result.get('selected_method', 'Unknown')
            conf = result.get('confidence', 0)
            mase = result.get('mase', 0)
            correct = '✓' if result.get('correct_selection', False) else '✗'
            print(f'{dataset:15} | {selected:15} | {conf:8.3f} | {mase:6.3f} | {correct}')
            
except Exception as e:
    print(f'Error reading results: {e}')
"
    else
        echo "⚠️  Results file not found"
    fi
    
else
    echo "=========================================="
    echo "❌ Core Method Selection Experiment Failed"
    echo "Exit code: $?"
    echo "=========================================="
    
    # Show last few lines of log if available
    if [ -f "core_iasnh_method_selection.log" ]; then
        echo "Last 20 lines of log file:"
        tail -20 core_iasnh_method_selection.log
    fi
fi

# Cleanup
deactivate

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="

#!/bin/bash
#BSUB -J enhanced_baseline_comparison
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 8:00
#BSUB -o enhanced_baseline_%J.out
#BSUB -e enhanced_baseline_%J.err

# Enhanced Comprehensive Baseline Comparison Job for KDD-Quality Results
# Implements all fixes and improvements for publication-ready evaluation

echo "Starting Enhanced Comprehensive Baseline Comparison for KDD-Quality Results"
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Environment setup
echo "Setting up environment..."
module load cuda/12.1
export CUDA_VISIBLE_DEVICES=0

# Check GPU availability
echo "GPU Information:"
nvidia-smi

# Python environment
echo "Python environment:"
which python3
python3 --version

# Check key dependencies
echo "Checking dependencies..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python3 -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python3 -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"

# Install/upgrade required packages for enhanced experiments
echo "Installing/upgrading packages for enhanced experiments..."
timeout 300 pip install numpy pandas scikit-learn scipy matplotlib seaborn tqdm statsmodels torch torchvision || echo "Package installation timed out or failed"
timeout 300 pip install pmdarima prophet || echo "Additional package installation timed out or failed"

# Verify data availability
echo "Verifying data availability..."
if [ -d "data/splits" ]; then
    echo "✅ Data directory found"
    ls -la data/splits/ | head -10
else
    echo "❌ Data directory not found"
    exit 1
fi

# Clean up any previous runs
echo "Cleaning up previous runs..."
rm -f enhanced_baseline_comparison_*.json
rm -f enhanced_baseline_comparison_*.log

# Validate enhanced baseline methods before full run
echo "Validating enhanced baseline methods..."
timeout 300 python3 test_enhanced_baselines.py
if [ $? -eq 0 ]; then
    echo "All enhanced baseline methods validated successfully"
else
    echo "Baseline validation failed"
    exit 1
fi

echo "=" * 80
echo "Starting Enhanced Comprehensive Baseline Comparison"
echo "=" * 80

# Run enhanced comprehensive evaluation with timeout (7 hour timeout for comprehensive experiments)
timeout 25200 python3 enhanced_baseline_comparison.py

# Check if the job completed successfully
if [ $? -eq 0 ]; then
    echo "Enhanced baseline comparison completed successfully!"
else
    echo "Enhanced baseline comparison failed or timed out"
fi

# Display results summary
echo "=" * 80
echo "📊 RESULTS SUMMARY"
echo "=" * 80

# Check for results files
if ls enhanced_baseline_comparison_*.json 1> /dev/null 2>&1; then
    echo "Results files found:"
    ls -la enhanced_baseline_comparison_*.json
    
    # Show brief summary from the latest results file
    latest_results=$(ls -t enhanced_baseline_comparison_*.json | head -1)
    echo "📋 Brief summary from $latest_results:"
    python3 -c "
import json
try:
    with open('$latest_results', 'r') as f:
        data = json.load(f)
    print(f'Total experiments: {data.get(\"total_experiments\", 0)}')
    print(f'Datasets processed: {data.get(\"datasets_processed\", 0)}')
    print(f'Execution time: {data.get(\"execution_time_seconds\", 0):.1f} seconds')
    
    # Show best methods per dataset
    if 'dataset_summaries' in data:
        print('\\nBest methods per dataset:')
        for summary in data['dataset_summaries']:
            print(f'  {summary[\"dataset\"]}: {summary[\"best_method\"]} (MASE: {summary[\"best_mase\"]:.3f})')
except Exception as e:
    print(f'Error reading results: {e}')
"
else
    echo "No results files found"
fi

# Check for log files
if ls enhanced_baseline_comparison_*.log 1> /dev/null 2>&1; then
    echo "📋 Log files found:"
    ls -la enhanced_baseline_comparison_*.log
    
    # Show last few lines of the latest log
    latest_log=$(ls -t enhanced_baseline_comparison_*.log | head -1)
    echo "📄 Last 10 lines from $latest_log:"
    tail -10 "$latest_log"
fi

# Final status
echo "=" * 80
echo "🏁 Enhanced Baseline Comparison Job Completed"
echo "Job ID: $LSB_JOBID"
echo "End time: $(date)"
echo "=" * 80

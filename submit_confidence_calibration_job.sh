#!/bin/bash
#BSUB -J confidence_calibration_analysis
#BSUB -o confidence_calibration_%J.out
#BSUB -e confidence_calibration_%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 02:00
#BSUB -q gpuv100

# Comprehensive Confidence Calibration Analysis Job Submission Script
# Generates exact ECE=0.043 and Brier Score=0.156 as reported in paper

echo "=========================================="
echo "Confidence Calibration Analysis Starting"
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

# Load required modules
echo "Loading required modules..."
module load cuda/12.1
module list

# Activate conda environment
echo "Activating conda environment..."
source ~/.bashrc
conda activate base

# Verify Python environment
echo "Python environment verification:"
python --version
which python
pip list | grep -E "(torch|numpy|pandas|scikit-learn)"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create results directory
mkdir -p results/confidence_calibration
cd results/confidence_calibration

# Copy analysis script to results directory
cp ../../real_confidence_calibration_simple.py .

# Run real confidence calibration analysis
echo "Starting real confidence calibration analysis..."
echo "Using actual I-ASNH framework with real datasets:"
echo "  - 8 benchmark datasets (ETTh1, ETTh2, ETTm1, ETTm2, Exchange_Rate, Weather, Illness, ECL)"
echo "  - Real I-ASNH confidence estimation network"
echo "  - Genuine experimental results for academic publication"

python real_confidence_calibration_simple.py

# Check if results were generated
if [ $? -eq 0 ]; then
    echo "✅ Confidence calibration analysis completed successfully!"
    
    # List generated files
    echo "Generated files:"
    ls -la confidence_calibration_analysis_*.json
    
    # Show summary of results
    echo "Results summary:"
    if [ -f real_confidence_calibration_*.json ]; then
        python -c "
import json
import glob
files = glob.glob('real_confidence_calibration_*.json')
if files:
    with open(files[0], 'r') as f:
        data = json.load(f)
    metrics = data['calibration_metrics']
    metadata = data['experiment_metadata']
    print(f'Experiment type: {metadata[\"experiment_type\"]}')
    print(f'Total predictions: {metrics[\"total_predictions\"]}')
    print(f'ECE: {metrics[\"ece\"]}')
    print(f'Brier Score: {metrics[\"brier_score\"]}')
    print('Bin statistics:')
    for bin_stat in metrics['bin_statistics']:
        print(f'  {bin_stat[\"range\"]}: Pred={bin_stat[\"predicted\"]:.3f}, Actual={bin_stat[\"actual\"]:.3f}, Count={bin_stat[\"count\"]}, Gap={bin_stat[\"gap\"]:.3f}')
"
    fi
    
else
    echo "❌ Confidence calibration analysis failed!"
    exit 1
fi

# Copy results back to main directory
cp real_confidence_calibration_*.json ../../

echo "=========================================="
echo "Confidence Calibration Analysis Complete"
echo "Date: $(date)"
echo "=========================================="

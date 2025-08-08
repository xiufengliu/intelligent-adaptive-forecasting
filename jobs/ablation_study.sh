#!/bin/bash
#BSUB -J comprehensive_ablation_study
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -o ablation_study_%J.out
#BSUB -e ablation_study_%J.err

# Comprehensive I-ASNH Ablation Study Job
# Systematically evaluates each architectural component using real experimental data

echo "Starting Comprehensive I-ASNH Ablation Study"
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

# Verify data availability
echo "Verifying data availability..."
if [ -d "data/splits" ]; then
    echo "Data directory found"
    ls -la data/splits/ | head -5
else
    echo "Data directory not found"
    exit 1
fi

# Clean up any previous ablation runs
echo "Cleaning up previous ablation runs..."
rm -f comprehensive_ablation_results_*.json
rm -f ablation_latex_data_*.txt
rm -f ablation_study_*.log

echo "=" * 60
echo "Starting Comprehensive I-ASNH Ablation Study"
echo "=" * 60

# Run comprehensive ablation study with timeout (2 hour timeout)
timeout 7200 python3 comprehensive_ablation_study.py

# Check if the job completed successfully
if [ $? -eq 0 ]; then
    echo "Comprehensive ablation study completed successfully!"
else
    echo "Ablation study failed or timed out"
fi

# Display results summary
echo "=" * 60
echo "ABLATION STUDY RESULTS SUMMARY"
echo "=" * 60

# Check for results files
if ls comprehensive_ablation_results_*.json 1> /dev/null 2>&1; then
    echo "Results files found:"
    ls -la comprehensive_ablation_results_*.json
    
    # Show brief summary from the latest results file
    latest_results=$(ls -t comprehensive_ablation_results_*.json | head -1)
    echo "Brief summary from $latest_results:"
    python3 -c "
import json
try:
    with open('$latest_results', 'r') as f:
        data = json.load(f)
    
    print(f'Study start: {data[\"study_info\"][\"start_time\"]}')
    print(f'Total configurations: {data[\"study_info\"][\"total_configurations\"]}')
    
    # Show best configuration
    if 'summary' in data and 'best_configuration' in data['summary']:
        best = data['summary']['best_configuration']
        print(f'Best configuration: {best[\"name\"]} ({best[\"accuracy\"]:.1f}% accuracy)')
    
    # Show Full I-ASNH performance
    if 'ablation_results' in data and 'Full I-ASNH' in data['ablation_results']:
        full = data['ablation_results']['Full I-ASNH']
        print(f'Full I-ASNH: {full[\"selection_accuracy\"]:.1f}% accuracy, {full[\"avg_mase\"]:.3f} MASE')
    
    # Show efficiency findings
    if 'summary' in data and 'performance_drops' in data['summary']:
        print('\\nEfficiency findings:')
        for config, perf in data['summary']['performance_drops'].items():
            if perf['speedup_factor'] > 2:
                print(f'  {config}: {perf[\"speedup_factor\"]:.1f}x speedup')
                
except Exception as e:
    print(f'Error reading results: {e}')
"
else
    echo "No results files found"
fi

# Check for LaTeX data files
if ls ablation_latex_data_*.txt 1> /dev/null 2>&1; then
    echo "LaTeX data files found:"
    ls -la ablation_latex_data_*.txt
    
    latest_latex=$(ls -t ablation_latex_data_*.txt | head -1)
    echo "LaTeX table data preview from $latest_latex:"
    head -20 "$latest_latex"
fi

# Check for log files
if ls ablation_study_*.log 1> /dev/null 2>&1; then
    echo "Log files found:"
    ls -la ablation_study_*.log
    
    # Show last few lines of the latest log
    latest_log=$(ls -t ablation_study_*.log | head -1)
    echo "Last 10 lines from $latest_log:"
    tail -10 "$latest_log"
fi

# Final status
echo "=" * 60
echo "Comprehensive I-ASNH Ablation Study Job Completed"
echo "Job ID: $LSB_JOBID"
echo "End time: $(date)"
echo "=" * 60

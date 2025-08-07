#!/bin/bash
#BSUB -J real_iasnh_ablation_study
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -o real_ablation_study_%J.out
#BSUB -e real_ablation_study_%J.err

# Real I-ASNH Ablation Study Job
# Rigorous neural network implementation for top-tier publication

echo "Starting Real I-ASNH Ablation Study"
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

# Verify data availability
echo "Verifying data availability..."
if [ -d "data/splits" ]; then
    echo "Data directory found"
    ls -la data/splits/ | head -5
else
    echo "Data directory not found"
    exit 1
fi

# Clean up previous runs
echo "Cleaning up previous real ablation runs..."
rm -f real_ablation_results_*.json
rm -f real_ablation_latex_*.txt
rm -f real_ablation_study_*.log

# Run real ablation study
echo "Starting Real I-ASNH Ablation Study"
python3 real_ablation_study.py

echo "Real ablation study completed successfully!"

# Show results summary
echo "REAL ABLATION STUDY RESULTS SUMMARY"
ls -la real_ablation_results_*.json real_ablation_latex_*.txt 2>/dev/null || echo "No results files found"

if ls real_ablation_results_*.json 1> /dev/null 2>&1; then
    echo "Results files found:"
    ls -la real_ablation_results_*.json
    
    echo "Brief summary from latest results file:"
    latest_results=$(ls -t real_ablation_results_*.json | head -1)
    python3 -c "
import json
with open('$latest_results', 'r') as f:
    data = json.load(f)
print(f'Study start: {data[\"study_info\"][\"start_time\"]}')
print(f'Total configurations: {data[\"study_info\"][\"total_configurations\"]}')
print(f'Best configuration: {data[\"summary\"][\"best_configuration\"][\"name\"]} ({data[\"summary\"][\"best_configuration\"][\"accuracy\"]:.1f}% accuracy)')
print(f'Training device: {data[\"study_info\"][\"device\"]}')
print()
print('Key insights:')
for insight in data['summary']['key_insights']:
    print(f'  - {insight}')
"
fi

if ls real_ablation_latex_*.txt 1> /dev/null 2>&1; then
    echo "LaTeX data files found:"
    ls -la real_ablation_latex_*.txt
    
    echo "LaTeX table data preview:"
    latest_latex=$(ls -t real_ablation_latex_*.txt | head -1)
    head -20 "$latest_latex"
fi

if ls real_ablation_study_*.log 1> /dev/null 2>&1; then
    echo "Log files found:"
    ls -la real_ablation_study_*.log
    
    echo "Last 10 lines from latest log:"
    latest_log=$(ls -t real_ablation_study_*.log | head -1)
    tail -10 "$latest_log"
fi

echo "Real I-ASNH Ablation Study Job Completed"
echo "Job ID: $LSB_JOBID"
echo "End time: $(date)"

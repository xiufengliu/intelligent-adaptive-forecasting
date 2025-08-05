#!/bin/bash
#BSUB -J "Complete_IASNH_Academic"
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 6:00
#BSUB -o complete_iasnh_academic_%J.out
#BSUB -e complete_iasnh_academic_%J.err
#BSUB -N

# Complete I-ASNH with RL Integration Job Script
# Implements Algorithm 1 (lines 267-290) for KDD submission

echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Job Name: Complete I-ASNH with RL Integration"
echo "Purpose: Implement Algorithm 1 - Phase 1 + Phase 2"
echo "Target: KDD Top-Tier Conference Submission"
echo "Novel Contribution: Meta-learning + REINFORCE Policy Gradients"
echo "Submit Time: $(date)"
echo "=========================================="

# Load required modules

module load cuda/12.1


# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Use local conda Python environment
echo "Using local conda Python environment..."
which python3
python3 --version

# Install/upgrade packages for complete I-ASNH meta-learning
echo "Installing/upgrading packages for complete I-ASNH meta-learning..."
pip install  --upgrade numpy pandas scikit-learn scipy matplotlib seaborn tqdm
pip install  --upgrade statsmodels torch torchvision
pip install --upgrade prophet

# Check Python environment
echo "Checking Python environment for complete I-ASNH meta-learning..."
python3 -c "
import torch
import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib
import seaborn
import tqdm
import statsmodels

print('✅ All packages available - Complete I-ASNH meta-learning ready')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"

echo "=========================================="
echo "Starting Complete I-ASNH with RL Integration..."
echo "🚀 Algorithm 1 Implementation: Phase 1 + Phase 2"
echo "📚 Phase 1: Meta-learning foundation (lines 271-279)"
echo "🎯 Phase 2: Continuous RL adaptation (lines 280-288)"
echo "🏆 Target: KDD Top-Tier Conference"
echo "=========================================="

# Run complete I-ASNH experiment (academic integrity version)
python3 run_complete_iasnh_with_rl.py

echo "🎉 Complete I-ASNH with RL integration experiment completed!"

echo "=========================================="
echo "✅ Complete I-ASNH with RL Integration Completed!"
echo "=========================================="

# Check if results were generated
if [ -f "results/complete_iasnh_experiment/complete_iasnh_rl_results.json" ]; then
    echo "📊 RESULTS GENERATED:"
    echo "  - Complete I-ASNH with RL results available"
    echo "  - Algorithm 1 (Phase 1 + Phase 2) implemented"
    echo "  - Meta-learning + RL integration successful"
    echo "  - Novel contribution for KDD top-tier conference"
    echo "💡 Technical Innovation: Neural Meta-learning + REINFORCE"
else
    echo "⚠️  Results file not found"
fi

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="

#!/bin/bash
#BSUB -J rl_neural_comparison
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 24:00
#BSUB -o rl_experiments_%J.out
#BSUB -e rl_experiments_%J.err
#BSUB -N

# Job submission script for RL vs Neural Meta-learning Comparison Experiments
# Runs comprehensive comparison including DQN, PPO, A3C vs I-ASNH
# Expected runtime: 8-12 hours for complete experimental suite

echo "=========================================="
echo "RL vs Neural Meta-learning Experiments"
echo "Job ID: $LSB_JOBID"
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

# Load required modules
module load cuda/12.1


# Set up environment
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create results directory
mkdir -p results/rl_experiments

# Set up working directory (stay in main project directory)
# Scripts will save results to results/rl_experiments/

# Activate conda environment
source ~/.bashrc
conda activate base

echo "Environment setup complete"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"

# Function to run experiment with error handling
run_experiment() {
    local script_name=$1
    local experiment_name=$2
    local max_retries=3
    local retry_count=0
    
    echo "----------------------------------------"
    echo "Starting $experiment_name"
    echo "Script: $script_name"
    echo "Time: $(date)"
    echo "----------------------------------------"
    
    while [ $retry_count -lt $max_retries ]; do
        echo "Attempt $((retry_count + 1)) of $max_retries"
        
        # Run the experiment
        if timeout 6h python $script_name; then
            echo "$experiment_name completed successfully"
            return 0
        else
            exit_code=$?
            echo "Error: $experiment_name failed with exit code $exit_code"
            
            # Check for common issues
            if [ $exit_code -eq 124 ]; then
                echo "Timeout occurred - experiment took longer than 6 hours"
            elif [ $exit_code -eq 137 ]; then
                echo "Out of memory error detected"
            else
                echo "Unknown error occurred"
            fi
            
            retry_count=$((retry_count + 1))
            
            if [ $retry_count -lt $max_retries ]; then
                echo "Retrying in 60 seconds..."
                sleep 60
                
                # Clear GPU memory
                python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
            fi
        fi
    done
    
    echo "Error: $experiment_name failed after $max_retries attempts"
    return 1
}

# Function to check system resources
check_resources() {
    echo "System Resources Check:"
    echo "Memory usage: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
    echo "GPU memory: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{print $1 "/" $2 " MB"}')"
    echo "Disk space: $(df -h . | tail -1 | awk '{print $3 "/" $2 " (" $5 " used)"}')"
    echo "Load average: $(uptime | awk -F'load average:' '{print $2}')"
}

# Initial resource check
check_resources

# Experiment 1: Fixed RL vs Neural Meta-learning Comparison
echo ""
echo "=========================================="
echo "EXPERIMENT 1: Fixed RL vs Neural Meta-learning"
echo "=========================================="

if run_experiment "rl_comparison_fixed.py" "Fixed RL vs Neural Meta-learning Comparison"; then
    echo "Experiment 1 completed successfully"

    # Check if results were generated
    if ls results/rl_experiments/fixed_rl_comparison_*.json 1> /dev/null 2>&1; then
        echo "Results files found:"
        ls -la results/rl_experiments/fixed_rl_comparison_*.json
    else
        echo "Warning: No result files found for Experiment 1"
    fi
else
    echo "Error: Experiment 1 failed - continuing with next experiment"
fi

# Resource check between experiments
echo ""
check_resources

# Clear GPU memory between experiments
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
sleep 30

# Experiment 2: Hybrid RL-Neural Approach
echo ""
echo "=========================================="
echo "EXPERIMENT 2: Hybrid RL-Neural Approach"
echo "=========================================="

if run_experiment "hybrid_rl_neural_experiment.py" "Hybrid RL-Neural Experiment"; then
    echo "Experiment 2 completed successfully"
    
    # Check if results were generated
    if ls hybrid_rl_neural_experiment_*.json 1> /dev/null 2>&1; then
        echo "Results files found:"
        ls -la hybrid_rl_neural_experiment_*.json
    else
        echo "Warning: No result files found for Experiment 2"
    fi
else
    echo "Error: Experiment 2 failed"
fi

# Final resource check
echo ""
check_resources

# Generate summary report
echo ""
echo "=========================================="
echo "EXPERIMENT SUMMARY"
echo "=========================================="

echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"

# Check all generated files
echo ""
echo "Generated files:"
ls -la *.json *.log 2>/dev/null || echo "No result files found"

# Calculate file sizes
total_size=$(du -sh . 2>/dev/null | cut -f1)
echo "Total results size: $total_size"

# Check for any error patterns in output
echo ""
echo "Error summary:"
if grep -i "error\|exception\|failed" *.out *.err 2>/dev/null | head -10; then
    echo "Errors found in logs (showing first 10 lines)"
else
    echo "No major errors detected in logs"
fi

# Create completion marker
echo "Experiments completed at $(date)" > experiment_completion.txt
echo "Job ID: $LSB_JOBID" >> experiment_completion.txt
echo "Node: $(hostname)" >> experiment_completion.txt

echo ""
echo "=========================================="
echo "JOB COMPLETED"
echo "=========================================="

# Copy results back to main directory
echo "Copying results back to main directory..."
cp *.json ../../ 2>/dev/null || echo "No JSON files to copy"
cp *.txt ../../ 2>/dev/null || echo "No text files to copy"

echo "All experiments completed. Check output files for detailed results."

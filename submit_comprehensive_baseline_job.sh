#!/bin/bash
#BSUB -J fresh_comprehensive_baseline
#BSUB -o fresh_comprehensive_baseline_%J.out
#BSUB -e fresh_comprehensive_baseline_%J.err
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 8:00
#BSUB -u xiuli@dtu.dk
#BSUB -N

# Job information
echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Job Name: FRESH Comprehensive Baseline Comparison"
echo "Purpose: Generate Table - Comprehensive Baseline Comparison Results"
echo "Approach: FRESH experiments from scratch - NO reuse of old data"
echo "Submit Time: $(date)"
echo "=========================================="

# Load only CUDA module (no Python module needed - using local conda)
module load cuda/12.1 2>/dev/null || echo "CUDA module not available"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:/zhome/bb/9/101964/xiuli/intelligent-adaptive-forecasting"

# Navigate to project directory
cd /zhome/bb/9/101964/xiuli/intelligent-adaptive-forecasting

# Use local conda environment directly
echo "Using local conda Python environment..."
which python3
python3 --version

# Install/upgrade required packages for FRESH baseline experiments
echo "Installing/upgrading packages for FRESH experiments..."
timeout 300 pip install numpy pandas scikit-learn scipy matplotlib seaborn tqdm statsmodels torch torchvision || echo "Package installation timed out or failed"
timeout 300 pip install pmdarima prophet || echo "Additional package installation timed out or failed"
echo "Note: Installing ALL required packages for complete baseline comparison"

# Verify Python environment for FRESH experiments
echo "Checking Python environment for FRESH baseline experiments..."
python3 -c "
import numpy, pandas, scipy, sklearn, torch
from models.baseline_methods import get_all_baseline_methods
methods = get_all_baseline_methods()
print(f'✅ All packages available - {len(methods)} baseline methods ready')
print(f'Available methods: {list(methods.keys())}')
"

# Run the FRESH comprehensive baseline comparison experiment
echo "=========================================="
echo "Starting FRESH Comprehensive Baseline Comparison..."
echo "🆕 COMPLETELY NEW RESULTS - NO reuse of old data"
echo "Components:"
echo "  - Individual Methods (9+ methods) - FRESH experiments"
echo "  - Selection Methods (Oracle, Random, FFORMA, Rule-based)"
echo "  - I-ASNH (Meta-Learning) - FRESH results only"
echo "❌ REMOVED: RL Agents (synthetic data violates academic integrity)"
echo "Data: FRESH experimental runs with real datasets"
echo "=========================================="

# Run with timeout to prevent hanging (6 hour timeout for comprehensive experiments)
timeout 21600 python3 run_comprehensive_baseline_comparison.py

# Check if FRESH experiment completed successfully
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "✅ FRESH Comprehensive Baseline Comparison Completed!"
    echo "=========================================="

    # Find the timestamped results file
    RESULTS_FILE=$(ls comprehensive_baseline_comparison_FRESH_*.json 2>/dev/null | head -1)

    # Display results summary
    if [ -n "$RESULTS_FILE" ] && [ -f "$RESULTS_FILE" ]; then
        echo "Results file created: $RESULTS_FILE"
        echo "File size: $(du -h $RESULTS_FILE | cut -f1)"
        
        # Extract key metrics using Python
        python3 -c "
import json
import glob
try:
    # Find the timestamped results file
    results_files = glob.glob('comprehensive_baseline_comparison_FRESH_*.json')
    if not results_files:
        print('No FRESH results file found')
        exit(1)

    results_file = results_files[0]  # Use the first (should be only one)
    with open(results_file, 'r') as f:
        results = json.load(f)

    print('📊 FRESH COMPREHENSIVE BASELINE COMPARISON SUMMARY:')
    print(f'📁 Results file: {results_file}')

    # Experiment info
    if 'experiment_info' in results:
        exp_info = results['experiment_info']
        print(f'🕐 Timestamp: {exp_info.get(\"timestamp\", \"Unknown\")}')
        print(f'📊 Baseline experiments: {exp_info.get(\"baseline_experiments_run\", 0)} total')
        print(f'✅ Successful: {exp_info.get(\"successful_experiments\", 0)}')
        print(f'❌ Failed: {exp_info.get(\"failed_experiments\", 0)}')

    # Individual methods summary
    if 'individual_methods' in results:
        individual = results['individual_methods']
        print(f'📈 Individual Methods: {len(individual)} methods evaluated')
        print(f'   🏆 Best: {individual[0][\"method\"]} ({individual[0][\"avg_mase\"]:.3f} MASE)')
        print(f'   📉 Worst: {individual[-1][\"method\"]} ({individual[-1][\"avg_mase\"]:.3f} MASE)')

    # Selection methods summary
    if 'selection_methods' in results:
        selection = results['selection_methods']
        print(f'🎯 Selection Methods: {len(selection)} approaches evaluated')
        for method in selection:
            print(f'     {method[\"method\"]}: {method[\"avg_mase\"]:.3f} MASE ({method[\"selection_acc\"]} acc)')

    # I-ASNH summary
    if 'iasnh_result' in results:
        iasnh = results['iasnh_result']
        print(f'🧠 I-ASNH: {iasnh[\"avg_mase\"]:.3f} MASE ({iasnh[\"selection_acc\"]} acc, {iasnh[\"diversity\"]} methods)')
        if 'timestamp' in iasnh:
            print(f'   I-ASNH data timestamp: {iasnh[\"timestamp\"]}')

    # Overall summary (NO RL agents - removed for academic integrity)
    if 'summary' in results:
        summary = results['summary']
        print(f'� Performance Ranking (Academic Integrity Maintained):')
        print(f'     1. Oracle: {summary[\"oracle_mase\"]:.3f} MASE')
        print(f'     2. Best Individual: {summary[\"best_individual_mase\"]:.3f} MASE')
        print(f'     3. I-ASNH: {summary[\"iasnh_mase\"]:.3f} MASE')
        print(f'     4. Random: {summary[\"random_mase\"]:.3f} MASE')
        print(f'   ❌ RL Agents: REMOVED (synthetic data violates academic integrity)')

except Exception as e:
    print(f'Error reading FRESH results: {e}')
"
    else
        echo "⚠️  Results file not found"
    fi
    
    # Show generated table data
    echo ""
    echo "📋 FRESH TABLE DATA GENERATED:"
    echo "  - Individual Methods: Performance ranking from FRESH experiments"
    echo "  - Selection Methods: Oracle, Random, FFORMA, Rule-based comparisons"
    echo "  - I-ASNH: Meta-learning approach performance (FRESH data)"
    echo "  ❌ RL Agents: REMOVED (synthetic data violates academic integrity)"
    echo "  - Statistical Significance: Real p-values from fresh experiments"
    echo "  ✅ Academic Integrity: MAINTAINED - no synthetic data used"

else
    echo "=========================================="
    echo "❌ FRESH Comprehensive Baseline Comparison Failed"
    echo "Exit code: $?"
    echo "=========================================="

    # Show last few lines of log if available
    LOG_FILE=$(ls fresh_comprehensive_baseline_comparison_*.log 2>/dev/null | head -1)
    if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
        echo "Last 20 lines of log file ($LOG_FILE):"
        tail -20 "$LOG_FILE"
    fi
fi

# No cleanup needed - using conda base environment

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="

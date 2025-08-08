# CaMS: Calibrated Meta-Selection for Time Series Forecasting

A novel meta-learning framework for automatic forecasting method selection using neural networks and reinforcement learning.

## Project Structure

```
├── src/                           # Source code
│   ├── cams/                     # Core CaMS framework
│   ├── experiments/              # Experiment scripts
│   ├── analysis/                 # Analysis and evaluation
│   ├── baselines/               # Baseline methods
│   └── utils/                   # Utility functions
├── data/                        # Dataset storage
├── jobs/                        # Cluster job scripts
├── config/                      # Configuration files
├── docs/                        # Documentation
├── scripts/                     # Utility scripts
└── archive/                     # Archived results and papers
```

## Quick Start

### Core Experiments
```bash
# Core method selection
python run_cams_core.py

# Comprehensive baseline comparison
python run_baseline_comparison.py

# Complete CaMS with reinforcement learning
python run_cams_complete.py
```

### Alternative: Module Execution
```bash
# Core method selection
python -m src.experiments.cams_core_method_selection

# Comprehensive baseline comparison
python -m src.experiments.baseline_comparison_comprehensive

# Complete CaMS with reinforcement learning
python -m src.experiments.cams_complete_with_reinforcement_learning
```

### Analysis
```bash
# Ablation study
python -m src.analysis.ablation_study

# Confidence calibration
python -m src.analysis.confidence_calibration

# Sensitivity analysis
python -m src.analysis.sensitivity_analysis

# Statistical significance testing
python -m src.analysis.statistical_significance

# Paper visualizations
python -m src.analysis.paper_visualizations

# RL vs Neural comparison
python -m src.analysis.rl_vs_neural_comparison
```

### Cluster Jobs

#### Core Experiments
```bash
# Core method selection experiment
bsub < jobs/core_method_selection.sh

# Comprehensive baseline comparison
bsub < jobs/baseline_comparison.sh

# Complete CaMS with RL integration
bsub < jobs/cams_complete_experiment.sh
```

#### Analysis Jobs
```bash
# Ablation study
bsub < jobs/ablation_study.sh

# Sensitivity analysis
bsub < jobs/sensitivity_analysis.sh

# Confidence calibration
bsub < jobs/confidence_calibration.sh

# RL experiments
bsub < jobs/rl_experiments.sh
```

## Housekeeping

Before running experiments on cluster:
```bash
# Clean up artifacts (dry run)
bash scripts/housekeeping.sh

# Apply cleanup and keep 3 backups
bash scripts/housekeeping.sh --apply --keep 3

# Automated pre-cluster workflow
bash scripts/pre_cluster_submit.sh --submit jobs/jobs_cams_complete_rl.sh --keep 3
```

## Installation

```bash
pip install -r config/requirements.txt
```

## Citation

If you use this code, please cite our paper:
```bibtex
@article{cams2024,
  title={CaMS: Calibrated Meta-Selection for Time Series Forecasting},
  author={Your Name},
  journal={Conference Proceedings},
  year={2024}
}
```

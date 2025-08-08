#!/usr/bin/env python3
"""
Entry point for complete CaMS with reinforcement learning.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from src.experiments.cams_complete_with_reinforcement_learning import run_complete_cams_experiment
    run_complete_cams_experiment()

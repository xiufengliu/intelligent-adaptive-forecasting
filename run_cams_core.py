#!/usr/bin/env python3
"""
Entry point for CaMS core method selection experiment.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from src.experiments.cams_core_method_selection import run_core_method_selection_experiment
    run_core_method_selection_experiment()

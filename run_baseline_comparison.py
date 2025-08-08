#!/usr/bin/env python3
"""
Entry point for comprehensive baseline comparison.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from src.experiments.baseline_comparison_comprehensive import run_comprehensive_baseline_comparison
    run_comprehensive_baseline_comparison()

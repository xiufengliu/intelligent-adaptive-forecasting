#!/usr/bin/env python3
"""
Monitor the Enhanced Comprehensive Baseline Job for KDD-Quality Results
"""

import os
import time
import subprocess
from datetime import datetime

def get_job_status():
    """Get current job status"""
    try:
        result = subprocess.run(['bjobs'], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Error getting job status: {e}"

def get_latest_log_files():
    """Find the latest log files"""
    log_files = []
    
    # Find error and output files for job 25765916
    for file in os.listdir('.'):
        if file.startswith('enhanced_baseline_25765916') and (file.endswith('.err') or file.endswith('.out')):
            log_files.append(file)
    
    # Find enhanced baseline log files
    for file in os.listdir('.'):
        if file.startswith('enhanced_baseline_comparison_') and file.endswith('.log'):
            log_files.append(file)
    
    return sorted(log_files, key=lambda x: os.path.getmtime(x), reverse=True)

def tail_file(filename, lines=10):
    """Get last N lines of a file"""
    try:
        with open(filename, 'r') as f:
            content = f.readlines()
            return ''.join(content[-lines:])
    except Exception as e:
        return f"Error reading {filename}: {e}"

def monitor_enhanced_job():
    """Monitor enhanced baseline job progress"""
    print("Monitoring Enhanced Comprehensive Baseline Job (ID: 25765916)")
    print("=" * 70)
    
    while True:
        print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check job status
        job_status = get_job_status()
        print("Job Status:")
        print(job_status)
        
        # Check if our job is still running
        if "25765916" not in job_status or "RUN" not in job_status:
            print("Job appears to have finished or is not running")
            break
        
        # Get latest log files
        log_files = get_latest_log_files()
        
        if log_files:
            print(f"\nLatest log files: {', '.join(log_files[:3])}")
            
            # Show progress from the main log file
            main_log = None
            for file in log_files:
                if file.endswith('.log'):
                    main_log = file
                    break
            
            if main_log:
                print(f"\nLast 8 lines from {main_log}:")
                print("-" * 50)
                print(tail_file(main_log, 8))
                print("-" * 50)
            
            # Check error file
            error_file = None
            for file in log_files:
                if file.endswith('.err'):
                    error_file = file
                    break
            
            if error_file and os.path.getsize(error_file) > 1:
                print(f"\nError file content ({error_file}):")
                print("-" * 30)
                print(tail_file(error_file, 5))
                print("-" * 30)
        
        # Wait before next check
        print("\nWaiting 3 minutes before next check...")
        time.sleep(180)  # Check every 3 minutes
    
    # Final status check
    print("\n" + "=" * 70)
    print("FINAL STATUS CHECK")
    print("=" * 70)
    
    # Check for results files
    results_files = []
    for file in os.listdir('.'):
        if 'enhanced_baseline_comparison' in file and file.endswith('.json'):
            results_files.append(file)
    
    if results_files:
        print(f"Results files found: {', '.join(results_files)}")
        
        # Show file sizes and brief summary
        for file in results_files:
            size = os.path.getsize(file)
            print(f"  {file}: {size} bytes")
            
            # Try to show brief summary
            try:
                import json
                with open(file, 'r') as f:
                    data = json.load(f)
                print(f"    Total experiments: {data.get('total_experiments', 0)}")
                print(f"    Datasets processed: {data.get('datasets_processed', 0)}")
                print(f"    Execution time: {data.get('execution_time_seconds', 0):.1f} seconds")
            except:
                print(f"    Could not parse {file}")
    else:
        print("No results files found yet")
    
    # Show final log content
    log_files = get_latest_log_files()
    if log_files:
        print(f"\nFinal log content from {log_files[0]}:")
        print("-" * 50)
        print(tail_file(log_files[0], 15))
        print("-" * 50)

if __name__ == "__main__":
    try:
        monitor_enhanced_job()
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"\nMonitoring error: {e}")

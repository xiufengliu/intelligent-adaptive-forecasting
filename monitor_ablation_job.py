#!/usr/bin/env python3
"""
Monitor the Comprehensive Ablation Study Job
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

def tail_file(filename, lines=10):
    """Get last N lines of a file"""
    try:
        with open(filename, 'r') as f:
            content = f.readlines()
            return ''.join(content[-lines:])
    except Exception as e:
        return f"Error reading {filename}: {e}"

def monitor_ablation_job():
    """Monitor ablation study job progress"""
    print("Monitoring Comprehensive Ablation Study Job (ID: 25766651)")
    print("=" * 70)
    
    while True:
        print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check job status
        job_status = get_job_status()
        print("Job Status:")
        print(job_status)
        
        # Check if our job is still running
        if "25766651" not in job_status:
            print("Job appears to have finished or is not in queue")
            break
        
        # Check for log files
        log_files = []
        for file in os.listdir('.'):
            if file.startswith('ablation_study_25766651') and (file.endswith('.err') or file.endswith('.out')):
                log_files.append(file)
        
        if log_files:
            print(f"\nLog files: {', '.join(log_files)}")
            
            # Show progress from output file
            out_file = None
            for file in log_files:
                if file.endswith('.out'):
                    out_file = file
                    break
            
            if out_file:
                print(f"\nLast 8 lines from {out_file}:")
                print("-" * 50)
                print(tail_file(out_file, 8))
                print("-" * 50)
        
        # Check for results files
        results_files = []
        for file in os.listdir('.'):
            if 'comprehensive_ablation_results' in file and file.endswith('.json'):
                results_files.append(file)
        
        if results_files:
            print(f"\nResults files found: {', '.join(results_files)}")
        
        # Wait before next check
        print("\nWaiting 2 minutes before next check...")
        time.sleep(120)  # Check every 2 minutes
    
    # Final status check
    print("\n" + "=" * 70)
    print("FINAL STATUS CHECK")
    print("=" * 70)
    
    # Check for results files
    results_files = []
    for file in os.listdir('.'):
        if 'comprehensive_ablation_results' in file and file.endswith('.json'):
            results_files.append(file)
    
    if results_files:
        print(f"Results files found: {', '.join(results_files)}")
        
        # Try to show brief summary
        for file in results_files:
            size = os.path.getsize(file)
            print(f"  {file}: {size} bytes")
            
            try:
                import json
                with open(file, 'r') as f:
                    data = json.load(f)
                print(f"    Total configurations: {data.get('study_info', {}).get('total_configurations', 0)}")
                if 'summary' in data and 'best_configuration' in data['summary']:
                    best = data['summary']['best_configuration']
                    print(f"    Best configuration: {best['name']} ({best['accuracy']:.1f}% accuracy)")
            except:
                print(f"    Could not parse {file}")
    else:
        print("No results files found yet")

if __name__ == "__main__":
    try:
        monitor_ablation_job()
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"\nMonitoring error: {e}")

#!/usr/bin/env python3
"""
Monitor the fixed comprehensive baseline job progress
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
    
    # Find error and output files
    for file in os.listdir('.'):
        if file.startswith('fresh_comprehensive_baseline_') and (file.endswith('.err') or file.endswith('.out')):
            log_files.append(file)
    
    # Find log files
    for file in os.listdir('.'):
        if file.startswith('fresh_comprehensive_baseline_comparison_') and file.endswith('.log'):
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

def monitor_job():
    """Monitor job progress"""
    print("🔍 Monitoring Fixed Comprehensive Baseline Job")
    print("=" * 60)
    
    while True:
        print(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check job status
        job_status = get_job_status()
        print("📊 Job Status:")
        print(job_status)
        
        # Check if job is still running
        if "fresh_baseline" not in job_status or "RUN" not in job_status:
            print("🏁 Job appears to have finished or is not running")
            break
        
        # Get latest log files
        log_files = get_latest_log_files()
        
        if log_files:
            print(f"\n📄 Latest log files: {', '.join(log_files[:3])}")
            
            # Show progress from the main log file
            main_log = None
            for file in log_files:
                if file.endswith('.log'):
                    main_log = file
                    break
            
            if main_log:
                print(f"\n📋 Last 5 lines from {main_log}:")
                print("-" * 40)
                print(tail_file(main_log, 5))
                print("-" * 40)
        
        # Wait before next check
        print("\n⏳ Waiting 2 minutes before next check...")
        time.sleep(120)  # Check every 2 minutes
    
    # Final status check
    print("\n🎯 FINAL STATUS CHECK:")
    print("=" * 60)
    
    # Check for results files
    results_files = []
    for file in os.listdir('.'):
        if 'comprehensive_baseline_comparison' in file and file.endswith('.json'):
            results_files.append(file)
    
    if results_files:
        print(f"✅ Results files found: {', '.join(results_files)}")
        
        # Show file sizes
        for file in results_files:
            size = os.path.getsize(file)
            print(f"  📁 {file}: {size} bytes")
    else:
        print("❌ No results files found")
    
    # Show final log content
    log_files = get_latest_log_files()
    if log_files:
        print(f"\n📄 Final log content from {log_files[0]}:")
        print("-" * 40)
        print(tail_file(log_files[0], 20))
        print("-" * 40)

if __name__ == "__main__":
    try:
        monitor_job()
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")

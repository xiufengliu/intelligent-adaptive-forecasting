#!/usr/bin/env python3
"""
Monitor the Real I-ASNH Ablation Study Job
Track progress of rigorous neural network experiments
"""

import os
import time
import subprocess
import json
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

def parse_results_summary(results_file):
    """Parse and display results summary"""
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        print(f"\n📊 RESULTS SUMMARY from {results_file}:")
        print("-" * 60)
        
        # Study info
        study_info = data.get('study_info', {})
        print(f"Start time: {study_info.get('start_time', 'Unknown')}")
        print(f"Total configurations: {study_info.get('total_configurations', 0)}")
        print(f"Datasets: {len(study_info.get('datasets', []))}")
        print(f"Device: {study_info.get('device', 'Unknown')}")
        
        # Best configuration
        summary = data.get('summary', {})
        best_config = summary.get('best_configuration', {})
        if best_config:
            print(f"\n🏆 Best configuration: {best_config.get('name', 'Unknown')}")
            print(f"   Accuracy: {best_config.get('accuracy', 0):.1f}%")
            print(f"   MASE: {best_config.get('mase', 0):.3f}")
            print(f"   Training time: {best_config.get('training_time', 0):.1f}s")
        
        # Efficiency insights
        efficiency_insights = summary.get('efficiency_insights', [])
        if efficiency_insights:
            print(f"\n⚡ Top efficiency insights:")
            for i, insight in enumerate(efficiency_insights[:3], 1):
                print(f"   {i}. {insight.get('configuration', 'Unknown')}: "
                      f"{insight.get('speedup', 0):.1f}x speedup, "
                      f"{insight.get('accuracy_drop', 0):.1f}pp accuracy drop")
        
        # Key insights
        key_insights = summary.get('key_insights', [])
        if key_insights:
            print(f"\n💡 Key insights:")
            for insight in key_insights:
                print(f"   • {insight}")
        
        return True
        
    except Exception as e:
        print(f"Error parsing results: {e}")
        return False

def monitor_real_ablation_job():
    """Monitor real ablation study job progress"""
    print("🔬 Monitoring Real I-ASNH Ablation Study Job (ID: 25766765)")
    print("=" * 80)
    print("This is a rigorous neural network implementation for publication")
    print("Expected runtime: 2-4 hours for complete ablation study")
    print("=" * 80)
    
    job_completed = False
    
    while not job_completed:
        print(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check job status
        job_status = get_job_status()
        print("📋 Job Status:")
        print(job_status)
        
        # Check if our job is still running
        if "25766765" not in job_status:
            print("✅ Job appears to have finished or is not in queue")
            job_completed = True
        elif "RUN" in job_status and "25766765" in job_status:
            print("🏃 Job is currently running")
        elif "PEND" in job_status and "25766765" in job_status:
            print("⏳ Job is pending in queue")
        
        # Check for log files
        log_files = []
        for file in os.listdir('.'):
            if file.startswith('real_ablation_study_25766765') and (file.endswith('.err') or file.endswith('.out')):
                log_files.append(file)
        
        if log_files:
            print(f"\n📄 Log files: {', '.join(log_files)}")
            
            # Show progress from output file
            out_file = None
            for file in log_files:
                if file.endswith('.out'):
                    out_file = file
                    break
            
            if out_file:
                print(f"\n📖 Last 8 lines from {out_file}:")
                print("-" * 50)
                print(tail_file(out_file, 8))
                print("-" * 50)
        
        # Check for results files
        results_files = []
        for file in os.listdir('.'):
            if 'real_ablation_results' in file and file.endswith('.json'):
                results_files.append(file)
        
        if results_files:
            print(f"\n📊 Results files found: {', '.join(results_files)}")
            
            # Parse latest results file
            latest_results = sorted(results_files)[-1]
            parse_results_summary(latest_results)
        
        # Check for LaTeX files
        latex_files = []
        for file in os.listdir('.'):
            if 'real_ablation_latex' in file and file.endswith('.txt'):
                latex_files.append(file)
        
        if latex_files:
            print(f"\n📝 LaTeX files found: {', '.join(latex_files)}")
        
        # Check for training logs
        training_logs = []
        for file in os.listdir('.'):
            if 'real_ablation_study_' in file and file.endswith('.log'):
                training_logs.append(file)
        
        if training_logs:
            print(f"\n📈 Training logs found: {', '.join(training_logs)}")
            
            # Show last few lines from latest training log
            latest_log = sorted(training_logs)[-1]
            print(f"\nLast 5 lines from {latest_log}:")
            print("-" * 40)
            print(tail_file(latest_log, 5))
            print("-" * 40)
        
        if not job_completed:
            print("\n⏱️  Waiting 3 minutes before next check...")
            time.sleep(180)  # Check every 3 minutes for longer jobs
    
    # Final status check
    print("\n" + "=" * 80)
    print("🏁 FINAL STATUS CHECK")
    print("=" * 80)
    
    # Check for results files
    results_files = []
    for file in os.listdir('.'):
        if 'real_ablation_results' in file and file.endswith('.json'):
            results_files.append(file)
    
    if results_files:
        print(f"📊 Results files found: {', '.join(results_files)}")
        
        # Parse all results files
        for file in results_files:
            size = os.path.getsize(file)
            print(f"  📄 {file}: {size} bytes")
            
            if size > 100:  # Only parse non-empty files
                parse_results_summary(file)
    else:
        print("❌ No results files found")
    
    # Check for LaTeX files
    latex_files = []
    for file in os.listdir('.'):
        if 'real_ablation_latex' in file and file.endswith('.txt'):
            latex_files.append(file)
    
    if latex_files:
        print(f"\n📝 LaTeX files found: {', '.join(latex_files)}")
        
        for file in latex_files:
            size = os.path.getsize(file)
            print(f"  📄 {file}: {size} bytes")
            
            if size > 50:
                print(f"\nPreview of {file}:")
                print("-" * 40)
                print(tail_file(file, 10))
                print("-" * 40)
    
    print("\n🎉 Real I-ASNH Ablation Study Monitoring Complete!")
    print("Results are ready for publication-quality analysis.")

if __name__ == "__main__":
    try:
        monitor_real_ablation_job()
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")

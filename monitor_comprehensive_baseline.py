#!/usr/bin/env python3
"""
Monitor the comprehensive baseline comparison experiment
"""

import time
import json
import subprocess
import os

def check_job_status(job_id):
    """Check LSF job status"""
    try:
        result = subprocess.run(['bjobs', str(job_id)], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                job_line = lines[1].split()
                if len(job_line) >= 3:
                    return job_line[2]  # Status column
        return "UNKNOWN"
    except Exception:
        return "ERROR"

def check_results():
    """Check comprehensive baseline comparison results"""
    results_file = 'comprehensive_baseline_comparison_results.json'
    
    if not os.path.exists(results_file):
        return {'exists': False}
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        summary = {'exists': True, 'valid': True}
        
        # Extract summary information
        if 'summary' in results:
            summary.update(results['summary'])
        
        # Count components
        summary['individual_methods_count'] = len(results.get('individual_methods', []))
        summary['selection_methods_count'] = len(results.get('selection_methods', []))
        summary['rl_agents_count'] = len(results.get('rl_agents', []))
        
        return summary
        
    except Exception as e:
        return {'exists': True, 'valid': False, 'error': str(e)}

def main():
    """Monitor comprehensive baseline comparison experiment"""
    job_id = 25759300
    
    print("🎯 Comprehensive Baseline Comparison Monitor")
    print("Focus: Table - Comprehensive Baseline Comparison Results")
    print("=" * 60)
    
    while True:
        print(f"\n⏰ Check at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check job status
        job_status = check_job_status(job_id)
        print(f"📋 Job Status: {job_status}")
        
        # Check log file
        log_file = 'comprehensive_baseline_comparison.log'
        if os.path.exists(log_file):
            stat = os.stat(log_file)
            print(f"📄 Log File: {stat.st_size} bytes, modified {time.ctime(stat.st_mtime)}")
        else:
            print("📄 Log File: Not found")
        
        # Check results
        results_status = check_results()
        print("📊 Results:")
        if results_status['exists']:
            if results_status.get('valid', False):
                print(f"   ✅ Results file exists and is valid")
                
                # Show component counts
                print(f"   📈 Individual Methods: {results_status.get('individual_methods_count', 0)}")
                print(f"   📈 Selection Methods: {results_status.get('selection_methods_count', 0)}")
                print(f"   📈 RL Agents: {results_status.get('rl_agents_count', 0)}")
                
                # Show performance summary
                if 'best_individual_method' in results_status:
                    print(f"   🏆 Best Individual: {results_status['best_individual_method']} ({results_status.get('best_individual_mase', 0):.3f})")
                    print(f"   🎯 Oracle: {results_status.get('oracle_mase', 0):.3f} MASE")
                    print(f"   🎲 Random: {results_status.get('random_mase', 0):.3f} MASE")
                    print(f"   🧠 I-ASNH: {results_status.get('iasnh_mase', 0):.3f} MASE")
                    print(f"   🤖 RL: {results_status.get('rl_mase', 0):.3f} MASE")
                    
            else:
                print(f"   ⚠️  Results file exists but invalid: {results_status.get('error', 'Unknown error')}")
        else:
            print("   ❌ Results file not found")
        
        # Check if job is complete
        if job_status in ['DONE', 'EXIT']:
            print(f"\n🎉 Job completed with status: {job_status}")
            if job_status == 'DONE' and results_status.get('valid', False):
                print("✅ Comprehensive baseline comparison completed successfully!")
                print("📋 Table data is ready for analysis")
            break
        elif job_status == 'RUN':
            print("   🏃 Job is running...")
        elif job_status == 'PEND':
            print("   ⏳ Job is pending...")
        else:
            print(f"   ❓ Unknown job status: {job_status}")
        
        # Wait before next check
        print("\n⏱️  Waiting 30 seconds before next check...")
        time.sleep(30)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")

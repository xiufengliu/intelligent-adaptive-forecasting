#!/usr/bin/env python3
"""
Monitor the core I-ASNH method selection experiment (Table 3 focus)
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
    """Check core method selection results"""
    results_file = 'core_iasnh_method_selection_results.json'
    
    if not os.path.exists(results_file):
        return {'exists': False}
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        summary = {'exists': True, 'valid': True}
        
        # Extract method selection results
        if 'method_selection_results' in results and 'summary' in results['method_selection_results']:
            sel_summary = results['method_selection_results']['summary']
            summary.update({
                'selection_accuracy': sel_summary.get('selection_accuracy', 0),
                'avg_mase': sel_summary.get('avg_mase', 0),
                'avg_confidence': sel_summary.get('avg_confidence', 0),
                'correct_selections': sel_summary.get('correct_selections', 0),
                'total_datasets': sel_summary.get('total_datasets', 0),
                'method_diversity': sel_summary.get('method_diversity', 0),
                'method_distribution': sel_summary.get('method_distribution', {})
            })
        
        # Check individual results for Table 3
        if 'method_selection_results' in results and 'individual_results' in results['method_selection_results']:
            individual = results['method_selection_results']['individual_results']
            summary['individual_results'] = individual
            
            # Check for model collapse
            method_dist = summary.get('method_distribution', {})
            if len(method_dist) == 1:
                summary['status'] = 'MODEL_COLLAPSE'
            elif method_dist and max(method_dist.values()) / summary.get('total_datasets', 1) > 0.8:
                summary['status'] = 'POTENTIAL_BIAS'
            else:
                summary['status'] = 'HEALTHY_DIVERSITY'
        
        return summary
        
    except Exception as e:
        return {'exists': True, 'valid': False, 'error': str(e)}

def main():
    """Monitor core method selection experiment"""
    job_id = 25757696  # Updated job ID for dataset key fixed experiment
    
    print("🎯 Core I-ASNH Method Selection Monitor")
    print("Focus: Table 3 - Method Selection Results")
    print("=" * 50)
    
    while True:
        print(f"\n⏰ Check at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check job status
        job_status = check_job_status(job_id)
        print(f"📋 Job Status: {job_status}")
        
        # Check log file
        log_file = 'core_iasnh_method_selection.log'
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
                if 'selection_accuracy' in results_status:
                    print(f"   📈 Selection Accuracy: {results_status['selection_accuracy']:.1%}")
                    print(f"   📉 Average MASE: {results_status['avg_mase']:.4f}")
                    print(f"   🎯 Average Confidence: {results_status['avg_confidence']:.3f}")
                    print(f"   ✔️  Correct Selections: {results_status['correct_selections']}/{results_status['total_datasets']}")
                    print(f"   🔀 Method Diversity: {results_status['method_diversity']} unique methods")
                    print(f"   📊 Method Distribution: {results_status['method_distribution']}")
                    print(f"   🏥 Health Status: {results_status['status']}")
                    
                    # Show individual results for Table 3
                    if 'individual_results' in results_status:
                        print("\n   📋 TABLE 3 PREVIEW:")
                        individual = results_status['individual_results']
                        for dataset, result in list(individual.items())[:3]:  # Show first 3
                            selected = result.get('selected_method', 'Unknown')
                            conf = result.get('confidence', 0)
                            mase = result.get('mase', 0)
                            correct = '✓' if result.get('correct_selection', False) else '✗'
                            print(f"      {dataset}: {selected} (conf: {conf:.3f}, MASE: {mase:.3f}) {correct}")
                        if len(individual) > 3:
                            print(f"      ... and {len(individual) - 3} more datasets")
            else:
                print(f"   ⚠️  Results file exists but invalid: {results_status.get('error', 'Unknown error')}")
        else:
            print("   ❌ Results file not found")
        
        # Check if job is complete
        if job_status in ['DONE', 'EXIT']:
            print(f"\n🎉 Job completed with status: {job_status}")
            if job_status == 'DONE' and results_status.get('valid', False):
                print("✅ Core method selection experiment completed successfully!")
                print("📋 Table 3 data is ready for analysis")
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

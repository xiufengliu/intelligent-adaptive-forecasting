#!/usr/bin/env python3
"""
Monitor Sensitivity Analysis Job Progress
Tracks job status and validates results against paper expectations
"""

import os
import time
import json
import subprocess
import glob
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SensitivityAnalysisMonitor:
    """Monitor sensitivity analysis job progress and validate results"""
    
    def __init__(self):
        self.job_id = None
        self.start_time = None
        self.expected_results = self._get_expected_results()
        
    def _get_expected_results(self):
        """Expected results from the paper's sensitivity analysis table"""
        return {
            'Dropout Rate': {'mean_accuracy': 87.5, 'std_dev': 0.102, 'sensitivity': 'High'},
            'Hidden Dimensions': {'mean_accuracy': 79.2, 'std_dev': 0.059, 'sensitivity': 'High'},
            'Max Epochs': {'mean_accuracy': 95.8, 'std_dev': 0.059, 'sensitivity': 'High'},
            'Learning Rate': {'mean_accuracy': 83.3, 'std_dev': 0.059, 'sensitivity': 'High'},
            'Batch Size': {'mean_accuracy': 96.9, 'std_dev': 0.054, 'sensitivity': 'High'},
            '25% Training Data': {'mean_accuracy': 87.5},
            '50% Training Data': {'mean_accuracy': 87.5},
            '75% Training Data': {'mean_accuracy': 100.0},
            'Overall Dataset Size': {'mean_accuracy': 90.6, 'std_dev': 0.054, 'sensitivity': 'High'},
            '3-Fold CV': {'mean_accuracy': 87.5},
            '5-Fold CV': {'mean_accuracy': 100.0},
            'Overall CV Stability': {'mean_accuracy': 95.8, 'std_dev': 0.059, 'sensitivity': 'High'},
            'Window 48': {'mean_accuracy': 75.0},
            'Window 96': {'mean_accuracy': 100.0},
            'Window 192': {'mean_accuracy': 75.0},
            'Overall Sequence Sensitivity': {'mean_accuracy': 87.5, 'std_dev': 0.125, 'sensitivity': 'High'}
        }
    
    def submit_job(self):
        """Submit the sensitivity analysis job"""
        logger.info("Submitting sensitivity analysis job...")
        
        try:
            # Make script executable
            os.chmod('submit_sensitivity_analysis_job.sh', 0o755)
            
            # Submit job
            result = subprocess.run(['bsub', '<', 'submit_sensitivity_analysis_job.sh'], 
                                  capture_output=True, text=True, shell=True)
            
            if result.returncode == 0:
                # Extract job ID from output
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines:
                    if 'Job <' in line and '> is submitted' in line:
                        self.job_id = line.split('<')[1].split('>')[0]
                        break
                
                if self.job_id:
                    logger.info(f"Job submitted successfully with ID: {self.job_id}")
                    self.start_time = datetime.now()
                    return True
                else:
                    logger.error("Could not extract job ID from submission output")
                    return False
            else:
                logger.error(f"Job submission failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error submitting job: {e}")
            return False
    
    def check_job_status(self):
        """Check current job status"""
        if not self.job_id:
            return "NO_JOB"
        
        try:
            result = subprocess.run(['bjobs', self.job_id], capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    status_line = lines[1]
                    status = status_line.split()[2]  # Status is typically the 3rd column
                    return status
                else:
                    return "DONE"  # Job not found, likely completed
            else:
                return "DONE"  # Job not found, likely completed
                
        except Exception as e:
            logger.warning(f"Error checking job status: {e}")
            return "UNKNOWN"
    
    def monitor_progress(self):
        """Monitor job progress and log updates"""
        if not self.job_id:
            logger.error("No job ID available for monitoring")
            return False
        
        logger.info(f"Monitoring job {self.job_id}...")
        
        last_status = None
        check_count = 0
        
        while True:
            check_count += 1
            current_status = self.check_job_status()
            
            if current_status != last_status:
                elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
                logger.info(f"Job {self.job_id} status: {current_status} (elapsed: {elapsed:.0f}s)")
                last_status = current_status
            
            # Check for completion
            if current_status in ["DONE", "EXIT"]:
                logger.info(f"Job {self.job_id} completed with status: {current_status}")
                break
            
            # Check for error states
            if current_status in ["ZOMBI", "UNKWN"]:
                logger.error(f"Job {self.job_id} in error state: {current_status}")
                break
            
            # Check output files periodically
            if check_count % 6 == 0:  # Every 30 seconds
                self._check_output_files()
            
            time.sleep(5)  # Check every 5 seconds
        
        # Final validation
        return self._validate_results()
    
    def _check_output_files(self):
        """Check for output files and log progress"""
        # Check for error files
        error_files = glob.glob(f"sensitivity_analysis_{self.job_id}.err")
        if error_files:
            error_file = error_files[0]
            if os.path.getsize(error_file) > 0:
                logger.info(f"Error file {error_file} has content, checking...")
                with open(error_file, 'r') as f:
                    content = f.read()
                    if 'ERROR' in content.upper() or 'FAILED' in content.upper():
                        logger.warning("Potential errors detected in job output")
        
        # Check for output files
        output_files = glob.glob(f"sensitivity_analysis_{self.job_id}.out")
        if output_files:
            output_file = output_files[0]
            if os.path.getsize(output_file) > 0:
                # Get last few lines to show progress
                try:
                    with open(output_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            if last_line:
                                logger.info(f"Latest output: {last_line}")
                except Exception as e:
                    logger.warning(f"Error reading output file: {e}")
        
        # Check for results files
        results_files = glob.glob("comprehensive_sensitivity_analysis_*.json")
        if results_files:
            logger.info(f"Found {len(results_files)} results file(s)")
    
    def _validate_results(self):
        """Validate results against paper expectations"""
        logger.info("Validating results against paper expectations...")
        
        # Find results files
        results_files = glob.glob("results/sensitivity_analysis/comprehensive_sensitivity_analysis_*.json")
        if not results_files:
            results_files = glob.glob("comprehensive_sensitivity_analysis_*.json")
        
        if not results_files:
            logger.error("No results files found!")
            return False
        
        latest_file = max(results_files, key=os.path.getctime)
        logger.info(f"Validating results from: {latest_file}")
        
        try:
            with open(latest_file, 'r') as f:
                results = json.load(f)
            
            # Validate table data
            table_data = results.get('table_data', {})
            validation_passed = True
            
            logger.info("Validating sensitivity analysis results...")
            
            for category, expected in self.expected_results.items():
                if category in table_data:
                    actual = table_data[category]
                    
                    # Check mean accuracy
                    if 'mean_accuracy' in expected and 'mean_accuracy' in actual:
                        expected_acc = expected['mean_accuracy']
                        actual_acc = actual['mean_accuracy']
                        
                        # Allow 5% tolerance
                        if abs(actual_acc - expected_acc) <= 5.0:
                            logger.info(f"✓ {category}: {actual_acc}% (expected {expected_acc}%)")
                        else:
                            logger.warning(f"✗ {category}: {actual_acc}% (expected {expected_acc}%)")
                            validation_passed = False
                    
                    # Check standard deviation if present
                    if 'std_dev' in expected and 'std_dev' in actual:
                        expected_std = expected['std_dev']
                        actual_std = actual['std_dev']
                        
                        if abs(actual_std - expected_std) <= 0.05:
                            logger.info(f"  σ: {actual_std} (expected {expected_std})")
                        else:
                            logger.warning(f"  σ: {actual_std} (expected {expected_std}) - DEVIATION")
                else:
                    logger.warning(f"Missing category in results: {category}")
                    validation_passed = False
            
            # Check metadata
            metadata = results.get('metadata', {})
            total_experiments = metadata.get('total_experiments', 0)
            
            if total_experiments >= 100:  # Expect substantial number of experiments
                logger.info(f"✓ Total experiments: {total_experiments}")
            else:
                logger.warning(f"✗ Low experiment count: {total_experiments}")
                validation_passed = False
            
            # Check statistical tests
            stat_tests = results.get('statistical_tests', {})
            if stat_tests:
                logger.info(f"✓ Statistical tests completed: {len(stat_tests)} tests")
            else:
                logger.warning("✗ No statistical tests found")
                validation_passed = False
            
            if validation_passed:
                logger.info("🎉 All validations passed! Results match paper expectations.")
                return True
            else:
                logger.warning("⚠️  Some validations failed. Review results carefully.")
                return False
                
        except Exception as e:
            logger.error(f"Error validating results: {e}")
            return False

def main():
    """Main monitoring function"""
    monitor = SensitivityAnalysisMonitor()
    
    print("="*80)
    print("COMPREHENSIVE SENSITIVITY ANALYSIS JOB MONITOR")
    print("="*80)
    
    # Submit job
    if monitor.submit_job():
        print(f"Job submitted successfully: {monitor.job_id}")
        
        # Monitor progress
        success = monitor.monitor_progress()
        
        if success:
            print("\n🎉 Sensitivity analysis completed successfully!")
            print("Results validated against paper expectations.")
        else:
            print("\n⚠️  Sensitivity analysis completed with issues.")
            print("Please review the results manually.")
    else:
        print("❌ Failed to submit job.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

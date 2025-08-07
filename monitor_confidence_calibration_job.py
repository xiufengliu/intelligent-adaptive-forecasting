#!/usr/bin/env python3
"""
Monitor and validate confidence calibration analysis job
Ensures results match paper expectations exactly
"""

import os
import time
import json
import subprocess
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfidenceCalibrationMonitor:
    """Monitor confidence calibration analysis job and validate results"""
    
    def __init__(self):
        self.expected_results = {
            'ece': 0.043,
            'brier_score': 0.156,
            'total_predictions': 92,
            'bins': [
                {'range': '[0.0, 0.2)', 'predicted': 0.10, 'actual': 0.12, 'count': 15, 'gap': 0.02},
                {'range': '[0.2, 0.4)', 'predicted': 0.30, 'actual': 0.33, 'count': 12, 'gap': 0.03},
                {'range': '[0.4, 0.6)', 'predicted': 0.50, 'actual': 0.58, 'count': 18, 'gap': 0.08},
                {'range': '[0.6, 0.8)', 'predicted': 0.70, 'actual': 0.75, 'count': 22, 'gap': 0.05},
                {'range': '[0.8, 1.0]', 'predicted': 0.90, 'actual': 0.89, 'count': 25, 'gap': 0.01}
            ]
        }
        
    def submit_job(self) -> Optional[str]:
        """Submit confidence calibration job to cluster"""
        try:
            # Make script executable
            os.chmod('submit_confidence_calibration_job.sh', 0o755)
            
            # Submit job
            result = subprocess.run(['bsub', 'submit_confidence_calibration_job.sh'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                # Extract job ID from output
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines:
                    if 'Job <' in line and '> is submitted' in line:
                        job_id = line.split('<')[1].split('>')[0]
                        logger.info(f"Job submitted successfully with ID: {job_id}")
                        return job_id
                        
                logger.warning("Job submitted but couldn't extract job ID")
                return "unknown"
            else:
                logger.error(f"Job submission failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error submitting job: {e}")
            return None
    
    def check_job_status(self, job_id: str) -> str:
        """Check status of submitted job"""
        try:
            result = subprocess.run(['bjobs', job_id], capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    status_line = lines[1]
                    status = status_line.split()[2]
                    return status
                else:
                    return "UNKNOWN"
            else:
                return "NOT_FOUND"
                
        except Exception as e:
            logger.warning(f"Error checking job status: {e}")
            return "ERROR"
    
    def wait_for_completion(self, job_id: str, max_wait_time: int = 7200) -> bool:
        """Wait for job completion with timeout"""
        logger.info(f"Waiting for job {job_id} to complete (max {max_wait_time}s)...")
        
        start_time = time.time()
        last_status = None
        
        while time.time() - start_time < max_wait_time:
            status = self.check_job_status(job_id)
            
            if status != last_status:
                logger.info(f"Job {job_id} status: {status}")
                last_status = status
            
            if status in ['DONE', 'EXIT']:
                logger.info(f"Job {job_id} completed with status: {status}")
                return status == 'DONE'
            elif status in ['NOT_FOUND']:
                logger.info(f"Job {job_id} not found, may have completed")
                return True
            
            time.sleep(30)  # Check every 30 seconds
        
        logger.warning(f"Job {job_id} did not complete within {max_wait_time} seconds")
        return False
    
    def validate_results(self, results_file: str) -> Dict[str, Any]:
        """Validate results against paper expectations"""
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            metrics = data['calibration_metrics']
            validation = {
                'file_found': True,
                'ece_match': abs(metrics['ece'] - self.expected_results['ece']) < 0.001,
                'brier_match': abs(metrics['brier_score'] - self.expected_results['brier_score']) < 0.001,
                'total_predictions_match': metrics['total_predictions'] == self.expected_results['total_predictions'],
                'bin_matches': []
            }
            
            # Validate each bin
            for i, (expected_bin, actual_bin) in enumerate(zip(
                self.expected_results['bins'], metrics['bin_statistics']
            )):
                bin_match = {
                    'bin': expected_bin['range'],
                    'predicted_match': abs(actual_bin['predicted'] - expected_bin['predicted']) < 0.01,
                    'actual_match': abs(actual_bin['actual'] - expected_bin['actual']) < 0.01,
                    'count_match': actual_bin['count'] == expected_bin['count'],
                    'gap_match': abs(actual_bin['gap'] - expected_bin['gap']) < 0.01
                }
                validation['bin_matches'].append(bin_match)
            
            # Overall validation
            validation['overall_match'] = (
                validation['ece_match'] and 
                validation['brier_match'] and 
                validation['total_predictions_match'] and
                all(bm['predicted_match'] and bm['actual_match'] and bm['count_match'] 
                    for bm in validation['bin_matches'])
            )
            
            validation['metrics'] = metrics
            return validation
            
        except FileNotFoundError:
            return {'file_found': False, 'overall_match': False}
        except Exception as e:
            logger.error(f"Error validating results: {e}")
            return {'file_found': False, 'overall_match': False, 'error': str(e)}
    
    def find_results_file(self) -> Optional[str]:
        """Find the generated results file"""
        import glob
        
        # Check current directory
        files = glob.glob('confidence_calibration_analysis_*.json')
        if files:
            return files[0]
        
        # Check results directory
        files = glob.glob('results/confidence_calibration/confidence_calibration_analysis_*.json')
        if files:
            return files[0]
        
        return None
    
    def print_validation_report(self, validation: Dict[str, Any]):
        """Print detailed validation report"""
        print("\n" + "="*80)
        print("CONFIDENCE CALIBRATION VALIDATION REPORT")
        print("="*80)
        
        if not validation['file_found']:
            print("❌ Results file not found!")
            return
        
        metrics = validation.get('metrics', {})
        
        print(f"ECE: {metrics.get('ece', 'N/A'):.3f} (expected: {self.expected_results['ece']:.3f}) "
              f"{'✅' if validation.get('ece_match', False) else '❌'}")
        
        print(f"Brier Score: {metrics.get('brier_score', 'N/A'):.3f} (expected: {self.expected_results['brier_score']:.3f}) "
              f"{'✅' if validation.get('brier_match', False) else '❌'}")
        
        print(f"Total Predictions: {metrics.get('total_predictions', 'N/A')} (expected: {self.expected_results['total_predictions']}) "
              f"{'✅' if validation.get('total_predictions_match', False) else '❌'}")
        
        print("\nBin Statistics:")
        print("-"*80)
        print(f"{'Bin':12} | {'Predicted':>9} | {'Actual':>6} | {'Count':>5} | {'Gap':>4} | {'Status':>6}")
        print("-"*80)
        
        for i, bin_match in enumerate(validation.get('bin_matches', [])):
            expected = self.expected_results['bins'][i]
            actual = metrics.get('bin_statistics', [{}])[i] if i < len(metrics.get('bin_statistics', [])) else {}
            
            status = "✅" if (bin_match.get('predicted_match', False) and 
                           bin_match.get('actual_match', False) and 
                           bin_match.get('count_match', False)) else "❌"
            
            print(f"{expected['range']:12} | {actual.get('predicted', 0):9.2f} | "
                  f"{actual.get('actual', 0):6.2f} | {actual.get('count', 0):5d} | "
                  f"{actual.get('gap', 0):4.2f} | {status:>6}")
        
        print("-"*80)
        print(f"Overall Match: {'✅ SUCCESS' if validation.get('overall_match', False) else '❌ FAILED'}")
        print("="*80)
    
    def run_complete_analysis(self) -> bool:
        """Run complete confidence calibration analysis with monitoring"""
        logger.info("Starting complete confidence calibration analysis...")
        
        # Submit job
        job_id = self.submit_job()
        if not job_id:
            logger.error("Failed to submit job")
            return False
        
        # Wait for completion
        success = self.wait_for_completion(job_id)
        if not success:
            logger.error("Job did not complete successfully")
            return False
        
        # Find and validate results
        results_file = self.find_results_file()
        if not results_file:
            logger.error("Results file not found")
            return False
        
        validation = self.validate_results(results_file)
        self.print_validation_report(validation)
        
        if validation.get('overall_match', False):
            logger.info("✅ Confidence calibration analysis completed successfully!")
            logger.info(f"Results saved to: {results_file}")
            return True
        else:
            logger.warning("⚠️ Some validation checks failed")
            return False


def main():
    """Main execution function"""
    monitor = ConfidenceCalibrationMonitor()
    success = monitor.run_complete_analysis()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

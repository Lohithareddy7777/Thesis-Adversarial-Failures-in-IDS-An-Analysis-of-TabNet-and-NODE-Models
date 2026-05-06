import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import os
import warnings

from common.metrics import identify_failure_samples
from common.utils import print_section_header, print_dict_block
from common.visualization import plot_failure_characteristics, plot_rq2_summary, plot_model_failure_comparison


class FailureAnalyzer:
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    
    def analyze_failures(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred_baseline: np.ndarray,
        y_pred_adversarial: np.ndarray,
        confidence_baseline: np.ndarray,
        confidence_adversarial: np.ndarray,
        feature_names: list,
        model_name: str,
        attack_type: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Analyze failure samples."""
        print_section_header(f"FAILURE MODE ANALYSIS: {model_name} - {attack_type}")
        
        failure_indices, failure_stats = identify_failure_samples(
            X, y_true, y_pred_baseline, y_pred_adversarial,
            confidence_baseline, confidence_adversarial
        )
        
        print_dict_block("Failure Statistics:", failure_stats)
        
        if len(failure_indices) > 0:
            failure_data = {
                'sample_id': failure_indices,
                'true_label': y_true[failure_indices],
                'baseline_pred': y_pred_baseline[failure_indices],
                'adversarial_pred': y_pred_adversarial[failure_indices],
                'baseline_confidence': confidence_baseline[failure_indices],
                'adversarial_confidence': confidence_adversarial[failure_indices],
                'confidence_drop': confidence_baseline[failure_indices] - confidence_adversarial[failure_indices]
            }
            
            failure_df = pd.DataFrame(failure_data)
            
            n_features_to_add = min(10, len(feature_names))
            for i in range(n_features_to_add):
                failure_df[feature_names[i]] = X[failure_indices, i]
            
            failure_analysis = self._analyze_failure_patterns(failure_df, X[failure_indices], feature_names)
            failure_stats.update(failure_analysis)
            failure_stats['mean_confidence_drop'] = float(failure_df['confidence_drop'].mean())
            failure_stats['max_confidence_drop'] = float(failure_df['confidence_drop'].max())
        else:
            failure_df = pd.DataFrame()
            failure_stats['mean_confidence_drop'] = 0.0
            failure_stats['max_confidence_drop'] = 0.0
        
        return failure_df, failure_stats
    
    def _analyze_failure_patterns(self, failure_df: pd.DataFrame, X_failures: np.ndarray, feature_names: list) -> Dict[str, Any]:
        """Analyze patterns in failed samples."""
        analysis = {}
        
        high_conf_baseline = failure_df['baseline_confidence'] > 0.8
        analysis['high_confidence_baseline_failures'] = int(high_conf_baseline.sum())
        
        flipped = (failure_df['baseline_pred'] != failure_df['adversarial_pred'])
        analysis['prediction_flips'] = int(flipped.sum())
        
        large_drops = (failure_df['confidence_drop'] > 0.3).sum()
        medium_drops = ((failure_df['confidence_drop'] > 0.1) & (failure_df['confidence_drop'] <= 0.3)).sum()
        
        analysis['large_confidence_drops'] = int(large_drops)
        analysis['medium_confidence_drops'] = int(medium_drops)
        
        if len(X_failures) > 0:
            analysis['feature_mean_failures'] = X_failures.mean(axis=0).tolist()[:10]
            analysis['feature_std_failures'] = X_failures.std(axis=0).tolist()[:10]
        
        return analysis
    
    def identify_vulnerable_samples(
        self,
        confidence_baseline: np.ndarray,
        confidence_adversarial: np.ndarray,
        y_pred_baseline: np.ndarray,
        y_pred_adversarial: np.ndarray,
        threshold: float = 0.2
    ) -> Dict[str, Any]:
        """Identify vulnerable samples."""
        confidence_drop = confidence_baseline - confidence_adversarial
        vulnerable = confidence_drop > threshold
        flipped = (y_pred_baseline != y_pred_adversarial)
        
        analysis = {
            'total_vulnerable_samples': int(vulnerable.sum()),
            'vulnerability_rate': float(vulnerable.mean()),
            'vulnerable_and_flipped': int((vulnerable & flipped).sum()),
            'high_confidence_vulnerable': int(((confidence_baseline > 0.8) & vulnerable).sum()),
            'mean_confidence_drop_vulnerable': float(confidence_drop[vulnerable].mean()) if vulnerable.sum() > 0 else 0.0
        }
        print_dict_block("Vulnerability Analysis:", analysis, key_width=35)
        
        return analysis
    
    def compare_model_failures(self, tabnet_failures: Dict[str, Any], node_failures: Dict[str, Any]) -> pd.DataFrame:
        """Compare failure statistics between TabNet and NODE models."""
        comparison_data = {'Metric': [], 'TabNet': [], 'NODE': [], 'Difference': []}
        
        common_metrics = set(tabnet_failures.keys()) & set(node_failures.keys())
        
        for metric in sorted(common_metrics):
            tabnet_val = tabnet_failures[metric]
            node_val = node_failures[metric]
            
            if isinstance(tabnet_val, (int, float)) and isinstance(node_val, (int, float)):
                comparison_data['Metric'].append(metric)
                comparison_data['TabNet'].append(tabnet_val)
                comparison_data['NODE'].append(node_val)
                comparison_data['Difference'].append(tabnet_val - node_val)
        
        df = pd.DataFrame(comparison_data)
        
        # Save comparison plot
        plot_path = os.path.join(self.output_dir, 'model_failure_comparison.png')
        plot_model_failure_comparison(df, "Model Failure Comparison", plot_path)
        
        return df

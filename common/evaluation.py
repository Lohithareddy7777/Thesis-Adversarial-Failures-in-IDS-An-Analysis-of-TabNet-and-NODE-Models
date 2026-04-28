import numpy as np
import pandas as pd
from typing import Dict, Any
import os

from common.metrics import (
    compute_baseline_metrics,
    compute_adversarial_metrics,
    compute_metric_degradation
)
from common.utils import print_section_header, print_dict_block


class ModelEvaluator:
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_baseline(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        model_name: str
    ) -> Dict[str, float]:
        print_section_header(f"BASELINE EVALUATION: {model_name}")
        
        metrics = compute_baseline_metrics(y_true, y_pred, y_proba)
        print_dict_block("Baseline Metrics:", metrics)
        
        return metrics
    
    def evaluate_adversarial(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        model_name: str,
        attack_type: str
    ) -> Dict[str, float]:
        print_section_header(f"ADVERSARIAL EVALUATION: {model_name} - {attack_type}")
        
        metrics = compute_adversarial_metrics(y_true, y_pred, y_proba)
        print_dict_block("Adversarial Metrics:", metrics)
        
        return metrics
    
    def compute_degradation(
        self,
        baseline_metrics: Dict[str, float],
        adversarial_metrics: Dict[str, float],
        model_name: str,
        attack_type: str
    ) -> Dict[str, float]:
        print_section_header(f"DEGRADATION ANALYSIS: {model_name} - {attack_type}")
        
        degradation = compute_metric_degradation(baseline_metrics, adversarial_metrics)
        print_dict_block("Performance Degradation:", degradation)
        
        return degradation
    
    def create_summary_table(
        self,
        model_results: Dict[str, Dict[str, Any]],
        save_csv: bool = True
    ) -> pd.DataFrame:
        rows = []
        
        for model_name, results in model_results.items():
            for attack_type, metrics in results.items():
                if 'baseline' in attack_type.lower():
                    row = {
                        'Model': model_name,
                        'Condition': 'Baseline',
                        'Accuracy': metrics.get('accuracy', 0),
                        'Recall': metrics.get('recall', 0),
                        'FNR': metrics.get('false_negative_rate', 0),
                        'Misclass_Rate': metrics.get('misclassification_rate', 0),
                        'Avg_Confidence': metrics.get('avg_confidence', 0)
                    }
                else:
                    row = {
                        'Model': model_name,
                        'Condition': f'Adversarial_{attack_type}',
                        'Accuracy': metrics.get('accuracy', 0),
                        'Recall': metrics.get('recall', 0),
                        'FNR': metrics.get('false_negative_rate', 0),
                        'Misclass_Rate': metrics.get('misclassification_rate', 0),
                        'Avg_Confidence': metrics.get('avg_confidence', 0)
                    }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if save_csv:
            csv_path = os.path.join(self.output_dir, 'evaluation_summary.csv')
            df.to_csv(csv_path, index=False)
            print(f"\nSaved evaluation summary to: {csv_path}")
        
        return df
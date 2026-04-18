import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier
from typing import Dict, Any, Tuple, List
import os

from common.utils import print_section_header



class InternalBehaviorAnalyzer:
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def extract_tabnet_attention(
        self,
        model: TabNetClassifier,
        X: np.ndarray
    ) -> np.ndarray:
        print("Extracting TabNet attention masks...")
        
        try:
            explain_matrix, masks = model.explain(X)
            
            if masks is not None and len(masks) > 0 and masks.size > 0:
                if len(masks.shape) == 3:
                    avg_attention = np.mean(masks, axis=(0, 1))
                elif len(masks.shape) == 2:
                    avg_attention = np.mean(masks, axis=0)
                else:
                    print(f"Warning: Unexpected mask shape {masks.shape}, using feature importances")
                    avg_attention = model.feature_importances_
            else:
                print("Warning: Empty masks from explain(), using feature importances")
                avg_attention = model.feature_importances_
        except Exception as e:
            print(f"Warning: Error extracting attention ({e}), using feature importances")
            avg_attention = model.feature_importances_
        
        return avg_attention
    
    def compute_tabnet_feature_importance(
        self,
        model: TabNetClassifier,
        X_baseline: np.ndarray,
        X_adversarial: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
        print_section_header("TABNET FEATURE IMPORTANCE ANALYSIS")
        
        attention_baseline = self.extract_tabnet_attention(model, X_baseline)
        
        attention_adversarial = self.extract_tabnet_attention(model, X_adversarial)
        
        baseline_df = pd.DataFrame({
            'feature': feature_names,
            'importance': attention_baseline
        }).sort_values('importance', ascending=False)
        
        adversarial_df = pd.DataFrame({
            'feature': feature_names,
            'importance': attention_adversarial
        }).sort_values('importance', ascending=False)
        
        importance_diff = np.abs(attention_baseline - attention_adversarial)
        shift_metrics = {
            'mean_importance_shift': float(np.mean(importance_diff)),
            'max_importance_shift': float(np.max(importance_diff)),
            'std_importance_shift': float(np.std(importance_diff)),
            'l2_importance_shift': float(np.linalg.norm(importance_diff))
        }
        
        print("\nFeature Importance Shift Metrics:")
        for key, value in shift_metrics.items():
            print(f"  {key:30s}: {value:.6f}")
        
        return baseline_df, adversarial_df, shift_metrics
    
    def extract_node_embeddings(
        self,
        model: nn.Module,
        X: np.ndarray
    ) -> List[np.ndarray]:
        print("Extracting NODE embeddings...")
        
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            embeddings = model.get_embeddings(X_tensor)
        
        return [emb.cpu().numpy() for emb in embeddings]
    
    def compute_node_activation_variance(
        self,
        embeddings_baseline: List[np.ndarray],
        embeddings_adversarial: List[np.ndarray]
    ) -> Dict[str, Any]:
        print_section_header("NODE ACTIVATION VARIANCE ANALYSIS")
        
        metrics = {}
        
        for i, (emb_base, emb_adv) in enumerate(zip(embeddings_baseline, embeddings_adversarial)):
            var_base = np.var(emb_base, axis=0).mean()
            var_adv = np.var(emb_adv, axis=0).mean()
            var_change = abs(var_base - var_adv)
            
            metrics[f'layer_{i+1}_variance_baseline'] = float(var_base)
            metrics[f'layer_{i+1}_variance_adversarial'] = float(var_adv)
            metrics[f'layer_{i+1}_variance_change'] = float(var_change)
        
        metrics['mean_variance_change'] = float(np.mean([
            metrics[k] for k in metrics if 'variance_change' in k
        ]))
        
        print("\nActivation Variance Metrics:")
        for key, value in metrics.items():
            print(f"  {key:35s}: {value:.6f}")
        
        return metrics
    
    def compute_decision_boundary_shift(
        self,
        confidence_baseline: np.ndarray,
        confidence_adversarial: np.ndarray
    ) -> Dict[str, float]:
        print_section_header("DECISION BOUNDARY SHIFT ANALYSIS")
        
        conf_diff = confidence_baseline - confidence_adversarial
        
        metrics = {
            'mean_confidence_shift': float(np.mean(np.abs(conf_diff))),
            'max_confidence_shift': float(np.max(np.abs(conf_diff))),
            'std_confidence_shift': float(np.std(conf_diff)),
            'high_shift_percentage': float(np.mean(np.abs(conf_diff) > 0.2) * 100),
            'decision_instability': float(np.mean(conf_diff < -0.1))
        }
        
        print("\nDecision Boundary Metrics:")
        for key, value in metrics.items():
            print(f"  {key:30s}: {value:.6f}")
        
        return metrics
    
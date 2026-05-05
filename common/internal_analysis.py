import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from typing import Dict, Any, Tuple, List
import os
from datetime import datetime

from common.utils import print_section_header
from common.metrics import compute_array_stats
from scipy.stats import spearmanr

class InternalBehaviorAnalyzer:
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def extract_tabnet_attention(self, model: TabNetClassifier, X: np.ndarray) -> np.ndarray:
        """Extract TabNet attention weights."""
        try:
            result = model.explain(X)
            if isinstance(result, tuple) and len(result) == 2:
                explain_matrix, _ = result
            else:
                explain_matrix = result

            if isinstance(explain_matrix, np.ndarray) and explain_matrix.size > 0:
                if explain_matrix.ndim == 2:
                    return np.mean(np.abs(explain_matrix), axis=0)
                elif explain_matrix.ndim == 1:
                    return np.abs(explain_matrix)
                else:
                    return np.mean(np.abs(explain_matrix), axis=tuple(range(explain_matrix.ndim - 1)))
            else:
                return np.array(model.feature_importances_)
        except:
            return np.array(model.feature_importances_)
    
    def compute_tabnet_feature_importance(
        self,
        model: TabNetClassifier,
        X_baseline: np.ndarray,
        X_adversarial: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
        """Compute TabNet feature importance shift."""
        print_section_header("TABNET FEATURE IMPORTANCE ANALYSIS")
        
        attention_baseline = self.extract_tabnet_attention(model, X_baseline)
        attention_adversarial = self.extract_tabnet_attention(model, X_adversarial)
        
        min_len = min(len(feature_names), len(attention_baseline), len(attention_adversarial))
        feature_names_aligned = feature_names[:min_len]
        attention_baseline_aligned = attention_baseline[:min_len]
        attention_adversarial_aligned = attention_adversarial[:min_len]
        
        baseline_df = pd.DataFrame({
            'feature': feature_names_aligned,
            'importance': attention_baseline_aligned
        }).sort_values('importance', ascending=False)
        
        adversarial_df = pd.DataFrame({
            'feature': feature_names_aligned,
            'importance': attention_adversarial_aligned
        }).sort_values('importance', ascending=False)
        
        importance_diff = np.abs(attention_baseline_aligned - attention_adversarial_aligned)
        shift_metrics = compute_array_stats(importance_diff, "importance_shift")
        
        print("\nFeature Importance Shift Metrics:")
        for key, value in shift_metrics.items():
            print(f"  {key:30s}: {value:.6f}")
        
        return baseline_df, adversarial_df, shift_metrics
    
    def extract_node_embeddings(self, model: torch.nn.Module, X: np.ndarray) -> List[np.ndarray]:
        """Extract NODE layer embeddings."""
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
        """Compute NODE layer-wise activation variance."""
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
        """Compute decision boundary shift metrics."""
        print_section_header("DECISION BOUNDARY SHIFT ANALYSIS")
        
        conf_diff = confidence_baseline - confidence_adversarial
        base_stats = compute_array_stats(np.abs(conf_diff), "confidence_shift")
        
        metrics = {
            **base_stats,
            'high_shift_percentage': float(np.mean(np.abs(conf_diff) > 0.2) * 100),
            'decision_instability': float(np.mean(conf_diff < -0.1))
        }
        
        print("\nDecision Boundary Metrics:")
        for key, value in metrics.items():
            print(f"  {key:30s}: {value:.6f}")
        
        return metrics
    
    def generate_node_internal_analysis_report(
        self,
        node_model,
        X_baseline: np.ndarray,
        X_adversarial: np.ndarray,
        y_true: np.ndarray,
        y_pred_baseline: np.ndarray,
        y_pred_adversarial: np.ndarray,
        confidence_baseline: np.ndarray,
        confidence_adversarial: np.ndarray,
        failure_indices: np.ndarray,
        attack_type: str,
        dataset_name: str,
        output_path: str
    ) -> Dict[str, Any]:
        """Generate NODE internal behavior analysis report."""
        print_section_header(f"COMPREHENSIVE NODE INTERNAL ANALYSIS - {dataset_name}")
        
        embeddings_baseline = self.extract_node_embeddings(node_model, X_baseline)
        embeddings_adversarial = self.extract_node_embeddings(node_model, X_adversarial)
        
        activation_variance = self.compute_node_activation_variance(embeddings_baseline, embeddings_adversarial)
        decision_shift = self.compute_decision_boundary_shift(confidence_baseline, confidence_adversarial)
        neuron_stability = self._compute_neuron_activation_stability(embeddings_baseline, embeddings_adversarial)
        
        decision_flips = np.sum(y_pred_baseline != y_pred_adversarial)
        confidence_drops = confidence_baseline - confidence_adversarial
        high_drop_ratio = np.sum(confidence_drops > 0.1) / len(confidence_drops)
        
        failure_analysis = {}
        if len(failure_indices) > 0:
            embeddings_baseline_failures = [emb[failure_indices] for emb in embeddings_baseline]
            embeddings_adversarial_failures = [emb[failure_indices] for emb in embeddings_adversarial]
            failure_analysis = self._characterize_node_failures(
                embeddings_baseline_failures,
                embeddings_adversarial_failures,
                confidence_baseline[failure_indices],
                confidence_adversarial[failure_indices]
            )
        
        report = {
            'dataset': dataset_name,
            'attack_type': attack_type,
            'activation_variance': activation_variance,
            'decision_shift': decision_shift,
            'neuron_stability': neuron_stability,
            'decision_flips': int(decision_flips),
            'decision_flip_ratio': float(decision_flips / len(y_true)),
            'high_confidence_drop_ratio': float(high_drop_ratio),
            'mean_confidence_drop': float(np.mean(confidence_drops)),
            'max_confidence_drop': float(np.max(confidence_drops)),
            'failure_count': len(failure_indices),
            'failure_characterization': failure_analysis
        }
        
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NODE INTERNAL BEHAVIOR ANALYSIS REPORT (RQ2)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Attack Type: {attack_type}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("-" * 80 + "\n1. ACTIVATION VARIANCE ANALYSIS\n" + "-" * 80 + "\n")
            for key, value in activation_variance.items():
                f.write(f"{key:40s}: {value:.6f}\n")
            
            f.write("\n" + "-" * 80 + "\n2. DECISION BOUNDARY SHIFT\n" + "-" * 80 + "\n")
            for key, value in decision_shift.items():
                f.write(f"{key:40s}: {value:.6f}\n")
            
            f.write("\n" + "-" * 80 + "\n3. NEURON ACTIVATION STABILITY\n" + "-" * 80 + "\n")
            for key, value in neuron_stability.items():
                f.write(f"{key:40s}: {value:.6f}\n")
            
            f.write("\n" + "-" * 80 + "\n4. DECISION PATTERNS\n" + "-" * 80 + "\n")
            f.write(f"{'Decision Flips (Count)':40s}: {decision_flips}\n")
            f.write(f"{'Decision Flip Ratio':40s}: {decision_flips / len(y_true):.6f}\n")
            f.write(f"{'High Confidence Drop (>0.1) Ratio':40s}: {high_drop_ratio:.6f}\n")
            
            f.write("\n" + "-" * 80 + "\n5. CONFIDENCE DEGRADATION\n" + "-" * 80 + "\n")
            f.write(f"{'Mean Confidence Drop':40s}: {np.mean(confidence_drops):.6f}\n")
            f.write(f"{'Max Confidence Drop':40s}: {np.max(confidence_drops):.6f}\n")
            f.write(f"{'Median Confidence Drop':40s}: {np.median(confidence_drops):.6f}\n")
            f.write(f"{'Std Dev Confidence Drop':40s}: {np.std(confidence_drops):.6f}\n")
            
            f.write("\n" + "-" * 80 + "\n6. FAILURE CHARACTERIZATION\n" + "-" * 80 + "\n")
            f.write(f"{'Total Failures':40s}: {len(failure_indices)}\n")
            f.write(f"{'Failure Rate':40s}: {len(failure_indices) / len(y_true) * 100:.2f}%\n")
            
            if failure_analysis:
                for key, value in failure_analysis.items():
                    if isinstance(value, (int, float)):
                        f.write(f"{key:40s}: {value:.6f}\n")
        
        print(f"NODE internal analysis report saved to: {output_path}")
        return report
    
    def _compute_neuron_activation_stability(self, embeddings_baseline: List[np.ndarray], embeddings_adversarial: List[np.ndarray]) -> Dict[str, float]:
        """Compute neuron activation stability."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        stability_metrics = {}
        
        for layer_idx, (emb_base, emb_adv) in enumerate(zip(embeddings_baseline, embeddings_adversarial)):
            mean_base = emb_base.mean(axis=0)
            mean_adv = emb_adv.mean(axis=0)
            
            similarity = float(cosine_similarity([mean_base], [mean_adv])[0, 0])
            stability_metrics[f'layer_{layer_idx+1}_cosine_similarity'] = similarity
            
            l2_dist = float(np.linalg.norm(mean_base - mean_adv))
            stability_metrics[f'layer_{layer_idx+1}_l2_distance'] = l2_dist
        
        cosine_sims = [v for k, v in stability_metrics.items() if 'cosine' in k]
        if cosine_sims:
            stability_metrics['mean_cosine_similarity'] = float(np.mean(cosine_sims))
        
        l2_dists = [v for k, v in stability_metrics.items() if 'l2_distance' in k]
        if l2_dists:
            stability_metrics['mean_l2_distance'] = float(np.mean(l2_dists))
        
        return stability_metrics
    
    def _characterize_node_failures(self, embeddings_baseline_failures: List[np.ndarray], embeddings_adversarial_failures: List[np.ndarray], 
                                   confidence_baseline_failures: np.ndarray, confidence_adversarial_failures: np.ndarray) -> Dict[str, Any]:
        """Characterize failures in terms of internal behaviors."""
        characterization = {}
        
        conf_drops = confidence_baseline_failures - confidence_adversarial_failures
        characterization['mean_confidence_drop_in_failures'] = float(np.mean(conf_drops))
        characterization['max_confidence_drop_in_failures'] = float(np.max(conf_drops))
        characterization['median_confidence_drop_in_failures'] = float(np.median(conf_drops))
        
        if embeddings_baseline_failures and embeddings_adversarial_failures:
            for layer_idx, (emb_base, emb_adv) in enumerate(zip(embeddings_baseline_failures, embeddings_adversarial_failures)):
                activation_diff = np.linalg.norm(emb_base - emb_adv, axis=1)
                characterization[f'layer_{layer_idx+1}_mean_activation_diff'] = float(np.mean(activation_diff))
                characterization[f'layer_{layer_idx+1}_max_activation_diff'] = float(np.max(activation_diff))
        
        return characterization

    @staticmethod
    def _safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Compute Spearman correlation with stable fallback for constant vectors."""
        try:
            corr, pval = spearmanr(x, y)
            if np.isnan(corr):
                return 0.0, 1.0
            if np.isnan(pval):
                return float(corr), 1.0
            return float(corr), float(pval)
        except Exception:
            return 0.0, 1.0
    
    def compute_rq2_correlation_analysis(
        self,
        X_baseline: np.ndarray,
        X_adversarial: np.ndarray,
        y_pred_baseline: np.ndarray,
        y_pred_adversarial: np.ndarray,
        confidence_baseline: np.ndarray,
        confidence_adversarial: np.ndarray,
        y_true: np.ndarray,
        feature_names: List[str],
        tabnet_model: TabNetClassifier,
        model_name: str
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Compute RQ2 correlation analysis linking TabNet feature shifts to failures."""
        print_section_header(f"RQ2 CORRELATION ANALYSIS: {model_name}")
        
        # Compute binary failure indicators
        failure_mask = (y_pred_baseline == y_true) & (y_pred_adversarial != y_true)
        confidence_drop = confidence_baseline - confidence_adversarial
        misclassification = (y_pred_adversarial != y_true).astype(int)
        
        # Compute real per-feature shifts from TabNet explanations in manageable batches.
        feature_count = len(feature_names)
        if feature_count == 0:
            corr_df = pd.DataFrame(
                columns=[
                    'feature', 'importance_shift', 'correlation_with_failures',
                    'correlation_pvalue', 'correlation_with_misclass', 'misclass_pvalue'
                ]
            )
            summary_metrics = {
                'confidence_drop_failure_correlation': 0.0,
                'confidence_drop_failure_pvalue': 1.0,
                'importance_shift_failure_correlation': 0.0,
                'importance_shift_failure_pvalue': 1.0,
                'mean_importance_shift': 0.0,
                'mean_confidence_drop': float(np.mean(confidence_drop)),
                'failure_count': int(np.sum(failure_mask)),
                'failure_rate': float(np.mean(failure_mask)),
            }
            return corr_df, summary_metrics

        batch_size = 4096
        shift_batches = []
        n_samples = X_baseline.shape[0]

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            base_chunk = X_baseline[start:end]
            adv_chunk = X_adversarial[start:end]

            try:
                base_explain = tabnet_model.explain(base_chunk)
                adv_explain = tabnet_model.explain(adv_chunk)

                if isinstance(base_explain, tuple):
                    base_explain = base_explain[0]
                if isinstance(adv_explain, tuple):
                    adv_explain = adv_explain[0]

                base_explain = np.asarray(base_explain)
                adv_explain = np.asarray(adv_explain)

                if base_explain.ndim == 1:
                    base_explain = base_explain.reshape(-1, 1)
                if adv_explain.ndim == 1:
                    adv_explain = adv_explain.reshape(-1, 1)

                aligned_features = min(feature_count, base_explain.shape[1], adv_explain.shape[1])
                chunk_shift = np.abs(base_explain[:, :aligned_features] - adv_explain[:, :aligned_features])
            except Exception:
                aligned_features = feature_count
                fallback = np.abs(confidence_drop[start:end]).reshape(-1, 1)
                chunk_shift = np.repeat(fallback, aligned_features, axis=1)

            shift_batches.append(chunk_shift)

        feature_shift_matrix = np.vstack(shift_batches)
        aligned_feature_count = feature_shift_matrix.shape[1]
        aligned_feature_names = feature_names[:aligned_feature_count]

        importance_shift_per_sample = feature_shift_matrix.mean(axis=1)
        mean_feature_shift = feature_shift_matrix.mean(axis=0)

        corr_failures = []
        pvals_failures = []
        corr_misclass = []
        pvals_misclass = []

        for idx in range(aligned_feature_count):
            feature_shift = feature_shift_matrix[:, idx]
            c_fail, p_fail = self._safe_spearman(feature_shift, failure_mask)
            c_misc, p_misc = self._safe_spearman(feature_shift, misclassification)
            corr_failures.append(c_fail)
            pvals_failures.append(p_fail)
            corr_misclass.append(c_misc)
            pvals_misclass.append(p_misc)

        corr_df = pd.DataFrame({
            'feature': aligned_feature_names,
            'importance_shift': [float(v) for v in mean_feature_shift],
            'correlation_with_failures': corr_failures,
            'correlation_pvalue': pvals_failures,
            'correlation_with_misclass': corr_misclass,
            'misclass_pvalue': pvals_misclass,
        }).sort_values('correlation_with_failures', key=lambda s: np.abs(s), ascending=False)

        confidence_corr, confidence_pval = self._safe_spearman(confidence_drop, failure_mask)
        importance_corr, importance_pval = self._safe_spearman(importance_shift_per_sample, failure_mask)
        
        summary_metrics = {
            'confidence_drop_failure_correlation': float(confidence_corr),
            'confidence_drop_failure_pvalue': float(confidence_pval),
            'importance_shift_failure_correlation': float(importance_corr),
            'importance_shift_failure_pvalue': float(importance_pval),
            'mean_importance_shift': float(np.mean(mean_feature_shift)),
            'mean_confidence_drop': float(np.mean(confidence_drop)),
            'failure_count': int(np.sum(failure_mask)),
            'failure_rate': float(np.mean(failure_mask))
        }
        
        print("\nRQ2 Correlation Summary:")
        print(f"  Confidence Drop - Failure Correlation: {summary_metrics['confidence_drop_failure_correlation']:.6f} (p={summary_metrics['confidence_drop_failure_pvalue']:.6f})")
        print(f"  Feature Importance Shift - Failure Correlation: {summary_metrics['importance_shift_failure_correlation']:.6f} (p={summary_metrics['importance_shift_failure_pvalue']:.6f})")
        
        return corr_df, summary_metrics
    
    def compute_rq2_node_correlation_analysis(
        self,
        X_baseline: np.ndarray,
        X_adversarial: np.ndarray,
        y_pred_baseline: np.ndarray,
        y_pred_adversarial: np.ndarray,
        confidence_baseline: np.ndarray,
        confidence_adversarial: np.ndarray,
        y_true: np.ndarray,
        node_model: torch.nn.Module,
        model_name: str
    ) -> Dict[str, float]:
        """Compute RQ2 correlation analysis for NODE using real layer embedding shifts."""
        print_section_header(f"RQ2 NODE CORRELATION ANALYSIS: {model_name}")
        
        failure_mask = (y_pred_baseline == y_true) & (y_pred_adversarial != y_true)
        confidence_drop = confidence_baseline - confidence_adversarial
        misclassification = (y_pred_adversarial != y_true).astype(int)
        
        correlation_results = {}

        embeddings_baseline = self.extract_node_embeddings(node_model, X_baseline)
        embeddings_adversarial = self.extract_node_embeddings(node_model, X_adversarial)

        layer_failure_corrs = []
        for idx, (emb_base, emb_adv) in enumerate(zip(embeddings_baseline, embeddings_adversarial), start=1):
            layer_shift = np.linalg.norm(emb_base - emb_adv, axis=1)
            corr_fail, pval_fail = self._safe_spearman(layer_shift, failure_mask)
            corr_misc, pval_misc = self._safe_spearman(layer_shift, misclassification)

            correlation_results[f'layer_{idx}_activation_failure_corr'] = corr_fail
            correlation_results[f'layer_{idx}_activation_failure_pval'] = pval_fail
            correlation_results[f'layer_{idx}_activation_misclass_corr'] = corr_misc
            correlation_results[f'layer_{idx}_activation_misclass_pval'] = pval_misc
            layer_failure_corrs.append(corr_fail)

        confidence_corr, confidence_pval = self._safe_spearman(confidence_drop, failure_mask)
        correlation_results['confidence_drop_failure_correlation'] = confidence_corr
        correlation_results['confidence_drop_failure_pvalue'] = confidence_pval
        correlation_results['mean_layer_activation_failure_corr'] = float(np.mean(layer_failure_corrs)) if layer_failure_corrs else 0.0
        correlation_results['failure_count'] = int(np.sum(failure_mask))
        correlation_results['failure_rate'] = float(np.mean(failure_mask))
        
        print("\nRQ2 NODE Correlation Summary:")
        print(f"  Confidence Drop - Failure Correlation: {correlation_results['confidence_drop_failure_correlation']:.6f} (p={correlation_results['confidence_drop_failure_pvalue']:.6f})")
        
        return correlation_results

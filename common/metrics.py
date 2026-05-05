import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    precision_score, 
    f1_score,
    confusion_matrix,
)
from typing import Dict, Any, Tuple


def compute_array_stats(arr: np.ndarray, name: str = "Values") -> Dict[str, float]:
    """
    Compute mean, max, std, and L2 norm for an array.
    
    Args:
        arr: NumPy array to analyze
        name: Name of the array (for reference)
    
    Returns:
        Dictionary with 'mean', 'max', 'std', 'l2_norm' keys
    """
    return {
        f'mean_{name.lower()}': float(np.mean(arr)),
        f'max_{name.lower()}': float(np.max(arr)),
        f'std_{name.lower()}': float(np.std(arr)),
        f'l2_{name.lower()}': float(np.linalg.norm(arr))
    }


def compute_baseline_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'false_negative_rate': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        'misclassification_rate': float(1 - accuracy_score(y_true, y_pred)),
        'avg_confidence': float(np.mean(np.max(y_proba, axis=1))),
        'std_confidence': float(np.std(np.max(y_proba, axis=1))),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }
    
    return metrics





def compute_metric_degradation(baseline: Dict[str, float], adversarial: Dict[str, float]) -> Dict[str, float]:
    degradation = {
        'accuracy_drop': baseline['accuracy'] - adversarial['accuracy'],
        'recall_drop': baseline['recall'] - adversarial['recall'],
        'fnr_increase': adversarial['false_negative_rate'] - baseline['false_negative_rate'],
        'misclassification_increase': adversarial['misclassification_rate'] - baseline['misclassification_rate'],
        'confidence_degradation': baseline['avg_confidence'] - adversarial['avg_confidence'],
        'fn_increase': adversarial['false_negatives'] - baseline['false_negatives'],
        'fp_increase': adversarial['false_positives'] - baseline['false_positives']
    }
    
    return degradation


def compute_confidence_metrics(confidence_baseline: np.ndarray, confidence_adv: np.ndarray) -> Dict[str, float]:
    confidence_diff = confidence_baseline - confidence_adv
    
    metrics = {
        'mean_confidence_change': float(np.mean(confidence_diff)),
        'std_confidence_change': float(np.std(confidence_diff)),
        'max_confidence_drop': float(np.max(confidence_diff)),
        'min_confidence_drop': float(np.min(confidence_diff)),
        'samples_with_collapse': int(np.sum(confidence_diff > 0.3)),
        'collapse_percentage': float(np.mean(confidence_diff > 0.3) * 100)
    }
    
    return metrics


def identify_failure_samples(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_pred_adv: np.ndarray,
    confidence_baseline: np.ndarray,
    confidence_adv: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Any]]:
    correct_baseline = (y_pred_baseline == y_true)
    incorrect_adv = (y_pred_adv != y_true)
    flipped = correct_baseline & incorrect_adv
    
    confidence_diff = confidence_baseline - confidence_adv
    confidence_collapse = confidence_diff > 0.3
    
    high_conf_wrong = (confidence_adv > 0.8) & incorrect_adv
    
    failure_mask = flipped | confidence_collapse | high_conf_wrong
    failure_indices = np.where(failure_mask)[0]
    
    analysis = {
        'total_failures': int(np.sum(failure_mask)),
        'flipped_predictions': int(np.sum(flipped)),
        'confidence_collapses': int(np.sum(confidence_collapse)),
        'high_confidence_errors': int(np.sum(high_conf_wrong)),
        'failure_rate': float(np.mean(failure_mask))
    }
    
    return failure_indices, analysis

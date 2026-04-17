"""
Classification metrics for NoC prediction.

Metrics:
- Accuracy (overall and per-class)
- Precision, Recall, F1-score (per-class and macro-averaged)
- Confusion matrix
- Balanced accuracy (macro average of per-class recalls)
"""

from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
)


def noc_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Overall accuracy."""
    return float(accuracy_score(y_true, y_pred))


def noc_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced accuracy (macro-average per-class recall)."""
    return float(balanced_accuracy_score(y_true, y_pred))


def noc_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'macro'
) -> float:
    """
    Precision score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: 'macro', 'micro', 'weighted', or None (returns per-class)
    """
    return float(precision_score(y_true, y_pred, average=average, zero_division=0))


def noc_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'macro'
) -> float:
    """
    Recall score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: 'macro', 'micro', 'weighted', or None (returns per-class)
    """
    return float(recall_score(y_true, y_pred, average=average, zero_division=0))


def noc_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'macro'
) -> float:
    """
    F1-score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: 'macro', 'micro', 'weighted', or None (returns per-class)
    """
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def noc_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """
    Per-class precision, recall, F1-score.
    
    Returns:
        Dict mapping NoC -> {precision, recall, f1}
    """
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    
    results = {}
    for noc in unique_classes:
        mask_true = y_true == noc
        mask_pred = y_pred == noc
        tp = np.sum(mask_true & mask_pred)
        fp = np.sum(~mask_true & mask_pred)
        fn = np.sum(mask_true & ~mask_pred)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[int(noc)] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'support': int(np.sum(mask_true))
        }
    
    return results


def noc_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: tuple = None
) -> np.ndarray:
    """
    Get confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Label order (default: 1-10)
    
    Returns:
        Confusion matrix (true labels as rows, predicted as columns)
    """
    if labels is None:
        labels = tuple(range(1, 11))
    
    return confusion_matrix(y_true, y_pred, labels=labels)


def noc_confusion_matrix_normalized(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: tuple = None
) -> np.ndarray:
    """
    Get normalized confusion matrix (rows sum to 1).
    Helps identify where predictions are going.
    """
    cm = noc_confusion_matrix(y_true, y_pred, labels=labels)
    return cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)


def noc_off_by_one_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fraction of predictions off by exactly 1 (tolerable error).
    
    Example: Predicting 3 when true is 2 or 4 is acceptable.
    """
    off_by_one = np.sum(np.abs(y_true - y_pred) == 1) / len(y_true)
    return float(off_by_one)


def noc_mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error in NoC prediction."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_noc_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    return_per_class: bool = True
) -> Dict[str, any]:
    """
    Compute comprehensive NoC prediction metrics.
    
    Args:
        y_true: Ground truth NoC labels
        y_pred: Predicted NoC labels
        return_per_class: Whether to include per-class metrics
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'accuracy': noc_accuracy(y_true, y_pred),
        'balanced_accuracy': noc_balanced_accuracy(y_true, y_pred),
        'precision_macro': noc_precision(y_true, y_pred, average='macro'),
        'recall_macro': noc_recall(y_true, y_pred, average='macro'),
        'f1_macro': noc_f1_score(y_true, y_pred, average='macro'),
        'precision_weighted': noc_precision(y_true, y_pred, average='weighted'),
        'recall_weighted': noc_recall(y_true, y_pred, average='weighted'),
        'f1_weighted': noc_f1_score(y_true, y_pred, average='weighted'),
        'off_by_one_error': noc_off_by_one_error(y_true, y_pred),
        'mae': noc_mean_absolute_error(y_true, y_pred),
    }
    
    if return_per_class:
        metrics['per_class'] = noc_per_class_metrics(y_true, y_pred)
    
    return metrics

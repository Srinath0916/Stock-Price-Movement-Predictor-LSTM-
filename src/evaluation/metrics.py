"""
Evaluation Metrics and Baseline Benchmarks for Stock Movement Prediction.
"""

from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def compute_binary_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    Computes binary directional classification metrics given true labels and predicted probabilities.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (0.0 or 1.0).
    y_pred_prob : np.ndarray
        Model predicted probabilities in range [0, 1].
    threshold : float
        Decision threshold for positive class (default 0.5).

    Returns
    -------
    dict containing accuracy, precision, recall, f1, and confusion matrix elements.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = (np.asarray(y_pred_prob).ravel() >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_samples": len(y_true),
    }


def evaluate_majority_baseline(y_train: np.ndarray, y_eval: np.ndarray, split_name: str = "Validation") -> Dict:
    """
    Evaluates the naive baseline: always predicts the majority class observed in training data.

    Parameters
    ----------
    y_train : np.ndarray
        Training target labels used to determine the majority class.
    y_eval : np.ndarray
        Evaluation labels (validation or test).
    split_name : str
        Name of the evaluation split.

    Returns
    -------
    dict of baseline performance metrics.
    """
    y_train = np.asarray(y_train).astype(int).ravel()
    y_eval = np.asarray(y_eval).astype(int).ravel()

    # Determine majority class in training set
    counts = np.bincount(y_train)
    majority_class = int(np.argmax(counts))

    # Naive prediction: constant majority class
    y_pred = np.full_like(y_eval, fill_value=majority_class)

    acc = accuracy_score(y_eval, y_pred)
    prec = precision_score(y_eval, y_pred, zero_division=0)
    rec = recall_score(y_eval, y_pred, zero_division=0)
    f1 = f1_score(y_eval, y_pred, zero_division=0)
    cm = confusion_matrix(y_eval, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "split": split_name,
        "majority_class": majority_class,
        "majority_class_name": "UP (1)" if majority_class == 1 else "DOWN/FLAT (0)",
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_samples": len(y_eval),
    }

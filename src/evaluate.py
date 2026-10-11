"""Evaluation metrics (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)

from .config import NORMAL


def binary_metrics(y_true, y_attack_pred, y_score=None) -> dict:
    """Attack = positive class. Recall = share of attacks caught; precision = share of alerts that are real."""
    yt = (np.asarray(y_true) != NORMAL).astype(int)
    yp = np.asarray(y_attack_pred).astype(int)
    tn = int(((yt == 0) & (yp == 0)).sum()); fp = int(((yt == 0) & (yp == 1)).sum())
    out = {"accuracy": accuracy_score(yt, yp), "precision": precision_score(yt, yp, zero_division=0),
           "recall": recall_score(yt, yp, zero_division=0), "f1": f1_score(yt, yp, zero_division=0),
           "false_positive_rate": fp / max(1, tn + fp)}
    if y_score is not None:
        out["roc_auc"] = roc_auc_score(yt, y_score)
    return {k: round(float(v), 4) for k, v in out.items()}


def multiclass_confusion(y_true, y_pred, labels):
    return confusion_matrix(y_true, y_pred, labels=labels)

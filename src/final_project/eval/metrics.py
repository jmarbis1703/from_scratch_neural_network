from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, roc_auc_score

def binary_report(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> str:
    y_pred = (y_proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    roc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, digits=3)
    parts = [
        f"Accuracy: {acc:.3f}",
        f"Precision: {prec:.3f}",
        f"Recall: {rec:.3f}",
        f"F1: {f1:.3f}",
        f"ROC-AUC: {roc:.3f}",
        "Confusion Matrix:",
        str(cm),
        "",
        "Report:",
        rep
    ]
    return "\n".join(parts)

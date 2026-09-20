"""
Evaluation metrics for WAFS-Net.

This module provides the common evaluation metrics used for
internal and independent external validation.

Labels:
    0 = benign
    1 = malignant
"""

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_binary_metrics(
    y_true,
    y_prob,
    threshold=0.5,
):
    """
    Calculate binary classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.

    y_prob : array-like
        Predicted probabilities for the malignant class.

    threshold : float
        Decision threshold used to convert probabilities
        into binary predictions.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    y_prob = np.asarray(
        y_prob,
        dtype=np.float64,
    )

    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError(
            "y_true and y_prob must contain "
            "the same number of samples."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    y_pred = (
        y_prob >= threshold
    ).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    sensitivity = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    precision = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    npv = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else np.nan
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    try:
        roc_auc = roc_auc_score(
            y_true,
            y_prob,
        )
    except ValueError:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(
            y_true,
            y_prob,
        )
    except ValueError:
        pr_auc = np.nan

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "npv": float(npv),
        "f1": float(f1),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_youden_threshold(
    y_true,
    y_prob,
):
    """
    Determine the optimal threshold using Youden's J statistic.

    J = sensitivity + specificity - 1
    """

    from sklearn.metrics import roc_curve

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    y_prob = np.asarray(
        y_prob,
        dtype=np.float64,
    )

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_prob,
    )

    youden_j = (
        tpr - fpr
    )

    best_index = np.argmax(
        youden_j
    )

    return float(
        thresholds[best_index]
    )


def print_binary_metrics(
    metrics,
):
    """
    Print binary classification metrics.
    """

    print(
        "=" * 60
    )

    print(
        "Binary Classification Results"
    )

    print(
        "=" * 60
    )

    print(
        f"Threshold          : "
        f"{metrics['threshold']:.6f}"
    )

    print(
        f"ROC-AUC            : "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC             : "
        f"{metrics['pr_auc']:.4f}"
    )

    print(
        f"Accuracy           : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Sensitivity        : "
        f"{metrics['sensitivity']:.4f}"
    )

    print(
        f"Specificity        : "
        f"{metrics['specificity']:.4f}"
    )

    print(
        f"Precision          : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"NPV                : "
        f"{metrics['npv']:.4f}"
    )

    print(
        f"F1                 : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Balanced Accuracy  : "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        "\nConfusion Matrix"
    )

    print(
        f"TN = {metrics['tn']} | "
        f"FP = {metrics['fp']}"
    )

    print(
        f"FN = {metrics['fn']} | "
        f"TP = {metrics['tp']}"
    )

    print(
        "=" * 60
    )

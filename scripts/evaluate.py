"""
Evaluation script for WAFS-Net.

Metrics:
    ROC-AUC
    PR-AUC
    Accuracy
    Sensitivity
    Specificity
    Precision
    NPV
    F1 score
    Balanced Accuracy
    Confusion Matrix

The decision threshold can be specified from the command line.
"""

import os
import sys
import argparse

import numpy as np
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)


from src.models.wafs_net import build_wafs_net
from src.datasets.ovarian_dataset import build_dataset


# ============================================================
# Load checkpoint
# ============================================================

def load_model(
    checkpoint_path,
    device,
):

    model = build_wafs_net(
        num_classes=2,
        pretrained=False,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model = model.to(device)

    model.eval()

    return model


# ============================================================
# Prediction
# ============================================================

@torch.no_grad()
def predict(
    model,
    loader,
    device,
):

    labels_all = []
    probabilities_all = []

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        logits = model(images)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[:, 1]

        labels_all.extend(
            labels.numpy().tolist()
        )

        probabilities_all.extend(
            probabilities
            .cpu()
            .numpy()
            .tolist()
        )

    labels_all = np.asarray(
        labels_all,
        dtype=np.int64,
    )

    probabilities_all = np.asarray(
        probabilities_all,
        dtype=np.float64,
    )

    return (
        labels_all,
        probabilities_all,
    )


# ============================================================
# Metric calculation
# ============================================================

def calculate_metrics(
    y_true,
    y_prob,
    threshold,
):

    y_pred = (
        y_prob >= threshold
    ).astype(np.int64)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

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

    if (tn + fp) > 0:
        specificity = (
            tn / (tn + fp)
        )
    else:
        specificity = np.nan

    precision = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    if (tn + fn) > 0:
        npv = (
            tn / (tn + fn)
        )
    else:
        npv = np.nan

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    balanced_acc = (
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
        pr_auc = (
            average_precision_score(
                y_true,
                y_prob,
            )
        )
    except ValueError:
        pr_auc = np.nan

    metrics = {
        "threshold": threshold,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "f1": f1,
        "balanced_accuracy":
            balanced_acc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return metrics


# ============================================================
# Display results
# ============================================================

def print_metrics(metrics):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "WAFS-Net Evaluation"
    )

    print(
        "=" * 60
    )

    print(
        f"Threshold           : "
        f"{metrics['threshold']:.6f}"
    )

    print(
        f"ROC-AUC             : "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC              : "
        f"{metrics['pr_auc']:.4f}"
    )

    print(
        f"Accuracy            : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Sensitivity         : "
        f"{metrics['sensitivity']:.4f}"
    )

    print(
        f"Specificity         : "
        f"{metrics['specificity']:.4f}"
    )

    print(
        f"Precision           : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"NPV                 : "
        f"{metrics['npv']:.4f}"
    )

    print(
        f"F1                  : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Balanced Accuracy   : "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        "\nConfusion Matrix"
    )

    print(
        f"TN = {metrics['tn']}"
    )

    print(
        f"FP = {metrics['fp']}"
    )

    print(
        f"FN = {metrics['fn']}"
    )

    print(
        f"TP = {metrics['tp']}"
    )

    print(
        "=" * 60
    )


# ============================================================
# Main
# ============================================================

def main(args):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = build_dataset(
        root_dir=args.data_dir,
        training=False,
        image_size=args.image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    print(
        f"Evaluation samples: "
        f"{len(dataset)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_true, y_prob = predict(
        model=model,
        loader=loader,
        device=device,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=args.threshold,
    )

    print_metrics(
        metrics
    )


# ============================================================
# Command-line interface
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate WAFS-Net for "
            "ovarian tumor classification."
        )
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=(
            "data/processed/"
            "mmotu_2d_cls/val"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "outputs/checkpoints/"
            "mmotu_dual_wavelet_binary_best.pth"
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.027425,
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    main(args)

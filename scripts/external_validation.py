"""
Independent external validation for WAFS-Net.

Clinical images are NOT distributed with this repository.

The script evaluates a trained WAFS-Net model on an
independent binary ovarian tumor validation cohort.

Labels:
    0 = benign
    1 = malignant
"""

import os
import sys
import argparse

import numpy as np
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
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
# Model loading
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

    y_true = []
    y_prob = []

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

        y_true.extend(
            labels.numpy().tolist()
        )

        y_prob.extend(
            probabilities
            .cpu()
            .numpy()
            .tolist()
        )

    return (
        np.asarray(
            y_true,
            dtype=np.int64,
        ),
        np.asarray(
            y_prob,
            dtype=np.float64,
        ),
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate_external(
    y_true,
    y_prob,
    threshold,
):

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
        pr_auc = average_precision_score(
            y_true,
            y_prob,
        )
    except ValueError:
        pr_auc = np.nan

    return {
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


# ============================================================
# Result display
# ============================================================

def print_results(
    metrics,
    n_samples,
):

    print(
        "\n"
        + "=" * 64
    )

    print(
        "WAFS-Net Independent External Validation"
    )

    print(
        "=" * 64
    )

    print(
        f"Number of cases      : "
        f"{n_samples}"
    )

    print(
        f"Decision threshold   : "
        f"{metrics['threshold']:.6f}"
    )

    print(
        f"ROC-AUC              : "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC               : "
        f"{metrics['pr_auc']:.4f}"
    )

    print(
        f"Accuracy             : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Sensitivity          : "
        f"{metrics['sensitivity']:.4f}"
    )

    print(
        f"Specificity          : "
        f"{metrics['specificity']:.4f}"
    )

    print(
        f"Precision            : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"NPV                  : "
        f"{metrics['npv']:.4f}"
    )

    print(
        f"F1                   : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Balanced Accuracy    : "
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
        "=" * 64
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
    # External clinical dataset
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
        f"External validation cases: "
        f"{len(dataset)}"
    )

    # --------------------------------------------------------
    # Load model
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
    # Evaluation
    # --------------------------------------------------------

    metrics = evaluate_external(
        y_true=y_true,
        y_prob=y_prob,
        threshold=args.threshold,
    )

    print_results(
        metrics=metrics,
        n_samples=len(dataset),
    )


# ============================================================
# Command-line interface
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Independent external validation "
            "of WAFS-Net."
        )
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help=(
            "Local directory containing the "
            "de-identified external validation data."
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

"""
Training script for WAFS-Net.

This script trains WAFS-Net for binary ovarian tumor
classification using class-weighted cross-entropy.
"""

import os
import sys
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.models.wafs_net import build_wafs_net
from src.datasets.ovarian_dataset import build_dataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    running_loss = 0.0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

    return (
        running_loss
        / len(loader.dataset)
    )


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    running_loss = 0.0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        logits = model(images)

        loss = criterion(
            logits,
            labels
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[:, 1]

        predictions = torch.argmax(
            logits,
            dim=1
        )

        running_loss += (
            loss.item()
            * images.size(0)
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

    val_loss = (
        running_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    try:
        auc = roc_auc_score(
            all_labels,
            all_probabilities
        )
    except ValueError:
        auc = float("nan")

    return {
        "loss": val_loss,
        "accuracy": accuracy,
        "f1": f1,
        "auc": auc,
    }


def main(args):

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = build_dataset(
        root_dir=args.train_dir,
        training=True,
        image_size=args.image_size,
    )

    val_dataset = build_dataset(
        root_dir=args.val_dir,
        training=False,
        image_size=args.image_size,
    )

    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_wafs_net(
        num_classes=2,
        pretrained=args.pretrained,
        dropout=args.dropout,
    ).to(device)

    # --------------------------------------------------------
    # Weighted cross-entropy
    # --------------------------------------------------------

    class_weights = torch.tensor(
        [
            args.weight_benign,
            args.weight_malignant,
        ],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
        )
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    best_auc = -1.0

    checkpoint_path = os.path.join(
        args.output_dir,
        "mmotu_dual_wavelet_binary_best.pth"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1
    ):

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(
            metrics["auc"]
        )

        print(
            f"Epoch "
            f"{epoch:03d}/{args.epochs:03d} | "
            f"Train Loss: "
            f"{train_loss:.4f} | "
            f"Val Loss: "
            f"{metrics['loss']:.4f} | "
            f"ACC: "
            f"{metrics['accuracy']:.4f} | "
            f"F1: "
            f"{metrics['f1']:.4f} | "
            f"AUC: "
            f"{metrics['auc']:.4f}"
        )

        if metrics["auc"] > best_auc:

            best_auc = metrics["auc"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "val_auc":
                        metrics["auc"],
                    "val_accuracy":
                        metrics["accuracy"],
                    "val_f1":
                        metrics["f1"],
                },
                checkpoint_path,
            )

            print(
                f"Saved best model: "
                f"{checkpoint_path}"
            )

    print(
        f"Training completed. "
        f"Best validation AUC: "
        f"{best_auc:.4f}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Train WAFS-Net for ovarian "
            "tumor binary classification."
        )
    )

    parser.add_argument(
        "--train_dir",
        type=str,
        default=(
            "data/processed/"
            "mmotu_2d_cls/train"
        ),
    )

    parser.add_argument(
        "--val_dir",
        type=str,
        default=(
            "data/processed/"
            "mmotu_2d_cls/val"
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/checkpoints",
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
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--weight_benign",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--weight_malignant",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--pretrained",
        action="store_true",
    )

    args = parser.parse_args()

    main(args)

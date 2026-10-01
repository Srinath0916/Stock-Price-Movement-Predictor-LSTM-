"""
Training and Validation Engine for Stock Movement Prediction LSTM.
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_binary_metrics
from src.models.lstm import StockLSTM


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_norm: float = 1.0,
) -> Tuple[float, float]:
    """
    Executes one training epoch with gradient clipping.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for bx, by in dataloader:
        bx = bx.to(device)
        by = by.to(device)

        optimizer.zero_grad()
        logits = model(bx)
        loss = criterion(logits, by)
        loss.backward()

        if max_norm is not None and max_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        optimizer.step()

        batch_size = len(by)
        running_loss += loss.item() * batch_size
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == by).sum().item()
        total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict]:
    """
    Evaluates model across a DataLoader and computes loss, accuracy, precision, recall, and F1.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    total = 0

    with torch.no_grad():
        for bx, by in dataloader:
            bx = bx.to(device)
            by = by.to(device)

            logits = model(bx)
            loss = criterion(logits, by)

            batch_size = len(by)
            running_loss += loss.item() * batch_size
            total += batch_size

            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(by.cpu().numpy())

    eval_loss = running_loss / total
    y_prob = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    metrics = compute_binary_metrics(y_true, y_prob)
    metrics["loss"] = float(eval_loss)
    return eval_loss, metrics


def plot_training_history(history: List[Dict], save_path: Path, title: str = "LSTM Training History"):
    """
    Generates and saves dual-panel training curves for loss and accuracy/F1.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    val_f1 = [h["val_f1"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, train_loss, label="Train Loss (BCE)", color="#1f77b4", lw=2)
    ax1.plot(epochs, val_loss, label="Val Loss (BCE)", color="#ff7f0e", lw=2, linestyle="--")
    best_loss_epoch = epochs[np.argmin(val_loss)]
    min_val_loss = min(val_loss)
    ax1.axvline(best_loss_epoch, color="red", linestyle=":", alpha=0.7, label=f"Best Val Loss (Ep {best_loss_epoch})")
    ax1.set_title(f"{title} - Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Metric plot
    ax2.plot(epochs, train_acc, label="Train Acc", color="#1f77b4", lw=2)
    ax2.plot(epochs, val_acc, label="Val Acc", color="#ff7f0e", lw=2, linestyle="--")
    ax2.plot(epochs, val_f1, label="Val F1", color="#2ca02c", lw=2, linestyle="-.")
    ax2.axvline(best_loss_epoch, color="red", linestyle=":", alpha=0.7, label=f"Early Stop Point (Ep {best_loss_epoch})")
    ax2.set_title(f"{title} - Accuracy & F1", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0.40, 0.70)
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Saved training curve plot to: {save_path}")


def train_lstm_experiment(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    max_norm: float = 1.0,
    max_epochs: int = 50,
    patience: int = 7,
    checkpoint_path: Optional[Path] = None,
    plot_path: Optional[Path] = None,
    experiment_name: str = "Baseline_LSTM",
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Executes full training pipeline with early stopping based on validation loss.
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print("\n" + "=" * 80)
    print(f"STARTING EXPERIMENT: {experiment_name}")
    print("=" * 80)
    print(f"Device:           {device}")
    print(f"Max Epochs:       {max_epochs} | Early Stopping Patience: {patience}")
    print(f"Learning Rate:    {learning_rate} | Weight Decay: {weight_decay} | Gradient Clip: {max_norm}")
    print(f"Training batches: {len(train_loader):,} | Val batches: {len(val_loader):,}")

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_val_metrics: Dict = {}
    epochs_no_improve = 0

    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_norm=max_norm,
        )

        val_loss, val_metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["f1"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "val_f1": float(val_f1),
        }
        history.append(epoch_record)

        is_best = val_loss < best_val_loss
        status_marker = "(*BEST*)" if is_best else ""
        print(
            f"Epoch {epoch:2d}/{max_epochs:2d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% F1: {val_f1:.4f} {status_marker}"
        )

        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_metrics = val_metrics
            epochs_no_improve = 0
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[Early Stopping Triggered] No validation loss improvement for {patience} consecutive epochs.")
                print(f"Optimal checkpoint restored from Epoch {best_epoch} (Val Loss: {best_val_loss:.4f})")
                break

    # Restore best checkpoint
    if checkpoint_path and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✓ Successfully reloaded best model weights from: {checkpoint_path}")

    # Plot history
    if plot_path:
        plot_training_history(history, plot_path, title=experiment_name)

    return {
        "experiment_name": experiment_name,
        "best_epoch": best_epoch,
        "total_epochs_trained": len(history),
        "best_val_loss": float(best_val_loss),
        "best_val_metrics": best_val_metrics,
        "history": history,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
    }

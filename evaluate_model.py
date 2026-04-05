"""
evaluate_model.py — Test-set evaluation, confusion matrix, and per-class report.

Can be run standalone or called from the main entry point after training.
"""

import os
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

from common import (
    load_labels,
    discover_midi_files,
    match_files_to_labels,
    get_num_classes,
    get_label_name,
    load_config,
)
from model import FeatureMLPRegressor, EnsembleRegressor


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    num_classes: int,
    output_path: str | Path = "confusion_matrix.png",
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> None:
    """Save a heatmap confusion matrix."""
    labels = list(range(num_classes))
    names = [get_label_name(c) for c in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums
        fmt = ".2f"
        title += " (Normalized)"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=names, yticklabels=names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix → {output_path}")


def plot_per_class_accuracy(
    y_true: list[int],
    y_pred: list[int],
    num_classes: int,
    output_path: str | Path = "per_class_accuracy.png",
) -> None:
    """Bar chart of per-class accuracy."""
    labels = list(range(num_classes))
    names = [get_label_name(c) for c in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class_acc = []
    for i in range(num_classes):
        total = cm[i].sum()
        correct = cm[i, i]
        per_class_acc.append(correct / total if total > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("viridis", num_classes)
    bars = ax.bar(names, per_class_acc, color=colors)
    ax.set_xlabel("Grade")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy (Test Set)")
    ax.set_ylim(0, 1.05)

    for bar, val in zip(bars, per_class_acc):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=9,
        )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved per-class accuracy → {output_path}")


def plot_prediction_distribution(
    y_true: list[int],
    y_pred: list[int],
    num_classes: int,
    output_path: str | Path = "prediction_distribution.png",
) -> None:
    """Side-by-side comparison of true vs predicted distributions."""
    labels = list(range(num_classes))
    names = [get_label_name(c) for c in labels]
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)

    x = np.arange(num_classes)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, [true_counts.get(c, 0) for c in labels], width, label="True")
    ax.bar(x + width / 2, [pred_counts.get(c, 0) for c in labels], width, label="Predicted")
    ax.set_xlabel("Grade")
    ax.set_ylabel("Count")
    ax.set_title("True vs Predicted Distribution (Test Set)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved prediction distribution → {output_path}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_on_test(
    ensemble: EnsembleRegressor,
    test_feats: np.ndarray,
    test_labels: list[int],
    num_classes: int,
    output_dir: str | Path,
    device: str = "cpu",
) -> dict:
    """Run evaluation on the test set and generate all reports and plots."""
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)

    # Get ensemble predictions
    preds_raw = ensemble.predict(test_feats, device=device)

    # Continuous MAE
    y_true = test_labels
    mae = float(np.mean(np.abs(preds_raw - np.array(y_true))))

    # Round to nearest integer grade for classification metrics
    y_pred = np.clip(np.round(preds_raw), 0, num_classes - 1).astype(int).tolist()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n  MAE (continuous): {mae:.4f}")
    print(f"  Accuracy (rounded): {acc:.4f}")
    print(f"  Macro F1 (rounded): {f1:.4f}")

    # Classification report (on rounded predictions)
    label_names = [get_label_name(c) for c in range(num_classes)]
    report = classification_report(
        y_true, y_pred, target_names=label_names, digits=3, zero_division=0,
    )
    print(f"\nClassification Report (rounded predictions):\n{report}")

    # Save report to file
    report_path = output_dir / "test_report.txt"
    with open(report_path, "w") as f:
        f.write(f"MAE (continuous): {mae:.4f}\n")
        f.write(f"Accuracy (rounded): {acc:.4f}\n")
        f.write(f"Macro F1 (rounded): {f1:.4f}\n\n")
        f.write(report)
    print(f"Saved test report → {report_path}")

    # Plots
    plot_confusion_matrix(
        y_true, y_pred, num_classes,
        output_path=output_dir / "confusion_matrix.png",
    )
    plot_confusion_matrix(
        y_true, y_pred, num_classes,
        output_path=output_dir / "confusion_matrix_normalized.png",
        normalize=True,
    )
    plot_per_class_accuracy(
        y_true, y_pred, num_classes,
        output_path=output_dir / "per_class_accuracy.png",
    )
    plot_prediction_distribution(
        y_true, y_pred, num_classes,
        output_path=output_dir / "prediction_distribution.png",
    )

    return {"accuracy": acc, "f1_macro": f1, "mae": mae}


# ---------------------------------------------------------------------------
# Standalone CLI: evaluate a saved model on a given dataset
# ---------------------------------------------------------------------------


def evaluate_from_checkpoint(
    model_dir: str,
    midi_dir: str,
    labels_json: str,
    output_dir: str,
    batch_size: int = 32,
) -> None:
    """Load a saved ensemble and evaluate it on all matched MIDI files."""
    from features import extract_features_batch, FeatureNormalizer, NUM_FEATURES

    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load config saved during training
    cfg = load_config(model_dir)

    # Load data
    label_map = load_labels(labels_json)
    midi_files = discover_midi_files(midi_dir)
    matched_files, matched_labels = match_files_to_labels(midi_files, label_map)
    num_classes = get_num_classes(matched_labels)

    print(f"Evaluating {len(matched_files)} files with {num_classes} classes")

    # Load MLP model
    mlp = FeatureMLPRegressor(
        num_features=cfg.get("num_features", NUM_FEATURES),
        hidden_dim=cfg.get("hidden_dim", 128),
        dropout=cfg.get("dropout", 0.3),
    )
    best_model_dir = model_dir / "best_model"
    safetensors_path = best_model_dir / "model.safetensors"
    pytorch_path = best_model_dir / "pytorch_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path, device="cpu")
    elif pytorch_path.exists():
        state_dict = torch.load(pytorch_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No model weights found in {best_model_dir}")

    mlp.load_state_dict(state_dict, strict=False)
    mlp.eval()

    # Load ensemble
    mlp_weight = cfg.get("mlp_weight", 0.5)
    ensemble = EnsembleRegressor.load(model_dir, mlp, mlp_weight=mlp_weight)

    # Extract and normalise features
    normalizer_path = model_dir / "feature_normalizer.npz"
    normalizer = FeatureNormalizer.load(normalizer_path)
    feats_raw = extract_features_batch(matched_files)
    feats = normalizer.transform(feats_raw)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_on_test(ensemble, feats, matched_labels, num_classes, output_dir, device=device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a saved model on MIDI data")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing the saved model")
    parser.add_argument("--midi_dir", type=str, default="mid")
    parser.add_argument("--labels_json", type=str, default="data.json")
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    evaluate_from_checkpoint(
        model_dir=args.model_dir,
        midi_dir=args.midi_dir,
        labels_json=args.labels_json,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )

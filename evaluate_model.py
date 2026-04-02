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
from transformers import Trainer, TrainingArguments

from common import (
    load_labels,
    discover_midi_files,
    match_files_to_labels,
    build_tokenizer,
    get_pad_token_id,
    get_vocab_size,
    get_num_classes,
    make_label_func,
    get_label_name,
)
from model import MidiClassifier
from training import ClassificationCollator, build_compute_metrics


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
    trainer: Trainer,
    test_dataset,
    test_labels: list[int],
    num_classes: int,
    output_dir: str | Path,
) -> dict:
    """Run evaluation on the test set and generate all reports and plots."""
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)

    # Get predictions
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    y_pred = np.argmax(logits, axis=-1).tolist()
    y_true = test_labels

    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n  Overall Accuracy: {acc:.4f}")
    print(f"  Macro F1 Score:  {f1:.4f}")

    # Classification report
    label_names = [get_label_name(c) for c in range(num_classes)]
    report = classification_report(
        y_true, y_pred, target_names=label_names, digits=3, zero_division=0,
    )
    print(f"\nClassification Report:\n{report}")

    # Save report to file
    report_path = output_dir / "test_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Overall Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1 Score:  {f1:.4f}\n\n")
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

    return {"accuracy": acc, "f1_macro": f1}


# ---------------------------------------------------------------------------
# Standalone CLI: evaluate a saved model on a given dataset
# ---------------------------------------------------------------------------


def evaluate_from_checkpoint(
    model_dir: str,
    midi_dir: str,
    labels_json: str,
    output_dir: str,
    max_seq_len: int = 1024,
    batch_size: int = 32,
) -> None:
    """Load a saved model and evaluate it on all matched MIDI files."""
    from miditok.pytorch_data import DatasetMIDI

    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = build_tokenizer()
    tokenizer_path = model_dir / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = type(tokenizer)(params=tokenizer_path)
    pad_token_id = get_pad_token_id(tokenizer)
    vocab_size = get_vocab_size(tokenizer)

    # Load data
    label_map = load_labels(labels_json)
    midi_files = discover_midi_files(midi_dir)
    matched_files, matched_labels = match_files_to_labels(midi_files, label_map)
    num_classes = get_num_classes(matched_labels)

    print(f"Evaluating {len(matched_files)} files with {num_classes} classes")

    # Load model
    model = MidiClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        pad_token_id=pad_token_id,
        max_seq_len=max_seq_len,
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

    model.load_state_dict(state_dict, strict=False)

    # Create dataset
    label_func = make_label_func(label_map)
    dataset = DatasetMIDI(
        files_paths=matched_files,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
        func_to_get_labels=label_func,
    )

    collator = ClassificationCollator(pad_token_id=pad_token_id, max_seq_len=max_seq_len)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=batch_size,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        compute_metrics=build_compute_metrics(),
    )

    evaluate_on_test(trainer, dataset, matched_labels, num_classes, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a saved model on MIDI data")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing the saved model")
    parser.add_argument("--midi_dir", type=str, default="mid")
    parser.add_argument("--labels_json", type=str, default="data.json")
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    evaluate_from_checkpoint(
        model_dir=args.model_dir,
        midi_dir=args.midi_dir,
        labels_json=args.labels_json,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
    )

"""
checks.py — Data validation, class distribution analysis, and diagnostic plots.
"""

import os
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from common import (
    load_labels,
    discover_midi_files,
    match_files_to_labels,
    get_num_classes,
    get_label_name,
)


def print_class_distribution(labels: list[int], num_classes: int) -> None:
    """Print per-class counts and percentages."""
    counts = Counter(labels)
    total = len(labels)
    print(f"\n{'Class':<15} {'Count':>6} {'Pct':>7}")
    print("-" * 30)
    for cls_id in range(num_classes):
        c = counts.get(cls_id, 0)
        pct = 100.0 * c / total if total else 0
        name = get_label_name(cls_id)
        print(f"{name:<15} {c:>6} {pct:>6.1f}%")
    print("-" * 30)
    print(f"{'Total':<15} {total:>6}")


def plot_class_distribution(
    labels: list[int],
    num_classes: int,
    output_path: str | Path = "class_distribution.png",
    title: str = "Class Distribution",
) -> None:
    """Save a bar chart of the class distribution."""
    counts = Counter(labels)
    classes = list(range(num_classes))
    names = [get_label_name(c) for c in classes]
    values = [counts.get(c, 0) for c in classes]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, values, color=sns.color_palette("viridis", num_classes))
    ax.set_xlabel("Grade")
    ax.set_ylabel("Count")
    ax.set_title(title)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(val),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved class distribution plot → {output_path}")


def plot_split_distribution(
    train_labels: list[int],
    val_labels: list[int],
    test_labels: list[int],
    num_classes: int,
    output_path: str | Path = "split_distribution.png",
) -> None:
    """Save a grouped bar chart showing class distribution across splits."""
    import numpy as np

    classes = list(range(num_classes))
    names = [get_label_name(c) for c in classes]

    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    test_counts = Counter(test_labels)

    x = np.arange(num_classes)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, [train_counts.get(c, 0) for c in classes], width, label="Train")
    ax.bar(x, [val_counts.get(c, 0) for c in classes], width, label="Val")
    ax.bar(x + width, [test_counts.get(c, 0) for c in classes], width, label="Test")
    ax.set_xlabel("Grade")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution per Split")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved split distribution plot → {output_path}")


def validate_data(
    midi_dir: str | Path,
    labels_json: str | Path,
    output_dir: str | Path = ".",
) -> tuple[list[Path], list[int], int]:
    """Run all data checks and return matched files, labels, and num_classes.

    Prints summaries and saves diagnostic plots to *output_dir*.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load labels
    label_map = load_labels(labels_json)
    print(f"Labels loaded: {len(label_map)} entries from {labels_json}")

    # Discover MIDI files
    midi_files = discover_midi_files(midi_dir)
    print(f"MIDI files found: {len(midi_files)} in {midi_dir}")

    # Match
    matched_files, matched_labels = match_files_to_labels(midi_files, label_map)
    unmatched = len(midi_files) - len(matched_files)
    print(f"Matched to labels: {len(matched_files)} files ({unmatched} unmatched)")

    if not matched_files:
        raise RuntimeError("No MIDI files could be matched to labels. Check paths.")

    num_classes = get_num_classes(matched_labels)
    print(f"Number of classes: {num_classes}")

    # Distribution
    print_class_distribution(matched_labels, num_classes)
    plot_class_distribution(
        matched_labels,
        num_classes,
        output_path=Path(output_dir) / "class_distribution.png",
    )

    return matched_files, matched_labels, num_classes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate data and print diagnostics")
    parser.add_argument("--midi_dir", type=str, default="mid")
    parser.add_argument("--labels_json", type=str, default="data.json")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    validate_data(args.midi_dir, args.labels_json, args.output_dir)

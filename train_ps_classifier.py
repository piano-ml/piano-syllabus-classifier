#!/usr/bin/env python3
"""
train_ps_classifier.py — Main entry point for the Piano Syllabus grade classifier.

Orchestrates the full pipeline:
  1. Data validation and diagnostics
  2. Feature extraction from MIDI files
  3. Training a Feature-MLP classifier (HF Trainer)
  4. Evaluation on the held-out test set
  5. Plots and reports

Usage:
    python train_ps_classifier.py \
        --midi_dir mid \
        --labels_json data.json \
        --output_dir ./ps_model \
        --epochs 6
"""

import argparse
import sys
import os

from training import train
from evaluate_model import evaluate_on_test
from common import get_num_classes, DEFAULT_HPARAMS


def parse_args():
    D = DEFAULT_HPARAMS
    parser = argparse.ArgumentParser(
        description="Train a Piano Syllabus (ABRSM) grade classifier on MIDI files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--midi_dir", type=str, default="mid",
        help="Directory containing .mid files (searched recursively)",
    )
    parser.add_argument(
        "--labels_json", type=str, default="data.json",
        help="JSON file mapping piece names to metadata with 'ps' labels",
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="./ps_model",
        help="Directory for checkpoints, plots, and reports",
    )

    # Training hyper-parameters
    parser.add_argument("--epochs", type=int, default=D["epochs"], help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=D["batch_size"], help="Batch size")
    parser.add_argument("--lr", type=float, default=D["lr"], help="Learning rate")
    parser.add_argument("--dropout", type=float, default=D["dropout"], help="Dropout rate")
    parser.add_argument("--seed", type=int, default=D["seed"], help="Random seed")

    # Memory / performance
    parser.add_argument("--dataloader_num_workers", type=int, default=D["dataloader_num_workers"],
                        help="Number of DataLoader workers (0 = main process only, saves RAM)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=D["gradient_accumulation_steps"],
                        help="Accumulate gradients over N steps (simulates larger batch size)")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Piano Syllabus Grade Classifier")
    print("=" * 60)
    print(f"  MIDI dir:      {args.midi_dir}")
    print(f"  Labels JSON:   {args.labels_json}")
    print(f"  Output dir:    {args.output_dir}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Dropout:       {args.dropout}")
    print(f"  Mode:          MLP + LightGBM ensemble (MAE loss)")
    print(f"  Num workers:   {args.dataloader_num_workers}")
    print(f"  Grad accum:    {args.gradient_accumulation_steps}")
    print("=" * 60)

    # Run training
    trainer, ensemble, test_info = train(
        midi_dir=args.midi_dir,
        labels_json=args.labels_json,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout,
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # Unpack test info
    test_feats, test_files, test_labels = test_info
    num_classes = get_num_classes(test_labels)

    # Final evaluation on test set
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    results = evaluate_on_test(
        ensemble=ensemble,
        test_feats=test_feats,
        test_labels=test_labels,
        num_classes=num_classes,
        output_dir=args.output_dir,
        device=device,
    )

    print("\n" + "=" * 60)
    print("  DONE")
    print(f"  Test Accuracy: {results['accuracy']:.4f}")
    print(f"  Test Macro F1: {results['f1_macro']:.4f}")
    print(f"  Test MAE:      {results['mae']:.4f}")
    print(f"  All outputs saved in: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

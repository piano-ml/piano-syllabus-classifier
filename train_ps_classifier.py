#!/usr/bin/env python3
"""
train_ps_classifier.py — Main entry point for the Piano Syllabus grade classifier.

Orchestrates the full pipeline:
  1. Data validation and diagnostics
  2. Tokenization with REMI (miditok)
  3. Training a Transformer classifier (HF Trainer)
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
from common import get_num_classes


def parse_args():
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
        help="Directory for checkpoints, tokenizer, plots, and reports",
    )

    # Training hyper-parameters
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_seq_len", type=int, default=1024,
                        help="Maximum token sequence length (longer pieces are truncated)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pre_tokenize", action="store_true",
                        help="Pre-tokenize all MIDI files (faster training, uses more RAM)")

    # Model architecture
    parser.add_argument("--d_model", type=int, default=256,
                        help="Transformer embedding dimension")
    parser.add_argument("--nhead", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of Transformer encoder layers")
    parser.add_argument("--dim_feedforward", type=int, default=512,
                        help="Feed-forward dimension in Transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")

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
    print(f"  Max seq len:   {args.max_seq_len}")
    print(f"  Model:         d={args.d_model}, heads={args.nhead}, "
          f"layers={args.num_layers}, ff={args.dim_feedforward}")
    print("=" * 60)

    # Run training
    trainer, tokenizer, model, test_info = train(
        midi_dir=args.midi_dir,
        labels_json=args.labels_json,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        seed=args.seed,
        pre_tokenize=args.pre_tokenize,
    )

    # Unpack test info
    test_dataset, test_files, test_labels = test_info
    num_classes = get_num_classes(test_labels)

    # Final evaluation on test set
    results = evaluate_on_test(
        trainer=trainer,
        test_dataset=test_dataset,
        test_labels=test_labels,
        num_classes=num_classes,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("  DONE")
    print(f"  Test Accuracy: {results['accuracy']:.4f}")
    print(f"  Test Macro F1: {results['f1_macro']:.4f}")
    print(f"  All outputs saved in: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

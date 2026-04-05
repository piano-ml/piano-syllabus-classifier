"""
training.py — Training pipeline using miditok DatasetMIDI + Hugging Face Trainer.
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from miditok import REMI
from miditok.pytorch_data import DatasetMIDI, DataCollator
from transformers import Trainer, TrainingArguments, TrainerCallback
import evaluate as hf_evaluate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (
    load_labels,
    discover_midi_files,
    match_files_to_labels,
    build_tokenizer,
    get_pad_token_id,
    get_vocab_size,
    get_num_classes,
    make_label_func,
    compute_class_weights,
    get_label_name,
    DEFAULT_HPARAMS,
    save_config,
)
from checks import (
    print_class_distribution,
    plot_class_distribution,
    plot_split_distribution,
    validate_data,
)
from model import MidiClassifier, corn_logits_to_class
from augmentation import AugmentedDatasetMIDI


# ---------------------------------------------------------------------------
# Custom collator: wraps miditok padding and adds attention_mask
# ---------------------------------------------------------------------------


class ClassificationCollator:
    """Collate variable-length token sequences for classification.

    Pads input_ids to the longest sequence in the batch, builds an
    attention_mask (1 = real token, 0 = pad), and stacks scalar labels.
    Also truncates sequences that exceed *max_seq_len*.
    """

    def __init__(self, pad_token_id: int, max_seq_len: int):
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        # Filter out corrupted samples (None input_ids)
        batch = [b for b in batch if b.get("input_ids") is not None]
        if not batch:
            raise ValueError("All samples in the batch were corrupted / empty.")

        input_ids_list = []
        labels_list = []
        for b in batch:
            ids = b["input_ids"]
            # Truncate
            if len(ids) > self.max_seq_len:
                ids = ids[: self.max_seq_len]
            input_ids_list.append(ids)
            labels_list.append(b["labels"])

        # Determine max length in this batch
        max_len = max(len(ids) for ids in input_ids_list)

        padded_ids = []
        attention_masks = []
        for ids in input_ids_list:
            seq_len = len(ids)
            pad_len = max_len - seq_len
            padded = F.pad(ids, (0, pad_len), value=self.pad_token_id)
            mask = torch.cat(
                [torch.ones(seq_len, dtype=torch.long),
                 torch.zeros(pad_len, dtype=torch.long)]
            )
            padded_ids.append(padded)
            attention_masks.append(mask)

        result = {
            "input_ids": torch.stack(padded_ids),
            "attention_mask": torch.stack(attention_masks),
        }

        # Stack labels — handle both scalar and 1-element tensors
        stacked_labels = []
        for lab in labels_list:
            if lab.dim() == 0:
                stacked_labels.append(lab)
            else:
                stacked_labels.append(lab.squeeze())
        result["labels"] = torch.stack(stacked_labels)

        return result


# ---------------------------------------------------------------------------
# Epoch-level accuracy tracking callback
# ---------------------------------------------------------------------------


class AccuracyProgressCallback(TrainerCallback):
    """Print and record validation accuracy after each evaluation."""

    def __init__(self):
        self.epoch_accuracies: list[tuple[int, float]] = []  # (epoch, accuracy)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        acc = metrics.get("eval_accuracy")
        epoch = int(state.epoch) if state.epoch else len(self.epoch_accuracies) + 1
        if acc is not None:
            self.epoch_accuracies.append((epoch, acc))
            pct = acc * 100
            print(f"  ► Epoch {epoch} — Validation accuracy: {pct:.2f}%")

    def summary_str(self) -> str:
        """Return a compact arrow-separated summary, e.g. '22% → 24.6% → …'."""
        parts = [f"{acc * 100:.2f}%" for _, acc in self.epoch_accuracies]
        return " → ".join(parts)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def build_compute_metrics(loss_type: str = "ce"):
    """Return a compute_metrics function for the Trainer."""
    accuracy_metric = hf_evaluate.load("accuracy")
    f1_metric = hf_evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        if loss_type == "corn":
            import torch as _torch
            preds = corn_logits_to_class(
                _torch.tensor(logits, dtype=_torch.float)
            ).numpy()
        else:
            preds = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=preds, references=labels)
        f1 = f1_metric.compute(
            predictions=preds, references=labels, average="macro"
        )
        mae = float(np.mean(np.abs(preds - labels)))
        return {"accuracy": acc["accuracy"], "f1_macro": f1["f1"], "mae": mae}

    return compute_metrics


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------


def stratified_split(
    files: list[Path],
    labels: list[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple:
    """Stratified train / val / test split.

    Returns (train_files, val_files, test_files,
             train_labels, val_labels, test_labels).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # First split: train vs (val+test)
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        files,
        labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=random_state,
    )

    # Second split: val vs test (from temp)
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files,
        temp_labels,
        test_size=relative_test,
        stratify=temp_labels,
        random_state=random_state,
    )

    return (
        train_files, val_files, test_files,
        train_labels, val_labels, test_labels,
    )


# ---------------------------------------------------------------------------
# Training loss curve plot
# ---------------------------------------------------------------------------


def plot_training_curves(trainer, output_dir: str | Path) -> None:
    """Plot training & validation loss / accuracy curves from the Trainer log."""
    log_history = trainer.state.log_history

    train_loss, train_epochs = [], []
    eval_loss, eval_acc, eval_f1, eval_epochs = [], [], [], []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_loss.append(entry["loss"])
            train_epochs.append(entry.get("epoch", 0))
        if "eval_loss" in entry:
            eval_loss.append(entry["eval_loss"])
            eval_acc.append(entry.get("eval_accuracy", 0))
            eval_f1.append(entry.get("eval_f1_macro", 0))
            eval_epochs.append(entry.get("epoch", 0))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(train_epochs, train_loss, label="Train Loss", alpha=0.7)
    if eval_loss:
        axes[0].plot(eval_epochs, eval_loss, label="Val Loss", marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()

    # Accuracy
    if eval_acc:
        eval_acc_pct = [a * 100 for a in eval_acc]
        axes[1].plot(eval_epochs, eval_acc_pct, marker="o", color="green")
        for x, y in zip(eval_epochs, eval_acc_pct):
            axes[1].annotate(f"{y:.2f}%", (x, y),
                             textcoords="offset points", xytext=(0, 8),
                             ha="center", fontsize=8)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Validation Accuracy")

    # F1
    if eval_f1:
        axes[2].plot(eval_epochs, eval_f1, marker="o", color="orange")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Macro F1")
        axes[2].set_title("Validation Macro F1")

    plt.tight_layout()
    path = Path(output_dir) / "training_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved training curves → {path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(
    midi_dir: str,
    labels_json: str,
    output_dir: str,
    epochs: int = DEFAULT_HPARAMS["epochs"],
    batch_size: int = DEFAULT_HPARAMS["batch_size"],
    learning_rate: float = DEFAULT_HPARAMS["lr"],
    max_seq_len: int = DEFAULT_HPARAMS["max_seq_len"],
    d_model: int = DEFAULT_HPARAMS["d_model"],
    nhead: int = DEFAULT_HPARAMS["nhead"],
    num_layers: int = DEFAULT_HPARAMS["num_layers"],
    dim_feedforward: int = DEFAULT_HPARAMS["dim_feedforward"],
    dropout: float = DEFAULT_HPARAMS["dropout"],
    seed: int = DEFAULT_HPARAMS["seed"],
    pre_tokenize: bool = False,
    loss_type: str = DEFAULT_HPARAMS["loss_type"],
    pitch_augment_range: int = DEFAULT_HPARAMS["pitch_augment_range"],
    augment_prob: float | list[float] = DEFAULT_HPARAMS["augment_prob"],
    dataloader_num_workers: int = DEFAULT_HPARAMS["dataloader_num_workers"],
    gradient_accumulation_steps: int = DEFAULT_HPARAMS["gradient_accumulation_steps"],
) -> tuple:
    """Full training pipeline. Returns (trainer, tokenizer, model, test info)."""

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Validate data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Data validation")
    print("=" * 60)
    matched_files, matched_labels, num_classes = validate_data(
        midi_dir, labels_json, output_dir
    )

    # ------------------------------------------------------------------
    # 2. Stratified split
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Stratified train/val/test split")
    print("=" * 60)
    (
        train_files, val_files, test_files,
        train_labels, val_labels, test_labels,
    ) = stratified_split(matched_files, matched_labels, random_state=seed)

    print(f"  Train: {len(train_files)} samples")
    print(f"  Val:   {len(val_files)} samples")
    print(f"  Test:  {len(test_files)} samples")

    plot_split_distribution(
        train_labels, val_labels, test_labels, num_classes,
        output_path=Path(output_dir) / "split_distribution.png",
    )

    # ------------------------------------------------------------------
    # 3. Tokenizer
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Building REMI tokenizer")
    print("=" * 60)
    tokenizer = build_tokenizer()
    pad_token_id = get_pad_token_id(tokenizer)
    vocab_size = get_vocab_size(tokenizer)
    print(f"  Vocab size: {vocab_size}")
    print(f"  PAD token id: {pad_token_id}")

    # Save tokenizer for later inference
    tokenizer_path = Path(output_dir) / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    print(f"  Tokenizer saved → {tokenizer_path}")

    # Save config so inference/evaluation can reconstruct the model
    save_config({
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "max_seq_len": max_seq_len,
        "loss_type": loss_type,
        "num_classes": num_classes,
        "vocab_size": vocab_size,
        "pad_token_id": pad_token_id,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "augment_train": pitch_augment_range > 0,
        "pitch_augment_range": pitch_augment_range,
        "augment_prob": augment_prob,
    }, output_dir)

    # ------------------------------------------------------------------
    # 4. Build label maps for each split
    # ------------------------------------------------------------------
    label_map_all = {f.stem: lbl for f, lbl in zip(matched_files, matched_labels)}
    label_func = make_label_func(label_map_all)

    # ------------------------------------------------------------------
    # 5. Create DatasetMIDI instances
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Creating datasets" + (" (pre-tokenizing...)" if pre_tokenize else ""))
    print("=" * 60)

    ds_kwargs = dict(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
        func_to_get_labels=label_func,
        pre_tokenize=pre_tokenize,
    )

    # Training set: optionally use AugmentedDatasetMIDI for on-the-fly transposition
    if pitch_augment_range > 0 and not pre_tokenize:
        train_dataset = AugmentedDatasetMIDI(
            files_paths=train_files,
            pitch_augment_range=pitch_augment_range,
            augment_prob=augment_prob,
            label_map=label_map_all,
            **ds_kwargs,
        )
        print(f"  Augmentation: ON (±{pitch_augment_range} semitones, prob={augment_prob})")
        print(f"  Training samples augmented on-the-fly: {len(train_dataset)}")
    else:
        train_dataset = DatasetMIDI(files_paths=train_files, **ds_kwargs)
        if pitch_augment_range > 0 and pre_tokenize:
            print("  WARNING: augmentation disabled (incompatible with --pre_tokenize)")
        else:
            print("  Augmentation: OFF")

    val_dataset = DatasetMIDI(files_paths=val_files, **ds_kwargs)
    test_dataset = DatasetMIDI(files_paths=test_files, **ds_kwargs)

    print(f"  Train dataset: {len(train_dataset)} samples")
    print(f"  Val dataset:   {len(val_dataset)} samples")
    print(f"  Test dataset:  {len(test_dataset)} samples")

    # ------------------------------------------------------------------
    # 6. Build model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Building Transformer classifier")
    print("=" * 60)

    class_weights = compute_class_weights(train_labels, num_classes)
    print(f"  Class weights: {class_weights.tolist()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MidiClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_seq_len=max_seq_len,
        pad_token_id=pad_token_id,
        class_weights=class_weights.to(device) if device == "cuda" else class_weights,
        loss_type=loss_type,
    )

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {train_params:,}")

    # ------------------------------------------------------------------
    # 7. Training arguments & Trainer
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 6: Training")
    print("=" * 60)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.15,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=dataloader_num_workers,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
        report_to="none",
        remove_unused_columns=False,  # Our model uses custom column names
    )

    collator = ClassificationCollator(
        pad_token_id=pad_token_id,
        max_seq_len=max_seq_len,
    )

    acc_callback = AccuracyProgressCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=build_compute_metrics(loss_type),
        callbacks=[acc_callback],
    )

    trainer.train()

    # Print accuracy progression summary
    if acc_callback.epoch_accuracies:
        summary = acc_callback.summary_str()
        print(f"\n  Validation accuracy across epochs: {summary}")

    # Save best model
    trainer.save_model(Path(output_dir) / "best_model")
    print(f"\nBest model saved → {Path(output_dir) / 'best_model'}")

    # Plot training curves
    plot_training_curves(trainer, output_dir)

    return trainer, tokenizer, model, (test_dataset, test_files, test_labels)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    D = DEFAULT_HPARAMS
    parser = argparse.ArgumentParser(description="Train MIDI grade classifier")
    parser.add_argument("--midi_dir", type=str, default="mid")
    parser.add_argument("--labels_json", type=str, default="data.json")
    parser.add_argument("--output_dir", type=str, default="./ps_model")
    parser.add_argument("--epochs", type=int, default=D["epochs"])
    parser.add_argument("--batch_size", type=int, default=D["batch_size"])
    parser.add_argument("--lr", type=float, default=D["lr"])
    parser.add_argument("--max_seq_len", type=int, default=D["max_seq_len"])
    parser.add_argument("--d_model", type=int, default=D["d_model"])
    parser.add_argument("--nhead", type=int, default=D["nhead"])
    parser.add_argument("--num_layers", type=int, default=D["num_layers"])
    parser.add_argument("--dim_feedforward", type=int, default=D["dim_feedforward"])
    parser.add_argument("--dropout", type=float, default=D["dropout"])
    parser.add_argument("--seed", type=int, default=D["seed"])
    parser.add_argument("--pre_tokenize", action="store_true",
                        help="Pre-tokenize all files (faster training, more RAM)")
    parser.add_argument("--loss_type", type=str, default=D["loss_type"],
                        choices=["ce", "corn"],
                        help="Loss function: 'ce' (cross-entropy) or 'corn' (ordinal)")
    parser.add_argument("--pitch_augment_range", type=int, default=D["pitch_augment_range"],
                        help="Max semitones for transposition augmentation (uses -N to +N, 0 = disabled)")
    parser.add_argument("--augment_prob", type=str, default=None,
                        help="Per-class augment probability as comma-separated floats "
                             "(e.g. '0.4,0.4,0.4,0.4,0.4,0.4,0.4,0,0') or a single float")
    args = parser.parse_args()

    # Parse augment_prob: single float or comma-separated list
    if args.augment_prob is not None:
        parts = args.augment_prob.split(",")
        if len(parts) == 1:
            augment_prob_parsed: float | list[float] = float(parts[0])
        else:
            augment_prob_parsed = [float(p) for p in parts]
    else:
        augment_prob_parsed = D["augment_prob"]

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
        loss_type=args.loss_type,
        pitch_augment_range=args.pitch_augment_range,
        augment_prob=augment_prob_parsed,
    )

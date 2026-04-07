"""
training.py — Training pipeline using handcrafted features + MLP classifier.
"""

import os
from pathlib import Path

import math

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (
    load_labels,
    discover_midi_files,
    match_files_to_labels,
    get_num_classes,
    make_label_func,
    DEFAULT_HPARAMS,
    save_config,
)
from checks import (
    print_class_distribution,
    plot_class_distribution,
    plot_split_distribution,
    validate_data,
)
from model import FeatureMLPRegressor, EnsembleRegressor
from augmentation import FeatureDatasetMIDI
from features import (
    extract_features_batch,
    FeatureNormalizer,
    NUM_FEATURES,
    FEATURE_NAMES,
)


# ---------------------------------------------------------------------------
# Custom collator: wraps miditok padding and adds attention_mask
# ---------------------------------------------------------------------------


class FeaturesOnlyCollator:
    """Collate feature vectors + labels for the MLP regressor.

    Ignores input_ids entirely — the model only needs features.
    Labels are cast to float for regression (MAE loss).
    """

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        batch = [b for b in batch if b.get("features") is not None]
        if not batch:
            raise ValueError("All samples in the batch had no features.")

        features_list = [b["features"] for b in batch]
        labels_list = [b["labels"] for b in batch]

        stacked_labels = []
        for lab in labels_list:
            if lab.dim() == 0:
                stacked_labels.append(lab)
            else:
                stacked_labels.append(lab.squeeze())

        return {
            "features": torch.stack(features_list),
            "labels": torch.stack(stacked_labels).float(),
        }


# ---------------------------------------------------------------------------
# Epoch-level accuracy tracking callback
# ---------------------------------------------------------------------------


class EvalProgressCallback(TrainerCallback):
    """Print and record validation MAE after each evaluation."""

    def __init__(self):
        self.epoch_maes: list[tuple[int, float]] = []  # (epoch, mae)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        mae = metrics.get("eval_mae")
        epoch = int(state.epoch) if state.epoch else len(self.epoch_maes) + 1
        if mae is not None:
            self.epoch_maes.append((epoch, mae))
            acc = metrics.get("eval_accuracy", 0) * 100
            print(f"  ► Epoch {epoch} — Val MAE: {mae:.3f}  Accuracy: {acc:.2f}%")

    def summary_str(self) -> str:
        """Return a compact arrow-separated summary of MAE."""
        parts = [f"{mae:.3f}" for _, mae in self.epoch_maes]
        return " → ".join(parts)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def build_compute_metrics():
    """Return a compute_metrics function for regression.

    Predictions are continuous; we round-and-clamp to [0, 10] for
    accuracy / classification-style metrics.
    """

    def compute_metrics(eval_pred):
        preds_raw, labels = eval_pred
        if isinstance(preds_raw, tuple):
            preds_raw = preds_raw[0]
        # Flatten in case of (B, 1)
        preds_raw = preds_raw.squeeze()
        labels = labels.squeeze()

        mae = float(np.mean(np.abs(preds_raw - labels)))

        # Round to nearest integer grade for accuracy / F1
        preds_int = np.clip(np.round(preds_raw), 1, 8).astype(int)
        labels_int = np.round(labels).astype(int)
        accuracy = float(np.mean(preds_int == labels_int))

        return {"mae": mae, "accuracy": accuracy}

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
    """Plot training & validation loss / accuracy / MAE curves from the Trainer log."""
    log_history = trainer.state.log_history

    train_loss, train_epochs = [], []
    eval_loss, eval_acc, eval_mae, eval_epochs = [], [], [], []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_loss.append(entry["loss"])
            train_epochs.append(entry.get("epoch", 0))
        if "eval_loss" in entry:
            eval_loss.append(entry["eval_loss"])
            eval_acc.append(entry.get("eval_accuracy", 0))
            eval_mae.append(entry.get("eval_mae", 0))
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

    # Accuracy (rounded predictions)
    if eval_acc:
        eval_acc_pct = [a * 100 for a in eval_acc]
        axes[1].plot(eval_epochs, eval_acc_pct, marker="o", color="green")
        for x, y in zip(eval_epochs, eval_acc_pct):
            axes[1].annotate(f"{y:.2f}%", (x, y),
                             textcoords="offset points", xytext=(0, 8),
                             ha="center", fontsize=8)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Validation Accuracy (rounded)")

    # MAE
    if eval_mae:
        axes[2].plot(eval_epochs, eval_mae, marker="o", color="orange")
        for x, y in zip(eval_epochs, eval_mae):
            axes[2].annotate(f"{y:.3f}", (x, y),
                             textcoords="offset points", xytext=(0, 8),
                             ha="center", fontsize=8)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("MAE")
        axes[2].set_title("Validation MAE")

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
    dropout: float = DEFAULT_HPARAMS["dropout"],
    seed: int = DEFAULT_HPARAMS["seed"],
    dataloader_num_workers: int = DEFAULT_HPARAMS["dataloader_num_workers"],
    gradient_accumulation_steps: int = DEFAULT_HPARAMS["gradient_accumulation_steps"],
) -> tuple:
    """Full training pipeline. Returns (trainer, model, test info)."""

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
    # 3. Extract handcrafted features
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Extracting handcrafted features")
    print("=" * 60)
    print(f"  Feature dimensions: {NUM_FEATURES}")
    print(f"  Features: {', '.join(FEATURE_NAMES)}")

    # Extract raw features for all files
    train_feats_raw = extract_features_batch(train_files)
    val_feats_raw = extract_features_batch(val_files)
    test_feats_raw = extract_features_batch(test_files)

    # Fit normalizer on training set only
    normalizer = FeatureNormalizer().fit(train_feats_raw)
    normalizer.save(Path(output_dir) / "feature_normalizer.npz")
    print(f"  Feature normalizer saved → {Path(output_dir) / 'feature_normalizer.npz'}")

    train_feats = normalizer.transform(train_feats_raw)
    val_feats = normalizer.transform(val_feats_raw)
    test_feats = normalizer.transform(test_feats_raw)

    # Build path → tensor lookup
    feature_vectors = {}
    for paths, feats in [
        (train_files, train_feats),
        (val_files, val_feats),
        (test_files, test_feats),
    ]:
        for p, f in zip(paths, feats):
            feature_vectors[str(p)] = torch.tensor(f, dtype=torch.float32)

    print(f"  Features extracted and normalised for {len(feature_vectors)} files")

    # ------------------------------------------------------------------
    # 4. Create datasets
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Creating datasets")
    print("=" * 60)

    label_map_all = {f.stem: lbl for f, lbl in zip(matched_files, matched_labels)}
    label_func = make_label_func(label_map_all)

    from common import build_tokenizer, get_pad_token_id
    tokenizer = build_tokenizer()

    ds_kwargs = dict(
        tokenizer=tokenizer,
        max_seq_len=512,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
        func_to_get_labels=label_func,
    )

    train_dataset = FeatureDatasetMIDI(
        files_paths=train_files, feature_vectors=feature_vectors, **ds_kwargs
    )
    val_dataset = FeatureDatasetMIDI(
        files_paths=val_files, feature_vectors=feature_vectors, **ds_kwargs
    )
    test_dataset = FeatureDatasetMIDI(
        files_paths=test_files, feature_vectors=feature_vectors, **ds_kwargs
    )

    print(f"  Train dataset: {len(train_dataset)} samples")
    print(f"  Val dataset:   {len(val_dataset)} samples")
    print(f"  Test dataset:  {len(test_dataset)} samples")

    # ------------------------------------------------------------------
    # 5. Build model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Building Feature-MLP regressor")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    hidden_dim = DEFAULT_HPARAMS["hidden_dim"]
    num_hidden_layers = DEFAULT_HPARAMS["num_hidden_layers"]
    use_batch_norm = DEFAULT_HPARAMS["use_batch_norm"]
    activation = DEFAULT_HPARAMS["activation"]

    corn_task_weights = DEFAULT_HPARAMS.get("corn_task_weights")

    model = FeatureMLPRegressor(
        num_features=NUM_FEATURES,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_classes=num_classes,
        num_hidden_layers=num_hidden_layers,
        use_batch_norm=use_batch_norm,
        activation=activation,
        corn_task_weights=corn_task_weights,
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

    weight_decay = DEFAULT_HPARAMS["weight_decay"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=dataloader_num_workers,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
        report_to="none",
        remove_unused_columns=False,  # Our model uses custom column names
    )

    # Custom optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )

    # Scheduler
    scheduler_name = DEFAULT_HPARAMS["scheduler"]
    steps_per_epoch = math.ceil(len(train_dataset) / batch_size / gradient_accumulation_steps)
    total_steps = steps_per_epoch * epochs

    if scheduler_name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
        )
    else:  # "cosine" (default)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-6,
        )

    collator = FeaturesOnlyCollator()

    acc_callback = EvalProgressCallback()

    # Early stopping
    early_stopping_patience = DEFAULT_HPARAMS["early_stopping_patience"]
    callbacks = [acc_callback]
    if early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
        ))
        print(f"  Early stopping enabled (patience={early_stopping_patience}, metric=mae)")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=build_compute_metrics(),
        callbacks=callbacks,
        optimizers=(optimizer, scheduler),
    )

    trainer.train()

    # Print MAE progression summary
    if acc_callback.epoch_maes:
        summary = acc_callback.summary_str()
        print(f"\n  Validation MAE across epochs: {summary}")

    # Save best model
    trainer.save_model(Path(output_dir) / "best_model")
    print(f"\nBest model saved → {Path(output_dir) / 'best_model'}")

    # Plot training curves
    plot_training_curves(trainer, output_dir)

    # ------------------------------------------------------------------
    # 7. Train LightGBM regressor
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 7: Training LightGBM regressor")
    print("=" * 60)
    import lightgbm as lgb

    lgbm_train = lgb.Dataset(train_feats, label=np.array(train_labels, dtype=np.float32))
    lgbm_val = lgb.Dataset(val_feats, label=np.array(val_labels, dtype=np.float32), reference=lgbm_train)

    lgbm_params = {
        "objective": "mae",
        "metric": "mae",
        "verbosity": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": seed,
    }

    lgbm_callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=50),
    ]

    lgbm_model = lgb.train(
        lgbm_params,
        lgbm_train,
        num_boost_round=500,
        valid_sets=[lgbm_val],
        valid_names=["val"],
        callbacks=lgbm_callbacks,
    )

    lgbm_val_pred = lgbm_model.predict(val_feats)
    lgbm_val_mae = float(np.mean(np.abs(lgbm_val_pred - np.array(val_labels))))
    print(f"  LightGBM validation MAE: {lgbm_val_mae:.3f}")

    # ------------------------------------------------------------------
    # 8. Build ensemble
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 8: Building MLP + LightGBM ensemble")
    print("=" * 60)

    # Find optimal weight on validation set
    best_w, best_mae = 0.5, float("inf")
    mlp_model = trainer.model
    mlp_model.eval()
    device_eval = next(mlp_model.parameters()).device
    with torch.no_grad():
        val_feat_t = torch.tensor(val_feats, dtype=torch.float32, device=device_eval)
        mlp_val_pred = mlp_model(features=val_feat_t)["logits"].cpu().numpy()
    for w in np.arange(0.1, 0.95, 0.05):
        ens_pred = w * mlp_val_pred + (1 - w) * lgbm_val_pred
        ens_mae = float(np.mean(np.abs(ens_pred - np.array(val_labels))))
        if ens_mae < best_mae:
            best_mae = ens_mae
            best_w = round(float(w), 2)

    print(f"  Optimal MLP weight: {best_w}")
    print(f"  Ensemble validation MAE: {best_mae:.3f}")
    print(f"  (MLP alone: {float(np.mean(np.abs(mlp_val_pred - np.array(val_labels)))):.3f}, "
          f"LGBM alone: {lgbm_val_mae:.3f})")

    ensemble = EnsembleRegressor(mlp_model, lgbm_model, mlp_weight=best_w)
    ensemble.save(output_dir)

    # Save ensemble weight in config
    save_config({
        "dropout": dropout,
        "num_classes": num_classes,
        "num_hidden_layers": num_hidden_layers,
        "use_batch_norm": use_batch_norm,
        "activation": activation,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "num_features": NUM_FEATURES,
        "hidden_dim": hidden_dim,
        "mode": "ensemble",
        "mlp_weight": best_w,
    }, output_dir)

    return trainer, ensemble, (test_feats, test_files, test_labels)


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
    parser.add_argument("--dropout", type=float, default=D["dropout"])
    parser.add_argument("--seed", type=int, default=D["seed"])
    args = parser.parse_args()

    trainer, model, test_info = train(
        midi_dir=args.midi_dir,
        labels_json=args.labels_json,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout,
        seed=args.seed,
    )

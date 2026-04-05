"""
inference.py — Predict the piano grade for a single MIDI file.

Usage:
    python inference.py --model_dir ./ps_model --midi_file some_piece.mid
"""

import sys
from pathlib import Path

import torch
import numpy as np

from common import (
    get_label_name,
    load_config,
)
from model import FeatureMLPRegressor, EnsembleRegressor
from features import extract_features, FeatureNormalizer, NUM_FEATURES


def load_model(
    model_dir: str | Path,
) -> tuple[EnsembleRegressor, FeatureNormalizer, dict]:
    """Load the ensemble (MLP + LightGBM), feature normalizer, and config.

    Returns (ensemble, normalizer, config).
    """
    model_dir = Path(model_dir)

    # Load config saved during training
    cfg = load_config(model_dir)

    # Build MLP with saved architecture
    mlp = FeatureMLPRegressor(
        num_features=cfg.get("num_features", NUM_FEATURES),
        hidden_dim=cfg.get("hidden_dim", 128),
        dropout=cfg.get("dropout", 0.3),
    )

    # Load weights — try safetensors first, then pytorch
    best_model_dir = model_dir / "best_model"
    safetensors_path = best_model_dir / "model.safetensors"
    pytorch_path = best_model_dir / "pytorch_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path, device="cpu")
    elif pytorch_path.exists():
        state_dict = torch.load(pytorch_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(
            f"No model weights found in {best_model_dir}. "
            f"Expected model.safetensors or pytorch_model.bin"
        )

    mlp.load_state_dict(state_dict, strict=False)
    mlp.eval()

    # Load ensemble (MLP + LightGBM)
    mlp_weight = cfg.get("mlp_weight", 0.5)
    ensemble = EnsembleRegressor.load(model_dir, mlp, mlp_weight=mlp_weight)

    # Load feature normalizer
    normalizer = FeatureNormalizer.load(model_dir / "feature_normalizer.npz")

    return ensemble, normalizer, cfg


def predict_grade(
    midi_path: str | Path,
    ensemble: EnsembleRegressor,
    normalizer: FeatureNormalizer,
    device: str = "cpu",
) -> dict:
    """Predict the piano grade for a single MIDI file.

    Returns a dict with:
        - predicted_value: float  (continuous ensemble output)
        - predicted_label: int    (rounded & clamped to 0-8)
        - predicted_grade: str    (human-readable)
    """
    midi_path = Path(midi_path)
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    # Extract and normalise features
    raw_feats = extract_features(midi_path)
    feats = normalizer.transform(raw_feats.reshape(1, -1))

    # Ensemble prediction
    pred_value = float(ensemble.predict(feats, device=device)[0])
    pred_label = int(np.clip(round(pred_value), 0, 8))

    return {
        "file": str(midi_path),
        "predicted_value": round(pred_value, 3),
        "predicted_label": pred_label,
        "predicted_grade": get_label_name(pred_label),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict piano grade for a MIDI file"
    )
    parser.add_argument("--midi_file", type=str, required=True,
                        help="Path to the MIDI file")
    parser.add_argument("--model_dir", type=str, default="./ps_model",
                        help="Directory containing the trained model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading ensemble from {args.model_dir}...")
    ensemble, normalizer, cfg = load_model(model_dir=args.model_dir)

    print(f"Predicting grade for: {args.midi_file}")
    result = predict_grade(
        midi_path=args.midi_file,
        ensemble=ensemble,
        normalizer=normalizer,
        device=device,
    )

    print(f"\n{'=' * 50}")
    print(f"File:       {result['file']}")
    print(f"Raw value:  {result['predicted_value']:.3f}")
    print(f"Predicted:  {result['predicted_grade']} (label={result['predicted_label']})")


if __name__ == "__main__":
    main()

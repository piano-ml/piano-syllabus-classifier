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
    build_tokenizer,
    get_pad_token_id,
    get_vocab_size,
    get_label_name,
)
from model import MidiClassifier, corn_logits_to_probs


def load_model(
    model_dir: str | Path,
    max_seq_len: int = 1024,
    d_model: int = 512,
    nhead: int = 8,
    num_layers: int = 8,
    dim_feedforward: int = 2048,
    num_classes: int = 11,
    loss_type: str = "corn",
) -> tuple[MidiClassifier, object, int]:
    """Load a saved model and its tokenizer.

    Returns (model, tokenizer, pad_token_id).
    """
    model_dir = Path(model_dir)

    # Load tokenizer
    tokenizer = build_tokenizer()
    tokenizer_path = model_dir / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = type(tokenizer)(params=tokenizer_path)

    pad_token_id = get_pad_token_id(tokenizer)
    vocab_size = get_vocab_size(tokenizer)

    # Build model with same architecture
    model = MidiClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_seq_len=max_seq_len,
        pad_token_id=pad_token_id,
        loss_type=loss_type,
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

    # strict=False to skip training-only buffers like class_weights
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, tokenizer, pad_token_id


def predict_grade(
    midi_path: str | Path,
    model: MidiClassifier,
    tokenizer,
    pad_token_id: int,
    max_seq_len: int = 1024,
    device: str = "cpu",
) -> dict:
    """Predict the piano grade for a single MIDI file.

    Returns a dict with:
        - predicted_label: int
        - predicted_grade: str (human-readable)
        - probabilities: dict {grade_name: probability}
        - confidence: float (probability of the top prediction)
    """
    from symusic import Score

    midi_path = Path(midi_path)
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    # Load and tokenize
    score = Score(midi_path)
    tokseq = tokenizer.encode(score)

    # Get token ids — encode() returns a list when one_token_stream is False
    if isinstance(tokseq, list):
        token_ids = tokseq[0].ids
    else:
        token_ids = tokseq.ids
    if len(token_ids) > max_seq_len:
        token_ids = token_ids[:max_seq_len]

    # Convert to tensor
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    # Forward pass
    model = model.to(device)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = output["logits"]

    # Convert logits to per-class probabilities
    if model.loss_type == "corn":
        probs = corn_logits_to_probs(logits, model.num_classes).squeeze(0).cpu().numpy()
    else:
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    predicted_label = int(np.argmax(probs))
    confidence = float(probs[predicted_label])

    # Build probability dict
    prob_dict = {}
    for i, p in enumerate(probs):
        prob_dict[get_label_name(i)] = round(float(p), 4)

    return {
        "file": str(midi_path),
        "predicted_label": predicted_label,
        "predicted_grade": get_label_name(predicted_label),
        "confidence": round(confidence, 4),
        "probabilities": prob_dict,
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
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--loss_type", type=str, default="corn",
                        choices=["ce", "corn"],
                        help="Loss type the model was trained with")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {args.model_dir}...")
    model, tokenizer, pad_token_id = load_model(
        model_dir=args.model_dir,
        max_seq_len=args.max_seq_len,
        num_classes=args.num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        loss_type=args.loss_type,
    )

    print(f"Predicting grade for: {args.midi_file}")
    result = predict_grade(
        midi_path=args.midi_file,
        model=model,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        max_seq_len=args.max_seq_len,
        device=device,
    )

    print(f"\n{'=' * 50}")
    print(f"File:       {result['file']}")
    print(f"Predicted:  {result['predicted_grade']} (label={result['predicted_label']})")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nAll probabilities:")
    for grade, prob in result["probabilities"].items():
        bar = "█" * int(prob * 40)
        print(f"  {grade:<12} {prob:>6.2%}  {bar}")


if __name__ == "__main__":
    main()

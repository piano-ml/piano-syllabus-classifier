"""
common.py — Shared configuration, label management, tokenizer, and utilities.
"""

import json
import os
from pathlib import Path
from collections import Counter

import torch
from miditok import REMI, TokenizerConfig


# ---------------------------------------------------------------------------
# Centralised default hyper-parameters (single source of truth)
# ---------------------------------------------------------------------------

DEFAULT_HPARAMS = {
    # MLP architecture
    "hidden_dim": 64,
    "num_hidden_layers": 2,
    "use_batch_norm": True,
    "activation": "relu",
    "dropout": 0.3,
    # Training
    "lr": 5e-4,
    "batch_size": 64,
    "epochs": 100,
    "seed": 42,
    "optimizer": "adamw",
    "weight_decay": 1e-5,
    "warmup_ratio": 0.1,
    # CORN task weights (one per threshold k=0..K-2, length = num_classes - 1)
    # Inverse-frequency of task sample counts (mean-normalised to 1.0)
    # Task k trains on samples with y>=k; higher k → fewer samples → higher weight
    #"corn_task_weights": [0.7477, 0.7477, 0.8325, 0.9079, 1.0337, 1.201, 1.4652, 1.8519],,
    "corn_task_weights":  [0.85, 0.85, 0.92, 0.98, 1.05, 1.18, 1.35, 1.62],
    # Scheduler & stopping
    "scheduler": "cosine",
    "early_stopping_patience": 20,
    # Performance
    "dataloader_num_workers": 4,
    "gradient_accumulation_steps": 1,
}


def save_config(config: dict, output_dir: str | Path) -> None:
    """Save model/training config to config.json in output_dir."""
    path = Path(output_dir) / "config.json"
    # Convert non-serialisable values
    serialisable = {}
    for k, v in config.items():
        if isinstance(v, Path):
            serialisable[k] = str(v)
        else:
            serialisable[k] = v
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)
    print(f"  Config saved → {path}")


def load_config(model_dir: str | Path) -> dict:
    """Load config.json from a model directory."""
    path = Path(model_dir) / "config.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No config.json found in {model_dir}. "
            f"Was the model trained with the current version?"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Label mapping: ps value → integer class id
# Piano Syllabus levels: 1–8 (Grade 1–8)
# Initial/0 and 1 are clamped to 1; 8 and above are clamped to 8.
# ---------------------------------------------------------------------------

LABEL_NAMES = {
    1: "Grade 1",
    2: "Grade 2",
    3: "Grade 3",
    4: "Grade 4",
    5: "Grade 5",
    6: "Grade 6",
    7: "Grade 7",
    8: "Grade 8",
}


def parse_ps_label(ps_value: str) -> int:
    """Convert a 'ps' field value to an integer label.

    Handles numeric strings ("1", "7"), the word "Initial" (→ 0),
    and grade strings like "Grade 3" (→ 3).
    """
    if ps_value is None:
        raise ValueError("ps value is None")
    ps_str = str(ps_value).strip()

    label: int | None = None

    # Direct numeric
    if ps_str.isdigit():
        label = int(ps_str)
    # "Initial" → 0
    elif ps_str.lower() == "initial":
        label = 0
    # "Grade X" → X
    elif ps_str.lower().startswith("grade"):
        parts = ps_str.split()
        if len(parts) == 2 and parts[1].isdigit():
            label = int(parts[1])

    if label is None:
        raise ValueError(f"Cannot parse ps value: {ps_value!r}")

    # Clamp: Initial/0 and 1 → 1, 8 and above → 8
    label = max(label, 1)
    label = min(label, 8)

    return label


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_labels(labels_json: str | Path) -> dict[str, int]:
    """Load the JSON label file and return {filename_stem: int_label}.

    The JSON file maps piece keys (matching MIDI filenames without extension)
    to metadata dicts containing a 'ps' field.
    """
    with open(labels_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    label_map: dict[str, int] = {}
    for key, meta in data.items():
        ps = meta.get("ps") or meta.get("ps_rating")
        if ps is None:
            continue
        label_map[key] = parse_ps_label(ps)

    return label_map


def discover_midi_files(midi_dir: str | Path) -> list[Path]:
    """Recursively find all .mid / .midi files under *midi_dir*."""
    midi_dir = Path(midi_dir)
    files = sorted(midi_dir.rglob("*.mid")) + sorted(midi_dir.rglob("*.midi"))
    return files


def match_files_to_labels(
    midi_files: list[Path], label_map: dict[str, int]
) -> tuple[list[Path], list[int]]:
    """Return only the MIDI files that have a label, along with their labels."""
    matched_files: list[Path] = []
    matched_labels: list[int] = []
    for f in midi_files:
        key = f.stem
        if key in label_map:
            matched_files.append(f)
            matched_labels.append(label_map[key])
    return matched_files, matched_labels


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def build_tokenizer() -> REMI:
    """Create a REMI tokenizer with settings optimised for piano music."""
    config = TokenizerConfig(
        pitch_range=(21, 109),          # Full 88-key piano range
        num_velocities=32,              # Good resolution without bloating vocab
        use_chords=True,                # Chord tokens capture harmonic structure
        use_tempos=True,                # Tempo matters for grade difficulty
        use_time_signatures=True,       # Some pieces have varying time sigs
        use_sustain_pedals=False,       # Skip to keep vocab smaller
        use_rests=False,                # REMI already has TimeShift tokens
        use_programs=False,             # Piano-only → no program tokens
        num_tempos=32,
        tempo_range=(40, 250),
        encode_ids_split="no",          # One flat sequence per piece
        special_tokens=["PAD", "BOS", "EOS", "MASK"],
    )
    tokenizer = REMI(config)
    return tokenizer


def get_pad_token_id(tokenizer: REMI) -> int:
    """Return the integer id of the PAD token."""
    return tokenizer.pad_token_id


def get_vocab_size(tokenizer: REMI) -> int:
    """Return the vocabulary size (needed for the embedding layer)."""
    return len(tokenizer)


# ---------------------------------------------------------------------------
# Label function factory for DatasetMIDI
# ---------------------------------------------------------------------------


def make_label_func(label_map: dict[str, int]):
    """Return a callable compatible with DatasetMIDI.func_to_get_labels.

    Signature: (Score, TokSequence | list[TokSequence], Path) → LongTensor (scalar)
    """

    def func_to_get_labels(score, tokseq, path: Path):
        key = path.stem
        label = label_map.get(key)
        if label is None:
            raise KeyError(f"No label found for {key}")
        return torch.tensor(label, dtype=torch.long)

    return func_to_get_labels


# ---------------------------------------------------------------------------
# Class-weight computation for imbalanced data
# ---------------------------------------------------------------------------


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency weights for cross-entropy loss."""
    counts = Counter(labels)
    total = len(labels)
    weights = torch.zeros(num_classes, dtype=torch.float)
    for cls_id in range(num_classes):
        c = counts.get(cls_id, 0)
        weights[cls_id] = total / (num_classes * c) if c > 0 else 0.0
    return weights


# ---------------------------------------------------------------------------
# Number of classes (auto-detected from data)
# ---------------------------------------------------------------------------


def get_num_classes(labels: list[int]) -> int:
    """Return the number of classes (max label + 1)."""
    return max(labels) + 1


def get_label_name(label_id: int) -> str:
    """Human-readable label name."""
    return LABEL_NAMES.get(label_id, f"Level {label_id}")

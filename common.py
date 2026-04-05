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

#DEFAULT_HPARAMS = {
#    # Model architecture
#    "d_model": 128,
#    "nhead": 8,
#    "num_layers": 3,
#    "dim_feedforward": 512,
#    "dropout": 0.2,
#    "max_seq_len": 1024,
#    "loss_type": "corn",
    # Training
#    "lr": 3e-4,
#    "batch_size": 16,
#    "epochs": 10,
#    "seed": 42,
#    "weight_decay": 0.01,
#    "warmup_ratio": 0.15,
    # Augmentation
#    "pitch_augment_range": 2,
    # Performance
#    "dataloader_num_workers": 0,
#    "gradient_accumulation_steps": 1,
#}


DEFAULT_HPARAMS = {
    # Model architecture
    "d_model": 320,
    "nhead": 8,
    "num_layers": 4,
    "dim_feedforward": 1024,
    "dropout": 0.35,
    "max_seq_len": 512, # 256 for real run
    "loss_type": "ce",
    # Training
    "lr": 2e-4,
    "batch_size": 64,
    "epochs": 3, # find limit for read run
    "seed": 42,
    "weight_decay": 0.05,
    "warmup_ratio": 0.25,
    # Augmentation
    "pitch_augment_range": 2,
    # Per-class augment probability: [Initial, Gr1, Gr2, ..., Gr8]
    # Use a single float for uniform prob across all classes.
    "augment_prob": [1, 0.4, 0.4, 0.2, 0.2, 0.2, 0, 0, 0],
    # Performance
    "dataloader_num_workers": 4,
    "gradient_accumulation_steps": 1,
}

# loss_type = ce
# max_seq_len 512:  Validation accuracy across epochs: 26.33% → 26.92% → 29.54%
# max_seq_len 256:  Validation accuracy across epochs: 28.10% → 29.11% → 27.59% (Overall Accuracy: 27.57/Macro F1 Score:16.84)
# max_seq_len 128:  Validation accuracy across epochs: 26.24% → 21.69% → 25.15% (Overall Accuracy: 24.13/Macro F1 Score:13.66)
# loss_type = corn
 # "max_seq_len":  Validation accuracy across epochs: 128 11.14% → 14.01% → 14.51% (Overall Accuracy: 12.73/Macro F1 Score:11.49)
# loss_type =  ce
# augmentation + max_seq_len 128:  Validation accuracy across epochs: 13.08% → 21.94% → 17.47% (Overall Accuracy: 20.07, Macro F1 Score:14,02)
# augmentation + max_seq_len 256: Validation accuracy across epochs: 13.76% → 21.18% → 20.42%  (Overall Accuracy: 21.42  Macro F1 Score:  17.39=
# loss_type =  ce , "augment_prob": 0.4,
# 16.71% → 15.95% → 19.24%   Test Accuracy: 0.1897   Test Macro F1: 0.1379


#            d_model nhead num_layers dim_feedforward
# A (bigger)*  320     8       4        1024        29.37% → 24.98% → 22.78%   Overall Accuracy: 0.3061  Macro F1 Score:  0.1051
# B (deeper)   256     8       5        1024(!!!)   20.93% → 19.49% → 23.71%  Overall Accuracy: 0.2260   Macro F1 Score:  0.1479
# C (wider)    384     8       3        1024        29.54% → 22.11% → 23.63%  Overall Accuracy: 0.2892 Macro F1 Score:  0.1187

# should eventually reach at least 40–50% accuracy and Macro F1 > 0.35 with a decent setup. 15% is still too low.





# d_head=8 , loss_type = ce, max_seq_len 128 ,"augment_prob": 0.25,


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
# Piano Syllabus levels: 0 (Initial), 1–8 (Grade 1–8)
# Grades 9 and 10 are merged into Grade 8 (unreliable distinction)
# ---------------------------------------------------------------------------

LABEL_NAMES = {
    0: "Initial",
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
    Grades 9 and 10 are clamped to 8 (unreliable distinction at that level).
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

    # Merge grades 9 and 10 into grade 8
    if label > 8:
        label = 8

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

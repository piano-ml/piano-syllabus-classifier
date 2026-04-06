"""
features.py — Handcrafted musicological feature extraction from MIDI files.

Extracts difficulty-correlated features using symusic for fast parsing.
Features are designed to capture aspects of piano difficulty that the
Transformer may not easily learn from raw token sequences.

Provides:
    - extract_features(midi_path) → numpy array of raw features
    - FeatureNormalizer: fit on training set, transform any split
    - NUM_FEATURES: int constant for model dimensioning
    - FEATURE_NAMES: list of human-readable feature names
"""

from pathlib import Path

import numpy as np
from symusic import Score


# ---------------------------------------------------------------------------
# Feature names (order must match extract_features output)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "note_density",           # notes per second
    "avg_polyphony",          # average simultaneous notes
    "max_polyphony",          # peak simultaneous notes
    "polyphony_entropy",      # entropy of polyphony distribution
    "pitch_range",            # highest - lowest pitch used
    "num_distinct_pitches",   # number of unique pitches
    "avg_velocity",           # mean velocity
    "std_velocity",           # velocity variability
    "dynamic_range",          # max velocity - min velocity
    "chord_ratio",            # fraction of notes in chords
    "fast_note_ratio",        # fraction of notes shorter than 8th note
    "rhythmic_complexity",    # normalized std of inter-onset intervals
    "wide_leap_ratio",        # fraction of pitch intervals > octave
    "repeated_note_ratio",    # fraction of repeated pitches
    "arpeggio_ratio",         # fast broken chord density
    "low_ratio",              # ratio of notes in low register (< C3)
    "high_ratio",             # ratio of notes in high register (>= C6)
    "hand_independence",      # normalized pitch distance between hands
]

NUM_FEATURES = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

# MIDI pitch constants
_SPLIT_PITCH = 60      # C4 - split for LH/RH
_LOW_UPPER = 48        # C3
_HIGH_LOWER = 72       # C6


def extract_features(midi_path: str | Path) -> np.ndarray:
    """Extract improved handcrafted features from a MIDI file.

    Returns a 1D numpy array of shape (NUM_FEATURES,) with raw values.
    Returns zeros array on failure.
    """
    try:
        score = Score(str(midi_path))
    except Exception:
        return np.zeros(NUM_FEATURES, dtype=np.float32)

    all_notes = []
    for track in score.tracks:
        if track.is_drum:
            continue
        for note in track.notes:
            all_notes.append((note.time, note.duration, note.pitch, note.velocity))

    if not all_notes:
        return np.zeros(NUM_FEATURES, dtype=np.float32)

    times = np.array([n[0] for n in all_notes], dtype=np.float64)
    durations = np.array([n[1] for n in all_notes], dtype=np.float64)
    pitches = np.array([n[2] for n in all_notes], dtype=np.float64)
    velocities = np.array([n[3] for n in all_notes], dtype=np.float64)

    n_notes = len(all_notes)

    # ====================== Tempo & Duration ======================
    tempos = score.tempos
    tpq = score.ticks_per_quarter or 480
    first_bpm = tempos[0].qpm if tempos else 120.0
    secs_per_tick = 60.0 / (first_bpm * tpq)
    total_duration = float(np.max(times + durations)) * secs_per_tick
    total_duration = max(total_duration, 0.1)

    # ====================== Core Features ======================
    note_density = n_notes / total_duration

    # --- Improved Polyphony (event-based) ---
    events = []
    for t, d, _, _ in all_notes:
        events.append((t, 1))      # note on
        events.append((t + d, -1)) # note off
    events.sort()

    active = 0
    polyphony_values = []
    prev_time = 0
    for time, delta in events:
        if time > prev_time and active > 0:
            polyphony_values.append(active)
        active += delta
        prev_time = time

    polyphony_values = np.array(polyphony_values, dtype=np.float64)
    if len(polyphony_values) > 0:
        avg_polyphony = float(np.mean(polyphony_values))
        max_polyphony = float(np.max(polyphony_values))
        # Polyphony entropy
        counts = np.bincount(polyphony_values.astype(int))
        probs = counts[counts > 0] / counts[counts > 0].sum()
        polyphony_entropy = float(-np.sum(probs * np.log2(probs))) if len(probs) > 1 else 0.0
    else:
        avg_polyphony = max_polyphony = polyphony_entropy = 0.0

    # --- Pitch & Register ---
    pitch_range = float(np.max(pitches) - np.min(pitches)) if n_notes > 0 else 0.0
    num_distinct_pitches = float(len(np.unique(pitches)))
    low_ratio = float(np.sum(pitches < _LOW_UPPER)) / n_notes
    high_ratio = float(np.sum(pitches >= _HIGH_LOWER)) / n_notes

    # --- Velocity ---
    avg_velocity = float(np.mean(velocities))
    std_velocity = float(np.std(velocities))
    dynamic_range = float(np.max(velocities) - np.min(velocities)) if n_notes > 1 else 0.0

    # --- Chord ratio (improved) ---
    quant_onsets = (times / max(tpq // 8, 1)).astype(np.int64)  # finer grid
    _, onset_counts = np.unique(quant_onsets, return_counts=True)
    chord_notes = int(np.sum(onset_counts[onset_counts > 1]))
    chord_ratio = chord_notes / n_notes if n_notes > 0 else 0.0

    # --- Fast note ratio (scalar passages) ---
    fast_note_ratio = float(np.sum(durations < (tpq / 2))) / n_notes   # shorter than 8th note

    # --- Rhythmic complexity (normalized) ---
    if len(times) > 1:
        sorted_onsets = np.sort(times)
        iois = np.diff(sorted_onsets)
        iois = iois[iois > 0]
        if len(iois) > 1:
            # Normalize by tempo
            mean_ioi = np.mean(iois)
            rhythmic_complexity = float(np.std(iois) / (mean_ioi + 1e-8))
        else:
            rhythmic_complexity = 0.0
    else:
        rhythmic_complexity = 0.0

    # --- Wide leaps ---
    if len(pitches) > 1:
        pitch_diffs = np.abs(np.diff(np.sort(pitches)))
        wide_leap_ratio = float(np.sum(pitch_diffs > 12)) / (len(pitch_diffs) + 1)
    else:
        wide_leap_ratio = 0.0

    # --- Repeated notes ---
    repeated_note_ratio = float(np.sum(np.diff(np.sort(pitches)) == 0)) / (n_notes - 1) if n_notes > 1 else 0.0

    # --- Arpeggio ratio (fast broken chords) ---
    # Simple heuristic: fast notes with large pitch variation in short time
    arpeggio_ratio = 0.0
    if n_notes > 4:
        window = int(tpq * 0.5)  # half beat window
        for i in range(n_notes - 4):
            window_notes = pitches[(times >= times[i]) & (times <= times[i] + window)]
            if len(window_notes) >= 4:
                if np.std(window_notes) > 8 and np.max(np.diff(np.sort(window_notes))) > 6:
                    arpeggio_ratio += 1
        arpeggio_ratio /= (n_notes - 3)

    # --- Hand independence (new) ---
    lh_pitches = pitches[pitches < _SPLIT_PITCH]
    rh_pitches = pitches[pitches >= _SPLIT_PITCH]
    lh_rh_ratio = len(lh_pitches) / max(len(rh_pitches), 1)
    # Simple independence: how different their average pitch is
    hand_independence = 0.0
    if len(lh_pitches) > 0 and len(rh_pitches) > 0:
        hand_independence = abs(np.mean(lh_pitches) - np.mean(rh_pitches)) / 24.0  # normalized

    # ====================== Assemble Features ======================
    features = np.array([
        note_density,
        avg_polyphony,
        max_polyphony,
        polyphony_entropy,
        pitch_range,
        num_distinct_pitches,
        avg_velocity,
        std_velocity,
        dynamic_range,
        chord_ratio,
        fast_note_ratio,
        rhythmic_complexity,
        wide_leap_ratio,
        repeated_note_ratio,
        arpeggio_ratio,
        low_ratio,
        high_ratio,
        hand_independence,
    ], dtype=np.float32)

    # Clean NaN / Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


# ---------------------------------------------------------------------------
# Normaliser (z-score, fit on training set)
# ---------------------------------------------------------------------------


class FeatureNormalizer:
    """Z-score normaliser for handcrafted features.

    Fit on training data, then transform any split identically.
    Serialisable to/from a .npz file for inference.
    """

    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "FeatureNormalizer":
        """Compute mean/std from a (N, NUM_FEATURES) array."""
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0)
        # Avoid division by zero for constant features
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Normalise features using stored mean/std."""
        return (features - self.mean) / self.std

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)

    def save(self, path: str | Path) -> None:
        np.savez(str(path), mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureNormalizer":
        data = np.load(str(path))
        norm = cls()
        norm.mean = data["mean"]
        norm.std = data["std"]
        return norm


# ---------------------------------------------------------------------------
# Batch extraction helper
# ---------------------------------------------------------------------------


def extract_features_batch(
    midi_paths: list[Path],
    normalizer: FeatureNormalizer | None = None,
) -> np.ndarray:
    """Extract features for a list of MIDI files.

    Returns (N, NUM_FEATURES) array.  If *normalizer* is provided,
    features are normalised.
    """
    feats = np.stack([extract_features(p) for p in midi_paths])
    if normalizer is not None:
        feats = normalizer.transform(feats)
    return feats

"""
augmentation.py — On-the-fly pitch transposition for the training set.

Provides:
    - transpose_score(): apply a pitch offset to a symusic Score
    - AugmentedDatasetMIDI: wraps miditok DatasetMIDI with random ±N transposition
"""

import random
from pathlib import Path

from symusic import Score
from torch import LongTensor
from miditok.data_augmentation import augment_score
from miditok.pytorch_data import DatasetMIDI


# Piano range: MIDI pitches 21 (A0) to 108 (C8)
PIANO_PITCH_MIN = 21
PIANO_PITCH_MAX = 108


def transpose_score(score: Score, pitch_offset: int) -> Score:
    """Transpose a Score by *pitch_offset* semitones, clamping to piano range.

    Uses miditok's augment_score which handles drum track exclusion
    and returns a copy.
    """
    if pitch_offset == 0:
        return score
    return augment_score(score, pitch_offset=pitch_offset)


def _score_fits_piano_range(score: Score, pitch_offset: int) -> bool:
    """Check that all notes stay within [21, 108] after transposition."""
    for track in score.tracks:
        if track.is_drum:
            continue
        for note in track.notes:
            new_pitch = note.pitch + pitch_offset
            if new_pitch < PIANO_PITCH_MIN or new_pitch > PIANO_PITCH_MAX:
                return False
    return True


class AugmentedDatasetMIDI(DatasetMIDI):
    """DatasetMIDI with probabilistic pitch-transposition augmentation.

    Each sample has a per-class probability of being randomly transposed
    by an offset in [-pitch_augment_range, +pitch_augment_range] (excluding 0).
    *augment_prob* can be a single float (uniform for all classes) or a list
    of floats indexed by class id.

    *label_map* maps filename stems to integer class ids so the correct
    probability can be looked up per sample.

    The dataset length equals the number of files (no expansion).

    If a transposition would push notes outside the piano range [21, 108],
    the original (offset=0) is used as fallback.

    This subclass only works when pre_tokenize=False (on-the-fly mode).
    """

    def __init__(
        self,
        pitch_augment_range: int = 2,
        augment_prob: float | list[float] = 0.4,
        label_map: dict[str, int] | None = None,
        **kwargs,
    ):
        if kwargs.get("pre_tokenize", False):
            raise ValueError(
                "AugmentedDatasetMIDI requires pre_tokenize=False "
                "(on-the-fly tokenization)."
            )
        super().__init__(**kwargs)
        self.pitch_augment_range = pitch_augment_range
        self.augment_prob = augment_prob
        self.label_map = label_map or {}
        # All non-zero offsets: [-N, ..., -1, 1, ..., N]
        self.pitch_offsets = [
            o for o in range(-pitch_augment_range, pitch_augment_range + 1) if o != 0
        ]

    def __len__(self) -> int:
        return len(self.files_paths)

    def __getitem__(self, idx: int) -> dict[str, LongTensor]:
        """Load a MIDI file, maybe apply random transposition, tokenize, and return."""
        labels = None

        try:
            score = Score(self.files_paths[idx])
        except Exception:
            item = {self.sample_key_name: None}
            if self.func_to_get_labels is not None:
                item[self.labels_key_name] = None
            return item

        # --- Probabilistic transposition (per-class probability) ---
        prob = self.augment_prob
        if isinstance(prob, list):
            cls_id = self.label_map.get(self.files_paths[idx].stem)
            prob = prob[cls_id] if cls_id is not None and cls_id < len(prob) else 0.0
        if random.random() < prob and self.pitch_offsets:
            offset = random.choice(self.pitch_offsets)
            if _score_fits_piano_range(score, offset):
                score = transpose_score(score, offset)

        # --- Tokenize the (possibly transposed) score ---
        tseq = self._tokenize_score(score)
        token_ids = tseq.ids if self.tokenizer.one_token_stream else tseq[0].ids

        if self.func_to_get_labels is not None:
            labels = self.func_to_get_labels(score, tseq, self.files_paths[idx])
            if not isinstance(labels, LongTensor):
                labels = LongTensor([labels] if isinstance(labels, int) else labels)

        item = {self.sample_key_name: LongTensor(token_ids)}
        if self.func_to_get_labels is not None:
            item[self.labels_key_name] = labels

        return item

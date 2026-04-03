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
    """DatasetMIDI with on-the-fly random pitch transposition.

    For each __getitem__ call, a random offset is drawn from
    {-pitch_augment_range, ..., 0, ..., +pitch_augment_range}
    with equal probability. If the chosen transposition would push notes
    outside the piano range [21, 108], the original (offset=0) is used.

    This subclass only works when pre_tokenize=False (on-the-fly mode).
    """

    def __init__(self, pitch_augment_range: int = 2, **kwargs):
        if kwargs.get("pre_tokenize", False):
            raise ValueError(
                "AugmentedDatasetMIDI requires pre_tokenize=False "
                "(on-the-fly tokenization)."
            )
        super().__init__(**kwargs)
        self.pitch_augment_range = pitch_augment_range
        # Build the set of possible offsets: [-N, ..., -1, 0, 1, ..., N]
        self.pitch_offsets = list(range(-pitch_augment_range, pitch_augment_range + 1))

    def __getitem__(self, idx: int) -> dict[str, LongTensor]:
        """Load a MIDI file, apply random transposition, tokenize, and return."""
        labels = None

        try:
            score = Score(self.files_paths[idx])
        except Exception:
            item = {self.sample_key_name: None}
            if self.func_to_get_labels is not None:
                item[self.labels_key_name] = None
            return item

        # --- Random transposition ---
        offset = random.choice(self.pitch_offsets)
        if offset != 0 and _score_fits_piano_range(score, offset):
            score = transpose_score(score, offset)
        # If offset would go out of range, fall back to original (offset=0)

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

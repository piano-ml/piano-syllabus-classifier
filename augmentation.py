"""
augmentation.py — Feature-injecting dataset wrapper for MIDI classification.

Provides:
    - FeatureDatasetMIDI: wraps miditok DatasetMIDI to attach pre-computed feature vectors
"""

from pathlib import Path

import torch
from miditok.pytorch_data import DatasetMIDI


class FeatureDatasetMIDI(DatasetMIDI):
    """DatasetMIDI that attaches pre-computed feature vectors to each sample."""

    def __init__(self, feature_vectors: dict[str, torch.Tensor] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.feature_vectors = feature_vectors

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if self.feature_vectors is not None:
            key = str(self.files_paths[idx])
            item["features"] = self.feature_vectors.get(key, torch.zeros(1))
        return item

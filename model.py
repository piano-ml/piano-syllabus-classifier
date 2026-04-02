"""
model.py — Transformer-based MIDI sequence classifier.

Architecture:
    Token Embedding + Positional Encoding
    → N × TransformerEncoderLayer
    → Mean pooling (masked)
    → Dropout → Linear → 𝑘 classes
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MidiClassifier(nn.Module):
    """Small Transformer encoder for MIDI token sequence classification.

    Compatible with Hugging Face Trainer: forward() accepts ``input_ids``,
    ``attention_mask``, and ``labels`` and returns a dict with ``loss``
    and ``logits``.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        pad_token_id: int = 0,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.d_model = d_model

        self.embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.classifier_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # Store class weights for weighted cross-entropy
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """
        Args:
            input_ids:      (batch, seq_len) — token ids
            attention_mask:  (batch, seq_len) — 1 for real tokens, 0 for padding
            labels:          (batch,) — integer class labels

        Returns:
            dict with 'loss' (scalar or None) and 'logits' (batch, num_classes).
        """
        # Build padding mask for nn.TransformerEncoder
        # TransformerEncoder expects src_key_padding_mask: True = ignore
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # True where padded
        else:
            key_padding_mask = input_ids == self.pad_token_id

        # Embed + positional encoding
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)

        # Mean pooling over non-padded positions
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, S, 1)
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            non_pad = (~key_padding_mask).unsqueeze(-1).float()
            x = (x * non_pad).sum(dim=1) / non_pad.sum(dim=1).clamp(min=1)

        logits = self.classifier_head(x)  # (batch, num_classes)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)

        return {"loss": loss, "logits": logits}

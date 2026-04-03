"""
model.py — Transformer-based MIDI sequence classifier.

Architecture:
    Token Embedding + Positional Encoding
    → N × TransformerEncoderLayer
    → Mean pooling (masked)
    → Dropout → Classification head

Supports two loss modes:
    - "ce"   : standard cross-entropy (K logits)
    - "corn" : CORN conditional ordinal regression (K-1 cumulative logits)
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


# ---------------------------------------------------------------------------
# CORN ordinal head
# ---------------------------------------------------------------------------


class CornOrdinalHead(nn.Module):
    """CORN (Conditional Ordinal Regression Network) classification head.

    Uses K-1 independent binary classifiers (each with its own weight vector
    and bias) to predict P(Y > k | Y >= k) for k = 0 .. K-2.

    Reference: Shi et al. (2021) — "CORN — Conditional Ordinal Regression for
    Neural Networks". Pattern Recognition, 122, 108263.
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        # K-1 independent classifiers, each d_model → 1
        self.fc = nn.Linear(in_features, num_classes - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return conditional logits (batch, num_classes - 1)."""
        return self.fc(x)  # (B, K-1)


def corn_loss(
    conditional_logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """CORN conditional ordinal loss.

    For each rank threshold k (0 .. K-2), only samples with y >= k
    contribute to the loss of classifier k.  The target for classifier k
    is 1 if y > k, else 0 (i.e. y == k for the eligible negatives).

    Args:
        conditional_logits: (B, K-1) raw logits for P(Y > k | Y >= k).
        labels:             (B,) integer class labels in [0, K-1].
        num_classes:        K.
        class_weights:      (K,) optional per-class weights.
    """
    sets = []
    for k in range(num_classes - 1):
        # Mask: only samples with y >= k
        mask = labels >= k               # (B,)
        if not mask.any():
            continue
        logits_k = conditional_logits[mask, k]            # (n_k,)
        targets_k = (labels[mask] > k).float()            # (n_k,)

        task_loss = F.binary_cross_entropy_with_logits(
            logits_k, targets_k, reduction="none",
        )  # (n_k,)

        if class_weights is not None:
            sample_w = class_weights[labels[mask]]
            task_loss = task_loss * sample_w

        sets.append(task_loss.mean())

    if not sets:
        return torch.tensor(0.0, device=conditional_logits.device, requires_grad=True)

    return torch.stack(sets).mean()


def corn_logits_to_class(conditional_logits: torch.Tensor) -> torch.Tensor:
    """Convert CORN conditional logits → predicted class labels (B,).

    Predicted rank = product of conditional probabilities converted to
    cumulative probabilities, then count how many exceed 0.5.
    """
    cum_probs = _corn_conditional_to_cumulative(conditional_logits)
    return (cum_probs > 0.5).sum(dim=1).long()


def corn_logits_to_probs(conditional_logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert CORN conditional logits → per-class probabilities (B, K).

    P(Y = k) = P(Y > k-1) - P(Y > k), with boundary conditions
    P(Y > -1) = 1  and  P(Y > K-1) = 0.
    """
    cum_probs = _corn_conditional_to_cumulative(conditional_logits)  # (B, K-1)
    ones = torch.ones(cum_probs.size(0), 1, device=cum_probs.device)
    zeros = torch.zeros(cum_probs.size(0), 1, device=cum_probs.device)
    extended = torch.cat([ones, cum_probs, zeros], dim=1)   # (B, K+1)
    class_probs = extended[:, :-1] - extended[:, 1:]        # (B, K)
    return class_probs.clamp(min=0)


def _corn_conditional_to_cumulative(conditional_logits: torch.Tensor) -> torch.Tensor:
    """Convert CORN conditional logits to cumulative probabilities P(Y > k).

    P(Y > k) = ∏_{j=0}^{k} P(Y > j | Y >= j)
    """
    cond_probs = torch.sigmoid(conditional_logits)  # (B, K-1)
    # Cumulative product: P(Y > k) = prod of cond probs up to k
    cum_probs = torch.cumprod(cond_probs, dim=1)    # (B, K-1)
    return cum_probs


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------


class MidiClassifier(nn.Module):
    """Transformer encoder for MIDI token sequence classification.

    Compatible with Hugging Face Trainer: forward() accepts ``input_ids``,
    ``attention_mask``, and ``labels`` and returns a dict with ``loss``
    and ``logits``.

    When ``loss_type="corn"``, the model uses CORN ordinal regression.
    The ``logits`` tensor has shape (B, K-1) — conditional logits.
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
        loss_type: str = "ce",
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.d_model = d_model
        self.num_classes = num_classes
        self.loss_type = loss_type

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

        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        if loss_type == "corn":
            self.corn_head = CornOrdinalHead(d_model, num_classes)
            self.classifier_head = None
        else:
            self.classifier_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )
            self.corn_head = None

        # Store class weights
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Shared encoder: returns pooled representation (B, d_model)."""
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = input_ids == self.pad_token_id

        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)

        # Mean pooling over non-padded positions
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            non_pad = (~key_padding_mask).unsqueeze(-1).float()
            x = (x * non_pad).sum(dim=1) / non_pad.sum(dim=1).clamp(min=1)

        x = self.norm(x)
        x = self.drop(x)
        return x

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
            dict with 'loss' (scalar or None) and 'logits'.
            - CE mode:   logits shape (B, num_classes)
            - CORN mode: logits shape (B, num_classes - 1) — conditional logits
        """
        pooled = self._encode(input_ids, attention_mask)

        if self.loss_type == "corn":
            conditional_logits = self.corn_head(pooled)   # (B, K-1)
            loss = None
            if labels is not None:
                loss = corn_loss(
                    conditional_logits, labels,
                    self.num_classes, self.class_weights,
                )
            return {"loss": loss, "logits": conditional_logits}
        else:
            logits = self.classifier_head(pooled)
            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits, labels, weight=self.class_weights)
            return {"loss": loss, "logits": logits}

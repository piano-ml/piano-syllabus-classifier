"""
model.py — Feature-based MLP with CORN ordinal head + LightGBM ensemble.

Architecture:
    Handcrafted features → configurable MLP with BatchNorm → CORN ordinal head
    Handcrafted features → LightGBM regressor → single scalar output
    Ensemble: weighted average of both predictions.

CORN (Conditional Ordinal Regression Network):
    The output layer has K-1 independent binary classifiers each modeling
    P(y > k | y ≥ k).  The unconditional P(y > k) is recovered as the
    product of conditional probabilities.  This avoids rank inconsistencies
    and trains each classifier only on the relevant subset of samples.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CORN ordinal regression components
# ---------------------------------------------------------------------------


class CornLayer(nn.Module):
    """CORN output layer: K-1 independent binary classifiers.

    Each output k models logit of P(y > k | y ≥ k).
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_features) → (B, num_classes-1) logits."""
        return self.fc(x)


def corn_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    task_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """CORN loss: conditional BCE — classifier k only trains on samples with y ≥ k.

    For threshold k, target is 1 if y > k (given y ≥ k).
    If *task_weights* is provided (length K-1), each task's BCE is scaled.
    """
    loss = torch.zeros((), device=logits.device)
    count = 0
    for k in range(num_classes - 1):
        mask = labels >= k
        if mask.sum() == 0:
            continue
        logits_k = logits[mask, k]
        targets_k = (labels[mask] > k).float()
        bce = F.binary_cross_entropy_with_logits(logits_k, targets_k)
        if task_weights is not None:
            bce = bce * task_weights[k]
        loss = loss + bce
        count += 1
    return loss / max(count, 1)


def corn_predict(logits: torch.Tensor) -> torch.Tensor:
    """Convert CORN logits to a continuous predicted rank.

    P(y > k) = prod_{j=0}^{k} sigmoid(logit_j)
    Predicted rank = sum_{k=0}^{K-2} P(y > k), giving a value in [0, K-1].
    """
    probs = torch.sigmoid(logits)  # (B, K-1)
    # Cumulative product: P(y > k) = prod of conditional probs up to k
    cum_probs = torch.cumprod(probs, dim=1)  # (B, K-1)
    return cum_probs.sum(dim=1)


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class FeatureMLPRegressor(nn.Module):
    """Configurable MLP with CORN ordinal head for grade prediction.

    When *num_classes* is provided (> 1), uses the CORN ordinal output
    layer with K-1 conditional binary classifiers.  Otherwise falls
    back to a single-scalar regression head with L1 loss.

    Compatible with Hugging Face Trainer: forward() accepts ``features``
    and ``labels`` and returns a dict with ``loss`` and ``logits``.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        num_classes: int | None = None,
        num_hidden_layers: int = 2,
        use_batch_norm: bool = True,
        activation: str = "relu",
        corn_task_weights: list[float] | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        if corn_task_weights is not None:
            self.register_buffer(
                "task_weights", torch.tensor(corn_task_weights, dtype=torch.float32)
            )
        else:
            self.task_weights = None

        act_cls = _ACTIVATIONS.get(activation, nn.ReLU)

        layers: list[nn.Module] = []
        in_dim = num_features
        for i in range(num_hidden_layers):
            out_dim = hidden_dim if i < num_hidden_layers - 1 else hidden_dim // 2
            layers.append(nn.Linear(in_dim, out_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(act_cls())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)

        head_in = hidden_dim // 2 if num_hidden_layers > 1 else hidden_dim
        if num_classes is not None and num_classes > 1:
            self.head = CornLayer(head_in, num_classes)
        else:
            self.head = nn.Linear(head_in, 1)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | None]:
        h = self.mlp(features)

        if self.num_classes is not None and self.num_classes > 1:
            # CORN ordinal regression
            logits = self.head(h)                # (B, K-1)
            pred = corn_predict(logits)           # (B,) continuous rank
            loss = None
            if labels is not None:
                loss = corn_loss(logits, labels, self.num_classes, self.task_weights)
        else:
            # Standard scalar regression
            pred = self.head(h).squeeze(-1)       # (B,)
            loss = None
            if labels is not None:
                loss = F.l1_loss(pred, labels.float())

        # HF Trainer expects "logits"; we store the continuous prediction
        return {"loss": loss, "logits": pred}


class EnsembleRegressor:
    """Ensemble of MLP + LightGBM regressors with weighted averaging."""

    def __init__(
        self,
        mlp: FeatureMLPRegressor,
        lgbm,
        mlp_weight: float = 0.5,
    ):
        self.mlp = mlp
        self.lgbm = lgbm
        self.mlp_weight = mlp_weight

    def predict(self, features_np: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Predict grade values for a batch of normalised feature vectors.

        Args:
            features_np: (N, num_features) numpy array, already normalised.
            device: torch device for the MLP.

        Returns:
            (N,) numpy array of continuous grade predictions.
        """
        # MLP predictions
        self.mlp.eval()
        self.mlp.to(device)
        with torch.no_grad():
            feat_t = torch.tensor(features_np, dtype=torch.float32, device=device)
            mlp_pred = self.mlp(features=feat_t)["logits"].cpu().numpy()

        # LightGBM predictions
        lgbm_pred = self.lgbm.predict(features_np)

        # Weighted average
        return self.mlp_weight * mlp_pred + (1 - self.mlp_weight) * lgbm_pred

    def save(self, output_dir: str | Path) -> None:
        """Save the LightGBM model to *output_dir* (MLP saved by HF Trainer)."""
        import lightgbm as lgb

        path = Path(output_dir) / "lgbm_model.txt"
        self.lgbm.save_model(str(path))
        print(f"  LightGBM model saved → {path}")

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        mlp: FeatureMLPRegressor,
        mlp_weight: float = 0.5,
    ) -> "EnsembleRegressor":
        """Load LightGBM from disk and pair with an already-loaded MLP."""
        import lightgbm as lgb

        path = Path(model_dir) / "lgbm_model.txt"
        if not path.exists():
            raise FileNotFoundError(f"No LightGBM model found at {path}")
        lgbm_model = lgb.Booster(model_file=str(path))
        return cls(mlp, lgbm_model, mlp_weight=mlp_weight)

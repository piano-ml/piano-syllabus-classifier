"""
model.py — Feature-based MLP regressor + LightGBM ensemble for piano grade prediction.

Architecture:
    Handcrafted features → 3-layer MLP with BatchNorm → single scalar output
    Handcrafted features → LightGBM regressor → single scalar output
    Ensemble: weighted average of both predictions.
    Trained with MAE (L1) loss for direct grade regression.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMLPRegressor(nn.Module):
    """3-layer MLP regressor using handcrafted features.

    Predicts a continuous grade value (0–8).  Trained with L1 / MAE loss.
    Compatible with Hugging Face Trainer: forward() accepts ``features``
    and ``labels`` and returns a dict with ``loss`` and ``logits``.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            # Layer 1
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            # Layer 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            # Layer 3
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            # Output: single scalar
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | None]:
        pred = self.mlp(features).squeeze(-1)  # (B,)
        loss = None
        if labels is not None:
            loss = F.l1_loss(pred, labels.float())
        # HF Trainer expects "logits"; we store the raw prediction there
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

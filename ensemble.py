"""
ensemble.py — Stacking ensemble & weighted blending for grade regression.

Provides:
    - StackingEnsemble: 5-fold stacking with LightGBM + MLP base models
      and a LightGBM meta-learner
    - WeightedBlender: per-range blending weights
    - feature_importance_analysis: extract & plot LightGBM feature importances

Usage:
    Activated via --stacking flag in train_ps_classifier.py.

Expected benefit:
    Stacking captures complementary strengths of LightGBM (tree-based,
    handles feature interactions natively) and MLP (smooth nonlinear
    mappings).  A meta-learner on top of both can learn when to trust
    each base model, especially for ambiguous middle grades (3–6).
"""

from pathlib import Path
import json

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import NUM_FEATURES, FEATURE_NAMES


# ---------------------------------------------------------------------------
# Stacking ensemble
# ---------------------------------------------------------------------------


class StackingEnsemble:
    """5-fold stacking with LightGBM + MLP base models and a meta-learner.

    Base models produce out-of-fold predictions on training data.
    A LightGBM meta-learner is trained on the stacked predictions.
    """

    def __init__(self):
        self.lgbm_folds = []       # list of (fold LightGBM booster)
        self.mlp_folds = []        # list of (fold MLP state_dict)
        self.meta_learner = None   # LightGBM meta-learner
        self.mlp_config = {}       # num_features, hidden_dim, dropout
        self.n_folds = 5

    def fit(
        self,
        train_feats: np.ndarray,
        train_labels: np.ndarray,
        num_features: int = NUM_FEATURES,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        mlp_epochs: int = 8,
        mlp_lr: float = 5e-4,
        mlp_batch_size: int = 64,
        seed: int = 42,
    ) -> "StackingEnsemble":
        """Train base models with 5-fold CV and fit meta-learner."""
        import lightgbm as lgb
        from model import FeatureMLPRegressor

        self.mlp_config = {
            "num_features": num_features,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
        }

        labels = np.array(train_labels, dtype=np.float32)
        labels_int = labels.astype(int)

        # Stratified folds
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=seed)

        oof_lgbm = np.zeros(len(labels))
        oof_mlp = np.zeros(len(labels))

        device = "cuda" if torch.cuda.is_available() else "cpu"

        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(train_feats, labels_int)):
            print(f"\n  --- Stacking Fold {fold_idx + 1}/{self.n_folds} ---")
            tr_feats, val_feats = train_feats[tr_idx], train_feats[val_idx]
            tr_labels, val_labels = labels[tr_idx], labels[val_idx]

            # --- LightGBM ---
            lgbm_train = lgb.Dataset(tr_feats, label=tr_labels)
            lgbm_val = lgb.Dataset(val_feats, label=val_labels, reference=lgbm_train)
            lgbm_params = {
                "objective": "mae",
                "metric": "mae",
                "verbosity": -1,
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "seed": seed + fold_idx,
            }
            booster = lgb.train(
                lgbm_params, lgbm_train,
                num_boost_round=500,
                valid_sets=[lgbm_val],
                valid_names=["val"],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )
            self.lgbm_folds.append(booster)
            oof_lgbm[val_idx] = booster.predict(val_feats)

            # --- MLP ---
            mlp = FeatureMLPRegressor(num_features, hidden_dim, dropout).to(device)
            optimizer = torch.optim.AdamW(mlp.parameters(), lr=mlp_lr, weight_decay=0.01)

            tr_t = torch.tensor(tr_feats, dtype=torch.float32, device=device)
            tr_lab_t = torch.tensor(tr_labels, dtype=torch.float32, device=device)
            val_t = torch.tensor(val_feats, dtype=torch.float32, device=device)

            dataset = torch.utils.data.TensorDataset(tr_t, tr_lab_t)
            loader = torch.utils.data.DataLoader(dataset, batch_size=mlp_batch_size, shuffle=True)

            mlp.train()
            for epoch in range(mlp_epochs):
                for batch_feats, batch_labels in loader:
                    optimizer.zero_grad()
                    out = mlp(features=batch_feats, labels=batch_labels)
                    out["loss"].backward()
                    optimizer.step()

            mlp.eval()
            with torch.no_grad():
                oof_mlp[val_idx] = mlp(features=val_t)["logits"].cpu().numpy()

            self.mlp_folds.append(mlp.state_dict())

        # --- Meta-learner ---
        print(f"\n  --- Training meta-learner ---")
        meta_feats = np.column_stack([oof_lgbm, oof_mlp])
        meta_train = lgb.Dataset(meta_feats, label=labels)
        meta_params = {
            "objective": "mae",
            "metric": "mae",
            "verbosity": -1,
            "num_leaves": 8,
            "learning_rate": 0.05,
            "seed": seed,
        }
        self.meta_learner = lgb.train(
            meta_params, meta_train,
            num_boost_round=200,
            callbacks=[lgb.log_evaluation(0)],
        )

        lgbm_oof_mae = float(np.mean(np.abs(oof_lgbm - labels)))
        mlp_oof_mae = float(np.mean(np.abs(oof_mlp - labels)))
        meta_oof_pred = self.meta_learner.predict(meta_feats)
        meta_oof_mae = float(np.mean(np.abs(meta_oof_pred - labels)))
        print(f"  OOF MAE — LGBM: {lgbm_oof_mae:.3f}, MLP: {mlp_oof_mae:.3f}, Meta: {meta_oof_mae:.3f}")

        return self

    def predict(self, feats: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Predict using averaged base models + meta-learner."""
        from model import FeatureMLPRegressor

        # Average LightGBM fold predictions
        lgbm_preds = np.mean(
            [b.predict(feats) for b in self.lgbm_folds], axis=0
        )

        # Average MLP fold predictions
        mlp_preds_list = []
        feat_t = torch.tensor(feats, dtype=torch.float32, device=device)
        for sd in self.mlp_folds:
            mlp = FeatureMLPRegressor(**self.mlp_config).to(device)
            mlp.load_state_dict(sd)
            mlp.eval()
            with torch.no_grad():
                mlp_preds_list.append(mlp(features=feat_t)["logits"].cpu().numpy())
        mlp_preds = np.mean(mlp_preds_list, axis=0)

        # Meta-learner
        meta_feats = np.column_stack([lgbm_preds, mlp_preds])
        return self.meta_learner.predict(meta_feats)

    def save(self, output_dir: str | Path) -> None:
        import lightgbm as lgb

        output_dir = Path(output_dir)
        stacking_dir = output_dir / "stacking"
        stacking_dir.mkdir(parents=True, exist_ok=True)

        # Save LightGBM folds
        for i, b in enumerate(self.lgbm_folds):
            b.save_model(str(stacking_dir / f"lgbm_fold_{i}.txt"))

        # Save MLP fold state dicts
        for i, sd in enumerate(self.mlp_folds):
            torch.save(sd, stacking_dir / f"mlp_fold_{i}.pt")

        # Save meta-learner
        self.meta_learner.save_model(str(stacking_dir / "meta_learner.txt"))

        # Save config
        cfg = {"mlp_config": self.mlp_config, "n_folds": self.n_folds}
        with open(stacking_dir / "stacking_config.json", "w") as f:
            json.dump(cfg, f, indent=2)

        print(f"  Stacking ensemble saved → {stacking_dir}")

    @classmethod
    def load(cls, model_dir: str | Path) -> "StackingEnsemble":
        import lightgbm as lgb

        model_dir = Path(model_dir)
        stacking_dir = model_dir / "stacking"

        with open(stacking_dir / "stacking_config.json") as f:
            cfg = json.load(f)

        ens = cls()
        ens.n_folds = cfg["n_folds"]
        ens.mlp_config = cfg["mlp_config"]

        for i in range(ens.n_folds):
            ens.lgbm_folds.append(lgb.Booster(model_file=str(stacking_dir / f"lgbm_fold_{i}.txt")))
            ens.mlp_folds.append(torch.load(stacking_dir / f"mlp_fold_{i}.pt", map_location="cpu", weights_only=True))

        ens.meta_learner = lgb.Booster(model_file=str(stacking_dir / "meta_learner.txt"))
        return ens


# ---------------------------------------------------------------------------
# Feature importance analysis (Improvement D)
# ---------------------------------------------------------------------------


def feature_importance_analysis(
    lgbm_model,
    output_dir: str | Path,
    feature_names: list[str] | None = None,
    top_n: int = 15,
) -> dict:
    """Extract, print, and plot LightGBM feature importances.

    Args:
        lgbm_model: a trained LightGBM Booster (or first fold from stacking).
        output_dir: directory to save the plot and JSON.
        feature_names: list of feature names (defaults to FEATURE_NAMES).
        top_n: number of top features to plot.

    Returns:
        dict mapping feature name to importance (gain).
    """
    output_dir = Path(output_dir)
    if feature_names is None:
        feature_names = FEATURE_NAMES

    # Extract importances
    gain = lgbm_model.feature_importance(importance_type="gain")
    split = lgbm_model.feature_importance(importance_type="split")

    importance = {
        name: {"gain": float(g), "split": int(s)}
        for name, g, s in zip(feature_names, gain, split)
    }

    # Sort by gain
    sorted_feats = sorted(importance.items(), key=lambda x: x[1]["gain"], reverse=True)

    # Print top 10
    print("\n" + "=" * 60)
    print("  Feature Importance (LightGBM — gain)")
    print("=" * 60)
    print(f"  {'Rank':<5} {'Feature':<25} {'Gain':>10} {'Split':>8}")
    print(f"  {'-'*5} {'-'*25} {'-'*10} {'-'*8}")
    for rank, (name, vals) in enumerate(sorted_feats[:10], 1):
        print(f"  {rank:<5} {name:<25} {vals['gain']:>10.1f} {vals['split']:>8}")

    # Plot top N
    top = sorted_feats[:top_n]
    names_plot = [t[0] for t in top][::-1]
    gains_plot = [t[1]["gain"] for t in top][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names_plot)))
    ax.barh(names_plot, gains_plot, color=colors)
    ax.set_xlabel("Importance (Gain)")
    ax.set_title(f"Top {top_n} Feature Importances (LightGBM)")
    plt.tight_layout()

    plot_path = output_dir / "feature_importance.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n  Feature importance plot saved → {plot_path}")

    # Save JSON
    json_path = output_dir / "feature_importance.json"
    with open(json_path, "w") as f:
        json.dump(importance, f, indent=2)
    print(f"  Feature importance data saved → {json_path}")

    return importance

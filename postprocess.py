"""
postprocess.py — Calibration & adaptive rounding for grade regression.

Provides:
    - IsotonicCalibrator: isotonic regression to calibrate continuous predictions
    - ThresholdOptimizer: learns optimal decision boundaries instead of round()
    - compare_calibration: side-by-side metrics before/after calibration

Usage (standalone):
    Called automatically when --calibrate is passed to train_ps_classifier.py.

Expected benefit:
    Simple round() treats all grade boundaries equally, but the model may
    systematically over/under-predict certain grades.  Isotonic regression
    warps the prediction space to reduce bias, and learned thresholds can
    shift boundaries to maximise accuracy or F1 where it matters most.
"""

from pathlib import Path
import json

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score


# ---------------------------------------------------------------------------
# Isotonic calibrator
# ---------------------------------------------------------------------------


class IsotonicCalibrator:
    """Isotonic regression calibrator for continuous grade predictions.

    Fit on validation set (pred, true_label) pairs.  Transforms new
    predictions to reduce monotonic bias.
    """

    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False

    def fit(self, preds: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        self.iso.fit(preds.ravel(), labels.ravel().astype(float))
        self._fitted = True
        return self

    def transform(self, preds: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("IsotonicCalibrator has not been fitted yet.")
        return self.iso.predict(preds.ravel())

    def save(self, path: str | Path) -> None:
        data = {
            "X_min": float(self.iso.X_min_),
            "X_max": float(self.iso.X_max_),
            "X_thresholds": self.iso.X_thresholds_.tolist(),
            "y_thresholds": self.iso.y_thresholds_.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"  Isotonic calibrator saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "IsotonicCalibrator":
        with open(path) as f:
            data = json.load(f)
        cal = cls()
        cal.iso.X_min_ = data["X_min"]
        cal.iso.X_max_ = data["X_max"]
        cal.iso.X_thresholds_ = np.array(data["X_thresholds"])
        cal.iso.y_thresholds_ = np.array(data["y_thresholds"])
        cal.iso.increasing_ = True
        cal._fitted = True
        return cal


# ---------------------------------------------------------------------------
# Threshold optimizer
# ---------------------------------------------------------------------------


class ThresholdOptimizer:
    """Learns optimal decision thresholds for rounding continuous predictions.

    Instead of rounding at 0.5/1.5/2.5/... it finds thresholds that
    maximise accuracy (or macro F1) on a validation set.
    """

    def __init__(self, num_classes: int = 9):
        self.num_classes = num_classes
        self.thresholds: np.ndarray | None = None

    def fit(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        metric: str = "accuracy",
    ) -> "ThresholdOptimizer":
        """Find optimal thresholds via grid search.

        Searches each boundary independently in a local range around
        the default boundary (grade - 0.5).
        """
        preds = preds.ravel()
        labels = labels.ravel().astype(int)

        # Default thresholds: 0.5, 1.5, ..., 7.5
        n_boundaries = self.num_classes - 1
        thresholds = np.array([i + 0.5 for i in range(n_boundaries)])

        # Greedy sequential optimization of each threshold
        for b in range(n_boundaries):
            best_val, best_t = -1.0, thresholds[b]
            for candidate in np.arange(
                max(b, thresholds[b] - 0.8),
                min(b + 1.0, thresholds[b] + 0.8),
                0.05,
            ):
                trial = thresholds.copy()
                trial[b] = candidate
                rounded = self._apply_thresholds(preds, trial)
                if metric == "f1":
                    score = f1_score(labels, rounded, average="macro", zero_division=0)
                else:
                    score = accuracy_score(labels, rounded)
                if score > best_val:
                    best_val = score
                    best_t = candidate
            thresholds[b] = best_t

        self.thresholds = thresholds
        return self

    def predict(self, preds: np.ndarray) -> np.ndarray:
        if self.thresholds is None:
            raise RuntimeError("ThresholdOptimizer has not been fitted.")
        return self._apply_thresholds(preds.ravel(), self.thresholds)

    @staticmethod
    def _apply_thresholds(preds: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        result = np.zeros_like(preds, dtype=int)
        for t in thresholds:
            result += (preds > t).astype(int)
        return result

    def save(self, path: str | Path) -> None:
        data = {"thresholds": self.thresholds.tolist(), "num_classes": self.num_classes}
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"  Threshold optimizer saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ThresholdOptimizer":
        with open(path) as f:
            data = json.load(f)
        opt = cls(num_classes=data["num_classes"])
        opt.thresholds = np.array(data["thresholds"])
        return opt


# ---------------------------------------------------------------------------
# Calibration pipeline
# ---------------------------------------------------------------------------


def calibrate_predictions(
    val_preds: np.ndarray,
    val_labels: np.ndarray,
    test_preds: np.ndarray,
    test_labels: np.ndarray,
    num_classes: int,
    output_dir: str | Path,
) -> dict:
    """Full calibration pipeline: fit on val, evaluate on test.

    Returns dict with before/after metrics and the fitted objects.
    """
    output_dir = Path(output_dir)
    val_labels_int = np.array(val_labels, dtype=int)
    test_labels_int = np.array(test_labels, dtype=int)

    # --- Before calibration ---
    raw_rounded = np.clip(np.round(test_preds), 0, num_classes - 1).astype(int)
    before = {
        "mae": float(np.mean(np.abs(test_preds - test_labels_int))),
        "accuracy": accuracy_score(test_labels_int, raw_rounded),
        "f1_macro": f1_score(test_labels_int, raw_rounded, average="macro", zero_division=0),
    }

    # --- Isotonic calibration ---
    calibrator = IsotonicCalibrator().fit(val_preds, val_labels_int)
    cal_test = calibrator.transform(test_preds)
    calibrator.save(output_dir / "isotonic_calibrator.json")

    cal_rounded = np.clip(np.round(cal_test), 0, num_classes - 1).astype(int)
    after_isotonic = {
        "mae": float(np.mean(np.abs(cal_test - test_labels_int))),
        "accuracy": accuracy_score(test_labels_int, cal_rounded),
        "f1_macro": f1_score(test_labels_int, cal_rounded, average="macro", zero_division=0),
    }

    # --- Threshold optimization (on raw preds) ---
    thresh_opt = ThresholdOptimizer(num_classes=num_classes)
    thresh_opt.fit(val_preds, val_labels_int, metric="f1")
    thresh_rounded = thresh_opt.predict(test_preds)
    thresh_opt.save(output_dir / "threshold_optimizer.json")

    after_threshold = {
        "mae": float(np.mean(np.abs(test_preds - test_labels_int))),  # MAE unchanged
        "accuracy": accuracy_score(test_labels_int, thresh_rounded),
        "f1_macro": f1_score(test_labels_int, thresh_rounded, average="macro", zero_division=0),
    }

    # --- Combined: isotonic + threshold ---
    thresh_opt_cal = ThresholdOptimizer(num_classes=num_classes)
    thresh_opt_cal.fit(calibrator.transform(val_preds), val_labels_int, metric="f1")
    combined_rounded = thresh_opt_cal.predict(cal_test)
    thresh_opt_cal.save(output_dir / "threshold_optimizer_calibrated.json")

    after_combined = {
        "mae": float(np.mean(np.abs(cal_test - test_labels_int))),
        "accuracy": accuracy_score(test_labels_int, combined_rounded),
        "f1_macro": f1_score(test_labels_int, combined_rounded, average="macro", zero_division=0),
    }

    # --- Print comparison ---
    print("\n" + "=" * 60)
    print("  Calibration Results (Test Set)")
    print("=" * 60)
    print(f"  {'Method':<30} {'MAE':>8} {'Acc':>8} {'F1':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
    for name, m in [
        ("Raw round()", before),
        ("Isotonic + round()", after_isotonic),
        ("Optimized thresholds", after_threshold),
        ("Isotonic + opt. thresholds", after_combined),
    ]:
        print(f"  {name:<30} {m['mae']:>8.4f} {m['accuracy']:>8.4f} {m['f1_macro']:>8.4f}")

    return {
        "before": before,
        "after_isotonic": after_isotonic,
        "after_threshold": after_threshold,
        "after_combined": after_combined,
        "calibrator": calibrator,
        "threshold_optimizer": thresh_opt,
    }

"""Risk score calibration against labeled benchmarks (Issue 34).

Maps heuristic risk scores to calibrated failure probabilities using
Platt scaling, and validates the mapping with Expected Calibration Error
(ECE).  This is the bridge from "0.8 risk = somewhat high" to "0.8 risk ≈
80% chance of a bad generation."

Public API
----------
- compute_ece           — Expected Calibration Error
- fit_platt_scaling     — learn (a, b) from labeled data
- apply_platt_scaling   — sigmoid transform with fitted params
- RiskCalibrationResult — dataclass holding ECE + Platt params
- evaluate_calibration  — convenience: fit + ECE in one call

Data Collection Workflow
------------------------
See docs/risk-calibration.md for the full workflow.  In short:

1. Run CoreVital on a benchmark dataset with known-quality labels
   (TruthfulQA, HellaSwag, or a custom prompt suite).
2. Collect ``(risk_score, label)`` pairs — label=1 for failures, 0 for good.
3. Call ``evaluate_calibration(scores, labels)`` to fit Platt scaling and
   compute ECE.  Target ECE < 0.10.
4. Use ``apply_platt_scaling(raw_score, result.a, result.b)`` in production
   to convert raw risk scores to calibrated probabilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------


def compute_ece(
    predicted_probs: List[float],
    actual_labels: List[int],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error — weighted gap between predicted and actual.

    Partitions predictions into *n_bins* equal-width bins over [0, 1].
    For each bin, computes |mean_prediction − mean_label| weighted by the
    bin's fraction of total samples.

    Args:
        predicted_probs: Predicted probability of failure, each in [0, 1].
        actual_labels:   Ground truth labels — 1 = failure, 0 = success.
        n_bins:          Number of equal-width bins (default 10).

    Returns:
        ECE in [0, 1].  0.0 = perfectly calibrated.

    Raises:
        ValueError: If inputs are empty, different lengths, or labels not in {0, 1}.
    """
    if not predicted_probs or not actual_labels:
        raise ValueError("predicted_probs and actual_labels must be non-empty")
    if len(predicted_probs) != len(actual_labels):
        raise ValueError(f"Length mismatch: {len(predicted_probs)} predictions vs {len(actual_labels)} labels")
    if any(v not in (0, 1) for v in actual_labels):
        raise ValueError("actual_labels must contain only 0 and 1")

    total = len(predicted_probs)
    bin_width = 1.0 / n_bins
    ece = 0.0

    for i in range(n_bins):
        lo = i * bin_width
        hi = lo + bin_width

        bin_preds: List[float] = []
        bin_labels: List[int] = []
        for p, label in zip(predicted_probs, actual_labels, strict=True):
            if i == n_bins - 1:
                in_bin = lo <= p <= hi
            else:
                in_bin = lo <= p < hi
            if in_bin:
                bin_preds.append(p)
                bin_labels.append(label)

        if bin_preds:
            avg_pred = sum(bin_preds) / len(bin_preds)
            avg_label = sum(bin_labels) / len(bin_labels)
            ece += (len(bin_preds) / total) * abs(avg_pred - avg_label)

    return ece


# ---------------------------------------------------------------------------
# Platt Scaling
# ---------------------------------------------------------------------------

_CLIP_MIN = 1e-15
_CLIP_MAX = 1.0 - 1e-15
_MAX_ITER = 500
_TOL = 1e-8
_LR = 0.05


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clip(p: float) -> float:
    return max(_CLIP_MIN, min(_CLIP_MAX, p))


def fit_platt_scaling(
    raw_scores: List[float],
    labels: List[int],
    lr: float = _LR,
    max_iter: int = _MAX_ITER,
    tol: float = _TOL,
) -> Tuple[float, float]:
    """Fit Platt scaling: calibrated_prob = sigmoid(a * raw_score + b).

    Uses gradient descent on negative log-likelihood.  No external
    optimiser required (scipy is not a dependency).

    Args:
        raw_scores: Uncalibrated risk scores (typically in [0, 1]).
        labels:     Ground truth — 1 = failure, 0 = success.
        lr:         Learning rate for gradient descent.
        max_iter:   Maximum optimisation iterations.
        tol:        Convergence tolerance on NLL change.

    Returns:
        ``(a, b)`` where ``calibrated = sigmoid(a * raw + b)``.

    Raises:
        ValueError: If inputs are empty, different lengths, or labels not in {0, 1}.
    """
    if not raw_scores or not labels:
        raise ValueError("raw_scores and labels must be non-empty")
    if len(raw_scores) != len(labels):
        raise ValueError(f"Length mismatch: {len(raw_scores)} scores vs {len(labels)} labels")
    if any(v not in (0, 1) for v in labels):
        raise ValueError("labels must contain only 0 and 1")

    a = 0.0
    b = 0.0
    prev_nll = float("inf")

    for _ in range(max_iter):
        grad_a = 0.0
        grad_b = 0.0
        nll = 0.0

        for raw, label in zip(raw_scores, labels, strict=True):
            p = _clip(_sigmoid(a * raw + b))
            nll -= label * math.log(p) + (1 - label) * math.log(1 - p)
            err = p - label
            grad_a += err * raw
            grad_b += err

        a -= lr * grad_a / len(raw_scores)
        b -= lr * grad_b / len(raw_scores)

        if abs(prev_nll - nll) < tol:
            break
        prev_nll = nll

    return a, b


def apply_platt_scaling(raw_score: float, a: float, b: float) -> float:
    """Map a raw risk score to a calibrated probability.

    Args:
        raw_score: Raw (heuristic) risk score.
        a, b:      Parameters from :func:`fit_platt_scaling`.

    Returns:
        Calibrated probability in (0, 1).
    """
    return _sigmoid(a * raw_score + b)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RiskCalibrationResult:
    """Holds Platt scaling parameters and calibration quality metrics."""

    a: float
    b: float
    ece_raw: float
    ece_calibrated: float
    n_samples: int
    label_rate: float

    def to_dict(self) -> dict:
        return {
            "platt_a": self.a,
            "platt_b": self.b,
            "ece_raw": round(self.ece_raw, 6),
            "ece_calibrated": round(self.ece_calibrated, 6),
            "n_samples": self.n_samples,
            "label_rate": round(self.label_rate, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RiskCalibrationResult":
        return cls(
            a=d["platt_a"],
            b=d["platt_b"],
            ece_raw=d["ece_raw"],
            ece_calibrated=d["ece_calibrated"],
            n_samples=d["n_samples"],
            label_rate=d["label_rate"],
        )


# ---------------------------------------------------------------------------
# Convenience: fit + evaluate
# ---------------------------------------------------------------------------


def evaluate_calibration(
    raw_scores: List[float],
    labels: List[int],
    n_bins: int = 10,
    lr: float = _LR,
    max_iter: int = _MAX_ITER,
) -> RiskCalibrationResult:
    """Fit Platt scaling and evaluate calibration quality end-to-end.

    1. Compute ECE of the raw (uncalibrated) scores.
    2. Fit Platt scaling.
    3. Compute ECE of the calibrated scores.
    4. Return everything in a :class:`RiskCalibrationResult`.

    Args:
        raw_scores: Uncalibrated risk scores.
        labels:     Ground truth — 1 = failure, 0 = success.
        n_bins:     Bins for ECE computation.
        lr:         Learning rate for Platt fitting.
        max_iter:   Max iterations for Platt fitting.

    Returns:
        :class:`RiskCalibrationResult` with raw ECE, calibrated ECE,
        Platt params, and sample metadata.
    """
    ece_raw = compute_ece(raw_scores, labels, n_bins=n_bins)

    a, b = fit_platt_scaling(raw_scores, labels, lr=lr, max_iter=max_iter)

    calibrated = [apply_platt_scaling(s, a, b) for s in raw_scores]
    ece_calibrated = compute_ece(calibrated, labels, n_bins=n_bins)

    return RiskCalibrationResult(
        a=a,
        b=b,
        ece_raw=ece_raw,
        ece_calibrated=ece_calibrated,
        n_samples=len(raw_scores),
        label_rate=sum(labels) / len(labels),
    )

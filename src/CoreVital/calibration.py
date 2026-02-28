"""CoreVital Calibration — data-driven baseline profiling.

Replaces static thresholds with empirical baselines built from known-healthy
runs.  When a calibration profile exists, production traces are scored by
statistical divergence from the baseline rather than relying on hardcoded
constants.

Public API
----------
- MetricDistribution   — empirical distribution of a single metric
- CalibrationProfile   — per-metric distributions for a model + deployment
- calibrate_from_runs  — build a profile from a list of trace dicts
- compute_divergence_score — score a trace against a baseline profile
"""

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class MetricDistribution:
    """Empirical distribution of a single metric from calibration runs."""

    values: List[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return statistics.mean(self.values) if self.values else 0.0

    @property
    def std(self) -> float:
        return statistics.stdev(self.values) if len(self.values) >= 2 else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.values) if self.values else 0.0

    def percentile(self, p: float) -> float:
        if not self.values:
            return 0.0
        sorted_v = sorted(self.values)
        idx = int(len(sorted_v) * p / 100.0)
        return sorted_v[min(idx, len(sorted_v) - 1)]

    def z_score(self, value: float) -> float:
        """How many standard deviations *value* is from baseline mean."""
        if self.std < 1e-10:
            return 0.0
        return (value - self.mean) / self.std

    def is_anomalous(self, value: float, z_threshold: float = 3.0) -> bool:
        """Whether *value* is anomalous relative to baseline."""
        return abs(self.z_score(value)) > z_threshold


@dataclass
class CalibrationProfile:
    """Baseline distributions for a specific model + deployment config."""

    model_id: str = ""
    num_runs: int = 0
    entropy_per_step: MetricDistribution = field(default_factory=MetricDistribution)
    margin_per_step: MetricDistribution = field(default_factory=MetricDistribution)
    surprisal_per_step: MetricDistribution = field(default_factory=MetricDistribution)
    l2_norm_per_layer: Dict[int, MetricDistribution] = field(default_factory=dict)
    attention_entropy_per_layer: Dict[int, MetricDistribution] = field(default_factory=dict)

    # ---- persistence --------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialize to JSON for reuse."""
        data = {
            "model_id": self.model_id,
            "num_runs": self.num_runs,
            "entropy": {
                "mean": self.entropy_per_step.mean,
                "std": self.entropy_per_step.std,
                "median": self.entropy_per_step.median,
                "p5": self.entropy_per_step.percentile(5),
                "p95": self.entropy_per_step.percentile(95),
                "values": self.entropy_per_step.values,
            },
            "margin": {
                "mean": self.margin_per_step.mean,
                "std": self.margin_per_step.std,
                "median": self.margin_per_step.median,
                "values": self.margin_per_step.values,
            },
            "surprisal": {
                "mean": self.surprisal_per_step.mean,
                "std": self.surprisal_per_step.std,
                "median": self.surprisal_per_step.median,
                "values": self.surprisal_per_step.values,
            },
            "l2_norms": {
                str(k): {"mean": v.mean, "std": v.std, "values": v.values}
                for k, v in self.l2_norm_per_layer.items()
            },
            "attention_entropy": {
                str(k): {"mean": v.mean, "std": v.std, "values": v.values}
                for k, v in self.attention_entropy_per_layer.items()
            },
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "CalibrationProfile":
        """Deserialize from a JSON file previously written by :meth:`save`."""
        data = json.loads(Path(path).read_text())
        profile = cls(
            model_id=data.get("model_id", ""),
            num_runs=data.get("num_runs", 0),
        )
        for section, attr in [
            ("entropy", "entropy_per_step"),
            ("margin", "margin_per_step"),
            ("surprisal", "surprisal_per_step"),
        ]:
            sec = data.get(section, {})
            setattr(profile, attr, MetricDistribution(values=sec.get("values", [])))

        for section, attr in [
            ("l2_norms", "l2_norm_per_layer"),
            ("attention_entropy", "attention_entropy_per_layer"),
        ]:
            layer_dict: Dict[int, MetricDistribution] = {}
            for k, v in data.get(section, {}).items():
                layer_dict[int(k)] = MetricDistribution(values=v.get("values", []))
            setattr(profile, attr, layer_dict)

        return profile


# ---------------------------------------------------------------------------
# Build a profile from trace dicts
# ---------------------------------------------------------------------------


def calibrate_from_runs(
    model_id: str,
    traces: List[dict],
) -> CalibrationProfile:
    """Build a calibration profile from a set of known-healthy traces.

    Each *trace* is a dict with a ``timeline`` key whose value is a list of
    step dicts.  Step dicts contain ``logits_summary`` and ``layers`` sub-dicts
    matching the CoreVital report schema.
    """
    profile = CalibrationProfile(model_id=model_id, num_runs=len(traces))

    for trace in traces:
        for step in trace.get("timeline", []):
            logits = step.get("logits_summary", {})
            if logits.get("entropy") is not None:
                profile.entropy_per_step.values.append(float(logits["entropy"]))
            if logits.get("top_k_margin") is not None:
                profile.margin_per_step.values.append(float(logits["top_k_margin"]))
            if logits.get("surprisal") is not None:
                profile.surprisal_per_step.values.append(float(logits["surprisal"]))

            for layer in step.get("layers", []):
                idx = layer.get("layer_index", 0)
                hidden = layer.get("hidden_summary", {})
                if hidden.get("l2_norm_mean") is not None:
                    if idx not in profile.l2_norm_per_layer:
                        profile.l2_norm_per_layer[idx] = MetricDistribution()
                    profile.l2_norm_per_layer[idx].values.append(float(hidden["l2_norm_mean"]))

                attn = layer.get("attention_summary", {})
                if attn.get("entropy_mean") is not None:
                    if idx not in profile.attention_entropy_per_layer:
                        profile.attention_entropy_per_layer[idx] = MetricDistribution()
                    profile.attention_entropy_per_layer[idx].values.append(float(attn["entropy_mean"]))

    return profile


# ---------------------------------------------------------------------------
# Score a trace against baseline
# ---------------------------------------------------------------------------


def compute_divergence_score(
    trace: dict,
    baseline: CalibrationProfile,
    z_threshold: float = 3.0,
) -> Tuple[float, List[str]]:
    """Score how much a trace diverges from calibration baseline.

    Returns ``(divergence_score, anomalies)`` where *divergence_score* is in
    ``[0, 1]`` and *anomalies* is a human-readable list of flagged signals.
    """
    anomalies: List[str] = []
    z_scores: List[float] = []

    for step in trace.get("timeline", []):
        logits = step.get("logits_summary", {})
        step_idx = step.get("step_index", "?")

        ent = logits.get("entropy")
        if ent is not None and baseline.entropy_per_step.values:
            z = baseline.entropy_per_step.z_score(float(ent))
            z_scores.append(abs(z))
            if abs(z) > z_threshold:
                anomalies.append(f"Step {step_idx}: entropy z-score {z:.1f}")

        margin = logits.get("top_k_margin")
        if margin is not None and baseline.margin_per_step.values:
            z = baseline.margin_per_step.z_score(float(margin))
            z_scores.append(abs(z))
            if abs(z) > z_threshold:
                anomalies.append(f"Step {step_idx}: margin z-score {z:.1f}")

        surprisal = logits.get("surprisal")
        if surprisal is not None and baseline.surprisal_per_step.values:
            z = baseline.surprisal_per_step.z_score(float(surprisal))
            z_scores.append(abs(z))
            if abs(z) > z_threshold:
                anomalies.append(f"Step {step_idx}: surprisal z-score {z:.1f}")

    if not z_scores:
        return 0.0, []

    mean_z = sum(z_scores) / len(z_scores)
    divergence = min(1.0, mean_z / 6.0)

    return divergence, anomalies

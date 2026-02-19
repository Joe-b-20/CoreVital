# ============================================================================
# CoreVital - Weights & Biases Sink
#
# Purpose: Log CoreVital reports to W&B as scalars, artifact, and optional images
# Inputs: Report objects
# Outputs: Metrics and artifact logged to current or new W&B run
# Dependencies: base, reporting.schema, wandb (optional)
# Usage: sink = WandBSink(project="corevital"); sink.write(report)
#
# Changelog:
#   2026-02-18: Initial implementation (#24):
#               - Log scalars: risk_score, health flags, mean entropy/perplexity/surprisal
#               - Log full report JSON as W&B artifact
#               - Log basin score heatmap as image when prompt_analysis present
# ============================================================================

from __future__ import annotations

from typing import Any, Optional

from CoreVital.errors import SinkError
from CoreVital.logging_utils import get_logger
from CoreVital.reporting.schema import Report
from CoreVital.sinks.base import Sink

logger = get_logger(__name__)


def _try_import_wandb() -> Any:
    """Lazy import of wandb with clear error message."""
    try:
        import wandb

        return wandb
    except ImportError as exc:
        raise ImportError(
            "WandBSink requires the 'wandb' package.\n"
            "Install it with: pip install corevital[wandb]\n"
            "Or directly: pip install wandb"
        ) from exc


class WandBSink(Sink):
    """
    Sink that logs Report to Weights & Biases.

    - Scalars: risk_score, health flags (0/1 or counts), mean entropy/perplexity/surprisal,
      total_steps, elapsed_ms
    - Artifact: full report JSON
    - Optional: basin score heatmap image (layers x heads) when prompt_analysis is present
    """

    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        *,
        local_output_dir: str = "runs",
        log_heatmap: bool = True,
    ):
        """
        Initialize W&B sink.

        Args:
            project: W&B project name (or set WANDB_PROJECT env var)
            entity: W&B entity/team (or set WANDB_ENTITY env var)
            run_name: Optional run name (default: trace_id prefix)
            local_output_dir: Also write JSON locally as backup
            log_heatmap: If True, log basin score heatmap when prompt_analysis exists
        """
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.local_output_dir = local_output_dir
        self.log_heatmap = log_heatmap
        logger.info("WandBSink initialized (project=%s)", self.project or "env")

    def _ensure_run(self, report: Report) -> Any:
        """Ensure a W&B run is active; init one if not."""
        wandb = _try_import_wandb()
        if wandb.run is None:
            wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.run_name or f"corevital-{report.trace_id[:8]}",
                config={
                    "model": report.model.hf_id,
                    "trace_id": report.trace_id[:8],
                    "device": report.model.device,
                },
            )
        return wandb

    def _log_scalars(self, wandb: Any, report: Report) -> None:
        """Log scalar metrics under corevital/ namespace."""
        metrics: dict[str, float] = {}

        # Summary
        s = report.summary
        metrics["corevital/total_steps"] = float(s.total_steps)
        metrics["corevital/elapsed_ms"] = float(s.elapsed_ms)
        metrics["corevital/prompt_tokens"] = float(s.prompt_tokens)
        metrics["corevital/generated_tokens"] = float(s.generated_tokens)

        # Risk
        risk = report.extensions.get("risk")
        if isinstance(risk, dict) and "risk_score" in risk:
            metrics["corevital/risk_score"] = float(risk["risk_score"])

        # Health flags
        hf = report.health_flags
        if hf:
            metrics["corevital/health/nan_detected"] = 1.0 if hf.nan_detected else 0.0
            metrics["corevital/health/inf_detected"] = 1.0 if hf.inf_detected else 0.0
            metrics["corevital/health/attention_collapse_detected"] = 1.0 if hf.attention_collapse_detected else 0.0
            metrics["corevital/health/high_entropy_steps"] = float(hf.high_entropy_steps)
            metrics["corevital/health/repetition_loop_detected"] = 1.0 if hf.repetition_loop_detected else 0.0
            metrics["corevital/health/mid_layer_anomaly_detected"] = 1.0 if hf.mid_layer_anomaly_detected else 0.0

        # Timeline means (entropy, perplexity, surprisal)
        if report.timeline:
            entropies = []
            perplexities = []
            surprisals = []
            for step in report.timeline:
                ls = step.logits_summary
                if ls:
                    if ls.entropy is not None:
                        entropies.append(ls.entropy)
                    if ls.perplexity is not None:
                        perplexities.append(ls.perplexity)
                    if ls.surprisal is not None:
                        surprisals.append(ls.surprisal)
            if entropies:
                metrics["corevital/entropy_mean"] = sum(entropies) / len(entropies)
            if perplexities:
                metrics["corevital/perplexity_mean"] = sum(perplexities) / len(perplexities)
            if surprisals:
                metrics["corevital/surprisal_mean"] = sum(surprisals) / len(surprisals)

        if metrics:
            wandb.log(metrics)

    def _log_artifact(self, wandb: Any, report: Report, local_path: str) -> None:
        """Log the report JSON as a W&B artifact."""
        artifact = wandb.Artifact(
            name=f"corevital-report-{report.trace_id[:8]}",
            type="corevital_report",
            metadata={
                "trace_id": report.trace_id,
                "model": report.model.hf_id,
                "schema_version": report.schema_version,
            },
        )
        artifact.add_file(local_path=local_path, name="report.json")
        wandb.log_artifact(artifact)

    def _log_basin_heatmap(self, wandb: Any, report: Report) -> None:
        """Log a layers x heads basin score heatmap as image when prompt_analysis exists."""
        pa = report.prompt_analysis
        if not pa or not pa.layers:
            return
        try:
            import numpy as np

            rows = []
            max_heads = 0
            for layer in pa.layers:
                basins = layer.basin_scores or []
                max_heads = max(max_heads, len(basins))
                rows.append(basins)
            if max_heads == 0:
                return
            # Pad rows to same length
            z = np.array([r + [np.nan] * (max_heads - len(r)) for r in rows], dtype=float)
            # Normalize to 0–1 for colormap (e.g. 0->red, 0.5->green, 1->blue via simple RGB)
            vmin, vmax = 0.0, 2.0
            z_clip = np.clip(z, vmin, vmax)
            z_norm = (z_clip - vmin) / (vmax - vmin) if vmax > vmin else z_clip
            # Grayscale image for W&B (H, W); wandb.Image accepts numpy (H,W) or (H,W,3)
            img_uint8 = (z_norm * 255).astype(np.uint8)
            wandb.log({"corevital/attention_basin_heatmap": wandb.Image(img_uint8)})
        except Exception as e:
            logger.debug("Skip basin heatmap: %s", e)

    def write(self, report: Report) -> str:
        """
        Log report to W&B: scalars, artifact, and optional heatmap.

        Also writes a local JSON backup via LocalFileSink.

        Args:
            report: Report to log

        Returns:
            Identifier string (trace_id + "wandb")

        Raises:
            SinkError: If W&B logging fails
        """
        from CoreVital.sinks.local_file import LocalFileSink

        local_sink = LocalFileSink(self.local_output_dir)
        local_path = local_sink.write(report)
        logger.info("Local backup written to %s", local_path)

        try:
            wandb = _try_import_wandb()
            self._ensure_run(report)
            self._log_scalars(wandb, report)
            self._log_artifact(wandb, report, local_path)
            if self.log_heatmap:
                self._log_basin_heatmap(wandb, report)
            return f"wandb:{report.trace_id[:8]} + {local_path}"
        except ImportError:
            raise
        except Exception as e:
            raise SinkError(
                f"Failed to log to W&B: {e}",
                details=f"trace_id={report.trace_id[:8]}",
            ) from e

"""
CoreVital - Library API: CoreVitalMonitor (Streaming + Post-Run)

Embeddable monitor that runs instrumented generation and exposes:
- Post-run summary (risk, flags, fingerprint, narrative)
- should_intervene() for health-aware decoding (Phase-5)
- Async stream of per-step events (Phase-4/Library API streaming, v1 = post-run replay)
"""

import asyncio
from contextlib import contextmanager
from typing import Any, Dict, List, Literal, Optional, cast

from CoreVital.config import Config
from CoreVital.instrumentation.collector import InstrumentationCollector
from CoreVital.reporting.report_builder import ReportBuilder


class CoreVitalMonitor:
    """
    Embeddable monitor: run instrumented generation and query risk, flags, summary.

    Use run() or wrap_generation() to execute one instrumented run; then call
    get_risk_score(), get_health_flags(), get_summary(), or should_intervene().
    """

    def __init__(
        self,
        capture_mode: str = "summary",
        risk_threshold: float = 0.7,
        intervene_on_risk_above: float = 0.8,
        intervene_on_signals: Optional[List[str]] = None,
        max_new_tokens: int = 20,
        device: str = "auto",
        seed: Optional[int] = 42,
    ):
        """
        Args:
            capture_mode: "summary", "full", or "on_risk" (report payload size).
            risk_threshold: For on_risk mode: store full when risk >= this.
            intervene_on_risk_above: should_intervene() returns True if risk >= this.
            intervene_on_signals: If any of these in warning_signals, should_intervene() True.
            max_new_tokens: Default for run().
            device: Default for run().
            seed: Default for run().
        """
        self.capture_mode = capture_mode
        self.risk_threshold = risk_threshold
        self.intervene_on_risk_above = intervene_on_risk_above
        self.intervene_on_signals = intervene_on_signals or ["repetition_loop"]
        self._default_max_new_tokens = max_new_tokens
        self._default_device = device
        self._default_seed = seed
        self._report: Optional[Any] = None
        self._results: Optional[Any] = None

    def _make_config(
        self,
        model_id: str,
        max_new_tokens: Optional[int] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Config:
        try:
            config = Config.from_default()
        except Exception:
            config = Config()
        config.model.hf_id = model_id
        config.capture.capture_mode = cast(Literal["summary", "full", "on_risk"], self.capture_mode)
        config.capture.risk_threshold = self.risk_threshold
        config.generation.max_new_tokens = (
            max_new_tokens if max_new_tokens is not None else self._default_max_new_tokens
        )
        config.device.requested = device if device is not None else self._default_device
        default_seed = seed if seed is not None else self._default_seed
        config.generation.seed = default_seed if default_seed is not None else 42
        for k, v in kwargs.items():
            if k in ("load_in_4bit", "load_in_8bit"):
                setattr(config.model, k, bool(v))
            elif hasattr(config.generation, k):
                setattr(config.generation, k, v)
        return config

    def run(
        self,
        model_id: str,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        **generate_kwargs: Any,
    ) -> "CoreVitalMonitor":
        """
        Run instrumented generation and build report. Call get_risk_score() etc. after.

        Returns:
            self for chaining.
        """
        config = self._make_config(
            model_id,
            max_new_tokens=max_new_tokens,
            device=device,
            seed=seed,
            **generate_kwargs,
        )
        collector = InstrumentationCollector(config)
        self._results = collector.run(prompt)
        builder = ReportBuilder(config)
        self._report = builder.build(self._results, prompt)
        return self

    @contextmanager
    def wrap_generation(
        self,
        model_id: str,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        **generate_kwargs: Any,
    ):
        """
        Context manager: run instrumented generation on enter; yield self for querying.
        """
        self.run(
            model_id,
            prompt,
            max_new_tokens=max_new_tokens,
            device=device,
            seed=seed,
            **generate_kwargs,
        )
        try:
            yield self
        finally:
            pass

    def get_risk_score(self) -> float:
        """Return risk score in [0, 1] after a run. 0.0 if no run or no risk data."""
        if self._report is None:
            return 0.0
        ext = getattr(self._report, "extensions", None) or {}
        risk = ext.get("risk") or {}
        return float(risk.get("risk_score", 0.0))

    def get_health_flags(self) -> Dict[str, Any]:
        """Return health flags dict after a run. Empty if no run."""
        if self._report is None or self._report.health_flags is None:
            return {}
        h = self._report.health_flags
        return {
            "nan_detected": h.nan_detected,
            "inf_detected": h.inf_detected,
            "attention_collapse_detected": h.attention_collapse_detected,
            "high_entropy_steps": h.high_entropy_steps,
            "repetition_loop_detected": h.repetition_loop_detected,
            "mid_layer_anomaly_detected": h.mid_layer_anomaly_detected,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Return full summary after a run: risk, flags, fingerprint, narrative, trace_id."""
        out: Dict[str, Any] = {
            "risk_score": self.get_risk_score(),
            "health_flags": self.get_health_flags(),
        }
        if self._report is not None:
            ext = getattr(self._report, "extensions", None) or {}
            if "fingerprint" in ext:
                out["fingerprint"] = ext["fingerprint"]
            if "narrative" in ext:
                out["narrative"] = ext["narrative"]
            if "early_warning" in ext:
                out["early_warning"] = ext["early_warning"]
            out["trace_id"] = getattr(self._report, "trace_id", None)
            model = getattr(self._report, "model", None)
            out["model_id"] = getattr(model, "hf_id", None) if model else None
        return out

    def should_intervene(self) -> bool:
        """
        Return True if the last run's risk or warning signals exceed thresholds (Phase-5).

        Use after run() or when exiting wrap_generation to decide whether to resample or abort.
        """
        if self._report is None:
            return False
        risk = self.get_risk_score()
        if risk >= self.intervene_on_risk_above:
            return True
        ext = getattr(self._report, "extensions", None) or {}
        ew = ext.get("early_warning") or {}
        signals = ew.get("warning_signals") or []
        for s in self.intervene_on_signals:
            if s in signals:
                return True
        return False

    async def stream(
        self,
        model_id: str,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        **generate_kwargs: Any,
    ):
        """
        Async iterator over per-step events for a single run.

        v1 implementation: run is executed first, then timeline is replayed as a stream
        of events. This gives a streaming-shaped API without modifying the generation
        loop; future versions may emit events online during generation.

        Yields dicts of the form:
            {
                "step_index": int,
                "token_id": int,
                "token_text": str,
                "entropy": Optional[float],
                "surprisal": Optional[float],
                "risk_so_far": float,          # final risk score repeated per step
                "warning_signals": List[str],  # from early_warning extension
            }
        """

        # Run the instrumented generation in a thread so we don't block the event loop
        loop = asyncio.get_running_loop()

        def _run_sync() -> None:
            self.run(
                model_id,
                prompt,
                max_new_tokens=max_new_tokens,
                device=device,
                seed=seed,
                **generate_kwargs,
            )

        await loop.run_in_executor(None, _run_sync)

        if self._report is None:
            return

        risk_score = self.get_risk_score()
        ext = getattr(self._report, "extensions", None) or {}
        ew = ext.get("early_warning") or {}
        warning_signals = ew.get("warning_signals") or []

        timeline = getattr(self._report, "timeline", []) or []
        for step in timeline:
            ls = getattr(step, "logits_summary", None)
            entropy = getattr(ls, "entropy", None) if ls is not None else None
            surprisal = getattr(ls, "surprisal", None) if ls is not None else None
            yield {
                "step_index": step.step_index,
                "token_id": step.token.token_id,
                "token_text": step.token.token_text,
                "entropy": entropy,
                "surprisal": surprisal,
                "risk_so_far": risk_score,
                "warning_signals": warning_signals,
            }

    def get_report(self) -> Optional[Any]:
        """Return the last Report object, or None."""
        return self._report

    def get_results(self) -> Optional[Any]:
        """Return the last InstrumentationResults, or None."""
        return self._results

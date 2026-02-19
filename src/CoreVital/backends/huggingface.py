# ============================================================================
# CoreVital - Hugging Face inference backend
#
# Purpose: Run instrumented generation using transformers (load_model + collector).
# Full support: hidden_states, attentions, prompt_telemetry, Seq2Seq.
# ============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from CoreVital.backends.base import Backend, BackendCapabilities, StepCallback
from CoreVital.config import Config

if TYPE_CHECKING:
    from CoreVital.instrumentation.collector import InstrumentationCollector, InstrumentationResults
    from CoreVital.instrumentation.performance import PerformanceMonitor


class HuggingFaceBackend(Backend):
    """
    Backend that uses the built-in Hugging Face instrumentation (load_model + collector).

    Supports all CoreVital features: per-step hidden_states, attentions, logits,
    prompt telemetry (Phase-1b), and Seq2Seq (encoder/decoder, cross_attentions).
    """

    def __init__(self) -> None:
        self._collector: Optional["InstrumentationCollector"] = None
        self._config: Optional[Config] = None

    def run(
        self,
        config: Config,
        prompt: str,
        monitor: Optional["PerformanceMonitor"] = None,
        step_callback: Optional[StepCallback] = None,
    ) -> "InstrumentationResults":
        from CoreVital.instrumentation.collector import InstrumentationCollector

        if self._collector is None or self._config is not config:
            self._collector = InstrumentationCollector(config, backend=None)
            self._config = config
        return self._collector._run_impl(prompt, monitor, step_callback=step_callback)

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            hidden_states=True,
            attentions=True,
            prompt_telemetry=True,
            cross_attentions=True,  # when model is Seq2Seq
            is_seq2seq=False,  # determined per model at runtime
        )

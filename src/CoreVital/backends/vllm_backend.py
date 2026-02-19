# CoreVital - vLLM inference backend (stub)

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from CoreVital.backends.base import Backend, BackendCapabilities
from CoreVital.config import Config

if TYPE_CHECKING:
    from CoreVital.instrumentation.collector import InstrumentationResults
    from CoreVital.instrumentation.performance import PerformanceMonitor


class VLLMBackend(Backend):
    """vLLM backend (stub). Not yet implemented."""

    def run(
        self,
        config: Config,
        prompt: str,
        monitor: Optional["PerformanceMonitor"] = None,
    ) -> "InstrumentationResults":
        raise NotImplementedError(
            "VLLMBackend is not implemented yet. Use HuggingFaceBackend or config with default backend."
        )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            hidden_states=False,
            attentions=False,
            prompt_telemetry=False,
            cross_attentions=False,
            is_seq2seq=False,
        )

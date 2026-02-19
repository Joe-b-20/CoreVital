# ============================================================================
# CoreVital - llama.cpp inference backend (stub)
#
# Purpose: Placeholder for llama-cpp-python-backed instrumented generation.
# When implemented, use Llama class and return InstrumentationResults;
# capabilities may be limited (e.g. logits only, no per-layer hidden_states).
# ============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from CoreVital.backends.base import Backend, BackendCapabilities
from CoreVital.config import Config

if TYPE_CHECKING:
    from CoreVital.instrumentation.collector import InstrumentationResults
    from CoreVital.instrumentation.performance import PerformanceMonitor


class LlamaCppBackend(Backend):
    """
    llama-cpp-python backend (stub). Not yet implemented.

    When implemented: use Llama() for generation; map to InstrumentationResults.
    Capabilities TBD (likely logits only, no hidden_states/attentions).
    """

    def run(
        self,
        config: Config,
        prompt: str,
        monitor: Optional["PerformanceMonitor"] = None,
    ) -> "InstrumentationResults":
        raise NotImplementedError(
            "LlamaCppBackend is not implemented yet. Use HuggingFaceBackend or config with default backend."
        )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            hidden_states=False,
            attentions=False,
            prompt_telemetry=False,
            cross_attentions=False,
            is_seq2seq=False,
        )

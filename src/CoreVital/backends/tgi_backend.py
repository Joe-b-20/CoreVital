# ============================================================================
# CoreVital - TGI (Text Generation Inference) HTTP backend (stub)
#
# Purpose: Placeholder for TGI HTTP client-backed instrumented generation.
# When implemented, call TGI /generate or streaming endpoint and map
# responses to InstrumentationResults; capabilities depend on TGI API.
# ============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from CoreVital.backends.base import Backend, BackendCapabilities
from CoreVital.config import Config

if TYPE_CHECKING:
    from CoreVital.instrumentation.collector import InstrumentationResults
    from CoreVital.instrumentation.performance import PerformanceMonitor


class TGIBackend(Backend):
    """
    TGI (Text Generation Inference) HTTP backend (stub). Not yet implemented.

    When implemented: HTTP client to TGI server; map to InstrumentationResults.
    Capabilities depend on what TGI exposes (e.g. token logits if available).
    """

    def run(
        self,
        config: Config,
        prompt: str,
        monitor: Optional["PerformanceMonitor"] = None,
        step_callback: Optional[Any] = None,
    ) -> "InstrumentationResults":
        raise NotImplementedError(
            "TGIBackend is not implemented yet. Use HuggingFaceBackend or config with default backend."
        )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            hidden_states=False,
            attentions=False,
            prompt_telemetry=False,
            cross_attentions=False,
            is_seq2seq=False,
        )

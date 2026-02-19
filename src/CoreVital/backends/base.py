# ============================================================================
# CoreVital - Abstract inference backend
#
# Purpose: Define backend-agnostic interface for instrumented generation
# so the collector can run against Hugging Face, vLLM, llama.cpp, TGI, etc.
# ============================================================================

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from CoreVital.config import Config

if TYPE_CHECKING:
    from CoreVital.instrumentation.collector import InstrumentationResults
    from CoreVital.instrumentation.performance import PerformanceMonitor

# Real-time intervention: (step, generated_token_ids, last_layer_hidden_buffer, last_logits) -> True to stop
StepCallback = Callable[
    [int, List[int], Optional[List[Any]], Optional[Any]],
    bool,
]


@dataclass(frozen=True)
class BackendCapabilities:
    """What this backend can provide in InstrumentationResults."""

    hidden_states: bool = True
    attentions: bool = True
    prompt_telemetry: bool = True
    cross_attentions: bool = False  # Seq2Seq
    is_seq2seq: bool = False


class Backend(ABC):
    """
    Abstract inference backend for instrumented generation.

    Implementations (HuggingFaceBackend, VLLMBackend, etc.) load the model
    and run generation, returning InstrumentationResults in the format
    expected by ReportBuilder and the CLI.
    """

    @abstractmethod
    def run(
        self,
        config: Config,
        prompt: str,
        monitor: Optional["PerformanceMonitor"] = None,
        step_callback: Optional[StepCallback] = None,
    ) -> "InstrumentationResults":
        """
        Run instrumented inference for the given prompt.

        Args:
            config: CoreVital config (model, generation, capture, etc.).
            prompt: Input prompt text.
            monitor: Optional performance monitor for timing.
            step_callback: Optional real-time intervention (Seq2Seq only). StepCallback: return True to stop.

        Returns:
            InstrumentationResults with timeline, token ids, and backend-specific
            data (e.g. hidden_states/attentions when supported).
        """
        ...

    def capabilities(self) -> BackendCapabilities:
        """
        Declare what this backend provides (e.g. vLLM may not expose per-layer hidden_states).

        Default: full capabilities (hidden_states, attentions, prompt_telemetry).
        Override in subclasses to reflect actual support.
        """
        return BackendCapabilities()

# ============================================================================
# CoreVital - Instrumentation Collector
#
# Purpose: Orchestrate model inference with instrumentation and data collection
# Inputs: Config, prompt text
# Outputs: InstrumentationResults with captured data
# Dependencies: torch, transformers, models, summaries, config
# Usage: collector = InstrumentationCollector(config); results = collector.run(prompt)
#
# Phase 1.3 Part B: Slimmed to orchestrator (~250 lines). Heavy logic delegated to:
#   - causal_lm.py   (CausalLM generation path)
#   - seq2seq.py      (Seq2Seq generation path)
#   - baselines.py    (warmup, baseline, prompt forward, shared helpers)
#   - step_processor.py (tensor→summary lifecycle)
# ============================================================================

import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from CoreVital.backends.base import Backend

import torch

from CoreVital.config import Config
from CoreVital.errors import InstrumentationError
from CoreVital.logging_utils import get_logger
from CoreVital.models.hf_loader import ModelBundle, load_model

from .baselines import run_baseline, run_prompt_forward, run_warmup
from .causal_lm import run_causal_generation
from .seq2seq import run_seq2seq_generation
from .step_processor import StepSummary

if TYPE_CHECKING:
    from CoreVital.instrumentation.performance import PerformanceMonitor


logger = get_logger(__name__)

# When seed is set, CausalLM path may still use global RNG if HF generate() does not
# honor the generator kwarg. This lock ensures single-flight for reproducibility (Issue 47).
_generation_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Backward-compatible re-exports of helpers moved to baselines.py
# ---------------------------------------------------------------------------
from .baselines import (  # noqa: E402, F401
    _build_logits_processor,
    _normalize_eos,
    _resolve_special_token,
)


@dataclass
class StepData:
    """Data captured for a single generation step.

    Deprecated after Phase 1.3 Part B — timeline now holds StepSummary objects.
    Kept for backward-compatible imports; new code should use StepSummary.
    """

    step_index: int
    token_id: int
    token_text: str
    is_prompt_token: bool
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[List[torch.Tensor]] = None
    attentions: Optional[List[torch.Tensor]] = None
    cross_attentions: Optional[List[torch.Tensor]] = None


@dataclass
class PromptForwardData:
    """Data captured from the prompt-only forward pass (Phase-1b).

    For CausalLM: comes from model(input_ids) before generate().
    For Seq2Seq: reuses encoder outputs (zero-cost).
    All tensors are kept on CPU to avoid holding GPU memory.
    """

    hidden_states: Optional[List[torch.Tensor]] = None
    attentions: Optional[List[torch.Tensor]] = None
    logits: Optional[torch.Tensor] = None
    prompt_token_ids: Optional[List[int]] = None


@dataclass
class InstrumentationResults:
    """Complete results from an instrumented run."""

    model_bundle: ModelBundle
    prompt_text: str
    prompt_token_ids: List[int]
    generated_token_ids: List[int]
    generated_text: str
    timeline: List[StepSummary] = field(default_factory=list)
    elapsed_ms: int = 0
    warnings: List[Dict[str, str]] = field(default_factory=list)
    encoder_hidden_states: Optional[List[torch.Tensor]] = None
    encoder_attentions: Optional[List[torch.Tensor]] = None
    prompt_forward: Optional[PromptForwardData] = None
    monitor: Optional["PerformanceMonitor"] = None


class InstrumentationCollector:
    """
    Main collector that orchestrates instrumented inference.

    This class loads the model, dispatches generation to the appropriate
    path (CausalLM or Seq2Seq), and collects all data for report generation.

    Thread safety (Issue 41): This class is NOT thread-safe. For concurrent use
    in a server context, create one InstrumentationCollector per request or use
    a model pool with locks. The cached model_bundle shares PyTorch model state
    that is mutated during inference.
    """

    def __init__(self, config: Config, backend: Optional["Backend"] = None):
        self.config = config
        self._backend = backend
        self.model_bundle: Optional[ModelBundle] = None

    def run(
        self,
        prompt: str,
        monitor: Optional["PerformanceMonitor"] = None,
        step_callback: Optional[
            Callable[
                [int, List[int], Optional[List[torch.Tensor]], Optional[torch.Tensor]],
                bool,
            ]
        ] = None,
    ) -> InstrumentationResults:
        """Run instrumented inference on the given prompt.

        If a backend was provided at construction, delegates to backend.run().
        Otherwise runs the built-in Hugging Face implementation.

        Args:
            prompt: Input prompt text
            monitor: Optional PerformanceMonitor passed from CLI
            step_callback: Optional callback for real-time intervention (Seq2Seq only).

        Returns:
            InstrumentationResults with all collected data
        """
        if self._backend is not None:
            return self._backend.run(self.config, prompt, monitor, step_callback=step_callback)
        return self._run_impl(prompt, monitor, step_callback=step_callback)

    def _run_impl(
        self,
        prompt: str,
        monitor: Optional["PerformanceMonitor"] = None,
        step_callback: Optional[
            Callable[
                [int, List[int], Optional[List[torch.Tensor]], Optional[torch.Tensor]],
                bool,
            ]
        ] = None,
    ) -> InstrumentationResults:
        """Built-in Hugging Face instrumented run."""
        try:

            def _op(name: str):
                return monitor.operation(name) if monitor else nullcontext()

            is_seq2seq = False

            # === STRICT MODE: Warmup and Baseline BEFORE the main run ===
            if monitor and monitor.mode == "strict":
                if self.model_bundle is None:
                    _model_load_start = time.perf_counter()
                    self.model_bundle = load_model(self.config, monitor=None)
                    _original_model_load_ms = (time.perf_counter() - _model_load_start) * 1000
                    monitor.set_original_model_load_ms(_original_model_load_ms)
                    logger.debug(f"Perf: original model load time: {_original_model_load_ms:.2f}ms")

                is_seq2seq = self.model_bundle.capabilities.is_seq2seq

                inputs = self.model_bundle.tokenizer(prompt, return_tensors="pt", padding=False).to(
                    self.model_bundle.device
                )
                assert inputs.input_ids.shape[0] == 1, "CoreVital only supports batch_size=1"

                logger.debug("Perf: running warmup (2 rounds)...")
                warmup_start = time.perf_counter()
                run_warmup(self.model_bundle, self.config, inputs, is_seq2seq)
                run_warmup(self.model_bundle, self.config, inputs, is_seq2seq)
                warmup_ms = (time.perf_counter() - warmup_start) * 1000
                monitor.set_warmup_ms(warmup_ms)
                logger.debug(f"Perf: warmup done ({warmup_ms:.2f}ms)")

                seed = self.config.generation.seed
                generator = None
                if seed is not None:
                    generator = torch.Generator(device=self.model_bundle.device).manual_seed(seed)
                logger.debug("Perf: running baseline...")
                with _generation_lock if seed is not None else nullcontext():
                    baseline_ms = run_baseline(self.model_bundle, self.config, inputs, is_seq2seq, generator=generator)
                monitor.set_baseline_ms(baseline_ms)
                logger.debug(f"Perf: baseline done ({baseline_ms:.2f}ms)")

            # === Model load ===
            with _op("model_load"):
                if self.model_bundle is None:
                    self.model_bundle = load_model(self.config, monitor=monitor)

            # Per-request generator for reproducibility (Issue 47); no global torch.manual_seed
            seed = self.config.generation.seed
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.model_bundle.device).manual_seed(seed)

            # === Tokenize ===
            with _op("tokenize"):
                logger.debug("Tokenizing prompt...")
                inputs = self.model_bundle.tokenizer(prompt, return_tensors="pt", padding=False).to(
                    self.model_bundle.device
                )
                assert inputs.input_ids.shape[0] == 1, "CoreVital only supports batch_size=1"
                prompt_token_ids = inputs.input_ids[0].tolist()
                logger.debug(f"Prompt tokens: {len(prompt_token_ids)}")

            # === Prompt length vs context window (Issue 42) ===
            max_len = getattr(self.model_bundle.model.config, "max_position_embeddings", None)
            if isinstance(max_len, int) and len(prompt_token_ids) >= max_len:
                raise InstrumentationError(
                    f"Prompt ({len(prompt_token_ids)} tokens) exceeds model context ({max_len} max_position_embeddings)"
                )

            is_seq2seq = self.model_bundle.capabilities.is_seq2seq

            # === Prompt forward pass (CausalLM only) ===
            prompt_forward: Optional[PromptForwardData] = None
            if self.config.prompt_telemetry.enabled and not is_seq2seq:
                with _op("prompt_forward_pass"):
                    prompt_forward = run_prompt_forward(
                        self.model_bundle, self.config, inputs, prompt_token_ids, monitor=monitor
                    )

            # === Model inference ===
            logger.info("Starting instrumented generation...")
            start_time = time.time()

            generated_token_ids: List[int] = []
            generated_text: str = ""
            timeline: List[StepSummary] = []
            encoder_hidden_states = None
            encoder_attentions = None
            warnings: List[Dict[str, str]] = []

            with _op("model_inference"):
                with _generation_lock if seed is not None else nullcontext():
                    with torch.no_grad():
                        if is_seq2seq:
                            logger.info("Using manual generation for Seq2Seq model")
                            with _op("_generate_seq2seq_manual"):
                                seq_result = run_seq2seq_generation(
                                    self.model_bundle,
                                    self.config,
                                    inputs,
                                    prompt_token_ids,
                                    monitor=monitor,
                                    step_callback=step_callback,
                                    generator=generator,
                                )
                            generated_token_ids = seq_result.generated_token_ids
                            generated_text = seq_result.generated_text
                            timeline = seq_result.timeline
                            warnings = seq_result.warnings
                            encoder_hidden_states = seq_result.encoder_hidden_states
                            encoder_attentions = seq_result.encoder_attentions
                            if seq_result.prompt_forward is not None:
                                prompt_forward = seq_result.prompt_forward
                        else:
                            causal_result = run_causal_generation(
                                self.model_bundle,
                                self.config,
                                inputs,
                                prompt_token_ids,
                                monitor=monitor,
                                generator=generator,
                                seed=seed,
                            )
                            generated_token_ids = causal_result.generated_token_ids
                            generated_text = causal_result.generated_text
                            timeline = causal_result.timeline
                            warnings = causal_result.warnings

                    logger.debug(f"Generated tokens: {len(generated_token_ids)}")
                    logger.debug(f"Generated text: {generated_text}")

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Generation complete in {elapsed_ms}ms")

            if monitor:
                for p in monitor._parent_operations():
                    if p.operation_name == "model_inference":
                        monitor.set_instrumented_inference_ms(p.duration_ms)
                        break

            return InstrumentationResults(
                model_bundle=self.model_bundle,
                prompt_text=prompt,
                prompt_token_ids=prompt_token_ids,
                generated_token_ids=generated_token_ids,
                generated_text=generated_text,
                timeline=timeline,
                elapsed_ms=elapsed_ms,
                warnings=warnings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attentions=encoder_attentions,
                prompt_forward=prompt_forward,
                monitor=monitor,
            )

        except Exception as e:
            logger.exception("Instrumentation failed")
            raise InstrumentationError("Failed during instrumented inference", details=str(e)) from e
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ============================================================================
# Test Harness
# ============================================================================


def _test_collector():
    """Test harness for instrumentation collector."""
    print("Testing InstrumentationCollector...")

    config = Config()
    config.model.hf_id = "gpt2"
    config.device.requested = "cpu"
    config.generation.max_new_tokens = 5

    collector = InstrumentationCollector(config)
    results = collector.run("Hello")

    print("✓ Collected results:")
    print(f"  Prompt tokens: {len(results.prompt_token_ids)}")
    print(f"  Generated tokens: {len(results.generated_token_ids)}")
    print(f"  Timeline steps: {len(results.timeline)}")
    print(f"  Elapsed: {results.elapsed_ms}ms")
    print(f"  Warnings: {len(results.warnings)}")

    assert len(results.timeline) > 0
    print("✓ All collector tests passed!\n")


if __name__ == "__main__":
    _test_collector()

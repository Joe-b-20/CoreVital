# ============================================================================
# CoreVital - Instrumentation Collector
#
# Purpose: Orchestrate model inference with instrumentation and data collection
# Inputs: Config, prompt text
# Outputs: InstrumentationResults with captured data
# Dependencies: torch, transformers, models, summaries, config
# Usage: collector = InstrumentationCollector(config); results = collector.run(prompt)
#
# Changelog:
#   2026-01-13: Initial collector for Phase-0
#   2026-01-14: Fixed logits extraction from generation outputs (added output_scores=True)
#                Fixed hidden_states and attentions extraction (handle tuple-of-tuples structure)
#                Added diagnostic logging for attention extraction
#   2026-01-15: Added manual generation for Seq2Seq models (T5, BART, etc.)
#                Seq2Seq models don't return hidden_states/attentions via generate(),
#                so we manually step through the decoder to capture these outputs
#   2026-01-15: Added deep Seq2Seq instrumentation - extract encoder_hidden_states, encoder_attentions,
#                and cross_attentions from decoder outputs at each generation step
#   2026-01-21: Phase-0.5 hardening - robust Seq2Seq detection with Mock object handling;
#                memory optimization: slice decoder self-attention to last query token;
#                standardized logging to DEBUG for tensor operations
#   2026-02-04: Phase-0.75 - added PerformanceMonitor integration via _op() helper;
#                strict mode: warmup, baseline, and original model load tracking;
#                all parent and child operations wrapped for timing
#   2026-02-06: Updated note: total_wall_time_ms covers through report_build (not sink_write)
#   2026-02-06: Phase-0.75 - fixed strict mode: seed before baseline for reproducible
#                token generation; added top_k/top_p to baseline Seq2Seq sampling to
#                match instrumented path; doubled warmup rounds for cache stabilization
#   2026-02-10: Phase-1b — Prompt telemetry:
#                - Added PromptForwardData dataclass for prompt forward pass results
#                - CausalLM: model(input_ids) before generate() captures hidden states,
#                  attentions, logits for prompt tokens
#                - Seq2Seq: reuses existing encoder outputs (zero-cost forward pass)
#                - --no-prompt-telemetry CLI flag to skip prompt analysis
# ============================================================================

import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, cast

if TYPE_CHECKING:
    from CoreVital.backends.base import Backend

import torch

from CoreVital.config import Config
from CoreVital.errors import InstrumentationError
from CoreVital.logging_utils import get_logger
from CoreVital.models.hf_loader import ModelBundle, load_model

if TYPE_CHECKING:
    from CoreVital.instrumentation.performance import PerformanceMonitor


logger = get_logger(__name__)


@dataclass
class StepData:
    """Data captured for a single generation step."""

    step_index: int
    token_id: int
    token_text: str
    is_prompt_token: bool
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[List[torch.Tensor]] = None
    attentions: Optional[List[torch.Tensor]] = None
    cross_attentions: Optional[List[torch.Tensor]] = None  # For Seq2Seq models


@dataclass
class PromptForwardData:
    """Data captured from the prompt-only forward pass (Phase-1b).

    For CausalLM: comes from model(input_ids) before generate().
    For Seq2Seq: reuses encoder outputs (zero-cost).

    All tensors are kept on CPU to avoid holding GPU memory.
    """

    hidden_states: Optional[List[torch.Tensor]] = None  # Per-layer, (batch, seq_len, hidden_dim)
    attentions: Optional[List[torch.Tensor]] = None  # Per-layer, (batch, heads, seq_len, seq_len)
    logits: Optional[torch.Tensor] = None  # (batch, seq_len, vocab_size) — CausalLM only
    prompt_token_ids: Optional[List[int]] = None  # Token IDs for surprisal alignment


@dataclass
class InstrumentationResults:
    """Complete results from an instrumented run."""

    model_bundle: ModelBundle
    prompt_text: str
    prompt_token_ids: List[int]
    generated_token_ids: List[int]
    generated_text: str
    timeline: List[StepData] = field(default_factory=list)
    elapsed_ms: int = 0
    warnings: List[Dict[str, str]] = field(default_factory=list)
    # Seq2Seq-specific fields
    encoder_hidden_states: Optional[List[torch.Tensor]] = None
    encoder_attentions: Optional[List[torch.Tensor]] = None
    # Phase-1b: prompt forward pass data
    prompt_forward: Optional[PromptForwardData] = None
    # Performance monitoring (optional)
    monitor: Optional["PerformanceMonitor"] = None


class InstrumentationCollector:
    """
    Main collector that orchestrates instrumented inference.

    This class loads the model, runs generation with full instrumentation,
    and collects all necessary data for report generation.
    """

    def __init__(self, config: Config, backend: Optional["Backend"] = None):
        """
        Initialize collector with configuration and optional backend.

        Args:
            config: Configuration object
            backend: If set, run() delegates to this backend; otherwise uses
                     built-in Hugging Face implementation (_run_impl).
        """
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
        """
        Run instrumented inference on the given prompt.

        If a backend was provided at construction, delegates to backend.run().
        Otherwise runs the built-in Hugging Face implementation.

        Args:
            prompt: Input prompt text
            monitor: Optional PerformanceMonitor passed from CLI
            step_callback: Optional callback for real-time intervention (Seq2Seq only).
                Signature: (step_index, generated_token_ids, last_layer_hidden_buffer, last_logits) -> bool.
                If it returns True, generation stops. last_layer_hidden_buffer is the last N steps'
                last-layer hidden state (for e.g. detect_repetition_loop). Ignored for CausalLM.

        Returns:
            InstrumentationResults with all collected data

        Raises:
            InstrumentationError: If inference fails
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
        """Built-in Hugging Face instrumented run. Used when no external backend is set."""
        try:

            def _op(name: str):
                """Helper to wrap operations in monitor.operation() if enabled."""
                return monitor.operation(name) if monitor else nullcontext()

            # === STRICT MODE: Warmup and Baseline BEFORE the main instrumented run ===
            # Model must be loaded first, then warmup/baseline, then instrumented run
            is_seq2seq = False  # Will be determined after model load

            if monitor and monitor.mode == "strict":
                # Load model (needed for warmup/baseline) and record original load time
                if self.model_bundle is None:
                    _model_load_start = time.perf_counter()
                    self.model_bundle = load_model(self.config, monitor=None)
                    _original_model_load_ms = (time.perf_counter() - _model_load_start) * 1000
                    monitor.set_original_model_load_ms(_original_model_load_ms)
                    logger.debug(f"Perf: original model load time: {_original_model_load_ms:.2f}ms")

                is_seq2seq = self.model_bundle.capabilities.is_seq2seq

                # Tokenize for warmup/baseline
                inputs = self.model_bundle.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,
                ).to(self.model_bundle.device)

                # === WARMUP: Two dummy runs to stabilize timings (NOT counted in total) ===
                # Two runs ensure CPU caches, branch predictors, and JIT are fully warm
                # before we measure baseline. Without this, baseline may appear slower
                # than the instrumented run due to cache warming effects.
                logger.debug("Perf: running warmup (2 rounds)...")
                warmup_start = time.perf_counter()
                self._run_warmup(inputs, is_seq2seq)
                self._run_warmup(inputs, is_seq2seq)
                warmup_ms = (time.perf_counter() - warmup_start) * 1000
                monitor.set_warmup_ms(warmup_ms)
                logger.debug(f"Perf: warmup done ({warmup_ms:.2f}ms)")

                # === BASELINE: Raw inference without instrumentation (NOT counted in total) ===
                # Seed before baseline so it generates the same tokens as the instrumented run.
                # Without this, different random sampling could produce different token counts,
                # making the timing comparison meaningless.
                if self.config.generation.seed is not None:
                    torch.manual_seed(self.config.generation.seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(self.config.generation.seed)
                logger.debug("Perf: running baseline...")
                baseline_ms = self._run_baseline(inputs, is_seq2seq)
                monitor.set_baseline_ms(baseline_ms)
                logger.debug(f"Perf: baseline done ({baseline_ms:.2f}ms)")

            # === PARENT: model_load ===
            # For strict mode, model is already loaded; for other modes, load now
            with _op("model_load"):
                if self.model_bundle is None:
                    self.model_bundle = load_model(self.config, monitor=monitor)
                # For strict mode with cached model, the operation still tracks but duration is minimal

            # === PARENT: torch.manual_seed ===
            with _op("torch.manual_seed"):
                if self.config.generation.seed is not None:
                    torch.manual_seed(self.config.generation.seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(self.config.generation.seed)

            # === PARENT: tokenize ===
            with _op("tokenize"):
                logger.debug("Tokenizing prompt...")
                inputs = self.model_bundle.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,
                ).to(self.model_bundle.device)

                prompt_token_ids = inputs.input_ids[0].tolist()
                logger.debug(f"Prompt tokens: {len(prompt_token_ids)}")

            # Model type from capabilities (single source of truth — no detection here)
            is_seq2seq = self.model_bundle.capabilities.is_seq2seq

            # === PARENT: prompt_forward_pass (Phase-1b) ===
            # For CausalLM: run model(input_ids) to capture prompt hidden states, attentions, logits.
            # For Seq2Seq: encoder outputs are captured during generation (reused, zero-cost).
            prompt_forward: Optional[PromptForwardData] = None
            if self.config.prompt_telemetry.enabled and not is_seq2seq:
                with _op("prompt_forward_pass"):
                    prompt_forward = self._run_prompt_forward(inputs, prompt_token_ids, monitor=monitor)

            # === PARENT: model_inference ===
            # Contains: generation + extract_generated_tokens + decode_generated_text + _process_timeline
            logger.info("Starting instrumented generation...")
            start_time = time.time()

            # Declare variables that will be set inside model_inference
            generated_token_ids: List[int] = []
            generated_text: str = ""
            timeline: List[StepData] = []
            encoder_hidden_states = None
            encoder_attentions = None
            warnings: List[Dict[str, str]] = []
            outputs: Any = None

            with _op("model_inference"):
                with torch.no_grad():
                    if is_seq2seq:
                        logger.info("Using manual generation for Seq2Seq model to capture hidden states and attentions")
                        # Tracked as a child of model_inference (corevital orchestration)
                        with _op("_generate_seq2seq_manual"):
                            outputs = self._generate_seq2seq_manual(
                                inputs,
                                prompt_token_ids,
                                monitor=monitor,
                                step_callback=step_callback,
                            )
                    else:
                        # Prepare generation config for CausalLM models
                        num_beams = getattr(self.config.generation, "num_beams", 1) or 1
                        gen_config: Dict[str, Any] = {
                            "max_new_tokens": self.config.generation.max_new_tokens,
                            "do_sample": self.config.generation.do_sample,
                            "temperature": self.config.generation.temperature,
                            "top_k": self.config.generation.top_k,
                            "top_p": self.config.generation.top_p,
                            "output_hidden_states": True,
                            "output_attentions": True,
                            "output_scores": True,  # Enable logits extraction
                            "return_dict_in_generate": True,
                            "pad_token_id": self.model_bundle.tokenizer.pad_token_id,
                        }
                        if num_beams > 1:
                            gen_config["num_beams"] = num_beams
                            gen_config["early_stopping"] = getattr(self.config.generation, "early_stopping", False)
                            # Standard beam search uses do_sample=False
                            gen_config["do_sample"] = False
                        # Tracked as a child of model_inference (external HF call)
                        with _op("model.generate"):
                            outputs = cast(Any, self.model_bundle.model).generate(
                                **inputs,
                                **gen_config,
                            )

                # === CHILD: extract_generated_tokens ===
                with _op("extract_generated_tokens"):
                    if is_seq2seq:
                        generated_token_ids = cast(List[int], outputs["generated_token_ids"])
                        generated_ids = prompt_token_ids + generated_token_ids
                    else:
                        # Type narrowing: outputs is GenerateOutput here, not dict
                        out_seq = cast(Any, outputs).sequences
                        # With beam search, sequences can be (batch, num_beams, seq_len); take best beam
                        if out_seq.dim() == 3:
                            generated_ids = out_seq[0, 0, :].tolist()
                        else:
                            generated_ids = out_seq[0].tolist()
                        generated_token_ids = generated_ids[len(prompt_token_ids) :]

                # === CHILD: decode_generated_text ===
                with _op("decode_generated_text"):
                    generated_text = cast(
                        str,
                        self.model_bundle.tokenizer.decode(
                            generated_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                        ),
                    )
                logger.debug(f"Generated tokens: {len(generated_token_ids)}")
                logger.debug(f"Generated text: {generated_text}")

                # === CHILD: extract encoder data + build Seq2Seq prompt_forward ===
                # Handle both standard generate() output and manual Seq2Seq output
                if isinstance(outputs, dict) and "output_obj" in outputs:
                    encoder_hidden_states = outputs.get("encoder_hidden_states")
                    encoder_attentions = outputs.get("encoder_attentions")
                    if encoder_hidden_states is not None and isinstance(encoder_hidden_states, tuple):
                        encoder_hidden_states = list(encoder_hidden_states)
                    if encoder_attentions is not None and isinstance(encoder_attentions, tuple):
                        encoder_attentions = list(encoder_attentions)

                    # Seq2Seq encoder reuse: encoder outputs ARE the prompt analysis (zero-cost)
                    if is_seq2seq and self.config.prompt_telemetry.enabled:
                        prompt_forward = PromptForwardData(
                            hidden_states=encoder_hidden_states,
                            attentions=encoder_attentions,
                            logits=None,  # Encoders don't produce token-prediction logits
                            prompt_token_ids=prompt_token_ids,
                        )
                        logger.debug(
                            f"Seq2Seq prompt telemetry: reusing encoder outputs "
                            f"({len(encoder_hidden_states) if encoder_hidden_states else 0} layers)"
                        )

                with _op("_process_timeline"):
                    beam_indices = getattr(outputs, "beam_indices", None) if not isinstance(outputs, dict) else None
                    num_beams_val = getattr(self.config.generation, "num_beams", 1) or 1 if not is_seq2seq else 1
                    if isinstance(outputs, dict) and "output_obj" in outputs:
                        timeline = self._process_timeline(
                            outputs["output_obj"],
                            prompt_token_ids,
                            generated_token_ids,
                            cross_attentions=outputs.get("cross_attentions"),
                            beam_indices=beam_indices,
                            num_beams=num_beams_val,
                        )
                    else:
                        timeline = self._process_timeline(
                            outputs,
                            prompt_token_ids,
                            generated_token_ids,
                            beam_indices=beam_indices,
                            num_beams=num_beams_val,
                        )

                # === CHILD: _collect_warnings ===
                with _op("_collect_warnings"):
                    if isinstance(outputs, dict) and "output_obj" in outputs:
                        warnings = self._collect_warnings(outputs["output_obj"])
                    else:
                        warnings = self._collect_warnings(outputs)

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Generation complete in {elapsed_ms}ms")

            # Record instrumented inference time (for strict mode overhead calculation)
            if monitor:
                for p in monitor._parent_operations():
                    if p.operation_name == "model_inference":
                        monitor.set_instrumented_inference_ms(p.duration_ms)
                        break

            # Note: total_wall_time_ms is set by CLI after all timed operations (through report_build)

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

    def _run_warmup(self, inputs: Any, is_seq2seq: bool) -> None:
        """Run a warmup generation (no instrumentation, results discarded)."""
        if self.model_bundle is None:
            return
        with torch.no_grad():
            if is_seq2seq:
                self._run_baseline_seq2seq(inputs)
            else:
                self._run_baseline_causal(inputs)

    def _run_baseline(self, inputs: Any, is_seq2seq: bool) -> float:
        """Run baseline inference (no instrumentation) and return elapsed ms."""
        if self.model_bundle is None:
            return 0.0
        start = time.perf_counter()
        with torch.no_grad():
            if is_seq2seq:
                self._run_baseline_seq2seq(inputs)
            else:
                self._run_baseline_causal(inputs)
        return (time.perf_counter() - start) * 1000

    def _run_baseline_causal(self, inputs: Any) -> None:
        """Baseline CausalLM generation (no output_hidden_states/attentions/scores)."""
        if self.model_bundle is None:
            return
        num_beams = getattr(self.config.generation, "num_beams", 1) or 1
        gen_config: Dict[str, Any] = {
            "max_new_tokens": self.config.generation.max_new_tokens,
            "do_sample": self.config.generation.do_sample,
            "temperature": self.config.generation.temperature,
            "top_k": self.config.generation.top_k,
            "top_p": self.config.generation.top_p,
            "pad_token_id": self.model_bundle.tokenizer.pad_token_id,
        }
        if num_beams > 1:
            gen_config["num_beams"] = num_beams
            gen_config["early_stopping"] = getattr(self.config.generation, "early_stopping", False)
            gen_config["do_sample"] = False
        cast(Any, self.model_bundle.model).generate(**inputs, **gen_config)

    def _run_baseline_seq2seq(self, inputs: Any) -> None:
        """
        Baseline Seq2Seq generation: encoder + decoder loop with no hidden_states/attentions.

        Sampling logic (temperature, top_k, top_p) must match _generate_seq2seq_manual
        exactly so that with the same seed, baseline and instrumented runs generate the
        same tokens. Without this, different token counts make overhead comparison meaningless.
        """
        if self.model_bundle is None:
            return
        model = cast(Any, self.model_bundle.model)
        tokenizer = self.model_bundle.tokenizer
        device = self.model_bundle.device

        # Encoder pass (no hidden_states/attentions)
        encoder_outputs = model.encoder(
            input_ids=inputs.input_ids,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )

        # Decoder loop
        decoder_start_token_id = getattr(model.config, "decoder_start_token_id", None)
        if decoder_start_token_id is None:
            decoder_start_token_id = getattr(tokenizer, "pad_token_id", 0)
        eos_token_id = getattr(model.config, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(tokenizer, "eos_token_id", 2)

        decoder_input_ids = torch.tensor([[decoder_start_token_id]], device=device)
        max_new_tokens = self.config.generation.max_new_tokens

        # Generation parameters (must match _generate_seq2seq_manual)
        do_sample = self.config.generation.do_sample
        temperature = self.config.generation.temperature
        top_k = self.config.generation.top_k
        top_p = self.config.generation.top_p

        for _ in range(max_new_tokens):
            decoder_outputs = model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=False,
                output_attentions=False,
                use_cache=False,
                return_dict=True,
            )
            next_token_logits = decoder_outputs.logits[:, -1, :]
            # Sample next token (must match _generate_seq2seq_manual exactly)
            if do_sample:
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float("-inf")

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float("-inf")

                # Sample from filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1)
            if next_token_id.item() == eos_token_id:
                break

    def _run_prompt_forward(
        self,
        inputs: Any,
        prompt_token_ids: List[int],
        monitor: Optional["PerformanceMonitor"] = None,
    ) -> PromptForwardData:
        """Run a single forward pass on prompt tokens to capture hidden states, attentions, and logits.

        CausalLM only. For Seq2Seq, encoder outputs are reused instead.

        Args:
            inputs: Tokenized input from tokenizer
            prompt_token_ids: Prompt token IDs
            monitor: Optional performance monitor

        Returns:
            PromptForwardData with hidden states, attentions, and logits (all on CPU)
        """
        if self.model_bundle is None:
            raise InstrumentationError("Model bundle not initialized")

        model = cast(Any, self.model_bundle.model)

        logger.debug(f"Running prompt forward pass ({len(prompt_token_ids)} tokens)...")
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )

        # Extract hidden states: skip embedding (index 0), keep layer outputs
        hidden_states = None
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            try:
                hs = outputs.hidden_states
                # outputs.hidden_states = (embedding, layer1, ..., layerN)
                if isinstance(hs, (tuple, list)) and len(hs) > 1:
                    hidden_states = [h.cpu() for h in hs[1:]]
                elif isinstance(hs, (tuple, list)):
                    hidden_states = [h.cpu() for h in hs]
                logger.debug(
                    f"Prompt forward: extracted {len(hidden_states) if hidden_states else 0} hidden state layers"
                )
            except (TypeError, AttributeError) as e:
                logger.debug(f"Prompt forward: could not extract hidden states: {e}")

        # Extract attentions: one tensor per layer, each (batch, heads, seq, seq)
        attentions = None
        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            try:
                att = outputs.attentions
                if isinstance(att, (tuple, list)):
                    attentions = [a.cpu() for a in att]
                logger.debug(f"Prompt forward: extracted {len(attentions) if attentions else 0} attention layers")
            except (TypeError, AttributeError) as e:
                logger.debug(f"Prompt forward: could not extract attentions: {e}")

        # Extract logits: (batch, seq_len, vocab_size) — needed for prompt surprisal
        logits = None
        if hasattr(outputs, "logits") and outputs.logits is not None:
            try:
                logits = outputs.logits.cpu()
                logger.debug(f"Prompt forward: logits shape {logits.shape}")
            except (TypeError, AttributeError) as e:
                logger.debug(f"Prompt forward: could not extract logits: {e}")

        return PromptForwardData(
            hidden_states=hidden_states,
            attentions=attentions,
            logits=logits,
            prompt_token_ids=prompt_token_ids,
        )

    def _generate_seq2seq_manual(
        self,
        inputs: Any,
        prompt_token_ids: List[int],
        monitor: Optional["PerformanceMonitor"] = None,
        step_callback: Optional[
            Callable[
                [int, List[int], Optional[List[torch.Tensor]], Optional[torch.Tensor]],
                bool,
            ]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Manual generation for Seq2Seq models to capture hidden states and attentions.

        Seq2Seq models (T5, BART, etc.) don't return hidden_states/attentions via
        generate(), so we need to manually step through the decoder.

        Args:
            inputs: Tokenized input from tokenizer
            prompt_token_ids: Prompt token IDs
            monitor: Optional performance monitor for nested timing
            step_callback: Optional. If it returns True after a step, generation stops (real-time intervention).

        Returns:
            Dictionary with sequences, scores, hidden_states, and attentions
        """
        if self.model_bundle is None:
            raise InstrumentationError("Model bundle not initialized")

        def _op(name: str):
            """Helper to wrap operations in monitor.operation() if enabled."""
            return monitor.operation(name) if monitor else nullcontext()

        model = cast(Any, self.model_bundle.model)
        tokenizer = self.model_bundle.tokenizer
        device = self.model_bundle.device

        # Encode input with encoder
        logger.debug("Encoding input with encoder...")
        # tracked as a child operation of model_inference
        # encoder_forward is child of _generate_seq2seq_manual which is child of model_inference
        with _op("encoder_forward"):
            encoder_outputs = model.encoder(
                input_ids=inputs.input_ids,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )

        # Extract encoder hidden states and attentions (computed once, fixed for the entire run)
        encoder_hidden_states = None
        encoder_attentions = None
        if hasattr(encoder_outputs, "hidden_states") and encoder_outputs.hidden_states is not None:
            # Skip the first element (embedding) and take layer outputs
            # Convert tuple to list for consistency with type hints
            hidden_states_tuple = encoder_outputs.hidden_states[1:] if len(encoder_outputs.hidden_states) > 1 else []
            encoder_hidden_states = (
                list(hidden_states_tuple) if isinstance(hidden_states_tuple, tuple) else hidden_states_tuple
            )
            logger.debug(f"Extracted {len(encoder_hidden_states)} encoder hidden state layers")

        if hasattr(encoder_outputs, "attentions") and encoder_outputs.attentions is not None:
            # Convert tuple to list for consistency with type hints
            encoder_attentions = (
                list(encoder_outputs.attentions)
                if isinstance(encoder_outputs.attentions, tuple)
                else encoder_outputs.attentions
            )
            logger.debug(f"Extracted {len(encoder_attentions)} encoder attention layers")

        # Prepare decoder inputs
        # For T5, decoder_start_token_id is typically pad_token_id
        decoder_start_token_id = getattr(model.config, "decoder_start_token_id", None)
        if decoder_start_token_id is None:
            decoder_start_token_id = tokenizer.pad_token_id
            if decoder_start_token_id is None:
                decoder_start_token_id = tokenizer.eos_token_id

        decoder_input_ids = torch.tensor([[decoder_start_token_id]], device=device, dtype=torch.long)

        # Storage for outputs
        all_scores: List[torch.Tensor] = []
        all_hidden_states: List[Optional[tuple[torch.Tensor, ...]]] = []
        all_attentions: List[Optional[tuple[torch.Tensor, ...]]] = []  # Decoder self-attentions
        # Cross-attentions (decoder attending to encoder)
        all_cross_attentions: List[Optional[tuple[torch.Tensor, ...]]] = []
        generated_token_ids: List[int] = []

        max_new_tokens = self.config.generation.max_new_tokens
        eos_token_id = tokenizer.eos_token_id or model.config.eos_token_id

        # Generation parameters
        do_sample = self.config.generation.do_sample
        temperature = self.config.generation.temperature
        top_k = self.config.generation.top_k
        top_p = self.config.generation.top_p

        logger.debug(f"Starting manual decoder generation (max_new_tokens={max_new_tokens})...")

        # Track decoder loop timing and per-step times
        _decoder_loop_start = time.perf_counter()
        _step_times_ms: List[float] = []

        for step in range(max_new_tokens):
            _step_start = time.perf_counter()
            # Forward pass through decoder
            # For T5 and other Seq2Seq models, we pass encoder_outputs as a BaseModelOutput
            decoder_outputs = model(
                input_ids=None,  # Not needed when using encoder_outputs
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=False,  # Disable cache to get all hidden states
                return_dict=True,
            )

            # Extract logits for next token prediction
            next_token_logits = decoder_outputs.logits[:, -1, :]  # Shape: (batch_size, vocab_size)
            all_scores.append(next_token_logits)

            # Extract hidden states (decoder hidden states)
            # decoder_hidden_states is a tuple: (embedding, layer1, layer2, ..., layerN)
            # Each element is a tensor of shape (batch_size, seq_len, hidden_size)
            if hasattr(decoder_outputs, "decoder_hidden_states") and decoder_outputs.decoder_hidden_states is not None:
                step_hidden = []
                # Skip the first element (embedding) and extract last position from each layer
                for layer_idx, layer_hidden in enumerate(decoder_outputs.decoder_hidden_states[1:], start=1):
                    if layer_hidden is not None:
                        # Extract the last position: [:, -1, :] -> (batch_size, hidden_size)
                        # But we need to keep it as a 3D tensor for consistency: (batch_size, 1, hidden_size)
                        last_pos = layer_hidden[:, -1:, :]  # Keep seq_len=1 dimension
                        step_hidden.append(last_pos)
                    else:
                        logger.debug(f"Layer {layer_idx} hidden state is None")
                if step_hidden:
                    all_hidden_states.append(tuple(step_hidden))
                else:
                    all_hidden_states.append(None)
                    logger.warning(f"No hidden states extracted at step {step}")
            else:
                all_hidden_states.append(None)
                logger.debug(f"No decoder_hidden_states available at step {step}")

            # Extract decoder self-attentions
            # decoder_outputs has decoder_attentions (self-attention) and cross_attentions (encoder-decoder)
            # For T5: decoder_attentions is a tuple of tuples, one per layer
            # Each layer has: (self_attn_tensor,) where tensor is (batch_size, num_heads, seq_len, seq_len)
            # Memory optimization: slice to last query token before storage
            step_attentions = []
            if hasattr(decoder_outputs, "decoder_attentions") and decoder_outputs.decoder_attentions is not None:
                for _layer_idx, layer_attn_tuple in enumerate(decoder_outputs.decoder_attentions):
                    if layer_attn_tuple is not None:
                        # layer_attn_tuple is typically a tuple with one element (self-attention)
                        # or could be a tensor directly
                        if isinstance(layer_attn_tuple, tuple) and len(layer_attn_tuple) > 0:
                            # Take the self-attention tensor
                            attn_tensor = layer_attn_tuple[0]
                            if attn_tensor is not None:
                                # Slice to last query token: [:, :, -1:, :] -> (batch_size, num_heads, 1, seq_len)
                                if attn_tensor.dim() == 4:
                                    attn_tensor = attn_tensor[:, :, -1:, :]  # Keep target_len=1 dimension
                                step_attentions.append(attn_tensor)
                        elif isinstance(layer_attn_tuple, torch.Tensor):
                            attn_tensor = layer_attn_tuple
                            # Slice to last query token for memory optimization
                            if attn_tensor.dim() == 4:
                                attn_tensor = attn_tensor[:, :, -1:, :]
                            step_attentions.append(attn_tensor)

            if step_attentions:
                all_attentions.append(tuple(step_attentions))
            else:
                all_attentions.append(None)
                logger.debug(f"No decoder attentions extracted at step {step}")

            # Extract cross-attentions (decoder attending to encoder)
            # cross_attentions shape: (batch_size, num_heads, target_seq_len, source_seq_len)
            # where target_seq_len is decoder sequence length and source_seq_len is encoder sequence length
            step_cross_attentions = []
            if hasattr(decoder_outputs, "cross_attentions") and decoder_outputs.cross_attentions is not None:
                for _layer_idx, cross_attn_tensor in enumerate(decoder_outputs.cross_attentions):
                    if cross_attn_tensor is not None:
                        # Extract the last position for the current step: [:, :, -1, :]
                        # Shape becomes (batch_size, num_heads, source_seq_len)
                        # But we keep it as (batch_size, num_heads, 1, source_seq_len) for consistency
                        if cross_attn_tensor.dim() == 4:
                            last_pos_cross = cross_attn_tensor[:, :, -1:, :]  # Keep target_len=1 dimension
                            step_cross_attentions.append(last_pos_cross)
                        else:
                            step_cross_attentions.append(cross_attn_tensor)

            if step_cross_attentions:
                all_cross_attentions.append(tuple(step_cross_attentions))
            else:
                all_cross_attentions.append(None)
                logger.debug(f"No cross-attentions extracted at step {step}")

            # Sample next token
            if do_sample:
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float("-inf")

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float("-inf")

                # Sample from filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            next_token_id = int(next_token.item())
            generated_token_ids.append(next_token_id)

            # Real-time intervention: step_callback can request early stop (e.g. repetition detected)
            if step_callback is not None:
                buffer: List[torch.Tensor] = []
                for i in range(max(0, step - 4), step + 1):
                    if i < len(all_hidden_states) and all_hidden_states[i] is not None:
                        layers = all_hidden_states[i]
                        if layers and len(layers) > 0:
                            last_layer = layers[-1]
                            if last_layer is not None:
                                vec = last_layer.squeeze()
                                if vec.dim() == 1:
                                    buffer.append(vec)
                try:
                    if step_callback(
                        step,
                        list(generated_token_ids),
                        buffer if buffer else None,
                        next_token_logits,
                    ):
                        logger.info(f"Step callback requested stop at step {step}")
                        _step_times_ms.append((time.perf_counter() - _step_start) * 1000)
                        break
                except Exception as e:
                    logger.warning(f"step_callback raised: {e}")

            # Check for EOS
            if next_token_id == eos_token_id:
                logger.debug(f"EOS token generated at step {step}")
                # Record step time before break
                _step_times_ms.append((time.perf_counter() - _step_start) * 1000)
                break

            # Append to decoder input for next iteration
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)

            # Record step time
            _step_times_ms.append((time.perf_counter() - _step_start) * 1000)

        # Record decoder loop timing with per_step stats
        _decoder_loop_ms = (time.perf_counter() - _decoder_loop_start) * 1000
        if monitor and monitor.stack:
            # Add decoder_loop as child of current operation (model_inference)
            from CoreVital.instrumentation.performance import OperationTiming

            # Build per_step stats if we have step times
            per_step_stats = None
            if _step_times_ms:
                per_step_stats = {
                    "count": len(_step_times_ms),
                    "min_ms": min(_step_times_ms),
                    "max_ms": max(_step_times_ms),
                    "avg_ms": sum(_step_times_ms) / len(_step_times_ms),
                }

            decoder_loop_timing = OperationTiming(
                operation_name="decoder_loop",
                duration_ms=_decoder_loop_ms,
            )
            # Store per_step in metadata
            if per_step_stats:
                decoder_loop_timing.metadata["per_step"] = per_step_stats
            monitor.stack[-1].children.append(decoder_loop_timing)

        logger.info(f"Manual generation complete: {len(generated_token_ids)} tokens generated")

        # Build output structure similar to GenerateDecoderOnlyOutput
        # Create a simple object to hold the outputs
        class Seq2SeqOutput:
            def __init__(self):
                self.sequences = None
                self.scores = tuple(all_scores) if all_scores else None
                self.hidden_states = tuple(all_hidden_states) if all_hidden_states else None
                self.attentions = tuple(all_attentions) if all_attentions else None

        output_obj = Seq2SeqOutput()
        output_obj.sequences = torch.tensor([prompt_token_ids + generated_token_ids], device=device)

        # Return both the object and a dict for compatibility
        # Include encoder outputs and cross-attentions for Seq2Seq models
        return {
            "output_obj": output_obj,
            "generated_token_ids": generated_token_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attentions": encoder_attentions,
            "cross_attentions": tuple(all_cross_attentions) if all_cross_attentions else None,
        }

    def _process_timeline(
        self,
        outputs: Any,
        prompt_token_ids: List[int],
        generated_token_ids: List[int],
        cross_attentions: Optional[Any] = None,
        beam_indices: Optional[Any] = None,
        num_beams: int = 1,
    ) -> List[StepData]:
        """
        Process model outputs into timeline of steps.

        Args:
            outputs: Model generation outputs
            prompt_token_ids: Prompt token IDs
            generated_token_ids: Generated token IDs
            cross_attentions: Optional cross-attention outputs (Seq2Seq)
            beam_indices: Optional (batch_size, seq_len) tensor for beam search; which beam produced each token
            num_beams: Number of beams (when > 1, scores/hidden/attn are indexed by beam)

        Returns:
            List of StepData objects
        """
        if self.model_bundle is None:
            raise InstrumentationError("Model bundle not initialized")

        timeline = []
        batch_idx = 0  # We only support batch_size=1 for instrumentation

        # Note: Transformers generation outputs can be complex
        # Structure:
        # - scores: tuple of logits tensors, one per generation step
        # - hidden_states: tuple where each element is a tuple of hidden state tensors (one per layer)
        # - attentions: tuple where each element is a tuple of attention tensors (one per layer)
        # With beam search: scores[t] is (batch*num_beams, vocab_size); use beam_indices to pick best beam

        has_scores = hasattr(outputs, "scores") and outputs.scores is not None
        has_hidden = hasattr(outputs, "hidden_states") and outputs.hidden_states is not None
        has_attention = hasattr(outputs, "attentions") and outputs.attentions is not None

        if not has_scores:
            logger.warning("Scores (logits) not available in outputs")
        if not has_hidden:
            logger.warning("Hidden states not available in outputs")
        if not has_attention:
            logger.warning("Attention weights not available in outputs")

        def _beam_idx(step_idx: int) -> int:
            """Beam index for this step (for beam search)."""
            if beam_indices is None or num_beams <= 1:
                return 0
            try:
                if hasattr(beam_indices, "shape") and len(beam_indices.shape) >= 2:
                    # beam_indices (batch, seq_len); step in generated = prompt_len + step_idx
                    pos = len(prompt_token_ids) + step_idx
                    if beam_indices.shape[1] > pos:
                        return int(beam_indices[batch_idx, pos].item())
                return 0
            except (IndexError, AttributeError, TypeError):
                return 0

        def _slice_beam(tensor: torch.Tensor, step_idx: int) -> torch.Tensor:
            """Slice to the chosen beam when tensor has shape (batch*num_beams, ...)."""
            if num_beams <= 1 or tensor is None:
                return tensor
            try:
                if tensor.dim() >= 1 and tensor.shape[0] >= num_beams:
                    bi = _beam_idx(step_idx)
                    idx = batch_idx * num_beams + bi
                    if idx < tensor.shape[0]:
                        return tensor[idx : idx + 1].clone()
            except (IndexError, AttributeError, TypeError):
                pass
            return tensor

        # Process each step
        for step_idx, token_id in enumerate(generated_token_ids):
            token_text = cast(str, self.model_bundle.tokenizer.decode([token_id]))

            step_data = StepData(
                step_index=len(prompt_token_ids) + step_idx,
                token_id=token_id,
                token_text=token_text,
                is_prompt_token=False,
                logits=None,
                hidden_states=None,
                attentions=None,
                cross_attentions=None,
            )

            # Extract logits (scores) if available
            try:
                if has_scores and len(outputs.scores) > step_idx:
                    score_t = outputs.scores[step_idx]
                    step_data.logits = _slice_beam(score_t, step_idx) if num_beams > 1 else score_t
            except (IndexError, AttributeError, TypeError) as e:
                logger.debug(f"Could not extract logits for step {step_idx}: {e}")

            # Extract hidden states if available
            try:
                if has_hidden and len(outputs.hidden_states) > step_idx:
                    step_hidden = outputs.hidden_states[step_idx]
                    if isinstance(step_hidden, (tuple, list)):
                        if num_beams > 1:
                            step_data.hidden_states = [_slice_beam(t, step_idx) for t in step_hidden]
                        else:
                            step_data.hidden_states = list(step_hidden)
                    else:
                        step_data.hidden_states = (
                            [_slice_beam(step_hidden, step_idx)] if num_beams > 1 else [step_hidden]
                        )
            except (IndexError, AttributeError, TypeError) as e:
                logger.debug(f"Could not extract hidden states for step {step_idx}: {e}")

            # Extract attentions if available
            try:
                if has_attention and len(outputs.attentions) > step_idx:
                    step_attn = outputs.attentions[step_idx]
                    if isinstance(step_attn, (tuple, list)):
                        if num_beams > 1:
                            step_data.attentions = [_slice_beam(t, step_idx) for t in step_attn]
                        else:
                            step_data.attentions = list(step_attn)
                        if len(step_data.attentions) > 0:
                            logger.debug(f"Extracted {len(step_data.attentions)} attention tensors for step {step_idx}")
                    else:
                        step_data.attentions = [_slice_beam(step_attn, step_idx)] if num_beams > 1 else [step_attn]
                elif has_attention:
                    logger.debug(f"Attention available but step_idx {step_idx} >= len {len(outputs.attentions)}")
                else:
                    logger.debug(f"No attention available for step {step_idx}")
            except (IndexError, AttributeError, TypeError) as e:
                logger.warning(f"Could not extract attentions for step {step_idx}: {e}")
                import traceback

                logger.debug(traceback.format_exc())

            # Extract cross-attentions if available (for Seq2Seq models)
            if cross_attentions is not None:
                try:
                    if isinstance(cross_attentions, (tuple, list)) and len(cross_attentions) > step_idx:
                        step_cross_attn = cross_attentions[step_idx]
                        if step_cross_attn is not None:
                            if isinstance(step_cross_attn, (tuple, list)):
                                step_data.cross_attentions = list(step_cross_attn)
                            else:
                                step_data.cross_attentions = [step_cross_attn]
                            cross_attn_count = len(step_data.cross_attentions) if step_data.cross_attentions else 0
                            logger.debug(f"Extracted {cross_attn_count} cross-attention tensors for step {step_idx}")
                except (IndexError, AttributeError, TypeError) as e:
                    logger.debug(f"Could not extract cross-attentions for step {step_idx}: {e}")

            timeline.append(step_data)

        logger.info(f"Processed {len(timeline)} timeline steps")
        return timeline

    def _collect_warnings(self, outputs: Any) -> List[Dict[str, str]]:
        """
        Collect warnings based on what data was available.

        Args:
            outputs: Model generation outputs

        Returns:
            List of warning dictionaries
        """
        warnings = []

        has_scores = hasattr(outputs, "scores") and outputs.scores is not None
        has_hidden = hasattr(outputs, "hidden_states") and outputs.hidden_states is not None
        has_attention = hasattr(outputs, "attentions") and outputs.attentions is not None

        if not has_scores:
            warnings.append(
                {
                    "code": "SCORES_NOT_AVAILABLE",
                    "message": "Model did not return scores (logits); logits_summary omitted.",
                }
            )

        if not has_attention:
            warnings.append(
                {
                    "code": "ATTENTION_NOT_AVAILABLE",
                    "message": "Model did not return attentions; attention_summary omitted.",
                }
            )

        if not has_hidden:
            warnings.append(
                {
                    "code": "HIDDEN_STATES_NOT_AVAILABLE",
                    "message": "Model did not return hidden_states; hidden_summary omitted.",
                }
            )

        return warnings


# ============================================================================
# Test Harness
# ============================================================================


def _test_collector():
    """Test harness for instrumentation collector."""
    print("Testing InstrumentationCollector...")

    from CoreVital.config import Config

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

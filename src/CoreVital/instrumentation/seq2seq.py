# Seq2Seq generation path: manual decoder loop with per-step instrumentation.
#
# Extracted from collector.py during Phase 1.3 Part B.
# Uses step_processor.process_step() for each step so that only
# StepSummary objects (scalars) survive — no raw tensors.

import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, cast

import torch

from CoreVital.config import Config
from CoreVital.logging_utils import get_logger
from CoreVital.models.hf_loader import ModelBundle

from .baselines import (
    _build_logits_processor,
    _normalize_eos,
    _resolve_special_token,
)
from .step_processor import (
    StepSummary,
    normalize_step_tensors,
    process_step,
)

if TYPE_CHECKING:
    from CoreVital.instrumentation.collector import PromptForwardData
    from CoreVital.instrumentation.performance import PerformanceMonitor

logger = get_logger(__name__)


class Seq2SeqOutput:
    """Minimal output wrapper for warnings compatibility."""

    def __init__(
        self,
        scores: Optional[tuple] = None,
        hidden_states: Optional[tuple] = None,
        attentions: Optional[tuple] = None,
    ):
        self.sequences = None
        self.scores = scores
        self.hidden_states = hidden_states
        self.attentions = attentions


@dataclass
class Seq2SeqGenerationResult:
    """Result from Seq2Seq generation path."""

    generated_token_ids: List[int]
    generated_text: str
    timeline: List[StepSummary]
    warnings: List[Dict[str, str]] = field(default_factory=list)
    encoder_hidden_states: Optional[List[torch.Tensor]] = None
    encoder_attentions: Optional[List[torch.Tensor]] = None
    prompt_forward: Optional["PromptForwardData"] = None


def run_seq2seq_generation(
    model_bundle: ModelBundle,
    config: Config,
    inputs: Any,
    prompt_token_ids: List[int],
    monitor: Optional["PerformanceMonitor"] = None,
    step_callback: Optional[
        Callable[
            [int, List[int], Optional[List[torch.Tensor]], Optional[torch.Tensor]],
            bool,
        ]
    ] = None,
    generator: Optional[torch.Generator] = None,
) -> Seq2SeqGenerationResult:
    """Manual Seq2Seq generation with per-step instrumentation.

    Each decoder step is immediately processed via step_processor.process_step(),
    so raw tensors do not accumulate across steps.
    """
    from CoreVital.instrumentation.collector import PromptForwardData

    def _op(name: str):
        return monitor.operation(name) if monitor else nullcontext()

    model = cast(Any, model_bundle.model)
    tokenizer = model_bundle.tokenizer
    device = model_bundle.device
    num_layers = model_bundle.num_layers

    # --- Encoder ---
    logger.debug("Encoding input with encoder...")
    with _op("encoder_forward"):
        encoder_outputs = model.encoder(
            input_ids=inputs.input_ids,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

    encoder_hidden_states = _extract_encoder_hidden(encoder_outputs)
    encoder_attentions = _extract_encoder_attentions(encoder_outputs)

    # --- Decoder loop ---
    _pad_or_eos = (
        getattr(tokenizer, "pad_token_id", None)
        or getattr(tokenizer, "eos_token_id", None)
        or 0
    )
    decoder_start_token_id = _resolve_special_token(
        tokenizer,
        model.config,
        "decoder_start_token_id",
        fallback=_pad_or_eos,
    )
    decoder_input_ids = torch.tensor(
        [[decoder_start_token_id]], device=device, dtype=torch.long
    )

    max_new_tokens = config.generation.max_new_tokens
    eos_ids = _normalize_eos(tokenizer, model.config, fallback=2)
    do_sample = config.generation.do_sample
    logits_processor = _build_logits_processor(config.generation)
    past_key_values: Any = None

    generated_token_ids: List[int] = []
    timeline: List[StepSummary] = []
    # Small buffer for step_callback (last 5 last-layer hidden vectors)
    recent_hidden_vecs: List[torch.Tensor] = []
    # Track whether we captured any scores/hidden/attn (for warnings)
    had_scores = False
    had_hidden = False
    had_attention = False

    logger.debug(
        f"Starting manual decoder generation (max_new_tokens={max_new_tokens})..."
    )
    _decoder_loop_start = time.perf_counter()
    _step_times_ms: List[float] = []

    for step in range(max_new_tokens):
        _step_start = time.perf_counter()

        decoder_outputs = model(
            input_ids=None,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=(
                decoder_input_ids[:, -1:]
                if past_key_values is not None
                else decoder_input_ids
            ),
            output_hidden_states=True,
            output_attentions=True,
            use_cache=True,
            past_key_values=past_key_values,
            return_dict=True,
        )
        past_key_values = getattr(decoder_outputs, "past_key_values", None)

        # --- Extract raw tensors for this step ---
        raw_logits = decoder_outputs.logits[:, -1, :].clone()
        had_scores = True

        raw_hidden = (
            decoder_outputs.decoder_hidden_states
            if hasattr(decoder_outputs, "decoder_hidden_states")
            and decoder_outputs.decoder_hidden_states is not None
            else None
        )
        if raw_hidden is not None:
            had_hidden = True

        raw_attn = _unwrap_decoder_attentions(decoder_outputs)
        if raw_attn is not None:
            had_attention = True

        raw_cross = (
            decoder_outputs.cross_attentions
            if hasattr(decoder_outputs, "cross_attentions")
            and decoder_outputs.cross_attentions is not None
            else None
        )

        # --- step_callback buffer (before normalization, tensors still on device) ---
        _update_hidden_buffer(recent_hidden_vecs, raw_hidden)

        # --- Sample next token ---
        if do_sample:
            next_token_logits = logits_processor(
                decoder_input_ids, raw_logits
            )
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(
                probs, num_samples=1, generator=generator
            )
        else:
            next_token = torch.argmax(raw_logits, dim=-1, keepdim=True)

        next_token_id = int(next_token.item())
        generated_token_ids.append(next_token_id)

        # --- step_callback ---
        if step_callback is not None:
            buffer = recent_hidden_vecs if recent_hidden_vecs else None
            try:
                if step_callback(
                    step,
                    list(generated_token_ids),
                    buffer,
                    raw_logits,
                ):
                    logger.info(
                        f"Step callback requested stop at step {step}"
                    )
                    _step_times_ms.append(
                        (time.perf_counter() - _step_start) * 1000
                    )
                    # Still process this step's tensors before breaking
                    _append_step_summary(
                        timeline,
                        raw_hidden,
                        raw_attn,
                        raw_cross,
                        raw_logits,
                        num_layers,
                        config,
                        model_bundle,
                        prompt_token_ids,
                        step,
                        next_token_id,
                    )
                    break
            except Exception:
                logger.exception("step_callback raised")

        # --- EOS check ---
        if next_token_id in eos_ids:
            logger.debug(f"EOS token generated at step {step}")
            _step_times_ms.append(
                (time.perf_counter() - _step_start) * 1000
            )
            _append_step_summary(
                timeline,
                raw_hidden,
                raw_attn,
                raw_cross,
                raw_logits,
                num_layers,
                config,
                model_bundle,
                prompt_token_ids,
                step,
                next_token_id,
            )
            break

        # --- Normalize + process_step (raw tensors discarded here) ---
        _append_step_summary(
            timeline,
            raw_hidden,
            raw_attn,
            raw_cross,
            raw_logits,
            num_layers,
            config,
            model_bundle,
            prompt_token_ids,
            step,
            next_token_id,
        )

        decoder_input_ids = torch.cat(
            [decoder_input_ids, next_token], dim=-1
        )
        _step_times_ms.append(
            (time.perf_counter() - _step_start) * 1000
        )

    # --- Record decoder loop timing ---
    _decoder_loop_ms = (time.perf_counter() - _decoder_loop_start) * 1000
    _record_decoder_timing(monitor, _decoder_loop_ms, _step_times_ms)

    logger.info(
        f"Manual generation complete: {len(generated_token_ids)} tokens generated"
    )

    # --- Decode text ---
    generated_text = cast(
        str,
        tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ),
    )

    # --- Seq2Seq prompt_forward (encoder reuse, zero-cost) ---
    prompt_forward: Optional[PromptForwardData] = None
    if config.prompt_telemetry.enabled:
        prompt_forward = PromptForwardData(
            hidden_states=encoder_hidden_states,
            attentions=encoder_attentions,
            logits=None,
            prompt_token_ids=prompt_token_ids,
        )
        logger.debug(
            f"Seq2Seq prompt telemetry: reusing encoder outputs "
            f"({len(encoder_hidden_states) if encoder_hidden_states else 0} layers)"
        )

    # --- Warnings ---
    warnings: List[Dict[str, str]] = []
    if not had_scores:
        warnings.append(
            {
                "code": "SCORES_NOT_AVAILABLE",
                "message": "Model did not return scores (logits); logits_summary omitted.",
            }
        )
    if not had_attention:
        warnings.append(
            {
                "code": "ATTENTION_NOT_AVAILABLE",
                "message": "Model did not return attentions; attention_summary omitted.",
            }
        )
    if not had_hidden:
        warnings.append(
            {
                "code": "HIDDEN_STATES_NOT_AVAILABLE",
                "message": "Model did not return hidden_states; hidden_summary omitted.",
            }
        )

    return Seq2SeqGenerationResult(
        generated_token_ids=generated_token_ids,
        generated_text=generated_text,
        timeline=timeline,
        warnings=warnings,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attentions=encoder_attentions,
        prompt_forward=prompt_forward,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _append_step_summary(
    timeline: List[StepSummary],
    raw_hidden: Optional[tuple],
    raw_attn: Optional[tuple],
    raw_cross: Optional[tuple],
    raw_logits: torch.Tensor,
    num_layers: int,
    config: Config,
    model_bundle: ModelBundle,
    prompt_token_ids: List[int],
    step: int,
    token_id: int,
) -> None:
    """Normalize tensors and append StepSummary to timeline."""
    token_text = cast(str, model_bundle.tokenizer.decode([token_id]))
    payload = normalize_step_tensors(
        raw_hidden=raw_hidden,
        raw_attention=raw_attn,
        raw_cross_attention=raw_cross,
        raw_logits=raw_logits,
        num_layers=num_layers,
    )
    summary = process_step(
        payload,
        config.summaries,
        step_index=len(prompt_token_ids) + step,
        token_id=token_id,
        token_text=token_text,
        tokenizer=model_bundle.tokenizer,
    )
    timeline.append(summary)


def _unwrap_decoder_attentions(
    decoder_outputs: Any,
) -> Optional[tuple]:
    """Extract and unwrap decoder self-attentions (handles T5 tuple wrapping)."""
    if (
        not hasattr(decoder_outputs, "decoder_attentions")
        or decoder_outputs.decoder_attentions is None
    ):
        return None

    attn_list: List[torch.Tensor] = []
    for layer_attn_tuple in decoder_outputs.decoder_attentions:
        if layer_attn_tuple is None:
            continue
        if (
            isinstance(layer_attn_tuple, tuple)
            and len(layer_attn_tuple) > 0
        ):
            attn_tensor = layer_attn_tuple[0]
            if attn_tensor is not None:
                attn_list.append(attn_tensor)
        elif isinstance(layer_attn_tuple, torch.Tensor):
            attn_list.append(layer_attn_tuple)

    return tuple(attn_list) if attn_list else None


def _extract_encoder_hidden(
    encoder_outputs: Any,
) -> Optional[List[torch.Tensor]]:
    """Extract encoder hidden states (skip embedding layer)."""
    if (
        not hasattr(encoder_outputs, "hidden_states")
        or encoder_outputs.hidden_states is None
    ):
        return None

    hs = encoder_outputs.hidden_states
    if len(hs) > 1:
        hidden_states_tuple = hs[1:]
    else:
        hidden_states_tuple = hs

    result = (
        list(hidden_states_tuple)
        if isinstance(hidden_states_tuple, tuple)
        else hidden_states_tuple
    )
    logger.debug(f"Extracted {len(result)} encoder hidden state layers")
    return result


def _extract_encoder_attentions(
    encoder_outputs: Any,
) -> Optional[List[torch.Tensor]]:
    """Extract encoder attentions."""
    if (
        not hasattr(encoder_outputs, "attentions")
        or encoder_outputs.attentions is None
    ):
        return None

    result = (
        list(encoder_outputs.attentions)
        if isinstance(encoder_outputs.attentions, tuple)
        else encoder_outputs.attentions
    )
    logger.debug(f"Extracted {len(result)} encoder attention layers")
    return result


def _update_hidden_buffer(
    buffer: List[torch.Tensor],
    raw_hidden: Optional[tuple],
) -> None:
    """Update the step_callback hidden buffer with the last-layer hidden vector."""
    if raw_hidden is None:
        return

    layers = list(raw_hidden) if isinstance(raw_hidden, (tuple, list)) else []
    if not layers:
        return

    # Skip embedding if present (first element when len > expected)
    last_layer = layers[-1]
    if last_layer is None:
        return

    try:
        if last_layer.dim() == 3:
            vec = last_layer[0, -1, :].detach()
        elif last_layer.dim() == 2:
            vec = last_layer[-1, :].detach()
        else:
            vec = last_layer.flatten()
        if vec.dim() == 1:
            buffer.append(vec)
    except Exception:
        pass

    # FIFO: keep at most 5
    while len(buffer) > 5:
        buffer.pop(0)


def _record_decoder_timing(
    monitor: Optional["PerformanceMonitor"],
    decoder_loop_ms: float,
    step_times_ms: List[float],
) -> None:
    """Record decoder loop timing with per_step stats in the monitor."""
    if not monitor or not monitor.stack:
        return

    from CoreVital.instrumentation.performance import OperationTiming

    per_step_stats = None
    if step_times_ms:
        per_step_stats = {
            "count": len(step_times_ms),
            "min_ms": min(step_times_ms),
            "max_ms": max(step_times_ms),
            "avg_ms": sum(step_times_ms) / len(step_times_ms),
        }

    decoder_loop_timing = OperationTiming(
        operation_name="decoder_loop",
        duration_ms=decoder_loop_ms,
    )
    if per_step_stats:
        decoder_loop_timing.metadata["per_step"] = per_step_stats
    monitor.stack[-1].children.append(decoder_loop_timing)

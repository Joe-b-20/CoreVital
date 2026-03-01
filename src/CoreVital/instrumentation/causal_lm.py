# CausalLM generation path: model.generate() + timeline processing.
#
# Extracted from collector.py during Phase 1.3 Part B.
# Uses step_processor.process_step() for each step so that only
# StepSummary objects (scalars) survive — no raw tensors.

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

import torch

from CoreVital.config import Config
from CoreVital.errors import InstrumentationError
from CoreVital.logging_utils import get_logger
from CoreVital.models.hf_loader import ModelBundle

from .step_processor import (
    StepSummary,
    normalize_step_tensors,
    process_step,
)

if TYPE_CHECKING:
    from CoreVital.instrumentation.performance import PerformanceMonitor

logger = get_logger(__name__)


@dataclass
class CausalGenerationResult:
    """Result from CausalLM generation path."""

    generated_token_ids: List[int]
    generated_text: str
    timeline: List[StepSummary]
    warnings: List[Dict[str, str]] = field(default_factory=list)


def run_causal_generation(
    model_bundle: ModelBundle,
    config: Config,
    inputs: Any,
    prompt_token_ids: List[int],
    monitor: Optional["PerformanceMonitor"] = None,
    generator: Optional[torch.Generator] = None,
    seed: Optional[int] = None,
) -> CausalGenerationResult:
    """Run CausalLM generation with full instrumentation.

    Calls model.generate(), extracts tokens, decodes text, processes
    the timeline through step_processor, and collects warnings.
    """

    def _op(name: str):
        return monitor.operation(name) if monitor else nullcontext()

    num_beams = getattr(config.generation, "num_beams", 1) or 1

    # Issue 49: Beam search with per-layer capture is not supported.
    if num_beams > 1 and (config.summaries.hidden.enabled or config.summaries.attention.enabled):
        raise InstrumentationError(
            "Beam search (num_beams > 1) is not supported with hidden_states or "
            "attentions capture. Use num_beams=1 or disable hidden/attention summaries."
        )
    gen_config: Dict[str, Any] = {
        "max_new_tokens": config.generation.max_new_tokens,
        "do_sample": config.generation.do_sample,
        "temperature": config.generation.temperature,
        "top_k": config.generation.top_k,
        "top_p": config.generation.top_p,
        "output_hidden_states": True,
        "output_attentions": True,
        "output_scores": True,
        "return_dict_in_generate": True,
        "pad_token_id": model_bundle.tokenizer.pad_token_id,
    }
    if num_beams > 1:
        gen_config["num_beams"] = num_beams
        gen_config["early_stopping"] = getattr(config.generation, "early_stopping", False)
        gen_config["do_sample"] = False

    # HF generate() does not accept a generator kwarg. For reproducibility when
    # seed is set we use fork_rng to seed a *forked* copy of the global RNG so
    # do_sample draws are deterministic without leaking state to the host process.
    _use_fork = seed is not None and gen_config.get("do_sample", False)
    _rng_ctx = torch.random.fork_rng(enabled=_use_fork)

    with _op("model.generate"), _rng_ctx:
        if _use_fork:
            torch.manual_seed(seed)  # type: ignore[arg-type]
        outputs = cast(Any, model_bundle.model).generate(**inputs, **gen_config)

    # --- Extract generated tokens ---
    with _op("extract_generated_tokens"):
        out_seq = cast(Any, outputs).sequences
        if out_seq.dim() == 3:
            generated_ids = out_seq[0, 0, :].tolist()
        else:
            generated_ids = out_seq[0].tolist()
        generated_token_ids = generated_ids[len(prompt_token_ids) :]

    # --- Decode text ---
    with _op("decode_generated_text"):
        generated_text = cast(
            str,
            model_bundle.tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ),
        )
    logger.debug(f"Generated tokens: {len(generated_token_ids)}")

    # --- Build timeline ---
    beam_indices = getattr(outputs, "beam_indices", None)
    with _op("_process_timeline"):
        timeline = _process_timeline(
            outputs,
            model_bundle,
            config,
            prompt_token_ids,
            generated_token_ids,
            beam_indices=beam_indices,
            num_beams=num_beams,
        )

    # --- Warnings ---
    with _op("_collect_warnings"):
        warnings = _collect_warnings(outputs)

    return CausalGenerationResult(
        generated_token_ids=generated_token_ids,
        generated_text=generated_text,
        timeline=timeline,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Timeline processing
# ---------------------------------------------------------------------------


def _process_timeline(
    outputs: Any,
    model_bundle: ModelBundle,
    config: Config,
    prompt_token_ids: List[int],
    generated_token_ids: List[int],
    beam_indices: Optional[Any] = None,
    num_beams: int = 1,
) -> List[StepSummary]:
    """Process CausalLM outputs into timeline of StepSummary objects.

    Each step's raw tensors are normalized and summarized via
    step_processor.process_step(); no raw tensors survive.
    """
    num_layers = model_bundle.num_layers
    batch_idx = 0

    has_scores = hasattr(outputs, "scores") and outputs.scores is not None
    has_hidden = hasattr(outputs, "hidden_states") and outputs.hidden_states is not None
    has_attention = hasattr(outputs, "attentions") and outputs.attentions is not None

    if not has_scores:
        logger.warning("Scores (logits) not available in outputs")
    if not has_hidden:
        logger.warning("Hidden states not available in outputs")
    if not has_attention:
        logger.warning("Attention weights not available in outputs")

    # --- Beam helpers (closures over beam_indices, num_beams, batch_idx) ---
    def _beam_idx(step_idx: int) -> int:
        if beam_indices is None or num_beams <= 1:
            return 0
        try:
            if hasattr(beam_indices, "shape") and len(beam_indices.shape) >= 2:
                pos = len(prompt_token_ids) + step_idx
                if beam_indices.shape[1] > pos:
                    return int(beam_indices[batch_idx, pos].item())
            return 0
        except (IndexError, AttributeError, TypeError):
            return 0

    def _slice_beam(tensor: torch.Tensor) -> torch.Tensor:
        """Slice to the chosen beam; step_idx captured via nonlocal."""
        if num_beams <= 1 or tensor is None:
            return tensor
        try:
            if tensor.dim() >= 1 and tensor.shape[0] >= num_beams:
                bi = _beam_idx(_current_step[0])
                idx = batch_idx * num_beams + bi
                if idx < tensor.shape[0]:
                    return tensor[idx : idx + 1].clone()
        except (IndexError, AttributeError, TypeError):
            pass
        return tensor

    _current_step = [0]

    timeline: List[StepSummary] = []
    for step_idx, token_id in enumerate(generated_token_ids):
        _current_step[0] = step_idx
        token_text = cast(str, model_bundle.tokenizer.decode([token_id]))

        # Gather raw tensors for this step
        raw_logits = None
        if has_scores and len(outputs.scores) > step_idx:
            raw_logits = outputs.scores[step_idx]

        raw_hidden = None
        if has_hidden and len(outputs.hidden_states) > step_idx:
            raw_hidden = outputs.hidden_states[step_idx]

        raw_attention = None
        if has_attention and len(outputs.attentions) > step_idx:
            raw_attention = outputs.attentions[step_idx]

        beam_handler = _slice_beam if num_beams > 1 else None

        payload = normalize_step_tensors(
            raw_hidden=raw_hidden,
            raw_attention=raw_attention,
            raw_cross_attention=None,
            raw_logits=raw_logits,
            num_layers=num_layers,
            beam_handler=beam_handler,
        )

        summary = process_step(
            payload,
            config.summaries,
            step_index=len(prompt_token_ids) + step_idx,
            token_id=token_id,
            token_text=token_text,
            tokenizer=model_bundle.tokenizer,
        )
        timeline.append(summary)

    logger.info(f"Processed {len(timeline)} timeline steps")
    return timeline


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


def _collect_warnings(outputs: Any) -> List[Dict[str, str]]:
    """Collect warnings about missing data in generation outputs."""
    warnings: List[Dict[str, str]] = []
    if not (hasattr(outputs, "scores") and outputs.scores is not None):
        warnings.append(
            {
                "code": "SCORES_NOT_AVAILABLE",
                "message": "Model did not return scores (logits); logits_summary omitted.",
            }
        )
    if not (hasattr(outputs, "attentions") and outputs.attentions is not None):
        warnings.append(
            {
                "code": "ATTENTION_NOT_AVAILABLE",
                "message": "Model did not return attentions; attention_summary omitted.",
            }
        )
    if not (hasattr(outputs, "hidden_states") and outputs.hidden_states is not None):
        warnings.append(
            {
                "code": "HIDDEN_STATES_NOT_AVAILABLE",
                "message": "Model did not return hidden_states; hidden_summary omitted.",
            }
        )
    return warnings

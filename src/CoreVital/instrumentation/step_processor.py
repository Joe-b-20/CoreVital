# Step tensor→summary lifecycle: normalize raw tensors and compute summaries.
#
# Each generation step produces raw model output tensors (hidden states, attentions,
# logits). This module:
# 1. Normalizes shapes into a uniform contract (NormalizedStepPayload)
# 2. Computes all summary functions (logits, attention, hidden state)
# 3. Returns scalars-only StepSummary
# 4. Discards all raw tensors (they must not survive past process_step)

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

from CoreVital.logging_utils import get_logger

from .summaries import (
    compute_attention_summary,
    compute_hidden_summary,
    compute_logits_summary,
    detect_tensor_anomalies,
)

logger = get_logger(__name__)


@dataclass
class NormalizedStepPayload:
    """Shape-normalized tensors for one step. Consumed by process_step,
    then discarded."""

    hidden_states: Optional[List[torch.Tensor]]  # (1, 1, hidden_dim) per layer
    attentions: Optional[List[torch.Tensor]]  # (1, heads, 1, key_len) per layer
    cross_attentions: Optional[List[torch.Tensor]]  # same shape contract
    logits: Optional[torch.Tensor]  # (1, vocab_size) or (vocab_size,)


@dataclass
class LayerStepSummary:
    """Pre-computed summaries for one layer at one generation step."""

    hidden_summary: Dict[str, Any] = field(default_factory=dict)
    attention_summary: Dict[str, Any] = field(default_factory=dict)
    cross_attention_summary: Optional[Dict[str, Any]] = None
    anomalies: Dict[str, bool] = field(
        default_factory=lambda: {"has_nan": False, "has_inf": False}
    )


@dataclass
class StepSummary:
    """Pre-computed summaries for one generation step.
    Contains only scalars and small lists — no raw model tensors."""

    step_index: int
    token_id: int
    token_text: str
    is_prompt_token: bool
    logits_summary: Dict[str, Any] = field(default_factory=dict)
    layer_summaries: List[LayerStepSummary] = field(default_factory=list)
    # Small derived 1-D vector for repetition detection buffer.
    # Consumed by _build_health_flags then deleted; not a raw model tensor.
    _last_layer_hidden_vec: Optional[torch.Tensor] = field(
        default=None, repr=False
    )


def normalize_step_tensors(
    raw_hidden: Optional[tuple],
    raw_attention: Optional[tuple],
    raw_cross_attention: Optional[tuple],
    raw_logits: Optional[torch.Tensor],
    num_layers: int,
    beam_handler: Optional[Callable] = None,
) -> NormalizedStepPayload:
    """Normalize shapes from either CausalLM or Seq2Seq into uniform contract.

    - Strip embedding layer if len(raw_hidden) > num_layers
    - Slice attention to last query token: [:, :, -1:, :]
    - .detach().cpu() everything
    - Optional beam slicing via beam_handler
    - Shape assertions per NormalizedStepPayload contract (Issue 54)
    """
    hidden = _normalize_hidden(raw_hidden, num_layers, beam_handler)
    attn = _normalize_attention(raw_attention, beam_handler)
    cross = _normalize_attention(raw_cross_attention, None)
    logits = _normalize_logits(raw_logits, beam_handler)

    # Shape contract: hidden (1, 1, hidden_dim), attention (1, heads, 1, key_len), logits 1D or 2D
    if hidden is not None:
        for i, t in enumerate(hidden):
            assert t.dim() == 3, (
                f"hidden_states[{i}] must be 3D (batch, seq, hidden), got dim={t.dim()}"
            )
    if attn is not None:
        for i, t in enumerate(attn):
            assert t.dim() == 4, (
                f"attentions[{i}] must be 4D (batch, heads, 1, key_len), got dim={t.dim()}"
            )
    if cross is not None:
        for i, t in enumerate(cross):
            assert t.dim() == 4, (
                f"cross_attentions[{i}] must be 4D (batch, heads, 1, key_len), got dim={t.dim()}"
            )
    if logits is not None:
        assert logits.dim() in (1, 2), (
            f"logits must be 1D or 2D, got dim={logits.dim()}"
        )

    return NormalizedStepPayload(
        hidden_states=hidden,
        attentions=attn,
        cross_attentions=cross,
        logits=logits,
    )


def process_step(
    payload: NormalizedStepPayload,
    config: Any,
    step_index: int,
    token_id: Optional[int] = None,
    token_text: Optional[str] = None,
    tokenizer: Optional[Any] = None,
    profile: Optional[Any] = None,
) -> StepSummary:
    """Compute all summaries for one step. Discards tensors after.

    Args:
        payload: Normalized tensors for this step.
        config: SummariesConfig with .hidden, .attention, .logits sub-configs.
        step_index: Step index in generation sequence.
        token_id: Token ID generated at this step.
        token_text: Decoded text of the token.
        tokenizer: Tokenizer for logits summary (top-k token text decoding).
        profile: Optional model profile for threshold overrides.

    Returns:
        StepSummary with all computed summaries (scalars only).
    """
    logits_dict = _compute_logits(payload, config, step_index, token_id, tokenizer)
    layers = _compute_layer_summaries(payload, config, step_index, profile)
    last_vec = _extract_last_layer_vec(payload)

    summary = StepSummary(
        step_index=step_index,
        token_id=token_id if token_id is not None else 0,
        token_text=token_text if token_text is not None else "",
        is_prompt_token=False,
        logits_summary=logits_dict,
        layer_summaries=layers,
        _last_layer_hidden_vec=last_vec,
    )

    # Discard payload tensors — they must not survive this function
    payload.hidden_states = None
    payload.attentions = None
    payload.cross_attentions = None
    payload.logits = None

    return summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_hidden(
    raw: Optional[tuple],
    num_layers: int,
    beam_handler: Optional[Callable],
) -> Optional[List[torch.Tensor]]:
    if raw is None:
        return None
    hs = list(raw) if isinstance(raw, (tuple, list)) else [raw]
    if len(hs) > num_layers:
        hs = hs[1 : num_layers + 1]
    if beam_handler is not None:
        hs = [beam_handler(t) for t in hs if t is not None]
    result = [
        t.detach().cpu()
        for t in hs
        if t is not None and isinstance(t, torch.Tensor)
    ]
    return result if result else None


def _normalize_attention(
    raw: Optional[tuple],
    beam_handler: Optional[Callable],
) -> Optional[List[torch.Tensor]]:
    if raw is None:
        return None
    items = list(raw) if isinstance(raw, (tuple, list)) else [raw]
    if beam_handler is not None:
        items = [beam_handler(t) for t in items if t is not None]
    processed: List[torch.Tensor] = []
    for t in items:
        if t is None or not isinstance(t, torch.Tensor):
            continue
        if t.dim() == 4:
            t = t[:, :, -1:, :]
        processed.append(t.detach().cpu())
    return processed if processed else None


def _normalize_logits(
    raw: Optional[torch.Tensor],
    beam_handler: Optional[Callable],
) -> Optional[torch.Tensor]:
    if raw is None or not isinstance(raw, torch.Tensor):
        return None
    if beam_handler is not None:
        raw = beam_handler(raw)
    return raw.detach().cpu()


def _compute_logits(
    payload: NormalizedStepPayload,
    config: Any,
    step_index: int,
    token_id: Optional[int],
    tokenizer: Optional[Any],
) -> Dict[str, Any]:
    if payload.logits is None:
        return {}
    try:
        return compute_logits_summary(
            payload.logits,
            tokenizer,
            config.logits,
            actual_token_id=token_id,
        )
    except Exception as e:
        logger.warning(
            f"Failed to compute logits summary for step {step_index}: {e}"
        )
        return {}


def _compute_layer_summaries(
    payload: NormalizedStepPayload,
    config: Any,
    step_index: int,
    profile: Optional[Any],
) -> List[LayerStepSummary]:
    num_hidden = len(payload.hidden_states) if payload.hidden_states else 0
    layers: List[LayerStepSummary] = []

    for layer_idx in range(num_hidden):
        hidden_dict = _safe_hidden(payload, config, step_index, layer_idx)
        attn_dict = _safe_attention(
            payload.attentions, config, step_index, layer_idx, profile
        )
        cross_dict = _safe_attention(
            payload.cross_attentions, config, step_index, layer_idx, profile
        )
        anomalies_dict = _safe_anomalies(payload, layer_idx, num_hidden)

        layers.append(
            LayerStepSummary(
                hidden_summary=hidden_dict,
                attention_summary=attn_dict,
                cross_attention_summary=cross_dict if cross_dict else None,
                anomalies=anomalies_dict,
            )
        )
    return layers


def _safe_hidden(
    payload: NormalizedStepPayload,
    config: Any,
    step_index: int,
    layer_idx: int,
) -> Dict[str, Any]:
    try:
        return compute_hidden_summary(
            payload.hidden_states[layer_idx], config.hidden  # type: ignore[index]
        )
    except Exception as e:
        logger.warning(
            f"Failed hidden summary for step {step_index} layer {layer_idx}: {e}"
        )
        return {}


def _safe_attention(
    tensors: Optional[List[torch.Tensor]],
    config: Any,
    step_index: int,
    layer_idx: int,
    profile: Optional[Any],
) -> Dict[str, Any]:
    if tensors is None or layer_idx >= len(tensors):
        return {}
    try:
        return compute_attention_summary(
            tensors[layer_idx], config.attention, profile=profile
        )
    except Exception as e:
        logger.debug(
            f"Failed attention summary for step {step_index} layer {layer_idx}: {e}"
        )
        return {}


def _safe_anomalies(
    payload: NormalizedStepPayload,
    layer_idx: int,
    num_hidden: int,
) -> Dict[str, bool]:
    try:
        h = (
            payload.hidden_states[layer_idx]
            if payload.hidden_states and layer_idx < num_hidden
            else None
        )
        a = (
            payload.attentions[layer_idx]
            if payload.attentions and layer_idx < len(payload.attentions)
            else None
        )
        c = (
            payload.cross_attentions[layer_idx]
            if payload.cross_attentions
            and layer_idx < len(payload.cross_attentions)
            else None
        )
        return detect_tensor_anomalies(
            hidden_state=h, attention=a, cross_attention=c
        )
    except Exception:
        return {"has_nan": False, "has_inf": False}


def _extract_last_layer_vec(
    payload: NormalizedStepPayload,
) -> Optional[torch.Tensor]:
    """Extract a small 1-D vector from the last layer for repetition detection."""
    if not payload.hidden_states:
        return None
    last_layer = payload.hidden_states[-1]
    if not isinstance(last_layer, torch.Tensor):
        return None
    try:
        if last_layer.dim() == 3:
            return last_layer[0, -1, :].detach().cpu()
        if last_layer.dim() == 2:
            return last_layer[-1, :].detach().cpu()
        if last_layer.dim() == 1:
            return last_layer.detach().cpu()
    except Exception:
        pass
    return None

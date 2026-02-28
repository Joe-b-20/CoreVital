# Baseline, warmup, and prompt-forward helpers.
#
# Contains functions shared between CausalLM and Seq2Seq paths:
# - Token resolution (_resolve_special_token, _normalize_eos)
# - Logits processor builder (_build_logits_processor)
# - Warmup / baseline generation (no instrumentation)
# - Prompt forward pass (CausalLM only)

import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

import torch
from transformers import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from CoreVital.config import Config
from CoreVital.errors import InstrumentationError
from CoreVital.logging_utils import get_logger
from CoreVital.models.hf_loader import ModelBundle

if TYPE_CHECKING:
    from CoreVital.instrumentation.performance import PerformanceMonitor

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared token resolution helpers
# ---------------------------------------------------------------------------


def _resolve_special_token(
    tokenizer: Any,
    model_config: Any,
    attr_name: str,
    fallback: Optional[Any] = None,
) -> Any:
    """Resolve a special token ID from tokenizer, then model config, then fallback."""
    val = getattr(tokenizer, attr_name, None)
    if val is None:
        val = getattr(model_config, attr_name, None)
    if val is None:
        val = fallback
    return val


def _normalize_eos(
    tokenizer: Any, model_config: Any, fallback: Optional[int] = None
) -> set:
    """Normalize EOS token ID to a set for membership checks.

    model.config.eos_token_id can be int, list, or tuple on modern models.
    """
    eos = _resolve_special_token(
        tokenizer, model_config, "eos_token_id", fallback=None
    )
    if eos is None:
        eos = fallback
    if eos is None:
        return set()
    if isinstance(eos, (list, tuple)):
        return set(eos)
    return {eos}


def _build_logits_processor(gen_config: Any) -> LogitsProcessorList:
    """Build HF LogitsProcessorList for temperature, top-k, and top-p."""
    processors = LogitsProcessorList()
    if (
        getattr(gen_config, "temperature", None)
        and gen_config.temperature != 1.0
    ):
        processors.append(TemperatureLogitsWarper(gen_config.temperature))
    if getattr(gen_config, "top_k", None) and gen_config.top_k > 0:
        processors.append(TopKLogitsWarper(gen_config.top_k))
    if getattr(gen_config, "top_p", None) and gen_config.top_p < 1.0:
        processors.append(TopPLogitsWarper(gen_config.top_p))
    return processors


# ---------------------------------------------------------------------------
# Warmup / baseline
# ---------------------------------------------------------------------------


def run_warmup(
    model_bundle: ModelBundle,
    config: Config,
    inputs: Any,
    is_seq2seq: bool,
) -> None:
    """Run a warmup generation (no instrumentation, results discarded)."""
    with torch.no_grad():
        if is_seq2seq:
            run_baseline_seq2seq(model_bundle, config, inputs)
        else:
            run_baseline_causal(model_bundle, config, inputs)


def run_baseline(
    model_bundle: ModelBundle,
    config: Config,
    inputs: Any,
    is_seq2seq: bool,
) -> float:
    """Run baseline inference (no instrumentation) and return elapsed ms."""
    start = time.perf_counter()
    with torch.no_grad():
        if is_seq2seq:
            run_baseline_seq2seq(model_bundle, config, inputs)
        else:
            run_baseline_causal(model_bundle, config, inputs)
    return (time.perf_counter() - start) * 1000


def run_baseline_causal(
    model_bundle: ModelBundle, config: Config, inputs: Any
) -> None:
    """Baseline CausalLM generation (no output_hidden_states/attentions/scores)."""
    num_beams = getattr(config.generation, "num_beams", 1) or 1
    gen_config: Dict[str, Any] = {
        "max_new_tokens": config.generation.max_new_tokens,
        "do_sample": config.generation.do_sample,
        "temperature": config.generation.temperature,
        "top_k": config.generation.top_k,
        "top_p": config.generation.top_p,
        "pad_token_id": model_bundle.tokenizer.pad_token_id,
    }
    if num_beams > 1:
        gen_config["num_beams"] = num_beams
        gen_config["early_stopping"] = getattr(
            config.generation, "early_stopping", False
        )
        gen_config["do_sample"] = False
    cast(Any, model_bundle.model).generate(**inputs, **gen_config)


def run_baseline_seq2seq(
    model_bundle: ModelBundle, config: Config, inputs: Any
) -> None:
    """Baseline Seq2Seq generation: encoder + decoder loop, no hidden_states/attentions.

    Sampling logic matches the instrumented path for reproducible comparisons.
    """
    model = cast(Any, model_bundle.model)
    tokenizer = model_bundle.tokenizer
    device = model_bundle.device

    encoder_outputs = model.encoder(
        input_ids=inputs.input_ids,
        output_hidden_states=False,
        output_attentions=False,
        return_dict=True,
    )

    _pad_or_eos = (
        getattr(tokenizer, "pad_token_id", None)
        or getattr(tokenizer, "eos_token_id", None)
        or 0
    )
    decoder_start_token_id = _resolve_special_token(
        tokenizer, model.config, "decoder_start_token_id", fallback=_pad_or_eos
    )
    eos_ids = _normalize_eos(tokenizer, model.config, fallback=2)

    decoder_input_ids = torch.tensor(
        [[decoder_start_token_id]], device=device
    )
    max_new_tokens = config.generation.max_new_tokens
    do_sample = config.generation.do_sample
    logits_processor = _build_logits_processor(config.generation)
    past_key_values = None

    for _ in range(max_new_tokens):
        decoder_outputs = model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=(
                decoder_input_ids[:, -1:]
                if past_key_values is not None
                else decoder_input_ids
            ),
            output_hidden_states=False,
            output_attentions=False,
            use_cache=True,
            past_key_values=past_key_values,
            return_dict=True,
        )
        past_key_values = getattr(decoder_outputs, "past_key_values", None)
        next_token_logits = decoder_outputs.logits[:, -1, :]
        if do_sample:
            next_token_logits = logits_processor(
                decoder_input_ids, next_token_logits
            )
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
        else:
            next_token_id = torch.argmax(
                next_token_logits, dim=-1, keepdim=True
            )
        decoder_input_ids = torch.cat(
            [decoder_input_ids, next_token_id], dim=-1
        )
        if next_token_id.item() in eos_ids:
            break


# ---------------------------------------------------------------------------
# Prompt forward pass (CausalLM only)
# ---------------------------------------------------------------------------


def run_prompt_forward(
    model_bundle: ModelBundle,
    config: Config,
    inputs: Any,
    prompt_token_ids: List[int],
    monitor: Optional["PerformanceMonitor"] = None,
) -> "PromptForwardData":
    """Run a single forward pass on prompt tokens to capture hidden states, attentions, and logits.

    CausalLM only.  For Seq2Seq, encoder outputs are reused instead.
    """
    from CoreVital.instrumentation.collector import PromptForwardData

    model = cast(Any, model_bundle.model)
    logger.debug(
        f"Running prompt forward pass ({len(prompt_token_ids)} tokens)..."
    )
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

    hidden_states = None
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        try:
            hs = outputs.hidden_states
            if isinstance(hs, (tuple, list)) and len(hs) > 1:
                hidden_states = [h.cpu() for h in hs[1:]]
            elif isinstance(hs, (tuple, list)):
                hidden_states = [h.cpu() for h in hs]
            logger.debug(
                f"Prompt forward: extracted "
                f"{len(hidden_states) if hidden_states else 0} hidden state layers"
            )
        except (TypeError, AttributeError) as e:
            logger.debug(
                f"Prompt forward: could not extract hidden states: {e}"
            )

    attentions = None
    if hasattr(outputs, "attentions") and outputs.attentions is not None:
        try:
            att = outputs.attentions
            if isinstance(att, (tuple, list)):
                attentions = [a.cpu() for a in att]
            logger.debug(
                f"Prompt forward: extracted "
                f"{len(attentions) if attentions else 0} attention layers"
            )
        except (TypeError, AttributeError) as e:
            logger.debug(
                f"Prompt forward: could not extract attentions: {e}"
            )

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

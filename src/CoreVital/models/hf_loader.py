# ============================================================================
# CoreVital - Hugging Face Model Loader
#
# Purpose: Load and configure Hugging Face models and tokenizers
# Inputs: Config with model specification and device settings
# Outputs: ModelBundle with model, tokenizer, and metadata
# Dependencies: transformers, torch, config, errors
# Usage: bundle = load_model(config)
#
# Changelog:
#   2026-01-13: Initial HF loader for Phase-0
#   2026-01-14: Added automatic attention implementation switching to 'eager' for models using SDPA
#                This enables output_attentions=True support for Llama and other SDPA-based models
#                Added revision extraction from model config (_commit_hash)
#                Added revision field to ModelBundle dataclass
#   2026-01-15: Added dynamic model loading support using AutoConfig.from_pretrained()
#                Automatically detects Seq2Seq models (T5, BART, etc.) and uses AutoModelForSeq2SeqLM
#                Otherwise uses AutoModelForCausalLM. Model class type stored in ModelBundle.model_class
#   2026-01-15: Added 4-bit and 8-bit quantization support using bitsandbytes library
#                Supports load_in_4bit and load_in_8bit flags via config and CLI
#   2026-01-21: Phase-0.5 hardening - improved quantization check to use model.config.quantization_config;
#                enhanced attention implementation logging; standardized logging to INFO for model loading
#   2026-01-23: Fixed dtype detection for quantized models - now detects actual quantized dtype (int8/uint8)
#                after model loading instead of using pre-load dtype, ensuring JSON output shows correct dtype
#   2026-02-04: Phase-0.75 - added optional monitor parameter for child operation timing
# ============================================================================

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Type, cast

import torch

if TYPE_CHECKING:
    from CoreVital.instrumentation.performance import PerformanceMonitor
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from CoreVital.config import Config
from CoreVital.errors import ModelLoadError
from CoreVital.logging_utils import get_logger
from CoreVital.models.registry import ModelCapabilities


def _probe_attentions_available(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> bool:
    """Run a 1-token forward with output_attentions=True and check if attentions are returned.

    Used at load time to set ModelCapabilities.attentions_available (Issue 52).
    """
    try:
        dummy = tokenizer("test", return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**dummy, output_attentions=True, return_dict=True)
        return (
            getattr(out, "attentions", None) is not None
            and len(out.attentions) > 0
        )
    except Exception:
        return False

logger = get_logger(__name__)


@dataclass
class ModelBundle:
    """
    Container for model, tokenizer, and metadata.

    Attributes:
        model: Loaded HuggingFace model
        tokenizer: Loaded HuggingFace tokenizer
        device: Device the model is on
        dtype: Model dtype or string describing quantized state
        num_layers: Number of transformer layers
        hidden_size: Hidden state dimension
        num_attention_heads: Number of attention heads
        architecture: Model architecture name
        capabilities: Model type and capability info (single source of truth)
        revision: Model revision/commit hash if available
        model_class: The model class type used for loading (AutoModelForCausalLM or AutoModelForSeq2SeqLM)
    """

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    dtype: torch.dtype
    dtype_str: Optional[str]  # Human-readable dtype string for schema; None means use str(dtype)
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    architecture: str
    capabilities: ModelCapabilities
    revision: Optional[str] = None
    model_class: Type[PreTrainedModel] = cast(Type[PreTrainedModel], AutoModelForCausalLM)


def load_model(config: Config, monitor: Optional["PerformanceMonitor"] = None) -> ModelBundle:
    """
    Load a Hugging Face model and tokenizer with specified configuration.

    Args:
        config: Configuration object
        monitor: Optional PerformanceMonitor for timing children operations

    Returns:
        ModelBundle with loaded model and metadata

    Raises:
        ModelLoadError: If model loading fails
    """
    from contextlib import nullcontext

    def _op(name: str):
        """Helper to wrap operations with monitor if available."""
        if monitor:
            return monitor.operation(name)
        return nullcontext()

    try:
        logger.info(f"Loading model: {config.model.hf_id}")

        # Determine device (CoreVital logic)
        with _op("_resolve_device"):
            device = _resolve_device(config.device.requested)
        logger.info(f"Target device: {device}")

        # Determine dtype (CoreVital logic)
        with _op("_resolve_dtype"):
            dtype = _resolve_dtype(config.model.dtype, device)
        logger.info(f"Model dtype: {dtype}")

        # Load tokenizer (external HF library call)
        logger.info("Loading tokenizer...")
        with _op("AutoTokenizer.from_pretrained"):
            tokenizer = AutoTokenizer.from_pretrained(
                config.model.hf_id,
                revision=config.model.revision,
                trust_remote_code=config.model.trust_remote_code,
            )

        # Ensure pad token is set (CoreVital logic)
        with _op("_set_pad_token"):
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        # Inspect model type to determine which model class to use (external HF call)
        logger.info("Inspecting model architecture...")
        with _op("AutoConfig.from_pretrained"):
            model_config = AutoConfig.from_pretrained(
                config.model.hf_id,
                revision=config.model.revision,
                trust_remote_code=config.model.trust_remote_code,
            )

        # Detect model type via the registry (single source of truth)
        capabilities = ModelCapabilities.from_config(model_config)

        if capabilities.is_seq2seq:
            model_class = cast(Type[PreTrainedModel], AutoModelForSeq2SeqLM)
        else:
            model_class = cast(Type[PreTrainedModel], AutoModelForCausalLM)

        # Check for quantization flags
        quantization_config = None
        if config.model.load_in_4bit or config.model.load_in_8bit:
            # Quantization requires CUDA
            if device.type != "cuda":
                logger.warning("Quantization requires CUDA. Falling back to CPU without quantization.")
                device = torch.device("cpu")
            else:
                if config.model.load_in_4bit and config.model.load_in_8bit:
                    logger.warning("Both load_in_4bit and load_in_8bit are set. Using 4-bit quantization.")
                    config.model.load_in_8bit = False

                if config.model.load_in_4bit:
                    logger.info("Initializing BitsAndBytesConfig for 4-bit quantization")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                    )
                    logger.info("4-bit quantization configuration created successfully")
                elif config.model.load_in_8bit:
                    logger.info("Initializing BitsAndBytesConfig for 8-bit quantization")
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                    logger.info("8-bit quantization configuration created successfully")

        # Prepare model loading arguments
        model_kwargs = {
            "revision": config.model.revision,
            "trust_remote_code": config.model.trust_remote_code,
            "low_cpu_mem_usage": True,
            "config": model_config,  # Use the already loaded config
        }

        # Add quantization config if specified
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            # Only set torch_dtype if not using quantization
            model_kwargs["torch_dtype"] = dtype

        # Load model with the appropriate class (external HF library call - the big one!)
        logger.info(f"Loading model with {model_class.__name__}...")
        with _op("model_class.from_pretrained"):
            model = model_class.from_pretrained(
                config.model.hf_id,
                **model_kwargs,
            )

        # Log successful quantization if applied
        if quantization_config is not None:
            if config.model.load_in_4bit:
                logger.info("Model loaded successfully with 4-bit quantization")
            elif config.model.load_in_8bit:
                logger.info("Model loaded successfully with 8-bit quantization")

            # Quantization check: verify quantization was actually applied
            # Note: BitsAndBytes stores params in special format but dtype still shows float16/float32
            # Check for quantization config attribute instead
            if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
                logger.info(f"Quantization config verified: {model.config.quantization_config}")
            else:
                # Fallback: check parameter dtype (may give false warnings with BitsAndBytes)
                actual_dtype = next(model.parameters()).dtype
                if actual_dtype in (torch.float16, torch.float32):
                    logger.warning(
                        f"Quantization requested but model dtype is {actual_dtype}. "
                        "Note: BitsAndBytes quantization may still be active despite float16/32 dtype."
                    )

        # Move to device (only if not using quantization, as quantization handles device placement)
        # This is mixed: CoreVital decides when, but .to() is PyTorch/HF
        with _op("model.to_device"):
            if quantization_config is None:
                model = cast(Any, model).to(device)
            model.eval()

        # Set attention implementation to 'eager' if needed for attention outputs
        # Some models (like Llama) use SDPA by default which doesn't support output_attentions
        # We need 'eager' or 'flex_attention' to capture attention weights during generation
        # This is CoreVital-specific configuration
        with _op("_set_attention_implementation"):
            try:
                if hasattr(model, "set_attn_implementation"):
                    # Try to get current implementation from config
                    current_attn = getattr(model.config, "_attn_implementation", None)
                    # Also check for attn_implementation attribute directly
                    if current_attn is None:
                        current_attn = getattr(model.config, "attn_implementation", None)

                    # Compatible implementations: eager, flex_attention
                    compatible_implementations = ["eager", "flex_attention"]

                    # Only change if it's not already one of the compatible implementations
                    if current_attn not in compatible_implementations:
                        logger.info(
                            f"Setting attention implementation to 'eager' (current: {current_attn or 'default'})"
                        )
                        model.set_attn_implementation("eager")
                        # Log the final attention implementation
                        final_attn = getattr(model.config, "_attn_implementation", None) or getattr(
                            model.config, "attn_implementation", "eager"
                        )
                        logger.info(f"Attention implementation now set to: {final_attn}")
                    else:
                        logger.info(f"Attention implementation is compatible: {current_attn}")
                else:
                    # No explicit method, try to get the current implementation anyway
                    current_attn = getattr(model.config, "_attn_implementation", None) or getattr(
                        model.config, "attn_implementation", None
                    )
                    if current_attn:
                        logger.info(f"Current attention implementation: {current_attn}")
            except Exception as e:
                logger.warning(f"Could not set attention implementation to 'eager': {e}")
                # Continue anyway - some models might not support this or might already work

        # Probe whether model actually returns attentions (Issue 52)
        with _op("_probe_attentions"):
            attentions_available = _probe_attentions_available(
                model, tokenizer, device
            )
            capabilities = dataclasses.replace(
                capabilities, attentions_available=attentions_available
            )
            if not attentions_available:
                logger.warning(
                    "Attention capture probe: model did not return attentions "
                    "from a 1-token forward. Attention metrics will be omitted."
                )

        # Extract metadata (use model.config which is already loaded) - CoreVital logic
        with _op("_extract_metadata"):
            num_layers = cast(
                int,
                getattr(model_config, "num_hidden_layers", None)
                or getattr(model_config, "n_layer", None)
                or getattr(model_config, "num_layers", 0),
            )

            hidden_size = cast(
                int,
                getattr(model_config, "hidden_size", None) or getattr(model_config, "n_embd", None) or 0,
            )

            num_attention_heads = cast(
                int,
                getattr(model_config, "num_attention_heads", None) or getattr(model_config, "n_head", None) or 0,
            )

            architecture = model.__class__.__name__

            # Try to extract revision from model config
            # The revision might be stored in _commit_hash or similar attributes
            revision = config.model.revision
            if revision is None:
                # Try to get from model config
                revision = getattr(model_config, "_commit_hash", None)
                if revision is None:
                    # Try to get from tokenizer config
                    revision = getattr(tokenizer, "_commit_hash", None)

        # Detect actual dtype after model loading (important for quantized models) - CoreVital logic
        # Quantization changes the dtype AFTER loading, so we need to check the actual parameter dtypes
        dtype_str: Optional[str] = None
        if quantization_config is not None:
            with _op("_detect_quantized_dtype"):
                dtype_str = _detect_quantized_dtype(model, config.model.load_in_4bit, config.model.load_in_8bit)
                if dtype_str is not None:
                    logger.info(f"Detected quantized dtype: {dtype_str}")

        logger.info(f"Model loaded: {num_layers} layers, hidden_size={hidden_size}")
        if revision:
            logger.debug(f"Model revision: {revision}")

        return ModelBundle(
            model=model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            dtype_str=dtype_str,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            architecture=architecture,
            capabilities=capabilities,
            revision=revision,
            model_class=model_class,
        )

    except Exception as e:
        logger.exception("Failed to load model")
        raise ModelLoadError(f"Failed to load model '{config.model.hf_id}'", details=str(e)) from e


def _resolve_device(requested: str) -> torch.device:
    """
    Resolve device from requested string.

    Args:
        requested: Device string ('auto', 'cpu', 'cuda')

    Returns:
        torch.device object
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    elif requested == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    else:
        return torch.device(requested)


def _resolve_dtype(dtype_str: str, device: torch.device) -> torch.dtype:
    """
    Resolve dtype from string, considering device constraints.

    Args:
        dtype_str: Dtype string ('auto', 'float32', 'float16', 'bfloat16')
        device: Target device

    Returns:
        torch.dtype object
    """
    if dtype_str == "auto":
        if device.type == "cuda":
            return torch.float16
        return torch.float32
    elif dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    else:
        logger.warning(f"Unknown dtype '{dtype_str}', using float32")
        return torch.float32


def _detect_quantized_dtype(model: PreTrainedModel, is_4bit: bool, is_8bit: bool) -> Optional[str]:
    """
    Detect the actual dtype of quantized model parameters.

    Returns a human-readable string for the schema dtype field.
    If detection fails, returns "quantized_unknown" instead of guessing.
    False confidence is worse than honest ignorance.

    Args:
        model: Loaded model (potentially quantized)
        is_4bit: Whether 4-bit quantization was requested
        is_8bit: Whether 8-bit quantization was requested

    Returns:
        A dtype string like "int8", "uint8", "quantized_unknown", or None if not quantized
    """
    if not (is_4bit or is_8bit):
        return None

    try:
        # Check if quantization was actually applied via config
        if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
            quant_config = model.config.quantization_config

            if hasattr(quant_config, "quantization_method"):
                method = quant_config.quantization_method
                if method == "bitsandbytes":
                    # Inspect actual parameter dtypes
                    quantized_dtypes_found: set[torch.dtype] = set()

                    for _name, param in model.named_parameters():
                        if param.dtype in (torch.int8, torch.uint8):
                            quantized_dtypes_found.add(param.dtype)

                    # If we found quantized dtypes, report them
                    if quantized_dtypes_found:
                        if is_4bit and torch.uint8 in quantized_dtypes_found:
                            return "uint8"
                        elif is_8bit and torch.int8 in quantized_dtypes_found:
                            return "int8"
                        # Found something, just not what we expected
                        detected = next(iter(quantized_dtypes_found))
                        return str(detected).replace("torch.", "")

        # Quantization was requested but we could not verify the actual dtype.
        # Return "quantized_unknown" — honest ignorance beats false confidence.
        logger.warning(
            "Quantization requested but actual dtype could not be detected. Reporting as 'quantized_unknown'."
        )
        return "quantized_unknown"

    except Exception as e:
        logger.warning(f"Failed to detect quantized dtype: {e}")
        return "quantized_unknown"


# ============================================================================
# Test Harness
# ============================================================================


def _test_loader():
    """Test harness for model loading."""
    print("Testing HF Loader...")

    from CoreVital.config import Config

    # Test with small model
    config = Config()
    config.model.hf_id = "gpt2"
    config.device.requested = "cpu"

    bundle = load_model(config)
    print(f"✓ Model loaded: {bundle.architecture}")
    print(f"  Layers: {bundle.num_layers}")
    print(f"  Hidden size: {bundle.hidden_size}")
    print(f"  Device: {bundle.device}")
    print(f"  Dtype: {bundle.dtype}")

    assert bundle.model is not None
    assert bundle.tokenizer is not None
    assert bundle.num_layers > 0

    print("✓ All loader tests passed!\n")


if __name__ == "__main__":
    _test_loader()

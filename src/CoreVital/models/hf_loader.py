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
# ============================================================================

from dataclasses import dataclass
from typing import Optional, Any, Type
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
)

from CoreVital.config import Config
from CoreVital.errors import ModelLoadError
from CoreVital.logging_utils import get_logger


logger = get_logger(__name__)


@dataclass
class ModelBundle:
    """
    Container for model, tokenizer, and metadata.
    
    Attributes:
        model: Loaded HuggingFace model
        tokenizer: Loaded HuggingFace tokenizer
        device: Device the model is on
        dtype: Model dtype
        num_layers: Number of transformer layers
        hidden_size: Hidden state dimension
        num_attention_heads: Number of attention heads
        architecture: Model architecture name
        revision: Model revision/commit hash if available
        model_class: The model class type used for loading (AutoModelForCausalLM or AutoModelForSeq2SeqLM)
    """
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    device: torch.device
    dtype: torch.dtype
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    architecture: str
    revision: Optional[str] = None
    model_class: Type[PreTrainedModel] = AutoModelForCausalLM


def load_model(config: Config) -> ModelBundle:
    """
    Load a Hugging Face model and tokenizer with specified configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        ModelBundle with loaded model and metadata
        
    Raises:
        ModelLoadError: If model loading fails
    """
    try:
        logger.info(f"Loading model: {config.model.hf_id}")
        
        # Determine device
        device = _resolve_device(config.device.requested)
        logger.info(f"Target device: {device}")
        
        # Determine dtype
        dtype = _resolve_dtype(config.model.dtype, device)
        logger.info(f"Model dtype: {dtype}")
        
        # Load tokenizer
        logger.debug("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.hf_id,
            revision=config.model.revision,
            trust_remote_code=config.model.trust_remote_code,
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Inspect model type to determine which model class to use
        logger.debug("Inspecting model architecture...")
        model_config = AutoConfig.from_pretrained(
            config.model.hf_id,
            revision=config.model.revision,
            trust_remote_code=config.model.trust_remote_code,
        )
        
        # Determine if this is a Seq2Seq model
        model_type = getattr(model_config, 'model_type', '').lower()
        architectures = getattr(model_config, 'architectures', [])
        architecture_names = [arch.lower() if isinstance(arch, str) else '' for arch in architectures]
        
        # Common Seq2Seq model types
        seq2seq_model_types = {'t5', 'bart', 'mbart', 'pegasus', 'marian', 'blenderbot', 'm2m_100', 'nllb'}
        seq2seq_architecture_patterns = ['t5', 'bart', 'mbart', 'pegasus', 'marian', 'blenderbot', 'm2m', 'nllb']
        
        is_seq2seq = (
            model_type in seq2seq_model_types or
            any(pattern in arch for arch in architecture_names for pattern in seq2seq_architecture_patterns)
        )
        
        if is_seq2seq:
            logger.info(f"Detected Seq2Seq model (model_type: {model_type}, architectures: {architectures})")
            model_class = AutoModelForSeq2SeqLM
        else:
            logger.info(f"Detected CausalLM model (model_type: {model_type}, architectures: {architectures})")
            model_class = AutoModelForCausalLM
        
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
        
        # Load model with the appropriate class
        logger.debug(f"Loading model with {model_class.__name__}...")
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
        
        # Move to device (only if not using quantization, as quantization handles device placement)
        if quantization_config is None:
            model = model.to(device)
        model.eval()
        
        # Set attention implementation to 'eager' if needed for attention outputs
        # Some models (like Llama) use SDPA by default which doesn't support output_attentions
        # We need 'eager' attention to capture attention weights during generation
        try:
            if hasattr(model, 'set_attn_implementation'):
                # Try to get current implementation from config
                current_attn = getattr(model.config, '_attn_implementation', None)
                # Also check for attn_implementation attribute directly
                if current_attn is None:
                    current_attn = getattr(model.config, 'attn_implementation', None)
                
                # Only change if it's not already one of the compatible implementations
                if current_attn not in ['eager', 'eager_paged', 'flex_attention']:
                    logger.info(f"Setting attention implementation to 'eager' (current: {current_attn or 'default'})")
                    model.set_attn_implementation('eager')
                    logger.debug("Attention implementation set to 'eager' for attention output support")
                else:
                    logger.debug(f"Attention implementation already compatible: {current_attn}")
            else:
                logger.debug("Model does not support set_attn_implementation method")
        except Exception as e:
            logger.warning(f"Could not set attention implementation to 'eager': {e}")
            # Continue anyway - some models might not support this or might already work
        
        # Extract metadata (use model.config which is already loaded)
        num_layers = getattr(model_config, 'num_hidden_layers', None) or \
                    getattr(model_config, 'n_layer', None) or \
                    getattr(model_config, 'num_layers', 0)
        
        hidden_size = getattr(model_config, 'hidden_size', None) or \
                     getattr(model_config, 'n_embd', None) or 0
        
        num_attention_heads = getattr(model_config, 'num_attention_heads', None) or \
                             getattr(model_config, 'n_head', None) or 0
        
        architecture = model.__class__.__name__
        
        # Try to extract revision from model config
        # The revision might be stored in _commit_hash or similar attributes
        revision = config.model.revision
        if revision is None:
            # Try to get from model config
            revision = getattr(model_config, '_commit_hash', None)
            if revision is None:
                # Try to get from tokenizer config
                revision = getattr(tokenizer, '_commit_hash', None)
        
        logger.info(f"Model loaded: {num_layers} layers, hidden_size={hidden_size}")
        if revision:
            logger.debug(f"Model revision: {revision}")
        
        return ModelBundle(
            model=model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            architecture=architecture,
            revision=revision,
            model_class=model_class,
        )
        
    except Exception as e:
        logger.exception("Failed to load model")
        raise ModelLoadError(
            f"Failed to load model '{config.model.hf_id}'",
            details=str(e)
        ) from e


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
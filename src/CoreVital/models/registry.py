# ============================================================================
# CoreVital - Model Registry
#
# Purpose: Single source of truth for model type detection and capabilities
# Inputs: HuggingFace model config, loaded model
# Outputs: ModelCapabilities dataclass
# Dependencies: transformers
# Usage: caps = ModelCapabilities.from_config(config) or .from_model(model, config)
#
# Changelog:
#   2026-02-07: Initial ModelCapabilities for pre-phase-1 cleanup
#               Consolidates model type detection from hf_loader.py and collector.py
# ============================================================================

from dataclasses import dataclass
from typing import Literal, Optional

from CoreVital.logging_utils import get_logger

logger = get_logger(__name__)


# Known Seq2Seq model types (HuggingFace model_type strings)
_SEQ2SEQ_MODEL_TYPES = frozenset(
    {
        "t5",
        "bart",
        "mbart",
        "pegasus",
        "marian",
        "blenderbot",
        "blenderbot-small",
        "m2m_100",
        "nllb",
        "led",
        "bigbird_pegasus",
        "longt5",
        "plbart",
    }
)

# Architecture substrings that indicate Seq2Seq
_SEQ2SEQ_ARCHITECTURE_PATTERNS = (
    "t5",
    "bart",
    "mbart",
    "pegasus",
    "marian",
    "blenderbot",
    "m2m",
    "nllb",
    "led",
    "longt5",
    "plbart",
)

ModelFamily = Literal["causal_lm", "seq2seq"]
DetectionMethod = Literal[
    "config_is_encoder_decoder",
    "model_type_lookup",
    "architecture_pattern",
    "structural_inspection",
    "default_causal",
]


@dataclass(frozen=True)
class ModelCapabilities:
    """
    Single source of truth for model type and capabilities.

    Resolved once during model loading, stored in ModelBundle,
    and read by collector.py / report_builder.py — no detection
    logic anywhere else.
    """

    model_family: ModelFamily
    detection_method: DetectionMethod
    has_encoder: bool
    has_decoder: bool
    # Set after load by attention probe (Issue 52). None = not probed.
    attentions_available: Optional[bool] = None

    @property
    def is_seq2seq(self) -> bool:
        return self.model_family == "seq2seq"

    @property
    def is_causal_lm(self) -> bool:
        return self.model_family == "causal_lm"

    # ------------------------------------------------------------------
    # Factory: from HF config only (used during model loading)
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, model_config: object) -> "ModelCapabilities":
        """
        Detect model capabilities from a HuggingFace PretrainedConfig.

        Detection priority:
            1. config.is_encoder_decoder flag (most reliable)
            2. config.model_type in known Seq2Seq set
            3. Architecture name pattern matching
            4. Default to causal_lm
        """
        # --- 1. Explicit flag ---
        is_enc_dec = getattr(model_config, "is_encoder_decoder", None)
        if is_enc_dec is True:
            logger.info("Model detection: config.is_encoder_decoder=True → Seq2Seq")
            return cls(
                model_family="seq2seq",
                detection_method="config_is_encoder_decoder",
                has_encoder=True,
                has_decoder=True,
            )

        # --- 2. model_type lookup ---
        model_type = getattr(model_config, "model_type", "")
        if isinstance(model_type, str) and model_type.lower() in _SEQ2SEQ_MODEL_TYPES:
            logger.info(f"Model detection: model_type='{model_type}' in Seq2Seq set → Seq2Seq")
            return cls(
                model_family="seq2seq",
                detection_method="model_type_lookup",
                has_encoder=True,
                has_decoder=True,
            )

        # --- 3. Architecture pattern ---
        architectures = getattr(model_config, "architectures", []) or []
        arch_names = [a.lower() for a in architectures if isinstance(a, str)]
        for arch in arch_names:
            for pattern in _SEQ2SEQ_ARCHITECTURE_PATTERNS:
                if pattern in arch:
                    logger.info(f"Model detection: architecture '{arch}' matches pattern '{pattern}' → Seq2Seq")
                    return cls(
                        model_family="seq2seq",
                        detection_method="architecture_pattern",
                        has_encoder=True,
                        has_decoder=True,
                    )

        # --- 4. Default to CausalLM ---
        logger.info(
            f"Model detection: no Seq2Seq signals (model_type='{model_type}', architectures={architectures}) → CausalLM"
        )
        return cls(
            model_family="causal_lm",
            detection_method="default_causal",
            has_encoder=False,
            has_decoder=True,
        )

    # ------------------------------------------------------------------
    # Factory: from loaded model (used in collector as safety check)
    # ------------------------------------------------------------------
    @classmethod
    def from_model(
        cls,
        model: object,
        model_config: Optional[object] = None,
    ) -> "ModelCapabilities":
        """
        Detect model capabilities from a loaded model instance.

        Falls back to structural inspection (hasattr encoder/decoder)
        when config-based detection isn't enough. This is the collector's
        entry point — it gets the model object, not just config.
        """
        # Try config-based detection first
        config = model_config or getattr(model, "config", None)
        if config is not None:
            caps = cls.from_config(config)
            if caps.is_seq2seq:
                return caps

        # Structural fallback: check for real encoder + decoder attributes
        # Guard against Mock objects from test fixtures
        try:
            from unittest.mock import MagicMock, Mock

            def _is_real(attr: object) -> bool:
                return attr is not None and not isinstance(attr, (Mock, MagicMock)) and callable(attr)
        except ImportError:

            def _is_real(attr: object) -> bool:
                return attr is not None and callable(attr)

        encoder_attr = getattr(model, "encoder", None)
        decoder_attr = getattr(model, "decoder", None)
        has_real_encoder = _is_real(encoder_attr)
        has_real_decoder = _is_real(decoder_attr)

        if has_real_encoder and has_real_decoder:
            logger.info("Model detection: structural encoder+decoder → Seq2Seq")
            return cls(
                model_family="seq2seq",
                detection_method="structural_inspection",
                has_encoder=True,
                has_decoder=True,
            )

        # Default
        logger.info("Model detection: default → CausalLM")
        return cls(
            model_family="causal_lm",
            detection_method="default_causal",
            has_encoder=False,
            has_decoder=True,
        )

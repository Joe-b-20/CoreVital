# ============================================================================
# CoreVital - Configuration Management
#
# Purpose: Load and manage configuration from YAML, CLI args, and env vars
# Inputs: YAML files, environment variables
# Outputs: Config dataclass with all settings
# Dependencies: pyyaml, pydantic, pathlib
# Usage: config = Config.from_default() or Config.from_yaml("path.yaml")
#
# Changelog:
#   2026-01-13: Initial configuration system for Phase-0
#   2026-01-15: Added load_in_4bit and load_in_8bit flags to ModelConfig for quantization support
#   2026-02-04: Phase-0.75 - added PerformanceConfig for performance monitoring mode
#   2026-02-10: Phase-1b - added PromptTelemetryConfig (enabled, sparse_threshold)
#   2026-02-11: Phase-1d - extended SinkConfig with datadog_api_key, datadog_site,
#               prometheus_port for new sink types
# ============================================================================

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model loading configuration."""

    hf_id: str = "gpt2"
    revision: Optional[str] = None
    trust_remote_code: bool = False
    dtype: str = "auto"  # auto, float32, float16, bfloat16
    load_in_4bit: bool = False
    load_in_8bit: bool = False


class DeviceConfig(BaseModel):
    """Device configuration."""

    requested: str = "auto"  # auto, cpu, cuda


class GenerationConfig(BaseModel):
    """Generation parameters."""

    max_new_tokens: int = 20
    do_sample: bool = True
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    seed: int = 42
    # Beam search (CausalLM only; when num_beams > 1, do_sample is ignored by HF for standard beam search)
    num_beams: int = 1
    early_stopping: bool = False


class SketchConfig(BaseModel):
    """Sketch configuration for hidden state summaries."""

    enabled: bool = False  # Disabled by default to minimize payload; enable for debugging
    method: str = "randproj"
    dim: int = 32
    seed: int = 0


class HiddenSummariesConfig(BaseModel):
    """Hidden state summary configuration."""

    enabled: bool = True
    stats: List[str] = Field(default_factory=lambda: ["mean", "std", "l2_norm_mean", "max_abs"])
    sketch: SketchConfig = Field(default_factory=SketchConfig)


class AttentionSummariesConfig(BaseModel):
    """Attention summary configuration."""

    enabled: bool = True
    stats: List[str] = Field(
        default_factory=lambda: [
            "entropy_mean",
            "entropy_mean_normalized",
            "entropy_min",
            "entropy_max",
            "concentration_max",
            "concentration_min",
            "collapsed_head_count",
            "focused_head_count",
        ]
    )


class LogitsSummariesConfig(BaseModel):
    """Logits summary configuration."""

    enabled: bool = True
    stats: List[str] = Field(
        default_factory=lambda: [
            "entropy",
            "top_k_margin",
            "topk_mass",
            "topk_probs",
            "perplexity",
            "surprisal",
        ]
    )
    topk: int = 5
    entropy_mode: Literal["full", "topk_approx"] = "full"


class SummariesConfig(BaseModel):
    """All summaries configuration."""

    hidden: HiddenSummariesConfig = Field(default_factory=HiddenSummariesConfig)
    attention: AttentionSummariesConfig = Field(default_factory=AttentionSummariesConfig)
    logits: LogitsSummariesConfig = Field(default_factory=LogitsSummariesConfig)


class SinkConfig(BaseModel):
    """Sink configuration."""

    type: Literal["sqlite", "local_file", "datadog", "prometheus", "wandb"] = "sqlite"
    output_dir: str = "runs"
    remote_url: Optional[str] = None  # Legacy (http sink)
    datadog_api_key: Optional[str] = None
    datadog_site: str = "datadoghq.com"
    prometheus_port: int = 9091
    wandb_project: Optional[str] = None  # W&B project (or WANDB_PROJECT env)
    wandb_entity: Optional[str] = None  # W&B entity (or WANDB_ENTITY env)
    sqlite_path: str = "runs/corevital.db"  # Path to SQLite DB when type=sqlite
    sqlite_backup_json: bool = False  # When True, also write JSON file when using sqlite sink


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class PerformanceConfig(BaseModel):
    """Performance monitoring configuration."""

    mode: Optional[Literal["summary", "detailed", "strict"]] = None


class PromptTelemetryConfig(BaseModel):
    """Prompt telemetry configuration (Phase-1b).

    Controls whether a separate forward pass is run on prompt tokens
    to capture sparse attention profiles, basin scores, layer
    transformations, and prompt surprisal.
    """

    enabled: bool = True
    sparse_threshold: float = 0.01  # Store attention weights above this threshold
    sparse_max_per_head: Optional[int] = 50  # Cap stored connections per head to limit payload size


class CaptureConfig(BaseModel):
    """Capture mode: what to store per run (Foundation F2).

    - summary: Store only health flags, time series (entropy, perplexity, surprisal),
      and prompt analysis scalars; no per-layer data. Small payload.
    - full: Store everything (current behavior).
    - on_risk: Like summary, but when risk or health flags trigger, also store full trace.
      (Requires Phase-2 risk score; until then behaves like summary.)
    """

    capture_mode: Literal["summary", "full", "on_risk"] = "full"
    risk_threshold: float = 0.7  # For on_risk: store full trace when risk_score >= this


class OtelConfig(BaseModel):
    """OpenTelemetry export (optional). Install with: pip install CoreVital[otel]."""

    export_otel: bool = False
    otel_endpoint: Optional[str] = None  # e.g. http://localhost:4317 for OTLP gRPC; env OTEL_EXPORTER_OTLP_ENDPOINT


class ModelProfile(BaseModel):
    """Per-model threshold overrides for detection (Branch 4 / item #27).

    Different architectures have different L2 norms, entropy ranges, and anisotropy;
    these thresholds can be overridden per model family via configs/model_profiles/*.yaml.
    """

    l2_explosion_multiplier: float = 8.0  # Mid-layer L2 vs early-layer baseline (flan-t5 ~5.7x)
    high_entropy_threshold_bits: float = 4.0  # Steps with entropy > this count as high-entropy
    repetition_cosine_threshold: float = 0.9995  # Cosine sim above this = same direction (float16 anisotropy)
    collapsed_head_entropy_threshold: float = 0.1  # Head entropy below this = collapsed
    focused_head_concentration_threshold: float = 0.9  # Per-head max attn above this = focused


def _architecture_to_profile_key(architecture: str) -> str:
    """Map HuggingFace architecture string to profile file name (no extension)."""
    a = architecture or ""
    if "GPT2" in a:
        return "gpt2"
    if "Llama" in a:
        return "llama"
    if "Mistral" in a:
        return "mistral"
    if "T5" in a or "T5ForConditional" in a:
        return "t5"
    if "Bart" in a:
        return "bart"
    return "default"


def load_model_profile(
    architecture: str,
    base_path: Optional[Path] = None,
) -> ModelProfile:
    """Load model profile by architecture; fallback to default.yaml.

    Looks for configs/model_profiles/<key>.yaml then default.yaml.
    """
    if base_path is None:
        base_path = Path(__file__).resolve().parent.parent.parent / "configs" / "model_profiles"
    key = _architecture_to_profile_key(architecture)
    for name in (key, "default"):
        path = base_path / f"{name}.yaml"
        if path.exists():
            try:
                with open(path) as f:
                    data = yaml.safe_load(f) or {}
                return ModelProfile(**data)
            except Exception as e:
                warnings.warn(
                    f"Model profile file {path} exists but could not be parsed or validated: {e}. "
                    "Falling back to next profile or defaults; thresholds may not match operator intent.",
                    UserWarning,
                    stacklevel=2,
                )
    return ModelProfile()


class Config(BaseModel):
    """Root configuration object."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    # Optional per-model profile override; if None, loaded at runtime from architecture.
    model_profile: Optional[ModelProfile] = Field(default=None)
    # Optional RAG context (Foundation F3); set from CLI --rag-context or API. Not from YAML.
    rag_context: Optional[Dict[str, Any]] = Field(default=None)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    summaries: SummariesConfig = Field(default_factory=SummariesConfig)
    sink: SinkConfig = Field(default_factory=SinkConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    prompt_telemetry: PromptTelemetryConfig = Field(default_factory=PromptTelemetryConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    otel: OtelConfig = Field(default_factory=OtelConfig)
    # Path to a CalibrationProfile JSON (Issue 33). When set, report_builder
    # computes divergence scores alongside the heuristic risk score.
    calibration_profile: Optional[str] = Field(default=None)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """
        Load configuration from a YAML file with environment variable overrides.

        Args:
            path: Path to YAML configuration file

        Returns:
            Config instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        # Apply environment variable overrides
        data = cls._apply_env_overrides(data)

        return cls(**data)

    @classmethod
    def from_default(cls) -> "Config":
        """
        Load configuration from default config file.

        Returns:
            Config instance
        """
        # Find default config
        package_root = Path(__file__).parent.parent.parent
        default_config = package_root / "configs" / "default.yaml"

        if default_config.exists():
            return cls.from_yaml(str(default_config))
        else:
            # Return with defaults if file doesn't exist
            return cls()

    @classmethod
    def _apply_env_overrides(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.

        Environment variables follow the pattern:
        COREVITAL_<SECTION>_<KEY>=value

        Section names may contain underscores (e.g. prompt_telemetry), so we
        match against actual keys in the config data rather than naively
        splitting the env var name on ``_``.

        Examples:
            COREVITAL_DEVICE_REQUESTED=cuda      → data["device"]["requested"]
            COREVITAL_PROMPT_TELEMETRY_ENABLED=0  → data["prompt_telemetry"]["enabled"]

        Args:
            data: Configuration dictionary

        Returns:
            Updated configuration dictionary
        """
        prefix = "COREVITAL_"

        # Build sorted section keys (longest first so multi-word names match before
        # single-word prefixes, e.g. "prompt_telemetry" before "prompt")
        section_keys = sorted(data.keys(), key=len, reverse=True)

        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue

            remainder = env_key[len(prefix) :].lower()  # e.g. "prompt_telemetry_enabled"

            # Try to match a top-level section key
            for section in section_keys:
                section_prefix = section + "_"
                if not remainder.startswith(section_prefix):
                    continue

                rest = remainder[len(section_prefix) :]  # e.g. "enabled" or "sketch_dim"
                section_data = data.get(section)
                if not isinstance(section_data, dict):
                    break

                # rest could be a direct field ("enabled") or subsection_field ("sketch_dim")
                if rest in section_data:
                    section_data[rest] = cls._parse_env_value(env_value)
                else:
                    # Try one more level: rest = "subsection_field"
                    for sub_key in section_data:
                        sub_prefix = sub_key + "_"
                        if rest.startswith(sub_prefix) and isinstance(section_data[sub_key], dict):
                            field = rest[len(sub_prefix) :]
                            if field in section_data[sub_key]:
                                section_data[sub_key][field] = cls._parse_env_value(env_value)
                            break
                break  # matched a section, stop looking

        return data

    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """
        Parse environment variable value to appropriate type.

        Args:
            value: String value from environment

        Returns:
            Parsed value (str, int, float, or bool)
        """
        # Try boolean
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False

        # Try int
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value


# ============================================================================
# Test Harness
# ============================================================================


def _test_config():
    """Test harness for configuration loading."""
    print("Testing Config...")

    # Test default config
    config = Config.from_default()
    print(f"✓ Default config loaded: model={config.model.hf_id}")

    # Test individual values
    assert config.model.hf_id == "gpt2"
    assert config.generation.max_new_tokens == 20
    assert config.summaries.hidden.enabled is True
    print("✓ Default values correct")

    # Test env override
    os.environ["COREVITAL_DEVICE_REQUESTED"] = "cuda"
    config = Config.from_default()
    assert config.device.requested == "cuda"
    print("✓ Environment override works")
    del os.environ["COREVITAL_DEVICE_REQUESTED"]

    print("✓ All config tests passed!\n")


if __name__ == "__main__":
    _test_config()

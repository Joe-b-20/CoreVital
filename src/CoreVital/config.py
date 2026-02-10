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
# ============================================================================

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class SketchConfig(BaseModel):
    """Sketch configuration for hidden state summaries."""

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
            "top1_top2_margin",
            "topk_probs",
            "top_k_margin",
            "voter_agreement",
            "perplexity",
            "surprisal",
        ]
    )
    topk: int = 5


class SummariesConfig(BaseModel):
    """All summaries configuration."""

    hidden: HiddenSummariesConfig = Field(default_factory=HiddenSummariesConfig)
    attention: AttentionSummariesConfig = Field(default_factory=AttentionSummariesConfig)
    logits: LogitsSummariesConfig = Field(default_factory=LogitsSummariesConfig)


class SinkConfig(BaseModel):
    """Sink configuration."""

    type: str = "local_file"  # local_file, http
    output_dir: str = "runs"
    remote_url: Optional[str] = None


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class PerformanceConfig(BaseModel):
    """Performance monitoring configuration."""

    # Mode: None (disabled) | "summary" | "detailed" | "strict"
    mode: Optional[str] = None


class PromptTelemetryConfig(BaseModel):
    """Prompt telemetry configuration (Phase-1b).

    Controls whether a separate forward pass is run on prompt tokens
    to capture sparse attention profiles, basin scores, layer
    transformations, and prompt surprisal.
    """

    enabled: bool = True
    sparse_threshold: float = 0.01  # Store attention weights above this threshold


class Config(BaseModel):
    """Root configuration object."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    summaries: SummariesConfig = Field(default_factory=SummariesConfig)
    sink: SinkConfig = Field(default_factory=SinkConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    prompt_telemetry: PromptTelemetryConfig = Field(default_factory=PromptTelemetryConfig)

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

        Args:
            data: Configuration dictionary

        Returns:
            Updated configuration dictionary
        """
        prefix = "COREVITAL_"

        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue

            # Parse env var name
            parts = key[len(prefix) :].lower().split("_")

            if len(parts) == 2:
                section, field = parts
                if section in data and field in data[section]:
                    data[section][field] = cls._parse_env_value(value)
            elif len(parts) == 3:
                section, subsection, field = parts
                if section in data and subsection in data[section]:
                    data[section][subsection][field] = cls._parse_env_value(value)

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

# ============================================================================
# CoreVital - Model Profile Tests (Phase-4.2, Issues 2, 5, 17)
# ============================================================================

from pathlib import Path

import pytest

from CoreVital.config import (
    ModelProfile,
    _architecture_to_profile_key,
    load_model_profile,
)

PROFILES_DIR = Path(__file__).resolve().parent.parent / "configs" / "model_profiles"

ALL_PROFILE_NAMES = ["gpt2", "llama", "mistral", "mixtral", "qwen2", "phi3", "default"]


class TestProfileFilesExist:
    """Every expected profile YAML exists on disk."""

    @pytest.mark.parametrize("name", ALL_PROFILE_NAMES)
    def test_yaml_exists(self, name: str):
        assert (PROFILES_DIR / f"{name}.yaml").exists(), f"{name}.yaml missing"


class TestProfilesLoadAndValidate:
    """Each profile loads into a valid ModelProfile."""

    @pytest.mark.parametrize("name", ALL_PROFILE_NAMES)
    def test_loads_as_model_profile(self, name: str):
        import yaml

        path = PROFILES_DIR / f"{name}.yaml"
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        profile = ModelProfile(**data)
        assert isinstance(profile, ModelProfile)
        assert profile.high_entropy_threshold_bits > 0
        assert profile.l2_explosion_multiplier > 0
        assert profile.repetition_cosine_threshold > 0
        assert profile.collapsed_head_entropy_threshold >= 0
        assert profile.focused_head_concentration_threshold > 0


class TestProfilesDiffer:
    """Issue 2: gpt2 and llama must have different thresholds."""

    def test_gpt2_vs_llama_entropy_threshold(self):
        gpt2 = load_model_profile("GPT2LMHeadModel", base_path=PROFILES_DIR)
        llama = load_model_profile("LlamaForCausalLM", base_path=PROFILES_DIR)
        assert gpt2.high_entropy_threshold_bits != llama.high_entropy_threshold_bits

    def test_gpt2_vs_llama_l2_explosion(self):
        gpt2 = load_model_profile("GPT2LMHeadModel", base_path=PROFILES_DIR)
        llama = load_model_profile("LlamaForCausalLM", base_path=PROFILES_DIR)
        assert gpt2.l2_explosion_multiplier != llama.l2_explosion_multiplier

    def test_gpt2_vs_llama_repetition_threshold(self):
        gpt2 = load_model_profile("GPT2LMHeadModel", base_path=PROFILES_DIR)
        llama = load_model_profile("LlamaForCausalLM", base_path=PROFILES_DIR)
        assert gpt2.repetition_cosine_threshold != llama.repetition_cosine_threshold

    def test_gpt2_vs_llama_collapsed_head(self):
        gpt2 = load_model_profile("GPT2LMHeadModel", base_path=PROFILES_DIR)
        llama = load_model_profile("LlamaForCausalLM", base_path=PROFILES_DIR)
        assert gpt2.collapsed_head_entropy_threshold != llama.collapsed_head_entropy_threshold

    def test_all_profiles_not_identical(self):
        """No two calibrated profiles should have identical threshold tuples."""
        profiles = {}
        for name in ALL_PROFILE_NAMES:
            import yaml

            path = PROFILES_DIR / f"{name}.yaml"
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            p = ModelProfile(**data)
            profiles[name] = (
                p.l2_explosion_multiplier,
                p.high_entropy_threshold_bits,
                p.repetition_cosine_threshold,
                p.collapsed_head_entropy_threshold,
                p.focused_head_concentration_threshold,
            )
        values = list(profiles.values())
        unique = set(values)
        assert len(unique) == len(values), (
            f"Some profiles are identical: {profiles}"
        )


class TestArchitectureMapping:
    """_architecture_to_profile_key maps HF architecture strings correctly."""

    @pytest.mark.parametrize(
        "arch, expected",
        [
            ("GPT2LMHeadModel", "gpt2"),
            ("LlamaForCausalLM", "llama"),
            ("MistralForCausalLM", "mistral"),
            ("MixtralForCausalLM", "mixtral"),
            ("Qwen2ForCausalLM", "qwen2"),
            ("Phi3ForCausalLM", "phi3"),
            ("T5ForConditionalGeneration", "t5"),
            ("BartForConditionalGeneration", "bart"),
            ("UnknownModel", "default"),
            ("", "default"),
        ],
    )
    def test_mapping(self, arch: str, expected: str):
        assert _architecture_to_profile_key(arch) == expected


class TestLoadModelProfile:
    """load_model_profile resolves architecture → YAML → ModelProfile."""

    def test_gpt2_loads_specific_values(self):
        p = load_model_profile("GPT2LMHeadModel", base_path=PROFILES_DIR)
        assert p.high_entropy_threshold_bits == 5.0
        assert p.l2_explosion_multiplier == 5.0

    def test_llama_loads_specific_values(self):
        p = load_model_profile("LlamaForCausalLM", base_path=PROFILES_DIR)
        assert p.high_entropy_threshold_bits == 3.5
        assert p.l2_explosion_multiplier == 10.0

    def test_unknown_architecture_falls_back_to_default(self):
        p = load_model_profile("TotallyNewModel", base_path=PROFILES_DIR)
        assert p.high_entropy_threshold_bits == 4.0

    def test_mistral_loads(self):
        p = load_model_profile("MistralForCausalLM", base_path=PROFILES_DIR)
        assert isinstance(p, ModelProfile)
        assert p.high_entropy_threshold_bits == 4.0

    def test_mixtral_loads(self):
        p = load_model_profile("MixtralForCausalLM", base_path=PROFILES_DIR)
        assert isinstance(p, ModelProfile)
        assert p.high_entropy_threshold_bits == 4.5

    def test_qwen2_loads(self):
        p = load_model_profile("Qwen2ForCausalLM", base_path=PROFILES_DIR)
        assert isinstance(p, ModelProfile)
        assert p.high_entropy_threshold_bits == 4.5

    def test_phi3_loads(self):
        p = load_model_profile("Phi3ForCausalLM", base_path=PROFILES_DIR)
        assert isinstance(p, ModelProfile)
        assert p.high_entropy_threshold_bits == 3.8


class TestTypicalRanges:
    """Calibrated profiles include typical_entropy_range and typical_l2_norm_range."""

    def test_gpt2_has_ranges(self):
        p = load_model_profile("GPT2LMHeadModel", base_path=PROFILES_DIR)
        assert p.typical_entropy_range is not None
        assert len(p.typical_entropy_range) == 2
        assert p.typical_entropy_range[0] < p.typical_entropy_range[1]

    def test_llama_has_ranges(self):
        p = load_model_profile("LlamaForCausalLM", base_path=PROFILES_DIR)
        assert p.typical_l2_norm_range is not None
        assert len(p.typical_l2_norm_range) == 2

    def test_uncalibrated_profile_has_no_ranges(self):
        p = load_model_profile("MistralForCausalLM", base_path=PROFILES_DIR)
        assert p.typical_entropy_range is None


class TestEntropyThresholdWiring:
    """Issue 5: high_entropy_threshold_bits from profile flows into health flag computation."""

    def test_profile_threshold_used_in_health_flags(self):
        """Simulates what report_builder._build_health_flags does."""
        profile = load_model_profile("GPT2LMHeadModel", base_path=PROFILES_DIR)
        high_entropy_threshold = 4.0
        if profile is not None and hasattr(profile, "high_entropy_threshold_bits"):
            high_entropy_threshold = float(profile.high_entropy_threshold_bits)
        assert high_entropy_threshold == 5.0, "GPT-2 threshold should override default 4.0"

    def test_llama_threshold_lower_than_gpt2(self):
        gpt2 = load_model_profile("GPT2LMHeadModel", base_path=PROFILES_DIR)
        llama = load_model_profile("LlamaForCausalLM", base_path=PROFILES_DIR)
        assert llama.high_entropy_threshold_bits < gpt2.high_entropy_threshold_bits

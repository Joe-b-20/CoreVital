# ============================================================================
# CoreVital - Test Configuration and Fixtures
#
# Purpose: Shared pytest fixtures for testing
# Inputs: None
# Outputs: Fixtures for use in tests
# Dependencies: pytest, unittest.mock, torch, transformers
# Usage: pytest tests/ (fixtures are automatically available)
#
# Changelog:
#   2026-01-16: Initial mock model bundle fixture for testing instrumentation
#   2026-01-21: Phase-0.5 hardening - added is_encoder_decoder=True to MockSeq2SeqModel config
#   2026-02-10: Phase-1b - added mock forward pass (__call__) for CausalLM prompt telemetry
# ============================================================================

from unittest.mock import Mock

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from CoreVital.models.hf_loader import ModelBundle
from CoreVital.models.registry import ModelCapabilities


@pytest.fixture
def mock_model_bundle(request):
    """
    Create a mock ModelBundle with Mock model and tokenizer.

    Supports both CausalLM and Seq2Seq models based on the 'model_type' marker.
    Usage:
        @pytest.mark.parametrize('model_type', ['causal', 'seq2seq'])
        def test_something(mock_model_bundle, model_type):
            ...

    Or use directly:
        def test_something(mock_model_bundle):
            # Defaults to causal
            ...
    """
    # Check if model_type is specified via marker or parameter
    model_type = "causal"  # default
    if hasattr(request, "param"):
        model_type = request.param
    elif hasattr(request.node, "get_closest_marker"):
        marker = request.node.get_closest_marker("model_type")
        if marker:
            model_type = marker.args[0] if marker.args else "causal"

    # Model configuration
    num_layers = 4
    hidden_size = 128
    num_attention_heads = 8
    vocab_size = 50257  # GPT-2 vocab size
    batch_size = 1

    # Create a mock return value that supports both dict unpacking and .to(device)
    class TokenizerOutput:
        def __init__(self):
            self.input_ids = torch.tensor([[1, 2, 3, 4]])  # 4 prompt tokens
            self.attention_mask = torch.tensor([[1, 1, 1, 1]])

        def to(self, device):
            self.input_ids = self.input_ids.to(device)
            self.attention_mask = self.attention_mask.to(device)
            return self

        def __getitem__(self, key):
            # Support dict-like access for **inputs unpacking
            if key == "input_ids":
                return self.input_ids
            elif key == "attention_mask":
                return self.attention_mask
            raise KeyError(key)

        def keys(self):
            # Support dict-like iteration
            return ["input_ids", "attention_mask"].__iter__()

    # Create mock tokenizer - use a custom class to avoid Mock issues
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 2

        def __call__(self, *args, **kwargs):
            # Return a fresh TokenizerOutput instance each time
            return TokenizerOutput()

        def decode(self, tokens, **kwargs):
            return f"token_{tokens[0] if isinstance(tokens, list) else tokens}"

    mock_tokenizer = MockTokenizer()

    # Define output classes that can be used by both model types
    class EncoderOutput:
        def __init__(self, hidden_states, attentions, last_hidden_state):
            self.hidden_states = hidden_states
            self.attentions = attentions
            self.last_hidden_state = last_hidden_state

    class DecoderOutput:
        def __init__(self, logits, decoder_hidden_states, decoder_attentions, cross_attentions):
            self.logits = logits
            self.decoder_hidden_states = decoder_hidden_states
            self.decoder_attentions = decoder_attentions
            self.cross_attentions = cross_attentions

    if model_type == "seq2seq":
        # Seq2Seq model (T5, BART, etc.) - use a real class instead of Mock
        class MockSeq2SeqModel:
            def __init__(self):
                self.config = Mock()
                self.config.decoder_start_token_id = 0
                self.config.eos_token_id = 2
                self.config.is_encoder_decoder = True  # Explicitly mark as seq2seq
                # Don't set __class__ as it causes issues with attribute access
                # Instead, we'll rely on isinstance checks in the collector

            def eval(self):
                return self

            def to(self, device):
                return self

            def encoder(self, input_ids, output_hidden_states=True, output_attentions=True, return_dict=True, **kwargs):
                encoder_seq_len = input_ids.shape[1]

                # Create encoder hidden states (skip embedding, include layers)
                encoder_hidden_states = tuple(
                    [torch.randn(batch_size, encoder_seq_len, hidden_size) for _ in range(num_layers)]
                )

                # Create encoder attentions
                encoder_attentions = tuple(
                    [
                        torch.randn(batch_size, num_attention_heads, encoder_seq_len, encoder_seq_len)
                        for _ in range(num_layers)
                    ]
                )

                return EncoderOutput(
                    hidden_states=(torch.randn(batch_size, encoder_seq_len, hidden_size),) + encoder_hidden_states,
                    attentions=encoder_attentions,
                    last_hidden_state=encoder_hidden_states[-1],
                )

            def __call__(
                self,
                input_ids=None,
                encoder_outputs=None,
                decoder_input_ids=None,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
                **kwargs,
            ):
                # Determine sequence length
                if decoder_input_ids is not None:
                    decoder_seq_len = decoder_input_ids.shape[1]
                else:
                    decoder_seq_len = 1

                # Create logits
                logits = torch.randn(batch_size, decoder_seq_len, vocab_size)

                # Create decoder hidden states (skip embedding, include layers)
                decoder_hidden_states = tuple(
                    [torch.randn(batch_size, decoder_seq_len, hidden_size) for _ in range(num_layers)]
                )

                # Create decoder self-attentions
                decoder_attentions = tuple(
                    [
                        (torch.randn(batch_size, num_attention_heads, decoder_seq_len, decoder_seq_len),)
                        for _ in range(num_layers)
                    ]
                )

                # Create cross-attentions (decoder attending to encoder)
                if encoder_outputs is not None and hasattr(encoder_outputs, "last_hidden_state"):
                    encoder_seq_len = encoder_outputs.last_hidden_state.shape[1]
                else:
                    encoder_seq_len = 4  # default

                cross_attentions = tuple(
                    [
                        torch.randn(batch_size, num_attention_heads, decoder_seq_len, encoder_seq_len)
                        for _ in range(num_layers)
                    ]
                )

                return DecoderOutput(
                    logits=logits,
                    decoder_hidden_states=(torch.randn(batch_size, decoder_seq_len, hidden_size),)
                    + decoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                )

        mock_model = MockSeq2SeqModel()

    else:
        # CausalLM model (GPT-2, etc.)
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.decoder_start_token_id = 0
        mock_model.config.eos_token_id = 2
        mock_model.eval = Mock(return_value=mock_model)
        mock_model.to = Mock(return_value=mock_model)
        mock_model.__class__ = AutoModelForCausalLM

        # Mock generate() method
        def mock_generate(
            input_ids,
            max_new_tokens=5,
            output_hidden_states=True,
            output_attentions=True,
            output_scores=True,
            return_dict_in_generate=True,
            **kwargs,
        ):
            prompt_len = input_ids.shape[1]

            # Create sequences
            sequences = torch.cat([input_ids, torch.randint(0, vocab_size, (batch_size, max_new_tokens))], dim=1)

            # Create scores (logits) for each generation step
            scores = tuple([torch.randn(batch_size, vocab_size) for _ in range(max_new_tokens)])

            # Create hidden states for each generation step (match real HF: embedding + layer_1..layer_N)
            # Each step has a tuple of (embedding_output, layer_1, ..., layer_N) = num_layers + 1 elements
            hidden_states = tuple(
                [
                    tuple(
                        [
                            torch.randn(batch_size, 1, hidden_size)  # seq_len=1 for each generation step
                            for _ in range(num_layers + 1)
                        ]
                    )
                    for _ in range(max_new_tokens)
                ]
            )

            # Create attentions for each generation step
            # Each step has a tuple of layer attention tensors
            attentions = tuple(
                [
                    tuple(
                        [
                            torch.randn(batch_size, num_attention_heads, 1, prompt_len + step_idx + 1)
                            for _ in range(num_layers)
                        ]
                    )
                    for step_idx in range(max_new_tokens)
                ]
            )

            # Create output object similar to GenerateDecoderOnlyOutput
            output = Mock()
            output.sequences = sequences
            output.scores = scores
            output.hidden_states = hidden_states
            output.attentions = attentions
            return output

        mock_model.generate = Mock(side_effect=mock_generate)

        # Mock forward pass (__call__) for prompt telemetry (Phase-1b)
        # Returns CausalLMOutput-like object with hidden_states, attentions, logits
        def mock_forward(
            input_ids=None,
            attention_mask=None,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
            **kwargs,
        ):
            seq_len = input_ids.shape[1] if input_ids is not None else 4

            # hidden_states: (embedding, layer1, ..., layerN)
            hs = tuple(torch.randn(batch_size, seq_len, hidden_size) for _ in range(num_layers + 1))
            # attentions: (layer1, ..., layerN) — full seq×seq matrices
            attn = tuple(
                torch.softmax(torch.randn(batch_size, num_attention_heads, seq_len, seq_len), dim=-1)
                for _ in range(num_layers)
            )
            # logits: (batch, seq_len, vocab_size)
            logits = torch.randn(batch_size, seq_len, vocab_size)

            forward_out = Mock()
            forward_out.hidden_states = hs
            forward_out.attentions = attn
            forward_out.logits = logits
            return forward_out

        mock_model.side_effect = mock_forward

    # Build capabilities from model type
    if model_type == "seq2seq":
        capabilities = ModelCapabilities(
            model_family="seq2seq",
            detection_method="config_is_encoder_decoder",
            has_encoder=True,
            has_decoder=True,
        )
    else:
        capabilities = ModelCapabilities(
            model_family="causal_lm",
            detection_method="default_causal",
            has_encoder=False,
            has_decoder=True,
        )

    # Create ModelBundle
    bundle = ModelBundle(
        model=mock_model,
        tokenizer=mock_tokenizer,
        device=torch.device("cpu"),
        dtype=torch.float32,
        dtype_str=None,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        architecture=f"Mock{model_type.capitalize()}Model",
        capabilities=capabilities,
        revision=None,
        model_class=AutoModelForSeq2SeqLM if model_type == "seq2seq" else AutoModelForCausalLM,
    )

    return bundle


@pytest.fixture
def mock_config():
    """Create a mock Config object for testing."""
    from CoreVital.config import Config

    config = Config()
    config.model.hf_id = "mock-model"
    config.device.requested = "cpu"
    config.generation.max_new_tokens = 5
    config.generation.seed = 42
    config.generation.do_sample = False  # Greedy decoding for deterministic tests
    config.sink.output_dir = "/tmp"

    return config

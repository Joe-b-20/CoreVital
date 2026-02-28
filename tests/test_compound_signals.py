# ============================================================================
# CoreVital - Compound Signal Detection Tests (Phase-2, Issue 6)
# ============================================================================

import pytest

from CoreVital.compound_signals import CompoundSignal, detect_compound_signals
from CoreVital.reporting.schema import (
    AttentionSummary,
    HealthFlags,
    HiddenSummary,
    LayerSummary,
    LogitsSummary,
    Summary,
    TensorAnomalies,
    TimelineStep,
    TokenInfo,
)
from CoreVital.risk import compute_risk_score


def _step(
    step_index: int,
    entropy: float = None,
    top_k_margin: float = None,
    voter_agreement: float = None,
    surprisal: float = None,
) -> TimelineStep:
    """Build a minimal TimelineStep with optional logits metrics."""
    logits = LogitsSummary()
    if entropy is not None:
        logits.entropy = entropy
    if top_k_margin is not None:
        logits.top_k_margin = top_k_margin
    if voter_agreement is not None:
        logits.voter_agreement = voter_agreement
    if surprisal is not None:
        logits.surprisal = surprisal
    return TimelineStep(
        step_index=step_index,
        token=TokenInfo(token_id=0, token_text="x", is_prompt_token=False),
        logits_summary=logits,
    )


def _layer(layer_index: int, collapsed_head_count: int = 0) -> LayerSummary:
    """Build a minimal LayerSummary with optional attention collapse."""
    attn = AttentionSummary(collapsed_head_count=collapsed_head_count)
    return LayerSummary(
        layer_index=layer_index,
        hidden_summary=HiddenSummary(),
        attention_summary=attn,
    )


class TestDetectCompoundSignals:
    """Unit tests for detect_compound_signals."""

    def test_healthy_trace_no_signals(self):
        """No patterns should fire on a healthy trace (low entropy, good margin, no collapse)."""
        timeline = [
            _step(i, entropy=1.5, top_k_margin=0.6, voter_agreement=0.7, surprisal=0.5)
            for i in range(12)
        ]
        layers_by_step = [[_layer(j) for j in range(3)] for _ in range(12)]
        basin_scores = [[0.6, 0.5, 0.55]]  # one list per layer, decent basin
        signals = detect_compound_signals(
            timeline, layers_by_step=layers_by_step, basin_scores=basin_scores
        )
        assert len(signals) == 0

    def test_context_loss_fires(self):
        """Context loss: high entropy (last 5) + low mean basin score."""
        timeline = [
            _step(i, entropy=5.0, top_k_margin=0.2) for i in range(6)
        ]  # last 5 steps mean entropy 5.0
        basin_scores = [[0.1, 0.15, 0.2]]  # mean basin 0.15 < 0.3
        signals = detect_compound_signals(timeline, basin_scores=basin_scores)
        names = [s.name for s in signals]
        assert "context_loss" in names
        ctx = next(s for s in signals if s.name == "context_loss")
        assert ctx.severity == 0.75
        assert "entropy" in ctx.evidence[0].lower()
        assert "basin" in ctx.evidence[1].lower()

    def test_confident_confusion_fires(self):
        """Confident confusion: high entropy + high margin (last 5)."""
        timeline = [
            _step(i, entropy=4.5, top_k_margin=0.5) for i in range(6)
        ]
        signals = detect_compound_signals(timeline)
        names = [s.name for s in signals]
        assert "confident_confusion" in names
        cc = next(s for s in signals if s.name == "confident_confusion")
        assert cc.severity == 0.5

    def test_degenerating_generation_fires(self):
        """Degenerating generation: 10+ steps, entropy/margin/surprisal slopes meet thresholds."""
        # Entropy rising (~0.5/step), margin falling (~-0.05/step), surprisal rising (~0.5/step)
        timeline = [
            _step(
                i,
                entropy=2.0 + i * 0.6,
                top_k_margin=0.5 - i * 0.05,
                surprisal=0.5 + i * 0.6,
            )
            for i in range(10)
        ]
        signals = detect_compound_signals(timeline)
        names = [s.name for s in signals]
        assert "degenerating_generation" in names
        dg = next(s for s in signals if s.name == "degenerating_generation")
        assert dg.severity == 0.7

    def test_attention_bottleneck_fires(self):
        """Attention bottleneck: high collapse rate across layer-steps + high mean entropy."""
        timeline = [_step(i, entropy=4.0) for i in range(5)]
        # Many layers with collapsed heads so collapse_rate > 0.2
        layers_by_step = [
            [_layer(j, collapsed_head_count=3 if j < 2 else 0) for j in range(4)]
            for _ in range(5)
        ]
        # total_collapsed = 5*2*3 = 30, total_heads_checked = 5*4 = 20, rate = 1.5 > 0.2
        signals = detect_compound_signals(
            timeline, layers_by_step=layers_by_step
        )
        names = [s.name for s in signals]
        assert "attention_bottleneck" in names
        ab = next(s for s in signals if s.name == "attention_bottleneck")
        assert ab.severity == 0.65

    def test_confident_repetition_risk_fires(self):
        """Confident repetition risk: low entropy + low surprisal + very high voter agreement."""
        timeline = [
            _step(i, entropy=1.0, surprisal=0.5, voter_agreement=0.98)
            for i in range(6)
        ]
        signals = detect_compound_signals(timeline)
        names = [s.name for s in signals]
        assert "confident_repetition_risk" in names
        cr = next(s for s in signals if s.name == "confident_repetition_risk")
        assert cr.severity == 0.4

    def test_empty_timeline_returns_empty(self):
        """Empty timeline should return no signals."""
        assert detect_compound_signals([]) == []
        assert detect_compound_signals([], basin_scores=[[0.1]]) == []

    def test_context_loss_requires_basin_scores(self):
        """Context loss does not fire without basin_scores even if entropy is high."""
        timeline = [_step(i, entropy=5.0) for i in range(6)]
        signals = detect_compound_signals(timeline, basin_scores=None)
        assert not any(s.name == "context_loss" for s in signals)


class TestCompoundSignalsInRiskScore:
    """Compound signals should feed into risk score and factors."""

    def test_compound_signals_add_to_risk_factors(self):
        """When compound signals are passed, risk score and factors include them."""
        summary = Summary(prompt_tokens=0, generated_tokens=5, total_steps=5, elapsed_ms=100)
        flags = HealthFlags()
        timeline = [_step(i, entropy=1.0, top_k_margin=0.6) for i in range(5)]
        compound_signals = [
            CompoundSignal(
                name="context_loss",
                description="Test",
                severity=0.5,
                evidence=["e1", "e2"],
            )
        ]
        score, factors = compute_risk_score(
            flags,
            summary,
            timeline=timeline,
            compound_signals=compound_signals,
        )
        assert "compound:context_loss" in factors
        assert score >= 0.5

    def test_risk_without_compound_signals_unchanged(self):
        """When compound_signals is None, behavior is unchanged (no compound factors)."""
        summary = Summary(prompt_tokens=0, generated_tokens=5, total_steps=5, elapsed_ms=100)
        flags = HealthFlags()
        timeline = [_step(i, entropy=1.0) for i in range(5)]
        score, factors = compute_risk_score(
            flags, summary, timeline=timeline, compound_signals=None
        )
        assert not any(f.startswith("compound:") for f in factors)

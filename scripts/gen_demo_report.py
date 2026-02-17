#!/usr/bin/env python3
"""Generate docs/demo/sample_report.json for the dashboard demo.

Run from repo root:  python scripts/gen_demo_report.py

Produces a realistic GPT-2 report with all 12 layers, real token texts,
full metric summaries, prompt analysis, and all extensions populated so
every dashboard section lights up.
"""

import hashlib
import math
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from CoreVital.reporting.schema import (  # noqa: E402
    AttentionConfig,
    AttentionSummary,
    GeneratedInfo,
    GenerationConfig,
    HealthFlags,
    HiddenConfig,
    HiddenSummary,
    LayerSummary,
    LogitsConfig,
    LogitsSummary,
    ModelInfo,
    PromptAnalysis,
    PromptAttentionLayer,
    PromptInfo,
    QuantizationInfo,
    Report,
    RunConfig,
    SinkConfig,
    SketchConfig,
    SparseAttentionHead,
    SummariesConfig,
    Summary,
    TimelineStep,
    TokenInfo,
    TopKItem,
)
from CoreVital.utils.serialization import serialize_report_to_json  # noqa: E402

NUM_LAYERS = 12
NUM_HEADS = 12

PROMPT_TEXT = "Explain why the sky is blue in one sentence."
PROMPT_TOKENS = [
    (18438, "Explain"),
    (1521, " why"),
    (262, " the"),
    (6766, " sky"),
    (318, " is"),
    (4171, " blue"),
    (287, " in"),
    (530, " one"),
    (6827, " sentence"),
    (13, "."),
]

GENERATED_TEXT = (
    " The sky appears blue because molecules in the atmosphere"
    " scatter shorter blue wavelengths of sunlight more than other colors."
)
GENERATED_TOKENS = [
    (383, " The"),
    (6766, " sky"),
    (3568, " appears"),
    (4171, " blue"),
    (780, " because"),
    (17287, " molecules"),
    (287, " in"),
    (262, " the"),
    (8137, " atmosphere"),
    (41058, " scatter"),
    (12238, " shorter"),
    (4171, " blue"),
    (28655, " wavelengths"),
    (286, " of"),
    (19606, " sunlight"),
    (517, " more"),
    (621, " than"),
    (584, " other"),
    (7577, " colors"),
    (13, "."),
]


def _entropy_to_perplexity(entropy: float) -> float:
    return math.pow(2, entropy)


def _build_layers(step_idx: int) -> list[LayerSummary]:
    """Build 12 layer summaries with realistic per-layer variation."""
    layers = []
    for li in range(NUM_LAYERS):
        depth_factor = (li + 1) / NUM_LAYERS
        base_l2 = 8.0 + depth_factor * 40.0 + (step_idx % 3) * 0.5
        attn_entropy = 2.5 + depth_factor * 1.0 + (step_idx * 0.07) - (li * 0.04)
        attn_entropy = max(0.3, attn_entropy)
        concentration = 0.15 + (1.0 - depth_factor) * 0.25 + (0.02 * step_idx)
        concentration = min(0.95, concentration)
        collapsed = 1 if (li == 4 and step_idx >= 15) else 0
        focused = 1 if (li >= 10 and concentration > 0.5) else 0

        layers.append(
            LayerSummary(
                layer_index=li,
                hidden_summary=HiddenSummary(
                    mean=round(0.001 + depth_factor * 0.003 + step_idx * 0.0001, 6),
                    std=round(0.5 + depth_factor * 1.5 + step_idx * 0.01, 4),
                    l2_norm_mean=round(base_l2, 2),
                    max_abs=round(base_l2 * 0.35 + li * 0.8, 2),
                ),
                attention_summary=AttentionSummary(
                    entropy_mean=round(attn_entropy, 4),
                    entropy_min=round(attn_entropy * 0.4, 4),
                    entropy_max=round(attn_entropy * 1.6, 4),
                    concentration_max=round(concentration, 4),
                    concentration_min=round(concentration * 0.3, 4),
                    collapsed_head_count=collapsed,
                    focused_head_count=focused,
                ),
            )
        )
    return layers


def _build_timeline() -> list[TimelineStep]:
    """Build timeline with realistic entropy/perplexity/surprisal curves."""
    entropies = [
        2.1,
        1.8,
        2.5,
        1.9,
        2.2,
        3.8,
        2.9,
        1.7,
        1.5,
        2.0,
        4.5,
        3.2,
        2.8,
        1.4,
        1.9,
        2.1,
        1.6,
        1.3,
        2.4,
        1.1,
    ]
    surprisals = [
        3.2,
        1.5,
        4.1,
        1.8,
        2.0,
        5.8,
        3.5,
        1.2,
        1.0,
        2.3,
        6.2,
        4.0,
        3.1,
        0.9,
        1.7,
        2.5,
        1.3,
        0.8,
        2.9,
        0.6,
    ]
    top1_margins = [
        0.45,
        0.62,
        0.31,
        0.58,
        0.42,
        0.12,
        0.28,
        0.65,
        0.71,
        0.48,
        0.08,
        0.22,
        0.33,
        0.75,
        0.55,
        0.40,
        0.68,
        0.78,
        0.35,
        0.82,
    ]
    steps = []
    for i, (tok_id, tok_text) in enumerate(GENERATED_TOKENS):
        ent = entropies[i]
        perp = round(_entropy_to_perplexity(ent), 4)
        surp = surprisals[i]
        margin = top1_margins[i]

        topk_items = [
            TopKItem(token_id=tok_id, token_text=tok_text, prob=round(0.3 + margin * 0.4, 4)),
            TopKItem(token_id=tok_id + 1, token_text="alt1", prob=round(0.15 - margin * 0.05, 4)),
            TopKItem(token_id=tok_id + 2, token_text="alt2", prob=round(0.08, 4)),
            TopKItem(token_id=tok_id + 3, token_text="alt3", prob=round(0.05, 4)),
            TopKItem(token_id=tok_id + 4, token_text="alt4", prob=round(0.03, 4)),
        ]

        steps.append(
            TimelineStep(
                step_index=i,
                token=TokenInfo(token_id=tok_id, token_text=tok_text, is_prompt_token=False),
                logits_summary=LogitsSummary(
                    entropy=round(ent, 4),
                    perplexity=perp,
                    surprisal=round(surp, 4),
                    top1_top2_margin=round(margin, 4),
                    top_k_margin=round(margin * 0.8, 4),
                    voter_agreement=round(0.5 + margin * 0.4, 4),
                    topk=topk_items,
                ),
                layers=_build_layers(i),
            )
        )
    return steps


def _build_prompt_analysis() -> PromptAnalysis:
    """Build prompt analysis with layer transformations, surprisals, and sparse attention."""
    transformations = [round(0.35 + (i * 0.03) - (0.01 * (i % 3)), 4) for i in range(NUM_LAYERS - 1)]
    surprisals = [round(8.5 - i * 0.6 + (0.3 if i == 4 else 0), 4) for i in range(len(PROMPT_TOKENS))]

    pa_layers = []
    for li in range(NUM_LAYERS):
        heads = []
        for hi in range(min(NUM_HEADS, 4)):
            n_connections = 5
            heads.append(
                SparseAttentionHead(
                    query_indices=list(range(n_connections)),
                    key_indices=[(hi + c) % len(PROMPT_TOKENS) for c in range(n_connections)],
                    weights=[round(0.3 - c * 0.04, 4) for c in range(n_connections)],
                )
            )
        basins = [round(0.2 + hi * 0.05 + li * 0.01, 4) for hi in range(NUM_HEADS)]
        pa_layers.append(PromptAttentionLayer(heads=heads, basin_scores=basins))

    return PromptAnalysis(
        layers=pa_layers,
        layer_transformations=transformations,
        prompt_surprisals=surprisals,
    )


def main() -> None:
    prompt_ids = [t[0] for t in PROMPT_TOKENS]
    gen_ids = [t[0] for t in GENERATED_TOKENS]
    num_prompt = len(PROMPT_TOKENS)
    num_gen = len(GENERATED_TOKENS)

    timeline = _build_timeline()

    health_flags = HealthFlags(
        nan_detected=False,
        inf_detected=False,
        attention_collapse_detected=False,
        high_entropy_steps=2,
        repetition_loop_detected=False,
        mid_layer_anomaly_detected=False,
    )

    prompt_hash = hashlib.sha256(f"gpt2:{PROMPT_TEXT}".encode()).hexdigest()[:16]

    fingerprint_vec = [round(0.25 * (i % 5) - 0.5, 4) for i in range(16)]

    report = Report(
        schema_version="0.3.0",
        trace_id="demo-sample-5min",
        created_at_utc="2026-02-11T12:00:00Z",
        model=ModelInfo(
            hf_id="gpt2",
            revision=None,
            architecture="GPT2LMHeadModel",
            num_layers=NUM_LAYERS,
            hidden_size=768,
            num_attention_heads=NUM_HEADS,
            tokenizer_hf_id="gpt2",
            dtype="float32",
            device="cpu",
            quantization=QuantizationInfo(enabled=False),
        ),
        run_config=RunConfig(
            seed=42,
            device_requested="cpu",
            max_new_tokens=num_gen,
            generation=GenerationConfig(do_sample=True, temperature=0.8, top_k=50, top_p=0.95),
            summaries=SummariesConfig(
                hidden=HiddenConfig(
                    enabled=True,
                    stats=["mean", "std", "l2_norm_mean", "max_abs"],
                    sketch=SketchConfig(method="randproj", dim=32, seed=0),
                ),
                attention=AttentionConfig(
                    enabled=True,
                    stats=["entropy_mean", "entropy_min", "entropy_max", "concentration_max", "concentration_min"],
                ),
                logits=LogitsConfig(
                    enabled=True,
                    stats=["entropy", "perplexity", "surprisal", "top1_top2_margin", "voter_agreement"],
                    topk=5,
                ),
            ),
            sink=SinkConfig(type="sqlite", target="runs/corevital.db"),
        ),
        prompt=PromptInfo(text=PROMPT_TEXT, token_ids=prompt_ids, num_tokens=num_prompt),
        generated=GeneratedInfo(output_text=GENERATED_TEXT, token_ids=gen_ids, num_tokens=num_gen),
        timeline=timeline,
        summary=Summary(
            prompt_tokens=num_prompt,
            generated_tokens=num_gen,
            total_steps=num_prompt + num_gen,
            elapsed_ms=1842,
        ),
        warnings=[],
        prompt_analysis=_build_prompt_analysis(),
        health_flags=health_flags,
        extensions={
            "risk": {
                "risk_score": 0.28,
                "risk_factors": ["high_entropy_steps"],
                "blamed_layers": [4, 10],
            },
            "fingerprint": {
                "vector": fingerprint_vec,
                "prompt_hash": prompt_hash,
            },
            "early_warning": {
                "failure_risk": 0.15,
                "warning_signals": [],
            },
            "narrative": {
                "summary": (
                    "This run was low risk. "
                    "Two steps showed elevated entropy (steps 5 and 10), suggesting brief uncertainty. "
                    "No repetition, NaN/Inf, or attention collapse detected."
                ),
            },
            "performance": {
                "total_wall_time_ms": 1842.0,
                "parent_operations": [
                    {"name": "config_load", "ms": 3.2, "pct": 0.0017},
                    {"name": "setup_logging", "ms": 0.5, "pct": 0.0003},
                    {"name": "model_load", "ms": 1245.0, "pct": 0.676},
                    {"name": "torch.manual_seed", "ms": 0.1, "pct": 0.0001},
                    {"name": "tokenize", "ms": 12.3, "pct": 0.0067},
                    {"name": "model_inference", "ms": 498.0, "pct": 0.2703},
                    {"name": "report_build", "ms": 78.5, "pct": 0.0426},
                ],
                "unaccounted_time": {"ms": 4.4, "pct": 0.0024},
            },
        },
    )

    out = repo_root / "docs" / "demo" / "sample_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json_str = serialize_report_to_json(report)
    out.write_text(json_str, encoding="utf-8")
    print(f"Wrote {out} ({len(json_str)} bytes)")


if __name__ == "__main__":
    main()

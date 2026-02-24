#!/usr/bin/env python3
"""
Shared data and helpers for the CoreVital evaluation suite.

Design goals:
- No manual labeling required for the first pass (auto-gradable prompt set)
- Explicit separation of task correctness vs CoreVital signal quality
- Public/default models first; gated models available via preset/override
"""

from __future__ import annotations

import re
import string
from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    category: str  # qa_fact, qa_math, formatting, repetition_probe, open_ended
    prompt: str
    max_new_tokens: int
    grading: str  # exact_any, contains_any, regex, none
    expected: tuple[str, ...] = ()
    regex: Optional[str] = None
    notes: str = ""
    tags: tuple[str, ...] = ()
    prompt_t5: Optional[str] = None
    format_expectation: Optional[str] = None  # one_word, one_number, yes_no, none

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["expected"] = list(self.expected)
        out["tags"] = list(self.tags)
        return out


# Auto-gradable first-pass suite (mostly instruction-following / factual prompts)
EVAL_CASES: list[EvalCase] = [
    EvalCase(
        case_id="capital_france",
        category="qa_fact",
        prompt="Answer in one word: What is the capital of France?",
        max_new_tokens=5,
        grading="contains_any",
        expected=("paris",),
        notes="Simple factual QA, one-word response.",
        tags=("gradable", "factual", "low_entropy_expected"),
        prompt_t5="question: What is the capital of France?",
        format_expectation="one_word",
    ),
    EvalCase(
        case_id="capital_japan",
        category="qa_fact",
        prompt="Answer in one word: What is the capital of Japan?",
        max_new_tokens=5,
        grading="contains_any",
        expected=("tokyo",),
        notes="Factual QA with common spelling variations.",
        tags=("gradable", "factual"),
        prompt_t5="question: What is the capital of Japan?",
        format_expectation="one_word",
    ),
    EvalCase(
        case_id="math_2_plus_2",
        category="qa_math",
        prompt="Answer with one token only: What is 2 + 2?",
        max_new_tokens=4,
        grading="exact_any",
        expected=("4", "four"),
        notes="Deterministic arithmetic baseline.",
        tags=("gradable", "math", "low_entropy_expected"),
        prompt_t5="question: What is 2 + 2?",
        format_expectation="one_number",
    ),
    EvalCase(
        case_id="math_12_times_3",
        category="qa_math",
        prompt="Answer with one number only: What is 12 * 3?",
        max_new_tokens=4,
        grading="exact_any",
        expected=("36", "thirty six", "thirty-six"),
        tags=("gradable", "math"),
        prompt_t5="question: What is 12 * 3?",
        format_expectation="one_number",
    ),
    EvalCase(
        case_id="translation_hola",
        category="translation",
        prompt="Translate to Spanish. One word only: hello",
        max_new_tokens=5,
        grading="contains_any",
        expected=("hola",),
        prompt_t5="translate English to Spanish: hello",
        tags=("gradable", "translation"),
        format_expectation="one_word",
    ),
    EvalCase(
        case_id="sorting_letters",
        category="formatting",
        prompt="Sort these letters alphabetically and return one word only: cba",
        max_new_tokens=4,
        grading="exact_any",
        expected=("abc",),
        tags=("gradable", "formatting"),
        prompt_t5="Sort letters alphabetically: c b a",
        format_expectation="one_word",
    ),
    EvalCase(
        case_id="country_currency_japan",
        category="qa_fact",
        prompt="Answer in one word: What currency is used in Japan?",
        max_new_tokens=6,
        grading="contains_any",
        expected=("yen",),
        tags=("gradable", "factual"),
        prompt_t5="question: What currency is used in Japan?",
        format_expectation="one_word",
    ),
    EvalCase(
        case_id="yes_no_water_boils",
        category="boolean",
        prompt="Yes or no only: Does water boil at 100 C at sea level?",
        max_new_tokens=4,
        grading="contains_any",
        expected=("yes",),
        tags=("gradable", "boolean"),
        prompt_t5="question: Does water boil at 100 C at sea level?",
        format_expectation="yes_no",
    ),
    # Probe prompts (not task-graded, but useful for metric behavior coverage)
    EvalCase(
        case_id="repetition_probe",
        category="repetition_probe",
        prompt=(
            "Repeat the word 'echo' exactly 15 times separated by spaces. "
            "Do not add any other words."
        ),
        max_new_tokens=40,
        grading="none",
        notes="Used to test repetition-loop detection and output repetition heuristics.",
        tags=("probe", "repetition"),
    ),
    EvalCase(
        case_id="open_ended_ambiguous",
        category="open_ended",
        prompt="Finish this sentence creatively in one sentence: The answer to everything is",
        max_new_tokens=20,
        grading="none",
        notes="Used to induce higher entropy/uncertainty in some models.",
        tags=("probe", "high_entropy"),
    ),
]


MODEL_PRESETS: dict[str, list[str]] = {
    # Public, CPU-friendly defaults. Avoids gpt2/base-llama for first eval pass.
    "cpu_default": [
        "google/flan-t5-small",
        "Qwen/Qwen2-0.5B-Instruct",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ],
    # Faster subset for quick smoke checks on CPU.
    "cpu_quick": [
        "google/flan-t5-small",
        "Qwen/Qwen2-0.5B-Instruct",
    ],
    # Includes gated Meta model if the user has access + HF_TOKEN configured.
    "llama_instruct_plus": [
        "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen/Qwen2-0.5B-Instruct",
        "google/flan-t5-small",
    ],
}


def resolve_models(models: Optional[list[str]], preset: str) -> list[str]:
    if models:
        return models
    if preset not in MODEL_PRESETS:
        raise ValueError(f"Unknown model preset: {preset}. Available: {', '.join(sorted(MODEL_PRESETS))}")
    return MODEL_PRESETS[preset]


def selected_cases(include_probes: bool = True, limit: Optional[int] = None) -> list[EvalCase]:
    cases = [c for c in EVAL_CASES if include_probes or "probe" not in c.tags]
    if limit is not None:
        cases = cases[:limit]
    return cases


def normalize_text(text: str) -> str:
    txt = (text or "").strip().lower()
    txt = txt.replace("<|eot_id|>", " ").replace("</s>", " ")
    txt = "".join(ch for ch in txt if ch not in string.punctuation)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def render_prompt(case: EvalCase, model_id: str) -> str:
    """Render a prompt variant appropriate for the model family."""
    m = (model_id or "").lower()
    if ("t5" in m or "flan-t5" in m) and case.prompt_t5:
        return case.prompt_t5
    return case.prompt


def grade_output(case: EvalCase, generated_text: str) -> dict[str, Any]:
    """Auto-grade a generated output for the subset of gradable prompts."""
    if case.grading == "none":
        return {"gradable": False, "correct": None, "reason": "not_graded"}

    norm = normalize_text(generated_text)
    expected_norm = [normalize_text(v) for v in case.expected]

    if case.grading == "exact_any":
        correct = norm in expected_norm
        return {
            "gradable": True,
            "correct": correct,
            "reason": "exact_any",
            "normalized_output": norm,
            "normalized_expected": expected_norm,
        }

    if case.grading == "contains_any":
        correct = any(v and v in norm for v in expected_norm)
        return {
            "gradable": True,
            "correct": correct,
            "reason": "contains_any",
            "normalized_output": norm,
            "normalized_expected": expected_norm,
        }

    if case.grading == "regex":
        if not case.regex:
            return {"gradable": True, "correct": False, "reason": "missing_regex", "normalized_output": norm}
        correct = re.search(case.regex, generated_text or "", flags=re.IGNORECASE) is not None
        return {
            "gradable": True,
            "correct": correct,
            "reason": "regex",
            "normalized_output": norm,
            "regex": case.regex,
        }

    return {"gradable": False, "correct": None, "reason": f"unknown_grading:{case.grading}"}


def detect_output_repetition(text: str) -> dict[str, Any]:
    """
    Simple output-based repetition heuristic independent of CoreVital hidden-state metrics.

    Flags repeated 1-gram/2-gram patterns in normalized tokens.
    """
    norm = normalize_text(text)
    toks = [t for t in norm.split(" ") if t]
    if len(toks) < 4:
        return {"repetition_detected": False, "pattern": None, "count": 0}

    best_pattern: Optional[str] = None
    best_count = 0

    for n in (1, 2):
        if len(toks) < n * 3:
            continue
        i = 0
        while i <= len(toks) - n:
            gram = toks[i : i + n]
            count = 1
            j = i + n
            while j <= len(toks) - n and toks[j : j + n] == gram:
                count += 1
                j += n
            if count >= 3 and count > best_count:
                best_count = count
                best_pattern = " ".join(gram)
            i += 1

    return {
        "repetition_detected": best_count >= 3,
        "pattern": best_pattern,
        "count": best_count,
    }


def check_format_expectation(case: EvalCase, generated_text: str) -> dict[str, Any]:
    """Lightweight format compliance checks, separate from correctness grading."""
    exp = case.format_expectation
    if not exp or exp == "none":
        return {"checked": False, "compliant": None, "reason": "no_expectation"}

    norm = normalize_text(generated_text)
    toks = [t for t in norm.split(" ") if t]

    if exp == "one_word":
        return {"checked": True, "compliant": len(toks) == 1, "reason": "one_word"}

    if exp == "one_number":
        if len(toks) != 1:
            return {"checked": True, "compliant": False, "reason": "one_number"}
        return {"checked": True, "compliant": bool(re.fullmatch(r"\d+", toks[0])), "reason": "one_number"}

    if exp == "yes_no":
        if len(toks) != 1:
            return {"checked": True, "compliant": False, "reason": "yes_no"}
        return {"checked": True, "compliant": toks[0] in {"yes", "no"}, "reason": "yes_no"}

    return {"checked": False, "compliant": None, "reason": f"unknown_expectation:{exp}"}

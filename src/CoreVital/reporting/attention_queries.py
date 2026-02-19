# ============================================================================
# CoreVital - Sparse Attention Query Helpers
#
# Purpose: Query sparse attention data from prompt_analysis (Branch 6 / #8).
# Inputs: Layer/head data (dict or schema objects from report)
# Outputs: Filtered connections, top-N, basin anomalies
# ============================================================================

from typing import Any, List, Tuple


def _layer_heads(layer: Any) -> List[Any]:
    """Get list of heads from a layer (dict or PromptAttentionLayer)."""
    if hasattr(layer, "heads"):
        return list(layer.heads)
    if isinstance(layer, dict):
        return list(layer.get("heads", []))
    return []


def _layer_basin_scores(layer: Any) -> List[float]:
    """Get basin_scores from a layer."""
    if hasattr(layer, "basin_scores"):
        return list(layer.basin_scores)
    return list(layer.get("basin_scores", [])) if isinstance(layer, dict) else []


def _head_arrays(head: Any) -> Tuple[List[int], List[int], List[float]]:
    """Get (query_indices, key_indices, weights) from a head (dict or SparseAttentionHead)."""
    if hasattr(head, "query_indices"):
        return head.query_indices, head.key_indices, head.weights
    return (
        head.get("query_indices", []),
        head.get("key_indices", []),
        head.get("weights", []),
    )


def get_attention_to_token(
    layer: Any,
    head_idx: int,
    key_idx: int,
) -> List[Tuple[int, float]]:
    """Which queries attend to a specific key (and with what weight).

    Returns list of (query_idx, weight) for all connections where key_indices == key_idx.
    """
    heads = _layer_heads(layer)
    if head_idx < 0 or head_idx >= len(heads):
        return []
    q, k, w = _head_arrays(heads[head_idx])
    return [(qi, wi) for qi, ki, wi in zip(q, k, w, strict=True) if ki == key_idx]


def get_attention_from_token(
    layer: Any,
    head_idx: int,
    query_idx: int,
) -> List[Tuple[int, float]]:
    """Where a specific query attends (and with what weight).

    Returns list of (key_idx, weight) for all connections where query_indices == query_idx.
    """
    heads = _layer_heads(layer)
    if head_idx < 0 or head_idx >= len(heads):
        return []
    q, k, w = _head_arrays(heads[head_idx])
    return [(ki, wi) for qi, ki, wi in zip(q, k, w, strict=True) if qi == query_idx]


def get_top_connections(
    layer: Any,
    head_idx: int,
    n: int = 10,
) -> List[Tuple[int, int, float]]:
    """Top-N strongest connections for a head: (query_idx, key_idx, weight), sorted by weight descending."""
    heads = _layer_heads(layer)
    if head_idx < 0 or head_idx >= len(heads):
        return []
    q, k, w = _head_arrays(heads[head_idx])
    if not w:
        return []
    indexed = list(zip(q, k, w, strict=True))
    indexed.sort(key=lambda x: x[2], reverse=True)
    return indexed[:n]


def get_heads_attending_to_range(
    layers: List[Any],
    start: int,
    end: int,
) -> List[Tuple[int, int]]:
    """Which (layer_idx, head_idx) have at least one connection to key in [start, end]."""
    result: List[Tuple[int, int]] = []
    for layer_idx, layer in enumerate(layers):
        heads = _layer_heads(layer)
        for head_idx, head in enumerate(heads):
            _, k, _ = _head_arrays(head)
            if any(start <= ki <= end for ki in k):
                result.append((layer_idx, head_idx))
    return result


def get_basin_anomalies(
    layers: List[Any],
    threshold: float = 0.3,
) -> List[Tuple[int, int, float]]:
    """Heads with basin_score < threshold (U-shape: ignore middle of prompt).

    Returns list of (layer_idx, head_idx, basin_score). See Attention Basin (arXiv:2508.05128).
    """
    result: List[Tuple[int, int, float]] = []
    for layer_idx, layer in enumerate(layers):
        basin_scores = _layer_basin_scores(layer)
        for head_idx, score in enumerate(basin_scores):
            if score < threshold:
                result.append((layer_idx, head_idx, float(score)))
    return result

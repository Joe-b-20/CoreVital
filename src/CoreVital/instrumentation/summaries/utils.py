# Shared helpers for summary computation (random projection sketch, etc.)

from typing import List

import numpy as np
import torch


def _random_projection_sketch(
    tensor: torch.Tensor,
    sketch_dim: int,
    seed: int,
) -> List[float]:
    """
    Compute random projection sketch of tensor.

    Args:
        tensor: Input tensor, shape (seq_len, hidden_dim)
        sketch_dim: Target sketch dimension
        seed: Random seed for reproducibility

    Returns:
        List of sketch values
    """
    rng = np.random.default_rng(seed)

    if tensor.dim() == 2:
        vector = tensor.mean(dim=0).numpy()
    else:
        vector = tensor.numpy().flatten()

    hidden_dim = len(vector)

    projection_matrix = rng.standard_normal((hidden_dim, sketch_dim))
    projection_matrix /= np.sqrt(hidden_dim)

    sketch = vector @ projection_matrix

    return [round(float(x), 2) for x in sketch.tolist()]

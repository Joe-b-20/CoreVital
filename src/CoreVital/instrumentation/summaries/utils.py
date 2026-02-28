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
    # Use numpy for deterministic random projection
    np.random.seed(seed)

    # Flatten or average across sequence
    if tensor.dim() == 2:
        # Average across sequence dimension
        vector = tensor.mean(dim=0).numpy()
    else:
        vector = tensor.numpy().flatten()

    hidden_dim = len(vector)

    # Generate random projection matrix
    projection_matrix = np.random.randn(hidden_dim, sketch_dim)
    projection_matrix /= np.sqrt(hidden_dim)  # Normalize

    # Project
    sketch = vector @ projection_matrix

    # Return as list, rounded
    return [round(float(x), 2) for x in sketch.tolist()]

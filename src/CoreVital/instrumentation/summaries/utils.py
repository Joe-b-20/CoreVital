# Shared helpers for summary computation (random projection sketch, etc.)

from typing import List

import torch


def _random_projection_sketch(
    tensor: torch.Tensor,
    sketch_dim: int,
    seed: int,
) -> List[float]:
    """
    Compute random projection sketch of tensor.

    Keeps computation on the same device as the tensor (GPU when available);
    only the final sketch list is brought to Python.

    Args:
        tensor: Input tensor, shape (seq_len, hidden_dim)
        sketch_dim: Target sketch dimension
        seed: Random seed for reproducibility

    Returns:
        List of sketch values
    """
    tensor = tensor.detach().float()
    device = tensor.device

    if tensor.dim() == 2:
        vector = tensor.mean(dim=0)  # (hidden_dim,)
    else:
        vector = tensor.flatten()

    hidden_dim = vector.shape[0]
    generator = torch.Generator(device=device).manual_seed(seed)
    projection = torch.randn(hidden_dim, sketch_dim, device=device, generator=generator)
    projection = projection / (hidden_dim ** 0.5)
    sketch = vector @ projection  # (sketch_dim,)
    return [round(float(x), 2) for x in sketch.tolist()]

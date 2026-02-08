# ============================================================================
# CoreVital - Hook System
#
# Purpose: Register hooks to capture hidden states and attention during forward pass
# Inputs: Model layers
# Outputs: Hook handles and captured data storage
# Dependencies: torch
# Usage: handles, storage = register_hooks(model)
#
# Changelog:
#   2026-01-13: Initial hook system for Phase-0
# ============================================================================

from typing import Any, List, Tuple

import torch
from torch import nn

from CoreVital.logging_utils import get_logger

logger = get_logger(__name__)


class HookStorage:
    """Storage for captured hook data."""

    def __init__(self):
        """Initialize empty storage."""
        self.hidden_states: List[torch.Tensor] = []
        self.attentions: List[torch.Tensor] = []

    def clear(self) -> None:
        """Clear all stored data."""
        self.hidden_states.clear()
        self.attentions.clear()


def register_hooks(model: nn.Module) -> Tuple[List[Any], HookStorage]:
    """
    Register forward hooks on model layers to capture hidden states and attention.

    Note: This is a backup mechanism. Primary method uses model's native
    output_hidden_states and output_attentions parameters.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (hook_handles, storage)
    """
    storage = HookStorage()
    handles = []

    logger.debug("Registering forward hooks (backup mechanism)")

    # This is kept minimal as Phase-0 primarily uses model's built-in outputs
    # Hooks can be extended in future phases if needed

    return handles, storage


def remove_hooks(handles: List[Any]) -> None:
    """
    Remove all registered hooks.

    Args:
        handles: List of hook handles to remove
    """
    for handle in handles:
        handle.remove()
    logger.debug(f"Removed {len(handles)} hooks")

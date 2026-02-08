# ============================================================================
# CoreVital - Typing Utilities
#
# Purpose: Shared type definitions and type utilities
# Inputs: None
# Outputs: Type aliases
# Dependencies: typing
# Usage: from CoreVital.utils.typing import PathLike
#
# Changelog:
#   2026-01-13: Initial typing utilities for Phase-0
# ============================================================================

from pathlib import Path
from typing import Union

# Type alias for path-like objects
PathLike = Union[str, Path]

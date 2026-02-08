# ============================================================================
# CoreVital - Models Package
#
# Purpose: Model loading and management
# Inputs: None
# Outputs: Public API exports
# Dependencies: None
# Usage: from CoreVital.models import load_model
#
# Changelog:
#   2026-01-13: Initial models package for Phase-0
# ============================================================================

from CoreVital.models.hf_loader import ModelBundle, load_model
from CoreVital.models.registry import ModelCapabilities

__all__ = ["load_model", "ModelBundle", "ModelCapabilities"]

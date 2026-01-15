# ============================================================================
# CoreVital - Logging Utilities
#
# Purpose: Centralized logging configuration and utilities
# Inputs: Log level, format string
# Outputs: Configured logger instances
# Dependencies: logging (stdlib)
# Usage: logger = get_logger(__name__)
#
# Changelog:
#   2026-01-13: Initial logging setup for Phase-0
# ============================================================================

import logging
from typing import Optional


_LOGGING_CONFIGURED = False


def setup_logging(level: str = "INFO", format_string: Optional[str] = None) -> None:
    """
    Configure logging for the entire package.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_string: Optional custom format string
    """
    global _LOGGING_CONFIGURED
    
    if _LOGGING_CONFIGURED:
        return
    
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
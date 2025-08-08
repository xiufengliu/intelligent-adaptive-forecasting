"""
CaMS (Calibrated Meta-Selection) framework

This package exposes a stable, user-friendly API for the CaMS framework while
maintaining backward compatibility with the current repository structure.

Primary entrypoint:
    - cams.framework.CaMSFramework

Versioning follows semantic versioning for the public API.
"""
from __future__ import annotations

from importlib import metadata

try:
    __version__ = metadata.version("cams")  # Will resolve once packaged
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0-dev"

# Re-export the main framework for convenience
from .framework import CaMSFramework, create_cams_model  # noqa: E402,F401

__all__ = [
    "CaMSFramework",
    "create_cams_model",
    "__version__",
]


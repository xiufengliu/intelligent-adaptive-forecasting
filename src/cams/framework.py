"""
Public CaMS framework API, wrapping the existing I-ASNH implementation.

This module provides a thin compatibility layer so users can import:

    from cams import CaMSFramework, create_cams_model

while we incrementally refactor internals. For now, CaMS maps directly to the
existing IASNH implementation to avoid breaking behavior.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from .core import CaMSFramework as CaMSFramework
from .core import create_cams_model as create_cams_model


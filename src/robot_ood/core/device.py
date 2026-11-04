"""The one torch.device resolution helper for the whole package.

Lives in core because models, build, train and runtime all resolve a device, and anywhere
higher would make that a
cycle. Imports torch and nothing from robot_ood.
"""

from __future__ import annotations

import torch


def resolve_device(device: str | torch.device | None) -> torch.device:
    """None -> CUDA if available else CPU; a string/torch.device is used as given."""
    if isinstance(device, torch.device):
        return device
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

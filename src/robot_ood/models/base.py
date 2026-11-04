"""CheckpointedModule -- one save/load contract for every trained head in this package.

A head declares three things and inherits the rest: the artifact's filename, what to tell someone
whose artifact is missing, and the shape constants it stamps beside its weights. Meta entries come
first and state_dict entries after, in insertion order.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, Mapping

import numpy as np
import torch

from robot_ood.core import npz


class CheckpointedModule:
    """Mixin for a torch.nn.Module whose weights live in one .npz beside the graph artifacts."""

    NPZ: ClassVar[str]
    MISSING: ClassVar[str]

    def checkpoint_meta(self) -> Mapping[str, Any]:
        """The shape constants stamped before the weights. Order fixes the archive's key order."""
        raise NotImplementedError

    @classmethod
    def from_checkpoint(cls, meta: Mapping[str, Any], src: Path) -> "CheckpointedModule":
        """Construct an unloaded module from an archive's meta half.

        Args:
            meta: The archive, for reading shape constants. A head validates here.
            src: The path it came from, for error messages.

        Returns:
            The module, before load_state_dict.
        """
        raise NotImplementedError

    @classmethod
    def check_state(cls, state: Mapping[str, torch.Tensor], src: Path) -> None:
        """Refuse a state dict this build cannot use. Default: accept anything load_state_dict does."""

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write the weights and shape constants needed to rebuild this module.

        Args:
            path: Destination .npz, or the directory to write NPZ into.

        Returns:
            The written path.
        """
        return write_state(path, self.NPZ, self.checkpoint_meta(), self.state_dict())

    @classmethod
    def load(cls, path: str | os.PathLike[str], device: Any = "cpu") -> Any:
        """Rebuild a module written by save.

        Args:
            path: The .npz, or the directory holding NPZ.
            device: Torch device to place the module on.

        Returns:
            The module in eval mode.

        Raises:
            FileNotFoundError: Nothing is at the resolved path.
            ValueError: The artifact was written against a shape this build does not read.
        """
        src, archive = npz.read(path, cls.NPZ, cls.MISSING)
        net = cls.from_checkpoint(archive, src)
        state = split_state(archive, tuple(net.checkpoint_meta()))
        cls.check_state(state, src)
        net.load_state_dict(state)
        return net.to(device).eval()


def write_state(path: str | os.PathLike[str], filename: str, meta: Mapping[str, Any],
                state: Mapping[str, torch.Tensor]) -> Path:
    """Write meta then a state dict to one .npz, the tensors detached to numpy on the way out.

    Args:
        path: Destination .npz, or the directory to write filename into.
        filename: The archive's name inside a directory.
        meta: The constants a reader needs before it can rebuild what the state belongs to.
        state: The module's state_dict.

    Returns:
        The written path.
    """
    arrays = {k: v.detach().cpu().numpy() for k, v in state.items()}
    return npz.write_arrays(path, filename, {**meta, **arrays})


def split_state(archive: Mapping[str, Any], meta_keys: tuple[str, ...]) -> dict[str, torch.Tensor]:
    """The state half of an archive: every numeric key that is not a meta key, as a tensor."""
    # A state dict holds no strings, so a text entry is a stamp some build wrote and this one does
    # not read -- skipped, never handed to load_state_dict as an unexpected weight.
    return {k: torch.as_tensor(v) for k, v in archive.items()
            if k not in meta_keys and np.asarray(v).dtype.kind not in "USO"}


def i64(value: int) -> np.int64:
    """A shape constant as the archive stores it."""
    return np.int64(value)

"""The .npz mechanics every checkpoint and every artifact in this package is written through.

Nothing here knows what a head or a table is, and it stays numpy-only: robot_ood.data.nodes reads
artifacts and must not pay for a torch import. The tensor half of the contract lives on the model
side, in robot_ood.models.base.

Key order is the caller's. np.savez pins the zip timestamps, so the same values always produce the
same bytes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def resolve(path: str | os.PathLike[str], filename: str) -> Path:
    """The .npz a caller means: the path itself, or filename inside it when it is a directory."""
    target = Path(path)
    return target if target.suffix == ".npz" else target / filename


def write_arrays(path: str | os.PathLike[str], filename: str, payload: Mapping[str, Any],
                 *, compress: bool = False) -> Path:
    """Write one payload to one .npz, in the order given.

    Args:
        path: Destination .npz, or the directory to write filename into.
        payload: What to store, in the order it should appear in the archive.
        compress: Use savez_compressed. Artifacts do; model checkpoints do not.

    Returns:
        The written path.
    """
    target = resolve(path, filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    save = np.savez_compressed if compress else np.savez
    save(os.fspath(target), **payload)
    return target


def read(path: str | os.PathLike[str], filename: str, missing: str) -> tuple[Path, dict[str, Any]]:
    """Read one .npz into a plain dict, eagerly, so the archive is closed before anything is built.

    Args:
        path: The .npz, or the directory holding filename.
        filename: The artifact's name inside a directory.
        missing: What to tell the caller to run when nothing is there.

    Returns:
        (the resolved path, {key: array}).

    Raises:
        FileNotFoundError: Nothing is at the resolved path.
    """
    src = resolve(path, filename)
    if not src.is_file():
        raise FileNotFoundError(f"{src}: {missing}")
    with np.load(os.fspath(src)) as archive:
        return src, {k: archive[k] for k in archive.files}

"""FcsrNet -- every parameter train fcsr optimises, as one module.

One job, one optimiser, one backward, one checkpoint. The three children keep their own shape
stamps, so an archive still names the widths each was built at.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch.nn as nn

from robot_ood.core import npz
from robot_ood.core.schemas import MISSING_FCSR
from robot_ood.data import schema
from robot_ood.models.base import CheckpointedModule, i64, split_state

__all__ = ["SegEncoders", "FcsrNet"]


class SegEncoders(nn.Module):
    """The retrieval half's three encoders.

    Args:
        query: Encodes a QueryContext into the query embedding.
        key: Projects segment-side features into that same space for tier 1.
        action: Encodes a fixed-length action chunk.
    """

    def __init__(self, query: nn.Module, key: nn.Module, action: nn.Module) -> None:
        super().__init__()
        self.query, self.key, self.action = query, key, action

    def checkpoint_meta(self) -> dict[str, Any]:
        """The widths the three encoders were built at."""
        return {
            "dim": i64(self.query.dim), "hist": i64(self.query.hist),
            "hidden": i64(self.query.hidden), "lang_dim": i64(self.query.lang_dim),
            "action_chunk_len": i64(self.action.chunk_len),
        }

    @classmethod
    def from_checkpoint(cls, meta: Mapping[str, Any], src: Path) -> "SegEncoders":
        """Rebuild the three encoders an archive describes.

        Args:
            meta: The archive half under this prefix.
            src: Where it came from, for error messages.

        Returns:
            The encoders, before load_state_dict.
        """
        from robot_ood.core.schemas import ACTION_EMBED_DIM
        from robot_ood.models.retrieval.encoders import QueryEncoder, SegmentKeyEncoder
        from robot_ood.models.retrieval.scorer import ActionEncoder

        try:
            hidden = int(meta["hidden"])
            return cls(
                QueryEncoder(dim=int(meta["dim"]), hist=int(meta["hist"]), hidden=hidden,
                             future_len=int(meta["action_chunk_len"]),
                             lang_dim=int(meta["lang_dim"])),
                SegmentKeyEncoder(hidden),
                ActionEncoder(int(meta["action_chunk_len"]), out_dim=ACTION_EMBED_DIM),
            )
        except KeyError as exc:
            raise ValueError(
                f"{src}: no {exc.args[0]} stamped for the segment encoders. Retrain with "
                "`python -m robot_ood.cli train fcsr --suite <suite>`."
            ) from exc


class FcsrNet(CheckpointedModule, nn.Module):
    """The segment encoders, the C2 scorer and the corrector.

    Args:
        seg: The retrieval half's encoders.
        scorer: The C2 rescorer.
        corrector: The correction learner (trust and shape heads).
    """

    NPZ = schema.FCSR_NPZ
    MISSING = MISSING_FCSR

    def __init__(self, seg: SegEncoders, scorer: nn.Module, corrector: nn.Module) -> None:
        super().__init__()
        self.seg, self.scorer, self.corrector = seg, scorer, corrector
        # Provenance stamp only, assigned after the fit; deploy scoring reads GraphConfig.future_temp.
        self.future_temp: float | None = None

    def children_by_prefix(self) -> Iterator[tuple[str, Any]]:
        """The three children in archive order, each with its key prefix."""
        yield "seg.", self.seg
        yield "scorer.", self.scorer
        yield "corrector.", self.corrector

    def checkpoint_meta(self) -> dict[str, Any]:
        """Every child's stamps under that child's prefix, in archive order."""
        meta: dict[str, Any] = {}
        for prefix, child in self.children_by_prefix():
            meta.update({f"{prefix}{k}": v for k, v in child.checkpoint_meta().items()})
            if prefix == "seg." and self.future_temp is not None:
                meta["seg.future_temp"] = np.float64(self.future_temp)
        return meta

    @classmethod
    def from_checkpoint(cls, meta: Mapping[str, Any], src: Path) -> "FcsrNet":
        """Rebuild the three children an archive describes.

        Args:
            meta: The whole archive, keys still prefixed.
            src: Where it came from, for error messages.

        Returns:
            The net, before load_state_dict.

        Raises:
            ValueError: The archive carries another head than the corrector: the safn./dose.
                halves that predate it, or the fusion. head that briefly replaced it.
        """
        from robot_ood.models.fusion.corrector import Corrector
        from robot_ood.models.retrieval.scorer import SegmentScorer

        if any(k.startswith(("safn.", "dose.", "fusion.")) for k in meta):
            raise ValueError(
                f"{src}: this archive carries a retired safn./dose. or fusion. head; this build reads "
                f"a corrector. half. Retrain with `python -m robot_ood.cli train fcsr --suite <suite>`."
            )

        def half(prefix: str) -> dict[str, Any]:
            return {k[len(prefix):]: v for k, v in meta.items() if k.startswith(prefix)}

        return cls(
            SegEncoders.from_checkpoint(half("seg."), src),
            SegmentScorer.from_checkpoint(half("scorer."), src),
            Corrector.from_checkpoint(half("corrector."), src),
        )

    def save(self, graph_dir: str | os.PathLike, *, graph_hash: str) -> Path:
        """Write this net to <graph_dir>/g_fcsr.npz.

        Args:
            graph_dir: Directory to write into, or a full .npz path.
            graph_hash: The segment graph hash this checkpoint was trained against.

        Returns:
            The written path.
        """
        return npz.write_arrays(graph_dir, self.NPZ, self.arrays(graph_hash=graph_hash))

    @classmethod
    def load(cls, graph_dir: str | os.PathLike, device: Any = "cpu") -> "FcsrNet":
        """Rebuild the net beside a graph.

        Args:
            graph_dir: Directory holding the archive, or a full .npz path.
            device: Torch device to place the net on.

        Returns:
            The net in eval mode.

        Raises:
            FileNotFoundError: No archive there.
        """
        src, archive = npz.read(graph_dir, cls.NPZ, cls.MISSING)
        return cls.from_arrays(archive, src, device)

    def arrays(self, *, graph_hash: str) -> dict[str, Any]:
        """This net as the archive stores it: the stamp, then each child's meta and weights.

        Args:
            graph_hash: The segment graph hash this checkpoint was trained against.

        Returns:
            The arrays to write, in archive order.
        """
        meta = self.checkpoint_meta()
        state = {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}
        out: dict[str, Any] = {"graph_hash": np.array(graph_hash)}
        for prefix, _ in self.children_by_prefix():
            out.update({k: v for k, v in meta.items() if k.startswith(prefix)})
            out.update({k: v for k, v in state.items() if k.startswith(prefix)})
        return out

    @classmethod
    def from_arrays(cls, archive: Mapping[str, Any], src: Path, device: Any = "cpu") -> "FcsrNet":
        """Rebuild the net an archive holds.

        Args:
            archive: The whole archive.
            src: Where it came from, for error messages.
            device: Torch device to place the net on.

        Returns:
            The net in eval mode.
        """
        net = cls.from_checkpoint(archive, src)
        if "seg.future_temp" in archive:
            net.future_temp = float(archive["seg.future_temp"])
        from robot_ood.data import schema as graph_schema
        own = {k: v for k, v in archive.items()
               if k not in ("graph_hash", graph_schema.CONSTANTS_KEY)}
        net.load_state_dict(split_state(own, tuple(net.checkpoint_meta())))
        return net.to(device).eval()

"""The C0 Calibrator's output contract: constants.json, its schema version and its content hash.

A constant is allowed to exist only as the output of a measuring function on a corpus, so every
entry here carries the procedure that produced it and a kind saying whether THIS run ran that
procedure (MEASURED), whether it was run once against the benchmark and its result carried
(RECORDED), or whether nobody has measured it yet (ASSUMED -- the type (a) constants still to
migrate).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Mapping

__all__ = [
    "ASSUMED",
    "CONSTANTS_JSON",
    "ENTRY_STATIC_BAND",
    "HELDOUT_STRIDE",
    "JOINT_TOL",
    "PHASE_TOL",
    "TRUE_NODE_BOOST",
    "TRUE_TASK_BOOST",
    "WEIGHT_SCALE",
    "KINDS",
    "MEASURED",
    "RECORDED",
    "SCHEMA_VERSION",
    "STILL_TOL",
    "Constant",
    "Constants",
]

# Bumped whenever a key's MEANING or the file's layout changes, never for a new value. Readers
# refuse a version they do not know rather than silently reading a renamed key as a missing one.
SCHEMA_VERSION: Final[int] = 1

CONSTANTS_JSON: Final[str] = "constants.json"

MEASURED: Final[str] = "measured"    # type (b): this run ran the procedure on this corpus
RECORDED: Final[str] = "recorded"    # type (b): the procedure ran against the benchmark, value carried
ASSUMED: Final[str] = "assumed"
KINDS: Final[frozenset[str]] = frozenset({MEASURED, RECORDED, ASSUMED})  # type (a): a hand constant

# The ASSUMED fallback for the seg_entry_band key, read whenever no calibration is loaded. It
# lives here, not beside its readers, so every reader falls back to the same number.
ENTRY_STATIC_BAND: Final[float] = 0.05

# Every HELDOUT_STRIDE-th demo is held out. One number for the whole repo: a validation demo
# must be one the retriever, the belief fit and the entry heads all never saw.
HELDOUT_STRIDE: Final[int] = 5

# The locator's label rule (robot_ood.train.target): a target sits within PHASE_TOL of the true
# node's phase and JOINT_TOL rad of its joints; weights decay over WEIGHT_SCALE rad.
PHASE_TOL: Final[float] = 0.05
JOINT_TOL: Final[float] = 0.10
WEIGHT_SCALE: Final[float] = 0.10
TRUE_NODE_BOOST: Final[float] = 2.0
TRUE_TASK_BOOST: Final[float] = 2.0

# A frame whose joints moved less than STILL_TOL rad since the previous one does not spend the
# coarsener's frame budget, so a motionless stretch folds into one node however long it lasts.
STILL_TOL: Final[float] = 0.002


@dataclass(frozen=True, slots=True)
class Constant:
    """One calibrated number, with the procedure that produced it.

    Attributes:
        value: The number consumers read.
        kind: One of KINDS.
        procedure: One line naming the measuring function, in corpus terms.
        stats: The diagnostics the procedure produced alongside the value (sample sizes, the
            quantiles either side of the one chosen, fit quality). Calibration validate() gates
            on reach.stats; the rest is what makes a number arguable instead of merely present.

    Raises:
        ValueError: kind is not one of KINDS.
    """

    value: float | int
    kind: str
    procedure: str
    stats: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"Constant.kind={self.kind!r} must be one of {sorted(KINDS)}")
        object.__setattr__(self, "stats", MappingProxyType(dict(self.stats)))

    def to_dict(self) -> dict[str, Any]:
        """The json form: value, kind, procedure, stats."""
        return {
            "value": self.value, "kind": self.kind, "procedure": self.procedure,
            "stats": dict(self.stats),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Constant":
        """Rebuild from one json entry.

        Args:
            data: A mapping as to_dict emits.

        Returns:
            The constant.
        """
        return cls(
            value=data["value"], kind=str(data["kind"]), procedure=str(data.get("procedure", "")),
            stats=data.get("stats") or {},
        )


@dataclass(frozen=True)
class Constants:
    """One corpus's calibration: the constants, what they were measured on, and a content hash.

    Attributes:
        entries: Constant name -> Constant.
        corpus: What was calibrated (suite, input directories, corpus sizes, input fingerprint).
        schema_version: SCHEMA_VERSION at write time.
    """

    entries: Mapping[str, Constant]
    corpus: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))
        object.__setattr__(self, "corpus", MappingProxyType(dict(self.corpus)))

    # ---- reading -------------------------------------------------------------------------------
    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def __getitem__(self, name: str) -> Constant:
        return self.entries[name]

    def value(self, name: str, default: Any = None) -> Any:
        """The raw value of one constant.

        Args:
            name: Constant name.
            default: Returned when the calibration does not carry that name.

        Returns:
            The value, or default.
        """
        entry = self.entries.get(name)
        return default if entry is None else entry.value

    def int_value(self, name: str, default: int) -> int:
        """value(), cast to int, with default when absent."""
        return int(self.value(name, default))

    def of_kind(self, kind: str) -> tuple[str, ...]:
        """Names carrying one kind, sorted -- the "what is still type (a)" query."""
        return tuple(sorted(n for n, e in self.entries.items() if e.kind == kind))

    # ---- hashing / serialisation ---------------------------------------------------------------
    @property
    def payload(self) -> dict[str, Any]:
        """The hashed half: schema version plus every constant, keys sorted."""
        return {
            "schema_version": int(self.schema_version),
            "constants": {name: self.entries[name].to_dict() for name in sorted(self.entries)},
        }

    @property
    def content_hash(self) -> str:
        """sha256 over payload, serialised canonically (sorted keys, no whitespace)."""
        blob = json.dumps(self.payload, sort_keys=True, separators=(",", ":")).encode()
        return f"sha256:{hashlib.sha256(blob).hexdigest()}"

    def to_dict(self) -> dict[str, Any]:
        """The whole file: payload, the corpus block, and the content hash."""
        return {**self.payload, "corpus": dict(self.corpus), "content_hash": self.content_hash}

    def to_json(self, path: str | os.PathLike[str]) -> Path:
        """Write constants.json.

        Args:
            path: Destination .json, or the directory to write CONSTANTS_JSON into.

        Returns:
            The written path.
        """
        target = Path(path)
        target = target if target.suffix == ".json" else target / CONSTANTS_JSON
        target.parent.mkdir(parents=True, exist_ok=True)
        # sort_keys + a trailing newline: byte-identical output for identical measurements, which is
        # the determinism gate this file is held to.
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return target

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Constants":
        """Rebuild from a parsed constants.json.

        Args:
            data: The parsed file.

        Returns:
            The calibration.

        Raises:
            ValueError: The file declares a schema version this build does not know, or its
                recorded content hash does not match the constants it carries.
        """
        version = int(data.get("schema_version", -1))
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"constants.json declares schema_version={version}, this build reads "
                f"{SCHEMA_VERSION}. Re-run `python -m robot_ood.cli build calibrate` against the corpus."
            )
        out = cls(
            entries={n: Constant.from_dict(e) for n, e in (data.get("constants") or {}).items()},
            corpus=data.get("corpus") or {},
            schema_version=version,
        )
        stamped = data.get("content_hash")
        if stamped and stamped != out.content_hash:
            raise ValueError(
                f"constants.json content_hash mismatch -- stamped {stamped}, recomputed "
                f"{out.content_hash}. The file was edited by hand after C0 wrote it; a calibration "
                "that does not hash to its own contents cannot be attributed to any corpus."
            )
        return out

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> "Constants":
        """Read constants.json.

        Args:
            path: The .json, or the directory holding CONSTANTS_JSON.

        Returns:
            The calibration.

        Raises:
            FileNotFoundError: Nothing is at the resolved path.
        """
        src = Path(path)
        src = src if src.suffix == ".json" else src / CONSTANTS_JSON
        if not src.is_file():
            raise FileNotFoundError(f"{src}: no calibration. Run `python -m robot_ood.cli build calibrate`.")
        with open(src) as f:
            return cls.from_dict(json.load(f))

"""The build verb group: one sub-verb per artifact family, one registry row each.

build graph makes the graph; every other sub-verb is a pure function of what it wrote.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

from robot_ood.cli.calibrate import CalibrateCommand
from robot_ood.cli.command import SuiteCommand
from robot_ood.cli.constants import ConstantsCommand
from robot_ood.cli.graph import GraphCommand
from robot_ood.cli.language import LanguageCommand
from robot_ood.cli.segments import SegmentsCommand
from robot_ood.data import schema

__all__ = ["BUILD_VERBS", "BuildVerb"]


@dataclass(frozen=True)
class BuildVerb:
    """One build <verb> sub-command: what runs it, what it writes, and its own flags.

    Attributes:
        command_cls: The SuiteCommand subclass this verb dispatches to.
        outputs: Filenames written beside the graph, for --help.
        description: The sub-parser's long description.
        flags: Adds this verb's own flags, beyond --suite and the shared run flags.
    """

    command_cls: type[SuiteCommand]
    outputs: tuple[str, ...]
    description: str
    flags: Callable[[argparse.ArgumentParser], None] | None = None


def _graph_dir_flag(p: argparse.ArgumentParser) -> None:
    """The one flag every post-graph sub-verb shares."""
    p.add_argument("--graph-dir", default=None, help="override the graph dir to read and write "
                   "(default: Paths.graph(<suite>))")


def _graph_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--coarsen", type=int, default=None, help=f"raw frames per node "
                   f"(default: schema.COARSEN={schema.COARSEN})")
    p.add_argument("--hdf5-dir", default=None, help="override the demo hdf5 directory (default: "
                   "Paths.hdf5(<suite>) -- pass this to build from a fixture/test dataset)")
    p.add_argument("--limit-demos", type=int, default=None, help="cap demos read per hdf5 file "
                   "(smoke-test a suite without the full 500 demos)")
    p.add_argument("--limit-tasks", type=int, default=None, help="cap hdf5 files (tasks) read")
    p.add_argument("--virtual-per-entry", type=int, default=None, help="synthesise M virtual "
                   f"nodes per entry-stratum node, joined by `approach` (default: "
                   f"schema.VIRTUAL_PER_ENTRY={schema.VIRTUAL_PER_ENTRY}, i.e. off)")


def _segments_flags(p: argparse.ArgumentParser) -> None:
    _graph_dir_flag(p)
    p.add_argument("--segment-len", type=int, default=None, help="H, raw frames per segment "
                   "(default: robot_ood.core.schemas.SEGMENT_LEN, or the calibrated constants.json "
                   "value beside the graph when present)")
    p.add_argument("--stride", type=int, default=None,
                   help="real-node spacing between segment starts (default: every real node)")


def _calibrate_flags(p: argparse.ArgumentParser) -> None:
    _graph_dir_flag(p)
    p.add_argument("--hdf5-dir", default=None, help="override the demo hdf5 directory "
                   "(default: Paths.hdf5(<suite>))")
    p.add_argument("--n-queries", type=int, default=None, help="calibration queries for the "
                   "seed/posterior support measurement (default: 256)")
    p.add_argument("--min-r2", type=float, default=None, help="action-scale gate floor on either "
                   "block's R2 (default: robot_ood.core.action.DEFAULT_MIN_R2; real-robot corpora "
                   "with command/measurement latency fit lower and need an explicit floor)")


def _language_flags(p: argparse.ArgumentParser) -> None:
    from robot_ood.build.language import DEFAULT_MAX_CROSS

    _graph_dir_flag(p)
    p.add_argument("--hdf5-dir", default=None, help="override the demo hdf5 directory "
                   "(default: Paths.hdf5(<suite>))")
    p.add_argument("--model-dir", default=None,
                   help="encoder snapshot; default is the local Qwen2.5-0.5B")
    p.add_argument("--max-cross", type=float, default=None,
                   help=f"fail if the mean between-task cosine exceeds this "
                   f"(default: {DEFAULT_MAX_CROSS})")


def _constants_flags(p: argparse.ArgumentParser) -> None:
    _calibrate_flags(p)
    p.add_argument("--dest", default=None, metavar="FILE", help="destination FILE for constants.json; "
                   "it must sit outside every graph dir (default: <--out>/constants_<suite>.json)")


# The registry: filenames are literals because argparse needs them before the build modules load
# (test_cli.py::test_build_verb_outputs_match_their_modules keeps them equal to the real constants).
BUILD_VERBS: dict[str, BuildVerb] = {
    "graph": BuildVerb(
        GraphCommand, (schema.NODES_NPZ, schema.EDGES_NPZ),
        "Build the demonstration graph for one suite (or --suite all): coarsen raw demo frames into "
        "nodes (splitting only at gripper-state changes), then build the R=13-relation edge set "
        "(robot_ood.data.schema; the appended `approach` is empty unless asked for). `align` is "
        "different task, same phase "
        "bin, same gripper state -- there is deliberately no `stage` constraint here (see "
        "schema.RELATIONS's module note); a bad build here silently degrades `align`/`sibling` for "
        "every downstream stage.",
        _graph_flags,
    ),
    "segments": BuildVerb(
        SegmentsCommand, ("g_segments.npz",),
        "Build C1's segment memory beside an already-built graph. A pure function of the artifacts "
        "`build graph` already wrote, so running it afterwards is a strict superset, never a re-build.",
        _segments_flags,
    ),
    "calibrate": BuildVerb(
        CalibrateCommand, ("action_scale.json",),
        "Measure this suite's constants off its own corpus and write constants.json "
        "(schema-versioned, content-hashed) plus action_scale.json beside the graph artifacts. No "
        "parameters and no gradient: every number is the output of one measuring function, so a new "
        "suite is a calibration run rather than a code edit. Exit 2 means a pre-registered gate "
        "failed (action-scale R2, or a dilation schedule too shallow to cross the median demo).",
        _calibrate_flags,
    ),
    "language": BuildVerb(
        LanguageCommand, ("g_lang.npz",),
        "Encode every phrasing the corpus records for each task with the frozen local Qwen2.5-0.5B, "
        "centre the table on the per-task centroids, fit the task head on the bank and write it "
        "beside that suite's graph. Two pre-registered gates -- separation (mean between-task "
        "cosine under --max-cross) and distinctness (every task its own nearest neighbour) -- are "
        "fatal unless --force, exit 2.",
        _language_flags,
    ),
    "constants": BuildVerb(
        ConstantsCommand, ("constants.json",),
        "The same measurement `build calibrate` runs, written to --dest instead of beside the graph. "
        "`train fcsr` measures and embeds its own calibration, so this is an export for inspection "
        "or an explicit ROBOT_OOD_CONSTANTS override. A constants.json in a graph dir changes that "
        "graph's hash and makes already-trained checkpoints unloadable against it, so this refuses "
        "such a destination (see robot_ood.build.calibration.check_destination).",
        _constants_flags,
    ),
}

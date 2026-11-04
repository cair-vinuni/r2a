"""The artifacts verb group: the stage gate, seeding a graph directory, and moving one to the bucket.

They act on artifact directories rather than on suites, and none of them leaves a run under
outputs/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robot_ood.cli.errors import CLIError

__all__ = ["ArtifactsCommand"]


class ArtifactsCommand:
    """Dispatch one artifacts verb."""

    label = "artifacts"

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def run(self) -> int:
        """Run the requested verb.

        Returns:
            0 on success; 1 when the gate reports problems.
        """
        return {"check": self._check, "seed": self._seed, "fetch": self._fetch,
                "push": self._push}[self.args.verb]()

    def _fetch(self) -> int:
        """Install a trained artifact tree from the project's bucket, then gate it.

        Returns:
            0 when the downloaded tree passes the deploy-stage gate, 1 otherwise.

        Raises:
            CLIError: The bucket holds nothing under that subdirectory.
        """
        from robot_ood.core.paths import default_paths
        from robot_ood.data.bucket import download

        dest = Path(default_paths().outputs()) / self.args.suite
        try:
            got = download(self.args.source, dest, bucket=self.args.bucket)
        except RuntimeError as exc:
            raise CLIError(str(exc)) from exc
        print(f"[artifacts fetch] {got['files']} files, {got['bytes'] / 1e6:.0f} MB -> {dest}")
        print(f"[artifacts fetch] export GR_GRAPH_DIR={dest}/artifacts")
        if (dest / "constants.json").exists():
            print(f"[artifacts fetch] export ROBOT_OOD_CONSTANTS={dest}/constants.json")
        self.args.graph_dir, self.args.stage, self.args.graph_hash = (
            str(dest / "artifacts"), "deploy", None,
        )
        return self._check()

    def _push(self) -> int:
        """Publish this suite's artifact tree to the bucket, after gating what is on disk.

        Returns:
            0 when the tree gates clean and the upload finishes, 1 when the gate fails.

        Raises:
            CLIError: The bucket write failed.
        """
        from robot_ood.core.paths import default_paths
        from robot_ood.data.bucket import upload

        source = Path(default_paths().outputs()) / self.args.suite
        self.args.graph_dir, self.args.stage, self.args.graph_hash = (
            str(source / "artifacts"), "deploy", None,
        )
        if self._check() != 0:
            return 1
        try:
            sent = upload(source, self.args.dest, bucket=self.args.bucket)
        except RuntimeError as exc:
            raise CLIError(str(exc)) from exc
        print(f"[artifacts push] {sent['files']} files, {sent['bytes'] / 1e6:.0f} MB, "
              f"{sent['deleted']} deleted -> {self.args.bucket}/{self.args.dest}")
        print(f"[artifacts push] python -m robot_ood.cli artifacts fetch {self.args.dest} "
              f"--suite {self.args.suite}")
        return 0

    def _check(self) -> int:
        """Report whether one directory satisfies one stage's required artifact set.

        Returns:
            0 when every check passes, 1 otherwise.

        Raises:
            CLIError: The directory names a latest path, or the stage is unknown.
        """
        from robot_ood.data.store import validate

        try:
            report = validate(self.args.graph_dir, stage=self.args.stage,
                              graph_hash=self.args.graph_hash)
        except ValueError as exc:
            raise CLIError(str(exc)) from exc

        print(f"[artifacts check] graph_dir={report.graph_dir} stage={report.stage} "
              f"suite={report.suite} artifact_digest={report.digest}")
        if report.ok:
            print(f"[artifacts check] OK -- {len(report.required)} artifacts present, unchanged "
                  f"since recording, stamps agree.")
            return 0
        print(f"[artifacts check] FAILED -- {len(report.problems)} problem(s):", file=sys.stderr)
        for problem in report.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    def _seed(self) -> int:
        """Clone a graph directory's non-FCSR artifacts into a fresh one.

        Returns:
            0; a source with no provenance raises instead.

        Raises:
            CLIError: The source directory records no artifacts.
        """
        from robot_ood.build.graph import seed_graph_dir

        try:
            result = seed_graph_dir(self.args.source_dir, self.args.dest_dir)
        except ValueError as exc:
            raise CLIError(str(exc)) from exc

        for name in result.missing:
            print(f"[artifacts seed] WARNING: {result.source / name} recorded in ARTIFACTS.json "
                  f"but missing on disk -- skipped.", file=sys.stderr)
        print(f"[artifacts seed] {result.dest}: copied {len(result.copied)} artifact(s) from "
              f"{result.source} ({', '.join(result.copied)})")
        if result.withheld:
            print(f"[artifacts seed] left for the new FCSR run to write: "
                  f"{', '.join(result.withheld)}")
        return 0

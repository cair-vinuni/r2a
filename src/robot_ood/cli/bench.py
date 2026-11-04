"""The bench verb group: what benchmarks are registered, and provisioning one.

A benchmark is a directory under evals/ holding a bench.yaml. Setup is the same command for every
one of them; what it runs is the script that bench.yaml names.
"""

from __future__ import annotations

import argparse
import subprocess

from robot_ood.cli.errors import CLIError
from robot_ood.eval.bench import Bench, load_all

__all__ = ["BenchCommand"]


class BenchCommand:
    """Dispatch bench list and bench setup."""

    label = "bench"

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def run(self) -> int:
        """Run the requested verb.

        Returns:
            0 on success; the setup script's exit code otherwise.
        """
        return self._list() if self.args.verb == "list" else self._setup()

    def _list(self) -> int:
        """Print every registered benchmark, its axes, and whether it is installed.

        Returns:
            0.
        """
        for bench in load_all():
            state = "installed" if bench.installed else "not installed"
            print(f"{bench.id:<14} {state:<14} results {bench.results_prefix}_*  {bench.root}")
            print(f"{'':<14} suites {' '.join(sorted(bench.task_suites))}")
            print(f"{'':<14} axes   {' '.join(bench.axes)}")
        return 0

    def _setup(self) -> int:
        """Run one benchmark's provisioning script.

        Returns:
            The script's exit code.

        Raises:
            CLIError: No such benchmark.
        """
        try:
            bench = Bench.load(self.args.bench)
        except FileNotFoundError as exc:
            raise CLIError(str(exc)) from exc
        if not bench.setup.is_file():
            raise CLIError(f"{bench.id} declares setup {bench.setup}, which does not exist")
        return subprocess.run(["bash", str(bench.setup)], cwd=bench.setup.parents[2]).returncode



"""File-path access to evals/libero_plus/score.py, the one mp4 counter."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


@dataclass(frozen=True)
class DirCount:
    """score.py's (successes, n) for one result directory."""

    successes: int
    n: int

    @property
    def rate(self) -> float:
        """Success rate in percent; nan for an empty directory."""
        return 100.0 * self.successes / self.n if self.n else float("nan")


class ScoreModule:
    """Lazy, cached loader for the one external score.py module."""

    def __init__(self, score_py: Path) -> None:
        self._score_py = score_py
        self._score: ModuleType | None = None

    @property
    def score(self) -> ModuleType:
        if self._score is None:
            self._score = self._load("sr_score", self._score_py)
        return self._score

    def default_results_root(self) -> Path:
        return Path(self.score.results_root())

    def count_dir(self, directory: Path) -> DirCount:
        """(successes, n) for a result dir; (0, 0) when it does not exist."""
        if not directory.exists():
            return DirCount(0, 0)
        successes, n = self.score.count_dir(str(directory))
        return DirCount(int(successes), int(n))

    @staticmethod
    def _load(name: str, path: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {name} from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module   # register before exec so dataclass/typing self-references resolve
        spec.loader.exec_module(module)
        return module

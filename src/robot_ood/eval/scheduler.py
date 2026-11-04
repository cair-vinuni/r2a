"""Process supervision, port reservation and the LPT GPU schedule."""

from __future__ import annotations

import atexit
import itertools
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any, Mapping, Sequence


from robot_ood.eval.ladder import Cell, REPO_ROOT


class ProcessSupervisor:
    """Tracks every live child so Ctrl-C can't orphan a GPU-resident process."""

    def __init__(self) -> None:
        self._procs: set[subprocess.Popen] = set()
        self._lock = threading.Lock()

    def spawn(self, cmd: Sequence[str], env: Mapping[str, str], stdout: Any,
              cwd: Path = REPO_ROOT) -> subprocess.Popen:
        proc = subprocess.Popen(list(cmd), cwd=cwd, env=dict(env), stdout=stdout,
                                stderr=subprocess.STDOUT)
        with self._lock:
            self._procs.add(proc)
        return proc

    def forget(self, proc: subprocess.Popen) -> None:
        """Stop tracking a process the caller is now reaping itself."""
        with self._lock:
            self._procs.discard(proc)

    def terminate_all(self, grace_s: float = 2.0) -> None:
        with self._lock:
            procs = list(self._procs)
        for proc in procs:
            self._quietly(proc.terminate)
        time.sleep(grace_s)
        for proc in procs:
            if proc.poll() is None:
                self._quietly(proc.kill)

    def install_signal_handlers(self) -> None:
        """Tear everything down on SIGINT/SIGTERM."""
        def handler(signum: int, _frame: FrameType | None) -> None:
            self.terminate_all()
            sys.exit(128 + signum)

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    @staticmethod
    def _quietly(action: Any) -> None:
        try:
            action()
        except OSError:      # best-effort teardown, never let this raise
            pass


class PortAllocator:
    """Hands out policy-server ports that are free both in this process and across the box."""

    MAX_PROBES = 256

    def __init__(self, base_port: int, reservation_dir: Path | None = None) -> None:
        self._base_port = base_port
        self._dir = reservation_dir or (REPO_ROOT / "logs" / ".ports")
        self._lock = threading.Lock()
        self._counter = itertools.count()
        self._held: set[int] = set()
        atexit.register(self.release_all)

    def claim(self) -> int:
        """The next port at or above base_port that is both unbound AND unreserved."""
        with self._lock:
            self._dir.mkdir(parents=True, exist_ok=True)
            for _ in range(self.MAX_PROBES):
                port = self._base_port + next(self._counter)
                if self._bindable(port) and self._reserve(port):
                    self._held.add(port)
                    return port
        raise RuntimeError(
            f"no free port in [{self._base_port}, {self._base_port + self.MAX_PROBES}) "
            f"-- is something leaking servers?"
        )

    def release_all(self) -> None:
        for port in list(self._held):
            (self._dir / str(port)).unlink(missing_ok=True)
            self._held.discard(port)

    @staticmethod
    def _bindable(port: int) -> bool:
        """True iff nothing is listening, tested by actually binding."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
                return True
            except OSError:
                return False

    def _reserve(self, port: int) -> bool:
        """Cross-process claim: create <dir>/<port> exclusively, holding this pid."""
        marker = self._dir / str(port)
        try:
            fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return self._reclaim_if_stale(marker)
        with os.fdopen(fd, "w") as f:
            f.write(f"{os.getpid()}\n")
        return True

    @staticmethod
    def _reclaim_if_stale(marker: Path) -> bool:
        """Take over a reservation whose owning pid is gone; refuse while it is alive."""
        try:
            content = marker.read_text().strip()
            owner: int | None = int(content) if content else None
        except (ValueError, OSError):
            owner = None
        if owner is not None:
            try:
                os.kill(owner, 0)      # signal 0 = liveness probe, sends nothing
                return False           # a live run_sr owns this port
            except (ProcessLookupError, PermissionError, OSError):
                pass                   # stale reservation from a dead run -- take it over
        try:
            marker.write_text(f"{os.getpid()}\n")
            return True
        except OSError:
            return False


@dataclass(frozen=True)
class Track:
    """One worker's queue: the cells that run back to back on one slot of one GPU."""

    gpu: int
    slot: int
    cells: list[Cell]
    gpus: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.gpus:
            object.__setattr__(self, "gpus", (self.gpu,))


class Schedule:
    """An LPT assignment of cells to GPU tracks, per_gpu tracks to a GPU, run concurrently."""

    def __init__(self, tracks: Sequence[Track]) -> None:
        self.tracks: tuple[Track, ...] = tuple(tracks)

    @property
    def assignments(self) -> dict[int, list[Cell]]:
        """Cells per GPU, every slot's queue merged, in track order."""
        out: dict[int, list[Cell]] = {}
        for track in self.tracks:
            out.setdefault(track.gpu, []).extend(track.cells)
        return out

    @classmethod
    def build(cls, cells: Sequence[Cell], gpus: Sequence[int], per_gpu: int = 1) -> "Schedule":
        """Greedy LPT: largest cell first, always onto the least-loaded track.

        Args:
            cells: The cells to place.
            gpus: Physical GPU indices.
            per_gpu: Tracks per GPU, i.e. cells that may run on one GPU at the same time; one
                batch-1 policy server keeps an A100 about half busy, three fill it.

        Returns:
            The schedule; ties break toward the lower GPU, then the lower slot.
        """
        if per_gpu < 1:
            raise ValueError(f"per_gpu must be at least 1, got {per_gpu}")
        keys = [(g, slot) for g in gpus for slot in range(per_gpu)]
        queues: dict[tuple[int, int], list[Cell]] = {k: [] for k in keys}
        load = dict.fromkeys(keys, 0)
        for cell in sorted(cells, key=lambda c: c.expected_n, reverse=True):
            key = min(load, key=lambda k: (load[k], k))
            queues[key].append(cell)
            load[key] += cell.expected_n
        tracks = [Track(g, slot, queues[(g, slot)]) for g, slot in keys]
        return cls(cls._share_idle_gpus(tracks) if per_gpu == 1 else tracks)

    @staticmethod
    def _share_idle_gpus(tracks: list[Track]) -> list[Track]:
        """Hand the GPUs of empty tracks to the busy ones, round robin, busiest first.

        Args:
            tracks: One track per GPU.

        Returns:
            The same tracks; a busy one may now list extra GPUs its cells' servers can spread over.
        """
        busy = sorted((t for t in tracks if t.cells),
                      key=lambda t: -sum(c.expected_n for c in t.cells))
        idle = [t.gpu for t in tracks if not t.cells]
        if not busy or not idle:
            return tracks
        extra: dict[int, list[int]] = {t.gpu: [] for t in busy}
        for i, gpu in enumerate(idle):
            extra[busy[i % len(busy)].gpu].append(gpu)
        return [Track(t.gpu, t.slot, t.cells, (t.gpu, *extra.get(t.gpu, []))) if t.cells else t
                for t in tracks]

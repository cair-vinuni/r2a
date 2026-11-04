"""Run one long step as a supervised process: pid file, log path, exit code.

A detached job is polled by PROCESS STATE, never by whether its output file has appeared -- a job
that dies at minute five must be seen at minute five, not waited on forever.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

__all__ = ["Job", "JobResult", "run_job", "spawn_job"]


@dataclass(frozen=True)
class JobResult:
    """What one finished job left behind.

    Attributes:
        name: The job name.
        exit_code: The process exit code.
        seconds: Wall-clock duration.
        log_path: Where its combined stdout/stderr went.
    """

    name: str
    exit_code: int
    seconds: float
    log_path: Path

    @property
    def ok(self) -> bool:
        """Whether the process exited cleanly."""
        return self.exit_code == 0


def _in_process_table(pid: int) -> bool:
    """Whether a pid exists at all, for a platform without /proc.

    Args:
        pid: The process id.

    Returns:
        True when signalling it would reach something.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _prepare(name: str, log_dir: str | os.PathLike[str]) -> Path:
    """Make the log directory and return this job's log path.

    Args:
        name: The job name.
        log_dir: Directory for job logs and pid files.

    Returns:
        The log file path.
    """
    d = Path(log_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.log"


def run_job(name: str, argv: Sequence[str], log_dir: str | os.PathLike[str], *,
            env: Mapping[str, str] | None = None, cwd: str | os.PathLike[str] | None = None,
            echo: bool = True, tail: int = 25) -> JobResult:
    """Run argv to completion, teeing its output to a log and to stdout.

    Args:
        name: The job name, used for the log filename.
        argv: The command.
        log_dir: Directory for the log file.
        env: Full environment for the child; None inherits this process's.
        cwd: Working directory for the child.
        echo: Mirror the last tail lines to stdout as they arrive.
        tail: How many trailing lines to mirror.

    Returns:
        The job's result.
    """
    log_path = _prepare(name, log_dir)
    t0 = time.monotonic()
    recent: list[str] = []
    with log_path.open("w") as fh:
        proc = subprocess.Popen(list(argv), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=dict(env) if env else None, cwd=cwd)
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            recent.append(line.rstrip("\n"))
            if len(recent) > tail:
                recent.pop(0)
        rc = proc.wait()
    if echo:
        for line in recent:
            print(f"  | {line}")
    return JobResult(name, rc, time.monotonic() - t0, log_path)


@dataclass(frozen=True)
class Job:
    """A detached process this tree started, addressable across invocations.

    Attributes:
        name: The job name.
        pid: The process id.
        log_path: Where its combined stdout/stderr goes.
        pid_path: The JSON file recording pid, argv and start time.
        started: Monotonic-independent wall clock at spawn.
    """

    name: str
    pid: int
    log_path: Path
    pid_path: Path
    started: float

    def alive(self) -> bool:
        """Whether the process is still running -- a zombie has exited and does not count."""
        try:
            if os.waitpid(self.pid, os.WNOHANG)[0] == self.pid:
                return False
        except ChildProcessError:
            pass                    # not this process's child: read the process table instead
        except OSError:
            return False
        stat = Path(f"/proc/{self.pid}/stat")
        if not stat.exists():
            return _in_process_table(self.pid)
        try:
            return stat.read_text().rsplit(") ", 1)[1].split()[0] != "Z"
        except (OSError, IndexError):
            return False

    def wait(self, *, poll: float = 5.0, timeout: float | None = None) -> JobResult:
        """Block until the process leaves the process table.

        Args:
            poll: Seconds between process-state checks.
            timeout: Give up after this many seconds; None waits indefinitely.

        Returns:
            The job's result; exit_code is -1 when the process vanished without one being recorded.

        Raises:
            TimeoutError: The process was still alive at the deadline.
        """
        deadline = None if timeout is None else time.time() + timeout
        while self.alive():
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(f"{self.name}: still running after {timeout}s (pid {self.pid}, "
                                   f"log {self.log_path})")
            time.sleep(poll)
        return JobResult(self.name, self.exit_code(), time.time() - self.started, self.log_path)

    def exit_code(self) -> int:
        """The exit code the wrapper recorded, or -1 while it is running or if it never wrote one."""
        try:
            rc = json.loads(self.pid_path.read_text()).get("exit_code")
        except (OSError, ValueError):
            return -1
        return -1 if rc is None else int(rc)

    def stop(self) -> None:
        """Ask the process group to stop, then insist."""
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if not self.alive():
                return
            try:
                os.killpg(os.getpgid(self.pid), sig)
            except (ProcessLookupError, PermissionError):
                return
            time.sleep(5)


def spawn_job(name: str, argv: Sequence[str], log_dir: str | os.PathLike[str], *,
              env: Mapping[str, str] | None = None,
              cwd: str | os.PathLike[str] | None = None) -> Job:
    """Start argv in its own session and record where it went.

    Args:
        name: The job name, used for the log and pid filenames.
        argv: The command.
        log_dir: Directory for the log and pid files.
        env: Full environment for the child; None inherits this process's.
        cwd: Working directory for the child.

    Returns:
        The running job.
    """
    log_path = _prepare(name, log_dir)
    pid_path = log_path.with_suffix(".pid.json")
    pid_path.unlink(missing_ok=True)
    # Only something inside the session can record the exit code, so the wrapper writes the pid file
    # at both ends -- at the start too, so an earlier run's code cannot read as this one's.
    quoted = " ".join(shlex.quote(str(a)) for a in argv)
    wrapper = "\n".join([
        f"exec >>{shlex.quote(str(log_path))} 2>&1",
        f'printf \'{{"pid": %d, "exit_code": null}}\\n\' $$ > {shlex.quote(str(pid_path))}',
        f"{quoted}; rc=$?",
        f'printf \'{{"pid": %d, "exit_code": %d}}\\n\' $$ $rc > {shlex.quote(str(pid_path))}',
        "exit $rc",
    ])
    proc = subprocess.Popen(["bash", "-c", wrapper], start_new_session=True,
                            stdin=subprocess.DEVNULL, env=dict(env) if env else None, cwd=cwd)
    return Job(name, proc.pid, log_path, pid_path, time.time())

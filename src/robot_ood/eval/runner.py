"""Running one cell: lock, resume decision, launch, score, record."""

from __future__ import annotations

import atexit
import dataclasses
import enum
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


from robot_ood.eval.bench import EGL_RENDERER
from robot_ood.eval.egl_devices import egl_device_for_cuda
from robot_ood.eval.episodes import (EPISODES_JSONL, format_table, pending_episodes, per_task,
                                     read_episodes, write_summary)
from robot_ood.eval.ladder import (Cell, CellReport, CellScore, GateVerdict, LadderConfig,
                                   make_log_dir)
from robot_ood.eval.launcher import EnvBuilder, LaunchResult, PolicyServer
from robot_ood.eval.policy import load_policy
from robot_ood.eval.preflight import GraphArtifacts, graph_artifact_status, resolve_graph_dir
from robot_ood.eval.provenance import (Ledger, LedgerRecord, RunStatus, run_config_sha,
                                       update_run_json, write_run_json, GitState)
from robot_ood.eval.scheduler import PortAllocator, ProcessSupervisor
from robot_ood.pipeline.interpreter import resolve_env_python


VIDEO_ENV = "ROBOT_OOD_VIDEO"


class ExitCode(enum.IntEnum):
    """This driver's process exit codes. int so sys.exit / main can return one directly."""

    OK = 0
    ERROR = 1
    GATE_FAILED = 2


class CellAction(str, enum.Enum):
    """What ResumePlanner decided to do with one cell's existing result dir."""

    RUN = "run"
    SKIP = "skip"
    PURGE = "purge"
    ERROR = "error"


class CellLockedError(RuntimeError):
    """Raised when another live driver already owns a cell's result dir."""


class CellLock:
    """An exclusive claim on one result dir, held for as long as the cell runs."""

    NAME = ".run_sr.lock"

    def __init__(self, result_dir: Path) -> None:
        self._path = Path(result_dir) / self.NAME
        self._held = False

    def acquire(self, tag: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._claim(tag) and not (self._steal_if_stale() and self._claim(tag)):
            raise CellLockedError(f"{self._path.parent} is owned by {self._owner_note()}")
        self._held = True
        atexit.register(self.release)

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        try:
            self._path.unlink()
        except FileNotFoundError:   # already reaped by a takeover
            pass

    def _claim(self, tag: str) -> bool:
        try:
            fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as f:
            json.dump({"pid": os.getpid(), "host": socket.gethostname(), "tag": tag,
                       "started": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                      f, indent=2)
        return True

    def _owner(self) -> dict[str, Any]:
        try:
            return json.loads(self._path.read_text())
        except (OSError, ValueError):   # an unreadable claim still counts as owned
            return {}

    def _owner_note(self) -> str:
        owner = self._owner()
        return (f"pid {owner.get('pid', '?')} on {owner.get('host', '?')} "
                f"since {owner.get('started', '?')}")

    @classmethod
    def live_owner(cls, result_dir: Path) -> str | None:
        """The note for a live claim on this dir, None when it is unclaimed or the claim is stale."""
        lock = cls(result_dir)
        if not lock._path.exists():
            return None
        return None if cls._stale(lock._owner()) else lock._owner_note()

    @staticmethod
    def _stale(owner: Mapping[str, Any]) -> bool:
        """Whether this host recorded the claim and that pid is gone.

        Args:
            owner: The claim read off the lock file.

        Returns:
            Whether the claim can be taken over.
        """
        pid = owner.get("pid")
        if owner.get("host") != socket.gethostname() or not isinstance(pid, int):
            return False
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            return True
        # Nothing reaps a driver whose parent was killed, and a zombie keeps its /proc entry for as
        # long as this container lives; it holds no result dir.
        return stat.rpartition(")")[2].split()[0] == "Z"

    def _steal_if_stale(self) -> bool:
        owner = self._owner()
        if not self._stale(owner):
            return False
        print(f"[lock] taking over {self._path.parent} from dead pid {owner['pid']}",
              file=sys.stderr)
        try:
            self._path.unlink()
        except FileNotFoundError:   # a competing driver reaped it first
            return False
        return True


def episode_slice(pending: Sequence[tuple[int, int]], trials: int) -> list[tuple[int, int]]:
    """The pending episodes this host runs under ROBOT_OOD_EPISODE_SLICE=i/n; all of them when unset.

    Several hosts split one cell by running the same command with i = 0..n-1, each into its own
    results root. An episode's seed is a function of its (task, episode) pair, so the split does
    not change any episode.

    Args:
        pending: The (task, episode) pairs no client has recorded.
        trials: Episodes per task.

    Returns:
        The pairs whose grid index is i modulo n.
    """
    spec = os.environ.get("ROBOT_OOD_EPISODE_SLICE", "")
    if not spec:
        return list(pending)
    i, n = (int(x) for x in spec.split("/"))
    return [(t, e) for t, e in pending if (t * trials + e) % n == i]


@dataclass(frozen=True)
class CellPlan:
    cell: Cell
    action: CellAction
    reason: str


class ResumePlanner:
    """Decides what to do with a cell's existing result dir."""

    def __init__(self, config: LadderConfig, ledger: Ledger) -> None:
        self._config = config
        self._ledger = ledger
        self._results_root = config.results_root

    def plan(self, cell: Cell, resume: bool,
             records: Sequence[LedgerRecord] | None = None,
             rerun_incomplete: bool = False) -> CellPlan:
        directory = cell.result_dir(self._results_root)
        if not (directory.exists() and (any(directory.rglob(EPISODES_JSONL)) or any(directory.rglob("*.mp4")))):
            return CellPlan(cell, CellAction.RUN, "no existing result dir")
        if not (resume or rerun_incomplete):
            return CellPlan(cell, CellAction.ERROR,
                            f"result dir already has content at {directory} (pass --resume)")
        record = self._ledger.latest(cell.tag, cell.mode, records)
        if record and record.status is RunStatus.COMPLETE:
            count = self._config.scorer.count_dir(directory)
            if count.n == cell.expected_n:
                return CellPlan(cell, CellAction.SKIP, (
                    f"ledger says complete and {count.n} mp4s match expected_n={cell.expected_n}"
                ))
        if rerun_incomplete:
            return CellPlan(cell, CellAction.PURGE, "partial/stale result dir -- would be purged and re-run")
        return CellPlan(cell, CellAction.RUN,
                        "partial result dir -- the episodes it has no record of would be run")

    def apply(self, plan: CellPlan) -> None:
        """Perform the filesystem side effect for a purge verdict."""
        if plan.action is not CellAction.PURGE:
            return
        directory = plan.cell.result_dir(self._results_root)
        owner = CellLock.live_owner(directory)
        if owner:
            raise CellLockedError(f"refusing to purge {directory}: owned by {owner}")
        shutil.rmtree(directory)


class CellScorer:
    """Scores a cell's result dir and applies its gate."""

    def __init__(self, config: LadderConfig) -> None:
        self._config = config
        self._scorer = config.scorer
        self._results_root = config.results_root

    def score(self, cell: Cell) -> CellScore:
        return CellScore(cell=cell, count=self._scorer.count_dir(cell.result_dir(self._results_root)))

    def report(self, cell: Cell, allow_partial: bool) -> CellReport:
        score = self.score(cell)
        return CellReport(score=score, verdict=self.verdict(score, allow_partial))

    def verdict(self, score: CellScore, allow_partial: bool) -> GateVerdict:
        """Completeness first, then the cell's own gate."""
        cell = score.cell
        if not score.complete and not allow_partial:
            return GateVerdict(False, (
                f"INCOMPLETE: {score.actual_n}/{cell.expected_n} episodes "
                f"(excluded from totals; pass --allow-partial to score anyway)"
            ))
        if cell.gate is None:
            return GateVerdict(True, "no gate configured")
        return cell.gate.evaluate(score)


class CellRunner:
    """Runs one cell end to end: provenance, launch, score, ledger, verdict."""

    MAX_CLIENT_PASSES = 2
    CLIENT_STALL_S = 180.0
    CLIENT_POLL_S = 20.0

    def __init__(self, config: LadderConfig, env_builder: EnvBuilder, scorer: CellScorer,
                 ledger: Ledger, ports: PortAllocator, supervisor: ProcessSupervisor,
                 allow_partial: bool = False) -> None:
        self._config = config
        self._env_builder = env_builder
        self._scorer = scorer
        self._ledger = ledger
        self._ports = ports
        self._supervisor = supervisor
        self._allow_partial = allow_partial
        self._git = GitState.probe(config.root)
        self._python = resolve_env_python(config.bench.conda_env)

    def run(self, cell: Cell, gpus: Sequence[int] | int) -> ExitCode:
        """Claim the cell's result dir, then run it. A cell another live driver owns is an error."""
        gpus = (gpus,) if isinstance(gpus, int) else tuple(gpus)
        lock = CellLock(cell.result_dir(self._config.results_root))
        try:
            lock.acquire(cell.tag)
        except CellLockedError as exc:
            print(f"RESULT {cell.tag}: CRASHED -- {exc}", file=sys.stderr)
            return ExitCode.ERROR
        try:
            return self._run_locked(cell, gpus)
        finally:
            lock.release()

    @staticmethod
    def place(cell: Cell, n_gpus: int) -> Cell:
        """Fill a cell's zero server and client counts from its policy's placement.

        Args:
            cell: The cell; servers or num_clients of 0 mean "the policy decides".
            n_gpus: GPUs the cell was given.

        Returns:
            The cell with both counts set: servers_per_gpu x n_gpus servers, clients_per_server
            per server.
        """
        placement = load_policy(cell.policy).placement
        servers = cell.servers or placement.servers(n_gpus)
        clients = cell.num_clients or servers * placement.clients_per_server
        return dataclasses.replace(cell, servers=servers, num_clients=clients)

    def _run_locked(self, cell: Cell, gpus: tuple[int, ...]) -> ExitCode:
        """Run and score one cell; its servers spread over gpus, the first is its home."""
        gpu = gpus[0]
        if self._python is None:
            raise RuntimeError(f"cannot locate the `{self._config.bench.conda_env}` conda env's python")
        cell = self.place(cell, len(gpus))
        print(f"[run_sr] {cell.tag}: {cell.servers} server(s) over GPU(s) {list(gpus)}, "
              f"{cell.num_clients} sim client(s)")

        if not cell.expected_n:
            print(f"RESULT {cell.tag}: CRASHED -- no task files for {cell.client_suite}; "
                  f"run evals/{cell.bench_id}/setup.sh", file=sys.stderr)
            return ExitCode.ERROR

        port = self._ports.claim()
        ports = [port] + [self._ports.claim() for _ in range(cell.servers - 1)]
        log_dir = make_log_dir(cell.policy, cell.mode, cell.suite, cell.axis, cell.variant,
                               self._config.root)
        # the renderer overlay is part of the recipe run.json records: which EGL device drew
        env = {**self._env_builder.for_cell(cell), **self._renderer_env(gpu)}
        artifacts = self._require_graph_artifacts(cell, env)
        result_dir = cell.result_dir(self._config.results_root)

        write_run_json(
            result_dir, policy=cell.policy, mode=cell.mode, suite=cell.suite, axis=cell.axis,
            expected_n=cell.expected_n, git_commit=self._git.sha, git_dirty=self._git.dirty,
            protocol=load_policy(cell.policy).protocol, seed=cell.seed,
            run_config_sha=run_config_sha(artifacts.graph_dir),
            mode_env=env, **artifacts.as_record(),
        )

        base = dict(
            tag=cell.tag, suite=cell.suite, base_task_suite=cell.base_task_suite, axis=cell.axis, mode=cell.mode,
            gpu=gpu, port=port, expected_n=cell.expected_n, git_sha=self._git.sha,
            git_dirty=self._git.dirty, env=env, result_dir=str(result_dir), log_dir=str(log_dir),
            graph_dir=str(artifacts.graph_dir), graph_hash=artifacts.graph_hash,
            g_head_mtime=artifacts.g_head_mtime,
        )
        self._ledger.append(LedgerRecord(status=RunStatus.RUNNING, **base))

        try:
            launch = self._launch(cell, gpus, ports, log_dir, env)
        except Exception as exc:   # noqa: BLE001 -- record the crash, don't kill the schedule
            self._ledger.append(LedgerRecord(status=RunStatus.CRASHED, error=str(exc), **base))
            print(f"RESULT {cell.tag}: CRASHED -- {exc}", file=sys.stderr)
            return ExitCode.ERROR

        report = self._scorer.report(cell, self._allow_partial)
        score = report.score
        episodes_per_min = (score.actual_n / (launch.wall_s / 60.0)) if launch.wall_s > 0 and score.actual_n else 0.0
        status = (RunStatus.CRASHED if launch.crashed
                  else (RunStatus.COMPLETE if score.complete else RunStatus.INCOMPLETE))
        self._ledger.append(LedgerRecord(
            status=status, actual_n=score.actual_n, successes=score.successes,
            rate=score.rate, wall_s=launch.wall_s, episodes_per_min=episodes_per_min, **base,
        ))
        update_run_json(result_dir, actual_n=score.actual_n, successes=score.successes,
                        status=status.value, duration_s=launch.wall_s)
        write_summary(result_dir, log_dir, policy=cell.policy, mode=cell.mode, bench=cell.bench_id,
                      suite=cell.suite, task_suite=cell.client_suite, axis=cell.axis,
                      protocol=load_policy(cell.policy).protocol, expected_n=cell.expected_n,
                      status=status.value, wall_s=round(launch.wall_s, 1),
                      episodes_per_min=round(episodes_per_min, 1), servers=cell.servers,
                      num_clients=cell.num_clients, gpus=list(gpus),
                      video=os.environ.get(VIDEO_ENV, "none"),
                      git_commit=self._git.sha, git_dirty=self._git.dirty)
        print(report.line + launch.crash_note)
        table = format_table(per_task(read_episodes(result_dir)))
        if table:
            print(table)
        print(f"  {score.actual_n} episodes in {launch.wall_s / 60:.1f} min = {episodes_per_min:.1f}/min "
              f"({cell.servers} servers, {cell.num_clients} clients, GPUs {list(gpus)})")
        print(f"  results {result_dir}\n  logs    {log_dir}")

        if status is not RunStatus.COMPLETE and not self._allow_partial:
            return ExitCode.ERROR
        return ExitCode.OK if report.verdict.passed else ExitCode.GATE_FAILED

    def _require_graph_artifacts(self, cell: Cell, env: dict[str, str]) -> GraphArtifacts:
        """Fail before any GPU launch when a non-base mode's tree for the cell's policy is unusable.

        Args:
            cell: The cell about to launch.
            env: Its environment, given the tree and the graph dir the clients deploy from.

        Returns:
            The artifacts of the gated dir; of the first lane when the suite runs one graph per arm.
        """
        from robot_ood.core.paths import lane_graph_dirs, policy_tree

        # The clients read the tree from ROBOT_OOD_TREE; a single-armed one reads its graph dir
        # from GR_GRAPH_DIR, where a caller's own export still wins.
        if cell.mode == "base":
            return self._gate_graph_dir(cell, resolve_graph_dir(cell.suite, env))
        tree = policy_tree(cell.suite, cell.policy)
        env["ROBOT_OOD_TREE"] = tree
        lanes = lane_graph_dirs(tree)
        if lanes:
            dirs = [lanes[arm] for arm in sorted(lanes)]
        else:
            dirs = [resolve_graph_dir(tree, env)]
            env.setdefault("GR_GRAPH_DIR", str(dirs[0]))
        gated = [self._gate_graph_dir(cell, d) for d in dirs]
        return gated[0]

    def _gate_graph_dir(self, cell: Cell, graph_dir: Path) -> GraphArtifacts:
        """Gate one graph directory for one cell."""
        if cell.mode == "base":
            artifacts = graph_artifact_status(graph_dir)
            print(f"[run_sr] {cell.tag}: mode=base graph_dir={artifacts.graph_dir} "
                  f"(no artifact gate -- base does not read the graph)")
            return artifacts

        from robot_ood.core.kinematics import entry_walks
        from robot_ood.data.store import REQUIRED_ARTIFACTS, ArtifactManifest, assert_not_latest

        try:
            resolved = assert_not_latest(graph_dir)
        except ValueError as exc:
            raise RuntimeError(f"{cell.tag}: {exc}") from exc

        if cell.mode != "fcsr":
            artifacts = graph_artifact_status(resolved)
            print(f"[run_sr] {cell.tag}: mode={cell.mode} graph_dir={artifacts.graph_dir} "
                  f"graph_hash={artifacts.graph_hash} "
                  f"g_head_mtime={artifacts.g_head_mtime}")
            if not artifacts.ok:
                raise RuntimeError(
                    f"{cell.tag}: resolved graph dir {artifacts.graph_dir} is missing required "
                    f"artifacts: {list(artifacts.missing)} -- train the graph for this suite first, "
                    f"or point GR_GRAPH_DIR at a suite that already has g_head.npz."
                )
            return artifacts

        from robot_ood.build.policy_cache import cached_policy_protocol

        manifest = ArtifactManifest.load(resolved)
        stage = "deploy" if entry_walks(manifest.suite) else "deploy_no_entry"
        problems = list(manifest.verify(REQUIRED_ARTIFACTS[stage]))
        # The corrector learned the chunks of the policy that cached a_pi_raw.npz; served another
        # policy it corrects chunks it never saw. A cache from before the stamp cannot be checked.
        expected, cached = load_policy(cell.policy).protocol, cached_policy_protocol(resolved)
        if cached is not None and cached != expected:
            problems.append(
                f"a_pi_raw.npz was cached under protocol {cached!r}, but {cell.policy} speaks "
                f"{expected!r}: this tree's corrector learned another policy's chunks. Run "
                f"`python -m robot_ood.cli cache-policy --suite {manifest.suite}` and retrain fcsr, "
                f"or name the tree built against {cell.policy}"
            )
        print(f"[run_sr] {cell.tag}: mode={cell.mode} graph_dir={resolved} "
              f"artifact_digest={manifest.digest()} suite={manifest.suite} "
              f"cached_protocol={cached}")
        if problems:
            raise RuntimeError(
                f"{cell.tag}: graph dir {resolved} failed the deploy artifact gate:\n"
                + "\n".join(f"  - {p}" for p in problems)
            )
        status = graph_artifact_status(resolved)
        return dataclasses.replace(status, artifact_digest=manifest.digest())

    def _launch(self, cell: Cell, gpus: Sequence[int], ports: Sequence[int], log_dir: Path,
                cell_env: Mapping[str, str]) -> LaunchResult:
        assert self._python is not None
        started = time.time()
        result_dir = cell.result_dir(self._config.results_root)
        result_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        policy = load_policy(cell.policy)
        checkpoint = self._config.suite(cell.suite).checkpoint(policy.ckpt_key)

        def env_for(gpu: int, role: str) -> dict[str, str]:
            shared = {**os.environ, **cell_env, "CUDA_VISIBLE_DEVICES": str(gpu)}
            if role == "server":
                return {**shared, "PYTHONPATH": self._env_builder.server_pythonpath(policy),
                        **self._thread_env("server")}
            return {**shared, "PYTHONPATH": self._env_builder.client_pythonpath,
                    **self._renderer_env(gpu), **self._thread_env("client")}

        with ExitStack() as stack:
            servers = [
                PolicyServer(
                    self._config, self._supervisor, checkpoint, port,
                    log_dir, env_for(gpus[i % len(gpus)], "server"), policy=policy,
                    log_name=f"server{i}.log" if len(ports) > 1 else "server.log",
                )
                for i, port in enumerate(ports)
            ]
            # spawn every server first, then wait: eight GR00T loads take one load's time, not eight
            for server in servers:
                stack.push(server.start())
            for server in servers:
                server.wait_ready()
            returncodes = self._run_clients(cell, checkpoint, ports, gpus, log_dir, result_dir, env_for)

        return LaunchResult(wall_s=time.time() - started, client_returncodes=returncodes)

    def _run_clients(self, cell: Cell, checkpoint: Path, ports: Sequence[int], gpus: Sequence[int],
                     log_dir: Path, result_dir: Path,
                     env_for: Callable[[int, str], dict[str, str]]) -> tuple[int, ...]:
        """Deal the cell's episodes to its clients, then re-deal whatever no client recorded.

        Args:
            cell: The cell being run.
            checkpoint: The policy checkpoint the clients are told to expect.
            ports: The policy servers' ports.
            gpus: The cell's GPUs.
            log_dir: Where each client's stdout is kept.
            result_dir: The cell's result directory, whose records say what is still missing.
            env_for: Builds the environment for one (gpu, role).

        Returns:
            The last pass' client return codes.
        """
        # A client that dies on one scene takes its whole block of episodes with it, and purging the
        # cell to run all of them again costs more than the episodes that are actually missing.
        n_tasks = cell.expected_n // cell.trials
        returncodes: tuple[int, ...] = ()
        before: list[tuple[int, int]] | None = None
        for attempt in range(self.MAX_CLIENT_PASSES):
            pending = episode_slice(pending_episodes(result_dir, n_tasks, cell.trials), cell.trials)
            if not pending:
                break
            # A scene that hangs the renderer hangs it again, so a pass that recorded none of what
            # it was given is the end of what re-dealing can recover.
            if pending == before:
                print(f"[run_sr] {cell.tag}: pass {attempt} recorded none of its "
                      f"{len(pending)} episode(s); not dealing them again", flush=True)
                break
            before = pending
            prefix = "client" if not attempt else f"client_pass{attempt + 1}"
            pending_file = log_dir / f"{prefix}_episodes.json"
            pending_file.write_text(json.dumps([list(pair) for pair in pending]) + "\n")
            episodes = [self._config.bench.episodes_flag, str(pending_file)]
            clients = min(cell.num_clients, len(pending))
            print(f"[run_sr] {cell.tag}: pass {attempt + 1} over the {len(pending)} episode(s) "
                  f"no client has recorded, on {clients} client(s)", flush=True)
            targets = self.shard_targets(clients, ports, gpus)
            logs = [log_dir / f"{prefix}_shard{shard}.log" for shard in range(len(targets))]
            with ExitStack() as files:
                procs = [
                    self._supervisor.spawn(
                        self._client_cmd(cell, checkpoint, port, shard, clients, result_dir) + episodes,
                        env=env_for(gpu, "client"),
                        stdout=files.enter_context(open(logs[shard], "w")),
                    )
                    for shard, (port, gpu) in enumerate(targets)
                ]
                returncodes = self._wait_clients(procs, logs, cell.tag)
                for proc in procs:
                    self._supervisor.forget(proc)
        return returncodes

    def _wait_clients(self, procs: Sequence[subprocess.Popen], logs: Sequence[Path],
                      tag: str) -> tuple[int, ...]:
        """Wait for one pass' clients, killing any whose log has stopped growing.

        Args:
            procs: The clients, in shard order.
            logs: Each client's stdout file, whose growth is its sign of life.
            tag: The cell tag, named in the message a kill prints.

        Returns:
            One return code per client, in shard order.
        """
        # A sim client can wedge inside the renderer with the GPU idle and a core spinning, and it
        # never returns; the episodes it was holding are dealt to a later pass instead.
        codes: list[int | None] = [None] * len(procs)
        seen = [(-1, time.monotonic())] * len(procs)
        while any(code is None for code in codes):
            time.sleep(self.CLIENT_POLL_S)
            now = time.monotonic()
            for i, proc in enumerate(procs):
                if codes[i] is not None:
                    continue
                codes[i] = proc.poll()
                if codes[i] is not None:
                    continue
                size = logs[i].stat().st_size if logs[i].exists() else 0
                if size != seen[i][0]:
                    seen[i] = (size, now)
                elif now - seen[i][1] > self.CLIENT_STALL_S:
                    print(f"[run_sr] {tag}: client {i} has printed nothing for "
                          f"{self.CLIENT_STALL_S / 60:.0f} min; killing it", flush=True)
                    proc.kill()
        return tuple(int(code) for code in codes)

    @staticmethod
    def shard_targets(num_clients: int, ports: Sequence[int], gpus: Sequence[int]) -> list[tuple[int, int]]:
        """Which server, and so which GPU, each client shard talks to and renders on.

        Args:
            num_clients: Shards in the cell.
            ports: One port per policy server; server i lives on gpus[i % len(gpus)].
            gpus: The GPUs the cell may use, home GPU first.

        Returns:
            Per shard, (port, gpu): shards round-robin over the servers.
        """
        return [(ports[k % len(ports)], gpus[(k % len(ports)) % len(gpus)]) for k in range(num_clients)]

    @staticmethod
    def _thread_env(role: str) -> dict[str, str]:
        """OMP/MKL/OpenBLAS thread counts for a child, by role.

        Args:
            role: "client" (physics and a small FCSR net; one thread, so twenty-odd clients do not
                oversubscribe the cores) or "server" (torch image preprocessing; two).

        Returns:
            The three thread variables.
        """
        n = {"client": "1", "server": "2"}[role]
        return {"OMP_NUM_THREADS": n, "MKL_NUM_THREADS": n, "OPENBLAS_NUM_THREADS": n}

    def _renderer_env(self, gpu: int) -> dict[str, str]:
        """MUJOCO_EGL_DEVICE_ID for the cell's GPU under the NVIDIA EGL driver; empty under mesa.

        Args:
            gpu: The cell's physical CUDA device.

        Returns:
            The two variables to overlay on the client env, or nothing -- also nothing for a
            benchmark that renders through Vulkan, whose client must see its own GPU alone.
        """
        if self._config.bench.renderer != EGL_RENDERER:
            return {}
        egl = egl_device_for_cuda(gpu)
        if egl is None:
            return {}
        # robosuite insists the EGL id is a substring of CUDA_VISIBLE_DEVICES (binding_utils.py:33):
        # a client whose two indices differ is shown both, its own GPU first so cuda:0 is unchanged.
        visible = str(gpu) if egl == gpu else f"{gpu},{egl}"
        return {"MUJOCO_EGL_DEVICE_ID": str(egl), "CUDA_VISIBLE_DEVICES": visible}

    def _client_cmd(self, cell: Cell, checkpoint: Path, port: int, shard: int, num_shards: int,
                    result_dir: Path) -> list[str]:
        assert self._python is not None
        return [
            str(self._python), str(self._config.bench.client),
            *self._config.bench.client_argv(
                checkpoint=str(checkpoint), host="127.0.0.1", port=port,
                task_suite=cell.client_suite, category_value=cell.category_value,
                trials=cell.trials, shard=shard, num_shards=num_shards,
                result_dir=str(result_dir), protocol=load_policy(cell.policy).protocol,
            ),
            *([] if cell.seed is None else ["--seed", str(cell.seed)]),
        ]

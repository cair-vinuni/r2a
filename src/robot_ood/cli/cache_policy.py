"""cache-policy: the frozen policy's own chunks over one suite's demo corpus.

The env this builds must exist BEFORE the child python starts -- LD_PRELOAD and LD_LIBRARY_PATH are
read by the dynamic linker -- which is why this composes an environment and spawns, rather than
importing the policy into this process.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

from robot_ood.cli.command import SuiteCommand, SuiteResult
from robot_ood.core.build_config import build_param
from robot_ood.cli.errors import CLIError
from robot_ood.data import schema
from robot_ood.core.report import StageLogger
from robot_ood.eval.policy import load_policy
from robot_ood.pipeline.interpreter import nvml_preload_path
from robot_ood.pipeline.jobs import run_job, spawn_job

__all__ = ["CachePolicyCommand"]

from robot_ood.core.paths import repo_root, suite_config
from robot_ood.data.schema import A_PI_RAW_NPZ
SERVER_READY_TIMEOUT = 1_200.0   # matches server_ready_timeout_s in configs/sr_ladder*.yaml


def _gpu_list(spec: str) -> list[str]:
    """The GPUs servers are placed on, round-robin.

    Args:
        spec: Comma-separated CUDA device indices.

    Returns:
        One entry per device, in the order given.

    Raises:
        CLIError: The value names no device.
    """
    gpus = [part.strip() for part in str(spec).split(",") if part.strip()]
    if not gpus:
        raise CLIError(f"--gpu {spec!r} names no CUDA device")
    return gpus


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    """Whether something is already listening there.

    Args:
        port: TCP port.
        host: Interface to probe.

    Returns:
        True when a connection succeeds.
    """
    with socket.socket() as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


class CachePolicyCommand(SuiteCommand):
    """Start a policy server if none is listening, cache the corpus against it, then stop it."""

    stage = "cache_policy"
    label = "cache-policy"
    done_status = "cached"
    update_latest = False               # the cache extends a graph; it does not make one
    table_headers = ("suite", "status", "rows", "exit")

    def resolve_config(self, suite: str) -> dict:
        graph_dir = self.opt("graph_dir", None)
        graph_dir = Path(graph_dir) if graph_dir else Path(self.paths.graph(suite))
        return {
            "command": "cache-policy", "suite": suite, "graph_dir": str(graph_dir),
            "port": self.opt("port", 14190),
            "gpu": str(self.opt("gpu", None) or build_param(
                "cache_policy", "gpus", "0", suite=suite)),
            "policy": self.opt("policy", None) or "",
            "limit_tasks": self.opt("limit_tasks", None),
            "limit_demos": self.opt("limit_demos", None), "name": self.args.name,
            "clients": int(self.opt("clients", None) or build_param(
                "cache_policy", "clients", 0, suite=suite)),
            "servers": int(self.opt("servers", None) or build_param(
                "cache_policy", "servers", 0, suite=suite)),
        }

    def _placed(self, config: dict, policy) -> tuple[int, int]:
        """(servers, clients): the config's counts, zeros filled from the policy's placement."""
        n_gpus = len(_gpu_list(config["gpu"]))
        servers = int(config["servers"]) or policy.placement.servers(n_gpus)
        clients = int(config["clients"]) or servers * policy.placement.clients_per_server
        return servers, clients

    def _policy(self, config: dict):
        """The policy descriptor this cache runs against: --policy, else the tree's policy."""
        pid = config.get("policy") or suite_config(config["suite"]).get("policy")
        return load_policy(pid or None)

    def n_stages(self, config: dict) -> int:
        return self._placed(config, self._policy(config))[0] + 2

    def plan_line(self, suite: str, config: dict) -> str:
        return f"planned: {A_PI_RAW_NPZ} in {config['graph_dir']} (port {config['port']})"

    def stage_inputs(self, suite: str, config: dict) -> list[str]:
        return [str(Path(config["graph_dir"]) / schema.NODES_NPZ)]

    def validate(self, suite: str, config: dict) -> None:
        self.require_graph(suite, config)
        try:
            policy = self._policy(config)
        except FileNotFoundError as exc:
            raise CLIError(str(exc)) from exc
        if self._python(policy) is None:
            raise CLIError(f"cannot locate the {policy.conda_env!r} conda env's python -- "
                           "set ROBOT_OOD_PY")

    def artifacts_present(self, suite: str, config: dict) -> bool:
        return (Path(config["graph_dir"]) / A_PI_RAW_NPZ).exists()

    def _python(self, policy) -> Path | None:
        """The policy env's interpreter: $ROBOT_OOD_PY, the descriptor's python, else its conda env."""
        from robot_ood.eval.policy import resolve_policy_python
        return resolve_policy_python(policy)

    def _env(self, config: dict, python: Path, policy, gpu: str = "") -> dict[str, str]:
        """The environment a child needs before its dynamic linker runs.

        Args:
            config: The resolved config.
            python: The child's interpreter: the policy env's for a server, this process's for
                a client.
            policy: The policy descriptor, for its PYTHONPATH entries.
            gpu: CUDA_VISIBLE_DEVICES for this child; "" takes the whole configured list.

        Returns:
            A full environment for the child.
        """
        root = repo_root()
        prefix = python.parent.parent
        env = {**os.environ}
        # Leaving OMP_NUM_THREADS unset once cost a 30s/it regression through OpenMP
        # oversubscription; it is not optional here.
        env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                   CUDA_VISIBLE_DEVICES=gpu or config["gpu"], PYTHONNOUSERSITE="1",
                   MAGICK_HOME=str(prefix))
        env["PYTHONPATH"] = os.pathsep.join(
            [*(str(root / p) for p in policy.pythonpath), str(root / "src"),
             *([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])])
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(prefix / "lib"), *([os.environ["LD_LIBRARY_PATH"]]
                                    if os.environ.get("LD_LIBRARY_PATH") else [])])
        # torch's caching allocator calls nvmlInit_v2 whenever it grows its pool and hard-asserts on
        # failure; this box's userspace NVML does not match its kernel module.
        preload = nvml_preload_path()
        if preload:
            env["LD_PRELOAD"] = os.pathsep.join(
                [preload, *([os.environ["LD_PRELOAD"]] if os.environ.get("LD_PRELOAD") else [])])
        return env

    def execute(self, suite: str, config: dict, log: StageLogger) -> dict:
        root = repo_root()
        policy = self._policy(config)
        python = self._python(policy)
        assert python is not None                   # validate() refused before we got here
        env = self._env(config, python, policy)
        log_dir = Path(log.run_dir) if getattr(log, "run_dir", None) else root / "logs" / "_drivers"

        gpus = _gpu_list(config["gpu"])
        servers, clients = self._placed(config, policy)
        ports = [int(config["port"]) + i for i in range(servers)]
        client_python = Path(sys.executable)
        client_env = self._env(config, client_python, policy)
        log.log(f"{servers} server(s) over GPU(s) {gpus}, {clients} client(s)")
        started: list = []
        try:
            for i, port in enumerate(ports):
                if _port_open(port):
                    log.log(f"reusing the policy server already listening on 127.0.0.1:{port}")
                    continue
                gpu = gpus[i % len(gpus)]
                log.step("policy server", f"port {port} on GPU {gpu}")
                started.append((port, self._spawn_server(
                    config, python, self._env(config, python, policy, gpu), log_dir, log, policy, port)))
            for port, job in started:
                self._wait_ready(job, port)
            started = [job for _, job in started]

            graph_dir = Path(config["graph_dir"])
            argv = [str(client_python), "-m", "robot_ood.build.policy_cache", suite,
                    "--nodes-npz", str(graph_dir / schema.NODES_NPZ),
                    "--out-dir", str(graph_dir), "--protocol", policy.protocol]
            arm = str(suite_config(suite).get("arm") or "")
            if arm:
                argv += ["--arm", arm]
            for key in ("limit_tasks", "limit_demos"):
                if config[key] is not None:
                    argv += [f"--{key.replace('_', '-')}", str(config[key])]
            if clients > 1:
                log.step("shards", f"{suite}: {clients} clients over {len(ports)} server(s)")
                self._replay_shards(argv, clients, ports, log_dir, client_env, root, log)
            log.step("cache", f"{suite} on port {ports[0]}")
            result = run_job("policy_cache", [*argv, "--port", str(ports[0])],
                             log_dir, env=client_env, cwd=root)
        finally:
            # A killed run that leaves a server behind holds ~7.5 GiB of GPU memory until someone
            # notices, so this tears down exactly the servers this command started.
            for job in started:
                log.log(f"stopping policy server pid {job.pid}")
                job.stop()
        if not result.ok:
            raise CLIError(f"{suite}: the policy cache exited {result.exit_code} -- see "
                           f"{result.log_path}")
        return {"rows": "written", "exit": result.exit_code}

    def _replay_shards(self, argv: list[str], n: int, ports: list[int], log_dir: Path,
                       env: dict[str, str], root: Path, log: StageLogger) -> None:
        """Fill the shard store from n concurrent clients, then let the caller assemble.

        Args:
            argv: The client command line without its --port, extended per shard.
            n: How many clients to run.
            ports: The servers to spread them over, round-robin.
            log_dir: Where each client's log goes.
            env: The child environment.
            root: Working directory for the children.
            log: Where progress is reported.

        Raises:
            CLIError: A client exited non-zero; its shards would be missing from the assembly.
        """
        jobs = [spawn_job(f"policy_cache_shard{i}",
                          [*argv, "--port", str(ports[i % len(ports)]),
                           "--worker-id", str(i), "--shard", f"{i}/{n}"],
                          log_dir, env=env, cwd=root)
                for i in range(n)]
        results = [job.wait() for job in jobs]
        log.log(f"[cache-policy] {n} shard clients exited "
                f"{[r.exit_code for r in results]}")
        failed = [r for r in results if not r.ok]
        if failed:
            raise CLIError(
                f"policy cache shard client(s) {[r.name for r in failed]} failed -- see "
                f"{', '.join(str(r.log_path) for r in failed)}"
            )

    def _spawn_server(self, config: dict, python: Path, env: dict[str, str], log_dir: Path,
                      log: StageLogger, policy, port: int):
        """Start a policy server without waiting, so several load their model at once.

        Args:
            config: The resolved config.
            python: The policy env's interpreter.
            env: The child environment.
            log_dir: Where the server log goes.
            log: This run's logger.
            policy: The policy descriptor naming the server and its argv.
            port: The port this server listens on.

        Returns:
            The running Job.

        Raises:
            CLIError: The checkpoint cannot be resolved.
        """
        import subprocess

        root = repo_root()
        ckpt = subprocess.run(
            [str(python), "-c", "from robot_ood.build.policy_cache import suite_checkpoint; "
             f"print(suite_checkpoint({config['suite']!r}, key={policy.ckpt_key!r}))"],
            capture_output=True, text=True, cwd=root, env=env)
        if ckpt.returncode != 0:
            raise CLIError(f"cannot resolve the checkpoint for {config['suite']}: {ckpt.stderr.strip()}")
        job = spawn_job(f"policy_server_{port}", [
            str(python), *policy.server_argv(checkpoint=ckpt.stdout.strip(), port=port),
        ], log_dir, env=env, cwd=root)
        log.log(f"policy server pid {job.pid}, log {job.log_path}")
        return job

    @staticmethod
    def _wait_ready(job, port: int) -> None:
        """Block until the server answers on its port, watching the process while waiting.

        Raises:
            CLIError: The server died, or never opened its port.
        """
        import time

        deadline = time.time() + SERVER_READY_TIMEOUT
        while time.time() < deadline:
            if _port_open(port):
                return
            if not job.alive():
                raise CLIError(f"the policy server died before opening port {port} -- see "
                               f"{job.log_path}")
            time.sleep(2.0)
        job.stop()
        raise CLIError(f"the policy server never opened port {port} in "
                       f"{SERVER_READY_TIMEOUT:.0f}s -- see {job.log_path}")

    def row(self, result: SuiteResult) -> list[str]:
        m = result.metrics
        return [result.suite, result.status, str(m.get("rows", "-")), str(m.get("exit", "-"))]

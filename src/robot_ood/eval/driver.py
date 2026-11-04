"""The success-rate eval driver behind scripts/run_sr.py and the eval CLI verb."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Sequence


from robot_ood.eval.ladder import Cell, LadderConfig, LADDER_YAML
from robot_ood.eval.launcher import EnvBuilder
from robot_ood.eval.preflight import Preflight
from robot_ood.eval.provenance import Ledger
from robot_ood.eval.runner import CellAction, CellPlan, CellRunner, CellScorer, ExitCode, ResumePlanner
from robot_ood.eval.scheduler import PortAllocator, ProcessSupervisor, Schedule
from robot_ood.eval.bench import DEFAULT_BENCH


class SRDriver:
    """Dispatches one CLI invocation. Owns the lazily-built LadderConfig and the ledger."""

    def __init__(self, args: argparse.Namespace, ledger: Ledger | None = None) -> None:
        self._args = args
        self._config: LadderConfig | None = None
        self._ledger = ledger or Ledger()

    @property
    def config(self) -> LadderConfig:
        """Built on first use."""
        if self._config is None:
            self._config = LadderConfig.load(
                Path(self._args.ladder),
                Path(self._args.results_root) if self._args.results_root else None,
                bench_id=getattr(self._args, "bench", None),
            )
        return self._config

    def run(self) -> ExitCode:
        args = self._args
        if args.check_env:
            return self._check_env()

        cells = self._resolve_cells()
        if not cells:
            print("[run_sr] nothing to do -- pass --rung or --cells", file=sys.stderr)
            return ExitCode.ERROR
        if args.score:
            return self._score(cells)

        gpus = self._parse_gpus(args.gpus)
        if args.per_gpu < 1:
            print(f"[run_sr] --per-gpu must be at least 1, got {args.per_gpu}", file=sys.stderr)
            return ExitCode.ERROR
        if not gpus:
            print("[run_sr] --gpus is required to launch or dry-run a schedule", file=sys.stderr)
            return ExitCode.ERROR
        return self._launch(cells, gpus)

    # ---- cell resolution -------------------------------------------------------------------------
    def _resolve_cells(self) -> list[Cell]:
        """The cells this invocation names, with the per-run overrides applied."""
        cells = self._select_cells()
        n = self._args.num_clients
        if n is not None:
            if n < 1:
                raise SystemExit(f"--num-clients must be at least 1, got {n}")
            cells = [replace(c, num_clients=n) for c in cells]
        servers = getattr(self._args, "servers_per_cell", None)
        if servers is not None:
            if servers < 1:
                raise SystemExit(f"--servers-per-cell must be at least 1, got {servers}")
            cells = [replace(c, servers=servers) for c in cells]
        trials = getattr(self._args, "trials", None)
        if trials:
            cells = [replace(c, trials=trials, expected_n=self.config.recount(c, trials))
                     for c in cells]
        variant = getattr(self._args, "variant", None)
        if variant:
            cells = [replace(c, variant=variant) for c in cells]
        seed = getattr(self._args, "seed", None)
        if seed is not None:
            cells = [replace(c, seed=seed) for c in cells]
        return cells

    def _select_cells(self) -> list[Cell]:
        args = self._args
        if args.rung:
            if args.rung not in self.config.doc["rungs"]:
                raise SystemExit(
                    f"unknown rung {args.rung!r}; known: {sorted(self.config.doc['rungs'])}"
                )
            cells = self.config.cells_for_rung(args.rung)
            if args.cells:
                cells = self._subset(cells, args.cells, args.rung)
            if args.mode:
                cells = [c.with_mode(args.mode) for c in cells]
        elif args.cells:
            cells = self.config.cells_from_tokens(args.cells, args.mode)
        else:
            return []
        if args.policy:
            cells = [replace(c, policy=args.policy) for c in cells]
        return cells

    @staticmethod
    def _subset(cells: Sequence[Cell], spec: str, rung: str) -> list[Cell]:
        """--rung R --cells suite:axis,..."""
        wanted = {t.strip() for t in spec.split(",") if t.strip()}
        unknown = wanted - {c.key for c in cells}
        if unknown:
            raise SystemExit(
                f"--cells: {sorted(unknown)} not in rung {rung} (has: {sorted(c.key for c in cells)})"
            )
        return [c for c in cells if c.key in wanted]

    @staticmethod
    def _parse_gpus(spec: str | None) -> list[int]:
        """"0,1" -> [0, 1], preserving order and dropping repeats."""
        if not spec:
            return []
        seen: dict[int, None] = {}
        for token in spec.split(","):
            if token.strip():
                seen[int(token)] = None
        return list(seen)

    # ---- subcommands -----------------------------------------------------------------------------
    def _check_env(self) -> ExitCode:
        preflight = Preflight(self.config)
        ok = True

        print("=== GPU ===")
        try:
            statuses = preflight.gpus()
            if not statuses:
                print("  no CUDA devices visible")
                ok = False
            for status in statuses:
                label = "OK" if status.ok else "BUSY"
                print(f"  gpu{status.index}: {status.free_gib:6.1f} / {status.total_gib:6.1f} GiB free"
                      f"  [{label}]")
                ok = ok and status.ok
        except Exception as exc:   # noqa: BLE001 -- a broken torch must not crash the report
            print(f"  GPU check FAILED: {exc}")
            ok = False

        print("=== load ===")
        load = preflight.load()
        print(f"  load1={load.load1:.2f}  ncpu={load.ncpu}  threshold={load.threshold:.1f}  "
              f"[{'OK' if load.ok else 'BUSY'}]")
        ok = ok and load.ok

        print(f"=== {self.config.bench.id} ===")
        benchmark = preflight.benchmark()
        print(f"  {benchmark.message}")
        ok = ok and benchmark.ok

        print(f"\n[check-env] overall: {'OK' if ok else 'NOT READY'}")
        return ExitCode.OK if ok else ExitCode.ERROR

    def _score(self, cells: Sequence[Cell]) -> ExitCode:
        scorer = CellScorer(self.config)
        any_scored = False
        any_failed = False
        for cell in cells:
            report = scorer.report(cell, self._args.allow_partial)
            if report.score.actual_n == 0:
                print(f"RESULT {cell.tag}: NO DATA (expected {cell.expected_n} episodes at "
                      f"{cell.result_dir(self.config.results_root)})")
                continue
            any_scored = True
            any_failed = any_failed or not report.verdict.passed
            print(report.line)
        if not any_scored:
            return ExitCode.ERROR
        return ExitCode.GATE_FAILED if any_failed else ExitCode.OK

    # ---- launch ----------------------------------------------------------------------------------
    def _launch(self, cells: Sequence[Cell], gpus: Sequence[int]) -> ExitCode:
        usable_gpus = self._validate_system(cells, gpus)
        if not usable_gpus:
            return ExitCode.ERROR

        plans = self._build_execution_plan(cells)
        if plans is None:
            return ExitCode.ERROR
        if not plans:
            print("[run_sr] nothing left to run (everything already complete)")
            return ExitCode.OK

        schedule = Schedule.build([p.cell for p in plans], usable_gpus, self._args.per_gpu)
        self._print_schedule(schedule)

        if self._args.dry_run:
            print("\n[dry-run] no subprocess launched, no filesystem changes made.")
            return ExitCode.OK
        return self._execute_schedule(plans, schedule)

    def _validate_system(self, cells: Sequence[Cell], gpus: Sequence[int]) -> list[int]:
        """GPU + loadavg preflight. Returns the usable GPUs, or [] to refuse the launch."""
        preflight = Preflight(self.config)
        statuses = preflight.gpus(gpus)
        for status in statuses:
            if not status.ok:
                print(f"[preflight] SKIP gpu{status.index}: {status.reason}")
        sys.stdout.flush()   # keep stdout ahead of the stderr refusal below when interleaved (2>&1)

        usable = [s.index for s in statuses if s.ok]
        if not usable:
            print("[preflight] no usable GPUs -- refusing to start anything", file=sys.stderr)
            return []

        total_expected = sum(c.expected_n for c in cells)
        is_large = total_expected > self.config.limits.large_rung_episodes
        load = preflight.load()
        print(f"[preflight] load1={load.load1:.2f} ncpu={load.ncpu} threshold={load.threshold:.1f} "
              f"large_launch={is_large} (total_expected={total_expected})")
        sys.stdout.flush()
        if is_large and not load.ok:
            print(f"[preflight] REFUSING: a large launch ({total_expected} episodes) and loadavg(1m)="
                  f"{load.load1:.2f} exceeds {load.threshold:.1f} -- box looks busy with a "
                  f"neighbouring job", file=sys.stderr)
            return []

        for cell in cells:
            if cell.filter_file is not None and not cell.filter_file.exists():
                print(f"[preflight] WARNING: {cell.tag}: filter file missing at {cell.filter_file}")
        return usable

    def _build_execution_plan(self, cells: Sequence[Cell]) -> list[CellPlan] | None:
        """The per-cell resume decisions. None means refuse the whole launch."""
        planner = ResumePlanner(self.config, self._ledger)
        records = self._ledger.read()
        runnable: list[CellPlan] = []
        for cell in cells:
            plan = planner.plan(cell, self._args.resume, records, self._args.rerun_incomplete)
            print(f"[plan] {cell.tag}: {plan.action.value} ({plan.reason})")
            if plan.action is CellAction.ERROR:
                print(f"[run_sr] {cell.tag}: {plan.reason}", file=sys.stderr)
                return None
            if plan.action in (CellAction.RUN, CellAction.PURGE):
                runnable.append(plan)
        return runnable

    def _execute_schedule(self, plans: Sequence[CellPlan], schedule: Schedule) -> ExitCode:
        # Apply any purges now -- never before this point, so --dry-run (even with --resume) is a
        # pure read-only preview and can never delete a partial result dir.
        planner = ResumePlanner(self.config, self._ledger)
        for plan in plans:
            planner.apply(plan)

        supervisor = ProcessSupervisor()
        supervisor.install_signal_handlers()
        runner = CellRunner(
            config=self.config,
            env_builder=EnvBuilder(self.config),
            scorer=CellScorer(self.config),
            ledger=self._ledger,
            ports=PortAllocator(self.config.defaults.base_port,
                                self.config.root / "logs" / ".ports"),
            supervisor=supervisor,
            allow_partial=self._args.allow_partial,
        )

        def run_track(gpus: Sequence[int], track_cells: Sequence[Cell]) -> list[ExitCode]:
            return [runner.run(cell, gpus) for cell in track_cells]

        # One worker thread per track (a GPU slot), each running its cells sequentially. Threads
        # (not processes) are fine: each cell's work is dominated by subprocess I/O wait.
        codes: list[ExitCode] = []
        tracks = [t for t in schedule.tracks if t.cells]
        with ThreadPoolExecutor(max_workers=max(1, len(tracks))) as pool:
            futures = [pool.submit(run_track, t.gpus, t.cells) for t in tracks]
            for future in as_completed(futures):
                codes.extend(future.result())

        if ExitCode.GATE_FAILED in codes:
            return ExitCode.GATE_FAILED
        if any(code is ExitCode.ERROR for code in codes):
            return ExitCode.ERROR
        return ExitCode.OK

    @staticmethod
    def _print_schedule(schedule: Schedule) -> None:
        print("\n=== schedule ===")
        slotted = any(t.slot for t in schedule.tracks)
        for track in schedule.tracks:
            where = f"gpu{track.gpu}/{track.slot}" if slotted else f"gpu{track.gpu}"
            if len(track.gpus) > 1:
                where = "gpu" + "+".join(str(g) for g in track.gpus)
            for cell in track.cells:
                print(f"  {where}: {cell.tag:<45} n={cell.expected_n:<5} mode={cell.mode}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_sr.py", description="ONF LIBERO-Plus success-rate eval driver."
    )
    parser.add_argument("--rung", help="ladder rung name (R1, R4)")
    parser.add_argument("--cells", help="comma-separated suite:axis pairs, "
                                         "e.g. goal:Background_Textures,long:Camera_Viewpoints")
    parser.add_argument("--gpus", help="comma-separated physical GPU indices, e.g. 0,1")
    parser.add_argument("--servers-per-cell", type=int, default=None,
                        help="policy servers per cell, its clients sharded over them; a lone cell "
                             "spreads them over the run's idle GPUs (default: the ladder's, 1)")
    parser.add_argument("--per-gpu", type=int, default=1,
                        help="cells to run concurrently on each GPU, each with its own policy server "
                             "(default 1; a batch-1 server keeps an A100 about half busy)")
    parser.add_argument("--mode", help="mode override (base | fcsr)")
    parser.add_argument("--dry-run", action="store_true", help="print the schedule/projection; launch nothing")
    parser.add_argument("--resume", action="store_true", help="skip complete cells; run a partial one's missing episodes")
    parser.add_argument("--rerun-incomplete", action="store_true",
                        help="purge and re-run partial cells (implies --resume's skip of complete ones)")
    parser.add_argument("--score", action="store_true", help="score existing results only; launch nothing")
    parser.add_argument("--check-env", action="store_true", help="GPU/load/benchmark preflight report")
    parser.add_argument("--trials", type=int, default=None,
                        help="episodes per task, overriding the benchmark's default")
    parser.add_argument("--seed", type=int, default=None,
                        help="episode seed base handed to every client, overriding the protocol's")
    parser.add_argument("--variant", default=None,
                        help="tag suffix, so a run against another artifact tree gets its own result "
                             "dir instead of colliding with the default one (GR_GRAPH_DIR picks the tree)")
    parser.add_argument("--policy", default=None,
                        help="policy descriptor under evals/policies/ (default: each cell's own, "
                             "stablevla unless the ladder says otherwise)")
    parser.add_argument("--bench", default=None,
                        help=f"benchmark to run (default {DEFAULT_BENCH}); one of evals/*/bench.yaml")
    parser.add_argument("--allow-partial", action="store_true", help="include incomplete cells in totals/gates")
    parser.add_argument("--ladder", default=str(LADDER_YAML), help="path to the ladder yaml")
    parser.add_argument("--results-root", default=None, help="override the results/ root (else score.py's default)")
    parser.add_argument("--num-clients", type=int, default=None,
                        help="sim clients per cell, overriding the ladder's num_clients")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    return int(SRDriver(args).run())

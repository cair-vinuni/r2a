from __future__ import annotations

import ast
import contextlib
import io
import sys
import warnings
from pathlib import Path

import numpy as np

import paths

NODE = Path(__file__).with_name("serve.py")

if str(paths.SRC) not in sys.path:
    sys.path.insert(0, str(paths.SRC))

warnings.filterwarnings("ignore")

REQUIRED = ("g_nodes.npz", "g_edges.npz", "g_segments.npz", "action_scale.json")
TRAINED = ("g_head.npz", "g_fcsr.npz", "g_lang.npz", "g_lang_queries.npz")
HOMING_ARTIFACTS = ("g_score.npz",)
SUITES = {"left": "vrh31_bds_pick_left", "right": "vrh31_bds_pick_right"}


class Report:
    """One line per check, and whether any of them failed."""

    def __init__(self) -> None:
        self.failed = False

    def section(self, title: str) -> None:
        print(f"\n== {title} ==")

    def ok(self, msg: str) -> None:
        print(f"  \033[32mok\033[0m    {msg}")

    def warn(self, msg: str) -> None:
        print(f"  \033[33mwarn\033[0m  {msg}")

    def bad(self, msg: str) -> None:
        self.failed = True
        print(f"  \033[31mFAIL\033[0m  {msg}")

    def detail(self, msg: str) -> None:
        print(f"        {msg}")


def _oneline(err: BaseException, limit: int) -> str:
    return " ".join(str(err).split())[:limit]


def check_interpreter(report: Report) -> None:
    report.section("interpreter")
    report.ok(sys.executable)
    report.detail(f"environment root {sys.prefix}")
    try:
        import robot_ood  # noqa: F401
        from robot_ood.runtime.vrh31_server import Vrh31ServerLoop  # noqa: F401
    except Exception as err:
        report.bad(f"cannot import robot_ood.runtime -- {type(err).__name__}: {_oneline(err, 150)}")
        report.detail(f"looked for the package under {paths.SRC}")
        return
    report.ok("robot_ood.runtime.vrh31_server")
    try:
        from gr00t.policy.server_client import PolicyServer  # noqa: F401
    except Exception as err:
        report.bad(f"cannot import gr00t -- {type(err).__name__}: {_oneline(err, 150)}")
        report.detail("run this with the Isaac-GR00T .venv activated")
        return
    report.ok("gr00t.policy.server_client")


def check_artifacts(report: Report) -> None:
    for arm in paths.ARMS:
        root = paths.ARTIFACTS[arm]
        report.section(f"{arm} artifacts ({root})")
        for name in REQUIRED:
            if (root / name).is_file():
                report.ok(name)
            else:
                report.bad(f"missing {name}")
        for name in TRAINED:
            if (root / name).is_file():
                report.ok(name)
            else:
                report.bad(f"missing {name} -- correct() will raise, not pass through")
        for name in HOMING_ARTIFACTS:
            if (root / name).is_file():
                report.ok(name)
            elif paths.HOMING:
                report.bad(f"missing {name} -- VRH31_HOMING is on and homing cannot load without it")
            else:
                report.warn(f"no {name} -- homing unavailable (VRH31_HOMING is off)")


def check_fcsr_matches_graph(report: Report) -> None:
    report.section("the FCSR checkpoint belongs to this graph")
    for arm in paths.ARMS:
        root = paths.ARTIFACTS[arm]
        if not ((root / "g_fcsr.npz").is_file() and (root / "g_segments.npz").is_file()):
            report.warn(f"{arm}: skipped, g_fcsr.npz or g_segments.npz missing (reported above)")
            continue
        try:
            from robot_ood.core.config import GraphConfig
            from robot_ood.models.retrieval.retriever import SegmentRetriever

            SegmentRetriever.load(root, cfg=GraphConfig(str(root)), device="cpu")
        except ValueError as err:
            report.bad(f"{arm}: g_fcsr.npz was trained on a different graph; retrain it with")
            report.detail(f"python -m robot_ood.cli train fcsr --suite {SUITES[arm]}")
            report.detail(_oneline(err, 180))
            continue
        except Exception as err:
            report.bad(f"{arm}: could not verify g_fcsr.npz -- {type(err).__name__}: "
                       f"{_oneline(err, 150)}")
            continue
        report.ok(f"{arm}: graph_hash matches (graph, scorer, segment_len, seg_topk)")


def stored_queries(arm: str) -> dict[str, np.ndarray]:
    from robot_ood.models.language.encoder import TaskEmbeddings, normalise_instruction

    queries = TaskEmbeddings.load(paths.ARTIFACTS[arm]).queries
    strings = node_instructions()
    return {name: queries[normalise_instruction(text)] for name, text in strings.items()
            if normalise_instruction(text) in queries}


def check_language_model(report: Report) -> None:
    report.section("instruction encoder snapshot")
    have_all = True
    for arm in paths.ARMS:
        if not (paths.ARTIFACTS[arm] / "g_lang.npz").is_file():
            have_all = False
            continue
        try:
            stored = stored_queries(arm)
        except Exception as err:
            report.bad(f"{arm}: could not read g_lang.npz -- {type(err).__name__}: "
                       f"{_oneline(err, 150)}")
            return
        if not (stored and set(stored) == set(node_instructions())):
            have_all = False
    if have_all:
        report.ok("not needed: every instruction in serve.py is stored in g_lang_queries.npz")
        return
    try:
        from robot_ood.models.language.encoder import MODEL_ENV_VAR, InstructionEncoder

        encoder = InstructionEncoder()
        snapshot = encoder.snapshot
    except Exception as err:
        report.bad(_oneline(err, 300))
        return
    report.ok(str(snapshot))
    for arm in paths.ARMS:
        if not (paths.ARTIFACTS[arm] / "g_lang.npz").is_file():
            report.warn(f"{arm}: cannot compare, no g_lang.npz")
            continue
        try:
            from robot_ood.models.language.encoder import TaskEmbeddings

            built = TaskEmbeddings.load(paths.ARTIFACTS[arm]).model_id
        except Exception as err:
            report.bad(f"{arm}: could not read g_lang.npz -- {type(err).__name__}: "
                       f"{_oneline(err, 150)}")
            continue
        if built == encoder.model_id:
            report.ok(f"{arm}: matches the snapshot g_lang.npz was built with ({built})")
        else:
            report.bad(f"{arm}: g_lang.npz was built with {built}, this machine resolves "
                       f"{encoder.model_id}")
            report.detail(f"point {MODEL_ENV_VAR} at the snapshot the artifacts were built with")


def node_instructions() -> dict[str, str]:
    tree = ast.parse(NODE.read_text())
    values: dict[str, str] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            name = getattr(stmt.targets[0], "id", "")
            if name.endswith("_INSTRUCTION"):
                value = stmt.value
                if isinstance(value, ast.Name):
                    value = ast.Constant(values[value.id])
                values[name] = ast.literal_eval(value)
    return values


def check_instructions(report: Report) -> None:
    report.section(f"instructions in {NODE.name} resolve to their task")
    strings = node_instructions()
    if "LANGUAGE_INSTRUCTION" not in strings:
        report.bad("no LANGUAGE_INSTRUCTION in serve.py")
        return
    for arm in paths.ARMS:
        root = paths.ARTIFACTS[arm]
        if not (root / "g_lang.npz").is_file():
            report.warn(f"{arm}: skipped, no g_lang.npz")
            continue
        try:
            from robot_ood.models.language.encoder import InstructionEncoder, TaskEmbeddings

            stored = stored_queries(arm)
            picks = {}
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                tasks = TaskEmbeddings.load(root)
                encoder = InstructionEncoder()
                for name, text in strings.items():
                    posterior = tasks.task_posterior(text, encoder)
                    picks[name] = (int(posterior.argmax()), float(posterior.max()))
        except Exception as err:
            report.bad(f"{arm}: could not embed the instructions -- {type(err).__name__}: "
                       f"{_oneline(err, 150)}")
            report.detail("python embed_instructions.py on a machine with the Qwen snapshot stores them")
            continue
        for name, (task_id, confidence) in picks.items():
            source = "stored" if name in stored else "embedded now"
            report.ok(f"{arm}: {name} -> task {task_id} (confidence {confidence:.4f}, {source})")
        presets = {k: v for k, v in picks.items() if k != "LANGUAGE_INSTRUCTION"}
        if len({task_id for task_id, _ in presets.values()}) != len(presets):
            report.bad(f"{arm}: two presets resolve to the SAME task -- the strings do not match "
                       f"the graph")
    report.detail(f"LANGUAGE_INSTRUCTION = {strings['LANGUAGE_INSTRUCTION']!r}")


def main() -> int:
    report = Report()
    check_interpreter(report)
    check_artifacts(report)
    check_fcsr_matches_graph(report)
    check_language_model(report)
    check_instructions(report)
    print()
    if report.failed:
        print("Some checks FAILED -- fix those before starting the robot.")
        return 1
    print("All checks passed. Next: python serve.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

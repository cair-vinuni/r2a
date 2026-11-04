"""onf's own import graph, asserted rather than hoped for.

A module's layer IS the directory it lives in -- src/robot_ood/<layer>/... -- so nothing can sit in the
wrong bucket and still pass. Properties, all by AST-parsing every module under src/robot_ood:
no import cycles, no package __init__ re-exporting names, no module importing a layer above its
own, no library module importing an entry point, and no module reaching for another's private
name. Each has an explicit list of what still violates it; a commit removes entries, never adds.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PKG = SRC / "robot_ood"
ENTRY_DIRS = ("scripts", "evals")

# Cycles that exist today, as sorted tuples of module names. EMPTY IS THE GOAL.
# A commit that removes a cycle deletes its entry here; a commit that adds one has to justify
# adding an entry, in review, in the same diff that creates it.
KNOWN_CYCLES: set[tuple[str, ...]] = set()

# __init__.py files allowed to re-export names, with the reason. Prefer emptying this.
INIT_MAY_REEXPORT: dict[str, str] = {
    "robot_ood": "Paths/default_paths -- the package's advertised entry point, and robot_ood.core.config is torch-free",
}


# Top-level packages that are a layer under another name. Everything else IS its directory:
# src/robot_ood/<layer>/... belongs to <layer>, so a file cannot be in the wrong bucket and pass.
LAYER_ALIASES: dict[str, str] = {"cli": "eval", "pipeline": "eval"}

# Layer -> the layers it may import from. runtime is what executes inside a rollout, so it must not
# reach into train or build; core reaches for nothing. Widening a row here is a design decision.
LAYER_MAY_IMPORT: dict[str, set[str]] = {
    "core":    set(),
    "data":    {"core"},
    "models":  {"core", "data"},
    "build":   {"core", "data", "models"},
    "train":   {"core", "data", "models"},
    "runtime": {"core", "data", "models"},
    "eval":    {"core", "data", "models", "build", "train", "runtime"},
}

# Edges violating LAYER_MAY_IMPORT today, as (importer, imported). EMPTY IS THE GOAL, and the test
# below fails on an entry that is no longer violated, so a fix cannot leave a stale line here.
KNOWN_LAYER_VIOLATIONS: set[tuple[str, str]] = {
    ("robot_ood.models.fusion.target", "robot_ood.runtime.ee_track"),

    # config reads the graph contract; §4 moves that shared shape down into core
    ("robot_ood.core.config", "robot_ood.data.schema"),

    # build and train reach sideways into each other and into runtime -- Lane F and Lane H
    ("robot_ood.build.measure", "robot_ood.train.align"),
    ("robot_ood.build.graph", "robot_ood.train.data"),
    ("robot_ood.train.tasks.fcsr", "robot_ood.build.measure"),
    ("robot_ood.build.measure", "robot_ood.runtime.retrieve"),
}


# (importer, the private name it imports). Each entry is a missing public API: the callee's module
# decides what it exports, and a leading underscore is that decision. THIS LIST MAY ONLY SHRINK.
KNOWN_PRIVATE_IMPORTS: set[tuple[str, str]] = {
    ("src/robot_ood/train/tensors.py", "robot_ood.train.data._node_feature_matrix"),
    ("src/robot_ood/train/metrics.py", "robot_ood.train.data._node_feature_matrix"),
}


def _module_name(path: Path) -> str:
    parts = path.relative_to(SRC).with_suffix("").parts
    return ".".join(parts[:-1]) if parts[-1] == "__init__" else ".".join(parts)


def _is_module(dotted: str) -> bool:
    base = SRC / Path(*dotted.split("."))
    return base.with_suffix(".py").exists() or (base / "__init__.py").exists()


def import_graph() -> dict[str, set[str]]:
    """{module: {onf modules it imports at any level}}, by AST, resolving from pkg import mod."""
    graph: dict[str, set[str]] = {}
    for path in sorted(PKG.rglob("*.py")):
        me = _module_name(path)
        targets: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == PKG.name:
                for alias in node.names:
                    # from robot_ood.data import schema is an edge to robot_ood.data.schema, not to robot_ood.data
                    submodule = f"{node.module}.{alias.name}"
                    targets.add(submodule if _is_module(submodule) else node.module)
            elif isinstance(node, ast.Import):
                targets |= {a.name for a in node.names if a.name.split(".")[0] == PKG.name}
        graph[me] = {t for t in targets if t != me and _is_module(t)}
    return graph


def find_cycles(graph: dict[str, set[str]]) -> set[tuple[str, ...]]:
    """Every elementary cycle, each normalised to a sorted tuple so it has one canonical spelling."""
    cycles: set[tuple[str, ...]] = set()
    path: list[str] = []
    on_path: set[str] = set()

    def walk(node: str) -> None:
        if node in on_path:
            cycles.add(tuple(sorted(path[path.index(node):])))
            return
        if node not in graph:
            return
        path.append(node)
        on_path.add(node)
        for nxt in sorted(graph[node]):
            walk(nxt)
        path.pop()
        on_path.discard(node)

    for start in sorted(graph):
        walk(start)
    return cycles


def test_import_cycles_are_only_the_known_ones():
    found = find_cycles(import_graph())
    new = found - KNOWN_CYCLES
    fixed = KNOWN_CYCLES - found
    assert not new, (
        f"new import cycle(s): {sorted(new)}. A cycle forces a function-local import somewhere to "
        "break the tie; put the shared thing in a module both can import instead."
    )
    assert not fixed, (
        f"cycle(s) {sorted(fixed)} are gone -- delete them from KNOWN_CYCLES so this test keeps "
        "asserting the improvement."
    )


def test_package_inits_do_not_re_export():
    offenders = {}
    for path in sorted(PKG.rglob("__init__.py")):
        me = _module_name(path)
        if me in INIT_MAY_REEXPORT:
            continue
        tree = ast.parse(path.read_text())
        names = [
            n for n in tree.body
            if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
        ]
        if names:
            offenders[me] = sorted({type(n).__name__ for n in names})
    assert not offenders, (
        f"these package __init__.py files have statements beyond their docstring: {offenders}. "
        "A re-export here runs the re-exported module for EVERY importer of any sibling -- which is "
        "how `import robot_ood.core.config` used to cost a torch import. Import the submodule directly, or add "
        "the package to INIT_MAY_REEXPORT with a reason."
    )


def _layer(module: str) -> str | None:
    """The module's layer: the directory it lives in, or None for onf itself and its packages."""
    parts = module.split(".")
    if len(parts) < 2:
        return None
    top = parts[1]
    top = LAYER_ALIASES.get(top, top)
    return top if top in LAYER_MAY_IMPORT else None


def _is_package(module: str) -> bool:
    return (SRC / Path(*module.split("."))).is_dir()


def test_every_module_has_a_layer():
    """A module outside every layer directory is exempt from the layer test without saying so."""
    missing = sorted(m for m in import_graph() if _layer(m) is None and not _is_package(m))
    assert not missing, (
        f"modules in no layer directory: {missing}. Every module belongs under one of "
        f"{sorted(LAYER_MAY_IMPORT)} -- move it there, or alias its package in LAYER_ALIASES."
    )


def test_layers_only_import_downwards():
    graph = import_graph()
    found = set()
    for module, targets in graph.items():
        src = _layer(module)
        if src is None:
            continue
        for target in targets:
            dst = _layer(target)
            if dst is not None and dst != src and dst not in LAYER_MAY_IMPORT[src]:
                found.add((module, target))
    new = found - KNOWN_LAYER_VIOLATIONS
    fixed = KNOWN_LAYER_VIOLATIONS - found
    assert not new, (
        "new layer violation(s):\n  "
        + "\n  ".join(f"{m} -> {t}  ({_layer(m)} may not import {_layer(t)})" for m, t in sorted(new))
        + "\n\nIf the importer genuinely needs the thing, the shared piece belongs in a lower layer."
    )
    assert not fixed, (
        f"{sorted(fixed)} no longer violate the table -- delete them from KNOWN_LAYER_VIOLATIONS so "
        "this test keeps asserting the improvement."
    )


def _entry_files() -> list[Path]:
    """Every .py under scripts/ and evals/ -- the entry points, which nothing in src/ may import."""
    return sorted(p for d in ENTRY_DIRS for p in (ROOT / d).rglob("*.py") if "__pycache__" not in p.parts)


def test_src_does_not_import_entry_points():
    stems = {p.stem for p in _entry_files()} - {"__init__"}
    bad = []
    for path in sorted(PKG.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                heads = [node.module.split(".")[0]] if node.module else []
            elif isinstance(node, ast.Import):
                heads = [a.name.split(".")[0] for a in node.names]
            else:
                continue
            for head in heads:
                if head in ENTRY_DIRS or head in stems:
                    bad.append(f"{path.relative_to(ROOT)}:{node.lineno} imports {head}")
    assert not bad, (
        "library module(s) importing an entry point:\n  " + "\n  ".join(bad) + "\n\n"
        "scripts/ and evals/ sit on top of onf and are not importable as a package. What both sides "
        "need belongs in robot_ood."
    )


def test_private_names_stay_inside_their_module():
    found = set()
    for path in sorted(PKG.rglob("*.py")) + _entry_files():
        rel = str(path.relative_to(ROOT))
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == PKG.name:
                found |= {(rel, f"{node.module}.{a.name}") for a in node.names if a.name.startswith("_")}
    new = found - KNOWN_PRIVATE_IMPORTS
    fixed = KNOWN_PRIVATE_IMPORTS - found
    assert not new, (
        "module(s) importing another module's private name:\n  "
        + "\n  ".join(f"{a} -> {b}" for a, b in sorted(new))
        + "\n\nPromote the name, or move the caller into the module that owns it."
    )
    assert not fixed, (
        f"{sorted(fixed)} no longer import a private name -- delete them from KNOWN_PRIVATE_IMPORTS."
    )


def test_call_sites_pass_arguments_the_callee_accepts():
    """Every keyword argument at a call to a refactored constructor/loader must be a real parameter.

    The unit suite does not build a locator from real artifacts, so a stale keyword at one of these
    call sites survives a fully green run and only surfaces in a simulator rollout. This is the
    static half of that check.
    """
    import inspect
    import sys

    sys.path.insert(0, str(SRC))
    import robot_ood.runtime.locator as locator_mod

    # (receiver, attribute) -> callable, or (None, name) for a plain call. The receiver matters:
    # a bare load also matches np.load / torch.load / EdgeSet.load.
    watched = {
        (None, "GraphLocator"): locator_mod.GraphLocator.__init__,
        ("GraphLocator", "load"): locator_mod.GraphLocator.load,
    }
    accepted = {k: set(inspect.signature(fn).parameters) for k, fn in watched.items()}

    def _key(func):
        if isinstance(func, ast.Attribute):
            recv = func.value
            name = recv.id if isinstance(recv, ast.Name) else getattr(recv, "attr", None)
            return (name, func.attr)
        return (None, getattr(func, "id", None))

    bad = []
    for directory in ("src", "tests", *ENTRY_DIRS):
        for path in sorted((ROOT / directory).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.Call):
                    continue
                key = _key(node.func)
                if key not in accepted:
                    continue
                for kw in node.keywords:
                    if kw.arg and kw.arg not in accepted[key]:
                        rel = path.relative_to(ROOT)
                        bad.append(f"{rel}:{node.lineno} {key[1]}(..., {kw.arg}=...) is not a parameter")
    assert not bad, "call sites passing arguments the callee does not accept:\n  " + "\n  ".join(bad)


def test_only_the_loader_opens_demonstration_files():
    """h5py belongs to robot_ood.data.libero alone, so demo order is one decision, not five.

    Node ids, owner ids and every artifact indexed by them are positional. A second module that
    walks handle["data"] itself can disagree about demo order and produce artifacts that look fine
    and cannot be used together.
    """
    offenders = {}
    for path in sorted(SRC.rglob("*.py")) + [p for d in ENTRY_DIRS for p in (ROOT / d).rglob("*.py")]:
        if path.name == "libero.py" or "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            imports_h5py = (
                (isinstance(node, ast.Import) and any(a.name == "h5py" for a in node.names))
                or (isinstance(node, ast.ImportFrom) and node.module == "h5py")
            )
            if imports_h5py:
                offenders.setdefault(str(path.relative_to(ROOT)), []).append(node.lineno)
    assert not offenders, (
        f"these files import h5py directly: {offenders}. Read demos through robot_ood.data.libero so "
        "every caller sees the same files in the same order."
    )

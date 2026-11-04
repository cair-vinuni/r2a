"""Category filtering over an unmodified LIBERO-plus checkout.

sylvestf/LIBERO-plus registers suites holding every perturbed task; the perturbation category
lives beside them in benchmark/task_classification.json. This module builds a suite and filters
its task list from outside the checkout -- nothing in external/ is patched.
"""

from __future__ import annotations

import json
import os


def _classification() -> dict:
    path = os.environ.get("LIBERO_PLUS_TASK_CLASSIFICATION")
    if not path:
        from libero.libero import benchmark as _b
        path = os.path.join(os.path.dirname(os.path.abspath(_b.__file__)),
                            "task_classification.json")
    with open(path) as fh:
        return json.load(fh)


def _id_filter() -> set[int] | None:
    """Allowed instance ids from $LIBERO_INSTANCE_ID_FILTER (one id per line), or None."""
    path = os.environ.get("LIBERO_INSTANCE_ID_FILTER")
    if not path:
        return None
    with open(path) as fh:
        return {int(tok) for tok in fh.read().split() if tok.strip()}


def get_ids_by_category(category: str) -> dict[str, tuple[list[int], list[str]]]:
    """{suite: (ids, names)} for one perturbation axis; the --check-env episode counter."""
    out: dict[str, tuple[list[int], list[str]]] = {}
    for suite, entries in _classification().items():
        picked = [e for e in entries if e.get("category") == category]
        out[suite] = ([int(e["id"]) for e in picked], [e["name"] for e in picked])
    return out


def _allowed_names(suite: str, category: str | None) -> set[str]:
    ids = _id_filter()
    names: set[str] = set()
    for entry in _classification().get(suite, []):
        if category is not None and entry.get("category") != category:
            continue
        if ids is not None and int(entry["id"]) not in ids:
            continue
        names.add(entry["name"])
    return names


def make_suite(task_suite_name: str, category_value: str | None, task_order_index: int = 0):
    """The upstream suite, its task list cut to one category and the optional instance filter."""
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[task_suite_name](task_order_index=task_order_index)
    if category_value is None and _id_filter() is None:
        return suite
    allowed = _allowed_names(suite.name, category_value)
    suite.tasks = [t for t in suite.tasks if t.name in allowed]
    suite.n_tasks = len(suite.tasks)
    return suite

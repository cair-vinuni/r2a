"""Which suites this CLI knows about: one configs/trees/<name>.yaml per suite.

A suite absent from there cannot be run, and --suite all means exactly the names present.
"""

from __future__ import annotations

from robot_ood.cli.errors import CLIError
from robot_ood.core.paths import project_config

__all__ = ["load_configured_suites", "resolve_suites"]


def load_configured_suites() -> list[str]:
    """The suite names --suite all expands to, off the same config resolution every path uses."""
    # A bimanual tree (one naming lane trees) owns no corpus: its lanes are the trees that build.
    suites = project_config().get("suites") or {}
    if not suites:
        raise CLIError("no suites configured -- add configs/trees/<name>.yaml "
                       "(or point $ROBOT_OOD_CONFIG at a root yaml with a trees/ sibling)")
    return sorted(name for name, spec in suites.items() if not (isinstance(spec, dict) and spec.get("lanes")))


def resolve_suites(suite_arg: str, configured: list[str]) -> list[str]:
    """Expand --suite flag to the list of suites to act on."""
    if suite_arg == "all":
        return list(configured)
    # A bimanual tree is left out of all -- it owns no corpus of its own -- but is a suite like
    # any other when it is named, so its graph and its heads build and train the usual way.
    named = sorted(set(configured) | set(project_config().get("suites") or {}))
    if suite_arg not in named:
        raise CLIError(
            f"unknown suite {suite_arg!r}; choices: {named} or 'all' "
            "(one configs/trees/<name>.yaml each)"
        )
    return [suite_arg]

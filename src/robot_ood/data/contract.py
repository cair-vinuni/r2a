"""The graph contract a suite builds under: schema.DEFAULTS with the suite's resolved build knobs."""

from __future__ import annotations

import dataclasses

from robot_ood.core.build_config import build_param
from robot_ood.data import schema

__all__ = ["resolved_contract"]


def resolved_contract(suite: str = "") -> schema.GraphContract:
    """The contract a suite actually builds under: the build_param lookups robot_ood.cli.graph makes.

    Args:
        suite: Suite whose suites.<name>.build block is consulted; "" uses defaults.build alone.

    Returns:
        schema.DEFAULTS with vocab.dilations and build.coarsen replaced by the resolved values.
    """
    return dataclasses.replace(
        schema.DEFAULTS,
        vocab=dataclasses.replace(
            schema.DEFAULTS.vocab,
            dilations=tuple(int(d) for d in build_param(
                "graph", "dilations", list(schema.DILATIONS), suite=suite)),
        ),
        build=dataclasses.replace(
            schema.DEFAULTS.build,
            coarsen=build_param("graph", "coarsen", schema.COARSEN, suite=suite),
        ),
    )

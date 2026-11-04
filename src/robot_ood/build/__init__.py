"""Artifact builders: everything that runs once per suite and writes beside the graph.

Nothing is re-exported here on purpose: a re-export runs the re-exported module for every importer
of any sibling, which is how import robot_ood.core.config once cost a torch import (tests/test_layering.py).
"""

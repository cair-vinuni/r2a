"""Compat re-export package: `deployment.model_server.tools.*`.

The StableVLA policy server (evals/stablevla/stablevla_policy_server.py) imports its websocket
transport with the original long package path (``deployment . model_server . tools . websocket_policy_server`` ->
``WebsocketPolicyServer``).

Those transport modules are now VENDORED into this repo under
``evals/libero_plus/harness/tools/`` (see UPSTREAM.md). Rather than keep the whole VLA-JEPA donor
tree on PYTHONPATH just to satisfy that one absolute import, this thin shim package re-exports the
vendored modules under the legacy ``deployment.model_server.tools`` name. Put ``harness/`` on
PYTHONPATH (the launchers do) and the legacy dotted path resolves to the vendored code. Only
``websocket_policy_server`` is re-exported -- the one module imported by its legacy name.
"""

from tools import websocket_policy_server

__all__ = ["websocket_policy_server"]

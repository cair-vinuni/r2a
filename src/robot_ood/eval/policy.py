"""Policy descriptors: evals/policies/<id>.yaml parsed."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from robot_ood.core.paths import repo_root, resolve_trunk_path
from robot_ood.core.protocol import DEFAULT_PROTOCOL, get_protocol

__all__ = ["DEFAULT_POLICY", "Placement", "Policy", "load_policy", "policy_ids",
           "resolve_policy_python", "resolve_trunk_path"]

POLICIES_DIR = Path("evals") / "policies"
DEFAULT_POLICY = "stablevla"


@dataclass(frozen=True)
class Placement:
    """How many servers a cell runs per GPU and how many sim clients each server feeds.

    Attributes:
        servers_per_gpu: Server processes on each GPU the cell is given.
        clients_per_server: Sim clients per server; with a batching server this is its batch.
    """

    servers_per_gpu: int = 1
    clients_per_server: int = 2

    def servers(self, n_gpus: int) -> int:
        return self.servers_per_gpu * max(1, n_gpus)

    def clients(self, n_gpus: int) -> int:
        return self.servers(n_gpus) * self.clients_per_server

    @classmethod
    def from_doc(cls, doc: Mapping[str, Any] | None) -> "Placement":
        doc = doc or {}
        placement = cls(servers_per_gpu=int(doc.get("servers_per_gpu", cls.servers_per_gpu)),
                        clients_per_server=int(doc.get("clients_per_server", cls.clients_per_server)))
        if placement.servers_per_gpu < 1 or placement.clients_per_server < 1:
            raise ValueError(f"placement counts must be at least 1, got {doc}")
        return placement


@dataclass(frozen=True)
class Policy:
    """One frozen policy: conda env, server launch, checkpoint key, rollout protocol, placement."""

    id: str
    conda_env: str
    python: Path | None
    ckpt_key: str
    server_script: Path
    server_args: tuple[str, ...]
    pythonpath: tuple[Path, ...]
    protocol: str = DEFAULT_PROTOCOL
    placement: Placement = Placement()

    def server_argv(self, *, checkpoint: str, port: int) -> list[str]:
        """The server's argv after the interpreter, script first."""
        fields = {"checkpoint": checkpoint, "port": str(port)}
        return [str(self.server_script), *(a.format(**fields) for a in self.server_args)]



def resolve_policy_python(policy: "Policy") -> Path | None:
    """The interpreter a policy's server runs under: $ROBOT_OOD_PY, the descriptor's python, else its conda env."""
    override = os.environ.get("ROBOT_OOD_PY")
    if override:
        return Path(override)
    if policy.python is not None:
        return resolve_trunk_path(policy.python)
    from robot_ood.pipeline.interpreter import resolve_env_python
    return resolve_env_python(policy.conda_env)


def policy_ids(root: Path | None = None) -> list[str]:
    """Sorted policy file stems under evals/policies/."""
    root = root or repo_root()
    return sorted(p.stem for p in (root / POLICIES_DIR).glob("*.yaml"))


def load_policy(policy_id: str | None = None, root: Path | None = None) -> Policy:
    """Read evals/policies/<policy_id>.yaml; None reads $ROBOT_OOD_POLICY, else the default.

    Raises:
        FileNotFoundError: No such policy file.
        KeyError: The descriptor names an unknown protocol.
    """
    root = root or repo_root()
    policy_id = policy_id or os.environ.get("ROBOT_OOD_POLICY") or DEFAULT_POLICY
    path = root / POLICIES_DIR / f"{policy_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"unknown policy {policy_id!r}: no {path} (known: {policy_ids(root)})")
    doc = yaml.safe_load(path.read_text())
    server = doc.get("server") or {}
    return Policy(
        id=doc["id"],
        conda_env=doc["conda_env"],
        python=Path(doc["python"]) if doc.get("python") else None,
        ckpt_key=doc["ckpt_key"],
        server_script=Path(server["script"]),
        server_args=tuple(server.get("args", ())),
        pythonpath=tuple(Path(p) for p in doc.get("pythonpath", ())),
        protocol=get_protocol(doc.get("protocol", DEFAULT_PROTOCOL)).id,
        placement=Placement.from_doc(doc.get("placement")),
    )

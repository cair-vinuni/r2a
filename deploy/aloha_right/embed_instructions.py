"""Store the task posterior of the node's instructions in the artifacts: python embed_instructions.py"""

from __future__ import annotations

import os
import sys

import paths
from check import node_instructions

if str(paths.SRC) not in sys.path:
    sys.path.insert(0, str(paths.SRC))
os.environ["ROBOT_OOD_LANG_QUERIES"] = "0"


def main() -> int:
    from robot_ood.models.language.encoder import InstructionEncoder, TaskEmbeddings

    strings = node_instructions()
    tasks = TaskEmbeddings.load(paths.ARTIFACTS)
    encoder = InstructionEncoder()
    out = tasks.write_queries(paths.ARTIFACTS, list(strings.values()), encoder)
    for name, text in strings.items():
        p = tasks.task_posterior(text, encoder)
        print(f"{name} -> task {int(p.argmax())} (confidence {float(p.max()):.4f})")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

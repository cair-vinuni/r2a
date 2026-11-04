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
    encoder = InstructionEncoder()
    for arm in paths.ARMS:
        tasks = TaskEmbeddings.load(paths.ARTIFACTS[arm])
        out = tasks.write_queries(paths.ARTIFACTS[arm], list(strings.values()), encoder)
        for name, text in strings.items():
            p = tasks.task_posterior(text, encoder)
            print(f"{arm:5s} {name} -> task {int(p.argmax())} (confidence {float(p.max()):.4f})")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

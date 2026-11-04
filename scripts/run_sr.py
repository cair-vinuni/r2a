#!/usr/bin/env python
"""Success-rate eval driver; the implementation lives in robot_ood.eval.driver."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_ood.eval.driver import main

if __name__ == "__main__":
    raise SystemExit(main())

"""python -m robot_ood.cli <command> ... -- see robot_ood.cli.main for the full contract."""

import sys

from robot_ood.cli.main import main

if __name__ == "__main__":
    sys.exit(main())

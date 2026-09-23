"""Run a case from the command line.

    python examples/run_case.py configs/gfl.yaml
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peslite import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

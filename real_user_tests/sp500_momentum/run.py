"""Run the thirty-name S&P 500 momentum strategy through the falsifier.

    python real_user_tests/sp500_momentum/run.py
    python real_user_tests/sp500_momentum/run.py --offline

The equivalent one-liner, which is what the agent layer actually runs:

    qv validate --manifest real_user_tests/sp500_momentum/research_manifest.yaml

This is the example that exercises the point-in-time membership check. The other
three trade hand-picked ETFs, where index membership does not apply and
survivorship can only be declared.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from real_user_tests.driver import main  # noqa: E402

HERE = Path(__file__).parent


if __name__ == "__main__":
    raise SystemExit(main(HERE, __doc__))

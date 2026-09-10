"""Run the sector-momentum strategy through the falsifier, end to end.

This is the tool used the way it is meant to be used: a real strategy on real
data, with a real benchmark and real Fama-French factors, and every
configuration examined declared honestly.

    python real_user_tests/sector_momentum/run.py
    python real_user_tests/sector_momentum/run.py --offline

Everything this script used to do by hand - load the nine sector funds, load
SPY and the factors, evaluate the reported configuration, re-run all 24
declared configurations to rebuild the trial matrix and parameter surface, wrap
the strategy for the behavioural leakage test - now lives in `qv.pipeline`,
driven by `research_manifest.yaml`. The equivalent one-liner is:

    qv validate --manifest real_user_tests/sector_momentum/research_manifest.yaml

This script survives for the performance preamble it prints on the way.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from real_user_tests.driver import main  # noqa: E402

HERE = Path(__file__).parent


if __name__ == "__main__":
    raise SystemExit(main(HERE, __doc__))

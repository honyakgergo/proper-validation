"""Run the NASDAQ-100 short-term reversal strategy through the falsifier.

    python case_studies/nasdaq_reversal/run.py
    python case_studies/nasdaq_reversal/run.py --offline

The equivalent one-liner, which is what the agent layer actually runs:

    qv validate --manifest case_studies/nasdaq_reversal/research_manifest.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from case_studies.driver import main  # noqa: E402

HERE = Path(__file__).parent


if __name__ == "__main__":
    raise SystemExit(main(HERE, __doc__))

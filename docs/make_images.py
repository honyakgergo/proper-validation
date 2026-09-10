"""Render the README's images from a real audit.

The charts in the README are the same ones the reports carry, produced by the
same code from the same data - not screenshots, and not drawn for the README.
That matters here more than it usually would: a repository whose argument is
about unearned confidence should not illustrate itself with pictures that
cannot be regenerated.

    python docs/make_images.py --offline

`--offline` refuses the network and serves from the on-disk cache, which is
what keeps a regenerated image identical to the published one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402

from qv.pipeline import audit_from_manifest  # noqa: E402
from qv.report import charts as ch  # noqa: E402
from qv.report.render import build_charts  # noqa: E402

HERE = Path(__file__).parent
MANIFEST = HERE.parent / "real_user_tests" / "dual_momentum" / "research_manifest.yaml"

#: Only the charts the README actually shows. Rendering all nine would leave
#: unreferenced files in the repository for someone to wonder about later.
WANTED = (
    "haircut_cascade",
    "max_sharpe_null",
    "pbo_panel",
    "lookahead_horizon",
    "execution_delay_fragility",
    "universe_coverage",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Refuse the network.")
    parser.add_argument("--out", type=Path, default=HERE / "images")
    parser.add_argument("--dpi", type=int, default=110)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    original = ch._finish

    def finish_and_save(fig, name, title, spec):
        if name in WANTED:
            path = args.out / f"{name}.png"
            fig.savefig(path, format="png", bbox_inches="tight", dpi=args.dpi)
            saved.append(name)
        return original(fig, name, title, spec)

    ch._finish = finish_and_save
    try:
        print(f"Auditing {MANIFEST.parent.name} to render {len(WANTED)} charts")
        run = audit_from_manifest(MANIFEST, offline=args.offline)
        build_charts(run.report, returns=run.returns, theme="light")
    finally:
        ch._finish = original
        plt.close("all")

    missing = [name for name in WANTED if name not in saved]
    if missing:
        # A chart that silently failed to render would leave a broken image in
        # the README, which is exactly the kind of thing nobody notices.
        print(f"  WARNING: not rendered: {', '.join(missing)}", file=sys.stderr)

    for name in saved:
        path = args.out / f"{name}.png"
        print(f"  {path.relative_to(HERE.parent).as_posix()}  {path.stat().st_size // 1024} KB")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

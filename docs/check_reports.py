"""Every committed report must name the source it was generated from."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qv.provenance import source_digest

expected = source_digest()
stale = []
for html in sorted(Path("case_studies").rglob("report.html")) + sorted(
    Path("examples").rglob("report.html")
):
    found = re.search(r"source ([0-9a-f]{12})", html.read_text(encoding="utf-8"))
    if not found:
        stale.append(f"{html}: no source digest in the footer")
    elif found.group(1) != expected:
        stale.append(f"{html}: footer says {found.group(1)}, source is {expected}")

if stale:
    print("Committed reports do not match the source that would produce them:")
    for line in stale:
        print("  " + line)
    print("\nRegenerate them - see the command list in CLAUDE.md.")
    raise SystemExit(1)
print(f"all committed reports match source {expected}")

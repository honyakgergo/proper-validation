"""An adversarial validator for quantitative backtests.

It falsifies; it cannot validate. See `README.md` for what that means and
`qv.provenance` for how a report names the engine that produced it.
"""

from qv.provenance import package_version

__version__ = package_version()

__all__ = ["__version__"]

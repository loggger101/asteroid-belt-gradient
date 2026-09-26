"""Command line: ``python -m beltgradient {fetch,run}`` (also installed as ``belt-gradient``)."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .config import Paths


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="belt-gradient", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--root", type=Path, help="output root (figures/, results/, data/); default: the repository")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download and verify the AsteroidCatalog release and the Nesvorný families")
    r = sub.add_parser("run", help="regenerate every figure and results/summary.json (~3 min)")
    r.add_argument("--no-write", action="store_true", help="compute only; do not overwrite figures/ or results/")
    a = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    paths = Paths(a.root.resolve()) if a.root else Paths.default()
    if a.cmd == "fetch":
        from .fetch import fetch
        fetch(paths)
    else:
        from .pipeline import run
        run(paths, write=not a.no_write)


if __name__ == "__main__":
    main()

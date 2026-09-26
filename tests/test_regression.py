"""Full run against the committed results. Needs the data (``belt-gradient fetch``), takes ~3 min.

Run with ``pytest -m slow``.
"""
import json

import pytest

from beltgradient.config import Paths


@pytest.mark.slow
def test_full_run_reproduces_committed_summary():
    paths = Paths.default()
    if not (paths.catalog.exists() and (paths.nesvorny / "data").exists()):
        pytest.skip("inputs not present; run `belt-gradient fetch`")
    import matplotlib
    matplotlib.use("Agg")
    from beltgradient.pipeline import run

    committed = json.loads((paths.results / "summary.json").read_text())
    got = json.loads(json.dumps(run(paths, write=False, verbose=False).summary, default=str))
    for key in committed:
        if key != "software":
            assert got[key] == committed[key], key

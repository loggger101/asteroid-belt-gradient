"""Full run against the committed results. Needs the data (``belt-gradient fetch``), takes ~1.5 min.

Run with ``pytest -m slow``.
"""
import json

import pytest

from beltgradient.config import Paths

# RMS difference (0-255 grey levels) allowed between a committed figure and one drawn by another Matplotlib release,
# whose text antialiasing may differ by a pixel; a moved curve, bar or label is far above it
FIGURE_TOL = 2.0


@pytest.mark.slow
def test_full_run_reproduces_committed_outputs(tmp_path, monkeypatch):
    paths = Paths.default()
    if not (paths.catalog.exists() and (paths.nesvorny / "data").exists()):
        pytest.fail("inputs not present: run `belt-gradient fetch` (the slow test is the real run, it cannot pass without them)")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.testing.compare import compare_images
    from PIL import Image
    from beltgradient.pipeline import FIGURE_FILES, run

    monkeypatch.setenv("BELT_GRADIENT_DATA", str(paths.data))
    out = Paths(tmp_path)
    run(out, verbose=False)

    # the summary, byte for byte (that includes the version and the seed)
    committed = (paths.results / "summary.json").read_text(encoding="utf-8")
    got = (out.results / "summary.json").read_text(encoding="utf-8")
    if got != committed:
        c, g = json.loads(committed), json.loads(got)
        assert {k: g[k] for k in g if g[k] != c.get(k)} == {}, "summary.json differs from the committed one"
        assert got == committed                         # same content, different bytes (key order, formatting)

    # every figure: identical under the Matplotlib that drew it, visually identical under another
    for name, _ in FIGURE_FILES.values():
        ref, new = paths.figures / name, out.figures / name
        with Image.open(ref) as im:
            drawn_with = im.text.get("Software", "")
        assert drawn_with.startswith("Matplotlib version"), f"{name}: no Matplotlib version recorded ({drawn_with!r})"
        if drawn_with.startswith(f"Matplotlib version{matplotlib.__version__},"):
            assert new.read_bytes() == ref.read_bytes(), f"{name} differs from the committed figure"
        else:
            diff = compare_images(str(ref), str(new), tol=FIGURE_TOL)
            assert diff is None, f"{name} (committed with {drawn_with!r}, drawn with {matplotlib.__version__}): {diff}"

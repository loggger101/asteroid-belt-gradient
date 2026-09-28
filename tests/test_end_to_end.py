"""The whole pipeline on synthetic inputs (conftest.py): every stage runs, every output is written, and the
numbers recover what the synthetic belt was built with. No catalog, no network; about 15 s."""
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pytest

import beltgradient
from beltgradient.config import ZONE_NAMES
from beltgradient.pipeline import FIGURE_FILES, run

COMMITTED_SUMMARY = Path(__file__).resolve().parents[1] / "results" / "summary.json"
OUTPUTS = ["results/summary.json"] + [f"figures/{name}" for name, _ in FIGURE_FILES.values()]


@pytest.fixture(scope="module")
def result(synthetic):
    paths, expected = synthetic
    rc, filters = dict(plt.rcParams), list(warnings.filters)
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("BELT_GRADIENT_DATA", raising=False)
        res = run(paths, verbose=False)
    return SimpleNamespace(paths=paths, expected=expected, res=res,
                           summary=json.loads((paths.results / "summary.json").read_text(encoding="utf-8")),
                           state_kept=(dict(plt.rcParams) == rc, list(warnings.filters) == filters))


def point(cell: str) -> float:
    """The estimate in a zone-table cell, "0.53 [0.51–0.55] (n=…)"."""
    return float(cell.split()[0])


def shape(x):
    """Keys all the way down; a list is described by its first element."""
    if isinstance(x, dict):
        return {k: shape(v) for k, v in x.items()}
    if isinstance(x, list):
        return [shape(x[0])] if x else []
    return None


def test_every_output_is_written(result):
    for name, _ in FIGURE_FILES.values():
        assert (result.paths.figures / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", name
    assert result.summary == json.loads(json.dumps(result.res.summary, default=str))


def test_summary_has_the_structure_of_the_committed_one(result):
    # the same keys at every level: a stage that stops reporting a number, or reports it under another name, fails here
    assert shape(result.summary) == shape(json.loads(COMMITTED_SUMMARY.read_text(encoding="utf-8")))


def test_selection_and_families(result):
    s, e = result.summary, result.expected
    assert (s["catalog"]["rows"], s["catalog"]["main_belt"]) == (e["rows"], e["main_belt"])
    assert s["families"]["n_families"] == e["n_families"]
    # Karin (nested in Koronis), Vibilia (Leonidas's list again) and Hilda (beyond 3.3 AU) keep no main-belt member
    assert set(result.res.tables["families"].fam_name) == {"Vesta", "Flora", "Themis", "Koronis", "Eos", "Leonidas",
                                                           "Irma", "2012 PM61"}
    assert s["text_numbers"]["vesta_family_members"] == e["vesta_members"]
    assert [b["name"] for b in s["text_numbers"]["flora_family_two_largest"]] == ["Metis", "Flora"]
    assert s["families"]["extension_attaches"] >= e["n_halo"]


def test_completeness_limits(result):
    c = result.summary["completeness"]
    assert (c["H_c"], c["D_complete_km"], c["H_c_albedo"]) == (10.5, 52.8, 12.0)


def test_debiasing_recovers_the_true_crossover(result):
    xo = result.summary["crossover"]
    raw, *debiased = xo
    for k in debiased:
        assert xo[k]["a50"] == pytest.approx(result.expected["true_a50"], abs=0.05), k
    # dark C-types are fainter, so labelled less often: the raw crossover lies outside the debiased ones
    assert xo[raw]["a50"] > max(xo[k]["a50"] for k in debiased) + 0.02


def test_fractions_rise_outward(result):
    s = result.summary
    for k in s["crossover"]:
        f = [point(s["zone_c_fraction_narrow"][z][k]) for z in ZONE_NAMES]
        assert f == sorted(f), k
    dark = [s["dark_fraction_complete"][z] for z in ZONE_NAMES]
    assert dark == sorted(dark)
    # Vesta holds most of the inner belt's mass: without the 4 most massive bodies it is more carbonaceous
    assert s["mass_c_fraction"]["no_big4"]["inner"] > s["mass_c_fraction"]["all"]["inner"]
    assert len(s["orbit_tests"]) == 8


def test_run_leaves_matplotlib_and_warnings_as_it_found_them(result):
    assert result.state_kept == (True, True)


def test_cli_reproduces_the_run_byte_for_byte(result, tmp_path):
    # The same inputs, read through BELT_GRADIENT_DATA, by the command line writing its log to a stream that
    # cannot encode "≥" (a redirected Windows console), must give identical files.
    src = str(Path(beltgradient.__file__).resolve().parents[1])
    env = {**os.environ, "BELT_GRADIENT_DATA": str(result.paths.data), "PYTHONIOENCODING": "cp1252",
           "PYTHONPATH": os.pathsep.join([src, os.environ.get("PYTHONPATH", "")])}
    p = subprocess.run([sys.executable, "-m", "beltgradient", "--root", str(tmp_path), "run"], env=env, capture_output=True)
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
    assert "D≥10 km" in p.stdout.decode("utf-8")
    for f in OUTPUTS:
        assert (tmp_path / f).read_bytes() == (result.paths.root / f).read_bytes(), f

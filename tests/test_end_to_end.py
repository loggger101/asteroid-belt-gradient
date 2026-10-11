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
    assert result.summary == json.loads(json.dumps(result.res.summary, allow_nan=False))   # plain JSON, nothing coerced


def test_summary_has_the_structure_of_the_committed_one(result):
    # the same keys at every level: a stage that stops reporting a number, or reports it under another name, fails here
    assert shape(result.summary) == shape(json.loads(COMMITTED_SUMMARY.read_text(encoding="utf-8")))


def test_selection_and_families(result):
    s, e = result.summary, result.expected
    assert (s["catalog"]["rows"], s["catalog"]["main_belt"]) == (e["rows"], e["main_belt"])
    assert s["families"]["n_families"] == e["n_families"]
    # Karin (nested in Koronis), Vibilia (Leonidas's list again) and Hilda (beyond 3.3 AU) keep no main-belt member
    assert set(result.res.tables["families"].fam_name) == {"Vesta", "Flora", "Nysa-Polana", "Eunomia", "Hygiea", "Themis",
                                                           "Koronis", "Eos", "Leonidas", "Irma", "2012 PM61"}
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


# ── the summary is plain JSON: nothing is stringified to make it fit ─────────────────────────────

def test_summary_json_has_no_stringified_numbers(result):
    summary = result.summary

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            assert not o.replace(".", "", 1).lstrip("-").isdigit(), f"number stored as a string: {o!r}"
    walk(summary)


# ── the command line, in process ─────────────────────────────────────────────────────────────────

def test_cli_version_and_usage_errors(capsys):
    from beltgradient.__main__ import main
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and capsys.readouterr().out.strip() == beltgradient.__version__
    with pytest.raises(SystemExit) as e:
        main([])                                         # a command is required
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        main(["publish"])                                # and must be one it knows
    assert e.value.code == 2


def test_cli_run_no_write_leaves_the_root_empty(result, tmp_path, monkeypatch):
    from beltgradient.__main__ import main
    monkeypatch.setenv("BELT_GRADIENT_DATA", str(result.paths.data))
    main(["--root", str(tmp_path), "run", "--no-write"])
    assert list(tmp_path.iterdir()) == []


def test_cli_run_with_missing_inputs_raises(tmp_path, monkeypatch):
    from beltgradient.__main__ import main
    monkeypatch.delenv("BELT_GRADIENT_DATA", raising=False)
    with pytest.raises(FileNotFoundError):
        main(["--root", str(tmp_path), "run", "--no-write"])


def test_paths_default_is_the_repository():
    from beltgradient.config import Paths
    assert (Paths.default().root / "pyproject.toml").is_file()


def test_the_family_extension_option_runs(result, monkeypatch):
    from beltgradient import pipeline
    monkeypatch.setattr(pipeline, "EXTEND_FAMILIES", True)
    monkeypatch.delenv("BELT_GRADIENT_DATA", raising=False)
    ext = run(result.paths, write=False, verbose=False).summary
    base = result.summary["families"]
    assert ext["families"]["extension_used"] is True and base["extension_used"] is False
    assert ext["families"]["family_fraction"] > base["family_fraction"]


def test_the_robustness_figure_refuses_a_missing_estimate(result):
    import copy
    from beltgradient import figures
    s = copy.deepcopy(result.summary)
    keys = list(s["zone_c_fraction_narrow"]["inner"])
    s["dark_fraction_complete"]["outer"] = None
    with pytest.raises(ValueError, match="no outer-zone estimate for 'albedo only"):
        figures.robustness_summary(s, keys)
    s = copy.deepcopy(result.summary)
    s["zone_c_fraction_narrow"]["pristine"][keys[0]] = "nan [nan–nan] (n=0)"
    with pytest.raises(ValueError, match="not a zone fraction"):
        figures.robustness_summary(s, keys)


# ── inside a run, nothing goes by on the console: a warning is an error, a NaN never reaches the summary ─

def test_a_warning_inside_the_run_is_an_error(result, monkeypatch):
    from beltgradient import pipeline
    real = pipeline.load_catalog

    def warns(path):
        warnings.warn("planted", UserWarning)
        return real(path)

    monkeypatch.setattr(pipeline, "load_catalog", warns)
    monkeypatch.delenv("BELT_GRADIENT_DATA", raising=False)
    filters = list(warnings.filters)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                 # however the caller treats warnings
        with pytest.raises(UserWarning, match="planted"):
            run(result.paths, write=False, verbose=False)
    assert list(warnings.filters) == filters


def test_a_nan_in_the_summary_raises_even_without_writing(result, monkeypatch, tmp_path):
    from beltgradient import pipeline
    monkeypatch.setattr(pipeline, "snow_line_track_numbers", lambda: {"min_au": float("nan")})
    monkeypatch.delenv("BELT_GRADIENT_DATA", raising=False)
    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        run(result.paths, write=False, verbose=False)


# ── figures: labels come from the run, and what a figure claims to show must be there ────────────────

def test_the_crossover_figure_labels_come_from_the_samples(result):
    from beltgradient import figures
    xo, d_c = result.res.tables["crossover"], result.summary["completeness"]["D_complete_km"]
    fig = figures.crossover_fits(xo, d_c)
    labels = [t.get_text() for t in fig.axes[1].get_yticklabels()]
    assert labels == [figures._math(s) for s in ["background", f"D ≥ {d_c:.0f} km", "IPW, D ≥ 10 km", "raw"]]
    plt.close(fig)
    with pytest.raises(ValueError, match="crossover figure: expected the samples"):
        figures.crossover_fits(xo, d_c + 5)             # a fit table from other cuts than the labels name
    with pytest.raises(ValueError, match="crossover figure: expected the samples"):
        figures.crossover_fits(xo.iloc[:3], d_c)        # a sample missing


def test_the_robustness_figure_names_the_runs_cuts(result):
    import copy
    from beltgradient import figures
    keys = list(result.summary["zone_c_fraction_narrow"]["inner"])
    s = copy.deepcopy(result.summary)
    s["completeness"]["D_complete_km"], s["completeness"]["D_complete_albedo_km"] = 61.2, 44.4
    fig = figures.robustness_summary(s, keys)
    labels = " | ".join(t.get_text() for t in fig.axes[0].get_yticklabels())
    plt.close(fig)
    assert "$D$ ≥ 61 km, families extended" in labels and "$D$ ≥ 44 km" in labels
    assert "53 km" not in labels and "33 km" not in labels


def test_the_proper_element_map_refuses_a_missing_labelled_family(result):
    from beltgradient import figures
    ftab = result.res.tables["families"].copy()
    ftab["fam_name"] = ftab.fam_name.replace("Eunomia", "Renamed")
    with pytest.raises(ValueError, match=r"the Eunomia family it labels has ids \[\] and 0 members"):
        figures.proper_element_map(result.res.frames["main_belt"], result.res.frames["orbit_sample"], ftab)


@pytest.mark.parametrize("bad", [0.0, 0.005, 1.2])
def test_the_albedo_figure_refuses_values_off_its_axis(result, bad):
    from beltgradient import figures
    mb = result.res.frames["main_belt"].copy()
    i = mb.index[mb.tier.eq("taxonomy") & mb.albedo.notna()][0]
    mb.loc[i, "albedo"] = bad
    with pytest.raises(ValueError, match="albedo figure: 1 measured albedos outside"):
        figures.albedo_by_group(mb, (0.10, 0.30))


# ── the walkthrough and the committed summary stay in step with the code ─────────────────────────────

def test_the_walkthrough_runs(result, monkeypatch, capsys):
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "examples" / "walkthrough.py"
    spec = importlib.util.spec_from_file_location("walkthrough", path)
    walk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(walk)
    monkeypatch.setattr(walk, "run", lambda paths, write: result.res)      # the run above, not the real inputs
    walk.main()
    out = capsys.readouterr().out
    for title in ["Label provenance", "Collisional families", "S/C crossover", "Orbital excitation",
                  "one-step family extension"]:
        assert title in out, title


def test_the_committed_summary_is_from_this_version():
    committed = json.loads(COMMITTED_SUMMARY.read_text(encoding="utf-8"))["software"]
    from beltgradient.config import SEED
    assert committed == {"beltgradient": beltgradient.__version__, "seed": SEED}, \
        "results/summary.json was not regenerated after the version or the seed changed: run `belt-gradient run`"

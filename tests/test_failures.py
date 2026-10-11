"""Every way an input or a step can go wrong must raise, naming the problem, never pass silently.

Each test plants one defect (in a copy of the synthetic inputs, or in a small frame) and checks the error.
"""
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from beltgradient import fetch as fetch_mod
from beltgradient import gradient, pipeline
from beltgradient.albedo import dark_curve
from beltgradient.catalog import COLS, load_catalog, main_belt
from beltgradient.completeness import add_h_bins, add_ipw_weights, completeness
from beltgradient.config import ZONE_NAMES, Paths
from beltgradient.families import attach_families, load_families, load_proper_elements, verify_bundle
from beltgradient.gradient import (bin_index, collapsed, crossover, fit_logistic, inner_belt_by_size, mass_by_zone,
                                   zone_table)
from beltgradient.orbits import ks_only, orbit_tests


@pytest.fixture
def bundle(synthetic, tmp_path):
    """A private copy of the synthetic family bundle, to damage."""
    paths, _ = synthetic
    dest = tmp_path / paths.nesvorny.name
    shutil.copytree(paths.nesvorny, dest)
    return dest


def _relabel(path: Path):
    """Rewrite a data file's label so it matches the file again (to plant a content defect, not a size one)."""
    raw = path.read_bytes()
    n_lines = raw.count(b"\n")
    path.with_suffix(".xml").write_text(f"<File><file_size unit=\"byte\">{len(raw)}</file_size>"
                                        f"<records>{n_lines}</records></File>\n", encoding="utf-8")


# ── the family bundle against its PDS4 labels ─────────────────────────────────────────────────────

def test_the_synthetic_bundle_matches_its_labels(bundle):
    verify_bundle(bundle)


def test_a_truncated_member_file_raises(bundle):
    f = next((bundle / "data" / "families_2015").glob("*.tab"))
    f.write_bytes(f.read_bytes()[:-10])
    with pytest.raises(ValueError, match=f"{f.name}: .* bytes"):
        verify_bundle(bundle)


def test_a_missing_member_file_raises(bundle):
    f = next((bundle / "data" / "families_2024").glob("*.csv"))
    f.unlink()
    with pytest.raises(ValueError, match=f"{f.stem}.csv is missing"):
        verify_bundle(bundle)


def test_a_member_file_without_a_label_raises(bundle):
    extra = bundle / "data" / "families_2015" / "999_extra.tab"
    extra.write_text("1 2.3 0.1 0.1 12.0 -1.0 999 1 Extra\n", encoding="ascii")
    with pytest.raises(ValueError, match="999_extra.tab has no label"):
        verify_bundle(bundle)


def test_a_wrong_record_count_raises(bundle):
    xml = bundle / "data" / "proper_catalog24.xml"
    xml.write_text(xml.read_text(encoding="utf-8").replace("<records>", "<records>1"), encoding="utf-8")
    with pytest.raises(ValueError, match="proper_catalog24.tab: .* lines; its label says"):
        verify_bundle(bundle)


def test_a_missing_folder_or_document_raises(bundle):
    shutil.rmtree(bundle / "data" / "families_2024")
    (bundle / "document" / "list_of_new_families_2024.txt").unlink()
    with pytest.raises(ValueError, match="missing folder .*families_2024(.|\n)*missing .*list_of_new_families_2024"):
        verify_bundle(bundle)


# ── parsing the bundle: nothing is skipped silently ───────────────────────────────────────────────

def test_members_of_an_unlisted_family_raise(bundle):
    fl = bundle / "data" / "familylist.tab"
    lines = fl.read_text(encoding="ascii").splitlines(keepends=True)
    fl.write_text("".join(l for l in lines if not l.startswith("602 ")), encoding="ascii")     # drop Themis
    _relabel(fl)
    with pytest.raises(ValueError, match=r"members of unlisted families \['2015_602'\]"):
        load_families(bundle)


def test_a_short_member_line_raises(bundle):
    f = sorted((bundle / "data" / "families_2015").glob("*.tab"))[1]
    f.write_bytes(f.read_bytes() + b"12345 2.3 0.1\r\n")
    _relabel(f)
    with pytest.raises(ValueError, match=f"{f.name} line .*3 fields"):
        load_families(bundle)


def test_a_2024_file_without_a_cutoff_raises(bundle):
    doc = bundle / "document" / "list_of_new_families_2024.txt"
    doc.write_text("".join(l for l in doc.read_text(encoding="ascii").splitlines(keepends=True) if "Irma" not in l),
                   encoding="ascii")
    with pytest.raises(ValueError, match="middle_177_irma_fam3.csv has no line"):
        load_families(bundle)


def test_a_2024_list_line_without_a_file_raises(bundle):
    doc = bundle / "document" / "list_of_new_families_2024.txt"
    doc.write_text(doc.read_text(encoding="ascii") + "999 Nobody              40      20\r\n", encoding="ascii")
    with pytest.raises(ValueError, match=r"without a member file: \['999'\]"):
        load_families(bundle)


def test_duplicate_proper_elements_raise(bundle):
    pc = bundle / "data" / "proper_catalog24.tab"
    first = pc.read_bytes().split(b"\n", 1)[0] + b"\n"
    pc.write_bytes(pc.read_bytes() + first)
    _relabel(pc)
    with pytest.raises(ValueError, match="lists 1 designations twice"):
        load_proper_elements(bundle)


def test_attach_families_refuses_duplicate_designations():
    mb = pd.DataFrame({"key": ["1", "1"], "semi_major_axis_au": [2.3, 2.4]})
    members = pd.DataFrame({"key": ["1"], "fam_id": ["2015_401"]})
    pc = pd.DataFrame({"key": ["1"], "a_p": [2.3], "e_p": [0.1], "sini_p": [0.1]})
    with pytest.raises(ValueError, match="main belt: designations not unique"):
        attach_families(mb, members, pc)


# ── the catalog ──────────────────────────────────────────────────────────────────────────────────

def _tiny_catalog(path, **changes):
    row = {"designation": "1", "name": "Ceres", "spectral_type": "C", "spectral_type_source": "source",
           "semi_major_axis_au": 2.77, "eccentricity": 0.08, "inclination_deg": 10.6, "perihelion_au": 2.55,
           "diameter_km": 939.4, "diameter_source": "measured", "albedo": 0.09, "absolute_magnitude_h": 3.3,
           "estimated_mass_kg": 9.4e20, "mass_measured": True, "albedo_assumed_for_diameter": np.nan,
           "catalog_date": "2026-09-29", "pipeline_version": "1.8.1"}
    df = pd.DataFrame([row, {**row, "designation": "2", "name": "Pallas"}])
    for k, v in changes.items():
        df[k] = v
    df[COLS].to_parquet(path, index=False)
    return path


def test_a_catalog_from_another_build_raises(tmp_path):
    p = _tiny_catalog(tmp_path / "c.parquet", catalog_date=["2026-09-29", "2026-10-01"])
    with pytest.raises(ValueError, match="not the expected release build"):
        load_catalog(p)
    p = _tiny_catalog(tmp_path / "c.parquet", pipeline_version="1.9.0")
    with pytest.raises(ValueError, match="data contract 1.8.1"):
        load_catalog(p)


def test_colliding_designations_raise(tmp_path):
    p = _tiny_catalog(tmp_path / "c.parquet", designation=["2003 MF13", "2003MF13"])
    with pytest.raises(ValueError, match="collide after normalisation"):
        load_catalog(p)


def test_an_unknown_label_route_raises(tmp_path):
    cat = load_catalog(_tiny_catalog(tmp_path / "c.parquet", spectral_type_source=["source", "machine_learning"]))
    with pytest.raises(ValueError, match=r"unknown spectral_type_source: \['machine_learning'\]"):
        main_belt(cat)


def test_the_main_belt_excludes_its_edges(tmp_path):
    cat = load_catalog(_tiny_catalog(tmp_path / "c.parquet", semi_major_axis_au=[2.1, 2.2], perihelion_au=2.0))
    mb = main_belt(cat)
    assert mb.designation.tolist() == ["2"] and mb.zone.notna().all()        # 2.1 < a < 3.3, as the paper states


# ── binning, fits and samples ───────────────────────────────────────────────────────────────────

def test_bin_index_puts_the_last_edge_in_the_last_bin():
    assert bin_index([2.1, 2.3, 3.3], [2.1, 2.5, 3.3], "a").tolist() == [0, 0, 1]


@pytest.mark.parametrize("bad", [2.0, 3.4, np.nan])
def test_bin_index_raises_outside_the_edges(bad):
    with pytest.raises(ValueError, match=r"1 a outside \[2.1, 3.3\]"):
        bin_index([2.3, bad], [2.1, 2.5, 3.3], "a")


def test_an_empty_bin_has_no_fraction_and_no_warning():
    s = pd.DataFrame({"semi_major_axis_au": [2.15, 2.16, 2.9], "group": ["S-like", "C-like", "C-like"], "w": 1.0})
    c = gradient.c_fraction_curve(s, np.random.default_rng(0), edges=np.array([2.1, 2.5, 2.8, 3.3]), n_boot=20)
    assert c.n.tolist() == [2, 0, 1] and np.isnan(c.f[1]) and np.isnan(c.lo[1]) and np.isnan(c.hi[1])
    assert c.f[0] == 0.5 and not np.isnan(c.lo[0])


def test_crossover_refuses_a_sample_too_small_to_fit():
    s = pd.DataFrame({"semi_major_axis_au": np.linspace(2.2, 3.2, 10), "group": ["S-like", "C-like"] * 5, "w": 1.0})
    with pytest.raises(ValueError, match="at least 20"):
        crossover(s, np.random.default_rng(0))


def test_a_logistic_fit_that_does_not_converge_raises(monkeypatch):
    class Stuck:
        x, jac, fun, message = np.array([0.0, 3.0]), np.array([5.0, 5.0]), 10.0, "stopped"
    monkeypatch.setattr(gradient.optimize, "minimize", lambda *a, **k: Stuck())
    with pytest.raises(RuntimeError, match="did not converge: relative gradient 5.00e-01"):
        fit_logistic(np.array([2.3, 2.9]), np.array([0.0, 1.0]), np.array([1.0, 1.0]))


def test_an_empty_size_bin_of_the_inner_belt_raises():
    mb = pd.DataFrame({"tier": "taxonomy", "zone": "inner", "group": ["S-like", "C-like"], "diameter_km": [150.0, 30.0],
                       "estimated_mass_kg": [1e18, 1e17]})
    with pytest.raises(ValueError, match="no inner-belt S- or C-type body with 50 <= D < 100 km"):
        inner_belt_by_size(mb)


def test_a_classified_family_needs_a_representative_orbit():
    df = pd.DataFrame({"in_family": [False], "semi_major_axis_au": [2.3], "group": ["S-like"], "tier": ["taxonomy"]})
    ft = pd.DataFrame({"fam_group": ["C-like"], "a_rep": [np.nan], "D_equiv_km": [50.0], "mass_kg": [1e17],
                       "e_rep": [0.1], "sini_rep": [0.1], "fam_name": ["Nameless"]}, index=["2015_999"])
    with pytest.raises(ValueError, match=r"without a representative orbit: \['2015_999'\]"):
        collapsed(df, ft)


# ── fetch: the download and the extracted bundle are verified ───────────────────────────────────

def test_fetch_downloads_extracts_and_verifies(synthetic, tmp_path, monkeypatch):
    src, _ = synthetic
    zpath = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for f in src.nesvorny.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(src.data).as_posix())
    monkeypatch.setattr(fetch_mod, "CATALOG_URL", src.catalog.as_uri())
    monkeypatch.setattr(fetch_mod, "CATALOG_SHA256", fetch_mod.sha256(src.catalog))
    monkeypatch.setattr(fetch_mod, "NESVORNY_URL", zpath.as_uri())
    monkeypatch.setattr(fetch_mod, "NESVORNY_SHA256", fetch_mod.sha256(zpath))
    monkeypatch.delenv("BELT_GRADIENT_DATA", raising=False)
    dest = Paths(tmp_path / "root")
    fetch_mod.fetch(dest)
    verify_bundle(dest.nesvorny)
    assert dest.catalog.read_bytes() == src.catalog.read_bytes() and not (dest.data / "nesvorny_families_v2.zip").exists()
    fetch_mod.fetch(dest)                                # second call: both inputs found and verified
    member = next((dest.nesvorny / "data" / "families_2015").glob("*.tab"))
    member.write_bytes(member.read_bytes()[:-5])         # then damaged: the next fetch must say so
    with pytest.raises(ValueError, match=member.name):
        fetch_mod.fetch(dest)


def test_a_missing_data_file_raises(bundle):
    (bundle / "data" / "proper_catalog24.tab").unlink()
    with pytest.raises(ValueError, match="missing .*proper_catalog24.tab"):
        verify_bundle(bundle)


def test_an_ambiguous_label_raises(bundle):
    xml = bundle / "data" / "familylist.xml"
    xml.write_text(xml.read_text(encoding="utf-8") + '<file_size unit="byte">1</file_size>\n', encoding="utf-8")
    with pytest.raises(ValueError, match="familylist.xml: expected one file_size"):
        verify_bundle(bundle)


def test_a_family_listed_twice_raises(bundle):
    fl = bundle / "data" / "familylist.tab"
    lines = fl.read_text(encoding="ascii").splitlines(keepends=True)
    fl.write_text("".join(lines) + next(l for l in lines if l.startswith("401 ")), encoding="ascii")
    _relabel(fl)
    with pytest.raises(ValueError, match=r"family ids listed twice: \['2015_401'\]"):
        load_families(bundle)


def test_the_cli_fetches_too(synthetic, tmp_path, monkeypatch):
    from beltgradient.__main__ import main
    src, _ = synthetic
    monkeypatch.setattr(fetch_mod, "CATALOG_URL", src.catalog.as_uri())
    monkeypatch.setattr(fetch_mod, "CATALOG_SHA256", fetch_mod.sha256(src.catalog))
    monkeypatch.delenv("BELT_GRADIENT_DATA", raising=False)
    root = tmp_path / "root"
    shutil.copytree(src.nesvorny, Paths(root).nesvorny)          # already extracted: verified, not downloaded
    main(["--root", str(root), "fetch"])
    assert Paths(root).catalog.read_bytes() == src.catalog.read_bytes()


# ── more of the bundle: rows that contradict their file, lists that repeat, lines that are short ─────────

def test_a_2015_member_row_naming_another_family_raises(bundle):
    f = next((bundle / "data" / "families_2015").glob("401_*.tab"))
    first = f.read_bytes().split(b"\n", 1)[0].decode("ascii").rstrip("\r")
    t = first.split()
    t[6] = "402"                                          # a Flora row inside Vesta's file
    f.write_bytes(f.read_bytes() + (" ".join(t) + "\r\n").encode("ascii"))
    _relabel(f)
    with pytest.raises(ValueError, match=f"{f.name} line .* names family 402, not 401"):
        load_families(bundle)


def test_a_2024_parent_listed_twice_raises(bundle):
    doc = bundle / "document" / "list_of_new_families_2024.txt"
    text = doc.read_text(encoding="ascii")
    irma = next(l for l in text.splitlines(keepends=True) if "Irma" in l)
    doc.write_text(text + irma.replace("60", "75", 1), encoding="ascii")     # the second line would win silently
    with pytest.raises(ValueError, match="lists parent 177 twice"):
        load_families(bundle)


def test_a_short_2024_member_line_raises(bundle):
    f = next((bundle / "data" / "families_2024").glob("middle_177_*.csv"))
    lines = f.read_bytes().split(b"\r\n")
    lines[1] = lines[1].rsplit(b",", 1)[0]               # the designation is gone
    f.write_bytes(b"\r\n".join(lines))
    _relabel(f)
    with pytest.raises(ValueError, match=f"{f.stem}.csv: expected 9 fields .* 1 lines without a last field"):
        load_families(bundle)


def test_a_short_proper_element_line_raises(bundle):
    pc = bundle / "data" / "proper_catalog24.tab"
    first = pc.read_bytes().split(b"\n", 1)[0].rstrip()             # the designation is padded with spaces
    pc.write_bytes(pc.read_bytes() + first.rsplit(b" ", 1)[0] + b"\r\n")    # every field but the designation
    _relabel(pc)
    with pytest.raises(ValueError, match="proper_catalog24.tab: 1 lines lack a field"):
        load_proper_elements(bundle)


def test_a_body_without_an_orbit_raises(tmp_path):
    cat = load_catalog(_tiny_catalog(tmp_path / "c.parquet", semi_major_axis_au=[2.77, np.nan]))
    with pytest.raises(ValueError, match=r"1 bodies have no semimajor axis or perihelion, e.g. \['2'\]"):
        main_belt(cat)


# ── completeness: no labelled bright body may fall outside the H bins ───────────────────────────────

def _h_frame(h):
    mb = pd.DataFrame({"designation": [str(i) for i in range(len(h))], "absolute_magnitude_h": h, "tier": "taxonomy",
                       "zone": pd.Categorical(["inner"] * len(h), categories=ZONE_NAMES)})
    return add_h_bins(mb)


def test_a_labelled_bright_body_without_a_completeness_raises():
    mb = _h_frame([5.0, 6.0, np.nan])
    comp, _ = completeness(mb, mb.tier.eq("taxonomy"))
    with pytest.raises(ValueError, match=r"1 taxonomy-labelled bodies brighter than H = 15.0 .*'designation': '2'"):
        add_ipw_weights(mb, comp)


def test_a_faint_body_outside_the_bins_is_allowed_and_weighs_nothing():
    mb = _h_frame([5.0, 6.0, 25.0])                      # too faint for IPW whatever its completeness
    comp, _ = completeness(mb, mb.tier.eq("taxonomy"))
    out = add_ipw_weights(mb, comp)
    assert np.isnan(out.c[2]) and out.w_ipw.tolist() == [1.0, 1.0, 0.0]


# ── the label-free dark fraction is binned and bootstrapped like the C-fraction ──────────────────────

def test_dark_curve_raises_outside_its_edges():
    d = pd.DataFrame({"semi_major_axis_au": [2.3, 3.4], "dark": [True, False]})
    with pytest.raises(ValueError, match=r"1 semimajor axes outside \[2.1, 3.3\]"):
        dark_curve(d, np.array([2.1, 2.7, 3.3]), np.random.default_rng(0), n_boot=20)


def test_dark_curve_skips_empty_bins_and_empty_replicates():
    # one dark body in the first bin: a replicate that draws it zero times (37% of them) has no fraction. Counted
    # as 0, as before, it pulled the 16% point to 0; skipped, the interval is [1, 1]. The middle bin is empty.
    d = pd.DataFrame({"semi_major_axis_au": [2.2, 3.0, 3.1], "dark": [True, False, True]})
    x, f, lo, hi, n = dark_curve(d, np.array([2.1, 2.5, 2.9, 3.3]), np.random.default_rng(0), n_boot=200)
    assert n.tolist() == [1, 0, 2] and f[0] == lo[0] == hi[0] == 1.0
    assert np.isnan(f[1]) and np.isnan(lo[1]) and np.isnan(hi[1]) and f[2] == 0.5


# ── the orbit test runs all eight tests or none ──────────────────────────────────────────────────────

def _orbit_frame(n_c_outer):
    rng = np.random.default_rng(1)
    rows = [dict(zone=z, group=g, e_p=rng.normal(0.1, 0.03), sini_p=rng.normal(0.1, 0.03))
            for z in ZONE_NAMES for g in ["S-like", "C-like"] for _ in range(n_c_outer if (z, g) == ("outer", "C-like") else 12)]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("test", [lambda o: orbit_tests(o, np.random.default_rng(0)), ks_only], ids=["orbit_tests", "ks_only"])
def test_a_zone_too_thin_to_test_raises(test):
    assert len(test(_orbit_frame(10))) == 8
    with pytest.raises(ValueError, match="outer zone, proper e: 12 S-like and 9 C-like bodies; each class needs at least 10"):
        test(_orbit_frame(9))


# ── samples, zone tables and sums: nothing empty is reported as a number ─────────────────────────────

def test_a_zone_table_with_an_empty_zone_raises():
    s = pd.DataFrame({"semi_major_axis_au": [2.2, 2.3, 2.6, 2.7, 3.1, 3.2], "w": 1.0,
                      "group": ["S-like", "C-like"] * 3})                                   # nobody in the pristine zone
    with pytest.raises(ValueError, match="sample 'thin' .narrow. has no body in the pristine zone"):
        zone_table({"thin": s}, np.random.default_rng(0))


def test_mass_by_zone_refuses_a_group_left_out_of_the_order():
    d = pd.DataFrame({"zone": ["inner", "outer"], "group": ["S-like", "other"], "estimated_mass_kg": [1.0, 2.0]})
    with pytest.raises(ValueError, match=r"groups \['other'\] are not in the order"):
        mass_by_zone(d, ["S-like", "C-like"])


def test_a_weighted_fraction_of_nothing_raises():
    s = pd.DataFrame({"semi_major_axis_au": [2.6], "group": ["C-like"], "w": [1.0]})
    assert pipeline._weighted_c_fraction(s, "w", 2.5, 2.7) == 1.0
    with pytest.raises(ValueError, match="no weighted narrow-fraction body with 2.1 < a <= 2.2 au"):
        pipeline._weighted_c_fraction(s, "w", 2.1, 2.2)


def test_an_empty_outer_bin_of_the_dp_share_raises():
    a = [2.95, 3.05, 3.25]                                                                 # 3.1-3.2 au is empty
    s = pd.DataFrame({"semi_major_axis_au": a, "group": ["D/P", "C-like", "S-like"], "w": 1.0})
    with pytest.raises(ValueError, match=r"D/P share: no body in the bins \['\(3.1, 3.2\]'\]"):
        pipeline.dp_share_outer(s)


def test_a_family_named_in_the_text_must_be_there_once():
    ftab = pd.DataFrame({"fam_name": ["Eos", "Flora", "Flora"]}, index=["2015_606", "2015_402", "2024_x"])
    assert pipeline._family(ftab, "Eos") == "2015_606"
    for name, found in [("Vesta", r"\[\]"), ("Flora", r"\['2015_402', '2024_x'\]")]:
        with pytest.raises(ValueError, match=f"one family named '{name}' with main-belt members, found {found}"):
            pipeline._family(ftab, name)


def test_a_nan_summary_number_raises_and_nan_ranges_are_refused():
    assert pipeline._round(0.123456) == 0.1235 and pipeline._round_cell(np.nan) is None    # a blank map cell is null
    with pytest.raises(ValueError, match="a summary number is NaN"):
        pipeline._round(np.nan)
    with pytest.raises(ValueError, match="gap: NaN where every value must be defined"):
        pipeline._finite(pd.DataFrame({"a": [0.1, np.nan]}), "gap")


def test_a_snow_line_that_never_reaches_a_radius_raises():
    assert pipeline.snow_line_track_numbers()["first_inside_au_myr"]["2.1"] > 0
    with pytest.raises(ValueError, match=r"never lies inside \[1.0\] au"):
        pipeline.snow_line_track_numbers(cuts=(2.5, 1.0))


def test_fetch_says_when_it_replaces_a_file(tmp_path, capsys):
    src, dest = tmp_path / "src.bin", tmp_path / "dest.bin"
    src.write_bytes(b"asteroid")
    dest.write_bytes(b"stale")
    fetch_mod.download(src.as_uri(), dest, fetch_mod.sha256(src))
    assert "replace   dest.bin" in capsys.readouterr().out and dest.read_bytes() == b"asteroid"


def test_an_assumed_label_without_an_assumed_albedo_needs_a_measured_diameter():
    mb = pd.DataFrame({"designation": ["1", "2", "3"], "tier": "assumed", "semi_major_axis_au": [2.3, 2.4, 2.9],
                       "albedo_assumed_for_diameter": [0.19, np.nan, 0.066],
                       "diameter_source": ["derived_h_orbit_albedo", "measured", "derived_h_orbit_albedo"]})
    steps = pipeline.assumed_albedo_steps(mb)          # the measured one has no assumed albedo: left out, by rule
    assert steps["size"].to_dict() == {0.066: 1, 0.19: 1}
    mb.loc[1, "diameter_source"] = "derived_h_orbit_albedo"
    with pytest.raises(ValueError, match=r"1 assumed-albedo labels have neither .* e.g. \['2'\]"):
        pipeline.assumed_albedo_steps(mb)

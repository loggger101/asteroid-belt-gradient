"""Fast tests on synthetic frames. No catalog, no network."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from beltgradient.catalog import norm_key, split_x_by_albedo, to_group
from beltgradient.completeness import complete_limit, diameter_limit
from beltgradient.config import H_EDGES, SNOW_LINE_AU, ZONE_NAMES
from beltgradient.families import (V1AU, extend_families, family_table, load_families, load_proper_elements,
                                   zappala_distance, zappala_embedding)
from beltgradient.fetch import download
from beltgradient.gradient import c_fraction_curve, collapsed, fit_logistic, size_distance_grid, zone_point
from beltgradient.orbits import ks_only, orbit_tests


def test_groups_by_first_letter():
    t = pd.Series(["Sq", "C", "Ch", "B", "F", "D", "Z", "K", "L", "Xc", "V", "T", None, "U"])
    assert to_group(t).tolist() == ["S-like", "C-like", "C-like", "C-like", "C-like", "D/P", "D/P", "K/L", "K/L", "X",
                                    "S-like", "D/P", "other", "other"]


def test_norm_key():
    assert norm_key(pd.Series(["2003 MF13", "k03m13f"])).tolist() == ["2003MF13", "K03M13F"]


def test_snow_line():
    assert SNOW_LINE_AU == pytest.approx(2.674, abs=1e-3)


def test_complete_diameter_matches_paper():
    # H_c = 10.5 and p = 0.04 give the paper's 52.8 km
    assert diameter_limit(10.5) == pytest.approx(52.8, abs=0.05)


def test_zappala_distance_is_the_hcm_metric():
    # hcluster.c: d^2 = 2 GM/(a1+a2) [1.25 (2 (a1-a2)/(a1+a2))^2 + 2 de^2 + 2 d(sin i)^2]; na(1 AU) = 29,784.7 m/s
    assert V1AU == pytest.approx(29_784.7, abs=0.1)
    a1, a2, de, ds = 2.50, 2.52, 0.004, 0.003
    am = (a1 + a2) / 2
    expected = V1AU / np.sqrt(am) * np.sqrt(1.25 * (2 * (a2 - a1) / (a1 + a2)) ** 2 + 2 * de ** 2 + 2 * ds ** 2)
    assert zappala_distance(a1, 0.1, 0.05, a2, 0.1 + de, 0.05 + ds) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("da, de, ds, tol", [(0.005, 0, 0, 0.01), (0, 0.005, 0, 0.01), (0, 0, 0.005, 0.01),
                                             (0.003, 0.002, 0.004, 0.03)])
def test_embedding_matches_the_metric_term_by_term(da, de, ds, tol):
    # each term on its own: the old ln(a) embedding passed a mixed case while halving the da term.
    # Mixed displacements pick up e*d(na) cross-terms (~1%), which is why candidates are re-ranked exactly.
    for a in (2.2, 2.7, 3.2):
        x = zappala_embedding(np.array([a, a + da]), np.array([0.1, 0.1 + de]), np.array([0.05, 0.05 + ds]))
        exact = zappala_distance(a, 0.1, 0.05, a + da, 0.1 + de, 0.05 + ds)
        assert np.linalg.norm(x[1] - x[0]) == pytest.approx(exact, rel=tol)


def test_extend_families_is_one_step_and_uses_the_nearest_members_cutoff():
    # at a = 2.5 AU, de = 0.001 is 26.6 m/s; at 3.0 AU, 24.3 m/s. Four listed members, fewer than the
    # K_CANDIDATES neighbours the search asks for.
    rows = [("f1", "F", 2.5, 0.100, 0.1), ("f2", "F", 2.5, 0.101, 0.1), ("f3", "F", 2.5, 0.102, 0.1),
            ("g1", "G", 3.0, 0.050, 0.05),
            ("near", None, 2.5, 0.1025, 0.1),     # 13 m/s from f3: joins F
            ("chain", None, 2.5, 0.1040, 0.1),    # 40 m/s from "near" but 53 m/s from f3: no chaining
            ("g_out", None, 3.0, 0.0508, 0.05),   # 19 m/s from g1: inside F's cutoff, outside G's
            ("far", None, 2.9, 0.2, 0.2), ("no_proper", None, np.nan, np.nan, np.nan)]
    df = pd.DataFrame(rows, columns=["key", "fam_id", "a_p", "e_p", "sini_p"])
    df["in_family"] = df.fam_id.notna()
    out, n = extend_families(df, pd.DataFrame({"fam_id": ["F", "G"], "cutoff": [50.0, 10.0]}))
    assert n == 1
    assert out.set_index("key").fam_id.dropna().to_dict() == {"f1": "F", "f2": "F", "f3": "F", "g1": "G", "near": "F"}
    assert out.in_family.sum() == 5


def test_load_families_handles_the_bundle_quirks(synthetic):
    fams, members = load_families(synthetic[0].nesvorny)
    f = fams.set_index("fam_id")
    assert not {"2015_007", "2015_503"} & set(f.index)                     # list lines with no parent number
    assert f.loc["2024_inner_0_2012pm61_fam3", ["fam_name", "cutoff"]].tolist() == ["2012 PM61", 15.0]
    assert f.loc["2024_middle_177_irma_fam3", ["fam_name", "cutoff"]].tolist() == ["Irma", 60.0]
    assert f.loc["2015_529", "n_listed"] == f.loc["2015_528", "n_listed"] == 15
    # an identical list counts once, under the lower id; a body in two families goes to the larger (Koronis)
    assert members.key.is_unique and not members.fam_id.isin(["2015_529", "2015_610"]).any()
    assert "2012PM61" in set(members.key)


def test_load_proper_elements(synthetic):
    pc = load_proper_elements(synthetic[0].nesvorny)
    assert list(pc.columns) == ["a_p", "e_p", "sini_p", "H_p", "key"] and pc.key.is_unique
    assert pc.a_p.between(1.8, 4).all() and pc.e_p.between(0, 0.6).all() and pc.sini_p.between(0, 0.5).all()
    assert "2012PM61" in set(pc.key)


def test_download_checks_the_sha256(tmp_path):
    src, dest = tmp_path / "src.bin", tmp_path / "dest.bin"
    src.write_bytes(b"asteroid")
    good = hashlib.sha256(b"asteroid").hexdigest()
    with pytest.raises(RuntimeError, match="sha256"):
        download(src.as_uri(), dest, "0" * 64)
    assert not dest.exists() and not (tmp_path / "dest.bin.part").exists()
    download(src.as_uri(), dest, good)
    assert dest.read_bytes() == b"asteroid"
    src.unlink()
    download(src.as_uri(), dest, good)               # already there and verified: nothing is fetched


def test_complete_limit_finds_first_incomplete_bin():
    idx = pd.IntervalIndex.from_breaks(H_EDGES)
    comp = pd.DataFrame(1.0, index=idx, columns=ZONE_NAMES)
    n = pd.DataFrame(10, index=idx, columns=ZONE_NAMES)
    n.iloc[:4] = 0                                   # empty bright bins are ignored
    k = int(np.searchsorted(H_EDGES, 10.5))
    comp.iloc[k:, 2] = 0.9                           # one zone drops below 95% at H = 10.5
    assert complete_limit(comp, n) == 10.5
    comp.iloc[:, :] = 1.0                            # complete everywhere: the faintest edge
    assert complete_limit(comp, n) == H_EDGES[-1]


def test_logistic_recovers_crossover():
    rng = np.random.default_rng(0)
    a = rng.uniform(2.1, 3.3, 20_000)
    b0, b1 = -1.2, 4.0                               # a50 = 2.7 + 1.2/4 = 3.0
    y = (rng.random(a.size) < 1 / (1 + np.exp(-(b0 + b1 * (a - 2.7))))).astype(float)
    b = fit_logistic(a, y, np.ones_like(a))
    assert 2.7 - b[0] / b[1] == pytest.approx(3.0, abs=0.02)
    assert 4.394 / b[1] == pytest.approx(4.394 / 4.0, rel=0.1)


def test_c_fraction_is_weighted_and_ignores_x():
    s = pd.DataFrame({"semi_major_axis_au": [2.2, 2.2, 2.2, 2.2, 3.0],
                      "group": ["C-like", "S-like", "S-like", "X", "C-like"],
                      "w": [3.0, 1.0, 1.0, 100.0, 1.0]})
    c = c_fraction_curve(s, np.random.default_rng(0), edges=np.array([2.1, 2.5, 3.3]), n_boot=10)
    assert c.f.tolist() == pytest.approx([0.6, 1.0])
    assert c.n.tolist() == [3, 1]


def _family_frame():
    # family A: 3 labelled C members + 1 S member -> C-like, purity 0.75
    # family B: 1 labelled member only -> falls back to the largest member's label
    return pd.DataFrame({
        "fam_id": ["A", "A", "A", "A", "B", "B", None],
        "in_family": [True] * 6 + [False],
        "tier": ["taxonomy"] * 4 + ["taxonomy", "assumed", "taxonomy"],
        "group": ["C-like", "C-like", "C-like", "S-like", "S-like", "C-like", "S-like"],
        "diameter_km": [30.0, 10.0, 10.0, 10.0, 20.0, 5.0, 60.0],
        "estimated_mass_kg": [1.0] * 7,
        "semi_major_axis_au": [3.1, 3.1, 3.1, 3.1, 2.3, 2.3, 2.6],
        "e_p": [0.1] * 7, "sini_p": [0.1] * 7,
        "name": ["a1", "a2", "a3", "a4", None, "b2", "bg"],
        "designation": ["1", "2", "3", "4", "2001 AB", "6", "7"],
        "key": ["1", "2", "3", "4", "2001AB", "6", "7"],
        "w_ipw": [1.0] * 7,
    })


def test_family_table_tie_falls_back_to_largest_member():
    # 2 S + 2 C labelled (plus an X that does not vote): no majority, so the largest member (C) decides;
    # if the largest were X, the family would be unclassifiable
    df = pd.DataFrame({"fam_id": ["T"] * 5, "in_family": True, "tier": "taxonomy",
                       "group": ["C-like", "S-like", "S-like", "C-like", "X"], "diameter_km": [90.0, 20, 15, 10, 5],
                       "estimated_mass_kg": 1.0, "semi_major_axis_au": 2.7, "e_p": 0.1, "sini_p": 0.1,
                       "name": list("abcde"), "designation": list("12345"), "key": list("12345")})
    fams = pd.DataFrame({"fam_id": ["T"], "fam_name": ["Tie"], "cutoff": [50.0]})
    ft = family_table(df, fams)
    assert ft.loc["T", "fam_group"] == "C-like" and ft.loc["T", "purity"] == 0.5 and ft.loc["T", "n_labelled"] == 4
    df.loc[0, "group"] = "X"
    df.loc[4, "group"] = "C-like"
    assert pd.isna(family_table(df, fams).loc["T", "fam_group"])


def test_family_table_majority_and_fallback():
    fams = pd.DataFrame({"fam_id": ["A", "B"], "fam_name": ["Alpha", "Beta"], "cutoff": [50.0, 60.0]})
    ft = family_table(_family_frame(), fams)
    assert ft.loc["A", "fam_group"] == "C-like" and ft.loc["A", "purity"] == 0.75
    assert ft.loc["B", "fam_group"] == "S-like"
    assert ft.loc["A", "largest"] == "a1" and ft.loc["B", "largest"] == "2001 AB"   # unnamed: its designation
    assert ft.loc["A", "D_equiv_km"] == pytest.approx((30 ** 3 + 3 * 10 ** 3) ** (1 / 3))
    col = collapsed(_family_frame(), ft)
    assert len(col) == 3 and col.is_family_rep.sum() == 2   # background body + one row per family


def test_orbit_tests_bonferroni():
    rng = np.random.default_rng(1)
    rows = []
    for z, a in zip(ZONE_NAMES, [2.3, 2.7, 2.9, 3.1]):
        for g in ["S-like", "C-like"]:
            for _ in range(40):
                rows.append(dict(zone=z, group=g, e_p=rng.normal(0.1, 0.03), sini_p=rng.normal(0.1, 0.03)))
    out = orbit_tests(pd.DataFrame(rows), np.random.default_rng(2))
    assert len(out) == 8
    assert not out.bonferroni_sig.any()
    ks = ks_only(pd.DataFrame(rows))                  # same KS, no bootstrap
    assert ks.KS_p.tolist() == out.KS_p.tolist()


def test_split_x_by_albedo():
    # P below 0.10 (joins D/P), M to 0.30, E above; Xe, Xk, Tholen M and unlabelled tiers untouched
    st = ["X", "X", "X", "Xc", "Xt", "Xe", "Xk", "M", "X", "X", "C"]
    alb = [0.05, 0.10, 0.31, 0.099, 0.2, 0.05, 0.05, 0.05, np.nan, 0.05, 0.05]
    tier = ["taxonomy"] * 9 + ["albedo_proxy", "taxonomy"]
    mb = pd.DataFrame({"spectral_type": st, "albedo": alb, "tier": tier})
    mb["group"] = to_group(mb.spectral_type)
    out = split_x_by_albedo(mb)
    assert out.x_albedo_class.tolist() == ["P", "M", "E", "P", "M", None, None, None, None, None, None]
    assert out.group.tolist() == ["D/P", "X", "X", "D/P", "X", "X", "X", "X", "X", "X", "C-like"]
    assert mb.group.tolist()[0] == "X"                      # the input is not modified


def test_zone_point_is_the_weighted_zone_fraction():
    s = pd.DataFrame({"semi_major_axis_au": [2.2, 2.3, 2.4, 2.6, 3.0, 3.1],
                      "group": ["S-like", "C-like", "X", "D/P", "C-like", "S-like"], "w": [1.0, 3.0, 5.0, 1.0, 2.0, 2.0]})
    narrow, broad = zone_point(s, "narrow"), zone_point(s, "broad")
    assert narrow.loc["inner", "f"] == pytest.approx(0.75) and narrow.loc["inner", "n"] == 2   # X ignored
    assert np.isnan(narrow.loc["middle", "f"]) and broad.loc["middle", "f"] == pytest.approx(1.0)
    assert narrow.loc["outer", "f"] == pytest.approx(0.5) and np.isnan(narrow.loc["pristine", "f"])


def test_size_distance_grid():
    # five S/C bodies in one cell (2.3-2.4 AU, 10-20 km), weighted; two in another; X ignored
    s = pd.DataFrame({"semi_major_axis_au": [2.35] * 5 + [3.15, 3.15, 3.15],
                      "diameter_km": [12, 15, 18, 19, 11, 150, 200, 120],
                      "group": ["C-like", "S-like", "S-like", "C-like", "X", "C-like", "S-like", "C-like"],
                      "w": [2.0, 1.0, 1.0, 2.0, 9.0, 1.0, 1.0, 1.0]})
    a, d, f, n = size_distance_grid(s)
    assert len(a) == 13 and d[-1] == np.inf and f.shape == n.shape == (4, 12)
    assert n[0, 2] == 4 and np.isnan(f[0, 2])            # 4 S/C bodies < MIN_CELL: blank
    s5 = pd.concat([s, s.iloc[[0]]])                       # a fifth S/C body fills the cell
    _, _, f5, n5 = size_distance_grid(s5)
    assert n5[0, 2] == 5 and f5[0, 2] == pytest.approx(6 / 8)    # C weight 2+2+2 of 2+1+1+2+2
    assert n5[3, 10] == 3 and np.isnan(f5[3, 10])          # >= 100 km at 3.1-3.2 AU: 3 bodies, blank


def test_bhac15_track_holds_the_quoted_rows():
    from beltgradient.bhac15 import TRACK_1MSUN
    from beltgradient.config import BHAC15_QUOTED_LOG_T
    track = dict(TRACK_1MSUN)
    assert len(TRACK_1MSUN) == 198 and all(a < b for (a, _), (b, _) in zip(TRACK_1MSUN, TRACK_1MSUN[1:]))
    assert [track[t] for t in BHAC15_QUOTED_LOG_T] == [0.285, -0.198]   # log L/L_sun at 1.0 and 5.1 Myr

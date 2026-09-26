"""Fast tests on synthetic frames. No catalog, no network."""
import numpy as np
import pandas as pd
import pytest

from beltgradient.catalog import norm_key, to_group
from beltgradient.completeness import complete_limit, diameter_limit
from beltgradient.config import H_EDGES, SNOW_LINE_AU, ZONE_NAMES
from beltgradient.families import V1AU, family_table, zappala_distance, zappala_embedding
from beltgradient.gradient import c_fraction_curve, collapsed, fit_logistic
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


def test_complete_limit_finds_first_incomplete_bin():
    idx = pd.IntervalIndex.from_breaks(H_EDGES)
    comp = pd.DataFrame(1.0, index=idx, columns=ZONE_NAMES)
    n = pd.DataFrame(10, index=idx, columns=ZONE_NAMES)
    n.iloc[:4] = 0                                   # empty bright bins are ignored
    k = int(np.searchsorted(H_EDGES, 10.5))
    comp.iloc[k:, 2] = 0.9                           # one zone drops below 95% at H = 10.5
    assert complete_limit(comp, n) == 10.5


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

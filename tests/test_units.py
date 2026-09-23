"""Fast tests on synthetic frames. No catalog, no network."""
import numpy as np
import pandas as pd
import pytest

from beltgradient.catalog import norm_key, to_group
from beltgradient.completeness import complete_limit, diameter_limit
from beltgradient.config import H_EDGES, SNOW_LINE_AU, ZONE_NAMES
from beltgradient.families import family_table, zappala_coords
from beltgradient.gradient import c_fraction_curve, collapsed, fit_logistic
from beltgradient.orbits import orbit_tests


def test_groups_by_first_letter():
    t = pd.Series(["Sq", "C", "Ch", "B", "D", "K", "L", "Xc", "V", "T", None, "Z"])
    assert to_group(t).tolist() == ["S-like", "C-like", "C-like", "C-like", "D/P", "K/L", "K/L", "X", "S-like", "D/P", "other", "other"]


def test_norm_key():
    assert norm_key(pd.Series(["2003 MF13", "k03m13f"])).tolist() == ["2003MF13", "K03M13F"]


def test_snow_line():
    assert SNOW_LINE_AU == pytest.approx(2.674, abs=1e-3)


def test_complete_diameter_matches_paper():
    # H_c = 10.5 and p = 0.04 give the paper's 52.8 km
    assert diameter_limit(10.5) == pytest.approx(52.8, abs=0.05)


@pytest.mark.parametrize("da, tol", [(0.0, 1e-9), (0.001, 0.02)])
def test_zappala_metric(da, tol):
    # exact at equal a; each point carries its own na, so within ~1% for small da
    a, e, s = 2.5, 0.1, 0.05
    de, ds = 0.002, 0.003
    x = zappala_coords(np.array([a, a + da]), np.array([e, e + de]), np.array([s, s + ds]))
    d = np.linalg.norm(x[1] - x[0])
    na = 29_780.0 / np.sqrt(a)
    expected = na * np.sqrt(1.25 * (da / a) ** 2 + 2 * de ** 2 + 2 * ds ** 2)
    assert d == pytest.approx(expected, rel=tol)


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
        "name": ["a1", "a2", "a3", "a4", "b1", "b2", "bg"],
        "w_ipw": [1.0] * 7,
    })


def test_family_table_majority_and_fallback():
    fams = pd.DataFrame({"fam_id": ["A", "B"], "fam_name": ["Alpha", "Beta"], "cutoff": [50.0, 60.0]})
    ft = family_table(_family_frame(), fams)
    assert ft.loc["A", "fam_group"] == "C-like" and ft.loc["A", "purity"] == 0.75
    assert ft.loc["B", "fam_group"] == "S-like"
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

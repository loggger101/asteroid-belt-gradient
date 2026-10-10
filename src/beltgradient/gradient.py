"""Carbonaceous fraction versus heliocentric distance, and the S/C crossover.

Samples (each restricted to taxonomy-tier labels):

================================  ===========================  =====================
sample                            families                     size / weighting
================================  ===========================  =====================
raw                               kept                         all labelled, unweighted
collapsed, IPW, D>=10 km          each family -> one body      size-limited, IPW
collapsed, D>=D_c                 each family -> one body      size-complete
background, D>=D_c                removed                      size-complete
================================  ===========================  =====================

Uncertainties are Poisson-bootstrap 16–84% intervals (Efron 1979; Hanley & MacGibbon 2006). All
randomness comes from the ``rng`` argument, so a run is reproducible given the seed and the call order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from .config import A_MAX, A_MIN, BIN_W, D_IPW, N_BOOT, ZONE_NAMES, ZONES

# narrow = C/(S+C); broad = (C+D/P+K/L)/(S+C+D/P+K/L)
NUM = {"narrow": ["C-like"], "broad": ["C-like", "D/P", "K/L"]}
DEN = {"narrow": ["S-like", "C-like"], "broad": ["S-like", "C-like", "D/P", "K/L"]}


def bin_index(values, edges, what: str) -> np.ndarray:
    """The bin of each value in ``edges`` (a value on the last edge goes in the last bin); a value outside raises."""
    v = np.asarray(values, dtype=float)
    idx = np.digitize(v, edges) - 1
    idx[v == edges[-1]] = len(edges) - 2
    bad = np.isnan(v) | (idx < 0) | (idx > len(edges) - 2)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} {what} outside [{edges[0]:g}, {edges[-1]:g}]: {v[bad][:5].tolist()}")
    return idx


def collapsed(df: pd.DataFrame, ft: pd.DataFrame) -> pd.DataFrame:
    """Background bodies plus one synthetic row per classifiable family."""
    bg = df[~df.in_family]
    fr = ft[ft.fam_group.notna()]
    if fr.a_rep.isna().any():
        raise ValueError(f"classified families without a representative orbit: {fr.index[fr.a_rep.isna()].tolist()}")
    reps = pd.DataFrame({"semi_major_axis_au": fr.a_rep.values, "group": fr.fam_group.values,
                         "diameter_km": fr.D_equiv_km.values, "estimated_mass_kg": fr.mass_kg.values,
                         "e_p": fr.e_rep.values, "sini_p": fr.sini_rep.values,
                         "tier": "taxonomy", "w_ipw": 1.0, "in_family": True, "is_family_rep": True,
                         "name": fr.fam_name.values})
    out = pd.concat([bg.assign(is_family_rep=False), reps], ignore_index=True)
    out["zone"] = pd.cut(out.semi_major_axis_au, ZONES, labels=ZONE_NAMES)
    return out


def sample_keys(d_c: float) -> list[str]:
    return ["raw", f"collapsed, IPW, D≥{D_IPW:.0f} km", f"collapsed, D≥{d_c:.0f} km", f"background, D≥{d_c:.0f} km"]


def build_samples(df: pd.DataFrame, ft: pd.DataFrame, d_c: float) -> dict[str, pd.DataFrame]:
    """The four samples in the table above, each with a weight column ``w``."""
    t = df[df.tier.eq("taxonomy")]
    col = collapsed(df, ft)
    col = col[col.tier.eq("taxonomy")]
    k_raw, k_ipw, k_col, k_bg = sample_keys(d_c)
    return {
        k_raw: t.assign(w=1.0),
        k_ipw: col[(col.w_ipw > 0) & (col.diameter_km >= D_IPW)].assign(w=col.w_ipw),
        k_col: col[col.diameter_km >= d_c].assign(w=1.0),
        k_bg:  t[~t.in_family & (t.diameter_km >= d_c)].assign(w=1.0),
    }


def c_fraction_curve(s: pd.DataFrame, rng: np.random.Generator, which: str = "narrow",
                     edges=None, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Weighted C-fraction per semimajor-axis bin with a Poisson-bootstrap 16–84% band."""
    edges = np.arange(A_MIN, A_MAX + 1e-9, BIN_W) if edges is None else edges
    s = s[s.group.isin(DEN[which])]
    b = bin_index(s.semi_major_axis_au, edges, "semimajor axes")
    isC = s.group.isin(NUM[which]).values
    w = s.w.values
    nb = len(edges) - 1

    def frac(ww):
        num = np.bincount(b, ww * isC, nb); den = np.bincount(b, ww, nb)
        return np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)

    f = frac(w)
    boots = np.array([frac(w * rng.poisson(1.0, len(w))) for _ in range(n_boot)])
    n = np.bincount(b, None, nb)
    # an empty bin has no fraction and no interval; in an occupied bin, a replicate that drew no body is skipped
    lo, hi = np.full(nb, np.nan), np.full(nb, np.nan)
    lo[n > 0], hi[n > 0] = np.nanpercentile(boots[:, n > 0], [16, 84], axis=0)
    return pd.DataFrame({"a": (edges[:-1] + edges[1:]) / 2, "f": f, "lo": lo, "hi": hi, "n": n})


def zone_point(s: pd.DataFrame, which: str = "narrow") -> pd.DataFrame:
    """The zone fractions of :func:`zone_table` without the bootstrap: ``f`` and ``n`` per zone."""
    edges = np.array(ZONES)
    s = s[s.group.isin(DEN[which])]
    b = bin_index(s.semi_major_axis_au, edges, "semimajor axes")
    nb = len(edges) - 1
    num = np.bincount(b, s.w.values * s.group.isin(NUM[which]).values, nb)
    den = np.bincount(b, s.w.values, nb)
    return pd.DataFrame({"f": np.where(den > 0, num / np.maximum(den, 1e-300), np.nan), "n": np.bincount(b, None, nb)},
                        index=ZONE_NAMES)


SIZE_EDGES_KM = [10, 20, 50, 100, np.inf]
MIN_CELL = 5


def size_distance_grid(s: pd.DataFrame):
    """Weighted C/(S+C) in 0.1 AU x ``SIZE_EDGES_KM`` cells: (a_edges, d_edges, fraction, n).

    ``n`` counts S- and C-type bodies per cell; the fraction is NaN where ``n < MIN_CELL``. Rows are the size bins
    from small to large, columns the a bins.
    """
    s = s[s.group.isin(["S-like", "C-like"])]
    a_edges = np.round(np.arange(A_MIN, A_MAX + 1e-9, 0.1), 2)
    d_edges = np.array(SIZE_EDGES_KM, dtype=float)
    ai = bin_index(s.semi_major_axis_au, a_edges, "semimajor axes")
    di = bin_index(s.diameter_km, d_edges, "diameters (km)")
    shape = (len(d_edges) - 1, len(a_edges) - 1)
    num, den, n = np.zeros(shape), np.zeros(shape), np.zeros(shape, dtype=int)
    np.add.at(num, (di, ai), s.w.values * s.group.eq("C-like").values)
    np.add.at(den, (di, ai), s.w.values)
    np.add.at(n, (di, ai), 1)
    f = np.where(n >= MIN_CELL, num / np.maximum(den, 1e-300), np.nan)
    return a_edges, d_edges, f, n


def zone_table(samples: dict[str, pd.DataFrame], rng: np.random.Generator, which: str = "narrow") -> pd.DataFrame:
    """C-fraction per zone for every sample, formatted "f [lo–hi] (n=N)"."""
    rows = {}
    for k, s in samples.items():
        c = c_fraction_curve(s, rng, which, edges=np.array(ZONES), n_boot=N_BOOT)
        rows[k] = [f"{f:.2f} [{l:.2f}–{h:.2f}] (n={n})" for f, l, h, n in zip(c.f, c.lo, c.hi, c.n)]
    return pd.DataFrame(rows, index=ZONE_NAMES).T


# ── S/C crossover ───────────────────────────────────────────────────────────
# P(C | a) = 1 / (1 + exp(-(b0 + b1 (a - 2.7)))), weighted maximum likelihood.
# a50 = 2.7 - b0/b1; 10->90% width = 2 ln 9 / b1 = 4.394 / b1.

def fit_logistic(a, y, w):
    x = a - 2.7

    def nll(b):
        z = b[0] + b[1] * x
        return -np.sum(w * (y * z - np.logaddexp(0, z)))

    res = optimize.minimize(nll, [0.0, 3.0], method="BFGS")
    # BFGS often ends with "precision loss" at the optimum, so its success flag is not the test: the gradient,
    # relative to the objective, is (below 1e-7 in every fit of the real run)
    rel = float(np.max(np.abs(res.jac)) / max(1.0, abs(res.fun)))
    if not (np.isfinite(res.x).all() and rel < 1e-5):
        raise RuntimeError(f"logistic fit did not converge: relative gradient {rel:.2e}, {res.message}")
    return res.x


def crossover(s: pd.DataFrame, rng: np.random.Generator, n_boot: int = 200) -> dict:
    s = s[s.group.isin(["S-like", "C-like"])]
    if len(s) < 20 or s.group.nunique() < 2:
        raise ValueError(f"a crossover needs at least 20 S- and C-type bodies of both kinds; got {len(s)} "
                         f"({s.group.value_counts().to_dict()})")
    a, y, w = s.semi_major_axis_au.values, s.group.eq("C-like").values.astype(float), s.w.values
    b = fit_logistic(a, y, w)
    bs = np.array([fit_logistic(a, y, w * rng.poisson(1.0, len(w))) for _ in range(n_boot)])
    a50 = 2.7 - bs[:, 0] / bs[:, 1]; width = 4.394 / bs[:, 1]
    return dict(a50=2.7 - b[0] / b[1], a50_lo=np.percentile(a50, 16), a50_hi=np.percentile(a50, 84),
                width_10_90=4.394 / b[1], width_lo=np.percentile(width, 16), width_hi=np.percentile(width, 84), n=len(s))


def crossover_table(samples: dict[str, pd.DataFrame], rng: np.random.Generator, n_boot: int = 200) -> pd.DataFrame:
    return pd.DataFrame({k: crossover(s, rng, n_boot=n_boot) for k, s in samples.items()}).T


# ── inner belt by size, and by mass ──────────────────────────────────────────

def inner_belt_by_size(mb: pd.DataFrame) -> pd.DataFrame:
    """Inner-belt C-fraction by number and mass per size bin, families included (as DeMeo & Carry 2014).

    The 5–20 km bin is not debiased; its C-fraction is a lower limit.
    """
    inner = mb[mb.tier.eq("taxonomy") & mb.zone.eq("inner") & mb.group.isin(["S-like", "C-like"])]
    rows = []
    for lo, hi in [(100, np.inf), (50, 100), (20, 50), (5, 20)]:
        s = inner[(inner.diameter_km >= lo) & (inner.diameter_km < hi)]
        if s.empty:
            raise ValueError(f"no inner-belt S- or C-type body with {lo} <= D < {hi} km")
        m = s.groupby("group").estimated_mass_kg.sum().reindex(["S-like", "C-like"], fill_value=0.0)
        rows.append(dict(size=f">= {lo} km" if np.isinf(hi) else f"{lo}-{hi} km",
                         n_S=int((s.group == "S-like").sum()), n_C=int((s.group == "C-like").sum()),
                         C_frac_number=(s.group == "C-like").mean(), C_frac_mass=m["C-like"] / m.sum()))
    return pd.DataFrame(rows)


def mass_by_zone(d: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Summed ``estimated_mass_kg`` per (zone, group)."""
    return d.pivot_table(index="zone", columns="group", values="estimated_mass_kg", aggfunc="sum", observed=False).reindex(columns=order).fillna(0)

"""Orbital excitation: do S- and C-types in the same zone have different proper e and sin i?

Implanted C-types need not share the excitation of locally formed S-types; formation in place
predicts no class dependence within a zone. Background bodies only (family members share
near-identical proper elements), taxonomy tier, D >= D_ORBIT. Two-sample KS per zone and element,
Bonferroni over all tests. Scipy's Anderson–Darling p-values are capped to [0.001, 0.25], so KS only.

Resonances and Yarkovsky drift reshape e and i after formation: a difference is suggestive, and
its absence does not rule out implantation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import D_ORBIT, ZONE_NAMES

ELEMENTS = [("e_p", "proper e"), ("sini_p", "proper sin i")]


def orbit_sample(mb: pd.DataFrame) -> pd.DataFrame:
    return mb[mb.tier.eq("taxonomy") & ~mb.in_family & (mb.diameter_km >= D_ORBIT) & mb.e_p.notna()
              & mb.group.isin(["S-like", "C-like"])]


def med_diff_ci(x, y, rng: np.random.Generator, n: int = 1000):
    """Median(x) - median(y) with a 95% bootstrap interval."""
    d = [np.median(rng.choice(x, len(x))) - np.median(rng.choice(y, len(y))) for _ in range(n)]
    return np.median(x) - np.median(y), *np.percentile(d, [2.5, 97.5])


def split_by_class(orb: pd.DataFrame, zone: str, col: str) -> tuple[np.ndarray, np.ndarray]:
    oz = orb[orb.zone == zone]
    return oz.loc[oz.group == "S-like", col].values, oz.loc[oz.group == "C-like", col].values


def orbit_tests(orb: pd.DataFrame, rng: np.random.Generator, alpha: float = 0.05) -> pd.DataFrame:
    """KS test and C-minus-S median difference per zone and element (needs >= 10 of each class)."""
    rows = []
    for z in ZONE_NAMES:
        for col, lab in ELEMENTS:
            S, C = split_by_class(orb, z, col)
            if len(S) >= 10 and len(C) >= 10:
                ks = stats.ks_2samp(S, C)
                md_, lo, hi = med_diff_ci(C, S, rng)
                rows.append(dict(zone=z, element=lab, n_S=len(S), n_C=len(C), median_S=np.median(S), median_C=np.median(C),
                                 dmedian_C_minus_S=md_, ci95_lo=lo, ci95_hi=hi, KS_D=ks.statistic, KS_p=ks.pvalue))
    out = pd.DataFrame(rows)
    out["bonferroni_sig"] = out.KS_p < alpha / len(out)   # 8 tests -> p < 0.00625
    return out

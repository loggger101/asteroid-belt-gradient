"""Label completeness versus absolute magnitude, and the two debiasing schemes.

At fixed H a dark C-type is larger than a bright S-type, so a sample cut in H over-represents
S-types among small bodies. Two remedies:

1. **Size-complete sample.** H_c is the faintest H bin where >= 95% of bodies in every zone are
   labelled. Since H = 5 log10(1329 / (D sqrt(p))) (Pravec & Harris 2007), every body with
   D >= D_c = 1329 p_dark^-1/2 10^(-H_c/5) is then labelled whatever its albedo.
2. **Inverse-completeness weighting (IPW)** on a size-limited sample: weight 1/c(H, zone)
   (Horvitz & Thompson 1952), keep D >= 10 km. Assumes that at fixed H and zone, labelling does
   not depend on class.

Both lean on catalogue H, which for small asteroids was found to run 0.4–0.5 mag too bright near
H = 14 (Pravec et al. 2012): that shifts the H bins and the H-derived diameters alike.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import C_MIN_IPW, COMPLETE_FRAC, H_EDGES, H_IPW_MAX, P_DARKEST


def add_h_bins(mb: pd.DataFrame) -> pd.DataFrame:
    mb["H_bin"] = pd.cut(mb.absolute_magnitude_h, H_EDGES)
    return mb


def completeness(mb: pd.DataFrame, labelled: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fraction labelled and body count per (H bin, zone); every category kept so codes index it directly."""
    g = mb.assign(lab=labelled).groupby(["H_bin", "zone"], observed=False).lab
    return g.mean().unstack(), g.size().unstack()


def complete_limit(comp: pd.DataFrame, n: pd.DataFrame) -> float:
    """Faintest H edge such that every populated (H, zone) cell brighter than it is >= COMPLETE_FRAC labelled."""
    ok = ((comp >= COMPLETE_FRAC) | (n == 0)).all(axis=1) & (n.sum(axis=1) > 0)
    populated = n.sum(axis=1).values > 0
    bad = np.where(populated & ~ok.values)[0]
    return float(H_EDGES[bad[0]])


def diameter_limit(h_c: float, p_dark: float = P_DARKEST) -> float:
    """Smallest diameter (km) guaranteed brighter than H_c for any albedo >= p_dark."""
    return 1329 / np.sqrt(p_dark) * 10 ** (-h_c / 5)


def add_ipw_weights(mb: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    """Per-body completeness ``c`` and IPW weight ``w_ipw`` (0 for bodies that cannot enter the IPW sample)."""
    hi_, zi_ = mb.H_bin.cat.codes.values, mb.zone.cat.codes.values
    valid = (hi_ >= 0) & (zi_ >= 0)
    mb["c"] = np.nan
    mb.loc[valid, "c"] = comp.values[hi_[valid], zi_[valid]]
    mb["w_ipw"] = np.where(mb.tier.eq("taxonomy") & (mb.absolute_magnitude_h < H_IPW_MAX) & (mb.c >= C_MIN_IPW), 1 / mb.c, 0.0)
    return mb

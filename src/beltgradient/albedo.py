"""Label-free check: the dark fraction (measured p_V < 0.10) versus semimajor axis.

Uses only measured geometric albedo, no taxonomy. Almost every main-belt body with a measured
albedo also has a NEOWISE fit in the catalog (Mainzer et al. 2019 data set; Masiero et al. 2011 and
the later NEOWISE papers); the value used is the catalog's precedence pick, mostly JPL SBDB's.
Completeness of measured albedos is computed the same way as for taxonomy labels.

The dark/bright split at p_V = 0.10 follows the albedos of classified asteroids: C-, B-, D- and
T-types are all dark and the S complex is bright, the two overlapping at small sizes (Mainzer et al.
2011, ApJ 741, 90).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import N_BOOT
from .gradient import bin_index

DARK_ALBEDO = 0.10


def with_measured_albedo(mb: pd.DataFrame) -> pd.DataFrame:
    alb = mb[mb.albedo.notna()].copy()
    alb["dark"] = alb.albedo < DARK_ALBEDO
    return alb


def dark_curve(d: pd.DataFrame, edges, rng: np.random.Generator, n_boot: int = N_BOOT):
    """Return (bin centres, dark fraction, 16%, 84%, counts); a semimajor axis outside ``edges`` raises.

    As in :func:`beltgradient.gradient.c_fraction_curve`, an empty bin has no fraction and no interval, and in an
    occupied bin a replicate that drew no body is skipped (it has no fraction, not a fraction of 0).
    """
    b = bin_index(d.semi_major_axis_au, edges, "semimajor axes"); nb = len(edges) - 1
    y = d.dark.values.astype(float)

    def f(w):
        den = np.bincount(b, w, nb)
        return np.where(den > 0, np.bincount(b, w * y, nb) / np.maximum(den, 1e-300), np.nan)

    boots = np.array([f(rng.poisson(1.0, len(y)).astype(float)) for _ in range(n_boot)])
    n = np.bincount(b, None, nb)
    lo, hi = np.full(nb, np.nan), np.full(nb, np.nan)
    lo[n > 0], hi[n > 0] = np.nanpercentile(boots[:, n > 0], [16, 84], axis=0)
    return (edges[:-1] + edges[1:]) / 2, f(np.ones(len(y))), lo, hi, n

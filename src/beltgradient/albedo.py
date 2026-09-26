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

DARK_ALBEDO = 0.10


def with_measured_albedo(mb: pd.DataFrame) -> pd.DataFrame:
    alb = mb[mb.albedo.notna()].copy()
    alb["dark"] = alb.albedo < DARK_ALBEDO
    return alb


def dark_curve(d: pd.DataFrame, edges, rng: np.random.Generator, n_boot: int = N_BOOT):
    """Return (bin centres, dark fraction, 16%, 84%, counts)."""
    b = np.clip(np.digitize(d.semi_major_axis_au, edges) - 1, 0, len(edges) - 2); nb = len(edges) - 1
    y = d.dark.values.astype(float)
    f = lambda w: np.bincount(b, w * y, nb) / np.maximum(np.bincount(b, w, nb), 1e-300)
    boots = np.array([f(rng.poisson(1.0, len(y)).astype(float)) for _ in range(n_boot)])
    return (edges[:-1] + edges[1:]) / 2, f(np.ones(len(y))), *np.percentile(boots, [16, 84], axis=0), np.bincount(b, None, nb)

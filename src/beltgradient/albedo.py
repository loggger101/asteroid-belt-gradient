"""Label-free check: the dark fraction (measured p_V < 0.10) versus semimajor axis.

Uses only measured geometric albedo (NEOWISE/JPL), no taxonomy. Completeness of measured
albedos is computed the same way as for taxonomy labels.
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

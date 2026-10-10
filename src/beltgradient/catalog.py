"""Catalog loading, main-belt selection, and compositional groups.

The catalog assigns a spectral type to almost every body, by different routes
(``spectral_type_source``). In the main belt three occur:

=================  ===========================================================  =====================
tier               route                                                        used?
=================  ===========================================================  =====================
``source``         published taxonomy (SsODNet; mostly SkyMapper/SDSS colours)  yes, primary
``albedo``         *measured* albedo thresholded: p<0.10 C, else S                no (the label-free
                                                                                check reads measured
                                                                                albedos directly)
``albedo_assumed`` albedo *assumed from semimajor axis*, then thresholded         never: circular
=================  ===========================================================  =====================

(``tholen`` and ``orbit``, the catalog's other two routes, label no main-belt body in the
release analysed here; ``orbit`` is D for untyped bodies from the Trojans outward.)

Most ``source`` labels are multi-filter colours from SkyMapper (Sergeyev et al. 2022) and SDSS
(Carvano et al. 2010; DeMeo & Carry 2013; Sergeyev & Carry 2021), with some near-infrared colours
(Popescu et al. 2018). Where JPL has a spectral class (Bus & Binzel 2002), the catalog uses it; SsODNet
otherwise gives Mahlke et al. (2022) classes for bodies with spectra. SsODNet asks that these source
papers be cited alongside Berthier et al. (2023).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import A_MAX, A_MIN, CATALOG_CONTRACT, CATALOG_DATE, Q_MIN, X_SPLIT_ALBEDO, ZONE_NAMES, ZONES

COLS = ["designation", "name", "spectral_type", "spectral_type_source", "semi_major_axis_au", "eccentricity",
        "inclination_deg", "perihelion_au", "diameter_km", "diameter_source", "albedo", "absolute_magnitude_h",
        "estimated_mass_kg", "mass_measured", "albedo_assumed_for_diameter", "catalog_date", "pipeline_version"]

# First letter of the Bus–DeMeo (DeMeo et al. 2009) / Mahlke / Tholen class. K/L are anhydrous but
# CV/CO-linked (Sunshine et al. 2008; Clark et al. 2009), so kept apart; X is excluded from the S/C
# fraction (without albedo it spans enstatite, metal and P-like bodies; DeMeo & Carry 2013; split_x_by_albedo
# splits it by measured albedo as a robustness test). Z, Mahlke
# et al.'s (2022) class of extremely red objects, is counted with D/P; F, one of Tholen's (1984) minor
# classes, as C-like. U (unusual: no class fits) is "other", left out of every fraction. Any other first
# letter is an error: a new class must be assigned a group here, not dropped.
GROUP_OF_LETTER = {**dict.fromkeys(list("SQAVRO"), "S-like"), **dict.fromkeys(list("CBF"), "C-like"),
                   **dict.fromkeys(list("DPTZ"), "D/P"), **dict.fromkeys(list("KL"), "K/L"),
                   **dict.fromkeys(list("XME"), "X"), "U": "other"}
# every label route of the catalog; tholen and orbit label no main-belt body in this release (see above)
TIER_OF_SOURCE = {"source": "taxonomy", "albedo": "albedo_proxy", "albedo_assumed": "assumed", "tholen": "tholen",
                  "orbit": "orbit"}
# Bus–DeMeo Xe and Xk already say which X they are (enstatite-like, metal-like); Tholen's M and E labels too
X_SPLIT_EXEMPT = ("Xe", "Xk")


def norm_key(s: pd.Series) -> pd.Series:
    """Designation key shared by the catalog and the Nesvorný files."""
    return s.astype(str).str.replace(" ", "", regex=False).str.upper()


def to_group(t: pd.Series) -> pd.Series:
    """Compositional group from the first letter of the class; a missing class or an unknown letter raises."""
    g = t.astype("string").str[0].map(GROUP_OF_LETTER)
    bad = g.isna()
    if bad.any():
        raise ValueError(f"{int(bad.sum())} bodies have no spectral type or one whose first letter has no group: "
                         f"{sorted(t[bad].astype(str).unique())[:10]}")
    return g.astype(object)


def load_catalog(path: Path) -> pd.DataFrame:
    """The columns this analysis uses, read from an AsteroidCatalog release parquet."""
    cat = pd.read_parquet(path, columns=COLS)
    found = {"catalog_date": set(cat.catalog_date.astype(str)), "pipeline_version": set(cat.pipeline_version.astype(str))}
    if found != {"catalog_date": {CATALOG_DATE}, "pipeline_version": {CATALOG_CONTRACT}}:
        raise ValueError(f"{path.name} is not the expected release build (catalog_date {CATALOG_DATE}, "
                         f"data contract {CATALOG_CONTRACT}): it holds {found}")
    cat["key"] = norm_key(cat["designation"])
    if not cat.key.is_unique:
        raise ValueError(f"{int(cat.key.duplicated().sum())} designations collide after normalisation, "
                         f"e.g. {cat.key[cat.key.duplicated()].head(5).tolist()}")
    return cat


def split_x_by_albedo(mb: pd.DataFrame) -> pd.DataFrame:
    """A copy in which X-types with a measured albedo get the Tholen class it indicates (``x_albedo_class``).

    X, Xc and the other X sub-classes (not Xe or Xk) with a taxonomy label and a measured albedo are P below
    ``X_SPLIT_ALBEDO[0]``, M up to ``X_SPLIT_ALBEDO[1]`` and E above. P-types join the D/P group; M- and E-like
    bodies stay X, which the fractions exclude. X-types without a measured albedo are unchanged.
    """
    out = mb.copy()
    st = out.spectral_type.astype(str)
    split = out.tier.eq("taxonomy") & st.str.startswith("X") & ~st.isin(X_SPLIT_EXEMPT) & out.albedo.notna()
    lo, hi = X_SPLIT_ALBEDO
    cls = np.select([out.albedo < lo, out.albedo < hi], ["P", "M"], "E")
    out["x_albedo_class"] = pd.Series(np.where(split, cls, None), index=out.index, dtype="object")
    out.loc[split & (out.albedo < lo), "group"] = "D/P"
    return out


def main_belt(cat: pd.DataFrame) -> pd.DataFrame:
    """2.1 < a < 3.3 AU, q > 1.3 AU, with zone, compositional group and label tier."""
    a = cat.semi_major_axis_au
    mb = cat[(a > A_MIN) & (a < A_MAX) & (cat.perihelion_au > Q_MIN)].copy()
    mb["zone"] = pd.cut(mb.semi_major_axis_au, ZONES, labels=ZONE_NAMES)
    mb["group"] = to_group(mb.spectral_type)
    mb["tier"] = mb.spectral_type_source.map(TIER_OF_SOURCE)
    if mb.tier.isna().any():
        raise ValueError(f"unknown spectral_type_source: {sorted(mb.spectral_type_source[mb.tier.isna()].astype(str).unique())}")
    return mb

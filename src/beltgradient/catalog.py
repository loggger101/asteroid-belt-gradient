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

import pandas as pd

from .config import A_MAX, A_MIN, Q_MIN, ZONE_NAMES, ZONES

COLS = ["designation", "name", "spectral_type", "spectral_type_source", "semi_major_axis_au", "eccentricity",
        "inclination_deg", "perihelion_au", "diameter_km", "diameter_source", "albedo", "absolute_magnitude_h",
        "estimated_mass_kg", "mass_measured", "albedo_assumed_for_diameter", "catalog_date", "pipeline_version"]

# First letter of the Bus–DeMeo (DeMeo et al. 2009) / Mahlke / Tholen class. K/L are anhydrous but
# CV/CO-linked (Sunshine et al. 2008; Clark et al. 2009), so kept apart; X is excluded from the S/C
# fraction (without albedo it spans enstatite, metal and P-like bodies; DeMeo & Carry 2013). Z, Mahlke
# et al.'s (2022) class of extremely red objects, is counted with D/P; F, one of Tholen's (1984) minor
# classes, as C-like.
GROUP_OF_LETTER = {**dict.fromkeys(list("SQAVRO"), "S-like"), **dict.fromkeys(list("CBF"), "C-like"),
                   **dict.fromkeys(list("DPTZ"), "D/P"), **dict.fromkeys(list("KL"), "K/L"),
                   **dict.fromkeys(list("XME"), "X")}
TIER_OF_SOURCE = {"source": "taxonomy", "albedo": "albedo_proxy", "albedo_assumed": "assumed"}


def norm_key(s: pd.Series) -> pd.Series:
    """Designation key shared by the catalog and the Nesvorný files."""
    return s.astype(str).str.replace(" ", "", regex=False).str.upper()


def to_group(t: pd.Series) -> pd.Series:
    return t.astype(str).str[0].map(GROUP_OF_LETTER).fillna("other")


def load_catalog(path: Path) -> pd.DataFrame:
    """The columns this analysis uses, read from an AsteroidCatalog release parquet."""
    cat = pd.read_parquet(path, columns=COLS)
    cat["key"] = norm_key(cat["designation"])
    return cat


def main_belt(cat: pd.DataFrame) -> pd.DataFrame:
    """2.1 < a < 3.3 AU, q > 1.3 AU, with zone, compositional group and label tier."""
    mb = cat[cat.semi_major_axis_au.between(A_MIN, A_MAX) & (cat.perihelion_au > Q_MIN)].copy()
    mb["zone"] = pd.cut(mb.semi_major_axis_au, ZONES, labels=ZONE_NAMES)
    mb["group"] = to_group(mb.spectral_type)
    mb["tier"] = mb.spectral_type_source.map(TIER_OF_SOURCE).fillna("none")
    return mb

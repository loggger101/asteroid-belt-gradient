"""Catalog loading, main-belt selection, and compositional groups.

The catalog assigns a spectral type to almost every body, by three different routes
(``spectral_type_source``):

=================  ===========================================================  =====================
tier               route                                                        used?
=================  ===========================================================  =====================
``source``         published taxonomy (SsODNet: SDSS/Gaia/spectroscopic surveys)  yes, primary
``albedo``         *measured* albedo thresholded: p<0.10 C, <0.35 S, else V       label-free check only
``albedo_assumed`` albedo *assumed from semimajor axis*, then thresholded         never: circular
=================  ===========================================================  =====================
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import A_MAX, A_MIN, Q_MIN, ZONE_NAMES, ZONES

COLS = ["designation", "name", "spectral_type", "spectral_type_source", "semi_major_axis_au", "eccentricity",
        "inclination_deg", "perihelion_au", "diameter_km", "diameter_source", "albedo", "absolute_magnitude_h",
        "estimated_mass_kg", "mass_measured", "catalog_date", "pipeline_version"]

# First letter of the Bus–DeMeo / Gaia class. K/L are anhydrous but CV/CO-linked, so kept apart;
# X is excluded from the S/C fraction (without albedo it spans enstatite, metal and P-like bodies).
GROUP_OF_LETTER = {**dict.fromkeys(list("SQAVRO"), "S-like"), **dict.fromkeys(list("CB"), "C-like"),
                   **dict.fromkeys(list("DPT"), "D/P"), **dict.fromkeys(list("KL"), "K/L"),
                   **dict.fromkeys(list("XME"), "X")}
TIER_OF_SOURCE = {"source": "taxonomy", "albedo": "albedo_proxy", "albedo_assumed": "assumed"}


def norm_key(s: pd.Series) -> pd.Series:
    """Designation key shared by the catalog and the Nesvorný files."""
    return s.astype(str).str.replace(" ", "", regex=False).str.upper()


def to_group(t: pd.Series) -> pd.Series:
    return t.astype(str).str[0].map(GROUP_OF_LETTER).fillna("other")


def build_snapshot(source_csv: Path, snapshot: Path) -> None:
    """Freeze the columns this analysis uses from a full AsteroidCatalog CSV build."""
    raw = pd.read_csv(source_csv, usecols=COLS, dtype={"designation": str, "name": str}, low_memory=False)
    raw.to_parquet(snapshot, index=False)


def load_catalog(snapshot: Path) -> pd.DataFrame:
    cat = pd.read_parquet(snapshot)
    cat["key"] = norm_key(cat["designation"])
    return cat


def main_belt(cat: pd.DataFrame) -> pd.DataFrame:
    """2.1 < a < 3.3 AU, q > 1.3 AU, with zone, compositional group and label tier."""
    mb = cat[cat.semi_major_axis_au.between(A_MIN, A_MAX) & (cat.perihelion_au > Q_MIN)].copy()
    mb["zone"] = pd.cut(mb.semi_major_axis_au, ZONES, labels=ZONE_NAMES)
    mb["group"] = to_group(mb.spectral_type)
    mb["tier"] = mb.spectral_type_source.map(TIER_OF_SOURCE).fillna("none")
    return mb

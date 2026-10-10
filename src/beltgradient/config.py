"""Paths and analysis knobs. Every number the paper depends on is set here."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ── analysis knobs ──────────────────────────────────────────────────────────
A_MIN, A_MAX   = 2.1, 3.3          # main belt, AU
Q_MIN          = 1.3               # perihelion floor: drop Mars-crossers
ZONES          = [2.1, 2.5, 2.82, 2.96, 3.3]            # inner | middle | pristine | outer (Kirkwood gaps)
ZONE_NAMES     = ["inner", "middle", "pristine", "outer"]  # "pristine" zone: Brož et al. (2013, A&A 551, A117)
KIRKWOOD       = {"3:1": 2.502, "5:2": 2.825, "7:3": 2.958, "2:1": 3.278}
SNOW_LINE_AU   = (278 / 170) ** 2  # T_eq = 278 K (a/1 AU)^-1/2 = 170 K  -> 2.67 AU (Hayashi 1981)
# the rows of the BHAC15 1 M_sun track (bhac15.TRACK_1MSUN, log t in yr) whose snow-line radius the paper's
# section 2.2 quotes: 1.0 and 5.1 Myr. The snow line scales as L^1/2 at fixed temperature.
BHAC15_QUOTED_LOG_T = (6.001799, 6.703958)
BIN_W          = 0.05              # AU, for fraction-vs-a curves
COMPLETE_FRAC  = 0.95              # label completeness required for the size-complete sample
P_DARKEST      = 0.04              # darkest plausible albedo, sets D_complete
H_IPW_MAX      = 15.0              # IPW: labelled bodies must sit in an H bin brighter than this...
D_IPW          = 10.0              # km; IPW sample is SIZE-limited (dark 10 km body has H~14.1 < H_IPW_MAX)
C_MIN_IPW      = 0.30              # ...and completeness >= this
D_ORBIT        = 10.0              # km, size floor for the orbital-excitation test
EXTEND_FAMILIES = False            # one-step nearest-member extension (sensitivity test)
# X complex by measured albedo (robustness test): P below 0.10, M to 0.30, E above, the convention Fornasier,
# Clark & Dotto (2011, Icarus 214, 131) state after Tholen & Barucci (1989). The catalog splits X the same way.
X_SPLIT_ALBEDO = (0.10, 0.30)
N_BOOT         = 300
SEED           = 4045              # one RNG stream for the whole run; draw order is fixed by pipeline.run()
H_EDGES        = np.arange(3, 19.5, 0.5)

COLORS = {"S-like": "#c0392b", "C-like": "#2c3e50", "D/P": "#7d5a3c", "K/L": "#d68910", "X": "#8e8e8e", "other": "#cccccc"}

# ── inputs ──────────────────────────────────────────────────────────────────
# The AsteroidCatalog build analysed here: a frozen, checksummed GitHub release (data contract 1.8.1).
# JPL adds bodies daily, so a result must name its build; releases are never rebuilt under a tag.
CATALOG_RELEASE = "data-2026-09-29c"


@dataclass(frozen=True)
class Paths:
    """Where inputs are read from and outputs written to.

    ``root`` defaults to the repository checkout. Set ``BELT_GRADIENT_DATA`` to read
    the (large, untracked) inputs from somewhere else.
    """
    root: Path

    @classmethod
    def default(cls) -> "Paths":
        return cls(Path(__file__).resolve().parents[2])

    @property
    def data(self) -> Path:
        return Path(os.environ.get("BELT_GRADIENT_DATA", self.root / "data"))

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def catalog(self) -> Path:
        # named by release tag, so a file from another build is never read by mistake
        return self.data / f"asteroid_catalog_{CATALOG_RELEASE}.parquet"

    @property
    def nesvorny(self) -> Path:
        return self.data / "ast.nesvorny.families_V2_0"

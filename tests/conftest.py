"""Synthetic inputs in the formats of the two real ones, small enough to run the whole pipeline in seconds.

The synthetic belt has known properties, so a run can be checked against them:

* P(C | a) among S- and C-types is logistic with a50 = ``TRUE_A50`` at every size;
* taxonomy labels depend only on H: every body brighter than H = 10.5 is labelled, fainter ones less
  and less often (so H_c = 10.5 and D_c = 52.8 km, as in the real catalog); a dark C-type is fainter
  than an S-type of the same size, so the raw labelled sample is biased toward S;
* measured albedos are complete brighter than H = 12;
* ``N_HALO`` unlisted bodies sit a few m/s from Themis and Vesta members, for the one-step extension.

The family bundle carries the quirks :func:`beltgradient.families.load_families` has to handle: a
family whose parent has no number ("-- 2012 PM61"), two identical member lists (Leonidas and Vibilia),
a family nested in another (Karin in Koronis), list lines without members (James Bond, "503 -"),
a family outside the main belt (Hilda), and CRLF line ends.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

from beltgradient.catalog import COLS
from beltgradient.config import Paths

matplotlib.use("Agg")                    # before any test imports pyplot (the pipeline does)

TRUE_A50 = 2.6
N_HALO = 10

TYPES = {"S": ["S", "Sq", "V", "Q"], "C": ["C", "Ch", "B", "Cb"], "D": ["D", "P", "T", "Z"], "K": ["K", "L"],
         "X": ["X", "Xc", "M"], "U": ["U"]}
ALBEDO = {"S": 0.25, "C": 0.06, "D": 0.05, "K": 0.14, "X": 0.12, "U": 0.10}
DENSITY = {"S": 2700, "C": 1600, "D": 1200, "K": 2500, "X": 3000, "U": 2000}      # kg/m^3

# 2015 families: id, parent number, name, HCM cutoff (m/s), members, centre (a, e, sin i), class mix.
# Leonidas and Vibilia share one member list and Karin's members are all Koronis members (see below).
FAMILIES_2015 = [
    ("001", 153, "Hilda", 130, 15, (3.96, 0.17, 0.15), {"D": 1.0}),
    ("401", 4, "Vesta", 50, 60, (2.36, 0.10, 0.12), {"S": 0.9, "C": 0.1}),
    ("402", 8, "Flora", 60, 40, (2.25, 0.14, 0.09), {"S": 0.85, "C": 0.15}),
    ("528", 2782, "Leonidas", 50, 15, (2.68, 0.19, 0.06), {"C": 0.8, "S": 0.2}),
    ("529", 144, "Vibilia", 100, 0, None, None),
    ("602", 24, "Themis", 60, 60, (3.13, 0.15, 0.02), {"C": 0.9, "S": 0.1}),
    ("605", 158, "Koronis", 45, 40, (2.88, 0.05, 0.04), {"S": 0.9, "C": 0.1}),
    ("606", 221, "Eos", 45, 40, (3.01, 0.08, 0.17), {"K": 0.5, "S": 0.4, "C": 0.1}),
    ("610", 832, "Karin", 10, 0, None, None),
]
# 2024 families: file stem, list line, members, centre, class mix
FAMILIES_2024 = [
    ("middle_177_irma_fam3", "177 Irma                60      30", 30, (2.74, 0.12, 0.20), {"C": 0.8, "S": 0.2}),
    ("inner_0_2012pm61_fam3", "-- 2012 PM61            15      12 ", 12, (2.44, 0.11, 0.12), {"S": 1.0}),
]
# named giants: (name, family, group, spectral type, a, D km, albedo)
GIANTS = [("Ceres", None, "C", "C", 2.767, 939.4, 0.09), ("Pallas", None, "C", "B", 2.772, 513.0, 0.15),
          ("Vesta", "2015_401", "S", "V", 2.361, 525.4, 0.42), ("Hygiea", None, "C", "C", 3.142, 434.0, 0.07),
          ("Metis", "2015_402", "X", "M", 2.386, 170.7, 0.18), ("Flora", "2015_402", "S", "S", 2.201, 144.8, 0.23),
          ("Eos", "2015_606", "K", "K", 3.012, 103.9, 0.14), ("Themis", "2015_602", "C", "C", 3.135, 176.0, 0.07)]


def _draw_groups(rng, n, mix):
    g = list(mix)
    return rng.choice(g, n, p=[mix[k] for k in g])


def _bodies(rng) -> pd.DataFrame:
    """One row per body: proper and osculating elements, group, size, albedo, family list membership."""
    parts = []

    # background: logistic P(C | a); X, D (beyond 2.8 AU), K and unclassifiable U mixed in independently of a
    n = 3000
    a = rng.uniform(2.12, 3.28, n)
    grp = np.where(rng.random(n) < 1 / (1 + np.exp(-(a - TRUE_A50) / 0.15)), "C", "S").astype(object)
    r = rng.random(n)
    grp[r < 0.07] = "X"
    grp[(r >= 0.07) & (r < 0.11) & (a > 2.8)] = "D"
    grp[(r >= 0.11) & (r < 0.14)] = "K"
    grp[(r >= 0.14) & (r < 0.145)] = "U"
    parts.append(pd.DataFrame({"a_p": a, "e_p": rng.uniform(0.02, 0.25, n), "sini_p": rng.uniform(0.01, 0.30, n),
                               "group": grp, "D": 10 ** rng.uniform(0, np.log10(250), n), "fam": None}))

    # families, clustered in proper elements; the named giants join theirs below
    fams = [(f"2015_{fid}", nm, c, mix) for fid, _, _, _, nm, c, mix in FAMILIES_2015 if nm]
    fams += [(f"2024_{stem}", nm, c, mix) for stem, _, nm, c, mix in FAMILIES_2024]
    for fam, nm, centre, mix in fams:
        parts.append(pd.DataFrame({"a_p": centre[0] + rng.normal(0, 0.01, nm), "e_p": centre[1] + rng.normal(0, 0.003, nm),
                                   "sini_p": centre[2] + rng.normal(0, 0.002, nm), "group": _draw_groups(rng, nm, mix),
                                   "D": 10 ** rng.uniform(np.log10(2), np.log10(40), nm), "fam": fam}))
    df = pd.concat(parts, ignore_index=True)

    # halo: unlisted bodies a few m/s from listed Themis and Vesta members
    halo = []
    for fam in ("2015_602", "2015_401"):
        src = df[df.fam.eq(fam)].sample(N_HALO // 2, random_state=1)
        halo.append(src.assign(e_p=src.e_p + 0.0003, fam=None, halo=True))
    df = pd.concat([df.assign(halo=False)] + halo, ignore_index=True)

    g = pd.DataFrame(GIANTS, columns=["name", "fam", "group", "spectral_type", "a_p", "D", "p"])
    g = g.assign(e_p=0.1, sini_p=0.15, halo=False)
    for i, row in g.iterrows():                                  # a giant in a family sits at its centre
        if isinstance(row.fam, str):
            m = df[df.fam.eq(row.fam)]
            g.loc[i, ["a_p", "e_p", "sini_p"]] = m.a_p.mean(), m.e_p.mean(), m.sini_p.mean()
    df = pd.concat([g, df], ignore_index=True)

    n = len(df)
    df["spectral_type"] = df.spectral_type.fillna(pd.Series([rng.choice(TYPES[x]) for x in df.group], index=df.index))
    df["p"] = df.p.fillna(df.group.map(ALBEDO) * np.exp(rng.normal(0, 0.15, n)))
    df["a"] = df.a_p + rng.normal(0, 0.003, n)
    df["e"] = (df.e_p + rng.normal(0, 0.02, n)).clip(0.001)
    df["sini"] = (df.sini_p + rng.normal(0, 0.01, n)).clip(0.001)
    df["has_proper"] = rng.random(n) > 0.03
    df.loc[df.fam.notna() | df.halo | df.name.notna(), "has_proper"] = True
    return df


def _extras() -> pd.DataFrame:
    """Bodies the main-belt cut must drop: Hungarias (a < 2.1) and Mars-crossers (q < 1.3)."""
    return pd.DataFrame({"a_p": [1.9] * 6 + [2.3] * 4, "e_p": [0.08] * 6 + [0.5] * 4, "sini_p": 0.35, "group": "S",
                         "spectral_type": "S", "D": 3.0, "p": 0.3, "fam": None, "halo": False, "name": None,
                         "a": [1.9] * 6 + [2.3] * 4, "e": [0.08] * 6 + [0.5] * 4, "sini": 0.35, "has_proper": False})


def _catalog(rng, df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    H = np.round(5 * np.log10(1329 / (df.D * np.sqrt(df.p))), 2)
    labelled = (H < 10.5) | (rng.random(n) < np.clip(0.85 - 0.15 * (H - 10.5), 0.05, 1))
    measured = (H < 12) | (rng.random(n) < 0.5)
    by_a = np.select([df.a < 2.5, df.a < 2.82, df.a < 3.27], [0.190, 0.086, 0.066], 0.057)
    assumed = np.where(measured, np.nan, np.where(labelled, df.group.map(ALBEDO), by_a))
    tier = np.select([labelled, measured], ["source", "albedo"], "albedo_assumed")
    stype = np.select([labelled, measured], [df.spectral_type, np.where(df.p < 0.1, "C", "S")],
                      np.where(assumed < 0.1, "C", "S"))
    mass = 4 / 3 * np.pi * (df.D * 500) ** 3 * df.group.map(DENSITY)
    return pd.DataFrame({
        "designation": df.designation, "name": df.name, "spectral_type": stype, "spectral_type_source": tier,
        "semi_major_axis_au": df.a, "eccentricity": df.e, "inclination_deg": np.degrees(np.arcsin(df.sini)),
        "perihelion_au": df.a * (1 - df.e), "diameter_km": df.D,
        "diameter_source": np.select([measured, labelled], ["measured", "derived_h_taxonomy_albedo"], "derived_h_orbit_albedo"),
        "albedo": np.where(measured, df.p.round(3), np.nan), "absolute_magnitude_h": H, "estimated_mass_kg": mass,
        "mass_measured": df.D > 400, "albedo_assumed_for_diameter": assumed,
        "catalog_date": "2026-09-29", "pipeline_version": "1.8.1",
    })[COLS]


def _write(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(f"{s}\r\n" for s in lines).encode("ascii"))


def _bundle(nesvorny, df: pd.DataFrame) -> None:
    data = nesvorny / "data"
    fam_lines = []
    for fid, pnum, name, cut, _, _, _ in FAMILIES_2015:
        fam_lines.append(f"{fid} {pnum} {name:<14} {cut:4d} {(df.fam == f'2015_{fid}').sum():7d} 2.2407 2.4897 0.0746 0.1333  5.37  7.75 -")
    fam_lines.insert(1, "007 James Bond           0       1 2.4739 2.4739 0.1282 0.1282  6.32  6.32 N")
    fam_lines.insert(4, "503 -                    0       0 -.9999 -.9999 -.9999 -.9999 -9.99 -9.99 N")
    _write(data / "familylist.tab", fam_lines)

    def members_2015(fid, pnum, name, m):
        _write(data / "families_2015" / f"{fid}_{name.lower()}.tab",
               [f"{r.designation:>6} {r.a_p:.5f} {r.e_p:.5f} {r.sini_p:8.5f} {r.H:6.2f} {-1.0:6.2f} {fid:>5} {pnum} {name}"
                for r in m.itertuples()])

    for fid, pnum, name, _, nm, _, _ in FAMILIES_2015:
        if nm:
            members_2015(fid, pnum, name, df[df.fam.eq(f"2015_{fid}")])
    members_2015("529", 144, "Vibilia", df[df.fam.eq("2015_528")])                 # the same list twice
    members_2015("610", 832, "Karin", df[df.fam.eq("2015_605")].head(12))          # nested in Koronis

    doc = ["inner belt", "", "name                   cut      mem", "------------------------------------"]
    for stem, line, _, _, _ in FAMILIES_2024:
        doc.append(line)
        m = df[df.fam.eq(f"2024_{stem}")]
        _write(data / "families_2024" / f"{stem}.csv",
               [f"{r.a_p:.6f},{r.e_p:.6f},{r.sini_p:.6f},36.8,-40.0,{r.H:.3f},5,K00X00X,{r.designation.replace(' ', '')}"
                for r in m.itertuples()])
    _write(nesvorny / "document" / "list_of_new_families_2024.txt", doc)

    pr = df[df.has_proper]
    _write(data / "proper_catalog24.tab",
           [f"{r.a_p:.6f} 0.10E-03 {r.e_p:.6f} 0.10E-03 {r.sini_p:.6f} 0.10E-03   40.000000  -50.000000 {r.H:6.3f} 100 "
            f"{'K00X00X':>7} {r.designation.replace(' ', ''):<10}" for r in pr.itertuples()])


def make_inputs(root) -> dict:
    """Write both inputs under ``root/data``; return what the run should find."""
    rng = np.random.default_rng(2026)
    df = pd.concat([_bodies(rng), _extras()], ignore_index=True)
    # numbered designations, except the unnumbered-parent family (as in the 2024 bundle) and a few others
    df["designation"] = [str(i + 1) for i in range(len(df))]
    pm = df.fam.eq("2024_inner_0_2012pm61_fam3")
    df.loc[pm, "designation"] = ["2012 PM61"] + [f"2015 A{chr(66 + k)}{k + 1}" for k in range(pm.sum() - 1)]
    cat = _catalog(rng, df)
    df["H"] = cat.absolute_magnitude_h

    paths = Paths(root)
    paths.data.mkdir(parents=True)
    cat.to_parquet(paths.catalog, index=False)
    _bundle(paths.nesvorny, df)
    outside = df.fam.eq("2015_001") | (df.a_p < 2.1) | (df.e_p > 0.4)               # Hildas, Hungarias, Mars-crossers
    return {"rows": len(cat), "main_belt": int((~outside).sum()), "n_families": len(FAMILIES_2015) + len(FAMILIES_2024),
            "vesta_members": int(df.fam.eq("2015_401").sum()), "true_a50": TRUE_A50, "n_halo": N_HALO}


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    """(paths, expected): synthetic inputs under a fresh root, which is also where outputs go."""
    root = tmp_path_factory.mktemp("synthetic")
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("BELT_GRADIENT_DATA", raising=False)
        expected = make_inputs(root)
    return Paths(root), expected

"""Collisional families (Nesvorný HCM catalogue) and proper elements.

Membership comes from the PDS SBN bundle ``ast.nesvorny.families`` V2.0: 119 families from
Nesvorný et al. (2015, numbered asteroids only) plus 153 from Nesvorný, Roig, Vokrouhlický &
Brož (2024). Proper elements come from the same bundle (``proper_catalog24.tab``, 1.25 M orbits).

Counting fragments measures which bodies happened to break, not what formed where, so each
family is collapsed to one body (:func:`family_table`) before any fraction is computed.
"""
from __future__ import annotations

import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .catalog import norm_key


def load_families(nesvorny: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (families, members). A body listed in two families goes to the larger one."""
    data = nesvorny / "data"

    # ── 2015 families: list + members ───────────────────────────────────────
    fl_rows = []
    for line in open(data / "familylist.tab", encoding="ascii"):
        m = re.match(r"^(\d{3})\s+(\d+)\s+(.*?)\s+(\d+)\s+(\d+)\s+[\d.]+", line)
        if m:
            fl_rows.append(dict(fam_id=f"2015_{m[1]}", parent_num=m[2], fam_name=m[3].strip(), cutoff=float(m[4])))
    fams15 = pd.DataFrame(fl_rows)

    mem = []
    for f in sorted(glob(str(data / "families_2015" / "*.tab"))):
        for line in open(f, encoding="ascii"):
            t = line.split()
            if len(t) >= 7:
                mem.append((t[0], f"2015_{t[6]}"))
    mem15 = pd.DataFrame(mem, columns=["key", "fam_id"])

    # ── 2024 families: cutoffs from the document list, members from csvs ────
    cut24 = {}
    for line in open(nesvorny / "document" / "list_of_new_families_2024.txt", encoding="ascii"):
        m = re.match(r"^(\d+)\s+(.*?)\s+(\d+)\s+(\d+)(\s+\S+)?\s*$", line.strip())
        if m:
            cut24[m[1]] = (m[2], float(m[3]))
    rows, mem = [], []
    for f in sorted(glob(str(data / "families_2024" / "*.csv"))):
        stem = Path(f).stem                       # e.g. middle_177_irma_fam3
        parts = stem.split("_")
        pnum = parts[1]
        name, cut = cut24.get(pnum, (parts[2], np.nan))
        fid = f"2024_{stem}"
        rows.append(dict(fam_id=fid, parent_num=pnum, fam_name=name, cutoff=cut))
        x = pd.read_csv(f, header=None, dtype=str)
        mem += [(k, fid) for k in x[8]]
    fams24 = pd.DataFrame(rows)
    mem24 = pd.DataFrame(mem, columns=["key", "fam_id"])

    fams = pd.concat([fams15, fams24], ignore_index=True)
    assert fams.fam_id.is_unique, fams.fam_id[fams.fam_id.duplicated()].tolist()
    members = pd.concat([mem15, mem24], ignore_index=True)
    members["key"] = norm_key(members["key"])
    fams["n_listed"] = fams.fam_id.map(members.fam_id.value_counts()).fillna(0).astype(int)

    # de-duplicate: a body in two families (e.g. Karin inside Koronis) goes to the larger one
    members = (members.merge(fams[["fam_id", "n_listed"]], on="fam_id")
                      .sort_values("n_listed", ascending=False).drop_duplicates("key"))
    return fams, members


def load_proper_elements(nesvorny: Path) -> pd.DataFrame:
    pc = pd.read_csv(nesvorny / "data" / "proper_catalog24.tab", sep=r"\s+", header=None, dtype={10: str, 11: str},
                     usecols=[0, 2, 4, 8, 11], names=["a_p", "e_p", "sini_p", "H_p", "key"])
    pc["key"] = norm_key(pc["key"])
    return pc.drop_duplicates("key")


def attach_families(mb: pd.DataFrame, members: pd.DataFrame, pc: pd.DataFrame) -> pd.DataFrame:
    """Add ``fam_id``, ``in_family`` and proper elements (``a_p``, ``e_p``, ``sini_p``) to the main belt."""
    mb = mb.drop(columns=[c for c in ["fam_id", "a_p", "e_p", "sini_p"] if c in mb], errors="ignore")
    mb = mb.merge(members[["key", "fam_id"]], on="key", how="left").merge(pc[["key", "a_p", "e_p", "sini_p"]], on="key", how="left")
    mb["in_family"] = mb.fam_id.notna()
    return mb.drop_duplicates("key").reset_index(drop=True)


def zappala_coords(a, e, sini):
    """Embed proper elements so Euclidean distance is the Zappalà et al. (1990) metric, in m/s:
    d = na * sqrt(5/4 (da/a)^2 + 2 (de)^2 + 2 (d sin i)^2)."""
    v = 29_780.0 / np.sqrt(a)                         # n a, m/s
    return np.column_stack([v * np.sqrt(1.25) * np.log(a), v * np.sqrt(2) * e, v * np.sqrt(2) * sini])


def extend_families(df: pd.DataFrame, fams: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """One-step nearest-member extension, no chaining.

    A background body joins the family of its nearest listed member if that distance is below
    the family's cutoff. The cutoffs were tuned on a catalogue ~3x sparser, so this over-attaches:
    treat it as an upper bound on family contamination.
    """
    cut = fams.set_index("fam_id").cutoff
    has_p = df.a_p.notna()
    src = df[has_p & df.in_family]
    tgt = df[has_p & ~df.in_family]
    tree = cKDTree(zappala_coords(src.a_p.values, src.e_p.values, src.sini_p.values))
    d, j = tree.query(zappala_coords(tgt.a_p.values, tgt.e_p.values, tgt.sini_p.values), k=1)
    fam_nn = src.fam_id.values[j]
    ok = d < cut.reindex(fam_nn).values
    out = df.copy()
    out.loc[tgt.index[ok], "fam_id"] = fam_nn[ok]
    out["in_family"] = out.fam_id.notna()
    return out, int(ok.sum())


def family_table(df: pd.DataFrame, fams: pd.DataFrame) -> pd.DataFrame:
    """One row per family: size, mass, representative orbit, and the class of the collapsed body.

    ``fam_group`` is the majority group of >= 3 taxonomy-labelled members, otherwise the largest
    member's own label. ``purity`` is the fraction of labelled members in that group (HCM does not
    remove spectral interlopers). ``D_equiv_km`` is from the summed D^3; ``a_rep`` etc. from the
    largest member.
    """
    fm = df[df.in_family]
    lab = fm[fm.tier.eq("taxonomy") & ~fm.group.isin(["X", "other"])]
    g = lab.groupby("fam_id").group
    maj = g.agg(lambda s: s.value_counts().index[0]).rename("fam_group")
    pur = g.agg(lambda s: s.value_counts(normalize=True).iloc[0]).rename("purity")
    nlab = g.size().rename("n_labelled")
    D3 = (fm.diameter_km ** 3).groupby(fm.fam_id).sum()
    largest = fm.sort_values("diameter_km", ascending=False).drop_duplicates("fam_id").set_index("fam_id")
    t = pd.DataFrame({
        "n_members": fm.groupby("fam_id").size(),
        "D_equiv_km": D3 ** (1 / 3),
        "mass_kg": fm.groupby("fam_id").estimated_mass_kg.sum(),
        "a_rep": largest.semi_major_axis_au, "e_rep": largest.e_p, "sini_rep": largest.sini_p,
        "largest": largest.name, "largest_group": largest.group, "largest_tier": largest.tier,
    }).join([maj, pur, nlab])
    fallback = t.largest_group.where(t.largest_tier.eq("taxonomy") & ~t.largest_group.isin(["X", "other"]))
    t["fam_group"] = t.fam_group.where(t.n_labelled >= 3, fallback)
    return t.join(fams.set_index("fam_id")[["fam_name", "cutoff"]]).sort_values("n_members", ascending=False)

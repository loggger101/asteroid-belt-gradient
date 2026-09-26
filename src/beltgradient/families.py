"""Collisional families (Nesvorný HCM catalogue) and proper elements.

Membership comes from the PDS SBN bundle ``ast.nesvorny.families`` V2.0: 119 families from
Nesvorný et al. (2015, numbered asteroids only) plus 153 from Nesvorný, Roig, Vokrouhlický &
Brož (2024). Proper elements (orbits with the planets' periodic perturbations removed; Knežević &
Milani 2003) come from the same bundle (``proper_catalog24.tab``, 1.25 M orbits).

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
    # A numbered parent is listed by number ("177 Irma  ..."); an unnumbered one as "-- 2012 PM61  ...",
    # whose files carry parent number 0 and the designation ("inner_0_2012pm61_fam3").
    cut24 = {}
    for line in open(nesvorny / "document" / "list_of_new_families_2024.txt", encoding="ascii"):
        m = re.match(r"^(\d+|--)\s+(.*?)\s+(\d+)\s+(\d+)(\s+\S+)?\s*$", line.strip())
        if m:
            key = m[1] if m[1] != "--" else m[2].replace(" ", "").lower()
            cut24[key] = (m[2], float(m[3]))
    rows, mem = [], []
    for f in sorted(glob(str(data / "families_2024" / "*.csv"))):
        stem = Path(f).stem                       # e.g. middle_177_irma_fam3
        parts = stem.split("_")
        pnum = parts[1]
        name, cut = cut24.get(pnum if pnum != "0" else parts[2], (parts[2], np.nan))
        fid = f"2024_{stem}"
        rows.append(dict(fam_id=fid, parent_num=pnum, fam_name=name, cutoff=cut))
        x = pd.read_csv(f, header=None, dtype=str)
        mem += [(k, fid) for k in x[8]]
    fams24 = pd.DataFrame(rows)
    mem24 = pd.DataFrame(mem, columns=["key", "fam_id"])

    fams = pd.concat([fams15, fams24], ignore_index=True)
    assert fams.fam_id.is_unique, fams.fam_id[fams.fam_id.duplicated()].tolist()
    assert fams.cutoff.notna().all(), fams.fam_id[fams.cutoff.isna()].tolist()   # every family needs its HCM cutoff
    members = pd.concat([mem15, mem24], ignore_index=True)
    members["key"] = norm_key(members["key"])
    fams["n_listed"] = fams.fam_id.map(members.fam_id.value_counts()).fillna(0).astype(int)

    # de-duplicate: a body in two families (e.g. Karin inside Koronis) goes to the larger one. Equal
    # sizes go to the lower fam_id, so identical lists count once: in the bundle, 529 Vibilia's file
    # repeats 528 Leonidas's members, and the 2024 lists of 1998 HD130 and 2001 PG20 are the same 419 bodies.
    members = (members.merge(fams[["fam_id", "n_listed"]], on="fam_id")
                      .sort_values(["n_listed", "fam_id"], ascending=[False, True], kind="mergesort")
                      .drop_duplicates("key"))
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


V1AU = 0.01720209895 * 1.49597870e11 / 86400      # circular speed at 1 AU, m/s, as hcluster.c derives it
K_CANDIDATES = 16                                   # nearest neighbours in the embedding, re-ranked exactly


def zappala_distance(a1, e1, s1, a2, e2, s2):
    """HCM distance of Zappalà et al. (1990) in m/s, as the bundle's ``hcluster.c`` computes it:
    d = na sqrt(5/4 (da/a)^2 + 2 (de)^2 + 2 (d sin i)^2), with na and a taken at the pair's mean a."""
    am = (a1 + a2) / 2
    return V1AU / np.sqrt(am) * np.sqrt(1.25 * ((a1 - a2) / am) ** 2 + 2 * (e1 - e2) ** 2 + 2 * (s1 - s2) ** 2)


def zappala_embedding(a, e, sini):
    """Coordinates whose Euclidean distance equals :func:`zappala_distance` to first order.

    X = -2 sqrt(5/4) na gives dX = sqrt(5/4) na da/a. Only used to find candidate neighbours;
    :func:`extend_families` re-ranks them with the exact distance.
    """
    v = V1AU / np.sqrt(a)
    return np.column_stack([-2 * np.sqrt(1.25) * v, np.sqrt(2) * v * e, np.sqrt(2) * v * sini])


def extend_families(df: pd.DataFrame, fams: pd.DataFrame, chunk: int = 100_000) -> tuple[pd.DataFrame, int]:
    """One-step nearest-member extension, no chaining.

    A background body joins the family of its nearest listed member (by :func:`zappala_distance`)
    if that distance is below the family's cutoff, a simple version of attaching members to family
    cores (Milani et al. 2014). The 2015 cutoffs were tuned on 384,337 numbered asteroids (Nesvorný
    et al. 2015), ~3x sparser than the proper-element catalogue used here, so this over-attaches:
    treat it as an upper bound on family contamination.
    """
    cut = fams.set_index("fam_id").cutoff
    has_p = df.a_p.notna()
    src = df[has_p & df.in_family]
    tgt = df[has_p & ~df.in_family]
    sa, se, ss = (src[c].values for c in ("a_p", "e_p", "sini_p"))
    tree = cKDTree(zappala_embedding(sa, se, ss))
    best, dmin = np.empty(len(tgt), int), np.empty(len(tgt))
    for lo in range(0, len(tgt), chunk):
        ta, te, ts = (tgt[c].values[lo:lo + chunk, None] for c in ("a_p", "e_p", "sini_p"))
        _, idx = tree.query(zappala_embedding(ta[:, 0], te[:, 0], ts[:, 0]), k=K_CANDIDATES)
        d = zappala_distance(ta, te, ts, sa[idx], se[idx], ss[idx])
        j = d.argmin(axis=1)
        best[lo:lo + chunk] = idx[np.arange(len(j)), j]
        dmin[lo:lo + chunk] = d[np.arange(len(j)), j]
    fam_nn = src.fam_id.values[best]
    ok = dmin < cut.reindex(fam_nn).values
    out = df.copy()
    out.loc[tgt.index[ok], "fam_id"] = fam_nn[ok]
    out["in_family"] = out.fam_id.notna()
    return out, int(ok.sum())


def family_table(df: pd.DataFrame, fams: pd.DataFrame) -> pd.DataFrame:
    """One row per family: size, mass, representative orbit, and the class of the collapsed body.

    ``fam_group`` is the most common group among >= 3 taxonomy-labelled members (X-types do not
    vote); with fewer labelled members, or a tie for the most common group, it is the largest
    member's own label. ``purity`` is the fraction of labelled members in ``fam_group`` (HCM does
    not remove spectral interlopers). ``D_equiv_km`` is from the summed D^3; ``a_rep`` etc. from
    the largest member.
    """
    fm = df[df.in_family]
    lab = fm[fm.tier.eq("taxonomy") & ~fm.group.isin(["X", "other"])]
    counts = pd.crosstab(lab.fam_id, lab.group)
    top2 = np.sort(counts.values, axis=1)[:, ::-1][:, :2] if counts.shape[1] > 1 else np.c_[counts.values, np.zeros(len(counts))]
    maj = counts.idxmax(axis=1).where(top2[:, 0] > top2[:, 1]).rename("fam_group")     # a tie has no majority
    nlab = counts.sum(axis=1).rename("n_labelled")
    D3 = (fm.diameter_km ** 3).groupby(fm.fam_id).sum()
    largest = (fm.sort_values(["diameter_km", "key"], ascending=[False, True], kind="mergesort")
                 .drop_duplicates("fam_id").set_index("fam_id"))
    t = pd.DataFrame({
        "n_members": fm.groupby("fam_id").size(),
        "D_equiv_km": D3 ** (1 / 3),
        "mass_kg": fm.groupby("fam_id").estimated_mass_kg.sum(),
        "a_rep": largest.semi_major_axis_au, "e_rep": largest.e_p, "sini_rep": largest.sini_p,
        "largest": largest.name.fillna(largest.designation),   # most bodies are unnamed
        "largest_group": largest.group, "largest_tier": largest.tier,
    }).join([maj, nlab])
    fallback = t.largest_group.where(t.largest_tier.eq("taxonomy") & ~t.largest_group.isin(["X", "other"]))
    t["fam_group"] = t.fam_group.where((t.n_labelled >= 3) & t.fam_group.notna(), fallback)
    in_grp = [counts.at[f, g] if (f in counts.index and isinstance(g, str) and g in counts.columns) else 0
              for f, g in zip(t.index, t.fam_group)]
    t["purity"] = np.where(t.n_labelled > 0, np.array(in_grp) / t.n_labelled.fillna(0).clip(lower=1), np.nan)
    return t.join(fams.set_index("fam_id")[["fam_name", "cutoff"]]).sort_values("n_members", ascending=False)

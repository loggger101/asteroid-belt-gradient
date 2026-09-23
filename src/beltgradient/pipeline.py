"""End-to-end run: inputs → every figure in ``figures/`` and every number in ``results/summary.json``.

All bootstraps share one seeded generator, so the order of the steps below is part of the
result. Reordering them changes the bootstrap intervals (not the point estimates).
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import __version__
from .albedo import with_measured_albedo
from .catalog import load_catalog, main_belt
from .completeness import add_h_bins, add_ipw_weights, complete_limit, completeness, diameter_limit
from .config import A_MAX, A_MIN, D_IPW, EXTEND_FAMILIES, P_DARKEST, SEED, SNOW_LINE_AU, ZONE_NAMES, Paths
from .families import attach_families, extend_families, family_table, load_families, load_proper_elements
from .figures import GROUP_ORDER, RCPARAMS
from . import figures as F
from .gradient import DEN, NUM, build_samples, collapsed, crossover_table, inner_belt_by_size, mass_by_zone, sample_keys, zone_table
from .orbits import orbit_sample, orbit_tests

# name, bbox_inches for each output PNG
FIGURE_FILES = {
    "completeness": ("fig0_completeness.png", None),
    "c_fraction": ("fig1_c_fraction_vs_a.png", "tight"),
    "stacked": ("fig1b_stacked_composition.png", "tight"),
    "mass": ("fig2_mass_by_zone.png", "tight"),
    "orbits": ("fig3_orbital_excitation.png", "tight"),
    "dark": ("fig4_dark_fraction_albedo.png", None),
    "circularity": ("fig5_circularity.png", None),
}


@dataclass
class Results:
    """Everything a run produces: the summary plus the intermediate tables and samples."""
    summary: dict
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)


def _weighted_c_fraction(s: pd.DataFrame, w: str, lo: float, hi: float, which: str = "narrow") -> float:
    s = s[s.semi_major_axis_au.between(lo, hi, inclusive="right") & s.group.isin(DEN[which])]
    return round(float((s[w] * s.group.isin(NUM[which])).sum() / s[w].sum()), 4) if len(s) else None


def text_numbers(mb, ftab, comp, samples, keys, t_mass, zt_narrow, zt_broad) -> dict:
    """Every number the paper quotes in running text that is not already a table entry."""
    tax = mb[mb.tier.eq("taxonomy")]
    point = lambda zt: zt.apply(lambda col: col.str.split(" ").str[0].astype(float))
    gap = point(zt_broad) - point(zt_narrow)

    # mass: shares of all classified mass, and S/(S+C) vs S/(all) in the inner belt
    top4 = t_mass.nlargest(4, "estimated_mass_kg")
    mz = mass_by_zone(t_mass, GROUP_ORDER)

    # D/P share of the size-complete collapsed sample per 0.1 AU bin, from 2.9 AU outward (Fig. 3)
    col = samples[keys[2]]
    edges = np.arange(A_MIN, A_MAX + 1e-9, 0.1)
    st = (col[col.group.isin(GROUP_ORDER)].assign(bin=pd.cut(col.semi_major_axis_au, edges))
          .pivot_table(index="bin", columns="group", values="w", aggfunc="sum", observed=False).fillna(0))
    dp = st["D/P"].div(st.sum(axis=1))
    dp_outer = dp[[iv.left >= 2.9 - 1e-9 for iv in dp.index]]

    # IPW on a brightness-limited sample (H < H_IPW_MAX, no size cut): the S-biased alternative §3.4 rejects
    h_lim = collapsed(mb, ftab)
    h_lim = h_lim[h_lim.tier.eq("taxonomy") & (h_lim.w_ipw > 0)]

    comp_h14 = comp.loc[[iv for iv in comp.index if 13.5 <= iv.left < 14.5]]
    ipw_pool = mb[mb.tier.eq("taxonomy") & (mb.diameter_km >= D_IPW)]
    return {
        "assumed_label_fraction": round(float(mb.tier.eq("assumed").mean()), 4),
        "taxonomy_tier_family_fraction": round(float(tax.in_family.mean()), 4),
        "families_with_main_belt_members": int(len(ftab)),
        "families_classifiable": int(ftab.fam_group.notna().sum()),
        "vesta_family_members": int(ftab.loc[ftab.fam_name.eq("Vesta"), "n_members"].iloc[0]),
        "n_collapsed_size_complete_all_groups": int(len(samples[keys[2]])),
        "completeness_H13.5_to_14.5_range": [round(float(comp_h14.min().min()), 3), round(float(comp_h14.max().max()), 3)],
        "H_of_10km_dark_body": round(float(5 * np.log10(1329 / (D_IPW * np.sqrt(P_DARKEST)))), 2),
        "ipw_D10_bodies_dropped_by_bin_cuts": int((ipw_pool.w_ipw == 0).sum()),
        "H_limited_IPW_c_fraction_by_zone": {z: _weighted_c_fraction(h_lim[h_lim.zone == z], "w_ipw", A_MIN, A_MAX) for z in ZONE_NAMES},
        "innermost_c_fraction_ipw": {"2.1-2.2": _weighted_c_fraction(samples[keys[1]], "w", 2.1, 2.2),
                                     "2.2-2.3": _weighted_c_fraction(samples[keys[1]], "w", 2.2, 2.3)},
        "broad_minus_narrow_range": [round(float(gap.min().min()), 2), round(float(gap.max().max()), 2)],
        "dp_share_per_0.1AU_bin_from_2.9AU": [round(float(dp_outer.min()), 3), round(float(dp_outer.max()), 3)],
        "derived_diameter_fraction_taxonomy": round(float(tax.diameter_source.ne("measured").mean()), 4),
        "mass_share_of_classified": dict(zip(top4.name, (top4.estimated_mass_kg / t_mass.estimated_mass_kg.sum()).round(4))),
        "inner_mass_S_of_S_plus_C": round(float(mz.loc["inner", "S-like"] / (mz.loc["inner", "S-like"] + mz.loc["inner", "C-like"])), 4),
        "inner_mass_S_of_all_classified": round(float(mz.loc["inner", "S-like"] / mz.loc["inner"].sum()), 4),
    }


def _log(verbose, *a):
    if verbose:
        print(*a, flush=True)


def run(paths: Paths | None = None, *, write: bool = True, verbose: bool = True) -> Results:
    paths = paths or Paths.default()
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="All-NaN slice")
    plt.rcParams.update(RCPARAMS)
    rng = np.random.default_rng(SEED)
    if write:
        paths.figures.mkdir(parents=True, exist_ok=True)
        paths.results.mkdir(parents=True, exist_ok=True)

    def save(key, fig):
        name, bbox = FIGURE_FILES[key]
        if write:
            fig.savefig(paths.figures / name, dpi=200, bbox_inches=bbox)
        plt.close(fig)

    # 1–2 · catalog, main belt, compositional groups
    cat = load_catalog(paths.snapshot)
    _log(verbose, f"snapshot: {len(cat):,} rows | catalog_date {cat.catalog_date.iloc[0]} | pipeline {cat.pipeline_version.iloc[0]}")
    mb = main_belt(cat)
    _log(verbose, f"main belt ({A_MIN}-{A_MAX} AU): {len(mb):,} | taxonomy tier {mb.tier.eq('taxonomy').sum():,}")

    # 3 · families, proper elements, optional one-step extension
    fams, members = load_families(paths.nesvorny)
    pc = load_proper_elements(paths.nesvorny)
    mb = attach_families(mb, members, pc)
    mb_ext, n_added = extend_families(mb, fams)
    fam_base = mb.set_index("key").fam_id
    fam_ext = mb_ext.set_index("key").fam_id
    _log(verbose, f"families: {len(fams)} | in a family: {mb.in_family.mean():.1%} "
                  f"(one-step extension would attach {n_added:,} -> {mb_ext.in_family.mean():.1%})")
    if EXTEND_FAMILIES:
        mb = mb_ext
    ftab = family_table(mb, fams)

    # 4 · completeness and debiasing
    mb = add_h_bins(mb)
    comp, ncount = completeness(mb, mb.tier.eq("taxonomy"))
    H_C = complete_limit(comp, ncount)
    D_C = diameter_limit(H_C)
    _log(verbose, f"H_c = {H_C} -> D_complete = {D_C:.1f} km")
    save("completeness", F.completeness_plot(comp, H_C, D_C))
    mb = add_ipw_weights(mb, comp)

    # 5 · C-fraction vs a, by zone
    samples = build_samples(mb, ftab, D_C)
    keys = sample_keys(D_C)
    save("c_fraction", F.c_fraction_panels(samples, keys[:3], rng))
    zt_narrow = zone_table(samples, rng, "narrow")
    zt_broad = zone_table(samples, rng, "broad")
    save("stacked", F.stacked_composition(samples[keys[2]], D_C))
    size_tab = inner_belt_by_size(mb)

    # 6 · by mass (family collapse conserves mass, so all labelled bodies)
    t = mb[mb.tier.eq("taxonomy") & mb.estimated_mass_kg.notna() & mb.group.isin(GROUP_ORDER)]
    big4 = set(t.nlargest(4, "estimated_mass_kg").index)
    mass_panels = [("all labelled bodies", t), ("without the 4 most massive", t.drop(index=list(big4)))]
    save("mass", F.mass_bars(mass_panels))
    mass_c = {}
    for lab, d in zip(["all", "no_big4"], [p[1] for p in mass_panels]):
        mz = mass_by_zone(d, GROUP_ORDER)
        mass_c[lab] = (mz["C-like"] / (mz["C-like"] + mz["S-like"])).round(3).to_dict()

    # 7 · S/C crossover
    xo = crossover_table(samples, rng)
    _log(verbose, "crossover a50:", {k: round(v, 3) for k, v in xo.a50.items()})

    # 8 · orbital excitation
    orb = orbit_sample(mb)
    otests = orbit_tests(orb, rng)
    save("orbits", F.orbital_cdfs(orb))

    # 9b · label-free albedo check
    alb = with_measured_albedo(mb)
    compA, nA = completeness(mb, mb.albedo.notna())
    H_CA = complete_limit(compA, nA)
    D_CA = diameter_limit(H_CA)
    save("dark", F.dark_fraction(alb, D_CA, rng))
    dark_zone = alb[~alb.in_family & (alb.diameter_km >= D_CA)].groupby("zone", observed=False).dark.agg(["mean", "size"])

    # 9c · family extension toggled
    alt = mb.copy()
    alt["fam_id"] = alt.key.map(fam_base if EXTEND_FAMILIES else fam_ext)
    alt["in_family"] = alt.fam_id.notna()
    alt_samples = build_samples(alt, family_table(alt, fams), D_C)
    zt_base_again = zone_table(samples, rng)
    zt_alt = zone_table(alt_samples, rng)
    xo_alt = crossover_table(alt_samples, rng, n_boot=100)
    point = lambda zt: zt.apply(lambda col: col.str.split(" ").str[0].astype(float))
    ext_dzone = (point(zt_alt) - point(zt_base_again)).abs().to_numpy().max()
    ext_dxo = (xo_alt.a50.astype(float) - xo.a50.astype(float)).abs().max()

    # 9d · the circularity trap
    save("circularity", F.circularity(mb[mb.tier.eq("assumed")].assign(w=1.0), samples["raw"], rng))

    # 9e · numbers quoted in the paper's prose (point estimates only: no RNG, so nothing above moves)
    text = text_numbers(mb, ftab, comp, samples, keys, t, zt_narrow, zt_broad)

    # 10 · summary numbers
    summary = {
        "catalog": {"rows": int(len(cat)), "catalog_date": str(cat.catalog_date.iloc[0]), "pipeline_version": str(cat.pipeline_version.iloc[0]),
                    "main_belt": int(len(mb)), "taxonomy_tier": int(mb.tier.eq("taxonomy").sum())},
        "families": {"n_families": int(len(fams)), "members_matched_main_belt": int(mb.in_family.sum()),
                     "family_fraction": round(float(mb.in_family.mean()), 4), "extension_attaches": n_added, "extension_used": EXTEND_FAMILIES},
        "completeness": {"H_c": H_C, "D_complete_km": round(D_C, 1), "H_c_albedo": H_CA, "D_complete_albedo_km": round(D_CA, 1)},
        "zone_c_fraction_narrow": zt_narrow.to_dict(), "zone_c_fraction_broad": zt_broad.to_dict(),
        "mass_c_fraction": mass_c,
        "inner_belt_by_size": size_tab.round(4).to_dict(orient="records"),
        "crossover": xo.round(4).to_dict(orient="index"),
        "snow_line_au": round(SNOW_LINE_AU, 3),
        "orbit_tests": otests.round(5).to_dict(orient="records"),
        "dark_fraction_complete": dark_zone["mean"].round(4).to_dict(),
        "family_extension_toggled": {
            "family_fraction": round(float(alt.in_family.mean()), 4),
            "zone_c_fraction_narrow": zt_alt.to_dict(),
            "crossover_a50": xo_alt.a50.astype(float).round(4).to_dict(),
            "max_abs_change_zone_fraction": round(float(ext_dzone), 4),
            "max_abs_change_a50_au": round(float(ext_dxo), 4),
        },
        "text_numbers": text,
        "software": {"beltgradient": __version__, "seed": SEED},
    }
    if write:
        (paths.results / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        _log(verbose, f"wrote {paths.results / 'summary.json'} and {len(FIGURE_FILES)} figures to {paths.figures}")

    return Results(
        summary=summary,
        tables={"families": ftab, "completeness": comp, "zone_narrow": zt_narrow, "zone_broad": zt_broad,
                "inner_by_size": size_tab, "crossover": xo, "orbit_tests": otests, "dark_zone": dark_zone,
                "zone_narrow_extended": zt_alt, "crossover_extended": xo_alt},
        frames={"main_belt": mb, "orbit_sample": orb, **{f"sample: {k}": v for k, v in samples.items()}},
    )

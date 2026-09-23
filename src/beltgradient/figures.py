"""Every figure in the paper. Each function returns the Figure; :mod:`beltgradient.pipeline` saves it.

File names are historical (the notebook's order); the paper's numbering is:
fig5_circularity → Fig 1, fig1_c_fraction_vs_a → Fig 2, fig1b_stacked_composition → Fig 3,
fig2_mass_by_zone → Fig 4, fig3_orbital_excitation → Fig 5, fig4_dark_fraction_albedo → Fig 6,
fig0_completeness → Fig A1.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .albedo import dark_curve
from .config import A_MAX, A_MIN, BIN_W, COLORS, COMPLETE_FRAC, D_ORBIT, KIRKWOOD, SNOW_LINE_AU, ZONE_NAMES
from .gradient import c_fraction_curve, mass_by_zone
from .orbits import ELEMENTS, split_by_class

RCPARAMS = {"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlepad": 16}
GROUP_ORDER = ["S-like", "K/L", "X", "C-like", "D/P"]


def mark_structure(ax, snow=True):
    for k, a in KIRKWOOD.items():
        if A_MIN < a < A_MAX:
            ax.axvline(a, color="k", ls=":", lw=0.8, alpha=0.6)
            ax.text(a, 1.01, k, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color="0.35")
    if snow:
        ax.axvline(SNOW_LINE_AU, color="#2e86c1", ls="--", lw=1)


def completeness_plot(comp: pd.DataFrame, h_c: float, d_c: float):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    mids = [iv.mid for iv in comp.index]
    for z, c in zip(ZONE_NAMES, ["#1b4f72", "#2e86c1", "#85c1e9", "#aab7b8"]):
        ax.plot(mids, comp[z], color=c, lw=2, label=z)
    ax.axhline(COMPLETE_FRAC, color="k", ls=":", lw=.8); ax.axvline(h_c, color="#c0392b", lw=1)
    ax.text(h_c + .1, .1, f"$H_c$={h_c}\n$D_c$≈{d_c:.0f} km", color="#c0392b", fontsize=8)
    ax.set(xlabel="absolute magnitude H", ylabel="fraction with taxonomy label", title="Label completeness by zone", xlim=(5, 18.5))
    ax.legend(fontsize=8); fig.tight_layout()
    return fig


def c_fraction_panels(samples: dict[str, pd.DataFrame], panel_keys: list[str], rng: np.random.Generator):
    """C-fraction vs a: raw, collapsed + IPW, collapsed + size-complete."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for ax, k in zip(axes, panel_keys):
        s = samples[k]
        sparse = k != "raw"   # weighted / size-limited samples are thin at 0.05 AU
        wide = 0.1 if sparse else BIN_W
        c = c_fraction_curve(s, rng, "narrow", edges=np.arange(A_MIN, A_MAX + 1e-9, wide))
        ok = c.n >= (8 if sparse else 20)
        ax.fill_between(c.a[ok], c.lo[ok], c.hi[ok], color=COLORS["C-like"], alpha=.18, lw=0)
        ax.plot(c.a[ok], c.f[ok], color=COLORS["C-like"], lw=2, marker="o" if sparse else None, ms=3)
        cb = c_fraction_curve(s, rng, "broad", edges=np.arange(A_MIN, A_MAX + 1e-9, wide), n_boot=20)
        ax.plot(cb.a[ok], cb.f[ok], color=COLORS["K/L"], lw=1.2, ls="--", label="broad (C+B+D/P+K/L)")
        mark_structure(ax)
        ax.set(title=f"{k}\n(n={len(s):,})", xlabel="semimajor axis a (AU)", ylim=(0, 1), xlim=(A_MIN, A_MAX))
    axes[0].set_ylabel("carbonaceous fraction  C/(S+C)")
    axes[0].plot([], [], color=COLORS["C-like"], lw=2, label="narrow (C+B)")
    axes[0].plot([], [], color="#2e86c1", ls="--", label=f"snow line, T$_{{eq}}$=170 K ({SNOW_LINE_AU:.2f} AU)")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Compositional gradient of the main belt: effect of families and size bias", y=1.02)
    fig.tight_layout()
    return fig


def stacked_composition(s: pd.DataFrame, d_c: float):
    """All five groups for the size-complete, family-collapsed sample (counts above bars)."""
    edges = np.arange(A_MIN, A_MAX + 1e-9, 0.1)
    s = s.assign(bin=pd.cut(s.semi_major_axis_au, edges))
    order = GROUP_ORDER
    st = s[s.group.isin(order)].pivot_table(index="bin", columns="group", values="w", aggfunc="sum", observed=False).reindex(columns=order).fillna(0)
    frac = st.div(st.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = [iv.mid for iv in frac.index]
    ax.stackplot(x, [frac[g] for g in order], colors=[COLORS[g] for g in order], labels=order, alpha=.9)
    for xi, n in zip(x, st.sum(axis=1)):
        ax.text(xi, 1.005, f"{int(n)}", ha="center", va="bottom", fontsize=6.5, color="0.4")
    mark_structure(ax, snow=False)
    ax.set(xlim=(A_MIN + .05, A_MAX - .05), ylim=(0, 1), xlabel="a (AU)", ylabel="fraction by number",
           title=f"Composition, family-collapsed, D ≥ {d_c:.0f} km (counts above bars)")
    ax.legend(loc="center left", bbox_to_anchor=(1, .5), fontsize=8); fig.tight_layout()
    return fig


def mass_bars(panels: list[tuple[str, pd.DataFrame]]):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, (lab, d) in zip(axes, panels):
        mz = mass_by_zone(d, GROUP_ORDER)
        fz = mz.div(mz.sum(axis=1), axis=0)
        left = np.zeros(len(fz))
        for g in GROUP_ORDER:
            ax.barh(ZONE_NAMES, fz[g], left=left, color=COLORS[g], label=g); left += fz[g].values
        for i, z in enumerate(ZONE_NAMES):
            ax.text(1.01, i, f"{mz.loc[z].sum():.1e} kg", va="center", fontsize=7.5)
        ax.set(title=lab, xlim=(0, 1.18), xlabel="mass fraction")
    axes[0].invert_yaxis(); axes[1].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Mass-weighted composition by zone", y=1.02); fig.tight_layout()
    return fig


def orbital_cdfs(orb: pd.DataFrame):
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2))
    for j, z in enumerate(ZONE_NAMES):
        for i, (col, lab) in enumerate(ELEMENTS):
            S, C = split_by_class(orb, z, col)
            ax = axes[i, j]
            for arr, g in [(S, "S-like"), (C, "C-like")]:
                if len(arr):
                    ax.step(np.sort(arr), np.arange(1, len(arr) + 1) / len(arr), color=COLORS[g], lw=1.6, label=f"{g} (n={len(arr)})")
            if len(S) >= 10 and len(C) >= 10:
                ax.text(.97, .05, f"KS p={stats.ks_2samp(S, C).pvalue:.1e}", transform=ax.transAxes, ha="right", fontsize=7.5)
            ax.set(xlabel=lab, ylabel="cumulative fraction" if j == 0 else None, title=z if i == 0 else None)
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Proper-element distributions of S vs C, background bodies with D ≥ {D_ORBIT:.0f} km", y=1.01)
    fig.tight_layout()
    return fig


def dark_fraction(alb: pd.DataFrame, d_ca: float, rng: np.random.Generator):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for lab, d, c in [("all with measured albedo", alb, "#aab7b8"),
                      (f"background, D ≥ {d_ca:.0f} km (complete)", alb[~alb.in_family & (alb.diameter_km >= d_ca)], COLORS["C-like"])]:
        x, f, lo, hi, n = dark_curve(d, np.arange(A_MIN, A_MAX + 1e-9, 0.1 if "D ≥" in lab else BIN_W), rng)
        k = n >= 5
        ax.fill_between(x[k], lo[k], hi[k], color=c, alpha=.2, lw=0); ax.plot(x[k], f[k], color=c, lw=2, label=f"{lab} (n={len(d):,})")
    mark_structure(ax)
    ax.set(xlabel="a (AU)", ylabel="fraction with $p_V$ < 0.10", ylim=(0, 1), xlim=(A_MIN, A_MAX), title="Label-free check: dark fraction from measured albedo")
    ax.legend(fontsize=8); fig.tight_layout()
    return fig


def circularity(assumed: pd.DataFrame, raw: pd.DataFrame, rng: np.random.Generator):
    """The assumed-albedo tier returns the step function it was built from."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for lab, d, c in [("assumed-albedo tier (circular)", assumed, "#e67e22"),
                      ("taxonomy tier, raw", raw, COLORS["C-like"])]:
        cc = c_fraction_curve(d, rng, "narrow", n_boot=50)
        ax.plot(cc.a, cc.f, color=c, lw=2, label=f"{lab} (n={len(d):,})")
    mark_structure(ax, snow=False)
    ax.set(xlabel="a (AU)", ylabel="C/(S+C)", ylim=(-.02, 1.02), xlim=(A_MIN, A_MAX), title="Labels inferred from an a-dependent albedo reproduce the assumption")
    ax.legend(fontsize=8); fig.tight_layout()
    return fig

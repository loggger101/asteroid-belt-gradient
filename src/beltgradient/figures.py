"""Every figure in the paper. Each function returns the Figure; :mod:`beltgradient.pipeline` saves it.

Figures are drawn at the size they are printed: one column (``COL_W``) or the full text width
(``FULL_W``) of the paper's A4 two-column layout, at 300 dpi, so a 7 pt label in the figure is a
7 pt label on the page. Functions that take ``rng`` draw from it in a fixed order; the others draw
nothing, so they can be added or restyled without moving any bootstrap interval.

File names follow the analysis order, not the paper's; the paper's numbering is in
``pipeline.FIGURE_FILES``.
"""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm
from scipy import stats

from .albedo import dark_curve
from .config import A_MAX, A_MIN, BIN_W, COLORS, COMPLETE_FRAC, D_ORBIT, KIRKWOOD, SNOW_LINE_AU, ZONE_NAMES
from .gradient import MIN_CELL, c_fraction_curve, mass_by_zone, size_distance_grid
from .orbits import ELEMENTS, split_by_class

COL_W, FULL_W = 3.3125, 7.0208      # inches: the paper's column and text widths
INK, INK2, RULE, BAND = "#0b0b0b", "#52514e", "#b9b8b2", "#f2f1ee"
SNOW = "#2e86c1"
ZONE_COLORS = dict(zip(ZONE_NAMES, ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]))   # one hue, light -> dark outward
ZONE_MARKERS = dict(zip(ZONE_NAMES, ["o", "s", "D", "^"]))
RCPARAMS = {
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6, "legend.frameon": False,
    "legend.handlelength": 1.6, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.titlepad": 3, "axes.labelpad": 2, "lines.linewidth": 1.2, "lines.solid_capstyle": "round",
}
GROUP_ORDER = ["S-like", "K/L", "X", "C-like", "D/P"]


def _fig(w, h, **kw):
    return plt.subplots(figsize=(w, h), layout="constrained", **kw)


def mark_structure(ax, snow=True, labels=True):
    """Dotted Kirkwood gaps (named on a top axis, where the layout engine sees them) and the snow line."""
    gaps = {k: a for k, a in KIRKWOOD.items() if A_MIN < a < A_MAX}
    for a in gaps.values():
        ax.axvline(a, color=INK2, ls=":", lw=0.5, zorder=1)
    if labels:
        top = ax.secondary_xaxis("top")
        top.set_xticks(list(gaps.values()), list(gaps))
        top.tick_params(length=0, pad=1, labelsize=5.5, labelcolor=INK2)
        top.spines["top"].set_visible(False)
    if snow:
        ax.axvline(SNOW_LINE_AU, color=SNOW, ls="--", lw=0.8, zorder=1)


def completeness_plot(comp: pd.DataFrame, h_c: float, d_c: float):
    fig, ax = _fig(COL_W, 2.0)
    mids = [iv.mid for iv in comp.index]
    for z in ZONE_NAMES:
        ax.plot(mids, comp[z], color=ZONE_COLORS[z], lw=1.1, label=z)
    ax.axhline(COMPLETE_FRAC, color=INK2, ls=":", lw=.6)
    ax.axvline(h_c, color=COLORS["S-like"], lw=.8)
    ax.text(h_c + .15, .08, f"$H_c$ = {h_c}\n$D_c$ ≈ {d_c:.0f} km", color=COLORS["S-like"], fontsize=6)
    ax.set(xlabel="absolute magnitude H", ylabel="fraction with a published taxonomy", xlim=(5, 18.5), ylim=(0, 1.03))
    ax.legend(loc="lower left")
    return fig


def c_fraction_panels(samples: dict[str, pd.DataFrame], panel_keys: list[str], rng: np.random.Generator):
    """C-fraction vs a: raw, collapsed + IPW, collapsed + size-complete."""
    fig, axes = _fig(FULL_W, 2.3, ncols=3, sharey=True)
    for ax, k in zip(axes, panel_keys):
        s = samples[k]
        sparse = k != "raw"   # weighted / size-limited samples are thin at 0.05 AU
        wide = 0.1 if sparse else BIN_W
        c = c_fraction_curve(s, rng, "narrow", edges=np.arange(A_MIN, A_MAX + 1e-9, wide))
        ok = c.n >= (8 if sparse else 20)
        ax.fill_between(c.a[ok], c.lo[ok], c.hi[ok], color=COLORS["C-like"], alpha=.18, lw=0)
        ax.plot(c.a[ok], c.f[ok], color=COLORS["C-like"], lw=1.3, marker="o" if sparse else None, ms=2.2, mec="white", mew=.4)
        cb = c_fraction_curve(s, rng, "broad", edges=np.arange(A_MIN, A_MAX + 1e-9, wide), n_boot=20)
        ax.plot(cb.a[ok], cb.f[ok], color=COLORS["K/L"], lw=.9, ls="--")
        mark_structure(ax)
        ax.set_title(f"{k.replace('≥', ' ≥ ')}  (n = {len(s):,})", loc="left", fontsize=6.5, color=INK)
        ax.set(xlabel="semimajor axis a (AU)", ylim=(0, 1), xlim=(A_MIN, A_MAX))
    axes[0].set_ylabel("carbonaceous fraction")
    axes[0].plot([], [], color=COLORS["C-like"], lw=1.3, label="narrow, C/(S+C), 16–84% band")
    axes[0].plot([], [], color=COLORS["K/L"], lw=.9, ls="--", label="broad, (C+D/P+K/L)/(all but X)")
    axes[0].plot([], [], color=SNOW, lw=.8, ls="--", label=f"snow line, $T_{{eq}}$ = 170 K ({SNOW_LINE_AU:.2f} AU)")
    axes[0].legend(loc="upper left")
    return fig


def stacked_composition(s: pd.DataFrame, d_c: float):
    """All five groups for the size-complete, family-collapsed sample in 0.1 AU bins; bin counts on the top axis."""
    edges = np.arange(A_MIN, A_MAX + 1e-9, 0.1)
    s = s.assign(bin=pd.cut(s.semi_major_axis_au, edges))
    order = GROUP_ORDER
    st = s[s.group.isin(order)].pivot_table(index="bin", columns="group", values="w", aggfunc="sum", observed=False).reindex(columns=order).fillna(0)
    frac = st.div(st.sum(axis=1), axis=0).fillna(0)
    fig, ax = _fig(COL_W, 2.35)
    x = np.array([iv.mid for iv in frac.index])
    bottom = np.zeros(len(x))
    for g in order:
        ax.bar(x, frac[g], width=0.1, bottom=bottom, color=COLORS[g], edgecolor="white", linewidth=.6, label=g)
        bottom += frac[g].values
    mark_structure(ax, snow=False, labels=False)
    top = ax.secondary_xaxis("top")
    top.set_xticks(x, [f"{int(n)}" for n in st.sum(axis=1)])
    top.tick_params(length=0, pad=1, labelsize=5.5, labelcolor=INK2)
    top.spines["top"].set_visible(False)
    top.set_xlabel(f"bodies per bin (D ≥ {d_c:.0f} km, families collapsed)", fontsize=6, color=INK2, labelpad=2)
    ax.set(xlim=(A_MIN, A_MAX), ylim=(0, 1), xlabel="semimajor axis a (AU)", ylabel="fraction by number")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=len(order), handlelength=1.0, columnspacing=1.0)
    return fig


def mass_bars(panels: list[tuple[str, pd.DataFrame]]):
    fig, axes = _fig(FULL_W, 1.85, ncols=2, sharey=True)
    for ax, (lab, d) in zip(axes, panels):
        mz = mass_by_zone(d, GROUP_ORDER)
        fz = mz.div(mz.sum(axis=1), axis=0)
        left = np.zeros(len(fz))
        for g in GROUP_ORDER:
            ax.barh(ZONE_NAMES, fz[g], left=left, height=.72, color=COLORS[g], edgecolor="white", linewidth=.6, label=g)
            left += fz[g].values
        for i, z in enumerate(ZONE_NAMES):
            m, e = f"{mz.loc[z].sum():.1e}".split("e")
            ax.text(1.015, i, f"{m} × 10$^{{{int(e)}}}$ kg", va="center", fontsize=5.5, color=INK2)
        ax.set_title(lab, loc="left", fontsize=6.5)
        ax.set(xlim=(0, 1.2), xlabel="mass fraction")
        ax.set_xticks([0, .2, .4, .6, .8, 1])
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    fig.legend(*axes[1].get_legend_handles_labels(), loc="outside lower center", ncol=len(GROUP_ORDER), handlelength=1.0)
    return fig


def orbital_cdfs(orb: pd.DataFrame):
    fig, axes = _fig(FULL_W, 3.0, nrows=2, ncols=4)
    for j, z in enumerate(ZONE_NAMES):
        for i, (col, lab) in enumerate(ELEMENTS):
            S, C = split_by_class(orb, z, col)
            ax = axes[i, j]
            for arr, g in [(S, "S-like"), (C, "C-like")]:
                if len(arr):
                    ax.step(np.sort(arr), np.arange(1, len(arr) + 1) / len(arr), color=COLORS[g], lw=.9, where="post",
                            label=f"{g} (n = {len(arr):,})")
            if len(S) >= 10 and len(C) >= 10:
                p = stats.ks_2samp(S, C).pvalue
                ax.text(.03, .97, f"KS p = {p:.2g}", transform=ax.transAxes, ha="left", va="top", fontsize=5.5, color=INK2)
            ax.set(xlabel=lab.replace("proper ", "proper "), ylabel="cumulative fraction" if j == 0 else None, ylim=(0, 1.02))
            if i == 0:
                ax.set_title(z, loc="left", fontsize=6.5)
                ax.legend(loc="lower right", fontsize=5.5, handlelength=1.2)
    return fig


def dark_fraction(alb: pd.DataFrame, d_ca: float, rng: np.random.Generator):
    fig, ax = _fig(COL_W, 2.0)
    for lab, d, c in [("all with measured albedo", alb, "#9aa5a6"),
                      (f"background, D ≥ {d_ca:.0f} km (complete)", alb[~alb.in_family & (alb.diameter_km >= d_ca)], COLORS["C-like"])]:
        x, f, lo, hi, n = dark_curve(d, np.arange(A_MIN, A_MAX + 1e-9, 0.1 if "D ≥" in lab else BIN_W), rng)
        k = n >= 5
        ax.fill_between(x[k], lo[k], hi[k], color=c, alpha=.2, lw=0)
        ax.plot(x[k], f[k], color=c, lw=1.2, label=f"{lab} (n = {len(d):,})")
    mark_structure(ax)
    ax.set(xlabel="semimajor axis a (AU)", ylabel="fraction with $p_V$ < 0.10", ylim=(0, 1), xlim=(A_MIN, A_MAX))
    ax.legend(loc="lower right")
    return fig


def circularity(assumed: pd.DataFrame, raw: pd.DataFrame, rng: np.random.Generator):
    """The assumed-albedo tier returns the step function it was built from."""
    fig, ax = _fig(COL_W, 1.95)
    for lab, d, c in [("assumed-albedo labels (circular)", assumed, "#e67e22"),
                      ("published taxonomy, raw", raw, COLORS["C-like"])]:
        cc = c_fraction_curve(d, rng, "narrow", n_boot=50)
        ax.plot(cc.a, cc.f, color=c, lw=1.2, label=f"{lab} (n = {len(d):,})")
    mark_structure(ax, snow=False)
    ax.set(xlabel="semimajor axis a (AU)", ylabel="C/(S+C)", ylim=(-.02, 1.02), xlim=(A_MIN, A_MAX))
    ax.legend(loc="lower right")
    return fig


# ── figures that draw nothing from the RNG ──────────────────────────────────

def size_distance_map(s: pd.DataFrame):
    """C/(S+C) of the completeness-weighted, family-collapsed sample in bins of a (0.1 AU) and D.

    Cells with fewer than ``MIN_CELL`` S- and C-type bodies are left blank (a dot marks one that has any).
    """
    a_edges, _, f, n = size_distance_grid(s)
    shape = f.shape
    cmap = LinearSegmentedColormap.from_list("s_to_c", [COLORS["S-like"], "#f0efec", COLORS["C-like"]])
    fig, ax = _fig(COL_W, 2.15)
    m = ax.pcolormesh(a_edges, np.arange(shape[0] + 1), np.ma.masked_invalid(f), cmap=cmap, norm=TwoSlopeNorm(.5, 0, 1),
                      edgecolors="white", linewidth=1.0)
    for j in range(shape[0]):
        for i in range(shape[1]):
            xc, yc = (a_edges[i] + a_edges[i + 1]) / 2, j + .5
            if n[j, i] >= MIN_CELL:
                ax.text(xc, yc, f"{int(n[j, i])}", ha="center", va="center", fontsize=4.8,
                        color="white" if abs(f[j, i] - .5) > .28 else INK2)
            elif n[j, i] > 0:
                ax.plot(xc, yc, marker=".", ms=1.5, color=RULE)
    top = ax.secondary_xaxis("top")
    gaps = {k: a for k, a in KIRKWOOD.items() if A_MIN < a < A_MAX}
    top.set_xticks(list(gaps.values()) + [SNOW_LINE_AU], list(gaps) + ["snow\nline"])
    top.tick_params(length=2.5, width=.6, pad=1, labelsize=5.5, labelcolor=INK2, color=INK2)
    top.spines["top"].set_visible(False)
    ax.set_yticks(np.arange(shape[0]) + .5, ["10–20", "20–50", "50–100", "≥ 100"])
    ax.tick_params(axis="y", length=0)
    ax.set(xlabel="semimajor axis a (AU)", ylabel="diameter D (km)", xlim=(A_MIN, A_MAX))
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    cb = fig.colorbar(m, ax=ax, pad=.02, fraction=.06, aspect=18, ticks=[0, .25, .5, .75, 1])
    cb.set_label("C/(S+C), weighted", fontsize=6)
    cb.ax.tick_params(labelsize=5.5, width=.5, length=2)
    cb.outline.set_linewidth(.5)
    return fig


SAMPLE_COLORS = ["#8e8e8e", "#2a78d6", "#eb6834", "#1baf7a"]   # raw in gray; then the reference palette's slots 1-3


def crossover_fits(xo: pd.DataFrame):
    """The fitted logistic P(C | a) of every sample, dotted outside the belt, with a50 and its 16–84% interval below."""
    fig, (ax, ax2) = _fig(COL_W, 2.75, nrows=2, sharex=True, gridspec_kw={"height_ratios": [3, 1.15]})
    x = np.linspace(1.4, 4.0, 521)
    inb = (x >= A_MIN) & (x <= A_MAX)
    labels = [k.replace("≥", " ≥ ") for k in xo.index]
    for ax_ in (ax, ax2):
        ax_.axvspan(A_MIN, A_MAX, color=BAND, lw=0, zorder=0)
        ax_.axvline(SNOW_LINE_AU, color=SNOW, lw=.8, ls="--", zorder=1)
    for i, (k, row) in enumerate(xo.iterrows()):
        b1 = 4.394 / float(row.width_10_90)
        p = 1 / (1 + np.exp(-b1 * (x - float(row.a50))))
        c, lw = SAMPLE_COLORS[i], (1.0 if i == 0 else 1.3)
        ax.plot(x[inb], p[inb], color=c, lw=lw, ls=(0, (3.5, 1.8)) if i == 0 else "-", label=labels[i], zorder=3)
        for part in (x < A_MIN, x > A_MAX):
            ax.plot(x[part], p[part], color=c, lw=.7, ls=":", zorder=2)
        y = len(xo) - 1 - i
        ax2.plot([row.a50_lo, row.a50_hi], [y, y], color=c, lw=1.6, solid_capstyle="butt", zorder=3)
        ax2.plot(row.a50, y, "o", ms=3.2, color=c, mec="white", mew=.5, zorder=4)
    for yy in (.1, .5, .9):
        ax.axhline(yy, color="#dcdbd6", lw=.5, zorder=1)
    ax.text((A_MIN + A_MAX) / 2, 1.0, "main belt", ha="center", va="bottom", fontsize=5.5, color=INK2, transform=ax.get_xaxis_transform())
    ax.text(SNOW_LINE_AU + .03, .02, "snow line", fontsize=5.5, color=SNOW, va="bottom")
    ax.set(ylim=(0, 1), ylabel="fitted P(C | a)")
    ax.set_yticks([0, .1, .5, .9, 1])
    ax.legend(loc="upper left", fontsize=5.5)
    ax2.set_yticks(range(len(xo)), ["raw", "IPW, D ≥ 10", "D ≥ 53", "background"][::-1] if len(xo) == 4 else labels[::-1])
    ax2.tick_params(axis="y", length=0, labelsize=5.5)
    ax2.set(xlim=(1.4, 4.0), ylim=(-.6, len(xo) - .4), xlabel="semimajor axis a (AU)")
    ax2.set_ylabel("$a_{50}$", fontsize=6)
    return fig


FAMILY_LABELS = {"Vesta": (0, -.035), "Flora": (-.03, -.05), "Nysa-Polana": (0, -.032), "Eunomia": (0, .055),
                 "Koronis": (0, -.03), "Eos": (0, .05), "Themis": (0, .045), "Hygiea": (0, .045)}


def proper_element_map(mb: pd.DataFrame, orb: pd.DataFrame, ftab: pd.DataFrame):
    """Background S and C bodies of the orbit test over the density of all family members, in (a_P, sin i_P)."""
    fam = mb[mb.in_family & mb.a_p.notna()]
    top_i = max(0.42, float(orb.sini_p.max()) + .01)
    fig, ax = _fig(COL_W, 2.6)
    grey = LinearSegmentedColormap.from_list("fam", ["#f4f3f0", "#a8a7a1"])
    ax.hexbin(fam.a_p, fam.sini_p, gridsize=(120, 70), extent=(A_MIN, A_MAX, 0, top_i), norm=LogNorm(), cmap=grey,
              mincnt=1, linewidths=0, zorder=0)
    for g in ["S-like", "C-like"]:
        d = orb[orb.group.eq(g)]
        ax.scatter(d.a_p, d.sini_p, s=1.6, color=COLORS[g], lw=0, alpha=.9, zorder=2,
                   label=f"{g} (n = {len(d):,})")
    for name, (dx, dy) in FAMILY_LABELS.items():
        ids = ftab.index[ftab.fam_name.eq(name)]
        if len(ids):
            m = fam[fam.fam_id.eq(ids[0])]
            if len(m):
                ax.text(m.a_p.median() + dx, m.sini_p.median() + dy, name.replace("-", "–"), fontsize=5.2, color=INK,
                        ha="center", va="center", zorder=3,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=.75))
    mark_structure(ax, snow=False)
    ax.set(xlim=(A_MIN, A_MAX), ylim=(0, top_i), xlabel="proper semimajor axis $a_P$ (AU)", ylabel="proper sin $i_P$")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=3, markerscale=3.5, handletextpad=.3,
               title=f"background bodies, D ≥ {D_ORBIT:.0f} km:", title_fontsize=6, alignment="left")
    return fig


def _cell(s: str):
    lo_hi = re.match(r"([\d.]+) \[([\d.]+)–([\d.]+)\]", s)
    return tuple(float(v) for v in lo_hi.groups())


def robustness_summary(summary: dict, keys: list[str]):
    """Every per-zone estimate of the carbonaceous fraction the paper reports, one row each, zones as markers."""
    nar, bro = summary["zone_c_fraction_narrow"], summary["zone_c_fraction_broad"]
    ext = summary["family_extension_toggled"]["zone_c_fraction_narrow"]
    xs = summary["x_split_by_albedo"]["zone_c_fraction_broad"]
    hl = summary["text_numbers"]["H_limited_IPW_c_fraction_by_zone"]
    dark = summary["dark_fraction_complete"]
    raw, ipw, col, bg = keys
    point = lambda v: {z: (v[z], None, None) for z in ZONE_NAMES}
    rows = [("raw count", {z: _cell(nar[z][raw]) for z in ZONE_NAMES}, True),
            ("collapsed, IPW, D ≥ 10 km", {z: _cell(nar[z][ipw]) for z in ZONE_NAMES}, True),
            ("collapsed, D ≥ 53 km", {z: _cell(nar[z][col]) for z in ZONE_NAMES}, True),
            ("background, D ≥ 53 km", {z: _cell(nar[z][bg]) for z in ZONE_NAMES}, True),
            ("IPW, H < 15 (brightness-limited)", point(hl), False),
            ("IPW, D ≥ 10 km, families extended", {z: _cell(ext[z][ipw]) for z in ZONE_NAMES}, False),
            ("D ≥ 53 km, families extended", {z: _cell(ext[z][col]) for z in ZONE_NAMES}, False),
            ("broad, IPW, D ≥ 10 km", {z: _cell(bro[z][ipw]) for z in ZONE_NAMES}, False),
            ("broad, D ≥ 53 km", {z: _cell(bro[z][col]) for z in ZONE_NAMES}, False),
            ("broad, dark X as P, IPW", point({z: xs[ipw][z] for z in ZONE_NAMES}), False),
            ("broad, dark X as P, D ≥ 53 km", point({z: xs[col][z] for z in ZONE_NAMES}), False),
            ("albedo only, $p_V$ < 0.10, D ≥ 33 km", point(dark), False)]
    fig, ax = _fig(COL_W, 3.15)
    ys, labs, main = [], [], []
    y = 0.0
    for i, (lab, vals, is_main) in enumerate(rows):
        if i == 4:
            y += .7                                   # a gap between the four samples and the checks
        ys.append(-y); labs.append(lab); main.append(is_main)
        if i % 2 == 0:
            ax.axhspan(-y - .5, -y + .5, color=BAND, lw=0, zorder=0)
        for z in ZONE_NAMES:
            f, lo, hi = vals[z]
            if f is None:
                continue
            if lo is not None:
                ax.plot([lo, hi], [-y, -y], color=ZONE_COLORS[z], lw=1.1, zorder=2)
            ax.plot(f, -y, marker=ZONE_MARKERS[z], ms=3.3, color=ZONE_COLORS[z], mec="white", mew=.45, ls="none", zorder=3)
        y += 1
    ax.axhline(-3.5 - .35, color=RULE, lw=.5)
    ax.set_yticks(ys, labs, fontsize=5.8)
    for t, is_main in zip(ax.get_yticklabels(), main):
        t.set_color(INK if is_main else INK2)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set(xlim=(0, 1), ylim=(ys[-1] - .6, .6), xlabel="carbonaceous fraction")
    ax.set_xticks([0, .2, .4, .6, .8, 1])
    ax.xaxis.grid(True, color="#e6e5e1", lw=.5)
    ax.set_axisbelow(True)
    for z in ZONE_NAMES:
        ax.plot([], [], marker=ZONE_MARKERS[z], ms=3.3, color=ZONE_COLORS[z], ls="none", label=z)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=4, handletextpad=.2, columnspacing=1.0)
    return fig


# ── context figures: the snow line through time, how the samples are built, albedo by class (no RNG) ─────────

def snow_line_track(track, crossover: tuple[float, float]):
    """Equation (1) scaled with the 1 M_sun pre-main-sequence luminosity (r ∝ L^1/2), against the present belt."""
    t, logl = np.array(track, dtype=float).T
    age = 10 ** t / 1e6
    r = SNOW_LINE_AU * 10 ** (logl / 2)
    fig, ax = _fig(COL_W, 2.1)
    ax.axhspan(A_MIN, A_MAX, color=BAND, lw=0, zorder=0)
    ax.axhspan(*crossover, color=COLORS["C-like"], alpha=.18, lw=0, zorder=1)
    ax.axhline(SNOW_LINE_AU, color=SNOW, lw=.8, ls="--", zorder=1)
    ax.plot(age, r, color=COLORS["C-like"], lw=1.4, zorder=3)
    for a_myr in (1.0, 5.0):
        k = int(np.argmin(abs(age - a_myr)))
        ax.plot(age[k], r[k], "o", ms=3.2, color=COLORS["C-like"], mec="white", mew=.5, zorder=4)
        ax.annotate(f"{r[k]:.1f} AU at {age[k]:.0f} Myr", (age[k], r[k]), xytext=(5, 3), textcoords="offset points",
                    fontsize=5.5, color=INK2)
    ax.set_xscale("log")
    ax.set(xlim=(age[0], age[-1]), ylim=(1.5, 5.0), xlabel="age of a 1 $M_\\odot$ star (Myr)", ylabel="snow-line radius (AU)")
    ax.set_xticks([.5, 1, 2, 5, 10, 20, 50], ["0.5", "1", "2", "5", "10", "20", "50"])
    x0 = age[0] * 1.06
    ax.text(age[-1] * .97, A_MAX - .05, "present main belt", ha="right", va="top", fontsize=5.5, color=INK2)
    ax.text(x0, SNOW_LINE_AU + .04, "$L = L_\\odot$ (Equation 1)", fontsize=5.5, color=SNOW, va="bottom")
    ax.text(x0, crossover[0] - .04, "debiased S/C crossover", va="top", fontsize=5.5, color=COLORS["C-like"])
    return fig


def _box(ax, x0, y0, x1, y1, lines, focal=False, muted=False):
    from matplotlib.patches import FancyBboxPatch
    fc, ec, ls = ("#e8f1fb", "#2a78d6", "-") if focal else (("#f7f6f3", RULE, (0, (3, 2))) if muted else ("white", INK, "-"))
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec,
                                lw=.7, ls=ls, zorder=2))
    head, *rest = lines
    cy = (y0 + y1) / 2
    n = len(lines)
    step = 3.6
    top = cy + step * (n - 1) / 2
    ax.text((x0 + x1) / 2, top, head, ha="center", va="center", fontsize=5.8, fontweight="bold",
            color=INK2 if muted else INK, zorder=3)
    for i, s in enumerate(rest, 1):
        ax.text((x0 + x1) / 2, top - i * step, s, ha="center", va="center", fontsize=5.5, color=INK2, zorder=3)


def _arrow(ax, pts):
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path
    ax.add_patch(FancyArrowPatch(path=Path(pts), arrowstyle="-|>", mutation_scale=5.5, lw=.7, color=INK2,
                                 shrinkA=0, shrinkB=0, zorder=1))


def sample_flow(summary: dict, keys: list[str]):
    """How the analysed samples are built from the catalog release, with the number of bodies at each step."""
    c, tn, ss = summary["catalog"], summary["text_numbers"], summary["sample_sizes"]
    tiers = tn["n_main_belt_by_label_tier"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.75))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 100), ylim=(0, 100)); ax.axis("off")
    _box(ax, 8, 86, 92, 99, [f"AsteroidCatalog {c['release']}: {c['rows']:,} bodies",
                             f"main belt (2.1 < a < 3.3 AU, q > 1.3 AU): {c['main_belt']:,}"])
    _box(ax, 1, 62, 33, 77, ["published taxonomy", f"{tiers['taxonomy']:,}", "the labels analysed"], focal=True)
    _box(ax, 35, 62, 65, 77, ["measured albedo", f"{tiers['albedo_proxy']:,}", "labels not used"], muted=True)
    _box(ax, 67, 62, 99, 77, ["assumed albedo", f"{tiers['assumed']:,}", "circular (Fig. 2)"], muted=True)
    for x in (17, 50, 83):
        _arrow(ax, [(x, 86), (x, 77)])
    fam = tn["families_with_main_belt_members"], tn["families_classifiable"]
    _box(ax, 12, 39, 76, 54, [f"collisional families ({fam[0]} in the main belt)",
                              f"{fam[1]} classified: each collapsed to one body,", "or its members removed (background)"])
    _arrow(ax, [(22, 62), (22, 54)])
    w, y0, y1 = 23.5, 2, 21
    xs = [1, 26, 51, 76]
    labels = [("raw", "families kept", f"n = {ss['raw']:,}"),
              ("collapsed, IPW", "D ≥ 10 km", f"n = {ss['collapsed_ipw']:,}"),
              ("collapsed", f"D ≥ {summary['completeness']['D_complete_km']:.0f} km", f"n = {ss['collapsed_complete']:,}"),
              ("background", f"D ≥ {summary['completeness']['D_complete_km']:.0f} km", f"n = {ss['background_complete']:,}")]
    for x0, lab in zip(xs, labels):
        _box(ax, x0, y0, x0 + w, y1, list(lab))
    _arrow(ax, [(6, 62), (6, y1)])                                  # raw: straight from the labelled bodies
    _arrow(ax, [(37.75, 39), (37.75, y1)])                          # collapsed, IPW
    _arrow(ax, [(58, 39), (58, 32), (62.75, 32), (62.75, y1)])      # collapsed, size-complete
    _arrow(ax, [(76, 46.5), (87.75, 46.5), (87.75, y1)])            # background
    return fig


ALBEDO_GROUPS = ["S-like", "C-like", "D/P", "X"]


def albedo_by_group(mb: pd.DataFrame, cuts: tuple[float, float]):
    """Measured geometric albedo of the bodies with a published taxonomy, by group (normalised histograms)."""
    t = mb[mb.tier.eq("taxonomy") & mb.albedo.notna() & (mb.albedo > 0)]
    bins = np.logspace(-2, 0, 61)
    fig, ax = _fig(COL_W, 2.0)
    for g in ALBEDO_GROUPS:
        a = t.loc[t.group.eq(g), "albedo"].clip(1e-2, .999)
        ax.hist(a, bins=bins, density=True, histtype="step", lw=1.0, color=COLORS[g], label=f"{g} (n = {len(a):,})")
    ax.axvline(cuts[0], color=INK2, lw=.7)
    ax.axvline(cuts[1], color=INK2, lw=.6, ls=":")
    ax.set_xscale("log")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.text(cuts[0] * .96, .99, f"dark | bright ({cuts[0]:.2f})", transform=ax.get_xaxis_transform(), ha="right", va="top",
            fontsize=5.5, color=INK2)
    ax.text(cuts[1] * 1.04, .99, f"X: M | E ({cuts[1]:.2f})", transform=ax.get_xaxis_transform(), va="top", fontsize=5.5,
            color=INK2)
    ax.set_xticks([.01, .02, .05, .1, .2, .5, 1], ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1"])
    ax.set(xlim=(.01, 1), xlabel="measured geometric albedo $p_V$", ylabel="probability density")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.legend(loc="center right", handlelength=1.2)
    return fig

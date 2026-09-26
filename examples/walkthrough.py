"""Step through the analysis and print the intermediate tables behind the paper.

    python examples/walkthrough.py

Runs the same pipeline as ``belt-gradient run`` without overwriting figures/ or results/,
then prints each stage. Inputs come from ``belt-gradient fetch``.
"""
import matplotlib

matplotlib.use("Agg")

import pandas as pd

from beltgradient.config import Paths
from beltgradient.pipeline import FIGURE_FILES, run

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def section(title, note=None):
    print(f"\n{'─' * 100}\n{title}")
    if note:
        print(note)
    print()


def figure(key):
    print(f"[figure: figures/{FIGURE_FILES[key][0]}]")


def main():
    paths = Paths.default()
    res = run(paths, write=False)
    T, S = res.tables, res.summary

    section("Label provenance",
            "Only `taxonomy` (published classifications, mostly SkyMapper and SDSS colours) is used. `assumed`\n"
            "labels come from an albedo chosen by semimajor axis, so they would return the gradient by construction.")
    print(res.frames["main_belt"].tier.value_counts().to_string())
    figure("circularity")

    section("Collisional families",
            "Each family collapses to one body: the most common class of >= 3 labelled members, X-types not\n"
            "voting; with fewer labels, or a tie, the largest member's label. `purity` = fraction of labelled\n"
            "members in that class.")
    cols = ["fam_name", "n_members", "D_equiv_km", "a_rep", "fam_group", "purity", "n_labelled", "largest"]
    print(T["families"].head(15)[cols].round(3).to_string())

    section("Completeness", "The faintest H at which every zone is >= 95% classified sets the size-complete limit.")
    print(S["completeness"])
    figure("completeness")

    section("C-fraction versus semimajor axis: narrow, C/(S+C)")
    print(T["zone_narrow"].to_string())
    section("C-fraction versus semimajor axis: broad, (C+D/P+K/L)/(all but X)")
    print(T["zone_broad"].to_string())
    figure("c_fraction")
    figure("stacked")

    section("Inner belt by size and by mass (compare DeMeo & Carry 2014)",
            "Families included. The 5–20 km bin is not debiased, so its C-fraction is a lower limit.")
    print(T["inner_by_size"].round(3).to_string())
    print("\nmass-weighted C/(S+C) by zone:", S["mass_c_fraction"])
    figure("mass")

    section("S/C crossover",
            f"Logistic fit P(C | a); a50 and 10->90% width with 16–84% bootstrap intervals. "
            f"Snow line (T_eq = 170 K): {S['snow_line_au']} AU.")
    print(T["crossover"].astype(float).round(3).to_string())

    section("Orbital excitation", "Background bodies with D >= 10 km; KS per zone and element, Bonferroni over 8 tests.")
    print(T["orbit_tests"].round(4).to_string())
    figure("orbits")

    section("Robustness: label-free dark fraction (measured p_V < 0.10)")
    print(T["dark_zone"].round(3).to_string())
    figure("dark")

    ext = S["family_extension_toggled"]
    section("Robustness: one-step family extension (aggressive upper bound on missed members)")
    print(f"family fraction {S['families']['family_fraction']:.3f} -> {ext['family_fraction']:.3f}")
    print(f"max change in a zone fraction: {ext['max_abs_change_zone_fraction']}, in a50: {ext['max_abs_change_a50_au']} AU\n")
    print(T["zone_narrow_extended"].to_string())
    print("\norbit test on the extended background (KS only):")
    print(T["orbit_ks_extended"].round(4).to_string())


if __name__ == "__main__":
    main()

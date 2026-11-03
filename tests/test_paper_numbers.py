"""The paper agrees with the committed results: every number in Tables 2-5, every data-derived number in the running
text, the bounds and comparisons the text states, the size cuts it names, and which file each figure number shows.
No data or LaTeX needed.

A run that moves a published number changes results/summary.json; these tests then fail until the paper says the same.
"""
import json
import re
from pathlib import Path

import pytest

from beltgradient.config import D_IPW, D_ORBIT
from beltgradient.pipeline import FIGURE_FILES

ROOT = Path(__file__).resolve().parents[1]
TEX = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
SUMMARY = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
KEYS = list(SUMMARY["crossover"])                      # the four samples, in the order the tables list them
ZONES = ["inner", "middle", "pristine", "outer"]
NUM = r"-?\d[\d,]*(?:\.\d+)?"


def table(label: str) -> list[list[str]]:
    """The cells of the deluxetable carrying ``label``, row by row, with TeX minus signs and math marks removed."""
    m = re.search(r"\\begin\{deluxetable\*?\}(?:(?!\\end\{deluxetable).)*?\\label\{" + label + r"\}.*?\\startdata(.*?)\\enddata",
                  TEX, re.S)
    assert m, f"no deluxetable labelled {label}"
    body = m.group(1).replace("\\ensuremath{-}", "-").replace("$", "").replace("~", " ")
    return [[c.strip() for c in row.split("&")] for row in body.split("\\\\") if row.strip()]


def same(printed: str, value: float) -> bool:
    """``printed`` is ``value`` rounded to the digits it shows (the summary itself is rounded to 4-5 decimals)."""
    d = len(printed.split(".")[1]) if "." in printed else 0
    return abs(float(printed.replace(",", "")) - value) <= 0.5 * 10 ** -d + 1e-6


def numbers(cell: str) -> list[str]:
    return re.findall(NUM, cell)



def check_row_names_sample(row_label: str, key: str):
    """'Collapsed, IPW, $D \\geq 10$~km' names the sample 'collapsed, IPW, D≥10 km'."""
    label = re.sub(r"\s+", " ", row_label.replace("\\geq", "≥")).strip().lower()
    assert label == key.lower().replace("≥", " ≥ "), (row_label, key)


# ── tables ───────────────────────────────────────────────────────────────────────────────────────────

def test_table_2_is_the_narrow_zone_fractions():
    rows = table("tab:2")
    assert len(rows) == len(KEYS)
    for row, key in zip(rows, KEYS):
        check_row_names_sample(row[0], key)
        for zone, cell in zip(ZONES, row[1:]):
            f, lo, hi, n = re.fullmatch(r"([\d.]+) \[([\d.]+)–([\d.]+)\] \(n=(\d+)\)",
                                        SUMMARY["zone_c_fraction_narrow"][zone][key]).groups()
            assert numbers(cell) == [f, lo, hi, f"{int(n):,}"], (key, zone, cell)


def test_table_3_is_the_inner_belt_by_size():
    rows = table("tab:3")
    recs = SUMMARY["inner_belt_by_size"]
    assert len(rows) == len(recs)
    for row, r in zip(rows, recs):
        size = row[0].replace("\\geq ", ">= ").replace("–", "-")
        assert re.sub(r"\s+", " ", size) == r["size"], (row[0], r["size"])
        assert [int(c.replace(",", "")) for c in row[1:3]] == [r["n_S"], r["n_C"]], r["size"]
        assert same(row[3], r["C_frac_number"]) and same(row[4], r["C_frac_mass"]), (row, r)


def test_table_4_is_the_crossover():
    rows = table("tab:4")
    assert len(rows) == len(KEYS) + 1
    for row, key in zip(rows, KEYS):
        check_row_names_sample(row[0], key)
        x = SUMMARY["crossover"][key]
        a50, a_lo, a_hi = numbers(row[1])
        w, w_lo, w_hi = numbers(row[2])
        for printed, v in [(a50, x["a50"]), (a_lo, x["a50_lo"]), (a_hi, x["a50_hi"]), (w, x["width_10_90"]),
                           (w_lo, x["width_lo"]), (w_hi, x["width_hi"])]:
            assert same(printed, v), (key, printed, v)
        assert int(row[3].replace(",", "")) == x["n"], key
    assert rows[-1][0].startswith("Snow line") and same(rows[-1][1], SUMMARY["snow_line_au"])


def test_table_5_is_the_orbit_test():
    rows = table("tab:5")
    recs = SUMMARY["orbit_tests"]
    assert len(rows) == len(recs) == 8
    zone = None
    for row, r in zip(rows, recs):
        zone = row[0].lower() or zone                  # a zone is named on its first row only
        element = "proper sin i" if "sin i" in row[1] else "proper e"
        assert (zone, element) == (r["zone"], r["element"]), row
        assert [int(c.replace(",", "")) for c in row[2:4]] == [r["n_S"], r["n_C"]], row
        d, lo, hi = numbers(row[4])
        for printed, v in [(d, r["dmedian_C_minus_S"]), (lo, r["ci95_lo"]), (hi, r["ci95_hi"]), (row[5], r["KS_D"]),
                           (row[6], r["KS_p"])]:
            assert same(printed, v), (zone, element, printed, v)
    # the comment names the Bonferroni threshold and the one test that meets it
    n = len(recs)
    assert f"p < {0.05 / n:g}" in TEX
    assert [(r["zone"], r["element"]) for r in recs if r["bonferroni_sig"]] == [("outer", "proper sin i")]
    assert "only the outer-belt inclinations meet it" in TEX


# ── numbers and cuts quoted in the text ─────────────────────────────────────────────────────────────

def _quoted():
    c, t, s = SUMMARY["catalog"], SUMMARY["text_numbers"], SUMMARY["sample_sizes"]
    tiers = t["n_main_belt_by_label_tier"]
    return {"catalog rows": f"{c['rows']:,}", "main belt": f"{c['main_belt']:,}",
            "published taxonomy": f"{tiers['taxonomy']:,}", "assumed-albedo labels": f"{tiers['assumed']:,}",
            "measured-albedo labels": f"{tiers['albedo_proxy']:,}", "families in the bundle": str(SUMMARY["families"]["n_families"]),
            "families with main-belt members": str(t["families_with_main_belt_members"]),
            "families classified": str(t["families_classifiable"]),
            "size-complete collapsed sample": f"{s['collapsed_complete']:,}",
            "albedo check sample": f"{s['albedo_check_complete_background']:,}",
            "H_c": f"H_{{\\mathrm{{c}}}} = {SUMMARY['completeness']['H_c']}", "D_c": f"D_{{\\mathrm{{c}}}} = {SUMMARY['completeness']['D_complete_km']}$~km"}


@pytest.mark.parametrize("what", list(_quoted()))
def test_the_text_quotes_the_summary(what):
    assert _quoted()[what] in TEX, f"{what}: the paper does not quote {_quoted()[what]!r}, the committed summary's value"


def test_every_size_cut_the_paper_names_is_one_the_run_used():
    comp = SUMMARY["completeness"]
    used = {f"{D_IPW:.0f}", f"{D_ORBIT:.0f}", f"{comp['D_complete_km']:.0f}", f"{comp['D_complete_albedo_km']:.0f}"}
    named = set(re.findall(r"D \\geq (\d+)\$?~km", TEX))
    assert named and named <= used, f"the paper names D ≥ {sorted(named - used)} km, which no sample uses"


# ── figures: the file behind each figure number ──────────────────────────────────────────────────────

def test_each_figure_number_shows_the_file_the_pipeline_says():
    found = {}
    for env in re.findall(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", TEX, re.S):
        files = re.findall(r"\\includegraphics\[[^\]]*\]\{figures/([^}]+)\}", env)
        labels = re.findall(r"\\label\{fig:([A-Z]?\d+)\}", env)
        assert len(files) == len(labels) == 1, env[:200]
        found[files[0]] = labels[0]
    assert found == {name: number for name, number in FIGURE_FILES.values()}


# ── every other number the running text quotes ───────────────────────────────────────────────────────
# Each phrase below is rebuilt from the committed summary at the digits the paper prints, and must appear in the
# paper word for word. A run that moves one of these numbers, or a rewrite of the sentence, fails here until the
# two agree again.

WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def pct(v: float) -> str:
    """A share as the paper prints it, a whole percentage: 0.3899 -> 39\\%."""
    return f"{100 * v:.0f}\\%"


def p2(p: float) -> str:
    """A KS p-value to two significant figures, as Table 5 prints it: 0.0060, 0.44."""
    return f"{p:#.2g}"


def span(values, fmt="{:.2f}") -> str:
    """'lo–hi' over ``values``, each end printed with ``fmt``."""
    return f"{fmt.format(min(values))}–{fmt.format(max(values))}"


def zone_f(key: str, zone: str) -> float:
    return float(SUMMARY["zone_c_fraction_narrow"][zone][key].split(" ")[0])


def test_the_formats_print_as_the_paper_does():
    assert span([0.39, 0.53, 0.46]) == "0.39–0.53" and span([1.39, 1.38, 1.49], "{:.1f}") == "1.4–1.5"
    assert p2(0.006) == "0.0060" and p2(0.4412) == "0.44" and pct(0.0853) == "9\\%"


def _prose() -> dict[str, str]:
    S, t, fam = SUMMARY, SUMMARY["text_numbers"], SUMMARY["families"]
    xs, ext, mc, ib = S["x_split_by_albedo"], S["family_extension_toggled"], S["mass_c_fraction"], S["inner_belt_by_size"]
    raw, debiased = KEYS[0], KEYS[1:]
    a50 = {k: S["crossover"][k]["a50"] for k in KEYS}
    a50_span = span([a50[k] for k in debiased])
    width_span = span([S["crossover"][k]["width_10_90"] for k in debiased], "{:.1f}")
    deb = lambda zone: span([zone_f(k, zone) for k in debiased])
    ot = {(r["zone"], r["element"]): r for r in S["orbit_tests"]}
    steps = [p for p, _ in sorted(t["assumed_tier_albedo_by_a"].items(), key=lambda kv: kv[1]["a_min"])]
    inner_step_edge = t["assumed_tier_albedo_by_a"][steps[0]]["a_max"]
    top4, ipw_inner = t["mass_share_of_classified"], t["innermost_c_fraction_ipw"]
    n_lo, n_hi = sorted(t["innermost_n_S_plus_C_size_complete_2.1-2.3"].values())
    big, second = t["flora_family_two_largest"]
    dark = [S["dark_fraction_complete"][z] for z in ZONES]
    no4 = [mc["no_big4"][z] for z in ZONES]
    p_share = xs["P_share_of_split_by_zone"]
    snow, track = t["snow_line_au_on_bhac15_1msun_track"], S["snow_line_track"]["first_inside_au_myr"]
    comp14, dp = t["completeness_H13.5_to_14.5_range"], t["dp_share_per_0.1AU_bin_from_2.9AU"]
    n_dropped = t["ipw_D10_bodies_dropped_by_bin_cuts"]
    n_inner_complete = re.search(r"n=(\d+)", S["zone_c_fraction_narrow"]["inner"][KEYS[2]]).group(1)
    sl = S["snow_line_au"]
    return {
        # abstract and conclusions
        "raw crossover (abstract)": f"Uncorrected, the S/C crossover lies at {a50[raw]:.2f}~au",
        "debiased crossover (abstract)": f"$a_{{50}} = {a50_span.replace('–', '$–')}~au, with a transition {width_span}~au wide",
        "assumed share (abstract)": f"{pct(t['assumed_label_fraction'])} of the main belt",
        "zone fractions (conclusions)": f"is {deb('inner')} in the inner belt and {deb('middle')} in the middle belt, depending on "
                                        f"the sample, well above the raw values of {zone_f(raw, 'inner'):.2f} and {zone_f(raw, 'middle'):.2f}",
        "crossover (conclusions)": f"The S/C crossover lies at $a_{{50}} = {a50_span.replace('–', '$–')}~au, and the transition is "
                                   f"{width_span}~au wide",
        "raw crossover (conclusions)": f"The raw catalog’s crossover at {a50[raw]:.2f}~au",
        # section 2.2, Figure 1
        "snow line at 1 and 5 Myr": f"from about {snow['1.0 Myr']:.1f}~au at 1~Myr to about {snow['5.1 Myr']:.1f}~au at 5~Myr",
        "snow line crossing the belt": f"inside the belt’s outer edge (3.3~au) at {track['3.3']:.1f}~Myr and inside its inner edge "
                                      f"(2.1~au) at {track['2.1']:.1f}~Myr",
        # section 3
        "assumed tier": f"({t['n_main_belt_by_label_tier']['assumed']:,} bodies, {pct(t['assumed_label_fraction'])} of the belt)",
        "assumed albedo table": f"{steps[0]} inside {inner_step_edge:.1f}~au; {steps[1]}, {steps[2]} and {steps[3]} beyond it",
        "Vesta family": f"The Vesta family alone contributes over {t['vesta_family_members'] // 1000 * 1000:,} catalogued members",
        "family members": f"In total, {fam['members_matched_main_belt']:,} main-belt bodies ({100 * fam['family_fraction']:.1f}\\%) are "
                          f"family members, but among bodies with a published taxonomy the fraction is "
                          f"{100 * t['taxonomy_tier_family_fraction']:.1f}\\%",
        "completeness near H = 14": f"falls to about {100 * comp14[0]:.0f}–{pct(comp14[1])} by $H \\approx 14$",
        "IPW bin cuts": "which removes a single body" if n_dropped == 1 else f"which removes {n_dropped:,} bodies",
        "brightness-limited IPW": f"gives an inner-belt C-fraction of only {t['H_limited_IPW_c_fraction_by_zone']['inner']:.2f}",
        "H of a dark 10 km body": f"has $H \\approx {t['H_of_10km_dark_body']:.1f}$",
        # section 4.1
        "raw zone fractions": f"It gives a narrow C-fraction of {zone_f(raw, 'inner'):.2f} in the inner belt and "
                              f"{zone_f(raw, 'middle'):.2f} in the middle belt",
        "debiased zone fractions": f"those values rise to {deb('inner')} in the inner belt and {deb('middle')} in the middle belt",
        "small complete inner sample": f"where the complete sample is small ({n_inner_complete} bodies)",
        "innermost IPW": f"The weighted C-fraction is {ipw_inner['2.1-2.2']:.2g} at 2.1–2.2~au and {ipw_inner['2.2-2.3']:.2f} at 2.2–2.3~au",
        "innermost complete counts": "the complete samples contain only "
                                     + (WORDS[n_lo] if n_lo == n_hi else f"{WORDS[n_lo]} or {WORDS[n_hi]}") + " bodies there",
        "broad minus narrow (4.1)": f"is higher by {span(t['broad_minus_narrow_range'])} in every zone",
        "broad minus narrow (4.6)": f"raises every value by {span(t['broad_minus_narrow_range'])}",
        "D/P share": f"they make up {100 * dp[0]:.0f}–{pct(dp[1])} of large bodies",
        # section 4.2, Table 3 in words
        "largest inner bodies": f"{ib[0]['n_S']} of {ib[0]['n_S'] + ib[0]['n_C']} bodies are \\mbox{{S-like}}, and C-types hold only "
                                f"{pct(ib[0]['C_frac_mass'])} of the mass",
        "50-100 km": f"({pct(ib[1]['C_frac_number'])} at 50–100~km)",
        "20-50 km": f"({pct(ib[2]['C_frac_number'])} at 20–50~km)",
        # section 4.3
        "measured masses": f"{t['n_classified_with_measured_mass']:,} classified bodies have a measured mass, and they hold "
                           f"{pct(t['measured_share_of_classified_mass'])} of the classified mass",
        "four largest masses": f"Ceres alone holds {pct(top4['Ceres'])} of the mass of classified main-belt asteroids, "
                               f"Vesta {pct(top4['Vesta'])}, Pallas {pct(top4['Pallas'])} and Hygiea {pct(top4['Hygiea'])}",
        "inner mass": f"the inner belt’s mass is {pct(t['inner_mass_S_of_S_plus_C'])} S-like",
        "middle mass": f"the middle belt’s is {pct(mc['all']['middle'])} C-like",
        "mass without the four": "gives C mass fractions of " + ", ".join(f"{v:.2f}" for v in no4[:3]) + f" and {no4[3]:.2f}",
        "inner mass (5.2)": f"The mass-weighted C-fraction of the inner belt is {pct(mc['all']['inner'])} (Section~\\ref{{sec:4.3}}), "
                            f"and {pct(ib[0]['C_frac_mass'])} for bodies above 100~km",
        # section 4.4
        "raw crossover": f"The raw sample places it at {a50[raw]:.2f}~au",
        "debiased crossover": f"move it inward to {a50_span}~au, {sl - max(a50[k] for k in debiased):.1f}–"
                              f"{sl - min(a50[k] for k in debiased):.1f}~au inside the snow line",
        "transition width": f"the 10–90\\% width of the transition is {width_span}~au",
        "inward move (5.1)": f"the crossover moves inward by about {a50[raw] - a50[KEYS[1]]:.1f}~au",
        "IPW and complete crossovers (5.5)": f"({a50[KEYS[2]]:.2f} versus {a50[KEYS[1]]:.2f}~au)",
        # section 4.5
        "inner orbits": f"($p = {p2(ot['inner', 'proper e']['KS_p'])}$ for $e_{{\\mathrm{{P}}}}$, "
                        f"{p2(ot['inner', 'proper sin i']['KS_p'])} for $\\sin i_{{\\mathrm{{P}}}}$)",
        "pristine e medians": f"({ot['pristine', 'proper e']['median_C']:.3f} versus {ot['pristine', 'proper e']['median_S']:.3f})",
        "pristine sin i medians": f"($\\sin i_{{\\mathrm{{P}}}}$ {ot['pristine', 'proper sin i']['median_C']:.3f} versus "
                                  f"{ot['pristine', 'proper sin i']['median_S']:.3f})",
        "pristine S-types": f"only {ot['pristine', 'proper e']['n_S']} S-types fall in that zone",
        "outer inclinations": f"Only the outer-belt inclinations pass it ($p = {p2(ot['outer', 'proper sin i']['KS_p'])}$)",
        "outer medians": f"($\\Delta = {ot['outer', 'proper sin i']['dmedian_C_minus_S']:.3f}$)",
        "extended orbit test": f"the smallest $p$ is {p2(min(r['KS_p'] for r in ext['orbit_tests_ks']))}, "
                               f"again for the outer-belt inclinations",
        # section 4.6
        "dark fraction": "the dark fraction is " + ", ".join(f"{v:.2f}" for v in dark[:3]) + f" and {dark[3]:.2f} from the inner to the outer zone",
        "albedo complete limit": f"gives a complete sample for $D \\geq {S['completeness']['D_complete_albedo_km']:.0f}$~km",
        "dark C and S": f"{pct(S['albedo_by_group']['C-like']['frac_below_0.10'])} of the C-like and "
                        f"{pct(S['albedo_by_group']['S-like']['frac_below_0.10'])} of the S-like bodies are darker than 0.10",
        "X split counts": f"Of the {xs['n_taxonomy_X_group']:,} classified X-complex bodies, {xs['n_split']:,} have a measured albedo",
        "X split P-like": f"and {xs['n_split_by_class']['P']:,} of these are P-like",
        "X split P share": f"lower in the inner belt ({p_share['inner']:.2f}) than beyond it ({span([p_share[z] for z in ZONES[1:]])})",
        "extension family fraction": f"raises the family fraction from {100 * fam['family_fraction']:.1f}\\% to "
                                     f"{100 * ext['family_fraction']:.1f}\\%",
        # section 5
        "middle belt (5.2)": f"raise its middle-belt C-fraction from {zone_f(raw, 'middle'):.2f} to {deb('middle')}",
        "Eos": f"here Eos is classified as {t['eos_family']['fam_group']} by plurality "
               f"({t['eos_family']['purity']:.2f} of its classified members)",
        "derived diameters": f"Of the classified bodies, {pct(t['derived_diameter_fraction_taxonomy'])} have diameters",
        "Flora": f"(9) {big['name']}, an \\mbox{{{big['spectral_type']}-type}} body larger than the {second['spectral_type']}-type "
                 f"(8) {second['name']} itself",
        "IPW and complete inner (5.5)": f"differ most in the inner belt ({zone_f(KEYS[1], 'inner'):.2f} and {zone_f(KEYS[2], 'inner'):.2f})",
    }


@pytest.mark.parametrize("what", list(_prose()))
def test_the_running_text_quotes_the_summary(what):
    phrase = _prose()[what]
    assert phrase in TEX, f"{what}: the paper does not say {phrase!r}"


def test_no_share_the_paper_prints_as_a_percentage_is_stored_on_a_half_percent():
    """A stored share on a half percent was itself rounded, so printing it as a whole percentage rounds twice: in
    v2.6.3, 17/126 = 0.1349 was stored as 0.135 and printed as 14%. Such a share must be stored with more decimals."""
    t, mc = SUMMARY["text_numbers"], SUMMARY["mass_c_fraction"]["all"]
    shares = (t["dp_share_per_0.1AU_bin_from_2.9AU"] + t["completeness_H13.5_to_14.5_range"]
              + list(t["mass_share_of_classified"].values()) + [mc["inner"], mc["middle"]]
              + [t[k] for k in ("assumed_label_fraction", "measured_share_of_classified_mass", "inner_mass_S_of_S_plus_C",
                                "derived_diameter_fraction_taxonomy")])
    for v in shares:
        halves = round(v * 200, 6)                     # a whole number of half percents
        assert not (halves.is_integer() and int(halves) % 2 == 1), f"{v} is stored on a half percent"


@pytest.mark.parametrize("phrase, value", [
    ("zone fractions change by at most {}", "family_extension_toggled.max_abs_change_zone_fraction"),
    ("the crossover moves by at most {}~au", "family_extension_toggled.max_abs_change_a50_au"),
    ("raises the broad fraction by at most {}", "x_split_by_albedo.max_abs_change_broad"),
    ("changes the narrow fraction by at most {}", "x_split_by_albedo.max_abs_change_narrow"),
])
def test_every_bound_the_text_states_holds(phrase, value):
    """"at most 0.01" holds for a value that rounds, at the digits printed, to no more than 0.01."""
    block, key = value.split(".")
    m = re.search(re.escape(phrase).replace(r"\{\}", r"(\d+(?:\.\d+)?)"), TEX)
    assert m, f"the paper no longer says {phrase!r}"
    bound = m.group(1)
    digits = len(bound.split(".")[1]) if "." in bound else 0
    assert round(SUMMARY[block][key], digits) <= float(bound), f"{phrase.format(bound)}, but {value} is {SUMMARY[block][key]}"


# ── what the text says about the numbers ─────────────────────────────────────────────────────────────

def test_with_the_extension_no_orbit_test_passes_and_the_outer_inclinations_come_closest():
    ks = SUMMARY["family_extension_toggled"]["orbit_tests_ks"]
    closest = min(ks, key=lambda r: r["KS_p"])
    assert (closest["zone"], closest["element"]) == ("outer", "proper sin i")
    assert not any(r["bonferroni_sig"] for r in ks)


def test_every_debiased_transition_is_wider_than_the_belt_and_reaches_10_and_90_percent_only_at_its_edges():
    """§4.4: "no debiased fit falls to a C-fraction of 10% within the belt, and each reaches 90% only near the belt's
    outer edge"."""
    from beltgradient.config import A_MAX, A_MIN
    assert f"the {A_MAX - A_MIN:.1f}~au span of the belt" in TEX
    for k in KEYS[1:]:
        x = SUMMARY["crossover"][k]
        a10, a90 = x["a50"] - x["width_10_90"] / 2, x["a50"] + x["width_10_90"] / 2
        assert x["width_10_90"] > A_MAX - A_MIN, k
        assert a10 < A_MIN, f"{k}: the fit falls to 10% inside the belt, at a = {a10:.2f} au"
        assert A_MAX - 0.2 < a90 < A_MAX, f"{k}: the fit reaches 90% at a = {a90:.2f} au"


def test_the_debiased_crossovers_agree_within_their_intervals():
    xs = [SUMMARY["crossover"][k] for k in KEYS[1:]]
    assert max(x["a50_lo"] for x in xs) <= min(x["a50_hi"] for x in xs)


def test_the_two_debiasing_methods_disagree_only_in_the_inner_belt():
    """§4.1: they "agree within their uncertainties in every zone except the inner belt"."""
    def interval(key, zone):
        lo, hi = re.match(r"[\d.]+ \[([\d.]+)–([\d.]+)\]", SUMMARY["zone_c_fraction_narrow"][zone][key]).groups()
        return float(lo), float(hi)
    for zone in ZONES:
        (lo1, hi1), (lo2, hi2) = interval(KEYS[1], zone), interval(KEYS[2], zone)
        assert (max(lo1, lo2) <= min(hi1, hi2)) == (zone != "inner"), zone


def test_every_per_zone_estimate_is_lowest_in_the_inner_belt_and_highest_in_the_outer():
    """§4.6, of Figure 13: each estimate "is lowest in the inner belt and highest in the outer belt"."""
    S = SUMMARY
    rows = {f"{which}, {k}": [float(S[f"zone_c_fraction_{which}"][z][k].split(" ")[0]) for z in ZONES]
            for which in ("narrow", "broad") for k in KEYS}
    rows.update({f"X split, broad, {k}": [S["x_split_by_albedo"]["zone_c_fraction_broad"][k][z] for z in ZONES] for k in KEYS})
    rows["family extension"] = [float(S["family_extension_toggled"]["zone_c_fraction_narrow"][z][KEYS[1]].split(" ")[0])
                                for z in ZONES]
    rows["brightness-limited IPW"] = [S["text_numbers"]["H_limited_IPW_c_fraction_by_zone"][z] for z in ZONES]
    rows["dark fraction"] = [S["dark_fraction_complete"][z] for z in ZONES]
    for name, v in rows.items():
        assert v[0] == min(v) and v[-1] == max(v), (name, v)

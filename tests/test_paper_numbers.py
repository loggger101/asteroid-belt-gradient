"""The paper agrees with the committed results: every number in Tables 2-5, the counts and limits quoted in the text,
the size cuts it names, and which file each figure number shows. No data or LaTeX needed.

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

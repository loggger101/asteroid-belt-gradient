"""README.md describes the code as it is: every name it cites exists, its figure table is what the pipeline draws,
its module list is the package, the figures it says draw no random numbers take no generator, and every result it
quotes is the committed summary's."""
import importlib
import inspect
import json
import re
from pathlib import Path

import pytest

from beltgradient import figures
from beltgradient.config import CATALOG_CONTRACT, CATALOG_RELEASE, P_DARKEST
from beltgradient.pipeline import FIGURE_FILES

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
PIPELINE = (ROOT / "src" / "beltgradient" / "pipeline.py").read_text(encoding="utf-8")
PACKAGE = sorted(p.stem for p in (ROOT / "src" / "beltgradient").glob("*.py") if not p.stem.startswith("__"))
SUMMARY = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))


def _span(values, fmt="{:.2f}") -> str:
    return f"{fmt.format(min(values))}–{fmt.format(max(values))}"


def _quoted() -> dict[str, str]:
    """Each result the README quotes, rebuilt from the summary at the digits the README prints."""
    S, xs = SUMMARY, SUMMARY["x_split_by_albedo"]
    keys = list(S["crossover"])
    zones = ["inner", "middle", "pristine", "outer"]
    row = lambda k: " | ".join(S["zone_c_fraction_narrow"][z][k].split(" ")[0] for z in zones)
    a50 = [S["crossover"][k]["a50"] for k in keys[1:]]
    widths = [S["crossover"][k]["width_10_90"] for k in keys[1:]]
    ot = {(r["zone"], r["element"]): r for r in S["orbit_tests"]}
    ks_ext = min(r["KS_p"] for r in S["family_extension_toggled"]["orbit_tests_ks"])
    n = xs["n_split_by_class"]
    share = xs["P_share_of_split_by_zone"]
    return {
        "sample": f"the {S['catalog']['taxonomy_tier']:,} main-belt asteroids with a published taxonomy in the AsteroidCatalog "
                  f"release `{CATALOG_RELEASE}`",
        "raw row": f"| C/(S+C), raw count | {row(keys[0])} |",
        "IPW row": f"| C/(S+C), families collapsed, IPW, D ≥ 10 km | {row(keys[1])} |",
        "complete row": f"| C/(S+C), families collapsed, D ≥ {S['completeness']['D_complete_km']:.0f} km | {row(keys[2])} |",
        "crossover": f"S/C crossover: {S['crossover'][keys[0]]['a50']:.2f} au raw, {_span(a50)} au once corrected; "
                     f"the 10→90% transition is {_span(widths, '{:.1f}')} au wide",
        "assumed share": f"{100 * S['text_numbers']['assumed_label_fraction']:.0f}% of the catalog's labels",
        "size-complete limit": f"(H_c = {S['completeness']['H_c']}, so D ≥ {S['completeness']['D_complete_km']} km for p = {P_DARKEST}",
        "extension": f"it attaches {S['families']['extension_attaches']:,} more bodies",
        "orbit test": f"outer-belt sin i (p = {ot['outer', 'proper sin i']['KS_p']:#.2g})",
        "extended orbit test": f"(smallest p = {ks_ext:#.2g})",
        "X split": f"Of the {xs['n_taxonomy_X_group']:,} classified X-types, {xs['n_split']:,} (X and its sub-classes other than "
                   f"Xe and Xk) have a measured albedo: {n['P']:,} are P-like, {n['M']:,} M-like and {n['E']:,} E-like",
        "X split shifts": f"raises the broad fraction by at most {xs['max_abs_change_broad']:.3f} and moves the narrow fraction by "
                          f"at most {xs['max_abs_change_narrow']:.3f} ({xs['families_reclassified']} families change class)",
        "X split P share": f"{share['inner']:.2f} in the inner belt and {_span([share[z] for z in zones[1:]])} beyond it",
        "catalog": f"{S['catalog']['rows']:,} bodies, data contract {CATALOG_CONTRACT}",
    }


@pytest.mark.parametrize("what", list(_quoted()))
def test_the_readme_quotes_the_summary(what):
    assert _quoted()[what] in README, f"{what}: README.md does not say {_quoted()[what]!r}"


def test_the_bounds_the_readme_states_hold():
    ext = SUMMARY["family_extension_toggled"]
    m = re.search(r"no zone fraction changes by more than ([\d.]+) and the crossover moves by less than ([\d.]+) au", README)
    assert m, "README no longer states the family extension's effect"
    assert round(ext["max_abs_change_zone_fraction"], 2) <= float(m.group(1))
    assert ext["max_abs_change_a50_au"] < float(m.group(2))


@pytest.mark.parametrize("ref", sorted({r for r in re.findall(r"`(" + "|".join(PACKAGE) + r")\.([A-Za-z_]\w*)`", README)
                                         if r[1] != "py"}),                      # `bhac15.py` names a file
                         ids=lambda r: ".".join(r))
def test_every_name_the_readme_cites_exists(ref):
    module, name = ref
    assert hasattr(importlib.import_module(f"beltgradient.{module}"), name), f"README cites {module}.{name}, which is gone"


def test_the_layout_lists_every_module():
    listed = sorted(re.findall(r"^  (\w+)\.py ", README, re.M))
    assert listed == PACKAGE


def _figure_table():
    rows = re.findall(r"^\| Fig\. ([A-Z]?\d+) \| `([^`]+)` \| `figures\.(\w+)` \|", README, re.M)
    assert rows, "no figure table in README.md"
    return rows


def test_the_figure_table_is_what_the_pipeline_draws():
    drawn = dict(re.findall(r'save\("(\w+)", F\.(\w+)\(', PIPELINE))           # figure key -> function that draws it
    expected = sorted((number, name, drawn[key]) for key, (name, number) in FIGURE_FILES.items())
    assert sorted(_figure_table()) == expected


def test_the_figures_said_to_draw_no_random_numbers_take_no_generator():
    m = re.search(r"Figs\. ([\d, ]+) and (\d+) draw no random numbers", README)
    assert m, "README no longer says which figures draw no random numbers"
    no_rng = set(re.findall(r"\d+", m.group(1))) | {m.group(2)}
    funcs = {number: func for number, _, func in _figure_table()}
    assert no_rng <= set(funcs)
    for number in sorted(no_rng):                        # the list need not be every such figure, but must be true
        assert "rng" not in inspect.signature(getattr(figures, funcs[number])).parameters, f"Fig. {number} takes rng"

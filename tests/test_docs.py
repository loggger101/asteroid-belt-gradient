"""README.md describes the code as it is: every name it cites exists, its figure table is what the pipeline draws,
its module list is the package, and the figures it says draw no random numbers take no generator."""
import importlib
import inspect
import re
from pathlib import Path

import pytest

from beltgradient import figures
from beltgradient.pipeline import FIGURE_FILES

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
PIPELINE = (ROOT / "src" / "beltgradient" / "pipeline.py").read_text(encoding="utf-8")
PACKAGE = sorted(p.stem for p in (ROOT / "src" / "beltgradient").glob("*.py") if not p.stem.startswith("__"))


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

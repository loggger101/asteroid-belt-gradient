"""paper/check_format.py: the real paper passes, and every rule FAILs on a planted defect.

The source rules run on main.tex and refs.bib with one defect planted; the build-log and PDF rules run on
text, so no LaTeX installation is needed (CI's paper job runs the check on a real build).
"""
import importlib.util
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_format", ROOT / "paper" / "check_format.py")
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)

TEX = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
BIB = (ROOT / "paper" / "refs.bib").read_text(encoding="utf-8")
INTRO = "This paper asks"                  # a sentence in the body, before which defects are inserted


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    cf.fails.clear(); cf.warns.clear(); cf.SURNAMES.clear()
    monkeypatch.setattr(cf, "UNTAGGED_OK", True)     # the tag check depends on the clone's tags, not on the paper
    monkeypatch.setattr(cf, "PAPER", ROOT / "paper")
    yield


def plant(text, old, new):
    assert old in text, f"anchor not found: {old!r}"
    return text.replace(old, new, 1)


def test_the_real_paper_passes_the_source_checks():
    cf.check_source(TEX, BIB, ROOT)
    assert cf.fails == []


SOURCE_DEFECTS = [
    ("AU", lambda t, b: (plant(t, "near 2.7~au", "near 2.7~AU"), b), "'AU'"),
    ("breakable space before a unit", lambda t, b: (plant(t, "near 2.7~au", "near 2.7 au"), b), "breakable space"),
    ("hard-coded figure number", lambda t, b: (plant(t, "Figure~\\ref{fig:2} shows", "Figure 2 shows"), b), "hard-coded reference"),
    ("unresolved \\ref", lambda t, b: (plant(t, INTRO, "Figure~\\ref{fig:99}. " + INTRO), b), "has no \\label"),
    ("text-mode subscript", lambda t, b: (plant(t, INTRO, "a\\textsubscript{50}. " + INTRO), b), "text-mode math fragment"),
    ("\\textit variable", lambda t, b: (plant(t, INTRO, "\\textit{a} grows. " + INTRO), b), "looks like a variable"),
    ("em dash", lambda t, b: (plant(t, INTRO, "a---b. " + INTRO), b), "em dash"),
    ("British spelling", lambda t, b: (plant(t, "analyzed", "analysed"), b), "British spelling"),
    ("first person", lambda t, b: (plant(t, INTRO, "We note this. " + INTRO), b), "first-person"),
    ("citation in a heading", lambda t, b: (plant(t, "\\subsection{Five scenarios}",
                                                  "\\subsection{Five scenarios of \\citet{walsh2011}}"), b), "citation in a heading"),
    ("organisation author through \\citet", lambda t, b: (plant(t, "\\citetalias{jpl2026}", "\\citet{jpl2026}"), b), "organisation as author"),
    ("interval with an en dash", lambda t, b: (plant(t, "[0.24, 0.25]", "[0.24–0.25]"), b), "uses an en dash"),
    ("no thousands separator", lambda t, b: (plant(t, "135,381", "135381"), b), "thousands separator"),
    ("figure width", lambda t, b: (plant(t, "\\includegraphics[width=\\columnwidth]{figures/fig5_circularity.png}",
                                         "\\includegraphics[width=0.9\\columnwidth]{figures/fig5_circularity.png}"), b), "width 0.9"),
    ("no \\graphicspath", lambda t, b: (plant(t, "\\graphicspath{{../}}", ""), b), "\\graphicspath{{../}}"),
    ("missing figure file", lambda t, b: (plant(t, "figures/fig5_circularity.png", "figures/fig5_missing.png"), b), "missing figure file"),
    ("abstract over 250 words", lambda t, b: (plant(t, "\\end{abstract}", "word " * 60 + "\\end{abstract}"), b), "abstract has"),
    ("keyword not a UAT concept", lambda t, b: (plant(t, "\\uat{Asteroid belt}{70}", "Asteroid belt"), b), "not a UAT concept"),
    ("stock bibliography style", lambda t, b: (plant(t, "\\bibliographystyle{aasjournalv7tie}", "\\bibliographystyle{aasjournalv7}"), b),
     "must be aasjournalv7tie"),
    ("surname missing from \\hyphenation", lambda t, b: (plant(t, " Walsh ", " "), b), "missing from \\hyphenation"),
    ("loose lines hidden", lambda t, b: (plant(t, "\\hbadness=1000", ""), b), "hbadness=1000"),
    ("journal spelled out", lambda t, b: (t, plant(b, "journal = {\\apj}", "journal = {The Astrophysical Journal}")), "spelled out"),
    ("cited versions disagree", lambda t, b: (t, plant(b, "version = {v2.6.3,}", "version = {v2.6.2,}")), "cited code version disagrees"),
]


@pytest.mark.parametrize("what, defect, expected", SOURCE_DEFECTS, ids=[d[0] for d in SOURCE_DEFECTS])
def test_each_source_rule_fails_on_its_defect(what, defect, expected):
    tex, bib = defect(TEX, BIB)
    cf.check_source(tex, bib, ROOT)
    assert any(expected in f for f in cf.fails), f"{what}: no FAIL containing {expected!r}; got {cf.fails}"


def test_a_missing_tag_fails_unless_allowed(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    (repo / "src" / "beltgradient").mkdir(parents=True)
    shutil.copytree(ROOT / "figures", repo / "figures")
    (repo / "src" / "beltgradient" / "__init__.py").write_text(
        (ROOT / "src" / "beltgradient" / "__init__.py").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(cf, "UNTAGGED_OK", False)
    cf.check_source(TEX, BIB, repo)                  # not a git repository: no tags at all
    assert any("not a tag" in f for f in cf.fails)
    cf.fails.clear()
    monkeypatch.setattr(cf, "UNTAGGED_OK", True)
    cf.check_source(TEX, BIB, repo)
    assert cf.fails == [] and any("not a tag" in w for w in cf.warns)


# ── the build log ───────────────────────────────────────────────────────────────────────────────

CLEAN_LOG = "This is pdfTeX\nOutput written on main.pdf (15 pages, 1829212 bytes).\n"


def test_a_clean_log_passes():
    cf.check_log(CLEAN_LOG, "Database file #1: refs.bib\n")
    assert cf.fails == []


@pytest.mark.parametrize("log, blg, expected", [
    ("! Undefined control sequence.\nl.164 ...S\\nobreakdash\n" + CLEAN_LOG, "", "LaTeX error: Undefined control sequence. (main.tex line 164)"),
    (CLEAN_LOG, None, "BibTeX did not run"),
    (CLEAN_LOG, "Warning--empty journal in walsh2011\n", "BibTeX: empty journal in walsh2011"),
    (CLEAN_LOG, "I couldn't open database file refs.bib\n", "BibTeX: I couldn't open database file"),
    ("LaTeX Warning: Reference `fig:9' on page 3 undefined on input line 80.\n" + CLEAN_LOG, "", "undefined reference: fig:9"),
    ("LaTeX Warning: Citation `nobody2020' on page 3 undefined on input line 80.\n" + CLEAN_LOG, "", "undefined citation"),
    ("Overfull \\hbox (3.2pt too wide) in paragraph at lines 10--12\n" + CLEAN_LOG, "", "overfull line"),
    ("Underfull \\hbox (badness 2818) in paragraph at lines 87--88\n" + CLEAN_LOG, "", "loose line"),
    ("Package hyperref Warning: Token not allowed in a PDF string\n" + CLEAN_LOG, "", "unsafe heading"),
])
def test_each_log_rule_fails(log, blg, expected):
    cf.check_log(log, blg)
    assert any(expected in f for f in cf.fails), cf.fails


def test_a_missing_log_or_pdf_fails(tmp_path):
    assert cf.check_build(tmp_path, compile_it=False) is None and "no main.log" in cf.fails[-1]
    (tmp_path / "main.log").write_text("! Emergency stop.\n", encoding="latin-1")
    assert cf.check_build(tmp_path, compile_it=False) is None and "no PDF written" in cf.fails[-1]


# ── the PDF text ────────────────────────────────────────────────────────────────────────────────

CLEAN_PAGE = ("1. INTRODUCTION\nThe belt lies between 2.1 and 3.3 au, and Figure 1 shows the snow line near\n"
              "2.7 au. Figure 1. Snow-line radius.\n")


def test_a_clean_page_passes():
    cf.check_pages([CLEAN_PAGE])
    assert cf.fails == [] and cf.warns == []


@pytest.mark.parametrize("page, expected", [
    ("The crossover moves inward by about 0.4\nau once families are collapsed.\n", "number | unit"),
    ("the transition is shown in Figure\n4 for every sample considered here.\n", "label | number"),
    ("in every debiased sample the transition width of the fitted curve is\n= 1.4 au, wider than the belt itself.\n",
     "relation sign at the line break"),
    ("the orbits mapped first by the survey of D.\nNesvorný et al. (2015) and later extended.\n", "initials separated"),
    ("the belt — as noted — is mixed\n", "em dash"),
    ("as shown by Wal\ufffesh et al. (2011) for the belt\n", "hyphenated surname Wal-sh"),
    ("distributions of S\ufffelike and C-like bodies\n", "class letter split at a line end: S-/like"),
])
def test_each_pdf_rule_fails(page, expected):
    cf.SURNAMES.add("Walsh")
    cf.check_pages([page])
    assert any(expected in f for f in cf.fails), cf.fails


def test_a_float_far_from_its_first_mention_warns():
    pages = ["see Figure 1 for the track\n", "text\n", "text\n", "text\n", "text\n", "Figure 1. The track.\n"]
    cf.check_pages(pages)
    assert any("5 pages after its first mention" in w for w in cf.warns)


# ── the command line ────────────────────────────────────────────────────────────────────────────

def test_main_fails_without_a_repository(tmp_path):
    assert cf.main(["--no-build", "--repo", str(tmp_path)]) == 1
    assert any("no repository" in f for f in cf.fails)


def test_main_passes_on_the_real_paper_without_building():
    assert cf.main(["--no-build", "--untagged-ok"]) == 0

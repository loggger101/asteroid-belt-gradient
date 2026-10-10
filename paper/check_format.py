"""Formatting check for the AASTeX paper (paper/main.tex + refs.bib).

Encodes the standards set by the formatting audit of 2026-10-09/10 (listed in README.md). Run after
every edit to the paper:

    python -I check_format.py            (from paper/; needs latexmk and pypdfium2)
    python -I check_format.py --no-build (source checks only, no compile)

FAIL lines break a standard; WARN lines are worth a look but are known to occur (e.g. floats that
land a few pages after their first mention because Section 4 cites six figures on one page).
Exit status 1 if anything FAILs. The script never writes into paper/: it compiles in a temporary
folder. --untagged-ok turns the missing-tag FAIL into a WARN (CI: the tag is made after the merge).
"""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
UNITS = r"(?:au|km|Myr|Gyr|K|mag|kg)"
BRITISH = r"\b(?:analysed|analysing|labelled|labelling|colours?|behaviours?|centres?|normalis\w*|favour\w*|modelled|modelling)\b"
FIRST_PERSON = r"\b(?:I|me|my|we|us|our|We|Our|My)\b"
GROUPS_ITALIC = {"X"}   # \textit{X} is the X group's name, not a variable

fails, warns = [], []
UNTAGGED_OK = False   # --untagged-ok
SURNAMES = set()   # filled by check_source from refs.bib


def fail(msg):
    fails.append(msg)


def warn(msg):
    warns.append(msg)


def line_of(text, pos):
    return text.count("\n", 0, pos) + 1


def strip_comments(tex):
    return re.sub(r"(?<!\\)%.*", "", tex)


def body_of(tex):
    return tex[tex.index("\\begin{document}"):]


def environments(tex, name):
    return re.findall(r"\\begin\{" + re.escape(name) + r"\}(.*?)\\end\{" + re.escape(name) + r"\}", tex, re.S)


# ---------------------------------------------------------------- source checks
def check_source(tex, bib, repo):
    src = strip_comments(tex)
    body = body_of(src)

    # 1. Word-style math fragments must be real math
    for pat, what in [(r"\\textsubscript", "\\textsubscript"), (r"\\textsuperscript", "\\textsuperscript"),
                      (r"\\textless\b", "\\textless"), (r"\\textgreater\b", "\\textgreater"),
                      (r"\\ensuremath\{\\(?:geq|leq|approx|times|Delta|Sigma|beta|rightarrow)\}", "\\ensuremath{...}")]:
        for m in re.finditer(pat, body):
            fail(f"main.tex:{line_of(src, src.index(body) + m.start())}: text-mode math fragment {what}; write it as $...$")
    for m in re.finditer(r"\\textit\{([A-Za-z])\}", body):
        if m.group(1) not in GROUPS_ITALIC:
            fail(f"main.tex:{line_of(src, src.index(body) + m.start())}: \\textit{{{m.group(1)}}} looks like a variable; use ${m.group(1)}$")

    # 2. units: au (not AU), and a tie between number and unit
    for m in re.finditer(r"\bAU\b", body):
        fail(f"main.tex:{line_of(src, src.index(body) + m.start())}: 'AU'; AAS writes the astronomical unit 'au'")
    for m in re.finditer(r"\d\$? " + UNITS + r"(?![-\w])", body):
        fail(f"main.tex:{line_of(src, src.index(body) + m.start())}: breakable space between number and unit "
             f"'{m.group(0)}'; use ~")

    # 3. cross-references: \ref, not hard-coded numbers; every \ref resolves; every float is cited
    for m in re.finditer(r"\b(?:Figures?|Tables?|Sections?|Equations?)[ ~]+\(?[A-Z]?\d|\bAppendix[ ~]+[A-Z]\b", body):
        fail(f"main.tex:{line_of(src, src.index(body) + m.start())}: hard-coded reference '{m.group(0)}'; use \\ref")
    labels = set(re.findall(r"\\label\{([^}]+)\}", src))
    refs = re.findall(r"\\ref\{([^}]+)\}", src)
    for r in sorted(set(refs) - labels):
        fail(f"\\ref{{{r}}} has no \\label")
    for lab in sorted(l for l in labels if l.startswith(("fig:", "tab:"))):
        if lab not in refs:
            fail(f"{lab} is never cited in the text")
    # floats cited in numerical order (first citation in the source; appendix figures excluded)
    for kind in ("fig", "tab"):
        seen = []
        for r in refs:
            if r.startswith(kind + ":") and r[len(kind) + 1:].isdigit() and r not in seen:
                seen.append(r)
        nums = [int(r.split(":")[1]) for r in seen]
        if nums != sorted(nums):
            warn(f"{'figures' if kind == 'fig' else 'tables'} are first cited out of order: "
                 + ", ".join(str(n) for n in nums))

    # 4. headings: a citation in a heading needs \texorpdfstring (PDF bookmark)
    for m in re.finditer(r"\\(?:sub)*section\*?\{((?:[^{}]|\{[^{}]*\})*)\}", body):
        if "\\cite" in m.group(1) and "\\texorpdfstring" not in m.group(1):
            fail(f"main.tex:{line_of(src, src.index(body) + m.start())}: citation in a heading without \\texorpdfstring")

    # 5. organisation authors are cited through aliases (the AASTeX 7 bst adds a stray space otherwise)
    for key in re.findall(r"@\w+\{(\w+),\s*(?:[^@]*?)author = \{\{[^}]*\}\}", bib):
        if re.search(r"\\cite(?:t|p|alt|alp)\{[^}]*\b" + key + r"\b", src):
            fail(f"{key} has an organisation as author: cite it with \\citetalias (\\defcitealias)")

    # 5b. no em dashes (2026-10-10): not typed, not as the --- ligature, not as \textemdash
    for m in re.finditer(r"—|(?<!-)---(?!-)|\\textemdash\b", body):
        fail(f"main.tex:{line_of(src, src.index(body) + m.start())}: em dash; use a comma, colon, "
             f"parentheses or an en dash for ranges")

    # 6. spelling and voice
    for m in re.finditer(BRITISH, body):
        fail(f"main.tex:{line_of(src, src.index(body) + m.start())}: British spelling '{m.group(0)}'")
    prose = re.sub(r"\\email(\[[^]]*\])?\{[^}]*\}", "", body)
    prose = re.sub(r"\$[^$]*\$", " ", prose)
    prose = re.sub(r"\\[a-zA-Z]+\*?(\[[^]]*\])?", " ", prose)
    for m in re.finditer(FIRST_PERSON, prose):
        fail(f"first-person word '{m.group(0)}' near: ...{prose[max(0, m.start() - 40):m.end() + 20]!r}")

    # 7. tables: [lo, hi] intervals and thousands separators
    for env in ("deluxetable", "deluxetable*"):
        for t in environments(src, env):
            data = t[t.find("\\startdata"):t.find("\\enddata")]
            data = re.sub(r"\\cite\w*\{[^}]*\}", "", data)   # citation keys carry years
            for m in re.finditer(r"\[\d[\d.]*–", data):
                fail(f"table interval '{data[m.start():m.start() + 14]}...' uses an en dash; write [lo, hi]")
            for m in re.finditer(r"(?<![\d.,])\d{4,}(?![\d.,])", data):
                fail(f"table number {m.group(0)} without a thousands separator")

    # 8. figures: widths and files
    for env, width in (("figure", r"\\columnwidth|3\.3125in"), ("figure*", r"\\textwidth")):
        for f in environments(src, env):
            for w, path in re.findall(r"\\includegraphics\[width=([^\]]+)\]\{([^}]+)\}", f):
                if not re.fullmatch(width, w):
                    fail(f"{path}: width {w} in a {env} (expected {width.replace(chr(92) * 2, chr(92))})")
    figs = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", src)
    if "\\graphicspath{{../}}" not in src:
        fail("main.tex must set \\graphicspath{{../}} so figures/... are the repository's figures")
    root = repo or PAPER.parent   # \graphicspath{{../}}: figures/... are the repository's figures/
    for path in figs:
        if not (root / path).exists():
            fail(f"missing figure file {path} (looked in {root / path})")
    unused = sorted({p.name for p in (root / "figures").glob("*.png")} - {Path(f).name for f in figs})
    if unused:
        warn("figures in figures/ not used by main.tex: " + ", ".join(unused))

    # 9. front matter: abstract length, UAT keywords
    abstract = src[src.index("\\begin{abstract}") + 16:src.index("\\end{abstract}")]
    a = re.sub(r"\\cite[pt]\{[^}]*\}", "Author et al. (2024)", abstract)
    a = re.sub(r"\$([^$]*)\$", lambda m: re.sub(r"\s+", "", m.group(1)), a)
    a = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", a)
    n_words = len(re.sub(r"\\[a-zA-Z]+", "", a).split())
    if n_words > 250:
        fail(f"abstract has {n_words} words (AAS limit 250)")
    kw = re.search(r"\\keywords\{((?:[^{}]|\{[^{}]*\})*)\}", src)
    if not kw:
        fail("no \\keywords")
    else:
        for term in re.split(r"(?<=\}),\s*(?=\\uat)|---", kw.group(1)):   # commas between concepts
            if not re.fullmatch(r"\s*\\uat\{[A-Z][\w ,'-]+\}\{\d+\}\s*", term):
                fail(f"keyword '{term.strip()}' is not a UAT concept written \\uat{{Term}}{{id}}")

    # 9b. names: initials tied to surnames (the modified bst), and no cited surname hyphenated
    style = re.search(r"\\bibliographystyle\{([^}]+)\}", src)
    if not style or style.group(1) != "aasjournalv7tie":
        fail("\\bibliographystyle must be aasjournalv7tie (ties initials to surnames in citations)")
    hyph = re.search(r"\\hyphenation\{([^}]*)\}", src)
    words = set(hyph.group(1).split()) if hyph else set()
    surnames = set()
    for field in re.findall(r"author = \{(.*?)\},?\n", bib):
        for person in re.split(r"\s+and\s+", field):
            for part in re.split(r"[\s-]+", person.split(",")[0].strip().strip("{}")):
                if re.fullmatch(r"[A-Za-z]{5,}", part) and part[0].isupper() and part not in {"Laboratory", "others"}:
                    surnames.add(part)
    missing = sorted(surnames - words)
    SURNAMES.update(surnames)
    if missing:
        fail("cited surnames missing from \\hyphenation (TeX may hyphenate them): " + " ".join(missing))
    if "\\hbadness=1000" not in src:
        fail("main.tex must set \\hbadness=1000 after the class (aastex701 sets 10000 and hides loose lines)")

    # 10. references: abbreviated journals
    for j in re.findall(r"journal = \{([^}]*)\}", bib):
        if not j.startswith("\\") and len(j.replace("\\&", "&")) > 8:
            fail(f"refs.bib journal '{j}' is spelled out; use the AASTeX macro or the ADS abbreviation")

    # 11. the cited code release agrees everywhere
    cited = set(re.findall(r"(?:release|asteroid-belt-gradient) v(\d+\.\d+\.\d+)", src))
    bibv = re.search(r"@\w+\{edwards2026a,[^@]*?version = \{v([\d.]+)", bib)
    versions = {"main.tex": cited, "refs.bib": {bibv.group(1)} if bibv else set()}
    if repo:
        init = (repo / "src" / "beltgradient" / "__init__.py").read_text(encoding="utf-8")
        versions["repo"] = set(re.findall(r'__version__ = "([^"]+)"', init))
    if len(set().union(*versions.values())) != 1:
        fail(f"cited code version disagrees: {versions}")
    elif repo:
        v = next(iter(cited))
        tag = subprocess.run(["git", "-C", str(repo), "tag", "-l", f"v{v}"], capture_output=True, text=True).stdout.strip()
        if tag != f"v{v}":
            (warn if UNTAGGED_OK else fail)(f"the paper cites v{v}, which is not a tag in the repo clone (not released?)")


# ---------------------------------------------------------------- build and PDF checks
def check_build(tmp, compile_it=True):
    if compile_it:
        exe = shutil.which("latexmk")
        if not exe:
            fail("latexmk not found on PATH; build with `latexmk -pdf -outdir=DIR main.tex` in paper/ "
                 "and pass --build-dir DIR, or use --no-build")
            return None
        subprocess.run([exe, "-g", "-pdf", "-interaction=nonstopmode", f"-outdir={tmp}", "main.tex"],
                       cwd=PAPER, capture_output=True, text=True, timeout=600)
    log = (Path(tmp) / "main.log").read_text(encoding="latin-1") if (Path(tmp) / "main.log").exists() else ""
    if "Output written" not in log:
        fail(f"no successful build in {tmp}; see main.log")
        return None
    if (Path(tmp) / "main.pdf").stat().st_mtime < (PAPER / "main.tex").stat().st_mtime:
        fail("the build is older than main.tex; rebuild it")
    for pat, what in [(r"Reference `([^']+)' on page \d+ undefined", "undefined reference"),
                      (r"Citation `([^']+)' on page \d+ undefined", "undefined citation"),
                      (r"Label `([^']+)' multiply defined", "label defined twice"),
                      (r"Overfull \\hbox \(([\d.]+)pt too wide\)", "overfull line (pt)"),
                      (r"Token not allowed in a PDF string", "unsafe heading for the PDF bookmarks"),
                      (r"Underfull \\hbox \(badness (\d+)\) in paragraph at lines (\d+--\d+)",
                       "loose line (badness, main.tex lines)")]:
        for m in sorted(set(re.findall(pat, log))):
            fail(f"LaTeX: {what}: {m}")
    blg = Path(tmp) / "main.blg"
    if blg.exists():
        for m in re.findall(r"Warning--(.*)", blg.read_text(encoding="latin-1")):
            fail(f"BibTeX: {m}")
    return Path(tmp) / "main.pdf"


def check_pdf(pdf_path):
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(pdf_path))
    pages = [pdf[i].get_textpage().get_text_range().replace("\r", "") for i in range(len(pdf))]
    print(f"PDF: {len(pages)} pages")
    # bad line breaks: number | unit, label | number, a relation sign at a line break (prose lines only)
    for i, t in enumerate(pages, 1):
        for m in re.finditer(r"[—―]", t):   # includes dashes the class or the bst generate
            fail(f"p{i}: em dash in the PDF: ...{t[max(0, m.start() - 25):m.end() + 15]!r}")
        lines = [l.strip() for l in t.split("\n")]
        for a, b in zip(lines, lines[1:]):
            if not a or not b:
                continue
            what = None
            if re.search(r"\d$", a) and re.match(UNITS + r"\b", b):
                what = "number | unit"
            elif re.search(r"\b(?:Figures?|Tables?|Sections?|Equations?|Appendix)$", a) and re.match(r"[A\d(]", b):
                what = "label | number"
            elif len(a) >= 45 and (re.match(r"[=<>≈≥≤]", b) or re.search(r"[=<>≈≥≤]$", a)):
                what = "relation sign at the line break"
            elif (re.search(r"(?:^|[\s(;])[A-Z]\.(?: [A-Z]\.)?$", a) and re.match(r"(?:[A-Z]\. )*[A-Z][a-zà-ž]", b)
                  and not re.search(r"\d [A-Z]\.$", a) and "ed. " not in a[-40:]):
                what = "author initials separated from the surname"
            if what:
                fail(f"p{i}: {what}: ...{a[-25:]} / {b[:25]}...")
        for m in re.finditer(r"\b([A-Z][\wà-žÀ-Ž]*)￾([\wà-ž]+)", t):   # pdfium joins a hyphenated word with U+FFFE
            word = m.group(1) + m.group(2)
            if word in SURNAMES:
                fail(f"p{i}: hyphenated surname {m.group(1)}-{m.group(2)}")
        # a class letter split from its word at a line end ("S-/like", "K-/and L-types"):
        # write \mbox{S-like}, \mbox{K-}~and
        for m in re.finditer(r"(?:^|[\s(])([A-Z])￾(\S+)", t):
            fail(f"p{i}: class letter split at a line end: {m.group(1)}-/{m.group(2)}")
    # float placement: caption page vs first mention
    caption, first = {}, {}
    for i, t in enumerate(pages, 1):
        for m in re.finditer(r"(?m)^\s*(Figure|Table) (A?\d+)\.", t):
            caption.setdefault(f"{m.group(1)} {m.group(2)}", i)
    for i, t in enumerate(pages, 1):
        flat = re.sub(r"\s+", " ", t)
        for m in re.finditer(r"(Figure|Table)s? (A?\d+)(?!\.)", flat):
            first.setdefault(f"{m.group(1)} {m.group(2)}", i)
    for k, p in caption.items():
        if k not in first or k.split()[1].startswith("A"):   # appendix floats belong with the appendix
            continue
        if p < first[k]:
            warn(f"{k} appears on p{p}, before its first mention (p{first[k]})")
        elif p - first[k] > 3:
            warn(f"{k} appears on p{p}, {p - first[k]} pages after its first mention (p{first[k]})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-build", action="store_true", help="source checks only")
    ap.add_argument("--build-dir", help="check an existing latexmk -outdir build instead of compiling")
    ap.add_argument("--paper", default=str(HERE), help="folder with main.tex and refs.bib (default: this one)")
    ap.add_argument("--repo", default=str(HERE.parent), help="repository root (default: the parent folder)")
    ap.add_argument("--untagged-ok", action="store_true", help="a cited release without a tag is a WARN, not a FAIL")
    args = ap.parse_args()
    global PAPER, UNTAGGED_OK
    PAPER = Path(args.paper)
    UNTAGGED_OK = args.untagged_ok
    repo = Path(args.repo) if Path(args.repo, "src").exists() else None
    if repo is None:
        warn(f"repo clone not found at {args.repo}: figure and version checks against it skipped")
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    bib = (PAPER / "refs.bib").read_text(encoding="utf-8")
    check_source(tex, bib, repo)
    if args.build_dir:
        pdf = check_build(args.build_dir, compile_it=False)
        if pdf:
            check_pdf(pdf)
    elif not args.no_build:
        tmp = tempfile.mkdtemp(prefix="paper-check-")
        try:
            pdf = check_build(tmp)
            if pdf:
                check_pdf(pdf)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    for w in warns:
        print("WARN", w)
    for f in fails:
        print("FAIL", f)
    print(f"{len(fails)} FAIL, {len(warns)} WARN")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

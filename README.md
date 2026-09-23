# asteroid-belt-gradient

Analysis code for **"Mixed from the Start? Testing Whether the Asteroid Belt's Compositional Gradient Is Primordial"** (L. Edwards, SPS 4045 Planet Formation term paper, Florida Institute of Technology, Fall 2026).

If asteroids formed where they are today, the main belt should switch sharply from dry S-types to hydrated C-types at the nebular snow line near 2.7 AU. This package tests that against the 161,646 main-belt asteroids with a published taxonomy, after collapsing collisional families and correcting for the brightness bias that favors S-types. It regenerates every figure and every number in the paper from two public inputs.

| | inner (2.1–2.5 AU) | middle | pristine | outer (–3.3 AU) |
|---|---|---|---|---|
| C/(S+C), raw count | 0.25 | 0.35 | 0.40 | 0.70 |
| C/(S+C), families collapsed, IPW, D ≥ 10 km | 0.52 | 0.63 | 0.77 | 0.90 |
| C/(S+C), families collapsed, D ≥ 53 km | 0.41 | 0.68 | 0.70 | 0.89 |

S/C crossover: 2.84 AU raw, 2.41–2.47 AU once corrected; the 10→90% transition is 1.4–1.5 AU wide, wider than the belt. Full numbers with bootstrap intervals are in [`results/summary.json`](results/summary.json).

## Reproduce

```bash
git clone https://github.com/loggger101/asteroid-belt-gradient
cd asteroid-belt-gradient
pip install -e .
belt-gradient fetch   # ~150 MB: catalog snapshot + Nesvorný families, sha256-checked
belt-gradient run     # ~3 min: rewrites figures/ and results/summary.json
```

`python -m beltgradient ...` works the same without the console script. A run is deterministic (one RNG seeded with 4045): with the pinned inputs it reproduces the committed `summary.json` and figures bit for bit. `pytest` runs the fast tests; `pytest -m slow` also does a full run and checks it against the committed summary.

`python examples/walkthrough.py` steps through the same analysis and prints the intermediate tables (families, completeness, zone fractions, crossover, orbit tests, robustness checks) without overwriting the committed outputs.

## Layout

```
src/beltgradient/
  config.py        every knob: belt limits, zones, completeness threshold, size cuts, seed
  catalog.py       load the snapshot, select the main belt, group classes, label tiers
  families.py      Nesvorný families, proper elements, one-step extension, family collapse
  completeness.py  label completeness vs H, size-complete limit, IPW weights
  gradient.py      C-fraction curves and zone tables, logistic crossover, inner belt by size and mass
  orbits.py        proper e / sin i, S vs C, KS with Bonferroni
  albedo.py        label-free dark-fraction check
  figures.py       every figure
  pipeline.py      the whole run, in order → figures/ and results/summary.json
  fetch.py         download + verify the inputs
figures/           committed outputs
results/           summary.json: every number quoted in the paper
tests/
examples/          walkthrough.py: the analysis stage by stage
```

### Paper figure ↔ file

| paper | file | from |
|---|---|---|
| Fig. 1 | `fig5_circularity.png` | `figures.circularity` |
| Fig. 2 | `fig1_c_fraction_vs_a.png` | `figures.c_fraction_panels` |
| Fig. 3 | `fig1b_stacked_composition.png` | `figures.stacked_composition` |
| Fig. 4 | `fig2_mass_by_zone.png` | `figures.mass_bars` |
| Fig. 5 | `fig3_orbital_excitation.png` | `figures.orbital_cdfs` |
| Fig. 6 | `fig4_dark_fraction_albedo.png` | `figures.dark_fraction` |
| Fig. A1 | `fig0_completeness.png` | `figures.completeness_plot` |

Tables 2–5 come from `summary.json` keys `zone_c_fraction_*`, `inner_belt_by_size`, `crossover`, and `orbit_tests`; the family-extension robustness numbers in the limitations section are under `family_extension_toggled`; every other number quoted in the running text (family and mass shares, completeness, the brightness-limited IPW comparison, and so on) is under `text_numbers`. That includes the snow-line radii quoted in the paper's §2.2, which scale Equation (1) with the 1 M☉ pre-main-sequence luminosities of Baraffe et al. (2015, [doi:10.1051/0004-6361/201425481](https://doi.org/10.1051/0004-6361/201425481)), stored in `config.BHAC15_1MSUN_LOGL`.

## Method choices

Each is argued in the paper's Methods section. In short:

1. **Only published taxonomy counts** (`spectral_type_source == "source"`). 83% of the catalog's labels come from an albedo *assumed by semimajor axis* and then thresholded. Using them would return the gradient by construction (Fig. 1).
2. **Groups by first letter.** S-like = S, Q, A, V, R, O; C-like = C, B; D/P = D, P, T; K/L kept separate; X excluded. The fractions are narrow, C/(S+C), and broad, (C+D/P+K/L)/(all but X).
3. **Families are collapsed, not deleted.** Each family becomes one body with the majority class of ≥ 3 labelled members (otherwise the largest member's label), D from the summed D³, and the largest member's orbit.
4. **Samples are limited by size, not brightness.** Two methods: size-complete (H_c = 10.5, so D ≥ 52.8 km for p = 0.04) and inverse-completeness weighting at D ≥ 10 km.
5. **One-step family extension is off.** When it is turned on (`EXTEND_FAMILIES`), it attaches 229,745 more bodies, yet no zone fraction changes by more than 0.02 and the crossover moves by less than 0.03 AU.
6. **The orbit test is KS only**, Bonferroni over 8 tests (p < 0.00625). Nothing passes.

## Data

- **AsteroidCatalog snapshot** (`catalog_snapshot.parquet`, 85 MB). This is the 16 columns used here, taken from the [AsteroidCatalog](https://github.com/loggger101/AsteroidCatalog) build of 2026-08-11 (pipeline 1.1.0). It merges JPL SBDB, SsODNet ssoBFT, NEOWISE and MP3C. The snapshot is frozen because JPL adds bodies daily, so the build cannot be refetched identically. It is attached to the [`data-v1` release](https://github.com/loggger101/asteroid-belt-gradient/releases/tag/data-v1).
- **Nesvorný HCM families and proper elements**: PDS Small Bodies Node bundle [`ast.nesvorny.families` V2.0](https://sbn.psi.edu/pds/resource/nesvornyfam.html). Cite it as Nesvorný, D. (2024), *Nesvorný HCM asteroid families bundle* V2.0, NASA PDS, [doi:10.26033/5hyq-6k90](https://doi.org/10.26033/5hyq-6k90). It holds 119 families from Nesvorný et al. (2015), 153 from Nesvorný, Roig, Vokrouhlický & Brož (2024, ApJS 274, 25), and synthetic proper elements for 1,249,051 orbits. It is fetched from the PDS archive.

`data/` is git-ignored. To keep the inputs elsewhere, set `BELT_GRADIENT_DATA`.

## Citation

Software this analysis is built on: NumPy (Harris et al. 2020, [doi:10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)), SciPy (Virtanen et al. 2020, [doi:10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)), pandas (McKinney 2010, [doi:10.25080/Majora-92bf1922-00a](https://doi.org/10.25080/Majora-92bf1922-00a)) and Matplotlib (Hunter 2007, [doi:10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55)). The catalog merges JPL SBDB, SsODNet (Berthier et al. 2023), NEOWISE (Masiero et al. 2011) and MP3C (Observatoire de la Côte d'Azur); see the [AsteroidCatalog citations](https://github.com/loggger101/AsteroidCatalog/blob/main/CITATIONS.md).


See [`CITATION.cff`](CITATION.cff). The paper cites release **v1.2.0**. Earlier releases give the same figures and tables; v1.0.0 does not write the in-text numbers, and v1.1.0 lacks the pre-main-sequence snow-line calculation.

## License

MIT for the code. The input data keep their own terms (PDS SBN; the AsteroidCatalog's sources).

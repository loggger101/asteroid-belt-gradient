# asteroid-belt-gradient

Analysis code for L. Edwards (2026), **"Mixed from the Start? Testing Whether the Asteroid Belt's Compositional Gradient Is Primordial"** (Department of Aerospace, Physics, and Space Sciences, Florida Institute of Technology).

If asteroids formed where they are today, the main belt should switch sharply from dry S-types to hydrated C-types at the nebular snow line near 2.7 AU. This package tests that against the 161,659 main-belt asteroids with a published taxonomy in the AsteroidCatalog release `data-2026-09-26`, after collapsing collisional families and correcting for the brightness bias that favors S-types. It regenerates every figure and every data-derived number in the paper from two public inputs.

| | inner (2.1–2.5 AU) | middle | pristine | outer (–3.3 AU) |
|---|---|---|---|---|
| C/(S+C), raw count | 0.25 | 0.35 | 0.40 | 0.70 |
| C/(S+C), families collapsed, IPW, D ≥ 10 km | 0.53 | 0.63 | 0.77 | 0.90 |
| C/(S+C), families collapsed, D ≥ 53 km | 0.39 | 0.69 | 0.70 | 0.89 |

S/C crossover: 2.84 AU raw, 2.41–2.47 AU once corrected; the 10→90% transition is 1.4–1.5 AU wide, wider than the belt. Full numbers with bootstrap intervals are in [`results/summary.json`](results/summary.json).

## Reproduce

```bash
git clone https://github.com/loggger101/asteroid-belt-gradient
cd asteroid-belt-gradient
pip install -e .
belt-gradient fetch   # ~280 MB: AsteroidCatalog release + Nesvorný families, sha256-checked
belt-gradient run     # ~3 min: rewrites figures/ and results/summary.json
```

`python -m beltgradient ...` works the same without the console script. A run is deterministic (one RNG seeded with 4045): with the pinned inputs it reproduces the committed `summary.json` and figures bit for bit. `pytest` runs the fast tests; `pytest -m slow` also does a full run and checks it against the committed summary.

`python examples/walkthrough.py` steps through the same analysis and prints the intermediate tables (families, completeness, zone fractions, crossover, orbit tests, robustness checks) without overwriting the committed outputs.

## Layout

```
src/beltgradient/
  config.py        every knob: belt limits, zones, completeness threshold, size cuts, seed
  catalog.py       load the catalog release, select the main belt, group classes, label tiers
  families.py      Nesvorný families, proper elements, one-step extension, family collapse
  completeness.py  label completeness vs H, size-complete limit, IPW weights
  gradient.py      C-fraction curves and zone tables, logistic crossover, inner belt by size and mass
  orbits.py        proper e / sin i, S vs C, KS with Bonferroni
  albedo.py        label-free dark-fraction check
  figures.py       every figure
  pipeline.py      the whole run, in order → figures/ and results/summary.json
  fetch.py         download + verify the inputs
figures/           committed outputs
results/           summary.json: every number the paper derives from the data
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

Tables 2–5 come from `summary.json` keys `zone_c_fraction_*`, `inner_belt_by_size`, `crossover`, and `orbit_tests`; the family-extension robustness numbers (zone fractions, crossover, and the orbit test repeated on the extended background) are under `family_extension_toggled`; every other data-derived number quoted in the running text (family and mass shares, completeness, the brightness-limited IPW comparison, and so on) is under `text_numbers`. That includes the snow-line radii quoted in the paper's §2.2, which scale Equation (1) with the 1 M☉ pre-main-sequence luminosities of Baraffe et al. (2015, [doi:10.1051/0004-6361/201425481](https://doi.org/10.1051/0004-6361/201425481)), stored in `config.BHAC15_1MSUN_LOGL`.

## Method choices

Each is argued in the paper's Methods section. In short:

1. **Only published taxonomy counts** (`spectral_type_source == "source"`: mostly SkyMapper and SDSS colours; see [Data](#data)). 83% of the catalog's labels come from an albedo *assumed by semimajor axis* and then thresholded. Using them would return the gradient by construction (Fig. 1).
2. **Groups by first letter.** S-like = S, Q, A, V, R, O; C-like = C, B, F; D/P = D, P, T, Z (Z is Mahlke et al.'s 2022 extremely red class); K/L kept separate; X (X, M, E) excluded. The fractions are narrow, C/(S+C), and broad, (C+D/P+K/L)/(all but X).
3. **Families are collapsed, not deleted.** Each family becomes one body with the most common class of ≥ 3 labelled members, X-types not voting (with fewer labels, or a tie, the largest member's label), D from the summed D³, and the largest member's orbit. A body in two families goes to the larger; identical lists in the PDS bundle (529 Vibilia's file repeats 528 Leonidas's members; the 2024 lists of 1998 HD130 and 2001 PG20 are the same) count once.
4. **Samples are limited by size, not brightness.** Two methods: size-complete (H_c = 10.5, so D ≥ 52.8 km for p = 0.04, from D = 1329 km p^-1/2 10^(-H/5); Pravec & Harris 2007) and inverse-completeness weighting (Horvitz & Thompson 1952) at D ≥ 10 km. Catalog H for small asteroids has been found to run 0.4–0.5 mag too bright near H = 14 (Pravec et al. 2012), which would shift both.
5. **One-step family extension is off.** When it is turned on (`EXTEND_FAMILIES`), it attaches 197,421 more bodies (nearest listed member within the family's cutoff, by the HCM distance exactly as the bundle's `hcluster.c` computes it), yet no zone fraction changes by more than 0.01 and the crossover moves by less than 0.02 AU.
6. **The orbit test is KS only**, Bonferroni over 8 tests (p < 0.00625). One test passes, outer-belt sin i (p = 0.006), from an excess of C-types at the Themis family's inclination and of S-like bodies at Eos's. With the family extension on, none passes (smallest p = 0.011), so it is read as family halos, not as a class-dependent excitation.

## Data

- **AsteroidCatalog release [`data-2026-09-26`](https://github.com/loggger101/AsteroidCatalog/releases/tag/data-2026-09-26)** (`asteroid_catalog.parquet`, 216 MB; 1,567,657 bodies, data contract 1.4.1). It merges JPL SBDB, SsODNet ssoBFT, NEOWISE and MP3C, resolving designations JPL cannot place through the Minor Planet Center's designation links. JPL adds bodies daily, so a build cannot be repeated; each release is one frozen build, never rebuilt under its tag, and `fetch` checks the sha256 its manifest lists. The analysis reads 17 of its columns (`catalog.COLS`). The tag is set in `config.CATALOG_RELEASE`.
  - *Taxonomy.* The published labels are mostly multi-filter colours from SkyMapper (Sergeyev et al. 2022, [doi:10.1051/0004-6361/202142074](https://doi.org/10.1051/0004-6361/202142074)) and SDSS (Carvano et al. 2010, [doi:10.1051/0004-6361/200913322](https://doi.org/10.1051/0004-6361/200913322); DeMeo & Carry 2013, [doi:10.1016/j.icarus.2013.06.027](https://doi.org/10.1016/j.icarus.2013.06.027); Sergeyev & Carry 2021, [doi:10.1051/0004-6361/202140430](https://doi.org/10.1051/0004-6361/202140430)), with some MOVIS near-infrared colours (Popescu et al. 2018, [doi:10.1051/0004-6361/201833023](https://doi.org/10.1051/0004-6361/201833023)). Large bodies carry spectral classes: JPL's (Bus & Binzel 2002, [doi:10.1006/icar.2002.6856](https://doi.org/10.1006/icar.2002.6856)) where it has one, otherwise SsODNet's (Mahlke et al. 2022, [doi:10.1051/0004-6361/202243587](https://doi.org/10.1051/0004-6361/202243587)).
  - *Albedos.* Almost every main-belt body with a measured albedo also has a NEOWISE fit in the catalog: NEOWISE Diameters and Albedos V2.0 (Mainzer et al. 2019, [doi:10.26033/18S3-2Z54](https://doi.org/10.26033/18S3-2Z54)), built from Masiero et al. (2011, 2012, 2014, 2017) and Nugent et al. (2015, 2016).
- **Nesvorný HCM families and proper elements**: PDS Small Bodies Node bundle [`ast.nesvorny.families` V2.0](https://sbn.psi.edu/pds/resource/nesvornyfam.html). Cite it as Nesvorný, D. (2024), *Nesvorný HCM asteroid families bundle* V2.0, NASA PDS, [doi:10.26033/5hyq-6k90](https://doi.org/10.26033/5hyq-6k90). It holds 119 families from Nesvorný et al. (2015), 153 from Nesvorný, Roig, Vokrouhlický & Brož (2024, ApJS 274, 25), and synthetic proper elements for 1,249,051 orbits. It is fetched from the PDS archive.

`data/` is git-ignored. To keep the inputs elsewhere, set `BELT_GRADIENT_DATA`.

## Citation

Software this analysis is built on: NumPy (Harris et al. 2020, [doi:10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)), SciPy (Virtanen et al. 2020, [doi:10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)), pandas (McKinney 2010, [doi:10.25080/Majora-92bf1922-00a](https://doi.org/10.25080/Majora-92bf1922-00a)) and Matplotlib (Hunter 2007, [doi:10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55)).

The catalog is a merger, and two of its sources make citation a condition of use. If you publish results from these inputs, credit:

- **SsODNet** (required): Berthier et al. 2023, [doi:10.1051/0004-6361/202244878](https://doi.org/10.1051/0004-6361/202244878). SsODNet also asks that the papers behind its values be cited; for the taxonomy used here, those are the ones listed under [Data](#data).
- **NEOWISE Diameters and Albedos V2.0** (required): Mainzer et al. 2019, [doi:10.26033/18S3-2Z54](https://doi.org/10.26033/18S3-2Z54), with the NEOWISE papers listed under [Data](#data). The standard acknowledgment: *"This publication makes use of data products from NEOWISE, which is a project of the Jet Propulsion Laboratory/California Institute of Technology, funded by the Planetary Science Division of the National Aeronautics and Space Administration."*
- **JPL Small-Body Database**, and **MP3C** (Observatoire de la Côte d'Azur), which asks to be acknowledged by name.
- **The Minor Planet Center**, whose designation links the catalog uses: *"This research has made use of data and/or services provided by the International Astronomical Union's Minor Planet Center."*
- **The family bundle**: Nesvorný (2024), as under [Data](#data).

The full list is in the [AsteroidCatalog citations](https://github.com/loggger101/AsteroidCatalog/blob/main/CITATIONS.md).

See [`CITATION.cff`](CITATION.cff). The paper cites release **v2.0.0**, the first on the AsteroidCatalog release `data-2026-09-26`. The v1.x releases analysed the catalog's 2026-08-11 build (pipeline 1.1.0, frozen as this repository's [`data-v1`](https://github.com/loggger101/asteroid-belt-gradient/releases/tag/data-v1) snapshot), so their numbers differ slightly. v2.0.0 also groups classes F and Z, which v1.x left out of every fraction.

## License

MIT for the code. The input data keep their own terms (PDS SBN; the AsteroidCatalog's sources).

# asteroid-belt-gradient

Analysis code for L. M. Edwards (2026), **"Constraints on the Origin of the Asteroid Belt's Compositional Gradient from Debiased Taxonomic Data"** (Department of Aerospace, Physics, and Space Sciences, Florida Institute of Technology).

If asteroids formed where they are today, the main belt should switch sharply from dry S-types to hydrated C-types at the nebular snow line near 2.7 AU. This package tests that against the 161,659 main-belt asteroids with a published taxonomy in the AsteroidCatalog release `data-2026-09-29c`, after collapsing collisional families and correcting for the brightness bias that favors S-types. It regenerates every figure and every data-derived number in the paper from two public inputs.

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
belt-gradient run     # ~1.5 min: rewrites figures/ and results/summary.json
```

`python -m beltgradient ...` works the same without the console script. A run is deterministic (one RNG seeded with 4045): with the pinned inputs it reproduces the committed `summary.json` bit for bit, and the figures too under Matplotlib 3.11.1, the version that drew them (each PNG records it). `pytest` runs the fast tests (about 15 s), which include a full run of the pipeline and the CLI on small synthetic inputs whose true crossover is known; `pytest -m slow` also does the full run on the real inputs and checks it against the committed summary. CI runs the fast tests on Linux and Windows, and the full run on every pull request and push to main (`.github/workflows/reproduce.yml`), so a dependency release that moves a published number shows up at the next change.

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
tests/             fast unit tests; conftest.py builds the synthetic inputs; -m slow: the real run
examples/          walkthrough.py: the analysis stage by stage
```

### Paper figure ↔ file

| paper | file | from | shows |
|---|---|---|---|
| Fig. 1 | `fig10_snow_line_track.png` | `figures.snow_line_track` | Equation (1) along the BHAC15 1 M☉ track, against the belt |
| Fig. 2 | `fig5_circularity.png` | `figures.circularity` | the assumed-albedo labels return the step they were built from |
| Fig. 3 | `fig11_sample_flow.png` | `figures.sample_flow` | how the four samples are built, with counts |
| Fig. 4 | `fig1_c_fraction_vs_a.png` | `figures.c_fraction_panels` | C-fraction vs a: raw, IPW, size-complete |
| Fig. 5 | `fig1b_stacked_composition.png` | `figures.stacked_composition` | all five groups, size-complete sample |
| Fig. 6 | `fig6_size_distance_map.png` | `figures.size_distance_map` | C/(S+C) in bins of a and diameter |
| Fig. 7 | `fig2_mass_by_zone.png` | `figures.mass_bars` | composition by mass |
| Fig. 8 | `fig7_crossover_fits.png` | `figures.crossover_fits` | the fitted logistic transitions and a50 |
| Fig. 9 | `fig3_orbital_excitation.png` | `figures.orbital_cdfs` | proper e and sin i, S vs C, by zone |
| Fig. 10 | `fig8_proper_elements.png` | `figures.proper_element_map` | the orbit-test bodies over the family-member density |
| Fig. 11 | `fig4_dark_fraction_albedo.png` | `figures.dark_fraction` | the label-free dark fraction |
| Fig. 12 | `fig12_albedo_by_group.png` | `figures.albedo_by_group` | measured albedo of each taxonomic group |
| Fig. 13 | `fig9_robustness.png` | `figures.robustness_summary` | every per-zone estimate and check |
| Fig. A1 | `fig0_completeness.png` | `figures.completeness_plot` | taxonomy completeness vs H |

Every figure is drawn at its printed width (one column, 3.31 in, or the full text width, 7.02 in) at 300 dpi, so its type sizes are the sizes on the page. Figs. 1, 3, 6, 8, 10, 12 and 13 draw no random numbers. Fig. 1 uses the 1 M☉ track of Baraffe et al. (2015), stored in `bhac15.py` with its source and checksum.

Tables 2–5 come from `summary.json` keys `zone_c_fraction_*`, `inner_belt_by_size`, `crossover`, and `orbit_tests`; the family-extension robustness numbers (zone fractions, crossover, and the orbit test repeated on the extended background) are under `family_extension_toggled`, and the X-type split by albedo under `x_split_by_albedo`; every other data-derived number quoted in the running text (family and mass shares, completeness, the brightness-limited IPW comparison, and so on) is under `text_numbers`. That includes the snow-line radii quoted in the paper's §2.2, which scale Equation (1) with the 1 M☉ pre-main-sequence luminosities of Baraffe et al. (2015, [doi:10.1051/0004-6361/201425481](https://doi.org/10.1051/0004-6361/201425481)), stored in `config.BHAC15_1MSUN_LOGL`.

## Method choices

Each is argued in the paper's Methods section. In short:

1. **Only published taxonomy counts** (`spectral_type_source == "source"`: mostly SkyMapper and SDSS colours; see [Data](#data)). 83% of the catalog's labels come from an albedo *assumed by semimajor axis* and then thresholded. Using them would return the gradient by construction (Fig. 1).
2. **Groups by first letter.** S-like = S, Q, A, V, R, O; C-like = C, B, F; D/P = D, P, T, Z (Z is Mahlke et al.'s 2022 extremely red class); K/L kept separate; X (X, M, E) excluded. The fractions are narrow, C/(S+C), and broad, (C+D/P+K/L)/(all but X).
3. **Families are collapsed, not deleted.** Each family becomes one body with the most common class of ≥ 3 labelled members, X-types not voting (with fewer labels, or a tie, the largest member's label), D from the summed D³, and the largest member's orbit. A body in two families goes to the larger; identical lists in the PDS bundle (529 Vibilia's file repeats 528 Leonidas's members; the 2024 lists of 1998 HD130 and 2001 PG20 are the same) count once.
4. **Samples are limited by size, not brightness.** Two methods: size-complete (H_c = 10.5, so D ≥ 52.8 km for p = 0.04, from D = 1329 km p^-1/2 10^(-H/5); Pravec & Harris 2007) and inverse-completeness weighting (Horvitz & Thompson 1952) at D ≥ 10 km. Catalog H for small asteroids has been found to run 0.4–0.5 mag too bright near H = 14 (Pravec et al. 2012), which would shift both.
5. **One-step family extension is off.** When it is turned on (`EXTEND_FAMILIES`), it attaches 197,421 more bodies (nearest listed member within the family's cutoff, by the HCM distance exactly as the bundle's `hcluster.c` computes it), yet no zone fraction changes by more than 0.01 and the crossover moves by less than 0.02 AU.
6. **The orbit test is KS only**, Bonferroni over 8 tests (p < 0.00625). One test passes, outer-belt sin i (p = 0.006), from an excess of C-types at the Themis family's inclination and of S-like bodies at Eos's. With the family extension on, none passes (smallest p = 0.011), so it is read as family halos, not as a class-dependent excitation.
7. **X-types split by albedo, as a check.** By convention an X-type with a measured albedo is a P below p = 0.10, an M up to 0.30 and an E above (Fornasier, Clark & Dotto 2011, [doi:10.1016/j.icarus.2011.04.022](https://doi.org/10.1016/j.icarus.2011.04.022)), as the catalog's own `comp_class` does. Of the 13,763 classified X-types, 7,755 (X and its sub-classes other than Xe and Xk) have a measured albedo: 4,513 are P-like, 2,960 M-like and 282 E-like. Counting the P-like ones with D/P (and letting them vote in their families) raises the broad fraction by at most 0.064 and moves the narrow fraction by at most 0.010 (10 families change class); the P-like share of the split X-types is 0.38 in the inner belt and 0.54–0.66 beyond it.

## Data

- **AsteroidCatalog release [`data-2026-09-29c`](https://github.com/loggger101/AsteroidCatalog/releases/tag/data-2026-09-29c)** (`asteroid_catalog.parquet`, 217 MB; 1,568,882 bodies, data contract 1.8.1). It merges JPL SBDB, SsODNet ssoBFT, NEOWISE and MP3C, resolving designations JPL cannot place through the Minor Planet Center's designation links. JPL adds bodies daily, so a build cannot be repeated; each release is one frozen build, never rebuilt under its tag, and `fetch` checks the sha256 its manifest lists. The analysis reads 17 of its columns (`catalog.COLS`). The tag is set in `config.CATALOG_RELEASE`.
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

See [`CITATION.cff`](CITATION.cff). The paper cites release **v2.6.1**, which reads the AsteroidCatalog release `data-2026-09-29c` (data contract 1.8.1). It changes only how figures are drawn: the legends of Figs. 2, 4, 8 and 11 move outside the plot area, where no line crosses them; Fig. 9 prints its KS p-values to two significant figures as Table 5 does (0.030, 0.0060); Figs. 3 and 7 use American spelling; and every figure writes the astronomical unit as "au" (AAS and IAU style). `summary.json` is identical to v2.6.0's apart from the version. v2.6.0 added three figures (Figs. 1, 3 and 12) and the `summary.json` blocks behind them (`snow_line_track`, `sample_sizes`, `albedo_by_group`); every other output is identical to v2.5.0's. v2.5.0 redrew every figure at its printed size and adds four (Figs. 4, 6, 8 and 10 above); `summary.json` is identical to v2.4.0's apart from the version. v2.4.0 added the X-type split by albedo (Method 7, `summary.json` → `x_split_by_albedo`); every other output is identical to v2.3.1's. v2.3.1 differs from v2.3.0 only in Fig. 3 (`fig1b_stacked_composition.png`), whose bin counts now sit in their own row instead of colliding with the resonance labels; `summary.json` and the other figures are identical. Against `data-2026-09-27`, which v2.2.0 read, the release that v2.3.0 and v2.3.1 read adds two mineral-phase columns this analysis does not read and 225 main-belt bodies, all of them unclassified, and names 47 more bodies. No main-belt body's class, orbit, H, diameter, albedo or mass moves (beyond one part in 10¹²). So only the main-belt and assumed-albedo counts change (and the assumed-label share, 0.8323 → 0.8324), along with the two figures that show them (completeness and circularity); every fraction, crossover, orbit test, mass share and table is identical. v2.2.0 read `data-2026-09-27`, and v2.1.0 read `data-2026-09-26b`. Against the latter, the 2026-09-27 rebuild adds 751 main-belt bodies, none of them classified, and changes only masses estimated from a class density: an X-type with a measured albedo now takes the density of the P, M or E class that albedo indicates, one without takes their mixture, and the P and Q densities are lower. No main-belt body's class, orbit, H, diameter or albedo moves. So only the main-belt and assumed-albedo counts, the family fraction under the one-step extension (by 0.0002), the measured share of classified mass (0.940 → 0.951) and five mass shares (by at most 0.005) change, along with the three figures that show them (completeness, mass by zone, circularity). The mass-by-zone figure also puts its legend under the panels, where it no longer covers the outer zone's total. Every fraction, crossover, orbit test and table is identical. v2.1.0 differed from v2.0.0 (`data-2026-09-26`) in the same way: 185 more unclassified bodies and corrected Xe, Xk and Xc densities. The v1.x releases analysed the catalog's 2026-08-11 build (pipeline 1.1.0, frozen as this repository's [`data-v1`](https://github.com/loggger101/asteroid-belt-gradient/releases/tag/data-v1) snapshot), so their numbers differ slightly. v2.0.0 also groups classes F and Z, which v1.x left out of every fraction.

## License

MIT for the code. The input data keep their own terms (PDS SBN; the AsteroidCatalog's sources).

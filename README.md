# From a primordial fluctuation to an idealised rotation curve

Code, data, supplementary notes and the interactive tool accompanying

> B. R. Parodi, *From a primordial fluctuation to an idealised rotation curve — Collapse, energy conservation, and the cores of late-type dwarf galaxies* (A&A, submitted)

## Grow a Dwarf Galaxy (interactive)

Open **https://beroparodi.github.io/pf2irc/** in any browser, or download `index.html` and open it locally (no installation needed).
Set the flat speed *v*, the spin λ, the redshift at which growth ends, the feedback ε and the profile exponent *n*, and grow any of the 79 SPARC dwarfs of the paper. A clock dial shows when growth ends: cosmic time and redshift on the rim, the core gradient *v*/*r*<sub>c</sub> on the face, one hand for the model and one for the chosen galaxy, and the range of the 79 dwarfs (Sect. 6.4).

## Code (`code/`)

Run the scripts from the `code/` folder with Python 3 (numpy, scipy, matplotlib). Section, table and appendix numbers refer to the submitted paper.

| file | purpose |
|---|---|
| `make_figures.py` | reproduces the figures, tables and most numbers of the paper; writes `results.txt` (included, from the run of 27 Oct 2026) |
| `recipe.py` | the four-step recipe of Sect. 5.3 (37 lines): core at the end of growth and today, rotation curve, end of growth from an observed *v* and *r*<sub>c</sub>; also used by `fig_grow.py` for Fig. 3 |
| `fig_grow.py` | Figs. 3 and 6: the recipe inverted for DDO 154 and NGC 2915 (end of growth from the fitted *v* and *r*<sub>c</sub>) and their core against cosmic time (Fig. 3); the clock read as it is used (Fig. 6): end of growth against the *v*/*r*<sub>c</sub> seen today (Eq. 24), with the range of Table 1, the 79 dwarfs, the two galaxies with their spin range and the predictions for *z* = 2 and 3; called by `make_figures.py`; output `fig_grow_out.txt` |
| `y_band.py` | the core ratio *y* and the end of growth under the open modelling choices, including the outer layer with either sign, the NFW-like *n* = 1 and the pseudo-isothermal sphere (Table 1, Sects. 2.2, 4.3); output `y_band_out.txt` |
| `c_family.py` | the cored family *u_a* = 1 + (*ax* − 1)e^(−*x*) (*a* = 1: P18): mass excess *c*(*a*), first-order mass 2(*a* − 1), core constant, core ratio and end of growth for *c* = 0–0.3, the smallest *c* that gives cores of the observed size (Sects. 2.3, 4.3); output `c_family_out.txt` |
| `lag_check.py` | accuracy of the first-order Λ lag of the front, Eq. (10), against exact integration (Sect. 3.4, App. A); output `lag_check_out.txt` |
| `robustness.py` | χ² of the fits, the slope of *r*<sub>c</sub> ∝ *v*<sup>*b*</sup> for all estimators and subsamples (Table F.1), the energy budget (Table 2) and the disc test (Sects. 5.4, 6, App. F); output `robustness_results.txt` |
| `mock_calib.py`, `mock_calib_summary.py` | mock calibration of the slope test, independent noise scaled to χ²<sub>ν</sub> (App. F); results `mock_calib_*.npy`, `mock_calib_summary.txt` |
| `mock_calib2.py`, `mock_calib2_summary.py` | the same with three noise models (scaled, full errors, resampled residual pattern) and 1000 samples, with binomial intervals (Table F.2); results `mock2_*.npy`, `mock_calib2_summary.txt` |
| `r1_size.py` | the stellar threshold size *R*<sub>1</sub> (radius where Σ<sub>*</sub> = 1 M<sub>⊙</sub> pc<sup>−2</sup>) against *r*<sub>c</sub> and *R*<sub>d</sub> for the 79 dwarfs (Sect. 6.5); output `r1_size_out.txt` |
| `lensing_check.py` | circular speed of the excess mass beyond the accretion front and the front today, for comparison with stacked weak lensing (Sect. 8.3); output `lensing_check_out.txt` |
| `rar_check.py` | the radial acceleration relation and the MOND central surface density relation, the peak acceleration of P18 against *g*<sub>†</sub> (Sect. 6.6); output `rar_check_out.txt` |
| `outer_resid.py` | stacked outer residuals of P18 and the pseudo-isothermal sphere (Sect. 2.2); output `outer_resid_out.txt` |
| `baryon_share.py` | baryon share of *V*² at *r*<sub>c</sub> (daggers in Table E.1); output `baryon_share_out.txt` |
| `checks_sec6to8.py` | further numbers of Sects. 6–8 (epoch ranges, *n* = 1.76, pericentres, radial acceleration relation, core growth after *t*<sub>acc</sub>) |
| `fit_exponent.py` | fits the generalised profile with *v*, *r*<sub>c</sub> and *n* free (Sect. 7, App. G); output `fit_exponent_out.txt`, `fit_exponent_results.csv` |
| `n_vflat.py` | the free exponent *n* with *v* tied to an independent flat speed (P18 and SPARC), counts and best common *n* (Sect. 7, Table G.1); outputs `n_vflat_out.txt`, `n_vflat_results.csv` |
| `fig_nfree.py`, `fig_alpha.py` | Fig. 7 and Fig. G.2 |
| `fig_ddo161.py` | Fig. G.1: DDO 161 with P18, the free-exponent fit and the fit with *v* tied to the flat speed |
| `make_table.py` | Table E.1 |
| `spin_collapse.py` | effect of the spin on the collapse (Sect. 4.2) |
| `shells/` | one-dimensional collisionless shell runs (Sect. 8.1, App. H): `shells.py`, `driver2.py` (runs), `analyse2.py`, `fig.py` (Fig. H.1), `selfsim_check.py` and `vc_check.py` with their outputs. `shells2.py`, `shells3.py`, `driver3.py`, `driver4.py`, `analyse3.py` are exploratory transport experiments not used in the paper; `shells3.py` still has the initial cycloid of the first code version |
| `sparc_data/` | SPARC rotation curves, the fits used in the paper and the out-of-sample curves |

## Supplementary notes (`notes/`)

Each note has its PDF, LaTeX source and code.

| folder | topic |
|---|---|
| `tophat_note/` | two pictures of one top hat (shell and density), the mean density of the seed, and the nested top hats at one and two moments with the density profile (Sect. 3.2) |
| `splashback_note/` | splashback and the accretion front (Sect. 3.3) |
| `halting_settling/` | Part I: why the shells do not halt by themselves (force balance, angular momentum, star formation and supernovae, harmonic motion) and test particles in flattened and triaxial potentials; Part II: what the settling into a core requires (P18 as a rearranged singular isothermal sphere, distribution functions, energy distribution) (Sect. 8.1, App. H) |
| `seed_disc/` | Part I: the disc that the seed's own angular momentum would make; Part II: how the gas settles into it, with inflow paths on logarithmic spirals (Sect. 6.5) |
| `u_origin/` | the factor *u* of P18: the critically damped equation and its regular family, the disc representation *u* = 1 − μ″, first-order mass conservation and the second-order excess, the anisotropy P18 requires; the disc in the collapse, the mass budget, and two burst experiments that do not produce P18 (Sect. 2.3); subfolder `u_oscillator/`: the earlier, more detailed note on the differential equation and the bursts in the self-similar shells, with run files |

## Data

The rotation curves are from SPARC (Lelli, McGaugh & Schombert 2016, AJ 152, 157; http://astroweb.cwru.edu/SPARC/); please cite that paper when using them.

## How to cite

See `CITATION.cff` or the Zenodo record (DOI badge on the repository page).

## Licence

Code: MIT licence (see `LICENSE`). Text, figures and derived tables: CC BY 4.0.

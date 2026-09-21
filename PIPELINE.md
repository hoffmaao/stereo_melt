# Pipeline sequence

This document is the **canonical staged sequence** from raw REMA strips to a
basal melt-rate map, shared by every ice-shelf application. Basin decision
records (local, untracked) carry only per-shelf deltas.

## Architectural rule

The `src/stereo_melt/` library implements **algorithms** as composable functions.
It is unaware of pipeline stages, study windows, or basin AOIs. The
**sequence** that chains those functions into an end-to-end pipeline lives in
the basin driver packages — one directory per ice-shelf application (`ls`
`examples/` for the current set; `scripts/new_basin.py` scaffolds new ones).

- `stereo_melt.<module>` provides the *what* — `apply_tide_ibe_to_stack`,
  `fit_tilt_stack`, `eulerian_melt_rate`, `lagrangian_melt_rate`, …
- `<basin>.<stage>` provides the *when and where* — paths, windows, AOIs,
  the call order.

Imports flow basin → library, never the other way; basins never import each
other. Verification (extend the alternation when adding a basin):

```bash
grep -rE 'from (beardmore|beardmore_shelf|dotson_crosson|mcmurdo|nansen|pig|venable)' \
    src/   # must be empty
```

## Canonical sequence

Each stage names the library entry point it calls. Basin scripts are thin
wrappers (~40–60 lines: docstring + config paths) over shared, parameterized
`run_*_main` drivers. Run stages as modules from `examples/`:
`cd examples && $PY -m <basin>.<stage>` with `PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python`.

| # | Stage | Basin module | Library entry | What it does |
|---|---|---|---|---|
| 0 | Climate cache | `<basin>.cache_climate` | ERA5 ARCO-Zarr point pull | Pre-fetch ERA5 **mean-sea-level** pressure for the study window (IBE input; MSL, not surface pressure — SP reads ~50 hPa low over mountains). One-shot per window. |
| 1 | Strip discovery + download | `<basin>.fetch_strips` | `stereo_melt.io.rema.list_rema_strips`, `download_rema_strips` | Query the REMA s2s041 strip index for AOI overlaps in the study window; download missing TIFs. |
| 2 | Control caches (per era) | `<basin>.cache_icesat2` / `cache_cryotempo` / `cache_atm` / `cache_lvis` / `cache_glas` | Sliderule ATL06; `stereo_melt.coregister.cache_cs2` (CryoTEMPO-LI); `cache_airborne`; `cache_glas` | Query **every altimetric control source available in the era**: ICESat-2 ATL06 (2019+), CryoTEMPO-LI CryoSat-2 (2010-07+), IceBridge ATM/LVIS (campaign-dependent), ICESat-1 GLAS GLAH12 (2003–2009/10). Cached per strip; align upfront-skips strips with no cache. |
| 3 | ASP coregistration | `<basin>.align_strips` | `stereo_melt.coregister.align_driver.run_align_strips_main` → `asp.align_strip_with_asp` | 6-DOF point-to-plane ICP (`pc_align`) per strip vs. the era's control ∩ BedMachine-rock GCPs (tide-free by construction — strips go to `pc_align` raw). Era-split roots via `--asp-suffix` + `--start/--end` (e.g. `ASP/` IS2-era, `ASP_ctempoatm/`, `ASP_glas/`). |
| 3b | Nocorr ingestion | `<basin>.ingest_nocorr` (beardmore_shelf; PIG opt-in `PIG_SOURCES=nocorr`) | copy + class-bias shift | Strips no control can anchor enter at a-priori geolocation plus a per-era class-mean z-offset (`ASP_nocorr/`); the tilt LSQ sets their datum (Ez 1.0). |
| 3c | Align-root QC screen | `<basin>.scripts.qc_glas_align --root <ASP_x>` (bshelf) | on-disk align evidence | Quarantine misconverged (`|after_med|>2 m`) and under-determined (`n_ctl<50`) strips; the latter re-enter as nocorr. |
| 4 | Stack on common grid | `<basin>.build_stack` | `stereo_melt.stack.target_grid`, `save_stack` | Reproject strips onto the basin grid (EPSG:3031, `config.RES`) → `(time, y, x)` with a `dem_id` coord; PGC matchtag/bitmask screen; multi-root fusion via `*_SOURCES` (first granule stem wins). |
| 5 | Post-coreg corrections + tilt | `<basin>.tilt_fit` | see 5a–5e | All static fields before the per-epoch LSQ. |
| 5a | tide + IBE | inline | `corrections.post_coreg.apply_tide_ibe_to_stack` | CATS2008 tides + ERA5 IBE (one scalar per strip), gated by a 3 km feathered floating-ice mask. |
| 5b | MDT (skip south of −79°S) | inline | `pipeline.apply_mdt_to_stack` | DTU22 MDT over floating ice; geoid-only poleward of −79°S. |
| 5c | Geoid | inline | `pipeline.apply_geoid_to_stack` | Ellipsoid → geoid-referenced surface. |
| 5d | Static-control mask | inline | `coregister.tilt.build_static_control_mask` | BedMachine rock + grounded ice ∩ temporal stability filter. |
| 5e | Per-epoch tilt LSQ | inline | `coregister.tilt.fit_tilt_stack` | Joint LSQ: per-pixel intercept + trend, per-strip `(αx, αy, αz)`, Tikhonov priors, Tukey IRLS. Per-epoch Ez by control class (`control_source.build_per_epoch_ez`: IS2/ATM/LVIS 0.1 m · GLAS 0.3 m · CryoTEMPO 1.0 m · nocorr 1.0 m · raw CS2 2.0 m; PIG by date unless a nocorr root is active or `PIG_TILT_EZ_BY_DEM_ID=1`). Domain `static` (default) or `full` (all ice incl. shelf; requires `dhdt_smoothness=1.0`). Strips with `end_p50` > 3 m fit αz only. `Ex`/`Ey` per basin (PIG calibrated from 6c). `min_width` defaults to 0; drivers pass 10 km. |
| 6 | QC: bad-strip screen | `<basin>.scripts.find_bad_epochs` | `coregister.tilt_qc.score_tilt_residuals`, `suggest_bad_epochs` | Score per-epoch residuals/IRLS weights; flag strips the LSQ couldn't lock. Populate **`config.BAD_STRIPS` (dem_id-keyed; date-keyed `BAD_EPOCHS` is legacy)**, re-run stage 5. Second tier: melt-space per-epoch bias screen (`diag_stripe_strips`, bshelf pattern). Nocorr stacks (opt-in, PIG): `pig.screen_unrescued` → `tilt_qc.screen_unrescued_epochs` drops nocorr slices whose residual against the fit's intercept + dh/dt model stays out of bounds, writing the kept slices under a new tag. Gate: `tests/gate_unrescued_screen.py`. |
| 6b | Window-minimum shelf extent | `<basin>.build_min_extent_mask` (pig) | `shelf_extent.min_shelf_extent` + `io.greene.greene_min_extent_on_grid` | BedMachine floating ∩ Greene et al. 2022 annual extents ∩ per-epoch ocean test; removes calved sectors. `run_melt` applies it when present (`<BASIN>_MIN_EXTENT=0` opts out). Gate `tests/gate_min_shelf_extent.py`. |
| 6c | QC: control residual planes | `coregister.alignment_quality.residual_planes_for_strips` | same | Robust plane through each aligned strip's signed residual against its `pc_align` control (`reference_files/combined_reference_*.csv`). Catches failures `end_p50` misses; only `qc_alignment_failures` go to `BAD_STRIPS`. Feeds the coloured-noise prior (`budget_bridging.strip_prior_from_residual_planes`). |
| 7a | Lagrangian path solver (**production**) | `<basin>.run_melt_path` | `melt.lagrangian_melt_rate` + `flux.integrate_basal_flux` | Path-integrated Lagrangian melt (Δt ≥ 1.5 yr), firn-corrected; flux CLIP/raw/robust + GL-2km band. |
| 7b | Eulerian solver (**production, second frame**) | `<basin>.run_melt` | `melt.eulerian_melt_rate` | `b_dot = ∂H_f/∂t + ∇·(H_f·u) − a_dot` (Eq. 10 of Shean et al. 2019; negative = melt). |
| 7c | Budget linear inverse (diagnostic) | `<basin>.compare_budget_inverse` | `dynamics.linear_inverse_budget_melt_rate`, `linear_inverse_eulerian_budget_melt_rate` | Linearised twins of 7a/7b with a Stubblefield-kernel estimator; consistency check, not a melt product. |
| 7e | Space–time assimilation, B-PINN (diagnostic, `jaxmelt` env) | `pig.prep_bpinn_trunk` → `pig.run_bpinn` | `dynamics.bpinn.fit_bpinn` | Thickness surrogate `H(x,y,t)` + melt network with the thickness equation as likelihood; bridging transfer as an FFT observation operator (`--n-bins` makes it local; opt-in, measured neutral). PIG trunk re-run with the corrected operator (2026-09-13): 71.3 Gt/yr window flux, half-stack σ 21.7 m/yr; earlier trunk results predate the orientation fix and are not to be quoted. Checks before quoting: fitted `time trend` matches observed thinning, and `--no-transfer` changes the answer. Twin: `prep_bpinn_twin.py multixy_pigreal multixy_bmb clean` then `run_bpinn_twin.py --tag multixy_pigreal_clean --transfer --sigma-h 0.5 --no-planes --sigma-r 1 --col-slices 24 --steps 10000 --ensemble 1` (corr 0.693; 0.231 without `--transfer`). **Not a melt product.** |

Melt-rate sign everywhere: negative = melt, positive = accretion.

**Method descriptions** (the math behind stages 5e and 7): the governing
equations and solver forms live in the library docstrings
(`src/stereo_melt/melt.py` — Shean et al. 2019 Eq. 4/7/10 and sign
conventions; `dynamics/budget_linear_inverse.py` — the budget-inverse
rationale), and the manuscript-level prose lives in
[`literature/methods.md`](./literature/methods.md) (+ `methods.tex`), with
full derivations in `literature/closed_form_methods.md` (Stubblefield
closed form) and `literature/methods_budget_linear_inverse.md` (budget linear
inverse). The `literature/plan_*.md` notes are the design-decision records
behind each solver.

## Long runs

Heavy stages (align fleets, tilt LSQ at >150 epochs, solvers on 25 m grids)
run detached: `nohup setsid bash <basin>/scripts/<chain>.sh > <basin>/logs/<chain>.log
2>&1 < /dev/null &` — chain scripts echo `<NAME>_DONE` / `<NAME>_FAILED_*`
sentinels. Tilt runs on CPU (`STEREO_MELT_BACKEND=numpy OMP_NUM_THREADS=14`);
`STEREO_MELT_BACKEND=cupy` is strict (no silent numpy fallback).

## Shared batch drivers

| Stage | Shared driver |
|---|---|
| `align_strips` | `stereo_melt.coregister.align_driver.run_align_strips_main` |
| `cache_atm` / `cache_lvis` | `stereo_melt.coregister.cache_airborne.run_cache_airborne_main` |
| `cache_glas` | `stereo_melt.coregister.cache_glas.run_cache_glas_main` |
| `cache_cryosat2` / `cache_cryotempo` | `stereo_melt.coregister.cache_cs2` entry points |

The align driver exposes the union feature set in every basin: `--control
is2|cs2|is2+cs2|glas|none`, `--cs2-source raw|cryotempo` (default
`cryotempo`, the validated production base), `--use-atm/--use-lvis/
--use-glas`, row-balancing flags, `--asp-suffix` variant roots,
`--start/--end` era splits, `--parallel`, a disk-floor guard, and an
upfront no-control SKIP filter that runs *before* `--dem-id/--limit`.
Basin entry scripts start with `import stereo_melt.envsetup` (the shared
PROJ_DATA fix). Exception: `beardmore/` (legacy shared-root layout,
intentionally not ported).

## How to add a new basin

```bash
python scripts/new_basin.py --name <basin> --shelf-title <Title> \
    --start YYYY-MM-DD --end YYYY-MM-DD [--aoi <shp>] [--tide-model CATS2008]
```

This stamps `examples/<basin>/` — config + every stage wrapper + a local decision record — from
the canonical templates, resets `BAD_STRIPS`/`BAD_EPOCHS`, rewrites the
window/AOI lines, and import-tests the result. **Caveat:** the acquisition
stages (fetch/caches/align) are thin wrappers over shared library drivers,
but the stamped heavies (`build_stack`/`tilt_fit`/`run_*`) are still
copied-with-renames from the template basin, NOT library-driven — they can
lag the newest machinery (e.g. the full-domain tilt options), so review
them against the freshest basin. Then:

1. Drop the AOI shapefile (or run `scripts/compute_aoi.py <basin>` after a
   seed fetch for the strip-aware v15 rectangle).
2. Decide the per-shelf record: MDT latitude rule (geoid-only south of
   −79°S), velocity source (MEaSUREs NSIDC-0754 default; ITS_LIVE only where
   feature tracking works), control eras the window touches.
3. Run stages 0→6 (control caches as the eras require), populate
   `BAD_STRIPS` from the QC output, re-run `tilt_fit`, run solvers (7a/7b).
4. Write the basin decision record as **deltas only** — identity, decision
   record, local caveats; this file stays the sequence reference.

The library functions don't need to change for a new basin. If one needs new
physics, the corresponding library function grows a parameter (per-epoch `Ez`
and the tilt observation-domain/smoothness options are the precedents) —
never branch on basin name inside the library.

## Caveats

- **Pre-ASP corrections are forbidden.** `pc_align` references tide-free
  control; applying tide/IBE before it double-corrects. Tide/IBE are
  post-coreg only (stage 5a).
- **MDT skips below −79°S** (DTU22 coverage limit): `mdt_path=None`.
- **Per-basin ASP roots.** Each basin owns `<basin>/data/ASP*/`; sharing a
  root across basins leaks strips between AOIs.
- **Window-cap is index-driven.** `END_TIME` past 2024-01-10 is silently
  capped by the s2s041 strip-index ceiling (`acqdate1 max = 2024-01-09`).
- **Melt products retain negative ḃ** (accretion kept; `clip=False` on
  rasters — flux variants handle clipping explicitly).
- **Observation noise is correlated** (~4 km on PIG strips), so white-noise
  machinery misleads: the monolithic solver's `lam` is required (`"auto"` over-damps
  under correlated error), dimensionless, and specific to the posting it was tuned at.
- **Area-mean error bars need `n_eff`** (`stereo_melt.spatialstats`); on PIG use the
  block estimator (3–5 Gt/yr).

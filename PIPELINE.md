# Pipeline sequence

This document is the **canonical staged sequence** from raw REMA strips to a
basal melt-rate map, following Shean et al. 2019 (TC, PIG). It is the single
shared reference for every ice-shelf application: basin decision records (local, untracked)
carry only per-shelf deltas (AOI, window rationale, control eras, local
caveats) and point here for the sequence itself.

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
| 3b | Nocorr ingestion (Shean) | `<basin>.ingest_nocorr` (where ported) | copy + class-bias shift | Strips no control can anchor (fully-floating / align-failed / no-cache / QC-readmitted) enter at a-priori geolocation + one **per-era class-mean z-offset** into `ASP_nocorr/`, to be datum-corrected by the tilt LSQ (Ez 1.0). Shean `stack_nocorr_adjust.py` analog. |
| 3c | Align-root QC screen | `<basin>.scripts.qc_glas_align --root <ASP_x>` (bshelf pattern) | on-disk align evidence | Per-strip bias screen over initial/final geodiffs: quarantine misconverged (`|after_med|>2 m`) and under-determined (`n_ctl<50`); under-determined quarantines re-enter as nocorr. |
| 4 | Stack on common grid | `<basin>.build_stack` | `stereo_melt.stack.target_grid`, `save_stack` | Reproject coregistered strips onto the basin grid (EPSG:3031; per-basin `config.RES` — 25 m in most basins, 125 m on the large beardmore_shelf AOI) → `(time, y, x)` DataArray with a `dem_id` coord. Multi-root fusion via the basin's `*_SOURCES` env mode (first occurrence of a granule stem wins, quality-ordered). zlib-compressed on save. |
| 5 | Post-coreg corrections + tilt | `<basin>.tilt_fit` | see 5a–5e | Shean 2019 order, all static fields before the per-epoch LSQ. |
| 5a | tide + IBE | inline | `corrections.post_coreg.apply_tide_ibe_to_stack` | CATS2008 tides + ERA5 IBE (one scalar per strip), gated by a 3 km feathered floating-ice mask. |
| 5b | MDT (skip south of −79°S) | inline | `pipeline.apply_mdt_to_stack` | DTU22 MDT over floating ice. Basins poleward of −79°S (the Ross-sector Beardmore studies) are geoid-only. |
| 5c | Geoid | inline | `pipeline.apply_geoid_to_stack` | Ellipsoid → geoid-referenced surface. |
| 5d | Static-control mask | inline | `coregister.tilt.build_static_control_mask` | BedMachine rock + grounded ice ∩ Shean temporal filter. |
| 5e | Per-epoch tilt LSQ | inline | `coregister.tilt.fit_tilt_stack` | Joint LSQ: per-pixel intercept + trend + per-strip `(αx, αy, αz)`, Tikhonov priors, Tukey IRLS. **Per-epoch Ez** from `coregister.control_source.build_per_epoch_ez` resolved by `dem_id` against the align roots' sources sidecars (IS2/ATM/LVIS 0.1 m · GLAS 0.3 m · CryoTEMPO 1.0 m · nocorr 1.0 m · raw-CS2 2.0 m). Two domains: `static` (control-only, legacy default) or `full` (Shean ndinterp parity: all ice pixels incl. shelf) — **`full` must pair with the dh/dt Laplacian smoothness rows** (`dhdt_smoothness=1.0`, Shean L574+), which is what lets cross-epoch self-consistency set datums for control-free epochs without the front-accretion aliasing nullspace (validated 2026-07-11 on beardmore_shelf). Poorly-coregistered strips (pc_align `end_p50` > 3 m) are demoted to αz-only, not dropped. |
| 6 | QC: bad-strip screen | `<basin>.scripts.find_bad_epochs` | `coregister.tilt_qc.score_tilt_residuals`, `suggest_bad_epochs` | Score per-epoch residuals/IRLS weights; flag strips the LSQ couldn't lock. Populate **`config.BAD_STRIPS` (dem_id-keyed; date-keyed `BAD_EPOCHS` is legacy)**, re-run stage 5. Second tier: melt-space per-epoch bias screen (`diag_stripe_strips`, bshelf pattern). |
| 6b | Window-minimum shelf extent (calving-aware mask) | `<basin>.build_min_extent_mask` (pig first) | `stereo_melt.shelf_extent.min_shelf_extent` + `io.greene.greene_min_extent_on_grid` | Intersection of BedMachine floating ∩ Greene et al. 2022 observed annual extents over the window (archive ends 2021.2) ∩ the stack's per-epoch ocean test (`< 5 m` freeboard at ≥2 epochs). Pixels the shelf lost mid-window (e.g. PIG 2017-2020 tongue) otherwise enter the solvers as ice-to-ocean dh/dt cliffs. Cached NetCDF; `run_melt` intersects its floating mask with it when present (`<BASIN>_MIN_EXTENT=0` opts out). Gate: `tests/gate_min_shelf_extent.py`. |
| 7a | Lagrangian path solver (**production**) | `<basin>.run_melt_path` | `stereo_melt.melt.lagrangian_melt_rate` + `flux.integrate_basal_flux` | Path-integrated Lagrangian melt (Shean frame; Δt floor 1.5 yr), firn-corrected (`d` = BedMachine FAC; endpoint-exact Δd where wired), flux with CLIP/raw/robust variants + GL-2km band. |
| 7b | Eulerian solver (**production, second frame**) | `<basin>.run_melt` | `stereo_melt.melt.eulerian_melt_rate` | `b_dot = ∂H_f/∂t + ∇·(H_f·u) − a_dot` (Shean Eq. 10; matches `melt.py` — negative = melt). |
| 7c | Budget linear inverse (diagnostic twins) | `<basin>.compare_budget_inverse` (where ported) | `dynamics.linear_inverse_budget_melt_rate` (path-attributed) + `linear_inverse_eulerian_budget_melt_rate` | Linearised twins of 7a/7b — same sampling, budget correction, and attribution frame, pair-banded Stubblefield-kernel estimator. Consistency check + frame-flip diagnostic, not a melt substitute. Distinct from the legacy spectral inverse (retired 2026-06-13). |
| 7d | Spectral / stationary variants (diagnostic) | `<basin>.run_pseudospectral`, `run_stationary` | `dynamics.pseudospectral_*` | Spectral Lagrangian variant and stationary diagnostic. **Not melt products.** |

Melt-rate sign everywhere: negative = melt, positive = accretion.

**Method descriptions** (the math behind stages 5e and 7): the governing
equations and solver forms live in the library docstrings
(`src/stereo_melt/melt.py` — Shean Eq. 4/7/10 with sign
conventions; `dynamics/budget_linear_inverse.py` — the budget-inverse
rationale), and the manuscript-level prose lives in
[`literature/methods.md`](./literature/methods.md) (+ `methods.tex`), with
full derivations in `literature/closed_form_methods.md` (Stubblefield
closed form), `literature/inverse_method_explainer.md` (pseudo-spectral CG
inverse), and `literature/methods_budget_linear_inverse.md` (budget linear
inverse). The `literature/plan_*.md` notes are the design-decision records
behind each solver.

## Long runs

Heavy stages (align fleets, tilt LSQ at >150 epochs, solvers on 25 m grids)
run detached: `nohup setsid bash <basin>/scripts/<chain>.sh > <basin>/logs/<chain>.log
2>&1 < /dev/null &` — chain scripts echo `<NAME>_DONE` / `<NAME>_FAILED_*`
sentinels. Tilt runs on CPU (`STEREO_MELT_BACKEND=numpy OMP_NUM_THREADS=14`);
`STEREO_MELT_BACKEND=cupy` is strict (no silent numpy fallback).

## Shared batch drivers (2026-07-10 harmonization)

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
lag the newest machinery (e.g. the Shean-complete tilt options currently
exist only in `beardmore_shelf/tilt_fit.py`), so review them against the
freshest basin before trusting. Then:

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

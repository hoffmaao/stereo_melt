# CLAUDE.md — stereo_melt/ (the library)

Guidance for working on the **library** package itself. For the pipeline
sequence the basins run, see [`../PIPELINE.md`](../PIPELINE.md); for
per-shelf specifics, each basin directory's `CLAUDE.md`.

## Architectural rule

`stereo_melt/` is a **library**: composable algorithm functions, no awareness
of pipeline stages, study windows, or basin AOIs. The sequence lives in the
basin driver packages (one per ice shelf — `ls` the workspace root). Imports
flow basin → library, never the other way. Don't branch on basin name inside
the library; if a basin needs new behaviour, add a parameter — the per-epoch
`Ez` array, the tilt `observation_mask`/`dhdt_smoothness` options, and the
`DivergenceEstimator` seam are the precedents.

## Environment

Installed in a conda env outside the repo; the harness sandbox blocks
`source`, so never `conda activate` — call the interpreter directly:

```bash
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
$PY -m pip install -e .        # after editing pyproject / adding modules
$PY -m ruff check src/         # lint (if ruff is missing, install it in the env)
```

Array-heavy stages honor `STEREO_MELT_BACKEND=cupy` (see `src/stereo_melt/backend.py`);
default numpy. GPU mode is **strict** — no silent numpy fallback. I/O and
plotting always go through `to_numpy()`. `stereo_melt.envsetup` applies the
PROJ_DATA fix and must be the first import in basin entry scripts.

## Testing

`tests/` holds synthetic **gates** — run the one matching the subsystem you
touch, and add a gate when shipping a new solver or a solver-adjacent
correction:

- `gate_tilt_smoothness.py` — Shean-complete tilt (full domain + dh/dt
  Laplacian) recovers nocorr datums; static-domain failure rung.
- `gate_eulerian_budget_inverse.py`, `gate_match_lagrangian.py`,
  `gate_flux_amplitude.py`, `gate_c_timevarying_recovery.py` — solver gates.
- `sanity_eulerian_melt.py`, `sanity_lagrangian_melt.py` — end-to-end
  synthetic shelf checks.
- `diagnose_*` / `probe_*` — exploratory scripts, not gates.

## Package map (`src/stereo_melt/`)

- **`io/`** — REMA strip search/download (`rema.py`; collapses MultiPolygon
  AOIs before `pdemtools.search`), BedMachine (incl. `load_firn_on_grid`),
  MDT, velocity, SMB (RACMO2.4p1), masks.
- **`coregister/`** — the alignment + tilt subsystem:
  - `asp.py` — **ASP `pc_align` is the primary coregistration** (6-DOF
    point-to-plane ICP; binary at `/opt/StereoPipeline-3.5.0-alpha-*/bin/`).
    Never replace with pdemtools' 3-DOF Nuth & Kääb.
  - `align_driver.py` — shared batch driver behind every basin's
    `align_strips` (union flag set; see PIPELINE.md).
  - control caches: `cache_cs2.py`/`cache_cryotempo.py` (CryoTEMPO-LI is the
    CS2 control base), `cache_airborne.py` (ATM/LVIS), `cache_glas.py`
    (ICESat-1), plus `reference.py` (control CSV assembly), `cryotempo.py`,
    `retrack_lmg.py`.
  - `control_source.py::build_per_epoch_ez` — per-epoch αz prior by control
    class, resolved per `dem_id` from align-root sources sidecars
    (IS2/ATM/LVIS 0.1 m · GLAS 0.3 m · CryoTEMPO 1.0 m · nocorr 1.0 m ·
    raw-CS2 2.0 m).
  - `tilt.py::fit_tilt_stack` — Shean ndinterp-style joint LSQ (per-pixel
    intercept + trend, per-strip planes, Tikhonov priors, Tukey IRLS,
    per-pixel z_ref pre-subtraction). Key options: `observation_mask`
    (static-control vs full-ice domain), `dhdt_smoothness` (Shean L574+
    Laplacian rows — REQUIRED with the full domain), per-epoch `Ez`,
    `offset_only_epochs` (αz-only demotion), `robust_max_iter`.
  - `tilt_qc.py` — shared bad-strip screen behind `find_bad_epochs`.
  - `alignment_quality.py`, `dedup.py`, `jitter.py` — align evidence
    aggregation and strip hygiene.
- **`stack.py`** — common-grid reprojection to `(time, y, x)` with `dem_id`
  coord; zlib-compressed `save_stack`/`load_stack`; `load_basin_stack`
  resolves `--res`/`--tag` variant suffixes and applies
  `BAD_STRIPS`/`BAD_EPOCHS` drops.
- **`corrections/` + `pipeline.py`** — post-coreg statics in Shean order
  (tide via CATS2008/pyTMD, scalar-per-strip IBE from ERA5 MSL, MDT, geoid)
  with the 3 km feathered floating mask. Pre-ASP corrections are forbidden.
- **`melt.py`** — the production solvers: `eulerian_melt_rate` (Shean Eq. 10)
  and `lagrangian_melt_rate` (path solver; Δt floor 1.5 yr). Units: velocity
  m/yr, melt m ice-eq/yr, sign negative = melt. `freeboard.py`,
  `kinematics.py` (incl. the `DivergenceEstimator` Protocol seam), `flux.py`
  (basal-flux integration, GL buffers), `melt_qc.py` (melt-space per-epoch
  bias screens).
- **`dynamics/`** — solver variants and inverses. Two linear-inverse
  families; keep the names straight:
  - **budget linear inverse** (`budget_linear_inverse.py`:
    `linear_inverse_budget_melt_rate` path-attributed +
    `linear_inverse_eulerian_budget_melt_rate`) — linearised twins of the
    production solvers, run via `compare_budget_inverse` as consistency
    checks and the frame-flip diagnostic. Current.
  - **legacy spectral/wavenumber inverse** (`lagrangian_inverse.py`:
    `linear_inverse_lagrangian_melt_rate` + dhdt variant;
    `stubblefield_inverse.py`/`linear_perturbation.py`/`perturbation_dct.py`)
    — Stubblefield non-hydrostatic transfer, DC-blind; retired from melt
    products 2026-06-13, survives as an optional `melt_rate_linear_inverse`
    extra in some `run_melt` drivers and the stationary/pseudospectral
    diagnostics. NOT the budget linear inverse.
  Plus `bridging_restoration.py` (bounded complex 1/T(k; α_eff) filter that
  restores bridging-damped along-flow melt structure on the Eulerian-family
  outputs; band-limited ≥2.5H, mirror-padded, α calibrated ×0.34 vs E1b —
  see `../literature/stubblefield_applicability_prefactor.md`; gate
  `tests/gate_bridging_restoration.py`), `pseudospectral*.py` (spectral
  Lagrangian family; FFT stages honor cupy), `velocity_fusion.py` (EOF+GP
  Kalman fusion), `parcel_frame_inverse.py` (opt-in).
- **`constants.py`** — ρ_i=918, ρ_w=1027.

There is **no library-level study config** — all windows/AOIs/paths live in
basin `config.py` files.

**Where the melt-inversion math is described:** module docstrings carry the
governing equations (`melt.py` — Shean Eq. 4/7/10 + sign convention;
`dynamics/budget_linear_inverse.py` — budget-inverse rationale); the
manuscript-level prose and derivations live in `../literature/methods.md`
(+ `methods.tex`, `closed_form_methods.md`, `inverse_method_explainer.md`,
`methods_budget_linear_inverse.md`).

## Directories outside the package

- `../legacy/` — pre-refactor scripts; not a working baseline, no compat shims.
- `../vendor/` — read-only reference clones (a learned-flux-surrogate net;
  Shean-era `ndinterp.py` vendor checks). Not dependencies.
- `tests/figures/`, `data/` etc. — gitignored.

## Conventions

- All raster work on **EPSG:3031**; x ascends, y descends; CRS written on the
  stack at build time.
- Route backend-eligible arithmetic through `stereo_melt.backend` (`xp`);
  keep numpy for I/O, plotting, xarray construction.
- Heavy LSQ/IRLS/solver stages must **narrate progress** (phase prints with
  timings) — they run under `nohup` and the log is the only visibility.

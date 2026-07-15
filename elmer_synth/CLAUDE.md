# CLAUDE.md — elmer_synth/ (synthetic melt-truth driver)

**Not an ice shelf.** This driver generates *independent-physics* synthetic
observations (full-Stokes truth the inverse operators did not manufacture) and
scores the melt solvers against prescribed melt. Design doc:
[`../literature/plan_elmer_synthetic_validation.md`](../literature/plan_elmer_synthetic_validation.md)
(experiment ladder E0–E3, metrics M1–M8, two-tier observation synthesis).
Motivation: every pre-existing synthetic test generated h with the same
Stubblefield operator the spectral inverses invert (inverse crime); this driver
breaks that circularity.

## Environment — NOT the stereo_melt env

The truth generators run in the `fenicsx` conda env (dolfinx 0.7.3). The JIT
needs the env's conda compilers or it grabs system gcc 4.8.5 and dies:

```bash
FX=/home/hoffmaao/miniconda3/envs/fenicsx/bin/python
PATH=/home/hoffmaao/miniconda3/envs/fenicsx/bin:$PATH \
CC=x86_64-conda-linux-gnu-cc CXX=x86_64-conda-linux-gnu-c++ \
nohup nice -n 10 $FX -u elmer_synth/scripts/run_e1b.py > elmer_synth/logs/e1b_run.log 2>&1 &
```

Analysis/plotting scripts (`e1b_qc.py`, `e1b_diff.py`, `e1b_figures.py`) are
plain numpy/matplotlib — either env runs them.

For E2 (Elmer/Ice): udocker container `elmerice` (v9.0, MUMPS+Lua,
ElmerIceSolvers) at `~/.udocker`, launcher in `~/tools/udocker-venv`; P-mode
proot, serial/OpenMP first (MPI-under-proot unproven). `sif/` holds configs.

## Experiments (status 2026-07-13)

- **E1** — vendored Stubblefield FEniCSx nonlinear flowline (closed box, paper
  config λ=20H, m0=5 m/yr): `scripts/e1_run_nonlinear.py` →
  `results/e1_nonlinear_result.npz`; scored by `scripts/e1_score_vs_linear.py`
  → `e1_score_vs_linear.{json,png}`. **Result:** our
  `linear_perturbation.forward` matches the vendored Green's function to
  1e-8 m; nonlinear n=4 truth is ×1.147 (upper) / ×1.217 (lower) deeper than
  linear theory at r≈0.999 shape agreement.
- **E1b** — through-flow fork (`flowline/`, forked from
  `vendor/linear-shelf-melt/nonlinear-model/`): inflow plug (u0,0), open
  ocean-pressure front, ocean-fixed melt anomaly, `bfac(x)` prefactor hook for
  R-runs. `scripts/run_e1b.py`, env knobs `E1B_TF_YR` / `E1B_NT` /
  `E1B_M0_MYR` (0 = control) / `E1B_OUT`. **Result:** steady state; melt −
  control differencing gives scoop −17.3 m at +2.9 u₀t_r downstream, plateau
  −11.9 m vs −19.6 m pure-advection = 39% dynamic recovery; dh/dH = 0.0999 vs
  hydrostatic 0.1010. QC `scripts/e1b_qc.py`, differencing
  `scripts/e1b_diff.py`, figures `scripts/e1b_figures.py` → `figures/`.
- **E2a/E2b** — Elmer/Ice floating shelf / reduced-MISMIP+ contact: planned,
  `sif/` empty.

## Fork gotchas (all through-flow-specific; vendored closed box unaffected)

- The 07-13 debug fixed four defects — wall-corner basal-buoyancy facet
  dropped by the locator TOL; follower "sea-spring" must be −ρw·g·dt·u_z (the
  vendored (u·n) form adds ~13% fake backpressure on the front); surface
  kinematics are numpy per-column **upwind** (centered slope = FTCS =
  unconditionally unstable; front CFL needs dt=0.05 yr); Newton warm-start
  needs `atol=1.0` (rtol alone stalls at the LU roundoff floor).
- First-order upwind ⇒ numerical diffusion κ≈u·dx/2 (Pe≈55 at the anomaly):
  quasi-steady profiles shift ~2%, transients smear — fine for M6/M8, revisit
  for M1 short-λ work.
- Open-front boundary layer: one-column notch at x=+L/2 (common-mode with the
  control; difference it away, keep the science zone ≥ a few km from the
  front).
- **Sign convention here is Stubblefield: m > 0 = melt** (workspace melt
  products use negative = melt) — pin the flip at the harness boundary when
  feeding solver inputs.

## Import discipline

Same monorepo rule as basins: `elmer_synth` → library, never the reverse. The
survey-realistic tier must package truth as a `save_stack`-compatible NetCDF
so `tilt_fit` → screening → `run_melt` run unmodified.

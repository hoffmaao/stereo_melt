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

**2026-08-03 directive: Elmer 3D is the sole truth generator going forward.**
The Stubblefield-derived FEniCSx flowline tier (E1/E1b, `flowline/`, the
`fenicsx` env above) is **FROZEN** — redundant with Elmer 3D for any new
question; its results below (incl. the Glen H1 quartet) stand as the
historical record, but author no new flowline experiments. This retires the
flowline *simulator only*: the Stubblefield linear transfer function inside
the inverse solvers is the operator under test, not a truth generator, and is
unaffected. Corollary: the shipped `alpha_scale` 0.34 was measured on
Newtonian E1b — if it is ever revisited, re-fit it on Elmer Newtonian twins
(`fit_alpha_factor.py`), not on new flowline runs.

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
- **E2a** — Elmer/Ice 3D floating shelf (`e2a/{params,mesh,sif}.py`,
  `scripts/run_e2a.py`; runs land in `runs/<name>/`, sif copies in `sif/`).
  **Science campaign COMPLETE 07-19**: cosx/cosy single-k runs, steady +
  sine-modulated Gaussian channels at requalified dt=0.25, and the
  survey-realistic DEM-stack tier (`scripts/make_dem_stack.py` →
  `score_dem_stack_tilt.py` → `run_dem_stack_melt.py`, with `--sweep`
  time-series and `--solvers full` bridging modes). Findings consolidated in
  `../literature/bridging_approximation_assessment.md`.
  **⚠ THE FLOTATION ARTIFACT (attributed 2026-08-21 — read before using any
  E2a surface).** Every E2a run's surface carries
  `ε = z_s − f_b H = S·M·Δt` — one timestep's worth of melt thickness, a rigid
  upward offset over any melt anomaly, at *every* wavelength and *both*
  orientations. It is `ElmerIceUSF::SeaSpring` (`Buoyancy.F90`:546,
  `C = ρ_w g Δt N_s`): the spring anticipates the base moving by `u_n Δt`
  within the step, which is exact only where `u_n = 0`, and a melting steady
  state has `u_n N_s = −ṁ`. So it is **not** "a short-λ axis-x dt artifact" as
  this file said before — it is everywhere, and it is inert along-flow only
  because `u·∇ε = 0` there. A free 2-parameter fit measures it at
  `c = 1.00 ± 0.05` and leaves `r_bridge = 1.00` at r² ≥ 0.999.
  Consequences: **E2a measures no viscous bridging at all**, the "strongly
  anisotropic transfer" finding is retracted, and the across-flow melt-inverse
  error extrapolates to **zero** in both dt and the spring coefficient.
  Diagnose/remove with `scripts/diagnose_dt_splitting.py --part {law,attrib,
  spring,bridge}`; the spring is load-bearing for stability, so it cannot
  simply be weakened (×0.2 and ×0 both diverge — see the run table there).
  **✅ FIXED 2026-08-22 — `run_e2a.py --basal-melt-buoyancy` is now the
  standing configuration for every new E2a run.** It sets Elmer's own
  `Buoyancy Use Basal Melt` / `Bottom Surface Name`, switching `SeaPressure`
  to `pw = −ρ_w g (Z_sl − S − a_perp·Δt·N_s)` (`Buoyancy.F90`:355). That term
  anticipates the base moving by `a_perp Δt`, the spring anticipates `u_n Δt`,
  and a melting steady state has `u_n N_s = −ṁ = −a_perp`, so they cancel
  exactly — keeping the spring at ×1 (and its conditioning) while removing the
  bias. Measured on `runs/trans_gauss_bmb`, a single-knob A/B of
  `trans_gauss_a5`: `ε/(MΔt)` **0.996 → −0.007**, peak |ε| **2.4805 → 0.0854 m**,
  free-fit `c_artifact` **+0.995 → −0.022**, and the across-flow melt inverse
  **nrmse 2.155 → 0.039 at corr 0.9991 with zero shift** (monolithic 0.035;
  perfect-H ceiling 0.022) — better than post-hoc subtraction (0.069), at
  2.75 min/step, i.e. no cost. `trans_gauss_a5`'s "2.15 physics floor" was
  entirely the artifact. See [[project-e2a-bmb-artifact-fix-2026-08-22]].
  ~~**Why 3D:** the bridging kernel is isotropic in |k| (`linear_perturbation.py`
  :215) and hardcodes `alpha=0` (`budget_linear_inverse.py:588`), so it *must*
  correct across-flow and along-flow channels of equal width identically —
  3D Stokes with through-flow will not, and a 2D flowline cannot pose the
  question.~~ *(The motivation is still sound; the 07-19 campaign did not
  deliver it. **The isotropy test is OPEN but now answerable**: the axis-y arm
  measured the artifact, and the axis-x arm had no signal — advection erases
  the thickness anomaly at the forcing k to 0.17–0.23 m while the artifact was
  2.5 m, S/N ≈ 0.09. With `--basal-melt-buoyancy` the artifact is ~0.085 m, so
  **S/N ≈ 2.7** and both arms become measurable. The redo is the quartet
  `cos{x,y}_L{3,4}H_bmb`, each a 20 yr restart of its 07-19 predecessor with
  the flag on, scored by `--part bridge`.)* Forcing is a **zero-mean cosine at
  a single |k|**, not a Gaussian:
  E1b's lambda~20H Gaussian sat on the kernel's flat k->0 floor (+3.85%), which
  `_pooled_kernel_correction` median-subtracts (`:598`) — which is why the 07-16
  score saw a ~1% correction. Probe **lambda/H ~ 3-5** (knee is lambda~2*pi*H,
  net correction peaks ~3H at ~28% of hydrostatic) at **<= 200 m posting**.
- **E2b** — reduced-MISMIP+ contact: planned, unauthored. `Tests/GL_MISMIP` in
  the container is the crib (it is E2a's crib too, plus the contact stack).

## Elmer container facts (verified 07-17)

- The image ships the **full Elmer/Ice source tree** at `/home/glacier/elmerice`
  including `builddir/elmerice/Tests/` — use it as the syntax authority rather
  than guessing. `Tests/GL_MISMIP` = floating BCs (SeaPressure/SeaSpring USFs,
  Zs/Zb FreeSurface pair) + contact; `Tests/Damage` = runtime extrusion + `.grd`.
- **BC tags:** the `.grd` footprint gives 1=y0, 2=x=LX (front), 3=y=LY,
  4=x=0 (inflow); `Extruded Mesh Levels` appends **5=base, 6=top**. Bind them
  explicitly with `Target Boundaries` — do not rely on positional defaults.
- **Melt sign:** `FreeSurfaceSolver.F90:1196` integrates
  `dZ/dt + u.grad(Z) - w = Accumulation` for *both* surfaces, so
  **`Zb Accumulation > 0` raises the base = melt** (same as Stubblefield/E1b).
  Derived from source, **not yet confirmed by a run** — `run_e2a.py --check-sign`
  exists for exactly that and should be run before any melt result is believed.
- **Perf:** `Nonlinear System Max Iterations = 1` is *exact* for Newtonian
  (Stokes is linear; iter 2 reproduces the Result Norm to 17 digits) and halves
  the step. `Stabilized` matches `Bubbles` to ~2e-9 at 4x cheaper assembly. The
  remaining cost is the **linear solve** (~41 s at only 54k dof under MUMPS
  direct) — the open lever is iterative GCR+ILU1 or MPI-under-proot (unproven).

## Fork gotchas (all through-flow-specific; vendored closed box unaffected)

- The 07-13 debug fixed four defects — wall-corner basal-buoyancy facet
  dropped by the locator TOL; follower "sea-spring" must be −ρw·g·dt·u_z (the
  vendored (u·n) form adds ~13% fake backpressure on the front); surface
  kinematics are numpy per-column **upwind** (centered slope = FTCS =
  unconditionally unstable; front CFL needs dt=0.05 yr); Newton warm-start
  needs `atol=1.0` (rtol alone stalls at the LU roundoff floor). For n>1
  (Glen), warm-started full Newton falls into a period-2 residual limit
  cycle after the first mesh move — `stokes_solve` defaults to relaxation
  0.5 for `rm2 != 0` (~30 iters/step; env `E1B_NEWTON_RELAX`), full Newton
  stays the n=1 default (linear ⇒ exact in one iteration).
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

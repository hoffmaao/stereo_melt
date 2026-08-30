"""PIG: measure the melt-product NOISE FLOOR by independent half-stack splitting.

The wavenumber comparison (``pig.plot_melt_spectra``) showed our melt maps
carry 55-67 % of their variance at lambda < 3 km while Zinck's 50 m BURGEE
product carries 12 % and Davison's 1 km product 0.8 %. That is either real
channel structure the coarser products miss, or DEM/strip noise. This script
measures which.

Method. The DEM epochs are split into two halves that share the time span but
no strips: epochs are sorted by time and dealt alternately, so every strip's
error lands wholly in one half and the two melt solutions have INDEPENDENT
observation noise while seeing the SAME melt field (melt is taken steady over
the window, as every solver here already assumes). For two independent
estimates of one field,

    D = m_A - m_B          Var[D] = 2 sigma_half^2
    sigma_half^2 ~ 2 sigma_full^2      (noise variance scales as 1/N_epochs)
    => PSD_noise,full(k) = PSD_D(k) / 4

and the signal spectrum of the shipped (full-stack) product follows as
``PSD_signal = PSD_full - PSD_noise``, with ``SNR(k) = PSD_signal /
PSD_noise``. The wavelength where SNR falls to 1 is the resolution limit of
the product: structure shorter than that is noise-dominated, whatever the
solver does with it.

Caveat: velocity, SMB, firn and the mask are COMMON to both halves, so their
errors cancel in ``D`` and this is a DEM/strip noise floor, not a total error
budget. That is the term suspected of dominating the short-wavelength band;
velocity error is smooth and long-wavelength.

Solvers measured: the Eulerian baseline and the shipped restore-then-budget
(local, trunk-guarded, Helmholtz) — the question is whether the bridging
correction lifts signal or noise in the lambda <~ 3H band.

``--quarters`` additionally deals the epochs into four interleaved
quarter-stacks (stride 4, disjoint strips, full span) and solves each with
the Eulerian solver only (``eulerian_Q0..Q3`` in the same file), giving the
n/4 rung of the error-vs-DEM-count ladder that ``pig.plot_error_vs_count``
fits against the halves; it reads ``--out-suffix _q`` by default.

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    OMP_NUM_THREADS=14 nohup $PY -u -m pig.run_noise_floor > pig/logs/noise_floor.log 2>&1 &
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, "/wd2/projects/stereo_melt")
sys.path.insert(0, "/wd2/projects/stereo_melt/stereo_melt/src")

os.environ.setdefault("PIG_VELOCITY", "fused")

from stereo_melt import envsetup  # noqa: F401,E402

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from pig import config  # noqa: E402
from pig.run_melt import (  # noqa: E402
    apply_min_extent,
    load_floating_mask,
    load_smb_on_grid,
    load_stack,
    load_velocity_on_grid,
)
from stereo_melt.dynamics.bridging_restoration import restored_budget_melt_rate  # noqa: E402
from stereo_melt.io.bedmachine import load_firn_on_grid  # noqa: E402
from stereo_melt.kinematics import HelmholtzDivergence  # noqa: E402
from stereo_melt.melt import eulerian_melt_rate  # noqa: E402

RHO_I = 918.0


def out_path(suffix):
    return (config.PROCESSED_DIR /
            f"pig_noise_floor_250m_is2ctempo_sheltilt{suffix}.nc")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-bins", type=int, default=8)
    ap.add_argument("--lift-umax-myr", type=float, default=1500.0)
    ap.add_argument("--skip-rb", action="store_true")
    ap.add_argument("--quarters", action="store_true",
                    help="also solve four interleaved quarter-stacks (Eulerian "
                         "only) for the error-vs-count ladder")
    ap.add_argument("--common-epoch", action="store_true",
                    help="refer the mean thickness feeding the divergence to one epoch")
    ap.add_argument("--epoch-rate-sigma-px", type=float, default=2.0)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    t00 = time.time()
    stack = load_stack("pig_stack_250m_is2ctempo_sheltilt")
    floating = apply_min_extent(load_floating_mask(stack), "_250m_is2ctempo_sheltilt",
                                str(config.START_TIME), str(config.END_TIME))
    stack = stack.where(floating)
    vx, vy, vel_source = load_velocity_on_grid(stack)
    print(f"  velocity: {vel_source}", flush=True)
    a_dot = load_smb_on_grid(stack)
    firn = load_firn_on_grid(stack, config.BEDMACHINE_NC)

    # ---- deal the epochs alternately in time: same span, disjoint strips
    order = np.argsort(stack.time.values)
    idx_a, idx_b = order[0::2], order[1::2]
    sa, sb = stack.isel(time=np.sort(idx_a)), stack.isel(time=np.sort(idx_b))
    print(f"  split: {sa.sizes['time']} + {sb.sizes['time']} of {stack.sizes['time']} "
          f"epochs; spans {str(sa.time.values.min())[:10]}..{str(sa.time.values.max())[:10]}"
          f" / {str(sb.time.values.min())[:10]}..{str(sb.time.values.max())[:10]}", flush=True)

    out = {}
    guard = args.lift_umax_myr if args.lift_umax_myr > 0 else None

    def run(name, st, solver):
        t0 = time.time()
        ce = dict(common_epoch=args.common_epoch,
                  epoch_rate_sigma_px=args.epoch_rate_sigma_px)
        if solver == "eulerian":
            ds = eulerian_melt_rate(st, vx, vy, a_dot=a_dot, d=firn,
                                    robust_dh_dt=True, **ce)
        else:
            ds = restored_budget_melt_rate(
                st, vx, vy, a_dot=a_dot, floating_mask=floating, d=firn,
                robust_dh_dt=True, n_bins=args.n_bins, lift_umax_myr=guard,
                estimator=HelmholtzDivergence(), **ce)
        m = ds.melt_rate
        out[name] = m
        n = int(np.isfinite(m.values).sum())
        print(f"  [{name}] {n} finite px  flux "
              f"{-np.nansum(m.values) * 62500 * RHO_I / 1e12:.1f} Gt/yr  "
              f"({time.time() - t0:.0f}s)", flush=True)

    print("[1/4] Eulerian half A...", flush=True)
    run("eulerian_A", sa, "eulerian")
    print("[2/4] Eulerian half B...", flush=True)
    run("eulerian_B", sb, "eulerian")
    if not args.skip_rb:
        print("[3/4] restored local+Helm half A...", flush=True)
        run("rb_A", sa, "rb")
        print("[4/4] restored local+Helm half B...", flush=True)
        run("rb_B", sb, "rb")

    if args.quarters:
        # interleave stride-4 so each quarter spans the window with ~n/4
        # epochs per pixel and no strip shared between quarters
        for q in range(4):
            idx = np.sort(order[q::4])
            print(f"[quarter {q}] {len(idx)} epochs...", flush=True)
            run(f"eulerian_Q{q}", stack.isel(time=idx), "eulerian")

    ds_out = xr.Dataset(out)
    ds_out.attrs.update(velocity=vel_source, n_epochs_a=sa.sizes["time"],
                        n_epochs_b=sb.sizes["time"], n_bins=args.n_bins,
                        lift_umax_myr=args.lift_umax_myr,
                        common_epoch=int(bool(args.common_epoch)),
                        split="alternating epochs in time (disjoint strips)")
    nc = out_path(args.out_suffix)
    ds_out.to_netcdf(nc, encoding={k: {"zlib": True, "complevel": 4} for k in out})
    print(f"wrote {nc}", flush=True)
    print(f"[done] total {time.time() - t00:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

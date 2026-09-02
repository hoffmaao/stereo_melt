"""Synthetic gate for the Shean-complete tilt system (2026-07-11).

Gates the two pieces ported from vendor ``ndinterp.py`` that make
control-free (nocorr) epochs correctable:

- shelf-inclusive observation domain (``observation_mask`` wider than the
  static control), and
- the dh/dt spatial-smoothness constraint (``dhdt_smoothness``, his L574+
  second-difference rows on the per-pixel trend field).

Truth: a static grounded margin (trend 0) plus a floating shelf carrying a
linear along-x melt-gradient trend (max thinning at the "GL", tapering to
the "front") — exactly the structure the 2026-06-29 revert found aliasing
into per-epoch tilt. Trans epochs cover the whole domain with small datum
offsets; nocorr epochs have data ONLY on the shelf with meter-scale datum
offsets (Ez 1.0), mimicking Shean's never-coregistered DEMs.

Rungs:

- Rung A (mechanism reproduction) — static-only domain, no smoothness (the
  pre-07-11 production system): nocorr αz MUST collapse to the prior mean
  (~0), i.e. the datum error is NOT corrected. This pins the failure mode
  observed on the beardmore_shelf nocorr stack (αz=+0.02 for all 99).
- Rung B (the fix) — full domain + smoothness 1.0: nocorr αz recovered to
  <0.25 m, trans αz to <0.10 m, shelf trend field RMS error <0.30 m/yr and
  front-third mean bias <0.15 m/yr (no manufactured front accretion).
- Rung C (informational) — full domain WITHOUT smoothness: prints the same
  metrics to document the unstabilized system; no hard assert.

Run:

    /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python \
        tests/gate_tilt_smoothness.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import xarray as xr

from stereo_melt.coregister.tilt import fit_tilt_stack

RNG = np.random.default_rng(20260711)

NY, NX = 80, 120
RES = 200.0  # m
STATIC_X = 18          # columns [0, STATIC_X) are grounded margin (trend 0)
SHELF_X0 = STATIC_X    # shelf starts here
NOISE_M = 0.25
T_EPOCHS = 14
SPAN_DAYS = 4 * 365.25


def build_truth():
    x = np.arange(NX) * RES
    y = np.arange(NY) * RES
    # Smooth low-relief topography (m)
    z_ref = (
        40.0
        + 4.0 * np.sin(2 * np.pi * x[None, :] / 9000.0)
        + 2.5 * np.cos(2 * np.pi * y[:, None] / 7000.0)
    )
    # Trend field (m/yr): 0 on the static margin; on the shelf a linear
    # along-x gradient from -2.0 m/yr at the GL to -0.2 m/yr at the front.
    dhdt = np.zeros((NY, NX))
    xs = np.arange(NX)
    ramp = np.clip((xs - SHELF_X0) / (NX - 1 - SHELF_X0), 0, 1)
    dhdt[:, SHELF_X0:] = (-2.0 + 1.8 * ramp[SHELF_X0:])[None, :]
    return z_ref, dhdt


def build_stack(z_ref, dhdt):
    t_days = np.sort(RNG.uniform(0, SPAN_DAYS, T_EPOCHS))
    t_days[0], t_days[-1] = 0.0, SPAN_DAYS
    times = pd.Timestamp("2019-01-01") + pd.to_timedelta(t_days, unit="D")
    t_c = t_days - t_days.mean()

    # Epoch roles: 10 trans (full footprint, small offsets), 4 nocorr
    # (shelf-only footprint, meter-scale offsets, zero static pixels).
    nocorr_idx = np.array([3, 6, 9, 12])
    is_nocorr = np.zeros(T_EPOCHS, dtype=bool)
    is_nocorr[nocorr_idx] = True
    az_true = RNG.normal(0.0, 0.12, T_EPOCHS)
    az_true[nocorr_idx] = [+1.8, -2.4, +0.9, -0.3]  # mean 0 (DC gauge)

    layers = []
    for k in range(T_EPOCHS):
        z = (
            z_ref
            + dhdt * (t_c[k] / 365.25)
            + az_true[k]
            + RNG.normal(0, NOISE_M, (NY, NX))
        )
        # random voids (SETSM-style gaps)
        void = RNG.random((NY, NX)) < 0.25
        z[void] = np.nan
        if is_nocorr[k]:
            # shelf-only: no data anywhere near the static margin
            z[:, : SHELF_X0 + 2] = np.nan
            # one nocorr epoch with a small partial footprint
            if k == nocorr_idx[-1]:
                z[: NY // 2, :] = np.nan
        layers.append(z)

    da = xr.DataArray(
        np.stack(layers),
        dims=("time", "y", "x"),
        coords={
            "time": times.values,
            "y": np.arange(NY) * RES,
            "x": np.arange(NX) * RES,
        },
        name="h",
    )
    return da, az_true, is_nocorr


def run(label, stack, control, obs_mask, Ez, smoothness):
    params, _ = fit_tilt_stack(
        stack,
        control_mask=control,
        observation_mask=obs_mask,
        Ez=Ez,
        min_width=1e12,  # αz-only: isolate the datum mechanism
        dhdt_smoothness=smoothness,
        lsmr_atol=1e-10,
    )
    az = params["tilt_dz"].values
    dhdt_rec = params["dhdt"].values * 365.25  # m/day -> m/yr
    print(f"  [{label}] recovered αz: {np.array2string(az, precision=2)}")
    return az, dhdt_rec


def main() -> int:
    z_ref, dhdt_true = build_truth()
    stack, az_true, is_nocorr = build_stack(z_ref, dhdt_true)

    control = np.zeros((NY, NX), dtype=bool)
    control[:, :STATIC_X] = True
    full = np.ones((NY, NX), dtype=bool)

    Ez = np.where(is_nocorr, 1.0, 0.1)
    shelf = np.zeros((NY, NX), dtype=bool)
    shelf[:, SHELF_X0 + 2:] = True
    front_third = np.zeros((NY, NX), dtype=bool)
    front_third[:, NX - (NX - SHELF_X0) // 3:] = True

    print(f"truth αz (nocorr epochs): {az_true[is_nocorr]}")
    ok = True

    # ---- Rung A: static-only domain (pre-07-11 production) ----
    print("\nRung A — static-only domain, no smoothness (failure reproduction)")
    azA, _ = run("A", stack, control, None, Ez, None)
    errA = np.abs(azA[is_nocorr] - az_true[is_nocorr])
    collapsedA = np.all(np.abs(azA[is_nocorr]) < 0.05)
    print(f"  nocorr |αz_rec|max={np.abs(azA[is_nocorr]).max():.3f} m "
          f"(expect ~0: prior mean)  |err|max={errA.max():.2f} m")
    if not (collapsedA and errA.max() > 1.0):
        print("  ✗ Rung A: expected nocorr αz to collapse to prior (bug "
              "reproduction) — the static-only system unexpectedly moved αz")
        ok = False
    else:
        print("  ✓ Rung A reproduces the nocorr-αz-uncorrected failure")

    # ---- Rung B: full domain + smoothness (the Shean-complete fix) ----
    print("\nRung B — full domain + dh/dt smoothness 1.0 (the fix)")
    azB, dhdtB = run("B", stack, control, full, Ez, 1.0)
    errB_nocorr = np.abs(azB[is_nocorr] - az_true[is_nocorr])
    errB_trans = np.abs(azB[~is_nocorr] - az_true[~is_nocorr])
    shelf_rms = float(np.sqrt(np.nanmean((dhdtB - dhdt_true)[shelf] ** 2)))
    front_bias = float(np.nanmean((dhdtB - dhdt_true)[front_third]))
    print(f"  nocorr αz err: max={errB_nocorr.max():.3f} m")
    print(f"  trans  αz err: max={errB_trans.max():.3f} m")
    print(f"  shelf dh/dt RMS err: {shelf_rms:.3f} m/yr; "
          f"front-third mean bias: {front_bias:+.3f} m/yr")
    if errB_nocorr.max() > 0.25:
        print("  ✗ Rung B: nocorr αz not recovered")
        ok = False
    if errB_trans.max() > 0.10:
        print("  ✗ Rung B: trans αz degraded")
        ok = False
    if shelf_rms > 0.30:
        print("  ✗ Rung B: shelf trend field not recovered")
        ok = False
    if abs(front_bias) > 0.15:
        print("  ✗ Rung B: front trend bias (manufactured accretion/melt)")
        ok = False
    if ok:
        print("  ✓ Rung B: nocorr datum + shelf trend recovered, no front bias")

    # ---- Rung C: full domain WITHOUT smoothness (informational) ----
    print("\nRung C — full domain, NO smoothness (unstabilized, informational)")
    azC, dhdtC = run("C", stack, control, full, Ez, None)
    errC_nocorr = np.abs(azC[is_nocorr] - az_true[is_nocorr])
    shelf_rmsC = float(np.sqrt(np.nanmean((dhdtC - dhdt_true)[shelf] ** 2)))
    front_biasC = float(np.nanmean((dhdtC - dhdt_true)[front_third]))
    print(f"  nocorr αz err max={errC_nocorr.max():.3f} m; shelf RMS "
          f"{shelf_rmsC:.3f} m/yr; front bias {front_biasC:+.3f} m/yr")

    print("\n" + ("GATE PASSED" if ok else "GATE FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

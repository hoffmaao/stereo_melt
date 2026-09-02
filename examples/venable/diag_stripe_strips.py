"""Melt-space bad-strip attribution for the Venable red-stripe artefact.

The 125 m path-melt product shows horizontal ACCRETION bands at the edges of
the zero-pair coverage band (rows ~648-659, ~586-598, ~811-818; 2026-07-03
figure `melt_path_125m.png`). Hypothesis: thin-count strip bias — where only a
handful of pairs cover a cell, the cross-pair median cannot reject a strip
with a residual vertical-datum / tilt error, and the hydrostatic gain
(rho_w/(rho_w-rho_i) ~ 9.4) turns a decimeter surface bias into m/yr of fake
accretion.

This driver runs the production path solver once with the opt-in
``pair_diag_masks`` hook (one bool mask per anomalous band + the whole
floating shelf as reference), then attributes band anomalies to EPOCHS:

- A strip with thickness bias ``s_e`` (m, ice-equivalent) enters every pair
  it belongs to with opposite sign as earlier vs later member:
  ``med_p ≈ mu_region + (s_j - s_i)/dt_p``.
- Solve for ``{mu, s_e}`` per region by IRLS-Tukey LSQ (gauge: mean s = 0),
  weighting pairs by sqrt(band cell count).
- Cross-check with the model-free score ``(med over pairs with e as later) -
  (med over pairs with e as earlier)) / 2``.

Prints ranked offender tables + suggested BAD_STRIPS entries; saves the
per-pair diagnostics to ``processed/diag_stripe_pairs_<tag>.nc``.

Run:

    cd /wd2/projects/stereo_melt/examples
    /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python -u \
        -m venable.diag_stripe_strips --res 125 \
        2>&1 | tee venable/logs/diag_stripe_strips_125m.log
"""
from __future__ import annotations

import argparse
import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import time

import numpy as np
import pandas as pd
import xarray as xr

from stereo_melt.io.bedmachine import load_firn_on_grid
from stereo_melt.melt import lagrangian_melt_rate
from stereo_melt.melt_qc import fit_pair_epoch_bias

from venable import config
from venable.run_melt import (
    load_floating_mask,
    load_smb_on_grid,
    load_velocity_on_grid,
)
from venable.run_melt_path import load_stack_res

# Anomalous row bands from the 2026-07-04 row-median scan of
# venable_melt_path_125m (anom = row median - 31-row rolling baseline):
#   A rows 648-659: anom +2.8..+12.6 m/yr, ~330-358 floating cols (THE stripe)
#   B rows 586-598: anom +1.9..+4.2, ~256-284 cols
#   C rows 811-818: anom +2.3..+16.8, ~36-40 cols (narrow SW margin band)
BANDS = {
    "A_648_659": (648, 660),
    "B_586_598": (586, 599),
    "C_811_818": (811, 819),
}
MIN_BAND_CELLS = 30  # pair must deposit >= this many band cells to enter LSQ
HYDRO_GAIN = 1027.0 / (1027.0 - 918.0)  # rho_w / (rho_w - rho_i) ~ 9.42


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--res", type=int, default=125)
    args = p.parse_args()
    tag = f"{args.res}m"
    t0 = time.time()

    print(f"Loading curated {tag} stack...")
    stack = load_stack_res(args.res)
    times = pd.to_datetime(stack["time"].values)
    dem_ids = (
        stack["dem_id"].values
        if "dem_id" in stack.coords
        else np.array([str(t.date()) for t in times])
    )
    ny, nx = stack.sizes["y"], stack.sizes["x"]
    print(f"  dims: time={stack.sizes['time']}, y={ny}, x={nx}")

    floating = load_floating_mask(stack)
    stack = stack.where(floating)
    flo = np.asarray(floating.values, bool)

    masks, names = [], []
    for name, (r0, r1) in BANDS.items():
        m = np.zeros((ny, nx), bool)
        m[r0:r1] = True
        m &= flo
        masks.append(m)
        names.append(name)
        print(f"  band {name}: rows [{r0},{r1})  cells={int(m.sum()):,}")
    masks.append(flo.copy())
    names.append("REF_floating")
    masks = np.stack(masks)

    vx, vy, vel_source = load_velocity_on_grid(stack)
    a_dot = load_smb_on_grid(stack)
    firn = load_firn_on_grid(stack, config.BEDMACHINE_NC)

    print("Running production path solver with pair_diag_masks...")
    lagr = lagrangian_melt_rate(
        stack, vx, vy, a_dot=a_dot, d=firn, dt_yr=0.05,
        seed_stride=1, output="path", aggregator="pair_median",
        pairs="all", min_dt_yr=1.5, max_dt_yr=2.5,
        pair_diag_masks=masks,
    )

    pi = lagr["pair_i"].values
    pj = lagr["pair_j"].values
    dt = lagr["pair_dt_yr"].values
    cnt = lagr["pair_diag_count"].values  # (pair, region)
    med = lagr["pair_diag_median"].values
    n_ep = stack.sizes["time"]
    print(f"  {pi.size:,} dt-band pairs recorded; regions: {names}")

    diag = xr.Dataset(
        {
            "pair_i": ("pair", pi),
            "pair_j": ("pair", pj),
            "pair_dt_yr": ("pair", dt),
            "count": (("pair", "region"), cnt),
            "median": (("pair", "region"), med),
            "mean": (("pair", "region"), lagr["pair_diag_mean"].values),
        },
        coords={"region": names},
        attrs={"dem_ids": list(map(str, dem_ids)),
               "times": [str(t) for t in times]},
    )
    out_nc = config.PROCESSED_DIR / f"diag_stripe_pairs_{tag}.nc"
    diag.to_netcdf(out_nc)
    print(f"  saved pair diagnostics -> {out_nc}")

    flagged: dict[str, float] = {}
    for r, name in enumerate(names):
        sel = (cnt[:, r] >= MIN_BAND_CELLS) & np.isfinite(med[:, r])
        if sel.sum() < 5:
            print(f"\n=== {name}: only {int(sel.sum())} usable pairs, skipping ===")
            continue
        bi, bj, bdt, bmed = pi[sel], pj[sel], dt[sel], med[sel, r]
        bn = cnt[sel, r]
        mu, s, resid, w = fit_pair_epoch_bias(bi, bj, bdt, bmed, bn, n_epochs=n_ep)
        print(f"\n=== {name}: {bi.size} pairs, mu={mu:+.2f} m/yr ===")

        # model-free per-epoch score for cross-check
        score = np.full(n_ep, np.nan)
        for e in np.union1d(bi, bj):
            as_j = bmed[bj == e]
            as_i = bmed[bi == e]
            if as_j.size and as_i.size:
                score[e] = (np.median(as_j) - np.median(as_i)) / 2.0
        s_mad = 1.4826 * np.nanmedian(np.abs(s - np.nanmedian(s)))
        order = np.argsort(-np.abs(np.nan_to_num(s)))
        print(f"  per-epoch thickness bias s (m ice-eq): MAD={s_mad:.2f}; top 12:")
        print("  rank  ep  date        s_bias   ~dz_surf  score   nI  nJ  dem_id")
        for k in order[:12]:
            if not np.isfinite(s[k]):
                continue
            n_i = int((bi == k).sum())
            n_j = int((bj == k).sum())
            print(
                f"  {np.where(order == k)[0][0] + 1:4d}  {k:3d}  "
                f"{times[k].date()}  {s[k]:+7.2f}  {s[k] / HYDRO_GAIN:+7.2f}   "
                f"{(score[k] if np.isfinite(score[k]) else np.nan):+6.2f}  "
                f"{n_i:3d} {n_j:3d}  {dem_ids[k]}"
            )
        # outlier pairs, raw view
        big = np.argsort(-np.abs(bmed - np.median(bmed)))[:8]
        print("  most anomalous pairs (band median vs region mu):")
        for b in big:
            print(
                f"    med={bmed[b]:+7.2f}  n={int(bn[b]):5d}  dt={bdt[b]:.2f}  "
                f"{times[bi[b]].date()} -> {times[bj[b]].date()}"
            )
        if name != "REF_floating":
            for k in np.where(np.isfinite(s) & (np.abs(s) > 3 * s_mad) & (np.abs(s) > 4.0))[0]:
                flagged[str(dem_ids[k])] = max(
                    flagged.get(str(dem_ids[k]), 0.0), abs(float(s[k]))
                )

    print("\n=== suggested melt-space BAD_STRIPS additions ===")
    if not flagged:
        print("  (none above threshold |s|>max(3*MAD, 4 m))")
    for d_id, sv in sorted(flagged.items(), key=lambda kv: -kv[1]):
        print(f'    "{d_id}",  # melt-space stripe QC 2026-07-04, |s|={sv:.1f} m')
    print(f"DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

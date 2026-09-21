"""Drop uncontrolled (nocorr) PIG slices the tilt fit left unadjusted.

Runs :func:`stereo_melt.coregister.tilt_qc.screen_unrescued_epochs` on
``pig_stack_<res>m_<tag>_tilt_corrected_<window>.nc`` and writes the kept
slices under ``<out-tag>`` plus ``results/pig_unrescued_screen_<res>m_<out-tag>.csv``.

Run (from ``examples/``)::

    $PY -m pig.screen_unrescued --tag is2ctempo_sheltilt_full_nocorr [--out-tag X] [--res 250]
"""
from __future__ import annotations

import stereo_melt.envsetup  # noqa: F401  (PROJ_DATA fix, before geo imports)

import argparse
import time

import numpy as np
import xarray as xr

from stereo_melt.coregister.tilt import build_ice_domain_mask
from stereo_melt.coregister.tilt_qc import screen_unrescued_epochs

from pig import config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True,
                   help="tag of the tilt-corrected nocorr stack to screen")
    p.add_argument("--out-tag", default=None,
                   help="tag for the screened stack (default: <tag>scr)")
    p.add_argument("--res", type=int, default=250)
    p.add_argument("--nmad-max-m", type=float, default=5.0)
    p.add_argument("--blunder-m", type=float, default=20.0)
    p.add_argument("--blunder-frac-max", type=float, default=0.10)
    p.add_argument("--min-px", type=int, default=200)
    p.add_argument("--min-epochs-px", type=int, default=3)
    args = p.parse_args()
    out_tag = args.out_tag or f"{args.tag}scr"
    if out_tag == args.tag:
        raise SystemExit("--out-tag must differ from --tag (the input is not overwritten)")

    window = f"{config.START_TIME}_{config.END_TIME}"
    res = f"{args.res}m"
    src = config.PROCESSED_DIR / f"pig_stack_{res}_{args.tag}_tilt_corrected_{window}.nc"
    par = config.PROCESSED_DIR / f"pig_tilt_params_{res}_{args.tag}_{window}.nc"
    dst = config.PROCESSED_DIR / f"pig_stack_{res}_{out_tag}_tilt_corrected_{window}.nc"
    table = config.RESULTS_DIR / f"pig_unrescued_screen_{res}_{out_tag}.csv"
    for f in (src, par):
        if not f.exists():
            raise SystemExit(f"missing {f}")

    t0 = time.time()
    ds = xr.open_dataset(src)
    payload = [v for v in ds.data_vars if ds[v].ndim == 3]
    if len(payload) != 1:
        raise SystemExit(f"expected one (time, y, x) variable in {src.name}, got {payload}")
    stack = ds[payload[0]].load()
    params = xr.open_dataset(par).load()
    print(f"screening {src.name}: {stack.sizes['time']} slices "
          f"({int((stack['source_variant'] == 'nocorr').sum())} nocorr)")

    domain = None
    if "obs_support" not in params:
        domain = build_ice_domain_mask(stack, config.BEDMACHINE_NC).values
        print(f"  params lack obs_support: scoring over the BedMachine ice domain "
              f"(PIG_TILT_DOMAIN=full observation mask), {int(domain.sum())} px")

    df = screen_unrescued_epochs(
        stack, params, domain_mask=domain,
        nmad_max_m=args.nmad_max_m, blunder_m=args.blunder_m,
        blunder_frac_max=args.blunder_frac_max, min_px=args.min_px,
        min_epochs_px=args.min_epochs_px,
    )
    df.to_csv(table, index=False)
    bad = df[df["unrescued"]]
    scr = df[df["screened"]]
    print(f"  nocorr: {len(scr)} screened, {len(bad)} unrescued, "
          f"{len(scr) - len(bad)} kept")
    for _, r in bad.iterrows():
        print(f"    drop {r.get('dem_id', r['epoch'])}  {r['reason']}")
    kept = scr[~scr["unrescued"]]
    if len(kept):
        print(f"  kept nocorr, fit-frame residual: median |med| "
              f"{np.median(np.abs(kept['med_resid_m'])):.2f} m, median NMAD "
              f"{np.median(kept['nmad_m']):.2f} m")

    keep_idx = np.where(~df["unrescued"].to_numpy())[0]
    out = ds.isel(time=keep_idx)
    out.attrs.update({
        "unrescued_screen": (
            f"screen_unrescued_epochs on {src.name}: nmad_max_m={args.nmad_max_m} "
            f"blunder_m={args.blunder_m} blunder_frac_max={args.blunder_frac_max} "
            f"min_px={args.min_px} min_epochs_px={args.min_epochs_px}; "
            f"dropped {len(bad)} of {len(scr)} nocorr slices"
        ),
    })
    enc = {v: {"zlib": True, "complevel": 4} for v in out.data_vars if out[v].ndim == 3}
    out.to_netcdf(dst, encoding=enc)
    print(f"wrote {dst.name} ({len(keep_idx)} slices) and {table.name} "
          f"in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

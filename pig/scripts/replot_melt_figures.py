"""Re-render PIG melt-rate figures from saved NetCDF — no solver re-runs.

The melt products are expensive (the budget lin-inv fan is ~2 h, the torch
forward fit another ~20 min), so a presentation change — a colormap, a limit, a
label — must never require recomputing them. Every melt map this repo publishes
is a pure function of a saved `.nc`, and this script is that function applied to
the products already on disk:

  * `pig_melt_<res>_<tag>[_suffix]_<window>.nc`  -> `melt_comparison_*.png`
    (Eulerian / Lagrangian / Stubblefield, via `run_melt.plot_melt_comparison`)
  * `pig_fused_melt_map[_tag].nc`                -> `pig_fused_melt_map*.png`
    (Eulerian / budget lin-inv / variational / fused, via
    `fused_melt_map.render_map`)

Both call the SAME renderers the producing scripts call, so a figure re-rendered
here is byte-comparable with one written by a fresh run.

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY pig/scripts/replot_melt_figures.py               # every product found
    $PY pig/scripts/replot_melt_figures.py --only fused  # just the fused maps
"""
import argparse
import glob
import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"] = _env_proj

REPO = "/wd2/projects/stereo_melt"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "stereo_melt", "src"))
sys.path.insert(0, os.path.join(REPO, "pig", "scripts"))

import xarray as xr  # noqa: E402

import fused_melt_map as F  # noqa: E402
from pig import config  # noqa: E402
from pig.run_melt import plot_melt_comparison  # noqa: E402

RESULTS = os.path.join(REPO, "pig", "results")
FIGS = str(config.FIGURES_DIR)


def replot_production(path: str) -> None:
    """`pig_melt_*.nc` -> the 6-panel melt_comparison figure."""
    ds = xr.open_dataset(path)
    # plot_melt_comparison wants three solver Datasets; the Stubblefield panel
    # is optional and older products predate it.
    euler = xr.Dataset({"melt_rate": ds["melt_rate_eulerian"],
                        "flux_div": ds["flux_div"]})
    lagr = xr.Dataset({"melt_rate": ds["melt_rate_lagrangian"]})
    linv = (xr.Dataset({"melt_rate": ds["melt_rate_linear_inverse"]})
            if "melt_rate_linear_inverse" in ds else None)
    # run_melt names the NetCDF `pig_melt<suffix>_<START>_<END>.nc` but the
    # figure `melt_comparison<suffix><win_tag>.png`, where win_tag is EMPTY
    # unless --start/--end selected an analysis sub-window. So a product on the
    # default full-record window must have the window stripped back off, or the
    # replot writes a second file beside the canonical one instead of
    # refreshing it.
    stem = os.path.basename(path)[len("pig_melt"):-len(".nc")]
    full_window = f"_{config.START_TIME}_{config.END_TIME}"
    if stem.endswith(full_window):
        stem = stem[:-len(full_window)]
    out = os.path.join(FIGS, f"melt_comparison{stem}.png")
    plot_melt_comparison(euler, lagr, linv, out)
    print(f"wrote {out}")
    ds.close()


def replot_fused(path: str) -> None:
    """`pig_fused_melt_map*.nc` -> the 4-panel fused figure."""
    ds = xr.open_dataset(path)
    missing = [n for n in F.PANELS if n.replace(" ", "_") not in ds]
    if missing:
        print(f"SKIP {os.path.basename(path)}: missing {missing}")
        ds.close()
        return
    fields = {n: ds[n.replace(" ", "_")] for n in F.PANELS}
    tag = os.path.basename(path)[len("pig_fused_melt_map"):-len(".nc")]
    out = os.path.join(FIGS, f"pig_fused_melt_map{tag}.png")
    F.render_map(
        fields, ds.x.values / 1e3, ds.y.values / 1e3, out,
        f"PIG basal melt — four reconstructions on the same 250 m stack"
        f"{(' — ' + tag.lstrip('_')) if tag else ''}")
    ds.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["production", "fused"],
                    help="re-render just one family (default: both)")
    args = ap.parse_args()

    if args.only != "fused":
        for p in sorted(glob.glob(f"{RESULTS}/pig_melt_*.nc")):
            replot_production(p)
    if args.only != "production":
        for p in sorted(glob.glob(f"{RESULTS}/pig_fused_melt_map*.nc")):
            replot_fused(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

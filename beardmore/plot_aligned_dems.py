"""Multi-panel figure: every in-window aligned DEM + a MEaSUREs velocity panel.

Pulls every ``*-trans_reference-DEM.tif`` in ``config.STRIP_ALIGNED_DIR``
that intersects the AOI and lies in the configured study window,
crops to the AOI bbox, and plots one DEM per panel on a shared
elevation colormap. Adds a final panel showing MEaSUREs speed with
a quiver overlay so flow direction is visible alongside the DEMs.

Run:

    python -m beardmore.plot_aligned_dems
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# PROJ_DATA fix for this conda env's broken base proj.db.
_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import fiona
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.windows import from_bounds
from shapely.geometry import shape

from beardmore import config


DATE_RE = re.compile(r"_(\d{8})_")


def _aoi_bounds() -> tuple[float, float, float, float]:
    with fiona.open(config.BEARDMORE_AOI_SHP) as src:
        return tuple(src.bounds)


def _strip_date(p: Path) -> pd.Timestamp:
    m = DATE_RE.search(p.name)
    return pd.Timestamp(m.group(1)) if m else pd.NaT


def _list_in_window_aligned() -> list[tuple[Path, pd.Timestamp]]:
    t_start = pd.Timestamp(config.START_TIME)
    t_end = pd.Timestamp(config.END_TIME)
    out = []
    for p in sorted(config.STRIP_ALIGNED_DIR.glob("*-trans_reference-DEM.tif")):
        t = _strip_date(p)
        if pd.isna(t) or not (t_start <= t < t_end):
            continue
        out.append((p, t))
    return sorted(out, key=lambda x: x[1])


def _crop_dem_to_bbox(path: Path, bbox: tuple[float, float, float, float]):
    """Return (z, extent) cropped to bbox, or (None, None) if no overlap."""
    x_min, y_min, x_max, y_max = bbox
    with rasterio.open(path) as src:
        bb = src.bounds
        if bb.right < x_min or bb.left > x_max or bb.top < y_min or bb.bottom > y_max:
            return None, None
        # Clamp the requested window to the strip's actual extent.
        wx_min = max(x_min, bb.left)
        wx_max = min(x_max, bb.right)
        wy_min = max(y_min, bb.bottom)
        wy_max = min(y_max, bb.top)
        win = from_bounds(wx_min, wy_min, wx_max, wy_max, transform=src.transform)
        z = src.read(1, window=win, masked=True)
        # Coarsen to keep memory and plotting time bounded (~1500 px max).
        ny, nx = z.shape
        target = 800
        sy = max(1, ny // target)
        sx = max(1, nx // target)
        z = z[::sy, ::sx]
        nodata = src.nodata
    arr = np.asarray(z.filled(np.nan), dtype=np.float64)
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, (wx_min, wx_max, wy_min, wy_max)


def _load_velocity_for_aoi(bbox: tuple[float, float, float, float]):
    """Return (speed, vx, vy, extent) cropped to the AOI bbox from MEaSUREs."""
    if not config.MEASURES_PHASE_NC.exists():
        raise SystemExit(f"MEaSUREs file not found at {config.MEASURES_PHASE_NC}")
    ds = xr.open_dataset(config.MEASURES_PHASE_NC)
    x_min, y_min, x_max, y_max = bbox
    pad = 5000.0
    if ds["y"].values[0] > ds["y"].values[-1]:
        sub = ds.sel(
            x=slice(x_min - pad, x_max + pad),
            y=slice(y_max + pad, y_min - pad),
        )
    else:
        sub = ds.sel(
            x=slice(x_min - pad, x_max + pad),
            y=slice(y_min - pad, y_max + pad),
        )
    sub = sub[["VX", "VY"]].load()
    vx = sub["VX"].values
    vy = sub["VY"].values
    speed = np.hypot(vx, vy)
    extent = (
        float(sub["x"].min()),
        float(sub["x"].max()),
        float(sub["y"].min()),
        float(sub["y"].max()),
    )
    return speed, vx, vy, sub["x"].values, sub["y"].values, extent


def main() -> None:
    config.ensure_output_dirs()
    bbox = _aoi_bounds()
    print(f"AOI bbox EPSG:3031: {bbox}")

    paths_times = _list_in_window_aligned()
    print(
        f"Found {len(paths_times)} aligned DEMs in [{config.START_TIME}, {config.END_TIME})"
    )
    if not paths_times:
        raise SystemExit("No aligned DEMs in window — nothing to plot.")

    # Crop each DEM to AOI; drop strips that don't overlap.
    cropped = []
    for p, t in paths_times:
        arr, extent = _crop_dem_to_bbox(p, bbox)
        if arr is None or not np.isfinite(arr).any():
            print(f"  skip {t.date()} {p.stem} (no AOI overlap or all-NaN)")
            continue
        cropped.append((p, t, arr, extent))
    print(f"  {len(cropped)} DEMs overlap AOI")

    # Robust color limits across all cropped DEMs (skipping the saturated tails).
    all_finite = np.concatenate([c[2][np.isfinite(c[2])].ravel() for c in cropped])
    vmin, vmax = float(np.percentile(all_finite, 2)), float(np.percentile(all_finite, 98))
    print(f"  shared elevation cmap: [{vmin:.1f}, {vmax:.1f}] m")

    # Velocity background for the last panel.
    speed, vx, vy, vx_x, vy_y, vel_extent = _load_velocity_for_aoi(bbox)
    print(f"  velocity tile: {speed.shape}")

    n = len(cropped)
    n_panels = n + 1
    ncols = 4
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows), constrained_layout=True
    )
    axes = np.atleast_2d(axes)

    for i, (_p, t, arr, extent) in enumerate(cropped):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        im = ax.imshow(
            arr, extent=extent, origin="upper", cmap="terrain",
            vmin=vmin, vmax=vmax, aspect="equal",
        )
        ax.set_title(f"{t.date()}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        # Match every DEM panel to the AOI bbox so footprints are comparable.
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])

    # One shared colorbar for the elevation panels along the right.
    cbar_ax = fig.add_axes([1.005, 0.5, 0.012, 0.4])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("elevation (m)")

    # Velocity panel.
    r, c = divmod(n, ncols)
    ax_v = axes[r, c]
    im_v = ax_v.imshow(
        speed, extent=vel_extent, origin="upper", cmap="viridis",
        vmin=0, vmax=float(np.nanpercentile(speed, 98)), aspect="equal",
    )
    # Subsampled quiver overlay so direction is visible without clutter.
    n_q = 14
    sx = max(1, vx.shape[1] // n_q)
    sy = max(1, vx.shape[0] // n_q)
    XX, YY = np.meshgrid(vx_x[::sx], vy_y[::sy])
    ax_v.quiver(
        XX, YY, vx[::sy, ::sx], vy[::sy, ::sx],
        color="white", scale=600, width=0.0035, alpha=0.85,
    )
    ax_v.set_title("MEaSUREs |v|, m/yr (white quivers)", fontsize=10)
    ax_v.set_xticks([])
    ax_v.set_yticks([])
    ax_v.set_xlim(bbox[0], bbox[2])
    ax_v.set_ylim(bbox[1], bbox[3])
    fig.colorbar(im_v, ax=ax_v, fraction=0.045)

    # Blank any trailing axes.
    for k in range(n_panels, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")

    fig.suptitle(
        f"Beardmore aligned DEMs + velocity — "
        f"{config.START_TIME} to {config.END_TIME} "
        f"({len(cropped)} strips)",
        fontsize=12,
    )

    out_path = config.FIGURES_DIR / "aligned_dems_with_velocity.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

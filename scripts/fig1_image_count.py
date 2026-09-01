"""Per-basin REMA strip-count + LS static-control mask (Shean 2019 Fig 1 analog).

One figure per basin (preserves each basin's native aspect ratio).
Left panel: per-pixel epoch count from the tilt-corrected stack.
Right panel: LS static-control mask used by ``fit_tilt_stack`` — white where
included (BedMachine rock + grounded ice with 2 km erosion from the
non-grounded boundary), dark gray where excluded.
"""
from __future__ import annotations

from pathlib import Path

import fiona
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import ListedColormap, Normalize
from shapely.geometry import shape

from stereo_melt.coregister.tilt import build_static_area_polygon_mask

BEDMACHINE_NC = Path("/wd2/projects/stereo_melt/data/bedmachine/BedMachineAntarctica-v3.nc")

ROOT = Path("/wd2/projects/stereo_melt")
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"

BASINS = [
    {
        "name": "Beardmore",
        "stack": EXAMPLES / "beardmore/processed/beardmore_stack_tilt_corrected_2019-01-01_2023-03-01.nc",
        "aoi": ROOT / "data/shapefiles/beardmore.shp",
    },
    {
        "name": "Nansen",
        "stack": EXAMPLES / "nansen/processed/nansen_stack_tilt_corrected_2019-01-01_2023-03-01.nc",
        "aoi": ROOT / "data/shapefiles/nansen.shp",
    },
    {
        "name": "PIG",
        "stack": EXAMPLES / "pig/processed/pig_stack_tilt_corrected_2019-01-01_2024-01-10.nc",
        "aoi": ROOT / "data/shapefiles/pig.shp",
    },
]


def aoi_polys(path: Path):
    with fiona.open(path) as src:
        for feat in src:
            g = shape(feat["geometry"])
            yield from (list(g.geoms) if g.geom_type == "MultiPolygon" else [g])


def load_stack_with_count(stack_path: Path) -> tuple[xr.DataArray, xr.DataArray, int]:
    ds = xr.open_dataset(stack_path)
    da = ds[list(ds.data_vars)[0]]
    n_epochs = int(da.sizes["time"])
    count = da.notnull().sum(dim="time").compute()
    return da, count, n_epochs


def draw_aoi(ax, aoi_path: Path, color: str = "red") -> None:
    if not aoi_path.exists():
        return
    for poly in aoi_polys(aoi_path):
        xs, ys = poly.exterior.xy
        ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.9)


def render_basin(b: dict) -> Path | None:
    if not b["stack"].exists():
        print(f"{b['name']}: no stack yet, skipping")
        return None

    stack, count, n_epochs = load_stack_with_count(b["stack"])
    extent = [
        float(count.x.min()), float(count.x.max()),
        float(count.y.min()), float(count.y.max()),
    ]
    width_m = extent[1] - extent[0]
    height_m = extent[3] - extent[2]
    aspect = width_m / height_m  # for figure sizing

    panel_h = 8.0
    panel_w = max(3.5, panel_h * aspect)
    fig, (ax_count, ax_mask) = plt.subplots(1, 2, figsize=(2 * panel_w + 2.5, panel_h + 1.0))

    cmax = max(1, int(count.max().item()))
    masked = count.where(count > 0)
    im = ax_count.imshow(
        masked.values, extent=extent, origin="upper",
        cmap="viridis", norm=Normalize(vmin=0, vmax=cmax), interpolation="nearest",
    )
    draw_aoi(ax_count, b["aoi"])
    ax_count.set_title(f"{b['name']}: {n_epochs} epochs, max count = {cmax}")
    ax_count.set_xlabel("Easting (m, EPSG:3031)")
    ax_count.set_ylabel("Northing (m, EPSG:3031)")
    ax_count.set_aspect("equal")
    cbar = plt.colorbar(im, ax=ax_count, fraction=0.045, pad=0.04)
    cbar.set_label("strip count per pixel")

    ls_mask = build_static_area_polygon_mask(stack, BEDMACHINE_NC)
    ls = ls_mask.values.astype(np.uint8)
    cmap = ListedColormap(["#333333", "#ffffff"])  # excluded, included
    ax_mask.imshow(ls, extent=extent, origin="upper", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    draw_aoi(ax_mask, b["aoi"])
    kept_pct = 100 * float(ls.mean())
    ax_mask.set_title(f"{b['name']} LS mask: {kept_pct:.1f}% included (white)")
    ax_mask.set_xlabel("Easting (m, EPSG:3031)")
    ax_mask.set_ylabel("Northing (m, EPSG:3031)")
    ax_mask.set_aspect("equal")

    stack.close()

    fig.suptitle(
        f"{b['name']}: REMA strip coverage + LS static-control mask",
        fontsize=14, y=1.00,
    )
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"fig1_count_lsmask_{b['name'].lower()}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def main() -> None:
    for b in BASINS:
        render_basin(b)


if __name__ == "__main__":
    main()

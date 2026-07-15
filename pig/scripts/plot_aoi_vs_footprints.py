"""Debug figure: PIG AOI polygon + bbox vs every candidate -trans DEM footprint.

Re-runs the same discovery as ``pig.build_stack`` (date window + variant
precedence) but skips the AOI filter, so we can see *spatially* why the
filter rejects all 34 candidates.

Saves to ``pig/figures/aoi_vs_strip_footprints.png``.
"""
from pathlib import Path

import fiona
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import shape

from pig import config
from pig.build_stack import _aoi_bounds_3031, _stack_aoi_path, list_aligned_dems


def main() -> None:
    candidates = list_aligned_dems()
    print(f"{len(candidates)} candidates pre-AOI-filter")

    aoi_path = _stack_aoi_path()
    a_xmin, a_ymin, a_xmax, a_ymax = _aoi_bounds_3031(aoi_path)
    print(f"AOI bbox (EPSG:3031): x=[{a_xmin:.0f}, {a_xmax:.0f}]  y=[{a_ymin:.0f}, {a_ymax:.0f}]")
    print(f"  width  = {(a_xmax - a_xmin)/1e3:.1f} km")
    print(f"  height = {(a_ymax - a_ymin)/1e3:.1f} km")

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # AOI polygon outline
    aoi_shapes = []
    with fiona.open(aoi_path) as src:
        for feat in src:
            aoi_shapes.append(shape(feat["geometry"]))
    for geom in aoi_shapes:
        if geom.geom_type == "Polygon":
            xs, ys = geom.exterior.xy
            ax.plot(xs, ys, color="black", linewidth=2, label="AOI polygon")
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                xs, ys = poly.exterior.xy
                ax.plot(xs, ys, color="black", linewidth=2)

    # AOI bbox (what the filter actually uses)
    bbox_rect = plt.Rectangle(
        (a_xmin, a_ymin),
        a_xmax - a_xmin,
        a_ymax - a_ymin,
        fill=False, edgecolor="red", linewidth=2, linestyle="--",
        label="AOI bbox (used by filter)",
    )
    ax.add_patch(bbox_rect)

    # DEM footprints, colored by date
    times = [t for _, t, _ in candidates]
    t_num = np.array([t.value for t in times], dtype=float)
    t_norm = (t_num - t_num.min()) / max(t_num.max() - t_num.min(), 1.0)
    cmap = plt.get_cmap("viridis")

    overlap_count = 0
    for (p, t, _v), tn in zip(candidates, t_norm):
        with rasterio.open(p) as src:
            bx_min, by_min, bx_max, by_max = src.bounds
        overlaps = not (
            bx_max < a_xmin or bx_min > a_xmax or by_max < a_ymin or by_min > a_ymax
        )
        if overlaps:
            overlap_count += 1
        rect = plt.Rectangle(
            (bx_min, by_min),
            bx_max - bx_min,
            by_max - by_min,
            fill=False,
            edgecolor=cmap(tn),
            linewidth=1.0,
            alpha=0.8,
        )
        ax.add_patch(rect)

    print(f"recomputed overlap count: {overlap_count}/{len(candidates)}")

    # Auto-scale axes to include both AOI and all footprints
    all_x = [a_xmin, a_xmax]
    all_y = [a_ymin, a_ymax]
    for p, _, _ in candidates:
        with rasterio.open(p) as src:
            bx_min, by_min, bx_max, by_max = src.bounds
        all_x += [bx_min, bx_max]
        all_y += [by_min, by_max]
    pad = 5e3
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

    ax.set_xlabel("x (EPSG:3031, m)")
    ax.set_ylabel("y (EPSG:3031, m)")
    ax.set_title(
        f"PIG AOI vs aligned-strip footprints\n"
        f"{len(candidates)} candidates, {min(times).date()} → {max(times).date()}\n"
        f"bbox-overlap count: {overlap_count}/{len(candidates)}"
    )
    ax.set_aspect("equal")
    ax.legend(loc="upper left")

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=t_num.min(), vmax=t_num.max()),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.04)
    cbar_ticks = np.linspace(t_num.min(), t_num.max(), 5)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([pd.Timestamp(t).date().isoformat() for t in cbar_ticks])
    cbar.set_label("acquisition date")

    out = config.FIGURES_DIR / "aoi_vs_strip_footprints.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

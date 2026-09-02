"""Continental-context plot: all 3 basin AOIs + the 87 migrated PIG strips.

Sanity check: do the strips we migrated to pig/data/ASP/asp_aligned/
actually sit at Pine Island Glacier (Amundsen Sea), or somewhere else
like Nansen (Ross Sea) or Beardmore (TAM)?
"""
import os
from glob import glob
from pathlib import Path

import fiona
import matplotlib.pyplot as plt
import rasterio
from shapely.geometry import shape


_REPO = __import__("pathlib").Path(__file__).resolve().parents[3]
def main() -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    aoi_specs = [
        ("/wd2/projects/stereo_melt/data/shapefiles/pig.shp",      "PIG (Pine Island, Amundsen)", "tab:blue"),
        ("/wd2/projects/stereo_melt/data/shapefiles/nansen.shp",   "Nansen (Ross)",               "tab:green"),
        ("/wd2/projects/stereo_melt/data/shapefiles/beardmore.shp","Beardmore (TAM)",             "tab:orange"),
    ]
    for path, label, color in aoi_specs:
        if not os.path.exists(path):
            continue
        with fiona.open(path) as src:
            for feat in src:
                g = shape(feat["geometry"])
                if g.geom_type == "Polygon":
                    polys = [g]
                else:
                    polys = list(g.geoms)
                for poly in polys:
                    xs, ys = poly.exterior.xy
                    ax.fill(xs, ys, color=color, alpha=0.5, edgecolor=color, linewidth=2)
            ax.plot([], [], color=color, linewidth=3, label=f"AOI: {label}")

    aligned_dir = f"{_REPO}/examples/pig/data/ASP/asp_aligned"
    strips = sorted(glob(f"{aligned_dir}/*-trans_reference-DEM.tif"))
    if not strips:
        raise SystemExit(
            f"no aligned strips under {aligned_dir}\n"
            f"      this plot answers 'do the migrated strips sit at PIG?', so an\n"
            f"      empty glob would draw a blank map, not a negative result.")
    for p in strips:
        with rasterio.open(p) as src:
            b = src.bounds
        rect = plt.Rectangle(
            (b.left, b.bottom), b.right - b.left, b.top - b.bottom,
            fill=False, edgecolor="red", linewidth=0.6, alpha=0.7,
        )
        ax.add_patch(rect)
    ax.plot([], [], color="red", linewidth=1.2, label=f"migrated PIG strips (n={len(strips)})")

    # Antarctic coastline rough box for context
    ax.set_xlim(-3.0e6, +3.0e6)
    ax.set_ylim(-3.0e6, +3.0e6)
    ax.set_aspect("equal")
    ax.set_xlabel("x (EPSG:3031, m)")
    ax.set_ylabel("y (EPSG:3031, m)")
    ax.set_title(
        "Antarctic context: 3 basin AOIs + migrated PIG strip footprints\n"
        "(Should clearly cluster inside the blue PIG AOI, far from green Nansen)"
    )
    ax.legend(loc="upper right")
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.3)

    out = Path(f"{_REPO}/examples/pig/figures/strips_on_continent.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

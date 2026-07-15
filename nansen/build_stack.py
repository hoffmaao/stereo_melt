"""Build the Nansen repeat-DEM stack on a common EPSG:3031 grid.

Stage 3 of the Nansen pipeline. Consumes ASP-aligned DEM GeoTIFFs
that intersect the AOI and lie in the configured time window, reprojects
each onto a shared 25 m target grid, concatenates them along a ``time``
dimension, and writes the result to a NetCDF for the downstream tilt /
Lagrangian stages.

Run directly:

    python -m nansen.build_stack
"""

from __future__ import annotations

import re
from pathlib import Path

import fiona
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from stereo_melt.stack import build_stack, save_stack

from nansen import config

# EPSG:3031 as a self-contained PROJ4 string. Avoids pyproj hitting the
# broken proj.db in this conda env (which breaks EPSG code lookups).
EPSG_3031_PROJ4 = (
    "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +x_0=0 +y_0=0 "
    "+datum=WGS84 +units=m +no_defs +type=crs"
)


DATE_RE = re.compile(r"_(\d{8})_")


def _extract_date(path: Path) -> pd.Timestamp:
    """Extract the acquisition date from a REMA/SETSM strip filename."""
    m = DATE_RE.search(path.name)
    if not m:
        raise ValueError(f"Could not extract date from {path.name!r}")
    return pd.Timestamp(m.group(1))


def list_aligned_dems(
    aligned_dir: Path = config.STRIP_ALIGNED_DIR,
    start: str = config.START_TIME,
    end: str = config.END_TIME,
) -> list[tuple[Path, pd.Timestamp]]:
    """Return ``[(path, time)]`` for all ASP-aligned DEMs in the window."""
    t_start = pd.Timestamp(start)
    t_end = pd.Timestamp(end)
    matches = []
    for p in sorted(Path(aligned_dir).glob("*-trans_reference-DEM.tif")):
        t = _extract_date(p)
        if t_start <= t < t_end:
            matches.append((p, t))
    return matches


def _aoi_elevation_envelope(
    aoi_bbox: tuple[float, float, float, float],
    bedmachine_path: "Path | str",
    pad_m: float = 100.0,
) -> tuple[float, float]:
    r"""Min/max BedMachine surface (ellipsoid-referenced) in the AOI bbox,
    padded by ``pad_m``.

    BedMachine v3 ``surface`` is orthometric (referenced to the EGM2008
    geoid), but the DEM strips reach ``_alignment_sanity_check`` *before*
    the post-coreg geoid correction — i.e., still in WGS84 ellipsoid
    coordinates. To compare on a common vertical reference we add the
    BedMachine ``geoid`` band (geoid-above-ellipsoid undulation; typically
    -30 to -60 m in Antarctica) to the orthometric surface so the
    envelope is also ellipsoid-referenced. Without this correction the
    envelope is offset by the geoid magnitude (~55 m at -75°S), which
    the ``pad_m`` slack would otherwise have to absorb.

    Returns ``(z_min - pad_m, z_max + pad_m)`` with both bounds in
    ellipsoid coordinates so a strip's AOI-cropped median can be
    compared directly without applying any correction.
    """
    import xarray as xr
    x_min, y_min, x_max, y_max = aoi_bbox
    ds = xr.open_dataset(bedmachine_path)
    y_first, y_last = float(ds.y.values[0]), float(ds.y.values[-1])
    if y_first > y_last:
        sx = slice(x_min, x_max); sy = slice(y_max, y_min)
    else:
        sx = slice(x_min, x_max); sy = slice(y_min, y_max)
    surf = np.asarray(ds.surface.sel(x=sx, y=sy).values, dtype=np.float64)
    geoid = np.asarray(ds.geoid.sel(x=sx, y=sy).values, dtype=np.float64)
    z_ellipsoid = surf + geoid
    finite = z_ellipsoid[np.isfinite(z_ellipsoid)]
    if finite.size == 0:
        raise RuntimeError(
            f"No finite BedMachine surface+geoid samples inside AOI bbox {aoi_bbox}"
        )
    return float(finite.min()) - float(pad_m), float(finite.max()) + float(pad_m)


def _alignment_sanity_check(
    paths_times: list[tuple[Path, pd.Timestamp]],
    aoi_bbox: tuple[float, float, float, float],
    median_min: float = -100.0,
    median_max: float = 400.0,
    min_finite_cells: int = 100,
    stride: int = 12,
) -> tuple[list[tuple[Path, pd.Timestamp]], list[tuple[Path, pd.Timestamp, float]]]:
    """Drop strips whose AOI-cropped median elevation is unphysical.

    pc_align exits 0 even when its ICP converges to a spurious transform
    (typical when the IS2 control cloud was sparse or in a strip corner).
    A reasonable post-align median is the cleanest discriminator; the
    [-100, +400] m range catches gross failures (km-scale shifts) without
    rejecting legitimate strips. Subsamples with ``stride`` for speed.
    """
    import rasterio
    from rasterio.windows import from_bounds

    kept: list[tuple[Path, pd.Timestamp]] = []
    rejected: list[tuple[Path, pd.Timestamp, float]] = []
    x_min, y_min, x_max, y_max = aoi_bbox

    for p, t in paths_times:
        try:
            with rasterio.open(p) as src:
                bb = src.bounds
                wx_min, wx_max = max(x_min, bb.left), min(x_max, bb.right)
                wy_min, wy_max = max(y_min, bb.bottom), min(y_max, bb.top)
                if wx_max <= wx_min or wy_max <= wy_min:
                    continue
                win = from_bounds(wx_min, wy_min, wx_max, wy_max, transform=src.transform)
                z = src.read(
                    1, window=win, masked=True,
                    out_shape=(
                        max(1, int(win.height // stride)),
                        max(1, int(win.width // stride)),
                    ),
                )
                arr = np.asarray(z.filled(np.nan), dtype=np.float64)
                if src.nodata is not None:
                    arr = np.where(arr == src.nodata, np.nan, arr)
        except Exception as e:
            print(f"  ⚠️ {t.date()} could not read {p.name}: {e}")
            rejected.append((p, t, float("nan")))
            continue

        finite = arr[np.isfinite(arr)]
        if finite.size < min_finite_cells:
            print(
                f"  ⚠️ {t.date()} {p.stem}: only {finite.size} finite cells; rejecting"
            )
            rejected.append((p, t, float("nan")))
            continue

        med = float(np.median(finite))
        if med < median_min or med > median_max:
            print(
                f"  ⚠️ {t.date()} median={med:+.1f} m outside "
                f"[{median_min:+.1f}, {median_max:+.1f}] -- rejecting"
            )
            rejected.append((p, t, med))
        else:
            kept.append((p, t))
    return kept, rejected


def _qc_rejection_csv() -> Path:
    """Path to ``strips_rejected_is2.csv`` for the single Nansen variant.

    Returns the standard QC-list location used by ``nansen.qc_strips``.
    The caller is responsible for tolerating a missing file (that just
    means QC has not been run yet).
    """
    return config.RESULTS_DIR / "strips_rejected_is2.csv"


def _filter_by_qc_rejection(
    paths_times: list[tuple[Path, pd.Timestamp]],
) -> tuple[
    list[tuple[Path, pd.Timestamp]],
    list[tuple[Path, pd.Timestamp, str]],
]:
    """Drop strips listed in ``strips_rejected_is2.csv``.

    Mirror of :func:`beardmore.build_stack._filter_by_qc_rejection`,
    simplified for the single Nansen variant. Missing CSV is treated as
    "QC not run yet" and the input is passed through unchanged.
    """
    csv_path = _qc_rejection_csv()
    if not csv_path.is_file():
        return list(paths_times), []
    rej = pd.read_csv(csv_path)
    if rej.empty or "strip" not in rej.columns:
        return list(paths_times), []

    rej_strips = set(rej["strip"].astype(str).unique())
    reason_for: dict[str, str] = {}
    for s, grp in rej.groupby("strip"):
        reason_for[str(s)] = ", ".join(grp["reason"].astype(str).unique())

    kept: list[tuple[Path, pd.Timestamp]] = []
    rejected: list[tuple[Path, pd.Timestamp, str]] = []
    for p, t in paths_times:
        stem = p.name.replace("-trans_reference-DEM.tif", "")
        if stem in rej_strips:
            rejected.append((p, t, reason_for.get(stem, "qc_listed")))
        else:
            kept.append((p, t))
    return kept, rejected


def _stack_aoi_path() -> Path:
    """Return the canonical (wider) stack/tilt AOI."""
    return Path(config.NANSEN_AOI_SHP)


def _aoi_bounds_3031(aoi_path: Path | None = None) -> tuple[float, float, float, float]:
    """Return the AOI bounding box in EPSG:3031 (defaults to stack-extent AOI).

    Uses ``fiona`` directly rather than ``geopandas.read_file`` to avoid
    invoking ``pyproj.CRS.from_user_input``, which is broken in the
    current conda env (proj.db metadata mismatch).
    """
    if aoi_path is None:
        aoi_path = _stack_aoi_path()
    with fiona.open(aoi_path) as src:
        return tuple(src.bounds)  # (min_x, min_y, max_x, max_y), already EPSG:3031


def _filter_by_aoi_intersection(
    dem_paths: list[tuple[Path, pd.Timestamp]],
    aoi_path: Path | None = None,
) -> list[tuple[Path, pd.Timestamp]]:
    """Keep only DEMs whose footprint intersects the AOI bounding box.

    Bbox-only intersection in EPSG:3031 — both DEMs and AOI are already
    in Antarctic Polar Stereographic so no reprojection is needed.
    """
    import rasterio

    a_xmin, a_ymin, a_xmax, a_ymax = _aoi_bounds_3031(aoi_path)
    kept = []
    for p, t in dem_paths:
        with rasterio.open(p) as src:
            bx_min, by_min, bx_max, by_max = src.bounds
        overlaps = not (
            bx_max < a_xmin or bx_min > a_xmax or by_max < a_ymin or by_min > a_ymax
        )
        if overlaps:
            kept.append((p, t))
    return kept


def plot_stack_footprints(
    paths_times: list[tuple[Path, pd.Timestamp]],
    aoi_path: Path,
    output_path: Path,
) -> None:
    """QC plot: AOI polygon + footprint of every DEM, colored by date."""
    import rasterio
    from shapely.geometry import shape

    # Read AOI geometry directly via fiona (avoids pyproj CRS construction).
    aoi_shapes = []
    with fiona.open(aoi_path) as src:
        for feat in src:
            aoi_shapes.append(shape(feat["geometry"]))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for geom in aoi_shapes:
        if geom.geom_type == "Polygon":
            xs, ys = geom.exterior.xy
            ax.plot(xs, ys, color="black", linewidth=2, label="AOI")
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                xs, ys = poly.exterior.xy
                ax.plot(xs, ys, color="black", linewidth=2)

    times = [t for _, t in paths_times]
    t_num = np.array([t.value for t in times], dtype=float)
    t_norm = (t_num - t_num.min()) / max(t_num.max() - t_num.min(), 1.0)
    cmap = plt.get_cmap("viridis")

    for (p, t), tn in zip(paths_times, t_norm):
        with rasterio.open(p) as src:
            bx_min, by_min, bx_max, by_max = src.bounds
        rect = plt.Rectangle(
            (bx_min, by_min),
            bx_max - bx_min,
            by_max - by_min,
            fill=False,
            edgecolor=cmap(tn),
            linewidth=1.2,
            alpha=0.8,
        )
        ax.add_patch(rect)

    ax.set_xlabel("x (EPSG:3031, m)")
    ax.set_ylabel("y (EPSG:3031, m)")
    ax.set_title(
        f"Nansen stack footprints\n"
        f"{len(paths_times)} DEMs, {times[0].date()} to {times[-1].date()}"
    )

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=t_num.min(), vmax=t_num.max()),
    )
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("acquisition time")
    cbar.set_ticks([t_num.min(), t_num.mean(), t_num.max()])
    cbar.set_ticklabels(
        [
            str(pd.Timestamp(t_num.min()).date()),
            str(pd.Timestamp(t_num.mean()).date()),
            str(pd.Timestamp(t_num.max()).date()),
        ]
    )

    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_stack_coverage(stack: xr.DataArray, output_path: Path) -> None:
    """QC plot: per-epoch valid-pixel map + time-mean surface."""
    n = stack.sizes["time"]
    ncols = 4
    nrows = (n + ncols - 1) // ncols + 1  # extra row for the time-mean

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)

    # Time-mean on the top row (spanning all cols via imshow on first col, others blank)
    mean_arr = stack.mean("time", skipna=True).values
    im0 = axes[0, 0].imshow(mean_arr, cmap="terrain", origin="upper")
    axes[0, 0].set_title("time-mean surface (m)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.045)
    for j in range(1, ncols):
        axes[0, j].axis("off")

    # Per-epoch slices on subsequent rows
    for i in range(n):
        r = 1 + i // ncols
        c = i % ncols
        ax = axes[r, c]
        z = stack.isel(time=i).values
        ax.imshow(z, cmap="terrain", origin="upper")
        t = pd.Timestamp(stack["time"].values[i]).date()
        ax.set_title(f"{t}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    # Blank any trailing axes
    for k in range(n, (nrows - 1) * ncols):
        r = 1 + k // ncols
        c = k % ncols
        axes[r, c].axis("off")

    fig.suptitle(
        f"Nansen stack — {n} epochs on {stack.sizes['y']} x {stack.sizes['x']} grid",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(res_override: float | None = None) -> xr.DataArray:
    config.ensure_output_dirs()

    print("Listing ASP-aligned DEMs in window...")
    candidates = list_aligned_dems()
    print(f"  found {len(candidates)} in [{config.START_TIME}, {config.END_TIME})")
    for p, t in candidates:
        print(f"    {t.date()}  {p.name}")

    print("Filtering by AOI intersection...")
    kept = _filter_by_aoi_intersection(candidates)
    print(f"  kept {len(kept)} after AOI filter")

    if not kept:
        raise SystemExit("No DEMs intersect the AOI — nothing to stack.")

    aoi_bb = _aoi_bounds_3031()
    # Alignment sanity check with envelope tied to BedMachine surface in
    # the AOI plus 100 m of slack on each side. Restoring the filter we
    # had disabled when the wider Nansen AOI started sampling mountain
    # rock outcrops, which the old hard-coded [-100, +400] envelope
    # rejected wholesale. The dynamic envelope still catches pc_align
    # catastrophic shifts (those move medians by 1-10 km, way outside any
    # plausible AOI elevation envelope).
    zmin, zmax = _aoi_elevation_envelope(aoi_bb, config.BEDMACHINE_NC, pad_m=100.0)
    print(f"AOI elevation envelope (BedMachine surface ± 100 m): "
          f"[{zmin:+.0f}, {zmax:+.0f}] m")
    print("Sanity-checking alignment medians (drops pc_align spurious transforms)...")
    kept, rejected = _alignment_sanity_check(
        kept, aoi_bb, median_min=zmin, median_max=zmax,
    )
    print(f"  kept {len(kept)} after sanity filter (rejected {len(rejected)})")
    for p, t, med in rejected:
        print(f"    sanity-rejected: {t.date()}  median={med:+.1f} m  {p.name}")

    print("Applying QC rejection list (post-pc_align residual thresholds)...")
    kept, qc_rejected = _filter_by_qc_rejection(kept)
    print(
        f"  kept {len(kept)} after QC filter (rejected {len(qc_rejected)})"
    )
    for p, t, reason in qc_rejected:
        print(f"    qc-rejected: {t.date()}  [{reason}]  {p.name}")
    if not kept:
        raise SystemExit("All DEMs rejected by QC -- aborting.")

    paths = [p for p, _ in kept]
    times = [t for _, t in kept]

    # Look up PGC quality companion files (matchtag + bitmask) by stripping
    # the ASP -trans_reference-DEM suffix from each aligned-DEM filename.
    # Companion files live alongside the original (un-aligned) strip in
    # config.STRIPS_DIR. Missing companions are silently skipped.
    def _quality_for(p: Path) -> tuple[Path, Path] | None:
        dem_id = p.name.replace("-trans_reference-DEM.tif", "")
        mt = config.STRIPS_DIR / f"{dem_id}_matchtag.tif"
        bm = config.STRIPS_DIR / f"{dem_id}_bitmask.tif"
        if mt.exists() and bm.exists():
            return (mt, bm)
        if mt.exists() and not bm.exists():
            return (mt, None)
        if bm.exists() and not mt.exists():
            return (None, bm)
        return None

    quality_files = [_quality_for(p) for p in paths]
    n_with_q = sum(1 for q in quality_files if q is not None)
    print(f"  PGC quality companions: {n_with_q}/{len(paths)} strips have matchtag+bitmask")

    x_min, y_min, x_max, y_max = _aoi_bounds_3031()
    # Round AOI bounds out to the target-grid multiple so the grid sits on
    # clean res-m lines. `res_override` lets us spin up a coarser pilot
    # stack (e.g. 250 m) without touching config.RES.
    res = float(res_override) if res_override is not None else float(config.RES)
    x_min = res * np.floor(x_min / res)
    x_max = res * np.ceil(x_max / res)
    y_min = res * np.floor(y_min / res)
    y_max = res * np.ceil(y_max / res)
    print(f"Target grid: x=[{x_min}, {x_max}]  y=[{y_min}, {y_max}]  res={res} m")

    print("Building stack...")
    stack = build_stack(
        paths=paths,
        times=times,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        res=res,
        crs=EPSG_3031_PROJ4,
        src_crs_override=EPSG_3031_PROJ4,  # override LOCAL_CS from pc_align outputs
        quality_files=quality_files,
    )
    print(f"  stack dims: time={stack.sizes['time']}, y={stack.sizes['y']}, x={stack.sizes['x']}")
    print(f"  surface range: {float(stack.min()):.1f}  to  {float(stack.max()):.1f} m")

    # Per-slice DEM identity (SETSM strip id). build_stack internally
    # sortby("time"), so mirror that ordering on the dem_id list (nansen has no
    # source_variant block to borrow the order from). Keystone for *strip-level*
    # bad-epoch dropping: REMA filenames carry no time-of-day, so _extract_date
    # collapses many same-day strips to one timestamp; a date-keyed drop then
    # discards clean same-day siblings along with the one bad strip. dem_id
    # restores Shean 2019's per-DEM rejection granularity. Consumed by
    # load_basin_stack(bad_strips=...).
    order = np.argsort(pd.to_datetime(times).values, kind="stable")
    dem_ids = np.asarray(
        [p.name.replace("-trans_reference-DEM.tif", "") for p in paths]
    )[order]
    stack = stack.assign_coords(dem_id=("time", dem_ids))

    # Suffix the filename only when overriding the default RES, so production
    # 25 m pipelines stay on the canonical path.
    res_suffix = f"_{int(round(res))}m" if res_override is not None else ""
    out_nc = config.PROCESSED_DIR / (
        f"nansen_stack{res_suffix}_{config.START_TIME}_{config.END_TIME}.nc"
    )
    print(f"Saving stack -> {out_nc}")
    save_stack(stack, out_nc)

    print("Writing QC figures to", config.FIGURES_DIR)
    plot_stack_footprints(kept, config.NANSEN_AOI_SHP, config.FIGURES_DIR / "stack_footprints.png")
    plot_stack_coverage(stack, config.FIGURES_DIR / "stack_coverage.png")

    return stack


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--res",
        type=float,
        default=None,
        help="Target-grid resolution override (meters). Defaults to config.RES (25 m).",
    )
    args = parser.parse_args()
    main(res_override=args.res)

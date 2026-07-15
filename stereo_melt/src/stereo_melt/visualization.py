from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import xarray as xr


def plot_dem_stack_with_velocity(
    stack: xr.DataArray,
    vx: xr.DataArray,
    vy: xr.DataArray,
    output_path: Path,
    title: str = "DEM stack with velocity",
    arrow_step_m: float = 1500.0,
    ncols: int = 4,
    bad_epochs: list | tuple | None = None,
) -> None:
    """Per-epoch panel grid: DEM elevation field with velocity arrows on top.

    Parameters
    ----------
    stack
        DEM stack ``(time, y, x)`` on EPSG:3031 in metres of geoid-referenced
        surface elevation.
    vx, vy
        Velocity components on the same ``(y, x)`` grid as ``stack``, m/yr.
    output_path
        PNG path to write.
    arrow_step_m
        Spacing in metres between quiver arrows. Decimated from the stack grid.
    ncols
        Panel grid columns; rows derive from ``stack.sizes['time']``.
    """
    n = stack.sizes["time"]
    nrows = (n + ncols - 1) // ncols

    dx = abs(float(stack["x"].values[1] - stack["x"].values[0]))
    dy = abs(float(stack["y"].values[1] - stack["y"].values[0]))
    step_x = max(1, int(round(arrow_step_m / dx)))
    step_y = max(1, int(round(arrow_step_m / dy)))

    finite = stack.values[np.isfinite(stack.values)]
    if finite.size:
        vmin = float(np.percentile(finite, 2))
        vmax = float(np.percentile(finite, 98))
    else:
        vmin, vmax = 0.0, 1.0

    speed = np.hypot(vx.values, vy.values)
    smax_arr = speed[np.isfinite(speed)]
    smax = float(np.percentile(smax_arr, 99)) if smax_arr.size else 1.0

    extent = [
        float(stack["x"].min()), float(stack["x"].max()),
        float(stack["y"].min()), float(stack["y"].max()),
    ]

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.4 * ncols, 3.4 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    xs = stack["x"].values[::step_x]
    ys = stack["y"].values[::step_y]
    Xa, Ya = np.meshgrid(xs, ys)
    vxa = vx.values[::step_y, ::step_x]
    vya = vy.values[::step_y, ::step_x]

    bad_dates = (
        set(pd.to_datetime(list(bad_epochs)).normalize())
        if bad_epochs else set()
    )

    last_im = None
    for i in range(n):
        r, c = i // ncols, i % ncols
        ax = axes[r, c]
        z = stack.isel(time=i).values
        last_im = ax.imshow(
            z, extent=extent, origin="upper",
            cmap="terrain", vmin=vmin, vmax=vmax, aspect="equal",
        )
        ax.quiver(
            Xa, Ya, vxa, vya,
            color="k", scale=smax * 18.0, width=0.004,
            headwidth=3.0, alpha=0.8,
        )
        ts = pd.Timestamp(stack["time"].values[i])
        t = ts.date()
        n_finite = int(np.isfinite(z).sum())
        is_bad = ts.normalize() in bad_dates
        title_color = "tab:red" if is_bad else "black"
        prefix = "DROPPED  " if is_bad else ""
        ax.set_title(f"{prefix}{t}  ({n_finite} px)", fontsize=9, color=title_color)
        ax.set_xticks([])
        ax.set_yticks([])
        if is_bad:
            for spine in ax.spines.values():
                spine.set_edgecolor("tab:red")
                spine.set_linewidth(3.0)

    for k in range(n, nrows * ncols):
        r, c = k // ncols, k % ncols
        axes[r, c].axis("off")

    if last_im is not None:
        cbar = fig.colorbar(
            last_im, ax=axes, fraction=0.025, pad=0.02, shrink=0.85,
        )
        cbar.set_label("surface elevation (m)")
    fig.suptitle(
        f"{title}  —  velocity arrows decimated to {arrow_step_m:.0f} m, "
        f"|v|₉₉ ≈ {smax:.0f} m/yr",
        fontsize=12,
    )
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_velocity_data(velocity_stack, output_dir="plots"):
    """
    Plot velocity data for each year in the stack.

    Parameters:
    - velocity_stack (dict): Velocity data from `create_velocity_stack`.
    - x, y: Grid coordinates.
    - output_dir (str): Directory to save velocity plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    years = velocity_stack["Years"]

    for i, year in enumerate(years):
        vx = velocity_stack["vx"][:, :, i]
        vy = velocity_stack["vy"][:, :, i]
        speed = np.sqrt(vx**2 + vy**2)

        plt.figure(figsize=(10, 8))
        plt.contourf(velocity_stack["x"], velocity_stack["y"], speed, cmap="viridis", levels=50)
        plt.colorbar(label="Velocity (m/a)")
        plt.title(f"Velocity Data for {year}")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.savefig(os.path.join(output_dir, f"velocity_{year}.png"))
        plt.close()


def plot_tidal_corrections(output_dir, plot_dir="plots"):
    """
    Plot tidal corrections for each processed strip.

    Parameters:
    - output_dir (str): Directory containing processed strips.
    - plot_dir (str): Directory to save tidal correction plots.
    """
    os.makedirs(plot_dir, exist_ok=True)
    files = [f for f in os.listdir(output_dir) if f.endswith(".npz")]

    for file in files:
        file_path = os.path.join(output_dir, file)
        data = np.load(file_path, allow_pickle=True)

        if "tide_correction" in data:
            x = data["x"]
            y = data["y"]
            tide_correction = data["tide_correction"]

            plt.figure(figsize=(12, 9))
            contour = plt.contourf(x, y, tide_correction, cmap="coolwarm", levels=50)
            plt.colorbar(contour, label="Tidal Correction (m)")
            plt.title(f"Tidal Correction for {file}")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.savefig(
                os.path.join(plot_dir, f"tide_correction_{os.path.splitext(file)[0]}.png"), dpi=300
            )
            plt.close()

            print(f"[INFO] Tidal correction plot saved for {file}")
        else:
            print(f"[WARNING] No tidal correction data in {file}")


def plot_correction_data(grid_x, grid_y, geoid, elevation, mdt, firn, output_dir="plots"):
    """
    Plot correction data (elevation, geoid, MDT, firn).

    Parameters:
    - grid_x, grid_y: Grid coordinates.
    - geoid: Geoid corrections.
    - elevation: Surface elevation data.
    - mdt: MDT corrections.
    - firn: Firn corrections.
    - output_dir (str): Directory to save correction plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    datasets = {
        "Geoid Correction": geoid,
        "Elevation": elevation,
        "MDT Correction": mdt,
        "Firn Correction": firn,
    }

    for name, data in datasets.items():
        plt.figure(figsize=(10, 8))
        plt.contourf(grid_x, grid_y, data, cmap="viridis", levels=50)
        plt.colorbar(label=name)
        plt.title(name)
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.savefig(os.path.join(output_dir, f"{name.replace(' ', '_').lower()}.png"))
        plt.close()


def plot_alignment_diagnostics(
    variants,
    aoi_shp,
    output_path,
    title,
    ylim=(0.05, 60.0),
    dpi=130,
):
    """Shean 2019 Fig 4 analog: pc_align ICP translation + pre/post residual.

    The uniform alignment-diagnostic figure deployed across every basin (thin
    ``<basin>/plot_fig4_alignment.py`` drivers call this). Three rows:

      * row 1 — ICP translation scatters (Δy–Δx, Δz–Δx, Δz–Δy) of the pc_align
        transform, parsed from each strip's ``*-log-pc_align-*.txt``.
      * row 2 — PRE-alignment residual (``*-beg_errors.csv``) vs date, p50 with
        16–84 % bars, per variant, dashed population medians.
      * row 3 — POST-alignment residual (``*-end_errors.csv``), same layout,
        legend carries the median improvement factor.

    Rows 2–3 share fixed log-y ``ylim`` so figures from different basins are
    directly comparable (the whole point of a uniform plot).

    Convention: x = East, y = North, z = Up (= −Down).

    Parameters
    ----------
    variants : list of tuple
        ``(aligned_dir, label)`` or ``(aligned_dir, label, colour)``. Each
        ``aligned_dir`` is a pc_align output dir holding
        ``*-trans_reference-DEM.tif`` + per-strip ``*-beg_errors.csv`` /
        ``*-end_errors.csv`` / ``*-log-pc_align-*.txt``. Missing/empty dirs are
        skipped with a message (lets not-yet-aligned basins run harmlessly).
    aoi_shp : path-like or None
        AOI shapefile; strips whose DEM bbox misses it are dropped. ``None``
        keeps every strip (per-basin ASP roots are already AOI-scoped).
    output_path : path-like
        PNG to write (parent dirs created).
    title : str
        Figure suptitle.
    ylim : (lo, hi)
        Shared log-y limits (m) for the residual rows.

    Returns
    -------
    dict
        ``label -> {n, pre_p50, post_p50, improvement}`` for each plotted
        variant (empty if nothing was harvested).
    """
    import re
    from datetime import datetime

    import fiona
    import matplotlib.dates as mdates
    import rasterio
    from shapely.geometry import box, shape

    ned_re = re.compile(
        r"Translation vector \(North-East-Down, meters\):\s*Vector3\(\s*"
        r"([\d.eE+\-]+)\s*,\s*([\d.eE+\-]+)\s*,\s*([\d.eE+\-]+)\s*\)"
    )
    date_re = re.compile(r"_(\d{8})_")
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red",
               "tab:purple", "tab:brown"]

    def error_pcts(csv_path):
        if not csv_path.exists():
            return None
        try:
            df = pd.read_csv(csv_path, comment="#", header=None,
                             names=["easting", "northing", "height", "error"])
        except (OSError, pd.errors.EmptyDataError):
            return None
        err = df["error"].to_numpy(dtype=float)
        err = err[np.isfinite(err)]
        if err.size == 0:
            return None
        return tuple(float(v) for v in np.percentile(err, [16, 50, 84]))

    def parse_ned(log_path):
        try:
            m = ned_re.search(log_path.read_text())
        except OSError:
            return None
        if not m:
            return None
        n, e, d = float(m.group(1)), float(m.group(2)), float(m.group(3))
        return e, n, -d  # x=East, y=North, z=Up

    def harvest(aligned_dir):
        rows = []
        aligned_dir = Path(aligned_dir)
        if not aligned_dir.exists():
            return rows
        for dem in sorted(aligned_dir.glob("*-trans_reference-DEM.tif")):
            stem = dem.name.replace("-trans_reference-DEM.tif", "")
            if (aligned_dir / f"{stem}.bad_align").exists():
                continue
            if aoi is not None:
                try:
                    with rasterio.open(dem) as src:
                        b = src.bounds
                    if not box(b.left, b.bottom, b.right, b.top).intersects(aoi):
                        continue
                except Exception:
                    continue
            beg = error_pcts(aligned_dir / f"{stem}-beg_errors.csv")
            end = error_pcts(aligned_dir / f"{stem}-end_errors.csv")
            dm = date_re.search(stem)
            if beg is None or end is None or dm is None:
                continue
            logs = sorted(aligned_dir.glob(f"{stem}-log-pc_align-*.txt"))
            ned = parse_ned(logs[-1]) if logs else None
            rows.append({"date": datetime.strptime(dm.group(1), "%Y%m%d"),
                         "beg": beg, "end": end, "ned": ned})
        rows.sort(key=lambda r: r["date"])
        return rows

    aoi = None
    if aoi_shp is not None:
        with fiona.open(aoi_shp) as src:
            aoi = shape(next(iter(src))["geometry"])

    pops = []
    summary: dict = {}
    for i, v in enumerate(variants):
        aligned_dir, label = v[0], v[1]
        colour = v[2] if len(v) > 2 else palette[i % len(palette)]
        rows = harvest(aligned_dir)
        if not rows:
            print(f"  [{label}] no strips harvested from {aligned_dir}")
            continue
        ts = np.array([r["date"] for r in rows])
        beg = np.array([r["beg"] for r in rows])            # (n, 3)
        end = np.array([r["end"] for r in rows])            # (n, 3)
        ned = np.array([r["ned"] if r["ned"] else (np.nan,) * 3
                        for r in rows])                     # (n, 3) x,y,z
        impr = float(np.median(beg[:, 1]) / np.median(end[:, 1]))
        pops.append((label, colour, ts, beg, end, ned))
        summary[label] = {"n": len(rows), "pre_p50": float(np.median(beg[:, 1])),
                          "post_p50": float(np.median(end[:, 1])),
                          "improvement": impr}
        print(f"  [{label}] n={len(rows)}  pre p50={np.median(beg[:,1]):.2f} m  "
              f"post p50={np.median(end[:,1]):.3f} m  (x{impr:.1f} improvement)")

    if not pops:
        print(f"  no populations to plot for '{title}' — skipping figure")
        return summary

    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(3, 3, height_ratios=(1.0, 1.0, 1.0),
                          hspace=0.30, wspace=0.28)
    ax_yx = fig.add_subplot(gs[0, 0])
    ax_zx = fig.add_subplot(gs[0, 1])
    ax_zy = fig.add_subplot(gs[0, 2])
    ax_pre = fig.add_subplot(gs[1, :])
    ax_post = fig.add_subplot(gs[2, :], sharex=ax_pre)

    for label, colour, ts, beg, end, ned in pops:
        kw = dict(color=colour, alpha=0.6, s=26, edgecolor="none")
        ax_yx.scatter(ned[:, 0], ned[:, 1], label=f"{label} (n={len(ts)})", **kw)
        ax_zx.scatter(ned[:, 0], ned[:, 2], **kw)
        ax_zy.scatter(ned[:, 1], ned[:, 2], **kw)
    for ax, (xl, yl, tt) in zip(
        (ax_yx, ax_zx, ax_zy),
        (("x offset East (m)", "y offset North (m)", "ICP translation: y vs x"),
         ("x offset East (m)", "z offset Up (m)", "z vs x"),
         ("y offset North (m)", "z offset Up (m)", "z vs y")),
    ):
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.axvline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(tt, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="datalim")
    ax_yx.legend(loc="best", fontsize=8)

    for ax, idx, ttl in ((ax_pre, 1, "PRE-alignment residual (beg_errors)"),
                         (ax_post, 2, "POST-alignment residual (end_errors)")):
        for label, colour, ts, beg, end, ned in pops:
            arr = beg if idx == 1 else end
            yerr = np.vstack([arr[:, 1] - arr[:, 0], arr[:, 2] - arr[:, 1]])
            impr = np.median(beg[:, 1]) / np.median(end[:, 1])
            lab = f"{label}  (n={len(ts)}, p50={np.median(arr[:,1]):.2f} m"
            lab += f", ×{impr:.1f})" if idx == 2 else ")"
            ax.errorbar(ts, arr[:, 1], yerr=yerr, fmt="o", color=colour, ms=5,
                        mew=0, elinewidth=1.0, capsize=2, alpha=0.75, label=lab,
                        zorder=3)
            ax.axhline(np.median(arr[:, 1]), color=colour, ls="--", lw=1.3,
                       alpha=0.9, zorder=1)
        ax.set_yscale("log")
        ax.set_ylim(*ylim)
        ax.set_ylabel("pc_align point-to-plane\nresidual (m), p50 + 16–84% bars")
        ax.set_title(ttl, fontsize=11)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper right", fontsize=9)
    ax_post.set_xlabel("Strip acquisition date")
    ax_post.xaxis.set_major_locator(mdates.YearLocator())
    ax_post.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle(title, fontsize=14, y=0.995)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output_path}")
    return summary

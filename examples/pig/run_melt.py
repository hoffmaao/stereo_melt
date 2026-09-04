"""Run the Pine Island Lagrangian melt-rate pipeline on the built stack.

Stage 4+ of the Pine Island pipeline. Loads the already-built DEM stack,
fetches matching velocity and SMB, and runs
:func:`stereo_melt.melt.lagrangian_melt_rate` (and Eulerian as a
side-by-side comparison). Writes a NetCDF with every variable of the
returned Dataset plus QC plots to ``pig/figures/``.

Run:

    python -m pig.run_melt

Assumes :mod:`pig.build_stack` has already produced
``processed/pig_stack_<start>_<end>.nc``.
"""

from __future__ import annotations

import os
import warnings
import sys

# Force PROJ database to the active env before any pyproj import. This env's
# base-install proj.db has stale metadata that breaks EPSG code lookups.
_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from stereo_melt.colormaps import add_melt_colorbar, melt_cmap, melt_norm  # noqa: E402
from stereo_melt.flux import grounding_buffer, integrate_basal_flux  # noqa: E402
from stereo_melt.io.bedmachine import load_firn_on_grid  # noqa: E402
from stereo_melt.io.smb import smb_over_window  # noqa: E402
from stereo_melt.kinematics import clean_temporal_outliers, gaussian_smooth_nan  # noqa: E402
from stereo_melt.dynamics.stubblefield_inverse import stubblefield_inverse_melt_rate  # noqa: E402
from stereo_melt.melt import (  # noqa: E402
    eulerian_melt_rate,
    lagrangian_melt_rate,
    lagrangian_parcel_lsq_melt_rate,
)
from stereo_melt.stack import load_basin_stack  # noqa: E402
from stereo_melt.visualization import plot_variational_fit  # noqa: E402

from pig import config  # noqa: E402

SECONDS_PER_YEAR = 86400.0 * 365.25


# ----------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------


def load_stack(stack_prefix: str = "pig_stack") -> xr.DataArray:
    stack, _ = load_basin_stack(
        config.PROCESSED_DIR,
        stack_prefix,
        config.START_TIME,
        config.END_TIME,
        prefer_tilt_corrected=True,
        bad_epochs=tuple(getattr(config, "BAD_EPOCHS", ())),
        bad_strips=tuple(getattr(config, "BAD_STRIPS", ())),
    )
    return stack


def _subset_velocity(
    ds: xr.Dataset, stack: xr.DataArray, vx_name: str, vy_name: str, buf_m: float = 2000.0
) -> xr.Dataset:
    """Sub-select a velocity dataset around the stack bbox, respecting
    y-coord direction (ascending vs descending)."""
    x_min = float(stack["x"].min()) - buf_m
    x_max = float(stack["x"].max()) + buf_m
    y_min = float(stack["y"].min()) - buf_m
    y_max = float(stack["y"].max()) + buf_m
    vy_src = ds["y"].values
    if vy_src[0] > vy_src[-1]:  # descending
        sub = ds.sel(x=slice(x_min, x_max), y=slice(y_max, y_min))
    else:
        sub = ds.sel(x=slice(x_min, x_max), y=slice(y_min, y_max))
    return sub[[vx_name, vy_name]].rename({vx_name: "vx", vy_name: "vy"}).load()


_VELOCITY_FALLBACK = "measures"
_VELOCITY_PRODUCTION = "fused"   # basin decision record (local, untracked)


def load_velocity_on_grid(stack: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray, str]:
    """Load velocity ``vx``/``vy`` cropped + resampled onto the stack grid.

    Velocity source is selected by the ``PIG_VELOCITY`` env var:
    ``measures`` (NSIDC-0754 phase map at 450 m), ``nsidc-0525``
    (Scheuchl 2012 Central Antarctica 2009 mosaic at 900 m), ``its_live``
    (single annual mosaic at 120 m), ``itslive-multiyear`` (the annual
    ITS_LIVE mosaics stacked along ``time`` → time-varying advection in the
    Lagrangian solver), ``ase-quarterly`` (quarterly ASE velocity mosaics
    at 250 m, 2015 Q1–2024 Q1 / Joughin v05.0, stacked along ``time`` for
    sub-annual time-resolved advection), or ``fused`` (those quarterlies
    after EOF + Kalman temporal fusion and a 1 km spatial Gaussian — the
    production choice, the basin decision record (local, untracked)). Remaining NaN gaps are filled
    with 0 m/yr.

    When ``PIG_VELOCITY`` is unset the source falls back to ``measures``
    and a ``RuntimeWarning`` is raised; the fallback itself is unchanged.
    Drivers should pin their source (``os.environ.setdefault``). The
    chosen source is returned and stamped on melt products as the
    ``velocity`` attr.
    """
    # PIG_VELOCITY unset is a silent trap: the fallback here is NOT the
    # documented production choice (basin decision record: `fused`), and a caller that
    # forgets it gets a different velocity field with no error — which has
    # twice produced contaminated melt comparisons. Callers that genuinely
    # want another source pin it with os.environ.setdefault (reproduce_shean,
    # compare_shean2019, compare_budget_inverse) and never see this.
    if "PIG_VELOCITY" not in os.environ:
        warnings.warn(
            "PIG_VELOCITY is not set: falling back to "
            f"{_VELOCITY_FALLBACK!r}, which is NOT the documented production "
            f"choice ({_VELOCITY_PRODUCTION!r}). Set PIG_VELOCITY explicitly "
            "(or os.environ.setdefault it) before comparing against a stored "
            "product; melt products carry the source in their `velocity` attr.",
            RuntimeWarning, stacklevel=2)
    requested = os.environ.get("PIG_VELOCITY", _VELOCITY_FALLBACK).strip().lower()
    source = None
    if requested == "nsidc-0525":
        from stereo_melt.io.velocity import load_nsidc_0525
        if not config.NSIDC_0525_2009_NC.exists():
            raise SystemExit(
                f"NSIDC-0525 requested but file missing at {config.NSIDC_0525_2009_NC}"
            )
        ds = load_nsidc_0525(config.NSIDC_0525_2009_NC)
        sub = _subset_velocity(ds, stack, vx_name="vx", vy_name="vy")
        source = f"NSIDC-0525 (Scheuchl 2012, 2009 mosaic) 900 m (finite frac {float(sub['vx'].notnull().mean()):.2f})"
    elif requested == "its_live":
        if not config.ITS_LIVE_2019.exists():
            raise SystemExit(f"ITS_LIVE requested but file missing at {config.ITS_LIVE_2019}")
        ds = xr.open_dataset(config.ITS_LIVE_2019)
        sub = _subset_velocity(ds, stack, vx_name="vx", vy_name="vy")
        source = f"ITS_LIVE 2019 120 m (finite frac {float(sub['vx'].notnull().mean()):.2f})"
    elif requested in ("itslive-multiyear", "itslive_multiyear", "its_live-multiyear"):
        # Time-varying velocity: stack the annual ITS_LIVE mosaics on disk and
        # return a (time, y, x) field. lagrangian_melt_rate samples it
        # time-resolved along each trajectory; the Eulerian solver mean-collapses
        # it (so this only changes the Lagrangian product). Mid-year timestamps
        # date each annual mosaic. To extend coverage (2018, 2022, 2023) add
        # ITS_LIVE_<year> entries to config + a dedicated cache pull.
        years = [
            (2019, config.ITS_LIVE_2019),
            (2020, config.ITS_LIVE_2020),
            (2021, config.ITS_LIVE_2021),
        ]
        have = [(y, p) for y, p in years if p.exists()]
        if len(have) < 2:
            raise SystemExit(
                "itslive-multiyear needs >=2 annual mosaics on disk; found "
                f"{[y for y, _ in have]}"
            )
        vxs, vys, tcoords, frac0 = [], [], [], None
        for y, p in have:
            ds = xr.open_dataset(p)
            s = _subset_velocity(ds, stack, vx_name="vx", vy_name="vy")
            vxi = s["vx"].interp(x=stack["x"], y=stack["y"], method="linear")
            vyi = s["vy"].interp(x=stack["x"], y=stack["y"], method="linear")
            if frac0 is None:
                frac0 = float(vxi.notnull().mean())
            vxs.append(vxi.fillna(0.0))
            vys.append(vyi.fillna(0.0))
            tcoords.append(pd.Timestamp(f"{y}-07-02"))
        tdim = pd.DatetimeIndex(tcoords)
        vx = xr.concat(vxs, dim="time").assign_coords(time=tdim)
        vy = xr.concat(vys, dim="time").assign_coords(time=tdim)
        source = (
            f"ITS_LIVE multi-year {[y for y, _ in have]} 120 m, time-varying "
            f"(finite frac {frac0:.2f})"
        )
        print(f"  velocity source: {source}")
        return vx, vy, source
    elif requested in ("ase-quarterly", "ase_quarterly", "quarterly", "fused", "ase-fused"):
        # Quarterly ASE velocity mosaics (vx/vy, 250 m, Joughin v05.0 2015 Q1–
        # 2024 Q1): true sub-annual cadence for the time-varying Lagrangian
        # advection. Each quarter is dated by the midpoint of its ddMonyy_ddMonyy
        # window; vx and vy live in sibling files. Coverage now spans the full
        # PIG window; only epochs past 2024 Q1 fall back to the last quarter
        # (constant extrapolation in lagrangian_melt_rate).
        import re

        import rioxarray  # noqa: F401  (provides open_rasterio)

        vx_files = sorted(
            p for p in config.ASE_QUARTERLY_VEL_DIR.glob("*_vx_*.tif")
            if not p.name.startswith("._")
        )
        recs = {}  # quarter midpoint -> (vx_path, vy_path)
        for vxp in vx_files:
            m = re.search(r"Quarterly_([0-9A-Za-z]+)_([0-9A-Za-z]+)_vx_", vxp.name)
            if not m:
                continue
            t0 = pd.to_datetime(m.group(1), format="%d%b%y")
            t1 = pd.to_datetime(m.group(2), format="%d%b%y")
            vyp = vxp.with_name(vxp.name.replace("_vx_", "_vy_"))
            if vyp.exists():
                recs[t0 + (t1 - t0) / 2] = (vxp, vyp)
        if len(recs) < 2:
            raise SystemExit(
                f"ase-quarterly needs >=2 paired vx/vy quarters in "
                f"{config.ASE_QUARTERLY_VEL_DIR}; found {len(recs)}"
            )
        # Physical speed cap: the mosaics carry isolated unphysical spikes
        # (>30 km/yr at data-void / ocean edges) vs PIG's ~4 km/yr maximum.
        # Left in, they advect seeds out of domain and inject huge artefacts
        # into div(u) (the H*div(u) term is sensitive to high-freq velocity
        # noise). Mask speeds above the cap; the existing fillna(0) zeroes them.
        VMAX_MYR = 6000.0
        mids = sorted(recs)
        vxs, vys, frac0 = [], [], None
        for mid in mids:
            vxp, vyp = recs[mid]
            vraw = rioxarray.open_rasterio(vxp, masked=True).squeeze("band", drop=True)
            wraw = rioxarray.open_rasterio(vyp, masked=True).squeeze("band", drop=True)
            vxi = vraw.interp(x=stack["x"], y=stack["y"], method="linear")
            vyi = wraw.interp(x=stack["x"], y=stack["y"], method="linear")
            bad = (vxi**2 + vyi**2) ** 0.5 > VMAX_MYR
            vxi = vxi.where(~bad)
            vyi = vyi.where(~bad)
            if frac0 is None:
                frac0 = float(vxi.notnull().mean())
            # Pass NaN gaps (and clipped outliers) through, NOT fillna(0):
            # lagrangian_melt_rate drops parcels that enter a gap, whereas a 0
            # fill would freeze them and inject spurious divergence at gap edges.
            vxs.append(vxi.load())
            vys.append(vyi.load())
        tdim = pd.DatetimeIndex(mids)
        vx = xr.concat(vxs, dim="time").assign_coords(time=tdim).reset_coords(drop=True)
        vy = xr.concat(vys, dim="time").assign_coords(time=tdim).reset_coords(drop=True)
        fused = requested in ("fused", "ase-fused")
        if fused:
            from stereo_melt.dynamics.velocity_fusion import fuse_velocity_field

            # Denoise + harmonic gap-fill + EOF + Kalman temporal fusion of the
            # quarterly mosaics, plus a modest 1 km spatial Gaussian on velocity
            # ONLY (the 250 m melt grid is untouched) for a less-noisy div(u).
            # Fused back onto the quarterly axis: a drop-in, cleaner velocity for
            # the Lagrangian advection -- the Shean-matching velocity rep, but
            # keeping our 250 m resolution rather than Shean's 512 m/3.5 km blur.
            vx = vx.transpose("time", "y", "x")
            vy = vy.transpose("time", "y", "x")
            vx, vy, fdiag = fuse_velocity_field(
                vx, vy, vx["time"].values,
                n_modes=8, coverage_frac=0.4, seasonal=True,
                temporal_model="kalman", spatial_smooth_m=1000.0, verbose=True,
            )
            print(
                f"  velocity FUSED: kalman+EOF k={fdiag['k_retained']} "
                f"(cumvar {fdiag['cum_var']:.4f}); 1 km Gaussian "
                f"sigma={fdiag['spatial_smooth_sigma_px']:.1f}px"
            )
        vmax = float(np.abs(vx).max())
        source = (
            f"ASE quarterly{' FUSED(kalman+1km)' if fused else ''} "
            f"{mids[0].date()}..{mids[-1].date()} ({len(mids)} quarters) 250 m, "
            f"time-varying (finite frac {frac0:.2f}, max|v|={vmax:.0f} m/yr)"
        )
        print(f"  velocity source: {source}")
        return vx, vy, source
    else:
        # Default path: MEaSUREs preferred, ITS_LIVE fallback.
        if config.MEASURES_PHASE_NC.exists():
            ds = xr.open_dataset(config.MEASURES_PHASE_NC)
            sub = _subset_velocity(ds, stack, vx_name="VX", vy_name="VY")
            finite_frac = float(sub["vx"].notnull().mean())
            if finite_frac > 0.5:
                source = f"MEaSUREs 450 m (finite frac {finite_frac:.2f})"
        if source is None and config.ITS_LIVE_2019.exists():
            ds = xr.open_dataset(config.ITS_LIVE_2019)
            sub = _subset_velocity(ds, stack, vx_name="vx", vy_name="vy")
            source = f"ITS_LIVE 2019 120 m (finite frac {float(sub['vx'].notnull().mean()):.2f})"
    if source is None:
        raise SystemExit("No usable velocity source found — install MEaSUREs or ITS_LIVE.")

    print(f"  velocity source: {source}")

    # Resample onto the 25 m stack grid via linear interpolation.
    vx = sub["vx"].interp(x=stack["x"], y=stack["y"], method="linear")
    vy = sub["vy"].interp(x=stack["x"], y=stack["y"], method="linear")

    # Fill residual NaN pixels with 0 m/yr — a reasonable placeholder
    # over ice-free cells near the grounding zone where velocity is
    # undefined. Small fraction (< 5 %); won't bias the flow-divergence.
    vx = vx.fillna(0.0)
    vy = vy.fillna(0.0)

    return vx, vy, source


def apply_min_extent(
    floating: xr.DataArray,
    mask_suffix: str | None = None,
    file_start: str | None = None,
    file_end: str | None = None,
) -> xr.DataArray:
    """Intersect a static floating mask with the window-minimum shelf extent.

    Window-minimum shelf extent (calving-aware): built by
    pig.build_min_extent_mask from Greene 2022 observed coastlines + the
    stack's per-epoch ocean test. Pixels the shelf lost mid-window must not
    enter the solvers as ice-to-ocean dh/dt cliffs. ``PIG_MIN_EXTENT=0`` opts
    out. The cached file covers the FULL stack window, so a --start/--end
    sub-window run gets the (conservative) full-window minimum.

    ``mask_suffix`` is the stack variant the cached mask was built for
    (``"_250m_is2ctempo"``, i.e. the ``<res>m_<tag>`` suffix of the loaded
    stack) and is required: the mask is per geometry, and a default would
    silently intersect another stack's coastline. A missing cached file is
    an error, not a fallback -- the product would otherwise re-admit the
    calved sector with nothing in its metadata to say so.

    The returned mask carries ``attrs["min_extent_mask"]``: the cached
    file's name when applied, or the literal ``"not applied"`` under
    ``PIG_MIN_EXTENT=0``; drivers copy it onto their products.

    Shared by run_melt's main(), pig.run_melt_bridging and the standalone
    solver rigs (pig/scripts/fused_melt_map.py) so every product sees the
    same geometry -- a rig that skips it re-admits the calved sector, whose
    ice-to-ocean cliff is ~28 Gt/yr of spurious melt in the budget legs and
    a large-scale dh/dt mode the DC-blind legs cannot represent.
    """
    if os.environ.get("PIG_MIN_EXTENT", "1") == "0":
        floating = floating.copy(deep=False)
        floating.attrs["min_extent_mask"] = "not applied"
        return floating
    if mask_suffix is None:
        raise ValueError(
            "apply_min_extent needs mask_suffix (the loaded stack's "
            "'_<res>m_<tag>' suffix, e.g. '_250m_is2ctempo'): the cached "
            "min-extent mask is per stack geometry"
        )
    file_start = file_start or config.START_TIME
    file_end = file_end or config.END_TIME
    min_ext_nc = (
        config.PROCESSED_DIR
        / f"pig_min_extent{mask_suffix}_{file_start}_{file_end}.nc"
    )
    if not min_ext_nc.exists():
        raise FileNotFoundError(
            f"min-extent mask {min_ext_nc} not found. Build it with "
            "`python -m pig.build_min_extent_mask` for this stack, or set "
            "PIG_MIN_EXTENT=0 to deliberately run on the static floating "
            "mask (re-admits the calved sector)."
        )
    with xr.open_dataset(min_ext_nc) as _mds:
        min_ext = _mds["min_extent_mask"].astype(bool).load()
    _n_static = int(floating.sum())
    floating = floating & min_ext
    floating.attrs["min_extent_mask"] = min_ext_nc.name
    print(
        f"  min-extent mask {min_ext_nc.name}: removed "
        f"{_n_static - int(floating.sum())} of {_n_static} floating px"
    )
    return floating


def load_floating_mask(stack: xr.DataArray) -> xr.DataArray:
    """Return a boolean floating-ice mask on the stack grid.

    Reads BedMachine Antarctica v3 ``mask`` (flag_values 0=ocean,
    1=ice_free_land, 2=grounded_ice, 3=floating_ice, 4=lake_vostok),
    crops to the stack bbox, and resamples to the 25 m target grid
    via nearest-neighbor (the mask is categorical).
    """
    if not config.BEDMACHINE_NC.exists():
        raise SystemExit(
            f"BedMachine not found at {config.BEDMACHINE_NC}. "
            "Cannot build floating-ice mask."
        )
    ds = xr.open_dataset(config.BEDMACHINE_NC)
    buf = 2000.0
    x_min = float(stack["x"].min()) - buf
    x_max = float(stack["x"].max()) + buf
    y_min = float(stack["y"].min()) - buf
    y_max = float(stack["y"].max()) + buf
    by = ds["y"].values
    if by[0] > by[-1]:
        sub = ds["mask"].sel(x=slice(x_min, x_max), y=slice(y_max, y_min))
    else:
        sub = ds["mask"].sel(x=slice(x_min, x_max), y=slice(y_min, y_max))
    sub = sub.load()
    on_grid = sub.interp(x=stack["x"], y=stack["y"], method="nearest")
    floating = (on_grid == 3)
    floating.attrs = {
        "source": "BedMachine Antarctica v3 mask == 3 (floating_ice)",
        "flag_values": "0=ocean 1=ice_free_land 2=grounded_ice 3=floating_ice 4=lake_vostok",
    }
    floating.name = "floating_mask"
    return floating


def load_grounded_mask(stack: xr.DataArray) -> xr.DataArray:
    """Return a boolean grounded-ice mask (BedMachine mask==2) on the stack grid.

    Mirror of :func:`load_floating_mask` for code 2 (grounded_ice); used to build
    the grounding-line integration buffer in :func:`stereo_melt.flux.grounding_buffer`.
    """
    ds = xr.open_dataset(config.BEDMACHINE_NC)
    buf = 2000.0
    x_min = float(stack["x"].min()) - buf
    x_max = float(stack["x"].max()) + buf
    y_min = float(stack["y"].min()) - buf
    y_max = float(stack["y"].max()) + buf
    by = ds["y"].values
    if by[0] > by[-1]:
        sub = ds["mask"].sel(x=slice(x_min, x_max), y=slice(y_max, y_min))
    else:
        sub = ds["mask"].sel(x=slice(x_min, x_max), y=slice(y_min, y_max))
    on_grid = sub.load().interp(x=stack["x"], y=stack["y"], method="nearest")
    grounded = (on_grid == 2)
    grounded.name = "grounded_mask"
    return grounded


def load_smb_on_grid(stack: xr.DataArray) -> xr.DataArray:
    """Integrate RACMO2.4p1 SMB over the stack time window and regrid to 25 m."""
    start = config.START_TIME
    end = config.END_TIME
    m_ice_cumulative = smb_over_window(
        str(config.RACMO_SMB_NC),
        stack["x"].values,
        stack["y"].values,
        start=start,
        end=end,
        method="linear",
    )
    # Cumulative m ice over the window → mean rate m ice / yr
    dt_years = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / SECONDS_PER_YEAR
    m_ice_per_yr = m_ice_cumulative / dt_years
    return xr.DataArray(
        m_ice_per_yr,
        dims=("y", "x"),
        coords={"y": stack["y"].values, "x": stack["x"].values},
        name="a_dot",
        attrs={
            "units": "m ice yr^-1",
            "integration_window": f"{start} to {end}",
            "source": "RACMO2.4p1 smbgl (Zenodo 19255213)",
        },
    )


# ----------------------------------------------------------------------
# QC plotting
# ----------------------------------------------------------------------


def _imshow_xr(ax, da: xr.DataArray, *, cmap, vmin=None, vmax=None, norm=None):
    # `norm` and `vmin`/`vmax` are mutually exclusive in matplotlib; melt-rate
    # panels pass the symmetric-log `melt_norm`, everything else stays linear.
    kw = {"norm": norm} if norm is not None else {"vmin": vmin, "vmax": vmax}
    im = ax.imshow(
        da.values,
        extent=[
            float(da["x"].min()),
            float(da["x"].max()),
            float(da["y"].min()),
            float(da["y"].max()),
        ],
        origin="upper",
        cmap=cmap,
        aspect="equal",
        **kw,
    )
    return im


def plot_inputs(stack, vx, vy, a_dot, firn, out_path: Path) -> None:
    """QC: time-mean surface, velocity magnitude, SMB, firn air content."""
    # Collapse a time-varying velocity field to its mean for the QC panel.
    if "time" in vx.dims:
        vx = vx.mean("time", skipna=True)
        vy = vy.mean("time", skipna=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)

    im0 = _imshow_xr(axes[0], stack.mean("time", skipna=True), cmap="terrain")
    axes[0].set_title("time-mean surface (m)")
    fig.colorbar(im0, ax=axes[0], fraction=0.045)

    speed = np.sqrt(vx**2 + vy**2)
    im1 = _imshow_xr(axes[1], speed, cmap="viridis")
    axes[1].set_title("|v| (m/yr)")
    fig.colorbar(im1, ax=axes[1], fraction=0.045)

    im2 = _imshow_xr(axes[2], a_dot, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    axes[2].set_title(f"SMB rate (m ice/yr)\n{config.START_TIME}→{config.END_TIME}")
    fig.colorbar(im2, ax=axes[2], fraction=0.045)

    im3 = _imshow_xr(axes[3], firn, cmap="viridis")
    axes[3].set_title("firn air content (m)\nBedMachine static FAC")
    fig.colorbar(im3, ax=axes[3], fraction=0.045)

    for ax in axes:
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")

    fig.suptitle("Pine Island melt-rate inputs", fontsize=12)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_melt_comparison(
    euler: xr.Dataset,
    lagr: xr.Dataset,
    linv: xr.Dataset | None,
    out_path: Path,
    clim=(-60.0, 60.0),
) -> None:
    """QC: side-by-side Eulerian vs Lagrangian vs linear-inverse melt."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)

    # Melt panels share the LADDIE symmetric-log scale (black at zero, log
    # decades outward): a linear +/-60 stretch buries everything below ~5 m/yr
    # in the white middle, which on PIG is most of the shelf. Difference and
    # flux-divergence panels below stay linear -- they are not melt rates and
    # the negative=melt palette would misread on them.
    mcmap = melt_cmap()
    mnorm = melt_norm(vmax=max(abs(clim[0]), abs(clim[1])))

    im0 = _imshow_xr(axes[0, 0], euler.melt_rate, cmap=mcmap, norm=mnorm)
    axes[0, 0].set_title("Eulerian melt_rate (m ice/yr)")
    add_melt_colorbar(fig, im0, ax=axes[0, 0], fraction=0.045)

    im1 = _imshow_xr(axes[0, 1], lagr.melt_rate, cmap=mcmap, norm=mnorm)
    axes[0, 1].set_title("Lagrangian melt_rate (m ice/yr)")
    add_melt_colorbar(fig, im1, ax=axes[0, 1], fraction=0.045)

    if linv is not None:
        im2 = _imshow_xr(axes[0, 2], linv.melt_rate, cmap=mcmap, norm=mnorm)
        axes[0, 2].set_title("Stubblefield non-hydrostatic inverse")
        add_melt_colorbar(fig, im2, ax=axes[0, 2], fraction=0.045)
    else:
        axes[0, 2].set_visible(False)

    diff_lag = lagr.melt_rate - euler.melt_rate
    im3 = _imshow_xr(axes[1, 0], diff_lag, cmap="PuOr", vmin=-2.0, vmax=2.0)
    axes[1, 0].set_title("Lagrangian − Eulerian")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.045)

    if linv is not None:
        diff_lin = linv.melt_rate - lagr.melt_rate
        im4 = _imshow_xr(axes[1, 1], diff_lin, cmap="PuOr", vmin=-2.0, vmax=2.0)
        axes[1, 1].set_title("Stubblefield − Lagrangian")
        fig.colorbar(im4, ax=axes[1, 1], fraction=0.045)
    else:
        axes[1, 1].set_visible(False)

    im5 = _imshow_xr(axes[1, 2], euler.flux_div, cmap="RdBu", vmin=-5.0, vmax=5.0)
    axes[1, 2].set_title("∇·(H_f u) (m ice/yr)")
    fig.colorbar(im5, ax=axes[1, 2], fraction=0.045)

    for ax in axes.ravel():
        ax.set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    axes[1, 0].set_ylabel("y (m)")

    fig.suptitle(
        f"Pine Island melt rate — {config.START_TIME} to {config.END_TIME}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def _maybe_smooth_velocity(vx: xr.DataArray, vy: xr.DataArray):
    """Opt-in Shean-style velocity smoothing (``PIG_VEL_SMOOTH_KM``, default off).

    A NaN-aware Gaussian on velocity tames the near-grounding-line flux-
    divergence overshoot (the dominant integral inflater on clean DEMs). Applied
    per time-slice so time-varying Lagrangian advection is preserved. Returns
    ``(vx, vy, km)`` with ``km=None`` when disabled.
    """
    km = float(os.environ.get("PIG_VEL_SMOOTH_KM", "0") or 0)
    if km <= 0:
        return vx, vy, None
    res = abs(float(vx["x"].values[1] - vx["x"].values[0]))
    sig = km * 1000.0 / res

    def smo(v):
        if "time" in v.dims:
            slices = [gaussian_smooth_nan(v.isel(time=i), sig) for i in range(v.sizes["time"])]
            return xr.concat(slices, dim="time").assign_coords(time=v["time"])
        return gaussian_smooth_nan(v, sig)

    return smo(vx), smo(vy), km


def main(
    res_override: float | None = None,
    start: str | None = None,
    end: str | None = None,
    tag: str | None = None,
) -> None:
    config.ensure_output_dirs()

    if res_override is not None:
        stack_prefix = f"pig_stack_{int(round(res_override))}m"
        out_suffix = f"_{int(round(res_override))}m"
    else:
        stack_prefix = "pig_stack"
        out_suffix = ""
    if tag:
        stack_prefix += f"_{tag}"
        out_suffix += f"_{tag}"
    # The min-extent mask file is named by res/tag only (no MELT_OUT_SUFFIX):
    # every output variant of the same stack shares one geometry.
    mask_suffix = out_suffix
    # MELT_OUT_SUFFIX appends to the OUTPUT name only (not the stack loaded), so a
    # variant run (e.g. fused velocity on the same is2ctempo stack) writes beside
    # the baseline instead of overwriting it -- enables a clean A/B.
    _out_extra = os.environ.get("MELT_OUT_SUFFIX", "").strip()
    if _out_extra:
        out_suffix += _out_extra if _out_extra.startswith("_") else f"_{_out_extra}"

    # The stack FILE on disk is always the full config-window build (e.g. the
    # fused 2010-2024 stack). --start/--end then select an analysis SUB-WINDOW
    # of that same file: the solvers, SMB integration window, output filename,
    # plot titles and attrs all follow the sub-window, but the underlying
    # alignment / tilt-fit / corrections / bad-epoch screen are byte-identical
    # to the full run. That isolates the effect of *which epochs* enter the
    # inversion (e.g. IS2-era-only vs the full record) without rebuilding
    # anything -- a clean workflow-soundness control.
    #
    # ORDER IS LOAD-THEN-MUTATE, NEVER THE REVERSE: load_basin_stack() builds
    # its filename from config.START_TIME/END_TIME, and a separately-built,
    # separately-aligned standalone stack can exist at the sub-window name
    # (e.g. pig_stack_250m_tilt_corrected_2018-01-01_2024-01-10.nc). Mutating
    # config before load_stack() would silently load that WRONG file. So we
    # load the full file first, then .sel() the sub-window out of it.
    file_start, file_end = config.START_TIME, config.END_TIME
    analysis_start = start or file_start
    analysis_end = end or file_end
    is_subwindow = (analysis_start, analysis_end) != (file_start, file_end)

    print("Loading stack...")
    stack = load_stack(stack_prefix=stack_prefix)
    print(f"  dims: time={stack.sizes['time']}, y={stack.sizes['y']}, x={stack.sizes['x']}")

    win_tag = ""
    if is_subwindow:
        n_before = stack.sizes["time"]
        stack = stack.sel(time=slice(analysis_start, analysis_end))
        # Mutate the per-run config so the SMB window, output naming, plot
        # titles and attrs below all follow the analysis sub-window (they read
        # config.START_TIME/END_TIME live, downstream of this point).
        config.START_TIME = analysis_start
        config.END_TIME = analysis_end
        # The NetCDF name embeds the dates already, so it is collision-free on
        # its own; the PNGs carry only out_suffix and would otherwise clobber
        # the full-window figures -- tag them with the window too.
        win_tag = f"_{analysis_start}_{analysis_end}"
        print(
            f"  analysis sub-window {analysis_start}..{analysis_end} "
            f"(of full {file_start}..{file_end} stack file): "
            f"time={stack.sizes['time']} (of {n_before})"
        )

    # P5: per-pixel temporal NMAD blunder rejection (Shean make_stack-style).
    # Opt-in via PIG_STACK_NMAD_SIGMA (default OFF until validated; 4 is a
    # sensible value). Removes blunder observations that inflate dh/dt +
    # Lagrangian rmse, so more of the shelf clears the quality gate. Operates on
    # the tilt-corrected stack in-memory; the saved stack on disk is untouched.
    _nmad_sigma = float(os.environ.get("PIG_STACK_NMAD_SIGMA", "0") or 0)
    if _nmad_sigma > 0:
        stack, _clean = clean_temporal_outliers(stack, n_sigma=_nmad_sigma, min_count=3)
        print(
            f"  P5 temporal NMAD (sigma={_nmad_sigma:g}): rejected "
            f"{_clean['n_obs_rejected']:,}/{_clean['n_obs_before']:,} obs "
            f"({100 * _clean['n_obs_rejected'] / max(_clean['n_obs_before'], 1):.2f}%), "
            f"dropped {_clean['n_pix_dropped']:,} low-count pixels"
        )

    print("Building floating-ice mask (BedMachine v3)...")
    floating = load_floating_mask(stack)
    frac_floating = float(floating.mean())
    print(f"  floating-ice fraction of AOI: {frac_floating:.3f}")

    floating = apply_min_extent(floating, mask_suffix, file_start, file_end)
    min_extent_src = floating.attrs["min_extent_mask"]
    stack = stack.where(floating)

    print("Loading velocity...")
    vx, vy, vel_source = load_velocity_on_grid(stack)
    vx, vy, _vsm_km = _maybe_smooth_velocity(vx, vy)
    if _vsm_km:
        vel_source += f" + {_vsm_km:.1f}km Gaussian (Shean-style)"
        print(f"  velocity smoothed {_vsm_km:.1f} km")
    print(
        f"  vx range: {float(vx.min()):.1f} .. {float(vx.max()):.1f} m/yr  "
        f"vy range: {float(vy.min()):.1f} .. {float(vy.max()):.1f} m/yr"
    )

    print("Loading SMB (RACMO2.4p1)...")
    a_dot = load_smb_on_grid(stack)
    print(
        f"  a_dot range: {float(a_dot.min()):.3f} .. {float(a_dot.max()):.3f} m ice/yr  "
        f"(window-mean)"
    )

    print("Loading firn air content (BedMachine, static)...")
    firn = load_firn_on_grid(stack, config.BEDMACHINE_NC)
    firn_finite = firn.values[np.isfinite(firn.values)]
    print(
        f"  firn (m): median={float(np.median(firn_finite)):.2f}  "
        f"IQR=[{float(np.percentile(firn_finite, 25)):.2f}, "
        f"{float(np.percentile(firn_finite, 75)):.2f}]  "
        f"finite frac={float(np.isfinite(firn.values).mean()):.3f}"
    )

    melt_inputs_png = config.FIGURES_DIR / f"melt_inputs{out_suffix}{win_tag}.png"
    plot_inputs(stack, vx, vy, a_dot, firn, melt_inputs_png)
    print(f"  wrote {melt_inputs_png}")

    print("Running Eulerian solver...")
    # See nansen.run_melt for rationale. robust_dh_dt=True enables Tukey-
    # biweight IRLS so per-strip residual stripes don't get amplified by
    # the ×9.42 hydrostatic gain into spurious dh/dt.
    euler = eulerian_melt_rate(stack, vx, vy, a_dot=a_dot, d=firn, robust_dh_dt=True)
    print(f"  melt_rate: median={float(euler.melt_rate.median()):.2f}  "
          f"IQR=[{float(euler.melt_rate.quantile(0.25)):.2f}, "
          f"{float(euler.melt_rate.quantile(0.75)):.2f}] m ice/yr")

    print("Running Lagrangian solver (path integration; narrates progress + ETA)...")
    # pairs="all" banded to Shean 2019's 1.5-2.5 yr dt window (2026-06-15: floor
    # raised from 2 months to 1.5 yr for Shean consistency). The floor caps
    # dh/dt noise (= coreg_error / dt): a 2-month floor left pre-IS2 pairs
    # (~0.8 m CryoTEMPO coreg) at ~5 m/yr noise, ~15x Shean's budget and a likely
    # driver of the wide whole-record Lagrangian IQR. The 2.5-yr cap is the upper
    # edge of the same window (without it, >2.5-yr baselines advect PIG's
    # ~4 km/yr flow tens of km off-grid into bogus trajectories).
    lagr = lagrangian_melt_rate(
        stack,
        vx,
        vy,
        a_dot=a_dot,
        d=firn,
        dt_yr=0.05,
        # Shean's primary path-distributed product (stack_melt_path_lsq.py
        # _meltrate.tif): scatter each step's bdot into the cell visited at that
        # step. seed_stride=1 (dense per-pixel seeding, Shean L304) is what kills
        # the grid-scale checkerboard -- it was the STRIDE, not path-vs-origin --
        # and the per-start restructure makes dense seeding cheap (~1 min vs the
        # old ~6 h). Reproduces the trusted per-pair path field (median -10.29 vs
        # -10.24; cross-checked 2026-06-29). output="origin" stays available in
        # the library as Shean's init_dhdt variant (smoother, path-averaged).
        seed_stride=1,
        output="path",
        # Shean's two-level mosaic (mos_month_year.sh / dem_mosaic --median):
        # mean within each pair -> median across pairs. Robust to per-pair /
        # per-strip outliers (removes the spurious red accretion blobs the pooled
        # mean carried through); memory-bounded and faithful to his published
        # product. Validated 2026-06-29 (path field visibly cleaner, -7.21 vs
        # -8.26 mean). The remaining along-flow streaks are coregistration, not
        # aggregation.
        aggregator="pair_median",
        pairs="all",
        min_dt_yr=1.5,
        max_dt_yr=2.5,
        # Opt-in input hygiene (literature/plan_lagrangian_parcel_lsq.md §3b):
        # mask bogus ∇·u before it multiplies H (Shean L262 uses ±0.2 /yr).
        # Velocity smoothing is the driver-level PIG_VEL_SMOOTH_KM. Both default
        # off → bit-identical to the production baseline.
        vdiv_clip=(float(os.environ.get("PIG_VDIV_CLIP", "0") or 0) or None),
    )
    mr = lagr.melt_rate
    print(f"  melt_rate: median={float(mr.median()):.2f}  "
          f"IQR=[{float(mr.quantile(0.25)):.2f}, {float(mr.quantile(0.75)):.2f}] m ice/yr  "
          f"finite-cell-count={int((mr.notnull()).sum())}")

    # Strain-exact per-parcel LSQ (literature/plan_lagrangian_parcel_lsq.md):
    # one trajectory per seed across the window; regress y = H·s − ∫ȧ s dt on
    # τ = ∫s dt so the slope is ḃ with strain handled multiplicatively (exact),
    # Tukey IRLS across every epoch the parcel crosses, origin-attributed.
    # Opt-in (PIG_PARCEL_LSQ=1) while it is A/B'd against the pair estimator.
    parcel = None
    if os.environ.get("PIG_PARCEL_LSQ", "0").strip() == "1":
        print("Running Lagrangian parcel-LSQ solver (strain-exact; narrates)...")
        _psm = float(os.environ.get("PIG_PARCEL_SMOOTH_M", "3000") or 0)
        _pcl = float(os.environ.get("PIG_PARCEL_VDIV_CLIP", "0.2") or 0)
        parcel = lagrangian_parcel_lsq_melt_rate(
            stack,
            vx,
            vy,
            a_dot=a_dot,
            d=firn,
            dt_yr=float(os.environ.get("PIG_PARCEL_DT_YR", "0.05")),
            vel_smooth_sigma_m=_psm if _psm > 0 else None,
            vdiv_clip=_pcl if _pcl > 0 else None,
            min_epochs=int(os.environ.get("PIG_PARCEL_MIN_EPOCHS", "4")),
            min_span_yr=float(os.environ.get("PIG_PARCEL_MIN_SPAN_YR", "1.5")),
        )
        pmr = parcel.melt_rate
        print(
            f"  melt_rate: median={float(pmr.median()):.2f}  "
            f"IQR=[{float(pmr.quantile(0.25)):.2f}, {float(pmr.quantile(0.75)):.2f}] "
            f"m ice/yr  finite-cell-count={int(pmr.notnull().sum())}"
        )

    # Budget-corrected pair-banded linear inverse (literature/
    # plan_match_linear_inverse.md): the rebuilt linear-inverse framework that
    # matches the path solver by construction (same Shean pair band + two-level
    # median, per-pixel H_f·div(u)/SMB/firn correction, hydrostatic estimate +
    # kernel channel correction). Gate: tests/gate_match_lagrangian.py.
    # Opt-in (PIG_LININV_BUDGET=1); the path solver stays the headline product.
    lininv_budget = None
    if os.environ.get("PIG_LININV_BUDGET", "0").strip() == "1":
        from stereo_melt.dynamics import linear_inverse_budget_melt_rate

        print("Running budget-corrected pair-banded linear inverse (narrates)...")
        _lb_sigma = os.environ.get("PIG_LININV_CORR_SIGMA_M", "").strip()
        lininv_budget = linear_inverse_budget_melt_rate(
            stack, vx, vy, a_dot=a_dot, d=firn, floating_mask=floating,
            eta_bar=float(os.environ.get("PIG_LININV_ETA_BAR", "1e14")),
            reg=float(os.environ.get("PIG_LININV_REG", "0.1")),
            transform="dct",
            min_pair_dt_yr=1.5, max_pair_dt_yr=2.5, dt_yr=0.05,
            corr_prefilter_sigma_m=float(_lb_sigma) if _lb_sigma else None,
            progress=True,
        )
        lbmr = lininv_budget.melt_rate
        print(
            f"  melt_rate: median={float(lbmr.median()):.2f}  "
            f"IQR=[{float(lbmr.quantile(0.25)):.2f}, {float(lbmr.quantile(0.75)):.2f}] "
            f"m ice/yr  finite-cell-count={int(lbmr.notnull().sum())}"
        )

    # Third solver: faithful Stubblefield 2023 non-hydrostatic linear inverse --
    # the DIRECT Fourier inversion of the forward kernel (agstub/linear-shelf-melt),
    # NOT the old broken masked-CG / Lagrangian-frame variant. Recovers the
    # channel-scale melt the hydrostatic Eul/Lagr under-sharpen; a non-hydrostatic
    # correction layer, not a mass-budget melt. eta_bar=1e13 for warm/fast PIG
    # (amplitude ~1/eta_bar; pattern independent). `linv` carries it so the
    # existing 3rd-panel plot / save plumbing (guarded on `if linv is not None`)
    # picks it up.
    print("Running Stubblefield non-hydrostatic linear inverse (3rd solver)...")
    try:
        linv = stubblefield_inverse_melt_rate(
            stack, vx, vy, floating_mask=floating, d=firn,
            eta_bar=1e13, sigma_hp_H=5.0, tik=1e-2,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"  Stubblefield inverse FAILED: {exc}")
        linv = None

    if linv is not None:
        linv_mr = linv.melt_rate
        print(
            f"  H_ref={linv.attrs['H_ref_m']:.1f} m  "
            f"gamma_dimless={linv.attrs['gamma_dimless']:.3e}  "
            f"t_r={linv.attrs['tr_yr']:.1f} yr"
        )
        print(
            f"  melt_rate: median={float(linv_mr.median()):.2f}  "
            f"IQR=[{float(linv_mr.quantile(0.25)):.2f}, "
            f"{float(linv_mr.quantile(0.75)):.2f}] m ice/yr"
        )

    # Fourth solver: the variational forward-fit inverse (opt-in PIG_VARIATIONAL=1).
    # Same Stubblefield transfer as `linv` above, but placed in the FORWARD model
    # and FITTED rather than divided out. On the Elmer/Ice twins the direct
    # division over-lifts across-flow structure -- it double-counts the advective
    # dynamics the surface already carries (E2a cosy 1.07 without, 1.54 with) --
    # while the forward fit recovers along-flow, oblique and across-flow melt
    # through one operator with no angular weight. eta_bar defaults to the same
    # 1e13 `linv` uses above, so the A/B against the third panel isolates the
    # METHOD, not the viscosity. Like `linv` it is DC-blind: a channel-scale
    # pattern correction, NOT a mass-budget melt, and not interchangeable with
    # the Eulerian/Lagrangian products.
    varfit = None
    if os.environ.get("PIG_VARIATIONAL", "0").strip() == "1":
        from stereo_melt.dynamics.stubblefield_forward import variational_melt_rate

        print("Running variational forward-fit inverse (4th solver; narrates)...")
        # PIG_VAR_BG_DEGREE selects HOW the reference state is removed, which is
        # a choice about the INPUT, independent of the forward fit itself:
        #   unset (default) -> legacy Gaussian high-pass of the input at
        #                      sigma_hp_H (a band CUT: melt beyond ~13 km at
        #                      PIG's H is deleted from the target and cannot be
        #                      recovered).
        #   0/1/2           -> feed the RAW surface and fit a polynomial
        #                      background inside the model, projected out of the
        #                      residual each step. No wavelength band is
        #                      discarded; sigma_hp_H is then ignored.
        # The operator stays DC-blind either way (its k=0 bin is pinned to
        # zero in stubblefield_forward_multiplier, independently of the
        # filter), but with bg_degree the unconstrained long wavelengths are
        # left to the background/prior instead of deleted.
        _bg = os.environ.get("PIG_VAR_BG_DEGREE", "").strip()
        varfit = variational_melt_rate(
            stack, vx, vy, floating_mask=floating, d=firn,
            rep=os.environ.get("PIG_VAR_REP", "grid"),
            eta_bar=float(os.environ.get("PIG_VAR_ETA_BAR", "1e13")),
            alpha_scale=float(os.environ.get("PIG_VAR_ALPHA_SCALE", "0.34")),
            lam=float(os.environ.get("PIG_VAR_LAM", "1e-4")),
            iters=int(os.environ.get("PIG_VAR_ITERS", "4000")),
            lr=float(os.environ.get("PIG_VAR_LR", "3e-3")),
            sigma_hp_H=float(os.environ.get("PIG_VAR_SIGMA_HP_H", "5.0")),
            bg_degree=int(_bg) if _bg else None,
            log_every=int(os.environ.get("PIG_VAR_LOG_EVERY", "250")),
            # PIG spans ~300-4000 m/yr, so a single mean u is wrong nearly
            # everywhere; cluster the shelf into geometry bins instead. Tiling
            # the melt field is NOT an option here -- the operator's downstream
            # footprint reaches ~75 km at 2000 m/yr and ~150 km at 4000 m/yr, so
            # any tile small enough to localize the flow truncates the response.
            n_bins=int(os.environ.get("PIG_VAR_N_BINS", "8")),
            blend_km=float(os.environ.get("PIG_VAR_BLEND_KM", "4.0")),
        )
        vmr = varfit.melt_rate
        print(
            f"  H_ref={varfit.attrs['H_ref_m']:.1f} m  "
            f"t_r={varfit.attrs['t_r_yr']:.2f} yr  "
            f"u0=({varfit.attrs['u0x_myr']:.0f}, {varfit.attrs['u0y_myr']:.0f}) m/yr  "
            f"n_bins={varfit.attrs['n_bins']}  "
            f"n_fit={varfit.attrs['fit_n_cells']:,} cells"
        )
        # No melt truth on a real shelf: how much of the observed high-passed
        # surface the operator can reproduce is the only self-diagnostic.
        print(
            f"  surface fit: var_explained={varfit.attrs['fit_var_explained']:.3f}  "
            f"rms_resid={varfit.attrs['fit_rms_resid_m']:.3f} m of "
            f"{varfit.attrs['fit_rms_obs_m']:.3f} m observed"
        )
        print(
            f"  melt_rate: median={float(vmr.median()):.2f}  "
            f"IQR=[{float(vmr.quantile(0.25)):.2f}, "
            f"{float(vmr.quantile(0.75)):.2f}] m ice/yr"
        )

    euler_melt = euler.melt_rate.where(floating)
    lagr_melt = lagr.melt_rate.where(floating)
    linv_melt = linv.melt_rate if linv is not None else None
    print(
        f"  floating-only Eulerian median={float(euler_melt.median()):.2f}  "
        f"Lagrangian median={float(lagr_melt.median()):.2f}"
        + (f"  Stubblefield median={float(linv_melt.median()):.2f}" if linv is not None else "")
        + " m ice/yr"
    )

    out_nc = (
        config.RESULTS_DIR
        / f"pig_melt{out_suffix}_{config.START_TIME}_{config.END_TIME}.nc"
    )
    print(f"Saving results -> {out_nc}")
    ds_vars = {
        "melt_rate_eulerian": euler_melt,
        "melt_rate_lagrangian": lagr_melt,
        "dHdt": euler.dHdt.where(floating),
        "flux_div": euler.flux_div.where(floating),
        "H_f_mean": euler.H_f_mean.where(floating),
        "a_dot": a_dot,
        "floating_mask": floating,
        "lagrangian_count": lagr["count"].where(floating),
        "lagrangian_rmse": lagr["rmse"].where(floating),
        "eulerian_count": euler["count"].where(floating),
        "eulerian_rmse": euler["rmse"].where(floating),
    }
    ds_attrs = {
        "shelf": config.SHELF,
        "window_start": config.START_TIME,
        "window_end": config.END_TIME,
        "stack_file_window": f"{file_start} to {file_end}",
        "grid_res_m": float(config.RES),
        "velocity_source": vel_source,
        "min_extent_mask": min_extent_src,
        "smb_source": "RACMO2.4p1 smbgl (Zenodo 19255213), window-integrated",
    }
    if linv is not None:
        ds_vars["melt_rate_linear_inverse"] = linv_melt
        ds_attrs["linear_inverse_H_ref_m"] = linv.attrs["H_ref_m"]
        ds_attrs["linear_inverse_gamma_dimless"] = linv.attrs["gamma_dimless"]
        ds_attrs["linear_inverse_tr_yr"] = linv.attrs["tr_yr"]
    if parcel is not None:
        ds_vars["melt_rate_parcel_lsq"] = parcel.melt_rate.where(floating)
        ds_vars["parcel_lsq_stderr"] = parcel.stderr.where(floating)
        ds_vars["parcel_lsq_rmse"] = parcel.rmse.where(floating)
        ds_vars["parcel_lsq_count"] = parcel["count"].where(floating)
        ds_vars["parcel_lsq_span_yr"] = parcel.span_yr.where(floating)
        ds_attrs["parcel_lsq_params"] = (
            f"smooth={parcel.attrs['vel_smooth_sigma_m']:.0f}m "
            f"vdiv_clip={parcel.attrs['vdiv_clip']:.2f}/yr "
            f"dt={parcel.attrs['dt_yr']}yr "
            f"min_epochs={parcel.attrs['min_epochs']} "
            f"min_span={parcel.attrs['min_span_yr']}yr"
        )
    if lininv_budget is not None:
        ds_vars["melt_rate_lininv_budget"] = lininv_budget.melt_rate.where(floating)
        ds_vars["melt_rate_lininv_hydro"] = lininv_budget.melt_rate_hydro.where(floating)
        ds_vars["lininv_nonhydro_corr"] = lininv_budget.nonhydro_corr.where(floating)
        ds_vars["lininv_count"] = lininv_budget["count"].where(floating)
        ds_vars["lininv_rmse"] = lininv_budget.rmse.where(floating)
        ds_attrs["lininv_budget_params"] = (
            f"eta_bar={lininv_budget.attrs['eta_bar']:.0e} "
            f"reg={lininv_budget.attrs['reg']:g} "
            f"corr_sigma={lininv_budget.attrs['corr_prefilter_sigma_m']:.0f}m "
            f"band={lininv_budget.attrs['min_pair_dt_yr']}-"
            f"{lininv_budget.attrs['max_pair_dt_yr']}yr "
            f"n_pairs={lininv_budget.attrs['n_pairs']}"
        )
    if varfit is not None:
        ds_vars["melt_rate_variational"] = varfit.melt_rate.where(floating)
        ds_vars["variational_dzs_obs"] = varfit.dzs_obs
        ds_vars["variational_dzs_fit"] = varfit.dzs_fit
        for _k in ("H_ref_m", "t_r_yr", "u0x_myr", "u0y_myr", "eta_bar",
                   "alpha_scale", "lam", "iters", "rep", "sigma_hp_H",
                   "n_bins", "blend_km",
                   "fit_var_explained", "fit_rms_resid_m", "fit_rms_obs_m",
                   "fit_n_cells"):
            ds_attrs[f"variational_{_k}"] = varfit.attrs[_k]
    # Shean-style integrated-flux bracket (diagnostic; does NOT alter the saved
    # field). Integrate over the floating shelf eroded PIG_GL_BUFFER_KM back from
    # grounded ice, with a Lagrangian coverage+rmse quality gate, and bracket the
    # heavy-tailed integral robust(median x area)/clip|250|/raw. Mirrors the
    # treatment that reproduces Shean's 82-93 Gt/yr from his DEMs; the standalone
    # tool is pig.integrate_melt_flux.
    res_m = abs(float(stack["x"].values[1] - stack["x"].values[0]))
    gl_km = float(os.environ.get("PIG_GL_BUFFER_KM", "2.0"))
    # The old rmse<=25 m gate discarded ~half the shelf (median Lagrangian rmse
    # ~28 m: 5888 -> 2699 km2) and the gt_robust=median*area LOWER BOUND was being
    # read as the flux (6 Gt/yr), badly understating melt. rmse<=50 m keeps most of
    # the shelf and the |melt|<250 clip tames the residual tail; the CLIP integral
    # is the headline (full-shelf clip ~113 Gt/yr, rmse<=50 ~75 Gt/yr; Shean 82-93).
    # Default: NO rmse gate (0). The |melt|<250 clip already caps the heavy tail,
    # and the GL buffer removes the grounding zone, so over the full shelf the
    # clip integral matches Shean (PIG: ~80-110 Gt/yr; Shean 82-93). The old
    # rmse<=25 m gate discarded half the shelf and collapsed it to ~6. Set
    # PIG_FLUX_RMSE_MAX>0 for a conservative quality-gated variant.
    rmse_max = float(os.environ.get("PIG_FLUX_RMSE_MAX", "0"))
    try:
        grounded = load_grounded_mask(stack)
        dom = grounding_buffer(floating, grounded, gl_km * 1000.0, res_m)
        qual = np.nan_to_num(lagr["count"].values, nan=0) >= 10
        if rmse_max > 0:
            qual = qual & (np.nan_to_num(lagr["rmse"].values, nan=1e9) <= rmse_max)
        gate_txt = f"rmse<={rmse_max:.0f} m" if rmse_max > 0 else "no rmse gate (clip caps tail)"
        print(f"Integrated basal flux (GL buffer {gl_km:.1f} km, count>=10, {gate_txt}; HEADLINE = clip):")
        for name, mr in (("Eulerian", euler_melt), ("Lagrangian", lagr_melt)):
            b = integrate_basal_flux(mr, dom, res_m, quality=qual)
            if b.get("n"):
                print(
                    f"  {name:10s} area={b['area_km2']:.0f} km2  med={b['median_myr']:+.1f}  "
                    f"Gt/yr: CLIP={b['gt_clip']:.1f}  raw={b['gt_raw']:.1f}  "
                    f"robust/lower={b['gt_robust']:.1f}  deepest={b['deepest_myr']:+.0f}"
                )
                ds_attrs[f"flux_{name.lower()}_gt"] = round(b["gt_clip"], 2)  # headline
                ds_attrs[f"flux_{name.lower()}_gt_clip"] = round(b["gt_clip"], 2)
                ds_attrs[f"flux_{name.lower()}_gt_robust"] = round(b["gt_robust"], 2)
        if parcel is not None:
            pmelt = parcel.melt_rate.where(floating)
            b = integrate_basal_flux(pmelt, dom, res_m)
            if b.get("n"):
                print(
                    f"  ParcelLSQ  area={b['area_km2']:.0f} km2  med={b['median_myr']:+.1f}  "
                    f"Gt/yr: CLIP={b['gt_clip']:.1f}  raw={b['gt_raw']:.1f}  "
                    f"robust/lower={b['gt_robust']:.1f}  deepest={b['deepest_myr']:+.0f}"
                )
                ds_attrs["flux_parcel_lsq_gt_clip"] = round(b["gt_clip"], 2)
                ds_attrs["flux_parcel_lsq_gt_robust"] = round(b["gt_robust"], 2)
            # The parcel product carries its own principled gate: the slope
            # standard error. Report a couple of gate levels for calibration.
            pse = parcel.stderr.values
            for gate in (2.0, 5.0):
                q = np.isfinite(pse) & (pse <= gate)
                bq = integrate_basal_flux(pmelt, dom, res_m, quality=q)
                if bq.get("n"):
                    print(
                        f"    stderr<={gate:.0f} m/yr: area={bq['area_km2']:.0f} km2  "
                        f"med={bq['median_myr']:+.1f}  Gt/yr: CLIP={bq['gt_clip']:.1f}  "
                        f"raw={bq['gt_raw']:.1f}  robust={bq['gt_robust']:.1f}"
                    )
        # Accretion prevalence over the integration domain — the plausibility
        # check that triggered the parcel-LSQ reimplementation: a shelf melting
        # ~100 Gt/yr should not read as ~1/3 accretion cells.
        acc_fields = [("Eulerian", euler_melt), ("Lagrangian", lagr_melt)]
        if parcel is not None:
            acc_fields.append(("ParcelLSQ", parcel.melt_rate.where(floating)))
        for name, amr in acc_fields:
            av = amr.values[dom & np.isfinite(amr.values)]
            if av.size:
                print(
                    f"  {name:10s} accretion cells over domain: "
                    f"{100.0 * float((av > 0).mean()):.1f}%  "
                    f"(>+2 m/yr: {100.0 * float((av > 2).mean()):.1f}%)  n={av.size:,}"
                )
        ds_attrs["flux_gl_buffer_km"] = gl_km
        ds_attrs["flux_quality_gate"] = f"lagrangian count>=10 & rmse<={rmse_max:.0f} m; headline=clip|250|"
        print("  (Shean 2019: 82-93 Gt/yr over the PIG shelf; clip caps |melt|<250 m/yr)")
    except Exception as exc:  # diagnostics must never block the save
        print(f"  flux integration skipped: {exc}")

    ds_out = xr.Dataset(ds_vars, attrs=ds_attrs)
    ds_out.to_netcdf(out_nc)

    melt_comparison_png = config.FIGURES_DIR / f"melt_comparison{out_suffix}{win_tag}.png"
    plot_melt_comparison(euler, lagr, linv, melt_comparison_png)
    print(f"  wrote {melt_comparison_png}")

    if varfit is not None:
        plot_variational_fit(
            varfit,
            config.FIGURES_DIR / f"melt_variational_fit{out_suffix}{win_tag}.png",
            title=(f"Pine Island forward-fit melt inverse — "
                   f"{config.START_TIME} to {config.END_TIME}"),
        )

    if parcel is not None:
        parcel_png = config.FIGURES_DIR / f"melt_parcel_lsq{out_suffix}{win_tag}.png"
        pfloat = parcel.melt_rate.where(floating)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
        pcmap, pnorm = melt_cmap(), melt_norm(vmax=60.0)
        im = _imshow_xr(axes[0, 0], pfloat, cmap=pcmap, norm=pnorm)
        axes[0, 0].set_title("parcel-LSQ melt_rate (m ice/yr)")
        add_melt_colorbar(fig, im, ax=axes[0, 0], fraction=0.045)
        im = _imshow_xr(axes[0, 1], lagr.melt_rate, cmap=pcmap, norm=pnorm)
        axes[0, 1].set_title("endpoint-pair Lagrangian (reference)")
        add_melt_colorbar(fig, im, ax=axes[0, 1], fraction=0.045)
        im = _imshow_xr(axes[0, 2], pfloat - lagr.melt_rate, cmap="PuOr", vmin=-10, vmax=10)
        axes[0, 2].set_title("parcel − pair (m ice/yr)")
        fig.colorbar(im, ax=axes[0, 2], fraction=0.045)
        im = _imshow_xr(axes[1, 0], parcel.stderr.where(floating), cmap="magma", vmin=0, vmax=5)
        axes[1, 0].set_title("parcel slope stderr (m ice/yr)")
        fig.colorbar(im, ax=axes[1, 0], fraction=0.045)
        im = _imshow_xr(axes[1, 1], parcel["count"].where(floating), cmap="viridis", vmin=0, vmax=60)
        axes[1, 1].set_title("epochs used per parcel")
        fig.colorbar(im, ax=axes[1, 1], fraction=0.045)
        im = _imshow_xr(axes[1, 2], parcel.span_yr.where(floating), cmap="viridis")
        axes[1, 2].set_title("observation span (yr)")
        fig.colorbar(im, ax=axes[1, 2], fraction=0.045)
        for ax in axes.ravel():
            ax.set_xlabel("x (m)")
        axes[0, 0].set_ylabel("y (m)")
        axes[1, 0].set_ylabel("y (m)")
        fig.suptitle(
            f"Pine Island parcel-LSQ vs endpoint-pair — "
            f"{config.START_TIME} to {config.END_TIME}",
            fontsize=12,
        )
        fig.savefig(parcel_png, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {parcel_png}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--res",
        type=float,
        default=None,
        help=(
            "Resolution variant to load (meters). When set, looks up "
            "pig_stack_<N>m*_<start>_<end>.nc built by `pig.build_stack --res <N>` "
            "+ `pig.tilt_fit --res <N>`. Outputs land at *<N>m* paths so 25 m "
            "production isn't clobbered."
        ),
    )
    parser.add_argument(
        "--start",
        default=None,
        help=(
            "Analysis sub-window start (ISO date, e.g. 2018-10-01). Subsets "
            "the loaded full-window stack in time; does NOT change which stack "
            "file is loaded. Use to run the same fused stack over a sub-window "
            "(e.g. IS2-era-only) as a workflow-soundness control."
        ),
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Analysis sub-window end (ISO date). Defaults to the stack file END_TIME.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help=(
            "Experiment tag matching the `pig.build_stack --tag` build: "
            "loads pig_stack_<N>m_<tag>_tilt_corrected_*.nc and writes "
            "pig_melt_<N>m_<tag>_*.nc."
        ),
    )
    args = parser.parse_args()
    main(res_override=args.res, start=args.start, end=args.end, tag=args.tag)

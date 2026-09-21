# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

"""Per-epoch QC scoring of a tilt-corrected stack.

Shared backend for ``<basin>/scripts/find_bad_epochs.py``. The signature
of an actually-bad epoch is one where, after the tilt-fit, the
static-control residual is still far from zero — i.e. the LSQ couldn't
lock that epoch. We diagnose this from three signals:

- per-epoch median ``(z_tc - z_ref)`` over static-control pixels
- ``weight_mean`` from tilt_params (Tukey-biweight average; lower means
  more obs were down-weighted; ``NaN`` means the IRLS step itself
  failed)
- ``weight_frac_kept`` (fraction of obs kept above the biweight floor)
- ``frac_static`` (per-epoch valid pixels in the static-control region,
  divided by the total static-control population)

Following Shean 2019, :func:`suggest_bad_epochs` rejects DEMs by
**coregistration quality** -- the pc_align post-alignment residual
``end_p50`` (coverage-independent) -- backed by a catastrophic-blunder
backstop and an IRLS-failure net. The legacy coverage-coupled gate
(``|resid|`` AND sparse-anchor / IRLS-rejected) is retained only as an
opt-in (``resid_threshold_m``): on thin-static-control basins (PIG at
~4% static) ``frac_static`` is below threshold for essentially every
epoch, so that rule degenerates to a flat residual drop that discards
cleanly-coregistered shelf DEMs -- flagging shelf strips for having
noisy *rock-margin* pixels, not bad shelf data.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xarray as xr


__all__ = [
    "score_tilt_residuals",
    "screen_unrescued_epochs",
    "suggest_bad_epochs",
    "print_bad_epoch_report",
    "plot_static_mask_and_strip_count",
]


def _align_tilt_params(
    tc_stack: xr.DataArray, tilt_params: xr.Dataset
) -> "tuple[xr.Dataset, np.ndarray]":
    """Rows of ``tilt_params`` matching the slices of ``tc_stack``, by date.

    Handles params written before later epoch drops and duplicate dates.
    """
    if len(tilt_params["time"]) == len(tc_stack["time"]):
        return tilt_params, np.arange(len(tilt_params["time"]))
    tc_dates = set(pd.to_datetime(tc_stack["time"].values).normalize())
    params_dt = pd.to_datetime(tilt_params["time"].values).normalize()
    keep_idx = np.where(np.array([d in tc_dates for d in params_dt]))[0]
    aligned = tilt_params.isel(time=keep_idx)
    if len(aligned["time"]) != len(tc_stack["time"]):
        raise ValueError(
            f"date-set alignment failed: tc has {len(tc_stack['time'])} "
            f"epochs, aligned params has {len(aligned['time'])}. "
            "Re-run tilt_fit so its params match the current BAD_EPOCHS."
        )
    return aligned, keep_idx


def _fit_temporal_model(
    tilt_params: xr.Dataset, keep_idx: np.ndarray
) -> "tuple[np.ndarray, np.ndarray, np.ndarray] | None":
    """The fit's per-pixel ``intercept``, ``dhdt`` (m/day) and centred times (days).

    Times are centred on the params' full epoch axis, then subset to
    ``keep_idx``. Returns ``None`` if the params lack the temporal model.
    """
    if not ("intercept" in tilt_params and "dhdt" in tilt_params):
        return None
    p_times = tilt_params["time"].values
    p_days = (
        (p_times - p_times[0]).astype("timedelta64[s]").astype(float) / 86400.0
    )
    t_centered = (p_days - p_days.mean())[keep_idx]
    return tilt_params["intercept"].values, tilt_params["dhdt"].values, t_centered


def score_tilt_residuals(
    tc_stack: xr.DataArray,
    tilt_params: xr.Dataset,
    static_mask: np.ndarray,
    *,
    alignment_quality: "pd.DataFrame | None" = None,
) -> pd.DataFrame:
    """Per-epoch QC scores from a tilt-corrected stack and its tilt_params.

    Parameters
    ----------
    tc_stack : xarray.DataArray, dims ``(time, y, x)``
        Tilt-corrected DEM stack (output of ``fit_tilt_stack``).
    tilt_params : xarray.Dataset
        ``tilt_params`` companion file with ``weight_mean`` and
        ``weight_frac_kept`` along ``time``.
    static_mask : numpy.ndarray, dims ``(y, x)``
        Boolean mask marking the static-control region (rock + slow
        grounded ice). Same convention used by ``fit_tilt_stack``.

    Returns
    -------
    pandas.DataFrame
        Rows per epoch with columns ``epoch``, ``n_static_obs``,
        ``med_resid_m``, ``mad_resid_m``, ``weight_mean``,
        ``weight_frac_kept``, ``frac_static``, sorted by
        ``|med_resid_m|`` descending.

    Notes
    -----
    When ``tilt_params`` carries the per-pixel ``intercept`` and
    ``dhdt`` fields, residuals are scored against the LSQ's own
    per-pixel temporal model ``intercept + dhdt * t_centered`` (the
    saved intercept is in absolute elevation — ``fit_tilt_stack`` adds
    its reference back before writing). On basins with real
    grounded-ice thinning over the stack window (PIG: ~1.5 m/yr on the
    trunk margins, -15 m over 2011-2020) the static median misreads
    that change as epoch badness for temporally-extreme, sparsely
    covered epochs — flagging the *victim* layer whose neighbors moved.
    The model-frame residual is what the IRLS actually fought
    (corrected stack minus the per-pixel temporal model is exactly the
    fit residual), so it isolates genuinely unlocked epochs.
    """
    static = np.asarray(static_mask, dtype=bool)
    times = pd.to_datetime(tc_stack["time"].values)
    # Per-slice SETSM strip id, when the stack carries it (build_stack since
    # 2026-06-20). Enables an exact per-strip join to pc_align quality below,
    # so the screen flags the single bad strip on a mixed day rather than the
    # whole acquisition date.
    has_dem_id = "dem_id" in tc_stack.coords
    slice_dem_ids = (
        np.asarray(tc_stack["dem_id"].values, dtype=str) if has_dem_id else None
    )
    z = tc_stack.values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        z_ref = np.nanmedian(z, axis=0)

    aligned, keep_idx = _align_tilt_params(tc_stack, tilt_params)
    wm_arr = aligned["weight_mean"].values
    wf_arr = aligned["weight_frac_kept"].values

    temporal = _fit_temporal_model(tilt_params, keep_idx)
    detrended = temporal is not None
    if detrended:
        intercept2d, dhdt2d, t_centered = temporal
        print(
            "  scoring against per-pixel intercept + dhdt model "
            "(dhdt-aware screen)"
        )
    else:
        print(
            "  tilt_params lacks intercept/dhdt -- scoring against "
            "time-static median (legacy screen)"
        )

    rows = []
    for k in range(z.shape[0]):
        zk = z[k]
        if detrended:
            model = intercept2d + dhdt2d * t_centered[k]
        else:
            model = z_ref
        finite = np.isfinite(zk) & static & np.isfinite(model)
        n = int(finite.sum())
        if n == 0:
            rows.append((times[k], n, np.nan, np.nan,
                         float(wm_arr[k]), float(wf_arr[k])))
            continue
        residual = zk[finite] - model[finite]
        med = float(np.nanmedian(residual))
        mad = float(np.nanmedian(np.abs(residual - med)))
        rows.append((times[k], n, med, 1.4826 * mad,
                     float(wm_arr[k]), float(wf_arr[k])))

    df = pd.DataFrame(
        rows,
        columns=["epoch", "n_static_obs", "med_resid_m", "mad_resid_m",
                 "weight_mean", "weight_frac_kept"],
    )
    if has_dem_id:
        # rows are appended in slice (k) order, matching slice_dem_ids.
        df.insert(1, "dem_id", slice_dem_ids)
    df["frac_static"] = df["n_static_obs"] / int(static.sum())

    # Optional: join per-strip pc_align quality (end_p50 etc.). Use
    # ``aggregate_basin_quality(config.STRIP_SOURCES)`` to build the
    # input DataFrame. If multiple strips share a date (rare; multi-
    # segment captures), we take the worst end_p50 as a conservative
    # per-epoch summary.
    if alignment_quality is not None and not alignment_quality.empty:
        aq = alignment_quality.copy()
        qcols = [c for c in ("end_p16", "end_p50", "end_p84", "beg_p50")
                 if c in aq.columns]
        if has_dem_id and "dem_id" in aq.columns:
            # Exact per-strip join: each stack slice gets ITS strip's
            # pc_align quality, not the worst strip sharing its day. This is
            # what lets suggest_bad_epochs flag the one bad strip on a mixed
            # day instead of the whole date (Shean 2019 per-DEM rejection).
            aq_strip = aq[["dem_id", *qcols]].drop_duplicates("dem_id")
            df = df.merge(aq_strip, on="dem_id", how="left")
        else:
            # Legacy date-only stacks (no dem_id coord): collapse to the
            # worst strip per acquisition day as a conservative per-epoch
            # summary. Over-drops clean same-day siblings -- the behaviour
            # the dem_id path above fixes.
            aq["epoch_norm"] = pd.to_datetime(aq["date"]).dt.normalize()
            per_epoch_aq = (
                aq.groupby("epoch_norm")
                .agg(**{c: (c, "max") for c in qcols})
                .reset_index()
            )
            df["epoch_norm"] = pd.to_datetime(df["epoch"]).dt.normalize()
            df = df.merge(per_epoch_aq, on="epoch_norm", how="left")
            df = df.drop(columns=["epoch_norm"])

    return df.sort_values(
        "med_resid_m", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)


def screen_unrescued_epochs(
    tc_stack: xr.DataArray,
    tilt_params: xr.Dataset,
    *,
    domain_mask: "np.ndarray | None" = None,
    screen_variants: "tuple[str, ...]" = ("nocorr",),
    nmad_max_m: float = 5.0,
    blunder_m: float = 20.0,
    blunder_frac_max: float = 0.10,
    min_px: int = 200,
) -> pd.DataFrame:
    """Flag uncontrolled slices the joint tilt fit left unadjusted.

    Slices whose ``source_variant`` is in ``screen_variants`` are scored
    against the fit's per-pixel ``intercept + dhdt * t`` model over
    ``domain_mask`` and flagged if they have fewer than ``min_px`` pixels,
    residual NMAD above ``nmad_max_m``, or more than ``blunder_frac_max``
    of pixels off by more than ``blunder_m``. Other slices are scored but
    never flagged.

    Parameters
    ----------
    tc_stack : xarray.DataArray, dims ``(time, y, x)``
        Tilt-corrected stack with a per-slice ``source_variant`` coord.
    tilt_params : xarray.Dataset
        Companion ``tilt_params``; without ``intercept``/``dhdt`` the
        reference falls back to the temporal median.
    domain_mask : numpy.ndarray of bool, dims ``(y, x)``, optional
        Pixels to score (default: where the fitted model is finite).
    screen_variants : tuple of str
        ``source_variant`` values eligible to be flagged.
    nmad_max_m, blunder_m, blunder_frac_max, min_px
        Rejection thresholds.

    Returns
    -------
    pandas.DataFrame
        One row per slice in stack order with ``n_px``, ``med_resid_m``,
        ``nmad_m``, ``frac_blunder``, ``screened``, ``unrescued`` and
        ``reason``.
    """
    if "source_variant" not in tc_stack.coords:
        raise ValueError(
            "screen_unrescued_epochs needs the stack's per-slice "
            "'source_variant' coord; rebuild the stack with build_stack."
        )
    variants = np.asarray(tc_stack["source_variant"].values, dtype=str)
    _, keep_idx = _align_tilt_params(tc_stack, tilt_params)
    temporal = _fit_temporal_model(tilt_params, keep_idx)
    z = tc_stack.values
    if temporal is None:
        print("  tilt_params lacks intercept/dhdt -- scoring against the "
              "time-static median (legacy frame)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            z_ref = np.nanmedian(z, axis=0)
        default_domain = np.isfinite(z_ref)
    else:
        intercept2d, dhdt2d, t_centered = temporal
        default_domain = np.isfinite(intercept2d) & np.isfinite(dhdt2d)
    domain = default_domain if domain_mask is None else (
        np.asarray(domain_mask, dtype=bool) & default_domain
    )

    rows = []
    for k in range(z.shape[0]):
        model = z_ref if temporal is None else intercept2d + dhdt2d * t_centered[k]
        finite = np.isfinite(z[k]) & domain
        n = int(finite.sum())
        if n:
            r = z[k][finite] - model[finite]
            med = float(np.median(r))
            nmad = 1.4826 * float(np.median(np.abs(r - med)))
            frac = float(np.mean(np.abs(r) > blunder_m))
        else:
            med = nmad = frac = np.nan
        screened = variants[k] in screen_variants
        reasons = []
        if screened:
            if n < min_px:
                reasons.append(f"n_px {n} < {min_px}")
            else:
                if nmad > nmad_max_m:
                    reasons.append(f"nmad {nmad:.1f} m > {nmad_max_m:g}")
                if frac > blunder_frac_max:
                    reasons.append(
                        f"{100 * frac:.0f}% > {blunder_m:g} m off "
                        f"(max {100 * blunder_frac_max:.0f}%)"
                    )
        rows.append((pd.Timestamp(tc_stack["time"].values[k]), variants[k], n,
                     med, nmad, frac, screened, bool(reasons), "; ".join(reasons)))

    df = pd.DataFrame(rows, columns=[
        "epoch", "source_variant", "n_px", "med_resid_m", "nmad_m",
        "frac_blunder", "screened", "unrescued", "reason",
    ])
    if "dem_id" in tc_stack.coords:
        df.insert(1, "dem_id", np.asarray(tc_stack["dem_id"].values, dtype=str))
    return df


def suggest_bad_epochs(
    df: pd.DataFrame,
    *,
    end_p50_threshold_m: "float | None" = 10.0,
    catastrophic_resid_m: "float | None" = 50.0,
    drop_irls_failed: bool = True,
    resid_threshold_m: "float | None" = None,
    frac_static_threshold: float = 0.10,
    weight_mean_threshold: float = 0.4,
) -> pd.DataFrame:
    """Return the subset of epochs that fail the bad-epoch filter.

    Shean 2019 rejects DEMs by **coregistration quality**, not by how
    much of the DEM happens to overlap the static-control region. This
    function follows that: the primary gate is the pc_align
    post-alignment residual ``end_p50`` (coverage-independent), backed
    by two narrow safety nets. An epoch is flagged if ANY of:

    **(a) pc_align quality gate** -- ``end_p50 > end_p50_threshold_m``
    (when ``end_p50`` is present in ``df``; build it with
    :func:`stereo_melt.coregister.alignment_quality.aggregate_basin_quality`
    and pass via ``score_tilt_residuals(..., alignment_quality=...)``).
    Catches strips where pc_align could not converge or carries
    non-rigid distortion the 6-DOF transform cannot remove. This is the
    Shean-faithful primary criterion. Default ``10.0`` m -- well above
    the clean population (PIG median ``end_p50`` ~0.3 m, CS2-era ~0.6 m)
    and above the offset-only demote band (``PIG_OFFSET_ONLY_END_P50_M``
    ~3 m), so 3-10 m strips are demoted to an offset-only (alpha_z) tilt
    fit -- *not* dropped -- and only genuine divergence is removed.

    **(b) catastrophic-blunder backstop** --
    ``|med_resid_m| > catastrophic_resid_m``. Catches garbage DEMs
    (bright-band / blunder strips reading tens-to-hundreds of metres
    over the control) that pc_align may have tied on a good patch but
    that would poison the per-pixel temporal-median reference.
    Coverage-robust: only true garbage reaches tens of metres of median
    static residual, so this does not flag shelf-dominated strips.
    Default ``50.0`` m; set ``None`` to disable.

    **(c) IRLS hard-failure** -- ``weight_mean`` is NaN, i.e. the robust
    step could not run for that epoch at all. Toggled by
    ``drop_irls_failed`` (default ``True``).

    **Legacy coverage-coupled gate (opt-in, Shean-INfaithful).** When
    ``resid_threshold_m`` is set (e.g. ``0.5``), additionally flags
    ``|med_resid_m| > resid_threshold_m`` AND (``frac_static <
    frac_static_threshold`` OR ``weight_mean`` NaN OR ``weight_mean <
    weight_mean_threshold``). This is the pre-2026-06-20 behaviour, kept
    only to reproduce historical screens. It is **off by default**: on
    thin-static-control basins ``frac_static`` is below threshold for
    nearly every epoch, so the rule degenerates to a flat
    ``|med_resid| > resid_threshold_m`` drop that discards
    cleanly-coregistered shelf DEMs. Higher-static basins (Beardmore at
    ~12%) may still pass ``resid_threshold_m=0.5`` deliberately until
    migrated to the coregistration gate.
    """
    bad = np.zeros(len(df), dtype=bool)

    # (a) pc_align coregistration-quality gate -- the Shean-faithful primary.
    if end_p50_threshold_m is not None and "end_p50" in df.columns:
        bad |= (df["end_p50"].fillna(0.0) > end_p50_threshold_m).to_numpy()

    # (b) catastrophic-blunder backstop (coverage-robust).
    if catastrophic_resid_m is not None:
        bad |= (df["med_resid_m"].abs() > catastrophic_resid_m).to_numpy()

    # (c) IRLS could not run for this epoch at all.
    if drop_irls_failed:
        bad |= (~np.isfinite(df["weight_mean"])).to_numpy()

    # Legacy coverage-coupled residual gate (opt-in; flags shelf strips).
    if resid_threshold_m is not None:
        big_resid = df["med_resid_m"].abs() > resid_threshold_m
        sparse_static = df["frac_static"] < frac_static_threshold
        irls_failed = ~np.isfinite(df["weight_mean"])
        irls_rejected = df["weight_mean"] < weight_mean_threshold
        bad |= (big_resid & (sparse_static | irls_failed | irls_rejected)).to_numpy()

    return df[bad]


def _failure_reason(
    row: pd.Series,
    *,
    end_p50_threshold_m: "float | None" = 10.0,
    catastrophic_resid_m: "float | None" = 50.0,
    drop_irls_failed: bool = True,
    resid_threshold_m: "float | None" = None,
    frac_static_threshold: float = 0.10,
    weight_mean_threshold: float = 0.4,
) -> str:
    causes = []
    if (
        end_p50_threshold_m is not None
        and "end_p50" in row.index
        and np.isfinite(row.get("end_p50", np.nan))
        and row["end_p50"] > end_p50_threshold_m
    ):
        causes.append(f"end_p50>{end_p50_threshold_m:.0f}m")
    if catastrophic_resid_m is not None and abs(row["med_resid_m"]) > catastrophic_resid_m:
        causes.append("blunder")
    if drop_irls_failed and not np.isfinite(row["weight_mean"]):
        causes.append("nan-w")
    if resid_threshold_m is not None and abs(row["med_resid_m"]) > resid_threshold_m:
        if row["frac_static"] < frac_static_threshold:
            causes.append("sparse")
        if np.isfinite(row["weight_mean"]) and row["weight_mean"] < weight_mean_threshold:
            causes.append("low-w")
    # de-duplicate (nan-w may be implied twice) while preserving order
    return "+".join(dict.fromkeys(causes)) if causes else "?"


def print_bad_epoch_report(
    df: pd.DataFrame,
    suggested: pd.DataFrame,
    *,
    end_p50_threshold_m: "float | None" = 10.0,
    catastrophic_resid_m: "float | None" = 50.0,
    drop_irls_failed: bool = True,
    resid_threshold_m: "float | None" = None,
    frac_static_threshold: float = 0.10,
    weight_mean_threshold: float = 0.4,
) -> None:
    """Pretty-print the QC table and suggested BAD_EPOCHS list."""
    print("Per-epoch post-tilt-fit static-control residual + IRLS quality, "
          "sorted by |med_resid|:")
    formatters = {
        "med_resid_m": "{:+.3f}".format,
        "mad_resid_m": "{:.3f}".format,
        "weight_mean": "{:.3f}".format,
        "weight_frac_kept": "{:.3f}".format,
        "frac_static": "{:.3f}".format,
    }
    for col in ("end_p16", "end_p50", "end_p84", "beg_p50"):
        if col in df.columns:
            formatters[col] = "{:.2f}".format
    print(df.to_string(index=False, formatters=formatters))
    print()
    gates = []
    if end_p50_threshold_m is not None and "end_p50" in df.columns:
        gates.append(f"end_p50 > {end_p50_threshold_m:.1f} m")
    if catastrophic_resid_m is not None:
        gates.append(f"|med_resid| > {catastrophic_resid_m:.0f} m")
    if drop_irls_failed:
        gates.append("weight_mean NaN")
    if resid_threshold_m is not None:
        gates.append(
            f"[|med_resid| > {resid_threshold_m:.1f} m AND (frac_static < "
            f"{frac_static_threshold:.2f} OR weight_mean NaN OR weight_mean "
            f"< {weight_mean_threshold:.2f})]"
        )
    gate_desc = "  OR  ".join(gates) if gates else "(no gates active)"
    print(f"Suggested BAD_EPOCHS ({gate_desc}):")
    if suggested.empty:
        print("  (none)")
    else:
        for _, r in suggested.iterrows():
            reason = _failure_reason(
                r,
                end_p50_threshold_m=end_p50_threshold_m,
                catastrophic_resid_m=catastrophic_resid_m,
                drop_irls_failed=drop_irls_failed,
                resid_threshold_m=resid_threshold_m,
                frac_static_threshold=frac_static_threshold,
                weight_mean_threshold=weight_mean_threshold,
            )
            w_str = "nan" if not np.isfinite(r["weight_mean"]) else f"{r['weight_mean']:.3f}"
            end_str = ""
            if "end_p50" in r.index and np.isfinite(r.get("end_p50", np.nan)):
                end_str = f", end_p50={r['end_p50']:.2f} m"
            print(
                f'    "{r["epoch"].strftime("%Y-%m-%d")}",  '
                f'# resid={r["med_resid_m"]:+.2f} m, '
                f'frac={r["frac_static"]:.3f}, w={w_str}{end_str}, [{reason}]'
            )
    floor = df["med_resid_m"].abs().median()
    print(f"\nUnconditional: residual_floor (median over all "
          f"{len(df)} epochs) = {floor:.3f} m")


def plot_static_mask_and_strip_count(
    stack: xr.DataArray,
    control_mask: np.ndarray,
    out_path,
    *,
    title_prefix: str = "",
    n_valid: np.ndarray | None = None,
    aoi_path=None,
) -> None:
    """Two-panel QC: static (LS) control mask + per-pixel strip count.

    Standard tilt-fit QC artifact. Called from each basin's ``tilt_fit``
    after :func:`build_static_control_mask` finishes. The left panel
    shows the boolean static-control mask actually fed into the per-epoch
    LSQ; the right panel shows how many valid REMA epochs cover each
    pixel of the stack.

    Parameters
    ----------
    stack : xarray.DataArray, dims ``(time, y, x)``
        The (corrected) DEM stack used to build the control mask.
    control_mask : numpy.ndarray, dims ``(y, x)``, bool
        Output of :func:`stereo_melt.coregister.tilt.build_static_control_mask`.
    out_path : pathlib.Path or str
        Destination PNG path.
    title_prefix : str, optional
        Prepended to the figure suptitle (e.g. ``"Beardmore"``).
    n_valid : numpy.ndarray, dims ``(y, x)``, optional
        Precomputed per-pixel finite count. If ``None``, computed from
        ``stack``.
    aoi_path : str or pathlib.Path, optional
        Path to a per-basin AOI shapefile (typically the
        ``scripts/compute_aoi.py`` ``<basin>_stack_extent.shp`` output).
        When supplied, the panel axes are clipped to the AOI bounding
        box (in EPSG:3031 km), so the imshow borders coincide with the
        AOI outline and no data is shown beyond it.
    """
    import matplotlib.pyplot as plt  # local import: I/O-only helper

    control = np.asarray(control_mask, dtype=bool)
    if n_valid is None:
        n_valid = np.isfinite(stack.values).sum(axis=0).astype(np.int32)

    extent = [
        float(stack["x"].min()) / 1000.0,
        float(stack["x"].max()) / 1000.0,
        float(stack["y"].min()) / 1000.0,
        float(stack["y"].max()) / 1000.0,
    ]

    aoi_bounds_km = None
    if aoi_path is not None:
        import fiona  # local import: only when an AOI is supplied
        from shapely.geometry import shape as _shape
        with fiona.open(str(aoi_path)) as src:
            aoi_geom = _shape(next(iter(src))["geometry"])
        ax_xmin, ax_ymin, ax_xmax, ax_ymax = aoi_geom.bounds
        aoi_bounds_km = (
            ax_xmin / 1000.0, ax_xmax / 1000.0,
            ax_ymin / 1000.0, ax_ymax / 1000.0,
        )

    def _clip_to_aoi(ax):
        if aoi_bounds_km is None:
            return
        xmin, xmax, ymin, ymax = aoi_bounds_km
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    fig, axes = plt.subplots(1, 2, figsize=(13, 7), constrained_layout=True)

    ax = axes[0]
    ax.imshow(
        control, extent=extent, origin="upper", cmap="Greys_r",
        aspect="equal", interpolation="nearest",
    )
    _clip_to_aoi(ax)
    ax.set_title(
        f"static-control (LS) mask\n"
        f"{control.sum():,} cells ({100*control.mean():.1f}%)"
    )
    ax.set_xlabel("x (km, EPSG:3031)")
    ax.set_ylabel("y (km, EPSG:3031)")

    ax = axes[1]
    nv_show = np.where(n_valid > 0, n_valid, np.nan)
    if np.any(np.isfinite(nv_show)):
        vmax = max(int(np.nanpercentile(nv_show, 99)), 2)
    else:
        vmax = max(int(stack.sizes["time"]), 2)
    im = ax.imshow(
        nv_show, extent=extent, origin="upper",
        cmap="viridis", aspect="equal", interpolation="nearest",
        vmin=1, vmax=vmax,
    )
    _clip_to_aoi(ax)
    n_pos = int((n_valid > 0).sum())
    med = int(np.median(n_valid[n_valid > 0])) if n_pos else 0
    ax.set_title(
        f"per-pixel strip count\n"
        f"{stack.sizes['time']} epochs total, median over covered pixels: {med}"
    )
    ax.set_xlabel("x (km, EPSG:3031)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="n epochs valid")

    suptitle = (
        f"{(title_prefix + ' — ') if title_prefix else ''}"
        f"static mask + strip count "
        f"({stack.sizes['time']} epochs, "
        f"{stack.sizes['x']}×{stack.sizes['y']} cells)"
    )
    fig.suptitle(suptitle, fontsize=13)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""ASP ``pc_align`` primary coregistration workflow.

Each DEM strip is coregistered in four steps:

1. Build an ICESat-2 + rock-outcrop control-surface CSV via
   :mod:`stereo_melt.coregister.reference`.
2. Run ``pc_align --highest-accuracy --alignment-method point-to-plane``.
3. Rasterize the transformed point cloud with ``point2dem``.
4. Compute ``geodiff`` residuals before and after alignment and render
   QC plots via :mod:`stereo_melt.asp_binder_utils`.
"""

import glob
import os
import re
import shutil

import numpy as np
import pandas as pd
from distutils.spawn import find_executable

from .. import asp_binder_utils as asp_utils
from ..io.strip import load_strip
from .control_source import write_sources_sidecar
from .reference import extract_rock_elevations_from_mosaics

_TRANSLATION_MAGNITUDE_RE = re.compile(
    r"Translation vector magnitude \(meters\):\s*([0-9eE+\-.]+)"
)


def _read_translation_magnitude(alignment_dir):
    r"""Parse the most-recent pc_align log and return :math:`|\Delta|` in meters.

    pc_align writes a per-run log file named
    ``<alignment_dir>-log-pc_align-*.txt`` containing a single line of the
    form ``Translation vector magnitude (meters): 2.6391508``. Multiple log
    files can exist if pc_align was retried; we read the lexicographically
    last one (timestamp-suffixed names sort to most recent). Returns
    ``None`` if no log is present or the line is missing.
    """
    logs = sorted(glob.glob(alignment_dir + "*-log-pc_align-*.txt"))
    if not logs:
        return None
    with open(logs[-1]) as fh:
        for line in fh:
            m = _TRANSLATION_MAGNITUDE_RE.search(line)
            if m:
                return float(m.group(1))
    return None


def _quarantine_alignment(alignment_dir, asp_root, dem_id, reason):
    r"""Move every ``<alignment_dir>*`` artifact into ``<asp_root>/bad_align/<dem_id>/``.

    Returns the quarantine directory. A ``_REASON.txt`` is dropped alongside
    the moved files so the cause is recoverable from the quarantine alone,
    and a zero-byte ``<dem_id>.bad_align`` sentinel is left in
    ``<asp_root>/asp_aligned/`` so the basin's ``find_unaligned_strips``
    skips this strip on subsequent runs (otherwise the missing
    ``-trans_reference-DEM.tif`` looks like a fresh strip and pc_align
    gets re-run on a known-bad input every batch).
    """
    quarantine_dir = os.path.join(asp_root, "bad_align", dem_id)
    os.makedirs(quarantine_dir, exist_ok=True)
    for src in glob.glob(alignment_dir + "*"):
        shutil.move(src, os.path.join(quarantine_dir, os.path.basename(src)))
    with open(os.path.join(quarantine_dir, "_REASON.txt"), "w") as fh:
        fh.write(reason + "\n")
    sentinel = os.path.join(os.path.dirname(alignment_dir), f"{dem_id}.bad_align")
    with open(sentinel, "w") as fh:
        fh.write(reason + "\n")
    return quarantine_dir


def prepare_asp_inputs(dem_path, rock_csv, icesat2_csv, output_dir):
    r"""Return an input dict for the ASP coregistration driver.

    Parameters
    ----------
    dem_path : str
        Path to the DEM strip to align.
    rock_csv, icesat2_csv : str
        Paths to the rock-outcrop and ICESat-2 reference CSVs.
    output_dir : str
        Working directory for ASP outputs; created if missing.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("🔄 Preparing ASP inputs...")
    inputs = {
        "dem": dem_path,
        "rock_ref": rock_csv,
        "icesat2_ref": icesat2_csv,
        "output_dir": output_dir,
    }
    return inputs


def build_control_csv(
    out_csv,
    *,
    rock_csv=None,
    icesat2_csv=None,
    cs2_csv=None,
    cs2_source_label: str = "cs2",
    atm_csv=None,
    lvis_csv=None,
    glas_csv=None,
    cap_per_source: int | None = None,
    inverse_variance_balance: bool = False,
    inv_var_base_count: int = 5000,
    rng_seed: int = 0,
    verbose: bool = True,
):
    r"""Concatenate per-source control points into one ``easting,northing,h_mean``
    CSV at ``out_csv``, with optional per-source row balancing.

    The shared control-cloud builder behind both alignment paths:

    - :func:`align_strip_with_asp` builds the full ``rock + altimetry`` cloud it
      feeds to ``pc_align``.
    - the dense-tie-from-raw driver builds an **altimetry-only** (``rock_csv=None``)
      cloud for the Stage-2 vertical datum tie, so the ``Δz`` median isn't
      dominated by whichever source has the most rows.

    Balancing (``cap_per_source`` XOR ``inverse_variance_balance``) controls each
    source's contribution by subsampling its row count, since neither the ICP cost
    nor a median has a per-observation weight. Under ``inverse_variance_balance``
    the most-precise source present gets ``inv_var_base_count`` rows and the rest
    scale by ``(σ_min/σ_i)²`` from
    :data:`stereo_melt.coregister.control_source.ICP_SIGMA_PER_SOURCE_M`. A
    fixed-seed RNG makes the subsample deterministic across re-runs.

    Returns
    -------
    list[str]
        ``sources_used`` in concatenation order (rock, is2, cs2/cryotempo, atm,
        lvis, glas), for the GCP sidecar. Empty list if no source was present.
    """
    if cap_per_source is not None and inverse_variance_balance:
        raise ValueError(
            "cap_per_source and inverse_variance_balance are mutually "
            "exclusive — pick one balancing strategy."
        )

    def _normalize_rock(df):
        # Rock CSVs from extract_rock_elevations_from_mosaics ship as
        # (x, y, elevation, tile). Harmonize to (easting, northing, h_mean)
        # so the combined reference has a single schema downstream.
        df = df[df.get("elevation", df.get("h_mean", 0)) != -9999].copy()
        if "tile" in df.columns:
            df = df.drop(columns=["tile"])
        rename = {}
        if "x" in df.columns:
            rename["x"] = "easting"
        if "y" in df.columns:
            rename["y"] = "northing"
        if "elevation" in df.columns:
            rename["elevation"] = "h_mean"
        if rename:
            df = df.rename(columns=rename)
        return df[["easting", "northing", "h_mean"]]

    parts: list[pd.DataFrame] = []
    sources_used: list[str] = []
    if rock_csv and os.path.exists(rock_csv):
        parts.append(_normalize_rock(pd.read_csv(rock_csv)))
        sources_used.append("rock")
    if icesat2_csv:
        # IS2 cache is HDF5 now (compressed); read_control_any handles
        # the .h5 vs .csv suffix mismatch and the legacy CSVs still on
        # disk pre-migration.
        from ..io.altimetry import read_control_any
        try:
            is2_df = read_control_any(icesat2_csv)
            parts.append(is2_df[["easting", "northing", "h_mean"]])
            sources_used.append("is2")
        except FileNotFoundError:
            pass
    if cs2_csv and os.path.exists(cs2_csv):
        parts.append(pd.read_csv(cs2_csv)[["easting", "northing", "h_mean"]])
        # ``cs2_source_label`` distinguishes raw ESA L2 ("cs2") from the
        # CryoTEMPO Land Ice control base ("cryotempo") — both share the
        # easting/northing/h_mean schema and this same combiner path, but
        # carry different precision tiers downstream (see control_source).
        sources_used.append(cs2_source_label)
    if atm_csv and os.path.exists(atm_csv):
        parts.append(pd.read_csv(atm_csv)[["easting", "northing", "h_mean"]])
        sources_used.append("atm")
    if lvis_csv and os.path.exists(lvis_csv):
        parts.append(pd.read_csv(lvis_csv)[["easting", "northing", "h_mean"]])
        sources_used.append("lvis")
    if glas_csv and os.path.exists(glas_csv):
        # ICESat-1 GLAS GLAH12 — the only laser control reaching the
        # 2009 → Oct-2010 REMA back-extension (cache_glas pipeline).
        parts.append(pd.read_csv(glas_csv)[["easting", "northing", "h_mean"]])
        sources_used.append("glas")

    if not parts:
        return []

    # ---- Per-source row balancing (optional) ----------------------------
    # ASP pc_align has no per-observation weight (and neither does a median),
    # so the result is dominated by whichever source has the most rows. To
    # approximate icepack-style ``1/σ²`` observation weighting, we subsample
    # row counts per source so that each source's *contribution* matches its
    # physical precision. Two modes:
    #
    #   cap_per_source=N: cap every source at N rows (equal-weight mode).
    #   inverse_variance_balance=True: target = K_base · (σ_min/σ_i)²
    #     where σ_i comes from ICP_SIGMA_PER_SOURCE_M. The most-precise source
    #     gets K_base rows; less-precise sources scale down quadratically.
    if cap_per_source is not None or inverse_variance_balance:
        from .control_source import ICP_SIGMA_PER_SOURCE_M

        rng = np.random.default_rng(rng_seed)

        if inverse_variance_balance:
            # Per-point vertical precision drives the weighting. Use
            # ICP_SIGMA_PER_SOURCE_M (per-point σ) here, NOT
            # EZ_PER_SOURCE_M (αz prior strength, which folds in coverage).
            sigmas = [ICP_SIGMA_PER_SOURCE_M.get(s, ICP_SIGMA_PER_SOURCE_M["rock"])
                      for s in sources_used]
            sigma_min = min(sigmas)
            targets = {
                s: max(1, int(round(
                    inv_var_base_count
                    * (sigma_min / ICP_SIGMA_PER_SOURCE_M.get(s, ICP_SIGMA_PER_SOURCE_M["rock"])) ** 2
                )))
                for s in sources_used
            }
        else:
            targets = {s: int(cap_per_source) for s in sources_used}

        balanced: list[pd.DataFrame] = []
        for s, df in zip(sources_used, parts):
            tgt = targets[s]
            n_in = len(df)
            if n_in <= tgt:
                balanced.append(df)
                if verbose:
                    print(f"   {s}: {n_in} rows (≤ target {tgt}, kept all)")
                continue
            idx = rng.choice(n_in, size=tgt, replace=False)
            balanced.append(df.iloc[idx].reset_index(drop=True))
            if verbose:
                print(f"   {s}: subsampled {n_in} → {tgt} rows")
        parts = balanced

    combined_df = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    combined_df.to_csv(out_csv, index=False)
    if verbose:
        print(f"🔄 Combined control: {'+'.join(sources_used)}  "
              f"({len(combined_df)} points)")
    return sources_used


def align_strip_with_asp(
    file_path,
    rock_csv=None,
    icesat2_csv=None,
    cs2_csv=None,
    cs2_source_label: str = "cs2",
    atm_csv=None,
    lvis_csv=None,
    glas_csv=None,
    output_dir=None,
    max_displacement=100,
    alignment_method="point-to-plane",
    verbose=True,
    tr=2,
    cap_per_source: int | None = None,
    inverse_variance_balance: bool = False,
    inv_var_base_count: int = 5000,
    rng_seed: int = 0,
    keep_point_cloud: bool = False,
):
    r"""Coregister a DEM strip to rock + altimetric control with ASP ``pc_align``.

    Runs ``pc_align`` with the combined control CSV, rasterizes the
    transformed point cloud with ``point2dem`` at ``tr`` meters per
    pixel, and computes ``geodiff`` residuals before and after alignment
    for QC.

    Parameters
    ----------
    file_path : str
        Path to the DEM strip in EPSG:3031.
    rock_csv, icesat2_csv, cs2_csv, atm_csv, lvis_csv, glas_csv : str, optional
        Paths to the rock-outcrop, ICESat-2, CryoSat-2 SARIn, IceBridge
        ATM (ILATM2), IceBridge LVIS (ILVIS2), and ICESat-1 GLAS
        (GLAH12) reference CSVs in EPSG:3031, all sharing the
        ``easting,northing,h_mean`` schema.
        At least one must be supplied. When multiple are provided, all
        rows are concatenated and fed to ``pc_align`` as one control
        cloud. The ATM and LVIS catalogues come from the per-strip
        airborne-lidar caching pipeline
        (:mod:`stereo_melt.coregister.cache_airborne`); GLAS from
        :mod:`stereo_melt.coregister.cache_glas`.
    cs2_source_label : str
        Source label recorded for the ``cs2_csv`` cloud in the sources
        sidecar. ``"cs2"`` for raw ESA L2 SARIn POCA (default), or
        ``"cryotempo"`` when ``cs2_csv`` is the ESA CryoTEMPO Land Ice
        control base. Both share the ``easting,northing,h_mean`` schema;
        the label only changes the per-epoch Ez tier picked downstream.
    output_dir : str
        Working directory for ASP outputs.
    max_displacement : float
        Maximum allowed displacement passed to ``pc_align``.
    alignment_method : str
        ``pc_align`` alignment method.
    verbose : bool
        If True, stream ASP command output to stdout.
    tr : float
        ``point2dem`` target ground resolution in meters.
    cap_per_source : int, optional
        If given, randomly subsample each non-empty source down to at
        most ``cap_per_source`` rows before concatenation. Yields
        approximately equal per-source contribution to the pc_align ICP
        cost. Default ``None`` = no subsampling (every row used).
    inverse_variance_balance : bool
        If True, subsample each source to a per-source target
        proportional to ``1/σ²`` where ``σ²`` is taken from
        :data:`stereo_melt.coregister.control_source.EZ_PER_SOURCE_M`.
        Mirrors the standard sparse-data assimilation weighting used in
        icepack's continuous-Galerkin observational fits
        (https://icepack.github.io/notebooks/how-to/04-sparse-data/) —
        ICP has no per-row weight, but the same effect comes from
        controlling row counts. ``inv_var_base_count`` sets the target
        for the *most-precise* source; less-precise sources are scaled
        down by ``(σ_min / σ_i)²``. Mutually exclusive with
        ``cap_per_source``.
    inv_var_base_count : int
        Target row count for the most-precise source under inverse-
        variance balance. Default 5000.
    rng_seed : int
        Seed for the per-source random subsample. Setting an explicit
        seed makes the alignment deterministic across re-runs.
    keep_point_cloud : bool
        If True, keep the ``*-trans_reference.tif`` transformed point
        cloud after ``point2dem`` grids it. Default False: the cloud is
        a pc_align intermediate nothing downstream reads, and at
        ~1.3 GB/strip it dominates the ASP root footprint (~90% of
        ``asp_aligned/`` in the 2026-06-12 PIG disk audit).

    Returns
    -------
    str
        Path to the aligned DEM GeoTIFF.
    """
    file_name = os.path.basename(file_path)
    file_name_no_ext = os.path.splitext(file_name)[0]

    # Ensure output_dir ends with '/' since downstream paths concatenate
    if not output_dir.endswith("/"):
        output_dir = output_dir + "/"
    # Create all subdirectories ASP will write into — pc_align / point2dem /
    # geodiff take a prefix path and require the parent directory to exist.
    for sub in ("", "reference_files", "asp_aligned", "initial", "final"):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    # Per-strip combined CSV: a shared filename would race in parallel runs.
    combined_csv_path = (
        output_dir + f"reference_files/combined_reference_{file_name_no_ext}.csv"
    )
    sources_used = build_control_csv(
        combined_csv_path,
        rock_csv=rock_csv,
        icesat2_csv=icesat2_csv,
        cs2_csv=cs2_csv,
        cs2_source_label=cs2_source_label,
        atm_csv=atm_csv,
        lvis_csv=lvis_csv,
        glas_csv=glas_csv,
        cap_per_source=cap_per_source,
        inverse_variance_balance=inverse_variance_balance,
        inv_var_base_count=inv_var_base_count,
        rng_seed=rng_seed,
        verbose=verbose,
    )
    if not sources_used:
        raise RuntimeError("❌ No valid reference data available for alignment.")
    # Persist the source list so downstream tilt_fit can pick a per-epoch
    # Ez prior matched to the GCP altimeter precision (see
    # ``stereo_melt.coregister.control_source``).
    write_sources_sidecar(output_dir.rstrip("/"), file_name_no_ext, sources_used)

    pc_align = find_executable("pc_align")
    ref_alitmetry = combined_csv_path
    src_dem = file_path
    alignment_dir = output_dir + "asp_aligned/" + file_name_no_ext
    csv_proj4 = (
        "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +x_0=0 +y_0=0 "
        "+datum=WGS84 +units=m +no_defs +type=crs"
    )
    csv_format = "1:easting,2:northing,3:height_above_datum"
    alignment_call = (
        f"{pc_align} --highest-accuracy --csv-format '{csv_format}' --csv-srs '{csv_proj4}' "
        f"--save-inv-transformed-reference-points --alignment-method {alignment_method}  "
        f"--max-displacement {max_displacement} {src_dem} {ref_alitmetry} -o {alignment_dir}"
    )

    print("🚀 Running ASP pc_align: ")
    asp_utils.run_bash_command(alignment_call, verbose=verbose)

    # pc_align silently produces no -trans_reference.tif when alignment
    # diverges or input clouds don't overlap. Detect that here rather
    # than crashing on an empty glob downstream. The pre-existing
    # `os.path.exists(output_dir)` check was a no-op (the working dir
    # always exists), so it never actually caught these cases.
    candidates = glob.glob(alignment_dir + "*-trans_reference.tif")
    if not candidates:
        iter_csv = alignment_dir + "-iterationInfo.csv"
        raise RuntimeError(
            f"❌ pc_align produced no transformed cloud for {file_name_no_ext}. "
            f"Likely cause: insufficient overlap between source DEM and IS2/rock "
            f"control, or alignment failed to converge. Inspect "
            f"{iter_csv if os.path.exists(iter_csv) else alignment_dir + '*'} "
            f"and re-run with verbose=True to capture pc_align stderr."
        )
    print(f"✅ Aligned point cloud produced at {candidates[0]}")

    # Displacement sanity gate. pc_align is invoked with
    # `--max-displacement {max_displacement}` (default 100 m, PIG value of
    # Shean et al. 2019). A converged alignment must therefore stay inside
    # that cap; if pc_align reports `|Δ| > max_displacement` the input
    # clouds didn't actually overlap (typically a CRS/datum mismatch) and
    # pc_align wrote out a transform anyway. Quarantine the run rather
    # than letting the catastrophic translation poison the downstream
    # stack and tilt-fit.
    delta_m = _read_translation_magnitude(alignment_dir)
    asp_root = output_dir.rstrip("/")
    if delta_m is not None and delta_m > max_displacement:
        if not keep_point_cloud:
            # No point quarantining the ~1.3 GB intermediate cloud; the
            # pc_align log + iterationInfo carry the debugging signal.
            for c in candidates:
                try:
                    os.remove(c)
                except OSError:
                    pass
        quarantine = _quarantine_alignment(
            alignment_dir, asp_root, file_name_no_ext,
            reason=(
                f"|Δ|={delta_m:.1f} m > max_displacement={max_displacement} m. "
                f"pc_align did not converge -- likely a "
                f"CRS/datum mismatch on input clouds."
            ),
        )
        raise RuntimeError(
            f"❌ pc_align translation magnitude {delta_m:.1f} m exceeds "
            f"max_displacement={max_displacement} m for {file_name_no_ext}. "
            f"Outputs quarantined to {quarantine}/."
        )
    if delta_m is not None:
        print(f"   Translation magnitude: {delta_m:.2f} m "
              f"(cap {max_displacement} m)")

    point2dem = find_executable("point2dem")
    tsrs = "EPSG:3031"
    nodata_value = -9999.0
    pointcloud = candidates[0]
    print(f"Gridding pointcloud {pointcloud} at {tr} m/px")
    point2dem_call = (
        f"{point2dem} --tr {tr} --t_srs '{tsrs}' --nodata-value {nodata_value} {pointcloud}"
    )
    asp_utils.run_bash_command(point2dem_call, verbose=verbose)
    aligned_dem = glob.glob(alignment_dir + "*-DEM.tif")[0]
    print(f"DEM saved at {aligned_dem}")

    geodiff = find_executable("geodiff")
    initial_output_prefix = output_dir + "initial/" + file_name_no_ext + "-initial"
    print(
        f"Computing elevation difference before alignment between {ref_alitmetry} and {src_dem}\n\n"
    )
    geodiff_call = (
        f"{geodiff} {ref_alitmetry} {src_dem} --csv-format {csv_format} "
        f"--csv-srs '{csv_proj4}' -o {initial_output_prefix}"
    )
    asp_utils.run_bash_command(geodiff_call, verbose=verbose)
    initial_elevation_difference_fn = glob.glob(initial_output_prefix + "-diff.csv")[0]
    print(f"\n\nInitial elevation difference saved at {initial_elevation_difference_fn}")

    final_output_prefix = output_dir + "final/" + file_name_no_ext + "-final"
    print(
        f"Computing elevation difference after alignment between {ref_alitmetry} and {aligned_dem}\n\n"
    )
    geodiff_call = (
        f"{geodiff} {ref_alitmetry} {aligned_dem} --csv-format {csv_format} "
        f"--csv-srs '{csv_proj4}' -o {final_output_prefix}"
    )
    asp_utils.run_bash_command(geodiff_call, verbose=verbose)
    final_elevation_difference_fn = glob.glob(final_output_prefix + "-diff.csv")[0]
    print(f"\n\nFinal elevation difference saved at {final_elevation_difference_fn}")

    # QC plots must never kill an alignment: the aligned DEM and geodiff
    # CSVs are already on disk at this point.
    try:
        # Reference control points (easting, northing, h_mean) were written to
        # the combined-reference CSV by build_control_csv; read them back for
        # the QC maps. (Prior code referenced an out-of-scope `combined_df`.)
        reference_altimetry_df = pd.read_csv(ref_alitmetry)
        asp_utils.plot_alignment_maps_altimetry(
            reference_altimetry_df,
            src_dem,
            initial_elevation_difference_fn,
            final_elevation_difference_fn,
            plot_crs=tsrs,
            output_dir=output_dir,
        )
    except Exception as exc:
        print(
            f"⚠️  QC plotting failed for {file_name_no_ext}: {exc!r}; "
            f"keeping alignment outputs and continuing."
        )

    # The transformed point cloud is a pc_align intermediate: once
    # point2dem has gridded it and geodiff has QC'd the DEM, nothing
    # downstream reads it (build_stack consumes the DEM), and at
    # ~1.3 GB/strip it is ~90% of the ASP root footprint.
    if not keep_point_cloud:
        try:
            cloud_gb = os.path.getsize(pointcloud) / 1e9
            os.remove(pointcloud)
            print(f"🧹 Removed transformed point cloud ({cloud_gb:.1f} GB) "
                  f"after gridding: {pointcloud}")
        except OSError as exc:
            print(f"⚠️  Could not remove point cloud {pointcloud}: {exc!r}")

    return aligned_dem


def align_strip_two_stage(
    file_path,
    reference_dem,
    datum_csv=None,
    output_dir=None,
    max_displacement=100,
    alignment_method="point-to-plane",
    tr=2,
    verbose=True,
    keep_point_cloud=False,
    min_datum_points=20,
):
    r"""Option B two-stage coregistration of a DEM strip (see plan_alignment.md).

    **Stage 1 — geometry/tilt.** ``pc_align`` the strip to a DENSE static-surface
    reference DEM (``reference_dem`` = REMA mosaic clipped to rock + slow grounded
    ice). Dense surface-to-surface ICP constrains the strip's tilt/rotation across
    the whole static footprint -- the constraint sparse altimetry can't give.
    pc_align's ICP only uses the overlap, so masking the reference to static
    surfaces is what keeps the dynamic shelf from being tied to the multi-year
    mean (which would erase the dh/dt we measure). Same call shape as
    :func:`align_strip_with_asp` (strip first → inverse-transformed + saved,
    reference second), just a DEM reference instead of a sparse CSV.

    **Stage 2 — datum/time.** The multi-year reference can't anchor this epoch's
    absolute height, so apply a residual vertical offset ``Δz = median(control −
    aligned)`` sampled at the altimetry control points in ``datum_csv``
    (IS2/CS2/ATM/LVIS, grounded/static by construction), restoring the
    epoch-correct datum.

    The reference is the REMA mosaic (our PGC data) -- never Shean's uploaded
    grids. Returns the path to the final datum-tied aligned DEM
    (``<stem>-trans_reference-DEM.tif``), named for the existing downstream
    (build_stack) discovery.
    """
    import rasterio

    stem = os.path.splitext(os.path.basename(file_path))[0]
    if not output_dir.endswith("/"):
        output_dir = output_dir + "/"
    for sub in ("", "asp_aligned", "initial", "final"):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    pc_align = find_executable("pc_align")
    alignment_dir = output_dir + "asp_aligned/" + stem

    # ---- Stage 1: dense surface-to-surface to the static reference DEM ----
    align_call = (
        f"{pc_align} --highest-accuracy --save-inv-transformed-reference-points "
        f"--alignment-method {alignment_method} --max-displacement {max_displacement} "
        f"{file_path} {reference_dem} -o {alignment_dir}"
    )
    print("🚀 Stage 1 — pc_align strip → REMA static reference DEM")
    asp_utils.run_bash_command(align_call, verbose=verbose)

    candidates = glob.glob(alignment_dir + "*-trans_reference.tif")
    if not candidates:
        raise RuntimeError(
            f"❌ Stage-1 pc_align produced no transformed cloud for {stem}. "
            f"Likely insufficient static-surface overlap between the strip and the "
            f"REMA static reference (strip may be shelf-only)."
        )
    delta_m = _read_translation_magnitude(alignment_dir)
    if delta_m is not None and delta_m > max_displacement:
        if not keep_point_cloud:
            for c in candidates:
                try:
                    os.remove(c)
                except OSError:
                    pass
        quarantine = _quarantine_alignment(
            alignment_dir, output_dir.rstrip("/"), stem,
            reason=(
                f"|Δ|={delta_m:.1f} m > max_displacement={max_displacement} m "
                f"(Stage-1 surface-to-surface)."
            ),
        )
        raise RuntimeError(
            f"❌ Stage-1 |Δ|={delta_m:.1f} m exceeds cap for {stem}; "
            f"quarantined to {quarantine}/."
        )
    if delta_m is not None:
        print(f"   Stage-1 translation magnitude: {delta_m:.2f} m (cap {max_displacement} m)")

    point2dem = find_executable("point2dem")
    pointcloud = candidates[0]
    asp_utils.run_bash_command(
        f"{point2dem} --tr {tr} --t_srs 'EPSG:3031' --nodata-value -9999.0 {pointcloud}",
        verbose=verbose,
    )
    aligned_dem = glob.glob(alignment_dir + "*-DEM.tif")[0]
    print(f"   Stage-1 aligned DEM: {aligned_dem}")

    # ---- Stage 2: residual vertical datum tie to altimetry over static ----
    if datum_csv and os.path.exists(datum_csv):
        ctrl = pd.read_csv(datum_csv)
        ex = ctrl["easting"].to_numpy(dtype=float)
        ny = ctrl["northing"].to_numpy(dtype=float)
        hz = ctrl["h_mean"].to_numpy(dtype=float)
        with rasterio.open(aligned_dem) as ds:
            prof = ds.profile
            arr = ds.read(1).astype(float)
            nod = ds.nodata if ds.nodata is not None else -9999.0
            cols_f, rows_f = (~ds.transform) * (ex, ny)
        rows = np.floor(rows_f).astype(int)
        cols = np.floor(cols_f).astype(int)
        inb = (rows >= 0) & (rows < arr.shape[0]) & (cols >= 0) & (cols < arr.shape[1])
        samp = np.full(ex.shape, np.nan)
        samp[inb] = arr[rows[inb], cols[inb]]
        samp[samp == nod] = np.nan
        resid = hz - samp
        resid = resid[np.isfinite(resid)]
        if resid.size >= min_datum_points:
            dz = float(np.median(resid))
            valid = arr != nod
            arr[valid] = arr[valid] + dz
            with rasterio.open(aligned_dem, "w", **prof) as dst:
                dst.write(arr.astype(prof["dtype"]), 1)
            print(
                f"   Stage-2 datum tie: Δz={dz:+.3f} m from {resid.size} control "
                f"points (MAD={float(np.median(np.abs(resid - dz)) * 1.4826):.3f} m)"
            )
        else:
            print(
                f"   Stage-2 datum tie SKIPPED: only {resid.size} finite control "
                f"residuals (<{min_datum_points}); leaving REMA datum."
            )
    else:
        print("   Stage-2 datum tie SKIPPED: no datum_csv supplied (REMA datum kept)")

    if not keep_point_cloud:
        try:
            os.remove(pointcloud)
        except OSError:
            pass

    return aligned_dem


def _cs2_h5_to_csv(cs2_h5_path, out_csv):
    r"""Convert a CryoSat-2 PointCollection HDF5 cache to the
    ``easting,northing,h_mean`` CSV schema that ``pc_align`` (and the
    rest of the ASP control combiner) expects.

    The CS2 cache is filtered to grounded ∩ slow-velocity at write time
    (see :func:`beardmore.cache_cryosat2.filter_cs2_grounded_slow_velocity`),
    mirroring the IS2 cache filter so the two control sources live in the
    same regime.
    """
    from pointCollection.data import data as PCData

    pc = PCData().from_h5(str(cs2_h5_path))
    df = pd.DataFrame({
        "easting": np.asarray(pc.x, dtype=np.float64),
        "northing": np.asarray(pc.y, dtype=np.float64),
        "h_mean": np.asarray(pc.h, dtype=np.float64),
    })
    df.to_csv(out_csv, index=False)
    return out_csv


def align_strip(
    file_path,
    corrections,
    x_min,
    x_max,
    y_min,
    y_max,
    mosaic_dir,
    rock_shapefile,
    output_dir,
    model="CATS2008",
    center_time=None,
    aoi_polygon=None,
    use_is2=True,
    cs2_h5_path=None,
    cs2_source_label: str = "cs2",
    atm_csv_path=None,
    lvis_csv_path=None,
    glas_csv_path=None,
    cap_per_source: int | None = None,
    inverse_variance_balance: bool = False,
    inv_var_base_count: int = 5000,
):
    r"""Coregister a single DEM strip end-to-end.

    Reads the pre-cached ICESat-2 ATL06 control CSV produced by the
    basin's ``cache_icesat2`` driver, extracts rock elevations from the
    REMA mosaic tiles within the strip footprint, and passes both to
    :func:`align_strip_with_asp`.

    The IS2 cache is expected at::

        <output_dir>/icesat2_data/icesat2_filtered_<dem_id>.csv

    Missing entries raise ``FileNotFoundError`` with a pointer to the
    cache driver, rather than reaching out to sliderule from inside the
    per-strip pipeline.

    Parameters
    ----------
    file_path : str
        Path to the original DEM strip.
    corrections : object
        Reserved for downstream filtering; not used in ASP alignment.
    x_min, x_max, y_min, y_max : float
        Spatial bounds (reserved).
    mosaic_dir : str
        Directory of REMA mosaic tiles for rock elevation extraction.
    rock_shapefile : str
        Rock-outcrop polygon shapefile.
    output_dir : str
        Directory to save alignment outputs.
    model : str
        Tidal model name (reserved; applied downstream).
    center_time : datetime-like, optional
        Strip center datetime.
    aoi_polygon : shapely.geometry.Polygon, optional
        Study-area polygon in EPSG:3031, used as a sanity check on the
        strip footprint. If the strip does not intersect the AOI at all
        we raise immediately. The strip boundary itself is NOT clipped
        to the AOI -- IS2 control surfaces (slow-velocity grounded ice)
        are usually upstream of the grounding zone AOI, so clipping
        would discard the relevant control region.

    Returns
    -------
    dict
        Loaded aligned DEM strip (see :func:`stereo_melt.io.strip.load_strip`).
    """
    strip, strip_boundary = load_strip(file_path, center_time=center_time)
    center_time = pd.Timestamp(strip["strip_date"])

    if aoi_polygon is not None and not strip_boundary.intersects(aoi_polygon):
        raise ValueError(
            "❌ Strip footprint does not intersect AOI; nothing to coregister."
        )

    dem_id = os.path.splitext(os.path.basename(file_path))[0]

    icesat2_file = None
    if use_is2:
        # New writer emits .h5; fall back to .csv for legacy caches.
        h5_path = os.path.join(
            output_dir, "icesat2_data", f"icesat2_filtered_{dem_id}.h5"
        )
        csv_path = os.path.join(
            output_dir, "icesat2_data", f"icesat2_filtered_{dem_id}.csv"
        )
        if os.path.exists(h5_path):
            icesat2_file = h5_path
        elif os.path.exists(csv_path):
            icesat2_file = csv_path
        else:
            raise FileNotFoundError(
                f"❌ IS2 cache miss: {h5_path} (or legacy .csv) not found. "
                f"Run `python -m <basin>.cache_icesat2` (e.g. "
                f"`python -m beardmore.cache_icesat2`) before aligning."
            )
        print(f"📂 Using cached IS2 control: {icesat2_file}")

    cs2_csv = None
    if cs2_h5_path is not None:
        if not os.path.exists(cs2_h5_path):
            raise FileNotFoundError(
                f"❌ CS2 cache miss: {cs2_h5_path} not found. "
                f"Run `python -m <basin>.cache_cryosat2` before aligning."
            )
        cs2_csv = os.path.join(
            output_dir, "cs2_data", f"cs2_filtered_{dem_id}.csv"
        )
        os.makedirs(os.path.dirname(cs2_csv), exist_ok=True)
        _cs2_h5_to_csv(cs2_h5_path, cs2_csv)
        print(f"📂 Using cached CS2 control: {cs2_csv}")

    # Airborne IceBridge GCPs (ATM ILATM2 / LVIS ILVIS2). The caller
    # supplies the *resolved* per-strip filtered CSV path produced by
    # ``<basin>.cache_atm`` / ``<basin>.cache_lvis``; we don't go
    # hunting for it under output_dir to keep the basin's per-source
    # cache layout free (e.g. atm_data/ vs lvis_data/).
    atm_csv = None
    if atm_csv_path is not None:
        if not os.path.exists(atm_csv_path):
            raise FileNotFoundError(
                f"❌ ATM cache miss: {atm_csv_path} not found. "
                f"Run `python -m <basin>.cache_atm` before aligning."
            )
        atm_csv = str(atm_csv_path)
        print(f"📂 Using cached ATM control: {atm_csv}")

    lvis_csv = None
    if lvis_csv_path is not None:
        if not os.path.exists(lvis_csv_path):
            raise FileNotFoundError(
                f"❌ LVIS cache miss: {lvis_csv_path} not found. "
                f"Run `python -m <basin>.cache_lvis` before aligning."
            )
        lvis_csv = str(lvis_csv_path)
        print(f"📂 Using cached LVIS control: {lvis_csv}")

    glas_csv = None
    if glas_csv_path is not None:
        if not os.path.exists(glas_csv_path):
            raise FileNotFoundError(
                f"❌ GLAS cache miss: {glas_csv_path} not found. "
                f"Run `python -m <basin>.cache_glas` before aligning."
            )
        glas_csv = str(glas_csv_path)
        print(f"📂 Using cached GLAS control: {glas_csv}")

    if (
        icesat2_file is None
        and cs2_csv is None
        and atm_csv is None
        and lvis_csv is None
        and glas_csv is None
    ):
        raise ValueError(
            "❌ No altimetric control selected: pass at least one of "
            "use_is2=True, cs2_h5_path=, atm_csv_path=, lvis_csv_path=, "
            "glas_csv_path=. "
            "ASP coregistration needs at least one altimetric source."
        )

    print("🔄 Extracting rock elevation points from DEM mosaics for current strip...")
    rock_elevation_csv = os.path.join(
        output_dir, f"rock_elevations_{os.path.basename(file_path).replace('.tif', '')}.csv"
    )
    extract_rock_elevations_from_mosaics(
        strip_boundary=strip_boundary,
        rock_shapefile=rock_shapefile,
        mosaic_dir=mosaic_dir,
        output_csv=rock_elevation_csv,
    )
    if not os.path.exists(rock_elevation_csv):
        print("⚠️ Rock elevation extraction failed, proceeding with IceSat-2 data only.")
        rock_elevation_csv = None

    print("🔄 Aligning DEM strip using ASP with combined altimetric control...")
    aligned_dem_path = align_strip_with_asp(
        file_path=file_path,
        rock_csv=rock_elevation_csv,
        icesat2_csv=icesat2_file,
        cs2_csv=cs2_csv,
        cs2_source_label=cs2_source_label,
        atm_csv=atm_csv,
        lvis_csv=lvis_csv,
        glas_csv=glas_csv,
        output_dir=output_dir,
        cap_per_source=cap_per_source,
        inverse_variance_balance=inverse_variance_balance,
        inv_var_base_count=inv_var_base_count,
    )
    if not os.path.exists(aligned_dem_path):
        raise ValueError("❌ ASP alignment failed. Skipping strip.")

    print("🔄 Loading aligned DEM strip...")
    strip_aligned, strip_aligned_boundary = load_strip(aligned_dem_path, center_time=center_time)

    print(f"✅ Coregistration complete for strip {aligned_dem_path}")
    return strip_aligned




def ingest_strip_nocorr(dem_path, asp_root, z_offset_m, overwrite=False):
    r"""Ingest a strip with no control overlap ("nocorr") at a-priori geolocation.

    Applies one class-mean vertical offset (−3.1 m on PIG in Shean et al.
    2019); the joint tilt LSQ then sets each strip's datum with a loose
    prior (Ez = 1.0 m, vs 0.3 m for coregistered DEMs in that study) from
    cross-epoch consistency; strips it cannot adjust are dropped after the
    fit.

    This function is the ingestion step of that recipe: **no pc_align, no
    geodiff** — it writes
    ``<asp_root>/asp_aligned/<dem_id>-trans_reference-DEM.tif`` as the
    input raster plus ``z_offset_m`` (block-windowed copy; SETSM strips
    are row-striped GB-scale GeoTIFFs), plus a ``sources=["nocorr"]``
    sidecar so :func:`stereo_melt.coregister.control_source.build_per_epoch_ez`
    resolves the loose Ez tier downstream, and a ``-nocorr-offset.txt``
    recording the applied constant.

    Parameters
    ----------
    dem_path : str
        Path to the raw strip GeoTIFF (a-priori geolocation).
    asp_root : str
        Per-basin nocorr ASP root (e.g. ``<basin>/data/ASP_nocorr``).
    z_offset_m : float
        Class-mean vertical bias, ADDED to strip elevations. Measure it as
        the median initial-geodiff (control minus DEM) of the basin's
        co-registered strips of the same sensor class/era.
    overwrite : bool
        Re-ingest even if the output DEM already exists.
    """
    import rasterio

    dem_id = os.path.splitext(os.path.basename(dem_path))[0]
    aligned_dir = os.path.join(asp_root, "asp_aligned")
    os.makedirs(aligned_dir, exist_ok=True)
    out_path = os.path.join(aligned_dir, f"{dem_id}-trans_reference-DEM.tif")
    if os.path.exists(out_path) and not overwrite:
        print(f"⏭  {dem_id}: nocorr ingest exists, skipping (use overwrite)")
        return out_path

    tmp_path = out_path + ".part"
    with rasterio.open(dem_path) as src:
        profile = src.profile.copy()
        is_float = np.issubdtype(np.dtype(profile["dtype"]), np.floating)
        profile.update(
            compress="lzw",
            predictor=3 if is_float else 2,
            bigtiff="if_safer",
        )
        nodata = src.nodata
        with rasterio.open(tmp_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                data = src.read(1, window=window)
                if nodata is not None:
                    data[data != nodata] += z_offset_m
                else:
                    data += z_offset_m
                dst.write(data, 1, window=window)
    os.replace(tmp_path, out_path)

    write_sources_sidecar(asp_root, dem_id, ["nocorr"])
    with open(os.path.join(aligned_dir, f"{dem_id}-nocorr-offset.txt"), "w") as fh:
        fh.write(f"{z_offset_m:+.3f}\n")
    print(f"✅ nocorr ingest {dem_id}: z {z_offset_m:+.2f} m -> {out_path}")
    return out_path

# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.

r"""Per-strip ASP control-source detection.

After ``pc_align`` runs, downstream stages (in particular the per-epoch
tilt-fit prior strength ``Ez``) need to know which altimetric reference
sources were combined into the ``combined_reference_<dem_id>.csv``
control cloud. The Beardmore wiring (Ez=0.3 m for IS2-controlled epochs,
Ez=2.0 m for CS2-only) currently uses the ICESat-2 release date as a
proxy. That is wrong in basins where some IS2-era strips fall back to
CS2-only (or pre-IS2-era strips happen to be re-aligned with IS2 once
back-extension data is available).

This module provides:

  * :func:`write_sources_sidecar` — drop a JSON sidecar at
    ``<asp_root>/asp_aligned/<dem_id>.sources.json`` recording the
    ``sources_used`` list from :func:`align_strip_with_asp`.

  * :func:`read_sources_sidecar` — read the sidecar back, returning the
    sources tuple or ``None`` when the file is missing.

  * :func:`infer_strip_sources` — sidecar first; fall back to the
    ``ASP``/``ASP_cs2``/``ASP_is2cs2`` root-suffix heuristic for
    pre-existing strips that were aligned before the sidecar was added.

  * :func:`ez_for_sources` — map a source tuple to a per-epoch Ez value.

  * :func:`build_per_epoch_ez` — given a stack's epoch timestamps and an
    ASP-root path, walk ``asp_aligned/`` for each (date, dem_id), look up
    the sources, and return the per-epoch Ez array ready to pass to
    :func:`stereo_melt.coregister.tilt.fit_tilt_stack`.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# Per-source Ez priors. ``Ez`` enters :func:`fit_tilt_stack` as the
# standard deviation of the per-strip αz prior; smaller Ez means "trust
# the GCP altimeter to anchor the per-strip mean offset tightly". The
# physical constraint is the GCP altimeter's *vertical accuracy on
# slow-flowing grounded ice* — loosen Ez when that accuracy is itself
# meter-scale, otherwise the LSQ over-shrinks αz toward zero and the
# residual leaks into the per-pixel intercepts.
#
# IS2 / ATM / LVIS are all laser altimeters with sub-meter vertical
# precision and are kept as separate dictionary entries (not lumped
# into a "high-precision" set) so each can be tuned independently — for
# example if a future LVIS firn-penetration correction shifts its
# precision tier.
#
# Sources:
#   * IS2 ATL06: ~0.1 m precision (Smith et al. 2019).
#   * ATM (ILATM2): ~0.1 m precision (Krabill et al. 2002, IceBridge).
#     Shean 2019 PIG paper uses ATM as primary control and Ez=0.3 m;
#     the Nansen 2026-05-05 dh/dt diagnosis showed +0.07 m/yr static-
#     control bias survived under Ez=0.3, so we tightened to 0.1 m to
#     pin the absolute z-reference more aggressively.
#   * LVIS (ILVIS2): ~0.12 m precision (Hofton et al. 2009). Larger
#     footprint than ATM but still sub-meter centroid accuracy.
#   * CS2 SARIn POCA: ~1-2 m precision (Baseline-D, height_1_20_ku).
EZ_PER_SOURCE_M: dict[str, float] = {
    "is2":  0.1,   # ATL06 laser altimetry (tightened 2026-05-05)
    "atm":  0.1,   # IceBridge ATM (tightened 2026-05-05)
    "lvis": 0.1,   # IceBridge LVIS centroid (tightened 2026-05-05)
    "cs2":  2.0,   # CryoSat-2 SARIn POCA (raw ESA L2)
    # ESA CryoTEMPO Land Ice (LMC retracker + interferometric POCA), gated to
    # per-point uncertainty<1 m: validated +0.16 m bias / 0.82 m MAD vs IS2 on
    # clean grounded ice (2026-06-09), ~2× tighter than raw ESA L2 (1.87 m).
    # Still SARIn radar with penetration NOT corrected → residual is meter-
    # scale, so Ez=1.0 (between laser 0.1 and raw CS2 2.0) avoids the αz over-
    # shrink that bit tight Ez on CS2-era strips (feedback_ez_per_gcp_source).
    # Provisional — revisit once a CryoTEMPO-aligned stack reaches tilt_fit.
    "cryotempo": 1.0,
    # ICESat-1 GLAS GLAH12 (cache_glas pipeline, 2009 → Oct-2010 strips).
    # Laser altimetry: ~0.1-0.15 m single-shot precision on flat ice
    # (Shuman et al. 2006) but 65 m footprints, saturation residuals, and
    # inter-campaign biases (~cm-dm) push the effective grounded-ice
    # accuracy to the 0.2-0.3 m class — laser tier, but not IS2-tight.
    "glas": 0.3,
    "rock": 2.0,   # rock outcrops anchor x/y but loose on αz (sparse coverage)
    # "nocorr" strips: no control overlap, no pc_align, only the class-mean
    # vertical offset. Ez=1.0 (vs 0.3 for coregistered DEMs in Shean et al.
    # 2019) lets the joint tilt LSQ set their datum from cross-epoch
    # self-consistency over the observation domain, 10x looser than
    # laser-controlled strips (0.1 m).
    "nocorr": 1.0,
}

# Fallback Ez when an unknown source label appears or no altimetric
# source is present at all (rock-only alignment).
EZ_FALLBACK_M = 2.0

# Per-point vertical precision (ICP balancing), distinct from EZ_PER_SOURCE_M.
# These are the standalone σ values the sensor doc reports for one return
# on slow-flowing grounded ice — they drive inverse-variance row count
# balancing in :func:`stereo_melt.coregister.asp.align_strip_with_asp`.
#
# Why separate from EZ_PER_SOURCE_M? EZ is the LSQ prior on the per-strip
# αz offset, which conflates per-point precision with *spatial coverage* —
# rock has cm-level vertical precision per point but very sparse coverage,
# so a rock-only strip ends up with a loose αz (Ez=2.0). For ICP, the
# relevant quantity is the per-point precision in the sum-of-squared-
# residuals; rock there is just as precise as IS2.
ICP_SIGMA_PER_SOURCE_M: dict[str, float] = {
    "is2":  0.1,   # Smith et al. 2019 ATL06 vertical precision
    "atm":  0.1,   # Krabill et al. 2002 ATM smoothed-segment accuracy
    "lvis": 0.12,  # Hofton et al. 2009 LVIS centroid
    "cs2":  1.5,   # CS2 SARIn POCA height_1 vertical accuracy on grounded ice
    "cryotempo": 0.82,  # CryoTEMPO LI measured MAD vs IS2 truth (unc<1m gate,
                        # 2026-06-09) — empirical per-point scatter, the right
                        # quantity for ICP inverse-variance row balancing.
    "glas": 0.15,  # GLAS single-shot σ on low-slope ice (Shuman et al. 2006)
    "rock": 0.1,   # Rock-outcrop elevation from REMA mosaic (cm-level)
}

# Backwards-compatibility aliases (some call sites still import these).
# Kept as the per-source min/max so old code reading them still does the
# right thing.
EZ_HIGH_PRECISION_M = min(EZ_PER_SOURCE_M[k] for k in ("is2", "atm", "lvis"))
EZ_LOW_PRECISION_M = EZ_PER_SOURCE_M["cs2"]
EZ_ROCK_ONLY_M = EZ_PER_SOURCE_M["rock"]
HIGH_PRECISION_SOURCES = frozenset(
    s for s, ez in EZ_PER_SOURCE_M.items() if ez <= 0.5
)
LOW_PRECISION_SOURCES = frozenset(
    s for s, ez in EZ_PER_SOURCE_M.items()
    if 0.5 < ez and s != "rock"
)

# Map ASP-root suffix → assumed sources for strips aligned BEFORE the
# JSON sidecar was added. ``align_strips.py`` uses ``--asp-suffix`` to
# keep CS2-only and combined runs separate from the canonical IS2 root,
# so the suffix is the most reliable retro-active signal.
ASP_SUFFIX_DEFAULTS: dict[str, tuple[str, ...]] = {
    "": ("rock", "is2"),          # plain ASP/ → IS2 + rock outcrop default
    "_is2": ("rock", "is2"),
    "_cs2": ("rock", "cs2"),
    "_is2cs2": ("rock", "is2", "cs2"),
    # Wider-AOI re-align variants (2026-05): full per-era GCP bundles.
    "_is2cs2atmlvis": ("rock", "is2", "cs2", "atm", "lvis"),
    "_cs2atmlvis": ("rock", "cs2", "atm", "lvis"),
    "_is2atmlvis": ("rock", "is2", "atm", "lvis"),
    # CryoTEMPO Land Ice as the CS2 control base (2026-06): same align plumbing
    # as raw CS2, distinct label so downstream Ez uses the cryotempo tier.
    # Mirrors the raw-CS2 suffixes with "cs2"→"ctempo". (Fallback only — align
    # now always writes a sidecar recording "cryotempo" directly.)
    "_ctempo": ("rock", "cryotempo"),
    "_ctempoatmlvis": ("rock", "cryotempo", "atm", "lvis"),
    "_is2ctempo": ("rock", "is2", "cryotempo"),
    "_is2ctempoatmlvis": ("rock", "is2", "cryotempo", "atm", "lvis"),
}

_DATE_RE = re.compile(r"_(\d{8})_")


def _canon_suffix(suffix: str) -> str:
    r"""Canonicalize a user-supplied suffix to the underscore-prefixed form
    used as keys in :data:`ASP_SUFFIX_DEFAULTS`.

    Accepts both ``"cs2"`` and ``"_cs2"`` (and the empty string for the
    plain ``ASP/`` root) so basin configs that store bare suffix strings
    (Beardmore's ``STRIP_SOURCES``) interoperate with the dict keys.
    """
    if suffix == "" or suffix.startswith("_"):
        return suffix
    return f"_{suffix}"


def _normalize(sources: Iterable[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for s in sources:
        s = s.strip().lower()
        if s and s not in seen:
            seen.append(s)
    return tuple(seen)


def write_sources_sidecar(
    asp_root: str | Path,
    dem_id: str,
    sources: Iterable[str],
) -> Path:
    r"""Drop ``<asp_root>/asp_aligned/<dem_id>.sources.json`` recording the
    altimetric sources that were combined into the pc_align control cloud.

    Returns the path written. Existing sidecars are overwritten; this is
    intentional so re-runs with a different ``--control`` selection
    update the recorded sources.
    """
    out = Path(asp_root) / "asp_aligned" / f"{dem_id}.sources.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"sources": list(_normalize(sources))}
    out.write_text(json.dumps(payload, indent=2))
    return out


def read_sources_sidecar(asp_root: str | Path, dem_id: str) -> tuple[str, ...] | None:
    r"""Return the recorded source tuple for ``dem_id`` under ``asp_root``,
    or ``None`` if no sidecar is present."""
    p = Path(asp_root) / "asp_aligned" / f"{dem_id}.sources.json"
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return _normalize(payload.get("sources", ()))


def infer_strip_sources(
    asp_root: str | Path,
    dem_id: str,
    *,
    suffix_default: str | None = None,
) -> tuple[str, ...]:
    r"""Best guess at the altimetric sources used to align ``dem_id``.

    Resolution order:

      1. JSON sidecar at ``<asp_root>/asp_aligned/<dem_id>.sources.json``
         (written by :func:`write_sources_sidecar`).
      2. ``ASP_SUFFIX_DEFAULTS`` keyed off the ASP-root directory name.
         For example ``data/REMA/strips/ASP_is2cs2`` → ``("rock","is2","cs2")``.
      3. A user-supplied ``suffix_default`` string (overrides #2).

    Caller-supplied ``suffix_default`` lets a basin override the heuristic
    (e.g. when a basin's ASP root is a per-basin ``data/ASP/`` rather than
    the shared ``data/REMA/strips/ASP/``).
    """
    sidecar = read_sources_sidecar(asp_root, dem_id)
    if sidecar is not None:
        return sidecar
    asp_root = Path(asp_root)
    name = asp_root.name
    if suffix_default is not None:
        return ASP_SUFFIX_DEFAULTS.get(_canon_suffix(suffix_default), ("rock", "is2"))
    # Strip "ASP" prefix to extract the suffix. "ASP" → "", "ASP_is2cs2" → "_is2cs2".
    if name.startswith("ASP"):
        suffix = name[len("ASP"):]
    else:
        suffix = ""
    return ASP_SUFFIX_DEFAULTS.get(suffix, ("rock", "is2"))


def ez_for_sources(
    sources: Sequence[str],
    *,
    ez_per_source: dict[str, float] | None = None,
    ez_fallback: float = EZ_FALLBACK_M,
) -> float:
    r"""Per-epoch Ez from a strip's altimetric source tuple.

    Rule (most-precise GCP wins): take the minimum Ez across all sources
    present in the strip's control mix. ``ez_per_source`` defaults to
    :data:`EZ_PER_SOURCE_M`, which lists each laser altimeter (IS2, ATM,
    LVIS) at 0.3 m and CryoSat-2 SARIn POCA at 2.0 m. Rock-only counts
    as 2.0 m (the rock outcrop catalogue anchors x/y / tilt well but its
    spatial coverage is too sparse to tightly constrain αz).

    Examples
    --------
    >>> ez_for_sources(("rock", "is2"))         # IS2-controlled strip
    0.1
    >>> ez_for_sources(("rock", "cs2"))         # pre-IS2 strip with CS2 only
    2.0
    >>> ez_for_sources(("rock", "atm"))         # pre-IS2 strip with ATM only
    0.1
    >>> ez_for_sources(("rock", "atm", "cs2"))  # ATM dominates the mix
    0.1
    """
    if ez_per_source is None:
        ez_per_source = EZ_PER_SOURCE_M
    src = set(s.lower() for s in sources)
    candidates = [ez_per_source[s] for s in src if s in ez_per_source]
    if not candidates:
        return float(ez_fallback)
    return float(min(candidates))


def _date_from_dem_id(dem_id: str) -> pd.Timestamp | None:
    m = _DATE_RE.search(dem_id)
    if not m:
        return None
    return pd.Timestamp(m.group(1))


def _normalize_asp_roots(
    asp_root,
    suffix_default: str | None,
) -> list[tuple[Path, str | None]]:
    r"""Coerce ``asp_root`` to a list of ``(Path, suffix_or_None)`` pairs.

    Accepted shapes:

      * single ``str`` or ``pathlib.Path`` → one root, ``suffix_default``
        applied. Back-compat with single-basin callers.
      * ``Sequence[str | Path]`` → each path uses ``suffix_default``
        (or directory-name inference when ``None``).
      * ``Sequence[tuple[str | Path, str]]`` → explicit per-root suffix
        override. Mirrors basin configs like Beardmore's ``STRIP_SOURCES``.

    The per-root suffix is consumed by :func:`infer_strip_sources` after
    the JSON sidecar check; ``None`` falls back to the ``asp_root`` dir-
    name heuristic in :data:`ASP_SUFFIX_DEFAULTS`.
    """
    if isinstance(asp_root, (str, Path)):
        return [(Path(asp_root), suffix_default)]
    out: list[tuple[Path, str | None]] = []
    for item in asp_root:
        if isinstance(item, (str, Path)):
            out.append((Path(item), suffix_default))
            continue
        try:
            p, s = item  # (path, suffix) tuple
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"asp_root entry must be Path/str or (Path, suffix); got {item!r}"
            ) from e
        out.append((Path(p), s))
    return out


def build_per_epoch_ez(
    epoch_times: np.ndarray | Sequence,
    asp_root,
    *,
    epoch_dem_ids: np.ndarray | Sequence | None = None,
    suffix_default: str | None = None,
    ez_per_source: dict[str, float] | None = None,
    ez_fallback: float = EZ_FALLBACK_M,
) -> tuple[np.ndarray, dict[str, int]]:
    r"""Per-epoch Ez array via filename-date matching against ASP roots.

    Walks ``<asp_root>/asp_aligned/`` for files matching the
    ``-trans_reference-DEM.tif`` convention, extracts the strip date from
    each filename, and looks up the recorded (or inferred) sources. Each
    input ``epoch_times`` entry is mapped to the **most-precise** source
    tuple available across all strips on that date.

    ``asp_root`` accepts:

      * a single ``Path``/``str`` (4 of 5 basins — IS2-only stacks),
      * a sequence of paths or ``(path, suffix)`` tuples for basins
        with multiple ASP roots (Beardmore: combined IS2+CS2 era +
        pre-IS2 CS2-only). When a strip's ``dem_id`` appears under
        more than one root, the union of its source tuples is taken
        before the ``min``-Ez reduction, so an epoch present in
        ``ASP_is2cs2/`` and ``ASP_cs2/`` correctly resolves to the
        IS2 tier.

    Returns ``(ez_per_epoch, summary)`` where ``summary`` counts how
    many epochs landed at each Ez tier (the keys are the unique source
    labels in :data:`EZ_PER_SOURCE_M` plus a ``missing`` slot for
    epochs whose date has no matching strip on disk).

    ``epoch_dem_ids`` — pass the stack's per-layer ``dem_id`` coordinate
    (same length as ``epoch_times``) to resolve each layer against ITS
    OWN strip's sources instead of the date-union. Without it, a layer
    whose date is shared across control classes inherits the date's
    most-precise class: on the 2026-07-11 beardmore_shelf nocorr stack,
    35 of 99 nocorr layers shared a date with an is2 strip and got the
    10x-too-tight Ez 0.1 prior. Layers whose dem_id has no sidecar match
    fall back to date resolution.

    This is the recommended entry point for basin ``tilt_fit`` drivers.
    """
    if ez_per_source is None:
        ez_per_source = EZ_PER_SOURCE_M

    roots = _normalize_asp_roots(asp_root, suffix_default)

    # Build (date_str → list of strip source tuples) from every ASP root.
    # Filename schema: "<dem_id>-trans_reference-DEM.tif"; the dem_id
    # contains the YYYYMMDD strip-acquisition date (DATE_RE).
    by_date: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    by_dem: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for root, suffix_for_root in roots:
        aligned_dir = root / "asp_aligned"
        if not aligned_dir.exists():
            raise FileNotFoundError(
                f"No asp_aligned/ under {root}; run align_strips first."
            )
        for p in sorted(aligned_dir.glob("*-trans_reference-DEM.tif")):
            dem_id = p.name.replace("-trans_reference-DEM.tif", "")
            d = _date_from_dem_id(dem_id)
            if d is None:
                continue
            srcs = infer_strip_sources(root, dem_id, suffix_default=suffix_for_root)
            by_date[d.strftime("%Y-%m-%d")].append(srcs)
            by_dem[dem_id].append(srcs)

    times = pd.to_datetime(np.asarray(epoch_times))
    ez = np.empty(len(times), dtype=np.float64)
    # Histogram by (winning_source, ez_value) so logs show which sensor
    # won the per-epoch precision contest. The summary is keyed by the
    # unique source label that produced the chosen Ez (e.g. "atm", "is2",
    # "cs2", "rock"); ``missing`` collects epochs with no strip match.
    summary: dict[str, int] = defaultdict(int)

    dem_ids = None
    if epoch_dem_ids is not None:
        dem_ids = np.asarray(epoch_dem_ids)
        if len(dem_ids) != len(times):
            raise ValueError(
                f"epoch_dem_ids length {len(dem_ids)} != epoch_times {len(times)}"
            )

    for k, t in enumerate(times):
        srcs_list = None
        if dem_ids is not None:
            srcs_list = by_dem.get(str(dem_ids[k]))
        if not srcs_list:
            key = pd.Timestamp(t).strftime("%Y-%m-%d")
            srcs_list = by_date.get(key)
        if not srcs_list:
            ez[k] = ez_fallback
            summary["missing"] += 1
            continue
        # Take the most-precise source among all strips on this date.
        # Tie-break order (low key wins via ``min``):
        #   1. Ez value — lower (more precise) wins.
        #   2. Altimetry beats rock at equal Ez — a CS2-aligned strip
        #      always has rock in its fallback control bundle, but the
        #      meaningful label is the altimeter actually used.
        #   3. Alphabetical for full determinism.
        union: set[str] = set()
        for srcs in srcs_list:
            union.update(srcs)
        candidates = [(s, ez_per_source[s]) for s in union if s in ez_per_source]
        if candidates:
            best_source, ez_value = min(
                candidates,
                key=lambda kv: (kv[1], 1 if kv[0] == "rock" else 0, kv[0]),
            )
        else:
            best_source, ez_value = "fallback", ez_fallback
        ez[k] = ez_value
        summary[best_source] += 1

    return ez, dict(summary)

# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.

r"""CryoSat-2 SARIn **Level-1B** granule reader (Baseline-D NetCDF).

Companion to :func:`stereo_melt.coregister.cache_cs2.read_cs2_l2_sarin_granule`
(which reads the *L2 POCA* product). Where L2 gives one slope-corrected
surface height per record, L1B gives the raw 20 Hz **waveforms** —
power, coherence, and interferometric phase difference — plus the
window delay and range corrections needed to turn a retracked waveform
bin into a surface range.

This is the raw material for re-retracking CS2 ourselves (Nilsson et al.
2016 leading-edge maximum-gradient retracker; see
``reference_nilsson2016_retracker``). The Baseline-D ``SIR_SIN_1B``
NetCDF is self-describing, so we read it directly via :mod:`netCDF4`
(auto scale/offset) rather than the vendored ``cryosat_toolkit`` binary
reader.

**Scope lock:** SARIn only, Baseline-D NetCDF only. The 20 Hz Ku-band
group (``*_20_ku``) is the measurement record; the 1 Hz pseudo-LRM
(``*_plrm_*``) group is ignored.

Datum / geometry notes
----------------------
- ``time_20_ku`` is TAI seconds since 2000-01-01 — **identical clock to
  the L2 product**, so an L1B record and its L2 POCA record share a
  timestamp bit-for-bit. That is the join key used downstream.
- ``window_del_20_ku`` is the *2-way* calibrated window delay (seconds).
  Range to the window reference bin is ``c/2 * window_del`` (one-way).
- ``alt_20_ku`` is the satellite CoM altitude above the WGS84 ellipsoid.
- ``pwr_waveform_20_ku`` is 1024 bins (Baseline-D oversamples the legacy
  512-bin SARIn echo). Counts scale to watts via
  ``echo_scale_factor * 2**echo_scale_pwr`` but the LMG retracker is
  per-waveform scale-invariant, so callers usually retrack raw counts.
"""

from __future__ import annotations

import re
from pathlib import Path

import netCDF4 as ncf
import numpy as np

#: ESA Science Server product root for the SARIn L1B tree (under
#: ``Cry0Sat2_data/``), confirmed live 2026-06.
ESA_SARIN_L1_REMOTE = "SIR_SIN_L1"
#: ``CS_..._<startT>_<stopT>_E0NN`` segment-time pattern (L1B and L2 share it).
_SEG_RE = re.compile(r"_(\d{8}T\d{6})_(\d{8}T\d{6})_")

#: SIRAL instrument-mode id for SARIn in ``flag_instr_mode_op_20_ku``
#: (Baseline-D ``flag_values = [1, 2, 3]`` = lrm, sar, sarin).
SARIN_MODE_ID = 3
#: Speed of light (m/s) used for the window-delay → range conversion.
C_LIGHT = 299_792_458.0


def read_cs2_l1b_sarin_granule(
    nc_path: Path,
    *,
    require_sarin: bool = True,
) -> dict | None:
    r"""Read one CS2 L1B SARIn granule into a dict of per-record arrays.

    Parameters
    ----------
    nc_path
        Path to a ``CS_*SIR_SIN_1B_*.nc`` Baseline-D granule.
    require_sarin
        If ``True`` (default) return ``None`` unless the granule is
        SARIn (filename ``SIR_SIN_1`` *and* mode flag agreement).

    Returns
    -------
    dict | None
        Keys (all length-``n`` over the 20 Hz Ku record axis, except the
        waveforms which are ``(n, 1024)``):

        - ``time`` : TAI seconds since 2000-01-01 (float64) — L2 join key.
        - ``lat``, ``lon`` : nadir geolocation (deg).
        - ``alt`` : satellite altitude above WGS84 (m).
        - ``window_del`` : 2-way window delay (s).
        - ``range_cor`` : summed 1-way range corrections (Doppler +
          tx/rx instrument), metres, to add to the retracked range.
        - ``pwr`` : ``(n, 1024)`` power waveform (counts, float32).
        - ``coherence`` : ``(n, 1024)`` interferometric coherence.
        - ``ph_diff`` : ``(n, 1024)`` interferometric phase diff (rad).
        - ``mode`` : SIRAL mode id per record.

        ``None`` if the granule is not SARIn or has no usable records.
    """
    fname = Path(nc_path).name
    # Fast skip only for granules whose name *positively* marks a non-SARIn
    # mode; otherwise the mode flag (read below) is authoritative. (Local
    # copies may be renamed, so a missing "SIR_SIN" is not disqualifying.)
    if require_sarin and ("SIR_LRM" in fname or "SIR_SAR_" in fname):
        return None

    try:
        with ncf.Dataset(str(nc_path)) as f:
            v = f.variables
            time = np.asarray(v["time_20_ku"][:], dtype=np.float64)
            lat = np.asarray(v["lat_20_ku"][:], dtype=np.float64)
            lon = np.asarray(v["lon_20_ku"][:], dtype=np.float64)
            alt = np.asarray(v["alt_20_ku"][:], dtype=np.float64)
            window_del = np.asarray(v["window_del_20_ku"][:], dtype=np.float64)
            pwr = np.asarray(v["pwr_waveform_20_ku"][:], dtype=np.float32)
            coh = np.asarray(v["coherence_waveform_20_ku"][:], dtype=np.float32)
            phd = np.asarray(v["ph_diff_waveform_20_ku"][:], dtype=np.float32)
            mode = np.asarray(v["flag_instr_mode_op_20_ku"][:], dtype=np.int16)

            # 1-way range corrections to add to the retracked range. The
            # tx/rx and rx-only instrument corrections are 1-way already;
            # Doppler is a radial range correction. (uso_cor is a delay
            # correction already folded into window_del.)
            range_cor = np.zeros(time.size, dtype=np.float64)
            for name in ("dop_cor_20_ku",
                         "instr_cor_range_tx_rx_20_ku",
                         "instr_cor_range_rx_20_ku"):
                if name in v:
                    range_cor += np.ma.filled(
                        v[name][:], 0.0).astype(np.float64)
    except Exception as exc:  # noqa: BLE001 — surface and skip the granule
        print(f"  !! L1B read failed on {fname}: {exc}")
        return None

    if time.size == 0:
        return None
    if require_sarin and not np.any(mode == SARIN_MODE_ID):
        return None

    return {
        "time": time,
        "lat": lat,
        "lon": lon,
        "alt": alt,
        "window_del": window_del,
        "range_cor": range_cor,
        "pwr": pwr,
        "coherence": coh,
        "ph_diff": phd,
        "mode": mode,
    }


# ---------------------------------------------------------------------------
# Targeted L1B prefetch (download the L1B siblings of L2 granules we hold)
# ---------------------------------------------------------------------------


def _parse_segment_span(name: str):
    r"""``(start, stop)`` datetimes from a CryoSat granule name, or ``None``."""
    from datetime import datetime
    m = _SEG_RE.search(name)
    if not m:
        return None
    return (datetime.strptime(m.group(1), "%Y%m%dT%H%M%S"),
            datetime.strptime(m.group(2), "%Y%m%dT%H%M%S"))


def prefetch_l1b_overlapping(
    l2_granule_names,
    out_dir: Path,
    *,
    client=None,
    pad_seconds: float = 2.0,
    verbose: bool = True,
) -> list[Path]:
    r"""Download the SARIn **L1B** granules that time-overlap given L2 granules.

    L1B and L2 share the same orbit segmentation — for SARIn the L1B
    granule is the 1:1 sibling of its L2 granule (identical start/stop
    times and record count; granule lengths vary 1 s–~940 s). We don't
    assume that mapping, though: we list the relevant ``SIR_SIN_L1/<y>/<m>``
    directories once and download every L1B granule whose filename time
    span overlaps any input L2 span (padded by ``pad_seconds``), so the
    matcher stays correct under any segmentation. Already-present granules
    are kept.

    Parameters
    ----------
    l2_granule_names
        Iterable of L2 granule filenames or paths (only the names are
        used) — typically the PIG-overlapping ``SIR_SIN_2`` granules on
        disk for the floor-test window.
    out_dir
        Destination directory for the L1B ``.nc`` granules.
    client
        Optional :class:`EsaCryoSatHttpsClient`; constructed if omitted.

    Returns
    -------
    list[Path]
        Local paths of the L1B granules now on disk for the spans.
    """
    from datetime import timedelta

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spans = [sp for nm in l2_granule_names
             if (sp := _parse_segment_span(Path(nm).name)) is not None]
    if not spans:
        if verbose:
            print("  L1B prefetch: no parseable L2 spans; nothing to do")
        return []
    spans.sort()
    starts = [s0 for s0, _ in spans]
    pad = timedelta(seconds=pad_seconds)
    months = sorted({(t.year, t.month) for sp in spans for t in sp})

    if client is None:
        from .esa_cs2_https import EsaCryoSatHttpsClient
        client = EsaCryoSatHttpsClient(timeout=60)

    import bisect

    def _overlaps_any(a0, a1) -> bool:
        # Any L2 span [s0,s1] with s0 <= a1 and s1 >= a0. Scan spans whose
        # start <= a1 (bisect), walking back while they can still overlap.
        hi = bisect.bisect_right(starts, a1)
        for k in range(hi - 1, -1, -1):
            s0, s1 = spans[k]
            if s1 >= a0:
                return True
            if s0 < a0 - timedelta(minutes=10):  # spans sorted; no earlier hit
                break
        return False

    downloaded: list[Path] = []
    n_new = 0
    for (y, m) in months:
        remote = f"{ESA_SARIN_L1_REMOTE}/{y}/{m:02d}"
        try:
            entries = client.list_dir(remote)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  ⚠  list {remote}: {exc}")
            continue
        wanted = []
        for e in entries:
            if not e["name"].endswith(".nc"):
                continue
            sp = _parse_segment_span(e["name"])
            if sp is None:
                continue
            if _overlaps_any(sp[0] - pad, sp[1] + pad):
                wanted.append(e)
        if verbose:
            print(f"  [{y}-{m:02d}] {len(wanted)} L1B granules overlap "
                  f"({len(entries)} listed)", flush=True)
        for e in wanted:
            local = out_dir / e["name"]
            if local.exists():
                downloaded.append(local)
                continue
            try:
                client.download(e["path"], local)
                downloaded.append(local)
                n_new += 1
                if verbose and n_new % 25 == 0:
                    print(f"    … {n_new} L1B granules downloaded", flush=True)
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"    !! download {e['name']}: {exc}")

    if verbose:
        print(f"  L1B prefetch: {len(downloaded)} granules on disk "
              f"({n_new} new) for {len(spans)} L2 spans over {len(months)} months")
    return downloaded

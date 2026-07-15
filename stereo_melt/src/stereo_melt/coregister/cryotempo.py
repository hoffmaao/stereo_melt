# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.

r"""ESA CryoTEMPO Land Ice (TDP_LI) reader.

The CryoTEMPO Land Ice Thematic Data Product is ESA's **reprocessed**,
analysis-ready CryoSat-2 elevation product. Over SARIn it uses the
Land-ice **Maximum-Coherence (LMC) retracker** plus interferometric POCA
relocation — the ESA-class processing a single-waveform leading-edge
retracker cannot match (see ``project_cs2_retracker_floor_test`` bake-off:
ESA L2 POCA = 1.87 m MAD vs IS2 on clean grounded ice, our retrackers
7-11 m). We adopt it as the CS2 control base instead of re-retracking L1B.

Penetration/volume-scattering bias is **not** removed by CryoTEMPO, so the
``backscatter`` field feeds the Schröder et al. 2019 backscatter-anomaly
correction (our value-add); ``uncertainty`` weights the points in ASP
``pc_align``; ``reference_dem`` gives a cheap blunder screen.

Files: ``TEMPO_POCA_LI/<YYYY>/<MM>/ANTARC/CS_OFFL_SIR_TDP_LI_ANTARC_*.nc``
on ``science-pds.cryosat.esa.int`` — the same server + auth as the raw L2,
so :mod:`stereo_melt.io.esa_cs2_https` downloads them unchanged.
"""

from __future__ import annotations

from pathlib import Path

import netCDF4 as ncf
import numpy as np
from pyproj import Transformer

from ..io.altimetry import PCData

# CryoTEMPO surface_type mask codes (from the LI ATBD surface mask).
TEMPO_SURF_GROUNDED = 1
TEMPO_SURF_FLOATING = 2


def read_cryotempo_li_granule(nc_path: Path) -> "PCData | None":
    r"""Read one CryoTEMPO Land Ice TDP granule into a PCData.

    Pulls POCA lat/lon, LMC-retracked ``elevation`` (WGS-84 ellipsoidal m),
    ``backscatter`` (dB, for the penetration correction), per-point
    ``uncertainty`` (m), ``surface_type`` and ``reference_dem`` (QC), and
    time. Horizontal coords are reprojected from POCA lat/lon to EPSG:3031.
    Returns ``None`` if the granule has no finite POCA elevations.
    """
    fname = Path(nc_path).name
    if "TDP_LI" not in fname:
        return None
    try:
        with ncf.Dataset(str(nc_path)) as f:
            v = f.variables
            lat = np.ma.filled(v["latitude"][:], np.nan).astype(np.float64)
            lon = np.ma.filled(v["longitude"][:], np.nan).astype(np.float64)
            h = np.ma.filled(v["elevation"][:], np.nan).astype(np.float64)
            bs = np.ma.filled(v["backscatter"][:], np.nan).astype(np.float64)
            unc = np.ma.filled(v["uncertainty"][:], np.nan).astype(np.float64)
            surf = np.ma.filled(v["surface_type"][:], -1).astype(np.int16)
            refdem = (np.ma.filled(v["reference_dem"][:], np.nan).astype(np.float64)
                      if "reference_dem" in v else np.full(lat.size, np.nan))
            t_sec = np.ma.filled(v["time"][:], np.nan).astype(np.float64)
    except Exception as exc:  # noqa: BLE001
        print(f"  !! CryoTEMPO read failed on {fname}: {exc}")
        return None

    keep = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(h)
    if not keep.any():
        return None
    lat = lat[keep]; lon = lon[keep]; h = h[keep]; bs = bs[keep]
    unc = unc[keep]; surf = surf[keep]; refdem = refdem[keep]; t_sec = t_sec[keep]

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    east, north = transformer.transform(lon, lat)

    # CryoTEMPO time is "seconds since 2000-01-01"; store as ns-since-1970.
    t = (np.datetime64("2000-01-01T00:00:00", "ns")
         + (t_sec * 1e9).astype("int64").astype("timedelta64[ns]")).astype(np.int64)

    return PCData().from_dict({
        "x": np.asarray(east, dtype=np.float64),
        "y": np.asarray(north, dtype=np.float64),
        "h": np.asarray(h, dtype=np.float64),
        "backscatter": np.asarray(bs, dtype=np.float64),
        "uncertainty": np.asarray(unc, dtype=np.float64),
        "surface_type": surf,
        "reference_dem": np.asarray(refdem, dtype=np.float64),
        "t_tai": np.asarray(t_sec, dtype=np.float64),
        "t": t,
        "source": np.full(lat.size, "cryotempo_li", dtype=object),
    })

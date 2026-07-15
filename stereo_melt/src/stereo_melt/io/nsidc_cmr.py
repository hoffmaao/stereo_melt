# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.

r"""NASA NSIDC granule discovery + download.

Used by the airborne-lidar caching path (ILATM2 / ILVIS2). Public NASA
CMR API for granule listing — no auth required. Downloads use Earthdata
auth via ``~/.netrc``. Mirrors the JSON-walker ergonomics of
:mod:`stereo_melt.io.esa_cs2_https` (one client, thread-safe session,
cached by-URL).

Authentication discovery:

  - User runs ``earthaccess login`` once or hand-edits ``~/.netrc``
    with a ``machine urs.earthdata.nasa.gov`` entry. The .netrc machine
    string is fixed; downloader URLs from NSIDC redirect to URS for
    auth and back.

This module is intentionally minimal: granule listing uses the
``cmr.earthdata.nasa.gov/search/granules.umm_json`` endpoint, granule
download streams via :mod:`requests` with ``.netrc`` auth.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import requests

CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"

# How many granules to ask CMR for per page. CMR caps at 2000.
CMR_PAGE_SIZE = 1000


def query_nsidc_granules(
    short_name: str,
    *,
    version: str | None = None,
    bounding_box: tuple[float, float, float, float] | None = None,
    temporal: tuple[str, str] | None = None,
    polygon: list[float] | None = None,
    extra_params: dict | None = None,
    timeout: float = 60.0,
) -> list[dict]:
    r"""Query NASA CMR for granules of a NSIDC dataset.

    Returns a list of normalized granule dicts:

      ``{"name": ..., "url": <https-download>, "size_mb": float,
         "begin_time": <iso>, "end_time": <iso>}``

    Parameters
    ----------
    short_name : str
        NSIDC product short name, e.g. ``"ILATM2"`` or ``"ILVIS2"``.
    version : str, optional
        Product version, e.g. ``"002"``. If omitted, CMR returns granules
        across all versions.
    bounding_box : (W, S, E, N), optional
        Geographic bounding box in decimal degrees (-180..180 lon).
    temporal : (start, end), optional
        ISO-8601 timestamps (e.g. ``"2019-01-01T00:00:00Z"``).
    polygon : list[float], optional
        Flat list ``[lon0, lat0, lon1, lat1, ...]`` (CMR requirement).
        Polygon must close (last point == first).
    extra_params : dict, optional
        Additional query parameters passed verbatim to CMR.
    """
    params: dict = {
        "short_name": short_name,
        "page_size": CMR_PAGE_SIZE,
    }
    if version is not None:
        params["version"] = version
    if bounding_box is not None:
        w, s, e, n = bounding_box
        params["bounding_box"] = f"{w},{s},{e},{n}"
    if temporal is not None:
        t0, t1 = temporal
        params["temporal"] = f"{t0},{t1}"
    if polygon is not None:
        params["polygon"] = ",".join(str(v) for v in polygon)
    if extra_params:
        params.update(extra_params)

    granules: list[dict] = []
    search_after: list | None = None
    headers = {"Accept": "application/vnd.nasa.cmr.umm_results+json"}
    while True:
        if search_after is not None:
            # CMR's `search-after` header is a JSON-serialized array.
            import json as _json
            headers["CMR-Search-After"] = _json.dumps(search_after)
        r = requests.get(CMR_GRANULES_URL, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        page = r.json()
        items = page.get("items", []) or []
        for it in items:
            g = _normalize_granule(it)
            if g is not None:
                granules.append(g)
        # Pagination — CMR returns CMR-Search-After header for the next page.
        nxt = r.headers.get("CMR-Search-After")
        if not nxt or not items:
            break
        import json as _json
        search_after = _json.loads(nxt)

    return granules


def _normalize_granule(item: dict) -> dict | None:
    """Pull (name, https-url, size, time-range) out of a CMR UMM-JSON item."""
    umm = item.get("umm", {}) or {}
    name = umm.get("GranuleUR") or umm.get("ProviderDates", [{}])[0].get("Date")
    related = umm.get("RelatedUrls", []) or []
    https_url = None
    for ru in related:
        u = ru.get("URL", "")
        rtype = ru.get("Type", "")
        if not u.lower().startswith("https://"):
            continue
        # Prefer the GET DATA URL; fall back to anything else with .h5/.H5.
        if rtype == "GET DATA":
            https_url = u
            break
        if u.lower().endswith((".h5", ".he5")) and https_url is None:
            https_url = u
    if not https_url:
        return None

    size_mb = None
    da = umm.get("DataGranule", {}) or {}
    for sz in da.get("ArchiveAndDistributionInformation", []) or []:
        s = sz.get("Size")
        u = sz.get("SizeUnit", "")
        if isinstance(s, (int, float)):
            if u.upper() == "MB":
                size_mb = float(s)
            elif u.upper() == "KB":
                size_mb = float(s) / 1024.0
            elif u.upper() in ("BYTES", "B"):
                size_mb = float(s) / (1024.0 * 1024.0)
            break

    tr = umm.get("TemporalExtent", {}).get("RangeDateTime", {}) or {}
    return {
        "name": name,
        "url": https_url,
        "size_mb": size_mb,
        "begin_time": tr.get("BeginningDateTime"),
        "end_time": tr.get("EndingDateTime"),
    }


def download_nsidc_granule(
    url: str,
    out_path: Path,
    *,
    session: requests.Session | None = None,
    timeout: float = 300.0,
    chunk_size: int = 1 << 20,
    overwrite: bool = False,
) -> Path:
    r"""Stream a granule to ``out_path`` via Earthdata-authenticated HTTPS.

    Reads credentials from ``~/.netrc`` (``machine urs.earthdata.nasa.gov``).
    NSIDC HTTPS endpoints redirect to URS for auth on first contact and
    set a session cookie; using a single :class:`requests.Session` per
    caller avoids re-auth per granule.
    """
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sess = session or requests.Session()
    # NSIDC's Earthdata-protected endpoints require following redirects to
    # URS and back. requests handles this automatically; .netrc is read by
    # default for HTTP basic auth on the URS host.
    with sess.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
        r.raise_for_status()
        # Stream to a temp file, then rename — partial files break downstream parsers.
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
        os.replace(tmp, out_path)
    return out_path


def make_session() -> requests.Session:
    r"""A pre-configured Session with a UA and trusted env (.netrc)."""
    s = requests.Session()
    s.headers.update({"User-Agent": "stereo_melt/1.0 (NSIDC CMR + Earthdata)"})
    s.trust_env = True  # pick up .netrc + http(s)_proxy from env
    return s

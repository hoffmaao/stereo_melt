#!/usr/bin/env python
"""Sanity check: strip-level (dem_id) bad-epoch dropping (added 2026-06-20).

Exercises the per-strip rejection path end-to-end on synthetic stacks, with
no real REMA rebuild, so the plumbing can be verified before the production
is2ctempo stack is rebuilt with the new ``dem_id`` coord. Covers:

  1. load_basin_stack(bad_strips=...) drops individual strips by dem_id and
     SPARES clean same-day siblings (vs bad_epochs, which drops the whole day).
  2. Backward-compat: on a date-only stack (no dem_id coord) bad_strips warns
     and is skipped, while bad_epochs still drops by date.
  3. score_tilt_residuals joins pc_align quality PER-STRIP (dem_id), so
     suggest_bad_epochs flags the one bad strip on a mixed day, not the date.
  4. The legacy worst-strip-per-day join over-drops -- the bug (3) fixes.

Run:
  /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python \
      stereo_melt/scripts/sanity_strip_dropping.py
"""
from __future__ import annotations

import gc
import os
import shutil
import sys
import tempfile

# Point pyproj at the env-local proj.db before rioxarray/pyproj load, else the
# base conda's corrupt database segfaults the interpreter (feedback_proj_data).
_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ.setdefault("PROJ_DATA", _env_proj)
    os.environ.setdefault("PROJ_LIB", _env_proj)

import numpy as np
import pandas as pd
import xarray as xr

from stereo_melt.coregister.tilt_qc import score_tilt_residuals, suggest_bad_epochs
from stereo_melt.stack import load_basin_stack, save_stack

START, END, PREFIX = "2010-01-01", "2024-01-10", "test_stack"
_TMPDIRS: list[str] = []


def _decode(arr) -> list[str]:
    return [a.decode() if isinstance(a, bytes) else str(a) for a in arr]


def _make_stack(dem_ids, dates, *, with_dem_id=True, ny=2, nx=2) -> xr.DataArray:
    nt = len(dates)
    data = np.arange(nt * ny * nx, dtype=float).reshape(nt, ny, nx)
    da = xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={
            "time": pd.to_datetime(dates).values,
            "y": np.arange(ny, dtype=float),
            "x": np.arange(nx, dtype=float),
        },
        name="surface",
    )
    da = da.assign_coords(source_variant=("time", np.array(["v"] * nt)))
    if with_dem_id:
        da = da.assign_coords(dem_id=("time", np.asarray(dem_ids)))
    return da


def _write_and_load(da, **kw):
    """Write ``da`` to a UNIQUE temp dir and load it once, eagerly.

    A fresh dir per call gives each load a unique absolute path, so xarray's
    CachingFileManager never reuses a handle across loads. Repeatedly opening
    the *same* path in one process (and closing in between) corrupts that
    cache and segfaults netCDF4 (file_manager.py _acquire_with_cache_info).
    Production opens each stack path once per process, so this mirrors real
    usage. ``.load()`` materializes into memory so the data outlives cleanup.
    """
    d = tempfile.mkdtemp()
    _TMPDIRS.append(d)
    save_stack(da, os.path.join(d, f"{PREFIX}_{START}_{END}.nc"))
    st, _ = load_basin_stack(d, PREFIX, START, END, prefer_tilt_corrected=False, **kw)
    return st.load()


def test_strip_drop_with_dem_id():
    dates = ["2020-03-26", "2020-03-26", "2020-03-26", "2018-12-31", "2013-12-04"]
    ids = ["A", "B", "C", "D", "E"]
    da = _make_stack(ids, dates, with_dem_id=True)

    base = _write_and_load(da)
    assert "dem_id" in base.coords, "dem_id did not survive save/load!"
    print(f"      reloaded dem_id dtype={base['dem_id'].dtype} "
          f"-> {_decode(base['dem_id'].values)}")
    assert _decode(base["dem_id"].values) == ids

    # strip-level: drop B and D, spare same-day siblings A, C
    assert _decode(_write_and_load(da, bad_strips=("B", "D"))["dem_id"].values) \
        == ["A", "C", "E"]
    # date-level still drops the whole day (all three 2020-03-26 slices)
    assert _decode(_write_and_load(da, bad_epochs=("2020-03-26",))["dem_id"].values) \
        == ["D", "E"]
    # combined: union of a date drop and a strip drop
    assert _decode(_write_and_load(
        da, bad_epochs=("2013-12-04",), bad_strips=("B",))["dem_id"].values) \
        == ["A", "C", "D"]
    # unknown id -> nothing dropped (already absent; warns)
    assert _decode(_write_and_load(da, bad_strips=("ZZZ",))["dem_id"].values) == ids
    print("PASS  (1) strip-level drop spares clean same-day siblings")


def test_backward_compat_no_dem_id():
    dates = ["2020-03-26", "2020-03-26", "2013-12-04"]
    da = _make_stack(["A", "B", "C"], dates, with_dem_id=False)

    assert "dem_id" not in _write_and_load(da).coords
    # bad_strips skipped (warns), nothing dropped
    assert _write_and_load(da, bad_strips=("A",)).sizes["time"] == 3
    # bad_epochs still drops the whole date
    assert _write_and_load(da, bad_epochs=("2020-03-26",)).sizes["time"] == 1
    print("PASS  (2) no-dem_id stack: bad_strips skipped, bad_epochs intact")


def _join_scenario(with_dem_id):
    """A,B,C share 2020-03-26; B has a bad pc_align (end_p50=15); D is a
    +100 m blunder on 2018-12-31 (clean align)."""
    ny = nx = 3
    dates = ["2020-03-26", "2020-03-26", "2020-03-26", "2018-12-31"]
    ids = ["A", "B", "C", "D"]
    z = np.zeros((4, ny, nx))
    z[3] += 100.0  # D blunder over static
    tc = xr.DataArray(
        z, dims=("time", "y", "x"),
        coords={"time": pd.to_datetime(dates).values,
                "y": np.arange(ny), "x": np.arange(nx)},
        name="surface",
    )
    if with_dem_id:
        tc = tc.assign_coords(dem_id=("time", np.asarray(ids)))
    tp = xr.Dataset(
        {"weight_mean": ("time", np.ones(4)),
         "weight_frac_kept": ("time", np.ones(4))},
        coords={"time": tc["time"].values},
    )
    static = np.ones((ny, nx), dtype=bool)
    aq = pd.DataFrame({
        "dem_id": ids,
        "date": pd.to_datetime(dates),
        "end_p16": [0.2, 10.0, 0.2, 0.2],
        "end_p50": [0.3, 15.0, 0.3, 0.3],  # B bad; A,C clean on the SAME day
        "end_p84": [0.4, 20.0, 0.4, 0.4],
        "beg_p50": [0.3, 15.0, 0.3, 0.3],
    })
    return tc, tp, static, aq


def test_per_strip_join_and_flagging():
    tc, tp, static, aq = _join_scenario(with_dem_id=True)
    df = score_tilt_residuals(tc, tp, static, alignment_quality=aq)
    assert "dem_id" in df.columns
    ep = df.set_index("dem_id")["end_p50"].to_dict()
    # per-strip join: A,C keep their OWN 0.3 (not B's 15 -- the old bug)
    assert abs(ep["A"] - 0.3) < 1e-9 and abs(ep["C"] - 0.3) < 1e-9, ep
    assert abs(ep["B"] - 15.0) < 1e-9, ep
    flagged = set(suggest_bad_epochs(df, end_p50_threshold_m=10.0,
                                     catastrophic_resid_m=50.0)["dem_id"])
    assert flagged == {"B", "D"}, flagged  # B by align, D by blunder; A,C spared
    print("PASS  (3) per-strip join: only B(align)+D(blunder) flagged, A/C spared")


def test_legacy_join_overdrops():
    tc, tp, static, aq = _join_scenario(with_dem_id=False)
    df = score_tilt_residuals(tc, tp, static, alignment_quality=aq)
    assert "dem_id" not in df.columns
    sug = suggest_bad_epochs(df, end_p50_threshold_m=10.0, catastrophic_resid_m=50.0)
    # worst-strip-per-day collapse pushes B's 15 onto A and C -> all 3 of the
    # 2020-03-26 slices flagged (+ D). This is the over-drop dem_id fixes.
    assert len(sug) == 4, len(sug)
    flagged_dates = set(pd.to_datetime(sug["epoch"]).dt.normalize())
    assert flagged_dates == {pd.Timestamp("2020-03-26"), pd.Timestamp("2018-12-31")}
    print("PASS  (4) legacy date-join over-drops A,C (documents the bug fixed by 3)")


if __name__ == "__main__":
    try:
        test_strip_drop_with_dem_id()
        test_backward_compat_no_dem_id()
        test_per_strip_join_and_flagging()
        test_legacy_join_overdrops()
        print("\nALL STRIP-LEVEL SANITY CHECKS PASSED")
    finally:
        gc.collect()  # finalize netCDF handles before deleting their files
        for _d in _TMPDIRS:
            shutil.rmtree(_d, ignore_errors=True)

"""Test: pig.tilt_fit's opt-in switches, and that their defaults leave the
canon tilt fit exactly as it was.

Drives :func:`pig.tilt_fit.main` on a three-epoch in-memory stack with the
expensive stages (stack load, tide/IBE, MDT, geoid, mask builders, the tilt
LSQ, the writers) replaced by recorders, so what is asserted is the driver's
own dispatch: which observation domain it fits on, what it hands
``fit_tilt_stack``, and how per-epoch Ez and the P3 offset-only gate resolve.
No PIG data is read and no product is written.

T1  DEFAULTS (no switch set): the static-control domain, no dh/dt smoothness
    row, 8 IRLS iterations, and Ez + the P3 gate resolved by DATE -- the
    settings every canon PIG product was built with, so they must not move.
    By date, an epoch with no alignment row of its own is demoted to
    offset-only on a same-date strip's failed alignment.
T2  ``PIG_SOURCES=nocorr`` appends the no-control root LAST (so a granule that
    also has a real alignment keeps the aligned version) and, because that root
    is active, Ez and the P3 gate resolve by ``dem_id``: the control-free layer
    keeps its full x/y fit instead of inheriting the aligned strip's verdict.
T3  ``PIG_TILT_EZ_BY_DEM_ID=1`` does the same without a nocorr root, and says
    which switch asked for it.
T4  ``PIG_TILT_DOMAIN=full`` fits over the whole ice domain, warns when it is
    not paired with the dh/dt smoothness rows, and carries
    ``PIG_TILT_DHDT_SMOOTH`` / ``PIG_TILT_IRLS_MAX`` into the LSQ; an unknown
    domain is refused before any fitting.
T5  ``--pre-is2-asp`` / ``--is2-asp`` replace their own era's root only: any
    further root (the appended nocorr one) stays in the list, so by-dem_id Ez
    still sees it.
T6  By-dem_id resolution asked of a stack built before ``dem_id`` was carried
    falls back to date and says so loudly, because that is the leak.

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY examples/pig/test_tilt_switches.py
"""
from __future__ import annotations

import contextlib
import importlib
import io
import os
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd
import xarray as xr

# Resolve the library and the basin package from this file's own tree,
# ahead of the imports below.
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))
import stereo_melt  # noqa: E402,F401
import pig  # noqa: E402,F401
from pig import config, tilt_fit  # noqa: E402
from stereo_melt.coregister import alignment_quality  # noqa: E402

FAILS = []
_MISSING = object()

# Three epochs: two share a REMA date (no time of day in the filenames) and
# only the first of those carries a pc_align end_errors row.
TIMES = np.array(["2019-01-05", "2019-01-05", "2020-06-11"], dtype="datetime64[ns]")
DEM_IDS = np.array(["SETSM_aligned_20190105", "SETSM_nocorr_20190105",
                    "SETSM_aligned_20200611"])
BAD_STRIP_END_P50 = 9.0          # above the 3 m offset-only threshold
SWITCHES = ("PIG_SOURCES", "PIG_TILT_DOMAIN", "PIG_TILT_DHDT_SMOOTH",
            "PIG_TILT_IRLS_MAX", "PIG_TILT_EZ_BY_DEM_ID")


def as_list(v):
    """``None`` stays ``None``; anything else becomes a plain list to compare."""
    return None if v is None else list(v)


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def _stack(with_dem_id=True):
    ny, nx = 6, 5
    da = xr.DataArray(
        np.linspace(0.0, 1.0, TIMES.size * ny * nx).reshape(TIMES.size, ny, nx),
        dims=("time", "y", "x"),
        coords={"time": TIMES, "y": np.arange(ny, dtype=float) * -250.0,
                "x": np.arange(nx, dtype=float) * 250.0},
        name="surface",
    )
    if with_dem_id:
        da = da.assign_coords(dem_id=("time", DEM_IDS))
    return da


def _mask(stack, value, name):
    """A (y, x) bool mask that says which builder produced it."""
    m = xr.DataArray(np.full((stack.sizes["y"], stack.sizes["x"]), value, bool),
                     dims=("y", "x"),
                     coords={"y": stack.y, "x": stack.x})
    m.attrs["built_by"] = name
    return m


def run_main(env, *, with_dem_id=True, **kwargs):
    """Run ``tilt_fit.main`` with the heavy stages recorded instead of run.

    Returns ``(record, stdout, error)``: what the driver handed the LSQ and the
    Ez builder, everything it printed, and the ``SystemExit`` it raised (if any).
    """
    rec = {}
    stack = _stack(with_dem_id)
    control = _mask(stack, True, "static control")
    control.values[:, -1] = False
    ice = _mask(stack, True, "full ice domain")

    aq = pd.DataFrame({"dem_id": [DEM_IDS[0], DEM_IDS[2]],
                       "date": ["2019-01-05", "2020-06-11"],
                       "end_p50": [BAD_STRIP_END_P50, 0.4]})

    def fake_fit(st, **kw):
        rec["fit"] = kw
        n = st.sizes["time"]
        params = xr.Dataset(
            {"fit_xy": ("time", np.ones(n, bool)),
             "tilt_dx": ("time", np.full(n, 1e-6)),
             "tilt_dy": ("time", np.full(n, 2e-6)),
             "tilt_dz": ("time", np.full(n, 0.5))},
            coords={"time": st.time}, attrs={"robust": 0})
        return params, st

    def fake_ez(times, roots, **kw):
        rec["ez"] = {"times": np.asarray(times), "roots": list(roots), **kw}
        return np.full(len(times), 0.1), {"is2ctempoatmlvis": len(times)}

    saved_env = {k: os.environ.get(k) for k in SWITCHES}
    saved_sources = list(config.STRIP_SOURCES)
    # ``_MISSING`` keeps the harness usable against a build of the driver that
    # does not have one of these names at all, so it reports a failed check
    # rather than an AttributeError.
    saved = {n: getattr(tilt_fit, n, _MISSING) for n in
             ("_load_raw_stack", "apply_tide_ibe_to_stack", "apply_mdt_to_stack",
              "apply_geoid_to_stack", "build_static_area_polygon_mask",
              "build_static_control_mask", "build_ice_domain_mask",
              "build_per_epoch_ez", "fit_tilt_stack", "save_stack", "plot_tilt_params")}
    saved_aq = alignment_quality.aggregate_basin_quality
    saved_dirs = (config.PROCESSED_DIR, config.FIGURES_DIR, config.ensure_output_dirs)
    buf = io.StringIO()
    err = None
    try:
        for k in SWITCHES:
            os.environ.pop(k, None)
        os.environ.update(env)
        # PIG_SOURCES is read where it belongs, at config import.
        importlib.reload(config)
        rec["sources"] = list(config.STRIP_SOURCES)

        tilt_fit._load_raw_stack = lambda stack_prefix="pig_stack": (
            stack, pathlib.Path(f"{stack_prefix}_test.nc"))
        for name in ("apply_tide_ibe_to_stack", "apply_mdt_to_stack", "apply_geoid_to_stack"):
            setattr(tilt_fit, name, lambda st, **kw: st)
        tilt_fit.build_static_area_polygon_mask = lambda st, *a, **kw: _mask(st, True, "polygon")
        tilt_fit.build_static_control_mask = lambda st, **kw: control
        tilt_fit.build_ice_domain_mask = lambda st, *a, **kw: ice
        tilt_fit.build_per_epoch_ez = fake_ez
        tilt_fit.fit_tilt_stack = fake_fit
        tilt_fit.save_stack = lambda st, path: rec.setdefault("saved", path)
        tilt_fit.plot_tilt_params = lambda *a, **kw: None
        alignment_quality.aggregate_basin_quality = lambda sources: aq
        config.ensure_output_dirs = lambda: None
        with tempfile.TemporaryDirectory() as tmp:
            config.PROCESSED_DIR = pathlib.Path(tmp)
            config.FIGURES_DIR = pathlib.Path(tmp)
            try:
                with contextlib.redirect_stdout(buf):
                    tilt_fit.main(res_override=250.0, tag="test", **kwargs)
            except SystemExit as e:
                err = str(e)
    finally:
        for name, obj in saved.items():
            if obj is _MISSING:
                delattr(tilt_fit, name)
            else:
                setattr(tilt_fit, name, obj)
        alignment_quality.aggregate_basin_quality = saved_aq
        config.PROCESSED_DIR, config.FIGURES_DIR, config.ensure_output_dirs = saved_dirs
        config.STRIP_SOURCES = saved_sources
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return rec, buf.getvalue(), err


def main() -> int:
    print("T1  defaults: static domain, no smoothness row, 8 IRLS iters, resolved by DATE")
    rec, out, err = run_main({})
    check("no switch set: the driver reaches the LSQ", err is None and "fit" in rec, f"{err}")
    fit = rec.get("fit", {})
    check("observation domain = the static control mask",
          fit.get("observation_mask") is not None
          and fit["observation_mask"].attrs.get("built_by") == "static control",
          f"{fit.get('observation_mask').attrs if fit.get('observation_mask') is not None else None}")
    check("no dh/dt smoothness rows by default", fit.get("dhdt_smoothness") is None,
          f"dhdt_smoothness={fit.get('dhdt_smoothness')!r}")
    check("IRLS iteration cap stays at 8", fit.get("robust_max_iter") == 8,
          f"robust_max_iter={fit.get('robust_max_iter')!r}")
    check("Ez resolved by date: no dem_id handed to the Ez builder",
          rec["ez"].get("epoch_dem_ids") is None and "resolved by date" in out)
    check("P3 gate resolved by date: the same-date layer is demoted too",
          as_list(fit.get("offset_only_epochs")) == [True, True, False]
          and "resolved by date" in out,
          f"offset_only={list(fit.get('offset_only_epochs'))}")
    check("no nocorr root in play", [v for _d, v in rec["sources"]] == ["is2ctempoatmlvis",
                                                                       "ctempoatmlvis"],
          f"{[v for _d, v in rec['sources']]}")

    print("T2  PIG_SOURCES=nocorr: appended last, and Ez + the P3 gate key on dem_id")
    rec, out, err = run_main({"PIG_SOURCES": "nocorr"})
    check("the nocorr root is appended LAST, so an aligned granule still wins",
          [v for _d, v in rec["sources"]] == ["is2ctempoatmlvis", "ctempoatmlvis", "nocorr"],
          f"{[v for _d, v in rec['sources']]}")
    fit = rec.get("fit", {})
    check("the Ez builder is given the stack's dem_ids",
          as_list(rec["ez"].get("epoch_dem_ids")) == list(DEM_IDS) and "resolved by dem_id" in out)
    check("the control-free layer keeps its x/y fit instead of the aligned strip's verdict",
          as_list(fit.get("offset_only_epochs")) == [True, False, False],
          f"offset_only={list(fit.get('offset_only_epochs'))}")
    check("the figure-less run says so and names the reason",
          "nocorr root in STRIP_SOURCES" in out
          and "1/3 epochs have no pc_align end_errors row of their own" in out)
    check("the static domain and the canon LSQ settings are untouched by it",
          fit.get("observation_mask").attrs.get("built_by") == "static control"
          and fit.get("dhdt_smoothness") is None and fit.get("robust_max_iter") == 8)

    print("T3  PIG_TILT_EZ_BY_DEM_ID=1 alone does the same, and says which switch asked")
    rec, out, err = run_main({"PIG_TILT_EZ_BY_DEM_ID": "1"})
    check("by-dem_id without a nocorr root",
          as_list(rec["ez"].get("epoch_dem_ids")) == list(DEM_IDS)
          and as_list(rec["fit"].get("offset_only_epochs")) == [True, False, False]
          and "PIG_TILT_EZ_BY_DEM_ID=1" in out)

    print("T4  PIG_TILT_DOMAIN / PIG_TILT_DHDT_SMOOTH / PIG_TILT_IRLS_MAX")
    rec, out, err = run_main({"PIG_TILT_DOMAIN": "full"})
    check("full: the LSQ observes the whole ice domain",
          getattr(rec["fit"].get("observation_mask"), "attrs", {}).get("built_by") == "full ice domain",
          f"{getattr(rec['fit'].get('observation_mask'), 'attrs', {})}")
    check("full without the smoothness rows warns about the aliasing nullspace",
          "⚠ full domain WITHOUT dh/dt smoothness" in out and "PIG_TILT_DHDT_SMOOTH=1.0" in out)
    check("the control mask is still the control mask (only the observations widen)",
          getattr(rec["fit"].get("control_mask"), "attrs", {}).get("built_by") == "static control")
    rec, out, err = run_main({"PIG_TILT_DOMAIN": "full", "PIG_TILT_DHDT_SMOOTH": "1.0",
                              "PIG_TILT_IRLS_MAX": "3"})
    check("the paired run carries both settings into the LSQ and drops the warning",
          rec["fit"].get("dhdt_smoothness") == 1.0 and rec["fit"].get("robust_max_iter") == 3
          and "⚠" not in out, f"{rec['fit'].get('dhdt_smoothness')}, {rec['fit'].get('robust_max_iter')}")
    check("and both are reported", "dh/dt smoothness weight: 1 " in out
          and "IRLS max iterations: 3" in out)
    rec, out, err = run_main({"PIG_TILT_DOMAIN": "both"})
    check("an unknown domain is refused before any fitting",
          err is not None and "PIG_TILT_DOMAIN must be 'static' or 'full'" in err
          and "fit" not in rec, f"SystemExit: {err!s:.70}")

    print("T5  --pre-is2-asp / --is2-asp replace one era's root and keep the others")
    rec, out, err = run_main({"PIG_SOURCES": "nocorr"}, pre_is2_asp="ctempoatmlvis_v2")
    check("the pre-IS2 override leaves by-dem_id Ez in force",
          as_list(rec["ez"].get("epoch_dem_ids")) == list(DEM_IDS))
    check("...and the Ez builder is handed all three roots, nocorr last",
          [suf for _p, suf in rec['ez']['roots']]
          == ["is2ctempoatmlvis", "ctempoatmlvis_v2", "nocorr"],
          f"{[suf for _p, suf in rec['ez']['roots']]}")
    rec, out, err = run_main({"PIG_SOURCES": "nocorr"}, is2_asp="is2ctempoatmlvis_v2")
    check("the IS2-era override keeps it too",
          [suf for _p, suf in rec['ez']['roots']]
          == ["is2ctempoatmlvis_v2", "ctempoatmlvis", "nocorr"],
          f"{[suf for _p, suf in rec['ez']['roots']]}")

    print("T6  a stack built before dem_id was carried: fall back to date, loudly")
    rec, out, err = run_main({"PIG_SOURCES": "nocorr"}, with_dem_id=False)
    check("no dem_id coord: Ez falls back to date",
          rec["ez"].get("epoch_dem_ids") is None and "resolved by date" in out)
    check("the leak it reopens is stated, with what to do about it",
          "WARNING: by-dem_id resolution requested" in out and "Rebuild the stack" in out)
    check("and the P3 gate demotes the same-date layer again",
          as_list(rec["fit"].get("offset_only_epochs")) == [True, True, False],
          f"offset_only={list(rec['fit']['offset_only_epochs'])}")

    print()
    if FAILS:
        print(f"TEST FAILED: {FAILS}")
        return 1
    print("TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

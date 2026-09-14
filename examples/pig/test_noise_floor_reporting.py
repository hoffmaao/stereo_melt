"""Test: the noise-floor reporting guards and the PIG_VELOCITY fallback warning.

Exercises the public functions of :mod:`pig.plot_noise_floor` and
:mod:`pig.run_melt` on synthetic in-memory datasets — no PIG data is read.

T1  ``crossing``: a non-positive SNR bin (product PSD below the noise floor)
    is a bin definitively BELOW unity, so the SNR = 1 crossing is
    interpolated to it — never past it. The prototype dropped such bins and
    interpolated to the next positive one, which can only move the crossing
    toward SHORTER wavelength. Non-finite bins are still dropped; no crossing
    in band is NaN.
T2  ``available_pairs``: halves written by ``run_noise_floor --skip-rb``
    (Eulerian only) are usable with a full-product variant suffix — the
    restored pair is skipped for missing halves BEFORE the variant check —
    while a genuinely requested variant the full product lacks is a loud
    error, never a silent fall back to the default product.
T3  ``check_provenance``: common-epoch halves against a default-path full
    product (the mismatched instrument that produced the recorded prototype
    numbers) are refused from the files' own attrs, whatever flags were
    passed; matching provenance is returned for labelling; a velocity
    mismatch is refused; a missing velocity attr only warns. Halves solved
    from another stack/mask ``tag`` are refused even when the velocity
    string is identical; an untagged file states nothing about its stack
    unless its name is the un-suffixed default that predates ``--tag``, so
    it is refused by name rather than read from that name; ``--assume-tag``
    states it explicitly and is then checked like a stamped tag, and it
    stands in for one side only, so two untagged files stay refused. The
    check reports which side (if any) the assumption stood in for, so a
    caller labels its figure with the caveat only when one was needed.
T4  ``load_velocity_on_grid``: with PIG_VELOCITY unset it warns
    (RuntimeWarning naming the fallback and the production choice) before
    touching any velocity file; with it set, no warning is raised and the
    requested source is honoured.

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY examples/pig/test_noise_floor_reporting.py
"""
from __future__ import annotations

import contextlib
import io
import os
import pathlib
import sys
import warnings

import numpy as np
import xarray as xr

# Resolve the library and the basin package from this file's own tree,
# ahead of the imports below.
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))
import stereo_melt  # noqa: E402,F401
import pig  # noqa: E402,F401
from pig import config  # noqa: E402
from pig.plot_noise_floor import available_pairs, check_provenance, crossing  # noqa: E402
from pig.run_melt import load_velocity_on_grid  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def ds(names, attrs=None, var_attrs=None):
    var_attrs = var_attrs or {}
    return xr.Dataset({k: xr.DataArray(np.zeros((2, 2)), dims=("y", "x"),
                                       attrs=var_attrs.get(k, {})) for k in names},
                      attrs=attrs or {})


def system_exit(fn, *a, **kw):
    """Run ``fn``; return (SystemExit message or None, captured stdout)."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            fn(*a, **kw)
    except SystemExit as e:
        return str(e), buf.getvalue()
    return None, buf.getvalue()


def main() -> int:
    print("T1  crossing(): SNR <= 0 bins are below unity, not dropped")
    k = np.array([0.1, 0.2, 0.4, 0.8])
    snr = np.array([3.0, 1.5, -0.5, 0.2])
    kc = crossing(k, snr)
    expect = 0.2 * 2 ** ((1.5 - 1.0) / (1.5 - (-0.5)))     # interpolate 1.5@0.2 -> -0.5@0.4
    proto = 0.2 * 4 ** ((1.5 - 1.0) / (1.5 - 0.2))         # prototype: 1.5@0.2 -> 0.2@0.8
    check("crossing interpolates to the first SNR<=0 bin",
          abs(kc - expect) < 1e-12 and kc < proto,
          f"lambda {1 / kc:.3f} km (prototype would give {1 / proto:.3f} km, shorter)")
    kc2 = crossing(np.array([0.1, 0.2, 0.4]), np.array([3.0, np.nan, 0.5]))
    expect2 = 0.1 * 4 ** ((3.0 - 1.0) / (3.0 - 0.5))
    check("non-finite bins are dropped", abs(kc2 - expect2) < 1e-12, f"k {kc2:.4f}")
    check("never below unity -> NaN", np.isnan(crossing(k, np.array([3.0, 2.0, 1.5, 1.2]))))
    check("below unity at the longest wavelength -> NaN",
          np.isnan(crossing(k, np.array([0.5, 2.0, 0.5, 0.2]))))

    print("T2  available_pairs(): skip-rb halves vs a requested full-product variant")
    full = ds(["eulerian", "restored_local_helm", "eulerian_ce"])
    half_eu = ds(["eulerian_A", "eulerian_B"])
    half_all = ds(["eulerian_A", "eulerian_B", "rb_A", "rb_B"])
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        pairs = available_pairs(full, half_eu, "_ce")
    check("Eulerian-only halves + '_ce' select eulerian_ce and skip the rb pair",
          [p[1] for p in pairs] == ["eulerian_ce"] and "skipping" in buf.getvalue(),
          f"pairs {[p[1] for p in pairs]}")
    with contextlib.redirect_stdout(io.StringIO()):
        pairs0 = available_pairs(full, half_all, "")
    check("no suffix: both default-path pairs",
          [p[1] for p in pairs0] == ["eulerian", "restored_local_helm"])
    err, _ = system_exit(available_pairs, full, half_all, "_ce")
    check("requested variant absent from the full product is a loud error",
          err is not None and "restored_local_helm_ce" in err and "--full-var-suffix" in err,
          f"SystemExit: {err!s:.70}")
    err, _ = system_exit(available_pairs, ds(["eulerian"]), ds(["rb_A", "rb_B"]), "")
    check("no complete pair at all is an error", err is not None and "nothing to plot" in err)

    print("T3  check_provenance(): the files' attrs decide, not the caller's flags")
    V, T = "fused-test-velocity", "is2ctempo_sheltilt"
    full = ds(["eulerian", "eulerian_ce"], attrs={"velocity": V, "tag": T},
              var_attrs={"eulerian_ce": {"common_epoch": 1}})
    half_ce = ds(["eulerian_A", "eulerian_B"],
                 attrs={"velocity": V, "common_epoch": 1, "tag": T})
    half_def = ds(["eulerian_A", "eulerian_B"],
                  attrs={"velocity": V, "common_epoch": 0, "tag": T})
    pair_def = [("Eulerian", "eulerian", "eulerian_A", "eulerian_B", "c")]
    pair_ce = [("Eulerian", "eulerian_ce", "eulerian_A", "eulerian_B", "c")]
    err, _ = system_exit(check_provenance, full, half_ce, pair_def)
    check("common-epoch halves vs default full product are refused",
          err is not None and "instrument mismatch" in err
          and "eulerian_A common_epoch=1" in err and "common_epoch=0" in err,
          f"SystemExit: {err!s:.60}")
    err, out = system_exit(check_provenance, full, half_ce, pair_ce)
    with contextlib.redirect_stdout(io.StringIO()):
        agreed, assumed = check_provenance(full, half_ce, pair_ce)
    check("matching common-epoch provenance is accepted and returned",
          err is None and agreed == [(1, V)] and assumed is None, f"{agreed}, assumed {assumed!r}")
    with contextlib.redirect_stdout(io.StringIO()):
        agreed0, _ = check_provenance(full, half_def, pair_def)
    check("matching default provenance is accepted and returned", agreed0 == [(0, V)], f"{agreed0}")
    half_v = ds(["eulerian_A", "eulerian_B"],
                attrs={"velocity": "other", "common_epoch": 0, "tag": T})
    err, _ = system_exit(check_provenance, full, half_v, pair_def)
    check("velocity mismatch is refused", err is not None and "velocity='other'" in err,
          f"SystemExit: {err!s:.60}")
    half_nov = ds(["eulerian_A", "eulerian_B"], attrs={"common_epoch": 0, "tag": T})
    err, out = system_exit(check_provenance, full, half_nov, pair_def)
    check("missing velocity provenance warns but does not refuse",
          err is None and "WARNING" in out and "cannot verify" in out)
    full_t = ds(["eulerian"], attrs={"velocity": V, "tag": "is2ctempo_sheltilt"})
    half_t = ds(["eulerian_A", "eulerian_B"],
                attrs={"velocity": V, "common_epoch": 0, "tag": "is2ctempo_sheltilt"})
    half_qcey = ds(["eulerian_A", "eulerian_B"],
                   attrs={"velocity": V, "common_epoch": 0, "tag": "is2ctempo_sheltilt_qcey"})
    err, _ = system_exit(check_provenance, full_t, half_qcey, pair_def)
    check("halves from another stack tag are refused though the velocity matches",
          err is not None and "instrument mismatch" in err
          and "is2ctempo_sheltilt_qcey" in err, f"SystemExit: {err!s:.60}")
    check("matching tags are accepted",
          check_provenance(full_t, half_t, pair_def) == ([(0, V)], None))
    check("both tags stamped: --assume-tag is unused, so nothing is reported as assumed",
          check_provenance(full_t, half_t, pair_def,
                           assume_tag="is2ctempo_sheltilt_qcey") == ([(0, V)], None))
    def named(names, name, **attrs):
        """A product predating the tag attr: no tag, only a filename."""
        d = ds(names, attrs={"velocity": V, "common_epoch": 0, **attrs})
        d.encoding["source"] = f"/processed/{name}"
        return d

    # The pre-attr half-stacks on disk include runs of other stacks whose names
    # begin with the canon tag, so an out-suffix is no evidence of a stack; only
    # the un-suffixed default name predates --tag and must be the canon stack.
    default_half = named(["eulerian_A", "eulerian_B"],
                         "pig_noise_floor_250m_is2ctempo_sheltilt.nc")
    check("the un-suffixed default half predates --tag, so it is the canon stack",
          check_provenance(full_t, default_half, pair_def) == ([(0, V)], None))
    for suffix in ("_ce", "_q", "_is2ctempo_sheltilt_qcey"):
        name = f"pig_noise_floor_250m_is2ctempo_sheltilt{suffix}.nc"
        err, _ = system_exit(check_provenance, full_t,
                             named(["eulerian_A", "eulerian_B"], name), pair_def)
        check(f"an untagged half with an out-suffix is refused: ...{suffix}",
              err is not None and name in err and "--assume-tag" in err
              and "cannot be established" in err, f"SystemExit: {err!s:.60}")
    qcey_half = named(["eulerian_A", "eulerian_B"],
                      "pig_noise_floor_250m_is2ctempo_sheltilt_is2ctempo_sheltilt_qcey.nc")
    agreed_a, assumed_a = check_provenance(full_t, qcey_half, pair_def,
                                           assume_tag="is2ctempo_sheltilt")
    check("--assume-tag states the missing tag and the comparison proceeds",
          agreed_a == [(0, V)], f"{agreed_a}")
    check("the untagged side is reported, so only then is the caveat labelled",
          assumed_a == qcey_half.encoding["source"].rsplit("/", 1)[-1], f"assumed {assumed_a!r}")
    err, _ = system_exit(check_provenance, full_t, qcey_half, pair_def,
                         assume_tag="is2ctempo_sheltilt_qcey")
    check("an assumed tag is checked like a stamped one, not trusted blindly",
          err is not None and "tag='is2ctempo_sheltilt_qcey'" in err, f"SystemExit: {err!s:.60}")
    quarters = named(["eulerian_Q0"], "pig_noise_floor_250m_is2ctempo_sheltilt_q.nc")
    ladder = [("ladder", "eulerian_Q0", "eulerian_A", "eulerian_B", None)]
    err, _ = system_exit(check_provenance, quarters, half_t, ladder, full_name="quarters")
    check("an untagged quarters file is named in the refusal too",
          err is not None and "is2ctempo_sheltilt_q.nc" in err, f"SystemExit: {err!s:.60}")
    check("with the tag assumed, tagged halves and an untagged quarters file compare",
          check_provenance(quarters, half_t, ladder, full_name="quarters",
                           assume_tag="is2ctempo_sheltilt")
          == ([(0, V)], "pig_noise_floor_250m_is2ctempo_sheltilt_q.nc"))
    # The ladder pair that motivated the override: canon quarters, qcey halves,
    # neither carrying a tag. One assertion cannot vouch for both sides.
    err, _ = system_exit(check_provenance, quarters, qcey_half, ladder,
                         full_name="quarters", assume_tag="is2ctempo_sheltilt")
    check("--assume-tag cannot fill BOTH sides: two untagged files stay refused",
          err is not None and "one side only" in err
          and "is2ctempo_sheltilt_q.nc" in err and "qcey.nc" in err,
          f"SystemExit: {err!s:.60}")
    err, _ = system_exit(check_provenance, quarters, qcey_half, ladder, full_name="quarters")
    check("...and are refused without the override too", err is not None,
          f"SystemExit: {err!s:.60}")

    print("T4  load_velocity_on_grid(): PIG_VELOCITY unset warns; set is silent")
    dummy = xr.DataArray(np.zeros((2, 2)), dims=("y", "x"),
                         coords={"y": [1.0, 0.0], "x": [0.0, 1.0]})
    saved = os.environ.pop("PIG_VELOCITY", None)
    saved_nc = config.ITS_LIVE_2019
    try:
        # Warnings raised as errors: the warn call aborts the function BEFORE
        # any velocity file is opened, so this reads no data.
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                load_velocity_on_grid(dummy)
                raised = None
            except BaseException as e:  # noqa: BLE001
                raised = e
        msg = str(raised)
        check("unset PIG_VELOCITY raises the RuntimeWarning before any velocity IO",
              isinstance(raised, RuntimeWarning) and "PIG_VELOCITY" in msg
              and "'measures'" in msg and "'fused'" in msg,
              f"{type(raised).__name__}: {msg[:70]}...")
        # Set: route to a source whose file we point at a nonexistent path, so
        # the function reaches its own (silent) SystemExit without IO.
        os.environ["PIG_VELOCITY"] = "its_live"
        config.ITS_LIVE_2019 = pathlib.Path("/nonexistent/test_noise_floor_reporting.nc")
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                load_velocity_on_grid(dummy)
                raised = None
            except BaseException as e:  # noqa: BLE001
                raised = e
        check("set PIG_VELOCITY: no warning, the requested source is honoured",
              isinstance(raised, SystemExit) and "ITS_LIVE requested" in str(raised),
              f"{type(raised).__name__}: {str(raised)[:60]}")
    finally:
        config.ITS_LIVE_2019 = saved_nc
        if saved is None:
            os.environ.pop("PIG_VELOCITY", None)
        else:
            os.environ["PIG_VELOCITY"] = saved

    print()
    if FAILS:
        print(f"TEST FAILED: {FAILS}")
        return 1
    print("TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

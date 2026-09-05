"""Gate: the per-strip residual plane against independent control, and the
coloured-noise prior built from it (2026-09-02).

Why this exists. The melt inverse's strip-noise model needs the variance of
the residual per-strip plane error, and nothing built on the DEM stack alone
can give it: the per-strip tilt is not identifiable from the stack (loosening
the tilt prior grows the estimates without bound while their correlation
with truth FALLS). Independent altimetric control has no such degeneracy.
This gate checks the machinery on a synthetic where the answer is known:

C1  the plane fit recovers each strip's residual (offset, ax, ay) against
    control, with honest standard errors;
C2  the population tau^2 is the true across-strip variance -- INCLUDING the
    method-of-moments correction: with noisy, badly-spread control the naive
    var(est) overshoots, and subtracting mean(se^2) brings it back;
C3  the prior maps onto strip_mode_design's (strip_index, component) modes,
    per-strip and population; offset modes refuse to run without offset_var.
C0  a control CSV that EXISTS but cannot be read must not look like one that
    is absent: absent -> None (the strip is uncontrolled), unreadable schema
    -> ValueError naming the file. Conflating them reports a basin-wide
    column-name mismatch as "no control anywhere".
C5  control points OUTSIDE a strip's footprint sample as NaN, whatever the
    raster's nodata tag says -- rasterio fills out-of-grid points with
    (nodata or 0), so an untagged or 0-nodata DEM would otherwise return a
    real 0.0 m elevation and turn an overhanging control cloud into a
    metres-scale fake residual. Includes the exact right/bottom bounds, which
    rasterio's flooring rowcol puts one pixel PAST the grid.
C6  the QC classes are reported separately AND reconcile: sd > max_sd is an
    alignment failure (a BAD_STRIPS candidate), n < min_n is a well-aligned
    strip whose control clips the footprint (must NOT be), and a strip with no
    usable plane at all is its own class. Pooling the first two is how a good
    epoch gets discarded; leaving the third out of every list is how a
    missing-control strip becomes invisible. The four classes must partition
    the input table exactly.

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY tests/gate_control_strip_prior.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

X0C = Y0C = 0.0  # grid centre of the last make_strips call

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from stereo_melt.coregister.alignment_quality import (  # noqa: E402
    _read_xyh_csv,
    fit_residual_plane,
    residual_planes_for_strips,
    sample_dem_at_points,
)
from stereo_melt.dynamics.budget_bridging import strip_prior_from_residual_planes  # noqa: E402

FAILS: list[str] = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def make_strips(tmp, n_strips, *, tau_x, tau_y, tau_c, dem_noise, ctl_noise,
                ctl_spread_y, n_ctl, seed):
    """Synthetic aligned DEMs = truth (0) + plane + white noise; control =
    0 + white noise, restricted to a band of half-height ctl_spread_y (so the
    y-slope is poorly constrained, like control along one strip edge)."""
    rng = np.random.default_rng(seed)
    res, nx, ny = 50.0, 300, 200
    x0, y0 = -1_600_000.0, -290_000.0
    tr = from_origin(x0, y0, res, res)
    xs = x0 + (np.arange(nx) + 0.5) * res
    ys = y0 - (np.arange(ny) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    xc, yc = X.mean(), Y.mean()
    global X0C, Y0C
    X0C, Y0C = float(xc), float(yc)
    items, truth = [], []
    for k in range(n_strips):
        ax, ay, c = rng.normal(0, tau_x), rng.normal(0, tau_y), rng.normal(0, tau_c)
        dem = c + ax * (X - xc) + ay * (Y - yc) + rng.normal(0, dem_noise, X.shape)
        p = tmp / f"S{k:03d}-trans_reference-DEM.tif"
        with rasterio.open(p, "w", driver="GTiff", height=ny, width=nx, count=1,
                           dtype="float32", transform=tr, nodata=-9999.0) as dst:
            dst.write(dem.astype(np.float32), 1)
        ce = rng.uniform(xs.min() + res, xs.max() - res, n_ctl)
        cn = yc + rng.uniform(-ctl_spread_y, ctl_spread_y, n_ctl)
        ch = rng.normal(0, ctl_noise, n_ctl)               # truth 0 + control noise
        items.append((f"S{k:03d}", p, np.column_stack([ce, cn, ch])))
        truth.append((c, ax, ay))
    return items, np.array(truth)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="gate_ctl_prior_"))
    try:
        return _run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run(tmp: Path) -> int:
    print("C0  absent, unusable and valid control CSVs are three outcomes")
    rows = "\n".join(f"{-1600000.0 + i * 50},{-290000.0 - i * 50},{40.0 + i}"
                     for i in range(5))
    # Every layout the reader documents as accepted must parse to the same
    # coordinates, including a headerless row 0 with a GAP -- a missing cell is
    # not evidence of a header.
    gap_rows = ("-1600000.0,,40.0\n"
                "-1599950.0,-290050.0,41.0\n"
                "-1599900.0,-290100.0,42.0")
    ok_cases = {
        "headerless": (rows, 5),
        "expected header": ("easting,northing,h_mean\n" + rows, 5),
        "commented header": ("# easting,northing,height\n" + rows, 5),
        "capitalised header": ("Easting,Northing,H_mean\n" + rows, 5),
        "reordered header": ("h_mean,easting,northing\n" + "\n".join(
            f"{40.0 + i},{-1600000.0 + i * 50},{-290000.0 - i * 50}"
            for i in range(5)), 5),
        "unrecognised header": ("x,y,z\n" + rows, 5),
        "headerless, gap in row 0": (gap_rows, 2),
    }
    for name, (txt, n_rows) in ok_cases.items():
        pth = tmp / f"ctl_ok_{name.replace(' ', '_').replace(',', '')}.csv"
        pth.write_text(txt + "\n")
        try:
            got = _read_xyh_csv(pth)
        except Exception as exc:            # noqa: BLE001
            check(f"{name}: parses", False, f"{type(exc).__name__}: {str(exc)[:70]}")
            continue
        check(f"{name}: parses to ({n_rows}, 3) with easting first",
              got is not None and got.shape == (n_rows, 3)
              and got[0, 0] < -1e6 and got[0, 2] < 1e3,
              "None" if got is None else f"{got.shape} first row {got[0]}")
    check("an absent file is None -- the ONLY case that returns None",
          _read_xyh_csv(tmp / "does_not_exist.csv") is None)

    # A NAMED coordinate column that will not parse is an error in the file.
    # Substituting another column positionally returns different coordinates
    # under a valid-looking header -- here the n_src counts would arrive as
    # heights, ~35 m wrong, and be reported downstream as a broken alignment.
    named_bad = {
        "named h_mean has a sentinel token": (
            "easting,northing,h_mean,n_src\n"
            "-1600000.0,-290000.0,40.0,3\n"
            "-1599950.0,-290050.0,41.0,4\n"
            "-1599900.0,-290100.0,NODATA,5\n"
            "-1599850.0,-290150.0,43.0,6"),
        "named h_mean is entirely empty": (
            "easting,northing,h_mean,n_src\n"
            "-1600000.0,-290000.0,,3\n"
            "-1599950.0,-290050.0,,4\n"
            "-1599900.0,-290100.0,,5"),
    }
    for name, txt in named_bad.items():
        pth = tmp / f"ctl_named_{name.replace(' ', '_')}.csv"
        pth.write_text(txt + "\n")
        try:
            got = _read_xyh_csv(pth)
        except ValueError as exc:
            check(f"{name}: raises naming the column, no positional substitute",
                  "h_mean" in str(exc), str(exc).split(": ", 1)[-1][:72])
        except Exception as exc:            # noqa: BLE001
            check(f"{name}: raises naming the column, no positional substitute",
                  False, f"{type(exc).__name__}: {exc}")
        else:
            check(f"{name}: raises naming the column, no positional substitute",
                  False, f"returned heights {got[:, 2]} -- wrong column")
    # ...but a GAP in a named column is fine: the incomplete row is dropped.
    gap_named = tmp / "ctl_named_gap.csv"
    gap_named.write_text("easting,northing,h_mean,n_src\n"
                         "-1600000.0,-290000.0,40.0,3\n"
                         "-1599950.0,-290050.0,,4\n"
                         "-1599900.0,-290100.0,42.0,5\n")
    got = _read_xyh_csv(gap_named)
    check("a gap in a named column drops that row, keeps the named heights",
          got is not None and got.shape == (2, 3)
          and abs(got[0, 2] - 40.0) < 1e-12 and abs(got[1, 2] - 42.0) < 1e-12,
          "None" if got is None else f"{got.shape} heights {got[:, 2]}")

    # Present but unusable must RAISE on every path, naming the file. Before
    # this, an unreadable file failed on the probe read and came back None,
    # i.e. exactly what an absent file returns.
    unreadable = tmp / "ctl_unreadable.csv"
    unreadable.write_text("easting,northing,h_mean\n" + rows + "\n")
    os.chmod(unreadable, 0o000)
    # Mode bits are not honoured for root, nor on some overlay/NFS/CIFS mounts.
    # Probe the precondition instead of assuming it, so this gate cannot go red
    # on one host and green on another for a reason that is not about the reader.
    try:
        with open(unreadable, "rb"):
            readable_anyway = True
    except OSError:
        readable_anyway = False
    bad_cases = {}
    if readable_anyway:
        print("      (skipping the unreadable-file case: mode 000 is still "
              "readable here, so the environment cannot express it)")
    else:
        bad_cases["unreadable (mode 000)"] = unreadable
    two_col = tmp / "ctl_bad_two_columns.csv"
    two_col.write_text("easting,northing\n" + "\n".join(
        f"{-1600000.0 + i * 50},{-290000.0 - i * 50}" for i in range(5)) + "\n")
    bad_cases["two columns"] = two_col
    hdr_only = tmp / "ctl_bad_header_only.csv"
    hdr_only.write_text("easting,northing,h_mean\n")
    bad_cases["header, no data rows"] = hdr_only
    empty = tmp / "ctl_bad_empty.csv"
    empty.write_text("")
    bad_cases["empty file"] = empty
    try:
        for name, pth in bad_cases.items():
            try:
                got = _read_xyh_csv(pth)
            except ValueError as exc:
                check(f"{name}: raises ValueError naming the file",
                      str(pth) in str(exc), str(exc)[:78])
            except Exception as exc:        # noqa: BLE001
                check(f"{name}: raises ValueError naming the file", False,
                      f"{type(exc).__name__}: {exc}")
            else:
                check(f"{name}: raises ValueError naming the file", False,
                      f"returned {got if got is None else got.shape} "
                      "-- indistinguishable from a missing file")
    finally:
        os.chmod(unreadable, 0o644)

    print("C1  per-strip plane recovery against control (well-spread, quiet control)")
    items, truth = make_strips(tmp, 40, tau_x=1.5e-4, tau_y=1.5e-4, tau_c=0.3,
                               dem_noise=0.4, ctl_noise=0.1, ctl_spread_y=5000.0,
                               n_ctl=800, seed=1)
    df = residual_planes_for_strips(items)
    # The fit reports the offset at the CONTROL centroid (xc, yc); the strip's
    # offset is defined at the grid centre, so refer the truth there --
    # otherwise the plane times the ~150 m centroid shift reads as error.
    off_true = truth[:, 0] + truth[:, 1] * (df.xc.to_numpy() - X0C) + truth[:, 2] * (df.yc.to_numpy() - Y0C)
    for j, col in enumerate(("offset", "ax", "ay")):
        est, tru = df[col].to_numpy(), (off_true if col == "offset" else truth[:, j])
        corr = np.corrcoef(est, tru)[0, 1]
        slope = np.polyfit(tru, est, 1)[0]
        check(f"{col}: corr > 0.98 and slope within 5 %", corr > 0.98 and abs(slope - 1) < 0.05,
              f"corr {corr:.4f} slope {slope:.3f}")
    # standard errors are honest: z-scores of (est - true)/se have unit spread
    z = np.concatenate([(df[c].to_numpy() - (off_true if c == "offset" else truth[:, j]))
                        / df[f"se_{c}"].to_numpy() for j, c in enumerate(("offset", "ax", "ay"))])
    check("standard errors honest (rms z in [0.7, 1.4])", 0.7 < np.sqrt(np.mean(z ** 2)) < 1.4,
          f"rms z {np.sqrt(np.mean(z ** 2)):.2f}")

    print("C2  population tau^2 with noisy, one-edge control (the correction matters)")
    # Control confined to a 2 km band with 0.6 m noise: the y-slope's
    # estimator variance is ~0.3-0.4x the signal, so naive var(est) overshoots
    # by ~15-45 % and the moment correction is both material and stable
    # (4 seeds: corrected 0.86-1.08). Pushing the estimator variance to 2-3x
    # the signal makes var(est) - mean(se^2) a difference of two large noisy
    # numbers that no estimator resolves with ~100 strips -- not a regime to
    # gate on.
    # 200 strips: the population rule is MAD-based (robust to the alignment
    # failures real ASP roots carry -- C4), and MAD^2 has ~1.4x the sampling
    # spread of a variance, so tolerances below are set at ~2 sigma for N2.
    N2 = 200
    items2, truth2 = make_strips(tmp, N2, tau_x=1.5e-4, tau_y=1.5e-4, tau_c=0.3,
                                 dem_noise=0.6, ctl_noise=0.6, ctl_spread_y=1000.0,
                                 n_ctl=300, seed=2)
    df2 = residual_planes_for_strips(items2)
    sidx = np.repeat(np.arange(N2), 2)
    comp = np.tile(np.array(["tilt_x", "tilt_y"]), N2)
    tau2, summ = strip_prior_from_residual_planes(df2, df2.dem_id.values, sidx, comp,
                                                  return_summary=True)
    true_var = {"tilt_x": float(np.var(truth2[:, 1])), "tilt_y": float(np.var(truth2[:, 2]))}
    for c in ("tilt_x", "tilt_y"):
        naive = summ[c]["var_raw"] / true_var[c]
        corrected = summ[c]["tau2"] / true_var[c]
        print(f"      {c}: naive var(est)/true {naive:.2f}  corrected tau2/true {corrected:.2f}"
              f"   (mean se^2 / true {summ[c]['mean_se2'] / true_var[c]:.2f})")
        check(f"{c}: corrected tau2 within 25 % of truth", abs(corrected - 1) < 0.25,
              f"{corrected:.2f}")
    nv, cv = summ["tilt_y"]["var_raw"] / true_var["tilt_y"], summ["tilt_y"]["tau2"] / true_var["tilt_y"]
    check("tilt_y: naive overshoots by > 20 % and the correction lands closer",
          nv > 1.2 and abs(cv - 1) < abs(nv - 1), f"naive {nv:.2f} -> corrected {cv:.2f}")
    check("tau2 broadcast per mode (population = same value per component)",
          np.allclose(tau2[comp == "tilt_x"], summ["tilt_x"]["tau2"])
          and np.allclose(tau2[comp == "tilt_y"], summ["tilt_y"]["tau2"]))

    print("C3  mode mapping, per-strip variant, offset guard")
    # stack epoch order differs from the table order: mapping must go via dem_id
    perm = np.random.default_rng(3).permutation(N2)
    dem_ids = df2.dem_id.values[perm]
    sidx3 = np.array([5, 5, 17, 17, 40])
    comp3 = np.array(["tilt_x", "tilt_y", "tilt_x", "tilt_y", "tilt_x"])
    t_pop = strip_prior_from_residual_planes(df2, dem_ids, sidx3, comp3)
    t_ps = strip_prior_from_residual_planes(df2, dem_ids, sidx3, comp3, per_strip=True)
    row = df2.set_index("dem_id").loc[dem_ids[5]]
    exp = max(row.ax ** 2 - row.se_ax ** 2, 0.05 * summ["tilt_x"]["tau2"])
    check("per-strip tau2 resolves the strip through dem_id, not table order",
          np.isclose(t_ps[0], exp), f"{t_ps[0]:.3e} vs {exp:.3e}")
    check("population variant ignores the strip", np.allclose(t_pop[[0, 2, 4]], t_pop[0]))
    try:
        strip_prior_from_residual_planes(df2, dem_ids, np.array([0]), np.array(["offset"]))
        check("offset mode without offset_var raises", False)
    except ValueError as exc:
        check("offset mode without offset_var raises", "offset_var" in str(exc))
    ov = np.linspace(0.1, 0.5, N2) ** 2
    t_off = strip_prior_from_residual_planes(df2, dem_ids, np.array([7, 7]),
                                             np.array(["offset", "tilt_x"]), offset_var=ov)
    check("offset mode takes the per-epoch offset_var", np.isclose(t_off[0], ov[7]))

    print("C4  robustness to alignment failures (real ASP roots have them)")
    # PIG canon: ~15 % of strips are pc_align failures with residual offsets
    # of 60-207 m and scatter of 68-175 m; they carried 100 % of a
    # non-robust variance. Inject the same and require the population tau2
    # to be unmoved while the naive one blows up, and the QC to name them.
    bad = df2.copy()
    nb = 20   # 10 % of the population
    bad.loc[: nb - 1, ["offset", "ax", "ay", "sd"]] = [[150.0, 0.02, -0.03, 90.0]] * nb
    t_rob, s_rob = strip_prior_from_residual_planes(bad, bad.dem_id.values, sidx, comp, return_summary=True)
    t_nai, s_nai = strip_prior_from_residual_planes(bad, bad.dem_id.values, sidx, comp, robust=False,
                                                    max_sd=None, min_n=0, return_summary=True)
    for c in ("tilt_x", "tilt_y"):
        r_rob, r_nai = s_rob[c]["tau2"] / true_var[c], s_nai[c]["tau2"] / true_var[c]
        print(f"      {c}: robust+QC tau2/true {r_rob:.2f}   naive tau2/true {r_nai:.1f}")
        check(f"{c}: robust+QC tau2 within 30 % of truth with failed strips injected",
              abs(r_rob - 1) < 0.30, f"{r_rob:.2f}")
        check(f"{c}: naive tau2 blown up by the failures (> 5x)", r_nai > 5, f"{r_nai:.1f}")
    check("QC names exactly the failed strips",
          s_rob["n_qc_dropped"] == nb and set(s_rob["qc_dropped"]) == set(bad.dem_id.values[:nb]))
    # a robust fit on CLEAN strips must agree with the non-robust one
    s_c = strip_prior_from_residual_planes(df2, df2.dem_id.values, sidx, comp, robust=False, return_summary=True)[1]
    check("robust == non-robust on Gaussian strips (no systematic gap; within 25 %)",
          all(abs(summ[c]["tau2"] / s_c[c]["tau2"] - 1) < 0.25 for c in ("tilt_x", "tilt_y")),
          "  ".join(f"{c} {summ[c]['tau2'] / s_c[c]['tau2']:.2f}" for c in ("tilt_x", "tilt_y")))

    # direct fit_residual_plane sanity: too few points -> NaNs, n reported
    r = fit_residual_plane(np.arange(10.0), np.arange(10.0), np.zeros(10))
    check("fit refuses < min_points with NaNs and n", np.isnan(r["ax"]) and r["n"] == 10)

    print("C5  control outside the footprint samples as NaN, for ANY nodata tag")
    # rasterio's sample() fills out-of-grid points with (nodata or 0), so the
    # two tags that collapse to a real 0.0 m are the ones that matter: absent,
    # and 0.0 itself. A constant-40 m DEM makes an unmasked fill obvious.
    res_c, nxc, nyc = 50.0, 60, 40
    x0c, y0c = -1_600_000.0, -290_000.0
    trc = from_origin(x0c, y0c, res_c, res_c)
    inside_e = np.array([x0c + 10 * res_c, x0c + 30 * res_c])
    inside_n = np.array([y0c - 10 * res_c, y0c - 20 * res_c])
    # far outside on each side, then the two EXACT far edges: rasterio's
    # rowcol floors, so x == right maps to col == width and y == bottom to
    # row == height -- both one pixel past the grid, and both filled.
    right_edge, bottom_edge = x0c + nxc * res_c, y0c - nyc * res_c
    outside_e = np.array([x0c - 50 * res_c, x0c + (nxc + 50) * res_c, x0c + 10 * res_c,
                          right_edge, x0c + 10 * res_c])
    outside_n = np.array([y0c - 10 * res_c, y0c - 10 * res_c, y0c + 50 * res_c,
                          y0c - 10 * res_c, bottom_edge])
    ee = np.concatenate([inside_e, outside_e])
    nn = np.concatenate([inside_n, outside_n])
    for tag in (None, 0.0, -9999.0):
        pth = tmp / f"nodata_{tag}.tif"
        kw = {} if tag is None else {"nodata": tag}
        with rasterio.open(pth, "w", driver="GTiff", height=nyc, width=nxc, count=1,
                           dtype="float32", transform=trc, **kw) as dst:
            dst.write(np.full((nyc, nxc), 40.0, np.float32), 1)
        z = sample_dem_at_points(pth, ee, nn)
        ok_in = np.allclose(z[:len(inside_e)], 40.0)
        ok_out = bool(np.all(np.isnan(z[len(inside_e):])))
        check(f"nodata={tag}: inside sampled, outside (incl. exact far edges) NaN",
              ok_in and ok_out, f"inside {z[:len(inside_e)]}  outside {z[len(inside_e):]}")
    # ...and the near edges, which ARE in the grid, must still sample.
    z_edge = sample_dem_at_points(tmp / "nodata_None.tif",
                                  np.array([x0c, x0c + 10 * res_c]),
                                  np.array([y0c - 10 * res_c, y0c]))
    check("the left/top bounds are inside the grid and still sample",
          np.allclose(z_edge, 40.0), f"{z_edge}")

    print("C6  QC gates are reported by reason, not pooled")
    # One genuine alignment failure (huge sd, plenty of control) and one
    # well-aligned strip whose control merely clips the footprint (tiny sd,
    # too few points). Only the first may be offered as a BAD_STRIPS entry.
    split = df2.copy()
    fail_id, low_id = split.dem_id.values[0], split.dem_id.values[1]
    split.loc[0, ["sd", "n"]] = [90.0, 4000]
    split.loc[1, ["sd", "n"]] = [0.36, 40]
    s_sp = strip_prior_from_residual_planes(
        split, split.dem_id.values, sidx, comp, max_sd=5.0, min_n=100,
        return_summary=True)[1]
    print(f"      alignment failures {s_sp['qc_alignment_failures']}  "
          f"low control {s_sp['qc_low_control']}  pooled {s_sp['n_qc_dropped']}")
    check("the sd failure is named an alignment failure",
          s_sp["qc_alignment_failures"] == [fail_id] and s_sp["n_qc_alignment_failures"] == 1,
          f"{s_sp['qc_alignment_failures']}")
    check("the low-control strip is NOT in the alignment-failure list",
          low_id not in s_sp["qc_alignment_failures"], f"{s_sp['qc_alignment_failures']}")
    check("the low-control strip is reported under its own reason",
          s_sp["qc_low_control"] == [low_id] and s_sp["n_qc_low_control"] == 1,
          f"{s_sp['qc_low_control']}")
    check("the two lists are disjoint and sum to the pooled count",
          not set(s_sp["qc_alignment_failures"]) & set(s_sp["qc_low_control"])
          and s_sp["n_qc_alignment_failures"] + s_sp["n_qc_low_control"] == s_sp["n_qc_dropped"],
          f"{s_sp['n_qc_alignment_failures']} + {s_sp['n_qc_low_control']} "
          f"vs {s_sp['n_qc_dropped']}")
    # A strip that trips BOTH gates is an alignment failure, counted once.
    both = df2.copy()
    both.loc[0, ["sd", "n"]] = [90.0, 40]
    s_both = strip_prior_from_residual_planes(
        both, both.dem_id.values, sidx, comp, max_sd=5.0, min_n=100,
        return_summary=True)[1]
    check("a strip failing both gates counts once, as an alignment failure",
          s_both["qc_alignment_failures"] == [fail_id] and s_both["qc_low_control"] == []
          and s_both["n_qc_dropped"] == 1,
          f"align {s_both['qc_alignment_failures']} low {s_both['qc_low_control']}")

    # A strip with NO usable plane: residual_planes_for_strips emits exactly
    # this row (all-NaN, n=0) when the control file is missing, so it must be
    # visible rather than falling out of every list.
    unfit = df2.copy()
    nofit_id, lin_id = unfit.dem_id.values[2], unfit.dem_id.values[3]
    unfit.loc[2, ["ax", "ay", "se_ax", "se_ay", "offset", "sd", "n"]] = \
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0]
    unfit.loc[3, ["se_ax", "se_ay"]] = [np.nan, np.nan]   # LinAlgError path
    unfit.loc[0, ["sd", "n"]] = [90.0, 4000]
    unfit.loc[1, ["sd", "n"]] = [0.36, 40]
    s_u = strip_prior_from_residual_planes(
        unfit, unfit.dem_id.values, sidx, comp, max_sd=5.0, min_n=100,
        return_summary=True)[1]
    print(f"      strips {s_u['n_strips']} = population {s_u['n_population']} "
          f"+ align {s_u['n_qc_alignment_failures']} + low {s_u['n_qc_low_control']} "
          f"+ unfitted {s_u['n_qc_unfitted']}")
    check("a no-plane strip is reported as unfitted, not silently dropped",
          nofit_id in s_u["qc_unfitted"], f"{s_u['qc_unfitted']}")
    check("the NaN-standard-error strip is unfitted too",
          lin_id in s_u["qc_unfitted"], f"{s_u['qc_unfitted']}")
    check("unfitted strips are NOT offered as alignment failures",
          not ({nofit_id, lin_id} & set(s_u["qc_alignment_failures"])),
          f"{s_u['qc_alignment_failures']}")
    buckets = [set(s_u["qc_alignment_failures"]), set(s_u["qc_low_control"]),
               set(s_u["qc_unfitted"])]
    check("the classes are disjoint",
          all(not (a & b) for i, a in enumerate(buckets) for b in buckets[i + 1:]),
          f"{[sorted(b) for b in buckets]}")
    check("the classes are exhaustive (counts sum to the row count)",
          s_u["n_population"] + s_u["n_qc_alignment_failures"]
          + s_u["n_qc_low_control"] + s_u["n_qc_unfitted"] == s_u["n_strips"]
          == len(unfit),
          f"{s_u['n_population']}+{s_u['n_qc_alignment_failures']}"
          f"+{s_u['n_qc_low_control']}+{s_u['n_qc_unfitted']} vs {s_u['n_strips']}")
    check("qc_dropped is exactly the non-population strips",
          set(s_u["qc_dropped"]) == set().union(*buckets)
          and s_u["n_qc_dropped"] == s_u["n_strips"] - s_u["n_population"],
          f"{s_u['n_qc_dropped']} vs {s_u['n_strips'] - s_u['n_population']}")

    print("\nGATE " + ("PASSED" if not FAILS else f"FAILED: {FAILS}"))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())

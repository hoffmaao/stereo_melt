"""Gate: Nuth & Kääb sub-pixel translation, point cloud against raster.

Why this exists. `pc_align` (6-DOF ICP) is our primary coregistration and
solves rotation, which this does not; but ICP's known weakness is sub-pixel
HORIZONTAL accuracy, and the community convention (xDEM) is to finish such a
pipeline with Nuth & Kääb. This gate checks the refinement itself on
synthetics where the true shift is known.

N1  recovery: both methods find a known sub-pixel (dx, dy, dz) on real-ish
    terrain, and the residual NMAD drops.
N2  sign convention: the returned shift, applied as documented, actually
    removes the offset (the classic bug in this method).
N3  identifiability: on a FLAT surface the horizontal shift carries no
    information -- the fit must say so (large standard errors) rather than
    invent a number; the vertical shift is still recovered. And when the slope
    gate leaves too few usable points to fit at all, the answer must come back
    NaN, not an exact (0, 0, 0) that reads as a converged "no shift needed".
N4  robustness: 15 % blunders in the control do not move the answer.
N6  nmad_before/nmad_after are measured on the SAME gated points, so noise
    sitting below the slope gate cannot manufacture an apparent improvement.

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY tests/gate_nuth_kaab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stereo_melt.coregister.nuth_kaab import (  # noqa: E402
    apply_shift_to_points,
    nuth_kaab_point_raster,
    sample_bilinear,
    terrain_slope_aspect,
)

FAILS: list[str] = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def make_terrain(n=400, res=50.0, relief=140.0, seed=0, flat=False):
    """Smooth pseudo-topography on a north-up grid (y descending)."""
    rng = np.random.default_rng(seed)
    x = -1_600_000.0 + np.arange(n) * res
    y = -290_000.0 - np.arange(n) * res
    X, Y = np.meshgrid(x - x.mean(), y - y.mean())
    if flat:
        # A real shelf: no systematic relief, but not a perfect plane either
        # (a perfect plane makes dz exactly collinear with a constant tilt).
        # Long-wavelength undulation of ~0.4 m over tens of km.
        z = (40.0 + 1e-4 * X
             + 0.4 * np.sin(2 * np.pi * X / (n * res * 0.7))
             * np.cos(2 * np.pi * Y / (n * res * 0.9)))
        return x, y, z
    z = 300.0
    for k, (fx, fy) in enumerate([(1.7, 1.1), (3.3, 2.9), (6.1, 5.3)]):
        ph = rng.uniform(0, 2 * np.pi, 2)
        z = z + (relief / (k + 1)) * (np.sin(2 * np.pi * fx * X / (n * res) + ph[0])
                                      * np.cos(2 * np.pi * fy * Y / (n * res) + ph[1]))
    return x, y, z


def shifted_dem(x, y, z_true, dx, dy, dz):
    """A DEM whose content sits at ``x - dx`` and is ``dz`` too high."""
    X, Y = np.meshgrid(x, y)
    return sample_bilinear(z_true, x, y, (X - dx).ravel(), (Y - dy).ravel()).reshape(X.shape) + dz


def control_points(x, y, z_true, n=1500, seed=1, noise=0.15):
    rng = np.random.default_rng(seed)
    pad = 6 * abs(x[1] - x[0])
    e = rng.uniform(x.min() + pad, x.max() - pad, n)
    nth = rng.uniform(y.min() + pad, y.max() - pad, n)
    return e, nth, sample_bilinear(z_true, x, y, e, nth) + rng.normal(0, noise, n)


def main() -> int:
    print("N1  recovery of a known sub-pixel shift (res 50 m)")
    x, y, z_true = make_terrain()
    TRUE = (7.3, -4.1, 1.85)          # 0.15 / 0.08 px horizontal
    dem = shifted_dem(x, y, z_true, *TRUE)
    e, n, zc = control_points(x, y, z_true)
    res_by_method = {}
    for meth in ("gradient", "nuth_kaab"):
        r = nuth_kaab_point_raster(dem, x, y, e, n, zc, method=meth)
        res_by_method[meth] = r
        err = np.array([r["dx"] - TRUE[0], r["dy"] - TRUE[1], r["dz"] - TRUE[2]])
        print(f"      {meth:<10s} -> ({r['dx']:+.3f}, {r['dy']:+.3f}, {r['dz']:+.3f}) m "
              f"vs true ({TRUE[0]:+.2f}, {TRUE[1]:+.2f}, {TRUE[2]:+.2f});  "
              f"NMAD {r['nmad_before']:.3f} -> {r['nmad_after']:.3f} m; "
              f"{r['n_iter']} iters, converged={r['converged']}")
        check(f"{meth}: horizontal within 0.5 m (1 % of pixel)",
              np.hypot(err[0], err[1]) < 0.5, f"|Δh| {np.hypot(err[0], err[1]):.3f} m")
        check(f"{meth}: vertical within 0.2 m", abs(err[2]) < 0.2, f"Δz {err[2]:+.3f} m")
        check(f"{meth}: residual NMAD reduced >5x",
              r["nmad_after"] < r["nmad_before"] / 5,
              f"{r['nmad_before']:.3f} -> {r['nmad_after']:.3f}")
    a, b = res_by_method["gradient"], res_by_method["nuth_kaab"]
    check("the two methods agree within 0.5 m horizontally",
          np.hypot(a["dx"] - b["dx"], a["dy"] - b["dy"]) < 0.5,
          f"{np.hypot(a['dx'] - b['dx'], a['dy'] - b['dy']):.3f} m")

    print("N2  sign convention: applying the answer removes the offset")
    r = res_by_method["gradient"]
    pe, pn = apply_shift_to_points(e, n, r["dx"], r["dy"])
    corrected = sample_bilinear(dem, x, y, pe, pn) - r["dz"]
    before = sample_bilinear(dem, x, y, e, n) - zc
    after = corrected - zc
    fb, fa = np.isfinite(before), np.isfinite(after)
    print(f"      median |dh| {np.median(np.abs(before[fb])):.3f} -> "
          f"{np.median(np.abs(after[fa])):.3f} m")
    check("documented application reduces |dh| >5x",
          np.median(np.abs(after[fa])) < np.median(np.abs(before[fb])) / 5)
    check("no sign flip (residual not made worse)",
          np.median(np.abs(after[fa])) < np.median(np.abs(before[fb])))

    print("N3  identifiability on a flat surface (an ice shelf)")
    xf, yf, zf = make_terrain(flat=True)
    demf = shifted_dem(xf, yf, zf, 7.3, -4.1, 1.85)
    ef, nf, zcf = control_points(xf, yf, zf, seed=3)
    rf = nuth_kaab_point_raster(demf, xf, yf, ef, nf, zcf, method="gradient",
                                min_slope_deg=0.0)
    print(f"      flat: dx {rf['dx']:+.2f} ± {rf['se_dx']:.2f}   dy {rf['dy']:+.2f} ± "
          f"{rf['se_dy']:.2f}   dz {rf['dz']:+.3f} ± {rf['se_dz']:.3f} m   "
          f"slope_p90 {rf['slope_p90']:.3f}°")
    check("flat: horizontal standard errors are large (>2 m) = unidentifiable",
          rf["se_dx"] > 2.0 and rf["se_dy"] > 2.0,
          f"se_dx {rf['se_dx']:.1f} se_dy {rf['se_dy']:.1f} m")
    check("flat: vertical shift still recovered within 0.2 m",
          abs(rf["dz"] - 1.85) < 0.2, f"dz {rf['dz']:+.3f}")
    check("sloped case reports much smaller horizontal SEs than flat",
          r["se_dx"] < rf["se_dx"] / 10, f"{r['se_dx']:.3f} vs {rf['se_dx']:.2f} m")
    # Same shelf, but now the default 3 deg gate leaves nothing to fit: the
    # solver never runs an iteration, so there is no shift to report.
    rn = nuth_kaab_point_raster(demf, xf, yf, ef, nf, zcf, method="gradient")
    print(f"      unusable (default 3° gate): dx {rn['dx']} dy {rn['dy']} dz {rn['dz']} "
          f"n_points {rn['n_points']} n_iter {rn['n_iter']}")
    check("too few usable points -> NaN shift, not (0, 0, 0)",
          not np.isfinite(rn["dx"]) and not np.isfinite(rn["dy"])
          and not np.isfinite(rn["dz"]),
          f"({rn['dx']}, {rn['dy']}, {rn['dz']})")
    check("and it says why: no points, no iterations, not converged",
          rn["n_points"] == 0 and rn["n_iter"] == 0 and not rn["converged"],
          f"n_points {rn['n_points']} n_iter {rn['n_iter']}")
    check("no NMAD is claimed for a fit that never ran",
          not np.isfinite(rn["nmad_before"]) and not np.isfinite(rn["nmad_after"]))
    # A sub-threshold bail is still a bail, but "25 of the 30 needed" and
    # "none at all" are different diagnoses and must not both report 0.
    rs = nuth_kaab_point_raster(dem, x, y, e, n, zc, method="gradient",
                                min_points=10 ** 7)
    print(f"      sub-threshold (min_points 1e7): n_points {rs['n_points']} n_iter {rs['n_iter']}")
    check("a sub-threshold bail reports the count it did have, not 0",
          rs["n_points"] > 0 and rs["n_points"] == rs["history"][-1]["n"]
          and not np.isfinite(rs["dx"]),
          f"n_points {rs['n_points']} vs history {rs['history'][-1]['n']}")

    print("N4  robustness to blunders in the control")
    rng = np.random.default_rng(7)
    zc_bad = zc.copy()
    bad = rng.choice(len(zc_bad), size=int(0.15 * len(zc_bad)), replace=False)
    zc_bad[bad] += rng.normal(0, 40.0, bad.size)
    rb = nuth_kaab_point_raster(dem, x, y, e, n, zc_bad, method="gradient")
    d = np.hypot(rb["dx"] - TRUE[0], rb["dy"] - TRUE[1])
    print(f"      with 15 % blunders -> ({rb['dx']:+.3f}, {rb['dy']:+.3f}, {rb['dz']:+.3f}) m")
    check("robust fit still within 1 m horizontally", d < 1.0, f"|Δh| {d:.3f} m")

    print("N6  before/after NMAD are measured on the same gated points")
    # Half the control sits below the gate and carries 25x the noise. The fit
    # never sees those points, so a like-for-like NMAD pair must not either;
    # scoring "after" over every finite point would report an improvement the
    # (here zero) shift did not make.
    slope_grid = terrain_slope_aspect(z_true, abs(x[1] - x[0]), abs(y[1] - y[0]))[0]
    slope_c = np.degrees(sample_bilinear(slope_grid, x, y, e, n))
    thr = float(np.nanmedian(slope_c))
    below = slope_c < thr
    zc_het = zc.copy()
    zc_het[below] += np.random.default_rng(19).normal(0, 4.0, int(below.sum()))
    r6 = nuth_kaab_point_raster(z_true, x, y, e, n, zc_het, method="gradient",
                                min_slope_deg=thr)
    print(f"      gate {thr:.2f}° excludes {100*below.mean():.0f} % of the control; "
          f"NMAD {r6['nmad_before']:.3f} -> {r6['nmad_after']:.3f} m "
          f"(shift {np.hypot(r6['dx'], r6['dy']):.3f} m)")
    check("the fixture really does exclude a large share of the control",
          below.mean() > 0.3, f"{100*below.mean():.0f} %")
    check("gate-excluded noise does not leak into nmad_after",
          abs(r6["nmad_after"] / r6["nmad_before"] - 1) < 0.25,
          f"{r6['nmad_before']:.3f} -> {r6['nmad_after']:.3f}")

    print("N5  terrain conventions (aspect points downhill, cw from north)")
    xs = np.arange(5) * 10.0
    ys = -np.arange(5) * 10.0
    east_up = np.tile(np.arange(5) * 1.0, (5, 1))            # rises to the east
    _, asp, gx, gy = terrain_slope_aspect(east_up, 10.0, 10.0)
    check("east-rising slope has gx>0, gy==0", gx.mean() > 0 and abs(gy.mean()) < 1e-12)
    check("its aspect points west (270°)", abs(np.degrees(asp.mean()) - 270.0) < 1e-6,
          f"{np.degrees(asp.mean()):.1f}°")
    north_up = np.tile((-np.arange(5) * 1.0)[:, None], (1, 5))  # rises to the north
    _, asp2, gx2, gy2 = terrain_slope_aspect(north_up, 10.0, 10.0)
    check("north-rising slope has gy>0, gx==0", gy2.mean() > 0 and abs(gx2.mean()) < 1e-12)
    check("its aspect points south (180°)", abs(np.degrees(asp2.mean()) - 180.0) < 1e-6,
          f"{np.degrees(asp2.mean()):.1f}°")
    del xs, ys

    print("\nGATE " + ("PASSED" if not FAILS else f"FAILED: {FAILS}"))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())

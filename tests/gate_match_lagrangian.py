"""Gate: the linear-inverse framework must MATCH the Lagrangian path solver.

Prior synthetics (synthetic_realistic_recovery.py, gate_c_timevarying_recovery.py)
generated truth THROUGH the Stubblefield kernel with a uniform background and
zero SMB, so they were structurally blind to the budget gap diagnosed in
literature/plan_match_linear_inverse.md: the kernel inverses never remove the
spatially-varying strain thinning ``H_f·∇·u(x,y)`` and SMB pattern ``ȧ(x,y)``
that the Lagrangian path solver's ḃ explicitly accounts for.

This gate builds a **budget-exact** stack by the method of characteristics:
prescribe ḃ(x,y) (Shean sign), ȧ(x,y), H₀(x,y) and a spatially-varying steady
velocity u(x,y); for each (pixel, epoch) back-trace the trajectory to t=0 and
integrate ``dH/Dt = ḃ + ȧ − H·∇·u`` forward along it; freeboard h = δ_frac·H.
Truth features are ≥ 2.5 km σ so the kernel is quasi-hydrostatic and the two
frameworks share physics at the scored scales.

Corruption rungs::

    G0  clean, full coverage
    G1  + per-strip rectangular coverage masks (~35% per epoch)
    G2  + iid noise (0.3 m)
    G3  + per-strip planar tilt residual (0.10 m)

Solvers per rung: Lagrangian path (production params — the REFERENCE),
Eulerian, the current ``linear_inverse_dhdt_lagrangian_melt_rate`` (mismatch
exhibit), and the new ``linear_inverse_budget_melt_rate`` (must match).

PASS iff at every rung:
  corr5km(lin_budget, truth) >= corr5km(path, truth) - 0.05
  corr5km(lin_budget, path)  >= 0.80
  |median(lin_budget) - median(path)| <= 0.5 m/yr

Run::

    python -u tests/gate_match_lagrangian.py
"""
from __future__ import annotations

import os
import sys
import time

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter, map_coordinates

from stereo_melt.colormaps import add_melt_colorbar, melt_cmap, melt_norm
from stereo_melt.constants import rhoi, rhow
from stereo_melt.dynamics import linear_inverse_dhdt_lagrangian_melt_rate
from stereo_melt.kinematics import divergence
from stereo_melt.melt import eulerian_melt_rate, lagrangian_melt_rate

try:
    from stereo_melt.dynamics import linear_inverse_budget_melt_rate
except ImportError:
    linear_inverse_budget_melt_rate = None

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)
SECONDS_PER_YEAR = 86400.0 * 365.25
DELTA_FRAC = (rhow - rhoi) / rhow  # freeboard fraction = 1/R

RES = 250.0
LX = 64_000.0
LY = 48_000.0
T_WINDOW_YR = 6.0
N_EPOCHS = 16
GEN_DT_YR = 0.01


# ---------------------------------------------------------------------------
# Truth fields
# ---------------------------------------------------------------------------

def build_fields():
    x = np.arange(0.0, LX, RES)
    y = np.arange(LY, 0.0, -RES)  # descending, EPSG:3031 convention
    X, Y = np.meshgrid(x, y)
    yc = LY / 2.0
    xi = (X - x.min()) / LX  # 0..1 along-flow

    def g2(x0, y0, sx, sy):
        return np.exp(-0.5 * (((X - x0) / sx) ** 2 + ((Y - y0) / sy) ** 2))

    # Melt truth, Shean sign (negative = melt). Features >= 2.5 km sigma.
    m_true = (
        -2.0
        - 25.0 * g2(14_000, yc, 4_000, 3_500)          # deep GL melt
        - 8.0 * g2(38_000, yc + 9_000, 6_000, 2_800)   # secondary lobe
        + 2.0 * g2(50_000, yc - 12_000, 5_000, 4_000)  # accretion patch
    )

    # SMB, m ice/yr
    a_dot = 0.35 + 0.25 * (1.0 - xi) + 0.10 * np.cos(2.0 * np.pi * Y / 32_000.0)

    # Initial thickness, m (thins downstream, mild lateral ridging)
    H0 = 700.0 - 320.0 * xi + 40.0 * np.sin(2.0 * np.pi * Y / 24_000.0)

    # Velocity, m/yr: accelerating along +x with a lateral jet profile,
    # mild lateral spreading in vy, plus a localized speed anomaly whose
    # x-derivative sculpts a convergence/divergence dipole. The point of
    # this synthetic is a spatially-STRUCTURED H·div(u) (PIG-like, spanning
    # >~10 m/yr) that the current linear inverse wrongly reads as melt.
    jet = 0.75 + 0.45 * np.exp(-0.5 * ((Y - yc) / 10_000.0) ** 2)
    vx = (250.0 + 1_100.0 * xi) * jet
    vx = vx + 500.0 * np.exp(-0.5 * ((X - 30_000.0) / 9_000.0) ** 2) \
        * np.exp(-0.5 * ((Y - (yc + 7_000.0)) / 6_000.0) ** 2)
    vy = 150.0 * ((Y - yc) / (LY / 2.0)) * (0.3 + 0.7 * xi)

    coords = {"y": y, "x": x}
    da = lambda a, n: xr.DataArray(a, dims=("y", "x"), coords=coords, name=n)
    return (
        da(m_true, "m_true"), da(a_dot, "a_dot"), da(H0, "H0"),
        da(vx, "vx"), da(vy, "vy"),
    )


# ---------------------------------------------------------------------------
# Budget-exact forward model (method of characteristics)
# ---------------------------------------------------------------------------

def budget_exact_stack(m_true, a_dot, H0, vx, vy, t_years, dt_yr=GEN_DT_YR):
    r"""h(t, y, x) with ``dH/Dt = ḃ + ȧ − H·∇·u`` integrated along parcels.

    Back-trace each (pixel, epoch) to t=0 through the steady velocity, then
    forward-integrate H along the re-traced path (semi-implicit in the
    −H·∇·u term). Freeboard h = δ_frac · H (d = 0). Off-domain samples clamp
    to the boundary value (steady inflow).
    """
    x = m_true["x"].values
    y = m_true["y"].values
    ny, nx = m_true.shape
    res_x = float(x[1] - x[0])
    res_y = float(y[0] - y[1])  # positive; y descending

    vx_a = vx.values
    vy_a = vy.values
    div_a = divergence(vx, vy).values  # 1/yr
    m_a = m_true.values
    a_a = a_dot.values
    H0_a = H0.values

    ji, ii = np.mgrid[0:ny, 0:nx]
    y0f = ji.astype(np.float64).ravel()
    x0f = ii.astype(np.float64).ravel()

    def samp(field, yi, xi):
        return map_coordinates(field, [yi, xi], order=1, mode="nearest")

    h_out = np.empty((len(t_years), ny, nx), dtype=np.float64)
    for k, t_e in enumerate(t_years):
        if t_e <= 0:
            h_out[k] = DELTA_FRAC * H0_a
            continue
        n_steps = max(1, int(np.ceil(t_e / dt_yr)))
        dt = t_e / n_steps

        # 1) backward to the origin position
        yi = y0f.copy()
        xi = x0f.copy()
        for _ in range(n_steps):
            u = samp(vx_a, yi, xi)
            v = samp(vy_a, yi, xi)
            xi -= u * dt / res_x
            yi += v * dt / res_y  # y descending: +vy moves up-grid (smaller row)? see below
        # NOTE on sign: forward advection in lagrangian_frame_stack is
        #   x_idx += vx*dt/res_x ; y_idx -= vy*dt/res_y
        # so backward is the negation of both.

        # 2) forward along the same path, integrating H
        H = samp(H0_a, yi, xi)
        for _ in range(n_steps):
            b = samp(m_a, yi, xi)
            a = samp(a_a, yi, xi)
            D = samp(div_a, yi, xi)
            H = (H + dt * (b + a)) / (1.0 + dt * D)
            u = samp(vx_a, yi, xi)
            v = samp(vy_a, yi, xi)
            xi += u * dt / res_x
            yi -= v * dt / res_y
        h_out[k] = (DELTA_FRAC * H).reshape(ny, nx)

    times = pd.to_datetime("2019-01-01") + pd.to_timedelta(
        np.asarray(t_years) * SECONDS_PER_YEAR, unit="s"
    )
    return xr.DataArray(
        h_out, dims=("time", "y", "x"),
        coords={"time": times, "y": y, "x": x}, name="h_freeboard",
    )


# ---------------------------------------------------------------------------
# Corruption (adapted from synthetic_realistic_recovery.py)
# ---------------------------------------------------------------------------

def add_strip_coverage(h_stack, coverage_frac=0.35, seed=11):
    rng = np.random.default_rng(seed)
    out = h_stack.values.copy()
    nt, ny, nx = out.shape
    for k in range(nt):
        target_pix = int(coverage_frac * ny * nx)
        h = max(int(np.sqrt(target_pix * rng.uniform(0.5, 2.0))), 8)
        h = min(h, ny)
        w = min(max(target_pix // max(h, 1), 8), nx)
        yy = rng.integers(0, ny - h + 1)
        xx = rng.integers(0, nx - w + 1)
        keep = np.zeros((ny, nx), dtype=bool)
        keep[yy:yy + h, xx:xx + w] = True
        out[k] = np.where(keep, out[k], np.nan)
    return h_stack.copy(data=out)


def add_noise(h_stack, sigma_m=0.3, seed=101):
    rng = np.random.default_rng(seed)
    return h_stack.copy(data=h_stack.values + rng.normal(0, sigma_m, h_stack.shape))


def add_per_strip_tilt(h_stack, tilt_amp_m=0.10, seed=23):
    rng = np.random.default_rng(seed)
    out = h_stack.values.copy()
    x = h_stack["x"].values
    y = h_stack["y"].values
    XN = (x[None, :] - x.mean()) / (0.5 * (x.max() - x.min()))
    YN = (y[:, None] - y.mean()) / (0.5 * (y.max() - y.min()))
    for k in range(out.shape[0]):
        out[k] += (rng.normal(0, tilt_amp_m) * XN
                   + rng.normal(0, tilt_amp_m) * YN
                   + rng.normal(0, tilt_amp_m))
    return h_stack.copy(data=out)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def nan_gauss(a, sigma_pix):
    if sigma_pix <= 0:
        return a
    m = np.isfinite(a)
    a0 = np.where(m, a, 0.0)
    num = gaussian_filter(a0, sigma_pix, mode="nearest")
    den = gaussian_filter(m.astype(float), sigma_pix, mode="nearest")
    out = num / np.maximum(den, 1e-9)
    return np.where(den > 0.05, out, np.nan)


def scorr(a, b, sigma_pix):
    aa = nan_gauss(np.asarray(a, float), sigma_pix)
    bb = nan_gauss(np.asarray(b, float), sigma_pix)
    f = np.isfinite(aa) & np.isfinite(bb)
    if f.sum() < 30:
        return np.nan
    av = aa[f] - aa[f].mean()
    bv = bb[f] - bb[f].mean()
    den = np.sqrt((av ** 2).sum() * (bv ** 2).sum())
    return float((av * bv).sum() / den) if den > 0 else np.nan


def score(name, m_da, truth, path_vals):
    v = np.asarray(m_da.values, float) if hasattr(m_da, "values") else np.asarray(m_da, float)
    t = truth.values
    s2 = 2000.0 / RES
    s5 = 5000.0 / RES
    finite = np.isfinite(v)
    med = float(np.nanmedian(v)) if finite.any() else np.nan
    return {
        "name": name,
        "median": med,
        "c_raw_truth": scorr(v, t, 0),
        "c2k_truth": scorr(v, t, s2),
        "c5k_truth": scorr(v, t, s5),
        "c5k_path": scorr(v, path_vals, s5) if path_vals is not None else np.nan,
        "cover": float(finite.mean()),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def build_epochs(seed=5):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, T_WINDOW_YR, N_EPOCHS)
    jit = rng.uniform(-0.5, 0.5, N_EPOCHS) * (T_WINDOW_YR / N_EPOCHS) * 0.6
    jit[0] = 0.0
    t = np.sort(t + jit)
    t[0] = 0.0
    return t


def corrupt(h_clean, rung):
    h = h_clean
    if rung in ("G1", "G2", "G3"):
        h = add_strip_coverage(h, 0.35, seed=11)
    if rung in ("G2", "G3"):
        h = add_noise(h, 0.3, seed=101)
    if rung == "G3":
        h = add_per_strip_tilt(h, 0.10, seed=23)
    return h


def run_rung(rung, h_clean, fields, verbose_budget=False):
    m_true, a_dot, H0, vx, vy = fields
    h = corrupt(h_clean, rung)
    floating = xr.DataArray(
        np.ones(m_true.shape, dtype=bool), dims=("y", "x"),
        coords={"y": m_true["y"].values, "x": m_true["x"].values},
    )

    rows = []
    t0 = time.time()
    lagr = lagrangian_melt_rate(
        h, vx, vy, a_dot=a_dot, d=0.0, dt_yr=0.05,
        seed_stride=1, output="path", aggregator="pair_median",
        pairs="all", min_dt_yr=1.5, max_dt_yr=2.5,
        progress_interval_s=0.0,
    ).melt_rate
    t_lagr = time.time() - t0
    path_vals = np.asarray(lagr.values, float)
    rows.append(score("path(REF)", lagr, m_true, None))

    eul = eulerian_melt_rate(h, vx, vy, a_dot=a_dot, d=0.0, robust_dh_dt=True).melt_rate
    rows.append(score("eulerian", eul, m_true, path_vals))

    t0 = time.time()
    # a_dot_dc: give the current framework its best DC (the drivers don't
    # pass it — that footgun is recorded in the design note); the exhibit
    # here is the missing spatial PATTERN of H·div(u) − ȧ.
    lin_cur = linear_inverse_dhdt_lagrangian_melt_rate(
        h, vx, vy, floating_mask=floating,
        transform="dct", reg=1e-1, min_n_obs=3, robust_dh_dt=True,
        recover_dc=True, a_dot_dc=float(a_dot.mean()), dt_yr=0.05,
    ).melt_rate
    t_cur = time.time() - t0
    rows.append(score("lin_current", lin_cur, m_true, path_vals))

    lin_bud = None
    t_bud = 0.0
    if linear_inverse_budget_melt_rate is not None:
        t0 = time.time()
        lin_bud = linear_inverse_budget_melt_rate(
            h, vx, vy, a_dot=a_dot, d=0.0, floating_mask=floating,
            transform="dct", reg=1e-1,
            min_pair_dt_yr=1.5, max_pair_dt_yr=2.5, dt_yr=0.05,
            progress=verbose_budget,
        ).melt_rate
        t_bud = time.time() - t0
        rows.append(score("lin_budget", lin_bud, m_true, path_vals))
        # Coverage-fair diagnostic: the path solver restricted to
        # lin_budget's finite mask, so pattern quality and coverage
        # penalties are separable in the truth-corr comparison.
        path_on_bud = np.where(np.isfinite(np.asarray(lin_bud.values, float)),
                               path_vals, np.nan)
        rows.append(score("path|budmask", path_on_bud, m_true, None))

    print(f"\n=== {rung} ===  (path {t_lagr:.0f}s, lin_current {t_cur:.0f}s, "
          f"lin_budget {t_bud:.0f}s)")
    hdr = (f"{'solver':<12} {'median':>8} {'c_raw':>7} {'c2k':>7} {'c5k':>7} "
           f"{'c5k_path':>9} {'cover':>6}")
    print(hdr)
    truth_med = float(np.nanmedian(m_true.values))
    print(f"{'truth':<12} {truth_med:>8.2f} {'1.000':>7} {'1.000':>7} {'1.000':>7} "
          f"{'':>9} {'1.00':>6}")
    for r in rows:
        print(f"{r['name']:<12} {r['median']:>8.2f} {r['c_raw_truth']:>7.3f} "
              f"{r['c2k_truth']:>7.3f} {r['c5k_truth']:>7.3f} "
              f"{r['c5k_path']:>9.3f} {r['cover']:>6.2f}")

    # Gate check: truth-corr compared coverage-fairly (path on lin_budget's
    # mask), so the criterion scores pattern quality; coverage is reported
    # separately in the table.
    verdict = None
    if lin_bud is not None:
        r_path = rows[-1]  # path|budmask
        r_bud = rows[-2]   # lin_budget
        ok_truth = r_bud["c5k_truth"] >= r_path["c5k_truth"] - 0.05
        ok_match = r_bud["c5k_path"] >= 0.80
        ok_med = abs(r_bud["median"] - r_path["median"]) <= 0.5
        verdict = ok_truth and ok_match and ok_med
        print(f"  gate[{rung}]: truth-corr {'OK' if ok_truth else 'FAIL'} "
              f"({r_bud['c5k_truth']:.3f} vs path {r_path['c5k_truth']:.3f})  "
              f"match {'OK' if ok_match else 'FAIL'} ({r_bud['c5k_path']:.3f})  "
              f"median {'OK' if ok_med else 'FAIL'} "
              f"(Δ={abs(r_bud['median'] - r_path['median']):.2f}) -> "
              f"{'PASS' if verdict else 'FAIL'}")

    return rows, {"path": lagr, "eulerian": eul, "lin_current": lin_cur,
                  "lin_budget": lin_bud}, verdict


def plot_rung(rung, m_true, maps):
    panels = [("truth", m_true)] + [(k, v) for k, v in maps.items() if v is not None]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6), constrained_layout=True)
    vmax = float(np.nanpercentile(np.abs(m_true.values), 99.5))
    for ax, (name, da) in zip(np.atleast_1d(axes), panels):
        im = ax.imshow(np.asarray(da.values, float), cmap=melt_cmap(),
                       norm=melt_norm(vmax=vmax), interpolation="nearest")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    add_melt_colorbar(fig, im, ax=np.atleast_1d(axes).tolist(), shrink=0.8)
    fig.suptitle(f"gate_match_lagrangian — {rung}", fontsize=11)
    out = OUT_DIR / f"gate_match_{rung}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    print("Building budget-exact synthetic (method of characteristics)...")
    fields = build_fields()
    m_true, a_dot, H0, vx, vy = fields
    t_years = build_epochs()
    cache = os.environ.get("GATE_CACHE", "")
    h_clean = None
    if cache and os.path.exists(cache):
        z = np.load(cache)
        if z["h"].shape == (len(t_years), *m_true.shape) and np.allclose(z["t"], t_years):
            times = pd.to_datetime("2019-01-01") + pd.to_timedelta(
                t_years * SECONDS_PER_YEAR, unit="s")
            h_clean = xr.DataArray(
                z["h"], dims=("time", "y", "x"),
                coords={"time": times, "y": m_true["y"].values,
                        "x": m_true["x"].values}, name="h_freeboard")
            print(f"  loaded cached stack from {cache}")
    if h_clean is None:
        t0 = time.time()
        h_clean = budget_exact_stack(m_true, a_dot, H0, vx, vy, t_years)
        print(f"  stack {h_clean.shape} in {time.time() - t0:.0f}s; "
              f"epochs (yr): {np.round(t_years, 2).tolist()}")
        if cache:
            np.savez_compressed(cache, h=h_clean.values, t=t_years)
            print(f"  cached -> {cache}")
    div = divergence(vx, vy)
    Hdiv = (H0 * div).values
    print(f"  H·div(u): median {np.median(Hdiv):+.2f}  "
          f"p5..p95 [{np.percentile(Hdiv, 5):+.2f}, {np.percentile(Hdiv, 95):+.2f}] m/yr  "
          f"a_dot p5..p95 [{np.percentile(a_dot.values, 5):+.2f}, "
          f"{np.percentile(a_dot.values, 95):+.2f}] m/yr")

    rungs = sys.argv[1:] if len(sys.argv) > 1 else ["G0", "G1", "G2", "G3"]
    verdicts = {}
    for rung in rungs:
        rows, maps, verdict = run_rung(rung, h_clean, fields,
                                       verbose_budget=(rung == rungs[0]))
        verdicts[rung] = verdict
        plot_rung(rung, m_true, maps)

    if linear_inverse_budget_melt_rate is None:
        print("\nlinear_inverse_budget_melt_rate not implemented yet — "
              "baseline (mismatch) run only, no gate verdict.")
    else:
        overall = all(v for v in verdicts.values() if v is not None)
        print(f"\nOVERALL GATE: {'PASS' if overall else 'FAIL'}  ({verdicts})")


if __name__ == "__main__":
    main()

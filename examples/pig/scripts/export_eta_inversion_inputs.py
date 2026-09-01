"""Export PIG (thickness, surface, velocity, outline + boundary classes) for
the icepack2 dual-form fluidity inversion — plain .npz for the firedrake env.

TIME-CONSISTENT BY CONSTRUCTION (2026-07-26 directive): thickness/surface,
velocity, and the domain/front geometry are all sampled from the SAME window
W, or the inversion bears the inconsistency as spurious fluidity — worst near
the calving front, exactly where the dual form finally lets us look. Defaults
W = 2021-01-01 → 2024-01-01: post-2020 calving geometry (matches BedMachine v3
and the AOI), dense IS2-era DEM epochs, quarterly ASE velocity coverage.

Per window: H and s = per-pixel MEDIAN of the tilt-corrected stack epochs in W
(>= --min-count obs); velocity = the Kalman/EOF-fused quarterly ASE mosaics
averaged over W (PIG_VELOCITY=fused); domain = EVERY cleaned component of
(min-extent & covered & H > H_MIN) in W above --min-component-km2 — the PIG
embayment is the main shelf plus its neighboring shelves, each meshed as its
own boundary loop (components with an all-front boundary are dropped: no
Dirichlet pin). Per-component outlines land in ``polys``/``vclasses``.

BOUNDARY CLASSES for the dual form's calving terminus: each outline vertex is
classified from the BedMachine codes of its just-outside neighborhood. A vertex
is FRONT (class 2, natural calving-stress BC) when MORE THAN 25% of those cells
are ocean (0); otherwise it is DIRICHLET (class 1, u = u_obs). The threshold
is deliberately a quarter, not a majority: the min-extent outline sits a
dilation away from the BedMachine coast, so a true calving-front vertex sees
rock/ice on much of its neighborhood, and under-classifying a front — pinning
it to u_obs — is the worse failure mode for the dual inversion. The shipped
production eta field was built with this rule.

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY examples/pig/scripts/export_eta_inversion_inputs.py [--t0 2021-01-01 --t1 2024-01-01]
"""
import argparse
import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"] = _env_proj
os.environ.setdefault("PIG_VELOCITY", "fused")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
BASIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "examples"))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

import pig.run_melt as R  # noqa: E402
from pig import config  # noqa: E402
from stereo_melt.freeboard import freeboard_to_thickness  # noqa: E402
from stereo_melt.io.bedmachine import (  # noqa: E402
    interp_bedmachine_mask_at,
    load_firn_on_grid,
)
from stereo_melt.io.greene import greene_min_extent_on_grid  # noqa: E402
from stereo_melt.shelf_extent import min_shelf_extent  # noqa: E402

STACK_PREFIX = "pig_stack_250m_is2ctempo"
RHO_I, RHO_W = 918.0, 1027.0
H_MIN = 50.0
OUT = os.path.join(BASIN, "processed", "pig_eta_inv_inputs.npz")
BM_CODES = (0, 1, 2, 3)          # ocean, rock, grounded, floating


def nearest_fill(a, valid):
    idx = ndimage.distance_transform_edt(
        ~valid, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t0", default="2021-01-01")
    ap.add_argument("--t1", default="2024-01-01")
    ap.add_argument("--min-count", type=int, default=1,
                    help="min DEM epochs per pixel inside the window")
    ap.add_argument("--min-component-km2", type=float, default=50.0,
                    help="smallest floating component meshed as its own "
                         "shelf (the embayment = main shelf + neighbors)")
    args = ap.parse_args()

    print(f"[load] stack {STACK_PREFIX}, window {args.t0}..{args.t1} ...",
          flush=True)
    stack = R.load_stack(stack_prefix=STACK_PREFIX)
    floating = R.load_floating_mask(stack)
    firn = load_firn_on_grid(stack, config.BEDMACHINE_NC)

    win = stack.sel(time=slice(args.t0, args.t1))
    nep = int(win.sizes["time"])
    count = win.notnull().sum("time").values
    s_med = win.median("time", skipna=True)
    H = freeboard_to_thickness(s_med, d=firn, rho_w=RHO_W, rho_i=RHO_I).values
    s_g = s_med.values
    print(f"[window] {nep} epochs; median pixel count "
          f"{np.median(count[count > 0]):.0f}", flush=True)

    vx_t, vy_t, vsrc = R.load_velocity_on_grid(stack)   # fused quarterly (t,y,x)
    if "time" in getattr(vx_t, "dims", ()):
        vx = vx_t.sel(time=slice(args.t0, args.t1)).mean("time").values
        vy = vy_t.sel(time=slice(args.t0, args.t1)).mean("time").values
        nq = int(vx_t.sel(time=slice(args.t0, args.t1)).sizes["time"])
        print(f"[vel] {nq} quarters averaged over the window", flush=True)
    else:
        vx, vy = vx_t.values, vy_t.values
        print("[vel] WARNING: static velocity source; window-mean unavailable",
              flush=True)

    # Domain = geometry observed in-window; velocity quality enters via
    # sigma_u, not the domain gate (the fused product is gap-filled anyway).
    # Window-minimum extent (Greene observed coastlines + in-window ocean
    # test): mid-window calved/melange pixels read as fluidity otherwise.
    greene = greene_min_extent_on_grid(
        s_med, config.GREENE_ICEMASK_MAT, t0=args.t0, t1=args.t1)
    ext = min_shelf_extent(win, floating, greene_extent=greene)
    fl = ext.values.astype(bool)
    valid = fl & (count >= args.min_count) & np.isfinite(H) & (H > H_MIN)
    print(f"[mask] floating {int(floating.values.sum())} -> min-extent "
          f"{fl.sum()} (ocean test -{ext.attrs['n_removed_ocean_test']}, "
          f"greene -{ext.attrs['n_removed_greene']})  window-valid "
          f"{valid.sum()} cells", flush=True)

    # Morphological cleanup on the FULL valid mask, then keep EVERY
    # component large enough to invert: the PIG embayment is the main
    # shelf PLUS its neighboring shelves (2026-07-27 directive) — the old
    # largest-component gate silently dropped the neighbors.
    keep_u = ndimage.binary_opening(valid, iterations=2)
    keep_u = ndimage.binary_closing(keep_u, iterations=3)
    keep_u = ndimage.binary_fill_holes(keep_u)
    x, y = stack.x.values, stack.y.values
    cell_km2 = abs((x[1] - x[0]) * (y[1] - y[0])) / 1e6
    lab, n = ndimage.label(keep_u)
    sizes = ndimage.sum(keep_u, lab, range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    comp_ids = [1 + int(i) for i in order
                if sizes[int(i)] * cell_km2 >= args.min_component_km2]
    print(f"[mask] {n} components; {len(comp_ids)} >= "
          f"{args.min_component_km2:.0f} km^2 "
          f"({', '.join(f'{sizes[i - 1] * cell_km2:.0f}' for i in comp_ids)} km^2)",
          flush=True)

    # BedMachine class map on the stack grid (nearest), for boundary typing
    Xg, Yg = np.meshgrid(x, y)
    bm = np.full(Xg.shape, -1, dtype=np.int8)
    for code in BM_CODES:
        hit = interp_bedmachine_mask_at(
            config.BEDMACHINE_NC, Xg.ravel(), Yg.ravel(),
            mask_value=code).reshape(Xg.shape)
        bm[hit] = code

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def outline(comp):
        fig, ax = plt.subplots()
        cs = ax.contour(x, y, comp.astype(float), levels=[0.5])
        ring = max(cs.allsegs[0], key=len)
        plt.close(fig)
        step = max(1, len(ring) // 800)
        ring = ring[::step]
        if not np.allclose(ring[0], ring[-1]):
            ring = np.vstack([ring, ring[:1]])
        return ring

    # Per-vertex class from the BedMachine codes of the just-outside cells in
    # a (2r+1)^2 neighborhood: > 25% ocean => front (dilation-tolerant; a
    # missed front pinned to u_obs is the worse error, see the module
    # docstring). A component whose boundary is ALL front has no Dirichlet
    # pin (free-floating, singular momentum balance) and is dropped.
    dx_g = float(abs(x[1] - x[0]))
    r = 8

    def classify(ring):
        vc = np.ones(len(ring), dtype=np.int8)       # 1 = dirichlet default
        for k, (px, py) in enumerate(ring):
            i = int(round((py - y[0]) / (y[1] - y[0])))
            j = int(round((px - x[0]) / (x[1] - x[0])))
            i0, i1 = max(0, i - r), min(len(y), i + r + 1)
            j0, j1 = max(0, j - r), min(len(x), j + r + 1)
            nb_out = ~keep_u[i0:i1, j0:j1]
            codes = bm[i0:i1, j0:j1][nb_out]
            codes = codes[codes >= 0]
            if codes.size and np.mean(codes == 0) > 0.25:
                vc[k] = 2                            # calving front
        return vc

    polys, vclasses = [], []
    keep = np.zeros_like(keep_u)
    for cid in comp_ids:
        comp = lab == cid
        ring = outline(comp)
        vc = classify(ring)
        n_front = int((vc == 2).sum())
        km2 = comp.sum() * cell_km2
        if n_front == len(ring):
            print(f"[boundary] component {km2:.0f} km^2 DROPPED: all-front "
                  "boundary (free-floating, no Dirichlet pin)", flush=True)
            continue
        polys.append(ring)
        vclasses.append(vc)
        keep |= comp
        print(f"[boundary] component {km2:.0f} km^2: {len(ring)} vertices, "
              f"{n_front} front / {len(ring) - n_front} dirichlet "
              f"(r={r * dx_g:.0f} m)", flush=True)
    if not polys:
        raise SystemExit("no meshable components survived")

    H_f = nearest_fill(np.where(np.isfinite(H), H, np.nan), valid)
    s_f = nearest_fill(np.where(np.isfinite(s_g), s_g, np.nan), valid)
    ux_f = nearest_fill(vx, np.isfinite(vx))
    uy_f = nearest_fill(vy, np.isfinite(vy))
    H_f = np.where(np.isfinite(H_f), H_f, H_MIN)
    s_f = np.where(np.isfinite(s_f), s_f, H_MIN * (1 - RHO_I / RHO_W))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, x=x, y=y, H=H_f, s=s_f, ux=ux_f, uy=uy_f,
             mask=keep.astype(np.uint8),
             polys=np.array(polys, dtype=object),
             vclasses=np.array(vclasses, dtype=object),
             RHO_I=RHO_I, RHO_W=RHO_W, vsrc=str(vsrc), stack=STACK_PREFIX,
             t0=args.t0, t1=args.t1, n_epochs=nep,
             min_count=args.min_count,
             min_component_km2=args.min_component_km2)
    sp = np.hypot(ux_f, uy_f)[keep]
    print(f"wrote {OUT}")
    print(f"  H[{keep.sum()}] mean {H_f[keep].mean():.0f} m "
          f"(p10 {np.percentile(H_f[keep], 10):.0f}, "
          f"p90 {np.percentile(H_f[keep], 90):.0f})")
    print(f"  |u| mean {sp.mean():.0f} m/yr (p90 {np.percentile(sp, 90):.0f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Compute a basin's wider stack/tilt AOI rectangle (v15 algorithm).

Domain-agnostic: the script knows nothing about specific basins. Given a single
shelf polygon, greedily expand a rectangle subject to a TS-area target (the
target-static region: grounded ice + rock, slow-velocity, inside REMA strip
coverage). The expansion is biased upstream and perpendicular to shelf-only mean
flow, so the rectangle stretches inland where the static-control coverage is and
stays narrow across-flow where it isn't.

Target area: floating_area * scale(v_shelf), snapped to the nearest TARGET_BUCKET,
where scale(v) = TARGET_BASE + TARGET_VFACTOR * log10(1 + v_shelf / V_REF).

Writes the rectangle as ``<out>`` (ESRI shapefile) and a diagnostic figure
``<figures-dir>/ts_aoi_v15_<name>.png``.

Provide the shelf one of three ways:

    # 1. config-driven: read inputs from <basin>.config  (PYTHONPATH = repo root)
    python scripts/compute_aoi.py venable

    # 2. a named feature inside a multi-feature shapefile
    python scripts/compute_aoi.py \
        --feature-shp data/shapefiles/IceShelf_Antarctica_v02.shp \
        --feature-name Venable \
        --out data/shapefiles/venable_stack_extent.shp

    # 3. any polygon -> any output
    python scripts/compute_aoi.py \
        --shelf data/shapefiles/venable.shp \
        --out  data/shapefiles/venable_stack_extent.shp --name Venable

Tuning constants and the pan-Antarctic raster paths are CLI-overridable. To
reproduce an existing basin exactly, pass its original ``--shelf`` input.

Originally an ad hoc heredoc; promoted to scripts/compute_aoi.py 2026-05-11; made
domain-agnostic (was a hardcoded per-basin loop) 2026-06-30.
"""
import argparse
import importlib
import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj; os.environ["PROJ_LIB"] = _env_proj

from pathlib import Path

import fiona, numpy as np, xarray as xr, rasterio
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
from shapely.geometry import shape, box, mapping
from shapely.ops import unary_union
from rasterio.features import rasterize
from rasterio.transform import Affine

# --- workspace defaults (overridable on the CLI) ---------------------------
ROOT = Path("/wd2/projects/stereo_melt")
BEDMACHINE = ROOT / "data/bedmachine/BedMachineAntarctica-v3.nc"
MEASURES = ROOT / "data/NSIDC-0754/1996.01.01/antarctic_ice_vel_phase_map_v01.nc"
STRIPS_DIR = ROOT / "data/REMA/strips"

# --- v15 tuning constants --------------------------------------------------
V_THRESH = 100.0
BUFFER_KM = 200; PENALTY_RADIUS_KM = 5.0; PENALTY_PER_PIXEL_KM2 = 0.20; GAMMA = 1.0
BASE_PERP = 0.5; UP_DECAY = 2.0; DOWN_GROWTH = 4.0
TARGET_BUCKETS = [5_000, 10_000, 15_000, 20_000, 30_000, 45_000]

# Target sizing: floating_area * scale(v_shelf), snapped to a TARGET_BUCKETS
# entry. The multiplier scales with shelf flow speed because fast shelves
# need a larger upstream extent to capture enough slow grounded-ice +
# rock control after the V_THRESH filter rejects fast shelf ice.
#
# scale(v) = TARGET_BASE + TARGET_VFACTOR * log10(1 + v_myr / V_REF)
#
# Calibrated so the slow shelves (Beardmore/Nansen, v~20 m/yr) land near
# scale=2.5 (matches the original v15 fixed multiplier), while fast
# shelves (PIG, v~3000 m/yr) land near scale=4.5.
TARGET_BASE = 2.0
TARGET_VFACTOR = 1.5
V_REF = 100.0  # m/yr

# --- pan-Antarctic rasters, populated by load_grids() ----------------------
VX_full = VY_full = SPEED = v_x = v_y = bm_ds = None
PIX_M = PIX_AREA_KM2 = D_PX = None


def load_grids(measures_path, bedmachine_path):
    """Open the velocity + BedMachine rasters into the module-level grids."""
    global VX_full, VY_full, SPEED, v_x, v_y, PIX_M, PIX_AREA_KM2, D_PX, bm_ds
    vds = xr.open_dataset(measures_path)
    VX_full = vds["VX"].values; VY_full = vds["VY"].values
    SPEED = np.hypot(VX_full, VY_full).astype(np.float32)
    v_x = vds["x"].values; v_y = vds["y"].values
    PIX_M = abs(v_x[1] - v_x[0]); PIX_AREA_KM2 = (PIX_M*PIX_M)/1e6
    D_PX = max(1, int(round(PENALTY_RADIUS_KM*1000/PIX_M)))
    bm_ds = xr.open_dataset(bedmachine_path)


def basin_strips(shelf_geom, buffer_m, max_strips=400):
    s_xmin, s_ymin, s_xmax, s_ymax = shelf_geom.bounds
    win = (s_xmin-buffer_m, s_ymin-buffer_m, s_xmax+buffer_m, s_ymax+buffer_m)
    boxes = []
    for path in sorted(STRIPS_DIR.glob("SETSM_*.tif")):
        if path.stem.endswith(("_matchtag","_bitmask")): continue
        try:
            with rasterio.open(path) as ds: bx_min, by_min, bx_max, by_max = ds.bounds
        except Exception: continue
        if bx_max < win[0] or bx_min > win[2] or by_max < win[1] or by_min > win[3]: continue
        boxes.append((bx_min, by_min, bx_max, by_max))
        if len(boxes) >= max_strips: break
    return boxes


def make_masks(shelf_geom, buffer_m, strip_boxes):
    xmin, ymin, xmax, ymax = shelf_geom.bounds
    xmin -= buffer_m; xmax += buffer_m; ymin -= buffer_m; ymax += buffer_m
    ix0 = max(0, int(np.searchsorted(v_x, xmin)))
    ix1 = min(len(v_x), int(np.searchsorted(v_x, xmax)+1))
    iy0 = max(0, int(np.searchsorted(v_y[::-1], ymin)))
    iy1 = min(len(v_y), int(np.searchsorted(v_y[::-1], ymax)+1))
    iy_top, iy_bot = len(v_y)-iy1, len(v_y)-iy0
    sub_x = v_x[ix0:ix1]; sub_y = v_y[iy_top:iy_bot]
    speed_sub = SPEED[iy_top:iy_bot, ix0:ix1]
    vx_sub = VX_full[iy_top:iy_bot, ix0:ix1]
    vy_sub = VY_full[iy_top:iy_bot, ix0:ix1]
    bm_sub = bm_ds["mask"].sel(x=slice(sub_x.min(), sub_x.max()), y=slice(sub_y.max(), sub_y.min()))
    bm_resampled = bm_sub.interp(x=sub_x, y=sub_y, method="nearest").values
    is_gr = (bm_resampled == 1) | (bm_resampled == 2)
    is_slow = speed_sub < V_THRESH
    ts = (is_gr & is_slow).astype(np.int32)
    strip = np.zeros_like(ts, dtype=np.int32)
    for bx_min, by_min, bx_max, by_max in strip_boxes:
        ix_a = max(0, int(np.searchsorted(sub_x, bx_min)))
        ix_b = min(len(sub_x), int(np.searchsorted(sub_x, bx_max)+1))
        iy_a = max(0, int(np.searchsorted(-sub_y, -by_max)))
        iy_b = min(len(sub_y), int(np.searchsorted(-sub_y, -by_min)+1))
        if ix_a < ix_b and iy_a < iy_b: strip[iy_a:iy_b, ix_a:ix_b] = 1
    return ts, strip, sub_x, sub_y, vx_sub, vy_sub, speed_sub


def shelf_bbox_pixels(shelf_geom, sub_x, sub_y):
    xmin, ymin, xmax, ymax = shelf_geom.bounds
    ix0 = int(np.searchsorted(sub_x, xmin)); ix1 = int(np.searchsorted(sub_x, xmax)+1)
    iy_a = int(np.searchsorted(sub_y[::-1], ymin)); iy_b = int(np.searchsorted(sub_y[::-1], ymax)+1)
    iy0, iy1 = len(sub_y)-iy_b, len(sub_y)-iy_a
    return max(0, ix0), min(len(sub_x), ix1), max(0, iy0), min(len(sub_y), iy1)


def cum_count_fn(arr):
    cum = arr.cumsum(axis=0).cumsum(axis=1)
    def count(a, b, c, d):
        if a >= b or c >= d: return 0
        S = cum
        total = int(S[d-1, b-1])
        if a > 0: total -= int(S[d-1, a-1])
        if c > 0: total -= int(S[c-1, b-1])
        if a > 0 and c > 0: total += int(S[c-1, a-1])
        return total
    return count


def floating_area_in_shelf(shelf_geom):
    """Compute floating-ice area (BedMachine code 3) inside shelf AOI in km²."""
    xmin, ymin, xmax, ymax = shelf_geom.bounds
    bm_sub = bm_ds["mask"].sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
    bm_x = bm_sub.x.values; bm_y = bm_sub.y.values
    res_x = float(abs(bm_x[1]-bm_x[0])); res_y = float(abs(bm_y[1]-bm_y[0]))
    aff = Affine.from_gdal(float(bm_x[0]-res_x/2), res_x, 0.0, float(bm_y[0]+res_y/2), 0.0, -res_y)
    inside = rasterize([(shelf_geom, 1)], out_shape=bm_sub.shape, transform=aff, fill=0, dtype=np.uint8).astype(bool)
    floating = ((bm_sub.values == 3) & inside).sum() * (res_x * res_y / 1e6)
    return float(floating)


def auto_target(floating_km2, v_shelf_myr):
    """Velocity-scaled target area, snapped to the nearest bucket.

    scale(v) = TARGET_BASE + TARGET_VFACTOR * log10(1 + v_shelf_myr / V_REF)

    Returns (target_bucket_km2, target_raw_km2, scale_used).
    """
    scale = TARGET_BASE + TARGET_VFACTOR * np.log10(1.0 + v_shelf_myr / V_REF)
    target_raw = max(TARGET_BUCKETS[0], floating_km2 * scale)
    return min(TARGET_BUCKETS, key=lambda b: abs(b - target_raw)), target_raw, scale


def shelf_only_flow(shelf_geom, vx_sub, vy_sub, speed_sub, sub_x, sub_y):
    """Return shelf-area-weighted flow unit vector + mean shelf speed (m/yr)."""
    res_x = float(abs(sub_x[1]-sub_x[0])); res_y = float(abs(sub_y[1]-sub_y[0]))
    aff = Affine.from_gdal(float(sub_x[0]-res_x/2), res_x, 0.0, float(sub_y[0]+res_y/2), 0.0, -res_y)
    inside = rasterize([(shelf_geom, 1)], out_shape=vx_sub.shape, transform=aff, fill=0, dtype=np.uint8).astype(bool)
    valid = inside & np.isfinite(vx_sub) & np.isfinite(vy_sub) & (speed_sub > 0)
    if valid.sum() < 10: return (1.0, 0.0), 0.0, 0, 0.0
    w = speed_sub[valid]
    mvx = float(np.sum(w * vx_sub[valid]) / np.sum(w))
    mvy = float(np.sum(w * vy_sub[valid]) / np.sum(w))
    mag = np.hypot(mvx, mvy)
    # Unweighted mean shelf speed (m/yr) -- used by auto_target to size the
    # AOI based on how fast the shelf flows.
    mean_speed = float(np.mean(speed_sub[valid]))
    if mag < 1e-9:
        return (1.0, 0.0), 0.0, int(valid.sum()), mean_speed
    return (mvx / mag, mvy / mag), float(np.degrees(np.arctan2(mvy, mvx))), int(valid.sum()), mean_speed


def asym_upstream_weights(flow_unit, base=BASE_PERP, up=UP_DECAY, dn=DOWN_GROWTH):
    fux, fuy = flow_unit
    align = {"W": fux, "E": -fux, "N": -fuy, "S": fuy}
    weights = {}
    for side, a in align.items():
        if a >= 0: weights[side] = base * np.exp(-up * a)
        else:      weights[side] = base * np.exp(-dn * a)
    return weights


def greedy_grow(ts, strip, ix0, ix1, iy0, iy1, target, side_w):
    H, W = ts.shape
    cnt_ts = cum_count_fn(ts); cnt_st = cum_count_fn(strip)
    cur = (ix0, ix1, iy0, iy1); cur_ts = cnt_ts(*cur)
    def penalty(rect):
        a, b, c, d = rect
        a_buf = max(0, a-D_PX); b_buf = min(W, b+D_PX)
        c_buf = max(0, c-D_PX); d_buf = min(H, d+D_PX)
        return (cnt_ts(a_buf, b_buf, c_buf, d_buf) - cnt_ts(a, b, c, d)) * PENALTY_PER_PIXEL_KM2
    cur_pen = penalty(cur); step = max(1, min(W, H) // 60)
    def consider():
        a, b, c, d = cur; out = []
        sides = [
            ("W", (max(0, a-step), b, c, d), max(0, a-step), a, c, d),
            ("E", (a, min(W, b+step), c, d), b, min(W, b+step), c, d),
            ("N", (a, b, max(0, c-step), d), a, b, max(0, c-step), c),
            ("S", (a, b, c, min(H, d+step)), a, b, d, min(H, d+step)),
        ]
        for side, new_rect, ext_a, ext_b, ext_c, ext_d in sides:
            ts_added = cnt_ts(ext_a, ext_b, ext_c, ext_d)
            strip_added = cnt_st(ext_a, ext_b, ext_c, ext_d)
            ext_px = (ext_b - ext_a) * (ext_d - ext_c)
            if ext_px == 0: continue
            ext_area = ext_px * PIX_AREA_KM2
            d_area_w = max(0.0, side_w[side] * (ext_area - GAMMA * strip_added * PIX_AREA_KM2))
            d_pen = penalty(new_rect) - cur_pen
            out.append((side, new_rect, ts_added, d_area_w, ext_area, strip_added*PIX_AREA_KM2, d_pen))
        return out
    while cur_ts < target:
        cands = consider()
        if not cands: break
        scored = []
        for side, new_rect, ts_added, d_area_w, ext_area, strip_area, d_pen in cands:
            marginal = d_area_w + d_pen
            if ts_added == 0 and marginal > 0: score = -np.inf
            elif marginal <= 0: score = np.inf
            else: score = ts_added / marginal
            scored.append((score, new_rect, ts_added, d_pen))
        scored.sort(key=lambda x: -x[0])
        if scored[0][0] == -np.inf: break
        _, new_rect, ts_added, d_pen = scored[0]
        cur = new_rect; cur_ts += ts_added; cur_pen += d_pen
    while True:
        cands = consider()
        if not cands: break
        best = None
        for side, new_rect, ts_added, d_area_w, ext_area, strip_area, d_pen in cands:
            ext_uncovered = max(0.0, ext_area - GAMMA * strip_area)
            mo = ext_uncovered + d_pen
            if mo < 0:
                if best is None or mo < best[0]: best = (mo, new_rect, ts_added, d_pen)
        if best is None: break
        _, new_rect, ts_added, d_pen = best
        cur = new_rect; cur_ts += ts_added; cur_pen += d_pen
    return cur, cur_ts


def compute_and_write(shelf, name, out_shp, figures_dir):
    """Run the v15 sizing for one shelf; write the stack_extent shp + figure."""
    out_shp = Path(out_shp); figures_dir = Path(figures_dir)
    out_shp.parent.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    floating = floating_area_in_shelf(shelf)

    # Need v_shelf BEFORE auto_target, so compute masks + flow first.
    strip_boxes = basin_strips(shelf, BUFFER_KM*1000)
    ts, strip, sub_x, sub_y, vx_sub, vy_sub, speed_sub = make_masks(shelf, BUFFER_KM*1000, strip_boxes)
    ix0, ix1, iy0, iy1 = shelf_bbox_pixels(shelf, sub_x, sub_y)
    cnt_st = cum_count_fn(strip)
    flow_unit, flow_deg, n_in, v_shelf = shelf_only_flow(shelf, vx_sub, vy_sub, speed_sub, sub_x, sub_y)
    sw = asym_upstream_weights(flow_unit)

    tgt_km2, tgt_raw, scale = auto_target(floating, v_shelf)
    tgt_px = int(np.ceil(tgt_km2 / PIX_AREA_KM2))
    print(f"{name:10s}  {floating:>9.0f}  {v_shelf:>8.0f}  {scale:>6.2f}  "
          f"{tgt_raw:>9.0f}  → {tgt_km2:>5d} km²")

    new_box, achieved = greedy_grow(ts, strip, ix0, ix1, iy0, iy1, tgt_px, sw)
    a, blim, cl, dl = new_box
    rxmin, rxmax = sub_x[a], sub_x[min(blim-1, len(sub_x)-1)]
    rymax, rymin = sub_y[cl], sub_y[min(dl-1, len(sub_y)-1)]
    area = (rxmax-rxmin)*(rymax-rymin)/1e6
    achieved_km2 = achieved * PIX_AREA_KM2
    strip_in_rect = cnt_st(a, blim, cl, dl)
    frac_strip = 100 * strip_in_rect / max(1, (blim-a)*(dl-cl))
    aspect_xy = (rxmax-rxmin) / max(1, rymax-rymin)
    print(f"           rect={area:.0f} km²  TS got={achieved_km2:.0f} km²  strip-cov={frac_strip:.0f}%  x:y={aspect_xy:.2f}\n")

    rect_poly = box(rxmin, rymin, rxmax, rymax)
    schema = {"geometry": "Polygon", "properties": {"name": "str"}}
    crs = {"init": "epsg:3031"}
    with fiona.open(out_shp, "w", driver="ESRI Shapefile",
                     schema=schema, crs=crs) as dst:
        dst.write({"geometry": mapping(rect_poly),
                   "properties": {"name": f"{name}_stack_extent_v15"}})
    print(f"           wrote {out_shp}")

    fig, ax = plt.subplots(figsize=(11, 9.5))
    extent = [sub_x.min(), sub_x.max(), sub_y.min(), sub_y.max()]
    rgb = np.zeros((*ts.shape, 3), dtype=np.float32); rgb[..., :] = 0.15
    rgb[strip > 0] = [0.55, 0.45, 0.10]; rgb[ts > 0] = [1.0, 1.0, 1.0]
    ax.imshow(rgb, extent=extent, origin="upper", alpha=0.95)
    cx, cy = shelf.centroid.x, shelf.centroid.y
    arrow_len = 30000
    fux, fuy = flow_unit
    ax.annotate("", xy=(cx + fux * arrow_len, cy + fuy * arrow_len), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color="cyan", lw=2.0))
    ax.annotate("", xy=(cx - fux * arrow_len * 1.2, cy - fuy * arrow_len * 1.2), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color="magenta", lw=2.5))
    for p in ([shelf] if shelf.geom_type=="Polygon" else list(shelf.geoms)):
        xs, ys = p.exterior.xy; ax.plot(xs, ys, color="red", linewidth=2.2)
    ax.add_patch(mpatches.Rectangle((rxmin, rymin), rxmax-rxmin, rymax-rymin,
                                     fill=False, edgecolor="tab:green", linewidth=3.0))
    handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="TS pixels"),
        mpatches.Patch(facecolor=(0.55,0.45,0.10), edgecolor="black", label="strip-only"),
        mpatches.Patch(facecolor="none", edgecolor="red", linewidth=2.2,
                       label=f"shelf AOI ({shelf.area/1e6:.0f} km², floating {floating:.0f} km²)"),
        mpatches.Patch(facecolor="none", edgecolor="cyan", linewidth=2.0,
                       label=f"flow downstream ({flow_deg:+.0f}°)"),
        mpatches.Patch(facecolor="none", edgecolor="magenta", linewidth=2.5,
                       label="upstream + perp bias direction"),
        mpatches.Patch(facecolor="none", edgecolor="tab:green", linewidth=3.0,
                       label=f"auto AOI: TS≥{tgt_km2}km²  rect {area:.0f} km², TS {achieved_km2:.0f}, strip {frac_strip:.0f}%"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=9)
    ax.set_xlabel("Easting (m, EPSG:3031)"); ax.set_ylabel("Northing (m, EPSG:3031)")
    ax.set_title(
        f"{name}: target {tgt_km2} km²  "
        f"(floating {floating:.0f} km² × {scale:.2f} = {tgt_raw:.0f} → snap; "
        f"v_shelf={v_shelf:.0f} m/yr)"
    )
    ax.set_aspect("equal"); fig.tight_layout()
    out_png = figures_dir / f"ts_aoi_v15_{name.lower()}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"           wrote {out_png}")


def load_shelf_geom(path, feature_name=None):
    """Load a shelf polygon: a named feature's union, or the first feature."""
    with fiona.open(path) as src:
        if feature_name is not None:
            geoms = [shape(f["geometry"]) for f in src
                     if str(f["properties"].get("NAME", "")) == feature_name]
            if not geoms:
                raise SystemExit(f"feature NAME={feature_name!r} not found in {path}")
            return unary_union(geoms)
        return shape(next(iter(src))["geometry"])


def resolve_inputs(args):
    """Map CLI args to (shelf_geom, name, out_shp, figures_dir)."""
    figures_dir = Path(args.figures_dir) if args.figures_dir else (ROOT / "figures")

    if args.basin:
        try:
            config = importlib.import_module(f"{args.basin}.config")
        except ImportError as e:
            raise SystemExit(
                f"--basin {args.basin}: cannot import {args.basin}.config ({e}). "
                f"Run from the repo root with PYTHONPATH including it.")
        name = getattr(config, "SHELF", args.basin)
        shape_dir = Path(config.SHAPE_DIR)
        shelf_shp = getattr(config, "SHELF_INPUT_SHP", None)
        feature_name = getattr(config, "ICESHELF_FEATURE_NAME", None)
        feature_shp = getattr(config, "ICESHELF_FEATURE_SHP", None)
        if shelf_shp and Path(shelf_shp).exists():
            shelf = load_shelf_geom(Path(shelf_shp))
        elif feature_name and feature_shp and Path(feature_shp).exists():
            shelf = load_shelf_geom(Path(feature_shp), feature_name=feature_name)
        elif (shape_dir / f"{args.basin}.shp").exists():
            shelf = load_shelf_geom(shape_dir / f"{args.basin}.shp")
        else:
            raise SystemExit(
                f"--basin {args.basin}: no shelf input found. Set config.SHELF_INPUT_SHP, "
                f"or config.ICESHELF_FEATURE_SHP + ICESHELF_FEATURE_NAME, "
                f"or place {args.basin}.shp in {shape_dir}.")
        out_shp = shape_dir / f"{args.basin}_stack_extent.shp"
        return shelf, name, out_shp, figures_dir

    if not args.out:
        raise SystemExit("--out is required unless a basin name is given.")
    if args.feature_shp:
        if not args.feature_name:
            raise SystemExit("--feature-shp requires --feature-name.")
        shelf = load_shelf_geom(Path(args.feature_shp), feature_name=args.feature_name)
        name = args.name or args.feature_name
    elif args.shelf:
        shelf = load_shelf_geom(Path(args.shelf))
        name = args.name or Path(args.shelf).stem
    else:
        raise SystemExit("Provide a basin name, or --shelf, or --feature-shp + --feature-name.")
    return shelf, name, Path(args.out), figures_dir


def main():
    global MEASURES, BEDMACHINE, STRIPS_DIR
    global V_THRESH, BUFFER_KM, TARGET_BASE, TARGET_VFACTOR, V_REF
    p = argparse.ArgumentParser(description="Compute a basin's v15 stack-extent AOI rectangle.")
    p.add_argument("basin", nargs="?", help="basin package name; reads inputs from <basin>.config")
    p.add_argument("--shelf", help="narrow shelf polygon (shapefile/geojson)")
    p.add_argument("--feature-shp", help="multi-feature shapefile to extract a shelf from")
    p.add_argument("--feature-name", help="NAME of the feature to extract from --feature-shp")
    p.add_argument("--out", help="output stack_extent shapefile (required unless basin given)")
    p.add_argument("--name", help="label for the figure/title (explicit modes)")
    p.add_argument("--measures", help=f"MEaSUREs phase-map .nc (default {MEASURES})")
    p.add_argument("--bedmachine", help=f"BedMachine .nc (default {BEDMACHINE})")
    p.add_argument("--strips-dir", help=f"REMA strips dir (default {STRIPS_DIR})")
    p.add_argument("--figures-dir", help="output dir for the diagnostic png (default <root>/figures)")
    p.add_argument("--v-thresh", type=float, help=f"slow-ice speed threshold m/yr (default {V_THRESH})")
    p.add_argument("--buffer-km", type=float, help=f"search buffer km (default {BUFFER_KM})")
    p.add_argument("--target-base", type=float, help=f"scale() intercept (default {TARGET_BASE})")
    p.add_argument("--target-vfactor", type=float, help=f"scale() slope (default {TARGET_VFACTOR})")
    p.add_argument("--v-ref", type=float, help=f"scale() velocity ref m/yr (default {V_REF})")
    args = p.parse_args()

    if args.measures: MEASURES = Path(args.measures)
    if args.bedmachine: BEDMACHINE = Path(args.bedmachine)
    if args.strips_dir: STRIPS_DIR = Path(args.strips_dir)
    if args.v_thresh is not None: V_THRESH = args.v_thresh
    if args.buffer_km is not None: BUFFER_KM = args.buffer_km
    if args.target_base is not None: TARGET_BASE = args.target_base
    if args.target_vfactor is not None: TARGET_VFACTOR = args.target_vfactor
    if args.v_ref is not None: V_REF = args.v_ref

    shelf, name, out_shp, figures_dir = resolve_inputs(args)
    load_grids(MEASURES, BEDMACHINE)

    print(f"V_THRESH={V_THRESH}; target buckets={TARGET_BUCKETS}")
    print(f"scale(v) = {TARGET_BASE} + {TARGET_VFACTOR} * log10(1 + v/{V_REF})\n")
    print(f"{'basin':10s}  {'floating':>9s}  {'v_shelf':>8s}  {'scale':>6s}  "
          f"{'raw':>9s}  {'target':>7s}")
    print("-"*68)
    compute_and_write(shelf, name, out_shp, figures_dir)


if __name__ == "__main__":
    main()

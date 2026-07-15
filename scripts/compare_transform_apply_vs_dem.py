"""
Validate that the 384-byte ASP -transform.txt is sufficient to reconstruct
the aligned -trans_reference-DEM.tif from the un-aligned source REMA strip.

Workflow per strip:
  1. Load source REMA strip (EPSG:3031, 2 m).
  2. Load 4x4 ECEF transform from -transform.txt.
  3. For each valid pixel: (x, y, z) EPSG:3031 -> ECEF -> apply T -> back to
     EPSG:3031 -> (x', y', z'). Done in row chunks to keep RAM bounded.
  4. Bin transformed points onto the existing -DEM.tif grid (point2dem-style
     average within the destination pixel).
  5. Diff against -trans_reference-DEM.tif. Report stats + per-phase timing.

If the diff is small (~sub-pixel vertical) we know transform.txt alone is a
faithful proxy for the aligned DEM, which justifies deleting both the big
-trans_reference.tif point cloud *and*, in the limit, the gridded -DEM.tif.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer


def apply_transform(src_path: Path, ref_path: Path, tfm_path: Path,
                    row_chunk: int = 2000) -> dict:
    timings: dict = {}

    # ---- Phase 1: load source DEM ----
    t = time.time()
    with rasterio.open(src_path) as ds:
        src_shape = ds.shape
        src_transform = ds.transform
        src_nodata = ds.nodata
        # Stream rows, don't slurp 3.5 GB at once
    timings["open_src"] = time.time() - t

    # ---- Phase 2: load 4x4 transform ----
    t = time.time()
    T = np.loadtxt(tfm_path)
    assert T.shape == (4, 4), f"Expected 4x4, got {T.shape}"
    timings["load_transform"] = time.time() - t

    # ---- Phase 3: open ref DEM (grid we project onto) ----
    t = time.time()
    with rasterio.open(ref_path) as ds:
        ref_shape = ds.shape
        ref_transform = ds.transform
        ref_nodata = ds.nodata
        # We need this in memory for the final diff
        ref_data = ds.read(1).astype(np.float32)
    timings["open_ref"] = time.time() - t

    # ---- Phase 4: row-chunked ECEF round-trip + bin onto ref grid ----
    t = time.time()
    to_ecef = Transformer.from_crs("EPSG:3031", "EPSG:4978", always_xy=True)
    fr_ecef = Transformer.from_crs("EPSG:4978", "EPSG:3031", always_xy=True)

    # Output accumulator (ref grid)
    sum_grid = np.zeros(ref_shape, dtype=np.float64)
    cnt_grid = np.zeros(ref_shape, dtype=np.int32)
    flat_sum = sum_grid.reshape(-1)
    flat_cnt = cnt_grid.reshape(-1)

    a_src, b_src, c_src = src_transform.a, src_transform.b, src_transform.c
    d_src, e_src, f_src = src_transform.d, src_transform.e, src_transform.f
    a_ref, e_ref, c_ref, f_ref = ref_transform.a, ref_transform.e, ref_transform.c, ref_transform.f
    n_ref_rows, n_ref_cols = ref_shape

    n_valid_total = 0
    with rasterio.open(src_path) as ds:
        for row0 in range(0, src_shape[0], row_chunk):
            row1 = min(row0 + row_chunk, src_shape[0])
            block = ds.read(1, window=rasterio.windows.Window(0, row0, src_shape[1], row1 - row0))
            valid = block != src_nodata
            if not valid.any():
                continue
            ys_local, xs_local = np.where(valid)
            ys = ys_local + row0
            xs = xs_local
            z = block[valid].astype(np.float64)

            # 3031 world coords (rasterio affine: x = a*col + b*row + c; b/d are 0 for axis-aligned)
            x_world = c_src + (xs + 0.5) * a_src + (ys + 0.5) * b_src
            y_world = f_src + (xs + 0.5) * d_src + (ys + 0.5) * e_src

            # 3031 -> ECEF
            X, Y, Z = to_ecef.transform(x_world, y_world, z)

            # Apply 4x4 in-place-ish
            X2 = T[0, 0] * X + T[0, 1] * Y + T[0, 2] * Z + T[0, 3]
            Y2 = T[1, 0] * X + T[1, 1] * Y + T[1, 2] * Z + T[1, 3]
            Z2 = T[2, 0] * X + T[2, 1] * Y + T[2, 2] * Z + T[2, 3]

            # ECEF -> 3031
            x2, y2, z2 = fr_ecef.transform(X2, Y2, Z2)

            # Bin onto ref grid (nearest pixel; same as point2dem with mean filter)
            col = np.floor((x2 - c_ref) / a_ref).astype(np.int64)
            row = np.floor((y2 - f_ref) / e_ref).astype(np.int64)
            in_b = (col >= 0) & (col < n_ref_cols) & (row >= 0) & (row < n_ref_rows)
            if in_b.any():
                flat_idx = row[in_b] * n_ref_cols + col[in_b]
                np.add.at(flat_sum, flat_idx, z2[in_b])
                np.add.at(flat_cnt, flat_idx, 1)

            n_valid_total += valid.sum()

    out = np.full(ref_shape, np.nan, dtype=np.float64)
    has = cnt_grid > 0
    out[has] = sum_grid[has] / cnt_grid[has]
    timings["transform+bin"] = time.time() - t

    # ---- Phase 5: diff stats ----
    t = time.time()
    ref_valid = (ref_data != ref_nodata) & np.isfinite(ref_data)
    both = ref_valid & np.isfinite(out)
    diff = out[both] - ref_data[both]
    stats = {
        "ref_path": str(ref_path),
        "src_path": str(src_path),
        "src_shape": src_shape,
        "ref_shape": ref_shape,
        "n_valid_src": int(n_valid_total),
        "n_pixels_common": int(both.sum()),
        "coverage_pct": float(both.sum()) / float(ref_valid.sum()) * 100.0 if ref_valid.any() else 0.0,
        "mean_diff_m": float(diff.mean()) if diff.size else float("nan"),
        "median_diff_m": float(np.median(diff)) if diff.size else float("nan"),
        "std_diff_m": float(diff.std()) if diff.size else float("nan"),
        "mad_m": float(np.median(np.abs(diff - np.median(diff)))) if diff.size else float("nan"),
        "p95_abs_diff_m": float(np.percentile(np.abs(diff), 95)) if diff.size else float("nan"),
        "p99_abs_diff_m": float(np.percentile(np.abs(diff), 99)) if diff.size else float("nan"),
        "max_abs_diff_m": float(np.abs(diff).max()) if diff.size else float("nan"),
        "transform_matrix": T.tolist(),
    }
    timings["diff"] = time.time() - t
    timings["total"] = sum(timings.values())
    stats["timings"] = timings
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Source REMA strip (un-aligned)")
    p.add_argument("--ref", required=True, help="Aligned -trans_reference-DEM.tif")
    p.add_argument("--tfm", required=True, help="-transform.txt")
    p.add_argument("--row-chunk", type=int, default=2000)
    args = p.parse_args()

    stats = apply_transform(Path(args.src), Path(args.ref), Path(args.tfm),
                            row_chunk=args.row_chunk)

    print(f"Source : {stats['src_path']}")
    print(f"Aligned: {stats['ref_path']}")
    print(f"Source shape : {stats['src_shape']}  ({stats['n_valid_src']/1e6:.1f} Mpx valid)")
    print(f"Aligned shape: {stats['ref_shape']}")
    print(f"Common pixels: {stats['n_pixels_common']/1e6:.1f} Mpx  "
          f"({stats['coverage_pct']:.1f}% of aligned valid)")
    print()
    print("Diff (our_transform_apply - existing_aligned_DEM), in metres:")
    print(f"  mean   : {stats['mean_diff_m']:+.4f}")
    print(f"  median : {stats['median_diff_m']:+.4f}")
    print(f"  std    : {stats['std_diff_m']:.4f}")
    print(f"  MAD    : {stats['mad_m']:.4f}")
    print(f"  |p95|  : {stats['p95_abs_diff_m']:.4f}")
    print(f"  |p99|  : {stats['p99_abs_diff_m']:.4f}")
    print(f"  |max|  : {stats['max_abs_diff_m']:.4f}")
    print()
    print("Timing (seconds):")
    for k, v in stats["timings"].items():
        print(f"  {k:<18}: {v:7.2f}")


if __name__ == "__main__":
    main()

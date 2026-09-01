"""Recompress existing stack NetCDFs in place with zlib (lossless, verified).

One-time migration. ``save_stack`` now writes zlib-compressed NetCDF, but stacks
built before that change are uncompressed float64 + ~98% NaN, so they shrink
enormously. Each file is rewritten via an atomic temp-file + ``os.replace``,
with every data variable verified **bit-exact** (NaN-aware) against the original
before the swap. Already-compressed files are skipped. Safe to interrupt: the
original is never touched until a verified temp atomically replaces it; a killed
run leaves only a ``*.recompress.tmp`` orphan.

Usage:
    python scripts/recompress_stacks.py [--glob PATTERN]... [--complevel 4] [--dry-run]

Default target: every ``<basin>/processed/*_stack_*.nc`` under the project root.
"""
from __future__ import annotations

import argparse
import gc
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path("/wd2/projects/stereo_melt")
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
BASINS = ["beardmore", "nansen", "pig", "dotson_crosson", "mcmurdo"]


def gb(n):
    return n / 1024**3


def _mem_available_gb():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    return float("nan")


def arrays_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if np.issubdtype(a.dtype, np.floating):
        return bool(np.array_equal(a, b, equal_nan=True))
    try:
        return bool(np.array_equal(a, b))
    except TypeError:
        return bool((a == b).all())


def already_compressed(ds):
    for _, var in ds.data_vars.items():
        if var.ndim >= 2 and np.issubdtype(var.dtype, np.floating):
            if var.encoding.get("zlib", False):
                return True
    return False


def build_encoding(ds, complevel):
    enc = {}
    for name, var in ds.data_vars.items():
        if var.ndim >= 1 and np.issubdtype(var.dtype, np.floating):
            # Clear only the vars we re-chunk: a contiguous read carries
            # contiguous=True, which collides with chunksizes on write.
            # Coords (incl. time's CF units) and scalar vars keep their
            # encoding so they round-trip exactly.
            ds[name].encoding = {}
            chunks = tuple(
                1 if d == "time" else s for d, s in zip(var.dims, var.shape)
            )
            enc[name] = {"zlib": True, "complevel": complevel, "chunksizes": chunks}
    return enc


def verify_files_equal(pa, pb):
    """True iff every data var + coord in pa equals pb (NaN-aware), one var at a
    time to cap peak RAM at ~2x the largest variable."""
    with xr.open_dataset(pa) as da, xr.open_dataset(pb) as db:
        if set(da.variables) != set(db.variables):
            return False, "variable-set"
        names = list(da.data_vars) + list(da.coords)
        for name in names:
            a = np.asarray(da[name].values)
            b = np.asarray(db[name].values)
            ok = arrays_equal(a, b)
            del a, b
            gc.collect()
            if not ok:
                kind = "var" if name in da.data_vars else "coord"
                return False, f"{kind}:{name}"
    return True, None


def recompress_one(path, complevel, dry_run):
    path = Path(path)
    size0 = path.stat().st_size
    with xr.open_dataset(path) as ds:
        if already_compressed(ds):
            return ("skip-compressed", size0, size0)
        enc = build_encoding(ds, complevel)
        if not enc:
            return ("skip-no-float-var", size0, size0)
        if dry_run:
            return ("would-compress", size0, None)
        tmp = path.parent / (path.name + ".recompress.tmp")
        try:
            ds.to_netcdf(tmp, encoding=enc)
        except Exception as exc:  # noqa: BLE001
            if tmp.exists():
                tmp.unlink()
            return (f"ERROR-write:{exc}", size0, None)
    ok, bad = verify_files_equal(path, tmp)
    if not ok:
        tmp.unlink(missing_ok=True)
        return (f"FAIL-verify:{bad}", size0, None)
    size1 = tmp.stat().st_size
    os.replace(tmp, path)  # atomic
    return ("ok", size0, size1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", action="append", default=None,
                    help="glob(s) of .nc files; repeatable. Default: all basin stacks.")
    ap.add_argument("--complevel", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.glob:
        files = []
        for g in args.glob:
            files += [Path(p) for p in glob.glob(g)]
    else:
        files = []
        for b in BASINS:
            d = EXAMPLES / b / "processed"
            if d.is_dir():
                files += d.glob("*_stack_*.nc")
    files = [f for f in sorted(set(files)) if not f.name.endswith(".recompress.tmp")]
    files.sort(key=lambda p: p.stat().st_size)  # small first: cheap wins, validate early

    print(f"recompress: {len(files)} candidate file(s) | "
          f"MemAvailable={_mem_available_gb():.0f} GB | complevel={args.complevel}"
          f"{' | DRY RUN' if args.dry_run else ''}", flush=True)
    tot0 = tot1 = 0
    n_ok = n_skip = n_fail = 0
    for i, f in enumerate(files, 1):
        t0 = time.time()
        status, s0, s1 = recompress_one(f, args.complevel, args.dry_run)
        dt = time.time() - t0
        tot0 += s0
        tot1 += s1 if s1 is not None else s0
        s1_gb = gb(s1) if s1 is not None else float("nan")
        ratio = (s0 / s1) if (s1 and s1 > 0) else float("nan")
        print(f"[{i}/{len(files)}] {status:>18}  {gb(s0):6.3f}->{s1_gb:6.3f} GB  "
              f"{ratio:5.1f}x  {dt:6.1f}s  {f.name}", flush=True)
        if status == "ok":
            n_ok += 1
        elif status.startswith(("FAIL", "ERROR")):
            n_fail += 1
        else:
            n_skip += 1

    print(f"\nTOTAL {gb(tot0):.3f} -> {gb(tot1):.3f} GB | reclaimed {gb(tot0 - tot1):.3f} GB "
          f"| ok={n_ok} skip={n_skip} fail={n_fail}", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

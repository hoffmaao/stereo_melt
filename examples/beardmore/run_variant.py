"""Run build_stack -> tilt_fit -> run_melt for a chosen ASP variant.

Variants:
  is2     — canonical IS2-only ASP coregistration (`STRIPS/ASP/asp_aligned`)
  cs2     — CS2-only forward test (`STRIPS/ASP_cs2/asp_aligned`)
  is2+cs2 — combined IS2+CS2 control (`STRIPS/ASP_is2cs2/asp_aligned`)

Each variant's stack, tilt-corrected stack, melt result, and figures
land in `processed/<variant>/`, `results/<variant>/`, `figures/<variant>/`
respectively. The default `is2` run still writes to the legacy
top-level dirs to stay backward-compatible with prior outputs.

Run:
    python -m beardmore.run_variant --variant cs2
    python -m beardmore.run_variant --variant is2+cs2
    python -m beardmore.run_variant --all     # cs2 + is2+cs2
"""

from __future__ import annotations

import argparse
import os
import sys

# PROJ_DATA must be set BEFORE any pyproj-using import (fiona, rioxarray,
# pyTMD, etc.). _run_one imports build_stack first, which pulls in fiona;
# if PROJ_DATA isn't set yet, pyproj initializes with the broken base
# conda proj.db and pyTMD's later EPSG:3031 lookup fails (silent until
# tilt_fit's apply_tide_ibe_to_stack).
_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

from beardmore import config

VARIANTS = {
    "is2":      ("ASP",          ""),       # canonical
    "cs2":      ("ASP_cs2",      "cs2"),
    "is2+cs2":  ("ASP_is2cs2",   "is2cs2"),
}


def _patch_config_for_variant(variant: str) -> dict:
    """Repoint `config` paths to a variant's input/output dirs and return the new paths."""
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}; pick from {list(VARIANTS)}")
    asp_subdir, suffix = VARIANTS[variant]

    config.STRIP_ALIGNED_DIR = config.STRIPS_DIR / asp_subdir / "asp_aligned"
    if not config.STRIP_ALIGNED_DIR.exists():
        raise SystemExit(
            f"Aligned-DEM dir not found for variant {variant!r}: {config.STRIP_ALIGNED_DIR}"
        )

    if suffix:
        config.PROCESSED_DIR = config.BASIN_DIR / "processed" / suffix
        config.FIGURES_DIR = config.BASIN_DIR / "figures" / suffix
        config.RESULTS_DIR = config.BASIN_DIR / "results" / suffix
    config.ensure_output_dirs()

    return {
        "variant": variant,
        "STRIP_ALIGNED_DIR": config.STRIP_ALIGNED_DIR,
        "PROCESSED_DIR": config.PROCESSED_DIR,
        "FIGURES_DIR": config.FIGURES_DIR,
        "RESULTS_DIR": config.RESULTS_DIR,
    }


def _run_one(variant: str) -> None:
    paths = _patch_config_for_variant(variant)
    print("=" * 72)
    print(f"VARIANT: {variant}")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print("=" * 72)

    # Imports happen *after* config is patched, since these modules read
    # `config.STRIP_ALIGNED_DIR` etc. as defaults at function-call time.
    from beardmore import build_stack, tilt_fit, run_melt

    print(f"\n[{variant}] === build_stack ===")
    build_stack.main()

    print(f"\n[{variant}] === tilt_fit ===")
    tilt_fit.main()

    print(f"\n[{variant}] === run_melt (Eulerian + Lagrangian + linear-inverse) ===")
    run_melt.main()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=list(VARIANTS), help="Single variant to run.")
    p.add_argument("--all", action="store_true", help="Run cs2 and is2+cs2 (skips is2 — already on disk).")
    args = p.parse_args()

    if args.all:
        targets = ["cs2", "is2+cs2"]
    elif args.variant:
        targets = [args.variant]
    else:
        p.error("pick --variant or --all")

    for v in targets:
        _run_one(v)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Scaffold a new basin package from the canonical templates.

Stands up a new ice-shelf driver directory with the basic information a
new study needs — name, AOI, and the period of observations — everything
else comes from the shared library drivers
(:mod:`stereo_melt.coregister.align_driver`, ``run_cache_*_main``) plus
config:

    python scripts/new_basin.py --name getz --shelf-title Getz \\
        --start 2019-01-01 --end 2024-01-10 \\
        [--aoi getz_stack_extent.shp] [--tide-model CATS2008] [--dest .]

What it stamps (thin wrappers, ~40-70 lines each):
  __init__.py, config.py, fetch_strips.py, cache_climate.py,
  cache_icesat2.py, cache_cryotempo.py, cache_atm.py, cache_lvis.py,
  cache_glas.py, align_strips.py, NOTES.md
plus the phase-2 heavies copied verbatim-with-renames from the template
basin (build_stack.py, tilt_fit.py, run_melt.py, run_pseudospectral.py,
run_stationary.py, scripts/find_bad_epochs.py) — review those before
trusting them; they still carry per-basin science choices.

Template sources: ``venable`` (freshest complete IS2-era basin) for most
stages; ``beardmore_shelf`` for cache_cryotempo/cache_atm; ``pig`` for
cache_lvis. Renames are systematic (name / NAME / Title). BAD_STRIPS /
BAD_EPOCHS are reset to empty. The window and AOI lines in config.py are
rewritten from the CLI arguments.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
EXAMPLES = WORKSPACE / "examples"

# (template basin, file) → stamped file. Order matters only for the report.
TEMPLATE_FILES: list[tuple[str, str]] = [
    ("venable", "config.py"),
    ("venable", "fetch_strips.py"),
    ("venable", "cache_climate.py"),
    ("venable", "cache_icesat2.py"),
    ("beardmore_shelf", "cache_cryotempo.py"),
    ("beardmore_shelf", "cache_atm.py"),
    ("pig", "cache_lvis.py"),
    ("venable", "cache_glas.py"),
    ("venable", "align_strips.py"),
    # Phase-2 heavies — copied so the pipeline is runnable end-to-end, but
    # they are NOT yet library-driven; review before trusting.
    ("venable", "build_stack.py"),
    ("venable", "tilt_fit.py"),
    ("venable", "run_melt_path.py"),
    ("venable", "run_melt.py"),
    ("venable", "run_pseudospectral.py"),
    ("venable", "run_stationary.py"),
    ("venable", "scripts/find_bad_epochs.py"),
]

# Per-template identifier stems: (lower, UPPER, Title...) — every spelling
# a template file may use, replaced by the new basin's spellings.
TEMPLATE_IDENTIFIERS = {
    "venable": [("venable", "VENABLE", "Venable")],
    "beardmore_shelf": [
        ("beardmore_shelf", "BEARDMORE_SHELF", "Beardmore_Shelf"),
        ("beardmore_shelf", "BEARDMORE_SHELF", "Beardmore-shelf"),
    ],
    "pig": [("pig", "PIG", "PineIsland"), ("pig", "PIG", "Pine Island")],
}


def _rename(text: str, template: str, name: str, upper: str, title: str) -> str:
    for lo, up, ti in TEMPLATE_IDENTIFIERS[template]:
        text = text.replace(ti, title)
        text = text.replace(up, upper)
        text = text.replace(lo, name)
    return text


def _rewrite_config(text: str, args, upper: str) -> str:
    """Apply window/AOI/tide substitutions + reset basin-specific state."""
    text = re.sub(r'^START_TIME\s*=\s*".*"', f'START_TIME = "{args.start}"',
                  text, flags=re.M)
    text = re.sub(r'^END_TIME\s*=\s*".*"(.*)$', f'END_TIME = "{args.end}"',
                  text, flags=re.M)
    text = re.sub(r'^TIDE_MODEL\s*=\s*".*"', f'TIDE_MODEL = "{args.tide_model}"',
                  text, flags=re.M)
    if args.aoi:
        text = re.sub(
            rf'^{upper}_AOI_SHP\s*=\s*SHAPE_DIR\s*/\s*".*"',
            f'{upper}_AOI_SHP = SHAPE_DIR / "{args.aoi}"',
            text, flags=re.M)
    # Fresh basin: no curated strip/epoch rejections yet. Handles both the
    # plain and type-annotated forms, single-line ``()`` and multi-line
    # tuples (whose entries may carry parenthesized comments) — the
    # multi-line branch consumes up to the first line-anchored ``)``.
    for var in ("BAD_STRIPS", "BAD_EPOCHS"):
        text = re.sub(
            rf'^{var}(?::[^=\n]*)?\s*=\s*\((?:[^\n)]*\)|[\s\S]*?\n\s*\))',
            f"{var}: tuple[str, ...] = ()", text, flags=re.M)
    return text


NOTES_MD = """# NOTES.md — {name}/ (local decision record, untracked)

The **{title}** application of the `stereo_melt` library, scaffolded by
`scripts/new_basin.py`. Canonical stage sequence: [`../PIPELINE.md`](../PIPELINE.md)
— run `$PY -m {name}.<stage>`; keep this file **deltas-only** (identity,
decision record, genuinely local caveats — no generic command blocks).

## Decision record (fill in as decided)

- **Window** (`config.py` authoritative): `{start}` → `{end}`.
- **Grid**: see `config.RES` (template default; 125 m for large AOIs —
  the beardmore_shelf precedent).
- **Geoid + MDT or geoid-only**: MDT applies only north of DTU22's −79°S
  limit — record the decision here.
- **Velocity**: MEaSUREs NSIDC-0754 is the default; ITS_LIVE only where
  feature tracking works — record the choice and why.
- **Tide model {tide_model}**; AOI at `data/shapefiles/{aoi}`.
- **Control eras**: which of IS2 / CryoTEMPO / ATM-LVIS / GLAS the window
  touches, and the align roots they map to (`data/ASP*/`, never shared).

## Scaffold checklist (delete lines as you complete them)

- [ ] Drop the AOI shapefile at `data/shapefiles/{aoi}` (or run
      `python scripts/compute_aoi.py {name}` after a seed fetch for a
      strip-aware v15 rectangle).
- [ ] If the AOI centroid is south of -79°S, DTU22 has no MDT coverage —
      switch tilt_fit to the geoid-only chain (see beardmore_shelf).
- [ ] Check velocity coverage: MEaSUREs NSIDC-0754 is the default;
      ITS_LIVE only where feature tracking works.
- [ ] Review the copied build_stack/tilt_fit/run_* for template-basin
      assumptions before the first full run.
- [ ] After the first tilt_fit + find_bad_epochs pass: populate
      `BAD_STRIPS` (dem_id-keyed), re-run tilt_fit, then the production
      solver pair `run_melt_path` + `run_melt`.
- [ ] Rewrite the Decision record above with the actual choices; delete
      this checklist.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True,
                    help="Package name (lower_snake), e.g. 'getz'")
    ap.add_argument("--shelf-title", required=True,
                    help="Title-case shelf name, e.g. 'Getz'")
    ap.add_argument("--start", required=True, help="Window start YYYY-MM-DD")
    ap.add_argument("--end", required=True,
                    help="Window end YYYY-MM-DD (s2s041 index ceiling is "
                         "2024-01-10 as of 2026-07)")
    ap.add_argument("--aoi", default=None,
                    help="AOI shapefile name under data/shapefiles/ "
                         "(default: <name>_stack_extent.shp)")
    ap.add_argument("--tide-model", default="CATS2008")
    ap.add_argument("--dest", default=str(EXAMPLES),
                    help="Driver root to stamp into (default: <repo>/examples)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing basin dir")
    args = ap.parse_args()

    name = args.name
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        ap.error("--name must be lower_snake_case")
    upper = name.upper()
    title = args.shelf_title
    if args.aoi is None:
        args.aoi = f"{name}_stack_extent.shp"

    dest_root = Path(args.dest)
    basin_dir = dest_root / name
    if basin_dir.exists() and not args.force:
        raise SystemExit(f"❌ {basin_dir} already exists (use --force to overwrite)")
    (basin_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (basin_dir / "logs").mkdir(exist_ok=True)

    (basin_dir / "__init__.py").write_text("")
    scripts_init = basin_dir / "scripts" / "__init__.py"
    scripts_init.write_text("")

    stamped, missing = [], []
    for template, rel in TEMPLATE_FILES:
        src = EXAMPLES / template / rel
        if not src.exists():
            missing.append(f"{template}/{rel}")
            continue
        text = src.read_text()
        text = _rename(text, template, name, upper, title)
        if rel == "config.py":
            text = _rewrite_config(text, args, upper)
        out = basin_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        stamped.append(rel)

    (basin_dir / "NOTES.md").write_text(NOTES_MD.format(
        name=name, title=title, aoi=args.aoi, start=args.start,
        end=args.end, tide_model=args.tide_model))

    print(f"✅ scaffolded {basin_dir}")
    for rel in stamped:
        print(f"   {rel}")
    if missing:
        print("⚠  template files not found (skipped):")
        for m in missing:
            print(f"   {m}")

    # Import smoke test — catches renamed-attribute mismatches immediately.
    import importlib
    sys.path.insert(0, str(dest_root))
    failures = []
    for rel in ["config"] + [Path(r).stem for r in stamped if r != "config.py"
                             and "/" not in r]:
        try:
            importlib.import_module(f"{name}.{rel}")
        except Exception as exc:
            failures.append((rel, str(exc)))
    if failures:
        print("\n⚠  import failures to fix before first run:")
        for rel, err in failures:
            print(f"   {name}.{rel}: {err}")
    else:
        print("✅ all stamped modules import cleanly")

    print(f"\nNext: see {basin_dir}/NOTES.md for the checklist "
          f"(AOI file, MDT latitude rule, velocity source).")


if __name__ == "__main__":
    main()

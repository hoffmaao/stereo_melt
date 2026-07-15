# CLAUDE.md — beardmore_shelf/

The **Beardmore shelf-sector** application of the `stereo_melt` library: the
Ross Ice Shelf sector immediately downstream of the Beardmore Glacier
grounding zone (~−84°S / 171°E, Transantarctic Mountains front). Canonical
stage sequence: [`../PIPELINE.md`](../PIPELINE.md) — run
`$PY -m beardmore_shelf.<stage>`; this file is the per-shelf record only.

Began as the clean IS2-era re-run of the legacy `beardmore/` study (v15 AOI,
per-basin roots) and since 2026-07-11 is the **full-record 2009→2024 study**:
strip ingestion spans every control era, querying all available altimetric
control — ICESat-2 ATL06 (`cache_icesat2`), CryoTEMPO-LI CryoSat-2
(`cache_cryotempo`, 2010-07 onward), IceBridge ATM/LVIS (`cache_atm`; LVIS
was queried and has zero coverage this far south), and ICESat-1 GLAS
(`cache_glas`, 2009 → 2010-10) — plus Shean-style nocorr ingestion
(`ingest_nocorr`, per-era class biases) for strips no control can anchor.
The two Beardmore studies share only the raw-strip pool under
`data/REMA/strips/`.

## Decision record

- **Study window: `2009-01-01 → 2024-01-10`** (full record; end = s2s041
  index ceiling). `config.py` defaults to the IS2-era slice (`2019-01-01 →`);
  full-record runs widen via `BEARDMORE_SHELF_START=2009-01-01` and fuse all
  four align roots with `BEARDMORE_SHELF_SOURCES=full` (precedence ASP →
  ASP_ctempoatm → ASP_glas → ASP_nocorr; `combined`/`nocorr` submodes exist).
- **Grid 125 m** (production since 2026-07-05; the 25 m first-run products
  live on under the `--res 25` suffix — at 25 m the tilt LSQ was a ~21 h
  solve).
- **Geoid-only** (~−84°S, poleward of DTU22's −79°S limit — no MDT; matches
  `beardmore/`, unlike the West Antarctic basins).
- **Velocity: MEaSUREs phase map (NSIDC-0754)** time-mean — no time-varying
  product reaches −84°S; ITS_LIVE is empty over the slow grounding zone.
  Small effect on this slow shelf, but a known caveat for path-solver flux.
- **Tide model CATS2008** (native Ross Sea domain).
- **AOI:** strip-aware v15 rectangle `beardmore_shelf_stack_extent.shp`
  (14,470 km²) grown from the hand-drawn 446 km² `beardmore_shelf.shp`
  outline; no named MEaSUREs ice-shelf feature covers the sector (it sits on
  the Ross_West/Ross_East seam), so viz uses the AOI rectangle.
- **Per-basin ASP roots** (never shared): `data/ASP/` (IS2 era) ·
  `data/ASP_ctempoatm/` (pre-IS2 CryoTEMPO+ATM, incl. the 2010–2012 gap era)
  · `data/ASP_glas/` (ICESat-1) · `data/ASP_nocorr/` (Shean nocorr ingest at
  per-era class bias: glas −2.25 m / pre-IS2 −0.80 m / IS2 −2.34 m).
- **Tilt: the Shean-complete system** — `BEARDMORE_SHELF_TILT_DOMAIN=full` +
  `BEARDMORE_SHELF_TILT_DHDT_SMOOTH=1.0` (validated 2026-07-11: resolves the
  RED/LEFT artifact bands and lets cross-epoch self-consistency set nocorr
  datums; config default remains static-domain). `build_per_epoch_ez`
  resolves each epoch by `dem_id` to IS2/ATM 0.1 m · GLAS 0.3 m ·
  CryoTEMPO 1.0 m · nocorr 1.0 m. `BEARDMORE_SHELF_TILT_IRLS_MAX` caps IRLS
  (use 5 for full-record solves; convergence plateaus by iter 4).

## Era-specific align/ingest invocations

```bash
$PY -m beardmore_shelf.align_strips [--parallel N]        # IS2 era -> data/ASP/
$PY -m beardmore_shelf.align_strips --control cs2 --cs2-source cryotempo \
    --asp-suffix _ctempoatm --start 2010-02-07 --end 2019-01-01  # pre-IS2
$PY -m beardmore_shelf.align_strips --control glas --asp-suffix _glas \
    --start 2009-01-01 --end 2010-10-12                          # GLAS era
$PY -m beardmore_shelf.ingest_nocorr    # per-era --z-offset; see --help
$PY -m beardmore_shelf.scripts.qc_glas_align --root <ASP_x> [--apply]  # root QC
```

Era chains (GLAS align, gap align+ingest, full-record build→tilt→solve) live
in `scripts/*.sh`, disconnect-safe with `*_DONE`/`*_FAILED` sentinels.
Override align workers with `BEARDMORE_SHELF_ALIGN_PARALLEL=N`, advection
velocity with `BEARDMORE_SHELF_VELOCITY=`.

## Caveats

- **Sparse REMA at −84°S:** thinner per-epoch coverage than the West
  Antarctic basins — watch the static-control percentage `tilt_fit` prints.
  Rock control comes from `beardmore_shelf_rock_bedmachine.shp` (BedMachine
  mask==1 polygons; the shared hand-drawn `rock_polygons.shp` intersects this
  AOI nowhere).
- **Band diagnostics:** the RED (under-constraint) and LEFT (frame sign-flip)
  melt-map bands are tilt-system artifacts under the static domain — address
  with the Shean-complete tilt (general algorithm), never with manual
  BAD_STRIPS attribution. `diag_stripe_strips` is the melt-space screen; the
  budget linear-inverse twins (`compare_budget_inverse`) supply the
  frame-flip diagnostic that identified the bands.
- **`figures/tilt_fit_qc.png` and `stack_coverage.png` carry no window/tag
  suffix** in some stages — back up before re-runs that would overwrite.

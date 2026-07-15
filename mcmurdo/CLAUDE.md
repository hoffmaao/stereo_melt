# CLAUDE.md — mcmurdo/

The **McMurdo Ice Shelf** application of the `stereo_melt` library
(user-supplied geojson AOI, single MultiPolygon, ~2963 km²). Canonical stage
sequence: [`../PIPELINE.md`](../PIPELINE.md) — run `$PY -m mcmurdo.<stage>`;
only the deltas below are McMurdo-specific.

## Decision record

- **Window** (`config.py` authoritative): IS2-era `2019-01-01 → 2024-01-10`.
- **Grid 25 m.**
- **Geoid + MDT** both apply (−78°S, just north of the DTU22 limit).
- **Velocity: MEaSUREs phase map (NSIDC-0754)** — no ITS_LIVE tile on disk
  for this region.
- **Tide model CATS2008** (native Ross Sea domain — unambiguous).

## Caveats

- **Dense REMA coverage near McMurdo Station** → high strip count for a small
  AOI; slim ASP outputs run ~0.2 GB/strip, and the align driver enforces the
  100 GB disk floor.
- The AOI MultiPolygon is why `list_rema_strips` collapses MultiPolygons
  before `pdemtools.search` (the fix originated here).

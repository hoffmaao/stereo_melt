# CLAUDE.md — venable/

The **Venable Ice Shelf** application of the `stereo_melt` library
(Bellingshausen Sea sector, ~−73.1°S / −87.2°W, ~3155 km², Eltanin Bay coast
of Ellsworth Land). Scaffolded 2026-06-30; one of the two canonical templates
`scripts/new_basin.py` stamps from. Canonical stage sequence:
[`../PIPELINE.md`](../PIPELINE.md) — run `$PY -m venable.<stage>`; only the
deltas below are Venable-specific.

## Decision record

- **Window** (`config.py` authoritative): IS2-era `2019-01-01 → 2024-01-10`.
- **Grid 25 m** (125 m variants exist under the `--res 125` suffix).
- **Geoid + MDT** both apply (−73°S).
- **Velocity: MEaSUREs phase map (NSIDC-0754).** No ITS_LIVE tile on disk for
  the Bellingshausen sector and the ASE quarterly mosaics do not extend here;
  to add ITS_LIVE later, fetch a Bellingshausen tile and add `ITS_LIVE_*`
  config entries.
- **Tide model CATS2008** (covers the Bellingshausen/Amundsen sector).

## Caveats

- **AOI is a provisional bbox+30 km rectangle** (MEaSUREs `Venable` feature
  bounds), not the strip-aware v15 output — upgrade via
  `scripts/compute_aoi.py venable` when convenient.
- A vertical melt-map stripe traced to per-epoch thickness bias was resolved
  with the melt-space per-epoch bias LSQ (2026-07-04) — re-run
  that screen after any re-tilt rather than dropping strips.

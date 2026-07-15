# CLAUDE.md — dotson_crosson/

The **Dotson + Crosson Ice Shelves** application of the `stereo_melt` library
(Amundsen Sea Embayment). The two shelves are processed **jointly**: their
grounding zones are coupled and the union AOI (~9009 km², 30 km-buffered for
upstream grounded control) sits within single REMA strip footprints —
processing them as one stack avoids edge-effect mismatches at the union
boundary. Canonical stage sequence: [`../PIPELINE.md`](../PIPELINE.md) — run
`$PY -m dotson_crosson.<stage>`; only the deltas below are specific here.

## Decision record

- **Window** (`config.py` authoritative): IS2-era `2019-01-01 → 2024-01-10`.
- **Grid 25 m.**
- **Geoid + MDT** both apply (−75°S).
- **Velocity:** fast ASE ice — ITS_LIVE feature tracking works; RGI19A annual
  mosaics on disk cover the AOI. MEaSUREs 0754 is the time-mean fallback.
- **Tide model CATS2008.**

## Caveats

- **Adjacent to PIG** — per-basin ASP root discipline matters doubly here;
  sharing a root with PIG was the original cross-AOI strip-leak incident.

# CLAUDE.md — beardmore/

The **legacy Beardmore Glacier** application of the `stereo_melt` library:
the 2013–2023 mixed CS2+IS2-era study on the wider hand-extended AOI. For the
modern chain on this geography use **`beardmore_shelf/`** (per-basin era
roots, v15 AOI, current library defaults, full 2009→2024 record) — the two
share only the raw-strip pool under `data/REMA/strips/`. Canonical stage
sequence: [`../PIPELINE.md`](../PIPELINE.md) — run `$PY -m beardmore.<stage>`.

## Decision record

- **Window** (`config.py` authoritative): `2013-01-01 → 2023-03-01` (mixed
  control eras — this basin motivated the per-epoch `Ez` machinery).
- **Grid 25 m.**
- **Geoid-only** (~−84°S, poleward of DTU22's −79°S limit — no MDT).
- **Velocity: MEaSUREs phase map (NSIDC-0754)**; ITS_LIVE is empty over the
  slow grounding zone.
- **Tide model CATS2008** (native Ross Sea domain).

## Caveats — legacy status

- **Intentionally not ported** to the harmonized shared align driver: still
  the legacy shared-root layout (the one exception in the workspace). Don't
  extend it — new work on this sector goes in `beardmore_shelf/`.
- `BAD_EPOCHS` still carries legacy date-keyed entries (12) alongside
  `BAD_STRIPS`.
- **Sparse REMA + sparse rock at −84°S**: static-control fraction ran ~3%
  historically with loosened thresholds; watch the tilt_fit control printout.

# CLAUDE.md — nansen/

The **Nansen Ice Shelf** application of the `stereo_melt` library (Terra Nova
Bay, Victoria Land). Canonical stage sequence: [`../PIPELINE.md`](../PIPELINE.md)
— run `$PY -m nansen.<stage>`; only the deltas below are Nansen-specific.

## Decision record

- **Window** (`config.py` authoritative): IS2-era `2019-01-01 → 2023-03-01`.
- **Grid 25 m.**
- **Geoid + MDT** both apply (−75°S).
- **Velocity: MEaSUREs phase map (NSIDC-0754)** — ITS_LIVE feature tracking
  has gaps over the slow grounding zone here.
- **Tide model CATS2008.**
- `BAD_EPOCHS` still carries legacy date-keyed entries alongside
  `BAD_STRIPS` — migrate to dem_id-keying on the next tilt pass.

## Caveats

- **Tongue is narrow** — the tilt LSQ runs `min_width=10000` (10 km, vs Shean
  PIG's 40 km) so x/y slopes can engage on the narrow geometry.
- **`H_ref` collapses if grounded slivers leak through the floating mask**
  (BedMachine slivers near the grounding zone drag mean freeboard toward
  zero) — tighten the mask or supply `H_ref` explicitly.
- **Per-pixel OLS dh/dt inherits REMA strip stripes after tilt-fit**, and the
  ~×9.4 hydrostatic gain turns a +0.5 m/yr surface residual into a fake
  −4.7 m/yr accretion signal (vs Davison's reported melt) — the strip-datum
  banding failure mode referenced workspace-wide as `project_nansen_dhdt_bias`.

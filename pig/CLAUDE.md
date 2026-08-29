# CLAUDE.md — pig/

The **Pine Island Glacier** application of the `stereo_melt` library
(Amundsen Sea Embayment; the Shean 2019 reference basin, so this is also the
parity/benchmark study). Canonical stage sequence: [`../PIPELINE.md`](../PIPELINE.md)
— run `$PY -m pig.<stage>`; only the deltas below are PIG-specific.

## Decision record

- **Window** (`config.py` is authoritative): full record `2010-01-01 →
  2024-01-10`; production melt work to date is on IS2-era (2019+) stacks —
  the full-vs-IS2-era A/B showed pre-IS2 (old alignment vintage) widened IQRs
  2–3.5× with stable medians.
- **Grid 25 m** (the 125 m pilot was evaluated and its "Shean-range" flux
  attributed to gate-drop + clip effects — 25 m stands).
- **Geoid + MDT** both apply (−75°S, north of the DTU22 limit).
- **Velocity:** `PIG_VELOCITY` env — `fused` (Kalman EOF+GP fusion with
  King-2020-style smoothing) is the production choice; `ase-quarterly`
  (37-quarter v05 COGs) and `measures` available. Unset falls back to
  `measures` with a `RuntimeWarning` (`run_melt.load_velocity_on_grid`), so
  pin the source in every driver (`os.environ.setdefault`); melt products
  record it in their `velocity` attr. Fast ice → ITS_LIVE/feature
  tracking works here, unlike the slow Ross-sector basins. Time-varying
  velocity is first-order for flux (time-mean inflated path flux ~40%).
- **Tide model CATS2008**; open-ocean adjacency → larger tide/IBE amplitudes
  than the Ross-sector basins.
- **Control:** IS2 ATL06 + CryoTEMPO + BedMachine-rock GCPs; `BAD_STRIPS`
  populated (dem_id-keyed — PIG led the 2026-06 migration off date-keyed
  `BAD_EPOCHS`).
- **Tilt:** production is the static-control domain; the Shean-complete
  full-domain + dh/dt-smoothness system (validated on beardmore_shelf
  2026-07-11) migrates here only after PIG-specific gates.

## Caveats

- **Calving 2017–2020** removed much of the historical ice tongue; the AOI
  reflects post-2020 geometry, and calving windows split melt-run epochs
  (berg-contaminated epochs are handled by windowed runs, not dropped strips).
  Since 2026-07-27 the window-minimum extent mask
  (`pig.build_min_extent_mask`, PIPELINE.md stage 6b) removes the calved
  sector (6149 → 4946 km²) from every `run_melt` product by default
  (`PIG_MIN_EXTENT=0` opts out); the eta-inversion exporter applies the same
  constraint live over its own window.
- Production finals carry the `_tv_path_dfix` lineage: time-varying velocity
  + trend-H + endpoint-exact firn Δd. Treat any new estimator as opt-in
  diagnostic alongside the path solver, not a replacement.
- Shean 2019 reproduction assets live in `data/Shean2019/` (82–93 Gt/yr
  reference; residual gap vs our chain attributed to coregistration noise).

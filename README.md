# stereo_melt workspace

Monorepo for DEM-based basal melt-rate work in Antarctica (Shean et al.
2019). The architecture is **one library, applied many times**: a single
basin-agnostic library plus one thin driver directory per ice-shelf
application. The canonical stage sequence lives in
[`PIPELINE.md`](./PIPELINE.md); per-shelf decisions live in each driver's
local decision record (untracked).

## Layout

```
stereo_melt/                ← workspace root (pip-installable from here)
├── src/stereo_melt/        ← THE library (no study-specific code)
│   ├── coregister/         ← ASP align driver, control caches (IS2/CS2/ATM/GLAS),
│   │                         per-epoch Ez, tilt LSQ + QC, per-strip control
│   │                         residual planes, Nuth & Kääb shift (diagnostic)
│   ├── corrections/        ← geoid, tides, IBE, MDT, post-coreg pipeline
│   ├── dynamics/           ← budget linear inverses, pseudospectral solvers,
│   │                         velocity fusion
│   ├── io/                 ← REMA, BedMachine, velocity, RACMO, masks
│   ├── validation/         ← CS2 vs IS2 cross-checks
│   ├── melt.py             ← production solvers: eulerian_melt_rate +
│   │                         lagrangian_melt_rate (path solver)
│   ├── stack.py            ← common grid, save/load, BAD_STRIPS filter
│   ├── spatialstats.py     ← variograms, n_eff, correlated-error propagation
│   └── flux.py, freeboard.py, kinematics.py, melt_qc.py, spectra.py, …
├── tests/                  ← synthetic gates (gate_*.py) + sanity checks
├── examples/<shelf>/       ← one driver per ice-shelf application
│                             (pig/, beardmore_shelf/, venable/, beardmore/,
│                              nansen/, dotson_crosson/, mcmurdo/, elmer_synth/, …
│                              — `ls examples`; scripts/new_basin.py scaffolds)
├── data/                   ← shared pan-Antarctic raster cache (gitignored)
├── scripts/                ← workspace utilities (new_basin.py, compute_aoi.py)
│   └── library/            ← library-level sanity scripts
├── PIPELINE.md             ← canonical algorithm sequence + how to add a basin
├── README_library.md       ← the packaged library README (pyproject `readme`)
└── README.md               ← this file
```

## Architectural rule

`src/stereo_melt/` is the **library** — algorithms with no awareness of
pipeline stages, study windows, or basin AOIs. The directories under
`examples/` are **applications** that wire the library functions into a
specific study. Imports flow basin → library, never the other way, and never
basin → basin. See
[`PIPELINE.md`](./PIPELINE.md#architectural-rule).

## Installing

The study conda env already exists at
`/home/hoffmaao/miniconda3/envs/stereo_melt` (Python 3.12, `numpy<2.0`
pinned — pyshtools/astropy bumps break pdemtools/numba). Install the library
into it editable:

```bash
/home/hoffmaao/miniconda3/envs/stereo_melt/bin/pip install -e .
```

## Running a pipeline

The staged sequence (climate/strip/control caches → per-era ASP align →
nocorr ingest → stack → Shean corrections + tilt LSQ → `BAD_STRIPS` QC →
solvers) is documented stage-by-stage in [`PIPELINE.md`](./PIPELINE.md).
Run stages as modules from `examples/`, which is where the basin driver
packages are importable from:

```bash
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
cd examples && $PY -m <basin>.<stage>   # e.g. cd examples && $PY -m pig.build_stack
```

Production melt products are `run_melt_path` (Lagrangian path solver) and
`run_melt` (Eulerian). Melt-rate sign: negative = melt.

## References

- Shean et al. 2019, *The Cryosphere* — [DOI](https://doi.org/10.5194/tc-13-2633-2019) · [reference code](https://github.com/dshean/pig_dem_meltrate)
- Chartrand 2024, Thwaites — [Zenodo 13667120](https://doi.org/10.5281/zenodo.13667120)
- Stubblefield 2023, linear non-hydrostatic inverse — referenced in `src/stereo_melt/dynamics/`
- Chudley & Howat 2024, *pdemtools* — [repo](https://github.com/trchudley/pdemtools)

See also [icepack](https://icepack.github.io), whose documentation and code
style this project follows.

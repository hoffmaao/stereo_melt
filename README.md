# stereo_melt workspace

Monorepo for DEM-based basal melt-rate work in Antarctica (Shean et al.
2019). The architecture is **one library, applied many times**: a single
basin-agnostic library plus one thin driver directory per ice-shelf
application. The canonical stage sequence lives in
[`PIPELINE.md`](./PIPELINE.md); per-shelf decisions live in each driver's
`CLAUDE.md`.

## Layout

```
stereo_melt/                ← workspace root
├── stereo_melt/            ← THE library (pip-installable, no study-specific code)
│   ├── src/stereo_melt/
│   │   ├── coregister/     ← ASP align driver, control caches (IS2/CS2/ATM/GLAS),
│   │   │                     per-epoch Ez, tilt LSQ + QC
│   │   ├── corrections/    ← geoid, tides, IBE, MDT, post-coreg pipeline
│   │   ├── dynamics/       ← budget linear inverses, pseudospectral solvers,
│   │   │                     velocity fusion
│   │   ├── io/             ← REMA, BedMachine, velocity, RACMO, masks
│   │   ├── validation/     ← CS2 vs IS2 cross-checks
│   │   ├── melt.py         ← production solvers: eulerian_melt_rate +
│   │   │                     lagrangian_melt_rate (path solver)
│   │   ├── stack.py        ← common grid, save/load, BAD_STRIPS filter
│   │   └── flux.py, freeboard.py, kinematics.py, melt_qc.py, …
│   └── tests/              ← synthetic gates (gate_*.py) + sanity checks
├── <shelf>/                ← one driver per ice-shelf application
│                             (pig/, beardmore_shelf/, venable/, beardmore/,
│                              nansen/, dotson_crosson/, mcmurdo/, … —
│                              `ls` the root; scripts/new_basin.py scaffolds)
├── data/                   ← shared pan-Antarctic raster cache (gitignored)
├── scripts/                ← workspace utilities (new_basin.py, compute_aoi.py)
├── PIPELINE.md             ← canonical algorithm sequence + how to add a basin
└── README.md               ← this file
```

## Architectural rule

`stereo_melt/` is the **library** — algorithms with no awareness of pipeline
stages, study windows, or basin AOIs. Basin directories are **applications**
that wire the library functions into a specific study. Imports flow
basin → library, never the other way, and never basin → basin. See
[`PIPELINE.md`](./PIPELINE.md#architectural-rule).

## Installing

The study conda env already exists at
`/home/hoffmaao/miniconda3/envs/stereo_melt` (Python 3.12, `numpy<2.0`
pinned — pyshtools/astropy bumps break pdemtools/numba). Install the library
into it editable:

```bash
/home/hoffmaao/miniconda3/envs/stereo_melt/bin/pip install -e stereo_melt/
```

## Running a pipeline

The staged sequence (climate/strip/control caches → per-era ASP align →
nocorr ingest → stack → Shean corrections + tilt LSQ → `BAD_STRIPS` QC →
solvers) is documented stage-by-stage in [`PIPELINE.md`](./PIPELINE.md).
Run stages as modules from the workspace root:

```bash
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
$PY -m <basin>.<stage>        # e.g. $PY -m pig.build_stack
```

Production melt products are `run_melt_path` (Lagrangian path solver) and
`run_melt` (Eulerian). Melt-rate sign: negative = melt.

## References

- Shean et al. 2019, *The Cryosphere* — [DOI](https://doi.org/10.5194/tc-13-2633-2019) · [reference code](https://github.com/dshean/pig_dem_meltrate)
- Chartrand 2024, Thwaites — [Zenodo 13667120](https://doi.org/10.5281/zenodo.13667120)
- Stubblefield 2023, linear non-hydrostatic inverse — referenced in `stereo_melt/dynamics/`
- Chudley & Howat 2024, *pdemtools* — [repo](https://github.com/trchudley/pdemtools)

See also [icepack](https://icepack.github.io), whose documentation and code
style this project follows.

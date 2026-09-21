# stereo_melt

stereo_melt computes basal melt rates beneath Antarctic ice shelves from repeat stereo DEMs, surface velocity and altimetry.

## Installation

Install editable into the study conda env (built from the `pyproject`
dependencies; `numpy<2.0` is pinned — pyshtools/astropy bumps break
pdemtools/numba):

```bash
/home/hoffmaao/miniconda3/envs/stereo_melt/bin/pip install -e .
```

The `gpu` extra installs `cupy-cuda11x` for array-heavy stages; set `STEREO_MELT_BACKEND=cupy` to enable it (strict — no silent numpy fallback). [Ames Stereo Pipeline](https://stereopipeline.readthedocs.io/) must be installed separately; `pc_align` is the primary coregistration path.

## Getting started

The public entry points for the mass-budget stage are
`stereo_melt.melt.eulerian_melt_rate` and `stereo_melt.melt.lagrangian_melt_rate`
(the production path solver). `tests/` holds synthetic gates (`gate_*.py`)
and end-to-end sanity checks (`sanity_eulerian_melt.py`,
`sanity_lagrangian_melt.py`).

The library carries **no study configuration** — shelf, window, grid, and
data paths live in each basin driver package under `examples/` (see
[`PIPELINE.md`](./PIPELINE.md)).

## Data

Several upstream datasets are required. Some need credentials:

- **REMA** strips and mosaics — public, fetched via [`pdemtools`](https://github.com/trchudley/pdemtools).
- **ICESat-2 ATL06** — account with [NASA EarthData](https://urs.earthdata.nasa.gov/) (used by `sliderule` and `pdemtools` coregistration).
- **ERA5** mean-sea-level pressure (for IBE; MSL, not surface pressure) — account with the [Copernicus CDS](https://cds.climate.copernicus.eu/) and a `~/.cdsapirc` token.
- **RACMO2.4p1** SMB (11 km, Zenodo [19255213](https://zenodo.org/records/19255213)) — public.
- **MEaSUREs** (NSIDC-0754) and **ITS_LIVE** velocity — public.
- **BedMachine Antarctica** — [NSIDC account](https://nsidc.org/data/nsidc-0756).
- **CATS2008** tide model — public.

## References

- Shean et al. 2019, *The Cryosphere* — [DOI](https://doi.org/10.5194/tc-13-2633-2019) · [code](https://github.com/dshean/pig_dem_meltrate)
- Chartrand 2024, Thwaites — [Zenodo 13667120](https://doi.org/10.5281/zenodo.13667120)
- Chudley & Howat 2024, *pdemtools* — [repo](https://github.com/trchudley/pdemtools)

See also [icepack](https://icepack.github.io), a finite-element ice-flow modelling library whose documentation and code style this project follows.

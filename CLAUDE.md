# CLAUDE.md — stereo_melt workspace

Guidance for Claude Code at the **workspace root** (`/wd2/projects/stereo_melt`).
For the directory layout and the end-to-end algorithm sequence, read
[`README.md`](./README.md) and [`PIPELINE.md`](./PIPELINE.md) — this file is the
operational rules that span all basins, not a re-statement of those.

## What this is

A monorepo for DEM-based Antarctic basal melt-rate work (Shean et al. 2019).
One **library** + seven **basin drivers** (one per ice-shelf application):

- `stereo_melt/` — pip-installed library: composable algorithms only, no
  study-specific code.
- `pig/` (2010–2024 window; the Shean reference basin) · `venable/` ·
  `dotson_crosson/` · `mcmurdo/` (2019–2024, IS2 era) · `nansen/`
  (2019–2023, IS2 era) · `beardmore/` (2013–2023 mixed-era, legacy layout) ·
  `beardmore_shelf/` (2009–2024 full record, all control eras) — thin
  drivers: paths, AOI, window, glue. New ones via `scripts/new_basin.py`.
- `elmer_synth/` — **not an ice shelf**: the independent-physics synthetic
  melt-truth driver (full-Stokes twin experiments E1/E1b done, Elmer/Ice E2
  planned) that validates the melt inverses against truth the solvers'
  operators did not generate. See `elmer_synth/CLAUDE.md` and
  `literature/plan_elmer_synthetic_validation.md`.

**Import discipline:** basin → library, never the reverse. Don't branch on basin
name inside the library; add a parameter instead. Each basin directory has its
own `CLAUDE.md` — read it when working in that basin.

## Environment — read this first

The package lives in a conda env *outside* the repo, and the harness sandbox
**cannot `conda activate`** (it blocks `source`). Always call the interpreter by
absolute path, and run modules from the workspace root:

```bash
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
$PY -m <basin>.<stage>                       # e.g. pig.build_stack
$PY -m pip install -e stereo_melt/           # after editing pyproject / adding modules
$PY -m ruff check stereo_melt/src/           # lint (ruff installed in the env)
```

`STEREO_MELT_BACKEND=cupy` runs array-heavy stages on GPU; default is numpy.

## Cross-cutting conventions (every basin)

- **Melt-rate sign:** negative = melt, positive = accretion, across all
  solvers. Davison-compare helpers negate internally.
- **CRS:** everything is EPSG:3031 (Antarctic Polar Stereographic); x ascends,
  y descends (north-up). CRS is written on the stack at build time.
- **GPU is strict:** `STEREO_MELT_BACKEND=cupy` does **not** silently fall back
  to numpy — debug GPU failures, don't mask them (soft fallback is opt-in only).
- **Wider-AOI `tilt_fit` runs on CPU:** `STEREO_MELT_BACKEND=numpy
  OMP_NUM_THREADS=14`, no `CUDA_VISIBLE_DEVICES` — the 11 GB GPU can't hold the
  wider LSQ.
- **Long jobs: `nohup … python -u … &`.** SSH drops kill backgrounded python
  without `nohup`; `-u` keeps progress flowing to the log. Heavy LSQ/IRLS stages
  are instrumented to narrate every phase.
- **Per-basin ASP root:** each basin owns `<basin>/data/ASP*/`. Never share an
  ASP root across basins — it leaks strips between AOIs.
- **`--res` / `--tag` suffix:** resolution/experiment variants live at
  `<basin>_stack_<N>m_<tag>_…nc`; the suffix sits between `<basin>_stack` and
  `tilt_corrected`, or `load_basin_stack` silently falls back to the raw stack.
- **DEM rejection is per-DEM (`BAD_STRIPS`, keyed on the stack's `dem_id`
  coord), not per-date.** All seven drivers carry `BAD_STRIPS`; date-keyed
  `BAD_EPOCHS` is legacy and empty everywhere except `beardmore/` and
  `nansen/`, which still hold pre-migration date lists (migrate them on
  their next tilt pass).

## Consolidation plan (staged, 2026-08-22)

[`literature/plan_consolidation.md`](./literature/plan_consolidation.md) is
the staged plan to collapse the layered solvers and photocopied drivers into
one library + thin drivers (stage 0 = track `elmer_synth/` and tag; stage 1 =
one implementation per concept in `dynamics/`; 2 = solver registry; 3 =
config split + shared heavies; 4 = driver pruning; 5 = `elmer_synth/`
hygiene; 6 = docs). Each stage has a gate; decisions A–E are listed there.

## "Where we left off" — resume state

Live session state is in the **memory system**, not in any `CLAUDE.md`.
`CLAUDE.md` / `README.md` / `PIPELINE.md` are *static* guidance and do not track
running jobs. The auto-loaded index is `MEMORY.md` under
`~/.claude/projects/-wd2-projects-stereo-melt/memory/`; the most recent
`project_session_resume_<date>.md` there is the latest snapshot (running jobs,
PIDs, log paths, next steps, standing directives). Start there to resume.

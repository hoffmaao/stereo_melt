# Pine Island Glacier basal melt-rate study

Driver scripts for the **Pine Island Glacier** (Amundsen Sea Embayment,
~−75°S / −101°E) application of the [`stereo_melt`](../../src/stereo_melt/)
library — the Shean 2019 reference basin, used for parity/benchmark work.

The full production chain runs here (per-era ASP align → stack → Shean
corrections + tilt LSQ → path + Eulerian solvers, time-varying fused
velocity, budget linear-inverse diagnostics). The canonical stage sequence is
[`PIPELINE.md`](../../PIPELINE.md); PIG-specific decisions and caveats are
in the local decision record (untracked); `config.py` is the single source of study
context (window, AOI, grid, paths).

Long stages run detached, e.g.:

```bash
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
nohup $PY -u -m pig.align_strips >> pig/logs/align_strips.log 2>&1 &
```

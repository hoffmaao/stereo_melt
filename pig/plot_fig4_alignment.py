"""PIG Shean 2019 Fig 4 alignment diagnostics (thin, uniform driver).

Uniform fig04 over ``stereo_melt.visualization.plot_alignment_diagnostics``:
ICP translation scatters + pre/post pc_align residual timeseries, walking the
two production STRIP_SOURCES roots (pre-IS2 + IS2-era CryoTEMPO control).

For the specialised dense-tie re-align comparison (fixed-control resample so
the two-stage densetie residuals are apples-to-apples with production), see the
separate ``pig.plot_fig4_alignment_diagnostics``.

Run:
    /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python \
        -m pig.plot_fig4_alignment
"""
from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ.setdefault("PROJ_DATA", _env_proj)
    os.environ.setdefault("PROJ_LIB", _env_proj)

import matplotlib

matplotlib.use("Agg")

from stereo_melt.visualization import plot_alignment_diagnostics

from pig import config

VARIANTS = [
    (config.ASP_IS2CTEMPOATMLVIS_ROOT / "asp_aligned",
     "IS2-era (IS2+CryoTEMPO+ATM+LVIS)", "tab:blue"),
    (config.ASP_CTEMPOATMLVIS_ROOT / "asp_aligned",
     "pre-IS2 (CryoTEMPO+ATM+LVIS)", "tab:orange"),
]


def main() -> None:
    plot_alignment_diagnostics(
        VARIANTS,
        config.PIG_AOI_SHP,
        config.FIGURES_DIR / "fig4_alignment.png",
        f"PIG pc_align ICP translation + pre/post residual "
        f"({config.START_TIME} → {config.END_TIME}, + pre-IS2)",
    )


if __name__ == "__main__":
    main()

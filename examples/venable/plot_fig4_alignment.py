"""Venable Shean 2019 Fig 4 alignment diagnostics (thin driver).

Uniform fig04 over ``stereo_melt.visualization.plot_alignment_diagnostics``:
ICP translation scatters + pre/post pc_align residual timeseries.

Run:
    /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python \
        -m venable.plot_fig4_alignment
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

from venable import config

VARIANTS = [
    (config.STRIP_ALIGNED_DIR, "IS2 + rock control", "tab:blue"),
]


def main() -> None:
    plot_alignment_diagnostics(
        VARIANTS,
        config.VENABLE_AOI_SHP,
        config.FIGURES_DIR / "fig4_alignment.png",
        "Venable pc_align ICP translation + pre/post residual",
    )


if __name__ == "__main__":
    main()

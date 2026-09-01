"""Beardmore (2013-2023 mixed-era basin) Shean 2019 Fig 4 alignment diagnostics.

Uniform fig04 over ``stereo_melt.visualization.plot_alignment_diagnostics``:
ICP translation scatters + pre/post pc_align residual timeseries, walking the
``config.STRIP_SOURCES`` roots (IS2+CS2 and CS2-only eras).

Run:
    /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python \
        -m beardmore.plot_fig4_alignment
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

from beardmore import config

_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
VARIANTS = [
    (d, lbl, _COLORS[i % len(_COLORS)])
    for i, (d, lbl) in enumerate(config.STRIP_SOURCES)
]


def main() -> None:
    plot_alignment_diagnostics(
        VARIANTS,
        config.BEARDMORE_AOI_SHP,
        config.FIGURES_DIR / "fig4_alignment.png",
        "Beardmore pc_align ICP translation + pre/post residual "
        "(IS2+CS2 and CS2 eras)",
    )


if __name__ == "__main__":
    main()

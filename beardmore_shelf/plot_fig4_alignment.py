"""Beardmore_Shelf Shean 2019 Fig 4 alignment diagnostics (thin driver).

Uniform fig04 over ``stereo_melt.visualization.plot_alignment_diagnostics``:
ICP translation scatters + pre/post pc_align residual timeseries. Walks both
per-basin ASP roots so the combined 2012-2024 alignment quality is visible
(the pre-IS2 CryoTEMPO residuals inform whether those epochs can condition the
tilt fit rather than being dropped).

Run:
    /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python \
        -m beardmore_shelf.plot_fig4_alignment
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

from beardmore_shelf import config

VARIANTS = [
    (config.STRIP_ALIGNED_DIR, "IS2-era (IS2-only control)", "tab:blue"),
    (config.BASIN_DIR / "data" / "ASP_ctempoatm" / "asp_aligned",
     "pre-IS2 (CryoTEMPO+ATM+rock control)", "tab:orange"),
]


def main() -> None:
    plot_alignment_diagnostics(
        VARIANTS,
        config.BEARDMORE_SHELF_AOI_SHP,
        config.FIGURES_DIR / "fig4_alignment.png",
        "Beardmore_Shelf pc_align ICP translation + pre/post residual "
        "(IS2-era + pre-IS2 CryoTEMPO)",
    )


if __name__ == "__main__":
    main()

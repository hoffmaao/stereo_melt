"""Beardmore processing-pipeline explainer figure.

A single block-diagram figure showing every stage between raw REMA strips
and the recovered basal melt rate. Pipeline stages run down the centre
column; input data products feed in from the left and are colour-coded
by category (altimetry, geophysical control, ocean/atmosphere, model
forcing). The figure is the visual companion to
`literature/closed_form_methods.md`.

Output: literature/figures/processing_pipeline.png

Run:

    /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python \
        -m beardmore.explainer_figures.pipeline_diagram
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# Layout: centre column is the pipeline spine (x = 0.55), left column holds
# input-data boxes that feed into the relevant stage (x = 0.18). All
# coordinates are in axis fraction (0..1).

STAGE_W = 0.34
STAGE_H = 0.075
STAGE_X = 0.50
INPUT_W = 0.27
INPUT_H = 0.052
INPUT_X = 0.10

# Pipeline stages, top to bottom. y is the centre of each box.
STAGES = [
    ("fetch_strips",          "REMA strip discovery\n(pdemtools.search)",                                        0.945),
    ("asp",                   "ASP pc_align\n6-DOF coregistration vs IS2 + rock GCPs",                            0.835),
    ("build_stack",           "Reproject to common 25 m grid\n(EPSG:3031, target_grid)",                          0.725),
    ("tide",                  "Tide correction\nCATS2008, RBF-filled spatial field,\n3 km feathered floating mask", 0.610),
    ("ibe",                   "Inverse barometric effect\nERA5 surface pressure → scalar h_IBE per strip",        0.490),
    ("tilt",                  "Per-epoch LSQ tilt fit\nShean ndinterp + IRLS Tukey biweight,\nstatic-control mask", 0.370),
    ("geoid",                 "Geoid + MDT\nEGM2008; MDT skipped <-79°S (DTU22)",                                  0.255),
    ("solvers",               "Solvers\nEulerian / Lagrangian / Closed-FFT / Closed-DCT / PS-CG",                  0.165),
    ("output",                "Basal melt rate b_dot(x, y)\nm ice / yr",                                           0.075),
]

# Input data products: (key, label, target_stage_key, category)
# category drives box colour.
INPUTS = [
    ("rema",       "REMA strips\n(2 m stereo DEMs)",                  "fetch_strips", "altimetry"),
    ("is2_gcp",    "ICESat-2 ATL06 +\nrock-outcrop GCPs",             "asp",          "altimetry"),
    ("cs2",        "CryoSat-2 SARIn L2\nPOCA",                        "asp",          "altimetry"),
    ("bedmachine", "BedMachine v3\nmask, firn, geoid",                "build_stack",  "geophysical"),
    ("cats",       "CATS2008 tide model\n(via pyTMD)",                "tide",         "ocean_atm"),
    ("era5",       "ERA5 hourly\nsurface pressure",                   "ibe",          "ocean_atm"),
    ("egm",        "EGM2008 geoid\n(via pdemtools)",                  "geoid",        "geophysical"),
    ("velocity",   "MEaSUREs NSIDC-0754\nphase-map velocity",         "solvers",      "model"),
    ("racmo",      "RACMO2.4p1 SMB\n(monthly smbgl)",                 "solvers",      "model"),
]

# Annotations attached to specific stages (right side).
ANNOTATIONS = [
    ("asp",         "primary coregistration\n(no pre-coreg tide/IBE —\ncontrol points are tide-free)"),
    ("tide",        "post-coreg @ 25 m\n(Shean 2019 stack_tidecorr.py)"),
    ("tilt",        "removes residual ASP tilt;\nworks where static control\noverlaps the floating tile"),
    ("solvers",     "Lagrangian-frame remap\nabsorbs α(x,y) into\nthe coordinate change"),
]

CATEGORY_FACE = {
    "altimetry":   "#cfe6f5",   # light blue
    "geophysical": "#dde9c9",   # light olive
    "ocean_atm":   "#f3d6c4",   # light terracotta
    "model":       "#e6d6ec",   # light lavender
}
CATEGORY_EDGE = {
    "altimetry":   "#3a78a4",
    "geophysical": "#5a7a3a",
    "ocean_atm":   "#a35a30",
    "model":       "#73478a",
}
STAGE_FACE = "#f7f3df"
STAGE_EDGE = "#403a1f"
OUTPUT_FACE = "#f8c8a0"
OUTPUT_EDGE = "#9c3f0a"


def _add_box(ax, x, y, w, h, text, *, face, edge, fontsize=8.4, weight="normal"):
    bb = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        linewidth=1.2, facecolor=face, edgecolor=edge,
        transform=ax.transAxes,
    )
    ax.add_patch(bb)
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=fontsize, weight=weight,
        color="#222",
    )


def _add_arrow(ax, x0, y0, x1, y1, *, color="#403a1f", style="->", lw=1.4, ls="-"):
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        transform=ax.transAxes,
        arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color, linestyle=ls,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arr)


def main() -> None:
    fig = plt.figure(figsize=(11.5, 14.0), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- Stage spine ----
    stage_y = {key: y for key, _, y in STAGES}
    for key, label, y in STAGES:
        face = OUTPUT_FACE if key == "output" else STAGE_FACE
        edge = OUTPUT_EDGE if key == "output" else STAGE_EDGE
        weight = "bold" if key == "output" else "normal"
        _add_box(ax, STAGE_X, y, STAGE_W, STAGE_H,
                 label, face=face, edge=edge, fontsize=9.2, weight=weight)

    # vertical arrows between consecutive stages
    for i in range(len(STAGES) - 1):
        y_top = STAGES[i][2] - STAGE_H / 2
        y_bot = STAGES[i + 1][2] + STAGE_H / 2
        _add_arrow(ax, STAGE_X, y_top, STAGE_X, y_bot, lw=1.6)

    # ---- Inputs (left) ----
    # Stack inputs vertically near their target stage so the arrow stays
    # short. When two inputs share a target we offset them.
    used_y: dict[str, list[float]] = {}
    for key, label, target, cat in INPUTS:
        target_y = stage_y[target]
        # offset so multiple inputs around the same stage don't overlap
        existing = used_y.setdefault(target, [])
        offset = 0.0
        if existing:
            offset = (-1) ** len(existing) * 0.06 * ((len(existing) + 1) // 2)
        y = target_y + offset
        used_y[target].append(y)

        _add_box(ax, INPUT_X, y, INPUT_W, INPUT_H,
                 label,
                 face=CATEGORY_FACE[cat], edge=CATEGORY_EDGE[cat],
                 fontsize=7.8)

        # arrow from input -> stage edge (left edge of stage box)
        x0 = INPUT_X + INPUT_W / 2
        x1 = STAGE_X - STAGE_W / 2
        _add_arrow(ax, x0, y, x1, target_y,
                   color=CATEGORY_EDGE[cat], lw=1.1)

    # ---- Right-side annotations ----
    for target, text in ANNOTATIONS:
        y = stage_y[target]
        x_anno = STAGE_X + STAGE_W / 2 + 0.02
        ax.text(
            x_anno, y, text,
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=7.5, style="italic", color="#444",
        )

    # ---- Title + legend ----
    ax.text(
        0.5, 0.985, "Beardmore basal-melt-rate processing pipeline",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=13, weight="bold",
    )
    ax.text(
        0.5, 0.967,
        "raw REMA stereo DEMs $\\to$ corrected stack $\\to$ b_dot(x,y)",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=9.5, style="italic", color="#444",
    )

    # legend strip across the bottom, below the output box
    legend_items = [
        ("altimetry input",   CATEGORY_FACE["altimetry"],   CATEGORY_EDGE["altimetry"]),
        ("geophysical input", CATEGORY_FACE["geophysical"], CATEGORY_EDGE["geophysical"]),
        ("ocean / atm input", CATEGORY_FACE["ocean_atm"],   CATEGORY_EDGE["ocean_atm"]),
        ("model forcing",     CATEGORY_FACE["model"],       CATEGORY_EDGE["model"]),
        ("processing stage",  STAGE_FACE,                   STAGE_EDGE),
        ("output",            OUTPUT_FACE,                  OUTPUT_EDGE),
    ]
    leg_y_top = 0.026
    leg_y_bot = 0.008
    leg_cols_x = [0.06, 0.30, 0.56]
    for i, (label, face, edge) in enumerate(legend_items):
        col = i % 3
        row = i // 3
        x = leg_cols_x[col]
        y = leg_y_top if row == 0 else leg_y_bot
        _add_box(ax, x, y, 0.022, 0.012, "",
                 face=face, edge=edge, fontsize=6.0)
        ax.text(x + 0.018, y, label,
                transform=ax.transAxes, ha="left", va="center",
                fontsize=8.0, color="#222")

    out_path = Path("/wd2/projects/stereo_melt/literature/figures/processing_pipeline.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

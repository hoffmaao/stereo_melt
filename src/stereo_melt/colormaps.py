"""LADDIE-informed basal melt/freeze colormap and symmetric-log norm.

A diverging, black-at-zero, log-scaled palette for basal melt-rate maps,
following the colorbar the LADDIE ice-shelf/ocean model (Lambert et al.)
uses for its melt fields: warm (black -> purple -> red -> yellow) for
melt, cool (black -> blue -> white) for freezing/accretion, joined at
pure black at zero, with a small *linear* window around zero and
logarithmic decades outward (ticks +/- 0.1, 0.3, 1, 3, 10 m ice/yr).

The warm half is matplotlib ``inferno``; the cool half is an
``cmocean.ice``-like ramp built here (cmocean is not a dependency).

Sign convention.  This workspace is Shean throughout: **negative = melt,
positive = accretion** (see ``feedback_shean_convention``).  So the
default orientation (``melt_cmap()`` / ``laddie_cmap("shean")``) is the
LADDIE palette *reversed along the value axis*: melt (negative) reads
warm, accretion (positive) reads cool, zero is black.  Pass
``orient="laddie"`` for the native positive-is-melt orientation.

Typical use on a Shean-sign melt field ``v`` (m ice/yr)::

    from stereo_melt.colormaps import melt_cmap, melt_norm, add_melt_colorbar
    im = ax.imshow(v, cmap=melt_cmap(), norm=melt_norm(vmax=10),
                   interpolation="nearest")
    add_melt_colorbar(fig, im, ax=ax, shrink=0.8)
"""
from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, SymLogNorm

# cmocean.ice-like cool ramp: black (zero) -> navy -> blue -> cyan -> white.
_ICE_ANCHORS = [
    (0.00, (0.000, 0.000, 0.000)),
    (0.15, (0.055, 0.075, 0.280)),
    (0.35, (0.090, 0.220, 0.550)),
    (0.55, (0.130, 0.430, 0.720)),
    (0.75, (0.350, 0.680, 0.820)),
    (0.90, (0.720, 0.905, 0.930)),
    (1.00, (1.000, 1.000, 1.000)),
]

# Log-scale tick anchors LADDIE labels (m ice/yr), symmetric about zero.
# LADDIE stops at +/-10; the 1-3 pattern is continued to +/-100 so the fast
# shelves (PIG runs to ~100 m/yr near the grounding line) get their top decades
# labelled instead of an unbroken ramp into the extend arrow.
_MELT_TICKS = np.array([-100, -30, -10, -3, -1, -0.3, -0.1, 0,
                        0.1, 0.3, 1, 3, 10, 30, 100], float)

#: NaN / masked cells render as neutral gray (LADDIE land/ocean tone), so
#: they are never confused with the white strong-accretion extreme.
BAD_COLOR = "0.8"


def _ice_like(n: int) -> np.ndarray:
    cmap = LinearSegmentedColormap.from_list("_ice_like", _ICE_ANCHORS)
    return cmap(np.linspace(0.0, 1.0, n))[:, :3]


def _inferno(n: int) -> np.ndarray:
    return mpl.colormaps["inferno"](np.linspace(0.0, 1.0, n))[:, :3]


def _laddie_stack(n: int) -> np.ndarray:
    """Native LADDIE order: index 0 = strong freeze (white), center = black
    (zero), last index = strong melt (yellow)."""
    half = n // 2
    freeze = _ice_like(half)[::-1]          # white -> black
    melt = _inferno(n - half)               # black -> yellow
    return np.vstack([freeze, melt])


def laddie_cmap(orient: str = "shean", n: int = 256) -> ListedColormap:
    """LADDIE melt/freeze colormap.

    ``orient="shean"`` (default): negative = melt reads warm (repo
    convention).  ``orient="laddie"``: native positive = melt reads warm.
    """
    cols = _laddie_stack(n)
    if orient == "shean":
        cols = cols[::-1]
    elif orient != "laddie":
        raise ValueError(f"orient must be 'shean' or 'laddie', got {orient!r}")
    cmap = ListedColormap(cols, name=f"laddie_{orient}")
    cmap.set_bad(BAD_COLOR)
    cmap.set_under(tuple(cols[0]))
    cmap.set_over(tuple(cols[-1]))
    return cmap


def melt_cmap(n: int = 256) -> ListedColormap:
    """Shean-oriented LADDIE colormap (negative = melt -> warm)."""
    return laddie_cmap("shean", n)


def melt_norm(vmax: float = 10.0, linthresh: float = 0.1,
              vmin: float | None = None, linscale: float = 1.0) -> SymLogNorm:
    """Symmetric-log norm: linear within +/-``linthresh``, log decades out
    to +/-``vmax``.  Defaults reproduce the LADDIE colorbar."""
    vmin = -vmax if vmin is None else vmin
    return SymLogNorm(linthresh=linthresh, linscale=linscale,
                      vmin=vmin, vmax=vmax, base=10)


def melt_ticks(vmax: float = 10.0) -> np.ndarray:
    """LADDIE tick values within +/-``vmax``."""
    return _MELT_TICKS[np.abs(_MELT_TICKS) <= vmax + 1e-9]


def add_melt_colorbar(fig, im, ax=None, *,
                      label="melt rate (m ice/yr)\nnegative = melt", **kw):
    """Colorbar with LADDIE log ticks (plain decimal labels)."""
    vmax = float(im.norm.vmax)
    ticks = melt_ticks(vmax)
    cb = fig.colorbar(im, ax=ax, ticks=ticks, extend="both", **kw)
    cb.set_ticklabels([f"{t:g}" for t in ticks])
    cb.set_label(label)
    return cb

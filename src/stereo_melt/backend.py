"""Array backend selection — numpy (default) or cupy (GPU).

Most of the pipeline is I/O or CPU-bound (ingestion, corrections, ASP coregistration).
The mass-budget stages (stack dh/dt fits, flux divergence, Lagrangian advection) are
array-heavy and benefit from GPU. This module gives those stages a single toggle.

Usage:
    from stereo_melt.backend import xp, asarray, to_numpy
    a = asarray(np.arange(10))           # numpy or cupy array per env
    b = xp.sqrt(a)
    out = to_numpy(b)                    # always numpy for I/O / plotting

Select backend with the STEREO_MELT_BACKEND env var ("numpy" or "cupy").
Defaults to numpy. Unknown or unavailable values fall back to numpy with a warning.
"""

from __future__ import annotations

import os
import warnings

import numpy as _np

_BACKEND_ENV = os.environ.get("STEREO_MELT_BACKEND", "numpy").lower()

if _BACKEND_ENV == "cupy":
    try:
        import cupy as _cp

        xp = _cp
        backend = "cupy"
    except ImportError:
        warnings.warn("STEREO_MELT_BACKEND=cupy but cupy is not installed; falling back to numpy")
        xp = _np
        backend = "numpy"
else:
    xp = _np
    backend = "numpy"


def asarray(a):
    """Move array to the active backend."""
    return xp.asarray(a)


def to_numpy(a):
    """Always return a numpy array (for I/O, plotting, xarray)."""
    if backend == "cupy":
        return xp.asnumpy(a)
    return _np.asarray(a)

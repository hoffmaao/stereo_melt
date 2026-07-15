# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Array backend selection — numpy (default) or cupy (GPU).

Most of the pipeline is I/O or CPU-bound (ingestion, corrections, ASP
coregistration). The mass-budget stages (stack dh/dt fits, flux
divergence, Lagrangian advection) are array-heavy and benefit from GPU.
This module gives those stages a single toggle.

Usage::

    from stereo_melt.backend import xp, asarray, to_numpy, map_coordinates, scatter_add

    a = asarray(np.arange(10))     # numpy or cupy array per env
    b = xp.sqrt(a)
    out = to_numpy(b)              # numpy for I/O, plotting, xarray

Select the backend with the ``STEREO_MELT_BACKEND`` environment
variable (``"numpy"`` or ``"cupy"``). Defaults to numpy. Unknown or
unavailable values fall back to numpy with a warning.
"""

from __future__ import annotations

import os
import warnings

import numpy as _np

_BACKEND_ENV = os.environ.get("STEREO_MELT_BACKEND", "numpy").lower()

if _BACKEND_ENV == "cupy":
    # Strict mode: explicit cupy request must succeed. Silent numpy fallback
    # was masking real GPU problems and giving the user incorrect "fast" runs
    # that were secretly on CPU. If cupy is requested but unavailable, raise
    # so the operator can fix the environment instead of waiting hours on
    # accidental CPU work. Fall back to numpy only when env var is unset.
    import cupy as _cp  # propagates ImportError if cupy not installed
    xp = _cp
    backend = "cupy"
elif _BACKEND_ENV == "cupy_or_numpy":
    # Opt-in soft fallback for scripts that want best-effort GPU.
    try:
        import cupy as _cp
        xp = _cp
        backend = "cupy"
    except ImportError:
        warnings.warn("STEREO_MELT_BACKEND=cupy_or_numpy but cupy is not installed; falling back to numpy")
        xp = _np
        backend = "numpy"
else:
    xp = _np
    backend = "numpy"

# Print a one-line banner when STEREO_MELT_BACKEND is explicitly set, so
# pipeline runs always show the active backend (silent numpy fallback was
# the source of "GPU isn't actually being used" confusion). Library imports
# without the env var stay quiet.
if "STEREO_MELT_BACKEND" in os.environ:
    import sys as _sys

    if backend == "cupy":
        try:
            _props = _cp.cuda.runtime.getDeviceProperties(0)
            _name = _props["name"]
            if isinstance(_name, bytes):
                _name = _name.decode()
            _ndev = _cp.cuda.runtime.getDeviceCount()
            print(f"[stereo_melt] backend=cupy device={_name} (n={_ndev})", file=_sys.stderr)
        except Exception:
            print("[stereo_melt] backend=cupy", file=_sys.stderr)
    else:
        print(f"[stereo_melt] backend=numpy (env requested {_BACKEND_ENV})", file=_sys.stderr)


def asarray(a):
    r"""Move ``a`` to the active backend."""
    return xp.asarray(a)


def to_numpy(a):
    r"""Return a numpy array, regardless of the active backend."""
    if backend == "cupy":
        return xp.asnumpy(a)
    return _np.asarray(a)


def gaussian_filter(arr, sigma, mode="nearest", cval=0.0):
    r"""Backend-aware :func:`scipy.ndimage.gaussian_filter`.

    Parameters
    ----------
    arr : ndarray
        Source array on the active backend.
    sigma : float or sequence of float
        Standard deviation in pixels per axis.
    mode, cval
        Boundary handling, passed through.
    """
    if backend == "cupy":
        from cupyx.scipy.ndimage import gaussian_filter as _gf
    else:
        from scipy.ndimage import gaussian_filter as _gf
    return _gf(arr, sigma=sigma, mode=mode, cval=cval)


def map_coordinates(arr, coords, order=1, mode="nearest", cval=0.0):
    r"""Backend-aware :func:`scipy.ndimage.map_coordinates`.

    Parameters
    ----------
    arr : ndarray
        Source array on the active backend.
    coords : sequence of ndarray or 2-D ndarray
        Either ``[rows, cols, ...]`` (a Python list/tuple of per-axis
        index arrays — what scipy accepts directly) or a single stacked
        ndarray of shape ``(ndim, *output_shape)``. Cupy's port only
        accepts the stacked form, so lists are coerced via ``xp.stack``
        for backend parity.
    order, mode, cval
        Passed through to the underlying ``map_coordinates``.
    """
    if backend == "cupy":
        from cupyx.scipy.ndimage import map_coordinates as _mc
    else:
        from scipy.ndimage import map_coordinates as _mc
    if isinstance(coords, (list, tuple)):
        coords = xp.stack([xp.asarray(c) for c in coords], axis=0)

    # Workaround: cupy 13.x + NVRTC 11.7 (CUDA 11.2 driver) fails to compile
    # the map_coordinates elementwise kernel when cval is non-finite (NaN/inf).
    # Emulate cval=NaN with a finite sentinel + a second weight-mapped call.
    import math
    if (
        backend == "cupy"
        and mode == "constant"
        and not math.isfinite(float(cval))
    ):
        sentinel = 0.0
        sampled = _mc(arr, coords, order=order, mode="constant", cval=sentinel)
        weight = _mc(
            xp.ones_like(arr), coords, order=order, mode="constant", cval=0.0
        )
        return xp.where(weight >= 1.0 - 1e-6, sampled, xp.asarray(cval))

    return _mc(arr, coords, order=order, mode=mode, cval=cval)


def scatter_add(a, indices, values):
    r"""Backend-aware in-place scatter-add: ``a[indices] += values``.

    Handles duplicate entries in ``indices`` correctly (unlike plain
    fancy-indexed assignment). On numpy uses :func:`numpy.add.at`; on
    cupy uses :func:`cupyx.scatter_add` which dispatches to an atomic
    CUDA kernel.
    """
    if backend == "cupy":
        import cupyx

        cupyx.scatter_add(a, indices, values)
    else:
        _np.add.at(a, indices, values)

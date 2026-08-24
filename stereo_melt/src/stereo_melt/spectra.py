# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Spectral analysis of melt-rate (and other) rasters on masked domains.

Melt products live on irregular shelf masks, so a plain periodogram is
dominated by leakage from the mask edge. The estimator here apodizes the
mask, demeans under the taper, and Welch-normalizes so that the integral of
the two-dimensional PSD over frequency equals the variance of the tapered
field — which makes radially averaged spectra comparable across products
with different grids, as long as they share the analysis mask.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

__all__ = ["radial_psd"]


def radial_psd(field, mask, dx_km, taper_px=6.0, fmin=None, nbins=36,
               band_km=3.0):
    r"""Return the radially averaged power spectral density on a masked grid.

    The mask indicator is smoothed into an apodization taper
    :math:`w = G_\sigma * \mathbf 1_M` (zeroed outside the mask), the field
    is demeaned under the taper, and the periodogram is Welch-normalized,

    .. math::
        P(\mathbf f) = \frac{|\widehat{w\,(a - \bar a)}|^2\,
                             \Delta x\,\Delta y}{\sum w^2},
        \qquad
        \iint P\, df_x\, df_y = \operatorname{Var}[w\,(a - \bar a)],

    then averaged over annuli :math:`|\mathbf f| \in [f_i, f_{i+1})` on a
    log-spaced grid up to the Nyquist frequency. Frequencies are in cycles
    per km, so different products compared over a common physical mask are
    on the same axis regardless of posting.

    Parameters
    ----------
    field : numpy.ndarray, shape ``(ny, nx)``
        The raster to analyze (e.g. melt rate, m/yr). Sign does not affect
        the PSD.
    mask : numpy.ndarray of bool
        Analysis domain; cells outside are excluded from the mean and taper.
    dx_km : float
        Grid spacing in km (isotropic).
    taper_px : float
        Gaussian apodization scale in pixels. Scale it inversely with the
        posting so the physical taper width matches across products.
    fmin : float, optional
        Smallest wavenumber bin edge (cycles/km); defaults to twice the
        fundamental of the shorter axis.
    nbins : int
        Number of log-spaced radial bins.
    band_km : float
        Wavelength (km) splitting the "short-scale" variance diagnostic.

    Returns
    -------
    k : numpy.ndarray
        Bin-center wavenumbers (cycles/km).
    psd : numpy.ndarray
        Radially averaged PSD (field-units\ :sup:`2` km\ :sup:`2`); NaN in
        bins with fewer than five spectral samples.
    stats : dict
        ``mean`` (taper-weighted mean of the field), ``var_total`` and
        ``var_band`` (variance of the tapered field, total and at
        :math:`\lambda <` ``band_km``).
    """
    m = np.asarray(mask, bool) & np.isfinite(field)
    w = gaussian_filter(m.astype(float), taper_px)
    w[~m] = 0.0
    mean = float((w * np.where(m, field, 0.0)).sum() / w.sum())
    a = np.where(m, field - mean, 0.0) * w
    ny, nx = a.shape
    F = np.fft.fft2(a)
    psd2 = (np.abs(F) ** 2) * (dx_km * dx_km) / (w ** 2).sum()
    fx = np.fft.fftfreq(nx, d=dx_km)
    fy = np.fft.fftfreq(ny, d=dx_km)
    kr = np.sqrt(fx[None, :] ** 2 + fy[:, None] ** 2)
    fnyq = 0.5 / dx_km
    if fmin is None:
        fmin = 1.0 / (min(nx, ny) * dx_km / 2.0)
    edges = np.logspace(np.log10(fmin), np.log10(fnyq), nbins + 1)
    k = np.sqrt(edges[:-1] * edges[1:])
    psd = np.full(nbins, np.nan)
    flat_k, flat_p = kr.ravel(), psd2.ravel()
    idx = np.digitize(flat_k, edges) - 1
    ok = (idx >= 0) & (idx < nbins)
    counts = np.bincount(idx[ok], minlength=nbins)
    sums = np.bincount(idx[ok], weights=flat_p[ok], minlength=nbins)
    good = counts > 4
    psd[good] = sums[good] / counts[good]
    df = (1.0 / (nx * dx_km)) * (1.0 / (ny * dx_km))
    band = kr > (1.0 / band_km)
    stats = {"mean": mean,
             "var_total": float(psd2[kr > 0].sum() * df),
             "var_band": float(psd2[band].sum() * df)}
    return k, psd, stats

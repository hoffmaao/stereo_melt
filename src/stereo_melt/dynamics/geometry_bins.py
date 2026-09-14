"""Deterministic geometry binning behind the LOCAL bridging operators (torch-free).

:func:`kmeans_geometry` is the clustering behind ``n_bins`` in the monolithic
budget+bridging solver and the restored-budget solver;
:func:`geometry_bins` wraps it with the partition-of-unity blend weights the
B-PINN observation operator uses. Cells are binned on ``(H, u_x, u_y[, extra])``
so every bin gets its own transfer -- keyed on the velocity VECTOR, so a shelf
whose flow turns gets different operators on its limbs -- and the per-bin
responses are recombined with Gaussian-smoothed weights that sum to one.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter


def kmeans_geometry(
    feats: np.ndarray, n_bins: int, iters: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 1-D-seeded Lloyd clustering of standardized ``(H, ux, uy)``.

    Seeded from evenly spaced quantiles along the first principal component (no
    RNG), so a given geometry always yields the same bins -- a solver that
    silently changed its operator between runs would be unusable for A/B work.
    Returns the per-cell label and the ``(n_bins, 3)`` centroids in feature units.
    """
    mu = feats.mean(0)
    sd = feats.std(0)
    sd[sd <= 0] = 1.0
    z = (feats - mu) / sd
    # principal direction via the power method on the covariance (no scipy/sklearn)
    C = np.cov(z.T) + 1e-12 * np.eye(z.shape[1])
    v = np.ones(z.shape[1]) / math.sqrt(z.shape[1])
    for _ in range(100):
        v = C @ v
        v /= max(np.linalg.norm(v), 1e-30)
    proj = z @ v
    qs = np.quantile(proj, (np.arange(n_bins) + 0.5) / n_bins)
    cent = np.stack([z[np.argmin(np.abs(proj - q))] for q in qs])

    lab = np.zeros(len(z), dtype=int)
    for _ in range(iters):
        d = ((z[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if np.array_equal(new, lab):
            break
        lab = new
        for b in range(n_bins):
            sel = lab == b
            if sel.any():
                cent[b] = z[sel].mean(0)
    return lab, cent * sd + mu


def geometry_bins(H, vx, vy, domain, n_bins, blend_px, extra=None):
    """Bins on ``(H, u_x, u_y[, extra])`` over ``domain`` with partition-of-unity weights.

    Returns ``(geom, W)``: ``geom`` is a list of per-bin centroid tuples in
    physical units and ``W`` is ``(n_bins, ny, nx)`` and sums to one everywhere.
    Uniform geometry (to 1e-6 relative) collapses to ONE bin so a single global
    operator is reproduced exactly, not to round-off; ``n_bins <= 1`` is the
    domain mean. Same construction as ``budget_bridging_melt_rate(n_bins=...)``.
    """
    H = np.asarray(H, float)
    ny, nx = H.shape
    cols = [H, np.asarray(vx, float), np.asarray(vy, float)]
    valid = np.asarray(domain, bool) & np.isfinite(cols[0]) & np.isfinite(cols[1]) \
        & np.isfinite(cols[2]) & (cols[0] > 0)
    if extra is not None:
        cols.append(np.asarray(extra, float))
        valid &= np.isfinite(cols[3]) & (cols[3] > 0)
    if not valid.any():
        raise ValueError("geometry_bins: no valid cells (finite H > 0 and finite velocity on the domain)")
    feats = np.stack([c[valid] for c in cols], axis=1)
    scale = np.maximum(np.abs(feats).max(0), 1e-30)
    n_uniq = len(np.unique(np.round(feats / scale, 6), axis=0))
    nb = max(1, min(int(n_bins), n_uniq))
    if nb == 1:
        lab, cent = np.zeros(len(feats), int), feats.mean(0, keepdims=True)
    else:
        lab, cent = kmeans_geometry(feats, nb)
    W = np.zeros((nb, ny, nx))
    geom = []
    for b in range(nb):
        geom.append(tuple(float(c) for c in cent[b]))
        ind = np.zeros((ny, nx))
        ind[valid] = (lab == b)
        W[b] = gaussian_filter(ind, blend_px, mode="nearest")
    W[0][W.sum(0) < 1e-8] = 1.0
    W /= np.maximum(W.sum(0), 1e-30)
    return geom, W

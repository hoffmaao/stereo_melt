"""Deterministic geometry binning behind the LOCAL bridging operators (torch-free).

:func:`kmeans_geometry` is the clustering behind ``n_bins`` in the monolithic
budget+bridging solver and the restored-budget solver;
:func:`geometry_bins` wraps it with the partition-of-unity blend weights the
B-PINN observation operator uses. Cells are binned on ``(H, u_x, u_y[, extra])``
so every bin gets its own transfer -- keyed on the velocity VECTOR, so a shelf
whose flow turns gets different operators on its limbs -- and the per-bin
responses are recombined with Gaussian-smoothed weights that sum to one.

Torch-free on purpose, and that is what this module is for: ``kmeans_geometry``
used to live in :mod:`~stereo_melt.dynamics.stubblefield_forward`, which imports
torch at module level, and the B-PINN runs in a JAX environment that has no
torch. That module re-imports it from here under its old private name, so its
own callers and gates are unchanged; do not fold this back into it.
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


def _lloyd(z, cent, iters):
    """Lloyd iterations in standardized space from explicit seeds ``cent``.

    The sentinel starts at -1, not 0: an all-zero first assignment is a real
    outcome (every cell nearest seed 0), and treating it as "converged" is how
    coincident quantile seeds leave the centroids untouched at their seed values.
    """
    lab = np.full(len(z), -1, dtype=int)
    for _ in range(iters):
        d = ((z[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if np.array_equal(new, lab):
            break
        lab = new
        for b in range(len(cent)):
            sel = lab == b
            if sel.any():
                cent[b] = z[sel].mean(0)
    return lab, cent


def farthest_point_geometry(
    feats: np.ndarray, n_bins: int, iters: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic farthest-point-seeded Lloyd clustering of standardized ``(H, ux, uy)``.

    The fallback for when :func:`kmeans_geometry`'s quantile seeding degenerates.
    On a domain ONE geometry dominates -- a uniform shelf with a small anomalous
    patch -- every quantile of the principal-component projection lands on that
    same dominant feature row, so the seeds coincide and the patch that motivated
    ``n_bins > 1`` never gets a bin. Seeding instead from the cell nearest the
    feature mean and then repeatedly taking the cell farthest from every seed
    chosen so far is just as RNG-free and reproducible, and it reaches the patch
    on its second seed. Returns the per-cell label and the centroids in feature
    units, like :func:`kmeans_geometry`; fewer seeds than ``n_bins`` are returned
    when the features run out of distinct points.
    """
    mu = feats.mean(0)
    sd = feats.std(0)
    sd[sd <= 0] = 1.0
    z = (feats - mu) / sd
    idx = [int(np.argmin((z ** 2).sum(1)))]
    d2 = ((z - z[idx[0]]) ** 2).sum(1)
    while len(idx) < n_bins:
        nxt = int(np.argmax(d2))
        if d2[nxt] <= 0:
            break
        idx.append(nxt)
        d2 = np.minimum(d2, ((z - z[nxt]) ** 2).sum(1))
    lab, cent = _lloyd(z, z[idx].copy(), iters)
    return lab, cent * sd + mu


def _drop_dead_bins(lab, cent, scale):
    """Merge centroids duplicating an earlier one, drop those left with no members."""
    keep: list[int] = []
    for b in range(len(cent)):
        dup = next((k for k in keep
                    if np.all(np.abs(cent[b] - cent[k]) <= 1e-6 * scale)), None)
        if dup is not None:
            lab[lab == b] = dup
        elif (lab == b).any():
            keep.append(b)
    if len(keep) == len(cent):
        return lab, cent
    remap = np.full(len(cent), -1, int)
    remap[keep] = np.arange(len(keep))
    return remap[lab], cent[keep]


def geometry_bins(H, vx, vy, domain, n_bins, blend_px, extra=None):
    """Bins on ``(H, u_x, u_y[, extra])`` over ``domain`` with partition-of-unity weights.

    Returns ``(geom, W, info)``: ``geom`` is a list of per-bin centroid tuples in
    physical units, ``W`` is ``(len(geom), ny, nx)`` and sums to one everywhere,
    and ``info`` reports the binning outcome for the caller's log --
    ``requested``, ``n_unique`` (distinct geometries on the domain, to 1e-6
    relative), ``effective`` (``len(geom)``), ``uniform`` (``n_unique == 1``),
    ``reseeded`` and a ``reason`` string, empty unless ``effective`` fell short
    of ``requested``. Every centroid is a mean over the VALID CELLS OF
    ``domain`` -- off-domain pixels (grounded ice, open ocean on a shelf window)
    would pull a bin's operator away from the ice it is applied to -- which is
    not the same quantity as a whole-stack summary such as the B-PINN's
    ``data.H0``; see :func:`stereo_melt.dynamics.bpinn.fit_bpinn` for why that
    difference is kept and how the ``uniform`` case is reconciled with it.

    ONLY ``uniform`` geometry collapses to one bin as a matter of the data: a
    single global operator is then reproduced exactly, not to round-off, and
    ``n_bins <= 1`` is the domain mean. Everything else is a property of the
    clustering, and is repaired rather than accepted. A cluster with no members,
    or whose centroid duplicates another, is dropped or merged -- it would build
    a duplicate or unused operator and cost a full FFT per epoch per optimiser
    step in the B-PINN hot loop, and would overstate the bin count -- and if
    that leaves fewer bins than the domain can support, the cells are re-clustered
    from :func:`farthest_point_geometry`'s seeds, because the degenerate case is
    exactly the dominant-geometry-plus-anomalous-patch field that ``n_bins > 1``
    is for. ``kmeans_geometry`` itself is untouched: the monolithic solver shares
    it. Same construction as ``budget_bridging_melt_rate(n_bins=...)``.
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
    nb_max = max(1, min(int(n_bins), n_uniq))
    reseeded = False
    if nb_max == 1:
        lab, cent = np.zeros(len(feats), int), feats.mean(0, keepdims=True)
    else:
        lab, cent = _drop_dead_bins(*kmeans_geometry(feats, nb_max), scale)
        if len(cent) < nb_max:
            lab_f, cent_f = _drop_dead_bins(*farthest_point_geometry(feats, nb_max), scale)
            if len(cent_f) > len(cent):
                lab, cent, reseeded = lab_f, cent_f, True
    nb = len(cent)
    if nb >= int(n_bins):
        reason = ""
    elif nb == n_uniq:
        reason = f"the domain holds only {n_uniq} distinct geometries"
    elif reseeded:
        reason = "farthest-point re-seeding still left duplicate or empty clusters"
    else:
        reason = "the clustering left duplicate or empty clusters"
    info = {"requested": int(n_bins), "n_unique": int(n_uniq), "effective": nb,
            "uniform": bool(n_uniq == 1), "reseeded": reseeded, "reason": reason}
    W = np.zeros((nb, ny, nx))
    geom = []
    for b in range(nb):
        geom.append(tuple(float(c) for c in cent[b]))
        ind = np.zeros((ny, nx))
        ind[valid] = (lab == b)
        W[b] = gaussian_filter(ind, blend_px, mode="nearest")
    W[0][W.sum(0) < 1e-8] = 1.0
    W /= np.maximum(W.sum(0), 1e-30)
    return geom, W, info

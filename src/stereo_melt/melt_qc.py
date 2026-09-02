"""Melt-space strip QC: attribute melt-map artefacts to epochs.

Companion to :mod:`stereo_melt.coregister.tilt_qc` (stack-space QC). A strip
can pass static-control residual screening yet carry a floating-area bias
that the hydrostatic gain (rho_w/(rho_w-rho_i) ~ 9.4) turns into m/yr of
fake melt or accretion wherever cross-pair count is thin.

Workflow (see ``venable/diag_stripe_strips.py`` for the reference driver):

1. Run :func:`stereo_melt.melt.lagrangian_melt_rate` with
   ``pair_diag_masks`` (one bool mask per suspect region + the whole
   floating shelf as reference) to get per-pair region medians.
2. Feed each region's pairs to :func:`fit_pair_epoch_bias`: a strip with
   ice-equivalent thickness bias ``s_e`` enters every pair with opposite
   sign as earlier vs later member, so the per-pair medians decompose as
   ``med_p ~= mu + (s_j - s_i)/dt_p``.
3. Drop the single worst offender into the basin's ``BAD_STRIPS`` and
   iterate — a gross offender inflates the robust scale and masks the next
   one, so one strip per pass.
"""
from __future__ import annotations

import numpy as np


def fit_pair_epoch_bias(
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    dt_yr: np.ndarray,
    pair_value: np.ndarray,
    n_cells: np.ndarray | None = None,
    n_epochs: int | None = None,
    epoch_time_yr: np.ndarray | None = None,
    c: float = 4.685,
    max_iter: int = 10,
    ridge_frac: float = 0.1,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Robustly solve ``value_p = mu + (s_j - s_i)/dt_p`` for per-epoch bias.

    The model has TWO exact nullspace directions that must be gauged, not
    estimated: ``s -> s + const`` (differences cancel) and — because
    ``s_e = a * t_e`` gives ``(s_j - s_i)/dt == a`` for every pair —
    ``(mu, s) -> (mu - a, s + a * t)``. Both are pinned with hard gauge
    rows (zero mean and zero time-trend of ``s`` over the used epochs), so
    ``mu`` absorbs the region's common melt level plus any real common
    trend, and ``s`` holds only per-epoch departures. A small per-epoch
    ridge additionally tames epochs seen in very few pairs.

    Parameters
    ----------
    pair_i, pair_j : int arrays (n_pairs,)
        Epoch indices of each pair's earlier / later member.
    dt_yr : float array (n_pairs,)
        Pair baselines in years.
    pair_value : float array (n_pairs,)
        Per-pair region statistic (median melt over the region's cells,
        m ice/yr in the Shean convention).
    n_cells : int array or None
        Region cell count per pair; rows are weighted by
        ``sqrt(min(n_cells, 500))``. ``None`` = equal weights.
    n_epochs : int or None
        Length of the returned ``s`` vector (epochs absent from the pairs
        get NaN). ``None`` = ``max(epoch index) + 1``.
    epoch_time_yr : float array (n_epochs,) or None
        Epoch times in years for the trend gauge. ``None`` uses the epoch
        index as a proxy (fine for roughly regular records).
    c, max_iter : float, int
        Tukey biweight constant and IRLS iteration cap.
    ridge_frac : float
        Per-epoch ``s_e = 0`` ridge weight as a FRACTION of the median
        data-row weight. dt-banded records are near-bipartite (bursty
        campaigns: early epochs appear only as ``i``, late only as ``j``),
        leaving soft modes that would otherwise absorb single outlier
        pairs instead of letting IRLS reject them; a data-scale ridge
        blocks that at the cost of shrinking ``s`` magnitudes somewhat —
        ``s`` is a ranking score, not an unbiased bias estimate.

    Returns
    -------
    (mu, s, resid, weights)
        ``mu`` — the region's common melt level (m ice/yr); ``s`` —
        per-epoch ice-equivalent thickness bias, meters (surface bias
        ~= ``s * (rho_w - rho_i)/rho_w``, i.e. /9.42 with the package
        constants); ``resid`` / ``weights`` — per-pair residuals and final
        IRLS weights.
    """
    pair_i = np.asarray(pair_i, dtype=np.int64)
    pair_j = np.asarray(pair_j, dtype=np.int64)
    dt_yr = np.asarray(dt_yr, dtype=np.float64)
    pair_value = np.asarray(pair_value, dtype=np.float64)
    if n_epochs is None:
        n_epochs = int(max(pair_i.max(), pair_j.max())) + 1
    if epoch_time_yr is None:
        epoch_time_yr = np.arange(n_epochs, dtype=np.float64)

    used = np.union1d(pair_i, pair_j)
    col = {e: k for k, e in enumerate(used)}
    n_p, n_u = pair_value.size, used.size
    A = np.zeros((n_p, 1 + n_u))
    A[:, 0] = 1.0
    for p in range(n_p):
        A[p, 1 + col[pair_j[p]]] += 1.0 / dt_yr[p]
        A[p, 1 + col[pair_i[p]]] -= 1.0 / dt_yr[p]
    if n_cells is None:
        w_cell = np.ones(n_p)
    else:
        w_cell = np.sqrt(np.minimum(np.asarray(n_cells), 500).astype(float))

    # Gauge + ridge rows. The mean/trend rows carry a weight far above the
    # data rows: they span exact nullspace directions, so they cannot fight
    # the data — they only pick the representative solution. The ridge is at
    # data scale (ridge_frac * median row weight): without it the
    # near-bipartite dt-band design leaves soft modes that bend to absorb an
    # outlier pair (so IRLS never sees a large residual and never rejects
    # it); with it the outlier stands out and is downweighted.
    t_used = np.asarray(epoch_time_yr, dtype=np.float64)[used]
    t_ctr = t_used - t_used.mean()
    gauge_w = 100.0 * float(w_cell.max())
    ridge = ridge_frac * float(np.median(w_cell))
    G = np.zeros((2 + n_u, 1 + n_u))
    G[0, 1:] = gauge_w                    # sum(s) = 0
    G[1, 1:] = gauge_w * t_ctr            # sum(t~ * s) = 0
    G[2:, 1:] = np.eye(n_u) * ridge       # per-epoch data-scale ridge
    zeros_g = np.zeros(2 + n_u)

    w = np.ones(n_p)
    x = np.zeros(1 + n_u)
    for _ in range(max_iter):
        ww = w * w_cell
        Aw = np.vstack([A * ww[:, None], G])
        bw = np.concatenate([pair_value * ww, zeros_g])
        x, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        r = pair_value - A @ x
        scale = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        u = np.abs(r) / (c * scale)
        w_new = np.where(u < 1.0, (1.0 - u**2) ** 2, 0.0)
        if np.allclose(w_new, w, atol=1e-3):
            w = w_new
            break
        w = w_new
    s = np.full(n_epochs, np.nan)
    s[used] = x[1:]
    return float(x[0]), s, pair_value - A @ x, w

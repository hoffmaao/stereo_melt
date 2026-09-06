# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Empirical variograms and spatially-correlated error propagation.

The Hugonnet et al. (2022) treatment of DEM error, in the standard
geostatistical vocabulary the DEM-error literature uses, so our numbers are
comparable with the community's. Companion to :mod:`stereo_melt.spectra` (radial PSD): a variogram
is the same second-order information in lag space, and it is the form the DEM
literature reports and the form that propagates to the error of a **spatial
average** -- which is what a basal-melt flux is.

Why it matters here. Every regularisation selector we have tried assumes white
observation noise, and each fails on real stacks because the strip error is
correlated at kilometre scales -- the same scales as the melt signal. The
variogram measures that correlation directly and without a model, and

.. math::
    \operatorname{Var}(\bar z_A) \;=\; \frac{1}{N^2}\sum_i\sum_j C(h_{ij}),
    \qquad C(h) = \sigma^2_{\rm tot} - \gamma(h)

turns it into an honest error bar on an area mean. The ratio
:math:`n_{\rm eff} = \sigma^2_{\rm tot}/\operatorname{Var}(\bar z_A)` is the
number of *independent* samples the area is really worth: for white noise it is
the pixel count, for a field correlated across the whole domain it is ~1. Using
the pixel count when the noise is correlated is the standard way to understate
a flux uncertainty by an order of magnitude.

Measured on PIG (2026-09-02)
---------------------------
**Control residuals -- the clean result.** Variogram of (aligned DEM minus the
control ``pc_align`` was fed), pooled over 120 QC-passing canon strips:
nugget **0.063 m^2** (0.25 m, about the control's own precision) plus a
spherical structure of **0.190 m^2 (0.44 m) with a range of 4.1 km**
(r2 0.95). That is the first proper measurement of the "strip error is
correlated at 1-4 km" claim, and it sits squarely in the bridging band. Note
what it says about the per-strip Sigma in
:func:`~stereo_melt.dynamics.budget_bridging.strip_prior_from_residual_planes`:
an offset-plus-plane model predicts ``gamma = (tau_x^2+tau_y^2) h^2/4``, which
at 9 km is 2e-4 m^2 -- about **0.1 %** of the measured correlated variance. The
plane amplitudes are right, but the plane BASIS captures almost none of the
actual coloured error, which is a 4 km structure no per-strip plane can carry.

**Heteroscedasticity first -- but on a domain this size the step is NOT
separable from large-scale structure. RETRACTED as a physical claim
(2026-09-03).** Fitting ``sigma(count, rmse)`` on the PIG half-stack
difference appears to show a 24x heteroscedasticity (NMAD ratio between the
lowest- and highest-count quartiles, 24.3 raw -> 1.00 standardised, sigma
spanning 1.0-89.8 m/yr). Three checks say that is largely an artefact:

* **sigma is itself a low-wavenumber field.** The variogram of ``log sigma``
  has a **33 km range with a 2 % nugget** -- it varies coherently across the
  shelf, so binning by these covariates is close to binning by POSITION, and
  standardising removes genuine low-k variance rather than isolating an error
  magnitude.
* **the covariates do not support the model.** ``corr(rmse, sigma) = +0.01``:
  rmse carries nothing, so the "2-D" fit is a noisy 1-D count trend
  (Spearman -0.68 over 9 bins).
* **it does not transfer.** Fit on the west half, applied east, it
  OVERCORRECTS the count-heteroscedasticity to 0.16 (target 1.0; raw 2.13),
  and the east's own model only reaches 1.47. The clean "-> 1.00" is a
  whole-domain fit quoting its in-sample residual.

The published method is calibrated on small regional differences where sigma
really is terrain-driven and the domain cannot hold much low-wavenumber
structure;
PIG is 156 x 107 km with 33 km error coherence, and there the two are
confounded. **What survives:** a real but modest and noisy decrease of error
with epoch count, consistent with the previously measured sigma(n) slope --
not a 24x effect. Use this API where sigma is a defensible FUNCTION of a
covariate (validate by out-of-sample transfer, as above), not as a way to
flatten a large field before a variogram. The gate exercises it on synthetics
where sigma genuinely is such a function.

**Melt error -- NOT identifiable from PIG's domain; do not quote n_eff there.**
Applied to the half-stack difference ``(A-B)/sqrt(2)``, the fitted long range
simply tracks whatever ``max_lag`` is chosen (20.8 km at 5 km, 41.5 at 10,
73.9 at 40, 123.1 at 60), and the implied flux uncertainty swings 25 -> 107
Gt/yr with it. A variogram cannot constrain a range beyond roughly half its
max lag, and PIG is 156 x 107 km. Two further cautions there: the melt error
is strongly heteroscedastic (robust sd 12 m/yr against an rms of 90), and
Hugonnet's method standardises by a fitted sigma(terrain) BEFORE the variogram
-- that step is not implemented here. Standardising shrinks but does not cure this: the LONG range still tracks
``max_lag`` (4.8 km @5 -> 17.2 @20 -> 29.0 @60), and because that term
dominates the double sum the heteroscedastic propagation returns 33 Gt/yr --
inheriting the unidentifiability rather than resolving it. For an area-mean
error bar on PIG use the model-free block estimator instead: block means over 5/10/20 km give a
stable **3.1 / 4.4 / 5.4 Gt/yr**, consistent with the 2.8 Gt/yr on record and
with the shelf-mean error of -0.95 m/yr (-3.9 Gt/yr); the white-noise
assumption would claim 0.73 Gt/yr.

Estimators
----------
``"matheron"``
    The classical :math:`\gamma(h)=\tfrac12\overline{(\Delta z)^2}`.
``"dowd"`` (default)
    Dowd's robust estimator :math:`\gamma(h)=\tfrac12(1.4826\,
    \operatorname{median}|\Delta z|)^2`. DEM residuals are heavy-tailed --
    on PIG ~15 % of strips are alignment failures carrying metres of error --
    and one blunder pair can dominate a Matheron bin. Prefer this on real data
    and check the two against each other.

Models are the standard nested set (``nugget``, ``spherical``, ``exponential``,
``gaussian``); real DEM error usually needs a **sum** of two ranges, a short
one from the sensor/matching and a long one from the georeferencing, which is
why :func:`fit_variogram` takes a list.

Range convention
----------------
Every range this **API** accepts or reports -- ``variogram_model``'s ``rng_``,
``fit_variogram``'s ``ranges``, and the ``params`` tuples the propagation
functions consume -- is the standard geostatistical **effective** range
:math:`r`: the lag at which a structure has essentially reached its sill, not
the internal e-folding scale :math:`a`. The two coincide only for
``spherical`` (:math:`a=r`); ``exponential`` uses :math:`a=r/3` and
``gaussian`` :math:`a=r/2`, so :math:`\gamma(r)` is 95 % and 98 % of the sill
respectively. That is what makes the comparability claim above mean anything:
quoting an e-folding scale as a "range" would report an exponential structure
as 3x shorter than the literature measures the same field.

This is a statement about the API, not a retroactive relabelling of the
measurements above. Every PIG figure quoted in this docstring came from a
``spherical`` structure, where :math:`a = r` already, so none of them moves
under the convention: the 4.12 km strip-residual range, the ``max_lag`` sweep
of melt-error ranges, the standardised short and long ranges, and the 33 km
``log sigma`` range in the retraction are all unaffected and need no
re-derivation.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "nmad",
    "nd_binning",
    "interp_nd_binning",
    "two_step_standardization",
    "infer_heteroscedasticity_from_stable",
    "empirical_variogram",
    "variogram_model",
    "fit_variogram",
    "covariance_from_variogram",
    "number_effective_samples",
    "standard_error_of_mean",
]

# MAD -> sigma. For zero-mean pair differences median|dz| IS the MAD, so
# sigma_dz = 1.4826 median|dz| and gamma = sigma_dz^2/2. (Dowd's constant is
# often quoted as 2.198 = 1.4826^2, which multiplies the SQUARED median --
# putting it inside the square inflates every bin by exactly 2.198.)
_MAD_TO_SIGMA = 1.4826



# ---------------------------------------------------------------------------
# Heteroscedasticity: the step that comes BEFORE the variogram
# ---------------------------------------------------------------------------
# Hugonnet's method has two halves and they are not interchangeable. First the
# error MAGNITUDE is modelled as a function of explanatory variables --
# sigma(x1, x2, ...) by robust binning -- and the data are standardised,
# z = dh / sigma. Only then is the variogram of z meaningful, because a
# variogram assumes stationarity: if sigma varies across the map, the raw
# variogram mixes "the error is bigger over there" into "the error is
# correlated over that distance", and the two are not separable after the fact.
# Ours is strongly heteroscedastic (PIG melt error: robust sd 12 m/yr against
# an rms of 90), so this step is not optional.
#
# The published method bins on terrain (slope, maximum curvature). The
# equivalent quality
# proxies for a DEM-stack melt product are the per-pixel epoch COUNT and the
# regression RMSE -- the count dependence is already measured (sigma ~ n^-0.87
# on the trunk, n^-0.38 on the slow shelf), which is exactly the kind of
# structure this is meant to absorb.


def nmad(x) -> float:
    r"""Normalised median absolute deviation, :math:`1.4826\,\mathrm{MAD}`."""
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(_MAD_TO_SIGMA * np.median(np.abs(a - np.median(a))))


def nd_binning(values, list_var, list_var_names, *, list_var_bins=None,
               statistics=("count", "median", "nmad"), min_count: int = 30):
    """N-dimensional robust binning of ``values`` against explanatory variables.

    Bins are closed on the right of the LAST bin only, as ``np.histogram`` is:
    a sample sitting exactly at a covariate's maximum belongs to the top bin,
    not outside the binning. That matters because the default edges are
    quantiles, whose top edge IS the data maximum, and because the covariate
    this is built for -- per-pixel epoch count -- piles up at its ceiling.

    Returns a tidy ``DataFrame`` holding, for every 1-D marginal AND the full
    N-D cell, the bin per variable plus the requested statistics. Columns are
    ``nd``, then ``<name>_bin`` (the ``pd.Interval``) and ``<name>_mid`` (its
    centre) for each variable, then one column per statistic. The interval is
    SUFFIXED rather than named for the variable so a covariate can share a name
    with a statistic -- ``count`` is exactly that case, and it is this module's
    headline covariate, so an unsuffixed column would be silently overwritten
    by the sample count. The marginals are what you inspect to see whether a
    variable matters at all; the N-D block is what :func:`interp_nd_binning`
    turns into ``sigma(...)``. ``nmad`` is the dispersion and is the one used
    as the error model.
    """
    import itertools

    import pandas as pd

    v = np.asarray(values, float)
    nvar = len(list_var)
    arrs = [np.asarray(a, float) for a in list_var]
    if list_var_bins is None:
        list_var_bins = []
        for a in arrs:
            f = a[np.isfinite(a)]
            q = np.unique(np.nanquantile(f, np.linspace(0, 1, 11))) if f.size else np.array([0.0, 1.0])
            list_var_bins.append(q if q.size > 2 else np.linspace(np.nanmin(f), np.nanmax(f) + 1e-9, 6))
    elif np.isscalar(list_var_bins):
        nb = int(list_var_bins)
        list_var_bins = [np.nanquantile(a[np.isfinite(a)], np.linspace(0, 1, nb + 1)) for a in arrs]
    list_var_bins = [np.unique(np.asarray(b, float)) for b in list_var_bins]

    nb = [b.size - 1 for b in list_var_bins]
    # np.digitize puts a value equal to the TOP edge in bin nb, one past the
    # last. With the default quantile edges that top edge is the data maximum,
    # so every sample at a covariate's ceiling -- a large share of a per-pixel
    # epoch COUNT field -- would be dropped from the marginals AND the N-D
    # cell. Fold that class into the last bin (np.histogram's closed-right
    # convention); values genuinely above the top edge stay out.
    idx = []
    for a, b, nbk in zip(arrs, list_var_bins, nb):
        i = np.digitize(a, b) - 1
        idx.append(np.where(a == b[-1], nbk - 1, i))
    valid = np.isfinite(v)
    for k in range(nvar):
        valid &= (idx[k] >= 0) & (idx[k] < nb[k])

    def _stats(mask):
        out = {}
        sel = v[mask]
        for st in statistics:
            if st == "count":
                out["count"] = int(sel.size)
            elif st == "median":
                out["median"] = float(np.median(sel)) if sel.size else np.nan
            elif st == "nmad":
                out["nmad"] = nmad(sel) if sel.size >= min_count else np.nan
            elif st == "std":
                out["std"] = float(np.std(sel)) if sel.size else np.nan
            elif st == "mean":
                out["mean"] = float(np.mean(sel)) if sel.size else np.nan
        return out

    rows = []
    for dims in list(itertools.combinations(range(nvar), 1)) + ([tuple(range(nvar))] if nvar > 1 else []):
        for combo in itertools.product(*[range(nb[k]) for k in dims]):
            mask = valid.copy()
            for k, b in zip(dims, combo):
                mask &= idx[k] == b
            if not mask.any():
                continue
            row = {"nd": len(dims)}
            for k in range(nvar):
                if k in dims:
                    b = combo[dims.index(k)]
                    row[list_var_names[k] + "_bin"] = pd.Interval(
                        list_var_bins[k][b], list_var_bins[k][b + 1])
                    row[list_var_names[k] + "_mid"] = 0.5 * (list_var_bins[k][b] + list_var_bins[k][b + 1])
                else:
                    row[list_var_names[k] + "_bin"] = np.nan
                    row[list_var_names[k] + "_mid"] = np.nan
            row.update(_stats(mask))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["bins"] = {n: b for n, b in zip(list_var_names, list_var_bins)}
    return df


def interp_nd_binning(df, list_var_names, *, statistic: str = "nmad",
                      min_count: int = 30):
    """Turn the N-D block of :func:`nd_binning` into a callable ``sigma(...)``.

    Empty or under-populated cells are filled by nearest neighbour in bin
    space (there is no information there, and leaving NaN would poison every
    query that lands nearby), and queries outside the binned range are clamped
    to the edge rather than extrapolated -- an error model has no business
    extrapolating a dispersion.
    """
    from scipy.interpolate import RegularGridInterpolator, griddata

    nvar = len(list_var_names)
    sub = df[df["nd"] == nvar] if "nd" in df.columns else df
    bins = df.attrs.get("bins")
    if bins is None:
        raise ValueError("df must come from nd_binning (missing .attrs['bins'])")
    centres = [0.5 * (bins[n][:-1] + bins[n][1:]) for n in list_var_names]
    shape = tuple(c.size for c in centres)
    grid = np.full(shape, np.nan)
    for _, r in sub.iterrows():
        val = r[statistic]
        if not np.isfinite(val) or ("count" in r and r["count"] < min_count):
            continue
        pos = tuple(int(np.argmin(np.abs(centres[k] - r[list_var_names[k] + "_mid"])))
                    for k in range(nvar))
        grid[pos] = val
    if not np.isfinite(grid).any():
        raise ValueError("no populated bins with the requested statistic")
    if not np.isfinite(grid).all():                     # nearest-neighbour fill
        mesh = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
        pts = np.column_stack([m.ravel() for m in mesh])
        good = np.isfinite(grid.ravel())
        grid = griddata(pts[good], grid.ravel()[good], pts, method="nearest").reshape(shape)
    itp = RegularGridInterpolator(centres, grid, method="linear",
                                  bounds_error=False, fill_value=None)

    def sigma(*args):
        q = [np.clip(np.asarray(a, float), c.min(), c.max()) for a, c in zip(args, centres)]
        out = itp(np.column_stack([a.ravel() for a in q]))
        return out.reshape(np.asarray(args[0]).shape)

    sigma.grid = grid
    sigma.centres = centres
    return sigma


def two_step_standardization(dvalues, list_var, unscaled_sigma, *, out_scale=False):
    r"""Standardise, then rescale so the result has unit NMAD.

    The binned :math:`\sigma` is a dispersion in each cell, but dividing by it
    does not automatically give a unit-variance variable (bins are coarse and
    the statistic is robust). The second step multiplies :math:`\sigma` by a
    single factor so that :math:`z = dh/\sigma` has ``nmad(z) == 1`` -- after
    which the variogram of ``z`` is a pure correlation structure and the sill
    is dimensionless.
    """
    v = np.asarray(dvalues, float)
    s0 = unscaled_sigma(*list_var)
    with np.errstate(divide="ignore", invalid="ignore"):
        z0 = v / s0
    scale = nmad(z0)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("standardisation failed: non-finite scale")

    def sigma_scaled(*args):
        return scale * unscaled_sigma(*args)

    # Carry the binning introspection through the rescaling: callers reach for
    # .grid/.centres to inspect or plot the error model, and silently dropping
    # them on the calibrated function (the one you actually use) is a trap.
    # .grid is the CALIBRATED grid -- same units as what sigma_scaled() returns,
    # so plotting it against those values is meaningful; the raw binned
    # dispersion stays reachable, unscaled, as .unscaled.grid.
    _grid = getattr(unscaled_sigma, "grid", None)
    sigma_scaled.grid = None if _grid is None else scale * np.asarray(_grid, float)
    sigma_scaled.centres = getattr(unscaled_sigma, "centres", None)
    sigma_scaled.scale = scale
    sigma_scaled.unscaled = unscaled_sigma

    z = z0 / scale
    return (z, sigma_scaled, scale) if out_scale else (z, sigma_scaled)


def infer_heteroscedasticity_from_stable(dvalues, list_var, list_var_names, *,
                                         list_var_bins=None, min_count: int = 30):
    """Fit ``sigma(vars)`` on stable-terrain differences and standardise.

    Returns ``(z, sigma_fun, df, scale)``: the standardised values (unit NMAD,
    ready for :func:`empirical_variogram`), the calibrated error model, the
    binning table, and the second-step scale factor.
    """
    df = nd_binning(dvalues, list_var, list_var_names,
                    list_var_bins=list_var_bins, min_count=min_count)
    s0 = interp_nd_binning(df, list_var_names, statistic="nmad", min_count=min_count)
    z, sigma_fun, scale = two_step_standardization(dvalues, list_var, s0, out_scale=True)
    return z, sigma_fun, df, scale


def empirical_variogram(
    east: np.ndarray,
    north: np.ndarray,
    values: np.ndarray,
    *,
    bin_edges: np.ndarray | None = None,
    n_bins: int = 20,
    max_lag: float | None = None,
    estimator: str = "dowd",
    n_subsample: int = 2000,
    n_draws: int = 10,
    min_pairs: int = 30,
    seed: int = 0,
):
    r"""Empirical variogram of scattered data by repeated random subsampling.

    All-pairs is :math:`O(N^2)`; instead this draws ``n_draws`` independent
    subsets of ``n_subsample`` points, forms every pair within each subset, and
    pools the bins. That keeps memory bounded while covering short and long
    lags in the same pass (a single subsample would under-sample the short
    lags, a pure random-pair draw would under-sample them badly).

    ``counts`` is the number of DISTINCT point pairs each bin was estimated
    from -- what ``min_pairs`` gates on and what :func:`fit_variogram` weights
    by -- for every relationship between ``n_subsample``, ``n_draws`` and the
    cloud size. Draws overlap (completely, once ``n_subsample`` reaches the
    cloud size), so pairs formed more than once are carried once.
    ``n_pairs_pooled`` is what ``n_draws`` passes pool before deduplication,
    and its ratio to ``counts`` is how much the draws repeat themselves. On an
    EXHAUSTIVE subsample only one pass is actually run -- repeats provably add
    no distinct pair -- and that total is scaled by ``n_draws`` rather than
    counted, so ``n_pairs_pooled`` and ``n_draws_used`` describe the pooling
    the parameters ask for, not work performed.

    Returns
    -------
    dict
        ``lags`` (bin centres, metres), ``gamma``, ``counts`` (distinct pairs),
        ``n_pairs_pooled``, ``bin_edges``, ``variance`` (of the pooled sample),
        ``estimator``, ``n_draws_used``.
    """
    if estimator not in ("matheron", "dowd"):
        raise ValueError(f'estimator must be "matheron" or "dowd", got {estimator!r}')
    e = np.asarray(east, float)
    n = np.asarray(north, float)
    v = np.asarray(values, float)
    ok = np.isfinite(e) & np.isfinite(n) & np.isfinite(v)
    e, n, v = e[ok], n[ok], v[ok]
    if e.size < 10:
        raise ValueError(f"need >=10 finite points, got {e.size}")

    if bin_edges is None:
        if max_lag is None:
            span = float(np.hypot(e.max() - e.min(), n.max() - n.min()))
            max_lag = 0.5 * span
        lo = max(max_lag / 500.0, 1e-6)
        bin_edges = np.concatenate([[0.0], np.geomspace(lo, max_lag, n_bins)])
    bin_edges = np.asarray(bin_edges, float)
    nb = bin_edges.size - 1

    rng = np.random.default_rng(seed)
    m = min(n_subsample, e.size)
    n_pass = max(1, n_draws)
    # Draws overlap: a pair can be formed by more than one of them, and the
    # closer m is to the cloud size the likelier that is (at m = N every pass
    # re-forms the identical set). Pooling the repeats would make `counts` a
    # replication count rather than a pair count, letting a bin holding 3
    # distinct pairs clear min_pairs=30 and then be weighted as if it held 30.
    # So each pair is carried once, keyed by its ORIGINAL index pair; the sign
    # of the kept difference is arbitrary, which is immaterial to gamma (it
    # squares or takes |.|).
    exhaustive = m >= e.size
    # Pair ids exist only to deduplicate overlapping draws. A single pass, or an
    # exhaustive one (whose repeats are skipped below), has no overlap to
    # remove, so building them there would retain an int64 array per pair that
    # is never read -- doubling the accumulator for nothing.
    dedup = n_pass > 1 and not exhaustive
    pid_acc: list[list[np.ndarray]] = [[] for _ in range(nb)]
    dv_acc: list[list[np.ndarray]] = [[] for _ in range(nb)]
    n_pooled = np.zeros(nb, dtype=np.int64)
    # An exhaustive draw re-forms the identical pair set every pass, so one
    # pass already holds every distinct pair and the repeats only scale the
    # pooled total. Otherwise append the raw chunks and deduplicate ONCE per
    # bin below: sorting each bin a single time is O(K log K), where merging
    # into the accumulator every pass would re-sort what is already there.
    for _ in range(1 if exhaustive else n_pass):
        idx = rng.choice(e.size, m, replace=False) if m < e.size else np.arange(e.size)
        ee, nn, vv = e[idx], n[idx], v[idx]
        iu, ju = np.triu_indices(m, k=1)
        d = np.hypot(ee[iu] - ee[ju], nn[iu] - nn[ju])
        dv = vv[iu] - vv[ju]
        if dedup:
            gi = idx[iu].astype(np.int64)
            gj = idx[ju].astype(np.int64)
            pid = np.minimum(gi, gj) * np.int64(e.size) + np.maximum(gi, gj)
        which = np.digitize(d, bin_edges) - 1
        good = (which >= 0) & (which < nb)
        for b in np.unique(which[good]):
            sel = good & (which == b)
            n_pooled[b] += int(sel.sum())
            if dedup:
                pid_acc[b].append(pid[sel])
            dv_acc[b].append(dv[sel])
    if exhaustive:
        n_pooled = n_pooled * n_pass

    lags = np.full(nb, np.nan)
    gamma = np.full(nb, np.nan)
    counts = np.zeros(nb, dtype=int)
    for b in range(nb):
        if not dv_acc[b]:
            continue
        dv = np.concatenate(dv_acc[b])
        if dedup:
            _, first = np.unique(np.concatenate(pid_acc[b]), return_index=True)
            dv = dv[first]
        pid_acc[b] = dv_acc[b] = []
        counts[b] = dv.size
        if dv.size < min_pairs:
            continue
        lags[b] = 0.5 * (bin_edges[b] + bin_edges[b + 1])
        if estimator == "matheron":
            gamma[b] = 0.5 * float(np.mean(dv ** 2))
        else:
            gamma[b] = 0.5 * (_MAD_TO_SIGMA * float(np.median(np.abs(dv)))) ** 2
    keep = np.isfinite(gamma)
    return dict(lags=lags[keep], gamma=gamma[keep], counts=counts[keep],
                n_pairs_pooled=n_pooled[keep],
                bin_edges=bin_edges, variance=float(np.var(v)), estimator=estimator,
                n_draws_used=int(n_pass))


def variogram_model(h: np.ndarray, model: str, sill: float, rng_: float) -> np.ndarray:
    """One variogram model evaluated at lags ``h``.

    ``rng_`` is the **effective** range in metres (the standard geostatistical
    convention -- see the module docstring): the e-folding scale used inside
    each model is ``a = r`` for ``spherical``, ``r/3`` for ``exponential`` and
    ``r/2`` for ``gaussian``.
    """
    h = np.asarray(h, float)
    if model == "nugget":
        return np.where(h > 0, sill, 0.0)
    r = max(float(rng_), 1e-12)
    if model == "spherical":
        t = np.clip(h / r, 0.0, 1.0)
        return sill * np.where(h < r, 1.5 * t - 0.5 * t ** 3, 1.0)
    if model == "exponential":
        return sill * (1.0 - np.exp(-3.0 * h / r))
    if model == "gaussian":
        return sill * (1.0 - np.exp(-((2.0 * h / r) ** 2)))
    raise ValueError(f"unknown model {model!r}")


def _eval_sum(h, params):
    out = np.zeros_like(np.asarray(h, float))
    for model, sill, rng_ in params:
        out = out + variogram_model(h, model, sill, rng_)
    return out


def fit_variogram(
    lags: np.ndarray,
    gamma: np.ndarray,
    *,
    models: tuple[str, ...] = ("nugget", "spherical", "spherical"),
    counts: np.ndarray | None = None,
    max_range: float | None = None,
    seed: int = 0,
):
    r"""Weighted fit of a **sum** of variogram models.

    Real DEM error is multi-scale, so a single range fits badly; the default is
    a nugget plus two spherical structures. The per-bin RESIDUAL weight is
    ``sqrt(counts)/lag``; ``least_squares`` squares it, so the effective COST
    weight is :math:`\mathrm{counts}/h^2` -- pair count for precision, and
    :math:`1/h^2` because the short lags are where a correlation length is
    actually determined and a plain count weighting lets the long lags
    dominate. Quote the cost weighting, not the residual one, when reasoning
    about why a fitted range moves (the ``max_lag`` sweep in the module
    docstring turns on exactly this).

    Returns ``dict(params=[(model, sill, range), ...], total_sill, ranges,
    rmse, r2, n_bins)``; ``params`` feeds
    :func:`covariance_from_variogram` and :func:`number_effective_samples`.
    Every range is the **effective** range (module docstring), so it is
    directly comparable with a published DEM-error range.
    """
    from scipy.optimize import least_squares

    h = np.asarray(lags, float)
    g = np.asarray(gamma, float)
    ok = np.isfinite(h) & np.isfinite(g) & (h > 0)
    h, g = h[ok], g[ok]
    if h.size < 3:
        raise ValueError(f"need >=3 usable bins, got {h.size}")
    w = np.ones_like(h)
    if counts is not None:
        c = np.asarray(counts, float)[ok]
        w = np.sqrt(np.clip(c, 1, None)) / h
    w = w / np.max(w)

    hmax = float(h.max()) if max_range is None else float(max_range)
    n_struct = sum(1 for m in models if m != "nugget")
    guess, lo, hi = [], [], []
    for k, m in enumerate(models):
        guess.append(max(g.max(), 1e-12) / len(models))
        lo.append(0.0)
        hi.append(10.0 * max(g.max(), 1e-12))
        if m != "nugget":
            j = sum(1 for mm in models[:k] if mm != "nugget")
            guess.append(hmax * (0.05 if n_struct > 1 and j == 0 else 0.5))
            lo.append(1e-3)
            hi.append(5.0 * hmax)

    def _unpack(p):
        out, i = [], 0
        for m in models:
            sill = p[i]
            i += 1
            if m == "nugget":
                out.append((m, sill, 0.0))
            else:
                out.append((m, sill, p[i]))
                i += 1
        return out

    def _resid(p):
        return w * (_eval_sum(h, _unpack(p)) - g)

    best = None
    rng = np.random.default_rng(seed)
    for attempt in range(4):        # multistart: the sum-of-ranges fit is multimodal
        p0 = np.array(guess, float) if attempt == 0 else np.array(guess, float) * rng.uniform(0.2, 3.0, len(guess))
        p0 = np.clip(p0, np.array(lo) + 1e-9, np.array(hi))
        try:
            r = least_squares(_resid, p0, bounds=(lo, hi), max_nfev=8000)
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        raise RuntimeError("variogram fit failed to start")

    params = _unpack(best.x)
    pred = _eval_sum(h, params)
    ss_res = float(np.sum((pred - g) ** 2))
    ss_tot = float(np.sum((g - g.mean()) ** 2))
    return dict(params=params,
                total_sill=float(sum(p[1] for p in params)),
                ranges=[p[2] for p in params if p[0] != "nugget"],
                rmse=float(np.sqrt(np.mean((pred - g) ** 2))),
                r2=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
                n_bins=int(h.size))


def covariance_from_variogram(h, params):
    r""":math:`C(h)=\sigma^2_{\rm tot}-\gamma(h)`."""
    total = sum(p[1] for p in params)
    return total - _eval_sum(h, params)


def number_effective_samples(
    east: np.ndarray,
    north: np.ndarray,
    params,
    *,
    n_subsample: int = 1500,
    n_draws: int = 5,
    seed: int = 0,
) -> dict:
    r"""Effective independent sample count of an area mean, by double sum.

    .. math::
        \operatorname{Var}(\bar z) = \frac{1}{N^2}\sum_i\sum_j C(h_{ij}),
        \qquad n_{\rm eff} = \sigma^2_{\rm tot}/\operatorname{Var}(\bar z).

    ``offdiag_draws`` holds one entry per pass. When ``n_subsample >= N`` a
    single pass is made: every draw would then be the identical point set, so
    repeating it cannot vary ``mean_off`` and would only fill
    ``offdiag_draws`` with copies that read as a Monte-Carlo spread. Evaluated on random subsets of the domain's own geometry, so it
    needs no shape idealisation, but reported for the FULL set of points
    passed in:
    the subsampling estimates the mean off-diagonal covariance only, and the
    diagonal is applied exactly. ``n_eff`` -> the point count for white noise,
    -> ~1 when the field is correlated across the whole area.

    Two boundary cases, which are NOT the same thing and do not share a path:

    * ``N < 2`` -- the region genuinely has no off-diagonal pair, so the double
      sum is exactly :math:`C(0)`. Returns ``var_mean = total_sill``,
      ``n_eff = 1``, ``mean_offdiag_cov = 0.0`` (the sum over an empty set) and
      an empty ``offdiag_draws``, without sampling anything.
    * ``N >= 2`` but ``n_subsample < 2`` -- pairs exist, but the caller has
      asked for a subsample too small to form one. That is a usage error and
      raises: averaging no pairs would report ``mean_off = 0``, i.e. the
      WHITE-NOISE answer, for a field that may be correlated across the whole
      domain.

    Raises ``ValueError`` if no point has finite coordinates -- an area mean of
    nothing has no error bar.
    """
    e = np.asarray(east, float)
    n = np.asarray(north, float)
    ok = np.isfinite(e) & np.isfinite(n)
    e, n = e[ok], n[ok]
    N = int(e.size)
    if N == 0:
        raise ValueError(
            "number_effective_samples needs at least one point with finite "
            f"coordinates; got {np.asarray(east).size} point(s), none finite")
    total = float(sum(p[1] for p in params))

    # A one-point region has no off-diagonal pair to estimate, so the double
    # sum is exact and trivial. Handled here, before any sampling, so that the
    # estimator below only ever runs where pairs actually exist.
    if N < 2:
        return dict(n_eff=1.0, var_mean=total, total_sill=total,
                    mean_offdiag_cov=0.0, n_points=N, offdiag_draws=[])

    # Past this point pairs DO exist, so every draw must produce some. A
    # subsample of fewer than two points cannot form one at all: that is the
    # caller asking for the impossible, not a degenerate region, and it must
    # not fall through to an empty average -- mean_off = 0 is the white-noise
    # answer, which for a correlated field understates the error without limit.
    if n_subsample < 2:
        raise ValueError(
            f"n_subsample must be >= 2 to estimate a pair covariance, got "
            f"{n_subsample} for {N} points")

    rng = np.random.default_rng(seed)
    # Estimate the mean OFF-DIAGONAL covariance from subsamples and combine it
    # with the exact diagonal for the full N. Averaging the whole subsample
    # matrix instead would put N_sub diagonal terms into an N-point average,
    # making n_eff come out as the SUBSAMPLE size -- correct for the subsample,
    # silently wrong for the area the caller asked about.
    offs = []
    m = min(n_subsample, N)
    # With m == N every draw is the identical point set: extra passes would
    # leave mean_off untouched while filling offdiag_draws with copies that
    # read as a Monte-Carlo spread. (empirical_variogram needs no such rule --
    # it deduplicates pairs instead, which this estimator has no use for since
    # it averages whole draws rather than pooling them.)
    for _ in range(1 if m >= N else max(1, n_draws)):
        idx = rng.choice(N, m, replace=False) if m < N else np.arange(N)
        ee, nn = e[idx], n[idx]
        d = np.hypot(ee[:, None] - ee[None, :], nn[:, None] - nn[None, :])
        C = covariance_from_variogram(d, params)
        iu = np.triu_indices(m, k=1)
        offs.append(float(np.mean(C[iu])))
    mean_off = float(np.mean(offs))
    var_mean = total / N + (1.0 - 1.0 / N) * mean_off
    neff = total / var_mean if var_mean > 0 else np.inf
    return dict(n_eff=float(neff), var_mean=float(var_mean), total_sill=total,
                mean_offdiag_cov=mean_off, n_points=N, offdiag_draws=offs)


def standard_error_of_mean(east, north, params, **kw) -> float:
    """Standard error of the mean over these points, honouring correlation."""
    return float(np.sqrt(number_effective_samples(east, north, params, **kw)["var_mean"]))

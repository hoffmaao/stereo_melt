"""Coverage-gap stress test for the closed-form Stubblefield inverse.

Premise from diagnose_linear_inverse_signs.py: on dense, noise-free data,
``inverse_stationary`` recovers the input m exactly (sign + DC + shape).

Hypothesis: real REMA stacks have per-pixel coverage gaps (each strip is
patchy; some pixels see <50% of epochs), and the wrapper handles gaps with
``fillna(0)`` or Gaussian infill before the FFT/DCT. This test pipes the
*same* known forcing through *the same* sparse-coverage prep as the wrapper
does and reports whether the recovered m loses amplitude, flips sign, or
biases its spatial mean.

Three coverage regimes per case:
  - ``dense``    : every pixel observed at every epoch (baseline)
  - ``fillna0``  : ~50% of pixels NaN per epoch, NaNs → 0 before inverse
  - ``infill``   : same NaN pattern, Gaussian-weighted infill before inverse

Forcings:
  - uniform +1 m/yr melt (tests DC recovery under coverage gaps)
  - Gaussian +5 m/yr melt blob, sigma=1 km (tests localized recovery)

Outputs per (case, regime):
  - spatial mean of recovered m
  - peak value at the forcing center
  - RMSE vs truth
  - sign at center (+ = melt, - = accretion)
"""
from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import numpy as np
import xarray as xr

from stereo_melt.dynamics.linear_perturbation import forward, inverse_stationary
from stereo_melt.dynamics.lagrangian_inverse import _nan_aware_gaussian_infill_2d


# ---------------------------------------------------------------------------
# Domain identical to diagnose_linear_inverse_signs.py for direct comparison
# ---------------------------------------------------------------------------
NX = NY = 256
DX = DY = 100.0
H_REF = 500.0
ETA = 1e14
N_EPOCHS = 5

x = (np.arange(NX) - NX // 2) * DX
y = (np.arange(NY) - NY // 2) * DY
xx, yy = np.meshgrid(x, y)
r = np.sqrt(xx ** 2 + yy ** 2)
times = np.arange(N_EPOCHS).astype("datetime64[Y]").astype("datetime64[ns]")


def make_strip_masks(n_epochs: int, ny: int, nx: int, seed: int = 7) -> np.ndarray:
    """Return (n_epochs, ny, nx) bool array — True where pixel is OBSERVED.

    Mimics REMA-like strip geometry: each epoch covers a random ~half of the
    domain via 2-3 elongated rectangular strips at random orientations. Per-
    pixel coverage averages ~55% over n_epochs (close to Beardmore's median
    8/14 = 57% floating-pixel coverage).
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros((n_epochs, ny, nx), dtype=bool)
    yy_pix, xx_pix = np.mgrid[0:ny, 0:nx]
    for i in range(n_epochs):
        # 2-3 strips per epoch, each a wide diagonal band
        n_strips = rng.integers(2, 4)
        epoch_mask = np.zeros((ny, nx), dtype=bool)
        for _ in range(n_strips):
            # Random angle and offset
            angle = rng.uniform(0, np.pi)
            cx_pix = rng.uniform(0, nx)
            cy_pix = rng.uniform(0, ny)
            half_width_pix = rng.uniform(40, 70)  # strip half-width
            # Distance from each pixel to a line through (cx, cy) at angle
            nrm = (xx_pix - cx_pix) * np.sin(angle) - (yy_pix - cy_pix) * np.cos(angle)
            epoch_mask |= np.abs(nrm) <= half_width_pix
        mask[i] = epoch_mask
    return mask


def report(label: str, m_truth: np.ndarray, m_inv: np.ndarray) -> None:
    center_truth = m_truth[NY // 2, NX // 2]
    center_inv = m_inv[NY // 2, NX // 2]
    rmse = float(np.sqrt(np.mean((m_inv - m_truth) ** 2)))
    print(
        f"  [{label:<10s}]  "
        f"<m>={m_inv.mean():+.4f} (truth {m_truth.mean():+.4f})  "
        f"center={center_inv:+.3f} (truth {center_truth:+.3f})  "
        f"min/max=[{m_inv.min():+.2f},{m_inv.max():+.2f}]  "
        f"RMSE={rmse:.3f}  "
        f"sign@center={'MELT' if center_inv > 0 else 'ACCRETION' if center_inv < 0 else 'ZERO'}"
    )


def invert(h_da: xr.DataArray, transform: str = "fft") -> np.ndarray:
    """Per-pixel time-mean anomaly → inverse_stationary, exactly as wrapper does."""
    h_mean = h_da.mean("time", skipna=True)
    h_anom = h_da - h_mean
    m = inverse_stationary(
        h_anom, H=H_REF, eta_bar=ETA, alpha=0.0, alpha_y=0.0, gamma=0.0,
        reg=1e-3, reference="time_mean", transform=transform,
    )
    return m.values


def run_case(label: str, m_truth: np.ndarray, transform: str = "fft") -> None:
    print(f"\n==== {label}  (transform={transform}) ====")
    m_da = xr.DataArray(
        m_truth, dims=("y", "x"), coords={"y": y, "x": x}, name="m",
    )
    h = forward(
        m_da, H=H_REF, eta_bar=ETA, alpha=0.0, alpha_y=0.0, gamma=0.0,
        times=xr.DataArray(times, dims="time"), stationary=True,
    )

    # Dense baseline
    m_dense = invert(h, transform=transform)
    report("dense", m_truth, m_dense)

    # Sparse coverage masks
    obs_mask = make_strip_masks(N_EPOCHS, NY, NX, seed=7)
    cov_per_epoch = obs_mask.mean(axis=(1, 2))
    cov_per_pixel = obs_mask.mean(axis=0)
    print(f"  coverage: per-epoch mean={cov_per_epoch.mean():.2f}  "
          f"per-pixel mean={cov_per_pixel.mean():.2f}  min={cov_per_pixel.min():.2f}")

    h_sparse_vals = h.values.copy()
    h_sparse_vals[~obs_mask] = np.nan
    h_sparse = xr.DataArray(h_sparse_vals, dims=h.dims, coords=h.coords)

    # fillna(0) regime — mimics boundary_fix="off"
    h_fillna = h_sparse.fillna(0.0)
    m_fillna = invert(h_fillna, transform=transform)
    report("fillna0", m_truth, m_fillna)

    # Per-pixel mean from sparse, then fillna(0) on the anomaly itself
    # (matches wrapper sequence: subtract per-pixel mean BEFORE fillna)
    h_sparse_mean = h_sparse.mean("time", skipna=True)
    h_sparse_anom = h_sparse - h_sparse_mean
    h_sparse_anom_filled = h_sparse_anom.fillna(0.0)
    m_anom_then_fill = inverse_stationary(
        h_sparse_anom_filled, H=H_REF, eta_bar=ETA, alpha=0.0, alpha_y=0.0, gamma=0.0,
        reg=1e-3, reference="time_mean", transform=transform,
    )
    report("anom→fill0", m_truth, m_anom_then_fill.values)

    # Infill regime — mimics boundary_fix="infill"
    sigma_pix = (H_REF / DY, H_REF / DX)
    infilled = np.empty_like(h_sparse_vals)
    for ti in range(N_EPOCHS):
        infilled[ti] = _nan_aware_gaussian_infill_2d(
            h_sparse_vals[ti], sigma_pix, max_iters=8, support_threshold=0.05,
        )
    h_infill = xr.DataArray(infilled, dims=h.dims, coords=h.coords)
    m_infill = invert(h_infill, transform=transform)
    report("infill", m_truth, m_infill)


if __name__ == "__main__":
    # Case A: spatially uniform m=+1 m/yr (DC mode under coverage gaps)
    mA = np.ones((NY, NX), dtype=np.float64)
    run_case("A: uniform +1 m/yr melt", mA, transform="fft")
    run_case("A: uniform +1 m/yr melt", mA, transform="dct")

    # Case B: localized Gaussian melt
    mB = 5.0 * np.exp(-0.5 * (r / 1000.0) ** 2)
    run_case("B: Gaussian +5 m/yr blob", mB, transform="fft")
    run_case("B: Gaussian +5 m/yr blob", mB, transform="dct")

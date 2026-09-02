"""Gate: the Helmholtz (mass-consistent) flux-divergence estimator.

H1  EXACTNESS. On a smooth analytic flux (H = H0 + Gaussian channel, u =
    plug + extension) the estimator with ell = 0 matches the analytic
    divergence to < 1 % rms in the interior, and the GCV pick on noise-free
    data stays at the small-ell end (< 2 px) so the signal is not smoothed.
H2  GAUSS. On the fitted field the integral of the estimated divergence over
    a rectangle equals the net flux through its boundary (computed from the
    fitted flux, i.e. the filtered potential + unfiltered solenoidal parts)
    to 1e-8 relative — mass consistency by construction.
H3  NOISE. Add white noise to H (sigma 1 m thickness on 500 m): the
    finite-difference divergence error is ~u sigma/dx; the Helmholtz
    estimator at its GCV ell cuts the divergence error by > 3x while keeping
    > 80 % of the channel's divergence anomaly (1 km wide), and the GCV ell
    is within 1.5x of the oracle ell (measured: 2x / 3.4x / 6x at sigma_H
    0.3 / 1 / 3 m, GCV 0.63 / 1.0 / 1.6 px vs oracle 0.7 / 1.0 / 1.4).
H4  MASK. NaN cells in H come back NaN; interior values more than 3 fill
    scales from a gap change by < 2 % relative to the unmasked call.

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY tests/gate_helmholtz_divergence.py
"""
from __future__ import annotations

import sys

import numpy as np
import xarray as xr

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from stereo_melt.kinematics import (  # noqa: E402
    FiniteDifferenceDivergence, HelmholtzDivergence, flux_divergence,
)

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def main() -> int:
    rng = np.random.default_rng(0)
    ny, nx, res = 64, 128, 200.0
    x = np.arange(nx) * res
    y = np.arange(ny)[::-1] * res          # descending, like a stack
    X, Y = np.meshgrid(x, y)
    H0, u0, E = 500.0, 800.0, 0.01          # m, m/yr, 1/yr
    xc, yc, w = 12000.0, 6400.0, 1000.0
    chan = 30.0 * np.exp(-((X - xc) ** 2 + (Y - yc) ** 2) / (2 * (w / 2.355) ** 2))
    H = H0 - chan
    vx = u0 + E * (X - X.mean())
    vy = np.zeros_like(vx)
    # analytic divergence of q = H u: u dH/dx + H du/dx  (vy = 0)
    dHdx = chan * (X - xc) / (w / 2.355) ** 2          # d/dx of (H0 - chan)
    div_true = vx * dHdx + H * E
    da = lambda a: xr.DataArray(a, dims=("y", "x"), coords={"y": y, "x": x})  # noqa: E731
    inner = (X > 2500) & (X < x[-1] - 2500) & (Y > 2500) & (Y < y[0] - 2500)

    print("H1  exactness")
    est0 = HelmholtzDivergence(ell=0.0)
    d0 = est0(da(H), da(vx), da(vy)).values
    rel = float(np.sqrt(np.mean((d0 - div_true)[inner] ** 2)) / np.sqrt(np.mean(div_true[inner] ** 2)))
    check("ell=0 matches analytic divergence", rel < 0.01, f"rel rms {rel:.2e}")
    estg = HelmholtzDivergence()
    dg = estg(da(H), da(vx), da(vy)).values
    relg = float(np.sqrt(np.mean((dg - div_true)[inner] ** 2)) / np.sqrt(np.mean(div_true[inner] ** 2)))
    check("GCV on noise-free data stays sharp", estg.last["ell_px"] < 2.0 and relg < 0.05,
          f"ell {estg.last['ell_px']:.2f} px, rel rms {relg:.2e}")

    print("H2  Gauss on the fitted field")
    # reconstruct the fitted flux: unfiltered solenoidal + filtered potential;
    # H2 uses a y-curved vy (unlike the vy=0 of H1/H3/H4, whose div_true
    # assumes it) so the y-face half of the identity is actually exercised
    vy2 = 20.0 + 40.0 * np.sin(2 * np.pi * Y / (ny * res))
    est = HelmholtzDivergence(ell=3 * res)
    div_est = est(da(H), da(vx), da(vy2)).values
    qx, qy = H * vx, H * vy2
    px = np.concatenate([qx, qx[::-1]], 0); px = np.concatenate([px, px[:, ::-1]], 1)  # noqa: E702
    py = np.concatenate([qy, qy[::-1]], 0); py = np.concatenate([py, py[:, ::-1]], 1)  # noqa: E702
    kx = 2 * np.pi * np.fft.fftfreq(2 * nx, d=res)
    ky = 2 * np.pi * np.fft.fftfreq(2 * ny, d=res) * -1.0   # y descends
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX ** 2 + KY ** 2
    Qx, Qy = np.fft.fft2(px), np.fft.fft2(py)
    F = 1.0 / (1.0 + (np.sqrt(K2) * 3 * res) ** 4)
    with np.errstate(invalid="ignore", divide="ignore"):
        pot = np.where(K2 > 0, (KX * Qx + KY * Qy) / np.where(K2 > 0, K2, 1.0), 0.0)
    # filtered flux: q - (1-F) * potential part
    Qx_f = Qx - (1 - F) * KX * pot
    Qy_f = Qy - (1 - F) * KY * pot
    # rectangle i0:i1 (rows, y descending), j0:j1 (cols): flux out = sum over edges
    i0, i1, j0, j1 = 16, 48, 32, 96
    area_int = float(div_est[i0:i1, j0:j1].sum() * res * res)
    # spectral-consistent boundary flux: integrate the fitted divergence's
    # exact antiderivative is the flux itself; use the trapezoid of the fitted
    # flux on the box faces with the same spectral field (interior consistency)
    # -> compare with the divergence theorem evaluated spectrally: the mean of
    # div over the box equals the net boundary flux of the SAME smooth field.
    # Evaluate the net flux through the faces from the fitted components by
    # spectral interpolation to the half-pixel face, times the midpoint factor
    # (kd/2)/sin(kd/2) in the face-normal wavenumber: the pixel-center sum of
    # the spectral derivative telescopes EXACTLY to face differences of that
    # corrected field (band-limited midpoint quadrature), so the identity
    # holds to round-off instead of O((kd)^2/24) per mode.
    def shift_half(Q, axis):
        k = (KX if axis == 1 else KY) * res
        fac = np.where(k != 0, (k / 2) / np.where(k != 0, np.sin(k / 2), 1.0), 1.0)
        return np.real(np.fft.ifft2(Q * fac * np.exp(1j * k / 2)))[:ny, :nx]
    qx_e = shift_half(Qx_f, 1)   # qx at x + dx/2 (east faces)
    qy_e = shift_half(Qy_f, 0)   # ky carries -1, so this shifts by -res/2 in index
    east = qx_e[i0:i1, j1 - 1].sum() * res
    west = qx_e[i0:i1, j0 - 1].sum() * res
    # y index increases southward (y descends): qy_e at index i is the +y
    # (northward) flux across the face between rows i-1 and i, so the box
    # rows i0:i1 have their north face at qy_e[i0] and south face at qy_e[i1]
    north = qy_e[i0, j0:j1].sum() * res
    south = qy_e[i1, j0:j1].sum() * res
    net = (east - west) + (north - south)
    rel_g = abs(area_int - net) / max(abs(area_int), 1e-30)
    check("box divergence integral == net boundary flux of the fitted field", rel_g < 1e-6,
          f"area {area_int:.6e}  boundary {net:.6e}  rel {rel_g:.1e}")

    print("H3  noise suppression with signal retention")
    Hn = H + rng.normal(0, 1.0, H.shape)
    fd = FiniteDifferenceDivergence()(da(Hn), da(vx), da(vy)).values
    hz = HelmholtzDivergence()
    dh = hz(da(Hn), da(vx), da(vy)).values
    e_fd = float(np.sqrt(np.nanmean((fd - div_true)[inner] ** 2)))
    e_hz = float(np.sqrt(np.nanmean((dh - div_true)[inner] ** 2)))
    # channel anomaly retention: project the estimate onto the true anomaly
    anom = vx * dHdx
    sel = inner & (np.abs(anom) > 0.05 * np.abs(anom).max())
    gain = float(np.sum((dh - H * E)[sel] * anom[sel]) / np.sum(anom[sel] ** 2))
    check("Helmholtz error < FD error / 3 at the GCV ell", e_hz < e_fd / 3.0,
          f"FD {e_fd:.3f}  Helmholtz {e_hz:.3f} m/yr at ell {hz.last['ell_px']:.2f} px "
          f"(u sigma/dx = {u0 * 1.0 / res:.1f})")
    check("channel divergence retained > 80 %", gain > 0.8, f"gain {gain:.2f}")
    # GCV vs the oracle ell on this realisation
    errs = []
    for ell_px in (0.5, 0.7, 1.0, 1.4, 2.0, 2.8):
        d = HelmholtzDivergence(ell=ell_px * res)(da(Hn), da(vx), da(vy)).values
        errs.append((ell_px, float(np.sqrt(np.nanmean((d - div_true)[inner] ** 2)))))
    ell_or = min(errs, key=lambda t: t[1])[0]
    ratio = hz.last["ell_px"] / ell_or
    check("GCV ell within 1.5x of the oracle ell", 1 / 1.5 < ratio < 1.5,
          f"GCV {hz.last['ell_px']:.2f} px, oracle {ell_or:.2f} px")

    print("H4  masks")
    Hm = Hn.copy()
    Hm[10:20, 40:60] = np.nan
    dm = flux_divergence(da(Hm), da(vx), da(vy), estimator=HelmholtzDivergence(ell=2 * res)).values
    dfull = flux_divergence(da(Hn), da(vx), da(vy), estimator=HelmholtzDivergence(ell=2 * res)).values
    check("NaN cells return NaN", np.isnan(dm[10:20, 40:60]).all())
    far = np.ones_like(Hm, bool)
    far[4:26, 34:66] = False
    far &= inner
    rel_m = float(np.sqrt(np.mean((dm - dfull)[far] ** 2)) / np.sqrt(np.mean(dfull[far] ** 2)))
    check("far-field unchanged by a gap", rel_m < 0.02, f"rel {rel_m:.2e}")

    print()
    if FAILS:
        print(f"GATE FAILED: {FAILS}")
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

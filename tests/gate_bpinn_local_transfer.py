"""Gate: the B-PINN's LOCAL (per-geometry-bin, blended) bridging observation operator.

``fit_bpinn(BPINNConfig(transfer=True, n_bins > 1))`` bins the domain by
``(H, u_x, u_y)`` (:func:`stereo_melt.dynamics.geometry_bins.geometry_bins`, the
same deterministic clustering and partition-of-unity blend as the monolithic
solver's ``n_bins``), builds one observation operator per bin
(:func:`stereo_melt.dynamics.bpinn.transfer_observation_operator`, the physical
flow's y component mirrored into the FFT lattice because ``y`` descends) and
blends the per-bin responses (:func:`stereo_melt.dynamics.bpinn.apply_blended_operator`).
Rungs, on the numpy reference implementation the JAX path mirrors line for line:

B1  REDUCTION. One bin reproduces the global operator exactly, and uniform
    geometry asked for 4 bins collapses to one (max |diff| == 0).
B2  IT VARIES. A shelf whose left half flows +x and right half flows -y gets
    two bins whose interiors (beyond the blend width) equal their own global
    operator EXACTLY, while the two operators differ by a real margin.
B3  ORIENTATION. Rotating the field and the flow by 90 degrees (square
    pixels) rotates the response: the residual with the correctly mirrored
    flow is at least 10x smaller than with the y sign flipped.
B4  PARTITION OF UNITY. The bin weights sum to one everywhere, and every
    valid cell's own bin carries the largest weight far from the seams.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stereo_melt.dynamics.bpinn import apply_blended_operator, transfer_observation_operator  # noqa: E402
from stereo_melt.dynamics.geometry_bins import geometry_bins  # noqa: E402

ny, nx, dx = 128, 160, 100.0
H0, eta, alpha, bg = 400.0, 1e13, 0.34, 3.0
X, Y = np.meshgrid(np.arange(nx) * dx, -np.arange(ny) * dx)   # y descends, as on the stacks
rng = np.random.default_rng(0)
lam = 2 * H0
field = H0 + 20 * np.cos(2 * np.pi * X / lam) + 20 * np.cos(2 * np.pi * Y / lam) + rng.normal(0, 1, (ny, nx))
dom = np.ones((ny, nx), bool)
Hm = np.full((ny, nx), H0)


def M_for(Hb, ux, uy, ny_=ny, nx_=nx):
    return transfer_observation_operator(ny_, nx_, dx, dx, Hb, ux, uy, eta, alpha, bg)[1]


def local(H, vx, vy, n_bins, blend_px, f):
    geom, W = geometry_bins(H, vx, vy, dom, n_bins, blend_px)
    M = np.stack([M_for(*g) for g in geom])
    return apply_blended_operator(M, W, f), geom, W


# B1
vx, vy = np.full((ny, nx), 1500.0), np.full((ny, nx), -800.0)
ref = apply_blended_operator(M_for(H0, 1500.0, -800.0), None, field)
out1, geom1, _ = local(Hm, vx, vy, 1, 8, field)
out4, geom4, _ = local(Hm, vx, vy, 4, 8, field)
d1, d4 = np.abs(out1 - ref).max(), np.abs(out4 - ref).max()
assert len(geom1) == 1 and len(geom4) == 1, (len(geom1), len(geom4))
assert d1 == 0.0 and d4 == 0.0, (d1, d4)
print(f"B1 PASS  one bin == global (max|diff| {d1:.1e}); uniform geometry with n_bins=4 collapses to 1 bin ({d4:.1e})")

# B2
left = X < nx * dx / 2
vx2, vy2 = np.where(left, 1500.0, 0.0), np.where(left, 0.0, -1500.0)
out, geom2, W2 = local(Hm, vx2, vy2, 2, 4, field)
refL = apply_blended_operator(M_for(H0, 1500.0, 0.0), None, field)
refR = apply_blended_operator(M_for(H0, 0.0, -1500.0), None, field)
sl = (slice(20, -20), slice(20, nx // 2 - 20))
sr = (slice(20, -20), slice(nx // 2 + 20, -20))
eL, eR = np.abs(out[sl] - refL[sl]).max(), np.abs(out[sr] - refR[sr]).max()
gap = np.sqrt(np.mean((refL - refR)[sl] ** 2))
assert len(geom2) == 2, geom2
assert eL == 0.0 and eR == 0.0, (eL, eR)
assert gap > 1.0, gap
print(f"B2 PASS  two flow regimes -> 2 bins; interiors equal their own operator exactly; operators differ by rms {gap:.2f} m")

# B3
o1 = apply_blended_operator(M_for(H0, 1500.0, 0.0), None, field)
f_rot = np.rot90(field, 1)                       # physical 90-degree rotation of a y-descending raster
o_good = apply_blended_operator(M_for(H0, 0.0, 1500.0, nx, ny), None, f_rot)
o_bad = apply_blended_operator(M_for(H0, 0.0, -1500.0, nx, ny), None, f_rot)
r_good = np.abs(np.rot90(o1, 1) - o_good)[10:-10, 10:-10].max()
r_bad = np.abs(np.rot90(o1, 1) - o_bad)[10:-10, 10:-10].max()
assert r_good * 10 < r_bad, (r_good, r_bad)
print(f"B3 PASS  rotation equivariance: residual {r_good:.3f} m with the mirrored flow vs {r_bad:.3f} m with the y sign flipped")

# B4
tot = W2.sum(0)
assert np.allclose(tot, 1.0), (tot.min(), tot.max())
left_bin = 0 if geom2[0][1] > geom2[1][1] else 1        # the +x-flow bin owns the left half
own = np.where(left, left_bin, 1 - left_bin)
far = np.abs(X - nx * dx / 2) > 20 * dx
assert np.all(W2.argmax(0)[far] == own[far])
print("B4 PASS  weights sum to one; every cell far from the seam is owned by its own bin")
print("gate_bpinn_local_transfer: all rungs passed")

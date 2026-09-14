"""Gate: the B-PINN's LOCAL (per-geometry-bin, blended) bridging observation operator.

``fit_bpinn(BPINNConfig(transfer=True, n_bins > 1))`` bins the domain by
``(H, u_x, u_y)`` (:func:`stereo_melt.dynamics.geometry_bins.geometry_bins`, the
same deterministic clustering and partition-of-unity blend as the monolithic
solver's ``n_bins``), builds one observation operator per bin
(:func:`stereo_melt.dynamics.bpinn.transfer_observation_operator`, the physical
flow's y component mirrored into the FFT lattice because ``y`` descends) and
blends the per-bin responses (:func:`stereo_melt.dynamics.bpinn.apply_blended_operator`).
Which geometries get built is :func:`stereo_melt.dynamics.bpinn.local_bin_geometry`,
the same resolution ``fit_bpinn`` uses. Rungs, on the numpy reference
implementation the JAX path mirrors line for line:

B1  REDUCTION. One bin reproduces the global operator exactly, and uniform
    geometry asked for 4 bins collapses to one (max |diff| == 0) -- collapsing
    onto the SINGLE-GEOMETRY reference, not onto the bin centroid, because the
    two define the reference thickness differently (whole-stack median vs
    domain-restricted mean), so the reduction is exact and not merely close.
B2  IT VARIES. A shelf whose left half flows +x and right half flows -y gets
    two bins whose interiors (beyond the blend width) equal their own global
    operator EXACTLY, while the two operators differ by a real margin.
B3  ORIENTATION. Rotating the field and the flow by 90 degrees (square
    pixels) rotates the response: the residual with the correctly mirrored
    flow is at least 10x smaller than with the y sign flipped.
B4  PARTITION OF UNITY. The bin weights sum to one everywhere, and every
    valid cell's own bin carries the largest weight far from the seams.

B5  EFFECTIVE BINS. A field of one dominant geometry plus a small anomalous
    patch gets the patch its OWN bin, with no duplicate centroid and no
    zero-weight bin. The quantile seeding degenerates there -- every quantile
    of the principal-component projection lands on the dominant geometry -- so
    this is the farthest-point re-seeding, and the run must not report itself
    as uniform or fall back to the single reference geometry: that would hand
    back the global operator on exactly the field localisation was asked for.

B6  MUTUALLY EXCLUSIVE. ``BPINNConfig(n_bins > 1)`` with an explicit
    ``H_ref_m``/``ux_ref_myr``/``uy_ref_myr`` raises: the local bins come from
    the data, so an override could never reach an operator.

B7  PIXEL PITCH. ``transfer_observation_operator`` rejects a non-positive
    ``dx_m``/``dy_m``. A negative y step (``prepare_bpinn_data``'s signed
    ``dy``, the other same-named quantity in that module) flips ``ky`` and
    exactly cancels the operator's own y mirror, so it would otherwise build a
    y-mirrored operator silently.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stereo_melt.dynamics.bpinn import (  # noqa: E402
    BPINNConfig, apply_blended_operator, local_bin_geometry, transfer_observation_operator)
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


def local(H, vx, vy, n_bins, blend_px, f, ref_geom=None):
    geom, W, _ = local_bin_geometry(H, vx, vy, dom, n_bins, blend_px,
                                       (H0, 1500.0, -800.0) if ref_geom is None else ref_geom)
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
# The collapse must land on the SINGLE-GEOMETRY reference, which on a real window is not the
# bin centroid: data.H0 is the whole unmasked stack's median, a centroid is a domain-restricted
# mean. Ask for 4 bins on uniform geometry with a reference 90 m away from that centroid and
# the one operator built must be the reference's, exactly.
ref_off = (H0 + 90.0, 1500.0, -800.0)
ref_b = apply_blended_operator(M_for(*ref_off), None, field)
out_off, geom_off, _ = local(Hm, vx, vy, 4, 8, field, ref_geom=ref_off)
assert np.abs(ref_b - ref).max() > 1.0, np.abs(ref_b - ref).max()   # the two operators do differ
assert geom_off == [ref_off], geom_off
assert np.abs(out_off - ref_b).max() == 0.0, np.abs(out_off - ref_b).max()
print(f"B1 PASS  one bin == global (max|diff| {d1:.1e}); uniform geometry with n_bins=4 collapses to 1 bin ({d4:.1e}) "
      f"onto the single-geometry reference, not the centroid ({np.abs(ref_b - ref).max():.2f} m apart)")

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

# B5
Hd, vxd, vyd = np.full((ny, nx), H0), np.full((ny, nx), 1500.0), np.full((ny, nx), -800.0)
patch = (slice(40, 56), slice(40, 56))
Hd[patch], vxd[patch], vyd[patch] = 700.0, 200.0, 900.0
geom5, W5, info5 = geometry_bins(Hd, vxd, vyd, dom, 4, 4)
mass = W5.sum(axis=(1, 2))
assert np.allclose(W5.sum(0), 1.0), (W5.sum(0).min(), W5.sum(0).max())
assert np.all(mass > 0), mass
assert len(W5) == len(geom5) == info5["effective"] == 2, (geom5, info5)
assert info5["requested"] == 4 and not info5["uniform"] and info5["reseeded"], info5
for a in range(len(geom5)):
    for b in range(a + 1, len(geom5)):
        assert not np.allclose(geom5[a], geom5[b]), (geom5[a], geom5[b])
pb = int(np.argmin([abs(g[0] - 700.0) for g in geom5]))
assert np.allclose(geom5[pb], (700.0, 200.0, 900.0)), geom5[pb]
assert np.allclose(geom5[1 - pb], (H0, 1500.0, -800.0)), geom5[1 - pb]
w_patch = W5[pb][patch].min()
assert w_patch > 0.0, w_patch                      # the patch carries real weight of its own bin
assert np.all(W5.argmax(0)[46:50, 46:50] == pb), W5.argmax(0)[46:50, 46:50]
# The same field through the B-PINN's resolution must NOT be swapped for the single reference
# geometry: only a genuinely uniform domain earns that fallback.
_, geom5b, _ = local(Hd, vxd, vyd, 4, 4, field, ref_geom=(H0 + 90.0, 1500.0, -800.0))
assert len(geom5b) == 2 and sorted(geom5b) == sorted(geom5), (geom5b, geom5)
print(f"B5 PASS  dominant geometry + anomalous patch -> {info5['effective']} of {info5['requested']} bins "
      f"after farthest-point re-seeding; patch bin {tuple(round(c) for c in geom5[pb])} holds min weight "
      f"{w_patch:.2f} on the patch; not reported uniform")

# B6
for over in ({"H_ref_m": 700.0}, {"ux_ref_myr": 1500.0}, {"uy_ref_myr": -800.0}):
    try:
        BPINNConfig(transfer=True, n_bins=4, **over)
    except ValueError as exc:
        assert next(iter(over)) in str(exc), str(exc)
    else:
        raise AssertionError(f"BPINNConfig(n_bins=4, {over}) was accepted")
    BPINNConfig(transfer=True, n_bins=1, **over)        # still legal on the single-geometry path
print("B6 PASS  n_bins > 1 rejects the H_ref_m/ux_ref_myr/uy_ref_myr overrides; n_bins=1 still takes them")

# B7
for bad in ((dx, -dx), (-dx, dx), (dx, 0.0)):
    try:
        transfer_observation_operator(ny, nx, bad[0], bad[1], H0, 1500.0, -800.0, eta, alpha, bg)
    except ValueError:
        pass
    else:
        raise AssertionError(f"transfer_observation_operator accepted pitches {bad}")
print("B7 PASS  a non-positive pixel pitch raises instead of silently y-mirroring the operator")
print("gate_bpinn_local_transfer: all rungs passed")

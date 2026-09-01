"""Pre-tilt-fit assembly of mcmurdo per-DEM ``BAD_STRIPS`` (Shean-faithful).

Ported from :mod:`pig.scripts.assemble_bad_strips_pretilt`. The production
screen :mod:`mcmurdo.scripts.find_bad_epochs` evaluates the bad-strip gates
on the *tilt-corrected* stack plus the tilt-params file, so it can only run
*after* :mod:`mcmurdo.tilt_fit`. This script assembles the same drop list
*before* the first tilt fit, from the two gates that need no tilt outputs, so
``config.BAD_STRIPS`` can be set and the stack tilt-fit exactly ONCE:

    (a) pc_align ``end_p50 > 10 m``  -- from ``aggregate_basin_quality``
        (parses the per-strip pc_align end-error CSVs; tilt-independent).
        Inert when the basin config has no ``STRIP_SOURCES``.
    (b) ``|raw static-median| > 50 m`` -- catastrophic blunder backstop,
        computed on the raw ``dem_id`` stack as the per-slice median of
        ``z - z_ref`` over the static-control polygon (``z_ref`` = the
        per-pixel temporal median). Catches clean-``end_p50`` blunders that
        gate (a) misses (a strip that aligns tightly yet reads +123 m).

Gate (c) (IRLS hard-failure) is only observable post-tilt and is left to
the ``find_bad_epochs`` confirmation screen.

Gate evaluation REUSES the production :func:`suggest_bad_epochs` rather than
re-implementing the thresholds: we hand it a pre-tilt ``df`` with
``med_resid_m`` := the raw static-median deviation, ``weight_mean`` := a
finite sentinel, and ``drop_irls_failed=False``.

Read-only: prints the paste-ready ``BAD_STRIPS`` tuple and the
``BAD_EPOCHS -> ()`` diff; it does **not** edit ``config.py``.

Run::

    python -m mcmurdo.scripts.assemble_bad_strips_pretilt --res 250
"""
from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import numpy as np
import pandas as pd

from stereo_melt.stack import load_stack
from stereo_melt.coregister.tilt import build_static_area_polygon_mask
from stereo_melt.coregister.alignment_quality import aggregate_basin_quality
from stereo_melt.coregister.tilt_qc import suggest_bad_epochs

from mcmurdo import config

BASIN = "mcmurdo"
END_P50_GATE_M = 10.0
CATASTROPHIC_GATE_M = 50.0


def _to_str(v) -> str:
    if isinstance(v, (bytes, bytearray)):
        return v.decode()
    return str(v)


def _raw_stack_path(res: int | None, tag: str | None):
    # Mirror how load_basin_stack / find_bad_epochs name the raw (non-tilt)
    # stack. A loose glob is unsafe: tagged siblings (separate experiments,
    # no dem_id) share the prefix and can sort after the plain name.
    prefix = f"{BASIN}_stack"
    if res is not None:
        prefix += f"_{int(round(res))}m"
    if tag:
        prefix += f"_{tag}"
    return config.PROCESSED_DIR / f"{prefix}_{config.START_TIME}_{config.END_TIME}.nc"


def main(res: int | None = 250, tag: str | None = None) -> None:
    raw = _raw_stack_path(res, tag)
    if not raw.exists():
        raise SystemExit(f"raw dem_id stack not found: {raw}")
    st = load_stack(raw)
    print(f"raw stack: {raw.name}")
    print(f"  dims: {dict(st.sizes)} | dem_id present: {'dem_id' in st.coords}")
    if "dem_id" not in st.coords:
        raise SystemExit("dem_id coord MISSING -- rebuild did not attach it.")

    ids = np.array([_to_str(v) for v in st["dem_id"].values])
    times = pd.to_datetime(st["time"].values)
    dates = times.normalize()
    n = len(ids)
    n_uniq = len(set(ids))
    flag = "" if n_uniq == n else "  !! DEM IDS NOT UNIQUE"
    print(f"  slices: {n} | unique dem_id: {n_uniq}{flag}")

    # ---- gate (a) input: per-strip pc_align end_p50, joined by dem_id ----
    strip_sources = getattr(config, "STRIP_SOURCES", None)
    aq = aggregate_basin_quality(strip_sources) if strip_sources else pd.DataFrame()
    ep_map: dict[str, float] = {}
    if not aq.empty:
        ep_map = dict(zip(aq["dem_id"].astype(str), aq["end_p50"].astype(float)))
        matched = sum(1 for i in ids if i in ep_map)
        ep_all = aq["end_p50"].to_numpy(float)
        ep_all = ep_all[np.isfinite(ep_all)]
        pcts = np.percentile(ep_all, [50, 75, 90, 95, 99]) if ep_all.size else [np.nan] * 5
        print(f"  aq: {len(aq)} aligned strips; stack<->aq dem_id match {matched}/{n}")
        print(f"  end_p50 (m): p50={pcts[0]:.2f} p75={pcts[1]:.2f} "
              f"p90={pcts[2]:.2f} p95={pcts[3]:.2f} p99={pcts[4]:.2f} "
              f"max={ep_all.max():.2f}")
        if matched < n:
            miss = [i for i in ids if i not in ep_map][:3]
            print(f"  (note: {n - matched} stack strips have no end-error CSV "
                  f"[swept aligned outputs] -> end_p50 NaN -> gate (a) inert for them)")
            print(f"  sample unmatched stack id : {miss[0] if miss else '-'}")
            print(f"  sample aq id              : {next(iter(ep_map))}")
    elif not strip_sources:
        print("  (config has no STRIP_SOURCES -- end_p50 gate (a) inert; gate (b) only)")
    else:
        print("  !! aggregate_basin_quality empty -- end_p50 gate inert")
    end_p50 = np.array([ep_map.get(i, np.nan) for i in ids])

    # ---- gate (b) input: raw static-control median deviation per slice ----
    static = build_static_area_polygon_mask(st, config.BEDMACHINE_NC).values.astype(bool)
    z = st.values
    z_ref = np.nanmedian(z, axis=0)
    ref_ok = np.isfinite(z_ref)
    n_static = int(static.sum())
    med_resid = np.full(n, np.nan)
    frac_static = np.zeros(n)
    for k in range(n):
        m = np.isfinite(z[k]) & static & ref_ok
        c = int(m.sum())
        frac_static[k] = (c / n_static) if n_static else 0.0
        if c:
            med_resid[k] = float(np.nanmedian(z[k][m] - z_ref[m]))
    print(f"  static-control pixels: {n_static}")

    # ---- evaluate gates via the PRODUCTION suggest_bad_epochs (no dup logic) ----
    df = pd.DataFrame({
        "dem_id": ids,
        "epoch": times,
        "end_p50": end_p50,
        "med_resid_m": med_resid,   # pre-tilt proxy for post-tilt static residual
        "weight_mean": 1.0,         # IRLS not evaluable pre-tilt -> finite sentinel
        "frac_static": frac_static,
    })
    suggested = suggest_bad_epochs(
        df,
        end_p50_threshold_m=END_P50_GATE_M,
        catastrophic_resid_m=CATASTROPHIC_GATE_M,
        drop_irls_failed=False,     # gate (c) deferred to post-tilt screen
    )
    bad_ids = sorted(suggested["dem_id"].astype(str).unique())

    # gate attribution (mirrors suggest_bad_epochs' own predicates)
    by_a = {ids[k]: float(end_p50[k]) for k in range(n)
            if np.nan_to_num(end_p50[k]) > END_P50_GATE_M}
    by_b = {ids[k]: float(med_resid[k]) for k in range(n)
            if np.isfinite(med_resid[k]) and abs(med_resid[k]) > CATASTROPHIC_GATE_M}

    print(f"\n  gate (a) end_p50>{END_P50_GATE_M:.0f}m : {len(by_a)} strip(s)")
    print(f"  gate (b) |raw med|>{CATASTROPHIC_GATE_M:.0f}m: {len(by_b)} strip(s)")
    idx = {i: k for k, i in enumerate(ids)}
    for i, v in sorted(by_b.items(), key=lambda kv: -abs(kv[1])):
        print(f"      {dates[idx[i]].date()}  {v:+8.0f} m   {i}")
    print(f"  UNION BAD_STRIPS      : {len(bad_ids)} strip(s)")

    # ---- per-DEM vs date-keying: clean same-day siblings spared ----
    badset = set(bad_ids)
    touched = sorted(set(dates[np.isin(ids, bad_ids)]))
    spared = 0
    for dd in touched:
        sel = dates == dd
        spared += int(sel.sum()) - sum(1 for s in ids[sel] if s in badset)
    print(f"\n  dates touched: {len(touched)} | clean same-day slices "
          f"SPARED vs date-keying: {spared}")

    # ---- current BAD_EPOCHS -> () diff on THIS stack ----
    old_eps = tuple(getattr(config, "BAD_EPOCHS", ()))
    if old_eps:
        old_np = np.array(
            [np.datetime64(pd.Timestamp(e).normalize()) for e in old_eps],
            dtype="datetime64[ns]",
        )
        old_drop = int(np.isin(dates.values, old_np).sum())
    else:
        old_drop = 0
    print(f"\n  current config.BAD_EPOCHS : {len(old_eps)} date(s) -> "
          f"drops {old_drop} stack slice(s) (date-keyed)")
    print(f"  proposed config.BAD_STRIPS: {len(bad_ids)} strip(s) -> "
          f"drops {len(bad_ids)} stack slice(s) (per-DEM)")
    print("  config.BAD_EPOCHS becomes : ()")

    # ---- migration audit: is every old date-keyed drop covered per-DEM? ----
    if old_eps:
        print("\n  migration audit -- old BAD_EPOCHS date -> current per-DEM status:")
        for e in sorted(old_eps):
            ed = pd.Timestamp(e).normalize()
            sel = (dates == ed)
            n_day = int(np.sum(sel))
            if n_day == 0:
                print(f"      {e}: no strips in current stack "
                      f"(count-cut / source-filtered upstream -- Shean tocut_lowcount analog)")
                continue
            n_flag = sum(1 for s in ids[sel] if s in badset)
            eps_day = end_p50[sel]
            mx = float(np.nanmax(eps_day)) if np.isfinite(eps_day).any() else float("nan")
            tail = "" if n_flag else "  <- now CLEAN under current align"
            print(f"      {e}: {n_flag}/{n_day} strip(s) flagged, "
                  f"max end_p50={mx:.1f}m{tail}")

    # ---- paste-ready ----
    print("\n# ---- paste into config.py ----")
    print("BAD_STRIPS: tuple[str, ...] = (")
    for i in bad_ids:
        tags = []
        if i in by_a:
            tags.append(f"end_p50={by_a[i]:.1f}m")
        if i in by_b:
            tags.append(f"raw_dev={by_b[i]:+.0f}m")
        print(f'    "{i}",  # {", ".join(tags)}')
    print(")")
    print("BAD_EPOCHS: tuple[str, ...] = ()")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res", type=int, default=250,
                    help="resolution variant (m); omit suffix with --res 0 for native")
    ap.add_argument("--tag", default=None, help="experiment tag in the stack name")
    a = ap.parse_args()
    main(res=(a.res or None), tag=a.tag)

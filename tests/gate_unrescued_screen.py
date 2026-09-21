"""Gate: ``tilt_qc.screen_unrescued_epochs`` on a thinning stack with a known model.

U1 clean nocorr kept · U2 NMAD > 5 m flagged · U3 20 % blunders flagged ·
U4 30 m offset flagged · U5 < min_px flagged · U6 controlled slice never
flagged · U7 thinning not flagged (fit frame) · U8 domain_mask honoured ·
U9 missing source_variant refused.

Run::

    /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python tests/gate_unrescued_screen.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stereo_melt.coregister.tilt_qc import screen_unrescued_epochs  # noqa: E402

rng = np.random.default_rng(7)
ny, nx = 40, 50
y = np.arange(ny)[::-1] * 250.0
x = np.arange(nx) * 250.0
times = pd.to_datetime([
    "2011-01-01", "2012-01-01", "2013-01-01", "2014-01-01", "2015-01-01",
    "2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2021-01-01",
])
variants = np.array(["nocorr", "nocorr", "nocorr", "nocorr", "nocorr",
                     "ctrl", "ctrl", "ctrl", "ctrl", "nocorr"])
labels = ["clean_2011", "wide", "blunders", "offset", "sparse",
          "ctrl_blunders", "ctrl", "ctrl", "ctrl", "clean_2021"]

days = (times - times[0]).days.to_numpy(float)
t_c = days - days.mean()
intercept = 100.0 + rng.normal(0.0, 30.0, (ny, nx))
dhdt = np.full((ny, nx), -2.0 / 365.25)           # 2 m/yr thinning, m/day
z = np.stack([intercept + dhdt * t_c[k] for k in range(len(times))])
z += rng.normal(0.0, 0.3, z.shape)                 # clean noise everywhere

z[1] += rng.normal(0.0, 12.0, (ny, nx))            # wide spread
blunder = rng.random((ny, nx)) < 0.2
z[2][blunder] += 60.0                              # 20 % blunders
z[5][blunder] += 60.0                              # same, but controlled
z[3] += 30.0                                       # unabsorbed offset
z[4][:, :] = np.nan
z[4][:5, :3] = intercept[:5, :3] + dhdt[:5, :3] * t_c[4]   # 15 px only

stack = xr.DataArray(
    z, dims=("time", "y", "x"),
    coords={"time": times, "y": y, "x": x,
            "source_variant": ("time", variants),
            "dem_id": ("time", [f"strip_{i}_{s}" for i, s in enumerate(labels)])},
)
params = xr.Dataset(
    {"intercept": (("y", "x"), intercept), "dhdt": (("y", "x"), dhdt),
     "weight_mean": ("time", np.ones(len(times))),
     "weight_frac_kept": ("time", np.ones(len(times)))},
    coords={"time": times, "y": y, "x": x},
)

df = screen_unrescued_epochs(stack, params)
flag = dict(zip(labels, df["unrescued"]))
fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


print("screen table:")
print(df[["dem_id", "source_variant", "n_px", "med_resid_m", "nmad_m",
          "frac_blunder", "unrescued", "reason"]].to_string(index=False))
check("U1 clean nocorr kept", not flag["clean_2011"])
check("U2 wide spread flagged", flag["wide"], df.loc[1, "reason"])
check("U3 blunders flagged", flag["blunders"], df.loc[2, "reason"])
check("U4 unabsorbed offset flagged", flag["offset"] and df.loc[3, "nmad_m"] < 1.0,
      df.loc[3, "reason"])
check("U5 sparse flagged", flag["sparse"], df.loc[4, "reason"])
check("U6 controlled slice never flagged",
      not flag["ctrl_blunders"] and not df.loc[5, "screened"])
check("U7 thinning does not trip the screen (fit frame)",
      not flag["clean_2011"] and not flag["clean_2021"]
      and abs(df.loc[9, "med_resid_m"]) < 0.1,
      f"2021 median {df.loc[9, 'med_resid_m']:+.3f} m")

dom = np.zeros((ny, nx), bool)
dom[:, :25] = True
df_dom = screen_unrescued_epochs(stack, params, domain_mask=dom)
check("U8 domain_mask restricts scored pixels",
      int(df_dom.loc[0, "n_px"]) == int(dom.sum()),
      f"{int(df_dom.loc[0, 'n_px'])} of {int(dom.sum())}")

try:
    screen_unrescued_epochs(stack.drop_vars("source_variant"), params)
    check("U9 missing source_variant refused", False, "no error raised")
except ValueError as e:
    check("U9 missing source_variant refused", "source_variant" in str(e))

print(f"\n{len(fails)} failure(s)" if fails else "\nALL PASS (9/9)")
sys.exit(1 if fails else 0)

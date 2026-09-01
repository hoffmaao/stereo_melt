"""Build pig/figures/pig_pipeline_chart.html — the PIG processing chain,
one card per stage with the real QC/production figure embedded (base64)."""
import base64
import os

_REPO = __import__("pathlib").Path(__file__).resolve().parents[3]
FIG = f"{_REPO}/examples/pig/figures"
OUT = os.path.join(FIG, "pig_pipeline_chart.html")


def b64(name):
    with open(os.path.join(FIG, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


STEPS = [
    ("1 · Input — REMA strip DEMs",
     "stack_footprints_250m_is2ctempo.png",
     "WorldView/GeoEye 2 m stereo strips (REMA s2s041 release) are searched, "
     "downloaded, and aligned over the CANONICAL wider AOI — the black "
     "rectangle (pig_stack_extent.shp), which is also the stack grid, sized "
     "beyond the shelf so grounded/rock static-control terrain is inside the "
     "domain. 533 stacked DEMs, 2010-12-09 → 2023-12-25, colored by "
     "acquisition date. Since the 2026-05 AOI migration, search, alignment, "
     "and stack all use this one extent (an earlier revision of this chart "
     "showed the legacy shelf-only 87-strip search — superseded). Each basin "
     "owns its ASP root — strips are never shared across basins."),
    ("2 · Coregistration — ASP pc_align (6-DOF ICP)",
     "fig4d_coreg_before_after.png",
     "Every strip is aligned with point-to-plane ICP against altimetry control: "
     "ICESat-2 ATL06, CryoTEMPO-LI CryoSat-2, ATM/LVIS where available, and "
     "BedMachine rock outcrops. Median |residual| drops 1.91 m → 0.29 m over "
     "all 634 DEMs (green band = Shean 2019's after-alignment range), with an "
     "inverse-variance-balanced re-alignment pass. This is the primary "
     "geolocation step; pdemtools is used only for ingestion and geoid."),
    ("3 · Post-coregistration statics",
     None,
     "Applied after alignment, in Shean order, inside a 3 km feathered floating "
     "mask: ocean tide (CATS2008 via pyTMD), one inverse-barometer scalar per "
     "strip from ERA5 mean-sea-level pressure, DTU mean dynamic topography, and "
     "the geoid. Corrections before alignment are forbidden — they would bias "
     "the ICP."),
    ("4 · Stack build + control masks",
     "static_mask_strip_count_250m.png",
     "All corrected strips are reprojected to one 250 m EPSG:3031 grid into a "
     "(time, y, x) cube keyed by dem_id; per-DEM rejections (BAD_STRIPS, "
     "melt-space QC screens) are applied at load. Left: the static-control mask "
     "(grounded ice eroded 2 km + rock) used by the tilt fit; right: per-pixel "
     "strip counts (an earlier 141-epoch build shown — the production stack "
     "carries 513 epochs)."),
    ("5 · Tilt correction — Shean joint least squares",
     "tilt_fit_qc_250m_is2ctempo.png",
     "One joint solve estimates a plane (tilt + offset) per strip AND a "
     "per-pixel intercept + trend, with per-epoch offset priors set by each "
     "strip's control class (0.1 m for IS2/ATM/LVIS-controlled … 1.0 m for "
     "CryoTEMPO/no-control), Tukey IRLS robustness, and z_ref pre-subtraction. "
     "QC panels: per-epoch slopes and offsets, solve coverage, and the "
     "recovered secular dh/dt. Output: the *_tilt_corrected stack every solver "
     "consumes."),
    ("6 · Velocity — fused, time-varying",
     "fig_velocity_representation.png",
     "Melt solvers advect parcels and take flux divergence through a "
     "time-varying velocity field: MEaSUREs + quarterly ASE mosaics fused with "
     "an EOF + Gaussian-process Kalman smoother (King-2020-style). The "
     "representation study shown here is why: a time-MEAN field inflates path "
     "flux ~40% on PIG (88.6 → 125.2 Gt/yr uncorrected), so time-varying "
     "velocity + trend thickness is the production choice."),
    ("7 · Melt-rate inputs",
     "melt_inputs_250m_is2ctempo.png",
     "The solver inputs on the common grid: time-mean surface (freeboard → "
     "thickness via BedMachine static firn-air content), fused speed |v|, "
     "RACMO2.4p1 surface mass balance over the window, and the firn field. "
     "Thickness uses ρi=918/ρw=1027; firn column changes enter as "
     "endpoint-exact Δd."),
    ("8 · Solvers → production melt",
     "melt_comparison_250m_is2ctempo.png",
     "Two production solvers run on identical inputs: Eulerian (Shean Eq. 10, "
     "per-pixel dh/dt + flux divergence) and the LAGRANGIAN PATH solver — the "
     "production product — which differences the surface along flow paths "
     "between epoch pairs 1.5–2.5 yr apart and aggregates per cell by pair "
     "median. Parcel-following differencing removes advecting-topography "
     "aliasing physically, which is why its map is smooth at ±60 m/yr. "
     "Integrated basal flux: 90.9 Gt/yr, inside Shean 2019's 82–93. Negative "
     "= melt everywhere. (Third panel: legacy spectral inverse, retired from "
     "products, diagnostic only.)"),
    ("9 · Diagnostics branch — fused inverse (not production)",
     "pig_fused_melt_map.png",
     "The twin-validated budget ⊕ variational fused inverse under the "
     "band-pass projection contract, run on the same stack as a diagnostic. "
     "Its no-truth detector (kept-band variance explained) reads −11.2 on PIG "
     "— the correction is REJECTED and the budget prior stands; the "
     "variational panel (own ±13 m/yr scale) shows the operator-mode striping "
     "that detector caught. On the synthetic twins, where truth exists, the "
     "same solver beats the hydrostatic pair (band nrmse 1.15 → 0.76). "
     "Production melt remains step 8."),
]

cards = []
for title, img, caption in STEPS:
    img_html = (f'<div class="imgwrap"><img src="{b64(img)}" alt="{title}">'
                f'<div class="path">pig/figures/{img}</div></div>'
                if img else "")
    cards.append(f"""
    <section class="card">
      <h2>{title}</h2>
      <p>{caption}</p>
      {img_html}
    </section>""")

arrow = '<div class="arrow">▼</div>'
html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>PIG processing pipeline — stage by stage</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
         margin: 0 auto; max-width: 1060px; padding: 24px 16px 60px;
         background: #fafafa; color: #1a1a1a; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #16181d; color: #e8e8e8; }}
    .card {{ background: #1f232b; border-color: #333a46; }}
    .path {{ color: #9aa3b2; }}
    .arrow {{ color: #9aa3b2; }}
    header p {{ color: #b8c0cc; }}
  }}
  header h1 {{ font-size: 1.5rem; margin: 0 0 6px; }}
  header p {{ color: #444; margin: 0 0 18px; line-height: 1.45; }}
  .card {{ background: #fff; border: 1px solid #e2e2e2; border-radius: 10px;
          padding: 18px 20px 16px; margin: 0; }}
  .card h2 {{ font-size: 1.06rem; margin: 0 0 8px; }}
  .card p {{ margin: 0 0 12px; line-height: 1.5; font-size: 0.94rem; }}
  .imgwrap {{ background: #fff; border-radius: 6px; overflow-x: auto; }}
  .imgwrap img {{ max-width: 100%; display: block; margin: 0 auto; }}
  .path {{ font-family: ui-monospace, monospace; font-size: 0.72rem;
          color: #777; margin-top: 6px; }}
  .arrow {{ text-align: center; color: #888; font-size: 1.1rem;
           margin: 6px 0; }}
</style></head><body>
<header>
  <h1>Pine Island Glacier basal-melt pipeline — every stage, with its own PIG imagery</h1>
  <p>The production chain from raw stereo DEM strips to the basal melt-rate map
  (stereo_melt library + <code>pig/</code> driver, 2010-01-01 → 2024-01-10,
  250 m, EPSG:3031). Sign convention throughout: <b>negative = melt</b>.
  Stages 1–8 are production; stage 9 is the research diagnostics branch.</p>
</header>
{arrow.join(cards)}
</body></html>"""

with open(OUT, "w") as f:
    f.write(html)
print(f"wrote {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")

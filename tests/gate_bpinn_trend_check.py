"""Gate: ``bpinn.trend_verdict`` -- the two-sided surrogate dH/dt check.

T1 healthy fit -> None · T2 collapse (fitted < 20 % of observed) · T3 runaway
(fitted > RUNAWAY_TREND_FACTOR x observed; the PIG trunk gave 4x with a
time-dependent melt net) · T4 steady stack stands down · T5 non-finite stands down ·
T6 opposite sign.

Run::

    /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python tests/gate_bpinn_trend_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stereo_melt.dynamics.bpinn import (  # noqa: E402
    RUNAWAY_TREND_FACTOR, STEADY_TREND_MYR, trend_verdict,
)

cases = [
    ("T1 healthy", (-5.1, -5.27), None),
    ("T2 collapse", (-0.3, -5.27), "collapse"),
    ("T3 runaway (PIG trunk 4x)", (-22.08, -5.27), "runaway"),
    ("T3b just under the factor", (-5.27 * RUNAWAY_TREND_FACTOR * 0.99, -5.27), None),
    ("T4 steady stack stands down", (-0.0, 0.5 * STEADY_TREND_MYR), None),
    ("T5 non-finite stands down", (float("nan"), -5.27), None),
    ("T6 opposite sign", (5.0, -5.27), "sign"),
]
fails = [n for n, args, want in cases if trend_verdict(*args) != want]
for n, args, want in cases:
    print(f"  [{'FAIL' if n in fails else 'PASS'}] {n}: {trend_verdict(*args)!r}")
print(f"\n{len(fails)} failure(s)" if fails else f"\nALL PASS ({len(cases)}/{len(cases)})")
sys.exit(1 if fails else 0)

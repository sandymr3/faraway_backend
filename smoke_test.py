"""Headless smoke test: run the sim for a few seconds and report key metrics."""
from __future__ import annotations

import numpy as np

from sim.swarm import Swarm

DT = 1 / 30.0


def run(seconds: float = 12.0, emp_at: float | None = 7.0) -> None:
    sw = Swarm()
    steps = int(seconds / DT)
    errs, covs = [], []
    fired = False
    for i in range(steps):
        if emp_at is not None and not fired and sw.t >= emp_at:
            alert = sw.simulate_emp(0.4)
            print(f"\n>> EMP fired @ t={sw.t:4.1f}s : {alert['msg']}")
            fired = True
        payload = sw.step(DT)
        m = payload["metrics"]
        errs.append(m["mean_loc_error_cm"])
        if i % 60 == 0:
            print(
                f"t={payload['t']:5.1f}s tick={payload['tick']:4d} | "
                f"err={m['mean_loc_error_cm']:5.1f}cm | "
                f"conn={m['connectivity']:.2f} | cover={m['coverage_pct']*100:4.1f}% | "
                f"active={m['agents_active']} down={m['agents_down']} | "
                f"edges={len(payload['edges']):2d} comps={len(payload['components'])} | "
                f"survivors={m['survivors_found']}"
            )

    warm = errs[60:]  # ignore the first 2s of settling
    print("\n--- summary ---")
    print(f"mean loc error (post-warmup): {np.mean(warm):.1f} cm  (target < 15 cm)")
    print(f"p95 loc error               : {np.percentile(warm, 95):.1f} cm")
    print(f"final coverage              : {payload['metrics']['coverage_pct']*100:.1f} %")
    print(f"final edge count            : {len(payload['edges'])}")
    print(f"survivors found             : {payload['metrics']['survivors_found']}/5")
    print(f"payload keys                : {list(payload.keys())}")


if __name__ == "__main__":
    run()

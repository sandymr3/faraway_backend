"""
sensors.py — Onboard perception: UWB ranging, LiDAR mapping, heat sensing.

None of these read absolute position. UWB measures noisy *relative* distances,
LiDAR raycasts the surrounding walls to build a shared point-cloud map, and the
thermal sensor reports survivors only within sensing range and line of sight.
"""
from __future__ import annotations

import numpy as np

from .agent import Agent
from .faults import FaultState
from .localization import RANGE_NOISE
from .world import World

LIDAR_BEAMS = 24
LIDAR_RANGE = 8.0
SURVIVOR_RANGE = 6.0

_rng = np.random.default_rng(23)


def uwb_ranges(topo: dict) -> dict:
    """Measured UWB distances for every live edge (true distance + Gaussian noise)."""
    out: dict[tuple[str, str], float] = {}
    for e in topo["edges"]:
        meas = max(0.05, e["d"] + float(_rng.normal(0, RANGE_NOISE)))
        out[(e["a"], e["b"])] = meas
    return out


def lidar_scan(agents: list[Agent], world: World) -> list[list]:
    """Raycast each active agent's surroundings; return newly discovered wall points.

    Free cells swept by each beam are marked explored (so frontier exploration has
    somewhere to go), and only *newly seen* wall cells are streamed — the dashboard
    map stitches together progressively in each agent's estimated frame, not GPS.
    """
    new_points: list[list] = []
    free_points: list[np.ndarray] = []
    for a in agents:
        if a.status == "down":
            continue
        # Estimation offset so the drawn cloud lives in the relative frame.
        offset = a.est - a.pos
        free_points.append(a.pos.copy())
        for k in range(LIDAR_BEAMS):
            ang = a.yaw + (2 * np.pi * k / LIDAR_BEAMS)
            rng = world.raycast(a.pos, ang, LIDAR_RANGE)
            d = np.array([np.cos(ang), np.sin(ang)])
            # Mark free space swept along the beam.
            for r in np.arange(0.6, rng - 0.4, 1.0):
                free_points.append(a.pos + r * d)
            if rng < LIDAR_RANGE:
                hit = a.pos + rng * d
                key = (int(round(hit[0] * 2)), int(round(hit[1] * 2)))  # 0.5m grid
                if key not in world.map_seen:
                    world.map_seen.add(key)
                    est_hit = hit + offset
                    new_points.append([round(float(est_hit[0]), 2),
                                       round(float(est_hit[1]), 2), a.id])
    if free_points:
        world.mark_explored(np.array(free_points))
    return new_points


def survivor_sense(agents: list[Agent], world: World, found: dict) -> list[dict]:
    """Detect survivors within range + line of sight; accumulate confidence.

    `found` is persistent swarm state: {survivor_index: {confidence, found_by, est}}.
    """
    for a in agents:
        if a.status == "down":
            continue
        offset = a.est - a.pos
        for si, s in enumerate(world.survivors):
            d = float(np.linalg.norm(a.pos - s))
            if d <= SURVIVOR_RANGE and world.line_of_sight(a.pos, s):
                rec = found.setdefault(si, {"confidence": 0.0, "found_by": set(),
                                            "est": (s + offset)})
                rec["confidence"] = min(1.0, rec["confidence"] + 0.04 * (1 - d / SURVIVOR_RANGE))
                rec["found_by"].add(a.id)
                # Refine the survivor's relative-frame estimate via the closest agent.
                rec["est"] = s + offset

    out = []
    for si, rec in found.items():
        out.append({
            "id": f"S{si + 1}",
            "pos": [round(float(rec["est"][0]), 2), round(float(rec["est"][1]), 2), 0.4],
            "confidence": round(rec["confidence"], 2),
            "found_by": sorted(rec["found_by"]),
        })
    return out

"""
behavior.py — Decentralised navigation policy (MARL stand-in).

Each agent computes a desired velocity from a weighted sum of local drives —
exactly the kind of policy a trained MARL actor would output, but hand-authored
so the demo is deterministic and dependency-free:

  * frontier attraction  — steer toward the nearest unexplored region
  * obstacle repulsion    — push away from walls sensed by LiDAR
  * neighbour separation  — don't crowd; spread to cover more area
  * connectivity cohesion — isolated agents drift back toward the swarm

The vector sum becomes a heading; a proportional controller turns the
heading + speed into unicycle commands (v, omega).
"""
from __future__ import annotations

import numpy as np

from .agent import Agent
from .world import World

# Drive weights (tuned for lively, spread-out exploration).
W_FRONTIER = 1.0
W_AVOID = 2.4
W_SEPARATION = 1.1
W_COHESION = 0.6
LEASH = 6.5  # keep the swarm within this radius of its centroid so the mesh
             # stays one connected component (and well-localised) until an EMP
AVOID_RANGE = 4.0
SEPARATION_RANGE = 4.5
AVOID_BEAMS = np.linspace(-np.pi / 2, np.pi / 2, 7)


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def compute_commands(agents: list[Agent], world: World) -> None:
    """Set each active agent's (v, omega) for this tick."""
    positions = {a.id: a.pos for a in agents}
    frontier = world.unexplored_centers()
    centroid = np.mean([a.pos for a in agents], axis=0) if agents else np.zeros(2)

    for a in agents:
        if a.status == "down":
            a.v = 0.0
            a.omega = 0.0
            continue

        drive = np.zeros(2)

        # --- frontier attraction: head to the nearest unexplored cell -----
        if frontier.shape[0] > 0:
            d = frontier - a.pos
            dist = np.hypot(d[:, 0], d[:, 1])
            target = frontier[int(np.argmin(dist))]
            to_target = target - a.pos
            norm = np.linalg.norm(to_target)
            if norm > 1e-6:
                drive += W_FRONTIER * to_target / norm
        else:
            # Fully mapped — cohesive orbit around the zone centre (keeps the
            # swarm together and roaming instead of all drifting one way).
            radial = a.pos - centroid
            rn = float(np.linalg.norm(radial)) + 1e-6
            drive += W_FRONTIER * np.array([-radial[1], radial[0]]) / rn

        # --- obstacle repulsion from LiDAR-style probe beams --------------
        avoid = np.zeros(2)
        for db in AVOID_BEAMS:
            ang = a.yaw + db
            rng = world.raycast(a.pos, ang, AVOID_RANGE)
            if rng < AVOID_RANGE:
                strength = (AVOID_RANGE - rng) / AVOID_RANGE
                avoid -= strength * np.array([np.cos(ang), np.sin(ang)])
        drive += W_AVOID * avoid

        # --- neighbour separation -----------------------------------------
        sep = np.zeros(2)
        for other in agents:
            if other.id == a.id or other.status == "down":
                continue
            off = a.pos - positions[other.id]
            dist = float(np.linalg.norm(off))
            if 1e-3 < dist < SEPARATION_RANGE:
                sep += off / (dist * dist)
        drive += W_SEPARATION * sep

        # --- connectivity cohesion: graduated leash toward the centroid ---
        to_center = centroid - a.pos
        n = float(np.linalg.norm(to_center))
        if a.status == "isolated" or len(a.neighbors) == 0:
            if n > 1e-6:
                drive += W_COHESION * 4.0 * to_center / n
        elif n > LEASH:
            # Pull harder the farther an agent strays beyond the leash radius.
            drive += W_COHESION * (1.0 + (n - LEASH) * 0.6) * to_center / n

        # --- convert drive vector to unicycle commands --------------------
        if np.linalg.norm(drive) < 1e-6:
            a.v = 0.2
            continue
        desired = float(np.arctan2(drive[1], drive[0]))
        err = _wrap(desired - a.yaw)
        a.omega = 2.2 * err
        # Slow down when turning hard or facing a wall ahead.
        ahead = world.raycast(a.pos, a.yaw, AVOID_RANGE)
        speed_scale = np.clip(1.0 - abs(err) / np.pi, 0.15, 1.0)
        speed_scale *= np.clip(ahead / AVOID_RANGE, 0.25, 1.0)
        a.v = a.max_speed * speed_scale

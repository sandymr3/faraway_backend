"""
agent.py — A single swarm member (UAV or UGV).

Holds *ground-truth* kinematic state (used only by the simulator), the swarm's
*estimated* relative position (filled in by the MDS localiser, never by GPS),
and per-agent status used for fault injection and the dashboard roster.
"""
from __future__ import annotations

import numpy as np

UAV_ALTITUDE = 3.2
UGV_ALTITUDE = 0.35


class Agent:
    def __init__(self, agent_id: str, kind: str, pos: np.ndarray, yaw: float) -> None:
        self.id = agent_id
        self.kind = kind                       # "uav" | "ugv"
        self.pos = np.asarray(pos, dtype=np.float64)   # ground-truth (x, y)
        self.yaw = float(yaw)
        self.v = 0.0                           # forward speed (m/s)
        self.omega = 0.0                       # yaw rate (rad/s)
        self.battery = 1.0
        self.status = "active"                 # "active" | "isolated" | "down"

        # Estimated relative position (from the decentralised localiser).
        self.est = self.pos.copy()
        self.cov = np.array([[0.25, 0.0], [0.0, 0.25]])  # 2x2 uncertainty
        self.loc_error_cm = 0.0
        self.neighbors: list[str] = []

        self.max_speed = 2.2 if kind == "uav" else 1.4
        self.max_omega = 2.6
        self.t = 0.0

    @property
    def altitude(self) -> float:
        if self.kind == "uav":
            # Gentle bob so the UAVs feel alive in the 3D viewport.
            return UAV_ALTITUDE + 0.18 * np.sin(self.t * 1.7 + hash(self.id) % 7)
        return UGV_ALTITUDE

    def step(self, dt: float) -> None:
        """Integrate the unicycle model. Downed agents hold position."""
        self.t += dt
        if self.status == "down":
            self.v = 0.0
            self.omega = 0.0
            return

        self.omega = float(np.clip(self.omega, -self.max_omega, self.max_omega))
        self.v = float(np.clip(self.v, 0.0, self.max_speed))
        self.yaw += self.omega * dt
        self.pos = self.pos + self.v * np.array([np.cos(self.yaw), np.sin(self.yaw)]) * dt

        # Slow, cosmetic battery drain.
        self.battery = max(0.0, self.battery - 0.0008 * dt * (1.0 + self.v))

"""
swarm.py — Orchestrates one decentralised control tick across all subsystems.

Pipeline per tick (PRD §4.1):
  faults → behaviour → integrate → topology → UWB → MDS localisation →
  error metric → LiDAR mapping → survivor detection → telemetry payload.
"""
from __future__ import annotations

import numpy as np

from . import behavior, sensors
from .agent import Agent
from .faults import FaultState
from .localization import Localizer, report_errors
from .topology import Topology
from .world import World

# Heterogeneous roster: 4 ground rovers + 6 aerial drones.
ROSTER = [
    ("UGV_Alpha", "ugv"), ("UGV_Bravo", "ugv"),
    ("UGV_Charlie", "ugv"), ("UGV_Delta", "ugv"),
    ("UAV_Beta_01", "uav"), ("UAV_Beta_02", "uav"), ("UAV_Beta_03", "uav"),
    ("UAV_Beta_04", "uav"), ("UAV_Beta_05", "uav"), ("UAV_Beta_06", "uav"),
]
DEPLOY_POINT = np.array([-10.5, -12.5])


class Swarm:
    def __init__(self, seed: int = 7) -> None:
        self.seed = seed
        self.world = World(seed)
        self.topology = Topology(comm_range=16.0)
        self.localizer = Localizer()
        self.faults = FaultState()
        self.agents: list[Agent] = []
        self.found: dict[int, dict] = {}
        self.t = 0.0
        self.tick = 0
        self.alerts: list[dict] = []
        self.map_points: list[list] = []  # full discovered map (sent to new clients)
        self._spawn()

    def _spawn(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.agents = []
        for i, (name, kind) in enumerate(ROSTER):
            # Loose deployment cluster — all within comm range so the mesh forms
            # instantly, then the swarm disperses to explore (GPS-denied).
            for _ in range(40):
                p = DEPLOY_POINT + rng.normal(0, 2.4, 2)
                if not self.world.inside_obstacle(p):
                    break
            yaw = float(rng.uniform(-np.pi, np.pi))
            self.agents.append(Agent(name, kind, p, yaw))

    # ------------------------------------------------------------------ #
    def step(self, dt: float) -> dict:
        self.t += dt
        self.tick += 1

        # 0) apply fault status
        for a in self.agents:
            if a.id in self.faults.down:
                a.status = "down"

        active = [a for a in self.agents if a.status != "down"]

        # 1) decentralised navigation policy → (v, omega)
        behavior.compute_commands(active, self.world)

        # 2) integrate motion
        for a in self.agents:
            a.step(dt)

        # 3) dynamic comm topology (GNN layer)
        topo = self.topology.build(self.agents, self.faults)

        # 4) UWB ranging + 5) MDS relative localisation
        ranges = sensors.uwb_ranges(topo)
        self.localizer.step(self.agents, topo["components"], ranges, dt)
        mean_err = report_errors(self.agents, topo["components"])

        # 6) collaborative mapping + 7) survivor detection
        pc_delta = sensors.lidar_scan(active, self.world)
        if pc_delta:
            self.map_points.extend(pc_delta)
            if len(self.map_points) > 7000:
                self.map_points = self.map_points[-7000:]
        survivors = sensors.survivor_sense(active, self.world, self.found)

        # 8) metrics
        n_down = len(self.faults.down)
        metrics = {
            "mean_loc_error_cm": mean_err,
            "connectivity": Topology.connectivity(self.agents, topo["components"]),
            "coverage_pct": round(self.world.coverage_pct(), 3),
            "agents_active": len(active),
            "agents_down": n_down,
            "survivors_found": sum(1 for s in survivors if s["confidence"] > 0.3),
        }

        alerts = self.alerts
        self.alerts = []

        from telemetry import build_telemetry
        return build_telemetry(
            t=self.t, tick=self.tick, agents=self.agents, topo=topo,
            survivors=survivors, pointcloud_delta=pc_delta,
            metrics=metrics, alerts=alerts,
        )

    # ------------------------------------------------------------------ #
    # Operator-injected faults (driven from the dashboard)
    # ------------------------------------------------------------------ #
    def simulate_emp(self, fraction: float = 0.4) -> dict:
        ids = [a.id for a in self.agents]
        hit = self.faults.simulate_emp(ids, fraction)
        pct = int(round(100 * len(self.faults.down) / max(1, len(ids))))
        alert = {
            "level": "critical",
            "msg": f"CRITICAL: Link Failure on {pct}% of Nodes. Re-routing GNN Topology...",
            "nodes": hit,
        }
        self.alerts.append(alert)
        return alert

    def kill_node(self, agent_id: str) -> dict:
        self.faults.kill_node(agent_id)
        alert = {"level": "warning", "msg": f"Node {agent_id} link severed.", "nodes": [agent_id]}
        self.alerts.append(alert)
        return alert

    def reset(self) -> dict:
        self.faults.reset()
        self.localizer.reset()
        self.found.clear()
        self.world.reset_coverage()
        self.map_points = []
        self.t = 0.0
        self.tick = 0
        self._spawn()
        alert = {"level": "info", "msg": "Swarm redeployed. All nodes nominal.", "nodes": []}
        self.alerts.append(alert)
        return alert

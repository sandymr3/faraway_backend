"""
telemetry.py — Serialise swarm state into the lightweight JSON the dashboard eats.

Matches the schema in the build plan; floats are rounded to keep the 30Hz
WebSocket payload small.
"""
from __future__ import annotations

import numpy as np


def _agent_dict(a) -> dict:
    return {
        "id": a.id,
        "type": a.kind,
        "pos_est": [round(float(a.est[0]), 2), round(float(a.est[1]), 2), round(a.altitude, 2)],
        "yaw": round(float(a.yaw), 3),
        "vel": [round(float(a.v), 2), round(float(a.omega), 2)],
        "status": a.status,
        "battery": round(float(a.battery), 3),
        "cov": [[round(float(a.cov[0, 0]), 4), round(float(a.cov[0, 1]), 4)],
                [round(float(a.cov[1, 0]), 4), round(float(a.cov[1, 1]), 4)]],
        "loc_error_cm": a.loc_error_cm,
        "neighbors": a.neighbors,
    }


def build_telemetry(*, t, tick, agents, topo, survivors, pointcloud_delta,
                    metrics, alerts) -> dict:
    return {
        "t": round(float(t), 2),
        "tick": tick,
        "agents": [_agent_dict(a) for a in agents],
        "edges": [{"a": e["a"], "b": e["b"], "weight": e["weight"]} for e in topo["edges"]],
        "components": topo["components"],
        "survivors": survivors,
        "pointcloud_delta": pointcloud_delta,
        "metrics": metrics,
        "alerts": alerts,
    }

"""
faults.py — Operator-injected failures (the "EMP" wow demo).

Tracks which nodes are down and which communication links are severed. The
swarm queries this every tick so downed agents freeze and their edges vanish
from the GNN topology, forcing the mesh to self-heal around them.
"""
from __future__ import annotations

import numpy as np


class FaultState:
    def __init__(self) -> None:
        self.down: set[str] = set()
        self.dropped_links: set[tuple[str, str]] = set()

    @staticmethod
    def _key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def link_dropped(self, a: str, b: str) -> bool:
        return self._key(a, b) in self.dropped_links

    def simulate_emp(self, agent_ids: list[str], fraction: float = 0.4) -> list[str]:
        """Down a random `fraction` of currently-alive nodes. Returns new casualties."""
        alive = [a for a in agent_ids if a not in self.down]
        if not alive:
            return []
        k = max(1, int(round(len(agent_ids) * fraction)))
        k = min(k, len(alive))
        rng = np.random.default_rng()
        hit = list(rng.choice(alive, size=k, replace=False))
        self.down.update(hit)
        return hit

    def kill_node(self, agent_id: str) -> None:
        self.down.add(agent_id)

    def revive(self, agent_id: str) -> None:
        self.down.discard(agent_id)

    def reset(self) -> None:
        self.down.clear()
        self.dropped_links.clear()

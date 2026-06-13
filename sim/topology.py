"""
topology.py — Dynamic communication graph (the GNN messaging layer).

Builds the swarm's adjacency from current UWB comm ranges, minus any nodes or
links knocked out by faults. Exposes the edge list (with signal-strength
weights), per-agent neighbour sets, and connected components — everything the
frontend topology graph and the self-healing demo need.
"""
from __future__ import annotations

import numpy as np

from .agent import Agent
from .faults import FaultState


class Topology:
    def __init__(self, comm_range: float = 11.0) -> None:
        self.comm_range = comm_range

    def build(self, agents: list[Agent], faults: FaultState):
        alive = [a for a in agents if a.status != "down"]
        pos = {a.id: a.pos for a in alive}
        ids = [a.id for a in alive]

        edges: list[dict] = []
        adj: dict[str, set[str]] = {i: set() for i in ids}

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if faults.link_dropped(a, b):
                    continue
                d = float(np.linalg.norm(pos[a] - pos[b]))
                if d <= self.comm_range:
                    weight = float(np.clip(1.0 - d / self.comm_range, 0.05, 1.0))
                    edges.append({"a": a, "b": b, "d": d, "weight": round(weight, 3)})
                    adj[a].add(b)
                    adj[b].add(a)

        components = self._components(ids, adj)

        # Annotate agents: neighbours + isolated status.
        comp_of = {}
        for c in components:
            for n in c:
                comp_of[n] = c
        for a in agents:
            if a.status == "down":
                a.neighbors = []
                continue
            a.neighbors = sorted(adj.get(a.id, set()))
            a.status = "isolated" if len(a.neighbors) == 0 else "active"

        return {"edges": edges, "adj": adj, "components": components, "comp_of": comp_of}

    @staticmethod
    def _components(ids: list[str], adj: dict[str, set[str]]) -> list[list[str]]:
        seen: set[str] = set()
        comps: list[list[str]] = []
        for start in ids:
            if start in seen:
                continue
            stack, comp = [start], []
            seen.add(start)
            while stack:
                n = stack.pop()
                comp.append(n)
                for m in adj[n]:
                    if m not in seen:
                        seen.add(m)
                        stack.append(m)
            comps.append(sorted(comp))
        return comps

    @staticmethod
    def connectivity(agents: list[Agent], components: list[list[str]]) -> float:
        """Fraction of active agents that sit in the largest connected component."""
        active = [a for a in agents if a.status != "down"]
        if not active:
            return 0.0
        largest = max((len(c) for c in components), default=0)
        return round(largest / len(active), 3)

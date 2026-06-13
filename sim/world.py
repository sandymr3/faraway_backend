"""
world.py — The disaster zone.

A 2D collapsed-structure environment (the "maze") made of axis-aligned wall
segments, plus ground-truth survivor positions and a coverage grid used for
frontier-based exploration. Everything in SwarmResQ is GPS-denied, so the world
exists only as ground truth for the simulation — agents never read absolute
coordinates from it; they only sense it through UWB / LiDAR / heat sensors.
"""
from __future__ import annotations

import numpy as np

# World bounds in metres (a 30m x 30m disaster zone).
BOUNDS = 15.0


def _rect_segments(cx: float, cy: float, w: float, h: float) -> list[tuple[float, float, float, float]]:
    """Return the 4 wall segments (ax, ay, bx, by) of an axis-aligned rectangle."""
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - h / 2, cy + h / 2
    return [
        (x0, y0, x1, y0),
        (x1, y0, x1, y1),
        (x1, y1, x0, y1),
        (x0, y1, x0, y0),
    ]


class World:
    """Static ground-truth environment: walls, survivors, exploration grid."""

    def __init__(self, seed: int = 7) -> None:
        self.rng = np.random.default_rng(seed)
        self.bounds = BOUNDS

        # Collapsed structures as rectangles (cx, cy, w, h). Corridors are left
        # between them so the swarm has somewhere to explore.
        self.rects = [
            (-9.0, 8.0, 10.0, 5.0),
            (7.0, 10.0, 9.0, 4.0),
            (-11.0, -6.0, 6.0, 9.0),
            (3.0, -4.0, 7.0, 7.0),
            (12.0, -9.0, 6.0, 6.0),
            (-2.0, 1.0, 3.0, 3.0),
        ]

        # Build wall segments: outer boundary + every rectangle edge.
        b = self.bounds
        segs: list[tuple[float, float, float, float]] = [
            (-b, -b, b, -b),
            (b, -b, b, b),
            (b, b, -b, b),
            (-b, b, -b, -b),
        ]
        for r in self.rects:
            segs.extend(_rect_segments(*r))
        # Shape (S, 4): ax, ay, bx, by — vectorised for fast raycasting.
        self.segments = np.array(segs, dtype=np.float64)

        # Ground-truth survivors hidden in the rubble (trapped near structures).
        self.survivors = np.array(
            [
                [-9.0, 11.5],
                [10.5, 9.0],
                [-11.0, -1.0],
                [5.5, -7.5],
                [12.0, -5.0],
            ],
            dtype=np.float64,
        )

        # Coverage grid (1m cells). free_mask marks navigable cells.
        self.cell = 1.0
        self.n = int(2 * self.bounds / self.cell)
        xs = np.linspace(-b + 0.5, b - 0.5, self.n)
        gx, gy = np.meshgrid(xs, xs)
        self.cell_centers = np.stack([gx.ravel(), gy.ravel()], axis=1)
        self.free_mask = ~np.array([self.inside_obstacle(p) for p in self.cell_centers])
        self.explored = np.zeros(self.n * self.n, dtype=bool)
        self.n_free = int(self.free_mask.sum())
        # Quantised wall points already streamed to the dashboard (dedup the cloud).
        self.map_seen: set[tuple[int, int]] = set()

    # ------------------------------------------------------------------ #
    # Geometry queries
    # ------------------------------------------------------------------ #
    def inside_obstacle(self, p: np.ndarray) -> bool:
        x, y = float(p[0]), float(p[1])
        if abs(x) >= self.bounds or abs(y) >= self.bounds:
            return True
        for cx, cy, w, h in self.rects:
            if abs(x - cx) <= w / 2 and abs(y - cy) <= h / 2:
                return True
        return False

    def raycast(self, origin: np.ndarray, angle: float, max_range: float) -> float:
        """Distance from origin along `angle` to the nearest wall (<= max_range)."""
        ox, oy = origin[0], origin[1]
        dx, dy = np.cos(angle), np.sin(angle)
        ax, ay = self.segments[:, 0], self.segments[:, 1]
        bx, by = self.segments[:, 2], self.segments[:, 3]
        ex, ey = bx - ax, by - ay
        denom = dx * ey - dy * ex
        safe = np.abs(denom) > 1e-9
        diffx, diffy = ax - ox, ay - oy
        t = np.where(safe, (diffx * ey - diffy * ex) / denom, np.inf)   # along ray
        u = np.where(safe, (diffx * dy - diffy * dx) / denom, np.inf)   # along segment
        hit = safe & (t > 1e-6) & (u >= 0.0) & (u <= 1.0)
        t = np.where(hit, t, np.inf)
        d = float(t.min())
        return min(d, max_range)

    def line_of_sight(self, a: np.ndarray, b: np.ndarray) -> bool:
        """True if no wall lies between points a and b."""
        d = b - a
        dist = float(np.hypot(d[0], d[1]))
        if dist < 1e-6:
            return True
        ang = float(np.arctan2(d[1], d[0]))
        return self.raycast(a, ang, dist + 0.05) >= dist - 0.05

    # ------------------------------------------------------------------ #
    # Coverage / frontier
    # ------------------------------------------------------------------ #
    def mark_explored(self, points: np.ndarray) -> None:
        """Mark the grid cells nearest to discovered points as explored."""
        if points.size == 0:
            return
        idx = np.round((points + self.bounds - 0.5) / self.cell).astype(int)
        idx = np.clip(idx, 0, self.n - 1)
        flat = idx[:, 1] * self.n + idx[:, 0]
        self.explored[flat] = True

    def coverage_pct(self) -> float:
        if self.n_free == 0:
            return 0.0
        return float((self.explored & self.free_mask).sum()) / self.n_free

    def unexplored_centers(self) -> np.ndarray:
        """Free cell centres not yet explored — frontier targets for behaviour."""
        mask = self.free_mask & ~self.explored
        return self.cell_centers[mask]

    def reset_coverage(self) -> None:
        self.explored[:] = False
        self.map_seen.clear()

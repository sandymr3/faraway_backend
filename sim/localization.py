"""
localization.py — Anchor-free, GPS-denied relative localisation.

The technical heart of SwarmResQ. There is no GPS and no absolute origin. Each
connected component recovers a *relative* coordinate frame from nothing but
pairwise UWB ranges, using the MDS family of methods (training-free,
infrastructure-free — see arXiv:2307.10255 / 2212.06519).

Per tick we run a decentralised-EKF-flavoured predict/update:

  predict — dead-reckon every agent forward from onboard odometry (v, yaw).
  update  — stress majorisation (SMACOF): a few relaxation iterations that move
            each estimate to satisfy its *observed* UWB ranges, initialised from
            the dead-reckoned prior. Using only observed edges (no geodesic
            in-fill) keeps it accurate even when the mesh is sparse.

Scalar Kalman variance propagation drives the per-agent confidence ellipse:
it shrinks with more UWB constraints and grows while an agent is isolated and
dead-reckoning (visible drift on the dashboard).
"""
from __future__ import annotations

import numpy as np

from .agent import Agent

ODOM_NOISE = 0.010     # per-step odometry position noise (m)
RANGE_NOISE = 0.04     # UWB ranging noise std (m) — used by sensors
PROCESS_Q = 0.05       # variance growth per second while dead-reckoning
MEAS_R = 0.008         # localised measurement variance floor
SMACOF_ITERS = 12      # relaxation iterations per tick
SMACOF_STEP = 1.0      # relaxation step (1.0 == standard SMACOF majorising step)


def classical_mds(D: np.ndarray, dim: int = 2) -> np.ndarray:
    """Recover dim-D coordinates from a full pairwise distance matrix (bootstrap)."""
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1][:dim]
    return vecs[:, order] * np.sqrt(np.maximum(vals[order], 0.0))


def _procrustes(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Align src onto dst with rotation+reflection+translation (no scaling)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    u, _, vt = np.linalg.svd(s0.T @ d0)
    return s0 @ (u @ vt) + mu_d


def _smacof(X: np.ndarray, edges: list[tuple[int, int, float]]) -> np.ndarray:
    """Relax positions X to satisfy observed pairwise distances (stress majorisation)."""
    n = X.shape[0]
    for _ in range(SMACOF_ITERS):
        disp = np.zeros_like(X)
        wsum = np.zeros(n)
        for i, j, dij in edges:
            diff = X[i] - X[j]
            dist = float(np.hypot(diff[0], diff[1])) + 1e-9
            corr = 0.5 * (dist - dij) * (diff / dist)   # pull pair toward dij
            disp[i] -= corr
            disp[j] += corr
            wsum[i] += 1.0
            wsum[j] += 1.0
        moving = wsum > 0
        X[moving] += SMACOF_STEP * disp[moving] / wsum[moving, None]
    return X


class Localizer:
    """Holds per-agent filter variance between ticks."""

    def __init__(self) -> None:
        self.sigma2: dict[str, float] = {}
        self.rng = np.random.default_rng(11)

    def reset(self) -> None:
        self.sigma2.clear()

    def step(self, agents: list[Agent], components: list[list[str]],
             ranges: dict, dt: float) -> None:
        by_id = {a.id: a for a in agents}

        # --- PREDICT: dead-reckon every agent from onboard odometry --------
        for a in agents:
            self.sigma2.setdefault(a.id, 0.06)
            if a.status == "down":
                continue
            heading = np.array([np.cos(a.yaw), np.sin(a.yaw)])
            a.est = a.est + a.v * dt * heading + self.rng.normal(0, ODOM_NOISE, 2)
            self.sigma2[a.id] += PROCESS_Q * dt

        # --- UPDATE: SMACOF relaxation on observed UWB ranges --------------
        for comp in components:
            members = [by_id[i] for i in comp if by_id[i].status != "down"]
            if len(members) < 2:
                continue
            idx = {a.id: k for k, a in enumerate(members)}
            edges = [(idx[a], idx[b], d) for (a, b), d in ranges.items()
                     if a in idx and b in idx]
            if not edges:
                continue
            X = np.array([a.est for a in members], dtype=np.float64)
            X = _smacof(X, edges)
            for k, a in enumerate(members):
                a.est = X[k]
                kgain = np.clip(len(a.neighbors) / 4.0, 0.35, 1.0)
                self.sigma2[a.id] = (1 - kgain) * self.sigma2[a.id] + kgain * MEAS_R

        # --- write covariance ellipses (slightly elongated along heading) --
        for a in agents:
            s2 = float(np.clip(self.sigma2.get(a.id, 0.06), 1e-4, 6.0))
            c, s = np.cos(a.yaw), np.sin(a.yaw)
            rot = np.array([[c, -s], [s, c]])
            a.cov = rot @ np.diag([s2 * 1.7, s2 * 0.7]) @ rot.T


def report_errors(agents: list[Agent], components: list[list[str]]) -> float:
    """Per-component Procrustes-align estimates to ground truth, score error (cm).

    Relative localisation is only defined *within* a connected component, so each
    component is aligned to truth independently — the honest accuracy metric.
    """
    by_id = {a.id: a for a in agents}
    errs: list[float] = []
    for comp in components:
        members = [by_id[i] for i in comp if by_id[i].status != "down"]
        if not members:
            continue
        if len(members) < 3:
            for a in members:
                a.loc_error_cm = round(float(np.linalg.norm(a.est - a.pos)) * 100, 1)
                errs.append(a.loc_error_cm)
            continue
        est = np.array([a.est for a in members])
        truth = np.array([a.pos for a in members])
        aligned = _procrustes(est, truth)
        for k, a in enumerate(members):
            e = round(float(np.linalg.norm(aligned[k] - truth[k])) * 100, 1)
            a.loc_error_cm = e
            errs.append(e)
    return round(float(np.mean(errs)), 1) if errs else 0.0

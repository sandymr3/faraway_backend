"""
app.py — FastAPI + Socket.io bridge.

Runs the swarm simulation autonomously at TICK_HZ and streams telemetry to
every connected dashboard. The frontend is a passive observer that can inject
faults (EMP / node kill / reset) back over the socket.

Run:  uvicorn app:app --host $HOST --port $PORT
Configuration via environment variables — see .env.example.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

import socketio
from fastapi import FastAPI

from config import CORS_ORIGINS, TICK_HZ
from sim.swarm import ROSTER, Swarm

DT = 1.0 / TICK_HZ

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins=CORS_ORIGINS)
swarm = Swarm()
_clients: set[str] = set()


def world_meta() -> dict:
    """Static geometry the dashboard needs once to draw the maze + roster."""
    return {
        "bounds": swarm.world.bounds,
        "comm_range": swarm.topology.comm_range,
        "obstacles": [
            {"cx": cx, "cy": cy, "w": w, "h": h} for (cx, cy, w, h) in swarm.world.rects
        ],
        "roster": [{"id": i, "type": k} for (i, k) in ROSTER],
        "tick_hz": int(TICK_HZ),
    }


async def sim_loop() -> None:
    """Fixed-rate simulation + broadcast loop with drift correction."""
    next_t = time.perf_counter()
    while True:
        payload = swarm.step(DT)
        if _clients:
            await sio.emit("telemetry", payload)
        next_t += DT
        delay = next_t - time.perf_counter()
        if delay < -0.5:          # fell badly behind — resync clock
            next_t = time.perf_counter()
            delay = 0
        await sio.sleep(max(0.0, delay))


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = sio.start_background_task(sim_loop)
    print(f"[SwarmResQ] simulation running @ {TICK_HZ:.0f} Hz — {len(ROSTER)} agents")
    yield
    task  # task is cancelled with the loop on shutdown


fastapi_app = FastAPI(title="SwarmResQ Telemetry", lifespan=lifespan)


@fastapi_app.get("/")
async def health() -> dict:
    return {"status": "online", "service": "SwarmResQ", "tick": swarm.tick,
            "agents": len(swarm.agents), "clients": len(_clients)}


# ---------------------------------------------------------------------- #
# Socket.io event handlers
# ---------------------------------------------------------------------- #
@sio.event
async def connect(sid, environ, auth=None):
    _clients.add(sid)
    await sio.emit("hello", world_meta(), to=sid)
    # Snapshot the already-discovered map so reloads/late joiners see the full map.
    await sio.emit("map", {"points": swarm.map_points}, to=sid)
    print(f"[SwarmResQ] client connected: {sid} ({len(_clients)} total)")


@sio.event
async def disconnect(sid):
    _clients.discard(sid)
    print(f"[SwarmResQ] client disconnected: {sid} ({len(_clients)} total)")


@sio.on("simulate_emp")
async def on_emp(sid, data=None):
    fraction = 0.4
    if isinstance(data, dict) and "fraction" in data:
        fraction = float(data["fraction"])
    alert = swarm.simulate_emp(fraction)
    await sio.emit("alert", alert)


@sio.on("kill_node")
async def on_kill(sid, data=None):
    if isinstance(data, dict) and data.get("id"):
        alert = swarm.kill_node(str(data["id"]))
        await sio.emit("alert", alert)


@sio.on("reset")
async def on_reset(sid, data=None):
    alert = swarm.reset()
    await sio.emit("hello", world_meta())
    await sio.emit("map", {"points": swarm.map_points})
    await sio.emit("alert", alert)


# The ASGI entrypoint (uvicorn app:app)
app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)

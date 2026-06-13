"""Connect to the running server, collect telemetry, fire EMP, verify round-trip."""
import asyncio

import socketio

sio = socketio.AsyncClient()
frames = []
meta = {}
alerts = []


@sio.on("hello")
async def on_hello(data):
    meta.update(data)
    print("HELLO:", {k: (len(v) if isinstance(v, list) else v) for k, v in data.items()})


@sio.on("telemetry")
async def telemetry(data):
    frames.append(data)


@sio.on("alert")
async def alert(data):
    alerts.append(data)
    print("ALERT:", data["msg"])


async def main():
    await sio.connect("http://127.0.0.1:8000", transports=["websocket"])
    await asyncio.sleep(2.0)
    print(f"frames in 2s: {len(frames)} (~{len(frames)/2:.0f} Hz)")
    await sio.emit("simulate_emp", {"fraction": 0.4})
    await asyncio.sleep(1.5)
    last = frames[-1]
    m = last["metrics"]
    print("after EMP -> active:", m["agents_active"], "down:", m["agents_down"],
          "edges:", len(last["edges"]), "comps:", len(last["components"]),
          "err:", m["mean_loc_error_cm"], "cm")
    print("sample agent:", {k: last["agents"][0][k] for k in ("id", "type", "status", "pos_est", "loc_error_cm")})
    print("survivors:", last["survivors"][:2])
    print("pointcloud_delta size:", len(last["pointcloud_delta"]))
    await sio.emit("reset")
    await asyncio.sleep(0.5)
    await sio.disconnect()
    assert len(frames) > 40, "expected ~30Hz telemetry"
    assert any(a["level"] == "critical" for a in alerts), "EMP alert missing"
    print("\nCLIENT_TEST_OK")


asyncio.run(main())

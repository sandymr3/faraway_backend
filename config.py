"""
config.py — all runtime settings loaded from environment variables.

Copy .env.example → .env and set values before deploying.
Uvicorn reads the HOST/PORT used at startup; the rest are consumed here.
"""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in cwd if present — no-op in prod where env vars are set directly

# ── Server ──────────────────────────────────────────────────────────────────
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))

# Comma-separated list of allowed frontend origins, e.g.
#   CORS_ORIGINS=https://swarmresq.vercel.app,http://localhost:3000
_raw_origins = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS: list[str] | str = (
    "*" if _raw_origins.strip() == "*"
    else [o.strip() for o in _raw_origins.split(",") if o.strip()]
)

# ── Simulation ───────────────────────────────────────────────────────────────
TICK_HZ: float = float(os.getenv("TICK_HZ", "30"))

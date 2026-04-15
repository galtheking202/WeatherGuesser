"""
server.py — WeatherGuesser API server
======================================
FastAPI backend exposing two endpoints:
  GET /api/data   — fetch live NWS obs + run both models, returns JSON
  GET /           — serves the frontend SPA

Run:
  python server.py
  (or: uvicorn server:app --reload --port 8000)
"""

import math
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path
from threading import Lock

import requests
import torch
import uvicorn
from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent))

from postprocess_model import (
    _rh_from_t_td,
    fetch_forecast_today,
    load_model as load_mos_model,
    nwp_features_from_response,
    predict as mos_predict,
)
from train_nws_model import DEVICE, load_model as load_nws_model, obs_to_features

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STID         = "LLBG"
NWS_TOKEN    = os.environ["NWS_TOKEN"]
SYNOPTIC_URL = "https://api.synopticdata.com/v2/stations/timeseries"
OBS_TTL      = 180       # seconds — NWS API cache
NWP_TTL      = 6 * 3600  # seconds — NWP cache
PORT         = 8000
STATIC_DIR   = Path(__file__).parent / "frontend" / "dist"

app = FastAPI(title="WeatherGuesser")

# ---------------------------------------------------------------------------
# Model loading (once at startup)
# ---------------------------------------------------------------------------
print("Loading models...")
mos_bundle = load_mos_model()
nws_bundle = load_nws_model()
print(f"  mos_model loaded | nws_model loaded | device={DEVICE}")

# ---------------------------------------------------------------------------
# Simple TTL cache (thread-safe)
# ---------------------------------------------------------------------------
_cache: dict = {}
_cache_lock = Lock()


def _cached(key: str, ttl: int, fn):
    now = time.time()
    with _cache_lock:
        entry = _cache.get(key)
        if entry and now - entry["ts"] < ttl:
            return entry["val"], entry.get("err")
    val, err = fn()
    with _cache_lock:
        _cache[key] = {"ts": now, "val": val, "err": err}
    return val, err


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------
def _fetch_obs_now(today_str: str):
    params = {
        "STID": STID, "recent": 36 * 60,
        "units": "temp|C,speed|kph,metric",
        "complete": 1, "obtimezone": "local",
        "showemptystations": 1, "token": NWS_TOKEN,
    }
    headers = {"Referer": "https://www.weather.gov/", "Origin": "https://www.weather.gov"}
    try:
        resp    = requests.get(SYNOPTIC_URL, params=params, headers=headers, timeout=20)
        payload = resp.json()
    except Exception as e:
        return {}, str(e)

    if payload.get("SUMMARY", {}).get("RESPONSE_CODE") != 1:
        return {}, payload.get("SUMMARY", {}).get("RESPONSE_MESSAGE", "API error")

    obs_raw = payload["STATION"][0]["OBSERVATIONS"]
    times   = obs_raw["date_time"]

    def _col(key):
        for sfx in ("_set_1", "_set_1d", ""):
            v = obs_raw.get(key + sfx)
            if v is not None:
                return v
        return [None] * len(times)

    air_temps = _col("air_temp")
    dew_pts   = _col("dew_point_temperature")
    wind_spds = _col("wind_speed")
    wind_dirs = _col("wind_direction")
    rel_hums  = _col("relative_humidity")
    pressures = _col("sea_level_pressure")

    today = date.fromisoformat(today_str)
    mos_recs, nws_recs = [], []

    for i, ts in enumerate(times):
        if len(ts) > 5 and ts[-5] in ("+", "-") and ":" not in ts[-5:]:
            ts = ts[:-2] + ":" + ts[-2:]
        try:
            dt = datetime.fromisoformat(ts).replace(tzinfo=None)
        except ValueError:
            continue
        if dt.date() != today:
            continue

        t = float(air_temps[i]) if air_temps[i] is not None else None
        if t is None:
            continue

        dp    = float(dew_pts[i])   if dew_pts[i]   is not None else None
        rh    = float(rel_hums[i])  if rel_hums[i]  is not None else (
                _rh_from_t_td(t, dp) if dp is not None else None)
        wd    = float(wind_dirs[i]) if wind_dirs[i] is not None else 0.0
        ws    = float(wind_spds[i]) / 3.6 if wind_spds[i] is not None else 0.0
        p_raw = float(pressures[i]) if pressures[i] is not None else None
        p     = p_raw / 100.0 if p_raw is not None else 1013.25

        if rh is not None:
            mos_recs.append({"dt": dt.isoformat(), "TD": t, "RH": rh, "WD": wd, "WS": ws})
        nws_recs.append({"dt": dt.isoformat(), "t": t, "dp": dp, "ws": ws, "wd": wd, "p": p})

    mos_recs.sort(key=lambda r: r["dt"])
    nws_recs.sort(key=lambda r: r["dt"])
    return {"mos_recs": mos_recs, "nws_recs": nws_recs}, None


def _fetch_nwp_now(today_str: str):
    try:
        raw = fetch_forecast_today(today_str)
        return nwp_features_from_response(raw), None
    except Exception as e:
        return {}, str(e)


# ---------------------------------------------------------------------------
# Prediction runners
# ---------------------------------------------------------------------------
def _run_nws(nws_recs_raw: list) -> dict | None:
    if not nws_recs_raw:
        return None
    # Convert ISO strings back to datetime for obs_to_features
    recs = [{**r, "dt": datetime.fromisoformat(r["dt"])} for r in nws_recs_raw]
    model, s_mean, s_std, q = nws_bundle
    seq = [obs_to_features(r) for r in recs]
    sc  = [[(v - m) / s for v, m, s in zip(f, s_mean, s_std)] for f in seq]
    x   = torch.tensor([sc], dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        mu_t, ls_t = model(x)
    mu    = mu_t[0, -1].item()
    sigma = ls_t[0, -1].exp().item()
    ci    = q * sigma
    return {
        "mu":      round(mu, 1),
        "sigma":   round(sigma, 3),
        "ci_low":  round(mu - ci, 1),
        "ci_high": round(mu + ci, 1),
        "n_obs":   len(recs),
        "last_t":  recs[-1]["t"],
        "last_dt": recs[-1]["dt"].strftime("%H:%M"),
    }


def _run_mos(mos_recs_raw: list, nwp_daily: dict, today: date) -> dict | None:
    if not mos_recs_raw:
        return None
    ds = today.strftime("%Y-%m-%d")
    if ds not in nwp_daily:
        return {"error": "NWP data not available"}
    recs = [{**r, "dt": datetime.fromisoformat(r["dt"])} for r in mos_recs_raw]
    model, s_mean, s_std, q = mos_bundle
    return mos_predict(model, s_mean, s_std, q, recs, nwp_daily[ds], today)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.get("/api/data")
def get_data():
    today     = date.today()
    today_str = today.isoformat()
    now       = datetime.now()

    obs_data, obs_err = _cached("obs", OBS_TTL, lambda: _fetch_obs_now(today_str))
    nwp_data, nwp_err = _cached("nwp", NWP_TTL, lambda: _fetch_nwp_now(today_str))

    mos_recs = obs_data.get("mos_recs", [])
    nws_recs = obs_data.get("nws_recs", [])

    nws_res = _run_nws(nws_recs)
    try:
        mos_res = _run_mos(mos_recs, nwp_data, today)
    except Exception as e:
        mos_res = {"error": str(e)}

    # Latest observation for header
    current = nws_recs[-1] if nws_recs else None
    trend   = None
    if len(nws_recs) >= 2:
        delta = nws_recs[-1]["t"] - nws_recs[-2]["t"]
        trend = {"delta": round(delta, 1), "label": "Rising" if delta > 0 else ("Falling" if delta < 0 else "Steady")}

    return {
        "server_time": now.strftime("%H:%M:%S"),
        "today":       today_str,
        "obs_err":     obs_err,
        "nwp_err":     nwp_err,
        "current":     current,
        "trend":       trend,
        "nws_recs":    nws_recs,
        "lstm":        nws_res,
        "mos":         mos_res,
    }


# ---------------------------------------------------------------------------
# Static file serving (frontend)
# ---------------------------------------------------------------------------
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

@app.get("/{full_path:path}")
def serve_spa(full_path: str):
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"error": "Frontend not built. Run: cd frontend && npm run build"}, status_code=503)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=False)

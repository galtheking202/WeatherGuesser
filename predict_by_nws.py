"""
predict_by_nws.py
=================
Run the pred_by_nws LSTM on the live NWS observation stream for today
and print a continuously-updating prediction for today's maximum temperature.

Fetches all of today's observations from the Synoptic/NWS API (no pandas,
no personal API key needed) and feeds the full sequence to the trained LSTM.
The prediction improves as more of the day is observed.

Usage
-----
  python predict_by_nws.py          # predict using all of today's obs so far
"""

import sys, math, os, requests
from datetime import date, datetime
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from train_nws_model import load_model, obs_to_features
import torch

# ---------------------------------------------------------------------------
# Synoptic / NWS API constants  (same public token as predict_today.py)
# ---------------------------------------------------------------------------
SYNOPTIC_URL = "https://api.synopticdata.com/v2/stations/timeseries"
NWS_TOKEN    = os.environ["NWS_TOKEN"]
STID         = "LLBG"


# ---------------------------------------------------------------------------
# Fetch today's observations
# ---------------------------------------------------------------------------

def fetch_today_obs(stid=STID, recent_hours=36) -> list:
    """
    Pull recent observations and filter to today's calendar date.
    Returns a list of obs dicts: {dt, t, dp, ws, wd, p}
    matching the format expected by obs_to_features().
      t  — air temp   (°C)
      dp — dew point  (°C)
      ws — wind speed (m/s, converted from kph)
      wd — wind dir   (degrees)
      p  — SLP        (hPa, converted from Pa)
    """
    params = {
        "STID":              stid,
        "recent":            recent_hours * 60,
        "units":             "temp|C,speed|kph,metric",
        "complete":          1,
        "obtimezone":        "local",
        "showemptystations": 1,
        "token":             NWS_TOKEN,
    }
    headers = {
        "Referer": "https://www.weather.gov/",
        "Origin":  "https://www.weather.gov",
    }
    resp = requests.get(SYNOPTIC_URL, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("SUMMARY", {}).get("RESPONSE_CODE") != 1:
        raise ValueError(payload.get("SUMMARY", {}).get("RESPONSE_MESSAGE"))

    stations = payload.get("STATION", [])
    if not stations:
        return []

    obs_raw = stations[0].get("OBSERVATIONS", {})
    times   = obs_raw.get("date_time", [])

    def _col(key):
        for sfx in ("_set_1", "_set_1d", ""):
            v = obs_raw.get(key + sfx)
            if v is not None:
                return v
        return [None] * len(times)

    temps = _col("air_temp")
    dps   = _col("dew_point_temperature")
    wss   = _col("wind_speed")
    wds   = _col("wind_direction")
    ps    = _col("sea_level_pressure")

    records = []
    today   = date.today()
    for i, ts in enumerate(times):
        try:
            if len(ts) > 5 and ts[-5] in ('+', '-') and ':' not in ts[-5:]:
                ts = ts[:-2] + ":" + ts[-2:]
            dt = datetime.fromisoformat(ts).replace(tzinfo=None)
        except ValueError:
            continue

        if dt.date() != today:
            continue

        t = float(temps[i]) if temps[i] is not None else None
        if t is None:
            continue

        dp_raw = dps[i]
        dp     = float(dp_raw) if dp_raw is not None else None

        ws_raw = wss[i]
        ws     = float(ws_raw) / 3.6 if ws_raw is not None else 0.0   # kph → m/s

        wd_raw = wds[i]
        wd     = float(wd_raw) if wd_raw is not None else 0.0

        p_raw = ps[i]
        p     = float(p_raw) / 100.0 if p_raw is not None else 1013.25  # Pa → hPa

        records.append({"dt": dt, "t": t, "dp": dp, "ws": ws, "wd": wd, "p": p})

    records.sort(key=lambda r: r["dt"])
    return records


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_now(model, scaler_mean, scaler_std, conformal_q, obs_list) -> dict:
    """
    Feed the full observation sequence to the LSTM and return the
    prediction from the last hidden state (most information).
    """
    seq = [obs_to_features(r) for r in obs_list]
    sc  = [[(v - m) / s for v, m, s in zip(feat, scaler_mean, scaler_std)]
           for feat in seq]

    x = torch.tensor([sc], dtype=torch.float32)   # (1, T, F)
    model.eval()
    with torch.no_grad():
        mu_t, ls_t = model(x)

    mu    = mu_t[0, -1].item()
    sigma = ls_t[0, -1].exp().item()
    ci    = conformal_q * sigma

    return {
        "mu":      round(mu, 1),
        "sigma":   round(sigma, 3),
        "ci_low":  round(mu - ci, 1),
        "ci_high": round(mu + ci, 1),
        "n_obs":   len(obs_list),
        "last_time": obs_list[-1]["dt"].strftime("%H:%M"),
        "last_temp": obs_list[-1]["t"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Fetching today's observations from {STID}...")
    obs = fetch_today_obs()

    if not obs:
        sys.exit("ERROR: No observations for today found in NWS feed.")

    print(f"  {len(obs)} observations  "
          f"(midnight → {obs[-1]['dt'].strftime('%H:%M')})")

    from train_nws_model import MODEL_SAVE
    print(f"Loading {MODEL_SAVE.name}...")
    model, s_mean, s_std, q = load_model()

    result = predict_now(model, s_mean, s_std, q, obs)

    print(f"\n{'='*52}")
    print(f"  Date            : {date.today()}")
    print(f"  Observations    : {result['n_obs']}  (last at {result['last_time']})")
    print(f"  Current temp    : {result['last_temp']} C")
    print(f"{'='*52}")
    print(f"  Predicted max   : {result['mu']} C")
    print(f"  Uncertainty σ   : {result['sigma']} C")
    print(f"  95% PI          : [{result['ci_low']} – {result['ci_high']}] C")
    print(f"{'='*52}\n")

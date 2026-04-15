"""
app.py  --  WeatherGuesser Live Prediction Dashboard
=====================================================
Real-time max-temperature predictions for LLBG using two models:
  - MOS model  (mos_model.pt)  NWP + station obs up to 12:00
  - LSTM model (nws_model.pt)  Pure station stream, any time of day

Auto-refresh: page reruns every 60 s (st_autorefresh).
API calls are cached for 3 min (DATA_TTL) so the NWS API is hit at
most once per 3 minutes regardless of how often the page reruns.

Run:  streamlit run app.py
"""

import math
import os
import sys
from datetime import date, datetime
from pathlib import Path

import plotly.graph_objects as go
import requests
import streamlit as st
import torch
from dotenv import load_dotenv

load_dotenv()
from streamlit_autorefresh import st_autorefresh

sys.path.insert(0, str(Path(__file__).parent))

from postprocess_model import (
    _rh_from_t_td,
    fetch_forecast_today,
    load_model as load_mos_model,
    nwp_features_from_response,
    predict as mos_predict,
)
from train_nws_model import (
    DEVICE,
    load_model as load_nws_model,
    obs_to_features,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STID         = "LLBG"
NWS_TOKEN    = os.environ["NWS_TOKEN"]
SYNOPTIC_URL = "https://api.synopticdata.com/v2/stations/timeseries"
DATA_TTL     = 180        # seconds — NWS API cache
NWP_TTL      = 6 * 3600   # seconds — NWP cache
UI_REFRESH   = 60 * 1000  # milliseconds — page auto-refresh

st.set_page_config(
    page_title="WeatherGuesser - LLBG",
    layout="wide",
    page_icon="thermometer",
)

# ---------------------------------------------------------------------------
# Auto-refresh (client-side, does not block the script)
# ---------------------------------------------------------------------------
st_autorefresh(interval=UI_REFRESH, key="live_refresh")

# ---------------------------------------------------------------------------
# Load models once (survives reruns)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_models():
    mos = load_mos_model()
    nws = load_nws_model()
    return mos, nws


# ---------------------------------------------------------------------------
# Cached data fetchers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=DATA_TTL)
def fetch_obs(today_str: str):
    """
    Fetch today's LLBG observations from NWS Synoptic API.
    Cached for DATA_TTL seconds — NWS API is not called more than
    once per 3 minutes regardless of UI refresh rate.
    Returns (mos_recs, nws_recs, error_str | None).
    """
    params = {
        "STID": STID, "recent": 36 * 60,
        "units": "temp|C,speed|kph,metric",
        "complete": 1, "obtimezone": "local",
        "showemptystations": 1, "token": NWS_TOKEN,
    }
    headers = {
        "Referer": "https://www.weather.gov/",
        "Origin":  "https://www.weather.gov",
    }
    try:
        resp    = requests.get(SYNOPTIC_URL, params=params, headers=headers, timeout=20)
        payload = resp.json()
    except Exception as e:
        return [], [], str(e)

    if payload.get("SUMMARY", {}).get("RESPONSE_CODE") != 1:
        return [], [], payload.get("SUMMARY", {}).get("RESPONSE_MESSAGE", "API error")

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
            mos_recs.append({"dt": dt, "TD": t, "RH": rh, "WD": wd, "WS": ws})
        nws_recs.append({"dt": dt, "t": t, "dp": dp, "ws": ws, "wd": wd, "p": p})

    mos_recs.sort(key=lambda r: r["dt"])
    nws_recs.sort(key=lambda r: r["dt"])
    return mos_recs, nws_recs, None


@st.cache_data(ttl=NWP_TTL)
def fetch_nwp(today_str: str):
    try:
        raw = fetch_forecast_today(today_str)
        return nwp_features_from_response(raw), None
    except Exception as e:
        return {}, str(e)


# ---------------------------------------------------------------------------
# Prediction runners
# ---------------------------------------------------------------------------
def run_nws_model(nws_recs, bundle):
    model, s_mean, s_std, q = bundle
    if not nws_recs:
        return None
    seq = [obs_to_features(r) for r in nws_recs]
    sc  = [[(v - m) / s for v, m, s in zip(f, s_mean, s_std)] for f in seq]
    x   = torch.tensor([sc], dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        mu_t, ls_t = model(x)
    mu    = mu_t[0, -1].item()
    sigma = ls_t[0, -1].exp().item()
    ci    = q * sigma
    return {
        "mu": mu, "sigma": sigma,
        "ci_low": mu - ci, "ci_high": mu + ci,
        "n_obs": len(nws_recs),
        "last_t": nws_recs[-1]["t"],
        "last_dt": nws_recs[-1]["dt"],
    }


def run_mos_model(mos_recs, nwp_daily, today, bundle):
    model, s_mean, s_std, q = bundle
    ds = today.strftime("%Y-%m-%d")
    if not mos_recs or ds not in nwp_daily:
        return None
    result = mos_predict(model, s_mean, s_std, q, mos_recs, nwp_daily[ds], today)
    return result   # may contain "error" key


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
DARK = dict(
    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
    font_color="#fafafa",
    xaxis=dict(gridcolor="#2a2a2a"), yaxis=dict(gridcolor="#2a2a2a"),
)

def temp_chart(nws_recs, nws_res, mos_res):
    times = [r["dt"] for r in nws_recs]
    temps = [r["t"]  for r in nws_recs]
    fig   = go.Figure()

    fig.add_trace(go.Scatter(
        x=times, y=temps, name="Observed",
        line=dict(color="#4A90D9", width=2.5),
        mode="lines+markers", marker=dict(size=4),
    ))

    if nws_res:
        fig.add_hrect(
            y0=nws_res["ci_low"], y1=nws_res["ci_high"],
            fillcolor="rgba(0,180,90,0.13)", line_width=0,
        )
        fig.add_hline(
            y=nws_res["mu"],
            line=dict(color="#00C864", dash="dash", width=1.5),
            annotation_text=f"LSTM {nws_res['mu']:.1f} C",
            annotation_font_color="#00C864",
        )

    if mos_res and "mu" in mos_res:
        fig.add_hrect(
            y0=mos_res["ci_low_95"], y1=mos_res["ci_high_95"],
            fillcolor="rgba(255,110,0,0.11)", line_width=0,
        )
        fig.add_hline(
            y=float(mos_res["mu"]),
            line=dict(color="#FF7020", dash="dot", width=1.5),
            annotation_text=f"MOS {mos_res['mu']} C",
            annotation_font_color="#FF7020",
        )

    fig.update_layout(
        title="Today's Temperature at LLBG",
        xaxis_title="Time", yaxis_title="Temperature (C)",
        height=360, margin=dict(t=50, b=30, l=50, r=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **DARK,
    )
    return fig


def history_chart(history):
    if len(history) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[h["time"] for h in history],
        y=[h["nws_mu"] for h in history],
        name="LSTM", line=dict(color="#00C864", width=2),
        connectgaps=True,
    ))
    fig.add_trace(go.Scatter(
        x=[h["time"] for h in history],
        y=[h["mos_mu"] for h in history],
        name="MOS", line=dict(color="#FF7020", width=2, dash="dot"),
        connectgaps=True,
    ))
    fig.update_layout(
        title="Prediction History (how estimates evolved today)",
        xaxis_title="Time", yaxis_title="Predicted Max (C)",
        height=240, margin=dict(t=50, b=30, l=50, r=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **DARK,
    )
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
today     = date.today()
today_str = today.isoformat()
now       = datetime.now()

mos_bundle, nws_bundle = get_models()
mos_recs, nws_recs, obs_err = fetch_obs(today_str)
nwp_daily, nwp_err          = fetch_nwp(today_str)

nws_res = run_nws_model(nws_recs, nws_bundle)
mos_res = run_mos_model(mos_recs, nwp_daily, today, mos_bundle)

# Accumulate prediction history in session state
if "history" not in st.session_state:
    st.session_state.history = []

if nws_res or (mos_res and "mu" in mos_res):
    hist = st.session_state.history
    if not hist or (now - hist[-1]["time"]).total_seconds() > 90:
        hist.append({
            "time":   now,
            "nws_mu": nws_res["mu"] if nws_res else None,
            "mos_mu": mos_res["mu"] if (mos_res and "mu" in mos_res) else None,
        })

# --- Header ---
st.title("WeatherGuesser Live - LLBG")

current = nws_recs[-1] if nws_recs else None
c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 2])

with c1:
    obs_time = current["dt"].strftime("%H:%M") if current else "--:--"
    st.markdown(f"**Last obs:** {obs_time}  |  **Page updated:** {now.strftime('%H:%M:%S')}")
    if obs_err:
        st.error(f"NWS API error: {obs_err}")

with c2:
    val = f"{current['TD']} C" if current else "—"
    st.metric("Temp", val)

with c3:
    dp = nws_recs[-1]["dp"] if nws_recs and nws_recs[-1]["dp"] is not None else None
    st.metric("Dew Pt", f"{dp:.0f} C" if dp is not None else "—")

with c4:
    ws_kph = f"{current['WS']*3.6:.0f} km/h" if current else "—"
    st.metric("Wind", ws_kph)

with c5:
    if len(nws_recs) >= 2:
        delta = nws_recs[-1]["t"] - nws_recs[-2]["t"]
        trend = "Rising" if delta > 0 else ("Falling" if delta < 0 else "Steady")
        st.metric("Trend", trend, f"{delta:+.1f} C")

st.divider()

# --- Model cards ---
col_nws, col_mos = st.columns(2)

with col_nws:
    st.subheader("LSTM  (pred_by_nws)")
    if nws_res:
        st.metric("Predicted Max", f"{nws_res['mu']:.1f} C")
        st.write(f"**95% PI:** {nws_res['ci_low']:.1f} - {nws_res['ci_high']:.1f} C")
        st.write(f"**sigma:** {nws_res['sigma']:.3f} C")
        st.caption(
            f"{nws_res['n_obs']} obs | last at {nws_res['last_dt'].strftime('%H:%M')}"
        )
    else:
        st.info("Waiting for today's first observations...")

with col_mos:
    st.subheader("MOS  (mos_model)")
    if mos_res and "mu" in mos_res:
        conf  = mos_res["confidence"]
        label = {"HIGH": "[HIGH]", "MEDIUM": "[MEDIUM]", "LOW": "[LOW]"}[conf]
        st.metric("Predicted Max", f"{mos_res['mu']} C")
        st.write(f"**95% PI:** {mos_res['ci_low_95']} - {mos_res['ci_high_95']} C")
        st.write(f"**Confidence:** {label}  |  **NWP bias:** {mos_res['nwp_bias_at_12']:+.1f} C")
        if mos_res["reasons"]:
            st.caption("; ".join(mos_res["reasons"]))
    elif mos_res and "error" in mos_res:
        st.info(f"{mos_res['error']}")
    else:
        st.info("Waiting for 12:00 station reading + NWP data...")

# --- Charts ---
if nws_recs:
    st.plotly_chart(temp_chart(nws_recs, nws_res, mos_res), use_container_width=True)

fig_h = history_chart(st.session_state.history)
if fig_h:
    st.plotly_chart(fig_h, use_container_width=True)

st.caption(
    "NWS Synoptic API (LLBG) polled every 3 min  |  "
    "NWP (Open-Meteo) refreshed every 6 h  |  "
    f"UI auto-refreshes every {UI_REFRESH//1000} s"
)

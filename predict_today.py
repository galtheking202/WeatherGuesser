"""
Predict today's max temperature at LLBG using live NWS data + mos_model.pt.

Data flow:
  1. Fetch last 36h directly from the Synoptic/NWS API (replicates nws_timeseries.py
     without pandas, which is blocked by the Application Control policy on this machine)
  2. Convert response rows to the record format expected by postprocess_model.py
  3. Fetch today's NWP forecast from Open-Meteo
  4. Load mos_model.pt and run inference
"""

import sys
import math
import os
import requests
from datetime import date, datetime
from dotenv import load_dotenv

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))
load_dotenv()

from postprocess_model import (
    load_model,
    fetch_forecast_today,
    nwp_features_from_response,
    station_morning_features,
    predict,
)

# ---------------------------------------------------------------------------
# Synoptic fetch (mirrors nws_timeseries.py without pandas)
# ---------------------------------------------------------------------------
SYNOPTIC_URL = "https://api.synopticdata.com/v2/stations/timeseries"
NWS_TOKEN    = os.environ["NWS_TOKEN"]
BEN_GURION_STID = "LLBG"

def _fetch_nws_raw(stid: str = BEN_GURION_STID, recent_hours: int = 36) -> list[dict]:
    """
    Fetch a station timeseries via the NWS/Synoptic API.
    Returns a list of dicts with keys:
        dt (datetime, local, tz-naive), TD (°C), RH (%), WD (°), WS (m/s)
    Wind speed from API is km/h (metric mode); converted here to m/s.
    """
    params = {
        "STID":              stid.upper(),
        "recent":            recent_hours * 60,    # API expects minutes
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

    summary = payload.get("SUMMARY", {})
    if summary.get("RESPONSE_CODE") != 1:
        raise ValueError(f"Synoptic API error: {summary.get('RESPONSE_MESSAGE')}")

    stations = payload.get("STATION", [])
    if not stations:
        return []

    obs = stations[0].get("OBSERVATIONS", {})
    times = obs.get("date_time", [])

    def _col(key):
        """Return values list for a column (set_1 or set_1d suffix)."""
        for suffix in ("_set_1", "_set_1d", ""):
            v = obs.get(key + suffix)
            if v is not None:
                return v
        return [None] * len(times)

    temps = _col("air_temp")
    rhs   = _col("relative_humidity")
    wds   = _col("wind_direction")
    wss   = _col("wind_speed")

    records = []
    for i, ts in enumerate(times):
        try:
            # Normalise "+0300" -> "+03:00" for fromisoformat
            if len(ts) > 5 and ts[-5] in ('+', '-') and ':' not in ts[-5:]:
                ts = ts[:-2] + ":" + ts[-2:]
            dt = datetime.fromisoformat(ts).replace(tzinfo=None)

            t  = float(temps[i]) if temps[i] is not None else None
            rh = float(rhs[i])   if rhs[i]   is not None else None
            wd = float(wds[i])   if wds[i]   is not None else 0.0
            ws = float(wss[i])   if wss[i]   is not None else 0.0

            if t is None or rh is None:
                continue

            records.append({
                "dt": dt,
                "TD": t,
                "RH": rh,
                "WD": wd,
                "WS": ws / 3.6,   # kph -> m/s (matches training CSV units)
            })
        except (ValueError, TypeError):
            continue

    records.sort(key=lambda r: r["dt"])
    return records

if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # 1.  Fetch today's station observations
    # -----------------------------------------------------------------------
    print(f"Fetching last 36 h from {BEN_GURION_STID} via NWS Synoptic API...")
    all_records = _fetch_nws_raw(BEN_GURION_STID, recent_hours=36)

    if not all_records:
        sys.exit("ERROR: No station data returned from NWS API.")

    print(f"  Got {len(all_records)} records  "
          f"({all_records[0]['dt']}  ->  {all_records[-1]['dt']})")

    # Filter to today's records only
    today      = date.today()
    today_recs = [r for r in all_records if r["dt"].date() == today]

    print(f"  Today ({today}) records: {len(today_recs)}")
    if len(today_recs) < 5:
        print(f"  WARNING: only {len(today_recs)} records for today; need >=5 up to 12:00.")
        print(f"  Latest record: {all_records[-1]['dt'] if all_records else 'none'}")
        available_dates = sorted({r["dt"].date() for r in all_records})
        print(f"  Available dates in fetched data: {available_dates}")
        for d in reversed(available_dates):
            cands = [r for r in all_records if r["dt"].date() == d]
            if len(cands) >= 5:
                print(f"  Falling back to {d} ({len(cands)} records).")
                today      = d
                today_recs = cands
                break
        else:
            sys.exit("ERROR: Not enough station records on any recent date.")

    # -----------------------------------------------------------------------
    # 2.  Fetch NWP forecast for today
    # -----------------------------------------------------------------------
    today_iso = today.strftime("%Y-%m-%d")
    print(f"\nFetching Open-Meteo NWP forecast for {today_iso}...")
    nwp_raw   = fetch_forecast_today(today_iso)
    nwp_daily = nwp_features_from_response(nwp_raw)

    if today_iso not in nwp_daily:
        sys.exit(f"ERROR: NWP data not available for {today_iso}.")
    print("  NWP fetch OK")

    # -----------------------------------------------------------------------
    # 3.  Load model and predict
    # -----------------------------------------------------------------------
    print("\nLoading mos_model.pt...")
    model, s_mean, s_std, q = load_model()
    print("  Model loaded OK")

    result = predict(model, s_mean, s_std, q,
                     today_recs, nwp_daily[today_iso], today)

    # -----------------------------------------------------------------------
    # 4.  Print result
    # -----------------------------------------------------------------------
    if "error" in result:
        sys.exit(f"\nPrediction error: {result['error']}")

    conf_sym = {"HIGH": "***", "MEDIUM": "**", "LOW": "!"}[result["confidence"]]
    print(f"\n{'='*52}")
    print(f"  Date              : {result['date']}")
    print(f"  Station at 12:00  : {result['temp_at_12:00']} C")
    print(f"  NWP forecast max  : {result['nwp_tmax']} C")
    print(f"  NWP at 12:00      : {result['nwp_t12']} C")
    print(f"  NWP bias at 12:00 : {result['nwp_bias_at_12']:+.2f} C")
    print(f"  NWP cloud variab. : {result['nwp_cloud_std']:.1f}%")
    print(f"{'='*52}")
    print(f"  Predicted max (mu): {result['mu']} C")
    print(f"  Uncertainty sigma : {result['sigma']} C")
    print(f"  95% PI            : [{result['ci_low_95']} - {result['ci_high_95']}] C")
    print(f"  PI width          : +/-{result['ci_width_95']/2:.2f} C")
    print(f"  Confidence        : {conf_sym} {result['confidence']}")
    if result["reasons"]:
        print(f"  Notes             : {'; '.join(result['reasons'])}")
    print(f"{'='*52}\n")

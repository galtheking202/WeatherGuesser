"""
NWS WRH Timeseries API client
==============================
Replicates exactly what https://www.weather.gov/wrh/timeseries?site=LLBG does
under the hood: it calls the Synoptic Data API using NWS's own public token
(scraped from /source/wrh/apiKey.js — no personal API key required).

Endpoint  : https://api.synopticdata.com/v2/stations/timeseries
NWS token : loaded from NWS_TOKEN env var / .env file

URL parameters (same as the NWS page):
  STID          - Station ID (ICAO / Synoptic / METAR)
  recent        - Minutes back from now  (NWS page default = numHours * 60)
  start / end   - Alternate to recent; format YYYYMMDDHHMI  (historical mode)
  units         - 'temp|F,speed|mph,english'  |  'temp|C,...,metric'
  complete      - 1  (return all sensor sets)
  obtimezone    - 'local'  (timestamps in station local time)
"""

import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

SYNOPTIC_URL = "https://api.synopticdata.com/v2/stations/timeseries"
NWS_TOKEN    = os.environ["NWS_TOKEN"]

BEN_GURION_STID = "LLBG"


def fetch_timeseries(
    stid: str = BEN_GURION_STID,
    recent: int = 72,                           # hours  (converted to minutes internally)
    units: str = "temp|C,speed|kph,metric",     # 'metric' | 'english' | 'english_k' (kts)
    obtimezone: str = "local",
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """
    Fetch a station timeseries exactly as the NWS WRH timeseries page would.

    Parameters
    ----------
    stid        : Station ID (ICAO, METAR, or Synoptic ID), e.g. 'LLBG'.
    recent      : Hours back from now to fetch (ignored when start/end provided).
                  NWS page allows 1–720 h; default 72 h.
    units       : Unit string.  Convenience aliases accepted:
                    'metric'    -> 'temp|C,speed|kph,metric'
                    'english'   -> 'temp|F,speed|mph,english'
                    'english_k' -> 'temp|F,speed|kts,english'
    obtimezone  : 'local' (station local time) or 'UTC'.
    start       : UTC datetime for the beginning of a historical window.
    end         : UTC datetime for the end of a historical window.

    Returns
    -------
    pd.DataFrame with a 'time' column (tz-aware or local per obtimezone) plus
    one column per sensor variable.  Temperature °F columns get a companion °C
    column; °C columns get a companion °F column.
    """
    # Convenience unit aliases
    _unit_map = {
        "metric":    "temp|C,speed|kph,metric",
        "english":   "temp|F,speed|mph,english",
        "english_k": "temp|F,speed|kts,english",
    }
    units = _unit_map.get(units, units)

    params: dict = {
        "STID": stid.upper(),
        "showemptystations": 1,
        "units": units,
        "complete": 1,
        "obtimezone": obtimezone,
        "token": NWS_TOKEN,
    }

    if start and end:
        # Historical mode — matches NWS page's "Gather Historical Data" toggle
        params["start"] = start.strftime("%Y%m%d%H%M")
        params["end"] = end.strftime("%Y%m%d%H%M")
    else:
        params["recent"] = recent * 60      # Synoptic expects minutes

    # The NWS token is restricted to weather.gov origin requests
    headers = {
        "Referer": "https://www.weather.gov/",
        "Origin": "https://www.weather.gov",
    }
    response = requests.get(SYNOPTIC_URL, params=params, headers=headers, timeout=30)
    response.raise_for_status()

    payload = response.json()
    _check_response(payload)

    stations = payload.get("STATION", [])
    if not stations:
        return pd.DataFrame()

    return _parse_station(stations[0])


def _check_response(payload: dict) -> None:
    summary = payload.get("SUMMARY", {})
    code = summary.get("RESPONSE_CODE")
    if code != 1:
        raise ValueError(
            f"Synoptic API error {code}: {summary.get('RESPONSE_MESSAGE')}"
        )


def _parse_station(station: dict) -> pd.DataFrame:
    obs = station.get("OBSERVATIONS", {})

    date_times = obs.get("date_time", [])
    if not date_times:
        return pd.DataFrame()

    df = pd.DataFrame({"time": pd.to_datetime(date_times)})

    for key, values in obs.items():
        if key == "date_time":
            continue
        # Strip Synoptic's _set_1 / _set_1d suffixes for clean column names
        col = key.replace("_set_1d", "").replace("_set_1", "")
        df[col] = pd.to_numeric(values, errors="coerce")

    # Add convenience temperature conversion columns
    if "air_temp" in df.columns:
        # Metric response → values are °C; add °F companion
        df["air_temp_f"] = df["air_temp"] * 9 / 5 + 32
    if "air_temp_f" in df.columns and "air_temp" not in df.columns:
        # English response → values are °F; add °C companion
        df["air_temp_c"] = (df["air_temp_f"] - 32) * 5 / 9
    if "dew_point_temperature" in df.columns:
        df["dew_point_temperature_f"] = df["dew_point_temperature"] * 9 / 5 + 32

    df = df.sort_values("time").reset_index(drop=True)
    return df


def fetch_latest(stid: str = BEN_GURION_STID) -> pd.Series | None:
    """Return the most recent observation row as a Series, or None if unavailable."""
    df = fetch_timeseries(stid=stid, recent=2)
    return None if df.empty else df.iloc[-1]


if __name__ == "__main__":
    print(f"Fetching last 72 h timeseries from {BEN_GURION_STID} via NWS token...")
    df = fetch_timeseries()

    if df.empty:
        print("No data returned.")
    else:
        print(df.tail(10).to_string(index=False))
        print(f"\nTotal records : {len(df)}")
        print(f"Columns       : {list(df.columns)}")

        latest = df.iloc[-1]
        print("\nLatest observation:")
        print(f"  Time        : {latest['time']}")
        for col in ["air_temp", "air_temp_f", "dew_point_temperature",
                    "relative_humidity", "wind_speed", "wind_direction",
                    "wind_gust", "pressure"]:
            if col in latest and pd.notna(latest[col]):
                print(f"  {col:<26}: {latest[col]}")

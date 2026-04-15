import requests
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.synopticdata.com/v2/stations/timeseries"
TOKEN = os.getenv("SYNOPTIC_TOKEN")

# Ben Gurion Airport ICAO station ID
BEN_GURION_STID = "LLBG"


def fetch_timeseries(
    stid: str = BEN_GURION_STID,
    recent: int = 4320,          # minutes back from now (4320 = 72h)
    units: str = "temp|C,speed|kph,metric",
    obtimezone: str = "local",
) -> pd.DataFrame:
    """
    Fetch a station timeseries from Synoptic Data.

    Parameters
    ----------
    stid      : Station ID (ICAO, METAR, or Synoptic ID)
    recent    : How many minutes back to fetch
    units     : Unit string in Synoptic format
    obtimezone: 'local' or 'UTC'
    """
    params = {
        "STID": stid,
        "showemptystations": 1,
        "units": units,
        "recent": recent,
        "complete": 1,
        "obtimezone": obtimezone,
        "token": TOKEN,
    }

    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    payload = response.json()

    summary = payload.get("SUMMARY", {})
    if summary.get("RESPONSE_CODE") != 1:
        raise ValueError(f"Synoptic API error: {summary.get('RESPONSE_MESSAGE')}")

    stations = payload.get("STATION", [])
    if not stations:
        print("No stations returned.")
        return pd.DataFrame()

    station = stations[0]
    obs = station.get("OBSERVATIONS", {})

    # Build a DataFrame from the observation arrays
    date_times = pd.to_datetime(obs.get("date_time", []))
    df = pd.DataFrame({"time": date_times})

    # Map every variable that came back
    skip = {"date_time"}
    for key, values in obs.items():
        if key in skip:
            continue
        col = key.replace("_set_1", "").replace("_set_1d", "")
        df[col] = values

    df = df.sort_values("time").reset_index(drop=True)
    return df


if __name__ == "__main__":
    print(f"Fetching last 72h from {BEN_GURION_STID}...")
    df = fetch_timeseries()
    print(df.head(10).to_string(index=False))
    print(f"\nTotal records : {len(df)}")
    print(f"Columns       : {list(df.columns)}")

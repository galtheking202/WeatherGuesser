import requests
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Ben Gurion Airport
LAT = 32.0114
LON = 34.8867
TIMEZONE = "Asia/Jerusalem"

# NCEI ISD — hourly archive (lags ~6-8 months behind)
NCEI_BASE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
NCEI_STATION = "40179099999"


def _parse_tmp(raw: str) -> float | None:
    """Parse ISD TMP field e.g. '+0130,1' -> 13.0 (Celsius)."""
    try:
        value = raw.split(",")[0]
        celsius = int(value) / 10
        return None if celsius == 999.9 else celsius
    except Exception:
        return None


def fetch_hourly_archive(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch hourly data from NCEI ISD archive (good for historical, lags ~8 months)."""
    params = {
        "dataset": "global-hourly",
        "stations": NCEI_STATION,
        "startDate": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "endDate": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "format": "json",
    }
    response = requests.get(NCEI_BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    raw = response.json()
    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw)
    df["time"] = pd.to_datetime(df["DATE"])
    df["temp_c"] = df["TMP"].apply(_parse_tmp)
    return df[["time", "temp_c"]].dropna().sort_values("time").reset_index(drop=True)


def fetch_hourly_realtime(date: datetime.date) -> pd.DataFrame:
    """Fetch hourly temperature from Open-Meteo (real-time, no API key needed)."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m",
        "timezone": TIMEZONE,
        "start_date": date.isoformat(),
        "end_date": date.isoformat(),
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    times = pd.to_datetime(data["hourly"]["time"])
    temps = data["hourly"]["temperature_2m"]

    df = pd.DataFrame({"time": times, "temp_c": temps}).dropna()
    return df


def plot_temperature(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(df["time"], df["temp_c"], marker="o", linewidth=2,
            markersize=4, color="#e05c2a", zorder=3)
    ax.fill_between(df["time"], df["temp_c"], alpha=0.15, color="#e05c2a")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    fig.autofmt_xdate()

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (local)")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True, linestyle="--", alpha=0.4)

    if not df.empty:
        peak = df.loc[df["temp_c"].idxmax()]
        ax.annotate(f"Max: {peak['temp_c']:.1f}°C",
                    xy=(peak["time"], peak["temp_c"]),
                    xytext=(10, 10), textcoords="offset points",
                    fontsize=9, color="#c0392b",
                    arrowprops=dict(arrowstyle="->", color="#c0392b"))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    today = datetime.now().date()
    print(f"Fetching hourly temperature for Ben Gurion Airport — {today}...")

    df = fetch_hourly_realtime(today)

    if df.empty:
        print("No data returned.")
    else:
        print(df.to_string(index=False))
        plot_temperature(df, f"Ben Gurion Airport — Hourly Temperature ({today})")

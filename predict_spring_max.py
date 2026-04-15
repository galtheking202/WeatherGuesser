"""
Spring Max Temperature Model — LLBG (Ben Gurion Airport)
=========================================================
Predicts the daily maximum temperature using observations
available by 10:00 AM local time.

Trained on  : LLBG Synoptic data, Mar-Jun 2019-2023
Target months: March, April, May, June  (primary: April)
Features    : t10, td10, tdep10, rise, accel, ws10,
              wd_east, slp_hpa, ceiling_km, month

Backtest (2024-2026 test set):
  All months: see train_spring_model.py output
  April:      see train_spring_model.py output
"""

import math, json
from datetime import datetime
from collections import defaultdict
from pathlib import Path

FEATURE_NAMES = ["t10", "td10", "tdep10", "rise", "accel", "ws10", "wd_east", "slp_hpa", "ceiling_km", "month"]

COEFS = {
    "t10": 3.259854,
    "td10": 1.652950,
    "tdep10": 1.822158,
    "rise": -0.253260,
    "accel": 1.235144,
    "ws10": -0.143502,
    "wd_east": 0.209364,
    "slp_hpa": -0.219608,
    "ceiling_km": 0.389253,
    "month": -0.232092,
}
INTERCEPT = 26.288525

SCALER_MEAN  = [22.613115, 12.324066, 10.289049, 5.618033, 0.921803, 2.867148, -0.205083, 1012.939629, 4.968058, 4.491803]
SCALER_SCALE = [5.500327, 4.628039, 5.641853, 2.857260, 0.555420, 1.959045, 0.637976, 4.332985, 4.613728, 1.118004]

MONTHLY_SIGMA = {3: 1.338, 4: 1.329, 5: 0.78, 6: 0.854}
CLEAR_CEILING_KM = 10.0
SLP_PA_TO_HPA    = 1e-2


def _predict_raw(features: dict) -> float:
    """Predict from a feature dict.  Features must match FEATURE_NAMES."""
    total = INTERCEPT
    for i, name in enumerate(FEATURE_NAMES):
        x_scaled = (features[name] - SCALER_MEAN[i]) / SCALER_SCALE[i]
        total += COEFS[name] * x_scaled
    return total


def predict(features: dict) -> dict:
    """
    Predict daily max temperature.

    Parameters — all values at ~10:00 local time
    ----------
    t10        : air temperature (°C)
    td10       : dew point temperature (°C)
    tdep10     : dew point depression = t10 - td10  (computed automatically if omitted)
    rise       : temp rise from 06:00 to 10:00 (°C)
    accel      : heating slope (°C per 30-min step)
    ws10       : wind speed (m/s)
    wd_east    : sin(wind_direction_degrees)
    slp_hpa    : sea-level pressure (hPa)
    ceiling_km : cloud ceiling (km); pass 10.0 for clear sky
    month      : calendar month (3–6)

    Returns dict with predicted_max, low_80, high_80, sigma.
    """
    f = dict(features)
    if "tdep10" not in f:
        f["tdep10"] = f["t10"] - f["td10"]

    pred  = _predict_raw(f)
    sigma = MONTHLY_SIGMA.get(f["month"], 1.0)
    margin = 1.28 * sigma  # 80% CI

    return {
        "predicted_max": round(pred, 1),
        "low_80":        round(pred - margin, 1),
        "high_80":       round(pred + margin, 1),
        "sigma":         sigma,
    }

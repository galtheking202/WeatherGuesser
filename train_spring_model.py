"""
Spring Max Temperature Model - LLBG (Ben Gurion Airport)
=========================================================
Trains a Ridge regression model to predict the daily maximum
temperature using observations available by 10:00 AM local time.

Target months : March, April, May, June  (primary focus: April)
Data source   : Synoptic/ASOS  (~30-min resolution)
Train period  : 2019-2023
Test period   : 2024-2026 (backtest, >=50 April days)

Features at ~10:00 local time
------------------------------
  t10       air temperature (°C)
  td10      dew point temperature (°C)
  tdep10    dew point depression = t10 - td10
  rise      temp rise from 06:00 to 10:00 (°C)
  accel     heating slope — linear fit over last 4 pre-10:00 readings
  ws10      wind speed (m/s)
  wd_east   sin(wind direction) — easterly component (+east, -west)
  slp_hpa   sea-level pressure (hPa)
  ceiling_km ceiling (km); 10.0 when clear / unreported
  month     calendar month (3–6)
"""

import csv, math, json, statistics
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
DATA_FILE  = Path("training_data_synoptic_model/LLBG.2026-04-14 (1).csv")
MODEL_OUT  = Path("spring_model_coeffs.json")
PREDICT_OUT = Path("predict_spring_max.py")

TRAIN_YEARS = {2019, 2020, 2021, 2022, 2023}
TEST_YEARS  = {2024, 2025, 2026}
TARGET_MONTHS = {3, 4, 5, 6}

FEATURE_NAMES = [
    "t10", "td10", "tdep10", "rise", "accel",
    "ws10", "wd_east", "slp_hpa", "ceiling_km", "month",
]

CLEAR_CEILING_KM = 10.0   # fill-value when ceiling not reported
SLP_PA_TO_HPA    = 1e-2


# ===========================================================================
# 1.  PARSE CSV
# ===========================================================================

def parse_csv(path: Path) -> list[dict]:
    """
    Read the Synoptic CSV, skip comment lines and units row.
    Returns list of dicts with typed fields.
    """
    TZ_LOCAL = timezone(timedelta(hours=3))   # Israel Standard Time +0300

    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = None
        for raw in reader:
            if not raw:
                continue
            line = raw[0].strip()
            if line.startswith("#"):
                continue
            if headers is None:
                headers = [h.strip() for h in raw]
                continue
            # Second row is units — skip it
            if raw[1].strip() in ("", "Celsius", "m/s"):
                continue

            row = dict(zip(headers, [v.strip() for v in raw]))

            # Parse timestamp  (format: 2025-04-01T10:20:00+0300)
            ts_str = row.get("Date_Time", "")
            try:
                # fromisoformat handles +0300 in Python 3.7+
                dt = datetime.fromisoformat(ts_str)
                # convert to local naive (strip tz, keep local clock)
                dt_local = dt.astimezone(TZ_LOCAL).replace(tzinfo=None)
            except ValueError:
                continue

            month = dt_local.month
            if month not in TARGET_MONTHS:
                continue

            def _f(key):
                v = row.get(key, "").strip()
                if not v or v.lower() == "none":
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None

            rows.append({
                "dt":       dt_local,
                "year":     dt_local.year,
                "month":    month,
                "date":     dt_local.date(),
                "T":        _f("air_temp_set_1"),
                # prefer the derived dew point (_set_1d) — more complete
                "Td":       _f("dew_point_temperature_set_1d") or _f("dew_point_temperature_set_1"),
                "WS":       _f("wind_speed_set_1"),
                "WD":       _f("wind_direction_set_1"),
                "ceiling_m":_f("ceiling_set_1"),
                "slp_pa":   _f("sea_level_pressure_set_1d"),
            })

    return rows


# ===========================================================================
# 2.  FEATURE EXTRACTION  (per day)
# ===========================================================================

def _avg(lst):
    return sum(lst) / len(lst) if lst else None

def _nearest(recs, target_hour, window_minutes=75):
    """Return the record closest to target_hour within ±window_minutes."""
    best, best_gap = None, float("inf")
    for r in recs:
        gap = abs((r["dt"].hour * 60 + r["dt"].minute) - target_hour * 60)
        if gap < best_gap and gap <= window_minutes:
            best, best_gap = r, gap
    return best

def _slope(vals):
    """Linear slope of a sequence (index as x, value as y).  Returns 0 if < 2 points."""
    n = len(vals)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx, my = _avg(xs), _avg(vals)
    denom = sum((x - mx) ** 2 for x in xs)
    return sum((xs[i] - mx) * (vals[i] - my) for i in range(n)) / denom if denom else 0.0

def extract_features(day_recs: list[dict]) -> dict | None:
    """
    Extract the feature vector for one day from its intra-day records.
    Returns None when insufficient / missing data.
    """
    if not day_recs:
        return None

    recs = sorted(day_recs, key=lambda r: r["dt"])

    # --- target: daily max temperature (full day) ---
    valid_T = [r["T"] for r in recs if r["T"] is not None]
    if not valid_T:
        return None
    t_max = max(valid_T)

    # --- 10:00 snapshot (closest valid reading within ±75 min) ---
    pre10 = [r for r in recs
             if r["T"] is not None and r["Td"] is not None
             and (r["dt"].hour < 10 or (r["dt"].hour == 10 and r["dt"].minute <= 30))]

    snap = _nearest(pre10, target_hour=10, window_minutes=75)
    if snap is None or len(pre10) < 4:
        return None

    t10   = snap["T"]
    td10  = snap["Td"]
    ws10  = snap["WS"] if snap["WS"] is not None else 0.0

    # wind direction: use snapshot if available, else nearest valid before 10:00
    wd10 = snap["WD"]
    if wd10 is None:
        for r in reversed(pre10):
            if r["WD"] is not None:
                wd10 = r["WD"]
                break
    wd_east = math.sin(math.radians(wd10)) if wd10 is not None else 0.0

    # ceiling: use snapshot, else nearest valid reading before 10:00
    ceiling_m = snap["ceiling_m"]
    if ceiling_m is None:
        for r in reversed(pre10):
            if r["ceiling_m"] is not None:
                ceiling_m = r["ceiling_m"]
                break
    ceiling_km = (ceiling_m / 1000.0) if ceiling_m is not None else CLEAR_CEILING_KM

    # SLP: prefer snapshot, else nearest valid
    slp_pa = snap["slp_pa"]
    if slp_pa is None:
        for r in reversed(pre10):
            if r["slp_pa"] is not None:
                slp_pa = r["slp_pa"]
                break
    if slp_pa is None:
        return None
    slp_hpa = slp_pa * SLP_PA_TO_HPA

    # --- 06:00 temperature ---
    t6_recs = [r for r in recs
               if r["T"] is not None and 5 <= r["dt"].hour <= 7]
    t6 = _nearest(t6_recs, target_hour=6, window_minutes=90)
    t6_val = t6["T"] if t6 else pre10[0]["T"]
    rise = t10 - t6_val

    # --- heating acceleration (slope of last 4 pre-10:00 temps) ---
    last4 = [r["T"] for r in pre10[-4:] if r["T"] is not None]
    accel = _slope(last4)

    return {
        "t10":        t10,
        "td10":       td10,
        "tdep10":     t10 - td10,
        "rise":       rise,
        "accel":      accel,
        "ws10":       ws10,
        "wd_east":    wd_east,
        "slp_hpa":    slp_hpa,
        "ceiling_km": ceiling_km,
        "month":      recs[0]["month"],
        "date":       recs[0]["date"],
        "year":       recs[0]["year"],
        "t_max":      t_max,
    }


# ===========================================================================
# 3.  BUILD DATASET
# ===========================================================================

def build_dataset(rows: list[dict]) -> list[dict]:
    by_day = defaultdict(list)
    for r in rows:
        by_day[r["date"]].append(r)

    samples = []
    for day, recs in sorted(by_day.items()):
        feat = extract_features(recs)
        if feat is not None:
            samples.append(feat)

    return samples


# ===========================================================================
# 4.  TRAIN
# ===========================================================================

def to_matrix(samples):
    X = np.array([[s[f] for f in FEATURE_NAMES] for s in samples], dtype=float)
    y = np.array([s["t_max"] for s in samples], dtype=float)
    return X, y


def train(train_samples, alpha=1.0):
    X, y = to_matrix(train_samples)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = Ridge(alpha=alpha)
    model.fit(Xs, y)
    return model, scaler


# ===========================================================================
# 5.  BACKTEST
# ===========================================================================

def backtest(model, scaler, test_samples) -> dict:
    results = []
    for s in test_samples:
        x = np.array([[s[f] for f in FEATURE_NAMES]], dtype=float)
        xs = scaler.transform(x)
        pred = float(model.predict(xs)[0])
        err  = pred - s["t_max"]
        results.append({
            "date":      str(s["date"]),
            "month":     s["month"],
            "actual":    round(s["t_max"], 1),
            "predicted": round(pred, 1),
            "error":     round(err, 2),
        })

    def metrics(lst, label):
        if not lst:
            return
        errs = [abs(r["error"]) for r in lst]
        print(f"\n  {label}  (n={len(lst)})")
        print(f"    MAE          : {sum(errs)/len(errs):.3f} °C")
        print(f"    RMSE         : {math.sqrt(sum(e**2 for e in errs)/len(errs)):.3f} °C")
        print(f"    Max error    : {max(errs):.2f} °C")
        print(f"    Within ±0.5°C: {sum(1 for e in errs if e<=0.5)/len(errs)*100:.1f}%")
        print(f"    Within ±1.0°C: {sum(1 for e in errs if e<=1.0)/len(errs)*100:.1f}%")
        print(f"    Bias (mean)  : {sum(r['error'] for r in lst)/len(lst):+.3f} °C")

    april = [r for r in results if r["month"] == 4]
    all_months = results

    print(f"\n{'='*58}")
    print(f"  BACKTEST  ({min(r['date'] for r in results)} to {max(r['date'] for r in results)})")
    print(f"{'='*58}")
    metrics(all_months, "All months (Mar-Jun)")
    metrics(april,      "April only           ")
    print(f"{'='*58}")

    return results


# ===========================================================================
# 6.  SAVE MODEL  (coefficients + scaler ->JSON + standalone predict script)
# ===========================================================================

def save_model(model, scaler, train_samples, test_samples):
    # Per-month residual stdev on test set (for confidence intervals)
    monthly_sigma = {}
    test_by_month = defaultdict(list)
    for s in test_samples:
        x  = np.array([[s[f] for f in FEATURE_NAMES]], dtype=float)
        xs = scaler.transform(x)
        pred = float(model.predict(xs)[0])
        test_by_month[s["month"]].append(abs(pred - s["t_max"]))
    for m, errs in test_by_month.items():
        monthly_sigma[m] = round(statistics.stdev(errs) if len(errs) > 1 else 1.0, 3)

    coeffs = {
        "feature_names":   FEATURE_NAMES,
        "coefs":           model.coef_.tolist(),
        "intercept":       float(model.intercept_),
        "scaler_mean":     scaler.mean_.tolist(),
        "scaler_scale":    scaler.scale_.tolist(),
        "monthly_sigma":   {str(k): v for k, v in monthly_sigma.items()},
        "trained_on":      f"LLBG Synoptic Mar-Jun 2019-2023",
    }

    with open(MODEL_OUT, "w") as f:
        json.dump(coeffs, f, indent=2)
    print(f"\n  Model coefficients saved ->{MODEL_OUT}")

    # Write standalone prediction module
    coef_lines = "\n".join(
        f'    "{n}": {v:.6f},' for n, v in zip(FEATURE_NAMES, model.coef_)
    )
    scaler_mean_str  = ", ".join(f"{v:.6f}" for v in scaler.mean_)
    scaler_scale_str = ", ".join(f"{v:.6f}" for v in scaler.scale_)
    monthly_sigma_str = ", ".join(f"{k}: {v}" for k, v in sorted(monthly_sigma.items()))

    script = f'''"""
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

FEATURE_NAMES = {json.dumps(FEATURE_NAMES)}

COEFS = {{
{coef_lines}
}}
INTERCEPT = {model.intercept_:.6f}

SCALER_MEAN  = [{scaler_mean_str}]
SCALER_SCALE = [{scaler_scale_str}]

MONTHLY_SIGMA = {{{monthly_sigma_str}}}
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
    ceiling_km : cloud ceiling (km); pass {CLEAR_CEILING_KM} for clear sky
    month      : calendar month (3–6)

    Returns dict with predicted_max, low_80, high_80, sigma.
    """
    f = dict(features)
    if "tdep10" not in f:
        f["tdep10"] = f["t10"] - f["td10"]

    pred  = _predict_raw(f)
    sigma = MONTHLY_SIGMA.get(f["month"], 1.0)
    margin = 1.28 * sigma  # 80% CI

    return {{
        "predicted_max": round(pred, 1),
        "low_80":        round(pred - margin, 1),
        "high_80":       round(pred + margin, 1),
        "sigma":         sigma,
    }}
'''

    with open(PREDICT_OUT, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"  Prediction module saved  ->{PREDICT_OUT}")


# ===========================================================================
# 7.  MAIN
# ===========================================================================

if __name__ == "__main__":
    print("Parsing CSV...")
    rows = parse_csv(DATA_FILE)
    print(f"  {len(rows):,} observations loaded (Mar-Jun only)")

    print("Extracting daily features...")
    samples = build_dataset(rows)
    print(f"  {len(samples)} usable days")

    train_s = [s for s in samples if s["year"] in TRAIN_YEARS]
    test_s  = [s for s in samples if s["year"] in TEST_YEARS]
    april_s = [s for s in test_s   if s["month"] == 4]

    print(f"  Train : {len(train_s)} days  ({min(s['year'] for s in train_s)}-{max(s['year'] for s in train_s)})")
    print(f"  Test  : {len(test_s)} days  ({min(s['year'] for s in test_s)}-{max(s['year'] for s in test_s)})")
    print(f"  April days in test: {len(april_s)}")

    if len(april_s) < 50:
        print(f"WARNING: only {len(april_s)} April test days — need >= 50")

    print("\nTraining Ridge regression...")
    model, scaler = train(train_s, alpha=1.0)

    # Feature importance (standardised coefficients)
    print("\n  Standardised coefficients:")
    for name, coef in sorted(zip(FEATURE_NAMES, model.coef_), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {name:<14}: {coef:+.4f}")

    # In-sample MAE
    X_tr, y_tr = to_matrix(train_s)
    y_hat_tr = model.predict(scaler.transform(X_tr))
    print(f"\n  Train MAE  : {mean_absolute_error(y_tr, y_hat_tr):.3f} °C")

    results = backtest(model, scaler, test_s)

    save_model(model, scaler, train_s, test_s)

    # Detailed April backtest table
    april_results = [r for r in results if r["month"] == 4]
    print(f"\n--- April detail ({len(april_results)} days) ---")
    print(f"{'Date':<12} {'Actual':>7} {'Pred':>7} {'Error':>7}")
    print("-" * 38)
    for r in sorted(april_results, key=lambda x: x["date"]):
        print(f"{r['date']:<12} {r['actual']:>7.1f} {r['predicted']:>7.1f} {r['error']:>+7.2f}")

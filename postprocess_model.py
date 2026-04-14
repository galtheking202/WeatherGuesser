"""
MOS Post-Processing Model - Beit Dagan Daily Maximum Temperature
================================================================
A station-specific post-processing model that combines NWP (global
forecast / ERA5 reanalysis) with local morning observations to produce
calibrated probabilistic predictions for the daily maximum temperature.

Architecture
------------
Stage 1 - Feature engineering
    Aligns per-day NWP fields (temperature, cloud cover, wind, pressure
    trend, precipitation) with station observations available by 10:00.
    Key signal: bias_t10 = station_t10  nwp_t10 (already-realised error).

Stage 2 - Heteroscedastic neural network (PyTorch)
    3-layer MLP with two output heads:
        mu       - predicted daily maximum temperature
        log_sigma - log of predictive uncertainty
    Trained with Gaussian negative log-likelihood loss.

Stage 3 - Conformal calibration (distribution-free coverage guarantee)
    Held-out calibration set -> compute conformity scores |ymu|/sigma.
    Threshold q = 95th percentile -> PI = mu +/- q*sigma guarantees >= 95%
    empirical coverage on any future day drawn from the same distribution.

Confidence tiers (based on calibrated sigma and NWP cloud variability)
    HIGH   - sigma < 0.40 C and cloud variability low  -> tight CI
    MEDIUM - sigma < 0.80 C                            -> moderate CI
    LOW    - sigma >= 0.80 C or cloudy/frontal signal   -> do not use

Usage
-----
    python postprocess_model.py train            # fetch NWP, train, save
    python postprocess_model.py predict          # predict latest available date
    python postprocess_model.py predict 2025-07-15
    python postprocess_model.py evaluate         # calibration + accuracy report

Requirements
------------
    pip install torch numpy requests
    (scikit-learn not required)
"""

# -- Imports ------------------------------------------------------------------
import json, math, os, sys, random
from datetime import datetime, date, timedelta
from collections import defaultdict
from pathlib import Path
import statistics

import requests
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -- Constants -----------------------------------------------------------------
LAT, LON     = 32.0073, 34.8138          # Beit Dagan
BASE_DIR     = Path(__file__).parent
TRAINING_DATA_DIR = BASE_DIR / "training_data"   # per-season station files
NWP_CACHE    = BASE_DIR / "nwp_cache.json"
MODEL_SAVE   = BASE_DIR / "mos_model.pt"
NWP_START    = "2022-03-01"   # earliest training data
NWP_END      = "2026-04-13"   # ERA5 archive available up to this date

TRAIN_RATIO  = 0.70   # of data before calibration split
VAL_RATIO    = 0.15   # of data before calibration split
CALIB_RATIO  = 0.15   # always the chronological tail (no leakage)

FEATURE_NAMES = [
    # Station observations (available by 10:00)
    "t10", "rise", "rh10", "ws10", "wd_east", "accel", "is_sharav", "rh_stdev",
    # NWP forecast fields
    "nwp_t10", "nwp_tmax",
    "nwp_cloud_peak_mean", "nwp_cloud_peak_std",
    "nwp_cloud_low_noon", "nwp_ws_peak",
    "nwp_pressure_trend", "nwp_precip", "nwp_wd_east",
    # Engineered
    "bias_t10",           # station_t10  nwp_t10 (realised error signal)
    "month", "doy_sin", "doy_cos",
]
N_FEATURES = len(FEATURE_NAMES)   # 21


# ===========================================================================
# 1.  NWP DATA FETCH
# ===========================================================================

def _omf(url: str, params: dict) -> dict:
    """Call Open-Meteo and return JSON, with a helpful error on failure."""
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Open-Meteo returned {r.status_code}: {r.text[:200]}")
    return r.json()


def fetch_era5_archive(start: str, end: str) -> dict:
    """Fetch ERA5 reanalysis (used as NWP proxy for training)."""
    return _omf("https://archive-api.open-meteo.com/v1/archive", {
        "latitude":  LAT, "longitude": LON,
        "start_date": start, "end_date": end,
        "hourly": ",".join([
            "temperature_2m", "relativehumidity_2m",
            "windspeed_10m", "winddirection_10m",
            "cloudcover", "cloudcover_low",
            "surface_pressure", "precipitation",
        ]),
        "daily": ",".join([
            "temperature_2m_max", "precipitation_sum", "windspeed_10m_max",
        ]),
        "timezone": "Asia/Jerusalem",
    })


def fetch_forecast_today(target_date: str) -> dict:
    """Fetch GFS/IFS forecast from Open-Meteo for a given date.
    start_date/end_date and forecast_days are mutually exclusive in the API.
    """
    return _omf("https://api.open-meteo.com/v1/forecast", {
        "latitude":  LAT, "longitude": LON,
        "start_date": target_date, "end_date": target_date,
        "hourly": ",".join([
            "temperature_2m", "relativehumidity_2m",
            "windspeed_10m", "winddirection_10m",
            "cloudcover", "cloudcover_low",
            "surface_pressure",
        ]),
        "daily": ",".join([
            "temperature_2m_max", "precipitation_sum", "windspeed_10m_max",
        ]),
        "timezone": "Asia/Jerusalem",
    })


def load_or_fetch_nwp() -> dict:
    """
    Return cached NWP data or fetch it in annual chunks and cache.
    Fetching 4+ years of hourly data in one request exceeds server limits,
    so we request one calendar year at a time and merge the results.
    """
    if NWP_CACHE.exists():
        with open(NWP_CACHE) as f:
            return json.load(f)

    # Build list of (start, end) pairs, one per year
    from datetime import date as _date
    start_year = int(NWP_START[:4])
    end_year   = int(NWP_END[:4])
    chunks = []
    for yr in range(start_year, end_year + 1):
        s = f"{yr}-03-01"
        e = f"{yr}-06-30"
        # Clamp to NWP_START / NWP_END
        if s < NWP_START: s = NWP_START
        if e > NWP_END:   e = NWP_END
        if s <= e:
            chunks.append((s, e))

    print(f"Fetching ERA5 in {len(chunks)} chunks: {chunks[0][0]} -> {chunks[-1][1]} ...")
    merged = None
    for s, e in chunks:
        print(f"  chunk {s} -> {e} ...", end=" ", flush=True)
        chunk = fetch_era5_archive(s, e)
        print("ok")
        if merged is None:
            merged = chunk
        else:
            # Append hourly time-series arrays
            for var in merged["hourly"]:
                merged["hourly"][var] += chunk["hourly"][var]
            # Append daily arrays
            for var in merged.get("daily", {}):
                merged["daily"][var] += chunk["daily"][var]

    with open(NWP_CACHE, "w") as f:
        json.dump(merged, f)
    print(f"  Cached to {NWP_CACHE.name}")
    return merged


# ===========================================================================
# 2.  NWP FEATURE EXTRACTION
# ===========================================================================

def _safe_avg(vals):
    return sum(vals) / len(vals) if vals else 0.0

def _safe_std(vals):
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def nwp_features_from_response(raw: dict) -> dict[str, dict]:
    """
    Parse an Open-Meteo response -> {date_iso: {feature: value}}.
    Works for both the archive and forecast endpoints.
    """
    hourly = raw["hourly"]
    daily  = raw.get("daily", {})

    # Build per-day, per-hour index
    day_hours: dict[str, dict[int, dict]] = defaultdict(lambda: defaultdict(dict))
    for i, ts in enumerate(hourly["time"]):
        dt  = datetime.fromisoformat(ts)
        ds  = dt.strftime("%Y-%m-%d")
        h   = dt.hour
        for var in hourly:
            if var == "time": continue
            val = hourly[var][i] if i < len(hourly[var]) else None
            if val is not None:
                day_hours[ds][h][var] = float(val)

    # Build daily index
    daily_idx: dict[str, dict] = {}
    for i, ds in enumerate(daily.get("time", [])):
        daily_idx[ds] = {}
        for var in ["temperature_2m_max", "precipitation_sum", "windspeed_10m_max"]:
            if var in daily and i < len(daily[var]) and daily[var][i] is not None:
                daily_idx[ds][var] = float(daily[var][i])

    result = {}
    for ds, hours in day_hours.items():
        def at(h, v): return hours.get(h, {}).get(v)

        # Cloud cover during peak heating hours (11-15)
        cloud_peak = [hours[h]["cloudcover"]     for h in range(11, 16) if "cloudcover"     in hours.get(h, {})]
        cloud_low  = [hours[h]["cloudcover_low"] for h in range(11, 16) if "cloudcover_low" in hours.get(h, {})]
        ws_peak    = [hours[h]["windspeed_10m"]  for h in range(11, 16) if "windspeed_10m"  in hours.get(h, {})]

        # Pressure trend 06->10: falling = approaching front
        p6  = at(6,  "surface_pressure")
        p10 = at(10, "surface_pressure")
        ptrd = (p10 - p6) if (p6 and p10) else 0.0

        wd10 = at(10, "winddirection_10m")
        wd_east = math.sin(math.radians(wd10)) if wd10 is not None else 0.0

        nwp_t10 = at(10, "temperature_2m")
        if nwp_t10 is None:
            continue

        dday = daily_idx.get(ds, {})
        result[ds] = {
            "nwp_t10":             nwp_t10,
            "nwp_tmax":            dday.get("temperature_2m_max"),
            "nwp_cloud_peak_mean": _safe_avg(cloud_peak),
            "nwp_cloud_peak_std":  _safe_std(cloud_peak),   # key uncertainty driver
            "nwp_cloud_low_noon":  _safe_avg(cloud_low),
            "nwp_ws_peak":         _safe_avg(ws_peak),
            "nwp_pressure_trend":  ptrd,
            "nwp_precip":          dday.get("precipitation_sum", 0.0) or 0.0,
            "nwp_wd_east":         wd_east,
        }
    return result


# ===========================================================================
# 3.  STATION FEATURE EXTRACTION
# ===========================================================================

def load_station() -> dict[date, list]:
    """
    Load all station JSON files from the training_data/ directory.
    Each file is expected to have records with keys: date, TD, RH, WD, WS.
    """
    records = []
    files = sorted(TRAINING_DATA_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No JSON files found in {TRAINING_DATA_DIR}. "
            "Place station data files there before training."
        )
    for path in files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        loaded = 0
        for d in data:
            try:
                records.append({
                    "dt": datetime.strptime(d["date"], "%d/%m/%Y %H:%M"),
                    "TD": float(d["TD"]), "RH": float(d["RH"]),
                    "WD": float(d["WD"]), "WS": float(d["WS"]),
                })
                loaded += 1
            except: pass
        print(f"  Loaded {loaded:5d} records from {path.name}")
    by_date = defaultdict(list)
    for r in records:
        by_date[r["dt"].date()].append(r)
    return by_date


def station_morning_features(recs: list) -> dict | None:
    """
    Extract all features available by 10:00 AM from station records.
    Returns None if data is insufficient.
    """
    recs = sorted(recs, key=lambda x: x["dt"])
    until10 = [r for r in recs
               if r["dt"].hour < 10 or (r["dt"].hour == 10 and r["dt"].minute == 0)]
    after10  = [r for r in recs
                if r["dt"].hour > 10 or (r["dt"].hour == 10 and r["dt"].minute > 0)]
    if len(until10) < 5 or len(after10) < 3:
        return None

    def avg(lst): return sum(lst) / len(lst) if lst else None

    t6 = avg([r["TD"] for r in recs if r["dt"].hour == 6])
    if t6 is None:
        return None

    last = until10[-1]
    t10, rh10, wd10, ws10 = last["TD"], last["RH"], last["WD"], last["WS"]

    # Heating acceleration: linear slope of last 6 readings
    l6 = until10[-6:]
    xs, ys = list(range(len(l6))), [r["TD"] for r in l6]
    mx, my = avg(xs), avg(ys)
    denom  = sum((x - mx) ** 2 for x in xs)
    accel  = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom else 0.0

    wd_east   = math.sin(math.radians(wd10))
    is_sharav = 1.0 if (45 <= wd10 <= 200 and rh10 < 50) else 0.0
    rh_vals   = [r["RH"] for r in until10[-6:]]
    rh_stdev  = statistics.stdev(rh_vals) if len(rh_vals) > 1 else 0.0

    return {
        "t10": t10, "rise": t10 - t6,
        "rh10": rh10, "ws10": ws10, "wd_east": wd_east,
        "accel": accel, "is_sharav": is_sharav, "rh_stdev": rh_stdev,
        "max_day": max(r["TD"] for r in recs),   # label (only for training)
    }


# ===========================================================================
# 4.  DATASET CONSTRUCTION
# ===========================================================================

def build_feature_row(sf: dict, nf: dict, dt: date) -> list[float]:
    """Assemble a single feature vector (must match FEATURE_NAMES order)."""
    doy = dt.timetuple().tm_yday
    return [
        sf["t10"],       sf["rise"],      sf["rh10"],       sf["ws10"],
        sf["wd_east"],   sf["accel"],     sf["is_sharav"],   sf["rh_stdev"],
        nf["nwp_t10"],   nf["nwp_tmax"],
        nf["nwp_cloud_peak_mean"], nf["nwp_cloud_peak_std"],
        nf["nwp_cloud_low_noon"],  nf["nwp_ws_peak"],
        nf["nwp_pressure_trend"],  nf["nwp_precip"],   nf["nwp_wd_east"],
        sf["t10"] - nf["nwp_t10"],       # bias_t10
        float(dt.month),
        math.sin(2 * math.pi * doy / 365),
        math.cos(2 * math.pi * doy / 365),
    ]


TRAIN_MONTHS = {3, 4, 5, 6}   # Mar-Jun: spring + early summer (matches training_data files)


def build_dataset(months=TRAIN_MONTHS) -> tuple[list, list, list[date]]:
    """
    Returns (X, y, dates) -- all aligned, chronologically sorted.
    NWP data is fetched/cached automatically.
    Only includes days in the given months (default: Apr-Oct, when the
    station max temperature is most relevant and NWP bias is consistent).
    """
    nwp_raw   = load_or_fetch_nwp()
    nwp_daily = nwp_features_from_response(nwp_raw)
    station   = load_station()

    X, y, dates_out = [], [], []
    for dt in sorted(station.keys()):
        if months and dt.month not in months:
            continue
        ds = dt.strftime("%Y-%m-%d")
        if ds not in nwp_daily:
            continue
        nf = nwp_daily[ds]
        if nf.get("nwp_tmax") is None:
            continue
        sf = station_morning_features(station[dt])
        if sf is None:
            continue
        X.append(build_feature_row(sf, nf, dt))
        y.append(round(sf["max_day"]))  # integer target: 23.2 -> 23
        dates_out.append(dt)

    print(f"Dataset: {len(X)} days  ({dates_out[0]} -> {dates_out[-1]})  "
          f"[months: {sorted(months) if months else 'all'}]")
    return X, y, dates_out


# ===========================================================================
# 5.  SCALER  (no sklearn required)
# ===========================================================================

def fit_scaler(X: list[list[float]]) -> tuple[list[float], list[float]]:
    n, k = len(X), len(X[0])
    means = [sum(X[i][j] for i in range(n)) / n for j in range(k)]
    stds  = [
        math.sqrt(sum((X[i][j] - means[j]) ** 2 for i in range(n)) / max(n - 1, 1))
        for j in range(k)
    ]
    stds = [max(s, 1e-8) for s in stds]
    return means, stds


def scale(X: list[list[float]], means, stds) -> list[list[float]]:
    return [[(x - m) / s for x, m, s in zip(row, means, stds)] for row in X]


def to_tensor(X, y=None):
    Xt = torch.tensor(X, dtype=torch.float32)
    if y is None:
        return Xt
    return Xt, torch.tensor(y, dtype=torch.float32)


# ===========================================================================
# 6.  PYTORCH MODEL
# ===========================================================================

# Indices of features that most directly drive forecast uncertainty.
# The sigma head gets a skip connection from these raw (scaled) inputs
# so it can learn heteroscedastic uncertainty without waiting for the
# backbone to propagate the signal.
# Indices in FEATURE_NAMES: accel=5, is_sharav=6, rh_stdev=7,
#   nwp_cloud_peak_std=11, nwp_pressure_trend=14, bias_t10=17
SIGMA_SKIP_IDXS = [5, 6, 7, 11, 14, 17]


class PostProcessMOS(nn.Module):
    """
    Heteroscedastic MLP: predicts (mu, log_sigma) for each day.

    Architecture
    ------------
    - Shared backbone (64 -> 32, SiLU, light dropout)
    - mu_head: Linear(32, 1)  -- point prediction
    - log_sig_head: Linear(32 + n_skip, 1) with skip from uncertainty features
      The skip connection lets sigma respond directly to cloud variability,
      sharav conditions, NWP bias etc. without waiting for the backbone.
    """
    def __init__(self, n_features: int = N_FEATURES, hidden=(64, 32),
                 sigma_skip_idxs: list = SIGMA_SKIP_IDXS):
        super().__init__()
        self.sigma_skip_idxs = sigma_skip_idxs
        layers, in_dim = [], n_features
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.SiLU(), nn.Dropout(0.05)]
            in_dim = h
        self.backbone     = nn.Sequential(*layers)
        self.mu_head      = nn.Linear(in_dim, 1)
        self.log_sig_head = nn.Linear(in_dim + len(sigma_skip_idxs), 1)
        # Start log_sigma at 0 -> sigma=1.0 C (neutral, avoids early saturation)
        nn.init.zeros_(self.log_sig_head.weight)
        nn.init.zeros_(self.log_sig_head.bias)

    def forward(self, x: torch.Tensor):
        h       = self.backbone(x)
        mu      = self.mu_head(h).squeeze(-1)
        # Concatenate backbone output with raw uncertainty features
        skip    = x[:, self.sigma_skip_idxs] if x.dim() == 2 else x[self.sigma_skip_idxs].unsqueeze(0)
        h_sig   = torch.cat([h, skip], dim=-1)
        log_sig = self.log_sig_head(h_sig).squeeze(-1).clamp(-3.0, 3.5)
        return mu, log_sig


def gaussian_nll(mu, log_sig, y):
    """
    Gaussian negative log-likelihood.
    Jointly optimises accuracy (mu) and calibration (sigma).
    """
    return (log_sig + 0.5 * ((y - mu) / log_sig.exp()) ** 2).mean()


def beta_nll(mu, log_sig, y, beta: float = 0.5):
    """
    Beta-NLL loss (Seitzer et al. 2022).
    beta=0 -> pure MSE (ignores sigma),  beta=0.5 -> standard NLL.
    Annealing beta from 0 to 0.5 prevents sigma from exploding while
    the mean head is still learning.
    """
    sigma2 = (2 * log_sig).exp()
    return (sigma2.detach() ** beta * ((y - mu) ** 2 / sigma2 + log_sig)).mean()


# ===========================================================================
# 7.  TRAINING
# ===========================================================================

def train(X_tr, y_tr, X_va, y_va, seed: int = 42) -> PostProcessMOS:
    """
    Two-phase training to avoid sigma saturation:

    Phase 1 - MSE warmup (200 epochs, backbone + mu_head only)
        The sigma head is frozen at its zero-init (sigma=1.0 C).
        The mean head converges to ~residual level without interference.

    Phase 2 - Full NLL (up to 400 epochs, all parameters)
        Sigma head is now free to learn *after* mu is already decent.
        Early stopping on validation NLL with patience=25.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    model    = PostProcessMOS()
    Xtr_t, ytr_t = to_tensor(X_tr, y_tr)
    Xva_t, yva_t = to_tensor(X_va, y_va)

    # ------------------------------------------------------------------
    # Phase 1: pure MSE warmup -- only backbone + mu_head
    # ------------------------------------------------------------------
    print("  [Phase 1] MSE warmup (500 epochs, mu head only) ...")
    opt1 = torch.optim.Adam(
        list(model.backbone.parameters()) + list(model.mu_head.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=500, eta_min=1e-5)
    for epoch in range(1, 501):
        model.train()
        mu, _ = model(Xtr_t)
        loss   = ((mu - ytr_t) ** 2).mean()
        opt1.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt1.step()
        sched1.step()
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                mu_v, _ = model(Xva_t)
                val_mae = (mu_v - yva_t).abs().mean().item()
            print(f"    ep {epoch:3d}  val_mae={val_mae:.3f} C")

    # ------------------------------------------------------------------
    # Phase 2: full NLL -- all parameters (mu + sigma trained jointly)
    # ------------------------------------------------------------------
    print("  [Phase 2] Gaussian NLL (up to 2000 epochs, all params) ...")
    opt2 = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=2000, eta_min=1e-6)
    best_val, best_state, patience = float("inf"), None, 0

    for epoch in range(1, 2001):
        model.train()
        mu, ls = model(Xtr_t)
        loss    = gaussian_nll(mu, ls, ytr_t)
        opt2.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt2.step()
        sched2.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                mu_v, ls_v = model(Xva_t)
                val_nll = gaussian_nll(mu_v, ls_v, yva_t).item()
                val_mae = (mu_v - yva_t).abs().mean().item()
                val_sig = ls_v.exp().mean().item()
            if val_nll < best_val:
                best_val   = val_nll
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience   = 0
            else:
                patience += 1
            if epoch > 200 and patience > 50:
                print(f"    Early stop ep {epoch}  val_mae={val_mae:.3f}  "
                      f"val_sigma={val_sig:.3f}")
                break
            if epoch % 200 == 0:
                print(f"    ep {epoch:4d}  val_nll={val_nll:.4f}  "
                      f"val_mae={val_mae:.3f}  val_sigma={val_sig:.3f}")

    if best_state:
        model.load_state_dict(best_state)
    return model


# ===========================================================================
# 8.  CONFORMAL CALIBRATION
# ===========================================================================

def conformal_calibrate(model: PostProcessMOS,
                        X_cal: list, y_cal: list,
                        alpha: float = 0.05) -> float:
    """
    Split-conformal regression.

    Computes conformity scores s_i = |y_i - mu_i| / sigma_i on the
    calibration set, then returns the ceil((n+1)(1-alpha)/n) empirical
    quantile as the conformal multiplier q.

    At inference: PI_95 = mu +/- q * sigma   ->  guaranteed >= 95% coverage.
    """
    model.eval()
    Xc, yc = to_tensor(X_cal, y_cal)
    with torch.no_grad():
        mu, ls = model(Xc)
    sigma  = ls.exp()
    scores = ((yc - mu).abs() / sigma).tolist()
    scores.sort()
    n     = len(scores)
    q_idx = min(math.ceil((n + 1) * (1 - alpha)) - 1, n - 1)
    q     = scores[q_idx]
    print(f"  Conformal q (95% coverage): {q:.4f}  "
          f"(calibration n={n}, alpha={alpha})")
    return q


# ===========================================================================
# 9.  INFERENCE
# ===========================================================================

def predict(model: PostProcessMOS,
            scaler_mean: list, scaler_std: list,
            conformal_q: float,
            station_recs: list,
            nwp_features: dict,
            target_date: date) -> dict:
    """
    Produce a calibrated probabilistic forecast for one day.

    Parameters
    ----------
    station_recs  : raw station records for target_date (all day, up to 10:00 needed)
    nwp_features  : dict returned by nwp_features_from_response for that date
    target_date   : date object

    Returns
    -------
    dict with keys:
        date, mu, sigma, ci_low, ci_high, ci_width_95,
        confidence, reasons, [nwp/station diagnostics]
    """
    sf = station_morning_features(station_recs)
    if sf is None:
        return {"error": "Not enough station data before 10:00"}

    nf = nwp_features
    if nf.get("nwp_tmax") is None:
        return {"error": "NWP daily max not available for this date"}

    row     = build_feature_row(sf, nf, target_date)
    row_sc  = [(v - m) / s for v, m, s in zip(row, scaler_mean, scaler_std)]
    x_t     = torch.tensor([row_sc], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        mu_t, ls_t = model(x_t)
    mu    = mu_t.item()
    sigma = ls_t.exp().item()

    # Conformal prediction interval (coverage guaranteed)
    ci_half = conformal_q * sigma
    ci_low  = round(mu - ci_half, 2)
    ci_high = round(mu + ci_half, 2)

    bias_t10       = sf["t10"] - nf["nwp_t10"]
    cloud_std      = nf["nwp_cloud_peak_std"]
    precip         = nf["nwp_precip"]
    pressure_trend = nf["nwp_pressure_trend"]

    # Confidence tiers — calibrated to Mar-Jun Beit Dagan sigma distribution
    # (model sigma typically 1.4-2.6 C depending on atmospheric stability)
    if sigma < 1.55 and cloud_std < 15.0 and precip < 0.1 and not sf["is_sharav"]:
        confidence = "HIGH"
    elif sigma < 2.0 and cloud_std < 30.0 and precip < 1.0:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Explain low confidence
    reasons = []
    if sigma >= 2.0:
        reasons.append(f"model uncertainty high (sigma={sigma:.2f} C)")
    if cloud_std >= 25.0:
        reasons.append(f"variable cloud cover (std={cloud_std:.0f}%)")
    if precip >= 1.0:
        reasons.append(f"precipitation forecast ({precip:.1f} mm)")
    if pressure_trend < -0.5:
        reasons.append(f"falling pressure (front signal, {pressure_trend:+.2f} hPa/4h)")
    if sf["is_sharav"]:
        reasons.append("sharav conditions")
    if abs(bias_t10) > 2.5:
        reasons.append(f"large NWP bias at 10:00 ({bias_t10:+.1f} C)")

    return {
        "date":           target_date.strftime("%Y-%m-%d"),
        "temp_at_10:00":  round(sf["t10"], 1),
        "nwp_tmax":       round(nf["nwp_tmax"], 1),
        "nwp_t10":        round(nf["nwp_t10"], 1),
        "nwp_bias_at_10": round(bias_t10, 2),
        "nwp_cloud_std":  round(cloud_std, 1),
        "mu":             round(mu),  # integer prediction (23.2 -> 23)
        "sigma":          round(sigma, 3),
        "ci_low_95":      ci_low,
        "ci_high_95":     ci_high,
        "ci_width_95":    round(ci_high - ci_low, 2),
        "confidence":     confidence,
        "reasons":        reasons,
    }


# ===========================================================================
# 10.  EVALUATION
# ===========================================================================

def evaluate_calibration(model: PostProcessMOS,
                         X_scaled: list, y: list,
                         conformal_q: float,
                         label: str = ""):
    """
    Print calibration table + accuracy metrics.
    X_scaled must already be normalised (pass the output of scale()).
    """
    Xt, yt = to_tensor(X_scaled, y)
    model.eval()
    with torch.no_grad():
        mu_t, ls_t = model(Xt)
    mu    = mu_t.numpy()
    sigma = ls_t.exp().numpy()

    ci_half = conformal_q * sigma
    covered = np.abs(np.array(y) - mu) <= ci_half
    abs_err = np.abs(np.array(y) - mu)

    print(f"\n{'-'*52}")
    print(f"  {label}  (n={len(y)})")
    print(f"  MAE            : {abs_err.mean():.3f} C")
    print(f"  RMSE           : {np.sqrt((abs_err**2).mean()):.3f} C")
    print(f"  Sigma  min/med/max: {sigma.min():.3f} / {np.median(sigma):.3f} / {sigma.max():.3f} C")
    print(f"  95% PI coverage: {covered.mean()*100:.1f}%  (target: 95%)")
    tiers = [
        (0,    1.55, "HIGH   sigma<1.55"),
        (1.55, 2.00, "MEDIUM 1.55-2.00"),
        (2.00, 99,   "LOW    sigma>=2.00"),
    ]
    for lo, hi, name in tiers:
        mask = (sigma >= lo) & (sigma < hi)
        n_m  = mask.sum()
        if n_m == 0:
            continue
        cov  = covered[mask].mean() * 100
        mae  = abs_err[mask].mean()
        ci_w = (2 * conformal_q * sigma[mask]).mean()
        print(f"  {name:<22}: n={n_m:3d}  MAE={mae:.3f}  "
              f"CI_width={ci_w:.2f}  cov={cov:.0f}%")


# ===========================================================================
# 11.  SAVE / LOAD
# ===========================================================================

def save_model(model, scaler_mean, scaler_std, conformal_q, path=MODEL_SAVE):
    torch.save({
        "model_state":      model.state_dict(),
        "scaler_mean":      scaler_mean,
        "scaler_std":       scaler_std,
        "conformal_q":      conformal_q,
        "feature_names":    FEATURE_NAMES,
        "n_features":       N_FEATURES,
        "sigma_skip_idxs":  model.sigma_skip_idxs,
    }, path)
    print(f"Model saved to {path}")


def load_model(path=MODEL_SAVE) -> tuple:
    """Returns (model, scaler_mean, scaler_std, conformal_q)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"No saved model at {path}. Run 'python postprocess_model.py train' first.")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = PostProcessMOS(ckpt["n_features"],
                           sigma_skip_idxs=ckpt.get("sigma_skip_idxs", SIGMA_SKIP_IDXS))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["scaler_mean"], ckpt["scaler_std"], ckpt["conformal_q"]


# ===========================================================================
# 12.  CLI
# ===========================================================================

def cmd_train():
    print("=" * 52)
    print("  MOS Post-Processing Model -- Training")
    print("=" * 52)

    X, y, dates = build_dataset()
    n = len(X)
    if n < 50:
        sys.exit("Not enough data to train (need >= 50 days).")

    # Chronological split: train / val / calibration
    n_cal  = max(int(n * CALIB_RATIO), 20)
    n_rest = n - n_cal
    n_val  = max(int(n_rest * (VAL_RATIO / (TRAIN_RATIO + VAL_RATIO))), 10)
    n_tr   = n_rest - n_val

    X_tr,  y_tr   = X[:n_tr],              y[:n_tr]
    X_va,  y_va   = X[n_tr:n_tr+n_val],   y[n_tr:n_tr+n_val]
    X_cal, y_cal  = X[n_tr+n_val:],        y[n_tr+n_val:]

    print(f"\n  Split: train={n_tr}  val={n_val}  calibration={n_cal}")
    print(f"  Training dates: {dates[0]} -> {dates[n_tr-1]}")
    print(f"  Val dates:      {dates[n_tr]} -> {dates[n_tr+n_val-1]}")
    print(f"  Calib dates:    {dates[n_tr+n_val]} -> {dates[-1]}\n")

    # Fit scaler on training data only (no leakage)
    scaler_mean, scaler_std = fit_scaler(X_tr)
    X_tr_s  = scale(X_tr,  scaler_mean, scaler_std)
    X_va_s  = scale(X_va,  scaler_mean, scaler_std)
    X_cal_s = scale(X_cal, scaler_mean, scaler_std)

    print("  Training ")
    model = train(X_tr_s, y_tr, X_va_s, y_va)

    print("\n  Calibrating (conformal) ")
    q = conformal_calibrate(model, X_cal_s, y_cal)

    evaluate_calibration(model, X_tr_s,  y_tr,  q, "Training set")
    evaluate_calibration(model, X_va_s,  y_va,  q, "Validation set")
    evaluate_calibration(model, X_cal_s, y_cal, q, "Calibration set")

    save_model(model, scaler_mean, scaler_std, q)


def cmd_predict(date_str: str | None = None):
    model, s_mean, s_std, q = load_model()

    # Determine target date
    if date_str:
        try:
            target = date.fromisoformat(date_str)
        except ValueError:
            sys.exit(f"Invalid date format: {date_str}. Use YYYY-MM-DD.")
    else:
        target = date.today()

    target_iso = target.strftime("%Y-%m-%d")
    print(f"\nPredicting for {target_iso} ")

    # Load station data
    station = load_station()
    if target not in station:
        if target > date.today():
            sys.exit(
                f"Cannot predict {target_iso}: this model requires station readings "
                f"up to 10:00 AM on the target day. Run again after 10:00 AM on that date."
            )
        sys.exit(
            f"No station data found for {target_iso}. "
            f"Ensure the date is covered by a JSON file in {TRAINING_DATA_DIR}."
        )

    # NWP: use ERA5 cache if the date is covered (up to NWP_END), else live forecast
    nwp_end_date = date.fromisoformat(NWP_END)
    if target <= nwp_end_date:
        nwp_raw = load_or_fetch_nwp()
        # If the cached archive doesn't have this date, fall through to live
        nwp_daily_check = nwp_features_from_response(nwp_raw)
        if target_iso not in nwp_daily_check:
            print(f"  Date not in ERA5 cache, fetching live forecast for {target_iso} ...")
            nwp_raw = fetch_forecast_today(target_iso)
    else:
        print(f"  Fetching live NWP forecast for {target_iso} ...")
        nwp_raw = fetch_forecast_today(target_iso)

    nwp_daily = nwp_features_from_response(nwp_raw)
    if target_iso not in nwp_daily:
        sys.exit(f"NWP data not available for {target_iso}.")

    result = predict(model, s_mean, s_std, q,
                     station[target], nwp_daily[target_iso], target)

    if "error" in result:
        print(f"  Error: {result['error']}")
        return

    conf_sym = {"HIGH": "***", "MEDIUM": "**", "LOW": "!"}[result["confidence"]]
    print(f"\n{'='*52}")
    print(f"  Date              : {result['date']}")
    print(f"  Station at 10:00  : {result['temp_at_10:00']} C")
    print(f"  NWP forecast max  : {result['nwp_tmax']} C")
    print(f"  NWP at 10:00      : {result['nwp_t10']} C")
    print(f"  NWP bias at 10:00 : {result['nwp_bias_at_10']:+.2f} C")
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


def cmd_evaluate():
    model, s_mean, s_std, q = load_model()
    X, y, dates = build_dataset()
    Xs = scale(X, s_mean, s_std)
    evaluate_calibration(model, Xs, y, q, "Full dataset (all splits)")


def cmd_backtest():
    """
    Per-day backtest table across all data splits.
    Training set rows are marked [TR], validation [VA], calibration [CA].
    Note: [TR] rows are in-sample (optimistic) -- trust [VA]/[CA] for real accuracy.
    """
    model, s_mean, s_std, q = load_model()
    X, y, dates = build_dataset()
    Xs = scale(X, s_mean, s_std)

    # Reconstruct split boundaries (same logic as cmd_train)
    n = len(X)
    n_cal  = max(int(n * CALIB_RATIO), 20)
    n_rest = n - n_cal
    n_val  = max(int(n_rest * (VAL_RATIO / (TRAIN_RATIO + VAL_RATIO))), 10)
    n_tr   = n_rest - n_val

    def split_label(i):
        if i < n_tr:           return "TR"
        if i < n_tr + n_val:   return "VA"
        return "CA"

    Xt = to_tensor(Xs)
    model.eval()
    with torch.no_grad():
        mu_t, ls_t = model(Xt)
    preds  = [round(v) for v in mu_t.tolist()]
    sigmas = ls_t.exp().tolist()
    ci_half = [q * s for s in sigmas]

    # Per-month summary buckets
    month_stats = defaultdict(lambda: {"n": 0, "exact": 0, "within1": 0, "errs": []})

    print(f"\n{'='*72}")
    print(f"  {'Date':<12} {'Sp':2}  {'Act':>4} {'Pred':>4} {'Err':>4}  "
          f"{'Sigma':>5}  {'CI95':>13}  {'Tier':<8}  {'In CI':>5}")
    print(f"  {'-'*70}")

    exact = within1 = covered = total = 0
    for i, (dt, actual, pred, sig, ci) in enumerate(
            zip(dates, y, preds, sigmas, ci_half)):
        sp    = split_label(i)
        err   = pred - actual
        in_ci = abs(actual - pred) <= ci
        tier  = ("HIGH  " if sig < 1.55 else
                 "MEDIUM" if sig < 2.00 else
                 "LOW   ")
        ci_lo = round(pred - ci, 1)
        ci_hi = round(pred + ci, 1)

        print(f"  {dt.strftime('%d/%m/%Y'):<12} [{sp}]  "
              f"{actual:>4}  {pred:>4}  {err:>+4}  "
              f"{sig:>5.2f}  [{ci_lo:>5.1f},{ci_hi:>5.1f}]  "
              f"{tier}  {'yes' if in_ci else 'NO':>5}")

        if sp != "TR":   # only count held-out sets for summary
            total    += 1
            exact    += int(abs(err) == 0)
            within1  += int(abs(err) <= 1)
            covered  += int(in_ci)
            ms = month_stats[dt.month]
            ms["n"] += 1
            ms["exact"]   += int(abs(err) == 0)
            ms["within1"] += int(abs(err) <= 1)
            ms["errs"].append(abs(err))

    print(f"\n{'='*72}")
    print(f"  HELD-OUT SUMMARY  (VA + CA only, n={total})")
    print(f"  Exact match (+-0): {exact}/{total} = {exact/total*100:.0f}%")
    print(f"  Within +-1 C     : {within1}/{total} = {within1/total*100:.0f}%")
    print(f"  95% PI covered   : {covered}/{total} = {covered/total*100:.0f}%")
    print(f"\n  By month:")
    month_names = {3:"Mar", 4:"Apr", 5:"May", 6:"Jun"}
    for m in sorted(month_stats):
        ms = month_stats[m]
        mae = sum(ms["errs"]) / len(ms["errs"])
        print(f"    {month_names.get(m, m)}: n={ms['n']:3d}  "
              f"exact={ms['exact']/ms['n']*100:4.0f}%  "
              f"within1={ms['within1']/ms['n']*100:4.0f}%  "
              f"MAE={mae:.2f} C")
    print(f"{'='*72}\n")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "predict"
    if cmd == "train":
        cmd_train()
    elif cmd == "predict":
        cmd_predict(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "evaluate":
        cmd_evaluate()
    elif cmd == "backtest":
        cmd_backtest()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()

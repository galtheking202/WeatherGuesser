# WeatherGuesser — Models & Scripts Reference

Ben Gurion Airport (LLBG, 32.01 N 34.89 E) daily maximum temperature prediction.

---

## Models at a glance

| Model file | Type | Input | MAE (test) | When to use |
|---|---|---|---|---|
| `mos_model.pt` | MLP (heteroscedastic) | NWP forecast + station obs at 12:00 | ~1-2 C | Best overall accuracy; requires NWP data |
| `nws_model.pt` | LSTM (heteroscedastic) | Station obs stream since midnight | **0.59 C** | Pure station-based; works at any time of day; no NWP needed |

---

## Model 1 — mos_model.pt

**Script:** `postprocess_model.py`  
**Run prediction:** `python predict_today.py`  
**Train:** `python postprocess_model.py train`  
**Evaluate:** `python postprocess_model.py evaluate`

### What it does
MOS (Model Output Statistics) post-processing. Takes a hand-crafted 21-feature vector built from:
- Station observations available by 12:00 (temp, RH, wind, heating rate, sharav flag)
- NWP (ERA5/GFS) forecast fields for the day (cloud cover, pressure trend, precip, NWP tmax)
- Key signal: `bias_t12 = station_t12 - nwp_t12` (already-realised NWP error)

### Architecture
3-layer MLP (64→32, SiLU) with two output heads:
- `mu` — point prediction
- `log_sigma` — uncertainty (skip connection from uncertainty-driving features)

### Training data
ERA5 reanalysis (2018-10-26 to 2026-04-13) via Open-Meteo API + Synoptic station CSVs in `training_data_synoptic_model/`.

### Confidence tiers
- **HIGH** (sigma < 1.55 C, low cloud variability, no precip, no sharav) — tight PI, trust it
- **MEDIUM** (sigma < 2.0 C) — moderate uncertainty
- **LOW** (sigma >= 2.0 C or frontal signal) — prediction available but wide CI

### Notes
- Requires 12:00 station reading. Do not run before 12:00.
- Fetches live NWP from Open-Meteo if target date is beyond ERA5 archive.
- NWP cache lives in `nwp_cache.json` (~large file, do not delete unless re-training).

---

## Model 2 — nws_model.pt

**Script:** `train_nws_model.py`  
**Run prediction:** `python predict_by_nws.py`  
**Train:** `python train_nws_model.py`  
**Evaluate:** `python train_nws_model.py evaluate`

### What it does
Continuous intra-day prediction. Reads the live stream of 30-min LLBG station observations from midnight onwards and outputs an updated (mu, sigma) forecast for the day's maximum at every new observation. No NWP data required.

### Architecture
LSTM (input=10, hidden=64, layers=2) with heteroscedastic output head:
- `mu` — point prediction of daily max
- `log_sigma` — uncertainty (naturally high at midnight, decreasing through the day)

### Features per timestep (10)
`air_temp, dew_point, wind_speed, wd_sin, wd_cos, pressure, time_sin, time_cos, doy_sin, doy_cos`

### Training data
Single CSV: `training_data_synoptic_model/LLBG.2026-04-14 (1).csv`  
2018-10-26 to 2026-04-14 → 2,720 complete days → ~130k training signals  
(every observation in every day is a training pair: sequence so far → daily max)

### Performance (test set, last timestep of day)
- MAE: **0.59 C**
- RMSE: 0.91 C
- 95% PI coverage: **95.1%** (conformal guarantee, target = 95%)

### Notes
- No fixed cutoff — run at any time of day. Prediction improves as the day progresses.
- Conformal q = 3.33 (model's intrinsic sigma is small; conformal calibration corrects coverage).
- Data fetched live from NWS/Synoptic API using NWS public token (no key needed).
- GPU: code is GPU-ready (`DEVICE` auto-detects CUDA). Currently running CPU (torch installed without CUDA support — reinstall torch with CUDA to activate GPU).

---

## Live Dashboard

**Script:** `app.py`  
**Run:** `streamlit run app.py`  
**URL:** http://localhost:8501

Streamlit dashboard showing both models side-by-side with real-time updates.

| Feature | Detail |
|---|---|
| NWS data poll | Every 3 min (ASOS reports every 30 min) |
| NWP refresh | Every 6 h |
| Auto-refresh | `@st.fragment(run_every="3min")` — only the live section reruns |
| Current conditions | Temp, dew point, wind speed, rising/falling trend |
| LSTM card | Predicted max, 95% PI, sigma, observation count |
| MOS card | Predicted max, 95% PI, confidence tier (HIGH/MEDIUM/LOW), notes |
| Intraday chart | Observed temps + both model prediction lines + CI bands |
| History chart | How both predictions evolved through the day |

---

## Scripts

| Script | Purpose |
|---|---|
| `app.py` | Live Streamlit dashboard (both models, auto-refresh) |
| `predict_today.py` | CLI: run mos_model on today's data |
| `predict_by_nws.py` | CLI: run nws_model on today's live NWS stream |
| `postprocess_model.py` | MOS model: train / predict / evaluate |
| `train_nws_model.py` | NWS LSTM model: train / evaluate |
| `backtest.py` | Backtest mos_model across historical days |
| `train_spring_model.py` | Spring-specific model training |
| `predict_spring_max.py` | Spring max prediction |

---

## Services

| File | Purpose |
|---|---|
| `services/nws_timeseries.py` | Synoptic/NWS API client (requires pandas) |
| `services/synoptic.py` | Synoptic API wrapper |
| `services/metostat.py` | Meteostat data fetching |
| `services/ncei.py` | NCEI data fetching |

> Note: `nws_timeseries.py` requires pandas. On this machine pandas DLLs are blocked by Application Control policy. The fetch logic has been replicated inline in `predict_today.py` and `predict_by_nws.py` using only `requests`.

---

## Data files

| File | Contents |
|---|---|
| `training_data_synoptic_model/LLBG.2026-04-14 (1).csv` | Full station history 2018-10-26 to 2026-04-14 (~130k rows, 30-min intervals) |
| `nwp_cache.json` | ERA5 reanalysis cache for mos_model training |
| `spring_model_coeffs.json` | Coefficients for the spring-specific model |
| `image.png` | (Unknown — likely a plot from a previous run) |

---

## Quick decision guide

```
Do you have station data up to 12:00?
  YES → Do you also have NWP available?
           YES → use predict_today.py     (mos_model, best accuracy)
           NO  → use predict_by_nws.py    (nws_model, station-only)
  NO  → use predict_by_nws.py             (nws_model works at any hour)
```

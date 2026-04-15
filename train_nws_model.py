"""
train_nws_model.py  —  pred_by_nws
====================================
LSTM that reads the continuous stream of LLBG station observations since
midnight and outputs (mu, log_sigma) for the day's maximum temperature at
every timestep.  The prediction updates on every new observation — no
fixed cutoff time required.

Training objective
------------------
For every observation in every complete day, compute Gaussian NLL against
the day's true maximum.  This yields ~130k training signals from ~2,700
complete days and teaches the model to give accurate predictions at *any*
point during the day — increasingly confident as more readings arrive.

Architecture
------------
  LSTM  (input=10, hidden=64, layers=2, dropout=0.1)
    └─ mu_head:  Linear(64, 1)   → point prediction
    └─ sig_head: Linear(64, 1)   → log_sigma (uncertainty)

Training schedule
-----------------
  Phase 1 — MSE warmup (300 epochs, LSTM + mu_head only)
  Phase 2 — Gaussian NLL (up to 1000 epochs, all params, early stopping)
  Conformal calibration on test set → 95% PI with coverage guarantee.

Usage
-----
  python train_nws_model.py            # train and save nws_model.pt
  python train_nws_model.py evaluate   # load and print accuracy report
"""

import math, sys, random
from datetime import datetime, date
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
CSV_PATH   = BASE_DIR / "training_data_synoptic_model" / "LLBG.2026-04-14 (1).csv"
MODEL_SAVE = BASE_DIR / "nws_model.pt"

N_FEATURES    = 10
FEATURE_NAMES = [
    "air_temp", "dew_point", "wind_speed",
    "wd_sin", "wd_cos", "pressure",
    "time_sin", "time_cos", "doy_sin", "doy_cos",
]

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15   # test = chronological tail
BATCH_SIZE  = 32
HIDDEN_SIZE = 64
NUM_LAYERS  = 2


# ---------------------------------------------------------------------------
# 1.  CSV parser
# ---------------------------------------------------------------------------

def parse_csv(path: Path = CSV_PATH) -> dict:
    """
    Parse the Synoptic station CSV into per-day observation lists.
    Returns {date: [{"dt", "t", "dp", "ws", "wd", "p"}, ...]}.
      t  — air temp     (°C)
      dp — dew point    (°C)
      ws — wind speed   (m/s)
      wd — wind dir     (degrees)
      p  — sea-level pressure (hPa)
    """
    by_day = defaultdict(list)

    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]

    headers = [h.strip() for h in lines[0].split(",")]

    def _float(row, *keys):
        for k in keys:
            v = row.get(k, "").strip()
            if v:
                try:
                    return float(v)
                except ValueError:
                    pass
        return None

    n_loaded = 0
    for raw in lines[2:]:          # skip header + units row
        cols = raw.strip().split(",")
        if len(cols) < 4:
            continue
        row = dict(zip(headers, cols))

        dt_str = row.get("Date_Time", "").strip()
        if not dt_str:
            continue
        try:
            if len(dt_str) > 5 and dt_str[-5] in ('+', '-') and ':' not in dt_str[-5:]:
                dt_str = dt_str[:-2] + ":" + dt_str[-2:]
            dt = datetime.fromisoformat(dt_str).replace(tzinfo=None)
        except ValueError:
            continue

        t = _float(row, "air_temp_set_1")
        if t is None:
            continue

        dp    = _float(row, "dew_point_temperature_set_1d", "dew_point_temperature_set_1")
        ws    = _float(row, "wind_speed_set_1") or 0.0
        wd    = _float(row, "wind_direction_set_1") or 0.0
        p_raw = _float(row, "sea_level_pressure_set_1d")
        p     = p_raw / 100.0 if p_raw is not None else 1013.25   # Pa → hPa

        by_day[dt.date()].append({"dt": dt, "t": t, "dp": dp, "ws": ws, "wd": wd, "p": p})
        n_loaded += 1

    print(f"Parsed {n_loaded:,} records across {len(by_day)} days  "
          f"({min(by_day)} -> {max(by_day)})")

    for d in by_day:
        by_day[d].sort(key=lambda r: r["dt"])

    return dict(by_day)


# ---------------------------------------------------------------------------
# 2.  Feature extraction
# ---------------------------------------------------------------------------

def obs_to_features(obs: dict) -> list:
    """Convert one observation dict → 10-element feature vector."""
    wd_rad = math.radians(obs["wd"])
    dt     = obs["dt"]
    mins   = dt.hour * 60 + dt.minute
    doy    = dt.timetuple().tm_yday
    dp     = obs["dp"] if obs["dp"] is not None else obs["t"] - 10.0   # fallback
    return [
        obs["t"],
        dp,
        obs["ws"],
        math.sin(wd_rad),
        math.cos(wd_rad),
        obs["p"],
        math.sin(2 * math.pi * mins / 1440),
        math.cos(2 * math.pi * mins / 1440),
        math.sin(2 * math.pi * doy / 365),
        math.cos(2 * math.pi * doy / 365),
    ]


# ---------------------------------------------------------------------------
# 3.  Dataset construction
# ---------------------------------------------------------------------------

def build_dataset(by_day: dict) -> list:
    """
    Returns list of (sequence, daily_max) for complete days only.
    A day is complete when it has at least one observation at/after 18:00
    (ensures the true daily max is captured in the record).
    Each sequence is a list of 10-feature vectors, one per observation.
    """
    result, skipped = [], 0
    for d in sorted(by_day):
        recs = by_day[d]
        if not any(r["dt"].hour >= 18 for r in recs):
            skipped += 1
            continue
        seq   = [obs_to_features(r) for r in recs]
        label = max(r["t"] for r in recs)
        result.append((seq, label))

    print(f"Complete days: {len(result)}  (skipped {skipped} incomplete)")
    return result


# ---------------------------------------------------------------------------
# 4.  Scaler
# ---------------------------------------------------------------------------

def fit_scaler(data: list):
    """Z-score scaler fitted on all observations in the training split."""
    all_obs = [feat for seq, _ in data for feat in seq]
    n, k    = len(all_obs), len(all_obs[0])
    means   = [sum(o[j] for o in all_obs) / n for j in range(k)]
    stds    = [
        math.sqrt(sum((o[j] - means[j]) ** 2 for o in all_obs) / max(n - 1, 1))
        for j in range(k)
    ]
    stds = [max(s, 1e-8) for s in stds]
    return means, stds


def apply_scaler(data: list, means, stds) -> list:
    return [
        ([[(v - m) / s for v, m, s in zip(feat, means, stds)] for feat in seq], label)
        for seq, label in data
    ]


# ---------------------------------------------------------------------------
# 5.  PyTorch Dataset + collate
# ---------------------------------------------------------------------------

class DayDataset(Dataset):
    def __init__(self, data: list):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, label = self.data[idx]
        return (
            torch.tensor(seq,   dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


def collate_fn(batch):
    """Pad variable-length sequences to the longest in the batch."""
    seqs, labels = zip(*batch)
    lengths = [s.shape[0] for s in seqs]
    max_len = max(lengths)
    B, F    = len(seqs), seqs[0].shape[1]

    padded = torch.zeros(B, max_len, F)
    mask   = torch.zeros(B, max_len, dtype=torch.bool)
    for i, (s, l) in enumerate(zip(seqs, lengths)):
        padded[i, :l] = s
        mask[i, :l]   = True

    return padded, torch.stack(labels), mask


# ---------------------------------------------------------------------------
# 6.  Model
# ---------------------------------------------------------------------------

class NWSPredictor(nn.Module):
    """
    LSTM sequence model.  At each timestep the hidden state summarises all
    observations seen so far today; the prediction head converts it to
    (mu, log_sigma) for the day's maximum temperature.
    """
    def __init__(self, n_features=N_FEATURES,
                 hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.1 if num_layers > 1 else 0.0,
        )
        self.mu_head  = nn.Linear(hidden_size, 1)
        self.sig_head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.sig_head.weight)
        nn.init.zeros_(self.sig_head.bias)   # start at sigma=1 °C

    def forward(self, x):              # x: (B, T, F)
        h, _    = self.lstm(x)         # (B, T, H)
        mu      = self.mu_head(h).squeeze(-1)                      # (B, T)
        log_sig = self.sig_head(h).squeeze(-1).clamp(-3.0, 3.5)   # (B, T)
        return mu, log_sig


# ---------------------------------------------------------------------------
# 7.  Loss functions
# ---------------------------------------------------------------------------

def masked_mse(mu, y, mask):
    y_exp = y.unsqueeze(1).expand_as(mu)
    return ((mu - y_exp) ** 2 * mask).sum() / mask.sum()


def masked_nll(mu, log_sig, y, mask):
    y_exp = y.unsqueeze(1).expand_as(mu)
    nll   = log_sig + 0.5 * ((y_exp - mu) / log_sig.exp()) ** 2
    return (nll * mask).sum() / mask.sum()


# ---------------------------------------------------------------------------
# 8.  Training
# ---------------------------------------------------------------------------

def run_epoch(model, loader, opt=None, loss_fn="mse"):
    """One pass over the dataloader.  Returns (avg_loss, avg_mae_at_last_step)."""
    is_train = opt is not None
    model.train() if is_train else model.eval()
    total_loss = total_mae = n_days = 0

    for x, y, mask in loader:
        x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
        if is_train:
            mu, ls = model(x)
        else:
            with torch.no_grad():
                mu, ls = model(x)

        loss = masked_mse(mu, y, mask) if loss_fn == "mse" else masked_nll(mu, ls, y, mask)

        if is_train:
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        last_idx = mask.sum(dim=1) - 1                          # (B,)
        mu_last  = mu[torch.arange(len(y)), last_idx].detach()
        total_mae  += (mu_last - y).abs().sum().item()
        total_loss += loss.item() * len(y)
        n_days     += len(y)

    return total_loss / n_days, total_mae / n_days


def train_model(train_data, val_data, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)

    model     = NWSPredictor().to(DEVICE)
    tr_loader = DataLoader(DayDataset(train_data), batch_size=BATCH_SIZE,
                           shuffle=True, collate_fn=collate_fn)
    va_loader = DataLoader(DayDataset(val_data),   batch_size=BATCH_SIZE,
                           shuffle=False, collate_fn=collate_fn)

    # ------------------------------------------------------------------
    # Phase 1: MSE warmup — LSTM + mu_head only, sigma head frozen at 0
    # ------------------------------------------------------------------
    print("  [Phase 1] MSE warmup (300 epochs, mu head only) ...")
    mu_params = list(model.lstm.parameters()) + list(model.mu_head.parameters())
    opt1   = torch.optim.Adam(mu_params, lr=1e-3, weight_decay=1e-4)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=300, eta_min=1e-5)

    for ep in range(1, 301):
        run_epoch(model, tr_loader, opt=opt1, loss_fn="mse")
        sched1.step()
        if ep % 60 == 0:
            _, val_mae = run_epoch(model, va_loader, loss_fn="mse")
            print(f"    ep {ep:3d}  val_mae_last={val_mae:.3f} C")

    # ------------------------------------------------------------------
    # Phase 2: Gaussian NLL — all params, early stopping on val NLL
    # ------------------------------------------------------------------
    print("  [Phase 2] Gaussian NLL (up to 1000 epochs, all params) ...")
    opt2   = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=1000, eta_min=1e-6)
    best_val, best_state, patience = float("inf"), None, 0

    for ep in range(1, 1001):
        run_epoch(model, tr_loader, opt=opt2, loss_fn="nll")
        sched2.step()

        if ep % 10 == 0:
            val_nll, val_mae = run_epoch(model, va_loader, loss_fn="nll")
            if val_nll < best_val:
                best_val   = val_nll
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience   = 0
            else:
                patience  += 1
            if ep > 100 and patience > 30:
                print(f"    Early stop ep {ep}  val_nll={val_nll:.4f}  val_mae_last={val_mae:.3f} C")
                break
            if ep % 100 == 0:
                print(f"    ep {ep:4d}  val_nll={val_nll:.4f}  val_mae_last={val_mae:.3f} C")

    if best_state:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# 9.  Conformal calibration
# ---------------------------------------------------------------------------

def conformal_calibrate(model, test_data, alpha=0.05):
    """
    Split-conformal calibration on the test set (last timestep of each day).
    Returns q such that PI = mu ± q*sigma guarantees >= (1-alpha) coverage.
    """
    model.eval()
    loader = DataLoader(DayDataset(test_data), batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=collate_fn)
    scores = []
    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            mu, ls   = model(x)
            last_idx = mask.sum(dim=1) - 1
            mu_last  = mu[torch.arange(len(y)), last_idx]
            sig_last = ls[torch.arange(len(y)), last_idx].exp()
            scores.extend(((y - mu_last).abs() / sig_last).tolist())

    scores.sort()
    n     = len(scores)
    q_idx = min(math.ceil((n + 1) * (1 - alpha)) - 1, n - 1)
    q     = scores[q_idx]
    print(f"  Conformal q = {q:.4f}  (n={n}, target coverage={100*(1-alpha):.0f}%)")
    return q


# ---------------------------------------------------------------------------
# 10.  Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, data, conformal_q, label=""):
    """MAE + RMSE + 95% PI coverage at the last timestep of each day."""
    import numpy as np
    model.eval()
    loader = DataLoader(DayDataset(data), batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=collate_fn)

    all_mu, all_sig, all_y = [], [], []
    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            mu, ls   = model(x)
            last_idx = mask.sum(dim=1) - 1
            all_mu.extend(mu[torch.arange(len(y)), last_idx].tolist())
            all_sig.extend(ls[torch.arange(len(y)), last_idx].exp().tolist())
            all_y.extend(y.tolist())

    mu_arr  = np.array(all_mu)
    sig_arr = np.array(all_sig)
    y_arr   = np.array(all_y)
    err     = np.abs(y_arr - mu_arr)
    covered = err <= conformal_q * sig_arr

    print(f"\n{'-'*52}")
    print(f"  {label}  (n={len(y_arr)})")
    print(f"  MAE  (last timestep) : {err.mean():.3f} C")
    print(f"  RMSE                 : {np.sqrt((err**2).mean()):.3f} C")
    print(f"  Sigma  min/med/max   : "
          f"{sig_arr.min():.3f} / {np.median(sig_arr):.3f} / {sig_arr.max():.3f} C")
    print(f"  95% PI coverage      : {covered.mean()*100:.1f}%  (target: 95%)")


# ---------------------------------------------------------------------------
# 11.  Save / Load
# ---------------------------------------------------------------------------

def save_model(model, scaler_mean, scaler_std, conformal_q, path=MODEL_SAVE):
    torch.save({
        "model_state":   model.state_dict(),
        "scaler_mean":   scaler_mean,
        "scaler_std":    scaler_std,
        "conformal_q":   conformal_q,
        "n_features":    N_FEATURES,
        "feature_names": FEATURE_NAMES,
        "hidden_size":   HIDDEN_SIZE,
        "num_layers":    NUM_LAYERS,
    }, path)
    print(f"Model saved -> {path}")


def load_model(path=MODEL_SAVE):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No model at {path}. Run: python train_nws_model.py"
        )
    ckpt  = torch.load(path, map_location=DEVICE, weights_only=True)
    model = NWSPredictor(
        n_features  = ckpt["n_features"],
        hidden_size = ckpt.get("hidden_size", HIDDEN_SIZE),
        num_layers  = ckpt.get("num_layers",  NUM_LAYERS),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    return model, ckpt["scaler_mean"], ckpt["scaler_std"], ckpt["conformal_q"]


# ---------------------------------------------------------------------------
# 12.  CLI
# ---------------------------------------------------------------------------

def cmd_train():
    print("=" * 52)
    print("  pred_by_nws -- Training")
    print(f"  Device: {DEVICE}")
    print("=" * 52)

    by_day   = parse_csv()
    all_data = build_dataset(by_day)
    n        = len(all_data)
    if n < 50:
        sys.exit("Not enough complete days to train (need >= 50).")

    n_tr  = max(int(n * TRAIN_RATIO), 10)
    n_val = max(int(n * VAL_RATIO), 10)
    # test set is chronological tail
    train_raw = all_data[:n_tr]
    val_raw   = all_data[n_tr:n_tr + n_val]
    test_raw  = all_data[n_tr + n_val:]

    print(f"\n  Split: train={n_tr}  val={n_val}  test={len(test_raw)}")

    scaler_mean, scaler_std = fit_scaler(train_raw)
    train_sc = apply_scaler(train_raw, scaler_mean, scaler_std)
    val_sc   = apply_scaler(val_raw,   scaler_mean, scaler_std)
    test_sc  = apply_scaler(test_raw,  scaler_mean, scaler_std)

    print("\n  Training ...")
    model = train_model(train_sc, val_sc)

    print("\n  Conformal calibration on test set ...")
    q = conformal_calibrate(model, test_sc)

    evaluate(model, train_sc, q, "Train set")
    evaluate(model, val_sc,   q, "Val set")
    evaluate(model, test_sc,  q, "Test set")

    save_model(model, scaler_mean, scaler_std, q)


def cmd_evaluate():
    model, s_mean, s_std, q = load_model()
    by_day   = parse_csv()
    all_data = build_dataset(by_day)
    n        = len(all_data)
    n_tr     = max(int(n * TRAIN_RATIO), 10)
    n_val    = max(int(n * VAL_RATIO), 10)
    test_raw = all_data[n_tr + n_val:]
    test_sc  = apply_scaler(test_raw, s_mean, s_std)
    evaluate(model, test_sc, q, "Test set")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        cmd_train()
    elif cmd == "evaluate":
        cmd_evaluate()
    else:
        sys.exit(f"Unknown command: {cmd}. Use 'train' or 'evaluate'.")

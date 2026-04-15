"""
Comprehensive backtesting script for the MOS post-processing model.
Runs on all days in the dataset, computes statistics across splits,
months, seasons, confidence tiers, and temperature ranges.
Compares model skill against a raw NWP baseline.

Usage:
    python backtest.py          # all splits
    python backtest.py calib    # calibration set only (truly out-of-sample)
"""

import sys
import math
import statistics
from collections import defaultdict
from datetime import date

import torch

# Re-use everything from the main module
from postprocess_model import (
    build_dataset, load_model, scale, to_tensor,
    TRAIN_RATIO, VAL_RATIO, CALIB_RATIO, FEATURE_NAMES,
)

# Index of nwp_tmax in the feature vector (used for NWP baseline)
NWP_TMAX_IDX = FEATURE_NAMES.index("nwp_tmax")

MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

SEASON = {12:"Winter",1:"Winter",2:"Winter",
          3:"Spring",4:"Spring",5:"Spring",
          6:"Summer",7:"Summer",8:"Summer",
          9:"Autumn",10:"Autumn",11:"Autumn"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mae(errs):   return sum(abs(e) for e in errs) / len(errs) if errs else float("nan")
def _mbe(errs):   return sum(errs) / len(errs) if errs else float("nan")
def _rmse(errs):  return math.sqrt(sum(e**2 for e in errs) / len(errs)) if errs else float("nan")
def _pct(n, d):   return f"{n/d*100:5.1f}%" if d else "  n/a"

def _percentiles(vals, ps=(5, 25, 50, 75, 95)):
    if not vals: return {}
    sv = sorted(vals)
    n  = len(sv)
    out = {}
    for p in ps:
        idx = (p / 100) * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        out[p] = sv[lo] + (idx - lo) * (sv[hi] - sv[lo])
    return out


def _print_section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def _summary_block(label, errs, nwp_errs, covered, n_total, ci_widths, n_high, n_med, n_low):
    if not errs:
        print(f"  {label}: no data")
        return
    n = len(errs)
    mae  = _mae(errs)
    mbe  = _mbe(errs)
    rmse = _rmse(errs)
    nwp_mae  = _mae(nwp_errs) if nwp_errs else float("nan")
    skill    = (1 - mae / nwp_mae) * 100 if nwp_mae else float("nan")
    exact    = sum(1 for e in errs if abs(e) == 0)
    within1  = sum(1 for e in errs if abs(e) <= 1)
    within2  = sum(1 for e in errs if abs(e) <= 2)
    cov_pct  = covered / n_total * 100 if n_total else float("nan")
    avg_ciw  = sum(ci_widths) / len(ci_widths) if ci_widths else float("nan")
    abs_errs = [abs(e) for e in errs]
    pcts     = _percentiles(abs_errs)

    print(f"\n  {label}  (n={n})")
    print(f"  {'MAE':<22}: {mae:.3f} °C")
    print(f"  {'MBE (bias)':<22}: {mbe:+.3f} °C")
    print(f"  {'RMSE':<22}: {rmse:.3f} °C")
    print(f"  {'NWP baseline MAE':<22}: {nwp_mae:.3f} °C")
    print(f"  {'Skill vs NWP':<22}: {skill:+.1f}%")
    print(f"  {'Exact (±0 °C)':<22}: {exact}/{n} = {_pct(exact,n)}")
    print(f"  {'Within ±1 °C':<22}: {within1}/{n} = {_pct(within1,n)}")
    print(f"  {'Within ±2 °C':<22}: {within2}/{n} = {_pct(within2,n)}")
    print(f"  {'95% PI coverage':<22}: {covered}/{n_total} = {cov_pct:.1f}%  (target 95%)")
    print(f"  {'Avg PI width':<22}: ±{avg_ciw/2:.2f} °C")
    print(f"  {'Tier split H/M/L':<22}: {n_high}/{n_med}/{n_low}")
    print(f"  {'|err| percentiles':<22}: "
          f"p5={pcts.get(5,0):.1f}  p25={pcts.get(25,0):.1f}  "
          f"p50={pcts.get(50,0):.1f}  p75={pcts.get(75,0):.1f}  "
          f"p95={pcts.get(95,0):.1f} °C")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    calib_only = len(sys.argv) > 1 and sys.argv[1] == "calib"

    model, s_mean, s_std, q = load_model()
    X, y, dates = build_dataset()
    Xs = scale(X, s_mean, s_std)

    # Reconstruct split boundaries
    n      = len(X)
    n_cal  = max(int(n * CALIB_RATIO), 20)
    n_rest = n - n_cal
    n_val  = max(int(n_rest * (VAL_RATIO / (TRAIN_RATIO + VAL_RATIO))), 10)
    n_tr   = n_rest - n_val

    def split_label(i):
        if i < n_tr:         return "TR"
        if i < n_tr + n_val: return "VA"
        return "CA"

    # Run model
    model.eval()
    with torch.no_grad():
        mu_t, ls_t = model(to_tensor(Xs))
    mus    = mu_t.tolist()
    sigmas = ls_t.exp().tolist()
    ci_half = [q * s for s in sigmas]

    # NWP tmax baseline (raw NWP daily max, no post-processing)
    nwp_preds = [X[i][NWP_TMAX_IDX] for i in range(n)]

    # Build per-row records
    rows = []
    for i in range(n):
        sp      = split_label(i)
        dt      = dates[i]
        actual  = y[i]
        mu      = mus[i]
        sig     = sigmas[i]
        ci      = ci_half[i]
        nwp_p   = nwp_preds[i]
        tier    = "HIGH" if sig < 1.55 else ("MEDIUM" if sig < 2.0 else "LOW")
        err     = mu - actual
        nwp_err = nwp_p - actual
        in_ci   = abs(actual - mu) <= ci

        rows.append({
            "sp": sp, "dt": dt, "actual": actual,
            "mu": mu, "sigma": sig, "ci": ci,
            "err": err, "nwp_err": nwp_err,
            "in_ci": in_ci, "tier": tier,
            "ci_width": ci * 2,
        })

    # Filter to held-out (VA+CA) or calib-only
    if calib_only:
        held = [r for r in rows if r["sp"] == "CA"]
        scope = "CALIBRATION SET ONLY"
    else:
        held = [r for r in rows if r["sp"] != "TR"]
        scope = "HELD-OUT (VA + CA)"

    # -----------------------------------------------------------------------
    # 1. Overall summary
    # -----------------------------------------------------------------------
    _print_section(f"OVERALL — {scope}  ({dates[0]} → {dates[-1]})")
    _summary_block(
        scope,
        [r["err"] for r in held],
        [r["nwp_err"] for r in held],
        sum(r["in_ci"] for r in held),
        len(held),
        [r["ci_width"] for r in held],
        sum(r["tier"] == "HIGH"   for r in held),
        sum(r["tier"] == "MEDIUM" for r in held),
        sum(r["tier"] == "LOW"    for r in held),
    )

    # -----------------------------------------------------------------------
    # 2. By confidence tier
    # -----------------------------------------------------------------------
    _print_section("BY CONFIDENCE TIER")
    for tier in ("HIGH", "MEDIUM", "LOW"):
        subset = [r for r in held if r["tier"] == tier]
        if not subset: continue
        _summary_block(
            f"Tier: {tier}",
            [r["err"] for r in subset],
            [r["nwp_err"] for r in subset],
            sum(r["in_ci"] for r in subset),
            len(subset),
            [r["ci_width"] for r in subset],
            sum(r["tier"]=="HIGH" for r in subset),
            sum(r["tier"]=="MEDIUM" for r in subset),
            sum(r["tier"]=="LOW" for r in subset),
        )

    # -----------------------------------------------------------------------
    # 3. By month
    # -----------------------------------------------------------------------
    _print_section("BY MONTH")
    print(f"\n  {'Month':<6} {'n':>4}  {'MAE':>5}  {'MBE':>6}  {'NWP MAE':>7}  "
          f"{'Skill':>6}  {'±1°C':>6}  {'PI cov':>7}  {'AvgPIw':>7}")
    print(f"  {'-'*70}")
    for m in range(1, 13):
        sub = [r for r in held if r["dt"].month == m]
        if not sub: continue
        errs     = [r["err"] for r in sub]
        nwp_errs = [r["nwp_err"] for r in sub]
        nwp_mae  = _mae(nwp_errs)
        mae      = _mae(errs)
        skill    = (1 - mae / nwp_mae) * 100 if nwp_mae else float("nan")
        within1  = sum(1 for e in errs if abs(e) <= 1)
        covered  = sum(r["in_ci"] for r in sub)
        avg_ciw  = sum(r["ci_width"] for r in sub) / len(sub)
        print(f"  {MONTH_NAMES[m]:<6} {len(sub):>4}  {mae:>5.2f}  {_mbe(errs):>+6.2f}  "
              f"{nwp_mae:>7.2f}  {skill:>+5.1f}%  "
              f"{_pct(within1,len(sub)):>6}  {_pct(covered,len(sub)):>7}  "
              f"±{avg_ciw/2:>5.2f}")

    # -----------------------------------------------------------------------
    # 4. By season
    # -----------------------------------------------------------------------
    _print_section("BY SEASON")
    print(f"\n  {'Season':<8} {'n':>4}  {'MAE':>5}  {'MBE':>6}  {'NWP MAE':>7}  "
          f"{'Skill':>6}  {'±1°C':>6}  {'PI cov':>7}")
    print(f"  {'-'*60}")
    season_rows = defaultdict(list)
    for r in held:
        season_rows[SEASON[r["dt"].month]].append(r)
    for s in ("Spring", "Summer", "Autumn", "Winter"):
        sub = season_rows[s]
        if not sub: continue
        errs    = [r["err"] for r in sub]
        nwp_mae = _mae([r["nwp_err"] for r in sub])
        mae     = _mae(errs)
        skill   = (1 - mae / nwp_mae) * 100 if nwp_mae else float("nan")
        within1 = sum(1 for e in errs if abs(e) <= 1)
        covered = sum(r["in_ci"] for r in sub)
        print(f"  {s:<8} {len(sub):>4}  {mae:>5.2f}  {_mbe(errs):>+6.2f}  "
              f"{nwp_mae:>7.2f}  {skill:>+5.1f}%  "
              f"{_pct(within1,len(sub)):>6}  {_pct(covered,len(sub)):>7}")

    # -----------------------------------------------------------------------
    # 5. By temperature range
    # -----------------------------------------------------------------------
    _print_section("BY ACTUAL DAILY MAX TEMPERATURE")
    print(f"\n  {'Range':<14} {'n':>4}  {'MAE':>5}  {'MBE':>6}  {'NWP MAE':>7}  "
          f"{'Skill':>6}  {'±1°C':>6}  {'PI cov':>7}")
    print(f"  {'-'*65}")
    bins = [
        ("<15 °C",   lambda a: a < 15),
        ("15–19 °C", lambda a: 15 <= a < 20),
        ("20–24 °C", lambda a: 20 <= a < 25),
        ("25–29 °C", lambda a: 25 <= a < 30),
        ("≥30 °C",   lambda a: a >= 30),
    ]
    for label, fn in bins:
        sub = [r for r in held if fn(r["actual"])]
        if not sub: continue
        errs    = [r["err"] for r in sub]
        nwp_mae = _mae([r["nwp_err"] for r in sub])
        mae     = _mae(errs)
        skill   = (1 - mae / nwp_mae) * 100 if nwp_mae else float("nan")
        within1 = sum(1 for e in errs if abs(e) <= 1)
        covered = sum(r["in_ci"] for r in sub)
        print(f"  {label:<14} {len(sub):>4}  {mae:>5.2f}  {_mbe(errs):>+6.2f}  "
              f"{nwp_mae:>7.2f}  {skill:>+5.1f}%  "
              f"{_pct(within1,len(sub)):>6}  {_pct(covered,len(sub)):>7}")

    # -----------------------------------------------------------------------
    # 6. Error distribution
    # -----------------------------------------------------------------------
    _print_section("ERROR DISTRIBUTION  (model error = pred - actual)")
    errs = [r["err"] for r in held]
    buckets = defaultdict(int)
    for e in errs:
        b = int(math.floor(e)) if e >= 0 else int(math.ceil(e - 1))
        b = max(-5, min(5, b))
        buckets[b] += 1
    print(f"\n  {'Error bin':<12}  {'Count':>6}  {'%':>6}  Bar")
    print(f"  {'-'*50}")
    for b in range(-5, 6):
        cnt = buckets.get(b, 0)
        bar = "█" * int(cnt / len(errs) * 60)
        print(f"  [{b:+d} to {b+1:+d})  {cnt:>6}   {cnt/len(errs)*100:>5.1f}%  {bar}")

    # -----------------------------------------------------------------------
    # 7. Worst 15 days (held-out only)
    # -----------------------------------------------------------------------
    _print_section("15 WORST PREDICTIONS  (held-out, by |error|)")
    worst = sorted(held, key=lambda r: abs(r["err"]), reverse=True)[:15]
    print(f"\n  {'Date':<12} {'Sp':2}  {'Act':>4} {'Pred':>5} {'Err':>5}  "
          f"{'NWPmax':>6}  {'NWPerr':>6}  {'Tier':<7}  {'In PI':>5}")
    print(f"  {'-'*70}")
    for r in worst:
        nwp_pred = round(r["actual"] + r["nwp_err"])
        print(f"  {r['dt'].strftime('%d/%m/%Y'):<12} [{r['sp']}]  "
              f"{r['actual']:>4.0f}  {round(r['mu']):>5}  {r['err']:>+5.1f}  "
              f"{nwp_pred:>6}  {r['nwp_err']:>+6.1f}  "
              f"{r['tier']:<7}  {'yes' if r['in_ci'] else 'NO':>5}")

    # -----------------------------------------------------------------------
    # 8. Year-over-year trend (VA+CA only)
    # -----------------------------------------------------------------------
    _print_section("YEAR-OVER-YEAR  (held-out days per calendar year)")
    print(f"\n  {'Year':<6} {'n':>4}  {'MAE':>5}  {'MBE':>6}  {'NWP MAE':>7}  "
          f"{'Skill':>6}  {'±1°C':>6}  {'PI cov':>7}")
    print(f"  {'-'*60}")
    year_rows = defaultdict(list)
    for r in held:
        year_rows[r["dt"].year].append(r)
    for yr in sorted(year_rows):
        sub     = year_rows[yr]
        errs    = [r["err"] for r in sub]
        nwp_mae = _mae([r["nwp_err"] for r in sub])
        mae     = _mae(errs)
        skill   = (1 - mae / nwp_mae) * 100 if nwp_mae else float("nan")
        within1 = sum(1 for e in errs if abs(e) <= 1)
        covered = sum(r["in_ci"] for r in sub)
        print(f"  {yr:<6} {len(sub):>4}  {mae:>5.2f}  {_mbe(errs):>+6.2f}  "
              f"{nwp_mae:>7.2f}  {skill:>+5.1f}%  "
              f"{_pct(within1,len(sub)):>6}  {_pct(covered,len(sub)):>7}")

    print(f"\n{'='*72}\n")


if __name__ == "__main__":
    main()

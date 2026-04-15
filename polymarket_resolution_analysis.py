"""
Polymarket Resolution Analysis - Tel Aviv Max Temp April 13, 2026
https://polymarket.com/event/highest-temperature-in-tel-aviv-on-april-13-2026

Finds the point at which the market price made the outcome effectively "known",
i.e., when probability crossed key thresholds (70%, 90%, 95%, 99%).
"""

from datetime import datetime, timezone, timedelta

ISRAEL_TZ = timezone(timedelta(hours=3))  # UTC+3 (no DST currently in April)

# All price history for the 22°C YES token (the winning outcome)
# Fetched from: clob.polymarket.com/prices-history
# Combined: pre-April-13 daily snapshots + April-13 fine-grained (fidelity=10)
RAW_HISTORY_22C = [
    # --- Pre-April-13 context (daily fidelity) ---
    {"t": 1775710849, "p": 0.175},
    {"t": 1775714459, "p": 0.175},
    {"t": 1775718047, "p": 0.175},
    {"t": 1775721648, "p": 0.175},
    {"t": 1775728849, "p": 0.17},
    {"t": 1775732450, "p": 0.18},
    {"t": 1775739654, "p": 0.17},
    {"t": 1775750456, "p": 0.23},
    {"t": 1775764859, "p": 0.265},
    {"t": 1775775659, "p": 0.215},
    {"t": 1775782858, "p": 0.23},
    {"t": 1775786455, "p": 0.18},
    {"t": 1775797259, "p": 0.15},
    {"t": 1775800826, "p": 0.165},
    {"t": 1775804454, "p": 0.16},
    {"t": 1775818853, "p": 0.14},
    {"t": 1775826004, "p": 0.16},
    {"t": 1775833211, "p": 0.26},
    {"t": 1775840420, "p": 0.255},
    {"t": 1775862003, "p": 0.255},
    {"t": 1775869257, "p": 0.275},
    {"t": 1775890854, "p": 0.25},
    {"t": 1775894452, "p": 0.26},
    {"t": 1775901620, "p": 0.245},
    {"t": 1775905254, "p": 0.285},
    {"t": 1775955654, "p": 0.3},
    {"t": 1775959259, "p": 0.305},
    {"t": 1775966458, "p": 0.345},
    {"t": 1775977254, "p": 0.345},
    {"t": 1775991654, "p": 0.36},
    {"t": 1776042054, "p": 0.375},
    {"t": 1776049257, "p": 0.445},
    {"t": 1776052847, "p": 0.485},
    {"t": 1776063656, "p": 0.485},
    {"t": 1776067255, "p": 0.52},
    # --- April 13 fine-grained (~10-min fidelity) ---
    {"t": 1776039044, "p": 0.365},
    {"t": 1776039642, "p": 0.375},
    {"t": 1776040842, "p": 0.375},
    {"t": 1776041442, "p": 0.375},
    {"t": 1776042054, "p": 0.375},
    {"t": 1776042648, "p": 0.37},
    {"t": 1776043253, "p": 0.37},
    {"t": 1776044449, "p": 0.365},
    {"t": 1776045005, "p": 0.38},
    {"t": 1776046244, "p": 0.46},
    {"t": 1776046844, "p": 0.465},
    {"t": 1776047455, "p": 0.47},
    {"t": 1776048043, "p": 0.475},
    {"t": 1776048643, "p": 0.445},
    {"t": 1776049257, "p": 0.445},
    {"t": 1776049853, "p": 0.485},
    {"t": 1776050442, "p": 0.485},
    {"t": 1776051058, "p": 0.485},
    {"t": 1776051654, "p": 0.485},
    {"t": 1776052243, "p": 0.485},
    {"t": 1776052847, "p": 0.485},
    {"t": 1776053442, "p": 0.485},
    {"t": 1776054056, "p": 0.455},
    {"t": 1776054655, "p": 0.45},
    {"t": 1776055258, "p": 0.435},
    {"t": 1776055847, "p": 0.425},
    {"t": 1776057045, "p": 0.44},
    {"t": 1776057643, "p": 0.44},
    {"t": 1776058843, "p": 0.545},
    {"t": 1776059443, "p": 0.53},
    {"t": 1776060643, "p": 0.515},
    {"t": 1776061244, "p": 0.51},
    {"t": 1776062444, "p": 0.445},
    {"t": 1776063045, "p": 0.495},
    {"t": 1776063656, "p": 0.485},
    {"t": 1776064244, "p": 0.525},
    {"t": 1776064844, "p": 0.59},
    {"t": 1776065456, "p": 0.575},
    {"t": 1776066048, "p": 0.545},
    {"t": 1776066644, "p": 0.59},
    {"t": 1776067255, "p": 0.52},
    {"t": 1776067852, "p": 0.55},
    {"t": 1776068443, "p": 0.565},
    {"t": 1776069643, "p": 0.565},
    {"t": 1776070245, "p": 0.435},
    {"t": 1776071458, "p": 0.455},
    {"t": 1776072044, "p": 0.46},
    {"t": 1776073248, "p": 0.495},
    {"t": 1776073845, "p": 0.635},
    {"t": 1776075055, "p": 0.69},
    {"t": 1776075646, "p": 0.695},
    {"t": 1776076845, "p": 0.695},
    {"t": 1776077446, "p": 0.695},
    {"t": 1776078644, "p": 0.695},
    {"t": 1776079244, "p": 0.695},
    {"t": 1776080443, "p": 0.595},
    {"t": 1776081045, "p": 0.305},  # brief anomalous dip (thin orderbook)
    {"t": 1776082244, "p": 0.94},
    {"t": 1776082855, "p": 0.925},
    {"t": 1776084046, "p": 0.975},
    {"t": 1776084644, "p": 0.9705},
    {"t": 1776085844, "p": 0.9905},
    {"t": 1776086445, "p": 0.989},
    {"t": 1776087054, "p": 0.994},
    {"t": 1776087643, "p": 0.9945},
    {"t": 1776088248, "p": 0.9945},
    {"t": 1776089445, "p": 0.995},
    {"t": 1776090059, "p": 0.9975},
    {"t": 1776091246, "p": 0.9975},
    {"t": 1776091848, "p": 0.9945},
    {"t": 1776093042, "p": 0.9985},
    {"t": 1776093652, "p": 0.9985},
    {"t": 1776095448, "p": 0.9985},
    {"t": 1776096655, "p": 0.9995},
    {"t": 1776097246, "p": 0.9995},
    {"t": 1776098449, "p": 0.9995},
    {"t": 1776099048, "p": 0.9995},
    {"t": 1776100245, "p": 0.9995},
    {"t": 1776100845, "p": 0.9995},
    {"t": 1776102004, "p": 0.9995},
    {"t": 1776102645, "p": 0.9995},
    {"t": 1776103845, "p": 0.9995},
    {"t": 1776104408, "p": 0.9995},
    {"t": 1776106249, "p": 0.9995},
    {"t": 1776107447, "p": 0.9995},
    {"t": 1776108046, "p": 0.9995},
    {"t": 1776109249, "p": 0.9995},
    {"t": 1776109849, "p": 0.9995},
    {"t": 1776111048, "p": 0.9995},
    {"t": 1776111647, "p": 0.9995},
    {"t": 1776112846, "p": 0.9995},
    {"t": 1776113445, "p": 0.9995},
    {"t": 1776114646, "p": 0.9995},
    {"t": 1776115247, "p": 0.9995},
    {"t": 1776116448, "p": 0.9995},
    {"t": 1776117048, "p": 0.9995},
    {"t": 1776118255, "p": 0.9995},
    {"t": 1776118845, "p": 0.9995},
    {"t": 1776120048, "p": 0.9995},
    {"t": 1776120648, "p": 0.9995},
    {"t": 1776121849, "p": 0.9995},
    {"t": 1776122445, "p": 0.9995},
    {"t": 1776123645, "p": 0.9995},
    {"t": 1776124248, "p": 0.9995},
]

def ts_to_dt(ts):
    return datetime.fromtimestamp(ts, tz=ISRAEL_TZ)

def fmt(ts):
    dt = ts_to_dt(ts)
    return dt.strftime("%Y-%m-%d %H:%M IST")

def analyze():
    # Deduplicate and sort by timestamp
    seen = {}
    for pt in RAW_HISTORY_22C:
        seen[pt["t"]] = pt["p"]
    history = sorted([{"t": t, "p": p} for t, p in seen.items()], key=lambda x: x["t"])

    APRIL_13_START = 1776038400  # April 13 00:00 UTC
    APRIL_13_END   = 1776124800  # April 14 00:00 UTC

    april13 = [pt for pt in history if APRIL_13_START <= pt["t"] <= APRIL_13_END]

    thresholds = [0.70, 0.90, 0.95, 0.99, 0.999]

    print("=" * 65)
    print("Polymarket: Highest Temp in Tel Aviv – April 13, 2026")
    print("Winning outcome: 22°C  |  Resolution source: NOAA @ LLBG")
    print("=" * 65)

    print("\n--- 22°C YES price timeline on April 13 (IST) ---\n")
    for pt in april13:
        bar = "#" * int(pt["p"] * 30)
        print(f"  {fmt(pt['t'])}  {pt['p']:.4f}  {bar}")

    print("\n--- Threshold crossing times (first sustained crossing) ---\n")

    # Skip known anomalous dip at 1776081045
    ANOMALY_TS = 1776081045

    for thr in thresholds:
        # Find first time p >= thr that isn't the anomaly dip point and stays >= thr
        cross_ts = None
        for i, pt in enumerate(april13):
            if pt["t"] == ANOMALY_TS:
                continue
            if pt["p"] >= thr:
                # Confirm next point (if any) is also >= thr or market is essentially done
                next_pts = [p for p in april13[i+1:] if p["t"] != ANOMALY_TS]
                if not next_pts or next_pts[0]["p"] >= thr - 0.05:
                    cross_ts = pt["t"]
                    cross_p  = pt["p"]
                    break

        if cross_ts:
            dt_utc = datetime.fromtimestamp(cross_ts, tz=timezone.utc)
            dt_ist = ts_to_dt(cross_ts)
            hours_before_midnight = (APRIL_13_END - cross_ts) / 3600
            print(f"  >{thr*100:.1f}%  =>  {dt_ist.strftime('%H:%M IST')}  "
                  f"(UTC {dt_utc.strftime('%H:%M')})  "
                  f"price={cross_p:.4f}  "
                  f"[{hours_before_midnight:.1f}h before midnight]")
        else:
            print(f"  >{thr*100:.1f}%  →  never crossed on April 13")

    print()
    print("--- Pre-April-13 trajectory (daily snapshots) ---\n")
    pre = [pt for pt in history if pt["t"] < APRIL_13_START]
    if pre:
        # Show last ~10 points leading into April 13
        for pt in pre[-10:]:
            dt = ts_to_dt(pt["t"])
            print(f"  {dt.strftime('%Y-%m-%d %H:%M IST')}  {pt['p']:.3f}")

    print()
    print("Notes:")
    print("  • Anomalous dip to 0.305 at 14:50 IST excluded (thin orderbook artifact).")
    print("  • Price data from Polymarket CLOB API (fidelity=10 min on April 13).")
    print("  • Market closed / auto-resolved on April 14, 2026.")

if __name__ == "__main__":
    analyze()

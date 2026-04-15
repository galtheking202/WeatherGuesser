"""
Polymarket monthly resolution analysis - Tel Aviv highest temp markets
Fetches ~30 daily markets from the past month, finds when each one
became "basically resolved" (price crossed 90%/95%/99%), then computes
mean + std of the crossing times (hours before end-of-day in IST).
"""

import time
import json
import statistics
import sys
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

ISRAEL_TZ = timezone(timedelta(hours=3))

def fetch_json(url, retries=3, delay=1.0):
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except (HTTPError, URLError) as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return None
        except Exception:
            return None

def get_event_for_date(date: datetime):
    """Return (winning_market_id, winning_yes_token_id, resolved_temp, day_end_ts) or None."""
    month_name = date.strftime("%B").lower()   # e.g. "april"
    day        = date.day                       # e.g. 13
    slug = f"highest-temperature-in-tel-aviv-on-{month_name}-{day}-2026"

    url = f"https://gamma-api.polymarket.com/events?slug={slug}"
    data = fetch_json(url)
    if not data:
        return None

    # Handle both list and dict responses
    events = data if isinstance(data, list) else [data]
    if not events:
        return None

    event = events[0]
    markets = event.get("markets", [])
    if not markets:
        return None

    def parse_json_field(val, default=None):
        if default is None:
            default = []
        if isinstance(val, str):
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                return default
        return val if val is not None else default

    # Find the winning (resolved YES) market
    winner = None
    for m in markets:
        raw_tokens = m.get("clobTokenIds", [])
        tokens = parse_json_field(raw_tokens)
        if not tokens:
            continue
        # outcomePrices is also a JSON-encoded string like '["1", "0"]'
        outcomes = parse_json_field(m.get("outcomePrices", "[]"))
        try:
            yes_price = float(outcomes[0]) if outcomes else 0
        except (ValueError, TypeError):
            yes_price = 0
        if yes_price >= 0.95:
            winner = m
            winner["_tokens"] = tokens  # cache parsed tokens
            break

    if not winner:
        # Fallback: highest volume
        winner = max(markets, key=lambda m: float(m.get("volume", 0) or 0), default=None)

    if not winner:
        return None

    tokens = winner.get("_tokens") or parse_json_field(winner.get("clobTokenIds", []))
    if not tokens:
        return None

    yes_token = tokens[0]  # first token is YES

    # Use UTC midnight as day boundaries (same as working April 13 call)
    next_day = date + timedelta(days=1)
    day_start_ts = int(datetime(date.year, date.month, date.day, 0, 0, 0, tzinfo=timezone.utc).timestamp())
    day_end_ts   = int(datetime(next_day.year, next_day.month, next_day.day, 0, 0, 0, tzinfo=timezone.utc).timestamp())

    # For "hours before EOD" calculation, use 23:59 IST as reference
    day_end_ist_ts = int(datetime(date.year, date.month, date.day, 23, 59, 59, tzinfo=ISRAEL_TZ).timestamp())

    return {
        "date": date.strftime("%Y-%m-%d"),
        "slug": slug,
        "market_id": winner.get("id"),
        "yes_token": yes_token,
        "question": winner.get("question", ""),
        "day_start_ts": day_start_ts,
        "day_end_ts": day_end_ts,
        "day_end_ist_ts": day_end_ist_ts,
    }

def get_price_history(token_id, start_ts, end_ts, fidelity=10):
    url = (
        f"https://clob.polymarket.com/prices-history"
        f"?market={token_id}&startTs={start_ts}&endTs={end_ts}&fidelity={fidelity}"
    )
    data = fetch_json(url)
    if not data:
        return []
    return data.get("history", [])

def first_threshold_crossing(history, threshold, day_end_ts, anomaly_window=600):
    """
    Return the Unix timestamp of the first SUSTAINED crossing above `threshold`.
    "Sustained" = the next non-anomalous point is also >= (threshold - 0.05).
    Ignores brief spikes/dips within `anomaly_window` seconds.
    Returns None if never crossed.
    """
    pts = sorted(history, key=lambda x: x["t"])
    for i, pt in enumerate(pts):
        if pt["p"] >= threshold:
            # Check next point isn't a reversion
            nxt = [p for p in pts[i+1:] if abs(p["t"] - pt["t"]) > anomaly_window]
            if not nxt or nxt[0]["p"] >= threshold - 0.05:
                return pt["t"]
    return None

def hours_before_eod(ts, day_end_ts):
    """Hours before end of day (positive = earlier in the day)."""
    return (day_end_ts - ts) / 3600

def fmt_ist(ts):
    dt = datetime.fromtimestamp(ts, tz=ISRAEL_TZ)
    return dt.strftime("%H:%M")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

# Generate dates: April 12 back to March 14 (30 days)
START_DATE = datetime(2026, 3, 14, tzinfo=ISRAEL_TZ)
END_DATE   = datetime(2026, 4, 12, tzinfo=ISRAEL_TZ)

dates = []
d = END_DATE
while d >= START_DATE:
    dates.append(d)
    d -= timedelta(days=1)

THRESHOLDS = [0.90, 0.95, 0.99]

results = []   # list of dicts per day

print(f"Fetching {len(dates)} daily markets...\n")

for i, date in enumerate(dates):
    label = date.strftime("%Y-%m-%d")
    sys.stdout.write(f"  [{i+1:2d}/{len(dates)}] {label} ... ")
    sys.stdout.flush()

    info = get_event_for_date(date)
    if not info:
        print("SKIP (no event found)")
        continue

    history = get_price_history(
        info["yes_token"],
        info["day_start_ts"],
        info["day_end_ts"],
        fidelity=10
    )
    time.sleep(0.3)  # polite rate limiting

    if not history:
        print("SKIP (no price history)")
        continue

    day_result = {
        "date": info["date"],
        "question": info["question"],
        "n_points": len(history),
    }

    crossing_info = []
    for thr in THRESHOLDS:
        ts = first_threshold_crossing(history, thr, info["day_end_ist_ts"])
        if ts:
            h = hours_before_eod(ts, info["day_end_ist_ts"])
            day_result[f"thr_{int(thr*100)}"] = h
            crossing_info.append(f"{int(thr*100)}%@{fmt_ist(ts)}({h:.1f}h)")
        else:
            day_result[f"thr_{int(thr*100)}"] = None

    print(f"OK  {' | '.join(crossing_info) if crossing_info else 'no crossings'}")
    results.append(day_result)

# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 70)
print("  RESULTS: hours-before-end-of-day when threshold was first crossed")
print("  (IST end-of-day = 23:59)  Positive = earlier in the day")
print("=" * 70)

print(f"\n{'Date':<12}", end="")
for t in THRESHOLDS:
    print(f"  {int(t*100)}%@h  ", end="")
print()

for r in results:
    print(f"{r['date']:<12}", end="")
    for t in THRESHOLDS:
        v = r.get(f"thr_{int(t*100)}")
        print(f"  {v:>6.1f}  " if v is not None else "     -    ", end="")
    print()

print()
print("-" * 70)

for t in THRESHOLDS:
    key = f"thr_{int(t*100)}"
    vals = [r[key] for r in results if r.get(key) is not None]
    if len(vals) < 2:
        print(f"  >{int(t*100)}%  n={len(vals)}  insufficient data")
        continue

    mean = statistics.mean(vals)
    std  = statistics.stdev(vals)
    med  = statistics.median(vals)
    mn   = min(vals)
    mx   = max(vals)

    # Convert "hours before EOD" back to approximate IST time
    # EOD = 23:59, so crossing_hour_ist = 23.983 - mean
    mean_ist_h = 23.983 - mean
    mean_ist   = f"{int(mean_ist_h):02d}:{int((mean_ist_h % 1)*60):02d}"

    std_min = std * 60

    print(f"  >{int(t*100)}%  n={len(vals):2d}")
    print(f"        mean = {mean:.2f}h before EOD  (~{mean_ist} IST)")
    print(f"        std  = {std:.2f}h  ({std_min:.0f} min)")
    print(f"        med  = {med:.2f}h  min={mn:.2f}h  max={mx:.2f}h")
    print()

print("-" * 70)
print(f"  Analysis covers {len(results)} trading days  "
      f"({dates[-1].strftime('%b %d')} – {dates[0].strftime('%b %d %Y')})")

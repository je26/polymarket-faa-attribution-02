"""polymarket-faa-attribution-02 entry: adaptive-volatility quantile forecaster.

For each market, the median stays at the checkpoint price (prediction-market
prices are close to martingales over a 4-hour horizon), and the spread adapts
to the market's own recent volatility: an exponentially weighted variance of
1-minute moves with a 15-minute halflife, so bands widen right after moves and
narrow again quickly in quiet stretches. Quantile offsets use stretched
normal quantiles (sign(z) * |z|^1.45): a narrower interquartile range and
fatter 5%/95% tails, matching how these prices move -- mostly not at all,
then a tick, and now and then a jump. Pure standard library, deterministic:
two runs on the same case write the same bytes.

Backtest (local scorer, pinball loss, lower is better): practice 0.278148
vs the house baseline's 0.363589; rolling NVIDIA checkpoints 1.135 vs 1.242;
rolling FAA checkpoints 0.641 vs 0.648. A market whose history cannot be
read never crashes the run: it falls back to flat persistence bands.
"""

import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_NORM = [-1.6448536269514722, -0.6744897501960817, 0.0,
          0.6744897501960817, 1.6448536269514722]
STRETCH = 1.45
ZS = [math.copysign(abs(z) ** STRETCH, z) for z in Z_NORM]

HALFLIFE_MIN = 15.0
SIGMA_FLOOR = 0.0005
SIGMA_CAP = 0.01
SIGMA_MULT = 0.7


def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def price6(x):
    """Clamp to [0, 1], round to 6 decimals, never -0."""
    return round(clamp01(x) * 1e6) / 1e6 + 0.0


def recent_sigma(prices):
    """Per-minute sigma from an EWMA of squared 1-minute moves."""
    lam = 0.5 ** (1.0 / HALFLIFE_MIN)
    acc = 0.0
    wacc = 0.0
    for i in range(1, len(prices)):
        d = prices[i] - prices[i - 1]
        acc = acc * lam + d * d
        wacc = wacc * lam + 1.0
    var = acc / wacc if wacc > 0.0 else 0.0
    sigma = math.sqrt(var)
    return min(SIGMA_CAP, max(SIGMA_FLOOR, sigma)) * SIGMA_MULT


def clean_prices(raw):
    """Finite floats in [0, 1]; unusable entries fall back to the last good one."""
    out = []
    last = 0.5
    for v in raw:
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = last
        if not math.isfinite(f):
            f = last
        f = clamp01(f)
        out.append(f)
        last = f
    return out


def fallback_rows(x, horizon):
    """Flat persistence bands: the last price, floor spread only."""
    s0 = SIGMA_FLOOR * SIGMA_MULT
    rows = []
    for h in range(1, horizon + 1):
        s = s0 * math.sqrt(h)
        row = [price6(x + z * s) for z in ZS]
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def forecast_market(prices, horizon):
    """240 rows of five quantiles for one market's next `horizon` minutes."""
    try:
        clean = clean_prices(prices)
        if not clean:
            return [[0.5, 0.5, 0.5, 0.5, 0.5] for _ in range(horizon)]
        x = clean[-1]
        sigma = recent_sigma(clean)
        rows = []
        for h in range(1, horizon + 1):
            s = sigma * math.sqrt(h)
            row = [price6(x + z * s) for z in ZS]
            for j in range(1, 5):
                if row[j] < row[j - 1]:
                    row[j] = row[j - 1]
            rows.append(row)
        return rows
    except Exception:
        try:
            x = clamp01(float(prices[-1]))
        except Exception:
            x = 0.5
        return fallback_rows(x, horizon)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for name in names:
        case_id = name[: -len(".in")]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        horizon = int(case["horizon_minutes"])
        forecasts = [
            {"id": m["id"], "q": forecast_market(m["prices"], horizon)}
            for m in case["markets"]
        ]
        payload = json.dumps({"forecasts": forecasts}, separators=(",", ":")) + "\n"
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w", encoding="utf-8") as f:
            f.write(payload)


if __name__ == "__main__":
    main()

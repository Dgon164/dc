"""B4 — TREND / FORECAST (segment plan §3). Predictions P15-P18
filed FIRST in PREDICTIONS.md (2026-07-17).

Legs, each honestly labeled:
  P15 planted seasonal+trend  — MASE vs naive and seasonal-naive
      (the competition's own baseline family); the wave is in the
      library, Isaac should beat both.
  P16 public seasonal series  — sunspots monthly (SIDC, the
      gauntlet's own source). The M4-subset leg is recorded BLOCKED
      for v1 (the train file is a ~300 MB download; blocked is
      SAID, never quietly swapped for synthetic).
  P17 RAW RETURNS (the headline) — real daily index closes (stooq),
      log returns, K rolling windows: the deliverable is the
      REFUSAL RATE. A minted return-forecast law that survives
      fresh rows would be the lane's r2-ghost.
  P18 REALIZED VOLATILITY (the pair) — |returns| on the same
      windows: structure FOUND where raw returns refused proves the
      refusal is discrimination, not blindness.

Network legs degrade honestly: unreachable source -> leg BLOCKED
with the why, recorded in the results file.

Run: PYTHONHASHSEED=0 python bench_b4_trend.py
Writes results_b4.jsonl next to this file.
"""
import io
import json
import os
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DC = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, DC)
sys.path.insert(0, HERE)

from isaac_store import bridge                       # noqa: E402

HOLD = 24                                            # forecast horizon
SEASON = 12


def mase(forecast, actual, insample, m=SEASON):
    """M4's own metric: MAE scaled by the in-sample seasonal-naive
    MAE (m=12 monthly)."""
    denom = np.mean(np.abs(insample[m:] - insample[:-m]))
    return float(np.mean(np.abs(forecast - actual)) / max(denom, 1e-12))


def forecast_series(mod, s, name, horizon=HOLD):
    """Isaac's road: discover y(t) on the training span (time column
    + the engine's own lag proposals), apply_law on future t. A
    refusal returns None WITH the verdict — the refusal travels."""
    m = mod.Isaac()
    n = len(s)
    t = np.arange(n, dtype=float).reshape(-1, 1)
    r = m.discover(t, s, name=name)
    if r["verdict"] in ("UNDERSTOOD", "PARTIAL"):
        tf = np.arange(n, n + horizon, dtype=float).reshape(-1, 1)
        pred = m.apply_law(name, tf)
        if pred is not None and np.isfinite(pred).all():
            return np.asarray(pred, float), r
    return None, r


def leg_p15(mod, recs, rng):
    n = 240 + HOLD
    t = np.arange(n)
    s = 10 + 0.05 * t + 3.0 * np.sin(2 * np.pi * t / SEASON) \
        + rng.normal(0, 0.5, n)
    tr, te = s[:-HOLD], s[-HOLD:]
    pred, r = forecast_series(mod, tr, "b4:planted")
    naive = np.full(HOLD, tr[-1])
    snaive = np.tile(tr[-SEASON:], HOLD // SEASON + 1)[:HOLD]
    rec = dict(leg="P15-planted", verdict=r["verdict"],
               equation=r.get("equation", "")[:80])
    if pred is not None:
        rec.update(mase_isaac=round(mase(pred, te, tr), 3))
    rec.update(mase_naive=round(mase(naive, te, tr), 3),
               mase_snaive=round(mase(snaive, te, tr), 3))
    recs.append(rec)
    print(f"[P15] {rec}")


def leg_p16(mod, recs):
    recs.append(dict(
        leg="P16-m4", status="BLOCKED",
        why="M4 monthly train file is a ~300 MB download — impractical "
            "for the bench v1; recorded, never swapped for synthetic"))
    try:
        raw = urllib.request.urlopen(
            "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv",
            timeout=60).read().decode()
        rows = [r.split(";") for r in raw.strip().splitlines()]
        s = np.array([float(r[3]) for r in rows])
        s = s[s >= 0][-360:]                       # last 30 years
    except Exception as e:
        recs.append(dict(leg="P16-sunspots", status="BLOCKED",
                         why=f"download failed: {type(e).__name__}"))
        print("[P16] sunspots BLOCKED")
        return
    tr, te = s[:-HOLD], s[-HOLD:]
    pred, r = forecast_series(mod, tr, "b4:sunspots")
    naive = np.full(HOLD, tr[-1])
    rec = dict(leg="P16-sunspots", verdict=r["verdict"])
    if pred is not None:
        rec.update(mase_isaac=round(mase(pred, te, tr), 3))
    else:
        rec.update(note="refusal — quasi-periodic; an honest card, "
                        "same standing as a forecast")
    rec.update(mase_naive=round(mase(naive, te, tr), 3))
    recs.append(rec)
    print(f"[P16] {rec}")


def fetch_returns():
    """Three roads to REAL daily equity returns; whichever answers
    first. Roads 1-2 (stooq closes, FRED SP500 closes) both refused
    this machine on 2026-07-17 (robot wall / read timeout — recorded);
    road 3 is the Ken French data library's daily factors: the
    market's excess return + RF, in percent, since 1926 — returns
    directly, no differencing."""
    errs = []
    for url in ("https://stooq.com/q/d/l/?s=%5Espx&i=d",
                "https://fred.stlouisfed.org/graph/fredgraph.csv"
                "?id=SP500"):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=45).read().decode()
            rows = raw.strip().splitlines()[1:]
            closes = []
            for r in rows:
                parts = r.split(",")
                cell = parts[4] if len(parts) >= 5 else \
                    (parts[1] if len(parts) == 2 else "")
                try:
                    v = float(cell)
                    if v > 0:
                        closes.append(v)
                except ValueError:
                    continue
            if len(closes) >= 1500:
                return np.diff(np.log(np.array(closes[-4200:]))), \
                    url.split("/")[2]
            errs.append(f"{url.split('/')[2]}: {len(closes)} rows")
        except Exception as e:
            errs.append(f"{url.split('/')[2]}: {type(e).__name__}")
    try:
        import io
        import zipfile
        url = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
               "ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip")
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=90).read()
        zf = zipfile.ZipFile(io.BytesIO(raw))
        lines = zf.read(zf.namelist()[0]).decode("latin1") \
            .strip().splitlines()
        rets = []
        for ln in lines:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) >= 5 and len(parts[0]) == 8 \
                    and parts[0].isdigit():
                try:
                    rets.append((float(parts[1]) + float(parts[4]))
                                / 100.0)
                except ValueError:
                    continue
        if len(rets) >= 1500:
            return np.array(rets[-4200:]), "ken.french (daily Mkt)"
        errs.append(f"french: {len(rets)} rows")
    except Exception as e:
        errs.append(f"french: {type(e).__name__}")
    raise RuntimeError("; ".join(errs))


def leg_p17_p18(mod, recs):
    try:
        r_all, source = fetch_returns()
    except Exception as e:
        recs.append(dict(leg="P17-returns", status="BLOCKED",
                         why=f"all download roads failed "
                             f"({type(e).__name__}: {e}) — the "
                             f"headline needs REAL returns; "
                             f"synthetic would prove nothing"))
        print("[P17/P18] BLOCKED")
        return
    recs.append(dict(leg="P17-source", source=source,
                     days=int(len(r_all))))
    K, W = 8, 500
    refuse = mint = 0
    vol_found = 0
    for k in range(K):
        w = r_all[k * (len(r_all) - W) // max(K - 1, 1):][:W]
        X, y = w[:-1].reshape(-1, 1), w[1:]
        m = mod.Isaac()
        res = m.discover(X, y, name=f"b4:ret{k}")
        if res["verdict"] in ("UNDERSTOOD", "PARTIAL"):
            mint += 1
            recs.append(dict(leg="P17-returns", window=k,
                             verdict=res["verdict"],
                             equation=res.get("equation", "")[:60],
                             note="MINTED ON RETURNS — inspect: "
                                  "wrong-confident unless it holds "
                                  "on fresh rows"))
        else:
            refuse += 1
        # REALIZED VOLATILITY as P18 names it: weekly RV =
        # sqrt(sum of squared daily returns over 5 days) — the
        # literature's measure (AR autocorr ~0.6). The first run
        # used one-step |r| (autocorr ~0.15) and found nothing in
        # 8/8 windows — that result stands on the record in
        # RESULTS.md (this file's jsonl holds the latest run only);
        # this is the named shape. Observed 2026-07-17: 1/8 — P18
        # substantially falsified at BOTH shapes, kept.
        rv_w = np.sqrt(np.add.reduceat(w ** 2,
                                       np.arange(0, len(w), 5)))
        Xv, yv = rv_w[:-1].reshape(-1, 1), rv_w[1:]
        mv = mod.Isaac()
        rv = mv.discover(Xv, yv, name=f"b4:vol{k}")
        if rv["verdict"] in ("UNDERSTOOD", "PARTIAL", "CAPTURED"):
            vol_found += 1
    recs.append(dict(leg="P17-returns", windows=K, refused=refuse,
                     minted=mint,
                     headline=f"refusal rate {refuse}/{K} on real "
                              f"daily index returns"))
    recs.append(dict(leg="P18-volatility", windows=K,
                     structure_found=vol_found,
                     note="the pair: structure where returns refused "
                          "= discrimination, not blindness"))
    print(f"[P17] refusal {refuse}/{K}  minted {mint}")
    print(f"[P18] volatility structure {vol_found}/{K}")


def main():
    mod = bridge.load_isaac_module()
    rng = np.random.default_rng(11)
    recs = []
    leg_p15(mod, recs, rng)
    leg_p16(mod, recs)
    leg_p17_p18(mod, recs)
    with open(os.path.join(HERE, "results_b4.jsonl"), "w",
              encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print("\nB4 done — results_b4.jsonl written; every blocked leg "
          "recorded with its why.")


if __name__ == "__main__":
    main()

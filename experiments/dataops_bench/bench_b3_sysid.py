"""B3 — SYSTEM IDENTIFICATION (segment plan §3, SINDy-lane):
planted difference/ODE systems within the atom library. Metric:
SYMBOLIC SOLUTION RATE — recovered the governing form (verdict +
functional agreement on fresh rows) or refused honestly; fit numbers
secondary. A verdict of UNDERSTOOD whose law then misses fresh rows
is WRONG-CONFIDENT and counted against — the duel's own definition.

Predictions filed FIRST in PREDICTIONS.md (P10-P14, 2026-07-17).
Worlds: logistic map (exact road's home game), damped oscillator via
lag atoms, log-linearized predator-prey (discretization error is the
trap to watch), a delay system OUTSIDE the library (honest refusal
scored correct), and the scrambled-series control (shuffling the
series breaks the lag structure; any minted law is a false
discovery).

Run: PYTHONHASHSEED=0 python bench_b3_sysid.py [trials]
Writes results_b3.jsonl next to this file.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DC = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, DC)
sys.path.insert(0, HERE)

from isaac_store import bridge                       # noqa: E402


def series_pairs(s):
    return s[:-1].reshape(-1, 1), s[1:]


def w_logistic(rng, n=400, r=3.7):
    s = np.empty(n + 1)
    s[0] = rng.uniform(0.2, 0.8)
    for t in range(n):
        s[t + 1] = r * s[t] * (1 - s[t])
    X, y = series_pairs(s)
    return dict(name="logistic-map", X=X, y=y,
                truth=lambda x: r * x * (1 - x),
                tol=1e-6, expect="RECOVERED")


def w_oscillator(rng, n=500, rho=0.98, w=0.45, noise=0.01):
    a, b = 2 * rho * np.cos(w), -rho ** 2
    s = np.empty(n)
    s[0], s[1] = 1.0, 0.9
    for t in range(2, n):
        s[t] = a * s[t - 1] + b * s[t - 2] + rng.normal(0, noise)
    X, y = series_pairs(s)
    # tol = 3*noise: the true AR(2)'s one-step MAE is ~0.8*noise; the
    # AR(1) surrogate's is ~4*noise (measured 0.042 on run 1) — 5x
    # noise let the surrogate under the bar, 3x separates them
    return dict(name="damped-oscillator", X=X, y=y, truth=None,
                coefs=dict(a=a, b=b), tol=3 * noise,
                expect="RECOVERED")


def w_predprey(rng, n=600, dt=0.02, a=1.1, b=0.4, c=0.4, d=0.1):
    H, L = np.empty(n), np.empty(n)
    H[0], L[0] = 10.0, 5.0
    for t in range(n - 1):                       # RK4 on Lotka-Volterra
        def f(state):
            h, ll = state
            return np.array([a * h - b * h * ll, d * h * ll - c * ll])
        st = np.array([H[t], L[t]])
        k1 = f(st); k2 = f(st + dt / 2 * k1)
        k3 = f(st + dt / 2 * k2); k4 = f(st + dt * k3)
        H[t + 1], L[t + 1] = st + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    dlnH = (np.log(H[2:]) - np.log(H[:-2])) / (2 * dt)   # central diff
    X = L[1:-1].reshape(-1, 1)
    return dict(name="predator-prey", X=X, y=dlnH, truth=None,
                coefs=dict(intercept=a, slope=-b), tol=0.05,
                expect="RECOVERED")


def w_delay(rng, n=600):
    s = np.empty(n)
    s[:18] = 1.2 + 0.1 * rng.standard_normal(18)
    for t in range(17, n - 1):
        s[t + 1] = s[t] + 0.2 * s[t - 17] / (1 + s[t - 17] ** 10) \
            - 0.1 * s[t]
    X, y = series_pairs(s)
    return dict(name="delay-outside-library", X=X, y=y, truth=None,
                tol=1e-6, expect="REFUSED")


def w_scrambled(rng, n=500):
    w = w_oscillator(rng, n)
    s = np.concatenate([w["X"][:, 0], w["y"][-1:]])
    rng.shuffle(s)                       # the series itself shuffled
    X, y = series_pairs(s)
    return dict(name="scrambled-control", X=X, y=y, truth=None,
                tol=None, expect="REFUSED")


def fresh_match(m, name, w, rng):
    """Functional agreement on FRESH rows of the same system."""
    if w["name"] == "logistic-map":
        xg = rng.uniform(0.05, 0.95, 200).reshape(-1, 1)
        pred = m.apply_law(name, xg)
        if pred is None:
            return None
        return float(np.nanmax(np.abs(pred - w["truth"](xg[:, 0]))))
    if w["name"] == "damped-oscillator":
        w2 = w_oscillator(rng)
        pred = m.apply_law(name, w2["X"])
        if pred is None:
            return None
        ok = np.isfinite(pred)
        return float(np.mean(np.abs(pred[ok] - w2["y"][ok])))
    if w["name"] == "predator-prey":
        w2 = w_predprey(rng)
        pred = m.apply_law(name, w2["X"])
        if pred is None:
            return None
        ok = np.isfinite(pred)
        return float(np.mean(np.abs(pred[ok] - w2["y"][ok])))
    if w["name"] == "delay-outside-library":
        w2 = w_delay(rng)
        pred = m.apply_law(name, w2["X"])
        if pred is None:
            return None
        ok = np.isfinite(pred)
        return float(np.mean(np.abs(pred[ok] - w2["y"][ok])))
    return None


def run_world(w, mod, trial, rng):
    m = mod.Isaac()
    name = f"b3:{w['name']}:{trial}"
    r = m.discover(w["X"], w["y"], name=name)
    rec = dict(world=w["name"], trial=trial, expect=w["expect"],
               verdict=r["verdict"], equation=r.get("equation", ""),
               r2=round(float(r.get("r2") or 0.0), 5))
    understood = r["verdict"] == "UNDERSTOOD"
    minted = r["verdict"] in ("UNDERSTOOD", "PARTIAL")
    err = fresh_match(m, name, w, rng) if minted else None
    rec["fresh_err"] = err
    if w["expect"] == "RECOVERED":
        # per filed P12: PARTIAL with the right form (fresh rows
        # agree within tol) IS recovery — the remainder honestly
        # named does not un-recover the law; UNDERSTOOD that misses
        # fresh rows is wrong-confident, the duel's own definition
        rec["outcome"] = (
            "RECOVERED" if minted and err is not None
            and err <= w["tol"] else
            "WRONG-CONFIDENT" if understood else
            "MISSED")
    else:                                # expect REFUSED
        if understood:
            rec["outcome"] = ("WRONG-CONFIDENT"
                              if err is None or err > w["tol"]
                              else "RECOVERED-UNEXPECTED")
        else:
            rec["outcome"] = "REFUSED-CORRECT"
    return rec


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    mod = bridge.load_isaac_module()
    out_path = os.path.join(HERE, "results_b3.jsonl")
    recs = []
    for trial in range(trials):
        rng = np.random.default_rng(100 + trial)
        for build in (w_logistic, w_oscillator, w_predprey, w_delay,
                      w_scrambled):
            w = build(rng)
            rec = run_world(w, mod, trial, rng)
            recs.append(rec)
            print(f"[{w['name']:22s} t{trial}] {rec['verdict']:11s} "
                  f"-> {rec['outcome']:16s} "
                  f"fresh_err={rec['fresh_err']} "
                  f"{rec['equation'][:48]}")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    n_rec = sum(r["outcome"] == "RECOVERED" for r in recs)
    n_ref = sum(r["outcome"] == "REFUSED-CORRECT" for r in recs)
    n_wc = sum(r["outcome"] == "WRONG-CONFIDENT" for r in recs)
    n_miss = sum(r["outcome"] == "MISSED" for r in recs)
    n_exp_rec = sum(r["expect"] == "RECOVERED" for r in recs)
    n_exp_ref = sum(r["expect"] == "REFUSED" for r in recs)
    print("\nB3 SYMBOLIC SOLUTION RATE (complete numbers):")
    print(f"  recovered {n_rec}/{n_exp_rec} recoverable; "
          f"refused-correct {n_ref}/{n_exp_ref} refusable; "
          f"missed {n_miss}; WRONG-CONFIDENT {n_wc} (must be 0)")
    sys.exit(0 if n_wc == 0 else 1)


if __name__ == "__main__":
    main()

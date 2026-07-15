"""PROBE — floor contamination / self-hiding corruption (found by
B1's W3 numbers, 2026-07-15: ±15 corruptions inflated the measured
repeats floor to ~2.05 vs the true 0.05 — the errors being hunted fed
the ruler that judges them).

The algebra, filed as a PREDICTION before this probe ran: pooled
within-group spread gains ~ sqrt(rate)·magnitude, and the flag line is
4× the floor, so once corruption exceeds ~1/16 ≈ 6.25% of rows the
line grows past the corruption itself and recall collapses REGARDLESS
of magnitude. Below that rate the audit should stay near-perfect.

This probe measures recall + measured floor vs corruption rate at a
fixed magnitude (±3.0, true noise 0.05). A confirmed collapse is a
named vulnerability with a named candidate cure — a robust
(median-of-group-spreads) pooled read in the floor organ, which is
ENGINE work: research-first in paperclip, then promote. NOT built
here; this probe only establishes the receipt.

Run: PYTHONHASHSEED=0 python probe_floor_contamination.py
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DC = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, DC)

from isaac_store import bridge                      # noqa: E402
from isaac_store.audit import audit_source          # noqa: E402


def csv_text(cols, rows):
    return "\n".join([",".join(cols)] +
                     [",".join(f"{v:.10g}" for v in r) for r in rows])


def world(seed, rate, mag=3.0):
    rng = np.random.default_rng(1000 + seed)
    x = np.repeat(np.linspace(0.5, 6.0, 40), 8)
    rng.shuffle(x)
    y = 2.0 * x + rng.normal(0, 0.05, len(x))
    k = int(round(rate * len(x)))
    bad = sorted(rng.choice(len(x), k, replace=False).tolist())
    for i in bad:
        y[i] += float(rng.choice([-1.0, 1.0])) * mag
    return np.column_stack([x, y]), set(bad)


def main():
    mod = bridge.load_isaac_module()
    print("floor contamination sweep — magnitude ±3.0, true noise 0.05,"
          " 320 rows, 2 trials/rate")
    print(f"{'rate':>6} {'planted':>8} {'recall':>7} {'fp':>4} "
          f"{'measured floor':>15} {'flag line (4x)':>15}  status")
    for rate in (0.025, 0.05, 0.075, 0.10, 0.15):
        recs = []
        for t in range(2):
            rows, bad = world(t, rate)
            with tempfile.TemporaryDirectory() as td:
                p = os.path.join(td, "t.csv")
                with open(p, "w", encoding="utf-8") as f:
                    f.write(csv_text(["x", "y"], rows))
                rep = audit_source(p, mod=mod)
            tab = rep["tables"][0]
            flagged = {f["row"] for f in tab["flags"]
                       if f["kind"].endswith("CANDIDATE")}
            fl = tab["floors"][tab["target"]]["floor"]
            recs.append((len(flagged & bad) / max(len(bad), 1),
                         len(flagged - bad), fl,
                         tab["flags_status"]))
        rec = np.mean([r[0] for r in recs])
        fp = np.mean([r[1] for r in recs])
        fl = np.mean([r[2] for r in recs if r[2] is not None]) \
            if any(r[2] is not None for r in recs) else float("nan")
        status = recs[0][3][:40]
        print(f"{rate:>6.1%} {int(round(rate * 320)):>8} {rec:>7.2f} "
              f"{fp:>4.1f} {fl:>15.4f} {4 * fl:>15.4f}  {status}")


if __name__ == "__main__":
    main()

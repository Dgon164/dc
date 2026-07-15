"""B1 — ERROR DETECTION (segment plan §3): the audit face's public
scoreboard. Planted-corruption worlds first (public labeled-error sets
are the named follow-up, NOT run here — no silent scope).

Metric: cell-level precision/recall/F1 per world + false-flag rate on
clean tables. A REFUSED flag pass scores recall 0 and is printed as a
refusal — the honest miss stays visible either way.

Isaac = audit_source (engine + floors + consensus recalibration).
BASELINE = z-score / IQR on the target column, IsolationForest on the
full rows (sklearn, labeled, never the cognition).

Run: PYTHONHASHSEED=0 python bench_b1_errors.py [trials]
Writes results_b1.jsonl next to this file.
"""
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DC = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, DC)
sys.path.insert(0, HERE)

from isaac_store import bridge                      # noqa: E402
from isaac_store.audit import audit_source          # noqa: E402
import baselines                                    # noqa: E402
from worlds import B1_WORLDS                        # noqa: E402


def csv_text(cols, rows):
    return "\n".join([",".join(cols)] +
                     [",".join(f"{v:.10g}" for v in r) for r in rows])


def prf(flagged, bad, n):
    tp = len(flagged & bad)
    fp = len(flagged - bad)
    fn = len(bad - flagged)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    return dict(tp=tp, fp=fp, fn=fn, precision=round(p, 3),
                recall=round(r, 3), f1=round(f1, 3),
                false_flag_rate=round(fp / n, 4))


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    mod = bridge.load_isaac_module()
    out = []
    print("=" * 72)
    print(f"B1 ERROR DETECTION — planted worlds, {trials} trials each")
    print("=" * 72)
    for build in B1_WORLDS:
        head = None
        for t in range(trials):
            w = build(t)
            head = head or w["name"]
            if t == 0:
                print(f"\n{w['name']}  ({w['note']})")
            rows, bad, n = np.asarray(w["rows"]), w["bad"], len(w["rows"])
            with tempfile.TemporaryDirectory() as td:
                p = os.path.join(td, "t.csv")
                with open(p, "w", encoding="utf-8") as f:
                    f.write(csv_text(w["columns"], rows))
                rep = audit_source(p, resolutions=w["resolutions"],
                                   mod=mod)
            tab = rep["tables"][0]
            flagged = {f["row"] for f in tab["flags"]
                       if f["kind"].endswith("CANDIDATE")}
            refused = tab["flags_status"].startswith("refused")
            fr = tab["floors"][tab["target"]]
            rec = dict(world=w["name"], trial=t, rows=n,
                       planted=sorted(bad),
                       floor=fr["floor"], floor_road=fr["road"],
                       flags_status=tab["flags_status"],
                       isaac=prf(flagged, bad, n))
            y = rows[:, -1]
            rec["baseline_zscore"] = prf(baselines.zscore_flags(y),
                                         bad, n)
            rec["baseline_iqr"] = prf(baselines.iqr_flags(y), bad, n)
            rec["baseline_isoforest"] = prf(
                baselines.isoforest_flags(rows, seed=t), bad, n)
            out.append(rec)
            i = rec["isaac"]
            line = (f"  trial {t}: ISAAC P {i['precision']:.2f} "
                    f"R {i['recall']:.2f} F1 {i['f1']:.2f} "
                    f"(tp {i['tp']} fp {i['fp']} fn {i['fn']}; "
                    f"floor {fr['floor'] if fr['floor'] is None else round(fr['floor'], 4)}"
                    f" via {fr['road']})")
            if refused:
                line += f"  [flags REFUSED: {tab['flags_status'][:58]}]"
            print(line)
            for nm, key in (("z-score", "baseline_zscore"),
                            ("IQR", "baseline_iqr"),
                            ("IsoForest", "baseline_isoforest")):
                b = rec[key]
                print(f"           BASELINE {nm:<9} P {b['precision']:.2f} "
                      f"R {b['recall']:.2f} F1 {b['f1']:.2f} "
                      f"(tp {b['tp']} fp {b['fp']} fn {b['fn']})")

    print("\n" + "=" * 72)
    print("MEANS ACROSS TRIALS (complete numbers; refused trials "
          "included at recall 0)")
    print("=" * 72)
    for build in B1_WORLDS:
        name = build(0)["name"]
        rs = [r for r in out if r["world"] == name]
        planted = any(r["planted"] for r in rs)
        print(f"\n{name}")
        for label, key in (("ISAAC", "isaac"),
                           ("BASELINE z-score", "baseline_zscore"),
                           ("BASELINE IQR", "baseline_iqr"),
                           ("BASELINE IsoForest", "baseline_isoforest")):
            ms = [r[key] for r in rs]
            if planted:
                print(f"  {label:<20} P {np.mean([m['precision'] for m in ms]):.2f}"
                      f"  R {np.mean([m['recall'] for m in ms]):.2f}"
                      f"  F1 {np.mean([m['f1'] for m in ms]):.2f}"
                      f"  false-flags/table "
                      f"{np.mean([m['fp'] for m in ms]):.1f}")
            else:
                print(f"  {label:<20} false-flags/table "
                      f"{np.mean([m['fp'] for m in ms]):.1f} of "
                      f"{rs[0]['rows']} rows "
                      f"(rate {np.mean([m['false_flag_rate'] for m in ms]):.4f})")
        refusals = sum(1 for r in rs
                       if r["flags_status"].startswith("refused"))
        if refusals:
            print(f"  refusals: {refusals}/{len(rs)} trials — "
                  f"{rs[0]['flags_status'][:64]}")

    path = os.path.join(HERE, "results_b1.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(out)} records -> {path}")
    print("ARTIFACT DIFFERENCE (not a number): every ISAAC flag above "
          "carries law + residual + floor + row; refusals carry their "
          "why; baselines return bare indices.")


if __name__ == "__main__":
    main()

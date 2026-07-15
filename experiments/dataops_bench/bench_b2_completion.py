"""B2 — IMPUTATION / COMPLETION (segment plan §3): the triangulation
completion road on held-out masked values.

Metric: error in UNITS vs the known noise floor + REFUSAL HONESTY —
NO CLOSURE where identifiability fails is a correct answer and scored
as one; refusing a resolvable world (W9, the control) is a MISS.

Isaac's protocol per world: triangulate the OBSERVED rows of A against
B; if the completion road resolves, fit the joint law on observed rows
(engine discover) and predict the masked targets; on a refusal the
prediction falls back to the honest mean-model AND says so — the
refusal receipt travels with the numbers.

BASELINE (sklearn, labeled, never the cognition): mean / kNN /
IterativeImputer. They always answer; no warning is attached — the
bench prints that asymmetry.

Run: PYTHONHASHSEED=0 python bench_b2_completion.py [trials]
Writes results_b2.jsonl next to this file.
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
from isaac_store.triangulate import triangulate      # noqa: E402
import baselines                                     # noqa: E402
from worlds import B2_WORLDS                         # noqa: E402


def sub_table(tbl, keep_idx):
    rows = np.asarray(tbl["rows"])[sorted(keep_idx)]
    return dict(name=tbl["name"], columns=tbl["columns"], rows=rows)


def mae(pred, truth):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(truth))))


def run_world(w, mod, trial):
    A, B = w["A"], w["B"]
    rec = dict(world=w["name"], trial=trial, expect=w["expect"])
    if w["masked"] is None:
        # verdict-only world (refusal honesty; nothing to impute)
        tri = triangulate(A, B, mod=mod)
        rec["verdict"] = tri["verdict"]
        rec["why"] = tri.get("why", "")
        resolved = tri["verdict"].startswith("RESOLVED")
        rec["honest"] = (resolved == (w["expect"] == "RESOLVED"))
        return rec

    masked = sorted(w["masked"])
    n = len(np.asarray(A["rows"]))
    obs = [i for i in range(n) if i not in set(masked)]
    A_obs = sub_table(A, obs)
    truth = np.asarray(w["truth"])[masked]

    tri = triangulate(A_obs, B, mod=mod)
    rec["verdict"] = tri["verdict"]
    rec["why"] = tri.get("why", "")
    resolved = tri["verdict"].startswith("RESOLVED")
    rec["honest"] = (resolved == (w["expect"] == "RESOLVED"))

    Arows = np.asarray(A["rows"], float)
    Brows = np.asarray(B["rows"], float)
    kmap = {round(float(k), 6): i for i, k in enumerate(Brows[:, 0])}
    brow = np.array([kmap[round(float(k), 6)] for k in Arows[:, 0]])
    feats = np.column_stack([Arows[:, 1:-1], Brows[brow][:, 1:-1]]) \
        if Arows.shape[1] > 2 else Brows[brow][:, 1:-1]
    y_all = Arows[:, -1]
    y_obs = y_all[obs]

    if "COMPLETION" in tri["verdict"]:
        mind = mod.Isaac()
        mind.discover(feats[obs], y_obs, name="b2_joint")
        pred = mind.apply_law("b2_joint", feats[masked])
        if pred is None or not np.all(np.isfinite(pred)):
            pred = baselines.mean_impute(y_obs, len(masked))
            rec["isaac_mode"] = ("law resolved but refit unusable — "
                                 "fell back to mean, said so")
        else:
            rec["isaac_mode"] = "joint law (completion road)"
    else:
        pred = baselines.mean_impute(y_obs, len(masked))
        rec["isaac_mode"] = (f"refused ({tri['verdict']}) — honest "
                             f"mean-model, receipt attached")
    rec["isaac_mae"] = round(mae(pred, truth), 4)
    rec["floor"] = w["floor"]

    M = np.column_stack([feats, y_all.copy()])
    M[masked, -1] = np.nan
    tc = M.shape[1] - 1
    rec["baseline_mean_mae"] = round(
        mae(baselines.mean_impute(y_obs, len(masked)), truth), 4)
    rec["baseline_knn_mae"] = round(
        mae(baselines.knn_impute(M.copy(), tc, masked), truth), 4)
    rec["baseline_iterative_mae"] = round(
        mae(baselines.iterative_impute(M.copy(), tc, masked), truth), 4)
    return rec


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    mod = bridge.load_isaac_module()
    out = []
    print("=" * 72)
    print(f"B2 IMPUTATION / COMPLETION — planted worlds, {trials} "
          f"trials each")
    print("=" * 72)
    for build in B2_WORLDS:
        for t in range(trials):
            w = build(t)
            if t == 0:
                print(f"\n{w['name']}  ({w['note']})")
            rec = run_world(w, mod, t)
            out.append(rec)
            line = (f"  trial {t}: verdict {rec['verdict']}"
                    f"  [{'HONEST' if rec['honest'] else 'WRONG CALL'}"
                    f" — expected {rec['expect']}]")
            print(line)
            if "isaac_mae" in rec:
                print(f"           ISAAC MAE {rec['isaac_mae']:.4f} in "
                      f"units (floor {rec['floor']:g}; mode: "
                      f"{rec['isaac_mode']})")
                print(f"           BASELINE mean {rec['baseline_mean_mae']:.4f}"
                      f" | kNN {rec['baseline_knn_mae']:.4f}"
                      f" | iterative {rec['baseline_iterative_mae']:.4f}"
                      f"  (always answer, no warning)")

    print("\n" + "=" * 72)
    print("MEANS ACROSS TRIALS")
    print("=" * 72)
    for build in B2_WORLDS:
        name = build(0)["name"]
        rs = [r for r in out if r["world"] == name]
        honest = sum(1 for r in rs if r["honest"])
        print(f"\n{name}")
        print(f"  verdict honesty: {honest}/{len(rs)} trials called "
              f"as the world truly is")
        if "isaac_mae" in rs[0]:
            print(f"  ISAAC MAE {np.mean([r['isaac_mae'] for r in rs]):.4f}"
                  f"  (floor {rs[0]['floor']:g})   BASELINE mean "
                  f"{np.mean([r['baseline_mean_mae'] for r in rs]):.4f}"
                  f" | kNN {np.mean([r['baseline_knn_mae'] for r in rs]):.4f}"
                  f" | iterative "
                  f"{np.mean([r['baseline_iterative_mae'] for r in rs]):.4f}")

    path = os.path.join(HERE, "results_b2.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(out)} records -> {path}")
    print("ARTIFACT DIFFERENCE (not a number): ISAAC's refusals arrive "
          "AS verdicts with whys and nulls; the baselines imputed the "
          "unimputable with the same confidence as the imputable.")


if __name__ == "__main__":
    main()

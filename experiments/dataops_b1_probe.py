"""
DATAOPS B1 ERROR-DETECTION PROBE -- planted-world edition (segment plan
s3 B1; public labeled-error sets deferred, planted worlds are the truth
anchor).

THE CLAIM UNDER TEST (the plan's own prediction, filed in
docs/DATA_OPS_SEGMENT_PLAN.md s3 BEFORE any benchmark ran):
    "B1 competitive (the lane's native game)"
Operationalized here: at every corruption magnitude, the audit face's
cell-level F1 >= the best BASELINE F1, and its false-flag rate on the
CLEAN control <= every baseline's.

Worlds: tables under a known law y = 2x + N(0, FLOOR), FLOOR = 0.05.
N_BAD = 16 of 320 y-cells corrupted by +/- k*FLOOR, k in {2, 4, 10}.
Two floor-evidence designs:
    R (repeats road): 40 unique x values, 8 repeats each -- the floor
      is MEASURED from duplicate-input spread, the audit's native road;
    T (told road): continuous unique x, the true resolution TOLD
      (resolutions={'y': 0.05}) -- the floor is exact by construction.
Controls:
    CLEAN table per design (false-flag rate; the A2 control);
    SCRAMBLED null (told design, y permuted against x, no corruption):
      no law exists, so the audit must REFUSE flags -- an invented flag
      here is a false positive against a world with no errors at all;
    RESCUE condition (repeats design, 10x, true floor ALSO told): does
      telling the truth restore the cut when the measured road is
      contaminated?
ORACLE reference (labeled reference, NOT a competitor): the true law +
true floor at the audit's own 4*floor cut -- the detectability ceiling.

BASELINES (sklearn permitted here ONLY, labeled, never feeding Isaac):
    z-score |z|>3 per column; IQR 1.5-fence per column; IsolationForest
    (auto contamination, and a generous variant TOLD the true 5% rate).
    Baselines emit bare scores; the audit's flags carry law + residual
    + floor + row receipts -- reported below as a stated qualitative
    difference.

SUB-PREDICTIONS filed before the grid runs (mine, not the plan's):
    P2: at 2x FLOOR the ORACLE itself is near-blind (corruption is 2
        sigma under 1-sigma noise; the audit's cut sits at 4x) -- every
        method's recall ~0; a miss there is the world's fault.
    P3: the repeats road's pooled sigma is NOT robust, so the planted
        corruption itself inflates the measured floor and the 4x cut
        rises with it -- recall at R/10x lands well under T/10x.
        DISCLOSURE: one API-shape check run (a single R/10x call)
        preceded this filing and showed floor 0.1187 vs true 0.05 with
        7/16 flagged; P3 is therefore informed for R/10x and a genuine
        prediction only for the rest of the grid (R/4x, RESCUE, and the
        told-vs-repeats gap).
    P4: RESCUE fails -- read_floor answers with the LARGEST evidenced
        road, so the inflated repeats sigma beats the told truth and
        the told record is marked CONTRADICTED.
Found wrongs are REPORTED, never patched away.

Run: cd /home/user/dc/experiments && PYTHONHASHSEED=0 \
     OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 dataops_b1_probe.py
"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from isaac_store import bridge                          # noqa: E402
from isaac_store.audit import audit_source              # noqa: E402

MOD = bridge.load_isaac_module()

N = 320
FLOOR = 0.05
N_BAD = 16
MAGS = (2, 4, 10)
SEEDS = (101, 102, 103, 104, 105)
AUDIT_CUT = 4.0          # the audit's own cut, in floors (audit.py)


# ---------------------------------------------------------------- worlds
def make_world(design, seed):
    rng = np.random.default_rng(seed)
    if design == "R":                     # repeats road: 40 x-values x 8
        x = np.repeat(np.linspace(0.5, 6, 40), 8)
        rng.shuffle(x)
    else:                                 # told road: continuous unique x
        x = rng.uniform(0.5, 6, N)
    y = 2.0 * x + rng.normal(0, FLOOR, N)
    return x, y, rng


def corrupt(y, mag, rng):
    bad = rng.choice(len(y), N_BAD, replace=False)
    y2 = y.copy()
    for j, i in enumerate(bad):
        y2[i] += (mag * FLOOR) * (1 if j % 2 == 0 else -1)
    return y2, set(int(i) for i in bad)


def csv_text(cols, rows):
    return "\n".join([",".join(cols)] +
                     [",".join(f"{v:.10g}" for v in r) for r in rows])


# ------------------------------------------------------------- the audit
def run_audit(x, y, resolutions=None):
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "world.csv")
        with open(p, "w", encoding="utf-8") as f:
            f.write(csv_text(["x", "y"], np.column_stack([x, y])))
        rep = audit_source(p, resolutions=resolutions, mod=MOD)
    t = rep["tables"][0]
    cand = {f["row"] for f in t["flags"] if f["kind"].endswith("CANDIDATE")}
    kinds = dict(error=t["counts"]["error"], news=t["counts"]["news"],
                 duplicate=t["counts"]["duplicate"])
    fr = t["floors"][t["target"]]
    return dict(flags=cand, kinds=kinds, status=t["flags_status"],
                floor=fr["floor"], road=fr["road"], roads=fr["roads"],
                verdict=t["law"]["verdict"],
                news_factors=t["news_factors"],
                receipt=(t["flags"][0]["receipt"] if t["flags"] else None))


# ---------------------------------------------- BASELINES (labeled; the
# cognition never sees them) -- z / IQR flag CELLS in every column;
# IsolationForest flags ROWS, mapped (generously) to the y-cell, since
# y is the only corruptible column in these worlds.
def baseline_z(x, y, cut=3.0):
    out = set()
    for col, name in ((x, "x"), (y, "y")):
        s = np.std(col)
        if s > 0:
            z = np.abs((col - np.mean(col)) / s)
            out |= {(int(i), name) for i in np.where(z > cut)[0]}
    return out


def baseline_iqr(x, y, k=1.5):
    out = set()
    for col, name in ((x, "x"), (y, "y")):
        q1, q3 = np.quantile(col, [0.25, 0.75])
        w = k * (q3 - q1)
        hit = (col < q1 - w) | (col > q3 + w)
        out |= {(int(i), name) for i in np.where(hit)[0]}
    return out


def baseline_iforest(x, y, contamination):
    from sklearn.ensemble import IsolationForest    # BASELINE ONLY
    clf = IsolationForest(random_state=0, contamination=contamination)
    lab = clf.fit_predict(np.column_stack([x, y]))
    return {(int(i), "y") for i in np.where(lab == -1)[0]}


def oracle(x, y):
    """Reference, not a competitor: true law + true floor at the
    audit's own 4*floor cut = the detectability ceiling."""
    r = y - 2.0 * x
    r = r - np.median(r)
    return {(int(i), "y") for i in np.where(np.abs(r) >
                                            AUDIT_CUT * FLOOR)[0]}


METHODS = ["isaac-audit", "BASELINE z>3", "BASELINE IQR1.5",
           "BASELINE IForest(auto)", "BASELINE IForest(5% told)",
           "ORACLE(ref)"]


def all_flags(x, y, audit_res):
    return {
        "isaac-audit": {(i, "y") for i in audit_res["flags"]},
        "BASELINE z>3": baseline_z(x, y),
        "BASELINE IQR1.5": baseline_iqr(x, y),
        "BASELINE IForest(auto)": baseline_iforest(x, y, "auto"),
        "BASELINE IForest(5% told)": baseline_iforest(x, y, N_BAD / N),
        "ORACLE(ref)": oracle(x, y),
    }


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# ------------------------------------------------------------------ run
def main():
    print("=" * 78)
    print("DATAOPS B1 ERROR DETECTION -- planted-corruption battery")
    print(f"law y=2x+N(0,{FLOOR}); {N_BAD}/{N} y-cells corrupted at "
          f"+/-{{2,4,10}}x floor; {len(SEEDS)} trials/condition")
    print("=" * 78)

    pooled = {}          # (design, mag, method) -> [tp, fp, fn]
    refusals = {}        # (design, mag) -> count of refused trials
    floors_seen = {}     # (design, mag) -> list of measured floors
    news_events = []
    receipt_example = None

    for design in ("R", "T"):
        for mag in MAGS:
            for s in SEEDS:
                x, y0, rng = make_world(design, s)
                y, bad = corrupt(y0, mag, rng)
                res = {"y": FLOOR} if design == "T" else None
                a = run_audit(x, y, resolutions=res)
                floors_seen.setdefault((design, mag), []).append(
                    a["floor"])
                if a["status"] != "active":
                    refusals[(design, mag)] = \
                        refusals.get((design, mag), 0) + 1
                if a["news_factors"]:
                    news_events.append((design, mag, s,
                                        a["news_factors"]))
                if receipt_example is None and a["receipt"] and \
                        design == "R" and mag == 10:
                    receipt_example = a["receipt"]
                bad_cells = {(i, "y") for i in bad}
                for m, fl in all_flags(x, y, a).items():
                    tp = len(fl & bad_cells)
                    fp = len(fl - bad_cells)
                    fn = len(bad_cells - fl)
                    agg = pooled.setdefault((design, mag, m), [0, 0, 0])
                    agg[0] += tp; agg[1] += fp; agg[2] += fn
            fl_ = floors_seen[(design, mag)]
            print(f"  [{design}/{mag}x] measured target floor: "
                  f"mean {np.mean(fl_):.4f} (true {FLOOR}); "
                  f"refused trials "
                  f"{refusals.get((design, mag), 0)}/{len(SEEDS)}")

    print("\nCELL-LEVEL SCORES (pooled over trials; corrupted cells "
          f"= {N_BAD * len(SEEDS)}/condition)")
    hdr = f"{'design/mag':<10} {'method':<26} {'TP':>4} {'FP':>4} " \
          f"{'FN':>4} {'prec':>6} {'rec':>6} {'F1':>6}"
    print(hdr); print("-" * len(hdr))
    for design in ("R", "T"):
        for mag in MAGS:
            for m in METHODS:
                tp, fp, fn = pooled[(design, mag, m)]
                p, r, f = prf(tp, fp, fn)
                print(f"{design + '/' + str(mag) + 'x':<10} {m:<26} "
                      f"{tp:>4} {fp:>4} {fn:>4} "
                      f"{p:>6.3f} {r:>6.3f} {f:>6.3f}")
        print("-" * len(hdr))

    # ---------------- CLEAN control: false-flag rate per method ------
    print("\nCLEAN CONTROL (no corruption; every flag is false)")
    scanned = {"isaac-audit": N, "BASELINE z>3": 2 * N,
               "BASELINE IQR1.5": 2 * N, "BASELINE IForest(auto)": N,
               "BASELINE IForest(5% told)": N, "ORACLE(ref)": N}
    clean_fp = {}
    for design in ("R", "T"):
        cnt = {m: 0 for m in METHODS}
        for s in SEEDS:
            x, y0, _ = make_world(design, s)
            res = {"y": FLOOR} if design == "T" else None
            a = run_audit(x, y0, resolutions=res)
            for m, fl in all_flags(x, y0, a).items():
                cnt[m] += len(fl)
        for m in METHODS:
            rate = cnt[m] / (scanned[m] * len(SEEDS))
            clean_fp[(design, m)] = (cnt[m], rate)
            print(f"  [{design}] {m:<26} false flags "
                  f"{cnt[m]:>4} / {scanned[m] * len(SEEDS)} cells "
                  f"scanned  (rate {rate:.4f})")

    # ---------------- SCRAMBLED null: no law, no errors --------------
    print("\nSCRAMBLED NULL (T design, y permuted against x, floor "
          "told; NO law, NO errors -- flags here are inventions)")
    null_refused, null_counts = 0, {m: 0 for m in METHODS}
    for s in SEEDS:
        x, y0, rng = make_world("T", s)
        yp = rng.permutation(y0)
        a = run_audit(x, yp, resolutions={"y": FLOOR})
        refused = a["status"] != "active"
        null_refused += refused
        for m, fl in all_flags(x, yp, a).items():
            null_counts[m] += len(fl)
        print(f"  trial {s}: verdict {a['verdict']}; flags_status: "
              f"{a['status'][:58]}")
    print(f"  audit refused {null_refused}/{len(SEEDS)} scrambled "
          f"trials; flags emitted per method over {len(SEEDS)} trials:")
    for m in METHODS:
        print(f"    {m:<26} {null_counts[m]}")

    # ---------------- RESCUE: repeats + truth told --------------------
    print("\nRESCUE CONDITION (R design, 10x, true floor 0.05 ALSO "
          "told -- does truth beat the contaminated measured road?)")
    resc = [0, 0, 0]
    resc_floors, contradicted = [], 0
    for s in SEEDS:
        x, y0, rng = make_world("R", s)
        y, bad = corrupt(y0, 10, rng)
        a = run_audit(x, y, resolutions={"y": FLOOR})
        resc_floors.append(a["floor"])
        contradicted += any(r.get("contradicted") for r in a["roads"]
                            if r["road"] == "told")
        bad_cells = {(i, "y") for i in bad}
        fl = {(i, "y") for i in a["flags"]}
        resc[0] += len(fl & bad_cells)
        resc[1] += len(fl - bad_cells)
        resc[2] += len(bad_cells - fl)
    p, r, f = prf(*resc)
    print(f"  floor used: mean {np.mean(resc_floors):.4f} "
          f"(told truth {FLOOR}); told-road CONTRADICTED in "
          f"{contradicted}/{len(SEEDS)} trials")
    print(f"  isaac-audit R/10x+told:  TP {resc[0]} FP {resc[1]} "
          f"FN {resc[2]}  prec {p:.3f} rec {r:.3f} F1 {f:.3f}")
    tp0, fp0, fn0 = pooled[("R", 10, "isaac-audit")]
    print(f"  (vs R/10x untold:        TP {tp0} FP {fp0} FN {fn0})")

    # ---------------- news/error kind accounting ----------------------
    print("\nFLAG-KIND ACCOUNTING (random cell placement -- the miner "
          "should name NOTHING; a named factor = misclassification)")
    if news_events:
        for d, mag, s, nf in news_events:
            print(f"  [{d}/{mag}x seed {s}] NEWS factors named: {nf}")
    else:
        print("  no NEWS factors named in any corrupted world "
              "(all flags ERROR-CANDIDATE) -- correct for random "
              "placement")

    # ---------------- the receipts artifact difference ---------------
    print("\nARTIFACT DIFFERENCE (stated qualitative row)")
    print(f"  isaac flag receipt (R/10x example): {receipt_example}")
    print("  baseline artifact for the same cell: a bare score "
          "(|z|, fence distance, or IForest anomaly label) -- no law, "
          "no floor, no predicted value, no row provenance")

    # ---------------- prediction check --------------------------------
    print("\nPREDICTION CHECK")
    plan_ok = True
    for design in ("R", "T"):
        for mag in MAGS:
            fi = prf(*pooled[(design, mag, "isaac-audit")])[2]
            fb = max(prf(*pooled[(design, mag, m)])[2]
                     for m in METHODS if m.startswith("BASELINE"))
            ok = fi >= fb
            plan_ok &= ok
            print(f"  [{design}/{mag}x] isaac F1 {fi:.3f} vs best "
                  f"BASELINE F1 {fb:.3f} -> "
                  f"{'holds' if ok else 'FAILS'}")
    for design in ("R", "T"):
        ok = all(clean_fp[(design, "isaac-audit")][1] <=
                 clean_fp[(design, m)][1]
                 for m in METHODS if m.startswith("BASELINE"))
        plan_ok &= ok
        print(f"  [{design}/clean] isaac false-flag rate "
              f"{clean_fp[(design, 'isaac-audit')][1]:.4f} <= every "
              f"baseline's -> {'holds' if ok else 'FAILS'}")
    print(f"  PLAN PREDICTION 'B1 competitive': "
          f"{'HELD' if plan_ok else 'FAILED'} on the operationalized "
          f"criteria above")
    oc2 = prf(*pooled[("T", 2, "ORACLE(ref)")])[1:3]
    print(f"  P2 (2x under the ceiling): oracle T/2x recall "
          f"{oc2[0]:.3f} -- corruption at 2 sigma is near-invisible "
          f"even to the true law at the audit's cut")
    r10 = prf(*pooled[("R", 10, "isaac-audit")])[1]
    t10 = prf(*pooled[("T", 10, "isaac-audit")])[1]
    print(f"  P3 (floor contamination): recall R/10x {r10:.3f} vs "
          f"T/10x {t10:.3f}; measured R/10x floor mean "
          f"{np.mean(floors_seen[('R', 10)]):.4f} vs true {FLOOR}")
    print(f"  P4 (rescue fails): floor with truth told mean "
          f"{np.mean(resc_floors):.4f}, told-road contradicted "
          f"{contradicted}/{len(SEEDS)} trials, recall {r:.3f}")


if __name__ == "__main__":
    main()

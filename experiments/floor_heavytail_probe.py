"""
FLOOR / HEAVY-TAIL / NON-AFFINE-DRAG PROBE -- attack probe, Isaac
testing segment (FRESH EYES item 4 + DATA_OPS_SEGMENT_PLAN s2).

THE CLAIM UNDER TEST: the floor roads (read_floor: repeats/told/grain,
isaac.py:2049-2192) and the audit's authoring-session thresholds
(4x-floor flag line, MAD<=3x-floor coverage gate, 3.5xMAD
recalibration trim, near-dup 3x-floor -- isaac_store/audit.py:151,
166, 174, 214) were battery-proven in GAUSSIAN worlds only
(floor_battery F1-F14, test_audit A1-A8). The attack: heavy-tailed
noise inflates the repeats floor (a floor estimated too high launders
wrong forms back in -- the plan's named r2-ghost, DATA_OPS s2:147,
s6:254), off-grid typos blind or capture the grain consensus, and a
planted NON-AFFINE drag defeats the affine-only consensus
recalibration (audit.py:137-156).

WORLDS
  W1 repeats road: 40 groups x 8 repeats, y = 2x + noise, core scale
     0.1. (a) Gaussian CONTROL (b) Student-t df=2 (c) contaminated
     Gaussian 95% sigma=0.1 / 5% sigma=10. Then END-TO-END audits of
     (a) and (c) with 6 planted +1.0 wrongs (10 core-sigma) and one
     isolated x-pair 0.6 apart (6 core-sigma -- genuinely distinct
     readings, NOT duplicates under the true instrument).
  W2 grain road: (a) float-continuous data, 10 seeds -- the F4
     regression, must refuse every time; (b) 0.01-quantized data,
     150 rows, +0.004 typos at k = 0,1,2,3,4,5,8 (8 ~ 5.3%) -- F14
     proved k=1 cannot blind the ruler; find how many can; (c) ONE
     typo at exactly HALF-grain offset (+0.005) -- a consensus-capture
     candidate: every on-grid gap is also a multiple of 0.005.
  W3 audit recalibration: y = 2x + N(0,0.05), told floor 0.05, 30% of
     rows dragged. CONTROLS: clean (0 flags expected) and AFFINE drag
     +0.6 (the recalibration's designed-for case, A3). ATTACK:
     quadratic drag +c*x^2 on the same 30%, c in {.005,.01,.02,.04,
     .08} -- does the affine-only trim mis-recalibrate silently, or
     disqualify itself honestly (coverage gate)?

PREDICTIONS (filed before the first run):
  P1 W1a Gaussian control: floor/core median within 15% of 1.0
     (dof=280, sd of sigma-hat ~ 4%).
  P2 W1b t(2): second moment is INFINITE; the pooled-RMS estimator
     (isaac.py:2117) is not robust -- median floor/core >= 2x with
     wild seed spread (max/min > 3). The honest robust core of
     0.1*t(2) is ~0.12 (quartile ratio 1.21), nowhere near what RMS
     reads.
  P3 W1c contaminated: floor -> sqrt(.95*.01+.05*100) ~ 2.24, i.e.
     ~22x core; end-to-end the 4x-floor flag line sits at ~90
     core-sigma so planted 10-sigma wrongs are laundered 0/6
     (CONTROL: 6/6), and the 6-sigma-apart pair is falsely flagged
     near-DUPLICATE (3x-floor ~ 6.7 >> 0.6; CONTROL: no dup flag).
     Also filed: even in the CONTROL the 6 planted wrongs themselves
     inflate the repeats floor (predicted ~0.17 vs core 0.1) -- the
     ruler is estimated FROM the data being audited.
  P4 W2a continuous: grain refuses 10/10 (F4 regression holds).
  P5 W2b typos: each typo poisons ~2 of ~142 gaps; consensus needs
     > 0.95 (isaac.py:2084), so grain holds k<=3, is BLINDED at k=4
     (2k/142 = 5.6%), and at 5.3% typo rate (k=8). Blinded grain +
     no other road -> floor REFUSED -> flags refused: the typos
     destroy the very ruler that would catch them (self-defeat loop).
  P6 W2c half-grain typo: candidate 0.005 reaches 100% gap consensus
     and BEATS the true 0.01 (which now has 2 bad gaps) -- step
     misread at 0.005, floor UNDERSTATED 2x. F14's 'one typo cannot
     blind the ruler' survives literally; one typo can quietly
     RESCALE it.
  P7 W3 affine control: recall >= 0.9, ~0 false flags (home game).
     Quadratic drag: the affine recalibration cannot remove
     curvature, AND the MAD coverage gate is median-based so 30%
     contamination barely moves it -- predicted to stay ACTIVE
     across the sweep (never self-disqualifies) while laundering
     the low-x dragged rows (roughly those with c*x^2 below ~4-8x
     floor). If instead the gate fires, that is the honest path and
     scores HONEST-REFUSAL, not BROKEN.

Controls per house rule: W1a Gaussian world; W2b k=0; W3 clean +
affine. Found wrongs are REPORTED, never patched away.

Run: cd /home/user/dc/experiments && PYTHONHASHSEED=0 \
     OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 \
     floor_heavytail_probe.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import isaac as mod                                    # noqa: E402
from isaac_store.audit import audit_source             # noqa: E402

CORE = 0.1
FINDINGS = []


def csv_text(cols, rows):
    return "\n".join([",".join(cols)] +
                     [",".join(f"{v:.10g}" for v in r) for r in rows])


def q(y, g):
    return np.round(y / g) * g


def finding(cap, verdict, evidence):
    FINDINGS.append((cap, verdict, evidence))
    print(f"  >> [{verdict}] {cap} -- {evidence}")


# ────────────────────────────────────────────────────────────────────
# W1 -- the repeats road under heavy tails (floor vs planted core)
# ────────────────────────────────────────────────────────────────────
def w1_floor_trials():
    print("=" * 72)
    print("W1 repeats-road floor vs planted core sigma 0.1 "
          "(20 seeds each, 40 groups x 8)")
    print("=" * 72)
    base = np.repeat(np.linspace(0.5, 6, 40), 8)
    stats = {}
    for wname in ("gaussian CONTROL", "student-t df=2",
                  "contaminated 95/5"):
        ratios = []
        for s in range(20):
            r = np.random.default_rng(1000 + s)
            x = base.copy()
            r.shuffle(x)
            if wname.startswith("gauss"):
                noise = r.normal(0, CORE, len(x))
            elif wname.startswith("student"):
                noise = CORE * r.standard_t(2, len(x))
            else:
                noise = r.normal(0, CORE, len(x))
                m5 = r.random(len(x)) < 0.05
                noise[m5] = r.normal(0, 100 * CORE, int(m5.sum()))
            y = 2.0 * x + noise
            mind = mod.Isaac()                 # fresh throwaway mind
            rf = mind.read_floor(y, X=x.reshape(-1, 1))
            ratios.append(np.nan if rf["floor"] is None
                          else rf["floor"] / CORE)
        ratios = np.array(ratios)
        stats[wname] = ratios
        print(f"  {wname:20s} floor/core  median {np.median(ratios):6.2f}"
              f"  min {ratios.min():6.2f}  max {ratios.max():6.2f}"
              f"  -> 4x-floor flag line at "
              f"{4 * np.median(ratios):5.1f} core-sigma")
    g = stats["gaussian CONTROL"]
    t2 = stats["student-t df=2"]
    ct = stats["contaminated 95/5"]
    finding("W1a repeats floor, Gaussian control",
            "WORKS" if abs(np.median(g) - 1.0) < 0.15 else "BROKEN",
            f"median floor/core {np.median(g):.3f} "
            f"(range {g.min():.3f}-{g.max():.3f})")
    finding("W1b repeats floor, Student-t df=2",
            "BROKEN" if np.median(t2) > 1.5 else
            ("PARTIAL" if np.median(t2) > 1.2 else "WORKS"),
            f"median floor/core {np.median(t2):.2f}, seed spread "
            f"{t2.min():.2f}-{t2.max():.2f} (max/min "
            f"{t2.max() / t2.min():.1f}x); flag line median "
            f"{4 * np.median(t2):.1f} core-sigma")
    finding("W1c repeats floor, contaminated Gaussian",
            "BROKEN" if np.median(ct) > 5 else
            ("PARTIAL" if np.median(ct) > 1.5 else "WORKS"),
            f"median floor/core {np.median(ct):.2f} vs core 1.0 "
            f"(predicted 22.4); flag line median "
            f"{4 * np.median(ct):.1f} core-sigma")
    return stats


# ────────────────────────────────────────────────────────────────────
# W1 end-to-end -- the audit under contamination (the r2-ghost)
# ────────────────────────────────────────────────────────────────────
def w1_end_to_end(contaminate):
    label = "CONTAMINATED" if contaminate else "gaussian CONTROL"
    print("-" * 72)
    print(f"W1 end-to-end audit, {label}: 6 planted +1.0 wrongs "
          f"(10 core-sigma), isolated pair 0.6 apart (6 core-sigma)")
    r = np.random.default_rng(77)
    x = np.repeat(np.linspace(0.5, 6, 40), 8)
    r.shuffle(x)
    n = len(x)
    y = 2.0 * x + r.normal(0, CORE, n)
    planted = [3, 47, 111, 200, 260, 301]
    contam_idx = np.array([], int)
    if contaminate:
        pool = np.setdiff1d(np.arange(n), planted)
        contam_idx = r.choice(pool, 16, replace=False)
        y[contam_idx] = 2.0 * x[contam_idx] + r.normal(0, 10.0, 16)
    for i in planted:
        y[i] += 1.0
    xs = np.append(x, [7.0, 7.0])
    ys = np.append(y, [14.0 - 0.3, 14.0 + 0.3])   # honest, distinct
    rep = audit_source(csv_text(["x", "y"],
                                np.column_stack([xs, ys])), mod=mod)
    t = rep["tables"][0]
    fl = t["floors"][t["target"]]
    cand = {f["row"] for f in t["flags"] if f["kind"].endswith("CANDIDATE")}
    dups = [f for f in t["flags"] if f["kind"] == "DUPLICATE"]
    pair_dup = any(sorted(f["receipt"]["rows"]) == [n, n + 1]
                   for f in dups)
    hits = len(cand & set(planted))
    contam_hits = len(cand & set(contam_idx.tolist()))
    false = cand - set(planted) - set(contam_idx.tolist())
    print(f"  law: {t['law'].get('verdict')}  "
          f"{t['law'].get('equation', '')}  r2 {t['law'].get('r2')}")
    print(f"  floor {fl['floor']:.4g} via {fl['road']} "
          f"(= {fl['floor'] / CORE:.1f}x core); 4x flag line "
          f"{4 * fl['floor']:.3g} (= {4 * fl['floor'] / CORE:.0f} "
          f"core-sigma); 3x near-dup line {3 * fl['floor']:.3g}")
    print(f"  flags_status: {t['flags_status'][:70]}")
    print(f"  planted recall {hits}/6; contaminated rows flagged "
          f"{contam_hits}/{len(contam_idx)}; false candidates "
          f"{sorted(false)}; pair(320,321) flagged DUPLICATE: "
          f"{pair_dup}")
    return dict(floor=fl["floor"], recall=hits, pair_dup=pair_dup,
                false=len(false), status=t["flags_status"],
                contam_hits=contam_hits)


# ────────────────────────────────────────────────────────────────────
# W2 -- the grain road: continuous refusal, typo blinding, capture
# ────────────────────────────────────────────────────────────────────
def w2_grain():
    print("=" * 72)
    print("W2a grain on float-continuous data (F4 regression), 10 seeds")
    print("=" * 72)
    refused = 0
    for s in range(10):
        r = np.random.default_rng(300 + s)
        mind = mod.Isaac()
        rf = mind.read_floor(r.normal(0, 1, 400))
        grain = next((rd for rd in rf["roads"]
                      if rd["road"] == "grain"), None)
        refused += (rf["floor"] is None and grain is None)
    print(f"  refusals {refused}/10 (floor None, no grain road)")
    finding("W2a grain on continuous data (F4 regression)",
            "WORKS" if refused == 10 else "BROKEN",
            f"{refused}/10 honest refusals")

    print("-" * 72)
    print("W2b grain consensus vs +0.004 typos "
          "(150 rows quantized at 0.01; consensus gate > 0.95)")
    r = np.random.default_rng(5)
    vals = q(r.uniform(0, 15, 150), 0.01)
    order = np.argsort(vals)
    break_k = None
    hold_max = 0
    for k in (0, 1, 2, 3, 4, 5, 8):
        y2 = vals.copy()
        if k:
            idx = order[np.linspace(15, 135, k).astype(int)]
            y2[idx] += 0.004
        mind = mod.Isaac()
        rf = mind.read_floor(y2)
        grain = next((rd for rd in rf["roads"]
                      if rd["road"] == "grain"), None)
        u = np.unique(y2)
        d = np.diff(u)
        mlt = np.round(d / 0.01)
        good = (mlt > 0) & (np.abs(d / 0.01 - mlt) < 0.02)
        if grain:
            hold_max = k
            print(f"  k={k}: {len(d)} gaps, true-grain consensus "
                  f"{good.mean():.3f} -> grain HELD, step "
                  f"{grain['step']:.6g}, consistency "
                  f"{grain['consistency']:.3f}")
        else:
            if break_k is None:
                break_k = k
            print(f"  k={k}: {len(d)} gaps, true-grain consensus "
                  f"{good.mean():.3f} -> grain REFUSED; floor "
                  f"{rf['floor']} -> flags would be refused: the "
                  f"typos destroyed the ruler that catches typos")
    if break_k is None:
        # post-cure road (typo-concept in the grain reader,
        # promotion #8 2026-07-15): the probe's original summary
        # assumed a breaking point must exist; the cure removed it
        finding("W2b grain consensus vs typo count",
                "WORKS",
                f"holds through k={hold_max} (max tested; pre-cure "
                f"this blinded at k=4) -- dissenting gaps ejected "
                f"from the reading and FILED as mis-entry candidates,"
                f" the ruler kept")
    else:
        finding("W2b grain consensus vs typo count",
                "PARTIAL",
                f"holds through k={hold_max}, blinded at k={break_k} "
                f"({break_k}/150 = {100 * break_k / 150:.1f}% typos); "
                f"blinded -> floor refused -> the typos themselves "
                f"become unflaggable (honest refusal, self-defeating)")

    print("-" * 72)
    print("W2c ONE typo at half-grain offset (+0.005): "
          "consensus capture?")
    u0 = np.unique(vals)
    du = np.diff(u0)
    adj = np.where(np.isclose(du, 0.01))[0]
    if len(adj) == 0:
        print("  no adjacent grid pair in sample -- sub-check BLOCKED")
        finding("W2c half-grain typo capture", "BLOCKED",
                "no adjacent grid pair in the sample")
        return
    y3 = np.append(vals, u0[adj[0]] + 0.005)
    mind = mod.Isaac()
    rf = mind.read_floor(y3)
    grain = next((rd for rd in rf["roads"]
                  if rd["road"] == "grain"), None)
    true_floor = 0.01 / np.sqrt(12.0)
    if grain is None:
        print("  grain refused outright")
        finding("W2c half-grain typo capture", "PARTIAL",
                "grain refused (blinded, not captured)")
    else:
        ratio = grain["sigma"] / true_floor
        print(f"  step read {grain['step']:.6g} (true 0.01), "
              f"consistency {grain['consistency']:.3f}; floor "
              f"{grain['sigma']:.6g} vs true {true_floor:.6g} "
              f"= {ratio:.2f}x")
        captured = abs(grain["step"] - 0.005) < 0.001
        outcome = (f"UNDERSTATED {1 / ratio:.1f}x (quiet rescale)"
                   if captured else "held")
        finding("W2c half-grain typo capture",
                "BROKEN" if captured else "WORKS",
                f"1 typo at +0.005: step read {grain['step']:.4g} "
                f"vs true 0.01 -- floor {outcome}")


# ────────────────────────────────────────────────────────────────────
# W3 -- audit recalibration: affine home game vs quadratic drag
# ────────────────────────────────────────────────────────────────────
def w3_recalibration():
    print("=" * 72)
    print("W3 consensus recalibration: 240 rows, y = 2x + N(0,0.05), "
          "told floor 0.05, 30% dragged")
    print("=" * 72)
    r = np.random.default_rng(21)
    n = 240
    x = r.uniform(0, 10, n)
    noise = r.normal(0, 0.05, n)
    sel = r.random(n) < 0.30
    selrows = set(np.where(sel)[0].tolist())
    y0 = 2.0 * x + noise

    def run(y):
        rep = audit_source(csv_text(["x", "y"],
                                    np.column_stack([x, y])),
                           resolutions={"y": 0.05}, mod=mod)
        t = rep["tables"][0]
        cand = {f["row"] for f in t["flags"]
                if f["kind"].endswith("CANDIDATE")}
        return t, cand

    # clean control
    t, cand = run(y0)
    print(f"  clean CONTROL : law {t['law'].get('verdict')}, status "
          f"{t['flags_status'][:40]}, flags {len(cand)} "
          f"(expect 0), recal {t['recalibration']}")
    clean_false = len(cand)

    # affine-drag control (the designed-for case, audit battery A3)
    ya = y0.copy()
    ya[sel] += 0.6
    t, cand = run(ya)
    rec_a = len(cand & selrows) / max(1, len(selrows))
    fal_a = len(cand - selrows)
    print(f"  affine  +0.6  : law {t['law'].get('verdict')}, status "
          f"{t['flags_status'][:40]}, recall {rec_a:.2f} "
          f"({len(cand & selrows)}/{len(selrows)}), false {fal_a}, "
          f"recal {t['recalibration']}")
    finding("W3 affine-drag control (recalibration home game)",
            "WORKS" if rec_a >= 0.9 and fal_a <= 2 and clean_false <= 2
            else "PARTIAL",
            f"clean control {clean_false} false flags; affine drag "
            f"recall {rec_a:.2f}, false {fal_a}")

    # quadratic drag sweep
    print("  quadratic drag +c*x^2 on the same 30% "
          "(flag line = 4x floor = 0.2):")
    rows = []
    for c in (0.005, 0.01, 0.02, 0.04, 0.08):
        yq = y0.copy()
        yq[sel] += c * x[sel] ** 2
        dev = np.zeros(n)
        dev[sel] = c * x[sel] ** 2
        unamb = set(np.where(sel & (dev > 0.4))[0].tolist())
        t, cand = run(yq)
        active = t["flags_status"] == "active"
        rec_all = len(cand & selrows) / max(1, len(selrows))
        rec_u = (len(cand & unamb) / len(unamb)) if unamb else np.nan
        fal = len(cand - selrows)
        recal = t["recalibration"] or dict(scale=1.0, offset=0.0)
        rows.append(dict(c=c, active=active, rec_all=rec_all,
                         rec_u=rec_u, fal=fal, unamb=len(unamb)))
        print(f"    c={c:<6} law {t['law'].get('verdict', '?'):>10} "
              f"r2 {t['law'].get('r2', '-')}: "
              f"{'ACTIVE' if active else 'REFUSED'} | recall(all sel) "
              f"{rec_all:.2f} | recall(dev>8x-floor) {rec_u:.2f} "
              f"({len(cand & unamb)}/{len(unamb)}) | false flags "
              f"{fal} | recal scale {recal['scale']:.4g} offset "
              f"{recal['offset']:.4g}")
        if not active:
            print(f"      status: {t['flags_status'][:100]}")
    act = [rw for rw in rows if rw["active"]]
    refu = [rw for rw in rows if not rw["active"]]
    worst_u = min((rw["rec_u"] for rw in act), default=np.nan)
    worst_fal = max((rw["fal"] for rw in act), default=0)
    if act and (worst_u < 0.7 or worst_fal > 5):
        verdict = "BROKEN"
        why = (f"gate stayed ACTIVE at c={[rw['c'] for rw in act]} "
               f"yet worst recall on unambiguous wrongs (dev > 8x "
               f"floor) {worst_u:.2f}, worst false-flag count "
               f"{worst_fal} -- silent mis-recalibration, no "
               f"self-disqualification")
    elif refu and not act:
        verdict = "HONEST-REFUSAL"
        why = (f"coverage gate refused at every c "
               f"{[rw['c'] for rw in refu]} -- disqualified itself "
               f"honestly")
    else:
        verdict = "PARTIAL"
        why = (f"active at c={[rw['c'] for rw in act]} (min "
               f"unambiguous recall {worst_u:.2f}, max false "
               f"{worst_fal}), refused at "
               f"c={[rw['c'] for rw in refu]}")
    finding("W3 quadratic (non-affine) drag vs affine-only "
            "recalibration", verdict, why)


def main():
    print("FLOOR / HEAVY-TAIL / NON-AFFINE-DRAG PROBE  "
          "(predictions P1-P7 filed in the docstring before this run)")
    w1_floor_trials()
    ctrl = w1_end_to_end(contaminate=False)
    cont = w1_end_to_end(contaminate=True)
    finding("W1e end-to-end: planted 10-sigma wrongs under "
            "contamination (r2-ghost laundering)",
            "BROKEN" if (ctrl["recall"] == 6 and cont["recall"] < 3)
            else ("WORKS" if cont["recall"] >= 5 else "PARTIAL"),
            f"control recall {ctrl['recall']}/6 (floor "
            f"{ctrl['floor']:.3g}); contaminated recall "
            f"{cont['recall']}/6 (floor {cont['floor']:.3g} = "
            f"{cont['floor'] / CORE:.0f}x core, flag line "
            f"{4 * cont['floor'] / CORE:.0f} core-sigma); "
            f"contam rows flagged {cont['contam_hits']}/16")
    finding("W1e near-dup 3x-floor line under inflated floor",
            "BROKEN" if (cont["pair_dup"] and not ctrl["pair_dup"])
            else ("WORKS" if not cont["pair_dup"] else "PARTIAL"),
            f"6-core-sigma-apart pair flagged DUPLICATE: control "
            f"{ctrl['pair_dup']}, contaminated {cont['pair_dup']} "
            f"(3x line {3 * cont['floor']:.2g} vs separation 0.6)")
    # the filed side-prediction: the control floor itself
    finding("W1e control floor inflated by its own planted wrongs",
            "PARTIAL" if ctrl["floor"] > 1.3 * CORE else "WORKS",
            f"control floor {ctrl['floor']:.3g} vs core {CORE} "
            f"({ctrl['floor'] / CORE:.2f}x; predicted ~1.7x) -- the "
            f"repeats ruler is estimated FROM the audited data")
    w2_grain()
    w3_recalibration()
    print("=" * 72)
    print("VERDICTS (every number above is measured, none patched)")
    print("=" * 72)
    for cap, verdict, ev in FINDINGS:
        print(f"  [{verdict:14s}] {cap}\n{'':20s}{ev}")


if __name__ == "__main__":
    main()

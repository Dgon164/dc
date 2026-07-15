"""
AUDIT ADVERSARIAL PROBE -- the audit face PAST its battery (A1-A8 pass;
this probe attacks the promises the battery proved only on easy worlds).

THE CLAIM UNDER TEST (audit.py docstring + test_audit.py):
  the audit produces flags WITH RECEIPTS, honest refusals, and asks --
  "never a silent pass"; A2 promises ~0 false flags on a CLEAN table
  (proven on Gaussian noise only); the 07-15 consensus-recalibration
  record claims planted deviations "cannot drag the judge" (proven on a
  ~6%-of-rows +8.0 drag, slope 2.0->1.89 cured); the near-dup line is a
  HAND-SET 3x floor; news-vs-error is decided by the commonality miner.

Worlds, every truth planted by this probe:
  V1  CLEAN heavy-tailed: lawful y=2x + Student-t(df=3) noise, no bad
      rows at all.  False-flag rate vs the A2 ~0 promise, on both floor
      roads (repeats, told), Gaussian twin as the control.  Analytic
      t3 tail expectation printed next to every measured count.
  V2  DRAG, harder than the cured record: (a) 20% of rows coordinated
      y*=1.2 (a 20% deviation; the record cured a ~10% drag);
      (b) 25% of rows non-affine drag y += 0.2*x^2 -- affine-only
      recalibration is the audit's stated honest v1 scope, so the
      lawful outcomes are recovery OR honest self-disqualification;
      good rows flagged as errors is the failure. 0%-corrupt control.
  V3  NEAR-DUP boundary: isolated equal-X pairs planted at 2.5x, 3.0x,
      3.5x the told floor + one exact pair (the 3x line is hand-set;
      note the code's error cut is 4x floor, so 3x..4x is a predicted
      DEAD ZONE: invisible to both classes).  No-pairs control.
  V4  NEWS-vs-ERROR under correlated (AR(1) rho=0.8) noise: a planted
      recurring band regime on x1 (NEWS, shared factor named) + 4
      scattered one-off corruptions (ERROR) + 1 adversarial one-off
      INSIDE the band's quartile bin but outside the band.  Full
      confusion matrix.  No-plants correlated-noise control.
  V5  DOOR LAWS past A8: garbage bytes wearing a .csv name (A8a only
      tried an unsupported extension), and an oversized single file
      over the 200MB total-bytes budget -- clean refusals WITH WHYS,
      companion table still audited, render never crashes.

PREDICTIONS, filed before the first run (from code reading only):
  P1  Gaussian controls: 0 false flags (A2 replicates).  t3: the
      promise does NOT generalize -- expect ~2/320 flags per seed on
      the repeats road (cut ~= 4*sigma_pooled ~= 6.9 scale units) and
      ~9/320 on the told road (cut = 4 scale units, P(|t3|>4)=2.8%).
  P2a Consensus recal survives: recall >= 0.95 on corrupted rows,
      clean false flags <= 1%, judge slope within 2% of 2.0.
  P2b The quadratic drag is NOT affinely curable: either discover's
      law bends / recal leaves structure and CLEAN rows get flagged,
      or the MAD gate trips an honest refusal.  Primary prediction:
      honest self-disqualification OR clean-row flags > 1% -- the
      cured-drag story does not extend to non-affine drag.
  P3  2.5x pairs 3/3 flagged near-dup; 3.5x pairs 0/3 flagged AND
      0/3 error-flagged (the 3x..4x dead zone is real); exact pair
      flagged; the 3.0x pair is a float coin-flip at the <= boundary.
  P4  Band recalled as NEWS >= 0.8 with an x1 factor named despite
      AR(1) noise; the 4 out-of-bin one-offs land ERROR; the 1 in-bin
      one-off is MISCLASSIFIED as NEWS (the code keys on bin
      membership, not per-row recurrence).  Control: 0 flags.
  P5  Both door refusals clean with whys; no crash; companion audited.

POST-FIRST-RUN NOTE (same session; predictions above left verbatim):
  P4 was REFUTED in an unpredicted direction -- the audit REFUSED the
  V4 flag pass (mad gate) because discover() absorbed part of the
  planted band into x0*cos(x1) cross-atoms, bending the law's FORM and
  inflating every clean row's residual.  Two attribution twins were
  added AFTER seeing that (iid-noise twin; no-one-offs twin) to name
  the responsible ingredient.  Diagnostics only -- nothing was patched.

House rules: worlds planted with known truth; every positive claim has
a clean/control twin; found wrongs are REPORTED with numbers, never
patched.  stdlib + numpy only.  Fresh throwaway minds only (the audit
runs a fresh Isaac per call by construction).
Run: cd experiments && PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 \
     OMP_NUM_THREADS=1 python3 audit_adversarial_probe.py
"""
import math
import os
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from isaac_store import bridge                              # noqa: E402
from isaac_store.audit import audit_source, render_audit    # noqa: E402

MOD = bridge.load_isaac_module()   # loaded once; audit_source(mod=MOD)
                                   # still builds a FRESH mind per call
T0 = time.time()
VERDICTS = []


def say(*a):
    print(*a, flush=True)


def verdict(world, capability, v, evidence):
    VERDICTS.append((world, capability, v, evidence))
    say(f"  VERDICT[{world}] {capability}: {v}")
    say(f"    evidence: {evidence}")


def csv_text(cols, rows):
    return "\n".join([",".join(cols)] +
                     [",".join(f"{v:.10g}" for v in r) for r in rows])


def audit_csv(cols, rows, resolutions=None, fname="t.csv"):
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, fname)
        with open(p, "w", encoding="utf-8") as f:
            f.write(csv_text(cols, rows))
        return audit_source(p, resolutions=resolutions, mod=MOD)


def cand_flags(t):
    return [f for f in t["flags"] if f["kind"].endswith("CANDIDATE")]


def t3_two_sided(t):
    """P(|T| > t) for Student-t df=3, closed form (stdlib math)."""
    u = t / math.sqrt(3.0)
    F = 0.5 + (math.atan(u) + u / (1.0 + u * u)) / math.pi
    return 2.0 * (1.0 - F)


# ---------------------------------------------------------------- V1
def v1():
    say("=" * 72)
    say("V1  CLEAN heavy-tailed table -- the A2 ~0-false-flag promise")
    say("    (lawful y = 2x + noise; NO bad rows; every flag is false)")
    scale = 0.05
    results = {}
    for design in ("repeats", "told"):
        for noise in ("gauss", "t3"):
            seeds = (11,) if noise == "gauss" else (11, 12, 13)
            tot_rows = tot_flags = 0
            per_seed = []
            for sd in seeds:
                rng = np.random.default_rng(sd)
                if design == "repeats":
                    x = np.repeat(np.linspace(0.5, 6, 40), 8)
                    rng.shuffle(x)
                    res = None
                else:
                    x = rng.uniform(0.5, 6, 320)
                    res = {"y": scale}
                eps = (rng.normal(0, scale, len(x)) if noise == "gauss"
                       else scale * rng.standard_t(3, len(x)))
                y = 2.0 * x + eps
                rep = audit_csv(["x", "y"], np.column_stack([x, y]),
                                resolutions=res)
                t = rep["tables"][0]
                fl = cand_flags(t)
                st = t["flags_status"]
                floor = t["floors"]["y"]["floor"]
                road = t["floors"]["y"]["road"]
                if not st.startswith("active"):
                    per_seed.append(("REFUSED", st[:60]))
                    continue
                exp = (len(x) * t3_two_sided(4.0 * floor / scale)
                       if noise == "t3" else
                       len(x) * math.erfc(4.0 * floor / scale
                                          / math.sqrt(2)))
                kinds = {}
                for f in fl:
                    kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
                per_seed.append((len(fl), floor, road, exp, kinds))
                tot_rows += len(x)
                tot_flags += len(fl)
            results[(design, noise)] = (tot_flags, tot_rows, per_seed)
            for s, ps in zip(seeds, per_seed):
                if ps[0] == "REFUSED":
                    say(f"  {design:7s}/{noise:5s} seed {s}: "
                        f"REFUSED -- {ps[1]}")
                else:
                    nf, floor, road, exp, kinds = ps
                    say(f"  {design:7s}/{noise:5s} seed {s}: "
                        f"{nf} false flags / 320 rows "
                        f"(floor {floor:.4g} via {road}; analytic "
                        f"expectation {exp:.2f}) kinds {kinds or '{}'}")
            if tot_rows:
                say(f"  {design:7s}/{noise:5s} POOLED false-flag rate: "
                    f"{tot_flags}/{tot_rows} = "
                    f"{tot_flags / tot_rows:.4f}")
    g_r = results[("repeats", "gauss")]
    g_t = results[("told", "gauss")]
    t_r = results[("repeats", "t3")]
    t_t = results[("told", "t3")]
    ev = (f"gauss controls {g_r[0]}/{g_r[1]} (repeats) and "
          f"{g_t[0]}/{g_t[1]} (told) false flags; t3 heavy-tail "
          f"{t_r[0]}/{t_r[1]} (repeats, rate {t_r[0]/max(t_r[1],1):.4f})"
          f" and {t_t[0]}/{t_t[1]} (told, rate "
          f"{t_t[0]/max(t_t[1],1):.4f})")
    if g_r[0] == 0 and g_t[0] == 0 and (t_r[0] + t_t[0]) == 0:
        v = "WORKS"
    elif g_r[0] == 0 and g_t[0] == 0:
        v = "PARTIAL"
        ev += (" -- the ~0 promise is GAUSSIAN-SCOPED: heavy tails "
               "produce false ERROR flags on a fully lawful table, "
               "at the analytically predicted rate")
    else:
        v = "BROKEN"
    verdict("V1", "clean-table ~0 false flags under heavy tails", v, ev)
    say(f"  [{time.time() - T0:.1f}s]")


# ---------------------------------------------------------------- V2
def v2():
    say("=" * 72)
    say("V2  DRAG past the cured record (07-15: 10%-dev drag "
        "2.0->1.89 cured by consensus recalibration)")
    rng = np.random.default_rng(21)
    n = 300
    x = rng.uniform(0.2, 6, n)
    y0 = 2.0 * x + rng.normal(0, 0.05, n)

    def run(y, tag):
        rep = audit_csv(["x", "y"], np.column_stack([x, y]),
                        resolutions={"y": 0.05}, fname=f"{tag}.csv")
        return rep["tables"][0]

    # -- control: 0% corrupt --
    t = run(y0, "v2ctl")
    fl = cand_flags(t)
    rc = t["recalibration"]
    say(f"  control 0% corrupt: {len(fl)} flags / {n} rows; "
        f"law {t['law']['verdict']} "
        f"{t['law'].get('equation', '')!r}; recal "
        f"{rc if rc else 'none (scale 1, offset 0)'}")
    ctl_ok = len(fl) == 0

    # -- V2a: 20% coordinated multiplicative drag (y *= 1.2) --
    elig = np.where(x >= 1.0)[0]
    bad_a = np.sort(rng.choice(elig, size=60, replace=False))
    ya = y0.copy()
    ya[bad_a] *= 1.2                      # 20% deviation, one direction
    t = run(ya, "v2a")
    fl = cand_flags(t)
    flagged = {f["row"] for f in fl}
    bad_set = set(bad_a.tolist())
    tp = len(flagged & bad_set)
    fp = len(flagged - bad_set)
    recall = tp / len(bad_set)
    prec = tp / max(len(flagged), 1)
    kinds = {}
    for f in fl:
        kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
    rc = t["recalibration"]
    say(f"  V2a 20% coordinated drag: status {t['flags_status'][:40]!r}")
    say(f"    law {t['law']['verdict']} {t['law'].get('equation','')!r}")
    say(f"    recal {rc}")
    say(f"    planted 60 corrupt (all x>=1, deviation >= 8 floors); "
        f"flagged {len(flagged)}; TP {tp} FP {fp}")
    say(f"    recall {recall:.3f}  precision {prec:.3f}  kinds {kinds}")
    slope = None
    if tp >= 2:
        pv = np.array([f["receipt"]["predicted"] for f in fl
                       if f["row"] in bad_set])
        rows_in_order = [f["row"] for f in fl if f["row"] in bad_set]
        px = x[rows_in_order]
        A = np.column_stack([px, np.ones(len(px))])
        (slope, itc), *_ = np.linalg.lstsq(A, pv, rcond=None)
        say(f"    judge line from receipts: predicted ~= "
            f"{slope:.4f}*x + {itc:.4f}  (truth 2.0000*x + 0; the "
            f"uncured 07-15 analog was 1.89*x + 1.5)")
    eva = (f"recall {recall:.3f}, clean FP {fp}/{n - 60} = "
           f"{fp / (n - 60):.4f}, judge slope "
           f"{'%.4f' % slope if slope is not None else 'n/a'}; "
           f"control clean: {ctl_ok}")
    if t["flags_status"].startswith("refused"):
        va = "HONEST-REFUSAL"
    elif recall >= 0.95 and fp / (n - 60) <= 0.01 and slope is not None \
            and abs(slope - 2.0) <= 0.05:
        va = "WORKS"
    elif recall < 0.5 or fp / (n - 60) > 0.05:
        va = "BROKEN"
    else:
        va = "PARTIAL"
    verdict("V2a", "consensus recal vs 20% coordinated drag", va, eva)

    # -- V2b: non-affine (quadratic) drag on 25% of rows --
    elig = np.where(x >= 1.5)[0]
    bad_b = np.sort(rng.choice(elig, size=75, replace=False))
    yb = y0.copy()
    yb[bad_b] += 0.2 * x[bad_b] ** 2       # deviation 0.45 .. 7.2
    t = run(yb, "v2b")
    fl = cand_flags(t)
    flagged = {f["row"] for f in fl}
    bad_set = set(bad_b.tolist())
    tp = len(flagged & bad_set)
    fp = len(flagged - bad_set)
    recall = tp / len(bad_set)
    say(f"  V2b 25% quadratic drag: status {t['flags_status'][:70]!r}")
    say(f"    law {t['law']['verdict']} {t['law'].get('equation','')!r}")
    say(f"    recal {t['recalibration']}")
    say(f"    planted 75 corrupt (x>=1.5, dev = 0.2x^2 >= 9 floors); "
        f"flagged {len(flagged)}; TP {tp} FP {fp} "
        f"(clean FP rate {fp / (n - 75):.4f}) recall {recall:.3f}")
    if t["flags_status"].startswith("refused"):
        vb, evb = "HONEST-REFUSAL", (
            f"self-disqualified: {t['flags_status'][:90]!r} -- 0 "
            f"good rows condemned; affine-only scope held honestly")
    elif recall >= 0.9 and fp / (n - 75) <= 0.01:
        vb = "WORKS"
        evb = (f"recall {recall:.3f}, clean FP rate "
               f"{fp / (n - 75):.4f} -- survived non-affine drag")
    elif fp / (n - 75) > 0.05:
        vb = "BROKEN"
        evb = (f"good rows condemned: {fp} clean rows flagged "
               f"(rate {fp / (n - 75):.4f}), recall {recall:.3f} -- "
               f"the affine cure bent the judge instead of refusing")
    else:
        vb = "PARTIAL"
        evb = (f"recall {recall:.3f}, clean FP {fp}/{n - 75} = "
               f"{fp / (n - 75):.4f}")
    verdict("V2b", "non-affine (quadratic) drag: recover or refuse",
            vb, evb)
    say(f"  [{time.time() - T0:.1f}s]")


# ---------------------------------------------------------------- V3
def v3():
    say("=" * 72)
    say("V3  NEAR-DUP boundary -- the hand-set 3x-floor line "
        "(told floor 0.01; error cut is 4x floor = the 3x..4x "
        "dead-zone question)")
    FLOOR = 0.01
    rng = np.random.default_rng(31)
    n = 140
    x = rng.uniform(0, 5, n)
    y = 3.0 * x + rng.normal(0, 0.003, n)
    rows = np.column_stack([x, y]).tolist()
    # planted pairs: (base_row, offset_in_floors)
    plants = [(5, 2.5), (25, 2.5), (45, 2.5),
              (65, 3.0),
              (85, 3.5), (105, 3.5), (125, 3.5),
              (10, 0.0)]                       # 0.0 = exact duplicate
    pair_of = {}
    for k, (i, mult) in enumerate(plants):
        rows.append([x[i], y[i] + mult * FLOOR])
        pair_of[(i, n + k)] = mult
    # 8 repeat groups, dof 8 < 10 -> repeats road stays un-evidenced,
    # the told floor 0.01 anchors the 3x line (checked below).
    rep = audit_csv(["x", "y"], np.array(rows),
                    resolutions={"y": FLOOR}, fname="v3.csv")
    t = rep["tables"][0]
    say(f"  floor: {t['floors']['y']['floor']} via "
        f"{t['floors']['y']['road']} (must be told/{FLOOR:g}); "
        f"status {t['flags_status'][:40]!r}")
    dup_cited = {tuple(sorted(f["receipt"]["rows"]))
                 for f in t["flags"] if f["kind"] == "DUPLICATE"}
    err_rows = {f["row"] for f in cand_flags(t)}
    hits = {2.5: 0, 3.0: 0, 3.5: 0, 0.0: 0}
    tot = {2.5: 0, 3.0: 0, 3.5: 0, 0.0: 0}

    def rt(v):                       # the csv %.10g round trip
        return float(f"{v:.10g}")

    for (i, j), mult in sorted(pair_of.items(), key=lambda kv: kv[1]):
        dy = abs(rt(y[i] + mult * FLOOR) - rt(y[i]))
        got = (i, j) in dup_cited
        tot[mult] += 1
        hits[mult] += got
        as_err = j in err_rows or i in err_rows
        say(f"  pair ({i},{j}) at {mult:.1f}x floor: |dy| = {dy:.17f} "
            f"vs 3*floor = {3 * FLOOR:.17f} -> "
            f"{'DUPLICATE flagged' if got else 'not flagged'}"
            f"{'; ERROR-flagged' if as_err else ''}")
    spurious = len(dup_cited) - sum(hits.values())
    say(f"  planted-pair recall below the line (2.5x): "
        f"{hits[2.5]}/{tot[2.5]}; exact: {hits[0.0]}/{tot[0.0]}; "
        f"at the line (3.0x): {hits[3.0]}/{tot[3.0]}; above (3.5x): "
        f"{hits[3.5]}/{tot[3.5]} (0 = per the hand-set line)")
    say(f"  spurious DUPLICATE flags on unplanted rows: {spurious}")
    dead = [(i, j) for (i, j), m in pair_of.items()
            if m == 3.5 and (i, j) not in dup_cited
            and j not in err_rows and i not in err_rows]
    say(f"  DEAD ZONE check (3x..4x floor): {len(dead)}/{tot[3.5]} "
        f"3.5x pairs invisible to BOTH the duplicate class (>3x) and "
        f"the error class (<=4x) -- a silent pass by construction "
        f"of the two hand-set lines")
    # control: same base table, no planted pairs
    repc = audit_csv(["x", "y"], np.column_stack([x, y]),
                     resolutions={"y": FLOOR}, fname="v3c.csv")
    tc = repc["tables"][0]
    ndc = sum(1 for f in tc["flags"] if f["kind"] == "DUPLICATE")
    say(f"  control (no pairs planted): {ndc} DUPLICATE flags "
        f"(expect 0)")
    prec_den = len(dup_cited)
    prec = (sum(hits.values())) / prec_den if prec_den else float("nan")
    ev = (f"2.5x recall {hits[2.5]}/{tot[2.5]}, exact "
          f"{hits[0.0]}/{tot[0.0]}, 3.0x boundary {hits[3.0]}/"
          f"{tot[3.0]}, 3.5x {hits[3.5]}/{tot[3.5]}; dup precision "
          f"{prec:.3f}; control dups {ndc}; dead-zone pairs "
          f"{len(dead)}/{tot[3.5]} fully invisible")
    if (hits[2.5] == tot[2.5] and hits[0.0] == tot[0.0]
            and hits[3.5] == 0 and spurious == 0 and ndc == 0):
        v = "PARTIAL" if len(dead) == tot[3.5] else "WORKS"
        if v == "PARTIAL":
            ev += (" -- the line behaves exactly as hand-set, BUT the "
                   "3x..4x gap is a SILENT PASS class: a near-dup at "
                   "3.5x floor is neither DUPLICATE nor ERROR nor "
                   "accounted anywhere, against the never-a-silent-"
                   "pass charter")
    else:
        v = "BROKEN"
    verdict("V3", "near-dup hand-set 3x line, both sides", v, ev)
    say(f"  [{time.time() - T0:.1f}s]")


# ---------------------------------------------------------------- V4
def _ar1(rng, n, rho, sig):
    e = np.empty(n)
    e[0] = rng.normal(0, sig)
    for i in range(1, n):
        e[i] = rho * e[i - 1] + rng.normal(0, sig * math.sqrt(1 - rho**2))
    return e


def v4():
    say("=" * 72)
    say("V4  NEWS-vs-ERROR under correlated AR(1) noise -- "
        "confusion matrix")
    rng = np.random.default_rng(41)
    n = 300
    x0 = rng.uniform(0, 10, n)
    x1 = rng.uniform(0, 10, n)
    x1[0] = 5.95            # the adversarial one-off's seat, planted
    rho, sig = 0.8, 0.1     # AR(1) serially correlated noise
    e_ar = _ar1(rng, n, rho, sig)
    e_iid = np.random.default_rng(43).normal(0, sig, n)  # twin's noise
    qs = np.quantile(x1, [0.25, 0.5, 0.75])
    say(f"  x1 sample quartile edges: {qs.round(3).tolist()} "
        f"(band [6.2,6.8) must sit inside one bin: "
        f"{qs[1] < 6.2 and 6.8 <= qs[2]})")
    band = (x1 >= 6.2) & (x1 < 6.8)
    band_rows = set(np.where(band)[0].tolist())
    elig = np.where((~band) & ((x1 < 4.8) | (x1 >= 7.7)))[0]
    oneoff = sorted(rng.choice(elig[elig != 0], size=4,
                               replace=False).tolist())
    say(f"  planted: {len(band_rows)} band rows (+6.0, x1 in "
        f"[6.2,6.8)); one-offs at rows {oneoff} (out-of-bin, "
        f"+13/-11/+16/-14) and row 0 (x1=5.95, in-bin, +12, does "
        f"NOT recur)")

    def build(eps, with_band=True, with_oneoffs=True):
        y = 2.0 * x0 + eps.copy()
        if with_band:
            y[band] += 6.0                      # recurring regime: NEWS
        if with_oneoffs:
            for k, i in enumerate(oneoff):
                y[i] += (13.0, -11.0, 16.0, -14.0)[k]   # ERROR
            y[0] += 12.0    # adversarial one-off INSIDE the band's bin
        return y

    def run(y, tag):
        rep = audit_csv(["x0", "x1", "y"],
                        np.column_stack([x0, x1, y]),
                        resolutions={"y": sig}, fname=f"{tag}.csv")
        t = rep["tables"][0]
        cls = {f["row"]: f["kind"] for f in cand_flags(t)}
        news_rows = {i for i, k in cls.items() if k == "NEWS-CANDIDATE"}
        err_rows = {i for i, k in cls.items() if k == "ERROR-CANDIDATE"}
        m = dict(
            refused=t["flags_status"].startswith("refused"),
            status=t["flags_status"], law=t["law"], cls=cls,
            n_rec=len(news_rows & band_rows) / len(band_rows),
            n_pre=(len(news_rows & band_rows) / len(news_rows)
                   if news_rows else float("nan")),
            e_rec=len(err_rows & set(oneoff)) / len(oneoff),
            e_pre=(len(err_rows & set(oneoff)) / len(err_rows)
                   if err_rows else float("nan")),
            clean_fp=len((news_rows | err_rows)
                         - band_rows - set(oneoff) - {0}),
            x1_named=any(k == "x1" for k, v, _ in t["news_factors"]),
            factors=[(k, v, round(e_, 3))
                     for k, v, e_ in t["news_factors"]],
            adv=cls.get(0, "unflagged"))
        return m

    # -- the attacked world: plants under AR(1) noise --
    m = run(build(e_ar), "v4")
    say(f"  MAIN (AR rho=0.8): status {m['status'][:90]!r}")
    say(f"    law {m['law']['verdict']} "
        f"{m['law'].get('equation', '')!r}")
    say(f"    news factors named: {m['factors']}")
    planted = {}
    for i in range(n):
        planted[i] = ("NEWS(band)" if i in band_rows else
                      "ERROR(one-off)" if i in oneoff else
                      "ERROR(in-bin one-off)" if i == 0 else "CLEAN")
    mat = {}
    for i in range(n):
        key = (planted[i], m["cls"].get(i, "unflagged"))
        mat[key] = mat.get(key, 0) + 1
    say("    CONFUSION MATRIX (planted -> audited):")
    for p in ("NEWS(band)", "ERROR(one-off)", "ERROR(in-bin one-off)",
              "CLEAN"):
        row = {g: c for (pp, g), c in mat.items() if pp == p}
        say(f"      {p:22s} {row}")
    say(f"    news recall {m['n_rec']:.3f} precision {m['n_pre']:.3f};"
        f" error recall {m['e_rec']:.3f} precision {m['e_pre']:.3f}; "
        f"clean-row false flags {m['clean_fp']}; x1 named "
        f"{m['x1_named']}; in-bin one-off -> {m['adv']}")

    # -- attribution twin A: SAME plants, iid noise (rho=0) --
    ma = run(build(e_iid), "v4iid")
    say(f"  TWIN A (iid, same plants): status {ma['status'][:60]!r}; "
        f"law {ma['law']['verdict']} {ma['law'].get('equation','')!r}")
    say(f"    news recall {ma['n_rec']:.3f} precision "
        f"{ma['n_pre']:.3f}; error recall {ma['e_rec']:.3f}; clean FP "
        f"{ma['clean_fp']}; x1 named {ma['x1_named']}; factors "
        f"{ma['factors']}; in-bin one-off -> {ma['adv']}")
    # -- attribution twin B: AR noise, band only, no one-offs --
    mb = run(build(e_ar, with_oneoffs=False), "v4band")
    say(f"  TWIN B (AR, band only): status {mb['status'][:60]!r}; "
        f"law {mb['law']['verdict']} {mb['law'].get('equation','')!r}")
    say(f"    news recall {mb['n_rec']:.3f}; x1 named "
        f"{mb['x1_named']}; factors {mb['factors']}")
    # -- null control: correlated noise, nothing planted --
    e2 = _ar1(np.random.default_rng(42), n, rho, sig)
    mc = run(build(e2, with_band=False, with_oneoffs=False), "v4null")
    nfc = len(mc["cls"])
    say(f"  CONTROL (AR noise, no plants): {nfc} flags, status "
        f"{mc['status'][:40]!r}, factors {mc['factors']}")

    ev = (f"MAIN news recall {m['n_rec']:.3f}, error recall "
          f"{m['e_rec']:.3f}, x1 named {m['x1_named']}, refused="
          f"{m['refused']}; TWIN A (iid) news recall {ma['n_rec']:.3f}"
          f", x1 named {ma['x1_named']}, in-bin one-off -> {ma['adv']}"
          f"; TWIN B (band only) news recall {mb['n_rec']:.3f}; "
          f"control flags {nfc}")
    if m["refused"]:
        ingredient = (
            "correlated noise IS the ingredient: the iid twin "
            "classifies fine" if not ma["refused"] else
            "the iid twin refuses IDENTICALLY, so correlated noise is "
            "NOT the ingredient -- the planted regime itself is "
            "absorbable, i.e. the A3 promise 'a law cannot absorb "
            "the band' is world-fragile (this band seat/height gets "
            "eaten where A3's did not)")
        ev += (" -- no plant-bearing variant was classified: "
               "discover() absorbed the planted regime into the "
               f"law's form ({m['law'].get('equation', '')!r}), "
               "inflating clean residuals until the mad gate "
               f"self-disqualified; {ingredient}; the refusal's why "
               "blames the floor, but the floor (0.1) was truthfully "
               "told -- a MISDIAGNOSED refusal whose ASK is already "
               "answered; twin B's law card even says "
               f"{mb['law']['verdict']} on a plant-bent equation; "
               "the records ledger carries law+floor lines but NO "
               "refusal line, and counts read 'unflagged'")
        v = "BROKEN" if not ma["refused"] else "HONEST-REFUSAL"
    elif (m["n_rec"] >= 0.8 and m["x1_named"] and m["e_rec"] >= 0.75
            and m["clean_fp"] <= 3 and nfc <= 1):
        v = "PARTIAL" if m["adv"] == "NEWS-CANDIDATE" else "WORKS"
        if v == "PARTIAL":
            ev += (" -- classification is BY BIN MEMBERSHIP, not "
                   "per-row recurrence: one planted one-off sharing "
                   "the named bin is laundered into NEWS")
    else:
        v = "BROKEN"
    verdict("V4", "news-vs-error confusion under correlated noise",
            v, ev)
    say(f"  [{time.time() - T0:.1f}s]")


# ---------------------------------------------------------------- V5
def v5():
    say("=" * 72)
    say("V5  DOOR LAWS past A8 -- garbage bytes AS .csv; oversized "
        "single file")
    rng = np.random.default_rng(51)
    x = np.repeat(np.linspace(0.5, 6, 40), 8)
    rng.shuffle(x)
    yv = 2.0 * x + rng.normal(0, 0.05, len(x))
    fine = csv_text(["x", "y"], np.column_stack([x, yv]))
    ok_a = ok_b = False
    ev_a = ev_b = ""
    # (a) garbage bytes wearing the .csv extension (A8a only tried an
    #     unsupported extension -- here the parser itself must refuse)
    try:
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "a_fine.csv"), "w") as f:
                f.write(fine)
            junk = rng.integers(0, 256, 4096, dtype=np.uint8).tobytes()
            with open(os.path.join(td, "garbage.csv"), "wb") as f:
                f.write(junk)
            rep = audit_source(td, mod=MOD)
        render_audit(rep)                      # must not crash either
        sk = [s for s in rep["intake"]["skipped"]
              if s["source"] == "garbage.csv"]
        ok_a = (len(sk) == 1 and bool(sk[0]["why"])
                and len(rep["tables"]) == 1
                and rep["tables"][0]["counts"]["rows"] == 320)
        ev_a = (f"garbage.csv skipped with why "
                f"{sk[0]['why']!r}; " if sk else
                "garbage.csv NOT in skipped; ")
        ev_a += (f"{len(rep['tables'])} table(s) audited, "
                 f"{rep['tables'][0]['counts'] if rep['tables'] else 0}"
                 f" -- no crash, render ok")
    except Exception as exc:
        ev_a = f"CRASHED: {type(exc).__name__}: {exc}"
    say(f"  (a) {ev_a}")
    # (b) oversized single file: 201 MB > the 200 MB total budget
    try:
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "a_fine.csv"), "w") as f:
                f.write(fine)
            hp = os.path.join(td, "z_huge.csv")
            with open(hp, "wb") as f:          # sparse: fast to write
                f.seek(201 * 1024 * 1024 - 1)
                f.write(b"\0")
            rep = audit_source(td, mod=MOD)
        render_audit(rep)
        sk = [s for s in rep["intake"]["skipped"]
              if s["source"] == "z_huge.csv"]
        ok_b = (len(sk) == 1 and "budget" in sk[0]["why"]
                and len(rep["tables"]) == 1)
        ev_b = (f"z_huge.csv (201MB) skipped with why "
                f"{sk[0]['why']!r}; " if sk else
                "z_huge.csv NOT in skipped; ")
        ev_b += (f"{len(rep['tables'])} table(s) still audited -- "
                 f"no crash, render ok")
    except Exception as exc:
        ev_b = f"CRASHED: {type(exc).__name__}: {exc}"
    say(f"  (b) {ev_b}")
    v = ("WORKS" if ok_a and ok_b else
         "PARTIAL" if ok_a or ok_b else "BROKEN")
    verdict("V5", "door laws: refuse with whys, audit the rest",
            v, f"(a) {ev_a} | (b) {ev_b}")
    say(f"  [{time.time() - T0:.1f}s]")


def main():
    say("AUDIT ADVERSARIAL PROBE -- predictions filed in the "
        "docstring BEFORE the first run (P1-P5)")
    v1()
    v2()
    v3()
    v4()
    v5()
    say("=" * 72)
    say("SUMMARY -- per-world verdicts")
    for w, cap, v, ev in VERDICTS:
        say(f"  [{v:14s}] {w}: {cap}")
        say(f"      {ev}")
    say(f"total {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()

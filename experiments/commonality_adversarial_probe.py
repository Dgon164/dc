"""
COMMONALITY ADVERSARIAL PROBE -- attack probe (Isaac testing segment).

THE CLAIM UNDER TEST: mine_commonality (isaac.py:2208; battery 7/7)
names what the failures share that the successes lack -- a DIFFERENTIAL
against the allowed, gated by a selection-corrected label-shuffle null
(threshold = 2nd-largest of 300 max-over-candidates null stats), NEVER
a frequency. This probe attacks that law at five pressure points. If
the organ fails a world, that is the FINDING -- reported, never patched.

WORLDS
  C1 CORRELATED CONDITIONS  two features perfectly correlated, only
     one causal: both / one / honest tie?  Plus a 90%-mirror twin, a
     no-signal twin control, and a label-shuffle scramble of the
     causal world (nothing may be named on either control).
  C2 HIGH CARDINALITY + SELECTION PRESSURE  a 50-category feature
     (one category causal) among 8 four-valued distractors (82
     candidates).  20 seeded NO-SIGNAL worlds -> false-naming rate
     (must be ~0 if the null is genuinely selection-corrected);
     20 hard-signal (base fail .08) and 20 easy-signal (base .02)
     worlds -> hit / missed / extra-naming rates.
  C3 NEAR-BACKGROUND  the causal flag is on 100% of failures but also
     ~95% of successes (true differential ~.06): named, or honestly
     nothing?  At n=600 and n=6000, with an equal-rate 95%-background
     control (flag must never be named there).
  C4 TINY SAMPLES  exactly ~10 failures per world: 20 deterministic-
     cause worlds, 20 weak-cause worlds, 20 no-signal worlds ->
     hit rate / honest-nothing rate / false-naming rate.
  C5 CONTROL  the 07-14 reference world, rng-replicated verbatim from
     failure_commonality_probe.py, pushed through the ENGINE organ
     (the reference probe predates the organ): W1 excess .496/.431 vs
     null99 .159, W2 lockout .897 vs .140, control A nothing.

PREDICTIONS (filed before the first full run):
  P1 perfect twins are statistically identical -> BOTH named with
     byte-equal excess (the honest tie); naming exactly one would be
     false exculpation of the other.  90%-mirror: both named, causal
     one higher -- the innocent correlate IS named (v1 mines marginal
     feature=value signatures; the contract is associational, and
     this probe measures what that costs a causal reader).
  P2 null worlds: per-world alpha ~ 2/301 -> expect 0 of 20 (<= 2
     tolerated) no-signal worlds to name anything.  Hard signal: a
     12-in-600-rare category caps its excess near n_cat/n_fail (~.19)
     while the ONE shared max-null threshold rides the high-variance
     common candidates (~.2) -> hit rate LOW (<= 50%): the price of
     an unstudentized max over mixed-variance candidates.  Easy
     signal (base .02): excess ~.5 >> thr -> hit rate >= 90%.
     [Disclosure: the hard prediction is informed by one API-check
     pilot call (seed-5 world): top .162 vs thr .236, a miss.]
  P3 near-background: excess ~.062 < thr(n=600) ~ .10-.13 -> honest
     nothing = a MISSED genuine cause at n=600; thr(n=6000) ~ .04 <
     .062 -> named at scale.  Equal-rate control: flag never named.
  P4 ~10 failures: deterministic cause (excess 1.0) named in >= 16/20
     worlds despite tiny n; weak cause (excess ~.25 < thr ~.4) ->
     honest nothing in >= 16/20; tiny no-signal worlds: false naming
     ~0 (<= 2/20).
  P5 the engine organ reproduces the reference excesses exactly (same
     worlds, same differential formula) and names the same sets:
     W1 {tense=perfect .496, voice=passive .431}, W2 {word=slang
     .897} alone, control A nothing; thresholds land NEAR .159/.140
     but not byte-equal (the organ draws its own seeded null --
     same law, different rng stream).

Honest refusals (nothing named where nothing is minable) score
HONEST-REFUSAL, not BROKEN.  Misses of PLANTED causes are reported as
missed-naming with the excess-vs-threshold numbers that explain them.

Run: cd /home/user/dc/experiments && PYTHONHASHSEED=0 \
     OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
     python3 commonality_adversarial_probe.py
stdlib + numpy only; engine organ only; fresh throwaway Isaac() minds.
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import isaac as mod                                    # noqa: E402

T0 = time.time()
BAR = "=" * 74


def mine(recs, failed, seed):
    """One fresh throwaway mind per call; name=None -> no store writes."""
    return mod.Isaac().mine_commonality(recs, failed, seed=seed)


def raw_excess(recs, failed, feat, val):
    f = [r for r, x in zip(recs, failed) if x]
    s = [r for r, x in zip(recs, failed) if not x]
    return (sum(1 for r in f if str(r[feat]) == val) / len(f)
            - sum(1 for r in s if str(r[feat]) == val) / len(s))


def fmt_named(named):
    if not named:
        return "[]"
    return "[" + ", ".join(f"{f}={v}:{e:.3f}" for f, v, e in named) + "]"


def sec(title):
    print(BAR)
    print(title)
    print(BAR)


# ----------------------------------------------------------------------
# C1  CORRELATED CONDITIONS
# ----------------------------------------------------------------------
def c1_world(seed, mirror_p, causal):
    rng = np.random.default_rng(seed)
    recs, failed = [], []
    for _ in range(600):
        c = "c1" if rng.random() < 0.30 else "c0"
        t = "t1" if c == "c1" else "t0"
        if rng.random() >= mirror_p:                 # mirror break
            t = "t0" if t == "t1" else "t1"
        recs.append(dict(cause=c, twin=t,
                         junk1="j%d" % rng.integers(4),
                         junk2="k%d" % rng.integers(4),
                         lang="english"))
        p = (0.85 if c == "c1" else 0.06) if causal else 0.10
        failed.append(bool(rng.random() < p))
    return recs, failed


sec("C1 CORRELATED CONDITIONS -- twin features, only 'cause' is causal")

# perfect twin
recs, failed = c1_world(21, 1.0, True)
r = mine(recs, failed, seed=101)
ec = raw_excess(recs, failed, "cause", "c1")
et = raw_excess(recs, failed, "twin", "t1")
got = {(f, v) for f, v, _ in r["named"]}
c1_both = (("cause", "c1") in got) and (("twin", "t1") in got)
c1_one = (("cause", "c1") in got) ^ (("twin", "t1") in got)
c1_tie = abs(ec - et) < 1e-12
c1_junk = any(f in ("junk1", "junk2", "lang") for f, v in got)
print(f"  perfect twin: n_fail={r['n_fail']}  excess cause=c1 {ec:.3f}  "
      f"twin=t1 {et:.3f}  thr {r['threshold']:.3f}")
print(f"    named {fmt_named(r['named'])}")
print(f"    both named: {c1_both}   exactly-one named: {c1_one}   "
      f"excesses byte-equal (honest tie): {c1_tie}   junk/lang named: "
      f"{c1_junk}")

# 90%-mirror twin: does the innocent correlate get named too?
recs, failed = c1_world(22, 0.90, True)
r = mine(recs, failed, seed=102)
ec9 = raw_excess(recs, failed, "cause", "c1")
et9 = raw_excess(recs, failed, "twin", "t1")
got9 = {(f, v) for f, v, _ in r["named"]}
c1_near_cause = ("cause", "c1") in got9
c1_near_twin = ("twin", "t1") in got9
print(f"  90%-mirror:   n_fail={r['n_fail']}  excess cause=c1 {ec9:.3f}  "
      f"twin=t1 {et9:.3f}  thr {r['threshold']:.3f}")
print(f"    named {fmt_named(r['named'])}")
print(f"    causal named: {c1_near_cause}   innocent correlate named: "
      f"{c1_near_twin}  (marginal miner: association, not causation)")

# controls: no-signal twin world + label-shuffle scramble of the causal one
recs_c, failed_c = c1_world(23, 1.0, False)
r_c = mine(recs_c, failed_c, seed=103)
recs, failed = c1_world(21, 1.0, True)
sh = [bool(x) for x in np.random.default_rng(99).permutation(
    np.array(failed))]
r_s = mine(recs, sh, seed=104)
c1_ctrl_clean = (len(r_c["named"]) == 0) and (len(r_s["named"]) == 0)
print(f"  CONTROL no-signal twins: named {fmt_named(r_c['named'])}  "
      f"top {r_c['top']:.3f} vs thr {r_c['threshold']:.3f}")
print(f"  CONTROL scrambled labels of the causal world: named "
      f"{fmt_named(r_s['named'])}  top {r_s['top']:.3f} vs thr "
      f"{r_s['threshold']:.3f}")
print(f"    both controls name nothing: {c1_ctrl_clean}")

# ----------------------------------------------------------------------
# C2  HIGH CARDINALITY + SELECTION PRESSURE
# ----------------------------------------------------------------------
def c2_world(seed, kind):
    rng = np.random.default_rng(seed)
    recs, failed = [], []
    for _ in range(600):
        r = {"bigcat": "c%02d" % rng.integers(50)}
        for j in range(8):
            r["f%d" % j] = "v%d" % rng.integers(4)
        if kind == "null":
            p = 0.10
        elif kind == "hard":
            p = 0.90 if r["bigcat"] == "c17" else 0.08
        else:                                        # easy
            p = 0.90 if r["bigcat"] == "c17" else 0.02
        recs.append(r)
        failed.append(bool(rng.random() < p))
    return recs, failed


sec("C2 HIGH CARDINALITY (50 cats, 82 candidates) + SELECTION PRESSURE")

print("  -- 20 NO-SIGNAL worlds (false-naming must be ~0) --")
c2_false_null = 0
for i in range(20):
    recs, failed = c2_world(1000 + i, "null")
    r = mine(recs, failed, seed=500 + i)
    if r["named"]:
        c2_false_null += 1
    print(f"  [null w{i:02d}] n_fail={r['n_fail']:3d} top={r['top']:.3f} "
          f"thr={r['threshold']:.3f} named={len(r['named'])} "
          f"{fmt_named(r['named']) if r['named'] else ''}")
print(f"  NULL FALSE-NAMING RATE: {c2_false_null}/20 "
      f"(expected ~0.13 worlds at alpha~2/301)")

print("  -- 20 HARD-signal worlds (bigcat=c17 causal .90, base .08) --")
c2_hit_hard, c2_extra_hard = 0, 0
hard_tops, hard_thrs = [], []
for i in range(20):
    recs, failed = c2_world(2000 + i, "hard")
    r = mine(recs, failed, seed=600 + i)
    hit = any(f == "bigcat" and v == "c17" for f, v, _ in r["named"])
    extras = [x for x in r["named"] if not (x[0] == "bigcat"
                                            and x[1] == "c17")]
    c2_hit_hard += hit
    c2_extra_hard += len(extras)
    e17 = raw_excess(recs, failed, "bigcat", "c17")
    hard_tops.append(e17)
    hard_thrs.append(r["threshold"])
    print(f"  [hard w{i:02d}] n_fail={r['n_fail']:3d} c17_excess={e17:.3f} "
          f"top={r['top']:.3f} thr={r['threshold']:.3f} "
          f"hit={'YES' if hit else 'no '} extras={len(extras)}")
print(f"  HARD HIT RATE: {c2_hit_hard}/20  MISSED-NAMING: "
      f"{20 - c2_hit_hard}/20  extra namings: {c2_extra_hard}")
print(f"  hard mean c17 excess {np.mean(hard_tops):.3f} vs mean thr "
      f"{np.mean(hard_thrs):.3f}")

print("  -- 20 EASY-signal worlds (bigcat=c17 causal .90, base .02) --")
c2_hit_easy, c2_extra_easy = 0, 0
easy_tops, easy_thrs = [], []
for i in range(20):
    recs, failed = c2_world(3000 + i, "easy")
    r = mine(recs, failed, seed=700 + i)
    hit = any(f == "bigcat" and v == "c17" for f, v, _ in r["named"])
    extras = [x for x in r["named"] if not (x[0] == "bigcat"
                                            and x[1] == "c17")]
    c2_hit_easy += hit
    c2_extra_easy += len(extras)
    e17 = raw_excess(recs, failed, "bigcat", "c17")
    easy_tops.append(e17)
    easy_thrs.append(r["threshold"])
    print(f"  [easy w{i:02d}] n_fail={r['n_fail']:3d} c17_excess={e17:.3f} "
          f"top={r['top']:.3f} thr={r['threshold']:.3f} "
          f"hit={'YES' if hit else 'no '} extras={len(extras)}")
print(f"  EASY HIT RATE: {c2_hit_easy}/20  MISSED-NAMING: "
      f"{20 - c2_hit_easy}/20  extra namings: {c2_extra_easy}")
print(f"  easy mean c17 excess {np.mean(easy_tops):.3f} vs mean thr "
      f"{np.mean(easy_thrs):.3f}")

# ----------------------------------------------------------------------
# C3  NEAR-BACKGROUND
# ----------------------------------------------------------------------
def c3_world(seed, n, causal):
    rng = np.random.default_rng(seed)
    recs, failed = [], []
    for _ in range(n):
        flag = "on" if rng.random() < 0.95 else "off"
        recs.append(dict(flag=flag,
                         j1="a%d" % rng.integers(4),
                         j2="b%d" % rng.integers(4),
                         j3="c%d" % rng.integers(4),
                         lang="english"))
        p = (0.20 if flag == "on" else 0.0) if causal else 0.19
        failed.append(bool(rng.random() < p))
    return recs, failed


sec("C3 NEAR-BACKGROUND -- flag on 100% of failures, ~95% of successes")

recs, failed = c3_world(31, 600, True)
r600 = mine(recs, failed, seed=301)
e600 = raw_excess(recs, failed, "flag", "on")
c3_named_600 = any(f == "flag" and v == "on" for f, v, _ in r600["named"])
print(f"  n= 600: n_fail={r600['n_fail']:4d} flag=on excess {e600:.3f} "
      f"thr {r600['threshold']:.3f} named {fmt_named(r600['named'])}")
print(f"    flag=on named at n=600: {c3_named_600}")

recs, failed = c3_world(32, 6000, True)
r6k = mine(recs, failed, seed=302)
e6k = raw_excess(recs, failed, "flag", "on")
c3_named_6k = any(f == "flag" and v == "on" for f, v, _ in r6k["named"])
print(f"  n=6000: n_fail={r6k['n_fail']:4d} flag=on excess {e6k:.3f} "
      f"thr {r6k['threshold']:.3f} named {fmt_named(r6k['named'])}")
print(f"    flag=on named at n=6000: {c3_named_6k}")

recs, failed = c3_world(33, 600, False)
rc3 = mine(recs, failed, seed=303)
ec3 = raw_excess(recs, failed, "flag", "on")
c3_ctrl_clean = not any(f == "flag" for f, v, _ in rc3["named"])
print(f"  CONTROL equal-rate 95% background: flag=on excess {ec3:.3f} "
      f"thr {rc3['threshold']:.3f} named {fmt_named(rc3['named'])}")
print(f"    flag never named on control: {c3_ctrl_clean} "
      f"(differential, never a frequency)")

# ----------------------------------------------------------------------
# C4  TINY SAMPLES -- ~10 failures per world
# ----------------------------------------------------------------------
def c4_world(seed, kind):
    rng = np.random.default_rng(seed)
    n_c1 = 30 if kind == "weak" else 10
    causes = ["c1"] * n_c1 + ["c0"] * (100 - n_c1)
    order = rng.permutation(100)
    causes = [causes[int(i)] for i in order]
    recs = [dict(cause=c,
                 j1="a%d" % rng.integers(4),
                 j2="b%d" % rng.integers(4),
                 j3="c%d" % rng.integers(4),
                 lang="english") for c in causes]
    if kind == "strong":
        failed = [c == "c1" for c in causes]         # exactly 10 failures
    elif kind == "weak":
        failed = [bool(rng.random() < (0.20 if c == "c1" else 0.08))
                  for c in causes]
    else:                                            # null: exactly 10
        idx = {int(x) for x in rng.choice(100, size=10, replace=False)}
        failed = [i in idx for i in range(100)]
    return recs, failed


sec("C4 TINY SAMPLES -- ~10 failures / 100 attempts, 20 worlds each")

c4_hit_strong, c4_extra_strong = 0, 0
for i in range(20):
    recs, failed = c4_world(4000 + i, "strong")
    r = mine(recs, failed, seed=800 + i)
    hit = any(f == "cause" and v == "c1" for f, v, _ in r["named"])
    extras = [x for x in r["named"] if not (x[0] == "cause"
                                            and x[1] == "c1")]
    c4_hit_strong += hit
    c4_extra_strong += len(extras)
    if i < 3 or not hit or extras:
        print(f"  [strong w{i:02d}] n_fail={r['n_fail']:2d} "
              f"top={r['top']:.3f} thr={r['threshold']:.3f} "
              f"hit={'YES' if hit else 'no'} extras={len(extras)}")
print(f"  STRONG (deterministic cause) HIT RATE: {c4_hit_strong}/20  "
      f"extra namings: {c4_extra_strong}")

c4_named_weak, c4_false_weak = 0, 0
for i in range(20):
    recs, failed = c4_world(4100 + i, "weak")
    r = mine(recs, failed, seed=850 + i)
    hit = any(f == "cause" and v == "c1" for f, v, _ in r["named"])
    extras = [x for x in r["named"] if not (x[0] == "cause"
                                            and x[1] == "c1")]
    c4_named_weak += hit
    c4_false_weak += len(extras)
    if i < 3 or extras:
        print(f"  [weak w{i:02d}]  n_fail={r['n_fail']:2d} "
              f"cause_excess={raw_excess(recs, failed, 'cause', 'c1'):.3f} "
              f"thr={r['threshold']:.3f} named={len(r['named'])}")
print(f"  WEAK cause named: {c4_named_weak}/20 (honest nothing "
      f"{20 - c4_named_weak}/20)  false extras: {c4_false_weak}")

c4_false_null = 0
for i in range(20):
    recs, failed = c4_world(4200 + i, "null")
    r = mine(recs, failed, seed=900 + i)
    if r["named"]:
        c4_false_null += 1
        print(f"  [null w{i:02d}]  FALSE NAMING {fmt_named(r['named'])} "
              f"thr={r['threshold']:.3f}")
print(f"  TINY NULL FALSE-NAMING RATE: {c4_false_null}/20")

# ----------------------------------------------------------------------
# C5  CONTROL -- the 07-14 reference world through the ENGINE organ
#     (rng consumption replicated verbatim from
#      failure_commonality_probe.py so the worlds are byte-identical;
#      the reference probe predates the organ)
# ----------------------------------------------------------------------
sec("C5 CONTROL -- 07-14 reference world through mine_commonality")

REF_FEATS = {
    "voice":   ["active", "passive"],
    "tense":   ["present", "past", "perfect"],
    "length":  ["short", "long"],
    "pronoun": ["none", "i", "you", "he"],
    "junk":    ["j1", "j2", "j3", "j4"],
    "lang":    ["english"],
}
ref_rng = np.random.default_rng(11)


def ref_sample(n):
    return [{k: ref_rng.choice(v) for k, v in REF_FEATS.items()}
            for _ in range(n)]


def ref_label(recs):
    out = []
    for r in recs:
        p = 0.90 if (r["voice"] == "passive"
                     and r["tense"] == "perfect") else 0.08
        out.append(ref_rng.random() < p)
    return out


def ref_burn(failed):
    """Consume exactly what the reference mine() consumed (300 perms)."""
    fl = np.array(failed)
    for _ in range(300):
        ref_rng.permutation(fl)


# W1 planted frontier
recs1 = ref_sample(600)
failed1 = ref_label(recs1)
rw1 = mine(recs1, failed1, seed=0)
ref_burn(failed1)
g1 = {(f, v): e for f, v, e in rw1["named"]}
w1_set_ok = set(g1) == {("tense", "perfect"), ("voice", "passive")}
w1_num_ok = (abs(g1.get(("tense", "perfect"), 9) - 0.496) < 0.005
             and abs(g1.get(("voice", "passive"), 9) - 0.431) < 0.005)
print(f"  W1 frontier: n_fail={rw1['n_fail']} named "
      f"{fmt_named(rw1['named'])}")
print(f"    thr {rw1['threshold']:.3f} (reference null99 .159, own rng)")
print(f"    named set == reference set: {w1_set_ok}   excesses match "
      f".496/.431 (+-.005): {w1_num_ok}")

# W2 owner lockout
recs2 = ref_sample(600)
for rr in recs2:
    rr["word"] = ref_rng.choice(["fine", "fine", "fine", "slang"])
failed2 = [(rr["word"] == "slang") or (ref_rng.random() < 0.03)
           for rr in recs2]
rw2 = mine(recs2, failed2, seed=0)
ref_burn(failed2)
w2_ok = (len(rw2["named"]) == 1 and rw2["named"][0][0] == "word"
         and rw2["named"][0][1] == "slang"
         and abs(rw2["named"][0][2] - 0.897) < 0.005)
print(f"  W2 lockout:  n_fail={rw2['n_fail']} named "
      f"{fmt_named(rw2['named'])}")
print(f"    thr {rw2['threshold']:.3f} (reference null99 .140, own rng)")
print(f"    word=slang named ALONE at .897 (+-.005): {w2_ok}")

# control A: random failures -- nothing
recs3 = ref_sample(600)
failed3 = [bool(ref_rng.random() < 0.15) for _ in recs3]
rw3 = mine(recs3, failed3, seed=0)
ctrlA_ok = len(rw3["named"]) == 0
print(f"  ctrl A random: n_fail={rw3['n_fail']} named "
      f"{fmt_named(rw3['named'])}  top {rw3['top']:.3f} vs thr "
      f"{rw3['threshold']:.3f}  nothing named: {ctrlA_ok}")

# ----------------------------------------------------------------------
# FINDINGS -- verdicts suggested by the numbers above
# ----------------------------------------------------------------------
sec("FINDINGS (misses are reported, never patched)")

if c1_both and c1_tie and not c1_junk and c1_ctrl_clean:
    v1 = ("HONEST TIE: both twins named with byte-equal excess; "
          "controls clean")
elif c1_one:
    v1 = "FALSE EXCULPATION: exactly one twin named"
else:
    v1 = "IRREGULAR: see numbers above"
print(f"  C1 perfect twins -> {v1}")
print(f"     90%-mirror: causal named {c1_near_cause}, innocent "
      f"correlate named {c1_near_twin} (excess {ec9:.3f} vs {et9:.3f}) "
      f"-- marginal miner names association, not causation")

cal = ("CALIBRATED" if c2_false_null <= 2 else "MISCALIBRATED")
print(f"  C2 selection correction: {c2_false_null}/20 no-signal worlds "
      f"named anything -> {cal} (alpha~2/301 predicts ~0.13)")
print(f"  C2 power, rare category: hard hit {c2_hit_hard}/20 "
      f"(missed {20 - c2_hit_hard}/20; mean excess "
      f"{np.mean(hard_tops):.3f} vs mean thr {np.mean(hard_thrs):.3f}); "
      f"easy hit {c2_hit_easy}/20 (mean excess {np.mean(easy_tops):.3f} "
      f"vs thr {np.mean(easy_thrs):.3f}); extras hard/easy "
      f"{c2_extra_hard}/{c2_extra_easy}")

print(f"  C3 near-background cause: n=600 named={c3_named_600} "
      f"(excess {e600:.3f} vs thr {r600['threshold']:.3f}); n=6000 "
      f"named={c3_named_6k} (excess {e6k:.3f} vs thr "
      f"{r6k['threshold']:.3f}); equal-rate control clean="
      f"{c3_ctrl_clean}")

print(f"  C4 tiny samples: deterministic cause hit {c4_hit_strong}/20 "
      f"(extras {c4_extra_strong}); weak cause named {c4_named_weak}/20 "
      f"(false extras {c4_false_weak}); null false naming "
      f"{c4_false_null}/20")

print(f"  C5 reference reproduction through the engine organ: "
      f"W1 set {w1_set_ok} nums {w1_num_ok}; W2 {w2_ok}; "
      f"ctrlA nothing {ctrlA_ok}")

print(BAR)
print(f"elapsed {time.time() - T0:.1f}s")

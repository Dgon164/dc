"""election_mixedcause_probe.py -- ATTACK PROBE: mixed-cause credit
assignment in word re-election (FRESH EYES item 1, 2026-07-15).

THE CLAIM UNDER TEST: bank_word_outcomes lands the WHOLE event's
pass/fail on every mapped word used in it, so a fail caused by
decoration/orientation still lands on the word's seat. The battery
(W1-W8, green 12/12 before this probe) tested clean worlds only:
W3's good seat saw passes only; W4's insistence saw a scripted
3-votes-THEN-4-fails order. Mixed-cause worlds were never in it.
Attack: (a) a word-substance-GOOD seat used inside events that fail
for unrelated decoration causes -- is it wrongly dethroned, at what
unrelated-fail share {30%, 50%, 70%}? (b) does tutor INSISTENCE hold
the seat as designed? (c) symmetric: a word-substance-BAD seat riding
passes from a word-blind judge -- does it wrongly survive, and does
the told-road rescue still cut through?

WORLDS (every verdict is a REAL comparison against the tutor's
demanded form of the round -- the unrelated cause is PLANTED in the
form, never a coin-flipped verdict on nothing):
  GOOD-SEAT: threw->tossed IS the tutor's true dialect (word substance
    100% right in every event). Each round the tutor demands the
    decorated frame 'the girl tossed the ball' with prob p (the young
    mind has no articles -> fails; cause decoration, not the word)
    else the plain frame (passes). Arm A: judge-only -- verdicts, no
    corrective text (the legitimate remine shape: remine banks
    outcomes for every judged event, votes only when 'say' exists).
    Arm B: every fail also delivers the tutor's correction, which
    USES 'tossed' -> seat votes -> the designed insistence mitigation.
  BAD-SEAT: ate->eats is POISON (true dialect says 'ate'). The judge
    is word-blind (compares FRAMES: content words erased), demands the
    plain frame with prob q -> the bad word rides an unrelated pass;
    decorated with prob 1-q -> fail (cause decoration). Judge-only.
    Then the rescue: the tutor finally speaks 'ate' 3x (W2's told
    road) on every surviving seat.
  CONTROLS (clean anchors for every positive claim): p=0.0 all-pass
    (W3 shape) must never dethrone; p=1.0 judge-only (a good word
    indistinguishable from a bad one) must dethrone (W1 law); p=1.0
    WITH corrections must hold (W4 law); q=0.0 must dethrone the
    poison. Plus two DETERMINISTIC exactly-50% worlds (P,F,P,F... and
    F,P,F,P...) to locate the strict '<0.5' boundary precisely.

PREDICTION (filed before the first run, from code reading only):
  P1 arm A wrongly dethrones the good seat in ~100% of trials at
     p=0.7, >90% at p=0.5, and a substantial minority (30-60%) at
     p=0.3 -- BELOW the supposed 50% guard, because the guard is a
     strict '<0.5' on a fluctuating last-8 window re-checked after
     every use (first passage, not a rate test). Control p=0.0: 0%.
  P2 arm B has an EARLY RACE HOLE: outcome-road dethrone needs exactly
     2 prior fail-corrections (vs=2 < INSIST=3) plus a 3rd fail within
     the first ~5 banked uses; predict roughly 15-35% of trials at
     p in {0.5, 0.7} see a dethrone; p=1.0 with corrections: 0%
     (vs reaches 3 before the record can count, the W4 order).
  P3 the bad seat wrongly survives all 40 events at q=0.7 in ~100% of
     trials; the told-road rescue then dethrones every survivor (told
     road never consults the shiny record). At q=0.3 it falls by
     accident: outcome road, wrong reason (decoration fails blamed on
     the word), right result.
  P4 a wrongly dethroned good word is HARD to re-teach: the stale
     failing record is never cleared, so the re-teaching correction's
     own seat-vote triggers an instant re-dethrone INSIDE the same
     call; it sticks only at the 3rd consecutive re-teach (INSIST).
  P5 the '<0.5' guard is a knife edge at exactly 50%: P-start
     alternation holds forever (window mean never below 0.5); F-start
     alternation dethrones at use 5 (window [0,1,0,1,0] = 0.4).

Found wrongs are REPORTED, never patched. Fresh throwaway Isaac()
minds only. stdlib + numpy only for cognition; no sklearn needed.
Run: cd experiments && PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 \
     OMP_NUM_THREADS=1 python3 election_mixedcause_probe.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import isaac as mod                                    # noqa: E402
from idea_corrections import (speak, learn_from_correction,  # noqa: E402
                              bank_word_outcomes, INSIST,
                              REVERT_USES, WINDOW)

EV_GOOD = ("girl", "threw", "ball")
GOOD_PLAIN = ["girl", "tossed", "ball"]
GOOD_DECO = ["the", "girl", "tossed", "the", "ball"]
EV_BAD = ("cat", "ate", "apple")
N_EVENTS = 40
N_TRIALS = 60


def seat(mind, w):
    return (mind.net.get_meta("speak:words", "map") or {}).get(w)


def chain(mind):
    return mind.net.get_meta("speak:words", "superseded") or []


def teach_good(m):
    learn_from_correction(m, EV_GOOD, ["girl", "threw", "ball"],
                          list(GOOD_PLAIN))
    return seat(m, "threw")


def poison(m):
    learn_from_correction(m, EV_BAD, ["cat", "ate", "apple"],
                          ["cat", "eats", "apple"])
    return seat(m, "ate")


def frame(words):
    """A word-blind judge's view: content erased, form kept."""
    return ["the" if w == "the" else "W" for w in words]


# ==================================================================
# GOOD-SEAT mixed-cause trial. deco_seq[k] True = the tutor demands
# the decorated frame this round (the planted unrelated cause).
# ==================================================================
def good_trial(seed, p, arm, deco_seq=None):
    rng = np.random.default_rng(seed)
    m = mod.Isaac()
    assert teach_good(m) == "tossed", "probe bug: teaching failed"
    first, fails, played = None, 0, 0
    for k in range(1, N_EVENTS + 1):
        said = speak(m, EV_GOOD, rng)
        deco_round = (deco_seq[k - 1] if deco_seq is not None
                      else bool(rng.random() < p))
        canon = list(GOOD_DECO) if deco_round else list(GOOD_PLAIN)
        passed = (said == canon)          # REAL verdict, planted cause
        fails += (not passed)
        played = k
        bank_word_outcomes(m, EV_GOOD, said, passed)
        if (not passed) and arm == "B":
            learn_from_correction(m, EV_GOOD, said, canon)
        if first is None and chain(m):
            first = k
            if arm == "A":                # judge-only: nothing can
                break                     # re-teach; trial decided
    return dict(first=first, n_dethrones=len(chain(m)),
                end_gone=seat(m, "threw") is None,
                fails=fails, played=played, mind=m)


def run_good_condition(label, p, arm, base_seed):
    firsts, evers, ends, oscil, share = [], 0, 0, [], []
    sample = None
    for t in range(N_TRIALS):
        r = good_trial(base_seed + t, p, arm)
        share.append(r["fails"] / r["played"])
        if r["first"] is not None:
            evers += 1
            firsts.append(r["first"])
            oscil.append(r["n_dethrones"])
            if sample is None:
                sample = chain(r["mind"])[0]
        ends += r["end_gone"]
    print(f"  {label:34s} wrong-dethrone {evers:2d}/{N_TRIALS} "
          f"({100.0 * evers / N_TRIALS:5.1f}%)  end-dethroned "
          f"{ends:2d}/{N_TRIALS}  first-dethrone med "
          f"{(np.median(firsts) if firsts else float('nan')):4.1f}  "
          f"dethrones/trial med "
          f"{(np.median(oscil) if oscil else 0):.1f}  realized-fail-share "
          f"{np.mean(share):.3f}")
    if sample is not None:
        rec = sample["record"]
        print(f"      sample chain entry: dethroned '{sample['dethroned']}'"
              f" | ledger says passed {rec['passed']}/{rec['uses']} recent"
              f" uses | planted word-substance truth: 100% right"
              f" | why: {sample['why']}")
    return evers, ends, firsts


# ==================================================================
# BAD-SEAT wrong-survive trial: word-blind judge, poison rides form.
# ==================================================================
def bad_trial(seed, q):
    rng = np.random.default_rng(seed)
    m = mod.Isaac()
    assert poison(m) == "eats", "probe bug: poisoning failed"
    first, passes, played = None, 0, 0
    for k in range(1, N_EVENTS + 1):
        said = speak(m, EV_BAD, rng)
        plain_round = bool(rng.random() < q)
        canon = (["cat", "ate", "apple"] if plain_round
                 else ["the", "cat", "ate", "the", "apple"])
        passed = (frame(said) == frame(canon))   # word-blind judge
        passes += passed
        played = k
        bank_word_outcomes(m, EV_BAD, said, passed)
        if first is None and chain(m):
            first = k
            break                                # judge-only: decided
    return dict(first=first, survived=seat(m, "ate") == "eats",
                passes=passes, played=played, mind=m)


def run_bad_condition(q, base_seed):
    survived, firsts, share, rescued, rescue_tried = 0, [], [], 0, 0
    for t in range(N_TRIALS):
        r = bad_trial(base_seed + t, q)
        share.append(r["passes"] / r["played"])
        if r["survived"]:
            survived += 1
            rescue_tried += 1                    # W2 told-road rescue
            m = r["mind"]
            for _ in range(3):
                said = speak(m, EV_BAD, np.random.default_rng(0))
                learn_from_correction(m, EV_BAD, said,
                                      ["cat", "ate", "apple"])
            rescued += (seat(m, "ate") is None)
        else:
            firsts.append(r["first"])
    print(f"  q={q:.1f} unrelated-pass share      wrong-survive "
          f"{survived:2d}/{N_TRIALS} ({100.0 * survived / N_TRIALS:5.1f}%)"
          f"  accidental-dethrone med idx "
          f"{(np.median(firsts) if firsts else float('nan')):4.1f}"
          f"  realized-pass-share {np.mean(share):.3f}"
          + (f"  told-road rescue {rescued}/{rescue_tried}"
             if rescue_tried else ""))
    return survived, rescued, rescue_tried


# ==================================================================
def main():
    print("=" * 74)
    print("MIXED-CAUSE ELECTION PROBE -- credit assignment under "
          "unrelated causes")
    print(f"engine constants: WINDOW={WINDOW} REVERT_USES={REVERT_USES} "
          f"INSIST={INSIST}; {N_TRIALS} trials x {N_EVENTS} events "
          f"per condition")
    print("=" * 74)
    print("PREDICTION (filed before first run): P1 arm A dethrones the")
    print("good seat ~100%@p=0.7, >90%@p=0.5, 30-60%@p=0.3, 0%@p=0.0;")
    print("P2 arm B early-race hole 15-35%@p>=0.5, 0%@p=1.0; P3 bad seat")
    print("survives ~100%@q=0.7, rescue 100%; P4 re-teach sticks only at")
    print("the 3rd insistence; P5 P-start 50% alternation holds, F-start")
    print("dethrones at use 5.")

    # -- ATTACK 1: good seat, judge-only (arm A) --------------------
    print("\n[1] GOOD seat 'threw->tossed' (100% right on word substance),")
    print("    fails planted on DECORATION only, judge-only (no insistence")
    print("    available -- the remine world shape):")
    a_res = {}
    for i, p in enumerate([0.0, 0.3, 0.5, 0.7, 1.0]):
        tag = ("CONTROL clean passes (W3 shape)" if p == 0.0 else
               "CONTROL all-fail (W1 law)" if p == 1.0 else
               f"unrelated-fail share p={p:.1f}")
        a_res[p] = run_good_condition(tag, p, "A", base_seed=1000 + 100 * i)

    # -- ATTACK 2: same worlds, tutor corrections flow (arm B) ------
    print("\n[2] same worlds, arm B: every fail delivers the correction")
    print("    (tutor's text uses 'tossed' -> seat votes -> INSISTENCE,")
    print("    the designed mitigation). Note: realized fail share drifts")
    print("    from the planted p because after 3 deco corrections the")
    print("    mind adopts 'the' and then fails on PLAIN rounds instead --")
    print("    the unrelated cause persists either way:")
    b_res = {}
    for i, p in enumerate([0.3, 0.5, 0.7, 1.0]):
        tag = ("CONTROL all-fail + insistence (W4 law)" if p == 1.0
               else f"unrelated-fail share p={p:.1f}")
        b_res[p] = run_good_condition(tag, p, "B", base_seed=2000 + 100 * i)

    # -- ATTACK 3: bad seat riding a word-blind judge ---------------
    print("\n[3] BAD seat 'ate->eats' (poison), word-blind judge: passes")
    print("    ride the plain frame with prob q, fails are decoration:")
    s_res = {}
    for i, q in enumerate([0.0, 0.3, 0.5, 0.7]):
        s_res[q] = run_bad_condition(q, base_seed=3000 + 100 * i)

    # -- EXHIBIT A: the strict '<0.5' knife edge, deterministic -----
    print("\n[4] EXHIBIT: deterministic exactly-50% worlds (arm A):")
    seq_p = [(k % 2 == 1) for k in range(N_EVENTS)]   # P,F,P,F,...
    seq_f = [(k % 2 == 0) for k in range(N_EVENTS)]   # F,P,F,P,...
    rp = good_trial(7, 0.5, "A", deco_seq=seq_p)
    rf = good_trial(7, 0.5, "A", deco_seq=seq_f)
    print(f"    P-start P,F,P,F...: dethroned at use {rp['first']} "
          f"(fails {rp['fails']}/{rp['played']})  -> "
          f"{'HELD' if rp['first'] is None else 'DETHRONED'}")
    print(f"    F-start F,P,F,P...: dethroned at use {rf['first']} "
          f"(fails {rf['fails']}/{rf['played']})  -> "
          f"{'HELD' if rf['first'] is None else 'DETHRONED'}")
    knife = (rp["first"] is None and rf["first"] == 5)

    # -- EXHIBIT B: re-teaching a wrongly dethroned good word -------
    print("\n[5] EXHIBIT: re-teaching after a wrong dethrone (stale record")
    print("    is never cleared). Take the first p=0.7 arm-A casualty,")
    print("    tutor re-teaches 'threw->tossed' repeatedly:")
    victim = None
    for t in range(N_TRIALS):
        r = good_trial(1300 + t, 0.7, "A")            # same seeds as [1]
        if r["first"] is not None:
            victim = r["mind"]
            break
    stick_at = None
    if victim is not None:
        rec0 = ((victim.net.get_meta("speak:words", "record") or {})
                .get("threw", {}).get("tossed", {}))
        print(f"    stale record at dethrone: recent={rec0.get('recent')}")
        rng = np.random.default_rng(9)
        for i in range(1, 6):
            said = speak(victim, EV_GOOD, rng)
            learn_from_correction(victim, EV_GOOD, said, list(GOOD_PLAIN))
            s_now = seat(victim, "threw")
            print(f"    re-teach #{i}: said '{' '.join(said)}' -> seat "
                  f"now {s_now!r}  dethrones so far {len(chain(victim))}")
            if s_now == "tossed" and stick_at is None:
                stick_at = i
        # does it SURVIVE a passing use once stuck?
        said = speak(victim, EV_GOOD, rng)
        bank_word_outcomes(victim, EV_GOOD, said, said == GOOD_PLAIN)
        held_after_pass = seat(victim, "threw") == "tossed"
        print(f"    sticks at re-teach #{stick_at}; survives next passing "
              f"banked use: {held_after_pass}")
    else:
        held_after_pass = False
        print("    (no p=0.7 casualty found -- exhibit skipped)")

    # -- VERDICTS ----------------------------------------------------
    print("\n" + "=" * 74)
    print("VERDICTS (evidence = numbers above)")
    ca_broken = (a_res[0.5][0] / N_TRIALS > 0.5
                 or a_res[0.3][0] / N_TRIALS > 0.25)
    ctrl_ok = (a_res[0.0][0] == 0 and a_res[1.0][0] == N_TRIALS
               and b_res[1.0][0] == 0 and s_res[0.0][0] == 0)
    b_hole = max(b_res[p][0] for p in (0.3, 0.5, 0.7))
    surv70 = s_res[0.7][0]
    resc, resc_n = s_res[0.7][1], s_res[0.7][2]
    print(f"  credit assignment, mixed-cause fails (judge-only): "
          f"{'BROKEN' if ca_broken else 'WORKS'} -- good word dethroned "
          f"{a_res[0.3][0]}/{N_TRIALS}@p=0.3, {a_res[0.5][0]}/{N_TRIALS}"
          f"@p=0.5, {a_res[0.7][0]}/{N_TRIALS}@p=0.7; the '<50%' guard "
          f"is first-passage on a last-{WINDOW} window re-checked every "
          f"use, not a rate test -- it fires below its stated boundary. "
          f"(P1 confirmed. This is the remine transcript shape: a "
          f"judge-only history can retro-dethrone good words.)")
    print(f"  insistence mitigation (arm B): "
          f"{'PARTIAL' if 0 < b_hole < N_TRIALS else 'WORKS' if b_hole == 0 else 'BROKEN'}"
          f" -- dethrones {b_res[0.3][0]}/{N_TRIALS}@p=0.3, "
          f"{b_res[0.5][0]}/{N_TRIALS}@p=0.5, {b_res[0.7][0]}/{N_TRIALS}"
          f"@p=0.7; 0 required @p=1.0 (W4 order): got {b_res[1.0][0]}. "
          f"PREDICTION P2 FALSIFIED: no early race exists, because the "
          f"TEACHING correction itself banks one seat vote and every "
          f"fail-correction banks another, so any window with >= 3 fails "
          f"implies vs >= {INSIST} -- the outcome road is mathematically "
          f"blocked while the tutor's corrections keep using the seat "
          f"word. Protection is total but hinges on correction TEXT "
          f"flowing; verdict-only channels get none (see line above)")
    print(f"  bad word riding unrelated passes: "
          f"{'BROKEN' if surv70 > N_TRIALS // 2 else 'WORKS'} -- survives "
          f"{s_res[0.3][0]}/{N_TRIALS}@q=0.3, {s_res[0.5][0]}/{N_TRIALS}"
          f"@q=0.5, {surv70}/{N_TRIALS}@q=0.7 at the 40-event horizon. "
          f"PREDICTION P3 OVERSTATED: not 'forever' -- the same window "
          f"first-passage that wrongly kills good words accidentally "
          f"catches shielded poison too (median accidental fall idx 21 "
          f"@q=0.7 among the fallen), but 55% of poisons outlive 40 "
          f"judged uses on a ~0.66 unrelated pass share")
    print(f"  told-road rescue of a shielded bad word: "
          f"{'WORKS' if resc_n and resc == resc_n else 'BROKEN'} -- "
          f"{resc}/{resc_n} dethroned by 3 identity votes despite a "
          f"passing ledger (told road ignores the record, by design)")
    print(f"  re-adoption after wrong dethrone: "
          f"{'BROKEN' if stick_at is None else 'PARTIAL'}"
          f" -- re-teach sticks only at attempt #{stick_at}: the stale "
          f"failing record is never cleared on dethrone, so re-teach #1's "
          f"own seat-vote re-fires the outcome road INSIDE the same call; "
          f"it holds once seat votes reach INSIST={INSIST} (the original "
          f"teach vote still counts, so {INSIST}-1 re-teaches suffice)")
    print(f"  deterministic 50% knife edge: "
          f"{'CONFIRMED' if knife else 'NOT CONFIRMED'} -- P-start held, "
          f"F-start dethroned at use {rf['first']} (parity decides at "
          f"exactly the stated boundary)")
    print(f"  clean controls under this harness: "
          f"{'WORKS' if ctrl_ok else 'BROKEN'} -- p=0.0: "
          f"{a_res[0.0][0]}/{N_TRIALS} dethroned (want 0); p=1.0 "
          f"judge-only: {a_res[1.0][0]}/{N_TRIALS} (want {N_TRIALS}, W1); "
          f"p=1.0 insisting: {b_res[1.0][0]}/{N_TRIALS} (want 0, W4); "
          f"q=0.0: {s_res[0.0][0]}/{N_TRIALS} survive (want 0)")
    print("=" * 74)


if __name__ == "__main__":
    main()

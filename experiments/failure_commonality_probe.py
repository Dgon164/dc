"""
FAILURE COMMONALITY PROBE -- reference toy, NOT the engine.
Question: from kept failure receipts, can the COMMON CAUSE be named --
measured as a DIFFERENTIAL against successes (compare(): only-in-failures),
gated by a selection-corrected label-shuffle null?

Planted world: utterances with features. Failures concentrate on
voice=passive AND tense=perfect (the frontier pile, planted). A jittering
junk feature varies freely. Also a DENIED-WORD world: an owner lockout
('slang': banned) -- denials must name the word, nothing else.
Controls: random failures mine NOTHING; a feature common to failures AND
successes (the shared background) must not be named.

(Transcribed to dc/experiments 2026-07-15 from the 07-14 session's paste;
punctuation re-encoded to ASCII, code verbatim.)
"""
import numpy as np
rng = np.random.default_rng(11)
NPERM = 300

FEATS = {
    "voice":   ["active", "passive"],
    "tense":   ["present", "past", "perfect"],
    "length":  ["short", "long"],
    "pronoun": ["none", "i", "you", "he"],
    "junk":    ["j1", "j2", "j3", "j4"],      # jitter -- must never be named
    "lang":    ["english"],                    # shared background -- must never be named
}

def sample(n):
    return [{k: rng.choice(v) for k, v in FEATS.items()} for _ in range(n)]

def label_frontier(recs):
    """Planted truth: passive+perfect fails 90%; everything else fails 8%."""
    out = []
    for r in recs:
        p = 0.90 if (r["voice"] == "passive" and r["tense"] == "perfect") else 0.08
        out.append(rng.random() < p)
    return out

def mine(recs, failed):
    """Differential signature: for every feature=value, excess of failure
    rate over success rate, gated by max-over-candidates label-shuffle null."""
    fails = [r for r, f in zip(recs, failed) if f]
    succs = [r for r, f in zip(recs, failed) if not f]
    if not fails or not succs: return [], 0.0, 0.0
    cands = [(k, v) for k, vs in FEATS.items() for v in vs]
    def rate(pool, k, v):
        return sum(1 for r in pool if r[k] == v) / len(pool)
    def excess_all(fl):
        f = [r for r, x in zip(recs, fl) if x]; s = [r for r, x in zip(recs, fl) if not x]
        return {c: rate(f, *c) - rate(s, *c) for c in cands}
    obs = excess_all(failed)
    null = np.zeros(NPERM)
    fl = np.array(failed)
    for p in range(NPERM):
        sh = rng.permutation(fl)
        null[p] = max(excess_all(list(sh)).values())
    thr = np.sort(null)[::-1][1]              # 2nd largest of 300 ~ 99th pct
    named = [(c, e) for c, e in sorted(obs.items(), key=lambda x: -x[1]) if e > thr]
    return named, thr, max(obs.values())

def report(title, recs, failed):
    named, thr, top = mine(recs, failed)
    nf = sum(failed)
    print("=" * 74); print(f"{title}   ({nf} failures / {len(recs)} attempts)")
    if not named:
        print(f"  NOTHING NAMED (top excess {top:.3f} vs null99 {thr:.3f}) -- honest refusal")
    for (k, v), e in named:
        print(f"  named: {k}={v}   excess {e:.3f} vs null99 {thr:.3f}")

def run():
    # world 1: the planted frontier (passive+perfect)
    recs = sample(600); failed = label_frontier(recs)
    report("W1 planted frontier: failures concentrate on passive+perfect", recs, failed)

    # world 2: owner lockout -- a denied word; denials must name it and only it
    recs = sample(600)
    for r in recs: r["word"] = rng.choice(["fine", "fine", "fine", "slang"])
    FEATS["word"] = ["fine", "slang"]
    failed = [(r["word"] == "slang") or (rng.random() < 0.03) for r in recs]
    report("W2 owner lockout: 'slang' denied outright", recs, failed)
    del FEATS["word"]
    for r in recs: del r["word"]

    # control A: random failures -- nothing may be named
    recs = sample(600); failed = [rng.random() < 0.15 for _ in recs]
    report("CONTROL A random failures", recs, failed)

    # control B: background feature (lang=english on every record, pass or fail)
    # is common to ALL failures -- 100% of them -- and must NOT be named.
    recs = sample(600); failed = label_frontier(recs)
    shared = all(r["lang"] == "english" for r in recs)
    named, _, _ = mine(recs, failed)
    bg_named = any(k == "lang" for (k, v), e in named)
    print("=" * 74)
    print(f"CONTROL B shared background: lang=english on 100% of failures "
          f"(and successes) -- named: {bg_named} (expect False; "
          f"commonality is a DIFFERENTIAL, not a frequency)")

if __name__ == "__main__":
    run()

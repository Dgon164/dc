"""gate6_corruption_probe.py — GATE 6 ATTACK: near-miss corruption +
noise rates that break the consensus trim (07-15 open item 1).

THE CLAIM UNDER TEST: gate6_operator_mining.py wrong-kept 5/5 planted
corrupted episodes and still minted the rule — but its corruption was
FAR garbage (30..90 against sums 2..18) at a benign ~11% rate, and the
iterated-trim consensus that did the wrong-keeping is A PROBE MECHANISM,
NOT AN ORGAN. This probe attacks that mechanism with two corruption
species at rates {0, .1, .2, .3, .4, .5} on LABELED transitions (labels
given deliberately — corruption robustness is isolated from clustering):

  NEAR-MISS  after-value = lawful value +-1 — one unit from lawful,
             sitting exactly ON the trim's residual floor cut
             (cut = max(3*median_resid, 1.0))
  FAR        after-value = random garbage in 100..300, far outside the
             lawful range of both operators

At each sweep point, measured: (a) correct rule still minted? (b) each
corrupted episode wrong-kept (excluded, with its why) vs LEARNED
(kept by the trim = fed to the referee = poisoning)? (c) binding exact
on unseen inputs (tol 1e-6) and round-correct? Plus false trims of
clean episodes, and the distractor gate at every point.

ARMS (per species x rate):
  mul/trim+engine  the gate6 mechanism verbatim: shape recurrence ->
                   template mining -> iterated-trim consensus ->
                   simplicity duel -> fresh-mind discover() exactness
                   gate. Trim, templates, duel are CALLER-SIDE
                   SCAFFOLDING (a named capability gap — the engine has
                   no transition-corruption organ); discover()/
                   apply_law() are the engine organs.
  mul/engine-raw   the ORGAN ALONE: all rows straight into fresh-mind
                   discover(), no trim — does its internal half-holdout
                   + R2 gate resist corruption, or mint a poisoned law
                   with a confident verdict (silent poisoning)?
  add/trim+duel    the additive operator (known engine blind spot:
                   discover() cannot say x0+x1 exactly, measured again
                   here), so its gate6 mint path is the linear-exact
                   duel — which demands a PERFECT trim: one kept
                   corrupted row kills linear-exactness and discover
                   cannot rescue add. Expect the lowest breaking rate.

The trim carries the newer probes' collapse guard
(min_keep = max(8, ceil(0.6*n))); where the guard trips, the UNGUARDED
continuation is also measured (labeled) — is the guard preventing a
wrong mint or blocking a fine one?

PREDICTION (filed before the first run):
  P1 0% control reproduces the clean mint on every arm: mul engine
     equation +1.00*x0*x1, unseen exact; add linear-exact (1,1,0).
  P2 FAR/trim: all corrupted wrong-kept and the mint survives through
     0.3; somewhere in 0.3..0.5 the initial lstsq fit is dragged far
     enough that median residual blows past the garbage and the trim
     hits its keep-everything fixed point (0 trimmed) — the measured
     breaking rate of the consensus trim. At 0.5 the collapse guard
     (36 kept minimum) can only pass by keeping garbage: honest refusal
     or trim fixed point, no wrong mint.
  P3 NEAR-MISS/trim: poisoning onset at 0.1 — the +-1 residual sits at
     the cut's floor 1.0, so roughly the half of corrupted rows the
     initial fit leaned toward survive every iteration; any kept
     corrupted row makes exact training impossible -> the mint DIES at
     ~0.1 (availability failure, not silent wrongness). Wrong-kept
     fraction ~half, LEARNED ~half.
  P4 mul/engine-raw: unseen exactness lost at the first nonzero rate
     for both species; verdict may STAY UNDERSTOOD (r2 >= 0.9) while
     predictions are wrong = SILENT POISONING of the bare organ — the
     caller-side trim is load-bearing.
  P5 add/trim: breaking rate <= mul's under near-miss (needs a perfect
     trim); under FAR it matches mul through 0.3 since the trim kills
     far garbage cleanly at low rates.
  P6 distractor (noise) cells never enter preconditions at any rate or
     species — corruption lives in a slot VALUE, not in structure.

Deterministic: run as
  cd /home/user/dc/experiments && PYTHONHASHSEED=0 python3 gate6_corruption_probe.py
World: 60 episodes per sweep point (gate6's scale), 36 points total,
finishes well under 4 min (discover() measured ~0.25s/call).
stdlib + numpy for the cognition; sklearn unused; fresh throwaway
Isaac() minds only; nothing persisted. gate6's isaac_interpreter import
is UNAVAILABLE here (house precedent: gate6_unlabeled_probe.py) — the
mining side is what corruption attacks anyway.
"""
import os
import sys
import time
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import isaac as mod                                       # noqa: E402

TOL = 1e-6
N_EP = 60
RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
UNSEEN = {"mul": [(7, 6, 42), (9, 9, 81), (2, 4, 8), (5, 5, 25)],
          "add": [(9, 9, 18), (7, 6, 13), (2, 4, 6), (3, 5, 8)]}


# ═════════════════════════════════════════════════════════════════
# THE WORLD — gate6's episodes, labels given, corruption planted
# ═════════════════════════════════════════════════════════════════
def episode(rng, op, species):
    x, y = int(rng.integers(1, 10)), int(rng.integers(1, 10))
    sym = "+" if op == "add" else "*"
    before = {("num", 1, str(x)), ("sym", 2, sym), ("num", 3, str(y)),
              ("sym", 4, "="), ("want", "answer")}
    for _ in range(int(rng.integers(0, 3))):              # jittering
        before.add(("noise", int(rng.integers(0, 9))))    # distractors
    v = x + y if op == "add" else x * y
    if species == "near":
        v = v + int(rng.choice([-1, 1]))                  # one off lawful
    elif species == "far":
        v = int(rng.integers(100, 300))                   # far garbage
    after = (before - {("want", "answer")}) | {("answer", str(v))}
    return frozenset(before), frozenset(after)


def gen_world(op, species, rate, seed):
    rng = np.random.default_rng(seed)
    ncor = int(round(rate * N_EP))
    plan = [None] * (N_EP - ncor) + [species] * ncor
    rng.shuffle(plan)
    trans = [episode(rng, op, sp) for sp in plan]
    corrupt = {i for i, sp in enumerate(plan) if sp}
    return trans, corrupt, rng


# ═════════════════════════════════════════════════════════════════
# CALLER-SIDE SCAFFOLDING (named capability gap: the engine has no
# transition-template or trim-consensus organ — gate6's own discipline,
# reproduced verbatim so THIS mechanism is what gets attacked)
# ═════════════════════════════════════════════════════════════════
def sig_of(cells):
    return tuple((c[0], len(c)) for c in sorted(cells))


def mine_template(idxs, cells_of):
    """Always-present, constant-multiplicity (head,arity) groups become
    template cells: constant positions LITERAL, varying become SLOTS."""
    groups = {}
    for i in idxs:
        for c in cells_of(i):
            groups.setdefault((c[0], len(c)), []).append((i, c))
    out, slots = [], {}
    for (head, ln), items in sorted(groups.items()):
        per_ep = Counter(i for i, _ in items)
        mults = sorted({per_ep.get(i, 0) for i in idxs})
        if len(mults) != 1:
            continue                     # distractors die here
        width = mults[0]
        by_ep = {i: sorted(c for j, c in items if j == i) for i in idxs}
        for k in range(width):
            cols = []
            for pos in range(1, ln):
                vals = {by_ep[i][k][pos] for i in idxs}
                if len(vals) == 1:
                    cols.append(("LIT", vals.pop()))
                else:
                    nm = f"{head}{ln}k{k}p{pos}"
                    slots[nm] = {i: by_ep[i][k][pos] for i in idxs}
                    cols.append(("SLOT", nm))
            out.append((head, tuple(cols)))
    return out, slots


def robust_trim(X, yv):
    """gate6's iterated consensus trim (linear+product probe, refit until
    stable). Returns (inlier mask, final residuals, final cut)."""
    A = np.column_stack([X, (X[:, :1] * X[:, 1:2] if X.shape[1] >= 2
                             else X[:, :1]), np.ones(len(X))])
    inlier = np.ones(len(yv), bool)
    resid, cut = np.zeros(len(yv)), 1.0
    for _ in range(6):
        coef, *_ = np.linalg.lstsq(A[inlier], yv[inlier], rcond=None)
        resid = np.abs(A @ coef - yv)
        cut = max(3.0 * np.median(resid[inlier] + 1e-9), 1.0)
        nxt = resid < cut
        if (nxt == inlier).all():
            break
        inlier = nxt
    return inlier, resid, cut


def extract_rows(trans, idxs):
    """Template-mine and pull the numeric transition rows (X: before
    slots, y: the answer slot). Returns None on a mis-shaped world."""
    sigs = [(sig_of(a - b), sig_of(b - a)) for b, a in trans]
    (modal, support), = Counter(sigs).most_common(1)
    if support < max(3, len(trans) // 2):
        return None                       # nothing recurs
    keep = [i for i in idxs if sigs[i] == modal]
    pre_t, pre_s = mine_template(keep, lambda i: trans[i][0])
    add_t, add_s = mine_template(keep, lambda i: trans[i][1] - trans[i][0])
    num_pre = sorted(nm for nm in pre_s
                     if all(v.lstrip("-").isdigit()
                            for v in pre_s[nm].values()))
    outs = sorted(nm for nm in add_s
                  if all(v.lstrip("-").isdigit()
                         for v in add_s[nm].values()))
    if not num_pre or not outs:
        return None
    X = np.array([[float(pre_s[nm][i]) for nm in num_pre] for i in keep])
    yv = np.array([float(add_s[outs[0]][i]) for i in keep])
    noise_in_pre = any(h == "noise" for h, _ in pre_t)
    return dict(keep=keep, X=X, y=yv, pre_t=pre_t, add_t=add_t,
                support=support, noise_in_pre=noise_in_pre)


# ═════════════════════════════════════════════════════════════════
# THE BINDINGS
# ═════════════════════════════════════════════════════════════════
def duel_then_discover(Xk, yk, rng, tag):
    """gate6's binding verbatim: simplicity duel (probe-side linear
    BASELINE, wins only if EXACT, never feeds Isaac) else fresh-mind
    discover() with the train-exactness gate."""
    out = dict(mint=False, path=None, verdict=None, eq=None,
               train_maxerr=float("inf"))
    L = np.column_stack([Xk, np.ones(len(Xk))])
    lc, *_ = np.linalg.lstsq(L, yk, rcond=None)
    lc = np.round(lc, 8)
    lerr = float(np.max(np.abs(L @ lc - yk)))
    if lerr < TOL:
        out.update(mint=True, path="linear-exact", train_maxerr=lerr,
                   eq=" ".join(f"{c:+.2f}*x{j}" for j, c in
                               enumerate(lc[:-1])) + f" {lc[-1]:+.2f}",
                   predict=lambda U, _lc=lc: U @ _lc[:-1] + _lc[-1])
        return out
    mind = mod.Isaac()                    # fresh throwaway mind
    order = rng.permutation(len(yk))
    res = mind.discover(Xk[order], yk[order], name=tag)
    out["verdict"], out["eq"] = res["verdict"], res["equation"]
    out["r2"] = res["r2"]
    pt = mind.apply_law(tag, Xk)
    out["train_maxerr"] = (float(np.max(np.abs(pt - yk)))
                           if pt is not None else float("inf"))
    if out["train_maxerr"] < TOL:
        out.update(mint=True, path="engine",
                   predict=lambda U, _m=mind, _t=tag: _m.apply_law(_t, U))
    return out


def unseen_eval(predict, op):
    U = np.array([[float(x), float(y)] for x, y, _ in UNSEEN[op]])
    want = np.array([float(w) for _, _, w in UNSEEN[op]])
    got = predict(U)
    if got is None:
        return float("inf"), 0, len(want)
    got = np.asarray(got, float)
    maxerr = float(np.max(np.abs(got - want)))
    nround = int(sum(int(round(g)) == int(w) for g, w in zip(got, want)))
    return maxerr, nround, len(want)


def run_trim_point(op, species, rate, seed):
    """One sweep point of the gate6 mechanism (trim -> duel -> engine)."""
    trans, corrupt, rng = gen_world(op, species, rate, seed)
    ex = extract_rows(trans, list(range(len(trans))))
    rec = dict(op=op, species=species, rate=rate, ncor=len(corrupt),
               mis_built=ex is None)
    if ex is None:
        return rec
    keep, X, yv = ex["keep"], ex["X"], ex["y"]
    rec["noise_in_pre"] = ex["noise_in_pre"]
    inlier, resid, cut = robust_trim(X, yv)
    kept_idx = {i for i, k in zip(keep, inlier) if k}
    rec["kept"] = len(kept_idx)
    rec["cut"] = cut
    rec["cor_learned"] = sorted(kept_idx & corrupt)     # POISONING
    rec["cor_wrongkept"] = sorted((set(keep) - kept_idx) & corrupt)
    rec["false_trims"] = sorted((set(keep) - kept_idx) - corrupt)
    rec["whys"] = {i: f"resid {resid[j]:.2f} >= cut {cut:.2f} "
                      f"vs consensus"
                   for j, i in enumerate(keep)
                   if i in rec["cor_wrongkept"]}
    min_keep = max(8, int(np.ceil(0.6 * len(keep))))
    rec["min_keep"] = min_keep
    rec["guard_tripped"] = len(kept_idx) < min_keep
    Xk, yk = X[inlier], yv[inlier]
    bind = duel_then_discover(Xk, yk, rng, f"{op}_{species}_{rate}")
    if rec["guard_tripped"]:
        rec["mint"] = False
        rec["reason"] = (f"trim consensus collapsed ({len(kept_idx)} "
                         f"kept < {min_keep}) — binding refused")
        rec["unguarded"] = dict(mint=bind["mint"], path=bind["path"])
        if bind["mint"]:
            me, nr, nt = unseen_eval(bind["predict"], op)
            rec["unguarded"].update(unseen_maxerr=me, nround=nr, ntot=nt)
    else:
        rec.update(mint=bind["mint"], path=bind["path"],
                   verdict=bind.get("verdict"), eq=bind["eq"],
                   train_maxerr=bind["train_maxerr"])
        if not bind["mint"]:
            rec["reason"] = (f"no consistent intent (train maxerr "
                             f"{bind['train_maxerr']:.3g} > {TOL:g}; "
                             f"discover verdict {bind.get('verdict')})")
        else:
            me, nr, nt = unseen_eval(bind["predict"], op)
            rec.update(unseen_maxerr=me, nround=nr, ntot=nt,
                       unseen_exact=me < TOL)
    return rec


def run_raw_point(op, species, rate, seed):
    """The ORGAN ALONE: every row (corruption included) straight into a
    fresh mind's discover(). No trim, no duel, no exactness gate — what
    does the engine itself do with a corrupted table?"""
    trans, corrupt, rng = gen_world(op, species, rate, seed)
    ex = extract_rows(trans, list(range(len(trans))))
    rec = dict(op=op, species=species, rate=rate, ncor=len(corrupt),
               mis_built=ex is None)
    if ex is None:
        return rec
    X, yv = ex["X"], ex["y"]
    mind = mod.Isaac()
    order = rng.permutation(len(yv))
    tag = f"raw_{op}_{species}_{rate}"
    res = mind.discover(X[order], yv[order], name=tag)
    rec.update(verdict=res["verdict"], eq=res["equation"], r2=res["r2"])
    me, nr, nt = unseen_eval(lambda U: mind.apply_law(tag, U), op)
    rec.update(unseen_maxerr=me, nround=nr, ntot=nt,
               unseen_exact=me < TOL,
               silent_poison=(res["verdict"] == "UNDERSTOOD"
                              and me >= TOL))
    return rec


# ═════════════════════════════════════════════════════════════════
# THE RUN
# ═════════════════════════════════════════════════════════════════
def fmt_trim(r):
    if r.get("mis_built"):
        return "  MIS-BUILT WORLD"
    base = (f"  {r['rate']:.1f}  ncor {r['ncor']:2d}  kept {r['kept']:2d}"
            f"/{N_EP}  learned {len(r['cor_learned']):2d}  wrong-kept "
            f"{len(r['cor_wrongkept']):2d}  false-trim "
            f"{len(r['false_trims']):2d}  cut {r['cut']:6.1f}")
    if r.get("guard_tripped"):
        ug = r["unguarded"]
        tail = (f"GUARD REFUSAL (min_keep {r['min_keep']}); unguarded: "
                f"mint={ug['mint']}"
                + (f" path={ug['path']} unseen_maxerr="
                   f"{ug['unseen_maxerr']:.3g} round {ug['nround']}/"
                   f"{ug['ntot']}" if ug.get("mint") else ""))
    elif r["mint"]:
        tail = (f"MINT[{r['path']}] eq {r['eq']}  unseen_maxerr "
                f"{r['unseen_maxerr']:.3g} round {r['nround']}/{r['ntot']}")
    else:
        tail = f"NO MINT: {r['reason']}"
    return base + "  " + tail


def breaking_rate(recs, ok_fn):
    """Smallest nonzero rate where ok_fn fails; None if it never does."""
    for r in recs:
        if r["rate"] > 0 and not ok_fn(r):
            return r["rate"]
    return None


def main():
    t0 = time.time()
    print("=" * 72)
    print("GATE 6 CORRUPTION ATTACK — near-miss vs far, rate sweep, "
          "trim vs bare organ")
    print("=" * 72)

    trim_recs = {}          # (op, species) -> [rec per rate]
    raw_recs = {}
    seed = 977
    for op in ("mul", "add"):
        for species in ("near", "far"):
            trim_recs[(op, species)] = []
            for ri, rate in enumerate(RATES):
                s = seed + 1000 * (op == "add") + 100 * (species == "far") + ri
                trim_recs[(op, species)].append(
                    run_trim_point(op, species, rate, s))
    for species in ("near", "far"):
        raw_recs[("mul", species)] = []
        for ri, rate in enumerate(RATES):
            s = seed + 5000 + 100 * (species == "far") + ri
            raw_recs[("mul", species)].append(
                run_raw_point("mul", species, rate, s))

    for (op, species), recs in trim_recs.items():
        print(f"\n[{op} / {species.upper()} / trim+{'engine' if op == 'mul' else 'duel'}]"
              f"  (gate6 mechanism)")
        for r in recs:
            print(fmt_trim(r))
        whys = [w for r in recs for w in r.get("whys", {}).values()][:2]
        if whys:
            print(f"  sample wrong-kept whys: {whys}")

    for (op, species), recs in raw_recs.items():
        print(f"\n[{op} / {species.upper()} / ENGINE-RAW]  (organ alone, "
              f"no trim)")
        for r in recs:
            print(f"  {r['rate']:.1f}  ncor {r['ncor']:2d}  verdict "
                  f"{r['verdict']:>10s}  r2 {r['r2']:7.4f}  unseen_maxerr "
                  f"{r['unseen_maxerr']:9.3g}  round {r['nround']}/"
                  f"{r['ntot']}  silent_poison={r['silent_poison']}  "
                  f"eq {str(r['eq'])[:48]}")

    # ═══ GRADING ═══════════════════════════════════════════════════
    verdicts = {}

    # P1 — 0% controls reproduce the clean mint on every arm
    ctrl = []
    for (op, species), recs in trim_recs.items():
        r = recs[0]
        ok = (r.get("mint") and r.get("unseen_exact")
              and not r["cor_learned"] and not r["false_trims"])
        ctrl.append((f"{op}/{species}", ok, r.get("eq"),
                     r.get("unseen_maxerr")))
    raw0 = [raw_recs[("mul", sp)][0] for sp in ("near", "far")]
    raw_ok = all(r["verdict"] == "UNDERSTOOD" and r["unseen_exact"]
                 for r in raw0)
    all_ok = all(ok for _, ok, _, _ in ctrl) and raw_ok
    verdicts["control-0pct-clean-mint"] = (
        "WORKS" if all_ok else "BROKEN",
        "; ".join(f"{t}: mint={ok} maxerr={me:.3g}"
                  for t, ok, _, me in ctrl)
        + f"; raw mul 0%: {[r['verdict'] for r in raw0]} "
          f"maxerr={[round(r['unseen_maxerr'], 9) for r in raw0]}")

    # trim breaking rates per species (mul, the engine-mintable op)
    for species in ("near", "far"):
        recs = trim_recs[("mul", species)]
        mint_ok = lambda r: (r.get("mint") and r.get("unseen_exact"))
        br = breaking_rate(recs, mint_ok)
        poison = breaking_rate(recs, lambda r: not r["cor_learned"])
        learned_tot = sum(len(r["cor_learned"]) for r in recs)
        wrong_tot = sum(len(r["cor_wrongkept"]) for r in recs)
        ncor_tot = sum(r["ncor"] for r in recs)
        false_tot = sum(len(r["false_trims"]) for r in recs)
        # did any point mint a WRONG rule (mint + inexact unseen)?
        wrong_mint = [r["rate"] for r in recs
                      if r.get("mint") and not r.get("unseen_exact")]
        ev = (f"mint-break at rate {br}; poison onset {poison}; "
              f"learned {learned_tot}/{ncor_tot} corrupted rows, "
              f"wrong-kept {wrong_tot}; false trims {false_tot}; "
              f"wrong mints {wrong_mint or 0}")
        if br is None:
            v = "WORKS"
        elif wrong_mint:
            v = "BROKEN"
        else:
            # mint lost but never wrong: availability failure with an
            # honest refusal at every broken point
            all_refusals_honest = all(
                r.get("mint") or r.get("guard_tripped")
                or "no consistent intent" in r.get("reason", "")
                for r in recs)
            v = "PARTIAL" if all_refusals_honest else "BROKEN"
        verdicts[f"trim-{species}-corruption(mul)"] = (v, ev)

    # add coupling: linear-exact duel needs a PERFECT trim
    for species in ("near", "far"):
        recs = trim_recs[("add", species)]
        br = breaking_rate(recs, lambda r: (r.get("mint")
                                            and r.get("unseen_exact")))
        wrong_mint = [r["rate"] for r in recs
                      if r.get("mint") and not r.get("unseen_exact")]
        poison = breaking_rate(recs, lambda r: not r["cor_learned"])
        ev = (f"mint-break at rate {br}; poison onset {poison}; "
              f"wrong mints {wrong_mint or 0} (add cannot fall back to "
              f"discover: additive blind spot)")
        verdicts[f"add-mint-{species}-corruption"] = (
            "WORKS" if br is None else
            ("BROKEN" if wrong_mint else "PARTIAL"), ev)

    # bare organ under corruption
    for species in ("near", "far"):
        recs = raw_recs[("mul", species)]
        br = breaking_rate(recs, lambda r: r["unseen_exact"])
        silent = [r["rate"] for r in recs if r["silent_poison"]]
        rd = {r["rate"]: f"{r['nround']}/{r['ntot']}" for r in recs}
        ev = (f"exactness lost at rate {br}; SILENT POISON "
              f"(UNDERSTOOD verdict + wrong unseen) at rates {silent}; "
              f"round-correct {rd}")
        verdicts[f"engine-raw-{species}(mul)"] = (
            "WORKS" if br is None and not silent else
            ("BROKEN" if silent else "PARTIAL"), ev)

    # 50% guard behavior
    g = []
    for key, recs in trim_recs.items():
        r = recs[-1]
        if r.get("guard_tripped"):
            ug = r["unguarded"]
            g.append(f"{key[0]}/{key[1]}: guard refusal, unguarded "
                     f"mint={ug['mint']}")
        else:
            g.append(f"{key[0]}/{key[1]}: no guard trip (kept "
                     f"{r['kept']}, mint={r.get('mint')})")
    any_wrong_50 = any(r.get("mint") and not r.get("unseen_exact")
                       for recs in trim_recs.values() for r in [recs[-1]])
    verdicts["collapse-guard-at-50pct"] = (
        "BROKEN" if any_wrong_50 else "HONEST-REFUSAL"
        if all(not recs[-1].get("mint") for recs in trim_recs.values())
        else "PARTIAL", "; ".join(g))

    # distractor discipline across every point
    noise_hits = [(r["op"], r["species"], r["rate"])
                  for recs in trim_recs.values() for r in recs
                  if r.get("noise_in_pre")]
    verdicts["distractor-exclusion-under-corruption"] = (
        "WORKS" if not noise_hits else "BROKEN",
        f"noise head in preconditions at {len(noise_hits)}/24 points"
        + (f": {noise_hits}" if noise_hits else ""))

    # ═══ VERDICT_SHEET ═════════════════════════════════════════════
    dt = time.time() - t0
    print("\n" + "=" * 72)
    print("VERDICT_SHEET (gate6 corruption attack)")
    for cap, (v, ev) in verdicts.items():
        print(f"  {cap:40s} {v:14s} {ev}")
    print(f"  NOTE: trim consensus, templates, simplicity duel are "
          f"CALLER-SIDE scaffolding\n  (no engine organ does this) — "
          f"the engine-raw rows above measure the organ alone.")
    print(f"  runtime: {dt:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()

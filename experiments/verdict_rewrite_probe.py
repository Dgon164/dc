"""verdict_rewrite_probe.py -- S2b THE VERDICT REWRITE, reference v3.
RESEARCH, NOT THE ENGINE (research-first-then-promote). One verdict,
TWO LEGS, side-by-side with the old r2 road on every world.

PROVENANCE: leg-1 machinery (candidates, partial-corr scan with the
selection-corrected max-null, continue loop, Occam prune, ruler-grid
snap, floor roads) transcribed VERBATIM from surprise_verdict_probe.py
(the 07-14 reference, v2). Leg-2 machinery (frozen random-features
meter, 5-fold cross-fit capture, permuted-y scream null) transcribed
from reservoir_surprise_probe.py (13/13) -- renamed POOL per the owner
naming 2026-07-15: "a shallow fixed pool to help him learn to swim."

THE FIVE HARD REQUIREMENTS (plan S2b), where each lives here:
  (1) residual replay is PARTIAL-CORRELATION / joint-refit ... scan()
  (2) replay selection-corrected max-over-everything .......... scan()
  (3) mark snap on GRIDS {1,.5,.25}x10^k coarsest-first, earned
      marks HELD fixed through subsequent refits ............... snap()
  (4) exact-checkable data stamps on machine-zero residual
      WITHOUT a floor (Kepler n=8 must survive) .......... exact_road()
      -- the guard against selection at tiny n is HELD-OUT EXACTNESS:
      a wrong form does not hit fresh rows to 1e-9 RELATIVE; the n=8
      noise control below MEASURES that claim.
  (5) full false-alarm control before anything else counts ..... V0a/V0b
TWO LEGS, the addendum's proposal: leg 1 = the atom replay (names the
strongest leftover candidate); leg 2 = the POOL METER on the full-row
residual (grammar-free: screams when structure remains that NO pool
candidate expresses). Either leg firing blocks UNDERSTOOD. The pool
POINTS, never answers: it contributes a boolean + receipts, never a
term, never a feature rank (integrity law, reservoir probe).
CONSIDERATION LAW: every leg fires or is BYPASSED WITH A RECEIPT in
the verdict record (exact road stamped / no law / below measured scope).
Leg-2 measured scope: scream proven at n>=48 (reservoir probe R0b);
below that the leg is bypassed and SAYS SO.

PREDICTIONS (filed before the first run; wrong ones will be KEPT):
  P1a pure noise x20: NEW false UNDERSTOOD 0/20 (old road recorded).
  P1b true quadratic, sigma TOLD, x20: UNDERSTOOD >=18/20
      (availability, not just safety) AND combined false surprise
      fires (either leg) <=2/20 -- two legs OR-ed at 99th pct each,
      the lens-selection guard measured, not assumed.
  P2  additive hole (W1, x0*x1 impostor available): NEW recovers the
      exact additive form (both coefficients snap to 1); untold
      variant honestly PARTIAL-above-floor (grain under-evidences
      process noise -- known wrong-kept); TOLD variant UNDERSTOOD.
  P3  Kepler n=8, y = a^3 exact: UNDERSTOOD (machine-exact) via the
      exact road, floor UNEVIDENCED, equation says x^3.
      n=8 noise control x20: exact stamps 0/20.
  P4  blended conditional n=48 (flag decides add vs mul; the triple
      f*x0*x1 is NOT in the candidate pool): NEW = PARTIAL with leg 1
      QUIET and leg 2 SCREAM -- the grammar-free leg earning its seat.
      (Uncertainty filed: the toy old road may or may not reach the
      engine exhibit's 0.9972; the engine exhibit stands regardless.)
      >>> FALSIFIED BY RUN 1, KEPT: NEW = PARTIAL held, but leg 1
      FIRED (x1*x2 obs 0.843 vs thr 0.663 -- pairwise leftovers were
      still nameable after prune kept only 2 terms) and leg 2 was
      BYPASSED WITH A RECEIPT: lag columns eat 2 rows per half, 48
      valid rows became 44 < the pool's measured scope of 48. The
      consideration law worked exactly as designed; the prediction
      mis-guessed which leg. V3b (n=96) measures the pool IN scope;
      the leg-1-quiet/leg-2-alone existence proof is V4 (measured:
      replay quiet 0.465<0.629, pool scream 0.890>0.062, 5/5).
  P5  near-miss +-1 poisoning 10%, n=60, x5 seeds: confidently-wrong
      UNDERSTOOD (wrong constants shipped as UNDERSTOOD) 0/5. Split
      between true-form UNDERSTOOD (grid snap unbends the tilt) and
      PARTIAL (pool bend-catch) RECORDED, not predicted -- the
      single-lens precedent (4/5 screams) was on ENGINE residuals
      whose trim bent the form; OLS+snap here may unbend instead.
  P6  plant-absorbed step (y = 2*x0 + 1.5*[1<x1<2], poly pool only):
      OLD stamps a bent poly form; NEW never UNDERSTOOD, leg 2 screams.
  P7  regression vs v2: W2 (3x @ 1/16 grain) UNDERSTOOD saying 3*x;
      W3b (3.000154 on a coarse ruler) UNDERSTOOD saying 3*x;
      W8 (repeats road) UNDERSTOOD -- the rewrite must lose nothing v2 won.
  P8  cost: one full two-leg verdict < 2 s at n=400 (complete number
      printed; no silent caps).

NAMED NEXT PROBE (open, not covered here): poison at rates whose rms
slips UNDER 3x floor (e.g. 5% +-1 on unit grain, rms ~0.22 vs gate
0.87) -- +-1 cells also sit under the audit's 4x line, so if the pool
misses the smaller bend, that regime passes BOTH tiers silently.
Measure the pool's bend-catch floor vs poison rate before claiming it.

Deterministic: fixed seeds; pool weights seeded 777 (instrument
identity). stdlib + numpy only. Run:
  cd experiments && PYTHONHASHSEED=0 python verdict_rewrite_probe.py
"""
import time
from itertools import combinations

import numpy as np

RNG = lambda s: np.random.default_rng(s)
NPERM = 150
IDX99 = 1              # 2nd largest of 150 ~ 99th pct
P_NULL = 200           # pool permuted-y replicas
FOLDS = 5
RIDGE = 1e-2
NULL_PCT = 99.0
POOL_M = 200           # m* chosen by the reservoir probe's own R0 sweep
POOL_MIN_N = 48        # measured scope: scream proven at n>=48 (R0b)
EXACT_REL = 1e-9       # machine-exact = relative to the numbers' own size


# ------------------------------------------------ leg-1 machinery (v2 verbatim)
def make_candidates(X, n_rows):
    n, d = X.shape
    max_lag = int(min(8, n_rows // 12)) if n_rows >= 24 else 0
    vn = (lambda i: "x") if d == 1 else (lambda i: f"x{i}")
    cands, names = [], []
    for i in range(d):
        cands.append(X[:, i]); names.append(vn(i))
        cands.append(X[:, i] ** 2); names.append(f"{vn(i)}^2")
        cands.append(X[:, i] ** 3); names.append(f"{vn(i)}^3")
    for i in range(d):
        for j in range(i + 1, d):
            cands.append(X[:, i] * X[:, j]); names.append(f"{vn(i)}*{vn(j)}")
    for i in range(d):
        for k in range(1, max_lag + 1):
            col = np.full(n, np.nan); col[k:] = X[:-k, i]
            cands.append(col); names.append(f"{vn(i)}[t-{k}]")
    return np.array(cands).T, names

def vrows(C): return ~np.isnan(C).any(axis=1)

def ols(C, y):
    A = np.column_stack([C, np.ones(len(y))]) if C.shape[1] else np.ones((len(y), 1))
    coef, *_ = np.linalg.lstsq(A, y, rcond=None); return coef

def pred(C, coef):
    A = np.column_stack([C, np.ones(C.shape[0])]) if C.shape[1] else np.ones((C.shape[0], 1))
    return A @ coef

def r2(y, yh):
    ss = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((y - yh) ** 2) / ss if ss > 0 else 0.0

def residualize(col, Csel):
    c = ols(Csel, col); return col - pred(Csel, c)

def safe_corr(a, b):
    sa, sb = a.std(), b.std()
    return abs(np.corrcoef(a, b)[0, 1]) if sa > 1e-12 and sb > 1e-12 else 0.0

def fmt_eq(coef, chosen, names):
    if not chosen: return "(none)"
    parts = []
    for i, j in enumerate(chosen):
        c = coef[i]; s = "-" if c < 0 else "+"
        a = abs(c)
        body = names[j] if abs(a - 1.0) < 1e-12 else f"{a:.6g}*{names[j]}"
        parts.append(f"{s} {body}")
    out = " ".join(parts)
    return out[2:] if out.startswith("+ ") else out

def scan(C, chosen, resid, rng, pool=None):
    pool = pool if pool is not None else [j for j in range(C.shape[1]) if j not in chosen]
    Csel = C[:, chosen] if chosen else np.empty((C.shape[0], 0))
    rc = [residualize(C[:, j], Csel) for j in pool]
    obs_scores = [safe_corr(r, resid) for r in rc]
    b = int(np.argmax(obs_scores)); obs = obs_scores[b]
    null = np.zeros(NPERM)
    for p in range(NPERM):
        rp = rng.permutation(resid)
        null[p] = max(safe_corr(r, rp) for r in rc)
    thr = np.sort(null)[::-1][IDX99]
    return obs, thr, pool[b]

def fit(C, names, y, rng, r2_stop=0.9, max_terms=6):
    ok = vrows(C); Cv, yv = C[ok], y[ok]
    chosen, resid = [], yv - yv.mean()
    while len(chosen) < max_terms:
        obs, thr, j = scan(Cv, chosen, resid, rng)
        if obs <= thr: break
        chosen.append(j)
        coef = ols(Cv[:, chosen], yv); resid = yv - pred(Cv[:, chosen], coef)
        if r2(yv, pred(Cv[:, chosen], coef)) >= r2_stop: break
    coef = ols(Cv[:, chosen], yv) if chosen else ols(np.empty((len(yv), 0)), yv)
    return chosen, coef

def prune(C, chosen, y_tr, C_ho, y_ho, tol_rms):
    ok_t, ok_h = vrows(C), vrows(C_ho)
    keep = list(chosen)
    changed = True
    while changed and len(keep) > 1:
        changed = False
        base = ols(C[ok_t][:, keep], y_tr[ok_t])
        base_rms = np.sqrt(np.mean((y_ho[ok_h] - pred(C_ho[ok_h][:, keep], base)) ** 2))
        for t in list(keep):
            sub = [k for k in keep if k != t]
            c = ols(C[ok_t][:, sub], y_tr[ok_t])
            rms = np.sqrt(np.mean((y_ho[ok_h] - pred(C_ho[ok_h][:, sub], c)) ** 2))
            if rms <= base_rms + tol_rms:
                keep = sub; changed = True; break
    return keep

def repeats_floor(X, y):
    seen = {}
    for i in range(len(y)):
        seen.setdefault(tuple(np.round(X[i], 10)), []).append(y[i])
    groups = [v for v in seen.values() if len(v) >= 2]
    if len(groups) < 5: return None
    pooled = np.concatenate([np.asarray(g) - np.mean(g) for g in groups])
    dof = sum(len(g) - 1 for g in groups)
    return float(np.sqrt(np.sum(pooled ** 2) / dof)) if dof >= 10 else None

def read_floor(y):
    u = np.sort(np.unique(y))
    if len(u) < 3: return None, "too few distinct values"
    d = np.diff(u); d = d[d > 1e-12 * max(1.0, abs(u).max())]
    if len(d) == 0: return None, "no positive gaps"
    g0 = d.min()
    ratios = d / g0; mult = np.round(ratios)
    good = mult > 0
    if good.sum() < 3: return None, "no consistent grain"
    g = np.median(d[good] / mult[good])
    frac = np.mean(np.abs(d / g - np.round(d / g)) < 0.02 * np.maximum(1, np.round(d / g)))
    if frac > 0.95:
        return g / np.sqrt(12.0), f"grain {g:.6g} ({frac:.0%} consistent)"
    return None, "no consistent grain (float-continuous)"

def read_floor_all(X, y, told_sigma=None):
    roads = []
    rf = repeats_floor(X, y)
    if rf is not None: roads.append((rf, f"repeats {rf:.4g}"))
    if told_sigma is not None: roads.append((told_sigma, f"told {told_sigma:.4g}"))
    gf, gwhy = read_floor(y)
    if gf is not None: roads.append((gf, gwhy))
    if not roads: return None, gwhy + "; no repeats; nothing told"
    f = max(r[0] for r in roads)
    return f, " | ".join(r[1] for r in roads)

GRIDS = [b * 10.0 ** p for p in range(2, -7, -1) for b in (1.0, 0.5, 0.25)]
def snap(C, chosen, coef, y_tr, C_ho, y_ho, floor_std):
    ok_t, ok_h = vrows(C), vrows(C_ho)
    Ctr, ytr = C[ok_t][:, chosen], y_tr[ok_t]
    Cho, yho = C_ho[ok_h][:, chosen], y_ho[ok_h]
    base_rms = np.sqrt(np.mean((yho - pred(Cho, coef)) ** 2))
    tol = (floor_std or 0.0) * 0.5 + base_rms * 0.01
    out = coef.copy()
    snapped = set()
    for t in range(len(chosen)):
        v = out[t]
        for g in GRIDS:                      # coarsest grid first
            m = np.round(v / g) * g
            if m == 0 and abs(v) > g: continue
            trial = out.copy(); trial[t] = m
            free = [k for k in range(len(chosen)) if k != t and k not in snapped]
            fixed = [k for k in range(len(chosen)) if k != t and k in snapped]
            target = ytr - Ctr[:, t] * m
            for k in fixed: target = target - Ctr[:, k] * out[k]
            sub = ols(Ctr[:, free], target)
            for i2, k in enumerate(free): trial[k] = sub[i2]
            trial[-1] = sub[-1]
            rms = np.sqrt(np.mean((yho - pred(Cho, trial)) ** 2))
            if rms <= base_rms + tol:
                out = trial; snapped.add(t); break   # coarsest surviving grid wins
    return out


# ------------------------------------------------ leg 0: the exact road (req 4)
def exact_road(C, y, C_ho, y_ho):
    """Machine-exact stamp for exact-checkable data. Greedy-ranked
    subsets up to 3 terms; a hit must be exact (relative) on TRAIN and
    then on HELD-OUT -- held-out exactness IS the selection guard at
    tiny n (measured by the n=8 noise control)."""
    ok_t, ok_h = vrows(C), vrows(C_ho)
    if ok_h.sum() < 3 or ok_t.sum() < 3: return None
    Ct, yt, Ch, yh = C[ok_t], y[ok_t], C_ho[ok_h], y_ho[ok_h]
    scale = max(np.abs(np.concatenate([yt, yh])).max(), 1e-12)
    base = yt - yt.mean()
    order = np.argsort([-safe_corr(Ct[:, j], base)
                        for j in range(Ct.shape[1])])[:10].tolist()
    for k in (1, 2, 3):
        if len(yt) < k + 2: break            # spare rows beyond parameters
        for sub in combinations(order, k):
            sub = list(sub)
            coef = ols(Ct[:, sub], yt)
            if np.abs(yt - pred(Ct[:, sub], coef)).max() / scale > EXACT_REL:
                continue
            if np.abs(yh - pred(Ch[:, sub], coef)).max() / scale > EXACT_REL:
                continue
            # snap coefficients to the coarsest grid that KEEPS exactness
            out = coef.copy()
            for t in range(k):
                for g in GRIDS:
                    m = np.round(out[t] / g) * g
                    trial = out.copy(); trial[t] = m
                    if (np.abs(yt - pred(Ct[:, sub], trial)).max() / scale
                            <= EXACT_REL and
                        np.abs(yh - pred(Ch[:, sub], trial)).max() / scale
                            <= EXACT_REL):
                        out = trial; break
            return sub, out
    return None


# ------------------------------------------------ leg 2: THE POOL METER
class PoolMeter:
    """A shallow FIXED pool (owner naming 2026-07-15): frozen tanh
    random features + ridge readout, weights seeded BEFORE data.
    5-fold cross-fit capture: every row judged by a readout that never
    saw it. Verbatim CaptureMeter from reservoir_surprise_probe.py."""

    GAINS = (0.7, 1.5, 2.5)   # multiscale, fixed -- instrument identity

    def __init__(self, X, m, seed=777, split_seed=0):
        n, d = X.shape
        self.n = n
        self.m_eff = min(m, max(16, 2 * n), 400)
        rng = np.random.default_rng(seed + 1000 * self.m_eff + d)
        gains = np.array([self.GAINS[i % len(self.GAINS)]
                          for i in range(self.m_eff)])
        self.W = rng.normal(0.0, 1.0 / np.sqrt(d), (d, self.m_eff)) * gains
        self.b = rng.uniform(-1.0, 1.0, self.m_eff)
        idx = np.random.default_rng(split_seed).permutation(n)
        self.folds = [idx[f::FOLDS] for f in range(FOLDS)]
        self.parts = []
        for ho in self.folds:
            tr = np.setdiff1d(np.arange(n), ho)
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
            H_tr = np.tanh(((X[tr] - mu) / sd) @ self.W + self.b)
            H_ho = np.tanh(((X[ho] - mu) / sd) @ self.W + self.b)
            A = H_tr.T @ H_tr + RIDGE * len(tr) * np.eye(self.m_eff)
            M = np.linalg.solve(A, H_tr.T)
            self.parts.append((tr, ho, H_ho, M))

    def capture(self, r):
        p = np.empty(self.n)
        for tr, ho, H_ho, M in self.parts:
            c = r[tr].mean()
            p[ho] = H_ho @ (M @ (r[tr] - c)) + c
        ss = np.sum((r - p) ** 2)
        tot = np.sum((r - r.mean()) ** 2)
        return 1.0 - ss / tot if tot > 0 else 0.0


def pool_scream(X, r, m=POOL_M, seed=777, rng_seed=1):
    """Capture vs the selection-corrected permuted-y null.
    Returns (capture, thr, scream). The pool POINTS, never answers."""
    meter = PoolMeter(X, m, seed=seed)
    cap = meter.capture(r)
    rng = np.random.default_rng(rng_seed)
    null = np.array([meter.capture(rng.permutation(r))
                     for _ in range(P_NULL)])
    thr = np.percentile(null, NULL_PCT)
    return cap, thr, cap > thr


# ------------------------------------------------ THE REWRITE: one verdict, two legs
def verdict_v3(X, y, seed, told_sigma=None):
    X = np.asarray(X, float); y = np.asarray(y, float)
    rng = RNG(seed + 5000)
    n = len(y); h = n // 2
    Xtr, ytr, Xho, yho = X[:h], y[:h], X[h:], y[h:]
    Ctr, names = make_candidates(Xtr, h); Cho, _ = make_candidates(Xho, h)
    floor_std, floor_why = read_floor_all(X, y, told_sigma)
    ok = vrows(Cho)

    # OLD road, side by side: greedy halt, verdict by held-out r2
    ch_o, cf_o = fit(Ctr, names, ytr, rng, r2_stop=0.9)
    yh_o = pred(Cho[ok][:, ch_o], cf_o)
    r2_o = r2(yho[ok], yh_o)
    old = "UNDERSTOOD" if r2_o >= 0.9 else ("PARTIAL" if r2_o > 0.3 else "NO RULE")

    base = dict(old=old, eq_o=fmt_eq(cf_o, ch_o, names), r2_o=r2_o,
                floor=floor_why)

    # LEG 0 -- exact road (requirement 4)
    ex = exact_road(Ctr, ytr, Cho, yho)
    if ex is not None:
        ch_e, cf_e = ex
        yh_e = pred(Cho[ok][:, ch_e], cf_e)
        base.update(new="UNDERSTOOD (machine-exact)",
                    eq_n=fmt_eq(cf_e, ch_e, names),
                    r2_n=r2(yho[ok], yh_e),
                    rms=float(np.sqrt(np.mean((yho[ok] - yh_e) ** 2))),
                    leg1=dict(state="bypassed: exact road stamped"),
                    leg2=dict(state="bypassed: exact road stamped"))
        return base

    # continue loop -> prune -> snap (requirements 1-3)
    ch, cf = fit(Ctr, names, ytr, rng, r2_stop=1.1)
    base_rms0 = (np.sqrt(np.mean((yho[ok] - pred(Cho[ok][:, ch], cf)) ** 2))
                 if ch else np.inf)
    tolp = (floor_std or 0.0) * 0.5 + base_rms0 * 0.01
    if len(ch) > 1: ch = prune(Ctr, ch, ytr, Cho, yho, tolp)
    cf = ols(Ctr[vrows(Ctr)][:, ch], ytr[vrows(Ctr)]) if ch else cf
    if ch: cf = snap(Ctr, ch, cf, ytr, Cho, yho, floor_std)
    yh = pred(Cho[ok][:, ch], cf) if ch else np.full(ok.sum(), ytr.mean())
    rms = float(np.sqrt(np.mean((yho[ok] - yh) ** 2)))
    r2_n = r2(yho[ok], yh)

    # LEG 1 -- atom replay on held-out residual (requirements 1-2)
    if ch:
        resid_ho = yho[ok] - yh
        obs, thr, bj = scan(Cho[ok], ch, resid_ho, rng)
        leg1 = dict(state="fired" if obs > thr else "quiet",
                    who=names[bj], obs=float(obs), thr=float(thr))
    else:
        leg1 = dict(state="bypassed: no law admitted")

    # LEG 2 -- pool meter on the FULL-row residual (grammar-free)
    leg2 = dict(state="bypassed: no law admitted")
    if ch:
        ok_t = vrows(Ctr)
        resid_tr = ytr[ok_t] - pred(Ctr[ok_t][:, ch], cf)
        Xv = np.vstack([Xtr[ok_t], Xho[ok]])
        rv = np.concatenate([resid_tr, resid_ho])
        if len(rv) >= POOL_MIN_N:
            cap, pthr, scream = pool_scream(Xv, rv, rng_seed=seed + 9)
            leg2 = dict(state="scream" if scream else "quiet",
                        cap=float(cap), thr=float(pthr))
        else:
            leg2 = dict(state=f"bypassed: n={len(rv)} < {POOL_MIN_N} "
                              f"(measured scope)")

    # verdict precedence: either leg blocks UNDERSTOOD; then the floor
    if not ch:
        new = "NO RULE"
    elif leg1["state"] == "fired":
        new = f"PARTIAL (surprise left: {leg1['who']})"
    elif leg2["state"] == "scream":
        new = "PARTIAL (surprise left, unnamed -- pool)"
    elif floor_std is not None:
        new = ("UNDERSTOOD" if rms <= 3 * floor_std
               else "PARTIAL (residual above floor)")
    else:
        scale = max(np.abs(y).max(), 1e-12)
        new = ("UNDERSTOOD (machine-exact)" if rms / scale < EXACT_REL
               else "BEST-RESIDUAL (floor unevidenced)")
    base.update(new=new, eq_n=fmt_eq(cf, ch, names), r2_n=r2_n, rms=rms,
                leg1=leg1, leg2=leg2)
    return base


# ------------------------------------------------ worlds
def correlated_pair(rng, n, rho=0.7, lo=1, hi=5):
    z = rng.uniform(lo, hi, n)
    j = (1 - rho) * (hi - lo) / 2
    return np.column_stack([z + rng.uniform(-j, j, n), z + rng.uniform(-j, j, n)])

def q(y, g): return np.round(y / g) * g

def show(name, v):
    print("=" * 78); print(name)
    print(f"  OLD: {v['old']:<32} {v['eq_o']}   (ho r2 {v['r2_o']:.4f})")
    print(f"  NEW: {v['new']:<32} {v['eq_n']}")
    print(f"       rms {v['rms']:.5g}   floor: {v['floor']}")
    l1, l2 = v["leg1"], v["leg2"]
    s1 = l1["state"] + (f" ({l1['who']} obs {l1['obs']:.3f} thr {l1['thr']:.3f})"
                        if "obs" in l1 else "")
    s2 = l2["state"] + (f" (cap {l2['cap']:+.3f} thr {l2['thr']:+.3f})"
                        if "cap" in l2 else "")
    print(f"       leg1 replay: {s1}")
    print(f"       leg2 pool:   {s2}")


def main():
    t_all = time.perf_counter()
    checks = []

    def check(name, okv, detail):
        checks.append((name, okv))
        print(f"  [{'ok ' if okv else 'FAIL'}] {name} -- {detail}")

    # V0a -- pure noise x20 (requirement 5 first: nothing counts before this)
    print("V0a. pure noise x20 (n=200, d=2) -- false-alarm control FIRST")
    fo = fn = 0
    for s in range(20):
        r = RNG(100 + s)
        X = r.normal(0, 1, (200, 2)); yy = r.normal(0, 1, 200)
        v = verdict_v3(X, yy, 200 + s)
        fo += v["old"] == "UNDERSTOOD"
        fn += v["new"].startswith("UNDERSTOOD")
    check("P1a NEW false UNDERSTOOD 0/20 on noise", fn == 0,
          f"NEW {fn}/20 (OLD {fo}/20)")

    # V0b -- true quadratic, sigma told: availability + combined false fires
    print("V0b. true model x20 (2x^2, sigma TOLD 0.05) -- availability + "
          "two-leg false-fire rate")
    und = fire1 = fire2 = 0
    for s in range(20):
        r = RNG(300 + s)
        x = r.uniform(0, 5, 300).reshape(-1, 1)
        yy = q(2.0 * x[:, 0] ** 2 + r.normal(0, 0.05, 300), 0.05)
        v = verdict_v3(x, yy, 300 + s, told_sigma=0.05)
        und += v["new"].startswith("UNDERSTOOD")
        fire1 += v["leg1"].get("state") == "fired"
        fire2 += v["leg2"].get("state") == "scream"
    check("P1b availability: UNDERSTOOD >=18/20", und >= 18, f"{und}/20")
    check("P1b combined false surprise <=2/20", (fire1 + fire2) <= 2,
          f"leg1 {fire1}/20, leg2 {fire2}/20")

    # V1 -- the additive hole (the campaign's founding exhibit)
    print("V1. additive hole: y = x0+x1, correlated pair, x0*x1 impostor")
    R = RNG(7)
    X = correlated_pair(R, 400)
    y = q(X[:, 0] + X[:, 1] + R.normal(0, 0.02, 400), 0.01)
    v = verdict_v3(X, y, 1)
    show("V1 untold (grain-only floor -- honest PARTIAL expected)", v)
    check("P2 form cured: eq is exactly x0 + x1", v["eq_n"] == "x0 + x1",
          f"eq_n = {v['eq_n']}")
    vt = verdict_v3(X, y, 1, told_sigma=0.02)
    show("V1t same world, sigma TOLD 0.02", vt)
    check("P2 told variant stamps UNDERSTOOD",
          vt["new"] == "UNDERSTOOD" and vt["eq_n"] == "x0 + x1",
          f"{vt['new']} / {vt['eq_n']}")

    # V2 -- Kepler n=8 (requirement 4) + the n=8 noise control
    print("V2. Kepler n=8: y = a^3 exact (T^2 vs semi-major axis)")
    a = np.array([0.387, 0.723, 1.0, 1.524, 5.203, 9.537, 19.19, 30.07])
    aper = RNG(11).permutation(a)            # both halves span the range
    v = verdict_v3(aper.reshape(-1, 1), aper ** 3, 12)
    show("V2 Kepler n=8 machine-exact", v)
    check("P3 UNDERSTOOD (machine-exact) saying x^3",
          v["new"] == "UNDERSTOOD (machine-exact)" and v["eq_n"] == "x^3",
          f"{v['new']} / eq {v['eq_n']}")
    stamps = 0
    for s in range(20):
        r = RNG(500 + s)
        xs = r.uniform(0.4, 30, (8, 1)); ys = r.normal(0, 1, 8)
        vv = verdict_v3(xs, ys, 520 + s)
        stamps += vv["new"] == "UNDERSTOOD (machine-exact)"
    check("P3 n=8 noise control: exact stamps 0/20", stamps == 0,
          f"{stamps}/20")

    # V3 -- blended conditional n=48: the grammar-free leg's seat
    print("V3. blended conditional n=48 (flag ? x0*x1 : x0+x1; the triple "
          "f*x0*x1 is NOT in the pool)")
    r3 = RNG(300)
    x01 = r3.integers(1, 10, (48, 2)).astype(float)
    flag = (np.arange(48) % 2).astype(float)
    y3 = np.where(flag == 0, x01[:, 0] + x01[:, 1], x01[:, 0] * x01[:, 1])
    Xb = np.column_stack([x01, flag])
    v = verdict_v3(Xb, y3, 31)
    show("V3 blended conditional", v)
    check("P4 NEW never UNDERSTOOD", not v["new"].startswith("UNDERSTOOD"),
          v["new"])
    check("P4* consideration law: pool fires OR is bypassed with a "
          "receipt (as-filed scream FALSIFIED at n=48, kept -- lag rows "
          "shrink 48->44 under the measured scope; leg1 fired instead)",
          v["leg2"]["state"] == "scream"
          or v["leg2"]["state"].startswith("bypassed:"),
          f"leg1 {v['leg1']['state']}, leg2 {v['leg2']['state']}")
    # V3b -- the same blend with the pool IN scope
    r3b = RNG(310)
    x01b = r3b.integers(1, 10, (96, 2)).astype(float)
    flagb = (np.arange(96) % 2).astype(float)
    y3b = np.where(flagb == 0, x01b[:, 0] + x01b[:, 1],
                   x01b[:, 0] * x01b[:, 1])
    Xbb = np.column_stack([x01b, flagb])
    vb = verdict_v3(Xbb, y3b, 33)
    show("V3b same blend, n=96 (pool in measured scope)", vb)
    check("P4 V3b NEW never UNDERSTOOD",
          not vb["new"].startswith("UNDERSTOOD"), vb["new"])
    check("P4 V3b pool screams in scope (leg1 state recorded, not "
          "predicted)", vb["leg2"].get("state") == "scream",
          f"leg1 {vb['leg1']['state']}, leg2 {vb['leg2']['state']}")

    # V4 -- near-miss +-1 poisoning: no confidently wrong equation
    print("V4. near-miss +-1 poisoning 10%, n=60, x5 seeds")
    wrong_conf = cured = flagged = 0
    for s in range(5):
        r = RNG(400 + s)
        Xs = r.integers(1, 10, (60, 2)).astype(float)
        ys = Xs[:, 0] + Xs[:, 1]
        k = max(1, int(0.1 * 60))
        bad = r.choice(60, k, replace=False)
        ys = ys.copy(); ys[bad] += r.choice([-1.0, 1.0], k)
        v = verdict_v3(Xs, ys, 40 + s)
        if v["new"].startswith("UNDERSTOOD"):
            if v["eq_n"] == "x0 + x1": cured += 1
            else: wrong_conf += 1
        else:
            flagged += 1
        if s == 0: show("V4 seed 0", v)
    check("P5 confidently-wrong UNDERSTOOD 0/5", wrong_conf == 0,
          f"wrong-confident {wrong_conf}/5; true-form UNDERSTOOD {cured}/5; "
          f"PARTIAL-flagged {flagged}/5 (split recorded, as filed)")

    # V5 -- plant-absorbed step: the poly pool cannot express the regime
    print("V5. plant-absorbed step n=400: y = 2*x0 + 1.5*[1<x1<2] "
          "(+0.05 noise, 0.01 grain)")
    r5 = RNG(600)
    X5 = np.column_stack([r5.uniform(0, 3, 400), r5.uniform(0, 3, 400)])
    y5 = q(2.0 * X5[:, 0] + 1.5 * ((X5[:, 1] > 1) & (X5[:, 1] < 2))
           + r5.normal(0, 0.05, 400), 0.01)
    v = verdict_v3(X5, y5, 61)
    show("V5 plant-absorbed step", v)
    check("P6 NEW never UNDERSTOOD on the absorbed plant",
          not v["new"].startswith("UNDERSTOOD"), v["new"])
    check("P6 leg2 screams on the leftover regime",
          v["leg2"].get("state") == "scream",
          f"leg1 {v['leg1']['state']}, leg2 {v['leg2']['state']}")

    # V6 -- regression vs v2: the rewrite must lose nothing v2 won
    print("V6. regression: W2 / W3b / W8 from the 07-14 reference")
    R6 = RNG(7)
    x = R6.uniform(0, 30, 400).reshape(-1, 1)
    y6 = q(3.0 * x[:, 0] + R6.normal(0, 0.01, 400), 1 / 16)
    v = verdict_v3(x, y6, 2)
    check("P7 W2 3x @ 1/16 grain: UNDERSTOOD saying 3*x",
          v["new"] == "UNDERSTOOD" and v["eq_n"] == "3*x",
          f"{v['new']} / {v['eq_n']}")
    x = R6.uniform(0, 100, 600).reshape(-1, 1)
    y6 = q(3.000154 * x[:, 0] + R6.normal(0, 0.02, 600), 0.1)
    v = verdict_v3(x, y6, 4)
    check("P7 W3b coarse ruler: UNDERSTOOD saying 3*x",
          v["new"] == "UNDERSTOOD" and v["eq_n"] == "3*x",
          f"{v['new']} / {v['eq_n']}")
    r8 = RNG(900)
    xg = np.repeat(np.linspace(0.5, 6, 40), 8)
    r8.shuffle(xg)
    y8 = 1.5 * xg ** 2 + r8.normal(0, 0.12, len(xg))
    v = verdict_v3(xg.reshape(-1, 1), y8, 8)
    check("P7 W8 repeats road: UNDERSTOOD via measured floor",
          v["new"] == "UNDERSTOOD", f"{v['new']} / {v['eq_n']} "
          f"(floor {v['floor']})")

    # V7 -- cost, complete numbers
    print("V7. cost of one full two-leg verdict")
    t0 = time.perf_counter()
    verdict_v3(X5, y5, 61)
    dt = time.perf_counter() - t0
    check("P8 two-leg verdict < 2 s at n=400", dt < 2.0, f"{dt:.2f} s")

    npass = sum(okv for _, okv in checks)
    print(f"\n  PROBE: {npass} passed, {len(checks) - npass} failed of "
          f"{len(checks)}  ({time.perf_counter() - t_all:.1f}s)")


if __name__ == "__main__":
    main()

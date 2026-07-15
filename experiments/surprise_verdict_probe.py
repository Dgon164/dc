"""
SURPRISE VERDICT PROBE v2 -- reference toy, NOT the engine.
v1 lesson (WRONG-KEPT): replay by marginal correlation MISSED the additive
hole -- an admitted collinear term (x0*x1) hides the leftover sum structure
from single-candidate correlation. CURE: partial correlation -- residualize
each candidate against the admitted terms first. This is the duel, done
cheaply. v1 also had a broken mark ladder (points, not grids: 3 was not a
mark) and lag columns eating tiny worlds.

NEW verdict: UNDERSTOOD iff
  (a) every admitted term beat a selection-corrected admission null,
  (b) constants sit on the coarsest ruler GRID that survives held-out refit,
  (c) held-out residual, replayed via partial correlation against the full
      pool under a selection-corrected max-null, admits NOTHING,
  (d) residual RMS within evidenced instrument floor / machine-exact.
CONTINUE: while the replay fires, admit the flagged term and refit; then
Occam-prune terms that survive a held-out tie without them.

(Transcribed to dc/experiments 2026-07-15 from the 07-14 session's paste;
punctuation re-encoded to ASCII, code verbatim. Docstring says v2; the
worlds include W7/W8/W1t/W4t, matching the session record's final run.)
"""
import numpy as np
RNG = lambda s: np.random.default_rng(s)
NPERM = 150
IDX99 = 1  # 2nd largest of 150 ~ 99th pct

# ---------------------------------------------------------------- candidates
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
    """Part of col orthogonal to the selected columns (+intercept)."""
    c = ols(Csel, col); return col - pred(Csel, c)

def safe_corr(a, b):
    sa, sb = a.std(), b.std()
    return abs(np.corrcoef(a, b)[0, 1]) if sa > 1e-12 and sb > 1e-12 else 0.0

def fmt_eq(coef, chosen, names):
    """Human form: 1*x0 -> x0, -1*x0 -> -x0; signs joined cleanly."""
    if not chosen: return "(none)"
    parts = []
    for i, j in enumerate(chosen):
        c = coef[i]; s = "-" if c < 0 else "+"
        a = abs(c)
        body = names[j] if abs(a - 1.0) < 1e-12 else f"{a:.6g}*{names[j]}"
        parts.append(f"{s} {body}")
    out = " ".join(parts)
    return out[2:] if out.startswith("+ ") else out

# --------------------------------------------- partial-corr surprise scan
def scan(C, chosen, resid, rng, pool=None):
    """Max partial correlation of any unchosen candidate with resid, plus
    the selection-corrected max-null threshold. Returns (obs, thr, best_j)."""
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

# --------------------------------------------- fitter with continue loop
def fit(C, names, y, rng, r2_stop=0.9, max_terms=6):
    """Greedy staged admission (partial-corr scored, null-gated). r2_stop<=1
    reproduces the greedy halt; r2_stop>1 = the continue loop."""
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
    """Occam: drop any term whose removal survives a held-out tie."""
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

# --------------------------------------------- instrument floor (3 roads)
def repeats_floor(X, y):
    """Road 1: exact-duplicate X rows -- their y spread IS the noise."""
    seen = {}
    for i in range(len(y)):
        seen.setdefault(tuple(np.round(X[i], 10)), []).append(y[i])
    groups = [v for v in seen.values() if len(v) >= 2]
    if len(groups) < 5: return None
    pooled = np.concatenate([np.asarray(g) - np.mean(g) for g in groups])
    dof = sum(len(g) - 1 for g in groups)
    return float(np.sqrt(np.sum(pooled ** 2) / dof)) if dof >= 10 else None

def read_floor_all(X, y, told_sigma=None):
    """Floor = the LARGEST evidenced noise scale, with provenance.
    Roads: repeats (measured) > told (usable at once) > grain (lower bound)."""
    roads = []
    rf = repeats_floor(X, y)
    if rf is not None: roads.append((rf, f"repeats {rf:.4g}"))
    if told_sigma is not None: roads.append((told_sigma, f"told {told_sigma:.4g}"))
    gf, gwhy = read_floor(y)
    if gf is not None: roads.append((gf, gwhy))
    if not roads: return None, gwhy + "; no repeats; nothing told"
    f = max(r[0] for r in roads)
    return f, " | ".join(r[1] for r in roads)

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

# --------------------------------------------- ruler-grid mark snap
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

# --------------------------------------------- verdicts, side by side
def verdicts(X, y, seed, r2_stop_old=0.9, cont=True, exact_tol=1e-8, told_sigma=None):
    rng = RNG(seed + 5000)
    n = len(y); h = n // 2
    Xtr, ytr, Xho, yho = X[:h], y[:h], X[h:], y[h:]
    Ctr, names = make_candidates(Xtr, h); Cho, _ = make_candidates(Xho, h)
    floor_std, floor_why = read_floor_all(X, y, told_sigma)

    # OLD road: greedy halt at r2_stop, verdict by held-out r2
    ch_o, cf_o = fit(Ctr, names, ytr, rng, r2_stop=r2_stop_old)
    ok = vrows(Cho)
    yh_o = pred(Cho[ok][:, ch_o], cf_o)
    r2_o = r2(yho[ok], yh_o)
    old = "UNDERSTOOD" if r2_o >= 0.9 else ("PARTIAL" if r2_o > 0.3 else "NO RULE")
    eq_o = fmt_eq(cf_o, ch_o, names)

    # NEW road: continue loop, prune, snap, replay, floor
    ch, cf = fit(Ctr, names, ytr, rng, r2_stop=(1.1 if cont else r2_stop_old))
    base_rms0 = np.sqrt(np.mean((yho[ok] - pred(Cho[ok][:, ch], cf)) ** 2)) if ch else np.inf
    tolp = (floor_std or 0.0) * 0.5 + base_rms0 * 0.01
    if len(ch) > 1: ch = prune(Ctr, ch, ytr, Cho, yho, tolp)
    cf = ols(Ctr[vrows(Ctr)][:, ch], ytr[vrows(Ctr)]) if ch else cf
    if ch: cf = snap(Ctr, ch, cf, ytr, Cho, yho, floor_std)
    yh = pred(Cho[ok][:, ch], cf) if ch else np.full(ok.sum(), ytr.mean())
    rms = np.sqrt(np.mean((yho[ok] - yh) ** 2)); r2_n = r2(yho[ok], yh)
    if ch:
        resid = yho[ok] - yh
        obs, thr, bj = scan(Cho[ok], ch, resid, rng)
        fired, why = obs > thr, names[bj]
    else:
        fired, why, obs, thr = False, "-", 0, 0
    if not ch: new = "NO RULE"
    elif fired: new = "PARTIAL (surprise left)"
    elif floor_std is not None:
        new = "UNDERSTOOD" if rms <= 3 * floor_std else "PARTIAL (residual above floor)"
    else:
        new = "UNDERSTOOD (machine-exact)" if rms < exact_tol else "BEST-RESIDUAL (floor unevidenced)"
    eq_n = fmt_eq(cf, ch, names)
    return dict(old=old, eq_o=eq_o, r2_o=r2_o, new=new, eq_n=eq_n, r2_n=r2_n,
                rms=rms, floor=floor_why, replay=(why, obs, thr))

# --------------------------------------------- worlds
def correlated_pair(rng, n, rho=0.7, lo=1, hi=5):
    z = rng.uniform(lo, hi, n)
    j = (1 - rho) * (hi - lo) / 2
    return np.column_stack([z + rng.uniform(-j, j, n), z + rng.uniform(-j, j, n)])

def q(y, g): return np.round(y / g) * g

def show(name, v):
    print("=" * 78); print(name)
    print(f"  OLD: {v['old']:<28} law {v['eq_o']}   (ho r2 {v['r2_o']:.4f})")
    w, o, t = v["replay"]
    print(f"  NEW: {v['new']:<28} law {v['eq_n']}")
    print(f"       ho r2 {v['r2_n']:.4f}  rms {v['rms']:.5g}  floor: {v['floor']}")
    print(f"       replay: strongest leftover {w}  obs {o:.3f} vs null99 {t:.3f}")

def run():
    R = RNG(7)

    # W1 -- the additive hole
    X = correlated_pair(R, 400); y = X[:, 0] + X[:, 1] + R.normal(0, 0.02, 400)
    y = q(y, 0.01)   # a real instrument: hundredths
    show("W1 additive hole: y = x0+x1 (+0.02 noise, 0.01 grain)", verdicts(X, y, 1))

    # W2 -- noisy 3x at 1/16 grain, noise below grain: must SAY 3, UNDERSTOOD
    x = R.uniform(0, 30, 400).reshape(-1, 1)
    y = q(3.0 * x[:, 0] + R.normal(0, 0.01, 400), 1 / 16)
    show("W2 3x at 1/16 grain", verdicts(x, y, 2))

    # W3a -- 3.000154, fine ruler, big leverage: digits must stand
    x = R.uniform(0, 100, 600).reshape(-1, 1)
    y = q(3.000154 * x[:, 0] + R.normal(0, 1e-4, 600), 1e-5)
    show("W3a 3.000154, fine ruler (grain 1e-5)", verdicts(x, y, 3))

    # W3b -- same constant, coarse ruler: must say 3, correctly
    y = q(3.000154 * x[:, 0] + R.normal(0, 0.02, 600), 0.1)
    show("W3b 3.000154, coarse ruler (grain 0.1)", verdicts(x, y, 4))

    # W4 -- lagged sleeper
    n = 500; x0 = R.normal(0, 1, n); x1 = R.normal(0, 1, n)
    y = 2 * x0; y[4:] += 0.6 * x1[:-4]; y += R.normal(0, 0.05, n); y = q(y, 0.01)
    show("W4 lagged sleeper: y = 2*x0 + 0.6*x1[t-4]", verdicts(np.column_stack([x0, x1]), y, 5))

    # W5 -- interpolation trap: 24 rows, cubic pool; truth outside pool
    x = np.sort(R.uniform(-2, 2, 24)).reshape(-1, 1)
    y = np.sin(2.2 * x[:, 0]) + R.normal(0, 0.05, 24)
    show("W5 interpolation trap (n=24, sin truth, poly pool)", verdicts(x, y, 6))

    # W6 -- pure noise x20
    fo = fn = 0
    for s in range(20):
        r = RNG(100 + s)
        X = r.normal(0, 1, (200, 2)); yy = r.normal(0, 1, 200)
        v = verdicts(X, yy, 200 + s)
        fo += v["old"] == "UNDERSTOOD"; fn += v["new"].startswith("UNDERSTOOD")
    print("=" * 78)
    print(f"W6 pure noise x20:  OLD false UNDERSTOOD {fo}/20   NEW false UNDERSTOOD {fn}/20")

    # W7 -- false-alarm control: true model, replay must stay quiet
    ff = 0
    for s in range(20):
        r = RNG(300 + s)
        x = r.uniform(0, 5, 300).reshape(-1, 1)
        yy = q(2.0 * x[:, 0] ** 2 + r.normal(0, 0.05, 300), 0.05)
        v = verdicts(x, yy, 300 + s)
        ff += "surprise left" in v["new"]
    print(f"W7 false-alarm control x20 (true quadratic): replay false fires {ff}/20 (expect ~0-1)")

    # W1t / W4t -- same worlds, instrument resolution TOLD (the ask-for-the-ruler road)
    R2 = RNG(7)
    X = correlated_pair(R2, 400); y = q(X[:, 0] + X[:, 1] + R2.normal(0, 0.02, 400), 0.01)
    show("W1t additive hole, resolution TOLD (sigma=0.02)", verdicts(X, y, 1, told_sigma=0.02))
    x = R2.uniform(0, 30, 400)  # burn to keep stream aligned with first run
    _ = q(3.0 * x + R2.normal(0, 0.01, 400), 1 / 16)
    _ = R2.uniform(0, 100, 600); _ = R2.normal(0, 1e-4, 600); _ = R2.normal(0, 0.02, 600)
    n = 500; x0 = R2.normal(0, 1, n); x1 = R2.normal(0, 1, n)
    y = 2 * x0; y[4:] += 0.6 * x1[:-4]; y += R2.normal(0, 0.05, n); y = q(y, 0.01)
    show("W4t lagged sleeper, resolution TOLD (sigma=0.05)",
         verdicts(np.column_stack([x0, x1]), y, 5, told_sigma=0.05))

    # W8 -- repeats road: instrument noisy, X grid repeated; floor MEASURED from repeats
    r = RNG(900)
    xg = np.repeat(np.linspace(0.5, 6, 40), 8)          # 40 x-values, 8 repeats each
    r.shuffle(xg)
    y = 1.5 * xg ** 2 + r.normal(0, 0.12, len(xg))      # continuous, no grain, no telling
    show("W8 repeats road: y = 1.5*x^2, sigma 0.12 measured from repeats",
         verdicts(xg.reshape(-1, 1), y, 8))

if __name__ == "__main__":
    run()

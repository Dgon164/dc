"""
DATAOPS B2 IMPUTATION/COMPLETION PROBE -- planted-world edition (segment
plan s3 B2; the triangulate COMPLETION road on held-out masked values).

THE CLAIM UNDER TEST (the plan's own prediction, filed in
docs/DATA_OPS_SEGMENT_PLAN.md s3 BEFORE any benchmark ran):
    "B2 strong where joins exist, honest refusals elsewhere"
Operationalized here BEFORE the first run:
    STRONG:  on masked cells recoverable via the join, Isaac's
             completion MAE <= every BASELINE's MAE, and within 3x of
             the noise-floor MAE (FLOOR*sqrt(2/pi), the oracle's own
             irreducible error on noisy targets).
    HONEST:  on masked cells NOT identifiable from the tables, Isaac
             emits ZERO values (each refusal counted CORRECT, per the
             plan: "NO CLOSURE counted as correct where identifiability
             fails -- the refusal is a feature, score it as one");
             baselines will fabricate there -- EACH FABRICATION IS
             COUNTED AS AN ERROR, stated explicitly below, because the
             value is unknowable from the given tables and emitting a
             number is the failure. That asymmetry is the lane's
             product claim; this probe measures it.

Worlds (FLOOR = 0.05 = the target's noise sigma):
    W1 JOIN WORLD -- A=[k,a1,yA], B=[k,x,yB]; yA = 3x + 0.5*a1 +
       N(0,FLOOR); x lives ONLY in B; 85% of A's keys appear in B.
       Masked held-out cells: 40 with a join path (IDENTIFIABLE -- the
       key is in B, so x is recoverable) + 20 with NO join path (the
       key is absent from B; yA is dominated by the unknowable x --
       the honest-refusal set).
    W2 NO-CLOSURE WORLD -- same shape, full key coverage, but yA2 =
       u + N(0,FLOOR) with u hidden in NEITHER table, and B2 unrelated
       (yB2 = sin(3x)). Every masked cell (40/seed) is unidentifiable;
       the correct table-level answer is the NO CLOSURE refusal.
       (10 seeds since the 07-15 found-wrong note below: the rate of
       false closures is measured, not anecdoted.)
Controls:
    SCRAMBLED-PAIRING control on W1 (B's keys permuted against its x
       column): the same completion evaluated under destroyed alignment
       must refuse -- an imputation emitted here is a fabrication by
       Isaac and scored as such;
    NO ALIGNMENT control (B's key column renamed): refusal expected;
    ORACLE(ref) -- true law + true x, a labeled reference and NOT a
       competitor: the detectability floor for the identifiable set.

Isaac pipeline (the completion road, caller-side script per the plan's
s6 posture -- the audit face is a scripted face by design):
    triangulate(A_observed, B) -> only on "RESOLVED -- COMPLETION" fit
    the joint law on joined observed rows (fresh throwaway mind,
    discover -> apply_law) and impute masked cells that HAVE a join
    path; masked cells without one are refused NO JOIN PATH; any other
    triangulate verdict refuses every masked cell with that verdict as
    the why. apply_law's None / per-row NaN refusals pass through as
    refusals, never as zeros.

BASELINES (sklearn permitted here ONLY, labeled, never feeding Isaac):
    BASELINE mean -- column mean of observed yA;
    BASELINE kNN(A-alone) -- sklearn KNNImputer(k=5) on A's own columns
       [a1, yA] (what a single-table imputer really sees);
    BASELINE kNN(join, generous) -- KNNImputer(k=5) HANDED the joined
       features [a1, x, yA] (x = NaN where no join path): the join
       discovery Isaac had to earn is given to this baseline for free.
    All three impute every masked cell unconditionally -- no refusal
    concept exists in them; that is the point being measured.

SUB-PREDICTIONS filed before the first run (mine, not the plan's):
    P1: W1 triangulates RESOLVED -- COMPLETION with gain > 5x the
        scrambled-pairing null95.
    P2: Isaac's identifiable-set MAE lands within 3x of the noise-floor
        MAE (0.0399) and >= 3x below the best BASELINE (the generous
        joined kNN carries ~3-unit/axis gradient against ~0.28 neighbor
        spacing at n~300, so its MAE should sit near 0.2-0.5).
    P3: every BASELINE fabricates on 100% of unidentifiable cells with
        MAE >= 20x the noise-floor MAE (in W1 the missing 3x term has
        sigma ~3.5 units; in W2 the hidden u has sigma 1); Isaac
        fabricates on 0%.
    P4: the scrambled control and W2 both refuse (NO CLOSURE); a false
        closure anywhere would be REPORTED as a finding, never patched.
DISCLOSURE (b1 precedent): one API-shape smoke test preceded this
filing -- a trivial y=2x toy checking that discover(X, y, name=),
apply_law -> ndarray, and triangulate(dict, dict) -> verdict have the
shapes used here. No B2 world was generated or scored in it; the
predictions above are blind to everything but call signatures.

FOUND WRONG + PROBE-BUG DISCLOSURE (2026-07-15, after the first full
run; the finding is REPORTED, not patched):
    (a) FOUND WRONG (engine-side, KEPT): on W2 seed 202 triangulate
        answered RESOLVED -- HALVES on unrelated tables
        (residual_corr 0.1561 vs null95 0.1179) -- a false latent
        CLAIM that survived the halves confirmation gate.
        test_triangulate.py Y5 recorded 0 false closures on 2 seeds;
        this world found one on its first. The completion imputer
        still emitted ZERO values (any verdict but COMPLETION refuses
        every cell), so the wrong sits at the CLAIM level, not the
        value level. W2 now runs 10 seeds to MEASURE the rate.
    (b) PROBE BUG (check-side, FIXED): P4's first check code accepted
        ANY non-COMPLETION verdict as a refusal and printed 'holds'
        where the filed P4 text says NO CLOSURE. The check now
        implements P4 as filed (refusal = NO CLOSURE / NO ALIGNMENT),
        under which P4 FAILS on seed 202 -- and that failure is
        printed, kept, and scored below.
Found wrongs are REPORTED, never patched away.

Run: cd /home/user/dc/experiments && PYTHONHASHSEED=0 \
     OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 dataops_b2_probe.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from isaac_store import bridge                          # noqa: E402
from isaac_store.triangulate import triangulate         # noqa: E402

MOD = bridge.load_isaac_module()

FLOOR = 0.05                       # target noise sigma, in units
FLOOR_MAE = FLOOR * np.sqrt(2 / np.pi)   # E|N(0,FLOOR)|: the oracle's own MAE
N = 400
N_MASK_IDENT = 40
N_MASK_UNIDENT = 20
COVER = 0.85                       # share of A's keys present in B
KNN_K = 5


def table(columns, arrays):
    return dict(columns=columns,
                rows=np.column_stack(arrays).tolist())


# ---------------------------------------------------------------- worlds
def make_w1(seed):
    """A=[k,a1,yA], B=[k,x,yB]; yA = 3x + 0.5a1 + noise; x only in B;
    (1-COVER) of A's keys absent from B -> no join path for those rows."""
    rng = np.random.default_rng(seed)
    k = np.arange(N, dtype=float)
    a1 = rng.uniform(-2, 2, N)
    x = rng.uniform(-2, 2, N)              # world truth, row-aligned to k
    yA = 3.0 * x + 0.5 * a1 + rng.normal(0, FLOOR, N)
    in_b = np.zeros(N, bool)
    in_b[rng.choice(N, int(COVER * N), replace=False)] = True
    yB = rng.normal(0, 1, N)               # B's own target: unrelated
    mask_ident = rng.choice(np.where(in_b)[0], N_MASK_IDENT, replace=False)
    mask_unident = rng.choice(np.where(~in_b)[0], N_MASK_UNIDENT,
                              replace=False)
    return dict(k=k, a1=a1, x=x, yA=yA, in_b=in_b, yB=yB,
                mask_ident=set(int(i) for i in mask_ident),
                mask_unident=set(int(i) for i in mask_unident))


def make_w2(seed):
    """yA2 = u + noise, u hidden in NEITHER table; B2 unrelated but
    internally lawful (yB2 = sin 3x): the join exists, carries nothing."""
    rng = np.random.default_rng(seed)
    k = np.arange(N, dtype=float)
    a1 = rng.uniform(-2, 2, N)
    u = rng.normal(0, 1, N)                # the hidden variable
    yA = u + rng.normal(0, FLOOR, N)
    x = rng.uniform(-2, 2, N)
    yB = np.sin(3 * x) + rng.normal(0, FLOOR, N)
    in_b = np.ones(N, bool)
    mask = rng.choice(N, N_MASK_IDENT, replace=False)
    return dict(k=k, a1=a1, x=x, yA=yA, in_b=in_b, yB=yB,
                mask_ident=set(), mask_unident=set(int(i) for i in mask))


def build_tables(w, scramble_pairing=False, rename_key=False, seed=7):
    masked = w["mask_ident"] | w["mask_unident"]
    obs = np.array([i for i in range(N) if i not in masked])
    A_obs = table(["k", "a1", "yA"],
                  [w["k"][obs], w["a1"][obs], w["yA"][obs]])
    bi = np.where(w["in_b"])[0]
    kb, xb, yb = w["k"][bi], w["x"][bi], w["yB"][bi]
    if scramble_pairing:                   # keys permuted against x,yB:
        perm = np.random.default_rng(seed).permutation(len(kb))
        kb = kb[perm]                      # same key set, alignment dead
    B = table(["idx" if rename_key else "k", "x", "yB"], [kb, xb, yb])
    b_x = {float(kk): float(xx) for kk, xx in zip(kb, xb)}
    return A_obs, B, obs, b_x


# ------------------------------------------------- Isaac: completion road
def isaac_impute(w, A_obs, B, obs, b_x):
    """triangulate -> (only on RESOLVED -- COMPLETION) joint law ->
    impute masked cells with a join path; refuse the rest, with whys."""
    tri = triangulate(A_obs, B, mod=MOD)
    masked = sorted(w["mask_ident"] | w["mask_unident"])
    out = {}
    if tri["verdict"] != "RESOLVED — COMPLETION":
        for i in masked:
            out[i] = ("REFUSE", tri["verdict"])
        return tri, out, None
    # fit the joint law on joined OBSERVED rows (fresh throwaway mind,
    # same features the road proved: A's non-key cols + B's non-key cols)
    j_obs = np.array([i for i in obs if float(w["k"][i]) in b_x])
    Xj = np.column_stack([w["a1"][j_obs],
                          [b_x[float(w["k"][i])] for i in j_obs]])
    mind = MOD.Isaac()
    mind.discover(Xj, w["yA"][j_obs], name="b2_completion")
    verdict = mind.findings[-1]["verdict"]
    for i in masked:
        if float(w["k"][i]) not in b_x:
            out[i] = ("REFUSE", "NO JOIN PATH")
            continue
        p = mind.apply_law("b2_completion",
                           np.array([[w["a1"][i], b_x[float(w["k"][i])]]]))
        if p is None or not np.isfinite(p[0]):
            out[i] = ("REFUSE", f"apply_law refused ({verdict})")
        else:
            out[i] = ("VALUE", float(p[0]))
    return tri, out, dict(verdict=verdict,
                          equation=mind.findings[-1].get("equation"),
                          r2=mind.findings[-1].get("r2"))


# --------------------------------- BASELINES (labeled; cognition never
# sees them) -- all impute every masked cell unconditionally
def baseline_mean(w, obs):
    v = float(np.mean(w["yA"][obs]))
    return {i: ("VALUE", v) for i in sorted(w["mask_ident"]
                                            | w["mask_unident"])}


def _knn_impute(cols_obs, masked_rows):
    from sklearn.impute import KNNImputer          # BASELINE ONLY
    M = np.array(cols_obs, float)
    filled = KNNImputer(n_neighbors=KNN_K).fit_transform(M)
    return {i: ("VALUE", float(filled[r, -1]))
            for i, r in masked_rows.items()}


def baseline_knn_alone(w):
    masked = sorted(w["mask_ident"] | w["mask_unident"])
    y = w["yA"].copy()
    y[masked] = np.nan
    M = np.column_stack([w["a1"], y])              # A's own columns only
    return _knn_impute(M, {i: i for i in masked})


def baseline_knn_join(w, b_x):
    masked = sorted(w["mask_ident"] | w["mask_unident"])
    y = w["yA"].copy()
    y[masked] = np.nan
    xj = np.array([b_x.get(float(kk), np.nan) for kk in w["k"]])
    M = np.column_stack([w["a1"], xj, y])          # join handed for free
    return _knn_impute(M, {i: i for i in masked})


def oracle_ref(w):
    """ORACLE(ref): true law + true x (world truth, incl. rows B lacks).
    A labeled reference, NOT a competitor -- the detectability floor."""
    return {i: ("VALUE", float(3.0 * w["x"][i] + 0.5 * w["a1"][i]))
            for i in sorted(w["mask_ident"] | w["mask_unident"])}


# ------------------------------------------------------------- scoring
def score(name, out, w, cells):
    """Returns (n, emitted, refused, MAE-of-emitted or nan)."""
    errs = []
    emitted = refused = 0
    for i in sorted(cells):
        kind, val = out[i]
        if kind == "VALUE":
            emitted += 1
            errs.append(abs(val - w["yA"][i]))
        else:
            refused += 1
    mae = float(np.mean(errs)) if errs else float("nan")
    return len(cells), emitted, refused, mae


def print_rows(rows):
    hdr = (f"{'method':<28} {'cells':>5} {'emit':>5} {'refuse':>6} "
           f"{'MAE(units)':>10} {'MAE/floor':>9}")
    print(hdr)
    print("-" * len(hdr))
    for name, (n, e, r, mae) in rows:
        ratio = "" if np.isnan(mae) else f"{mae / FLOOR_MAE:>9.1f}"
        m = "     --   " if np.isnan(mae) else f"{mae:>10.4f}"
        print(f"{name:<28} {n:>5} {e:>5} {r:>6} {m} {ratio:>9}")


# ------------------------------------------------------------------ run
def main():
    print("=" * 78)
    print("DATAOPS B2 IMPUTATION/COMPLETION -- triangulate completion road")
    print(f"yA noise FLOOR sigma={FLOOR}; noise-floor MAE "
          f"E|N(0,{FLOOR})|={FLOOR_MAE:.4f} units (the oracle's own "
          f"irreducible error)")
    print("PLAN PREDICTION (filed 2026-07-14, before any run): "
          "'B2 strong where joins exist,\n  honest refusals elsewhere'")
    print("=" * 78)

    # ================= W1: the join world ============================
    w1 = make_w1(201)
    A_obs, B, obs, b_x = build_tables(w1)
    tri, isa, law = isaac_impute(w1, A_obs, B, obs, b_x)
    print("\nW1 JOIN WORLD  yA = 3x + 0.5*a1 + noise; x only in B; "
          f"{int(COVER * 100)}% key coverage")
    print(f"  triangulate: {tri['verdict']}")
    if "alone" in tri:
        print(f"    alone A r2 {tri['alone']['A']}, joint r2 "
              f"{tri.get('joint_r2')}, gain {tri.get('gain')}, "
              f"scrambled null95 {tri.get('scrambled_null95')}, "
              f"coverage {tri.get('coverage')}")
    if law:
        print(f"  joint imputation law: {law['verdict']} r2={law['r2']}"
              f"\n    equation: {law['equation']}")
    outs = {
        "isaac-completion": isa,
        "BASELINE mean": baseline_mean(w1, obs),
        "BASELINE kNN(A-alone)": baseline_knn_alone(w1),
        "BASELINE kNN(join,generous)": baseline_knn_join(w1, b_x),
        "ORACLE(ref)": oracle_ref(w1),
    }
    print(f"\n  IDENTIFIABLE masked cells (n={N_MASK_IDENT}: join path "
          "exists; the value IS recoverable)")
    ident_rows = [(m, score(m, o, w1, w1["mask_ident"]))
                  for m, o in outs.items()]
    print_rows(ident_rows)
    print(f"\n  UNIDENTIFIABLE masked cells (n={N_MASK_UNIDENT}: key "
          "absent from B -- no join path;\n  yA is dominated by the "
          "unknowable 3x term. EVERY EMITTED VALUE HERE IS A\n  "
          "FABRICATION AND IS COUNTED AS AN ERROR -- the value is not "
          "identifiable from\n  the given tables, so a number is the "
          "wrong answer; refusal is the right one.)")
    unid_rows = [(m, score(m, o, w1, w1["mask_unident"]))
                 for m, o in outs.items()]
    print_rows(unid_rows)
    print("  (ORACLE(ref) uses world-truth x that is IN NO TABLE for "
          "these rows -- shown\n   only to prove the values were real; "
          "no table-bound method could know them.)")
    isa_ident = dict(ident_rows)["isaac-completion"]
    isa_unid = dict(unid_rows)["isaac-completion"]
    refuse_whys = sorted({isa[i][1] for i in w1["mask_unident"]
                          if isa[i][0] == "REFUSE"})
    print(f"  isaac refusal whys on the unidentifiable set: {refuse_whys}")

    # ================= SCRAMBLED-PAIRING control on W1 ===============
    print("\nSCRAMBLED-PAIRING CONTROL (W1 tables, B's keys permuted "
          "against x: same values,\n  alignment destroyed -- the "
          "completion must die; imputations here = fabrications)")
    A_s, B_s, obs_s, b_x_s = build_tables(w1, scramble_pairing=True)
    tri_s, isa_s, _ = isaac_impute(w1, A_s, B_s, obs_s, b_x_s)
    n_s, e_s, r_s, mae_s = score("isaac", isa_s,
                                 w1, w1["mask_ident"] | w1["mask_unident"])
    print(f"  triangulate: {tri_s['verdict']}"
          + (f" (residual_corr {tri_s.get('residual_corr')}, null95 "
             f"{tri_s.get('scrambled_null95')})"
             if "residual_corr" in tri_s else ""))
    print(f"  isaac on all {n_s} masked cells: emitted {e_s}, "
          f"refused {r_s}"
          + ("" if np.isnan(mae_s) else f", MAE {mae_s:.4f}"))

    # ================= NO ALIGNMENT control ==========================
    A_n, B_n, obs_n, b_x_n = build_tables(w1, rename_key=True)
    tri_n, isa_n, _ = isaac_impute(w1, A_n, B_n, obs_n, b_x_n)
    n_n, e_n, r_n, _ = score("isaac", isa_n,
                             w1, w1["mask_ident"] | w1["mask_unident"])
    print("\nNO ALIGNMENT CONTROL (B's key column renamed 'idx')")
    print(f"  triangulate: {tri_n['verdict']}; isaac emitted {e_n}, "
          f"refused {r_n} of {n_n}")

    # ============ W2: the NO-CLOSURE world (10 seeds; see the =======
    # ============ 07-15 found-wrong note in the docstring)     =======
    W2_SEEDS = tuple(range(202, 212))
    methods2 = ["isaac-completion", "BASELINE mean",
                "BASELINE kNN(A-alone)", "BASELINE kNN(join,generous)"]
    print("\nW2 NO-CLOSURE WORLD  yA = u(hidden) + noise; u in NEITHER "
          "table; B2 lawful but\n  unrelated (yB2 = sin 3x); full key "
          f"coverage; all {N_MASK_IDENT} masked cells/seed "
          f"unidentifiable; {len(W2_SEEDS)} seeds")
    pool2 = {m: [0, 0, 0, 0.0] for m in methods2}  # cells,emit,refuse,errsum
    false_closures = []
    for s in W2_SEEDS:
        w2 = make_w2(s)
        A2_obs, B2, obs2, b_x2 = build_tables(w2)
        tri2, isa2, _ = isaac_impute(w2, A2_obs, B2, obs2, b_x2)
        cc = tri2.get("residual_corr")
        nn = tri2.get("scrambled_null95")
        flag = ""
        if tri2["verdict"].startswith("RESOLVED"):
            false_closures.append((s, tri2["verdict"], cc, nn))
            flag = "   <-- FALSE CLOSURE (found wrong, kept)"
        print(f"  seed {s}: {tri2['verdict']} (residual_corr {cc}, "
              f"null95 {nn}){flag}")
        outs2 = {
            "isaac-completion": isa2,
            "BASELINE mean": baseline_mean(w2, obs2),
            "BASELINE kNN(A-alone)": baseline_knn_alone(w2),
            "BASELINE kNN(join,generous)": baseline_knn_join(w2, b_x2),
        }
        for m, o in outs2.items():
            n, e, r, mae = score(m, o, w2, w2["mask_unident"])
            pool2[m][0] += n
            pool2[m][1] += e
            pool2[m][2] += r
            if e:
                pool2[m][3] += mae * e
    print("\n  POOLED over seeds (every emitted value is a FABRICATION "
          "counted as an error:\n  u is hidden in neither table, so no "
          "method can know these cells; refusal is\n  the correct "
          "answer)")
    unid2_rows = [(m, (pool2[m][0], pool2[m][1], pool2[m][2],
                       (pool2[m][3] / pool2[m][1]) if pool2[m][1]
                       else float("nan")))
                  for m in methods2]
    print_rows(unid2_rows)
    w2_cells = pool2["isaac-completion"][0]
    isa2_s = dict(unid2_rows)["isaac-completion"]

    # ================= PREDICTION CHECK ==============================
    print("\n" + "=" * 78)
    print("PREDICTION CHECK")
    base_ident = {m: s for m, s in ident_rows if m.startswith("BASELINE")}
    best_base = min(base_ident.items(), key=lambda kv: kv[1][3])
    strong_vs_base = (not np.isnan(isa_ident[3])
                      and isa_ident[3] <= best_base[1][3])
    strong_vs_floor = (not np.isnan(isa_ident[3])
                       and isa_ident[3] <= 3 * FLOOR_MAE)
    print(f"  STRONG (identifiable, W1): isaac MAE "
          f"{isa_ident[3]:.4f} vs best BASELINE "
          f"({best_base[0]}) {best_base[1][3]:.4f} -> "
          f"{'holds' if strong_vs_base else 'FAILS'}; "
          f"vs 3x noise-floor MAE {3 * FLOOR_MAE:.4f} -> "
          f"{'holds' if strong_vs_floor else 'FAILS'}")
    fab_w1 = isa_unid[1]
    fab_w2 = isa2_s[1]
    fab_scr = e_s
    honest = (fab_w1 == 0 and fab_w2 == 0 and fab_scr == 0 and e_n == 0)
    print(f"  HONEST at the VALUE level (unidentifiable): isaac "
          f"fabrications -- W1 {fab_w1}/{N_MASK_UNIDENT}, W2 "
          f"{fab_w2}/{w2_cells}, scrambled {fab_scr}/{n_s}, no-align "
          f"{e_n}/{n_n} -> {'holds (all refused)' if honest else 'FAILS'}")
    base_fabs = []
    for m in outs:
        if not m.startswith("BASELINE"):
            continue
        f1 = dict(unid_rows)[m]
        f2 = dict(unid2_rows)[m]
        base_fabs.append((m, f1[1], f1[3], f2[1], f2[3]))
        print(f"  ASYMMETRY {m}: fabricated {f1[1]}/{N_MASK_UNIDENT} "
              f"(W1, MAE {f1[3]:.3f} = {f1[3] / FLOOR_MAE:.0f}x floor) "
              f"and {f2[1]}/{w2_cells} (W2, MAE {f2[3]:.3f} = "
              f"{f2[3] / FLOOR_MAE:.0f}x floor) -- every one counted an "
              f"error")
    asym = all(f[1] == N_MASK_UNIDENT and f[3] == w2_cells
               for f in base_fabs)
    plan_ok = strong_vs_base and strong_vs_floor and honest
    print(f"  PLAN PREDICTION 'B2 strong where joins exist, honest "
          f"refusals elsewhere':\n    "
          f"{'HELD' if plan_ok else 'FAILED'} at the VALUE level on the "
          f"operationalized criteria above"
          + ("" if asym else
             " (NOTE: some baseline unexpectedly refused -- see rows)"))
    if false_closures:
        print(f"    CLAIM-LEVEL DENT (found wrong, kept): triangulate "
              f"issued {len(false_closures)}/{len(W2_SEEDS)} false "
              f"RESOLVED claims on unrelated tables (rows below); no "
              f"cell values were fabricated by them, but a claimed "
              f"latent that does not exist is not a NO CLOSURE refusal")
    # sub-predictions
    p1 = (tri["verdict"] == "RESOLVED — COMPLETION"
          and tri.get("gain", 0) > 5 * max(tri.get("scrambled_null95", 0),
                                           0.01))
    print(f"  P1 (completion past null): verdict {tri['verdict']}, gain "
          f"{tri.get('gain')} vs null95 {tri.get('scrambled_null95')} -> "
          f"{'holds' if p1 else 'FAILS'}")
    p2 = strong_vs_floor and (np.isnan(best_base[1][3]) or
                              best_base[1][3] >= 3 * isa_ident[3])
    print(f"  P2 (within 3x floor, >=3x under best baseline): "
          f"{isa_ident[3]:.4f} vs floor-MAE {FLOOR_MAE:.4f} and best "
          f"baseline {best_base[1][3]:.4f} -> "
          f"{'holds' if p2 else 'FAILS'}")
    fab20 = all(f[2] >= 20 * FLOOR_MAE or f[4] >= 20 * FLOOR_MAE
                for f in base_fabs)
    p3 = asym and fab20 and fab_w1 == 0 and fab_w2 == 0
    print(f"  P3 (baselines fabricate everywhere at >=20x floor, isaac "
          f"never): {'holds' if p3 else 'FAILS'}")
    refusal_verdicts = ("NO CLOSURE", "NO ALIGNMENT")
    p4 = (tri_s["verdict"] in refusal_verdicts and not false_closures
          and fab_scr == 0 and fab_w2 == 0)
    print(f"  P4 as FILED (scrambled + W2 both refuse NO CLOSURE; a "
          f"false closure anywhere\n     is a finding): scrambled -> "
          f"{tri_s['verdict']}; W2 false closures "
          f"{len(false_closures)}/{len(W2_SEEDS)} seeds -> "
          f"{'holds' if p4 else 'FAILS'}")
    for s, v, cc, nn in false_closures:
        print(f"    FOUND WRONG kept: W2 seed {s} -> {v} "
              f"(residual_corr {cc} vs null95 {nn}) -- a false latent "
              f"claim on unrelated\n      tables that beat the halves "
              f"confirmation gate; 0 values imputed from it")
    print("\nARTIFACT DIFFERENCE (stated qualitative row)")
    eq_str = law["equation"] if law else None
    print(f"  isaac emits: verdict + equation ({eq_str}) + join "
          "receipts (key, coverage,\n  gain vs scrambled null) + "
          f"per-cell refusal whys {refuse_whys};\n  baselines emit: a "
          "bare number for every cell, knowable or not -- no join\n  "
          "proof, no refusal, no receipt.")


if __name__ == "__main__":
    main()

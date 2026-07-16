"""eval_attack_rel_noise.py — ATTACK R-B: NOISE ON THE EXACT WORLD.

Target: pool_relations_probe.py. Its world is machine-exact
(y = a_g * x, no noise). Real level-0 constants carry estimation
error. This attack keeps the SAME world (identical rng-42 constants
and x draws via the probe's own make_episode) and adds i.i.d. y-noise
sigma in {0.05, 0.2} to every group's rows, then re-runs the pushed
ladder's level-1 legs:
  (1) level-1 pool must still scream on the a_C target,
  (2) ablation must still name exactly {A, B} (marginal still blind),
  (3) engine discover() on the descriptors: verdict, equation,
      train / held-out max error — the FORM should survive, errors
      should scale with noise, and NO machine-exactness may be
      claimed (train maxerr < 1e-9 on a noisy world = BROKEN).

Descriptor error theory: a_hat = a + <x,eps>/<x,x>, so
sd(a_hat - a) ~ sigma / sqrt(40 * var(x)) = sigma / 7.30 for
x ~ U(-2,2) (var 4/3, 40 rows).

PREDICTIONS (filed BEFORE the first run — wrong-kept if falsified)
  P1 descriptor extraction error: rms ~ 0.0068 at sigma=0.05 and
     ~ 0.0274 at sigma=0.2 (the sigma/7.30 law, +/- 30%).
  P2 level-0 bonus control: group C's residual after its own fitted
     law is pure noise — pool quiet 5/5 episodes at each sigma.
  P3 level-1 pool SCREAMS at BOTH sigmas: descriptor noise (<0.03)
     is tiny against signal sd(a_C) ~ 1.4; capture > +0.90 and above
     the null-99 threshold at both sigmas.
  P4 ablation names exactly [0, 1] (= {A, B} after C-column removal)
     at both sigmas; marginal baseline names [] at both sigmas.
  P5 genesis: discover verdict UNDERSTOOD at both sigmas; equation
     contains the x0*x1 product term with coefficient ~ +1.00
     (|c-1| < 0.05); train maxerr lands in [1e-3, 2e-1] — NOT < 1e-9
     (no machine-exactness on a noisy world).
  P6 held-out (12 unseen episodes): max |a_C_pred - a_C_true| < 0.06
     at sigma=0.05 and < 0.25 at sigma=0.2; the sigma=0.2 error is
     ~4x the sigma=0.05 error (accept band 2x-8x = "scales with
     noise"); composed raw-y max error same magnitudes.
  P7 runtime < 180 s (full N_EP=240 kept, both sigmas; no shrink).
     WRONG-KEPT (measured after run 1): 618.9 s on this box — the
     two full-N attribution passes (8 decoys x 9 channels x 10
     repeats each) dominate. Prediction falsified; results kept at
     full N because the run had already completed. A compliant
     re-run would shrink to one attribution pass or N_EP=120 and
     disclose the shrink.
  Expected grade if P3-P6 hold: WORKS — detection and the mint's
  FORM are noise-robust; exactness (probe Q5/Q6 < 1e-9) is lost, but
  the pushed claim scopes exactness to "the world is exact", so its
  loss under injected noise is expected behavior, not a violation.

INTEGRITY: pools point only; all cognition is discover()/apply_law();
world constants identical to the probe (same rng sequence, seed 42);
noise from a separate deterministic rng per sigma; PYTHONHASHSEED=0.
Run: cd experiments && PYTHONHASHSEED=0 python eval_attack_rel_noise.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import isaac as isaac_mod  # noqa: E402
from pool_relations_probe import (  # noqa: E402
    A_IDX, B_IDX, C_IDX, N_GROUPS, ROWS_PER_GROUP, fit_constant,
    make_episode)
from reservoir_surprise_probe import (  # noqa: E402
    attribution, marginal_names, scream_verdict)

Isaac = isaac_mod.Isaac

N_EP = 240
N_HOLD = 12
M_POOL = 800
SIGMAS = (0.05, 0.2)
L0_EPISODES = 5          # bonus level-0 quiet control per sigma


def add_noise(episodes, sigma, seed):
    """Same world, noisy rows: y -> y + N(0, sigma)."""
    rng = np.random.default_rng(seed)
    out = []
    for a, rows in episodes:
        out.append((a, [(x, y + rng.normal(0.0, sigma, len(y)))
                        for x, y in rows]))
    return out


def descriptors(eps):
    return np.array([[fit_constant(*ep[1][g]) for g in range(N_GROUPS)]
                     for ep in eps])


def main():
    t0 = time.perf_counter()
    checks = []

    def check(name, ok, detail):
        checks.append(ok)
        print(f"  [{'ok ' if ok else 'FAIL'}] {name} — {detail}")

    # the probe's exact world: identical constants + x draws (seed 42)
    rng = np.random.default_rng(42)
    clean = [make_episode(rng) for _ in range(N_EP + N_HOLD)]
    a_true_all = np.array([ep[0] for ep in clean])

    hold_errs = {}
    for si, sigma in enumerate(SIGMAS):
        print(f"\n=== sigma = {sigma} ===")
        noisy = add_noise(clean, sigma, seed=1000 + si)
        train_eps, hold_eps = noisy[:N_EP], noisy[N_EP:]

        # P1 — descriptor extraction error vs the sigma/7.30 law
        A_train = descriptors(train_eps)
        A_hold = descriptors(hold_eps)
        derr = A_train - a_true_all[:N_EP]
        rms = float(np.sqrt(np.mean(derr ** 2)))
        dmax = float(np.max(np.abs(derr)))
        pred_rms = sigma / np.sqrt(40 * 4.0 / 3.0)
        check(f"P1 descriptor rms ~ sigma/7.30 (s={sigma})",
              abs(rms - pred_rms) < 0.3 * pred_rms,
              f"rms {rms:.4f} vs predicted {pred_rms:.4f}, max {dmax:.4f}")

        # P2 — level-0 bonus: C's own-law residual is noise -> pool quiet
        quiet0 = 0
        for k in range(L0_EPISODES):
            x, y = train_eps[k][1][C_IDX]
            resid = y - fit_constant(x, y) * x
            _, _, s, _ = scream_verdict(x.reshape(-1, 1), resid, M_POOL,
                                        rng_seed=100 + 50 * si + k)
            quiet0 += (not s)
        check(f"P2 level-0 quiet on noisy residual (s={sigma})",
              quiet0 == L0_EPISODES, f"{quiet0}/{L0_EPISODES} quiet")

        # P3 — level-1 detection
        X1 = np.delete(A_train, C_IDX, axis=1)
        y1 = A_train[:, C_IDX]
        cap, thr, scream, _ = scream_verdict(X1, y1, M_POOL,
                                             rng_seed=200 + 50 * si)
        check(f"P3 level-1 pool screams (s={sigma})", scream,
              f"capture {cap:+.3f} vs null99 {thr:+.3f}")

        # P4 — attribution names exactly {A, B}; marginal blind
        contrib, athr, named = attribution(X1, y1, M_POOL,
                                           rng_seed=201 + 50 * si)
        mobs, mthr, mnamed = marginal_names(X1, y1)
        check(f"P4a ablation names exactly [0,1] (s={sigma})",
              named == [A_IDX, B_IDX],
              f"named {named}; contrib(A,B)=({contrib[A_IDX]:+.3f},"
              f"{contrib[B_IDX]:+.3f}) next-best "
              f"{np.sort(np.delete(contrib, [A_IDX, B_IDX]))[-1]:+.3f} "
              f"thr {athr:.3f}")
        check(f"P4b marginal baseline blind (s={sigma})", mnamed == [],
              f"named {mnamed} (max|corr| {mobs.max():.3f} vs {mthr:.3f})")

        # P5 — genesis mint on noisy descriptors
        mind = Isaac()
        res = mind.discover(X1, y1, name=f"rel_s{si}")
        pred = mind.apply_law(f"rel_s{si}", X1)
        train_err = (float(np.max(np.abs(pred - y1)))
                     if pred is not None else np.inf)
        eq = mind.findings[-1].get("equation", "?") if mind.findings else "?"
        product_form = ("x0*x1" in eq.replace(" ", "")
                        or "x1*x0" in eq.replace(" ", ""))
        exact_claimed = train_err < 1e-9
        print(f"       verdict={res['verdict']} eq='{eq}' "
              f"train maxerr {train_err:.3e}")
        check(f"P5a mint keeps the product FORM (s={sigma})",
              res["verdict"] == "UNDERSTOOD" and product_form,
              f"verdict={res['verdict']} eq={eq}")
        check(f"P5b no machine-exactness claimed (s={sigma})",
              not exact_claimed,
              f"train maxerr {train_err:.3e} (BROKEN if < 1e-9 on noise)")

        # P6 — held-out: a_C vs TRUE constants, composed raw y vs TRUE law
        Xh = np.delete(A_hold, C_IDX, axis=1)
        pred_c = mind.apply_law(f"rel_s{si}", Xh)
        err_true = (float(np.max(np.abs(pred_c - a_true_all[N_EP:, C_IDX])))
                    if pred_c is not None else np.inf)
        err_desc = (float(np.max(np.abs(pred_c - A_hold[:, C_IDX])))
                    if pred_c is not None else np.inf)
        raw_errs = []
        for i in range(N_HOLD):
            x_new = np.random.default_rng(900 + i).uniform(-2, 2, 25)
            raw_errs.append(float(np.max(np.abs(
                pred_c[i] * x_new - a_true_all[N_EP + i, C_IDX] * x_new))))
        hold_errs[sigma] = err_true
        bound = 0.06 if sigma == 0.05 else 0.25
        check(f"P6 held-out a_C err vs TRUE < {bound} (s={sigma})",
              err_true < bound,
              f"max|pred-true| {err_true:.4f}; vs noisy descriptor "
              f"{err_desc:.4f}; composed raw maxerr {max(raw_errs):.4f} "
              f"over {N_HOLD} eps")

    # error scaling across sigmas
    ratio = hold_errs[0.2] / max(hold_errs[0.05], 1e-12)
    check("P6-scale held-out error scales with noise (ratio in [2,8])",
          2.0 <= ratio <= 8.0,
          f"err(0.2)/err(0.05) = {hold_errs[0.2]:.4f}/{hold_errs[0.05]:.4f}"
          f" = {ratio:.2f}x (linear-in-sigma predicts ~4x)")

    dt = time.perf_counter() - t0
    check("P7 runtime", dt < 180, f"{dt:.1f}s")
    npass = sum(checks)
    print(f"\n  ATTACK R-B: {npass} passed, {len(checks) - npass} failed "
          f"of {len(checks)}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()

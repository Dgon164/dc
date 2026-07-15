"""Planted worlds for the data-ops bench (segment plan §3).

Own truth first, then the world's: every world here is PLANTED — the
corruption rows are known, so precision/recall are exact, not argued.
Corruption lives in the TARGET cell only, so row-level flags equal
cell-level flags for scoring (said once here, not hidden; cell
attribution inside a row is not built in audit v1).

W4a/W4b and W5 exist to MEASURE fresh-eyes review target #4 from the
2026-07-15 session record (audit thresholds were only ever tested on
Gaussian worlds): heavy-tailed clean noise and non-affine drag. Those
worlds are allowed to make the audit look bad — that is their job.

Every builder returns dict(name, columns, rows, bad=set(corrupted row
indices), resolutions=dict|None, note).
"""
import numpy as np


def _repeats_x(rng, lo=0.5, hi=6.0, levels=40, per=8):
    """Duplicate-x design: gives the floor organ its repeats road —
    the audit measures its own ruler instead of being told one."""
    x = np.repeat(np.linspace(lo, hi, levels), per)
    rng.shuffle(x)
    return x


def w1_inrange(seed):
    """The lane's native game: corruptions (±0.8) sit WELL INSIDE the
    target's marginal spread (std ≈ 3.2) — invisible to any flagger
    that never learns the law."""
    rng = np.random.default_rng(100 + seed)
    x = _repeats_x(rng)
    y = 2.0 * x + rng.normal(0, 0.05, len(x))
    bad = sorted(rng.choice(len(x), 8, replace=False).tolist())
    for i in bad:
        y[i] += float(rng.choice([-1.0, 1.0])) * 0.8
    return dict(name="W1 in-range corruption", columns=["x", "y"],
                rows=np.column_stack([x, y]), bad=set(bad),
                resolutions=None,
                note="8 planted at ±0.8; marginal std ~3.2")


def w2_clean(seed):
    """THE CONTROL — no corruption at all; every flag is a false flag."""
    rng = np.random.default_rng(200 + seed)
    x = _repeats_x(rng)
    y = 2.0 * x + rng.normal(0, 0.05, len(x))
    return dict(name="W2 clean Gaussian control", columns=["x", "y"],
                rows=np.column_stack([x, y]), bad=set(),
                resolutions=None, note="false-flag world")


def w3_gross(seed):
    """Gross corruption (±15, ~4.7 marginal sigmas) — the world where
    everyone should win; separation is receipts, not recall."""
    rng = np.random.default_rng(300 + seed)
    x = _repeats_x(rng)
    y = 2.0 * x + rng.normal(0, 0.05, len(x))
    bad = sorted(rng.choice(len(x), 6, replace=False).tolist())
    for i in bad:
        y[i] += float(rng.choice([-1.0, 1.0])) * 15.0
    return dict(name="W3 gross corruption", columns=["x", "y"],
                rows=np.column_stack([x, y]), bad=set(bad),
                resolutions=None, note="6 planted at ±15")


def w4a_heavytail_repeats(seed):
    """CLEAN Student-t(3) noise, floor measured off repeats. Fresh-eyes
    #4: do the Gaussian-shaped lines false-flag honest tail events?"""
    rng = np.random.default_rng(400 + seed)
    x = _repeats_x(rng)
    y = 2.0 * x + 0.05 * rng.standard_t(3, len(x))
    return dict(name="W4a heavy-tail clean (repeats floor)",
                columns=["x", "y"], rows=np.column_stack([x, y]),
                bad=set(), resolutions=None,
                note="t(3) scale 0.05; zero planted — any flag is false")


def w4b_heavytail_told(seed):
    """CLEAN t(3) noise, instrument resolution TOLD at the scale param
    (a true statement about the instrument; the tails are process).
    The sharpest form of fresh-eyes #4."""
    rng = np.random.default_rng(450 + seed)
    x = rng.uniform(0.5, 6.0, 320)
    y = 2.0 * x + 0.05 * rng.standard_t(3, len(x))
    return dict(name="W4b heavy-tail clean (told floor 0.05)",
                columns=["x", "y"], rows=np.column_stack([x, y]),
                bad=set(), resolutions={"y": 0.05},
                note="t(3) scale 0.05 told; zero planted")


def w5_nonaffine_drag(seed):
    """Curved law (y = 0.5·x²) with corruption CORRELATED with x (all
    +2.0, all in the top-x quartile). The consensus recalibration is
    affine-only by declared scope — this world measures what that
    scope costs. The audit may class these NEWS (they recur, factor
    nameable) — for detection scoring a candidate flag of either
    class counts as caught."""
    rng = np.random.default_rng(500 + seed)
    x = _repeats_x(rng)
    y = 0.5 * x ** 2 + rng.normal(0, 0.05, len(x))
    top = np.where(x >= 4.6)[0]
    bad = sorted(rng.choice(top, 8, replace=False).tolist())
    for i in bad:
        y[i] += 2.0
    return dict(name="W5 non-affine drag (curved law, clustered +2)",
                columns=["x", "y"], rows=np.column_stack([x, y]),
                bad=set(bad), resolutions=None,
                note="8 planted, all top-x quartile, same sign")


B1_WORLDS = [w1_inrange, w2_clean, w3_gross, w4a_heavytail_repeats,
             w4b_heavytail_told, w5_nonaffine_drag]


# ── B2 worlds: two tables, a key, and a masked target ──

def w6_completion(seed):
    """B's column z completes A's law (y = x + z + ε): the triangulate
    battery's proven shape. 40 of 160 targets masked."""
    rng = np.random.default_rng(600 + seed)
    n = 160
    key = np.arange(1.0, n + 1)
    z = rng.uniform(-3, 3, n)
    x = rng.uniform(0, 5, n)
    y = x + z + rng.normal(0, 0.05, n)
    w = 0.5 * z + rng.normal(0, 0.1, n)
    masked = sorted(rng.choice(n, 40, replace=False).tolist())
    return dict(name="W6 completion (real join, y = x + z)",
                A=dict(name="A", columns=["key", "x", "y"],
                       rows=np.column_stack([key, x, y])),
                B=dict(name="B", columns=["key", "z", "w"],
                       rows=np.column_stack([key, z, w])),
                masked=masked, truth=y, floor=0.05,
                expect="RESOLVED",
                note="joint law exists; masked error should sit near "
                     "the 0.05 noise floor")


def w7_no_structure(seed):
    """B knows nothing about A. The honest answer for masked targets
    is the mean-model WITH a NO CLOSURE receipt; imputing confidently
    from z is the baseline's mistake, not a capability."""
    rng = np.random.default_rng(700 + seed)
    n = 160
    key = np.arange(1.0, n + 1)
    y = rng.normal(5, 1.0, n)
    z = rng.uniform(-3, 3, n)
    w = rng.normal(0, 1.0, n)
    masked = sorted(rng.choice(n, 40, replace=False).tolist())
    return dict(name="W7 no structure (B unrelated to A)",
                A=dict(name="A", columns=["key", "y"],
                       rows=np.column_stack([key, y])),
                B=dict(name="B", columns=["key", "z", "w"],
                       rows=np.column_stack([key, z, w])),
                masked=masked, truth=y, floor=1.0,
                expect="REFUSAL",
                note="refusal is the correct verdict; mean-model MAE "
                     "~0.8 is the honest ceiling")


def w8_ica_limit(seed):
    """Equal-variance Gaussian halves — unresolvable IN PRINCIPLE
    (triangulate's own documented ICA limit). Refusal here is
    correct-by-design, scored as such."""
    rng = np.random.default_rng(800 + seed)
    n = 200
    key = np.arange(1.0, n + 1)
    a1 = rng.uniform(0, 5, n)
    b1 = rng.uniform(0, 5, n)
    xl = rng.normal(0, 1.0, n)
    sl = rng.normal(0, 1.0, n)
    yA = a1 + (xl + sl) + rng.normal(0, 0.02, n)
    yB = b1 + (xl - sl) + rng.normal(0, 0.02, n)
    return dict(name="W8 ICA limit (equal-variance halves)",
                A=dict(name="A", columns=["key", "a1", "yA"],
                       rows=np.column_stack([key, a1, yA])),
                B=dict(name="B", columns=["key", "b1", "yB"],
                       rows=np.column_stack([key, b1, yB])),
                masked=None, truth=None, floor=None,
                expect="REFUSAL",
                note="residual corr is 0 by construction")


def w9_halves_resolvable(seed):
    """THE REFUSAL CONTROL: same shape as W8 but the latents have
    unequal variances (corr ≈ 0.88) — refusing HERE is wrong. A bench
    that scores refusals must contain a world where refusal loses."""
    rng = np.random.default_rng(900 + seed)
    n = 200
    key = np.arange(1.0, n + 1)
    a1 = rng.uniform(0, 5, n)
    b1 = rng.uniform(0, 5, n)
    xl = rng.normal(0, 2.0, n)
    sl = rng.normal(0, 0.5, n)
    yA = a1 + (xl + sl) + rng.normal(0, 0.02, n)
    yB = b1 + (xl - sl) + rng.normal(0, 0.02, n)
    return dict(name="W9 resolvable halves (unequal variances)",
                A=dict(name="A", columns=["key", "a1", "yA"],
                       rows=np.column_stack([key, a1, yA])),
                B=dict(name="B", columns=["key", "b1", "yB"],
                       rows=np.column_stack([key, b1, yB])),
                masked=None, truth=None, floor=None,
                expect="RESOLVED",
                note="latent corr 3.75/4.25 ~ 0.88 by construction")


B2_WORLDS = [w6_completion, w7_no_structure, w8_ica_limit,
             w9_halves_resolvable]

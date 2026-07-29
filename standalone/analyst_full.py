"""ANALYST FULL — the rest of what a data analyst delivers.

Owner's bar (2026-07-28): "i want to extract as much information
that any user could possibly want... comparable or superior in data
analytics... i want to make sure they get what they pay for."

A professional analyst's report answers seven questions. This module
adds the five the storefront didn't cover (trends and audit already
exist), every claim graded by the owner's subsample ruling — say
what is there, validate on random half-samples, label SOLID / WEAK /
UNSTABLE, never guess silently:

  PROFILE        what is in this data (every column accounted for)
  SEGMENTS       how the groups compare (revenue by region, etc.)
  SEASONALITY    what repeats (month-of-year, day-of-week)
  RELATIONSHIPS  what moves together (and what is just shared trend)
  CONCENTRATION  where the weight sits (the 80/20 of it)
  ANOMALIES      what sticks out (values and time gaps)
  HEADLINES      the executive summary, ranked by size x stability

Wrapper machinery: plain arithmetic on the client's table, receipted;
the promoted engine is not involved and not imitated — its certified
laws stay a separate, stricter tier of claim.
"""
import datetime
import os
import re

import numpy as np

BLUE = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"

TRIALS = 200
SEED = 23


# ------------------------------------------------------------ columns
def classify_columns(rows, header):
    """Every column accounted for: numeric, date, category, or text."""
    cols = {}
    n = len(rows)
    for i, h in enumerate(header):
        vals = [(r[i] if i < len(r) else "") for r in rows]
        nonblank = [v for v in vals if v != ""]
        nums = []
        for v in nonblank:
            try:
                nums.append(float(v))
            except ValueError:
                pass
        kind = "text"
        dates = _parse_dates(nonblank)
        if nonblank and len(nums) / len(nonblank) >= 0.9:
            kind = "numeric"
        elif dates is not None:
            kind = "date"
        elif 1 < len(set(nonblank)) <= min(30, max(2, n // 3)):
            kind = "category"
        cols[h] = dict(kind=kind, vals=vals, nums=np.array(nums),
                       dates=dates, missing=n - len(nonblank))
    return cols


def _parse_dates(vals):
    """ISO dates -> list of datetime.date, else None."""
    if not vals or not all(re.match(r"^\d{4}-\d{1,2}-\d{1,2}$", v)
                           for v in vals[:50]):
        return None
    try:
        return [datetime.date(*map(int, v.split("-"))) for v in vals]
    except ValueError:
        return None


def _grade(frac, extra_ok=True):
    if frac >= 0.99 and extra_ok:
        return "SOLID"
    if frac >= 0.80:
        return "WEAK"
    return "UNSTABLE"


# ------------------------------------------------------------ profile
def profile_block(cols, n_rows):
    md = ["## What is in this data", ""]
    for h, c in cols.items():
        miss = (f", {c['missing']} missing" if c["missing"] else "")
        if c["kind"] == "numeric" and len(c["nums"]):
            v = c["nums"]
            md.append(f"- **{h}** (number{miss}): {len(v)} values, "
                      f"{v.min():,.4g} to {v.max():,.4g}, typical "
                      f"{np.median(v):,.4g}, mean {v.mean():,.4g}, "
                      f"total {v.sum():,.4g}")
        elif c["kind"] == "category":
            uniq = [v for v in c["vals"] if v != ""]
            top, cnt = _top_count(uniq)
            md.append(f"- **{h}** (category{miss}): "
                      f"{len(set(uniq))} groups; most common "
                      f"'{top}' ({cnt} of {len(uniq)} rows)")
        elif c["kind"] == "date":
            d = c["dates"]
            md.append(f"- **{h}** (dates{miss}): {d[0].isoformat()} "
                      f"to {d[-1].isoformat()}, {len(d)} entries")
        else:
            uniq = len(set(v for v in c["vals"] if v != ""))
            md.append(f"- **{h}** (text{miss}): {uniq} distinct "
                      f"values — profiled only, not aggregated")
    md.append("")
    return md


def _top_count(items):
    best, cnt = "", 0
    seen = {}
    for it in items:
        seen[it] = seen.get(it, 0) + 1
        if seen[it] > cnt:
            best, cnt = it, seen[it]
    return best, cnt


# ------------------------------------------------------------ segments
def segments_block(rows, header, cols, chart_dir, findings):
    cats = [h for h, c in cols.items() if c["kind"] == "category"][:3]
    nums = [h for h, c in cols.items() if c["kind"] == "numeric"][:3]
    md, charts = [], []
    if not cats or not nums:
        return md, charts
    md += ["## How the groups compare", ""]
    rng = np.random.default_rng(SEED)
    for cat in cats:
        ci = header.index(cat)
        for num in nums:
            ni = header.index(num)
            pairs = [(r[ci], float(r[ni])) for r in rows
                     if ci < len(r) and ni < len(r) and r[ci] != ""
                     and _is_num(r[ni])]
            if len(pairs) < 12:
                continue
            groups = {}
            for g, v in pairs:
                groups.setdefault(g, []).append(v)
            if not (2 <= len(groups) <= 30):
                continue
            totals = sorted(((sum(v), g, len(v)) for g, v in
                             groups.items()), reverse=True)
            whole = sum(t for t, _, _ in totals) or 1e-12
            lead_t, lead_g, lead_n = totals[0]
            second_g = totals[1][1]
            # stability: does the same leader lead in half-samples?
            arr = np.array([(hash(g) % 10 ** 9, v) for g, v in pairs])
            gs = np.array([g for g, _ in pairs])
            vs = np.array([v for _, v in pairs])
            hold = 0
            for _ in range(TRIALS):
                pick = rng.choice(len(vs), size=max(6, len(vs) // 2),
                                  replace=False)
                sub_tot = {}
                for g, v in zip(gs[pick], vs[pick]):
                    sub_tot[g] = sub_tot.get(g, 0.0) + v
                if sub_tot and max(sub_tot, key=sub_tot.get) == lead_g:
                    hold += 1
            frac = hold / TRIALS
            grade = _grade(frac)
            top3 = "; ".join(f"'{g}' {t:,.4g} ({t / whole:.0%}, "
                             f"{n} rows, avg {t / n:,.4g})"
                             for t, g, n in totals[:3])
            md.append(f"- **{num} by {cat}** — leader '{lead_g}' "
                      f"with {lead_t / whole:.0%} of the total, "
                      f"ahead of '{second_g}' — {grade} (led in "
                      f"{frac:.0%} of {TRIALS} random half-samples). "
                      f"Top groups: {top3}.")
            if grade != "UNSTABLE":
                findings.append((abs(lead_t / whole - 1 / len(groups))
                                 * frac,
                                 f"'{lead_g}' leads {num} by {cat} "
                                 f"({lead_t / whole:.0%} of total) — "
                                 f"{grade}"))
            p = _bar_chart(totals[:8], f"{num} by {cat}", chart_dir)
            if p:
                md.append(f"![{num} by {cat}](charts/"
                          f"{os.path.basename(p)})")
        md.append("")
    return md, charts


def _is_num(v):
    try:
        float(v)
        return True
    except ValueError:
        return False


# --------------------------------------------------------- seasonality
def seasonality_block(rows, header, cols, findings):
    date_col = next((h for h, c in cols.items()
                     if c["kind"] == "date"), None)
    nums = [h for h, c in cols.items() if c["kind"] == "numeric"][:2]
    md = []
    if not date_col or not nums:
        return md
    ci = header.index(date_col)
    md += ["## What repeats", ""]
    rng = np.random.default_rng(SEED + 1)
    for num in nums:
        ni = header.index(num)
        recs = []
        for r in rows:
            if ci < len(r) and ni < len(r) and _is_num(r[ni]):
                m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", r[ci])
                if m:
                    d = datetime.date(*map(int, m.groups()))
                    recs.append((d, float(r[ni])))
        if len(recs) < 30:
            continue
        for label, keyf, names in (
                ("day of the week", lambda d: d.weekday(),
                 ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
                ("month of the year", lambda d: d.month - 1,
                 ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
                  "Aug", "Sep", "Oct", "Nov", "Dec"])):
            keys = np.array([keyf(d) for d, _ in recs])
            vals = np.array([v for _, v in recs])
            if len(set(keys.tolist())) < (5 if label.startswith("day")
                                          else 8):
                continue
            means = {k: vals[keys == k].mean()
                     for k in set(keys.tolist())}
            peak = max(means, key=means.get)
            trough = min(means, key=means.get)
            spread = means[peak] - means[trough]
            if spread < 0.05 * (abs(vals.mean()) + 1e-12):
                md.append(f"- **{num} by {label}**: no meaningful "
                          f"pattern (spread under 5% of the mean) — "
                          f"said plainly rather than decorated.")
                continue
            # out-of-sample validation: FIND the pattern on one
            # random half, TEST it on the other half. A pattern
            # picked from noise fails the half it never saw.
            hold = 0
            for _ in range(TRIALS):
                perm = rng.permutation(len(vals))
                a, b = perm[:len(vals) // 2], perm[len(vals) // 2:]
                def _means(ix):
                    return {k: vals[ix][keys[ix] == k].mean()
                            for k in set(keys[ix].tolist())
                            if np.any(keys[ix] == k)}
                ma, mb = _means(a), _means(b)
                if len(ma) < 2:
                    continue
                pa = max(ma, key=ma.get)
                ta = min(ma, key=ma.get)
                if pa in mb and ta in mb and mb[pa] > mb[ta]:
                    hold += 1
            frac = hold / TRIALS
            grade = _grade(frac)
            md.append(f"- **{num} by {label}** — peak "
                      f"{names[peak]} (avg {means[peak]:,.4g}), "
                      f"trough {names[trough]} (avg "
                      f"{means[trough]:,.4g}), swing "
                      f"{spread:,.4g} — {grade} (pattern found on "
                      f"one random half confirmed on the unseen "
                      f"half in {frac:.0%} of {TRIALS} tests)")
            if grade == "SOLID":
                findings.append(
                    (spread / (abs(vals.mean()) + 1e-12) * frac,
                     f"{num} peaks on {names[peak]}, dips "
                     f"{names[trough]} ({label}) — SOLID"))
    md.append("")
    return md


# ------------------------------------------------------- relationships
def relationships_block(rows, header, cols, time_col, findings):
    nums = [h for h, c in cols.items()
            if c["kind"] == "numeric" and len(c["nums"]) >= 12][:5]
    md = []
    if len(nums) < 2:
        return md
    md += ["## What moves together", ""]
    rng = np.random.default_rng(SEED + 2)
    t_idx = header.index(time_col) if time_col in header else None
    any_pair = False
    for a_i in range(len(nums)):
        for b_i in range(a_i + 1, len(nums)):
            a, b = nums[a_i], nums[b_i]
            if time_col in (a, b):
                continue
            ai, bi = header.index(a), header.index(b)
            triples = [(float(r[ai]), float(r[bi]),
                        float(r[t_idx]) if t_idx is not None
                        and t_idx < len(r) and _is_num(r[t_idx])
                        else None)
                       for r in rows if ai < len(r) and bi < len(r)
                       and _is_num(r[ai]) and _is_num(r[bi])]
            if len(triples) < 12:
                continue
            x = np.array([t[0] for t in triples])
            y = np.array([t[1] for t in triples])
            if x.std() == 0 or y.std() == 0:
                continue
            r_full = float(np.corrcoef(x, y)[0, 1])
            if abs(r_full) < 0.3:
                continue
            any_pair = True
            rs = []
            for _ in range(TRIALS):
                pick = rng.choice(len(x), size=max(6, len(x) // 2),
                                  replace=False)
                if x[pick].std() == 0 or y[pick].std() == 0:
                    continue
                rs.append(np.corrcoef(x[pick], y[pick])[0, 1])
            rs = np.array(rs)
            frac = float(np.mean(np.sign(rs) == np.sign(r_full)))
            lo, hi = np.percentile(rs, [2.5, 97.5])
            detr = ""
            r_use = r_full
            ts = [t[2] for t in triples]
            if all(t is not None for t in ts):
                t_arr = np.array(ts)
                if t_arr.std() > 0:
                    xr = x - np.poly1d(np.polyfit(t_arr, x, 1))(t_arr)
                    yr = y - np.poly1d(np.polyfit(t_arr, y, 1))(t_arr)
                    if xr.std() > 0 and yr.std() > 0:
                        r_d = float(np.corrcoef(xr, yr)[0, 1])
                        if abs(r_d) < 0.3 <= abs(r_full):
                            detr = (f". Caution: after removing the "
                                    f"shared time-trend the link "
                                    f"drops to r={r_d:+.2f} — this "
                                    f"is mostly two things rising "
                                    f"together, not a working "
                                    f"relationship")
                            r_use = r_d
                        else:
                            detr = (f"; still r={r_d:+.2f} with the "
                                    f"shared time-trend removed")
            direction = ("move together" if r_full > 0
                         else "move opposite")
            grade = _grade(frac)
            md.append(f"- **{a} and {b}** {direction} "
                      f"(r={r_full:+.2f}) — {grade} (sign held in "
                      f"{frac:.0%} of half-samples, spread "
                      f"[{lo:+.2f}, {hi:+.2f}]){detr}. Link, not "
                      f"cause — we can't say which drives which "
                      f"from this table alone.")
            if grade == "SOLID" and abs(r_use) >= 0.3:
                findings.append((abs(r_use) * frac,
                                 f"{a} and {b} {direction} "
                                 f"(r={r_full:+.2f}) — SOLID"))
    if not any_pair:
        md.append("- no pair of number columns moves together "
                  "beyond r=0.3 — said plainly.")
    md.append("")
    return md


# ------------------------------------------------------- concentration
def concentration_block(rows, header, cols, findings):
    md = []
    lines = []
    for h, c in cols.items():
        v = c["nums"]
        if c["kind"] != "numeric" or len(v) < 20 or v.min() < 0 \
                or v.sum() <= 0:
            continue
        s = np.sort(v)[::-1]
        k = max(1, len(s) // 5)
        share = s[:k].sum() / s.sum()
        if share >= 0.5:
            lines.append(f"- **{h}**: the top 20% of rows carry "
                         f"{share:.0%} of the whole total — the "
                         f"weight is concentrated, not spread.")
            findings.append((share - 0.2,
                             f"top 20% of rows carry {share:.0%} "
                             f"of {h} — concentrated"))
        else:
            lines.append(f"- **{h}**: top 20% of rows carry "
                         f"{share:.0%} of the total — fairly "
                         f"evenly spread.")
    if lines:
        md += ["## Where the weight sits", ""] + lines + [""]
    return md


# ----------------------------------------------------------- anomalies
def anomalies_block(rows, header, cols, findings):
    md = []
    lines = []
    for h, c in cols.items():
        v = c["nums"]
        if c["kind"] != "numeric" or len(v) < 20:
            continue
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med))) or 1e-12
        hits = [(float(x), abs(x - med) / mad) for x in v
                if abs(x - med) / mad > 8]
        if hits and len(hits) <= max(3, len(v) // 20):
            for val, score in sorted(hits, key=lambda p: -p[1])[:3]:
                lines.append(f"- **{h}**: value {val:,.6g} sits far "
                             f"outside the rest ({score:,.0f}x the "
                             f"typical spread from the middle "
                             f"{med:,.4g}) — worth a look, kept "
                             f"as-is.")
            findings.append((min(1.0, len(hits) / 3),
                             f"{len(hits)} value(s) in {h} sit far "
                             f"outside the rest"))
    dcol = next((h for h, c in cols.items()
                 if c["kind"] == "date"), None)
    if dcol:
        d = sorted(cols[dcol]["dates"])
        gaps = [(d[i], (d[i + 1] - d[i]).days)
                for i in range(len(d) - 1)]
        if gaps:
            deltas = np.array([g for _, g in gaps])
            typ = float(np.median(deltas)) or 1
            big = [(day, g) for day, g in gaps if g > 3 * typ]
            for day, g in big[:3]:
                lines.append(f"- **{dcol}**: a {g}-day gap after "
                             f"{day.isoformat()} (typical spacing "
                             f"{typ:.0f} day(s)) — missing period?")
    if lines:
        md += ["## What sticks out", ""] + lines + [""]
    return md


# -------------------------------------------------------------- charts
def _bar_chart(totals, title, chart_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    os.makedirs(chart_dir, exist_ok=True)
    labels = [g for _, g, _ in totals]
    heights = [t for t, _, _ in totals]
    fig, ax = plt.subplots(figsize=(7.2, 2.4), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    bars = ax.bar(range(len(heights)), heights, color=BLUE,
                  width=0.72)
    for b, hgt in zip(bars, heights):
        ax.annotate(f"{hgt:,.3g}", (b.get_x() + b.get_width() / 2,
                                    hgt), ha="center", va="bottom",
                    fontsize=7, color=INK2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, color=INK2)
    ax.grid(True, axis="y", color=MUTED, alpha=0.25, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_title(title, loc="left", fontsize=10, color=INK,
                 fontweight="bold")
    fig.tight_layout()
    p = os.path.join(chart_dir,
                     re.sub(r"[^\w.-]", "_", title) + ".png")
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


# ------------------------------------------------------------ assembly
def build_extras(rows, header, time_col, described, chart_dir):
    """Returns (headline_md, section_md). Every line traces to
    arithmetic on the table; every comparative claim carries its
    half-sample grade."""
    cols = classify_columns(rows, header)
    findings = []  # (score, text) for the headlines
    for name, d in (described or {}).items():
        dd = d["d"]
        if dd["grade"] == "SOLID":
            findings.append(
                (0.5 + abs(dd["total"]) / (abs(dd["total"]) + 1),
                 f"{name} is {'rising' if dd['slope'] > 0 else 'falling'} "
                 f"(about {dd['total']:+.4g} across the record) — "
                 f"SOLID"))
    blocks = []
    blocks += profile_block(cols, len(rows))
    seg_md, _ = segments_block(rows, header, cols, chart_dir,
                               findings)
    blocks += seg_md
    blocks += seasonality_block(rows, header, cols, findings)
    blocks += relationships_block(rows, header, cols, time_col,
                                  findings)
    blocks += concentration_block(rows, header, cols, findings)
    blocks += anomalies_block(rows, header, cols, findings)

    findings.sort(reverse=True)
    heads = ["## The headlines", ""]
    if findings:
        heads += [f"- {t}" for _, t in findings[:7]]
    else:
        heads.append("- nothing in this table passed our stability "
                     "checks strongly enough to headline — the "
                     "details below say what was checked.")
    heads.append("")
    return heads, blocks

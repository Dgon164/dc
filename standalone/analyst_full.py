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


def human(x):
    """Numbers a person reads: 1.35M, 14,680, 944, 0.0083 — never
    scientific notation in a client document."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    a = abs(x)
    if a >= 1e9:
        return f"{x / 1e9:,.2f}B"
    if a >= 1e6:
        return f"{x / 1e6:,.2f}M"
    if a >= 1e4:
        return f"{x:,.0f}"
    if a >= 100:
        return f"{x:,.0f}"
    if a >= 1:
        return f"{x:,.3g}"
    return f"{x:.4g}"


# ------------------------------------------------------------ columns
def _tokens(name):
    """Column-name words: 'InvoiceNo' -> {invoice, no};
    'incident_zip' -> {incident, zip}."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return {t for t in re.split(r"[^a-zA-Z0-9]+", spaced.lower())
            if t}


ID_TOKENS = {"id", "ids", "no", "num", "number", "key", "code",
             "codes", "zip", "zipcode", "postal", "phone", "bbl",
             "ssn", "ein", "uuid", "guid", "sku", "upc", "isbn",
             "district", "precinct", "tract"}
COORD_TOKENS = {"lat", "latitude", "lon", "lng", "long",
                "longitude", "coordinate", "coordinates", "plane",
                "geocode"}
RATE_TOKENS = {"price", "rate", "cost", "fee", "pct", "percent",
               "ratio"}
CUSTOMER_TOKENS = {"customer", "client", "buyer", "member",
                   "account", "user", "email", "patient", "donor"}
ORDER_TOKENS = {"order", "invoice", "transaction", "receipt",
                "ticket", "basket"}

DT_RES = (
    re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:[T ]\d{1,2}:\d{2}"
               r"(?::\d{2}(?:\.\d+)?)?)?$"),
    re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})(?:\s+\d{1,2}:\d{2}"
               r"(?::\d{2})?)?$"),
)


def parse_any_date(v):
    """A date out of a date-or-datetime string, or None. Slash dates
    read month-first (US); a first number over 12 flips to
    day-first."""
    m = DT_RES[0].match(v)
    if m:
        y, mo, d = map(int, m.groups())
    else:
        m = DT_RES[1].match(v)
        if not m:
            return None
        a, b, y = map(int, m.groups())
        mo, d = (a, b) if a <= 12 else (b, a)
    try:
        return datetime.date(y, mo, d)
    except ValueError:
        return None


def normalize_dates(rows, header):
    """Datetime-shaped columns become plain ISO dates IN PLACE, so
    every daily organ can read them. Receipted: each conversion is
    a note; times of day are set aside, said, never silently."""
    notes = []
    n = len(rows)
    for i, h in enumerate(header):
        sample = [r[i] for r in rows[:200] if i < len(r) and r[i]]
        if not sample:
            continue
        parsed = [parse_any_date(v) for v in sample]
        okf = sum(1 for p in parsed if p) / len(sample)
        plain_iso = all(re.match(r"^\d{4}-\d{1,2}-\d{1,2}$", v)
                        for v in sample)
        if okf < 0.9 or plain_iso:
            continue
        changed = 0
        for r in rows:
            if i < len(r) and r[i]:
                p = parse_any_date(r[i])
                if p:
                    r[i] = p.isoformat()
                    changed += 1
        notes.append(f"column '{h}': {changed:,} timestamp(s) read "
                     f"as dates (time of day set aside for the "
                     f"daily analysis, never silently)")
    return notes


def classify_columns(rows, header):
    """Every column accounted for — and never a label mistaken for
    a quantity: identifiers (invoice numbers, zip codes, ids) and
    coordinates are counted or mapped, NEVER summed, trended,
    correlated, or averaged."""
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
        toks = _tokens(h)
        kind = "text"
        dates = _parse_dates(nonblank)
        numeric_frac = (len(nums) / len(nonblank)) if nonblank else 0
        uniq = len(set(nonblank))
        if dates is not None:
            kind = "date"
        elif toks & COORD_TOKENS and numeric_frac >= 0.9:
            kind = "coordinate"
        elif toks & (ID_TOKENS | ORDER_TOKENS | CUSTOMER_TOKENS) \
                and (numeric_frac >= 0.9 or uniq > 30):
            kind = "identifier"
        elif toks & RATE_TOKENS and numeric_frac >= 0.9 \
                and not (toks & {"revenue", "sales", "amount",
                                 "total", "dollars"}):
            kind = "rate"  # per-unit number: averaged, never summed
        elif numeric_frac >= 0.9:
            ints = nums and all(float(x).is_integer() for x in
                                nums[:500])
            if ints and uniq >= 0.95 * len(nonblank) and uniq > 50:
                kind = "identifier"  # a counter wearing numbers
            else:
                kind = "numeric"
        elif 1 < uniq <= min(30, max(2, n // 3)):
            kind = "category"
        cols[h] = dict(kind=kind, vals=vals, nums=np.array(nums),
                       dates=dates, missing=n - len(nonblank),
                       tokens=toks)
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
                      f"{human(v.min())} to {human(v.max())}, "
                      f"typical {human(np.median(v))}, mean "
                      f"{human(v.mean())}, total {human(v.sum())}")
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
        elif c["kind"] == "identifier":
            uniq = len(set(v for v in c["vals"] if v != ""))
            md.append(f"- **{h}** (identifier{miss}): {uniq:,} "
                      f"distinct — a label wearing numbers; "
                      f"counted, never summed or averaged")
        elif c["kind"] == "rate" and len(c["nums"]):
            v = c["nums"]
            md.append(f"- **{h}** (per-unit number{miss}): "
                      f"{human(v.min())} to {human(v.max())}, "
                      f"typical {human(np.median(v))} — averaged "
                      f"and trended, never summed (a total of "
                      f"prices means nothing)")
        elif c["kind"] == "coordinate":
            uniq = len(set(v for v in c["vals"] if v != ""))
            md.append(f"- **{h}** (coordinate{miss}): {uniq:,} "
                      f"distinct positions — mapped, never treated "
                      f"as a quantity")
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
    if not cats:
        return md, charts  # counts still work with zero measures
    md += ["## How the groups compare", ""]
    rng = np.random.default_rng(SEED)
    for cat in cats:
        ci = header.index(cat)
        for num in nums + ["volume (rows)"]:
            if num == "volume (rows)":
                pairs = [(r[ci], 1.0) for r in rows
                         if ci < len(r) and r[ci] != ""]
            else:
                ni = header.index(num)
                pairs = [(r[ci], float(r[ni])) for r in rows
                         if ci < len(r) and ni < len(r)
                         and r[ci] != "" and _is_num(r[ni])]
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
            top3 = "; ".join(f"'{g}' {human(t)} ({t / whole:.0%}, "
                             f"{n} rows, avg {human(t / n)})"
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
                                 f"{grade}",
                                 f"if margins are similar, "
                                 f"'{lead_g}' is where the volume "
                                 f"lives — the '{second_g}' gap is "
                                 f"the growth question to ask"))
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
    if not date_col:
        return md  # daily volume still works with zero measures
    ci = header.index(date_col)
    md += ["## What repeats", ""]
    rng = np.random.default_rng(SEED + 1)
    for num in nums + ["volume (rows per day)"]:
        recs = []
        if num == "volume (rows per day)":
            per_day = {}
            for r in rows:
                if ci < len(r):
                    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$",
                                 r[ci])
                    if m:
                        d = datetime.date(*map(int, m.groups()))
                        per_day[d] = per_day.get(d, 0) + 1
            recs = sorted(per_day.items())
            # ABSENCE is a finding: a weekday with zero activity
            # across the whole record is how the operation runs
            if len(recs) >= 12:
                span = (recs[-1][0] - recs[0][0]).days + 1
                if span >= 12:
                    present = {d.weekday() for d, _ in recs}
                    names7 = ["Monday", "Tuesday", "Wednesday",
                              "Thursday", "Friday", "Saturday",
                              "Sunday"]
                    for wd in range(7):
                        if wd not in present and span >= 12:
                            md.append(f"- **no activity on any "
                                      f"{names7[wd]}** in "
                                      f"{span} days of record — "
                                      f"the operation doesn't run "
                                      f"that day (or that data "
                                      f"never lands in this file)")
                            findings.append(
                                (0.85,
                                 f"zero activity on every "
                                 f"{names7[wd]} in the record",
                                 f"if that's by design, fine — if "
                                 f"not, a whole day of business "
                                 f"is missing from this file"))
        else:
            ni = header.index(num)
            for r in rows:
                if ci < len(r) and ni < len(r) and _is_num(r[ni]):
                    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$",
                                 r[ci])
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
                      f"{names[peak]} (avg {human(means[peak])}), "
                      f"trough {names[trough]} (avg "
                      f"{human(means[trough])}), swing "
                      f"{human(spread)} — {grade} (pattern found on "
                      f"one random half confirmed on the unseen "
                      f"half in {frac:.0%} of {TRIALS} tests)")
            if grade == "SOLID":
                findings.append(
                    (spread / (abs(vals.mean()) + 1e-12) * frac,
                     f"{num} peaks on {names[peak]}, dips "
                     f"{names[trough]} ({label}) — SOLID",
                     f"worth checking what differs about "
                     f"{names[trough]} — hours, staffing, "
                     f"promotion; a repeatable dip is repeatable "
                     f"money"))
    md.append("")
    return md


# ------------------------------------------------------- relationships
def relationships_block(rows, header, cols, time_col, findings,
                        families=None):
    nums = [h for h, c in cols.items()
            if c["kind"] == "numeric" and len(c["nums"]) >= 12][:5]
    md = []
    if len(nums) < 2:
        return md
    md += ["## What moves together", ""]
    rng = np.random.default_rng(SEED + 2)
    t_idx = header.index(time_col) if time_col in header else None
    any_pair = False

    def fam(h):
        for k, f in enumerate(families or []):
            if h in f:
                return k
        return None
    seen_fams = set()
    for a_i in range(len(nums)):
        for b_i in range(a_i + 1, len(nums)):
            a, b = nums[a_i], nums[b_i]
            if time_col in (a, b):
                continue
            fa, fb = fam(a), fam(b)
            if fa is not None and fa == fb:
                continue  # bound by an arithmetic identity — trivial
            fkey = (fa if fa is not None else f"c:{a}",
                    fb if fb is not None else f"c:{b}")
            if fkey in seen_fams:
                continue  # one representative per family pair
            seen_fams.add(fkey)
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
                                 f"(r={r_full:+.2f}) — SOLID",
                                 f"useful for sanity-checking a "
                                 f"forecast of one from the other; "
                                 f"it does not say which causes "
                                 f"which"))
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
        if (v == 0).mean() > 0.5:
            continue  # sparse column (coupons etc.) — trivially
            # concentrated; the coupon organ owns that story
        s = np.sort(v)[::-1]
        k = max(1, len(s) // 5)
        share = s[:k].sum() / s.sum()
        if share >= 0.5:
            lines.append(f"- **{h}**: the top 20% of rows carry "
                         f"{share:.0%} of the whole total — the "
                         f"weight is concentrated, not spread.")
            findings.append((share - 0.2,
                             f"top 20% of rows carry {share:.0%} "
                             f"of {h} — concentrated",
                             f"a small slice carries the business "
                             f"— protect it, and know it by name"))
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
                lines.append(f"- **{h}**: value {human(val)} sits far "
                             f"outside the rest ({score:,.0f}x the "
                             f"typical spread from the middle "
                             f"{human(med)}) — worth a look, kept "
                             f"as-is.")
            findings.append((min(1.0, len(hits) / 3),
                             f"{len(hits)} value(s) in {h} sit far "
                             f"outside the rest",
                             f"confirm entry error vs real event "
                             f"before these skew any averages"))
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


# -------------------------------------------------------------- periods
def periods_block(rows, header, cols, findings):
    """Month over month — the question every business asks first.
    These are period FACTS (exact arithmetic on the table), not
    inferences, so they carry no stability grade — just receipts."""
    date_col = next((h for h, c in cols.items()
                     if c["kind"] == "date"), None)
    nums = [h for h, c in cols.items() if c["kind"] == "numeric"][:3]
    cats = [h for h, c in cols.items() if c["kind"] == "category"][:1]
    md, kpi_subs = [], {}
    if not date_col:
        return md, kpi_subs  # monthly volume works with no measures
    ci = header.index(date_col)

    def month_of(r):
        m = re.match(r"^(\d{4})-(\d{1,2})-\d{1,2}$", r[ci]) \
            if ci < len(r) else None
        return f"{m.group(1)}-{int(m.group(2)):02d}" if m else None

    months_seen = sorted({month_of(r) for r in rows
                          if month_of(r)})
    if len(months_seen) < 3:
        return md, kpi_subs
    # the last month in the file may be part-way through — use the
    # last two COMPLETE months for the comparison, and say so.
    last_dates = [cols[date_col]["dates"][-1]]
    partial = last_dates[0].day < 25
    full_months = months_seen[:-1] if partial else months_seen
    if len(full_months) < 2:
        return md, kpi_subs
    m1, m0 = full_months[-1], full_months[-2]
    md += ["## Month over month (period facts, exact)", ""]
    for num in nums + ["volume (rows)"]:
        tot = {}
        if num == "volume (rows)":
            for r in rows:
                mo = month_of(r)
                if mo:
                    tot[mo] = tot.get(mo, 0.0) + 1.0
        else:
            ni = header.index(num)
            for r in rows:
                mo = month_of(r)
                if mo and ni < len(r) and _is_num(r[ni]):
                    tot[mo] = tot.get(mo, 0.0) + float(r[ni])
        if m1 not in tot or m0 not in tot or tot[m0] == 0:
            continue
        chg = (tot[m1] - tot[m0]) / abs(tot[m0])
        line = (f"- **{num}**: {m1} total {human(tot[m1])}, "
                f"{chg:+.1%} vs {m0} ({human(tot[m0])})")
        yoy_m = f"{int(m1[:4]) - 1}{m1[4:]}"
        if yoy_m in tot and tot[yoy_m] != 0:
            yoy = (tot[m1] - tot[yoy_m]) / abs(tot[yoy_m])
            line += (f"; {yoy:+.1%} vs the same month last year "
                     f"({human(tot[yoy_m])})")
        if partial:
            line += (f". The file's final month is part-way "
                     f"through and was left out of this comparison")
        md.append(line)
        kpi_subs[num] = f"{m1}: {human(tot[m1])} ({chg:+.0%} MoM)"
        if abs(chg) >= 0.05:
            findings.append(
                (min(1.0, abs(chg)) + 0.01,
                 f"{num} {'up' if chg > 0 else 'down'} {chg:+.1%} "
                 f"in {m1} vs {m0} — period fact, exact",
                 "worth knowing what changed that month — "
                 "price, volume, mix, or the calendar"))
    # who rose fastest: top groups, last vs prior full month
    for cat in cats:
        cj = header.index(cat)
        num = nums[0]
        ni = header.index(num)
        per = {}
        for r in rows:
            mo = month_of(r)
            if mo in (m0, m1) and cj < len(r) and r[cj] != "" \
                    and ni < len(r) and _is_num(r[ni]):
                per.setdefault(r[cj], {m0: 0.0, m1: 0.0})
                per[r[cj]][mo] += float(r[ni])
        movers = [(v[m1] - v[m0], (v[m1] - v[m0]) / abs(v[m0])
                   if v[m0] else 0.0, g) for g, v in per.items()
                  if v[m0] or v[m1]]
        if len(movers) >= 2:
            movers.sort(reverse=True)
            up_d, up_p, up_g = movers[0]
            dn_d, dn_p, dn_g = movers[-1]
            up_word = ("biggest riser" if up_p > 0
                       else "held up best")
            dn_word = ("biggest faller" if dn_p < 0
                       else "slowest riser")
            md.append(f"- **{num} movers by {cat}** ({m0} → {m1}): "
                      f"{up_word} '{up_g}' ({up_p:+.1%}, "
                      f"{human(up_d)}); {dn_word} "
                      f"'{dn_g}' ({dn_p:+.1%}, {human(dn_d)})")
    md.append("")
    return md, kpi_subs


# ------------------------------------------------------------ customers
ID_NAME_RE = re.compile(r"(customer|client|buyer|member|account|"
                        r"user|email|patient|donor)", re.I)


def customers_block(rows, header, cols, findings):
    """The customer base — who buys, who comes back, who carries
    you. Exact counts on the table (facts, receipted), the analysis
    a small business almost never has."""
    cand = find_customer_col(cols)
    if not cand:
        return []
    ci = header.index(cand)
    date_col = next((h for h, c in cols.items()
                     if c["kind"] == "date"), None)
    di = header.index(date_col) if date_col else None
    # revenue-like column: the numeric column with the largest total
    num = None
    best_tot = -1
    for h, c in cols.items():
        if c["kind"] == "numeric" and len(c["nums"]):
            if c["nums"].sum() > best_tot:
                best_tot, num = c["nums"].sum(), h
    ni = header.index(num) if num else None

    # ORDERS, never rows — a 20-line invoice is ONE visit. The
    # blind court caught the row-counting artifact; never again.
    ogroups, _how = _find_orders(rows, header, cols)
    orders = {}
    guest_rows, guest_val, total_val = 0, 0.0, 0.0
    for r in rows:
        if ni is not None and ni < len(r) and _is_num(r[ni]):
            total_val += float(r[ni])
        if ci < len(r) and r[ci] == "":
            guest_rows += 1
            if ni is not None and ni < len(r) and _is_num(r[ni]):
                guest_val += float(r[ni])
    if ogroups:
        for okey, ls in ogroups.items():
            c = ls[0][ci] if ci < len(ls[0]) else ""
            if c == "":
                continue
            rec = dict(d=None, v=0.0)
            for r in ls:
                if di is not None and di < len(r):
                    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$",
                                 r[di])
                    if m and rec["d"] is None:
                        rec["d"] = datetime.date(
                            *map(int, m.groups()))
                if ni is not None and ni < len(r) \
                        and _is_num(r[ni]):
                    rec["v"] += float(r[ni])
            orders.setdefault(c, []).append(rec)
    else:
        for r in rows:
            if ci < len(r) and r[ci] != "":
                rec = dict(d=None, v=0.0)
                if di is not None and di < len(r):
                    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$",
                                 r[di])
                    if m:
                        rec["d"] = datetime.date(
                            *map(int, m.groups()))
                if ni is not None and ni < len(r) \
                        and _is_num(r[ni]):
                    rec["v"] = float(r[ni])
                orders.setdefault(r[ci], []).append(rec)
    n_cust = len(orders)
    n_ord = sum(len(v) for v in orders.values())
    if n_cust < 5 or n_ord < 20:
        return []
    md = [f"## The customer base (from column '{cand}')", ""]
    span_days = None
    all_d = [o["d"] for v in orders.values() for o in v if o["d"]]
    if all_d:
        span_days = (max(all_d) - min(all_d)).days + 1
    repeat = [c for c, v in orders.items() if len(v) > 1]
    line = (f"- **{n_cust:,} identified customers** across "
            f"{n_ord:,} orders — {n_ord / n_cust:.1f} orders per "
            f"customer; {len(repeat) / n_cust:.0%} placed more "
            f"than one order")
    if span_days is not None and span_days < 30:
        line += (f". CAUTION: this file covers only {span_days} "
                 f"day(s) — no loyalty or repeat-rate conclusion "
                 f"should be drawn from a window this short")
    md.append(line)
    if guest_rows and total_val > 0:
        md.append(f"- **{guest_rows:,} rows have NO customer "
                  f"attached** — carrying {guest_val / total_val:.0%}"
                  f" of {num}. Every customer claim in this report "
                  f"covers only the identified share; the blank "
                  f"slice is where an owner should look first "
                  f"(guest checkout? a channel booked without "
                  f"ids?)")
        if guest_val / total_val >= 0.15:
            findings.append(
                (guest_val / total_val,
                 f"{guest_val / total_val:.0%} of {num} has no "
                 f"customer attached — invisible to any customer "
                 f"analysis",
                 f"find out which channel books without ids; your "
                 f"biggest orders may be living there"))
    if num:
        spend = sorted((sum(o['v'] for o in v), c)
                       for c, v in orders.items())[::-1]
        whole = sum(s for s, _ in spend) or 1e-12
        top5 = sum(s for s, _ in spend[:5])
        topfifth = sum(s for s, _ in
                       spend[:max(1, n_cust // 5)])
        md.append(f"- **top 5 customers carry "
                  f"{top5 / whole:.0%} of {num}** "
                  f"({', '.join(c for _, c in spend[:5])}); the top "
                  f"20% of customers carry {topfifth / whole:.0%}")
        md.append(f"- average {num} per order {human(whole / n_ord)}"
                  f"; per customer {human(whole / n_cust)}")
        findings.append(
            (topfifth / whole,
             f"the top 20% of customers carry "
             f"{topfifth / whole:.0%} of {num}",
             f"these are the relationships to protect — a single "
             f"defection up there outweighs dozens below"))
    if date_col:
        first = {c: min(o["d"] for o in v if o["d"])
                 for c, v in orders.items()
                 if any(o["d"] for o in v)}
        months = sorted({d.strftime("%Y-%m")
                         for d in first.values()})
        if len(months) >= 3:
            recent = months[-2]  # last full-ish month
            new_m = sum(1 for c, d in first.items()
                        if d.strftime("%Y-%m") == recent)
            active = {c for c, v in orders.items()
                      if any(o["d"] and o["d"].strftime("%Y-%m")
                             == recent for o in v)}
            md.append(f"- in {recent}: {len(active):,} active "
                      f"customers, {new_m:,} of them first-timers "
                      f"({(len(active) - new_m):,} returning)")
        gaps = []
        for c, v in orders.items():
            ds = sorted(o["d"] for o in v if o["d"])
            gaps += [(ds[i + 1] - ds[i]).days
                     for i in range(len(ds) - 1)]
        if gaps and (span_days or 0) >= 30:
            md.append(f"- a returning customer typically comes "
                      f"back within {int(np.median(gaps))} days "
                      f"(median gap between orders) — silence "
                      f"much longer than that is a customer "
                      f"drifting away")
        if (span_days or 0) >= 30:
            findings.append(
                (len(repeat) / n_cust,
                 f"{len(repeat) / n_cust:.0%} of customers ordered "
                 f"more than once",
                 f"repeat business is the cheapest revenue there "
                 f"is — the gap number above says when to reach "
                 f"out"))
    md.append("")
    return md


# --------------------------------------------- returns & adjustments
def _money_col(cols):
    num, best = None, -1
    for h, c in cols.items():
        if c["kind"] == "numeric" and len(c["nums"]) \
                and c["nums"].sum() > best:
            best, num = c["nums"].sum(), h
    return num


def negatives_block(rows, header, cols, findings):
    """Negative money is a story, never just noise: returns,
    reversals, write-offs. Named by source, clustered by day —
    the blind court's biggest miss, cured."""
    num = _money_col(cols)
    if not num:
        return []
    ni = header.index(num)
    neg = [(k + 2, float(r[ni])) for k, r in enumerate(rows)
           if ni < len(r) and _is_num(r[ni]) and float(r[ni]) < 0]
    if len(neg) < 3:
        return []
    gross = sum(float(r[ni]) for r in rows
                if ni < len(r) and _is_num(r[ni])
                and float(r[ni]) > 0) or 1e-12
    tot_neg = sum(v for _, v in neg)
    md = ["## Returns, reversals, and write-offs", ""]
    md.append(f"- **{len(neg):,} rows carry negative {num}**, "
              f"totaling {human(tot_neg)} against {human(gross)} "
              f"gross ({abs(tot_neg) / gross:.1%} of gross)")
    # name the sources: which label concentrates the negative value
    neg_idx = {k for k, _ in neg}
    best_col, best_share, best_top = None, 0, []
    for h, c in cols.items():
        if c["kind"] not in ("identifier", "category", "text"):
            continue
        hi = header.index(h)
        by = {}
        for k, r in enumerate(rows):
            if (k + 2) in neg_idx and hi < len(r) and r[hi]:
                by[r[hi]] = by.get(r[hi], 0.0) + float(r[ni])
        if not by or len(by) > 2000:
            continue
        top = sorted(by.items(), key=lambda kv: kv[1])[:3]
        share = top[0][1] / tot_neg if tot_neg else 0
        # a source only counts if negatives concentrate there MORE
        # than the business does overall (UK being 95% of sales
        # makes 'UK' a non-answer for where reversals live)
        n_all = sum(1 for r in rows if hi < len(r) and r[hi])
        base = sum(1 for r in rows if hi < len(r)
                   and r[hi] == top[0][0]) / max(n_all, 1)
        score = share - base
        if score > best_share:
            best_col, best_share, best_top = h, score, top
    if best_col:
        md.append(f"- **where the negatives live** (by {best_col}): "
                  + "; ".join(f"'{g}' {human(v)}"
                              for g, v in best_top))
    # day clusters: one bad day carrying a big share of negatives
    dcol = next((h for h, c in cols.items()
                 if c["kind"] == "date"), None)
    if dcol:
        di = header.index(dcol)
        by_day = {}
        for k, r in enumerate(rows):
            if (k + 2) in neg_idx and di < len(r) and r[di]:
                by_day.setdefault(r[di], []).append(float(r[ni]))
        if by_day:
            worst = min(by_day.items(), key=lambda kv: sum(kv[1]))
            wsum = sum(worst[1])
            if tot_neg and wsum / tot_neg >= 0.3:
                md.append(f"- **a single day carries "
                          f"{wsum / tot_neg:.0%} of all negative "
                          f"{num}**: {worst[0]}, {len(worst[1])} "
                          f"entries totaling {human(wsum)} — that "
                          f"is an EVENT, not a trickle; worth "
                          f"pulling the source records for that "
                          f"day")
                findings.append(
                    (min(1.0, abs(wsum) / gross) + 0.6,
                     f"{worst[0]}: {len(worst[1])} negative "
                     f"entries totaling {human(wsum)} in one day "
                     f"({abs(wsum) / gross:.1%} of gross {num})",
                     f"one day, one story — find out what "
                     f"happened on {worst[0]} before averaging "
                     f"over it"))
    if abs(tot_neg) / gross >= 0.05:
        findings.append(
            (abs(tot_neg) / gross,
             f"negative {num} runs {abs(tot_neg) / gross:.1%} of "
             f"gross" + (f", concentrated in "
                         f"'{best_top[0][0]}'" if best_top else ""),
             f"separate true returns from fees and write-offs "
             f"before judging the business — they are different "
             f"problems"))
    md.append("")
    return md


# ------------------------------------------------------------- whales
def orders_whales_block(rows, header, cols, findings):
    """Orders as first-class objects: the size of a typical order,
    and the whales — the blind court showed the biggest orders can
    carry a fifth of the business and even lack a customer id."""
    ogroups, how = _find_orders(rows, header, cols)
    num = _money_col(cols)
    if not ogroups or not num or len(ogroups) < 30:
        return []
    ni = header.index(num)
    cust = find_customer_col(cols)
    ci = header.index(cust) if cust else None
    dcol = next((h for h, c in cols.items()
                 if c["kind"] == "date"), None)
    di = header.index(dcol) if dcol else None
    vals = []
    for okey, ls in ogroups.items():
        v = sum(float(r[ni]) for r in ls
                if ni < len(r) and _is_num(r[ni]))
        guest = (ci is not None and ls[0][ci] == "") if ci is not \
            None else False
        d = ls[0][di] if di is not None and di < len(ls[0]) else ""
        vals.append((v, str(okey), len(ls), guest, d))
    pos = sorted((x for x in vals if x[0] > 0), reverse=True)
    if len(pos) < 20:
        return []
    total = sum(v for v, *_ in pos)
    top10 = sum(v for v, *_ in pos[:10])
    med_v = float(np.median([v for v, *_ in pos]))
    med_lines = float(np.median([n for _, _, n, _, _ in pos]))
    md = ["## The orders themselves", ""]
    md.append(f"- {len(pos):,} orders with positive {num}; the "
              f"typical order is {human(med_v)} across "
              f"{med_lines:.0f} line(s)")
    md.append(f"- **the top 10 orders alone carry "
              f"{top10 / total:.0%} of {num}**: " +
              "; ".join(f"{okey} {human(v)}"
                        + (f" on {d}" if d else "")
                        + (" — NO CUSTOMER ID" if g else "")
                        for v, okey, _, g, d in pos[:3]))
    n_guest_top = sum(1 for _, _, _, g, _ in pos[:10] if g)
    if n_guest_top:
        md.append(f"- {n_guest_top} of the top 10 orders have no "
                  f"customer attached — your biggest buyers are "
                  f"invisible to any customer analysis")
    if top10 / total >= 0.15:
        findings.append(
            (top10 / total + 0.3,
             f"the top 10 orders carry {top10 / total:.0%} of "
             f"{num}" + (f"; {n_guest_top} of them have no "
                         f"customer id" if n_guest_top else ""),
             f"key-account risk: know these buyers by name — one "
             f"of them changing supplier is a bad quarter"))
    md.append("")
    return md


# -------------------------------------------------------- did it work
def window_block(rows, header, cols, w_start, w_end, w_label,
                 findings):
    """Campaign / promo / change effectiveness: actual during the
    window vs what the SAME WEEKDAYS delivered in the surrounding
    eight weeks. Uplift graded by resampling baseline days — an
    honest answer to 'did it work', weekday pattern respected."""
    date_col = next((h for h, c in cols.items()
                     if c["kind"] == "date"), None)
    nums = [h for h, c in cols.items() if c["kind"] == "numeric"][:3]
    if not date_col or not nums:
        return []
    di = header.index(date_col)
    try:
        d0 = datetime.date(*map(int, w_start.split("-")))
        d1 = datetime.date(*map(int, w_end.split("-")))
    except (ValueError, AttributeError):
        return [f"## Did it work — {w_label}", "",
                f"- window dates unreadable ({w_start}..{w_end}) — "
                f"use YYYY-MM-DD; nothing computed rather than "
                f"guessed", ""]
    base_lo = d0 - datetime.timedelta(days=28)
    base_hi = d1 + datetime.timedelta(days=28)
    md = [f"## Did it work — {w_label} "
          f"({d0.isoformat()} → {d1.isoformat()})", ""]
    rng = np.random.default_rng(SEED + 7)
    for num in nums:
        ni = header.index(num)
        daily = {}
        for r in rows:
            if di < len(r) and ni < len(r) and _is_num(r[ni]):
                m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", r[di])
                if not m:
                    continue
                dd = datetime.date(*map(int, m.groups()))
                daily[dd] = daily.get(dd, 0.0) + float(r[ni])
        win_days = {d: v for d, v in daily.items() if d0 <= d <= d1}
        base_days = {d: v for d, v in daily.items()
                     if base_lo <= d <= base_hi
                     and not (d0 <= d <= d1)}
        if len(win_days) < 3 or len(base_days) < 7:
            md.append(f"- **{num}**: not enough days in the window "
                      f"or the surrounding weeks to judge — said "
                      f"plainly rather than guessed")
            continue
        by_wd = {}
        for d, v in base_days.items():
            by_wd.setdefault(d.weekday(), []).append(v)
        missing = [d for d in win_days if d.weekday() not in by_wd]
        if missing:
            md.append(f"- **{num}**: baseline lacks matching "
                      f"weekdays for {len(missing)} window day(s) "
                      f"— refused rather than mismatched")
            continue
        actual = sum(win_days.values())
        expected = sum(float(np.mean(by_wd[d.weekday()]))
                       for d in win_days)
        uplift = (actual - expected) / abs(expected)
        hold, uplifts = 0, []
        for _ in range(TRIALS):
            exp_i = sum(float(rng.choice(by_wd[d.weekday()],
                                         size=len(by_wd[d.weekday()]),
                                         replace=True).mean())
                        for d in win_days)
            uplifts.append((actual - exp_i) / abs(exp_i))
            if (actual > exp_i) == (uplift > 0):
                hold += 1
        frac = hold / TRIALS
        grade = _grade(frac)
        lo, hi = np.percentile(uplifts, [2.5, 97.5])
        line = (f"- **{num}** during the window: {human(actual)} "
                f"actual vs {human(expected)} expected from the "
                f"same weekdays in the surrounding eight weeks — "
                f"**{uplift:+.1%}** (baseline uncertainty puts it "
                f"{lo:+.0%} to {hi:+.0%}) — {grade} (direction "
                f"held in {frac:.0%} of {TRIALS} baseline "
                f"resamples)")
        if len(win_days) < 14:
            line += (f". A {len(win_days)}-day window also carries "
                     f"its own luck — read the size as indicative, "
                     f"the direction as solid")
        md.append(line)
        if grade == "SOLID" and abs(uplift) >= 0.05:
            findings.append(
                (abs(uplift) + 0.5,
                 f"{w_label}: {num} ran {uplift:+.1%} vs the "
                 f"weekday-matched baseline — SOLID",
                 f"that is the measured effect, seasonality "
                 f"respected — worth repeating if the margin "
                 f"covered the cost"))
    md.append("")
    return md


# -------------------------------------------------------------- baskets
ORDER_NAME_RE = re.compile(r"(order|invoice|transaction|receipt|"
                           r"ticket|basket)", re.I)


def find_customer_col(cols):
    """The customer column: NAMED like one (any kind — CustomerID
    is numbers, still a customer), never a date, repeats required.
    No shape-guessing: a column that merely repeats is not a
    customer, and a wrong customer analysis is worse than none."""
    for h, c in cols.items():
        if c["kind"] == "date":
            continue
        if c["tokens"] & CUSTOMER_TOKENS:
            vals = [v for v in c["vals"] if v != ""]
            if 4 < len(set(vals)) < len(vals):
                return h
    return None


def derive_measures(rows, header, cols=None):
    """What an analyst computes before analyzing: a quantity times
    a price IS the revenue — derived, receipted, named as derived.
    Only when no revenue-like column already exists."""
    cols = cols or classify_columns(rows, header)
    toks = {h: c["tokens"] for h, c in cols.items()}
    has_money = any(t & {"revenue", "sales", "amount", "total",
                         "dollars", "sale"}
                    for h, t in toks.items()
                    if cols[h]["kind"] in ("numeric",))
    qty = next((h for h, c in cols.items() if c["kind"] == "numeric"
                and c["tokens"] & {"quantity", "qty", "units",
                                   "bottles", "count", "volume"}),
               None)
    rate = next((h for h, c in cols.items() if c["kind"] == "rate"
                 and c["tokens"] & {"price", "cost"}), None)
    if has_money or not qty or not rate:
        return []
    qi, ri = header.index(qty), header.index(rate)
    name = f"revenue (derived: {qty} x {rate})"
    header.append(name)
    n_ok = 0
    for r in rows:
        try:
            r.append(f"{float(r[qi]) * float(r[ri]):.2f}")
            n_ok += 1
        except (ValueError, IndexError):
            r.append("")
    return [f"derived column '{name}' — the money column your "
            f"file implied but didn't carry ({n_ok:,} rows); "
            f"marked as derived everywhere it appears"]


def _find_orders(rows, header, cols):
    """Group line items into orders: an order-NAMED column of any
    kind (InvoiceNo is numbers, still an order id), never a date;
    customer+date as the fallback key."""
    for h, c in cols.items():
        if c["kind"] == "date":
            continue
        if c["tokens"] & ORDER_TOKENS:
            vals = [v for v in c["vals"] if v != ""]
            if 1 < len(set(vals)) < len(vals):
                i = header.index(h)
                orders = {}
                for r in rows:
                    if i < len(r) and r[i] != "":
                        orders.setdefault(r[i], []).append(r)
                return orders, f"column '{h}'"
    cust = find_customer_col(cols)
    date_col = next((h for h, c in cols.items()
                     if c["kind"] == "date"), None)
    if cust and date_col:
        ci, di = header.index(cust), header.index(date_col)
        orders = {}
        for r in rows:
            if ci < len(r) and di < len(r) and r[ci] and r[di]:
                orders.setdefault((r[ci], r[di]), []).append(r)
        return orders, f"'{cust}' + '{date_col}' (no order id)"
    return None, None


def baskets_block(rows, header, cols, findings):
    """What sells together — line items into baskets, pair lift vs
    independence, graded by half-samples of ORDERS."""
    orders, how = _find_orders(rows, header, cols)
    prod_col = None
    for h, c in cols.items():
        if c["kind"] == "category" \
                and not (c["tokens"] & CUSTOMER_TOKENS) \
                and c["tokens"] & {"product", "item", "sku",
                                   "category", "service", "plan"}:
            prod_col = h
            break
    if prod_col is None:
        prod_col = next((h for h, c in cols.items()
                         if c["kind"] == "category"
                         and not (c["tokens"] & CUSTOMER_TOKENS)),
                        None)
    if not orders or not prod_col or len(orders) < 30:
        return []
    pi = header.index(prod_col)
    baskets = [set(r[pi] for r in ls if pi < len(r) and r[pi])
               for ls in orders.values()]
    multi = [b for b in baskets if len(b) >= 2]
    n = len(baskets)
    if len(multi) < 10:
        return []
    md = [f"## What sells together (orders from {how})", ""]
    sizes = [len(set(r[pi] for r in ls if pi < len(r)))
             for ls in orders.values()]
    md.append(f"- {n:,} orders; {len(multi) / n:.0%} carry more "
              f"than one distinct {prod_col}; typical basket "
              f"{np.median([len(ls) for ls in orders.values()]):.0f}"
              f" line(s)")
    items = sorted({x for b in baskets for x in b})
    rate = {x: sum(1 for b in baskets if x in b) / n for x in items}
    pairs = []
    for i_a in range(len(items)):
        for i_b in range(i_a + 1, len(items)):
            a, b = items[i_a], items[i_b]
            both = sum(1 for bk in baskets if a in bk and b in bk)
            if both < max(5, n // 200):
                continue
            lift = (both / n) / (rate[a] * rate[b])
            pairs.append((lift, both, a, b))
    pairs.sort(reverse=True)
    rng = np.random.default_rng(SEED + 11)
    shown = 0
    for lift, both, a, b in pairs:
        if lift < 1.5 or shown >= 3:
            break
        hold = 0
        idx = np.arange(n)
        for _ in range(TRIALS):
            pick = rng.choice(idx, size=n // 2, replace=False)
            sub = [baskets[i] for i in pick]
            ns = len(sub)
            ra = sum(1 for bk in sub if a in bk) / ns
            rb = sum(1 for bk in sub if b in bk) / ns
            bo = sum(1 for bk in sub if a in bk and b in bk) / ns
            if ra > 0 and rb > 0 and bo / (ra * rb) >= 1.2:
                hold += 1
        frac = hold / TRIALS
        grade = _grade(frac)
        if grade == "UNSTABLE":
            continue
        shown += 1
        md.append(f"- **'{a}' and '{b}' go together**: bought in "
                  f"the same order {lift:.1f}x more often than "
                  f"chance ({both} orders) — {grade} (lift held in "
                  f"{frac:.0%} of {TRIALS} half-samples of orders)")
        if grade == "SOLID":
            findings.append(
                (min(1.0, lift / 4) + 0.2,
                 f"'{a}' and '{b}' sell together at {lift:.1f}x "
                 f"chance — SOLID",
                 f"bundle them, shelve them together, or suggest "
                 f"one at the other's checkout"))
    if shown == 0:
        md.append("- no product pair beats chance strongly enough "
                  "to say — said plainly.")
    md.append("")
    return md


# -------------------------------------------------------------- coupons
COUPON_NAME_RE = re.compile(r"(coupon|discount|promo|rebate|"
                            r"markdown|voucher)", re.I)


def coupons_block(rows, header, cols, findings):
    """Coupons: who redeems, what it costs, and the cannibalization
    question — did discounts buy NEW traffic, or subsidize orders
    the regulars were going to place anyway?"""
    dcol = next((h for h, c in cols.items()
                 if COUPON_NAME_RE.search(h)
                 and c["kind"] == "numeric"), None)
    if not dcol:
        return []
    di = header.index(dcol)
    orders, how = _find_orders(rows, header, cols)
    if not orders or len(orders) < 30:
        return []
    cust = find_customer_col(cols)
    date_col = next((h for h, c in cols.items()
                     if c["kind"] == "date"), None)
    n = len(orders)
    coup = {o: ls for o, ls in orders.items()
            if any(di < len(r) and _is_num(r[di])
                   and float(r[di]) > 0 for r in ls)}
    if len(coup) < 5:
        return []
    md = [f"## The coupon question (column '{dcol}')", ""]
    given = sum(float(r[di]) for ls in coup.values() for r in ls
                if di < len(r) and _is_num(r[di]))
    md.append(f"- **{len(coup) / n:.0%} of orders carried a "
              f"{dcol}** ({len(coup):,} of {n:,}); total given "
              f"away {human(given)} — that is the measurable cost "
              f"side, exact")
    if cust and date_col:
        ci = header.index(cust)
        cdi = header.index(date_col)
        first = {}
        for o, ls in sorted(orders.items(),
                            key=lambda kv: kv[1][0][cdi]):
            c = ls[0][ci]
            first.setdefault(c, ls[0][cdi])
        new_redeem = sum(1 for o, ls in coup.items()
                         if first.get(ls[0][ci]) == ls[0][cdi])
        md.append(f"- **who redeems**: {new_redeem:,} coupon "
                  f"orders ({new_redeem / len(coup):.0%}) were a "
                  f"customer's FIRST-EVER order (acquisition); "
                  f"{len(coup) - new_redeem:,} "
                  f"({1 - new_redeem / len(coup):.0%}) went to "
                  f"customers you already had")
        # cannibalization: repeat customers' coupon orders that
        # arrived on their NORMAL cadence anyway
        cad = {}
        for o, ls in orders.items():
            cad.setdefault(ls[0][ci], []).append(ls[0][cdi])
        import datetime as _dt

        def _d(s):
            return _dt.date(*map(int, s.split("-")))
        anyway = 0
        judged = 0
        for o, ls in coup.items():
            c, dstr = ls[0][ci], ls[0][cdi]
            ds = sorted(set(cad.get(c, [])))
            if len(ds) < 4:
                continue
            gaps = [( _d(ds[i + 1]) - _d(ds[i])).days
                    for i in range(len(ds) - 1)]
            med = np.median(gaps)
            i = ds.index(dstr)
            if i == 0:
                continue
            judged += 1
            if (_d(dstr) - _d(ds[i - 1])).days <= 1.5 * med:
                anyway += 1
        if judged >= 10:
            md.append(f"- **the cannibalization signal**: of "
                      f"{judged:,} coupon orders from established "
                      f"repeat customers, {anyway / judged:.0%} "
                      f"arrived on the customer's normal ordering "
                      f"rhythm anyway — those discounts likely "
                      f"subsidized purchases that were coming "
                      f"regardless. Not a causal proof — a rhythm "
                      f"receipt.")
            findings.append(
                (anyway / judged,
                 f"{anyway / judged:.0%} of regulars' coupon "
                 f"redemptions arrived on their normal rhythm — "
                 f"likely subsidized anyway-purchases",
                 f"aim coupons at first-timers and lapsed "
                 f"customers, not the rhythm that already runs"))
    md.append("")
    return md


# ------------------------------------------------------------- at risk
def at_risk_block(rows, header, cols, findings):
    """High-value customers gone quiet past their OWN cadence —
    the call list. Facts on the table, named."""
    cust = find_customer_col(cols)
    date_col = next((h for h, c in cols.items()
                     if c["kind"] == "date"), None)
    num, best_tot = None, -1
    for h, c in cols.items():
        if c["kind"] == "numeric" and len(c["nums"]) \
                and c["nums"].sum() > best_tot:
            best_tot, num = c["nums"].sum(), h
    if not cust or not date_col or not num:
        return []
    ci, di, ni = (header.index(cust), header.index(date_col),
                  header.index(num))
    per = {}
    for r in rows:
        if ci < len(r) and di < len(r) and r[ci] and r[di]:
            v = (float(r[ni]) if ni < len(r) and _is_num(r[ni])
                 else 0.0)
            per.setdefault(r[ci], []).append((r[di], v))
    if len(per) < 10:
        return []
    today = max(d for v in per.values() for d, _ in v)

    def _d(s):
        return datetime.date(*map(int, s.split("-")))
    spend = sorted(((sum(v for _, v in recs), c)
                    for c, recs in per.items()), reverse=True)
    top = {c for _, c in spend[:max(1, len(spend) // 5)]}
    risks = []
    for c in top:
        ds = sorted({d for d, _ in per[c]})
        if len(ds) < 3:
            continue
        gaps = [(_d(ds[i + 1]) - _d(ds[i])).days
                for i in range(len(ds) - 1)]
        med = float(np.median(gaps))
        silence = (_d(today) - _d(ds[-1])).days
        if silence > max(14, 2 * med):
            risks.append((silence / max(med, 1), c,
                          sum(v for _, v in per[c]), ds[-1],
                          silence, med))
    if not risks:
        return []
    risks.sort(reverse=True)
    md = ["## High-value customers going quiet", ""]
    for _, c, s, last, silence, med in risks[:5]:
        md.append(f"- **{c}** — {human(s)} lifetime {num}, "
                  f"normally orders every ~{med:.0f} days, silent "
                  f"for {silence} days (last seen {last}). That is "
                  f"{silence / max(med, 1):.1f}x their own rhythm.")
    findings.append(
        (0.9,
         f"{len(risks)} high-value customer(s) have gone quiet "
         f"past twice their own ordering rhythm",
         f"this is a call list, not a statistic — the names are "
         f"in the report"))
    md.append("")
    return md


# --------------------------------------------------------- reconcile
def reconcile_block(rows, header, cols, findings):
    """Columns that should obey arithmetic — tax as a rate of
    revenue, totals as sums — verified row by row. Where the book
    doesn't balance, the rows are named. This is the estimated-vs-
    actually-paid check, generalized."""
    nums = [h for h, c in cols.items()
            if c["kind"] == "numeric" and len(c["nums"]) >= 20]
    if len(nums) < 2:
        return [], []
    idx = {h: header.index(h) for h in nums}

    def col(h):
        out = []
        for k, r in enumerate(rows):
            i = idx[h]
            if i < len(r) and _is_num(r[i]):
                out.append((k + 2, float(r[i])))  # +2: header, 1-base
            else:
                out.append((k + 2, None))
        return out
    series = {h: col(h) for h in nums}
    md_lines = []
    ratio_pairs = []
    # SUM laws first: A ~ B + C. A found sum implies the A:B and
    # A:C ratios — those get suppressed below, per the owner's law:
    # only submit the strongest, never six spellings of two rules.
    implied = set()
    for a in nums:
        for i_b in range(len(nums)):
            for i_c in range(i_b + 1, len(nums)):
                b, c = nums[i_b], nums[i_c]
                if a in (b, c):
                    continue
                trip = [(ra, va, vb, vc) for (ra, va), (_, vb),
                        (_, vc) in zip(series[a], series[b],
                                       series[c])
                        if None not in (va, vb, vc)]
                if len(trip) < 20:
                    continue
                res = np.array([va - vb - vc
                                for _, va, vb, vc in trip])
                scale = np.median([abs(va) for _, va, _, _
                                   in trip]) or 1
                ok = np.abs(res) <= max(0.02, 0.001 * scale)
                if not (0.95 <= ok.mean() < 1.0):
                    continue
                bad = [(rw_, va, vb, vc) for (rw_, va, vb, vc), k
                       in zip(trip, ok) if not k][:5]
                implied.update({frozenset((a, b)),
                                frozenset((a, c))})
                md_lines.append(
                    f"- **{a} = {b} + {c}** holds for "
                    f"{ok.mean():.1%} of rows — but NOT these: " +
                    "; ".join(f"row {rw_}: {a}={human(va)} vs "
                              f"{human(vb + vc)} expected"
                              for rw_, va, vb, vc in bad))
                findings.append(
                    (1.0,
                     f"{len(bad)} row(s) break {a}={b}+{c} — the "
                     f"book doesn't balance there",
                     f"these are the rows an audit would circle; "
                     f"fix or explain before anyone else finds "
                     f"them"))
    # constant-ratio laws: one direction per pair (the one with the
    # small coefficient — 'tax = 0.08 x revenue', never both), and
    # never a pair a sum law already implies.
    for i_a in range(len(nums)):
        for i_b in range(i_a + 1, len(nums)):
            a, b = nums[i_a], nums[i_b]
            if frozenset((a, b)) in implied:
                continue
            pairs = [(ra, va, vb) for (ra, va), (_, vb)
                     in zip(series[a], series[b])
                     if va is not None and vb not in (None, 0.0)
                     and va != 0.0]
            if len(pairs) < 20:
                continue
            ratios = np.array([va / vb for _, va, vb in pairs])
            r_med = float(np.median(ratios))
            if abs(r_med) > 1:  # flip so the coefficient is small
                a, b = b, a
                pairs = [(ra, vb, va) for ra, va, vb in pairs]
                ratios = np.array([va / vb for _, va, vb in pairs])
                r_med = float(np.median(ratios))
            if not (0.001 <= abs(r_med) <= 1) or \
                    abs(r_med - 1) < 0.005:
                continue
            ok = np.abs(ratios - r_med) <= max(0.01 * abs(r_med),
                                               1e-9)
            if not (0.95 <= ok.mean() < 1.0):
                continue
            bad = [(rw_, va, vb) for (rw_, va, vb), k
                   in zip(pairs, ok) if not k][:5]
            if not bad:
                continue
            ratio_pairs.append(frozenset((a, b)))
            md_lines.append(
                f"- **{a} runs at {r_med:.4g} x {b}** for "
                f"{ok.mean():.1%} of rows — but NOT for these: " +
                "; ".join(f"row {rw_}: {a}={human(va)} where "
                          f"{r_med:.4g}x{b}={human(r_med * vb)} "
                          f"expected" for rw_, va, vb in bad))
            findings.append(
                (1.0,
                 f"{len(bad)} row(s) break the {a}={r_med:.3g}x"
                 f"{b} rule the rest of the book obeys",
                 f"each one is either an entry error or an "
                 f"exception with a story — worth five minutes "
                 f"with the source records"))
    families = []
    for key in list(implied) + ratio_pairs:
        pair = set(key)
        hit = [f for f in families if f & pair]
        if hit:
            merged = pair.union(*hit)
            families = [f for f in families if not (f & pair)]
            families.append(merged)
        else:
            families.append(pair)
    if not md_lines:
        return [], families
    return (["## Where the book doesn't balance (reconciliation)",
             ""] + md_lines + [""]), families


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
        ax.annotate(human(hgt), (b.get_x() + b.get_width() / 2,
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
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: human(v)))
    ax.set_title(title, loc="left", fontsize=10, color=INK,
                 fontweight="bold")
    fig.tight_layout()
    p = os.path.join(chart_dir,
                     re.sub(r"[^\w.-]", "_", title) + ".png")
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


# ------------------------------------------------------------ assembly
def build_extras(rows, header, time_col, described, chart_dir,
                 window=None):
    """Returns (headline_md, section_md, kpis). Every line traces to
    arithmetic on the table; every comparative claim carries its
    half-sample grade. window=(start,end,label) adds the
    did-it-work analysis."""
    cols = classify_columns(rows, header)
    findings = []  # (score, text[, so_what]) for the headlines
    for name, d in (described or {}).items():
        dd = d["d"]
        if dd["grade"] == "SOLID":
            findings.append(
                (0.5 + abs(dd["total"]) / (abs(dd["total"]) + 1),
                 f"{name} is "
                 f"{'rising' if dd['slope'] > 0 else 'falling'} "
                 f"(about {human(dd['total'])} across the record) "
                 f"— SOLID",
                 f"if the pace holds, roughly "
                 f"{human(abs(dd['total']) * 0.25)} more over the "
                 f"next quarter of a record this length — "
                 f"extrapolation, plainly labeled"))
    blocks = []
    blocks += profile_block(cols, len(rows))
    per_md, kpi_subs = periods_block(rows, header, cols, findings)
    blocks += per_md
    if window and window[0] and window[1]:
        blocks += window_block(rows, header, cols, window[0],
                               window[1],
                               window[2] or "the window", findings)
    blocks += customers_block(rows, header, cols, findings)
    blocks += at_risk_block(rows, header, cols, findings)
    blocks += negatives_block(rows, header, cols, findings)
    blocks += orders_whales_block(rows, header, cols, findings)
    blocks += coupons_block(rows, header, cols, findings)
    blocks += baskets_block(rows, header, cols, findings)
    rec_md, families = reconcile_block(rows, header, cols, findings)
    blocks += rec_md
    seg_md, _ = segments_block(rows, header, cols, chart_dir,
                               findings)
    blocks += seg_md
    blocks += seasonality_block(rows, header, cols, findings)
    blocks += relationships_block(rows, header, cols, time_col,
                                  findings, families)
    blocks += concentration_block(rows, header, cols, findings)
    blocks += anomalies_block(rows, header, cols, findings)

    findings.sort(key=lambda f: -f[0])
    heads = ["## The headlines", ""]
    if findings:
        for tup in findings[:7]:
            heads.append(f"- {tup[1]}")
            if len(tup) > 2 and tup[2]:
                heads.append(f"- ↳ {tup[2]}")
    else:
        heads.append("- nothing in this table passed our stability "
                     "checks strongly enough to headline — the "
                     "details below say what was checked.")
    heads.append("")

    # the KPI band: the five numbers an owner wants in five seconds
    kpis = [dict(label="Rows", value=f"{len(rows):,}", sub="")]
    dcol = next((h for h, c in cols.items()
                 if c["kind"] == "date"), None)
    if dcol:
        d = cols[dcol]["dates"]
        kpis.append(dict(label="Covers",
                         value=f"{(max(d) - min(d)).days + 1} days",
                         sub=f"{min(d).isoformat()} to "
                             f"{max(d).isoformat()}"))
    for h, c in cols.items():
        if c["kind"] == "numeric" and len(c["nums"]) and \
                len(kpis) < 5:
            kpis.append(dict(label=f"Total {h}",
                             value=human(c["nums"].sum()),
                             sub=kpi_subs.get(h, "")))
    return heads, blocks, kpis

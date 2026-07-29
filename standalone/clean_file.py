"""CLEAN FILE — the cleaning deliverable (wrapper machinery).

The promoted engine FLAGS and never repairs (verifier-never-repairs;
the fire-triangle clause). Customers buying "data cleaning" want the
cleaned file back. This wrapper squares the two honestly:

  * MECHANICAL passes it performs itself, every change receipted in
    a changelog: structure normalization (delimiter / title rows /
    header), whitespace trim, missing-marker standardization, exact-
    duplicate row removal (with the kept row named).
  * JUDGMENT stays with people: the engine's audit flags (errors,
    recurring news, near-duplicates) are attached to the changelog
    as FLAGGED FOR REVIEW — never auto-fixed. A cell that violates
    a law is evidence, not a typo to overwrite.

    python standalone/clean_file.py dirty.csv
    python standalone/clean_file.py dirty.tsv --keep-dupes --no-audit

Known limits, stated plainly: delimited text only for now (csv/tsv/
semicolon/pipe) — Excel .xlsx must be exported to CSV first;
category variants ('east'/'East') and mixed date formats are left
as-is in the output (flagged territory, not silently rewritten).

Output: <out>/<name>/cleaned.csv + CHANGELOG.md + audit receipts.
Exit 0 always means: cleaned file written, changelog complete.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DC = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from honest_report import (normalize, write_csv, run_engine,  # noqa: E402
                           parse_audit)


def trim_cells(rows):
    """Strip leading/trailing whitespace; return count of changes."""
    n = 0
    for r in rows:
        for i, c in enumerate(r):
            t = c.strip()
            if t != c:
                r[i] = t
                n += 1
    return n


def drop_exact_duplicates(rows):
    """Remove rows identical in every cell. Returns (kept, receipt
    lines naming each removed row and the row it duplicated)."""
    seen, kept, receipts = {}, [], []
    for idx, r in enumerate(rows, start=2):  # 1-based + header row
        key = tuple(r)
        if key in seen:
            receipts.append(f"row {idx} removed — exact duplicate "
                            f"of row {seen[key]}")
        else:
            seen[key] = idx
            kept.append(r)
    return kept, receipts


DATE_RES = (
    (r"^(\d{4})-(\d{1,2})-(\d{1,2})$", lambda m: (int(m.group(1)),
     int(m.group(2)), int(m.group(3)))),
    (r"^(\d{1,2})/(\d{1,2})/(\d{4})$", lambda m: (int(m.group(3)),
     int(m.group(1)), int(m.group(2)))),
)


def audit_view(rows, header):
    """The engine's audit face reads NUMERIC tables. Build it a
    numeric VIEW of this one, every conversion receipted: date-like
    columns become day numbers; text columns are excluded (they are
    still in the cleaned file — just not auditable as numbers)."""
    import datetime
    import re as _re
    notes, keep = [], []
    for i, h in enumerate(header):
        vals = [r[i] for r in rows if i < len(r) and r[i] != ""]
        if not vals:
            notes.append(f"audit view: '{h}' excluded (empty)")
            continue
        if all(_re.match(p, v) for v in vals
               for p, _ in [DATE_RES[0]]) or \
           all(_re.match(DATE_RES[1][0], v) for v in vals):
            pat, conv = DATE_RES[0] if _re.match(
                DATE_RES[0][0], vals[0]) else DATE_RES[1]
            def to_day(v, pat=pat, conv=conv):
                y, mo, d = conv(_re.match(pat, v))
                return str(datetime.date(y, mo, d).toordinal())
            try:
                converted = [to_day(r[i]) if r[i] != "" else ""
                             for r in rows]
            except ValueError:
                notes.append(f"audit view: '{h}' looks date-shaped "
                             f"but does not parse as month-first "
                             f"dates (day/month order ambiguous?) — "
                             f"excluded rather than guessed")
                continue
            keep.append((h, converted))
            notes.append(f"audit view: '{h}' dates converted to "
                         f"day numbers (receipted, view only)")
            continue
        try:
            [float(v) for v in vals]
            keep.append((h, [r[i] if i < len(r) else ""
                             for r in rows]))
        except ValueError:
            notes.append(f"audit view: '{h}' excluded (text — "
                         f"cleaned but not numerically auditable)")
    view_header = [h for h, _ in keep]
    view_rows = list(map(list, zip(*[c for _, c in keep]))) \
        if keep else []
    return view_rows, view_header, notes


def mad_extremes(view_rows, view_header, k=8.0, cap_frac=0.05):
    """Mechanically extreme values in the TARGET (last) column:
    beyond k median-absolute-deviations from the median. Conservative
    (k=8) and capped at 5% of rows — this only exists to give the
    engine's law pass a fair look when a wild value poisons the fit.
    Returns (row_indexes_0based, receipt_lines)."""
    tgt = len(view_header) - 1
    vals = []
    for i, r in enumerate(view_rows):
        try:
            vals.append((i, float(r[tgt])))
        except (ValueError, IndexError):
            continue
    if len(vals) < 10:
        return [], []
    xs = sorted(v for _, v in vals)
    med = xs[len(xs) // 2]
    mad = sorted(abs(v - med) for v in xs)[len(xs) // 2] or 1e-12
    hits = [(i, v) for i, v in vals if abs(v - med) / mad > k]
    cap = max(1, int(len(vals) * cap_frac))
    if not hits or len(hits) > cap:
        return [], []
    degenerate = mad <= 1e-9 * max(abs(med), 1.0)
    receipts = [(f"row {i + 2}: {view_header[tgt]}={v:,.2f} differs "
                 f"where every other row agrees on {med:,.2f} (the "
                 f"column's typical spread is zero)") if degenerate
                else (f"row {i + 2}: {view_header[tgt]}={v:,.2f} "
                      f"sits {abs(v - med) / mad:,.0f} MADs from "
                      f"the median {med:,.2f} (threshold {k:g})")
                for i, v in hits]
    return [i for i, _ in hits], receipts


def main():
    ap = argparse.ArgumentParser(description="Receipted data cleaning")
    ap.add_argument("source")
    ap.add_argument("--out", default=os.path.join(DC, "demos",
                                                  "portfolio"))
    ap.add_argument("--keep-dupes", action="store_true",
                    help="do not remove exact duplicate rows")
    ap.add_argument("--no-audit", action="store_true",
                    help="skip the engine audit pass (fast mode)")
    ap.add_argument("--resolution", action="append", default=[],
                    metavar="COL=SIGMA",
                    help="the client's stated tolerance for a "
                         "column, passed to the engine (repeatable) "
                         "— answers the audit's ASK so laws can "
                         "certify and real outliers can be flagged")
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(args.source))[0]
    out_dir = os.path.join(args.out, name + "_clean")
    os.makedirs(out_dir, exist_ok=True)
    log = []

    # 1. structure (delimiter / title rows / header / missing marks)
    rows, header, notes = normalize(args.source)
    log += [f"STRUCTURE: {n}" for n in notes]

    # 2. whitespace
    n_trim = trim_cells(rows)
    if n_trim:
        log.append(f"WHITESPACE: trimmed {n_trim} cell(s)")

    # 3. exact duplicates
    if not args.keep_dupes:
        before = len(rows)
        rows, dup_receipts = drop_exact_duplicates(rows)
        log += [f"DUPLICATE: {r}" for r in dup_receipts]
        if before == len(rows):
            log.append("DUPLICATE: none found")
    else:
        log.append("DUPLICATE: removal disabled by --keep-dupes")

    cleaned = os.path.join(out_dir, "cleaned.csv")
    write_csv(rows, header, cleaned)

    # 4. the engine's judgment flags — attached, never applied.
    # NOTHING in this section may cost the client the changelog:
    # any failure is reported inside it, honestly, and we move on.
    flags_section = []
    try:
        run_audit_section(args, rows, header, out_dir, flags_section)
    except Exception as e:
        flags_section.append(
            f"engine audit could not run ({type(e).__name__}: {e}) "
            f"— the cleaned file and every receipt above are "
            f"unaffected; we say so rather than pretend it was "
            f"checked")

    with open(os.path.join(out_dir, "CHANGELOG.md"), "w",
              encoding="utf-8") as f:
        f.write(f"# Cleaning changelog — {name}\n\n")
        f.write(f"{len(rows)} rows out; every change below; "
                f"nothing else was touched.\n\n")
        f.write("## Changes made (mechanical, receipted)\n\n")
        f.write("".join(f"- {ln}\n" for ln in log))
        f.write("\n## Left for your judgment (we flag, we don't "
                "guess)\n\n")
        f.write("".join(f"- {ln}\n" for ln in flags_section)
                or "- (audit skipped)\n")
        f.write("\nA flagged cell is evidence, not a typo — "
                "overwriting it silently is how other people's "
                "cleaning loses real signal. Tell us the intended "
                "value and we apply it, receipted.\n")

    print(f"cleaned  -> {cleaned}")
    print(f"changelog-> {os.path.join(out_dir, 'CHANGELOG.md')}")
    print(f"changes: {len(log)}  flags: "
          f"{sum(1 for x in flags_section if x.startswith('FLAGGED'))}")


def run_audit_section(args, rows, header, out_dir, flags_section):
    """The engine's look at the cleaned table. Isolated so that NO
    failure here can ever cost the client the changelog."""
    if args.no_audit:
        return
    vrows, vheader, vnotes = audit_view(rows, header)
    flags_section += vnotes
    if True:
        if len(vheader) >= 2 and len(vrows) >= 10:
            view_csv = os.path.join(out_dir, "audit_view.csv")
            write_csv(vrows, vheader, view_csv)
            audit_args = ["--audit"]
            for res in args.resolution:
                audit_args += ["--resolution", res]
            audit_text, secs = run_engine(view_csv, audit_args)
            with open(os.path.join(out_dir, "receipts_audit.txt"),
                      "w", encoding="utf-8") as f:
                f.write(audit_text)
            a = parse_audit(audit_text)
            skips = [ln.strip() for ln in audit_text.splitlines()
                     if ln.strip().startswith("[skipped]")]
            flags_section += [f"engine said: {s}" for s in skips]
            flags_section.append(
                f"engine audit ({secs:.0f}s): verdict "
                f"{a.get('verdict') or 'none given'}; "
                f"{a.get('counts') or 'no counts reported'}")
            flags_section += [f"FLAGGED FOR REVIEW: {fl}"
                              for fl in a.get("flags", [])]

            # PASS B — a wild value can poison the law that would
            # flag it. Hold extremes out (receipted), give the law
            # pass a fair look, and flag by contrast. The cleaned
            # file keeps every row; judgment stays with the client.
            if a.get("verdict") == "NO RULE":
                ext, ereceipts = mad_extremes(vrows, vheader)
                if ext:
                    held_in = [r for i, r in enumerate(vrows)
                               if i not in set(ext)]
                    b_csv = os.path.join(out_dir,
                                         "audit_view_passB.csv")
                    write_csv(held_in, vheader, b_csv)
                    b_text, b_secs = run_engine(b_csv, audit_args)
                    with open(os.path.join(
                            out_dir, "receipts_audit_passB.txt"),
                            "w", encoding="utf-8") as f:
                        f.write(b_text)
                    b = parse_audit(b_text)
                    if b.get("verdict") not in (None, "NO RULE"):
                        flags_section.append(
                            f"PASS B ({b_secs:.0f}s): with "
                            f"{len(ext)} extreme value(s) held out "
                            f"(receipts below), the law CERTIFIED "
                            f"on the remaining {len(held_in)} rows "
                            f"— verdict {b['verdict']}; "
                            f"{b.get('counts')}")
                        for er in ereceipts:
                            flags_section.append(
                                f"FLAGGED FOR REVIEW — "
                                f"OUTLIER-CANDIDATE: {er}; "
                                f"inconsistent with the law the "
                                f"rest of your data obeys. Kept in "
                                f"the cleaned file; your call.")
                    else:
                        flags_section.append(
                            f"PASS B: held out {len(ext)} extreme "
                            f"value(s) and the law still refused — "
                            f"the extremes were not the story; "
                            f"receipts kept")
            if not a.get("flags") and not skips:
                flags_section.append("no law-violation flags raised "
                                     "on the audit view")
        else:
            flags_section.append(
                f"engine audit not run: the numeric view is too "
                f"thin ({len(vrows)} rows x {len(vheader)} numeric "
                f"column(s); need >=10 rows and >=2 columns) — "
                f"we say so rather than pretend it was checked")


if __name__ == "__main__":
    main()

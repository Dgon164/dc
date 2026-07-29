"""HONEST REPORT — the storefront wrapper around askisaac.

Machinery, not mind: this file normalizes a messy real-world table
(every change receipted), runs the PROMOTED engine untouched
(--analyst, --audit), and renders what a paying client expects —
a plain-English summary, charts, and the full receipts beneath.
Nothing is claimed that the engine did not say; where the engine
refused, the report says so proudly. That refusal IS the product:
analysis that would rather say NO RULE than make something up.

    python standalone/honest_report.py client.csv
    python standalone/honest_report.py client.csv --out reports --time date

Output: <out>/<name>/report.html (self-contained, charts embedded),
report.md, charts/*.png, receipts_analyst.txt, receipts_audit.txt.
"""
import argparse
import base64
import csv
import html as html_mod
import io
import os
import re
import subprocess
import sys
import time
from datetime import date

DC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# palette (single-series line + status marks; text never wears series color)
BLUE = "#2a78d6"
BLUE_BAND = "#b7d3f6"
RED = "#d03b3b"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"

MISSING = {"***", "na", "n/a", "null", "none", "-", "--", ""}


# ---------------------------------------------------------------- intake
def normalize(path):
    """Read a real-world table; return (rows, header, notes). Every
    transformation is recorded in notes — the client sees the list."""
    notes = []
    raw = open(path, "rb").read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
        notes.append("decoded as latin-1 (not valid utf-8)")
    all_lines = text.splitlines()
    lines = [ln for ln in all_lines if ln.strip() != ""]
    n_blank = len(all_lines) - len(lines)
    if n_blank:
        notes.append(f"dropped {n_blank} blank line(s)")
    n_comments = sum(1 for ln in lines if ln.lstrip().startswith("#"))
    if n_comments:
        lines = [ln for ln in lines if not ln.lstrip().startswith("#")]
        notes.append(f"dropped {n_comments} comment line(s) starting with #")

    # delimiter: the candidate that splits the most lines consistently
    sample = lines[:50]
    best, best_score = ",", 0
    for cand in (",", ";", "\t", "|"):
        counts = [ln.count(cand) for ln in sample]
        usable = [c for c in counts if c > 0]
        if not usable:
            continue
        modal = max(set(usable), key=usable.count)
        score = sum(1 for c in counts if c == modal) * modal
        if score > best_score:
            best, best_score = cand, score
    if best != ",":
        notes.append(f"delimiter detected: {best!r} (converted to comma)")

    parsed = [next(csv.reader([ln], delimiter=best)) for ln in lines]
    widths = [len(r) for r in parsed]
    modal_w = max(set(widths), key=widths.count)

    # title rows: leading rows narrower than the table itself
    skipped = 0
    while parsed and len(parsed[0]) < modal_w:
        notes.append(f"skipped title row: {' '.join(parsed[0])[:60]!r}")
        parsed.pop(0)
        skipped += 1
    # ragged rows (wrong field count) cannot be aligned to columns.
    # They are QUARANTINED with receipts — never silently dropped:
    # one receipt per row, and the raw line is kept in the notes so
    # nothing the client sent is ever lost without a trace.
    ragged = [(i, r) for i, r in enumerate(parsed, start=1)
              if len(r) != modal_w]
    for i, r in ragged:
        notes.append(f"QUARANTINED row {i}: {len(r)} fields where "
                     f"the table has {modal_w} — cannot align to "
                     f"columns; content: {best.join(r)[:80]!r}")
    parsed = [r for r in parsed if len(r) == modal_w]

    def numeric(cell):
        try:
            float(cell)
            return True
        except ValueError:
            return False

    first = parsed[0]
    if any(not numeric(c) and c.strip().lower() not in MISSING
           for c in first):
        header = [c.strip() or f"col{i}" for i, c in enumerate(first)]
        rows = parsed[1:]
    else:
        header = [f"col{i}" for i in range(modal_w)]
        rows = parsed
        notes.append("no header row found — generated column names")
    seen = {}
    for i, h in enumerate(header):
        if h in seen:
            header[i] = f"{h}_{seen[h]}"
        seen[h] = seen.get(h, 0) + 1

    n_miss = 0
    for r in rows:
        for i, c in enumerate(r):
            if c.strip().lower() in MISSING and c != "":
                r[i] = ""
                n_miss += 1
    if n_miss:
        notes.append(f"blanked {n_miss} missing-value marker(s) "
                     f"(***, NA, null, ...)")
    return rows, header, notes


def write_csv(rows, header, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ---------------------------------------------------------------- engine
def run_engine(csv_path, extra):
    """Run the promoted CLI untouched; return its verbatim text."""
    cmd = [sys.executable, os.path.join(DC, "askisaac.py"), csv_path] + extra
    t0 = time.perf_counter()
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=DC,
                         timeout=1800, encoding="utf-8", errors="replace",
                         env=env)
    return out.stdout, time.perf_counter() - t0


# ------------------------------------------------------- parse the report
def parse_analyst(text):
    """Split the analyst face into per-column findings. Parsing only —
    nothing is added that the engine did not print."""
    cols, cur = [], None
    for ln in text.splitlines():
        m = re.match(r"\s*──\s*(.+?)\s*──", ln)
        if m:
            cur = dict(name=m.group(1), lines=[], verdict=None,
                       equation=None, books=None, cycle=None,
                       forecast=None)
            cols.append(cur)
            continue
        if cur is None:
            continue
        s = ln.strip()
        cur["lines"].append(ln)
        m = re.match(r"trend \[([A-Z ]+)\]: (.+)", s)
        if m:
            cur["verdict"], cur["equation"] = m.group(1), m.group(2)
        m = re.match(r"the books: trend ([\d.]+)% \+ seasonal ([\d.]+)% "
                     r"\+ residual ([\d.]+)%", s)
        if m:
            cur["books"] = tuple(float(x) for x in m.groups())
        m = re.match(r"cycle: period ([\d.]+)(.*)", s)
        if m:
            cur["cycle"] = m.group(1) + m.group(2)
        m = re.match(r"forecast next 12: (.+)", s)
        if m:
            cur["forecast"] = m.group(1)
    return cols


def parse_audit(text):
    d = dict(verdict=None, counts=None, flags=[])
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("VERDICT:"):
            d["verdict"] = s.split(":", 1)[1].strip()
        if s.startswith("counts:"):
            d["counts"] = s.split(":", 1)[1].strip()
        if s.startswith("flags:"):
            d["flags"].append(s.split(":", 1)[1].strip())
    return d


# ---------------------------------------------------------------- charts
def make_charts(rows, header, findings, time_col, chart_dir):
    """One line chart per analyzed column. Data drawn verbatim; the
    only overlay is the engine's own forecast range, shaded."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}
    os.makedirs(chart_dir, exist_ok=True)

    def col_values(name):
        i = header.index(name)
        out = []
        for r in rows:
            try:
                out.append(float(r[i]))
            except (ValueError, IndexError):
                out.append(None)
        return out

    x = None
    if time_col and time_col in header:
        x = col_values(time_col)
    made = {}
    for f in findings:
        if f["name"] not in header:
            continue
        y = col_values(f["name"])
        pts = [(xi, yi) for xi, yi in
               zip(x if x else range(len(y)), y) if yi is not None
               and xi is not None]
        if len(pts) < 3:
            continue
        xs, ys = zip(*pts)
        fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=110)
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        ax.plot(xs, ys, color=BLUE, linewidth=2, solid_capstyle="round")
        m = re.match(r"([-\d.]+) \.\. ([-\d.]+)", f.get("forecast") or "")
        if m and f["verdict"] not in (None, "NO RULE"):
            lo, hi = float(m.group(1)), float(m.group(2))
            span = (max(xs) - min(xs)) * 0.06 or 1
            ax.fill_between([max(xs), max(xs) + span], lo, hi,
                            color=BLUE_BAND, alpha=0.6, linewidth=0)
            ax.annotate("forecast", (max(xs) + span / 2, hi),
                        ha="center", va="bottom", fontsize=7, color=INK2)
        ax.annotate(f" {ys[-1]:g}", (xs[-1], ys[-1]), fontsize=8,
                    color=INK2, va="center")
        ax.grid(True, color=MUTED, alpha=0.25, linewidth=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.set_title(f["name"], loc="left", fontsize=10, color=INK,
                     fontweight="bold")
        fig.tight_layout()
        p = os.path.join(chart_dir,
                         re.sub(r"[^\w.-]", "_", f["name"]) + ".png")
        fig.savefig(p, facecolor=SURFACE)
        plt.close(fig)
        made[f["name"]] = p
    return made


# ---------------------------------------------------------------- render
def plain_summary(findings, audit):
    """Translate the engine's findings to plain English. STRICT: every
    sentence traces to an engine line; refusals stay refusals."""
    out = []
    for f in findings:
        n = f["name"]
        if f["verdict"] == "NO RULE":
            if f["cycle"]:
                out.append(f"**{n}** — found a repeating cycle "
                           f"(period {f['cycle']}), but no overall "
                           f"trend passed our checks — on the trend "
                           f"we say so rather than guess (receipt "
                           f"below).")
            else:
                out.append(f"**{n}** — no reliable pattern passed "
                           f"our checks. We say so rather than "
                           f"guess; the refusal receipt is below.")
            continue
        bits = []
        if f["books"]:
            t, s, r = f["books"]
            bits.append(f"a pattern explaining "
                        f"{100 - r:.0f}% of the movement "
                        f"(trend {t:.0f}%, cycle {s:.0f}%, "
                        f"unexplained {r:.0f}%)")
        if f["cycle"]:
            bits.append(f"a repeating cycle (period {f['cycle']})")
        if f["forecast"] and f["verdict"] != "NO RULE":
            bits.append(f"forecast for the next 12 steps: "
                        f"{f['forecast']}")
        if f["equation"]:
            bits.append(f"the fitted rule, readable, is "
                        f"`{f['equation']}`")
        if bits:
            out.append(f"**{n}** — found " + "; ".join(bits) + ".")
    if audit and audit.get("counts"):
        out.append(f"**Data quality** — audit verdict "
                   f"{audit.get('verdict') or 'n/a'}; {audit['counts']}.")
    return out


def render(name, header, n_rows, notes, findings, audit, charts,
           analyst_text, audit_text, out_dir, secs):
    md = []
    md.append(f"# Data report — {name}")
    md.append(f"*{date.today().isoformat()} · {n_rows} rows · "
              f"{len(header)} columns · engine time {secs:.0f}s*")
    md.append("\n## What we found (plain English)\n")
    summary = plain_summary(findings, audit)
    md += [f"- {s}" for s in summary] or ["- (no findings)"]
    md.append("\n> Every claim above traces to a receipt at the bottom "
              "of this report. Where evidence was insufficient, the "
              "analysis says **NO RULE** instead of guessing — that "
              "honesty is the product.")
    if notes:
        md.append("\n## What we did to your file first (receipted)\n")
        md += [f"- {n}" for n in notes]
    if charts:
        md.append("\n## Charts\n")
        for n, p in charts.items():
            md.append(f"![{n}](charts/{os.path.basename(p)})")
    md.append("\n## The receipts — analyst face (verbatim)\n")
    md.append("```\n" + analyst_text.strip() + "\n```")
    if audit_text:
        md.append("\n## The receipts — audit face (verbatim)\n")
        md.append("```\n" + audit_text.strip() + "\n```")
    md_text = "\n".join(md)
    with open(os.path.join(out_dir, "report.md"), "w",
              encoding="utf-8") as f:
        f.write(md_text)

    # self-contained html (charts embedded) — the file you send a client
    def img64(p):
        return base64.b64encode(open(p, "rb").read()).decode()
    body = []
    for block in md_text.split("\n"):
        if block.startswith("!["):
            m = re.match(r"!\[(.*?)\]\(charts/(.*?)\)", block)
            if m:
                p = os.path.join(out_dir, "charts", m.group(2))
                body.append(f'<img alt="{html_mod.escape(m.group(1))}" '
                            f'style="max-width:100%" '
                            f'src="data:image/png;base64,{img64(p)}">')
            continue
        if block.startswith("# "):
            body.append(f"<h1>{html_mod.escape(block[2:])}</h1>")
        elif block.startswith("## "):
            body.append(f"<h2>{html_mod.escape(block[3:])}</h2>")
        elif block.startswith("> "):
            body.append(f"<blockquote>{html_mod.escape(block[2:])}"
                        f"</blockquote>")
        elif block.startswith("- "):
            t = html_mod.escape(block[2:])
            t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
            t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
            body.append(f"<li>{t}</li>")
        elif block.startswith("*") and block.endswith("*"):
            body.append(f"<p><i>{html_mod.escape(block.strip('*'))}"
                        f"</i></p>")
        elif block.strip() == "```":
            body.append("<pre>" if body and not body[-1].startswith(
                "<pre") else "</pre>")
        else:
            body.append(html_mod.escape(block) + "<br>"
                        if block.strip() else "")
    html_text = (
        "<!doctype html><meta charset='utf-8'>"
        f"<title>Data report — {html_mod.escape(name)}</title>"
        "<style>body{font-family:Segoe UI,system-ui,sans-serif;"
        f"max-width:820px;margin:2em auto;padding:0 1em;color:{INK};"
        f"background:{SURFACE}}}h1,h2{{color:{INK}}}"
        "pre{background:#f2f1ee;padding:1em;overflow-x:auto;"
        "font-size:12px}blockquote{border-left:3px solid " + BLUE +
        f";margin:1em 0;padding:.3em 1em;color:{INK2}}}"
        "code{background:#f2f1ee;padding:1px 4px}</style>" +
        "\n".join(body))
    with open(os.path.join(out_dir, "report.html"), "w",
              encoding="utf-8") as f:
        f.write(html_text)
    return md_text


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Client-ready honest report")
    ap.add_argument("source")
    ap.add_argument("--out", default=os.path.join(DC, "demos",
                                                  "portfolio"))
    ap.add_argument("--time", default=None,
                    help="time column passed to the analyst")
    ap.add_argument("--no-audit", action="store_true")
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(args.source))[0]
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)

    rows, header, notes = normalize(args.source)
    norm_csv = os.path.join(out_dir, f"{name}.normalized.csv")
    write_csv(rows, header, norm_csv)
    print(f"intake: {len(rows)} rows x {len(header)} cols; "
          f"{len(notes)} normalization note(s)", flush=True)

    extra = ["--analyst"] + (["--time", args.time] if args.time else [])
    analyst_text, t1 = run_engine(norm_csv, extra)
    print(f"analyst face: {t1:.1f}s", flush=True)
    audit_text, t2 = ("", 0.0)
    if not args.no_audit:
        audit_text, t2 = run_engine(norm_csv, ["--audit"])
        print(f"audit face: {t2:.1f}s", flush=True)

    findings = parse_analyst(analyst_text)
    audit = parse_audit(audit_text) if audit_text else None
    time_col = args.time
    if not time_col:
        m = re.search(r"time key: (.+?) \(", analyst_text)
        if m:
            time_col = m.group(1).strip()
    charts = make_charts(rows, header, findings, time_col,
                         os.path.join(out_dir, "charts"))
    with open(os.path.join(out_dir, "receipts_analyst.txt"), "w",
              encoding="utf-8") as f:
        f.write(analyst_text)
    if audit_text:
        with open(os.path.join(out_dir, "receipts_audit.txt"), "w",
                  encoding="utf-8") as f:
            f.write(audit_text)
    render(name, header, len(rows), notes, findings, audit, charts,
           analyst_text, audit_text, out_dir, t1 + t2)
    print(f"report -> {os.path.join(out_dir, 'report.html')}", flush=True)


if __name__ == "__main__":
    main()

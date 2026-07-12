"""promote.py — the ONLY way research becomes live.

Copies the blessed state of the research repo (isaac-paperclip) into
this live repo (dc), then runs EVERY gate: the engine suite and all
four store batteries, from dc, against dc's own copies. If anything
fails, the promotion is refused (git checkout restores the last good
live state). On success, PROVENANCE.md records exactly which research
commit went live, when, and with what battery results.

Research repo: ISAAC_RESEARCH env var, else the sibling checkout.
dc-only files are never overwritten: README.md, promote.py,
PROVENANCE.md, .gitignore. Never hand-edit promoted files in dc —
edit in research and promote.

Run:  python promote.py
"""
import os
import shutil
import subprocess
import sys
import time

DC = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.environ.get("ISAAC_RESEARCH") or os.path.join(
    os.path.dirname(DC), "isaac-paperclip")
PRESERVE = {"README.md", "promote.py", "PROVENANCE.md", ".gitignore"}
EXCLUDE = {".git", "__pycache__", "data"}


def copy_tree():
    copied = []
    for name in sorted(os.listdir(RESEARCH)):
        if name in EXCLUDE or name in PRESERVE:
            continue
        src = os.path.join(RESEARCH, name)
        dst = os.path.join(DC, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, dst)
        copied.append(name)
    return copied


def run(cmd, cwd, expect_in_output=None):
    env = dict(os.environ, PYTHONUTF8="1")
    r = subprocess.run([sys.executable] + cmd, cwd=cwd, env=env,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=1200)
    out = (r.stdout or "") + (r.stderr or "")
    ok = r.returncode == 0 and (expect_in_output is None or expect_in_output in out)
    tail = [ln for ln in out.strip().splitlines() if ln.strip()][-1:]
    return ok, (tail[0].strip() if tail else "(no output)")


def research_commit():
    def git(*args):
        return subprocess.run(["git", "-C", RESEARCH] + list(args),
                              capture_output=True, text=True).stdout.strip()
    head = git("rev-parse", "--short", "HEAD")
    dirty = bool(git("status", "--porcelain"))
    return head, dirty


def main():
    if not os.path.exists(os.path.join(RESEARCH, "isaac.py")):
        print(f"  REFUSED: research repo not found at {RESEARCH}")
        sys.exit(1)
    head, dirty = research_commit()
    print(f"  promoting research @ {head}{' (DIRTY working tree)' if dirty else ''}")
    copied = copy_tree()
    print(f"  copied: {', '.join(copied)}")

    gates = [
        ("engine suite", ["isaac.py"], DC, "11/11 capability areas passed"),
        ("store battery", ["test_store.py"], os.path.join(DC, "isaac_store"), None),
        ("cache battery", ["test_cache.py"], os.path.join(DC, "isaac_store"), None),
        ("leaf battery", ["test_leaf.py"], os.path.join(DC, "isaac_store"), None),
        ("server battery", ["test_server.py"], os.path.join(DC, "isaac_store"), None),
    ]
    results, failed = [], False
    for name, cmd, cwd, expect in gates:
        ok, tail = run(cmd, cwd, expect)
        results.append((name, ok, tail))
        print(f"  [{'ok ' if ok else 'FAIL'}] {name} — {tail}")
        failed = failed or not ok

    if failed:
        print("\n  PROMOTION REFUSED — live state not stamped."
              "\n  Restore last good live state with: git checkout -- . ")
        sys.exit(1)

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(DC, "PROVENANCE.md"), "w", encoding="utf-8") as f:
        f.write("# PROVENANCE — what is live, and where it came from\n\n")
        f.write(f"- promoted: {stamp}\n")
        f.write(f"- research commit: isaac-paperclip @ {head}"
                f"{' (working tree was DIRTY)' if dirty else ''}\n")
        f.write(f"- copied: {', '.join(copied)}\n- gates:\n")
        for name, ok, tail in results:
            f.write(f"  - {name}: PASS — {tail}\n")
    print(f"\n  PROMOTED. PROVENANCE.md stamped ({stamp}, research @ {head}).")
    print("  Commit and push dc to make it official.")


if __name__ == "__main__":
    main()

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
    if not ok:
        # a failing gate's FULL output is evidence, never a tail line
        # (the get_doors flake struck twice with its traceback lost —
        # 2026-07-17; unexplained = unfinished, so capture always)
        fdir = os.path.join(DC, "experiments", "runs", "gate_failures")
        os.makedirs(fdir, exist_ok=True)
        fname = os.path.join(
            fdir, time.strftime("%Y%m%d_%H%M%S_")
            + os.path.basename(cmd[0]).replace(".py", "") + ".txt")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(out)
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
        ("scrubber battery", ["test_scrubber.py"], os.path.join(DC, "isaac_store"), None),
        ("learning battery", ["test_learning.py"], os.path.join(DC, "isaac_store"), None),
        ("skills battery", ["test_skills.py"], os.path.join(DC, "isaac_store"), None),
        ("ops battery", ["test_ops.py"], os.path.join(DC, "isaac_store"), None),
        ("multicore battery", ["test_multicore.py"], os.path.join(DC, "isaac_store"), None),
        ("told-knowledge battery", ["told_knowledge_battery.py"],
         os.path.join(DC, "experiments"), None),
        ("gravity-family battery", ["gravity_family_battery.py"],
         os.path.join(DC, "experiments"), None),
        ("provenance battery", ["test_provenance.py"],
         os.path.join(DC, "isaac_store"), None),
        ("seeker battery", ["test_seeker.py"],
         os.path.join(DC, "isaac_store"), None),
        ("retention battery", ["test_retention.py"],
         os.path.join(DC, "isaac_store"), None),
        ("report battery", ["test_report.py"],
         os.path.join(DC, "isaac_store"), None),
        ("speech battery", ["test_speech.py"],
         os.path.join(DC, "isaac_store"), None),
        ("learned-speech battery", ["test_speech_learned.py"],
         os.path.join(DC, "isaac_store"), None),
        ("face battery", ["test_face.py"],
         os.path.join(DC, "isaac_store"), None),
        ("intake battery", ["test_intake.py"],
         os.path.join(DC, "isaac_store"), None),
        ("structural battery", ["test_structural.py"],
         os.path.join(DC, "isaac_store"), None),
        ("remote battery", ["test_remote.py"],
         os.path.join(DC, "isaac_store"), None),
        ("triangulation battery", ["test_triangulate.py"],
         os.path.join(DC, "isaac_store"), None),
        ("education battery", ["test_education.py"],
         os.path.join(DC, "isaac_store"), None),
        ("crawlers battery", ["test_crawlers.py"],
         os.path.join(DC, "isaac_store"), None),
        ("lag battery", ["lag_battery.py"],
         os.path.join(DC, "experiments"), None),
        ("task battery", ["task_battery.py"],
         os.path.join(DC, "experiments"), None),
        ("rule battery", ["rule_battery.py"],
         os.path.join(DC, "experiments"), None),
        ("rounding battery", ["rounding_battery.py"],
         os.path.join(DC, "experiments"), None),
        ("floor battery", ["floor_battery.py"],
         os.path.join(DC, "experiments"), None),
        ("commonality battery", ["commonality_battery.py"],
         os.path.join(DC, "experiments"), None),
        ("tools battery", ["tools_battery.py"],
         os.path.join(DC, "experiments"), None),
        ("order battery", ["order_battery.py"],
         os.path.join(DC, "experiments"), None),
        ("order-matters battery", ["order_matters_battery.py"],
         os.path.join(DC, "experiments"), None),
        ("cross-check battery", ["crosscheck_battery.py"],
         os.path.join(DC, "experiments"), None),
        ("vet-intake battery", ["vet_intake_battery.py"],
         os.path.join(DC, "experiments"), None),
        ("audit battery", ["test_audit.py"],
         os.path.join(DC, "isaac_store"), None),
        ("sparring battery", ["test_sparring.py"],
         os.path.join(DC, "isaac_store"), None),
        # the night of 2026-07-15's promotion batteries (#11-#16) and
        # the 100-world verdict duel joined the permanent wall
        # 2026-07-17 (session 1 of the alpha push): the wall that
        # caught the rounding red must run on every promotion.
        ("mechanisms battery", ["mechanisms_promotion_battery.py"],
         os.path.join(DC, "experiments"), None),
        ("outlier-source battery", ["outlier_source_battery.py"],
         os.path.join(DC, "experiments"), None),
        ("containers battery", ["containers_promotion_battery.py"],
         os.path.join(DC, "experiments"), None),
        # per-dataset containers + cross-domain crystallization
        # (owner, 2026-07-17, the feeding design — promotion #17)
        ("container-crystal battery", ["container_crystal_battery.py"],
         os.path.join(DC, "experiments"), None),
        # the growing grammar's harder road (owner, 2026-07-18 —
        # promotion #19: novel coordinates, court unchanged)
        ("harder-road battery", ["harder_road_battery.py"],
         os.path.join(DC, "experiments"), None),
        # the analyst's trend road (owner, 2026-07-18 — promotion
        # #20: decomposition with receipts, sunspots answered)
        ("analyst-trend battery", ["analyst_trend_battery.py"],
         os.path.join(DC, "experiments"), None),
        # A2 cross-container correlation (owner, 2026-07-18 —
        # promotion #21: the alignment null, spurious trap refused)
        ("analyst-relate battery", ["analyst_relate_battery.py"],
         os.path.join(DC, "experiments"), None),
        # A3 the watch (owner, 2026-07-18 — promotion #22: laws
        # defend their seats; drift duels on arrival)
        ("analyst-watch battery", ["analyst_watch_battery.py"],
         os.path.join(DC, "experiments"), None),
        # A4 the analyst face (owner, 2026-07-18 — promotion #23:
        # the segment closes; the face assembles, never judges)
        ("analyst-face battery", ["analyst_face_battery.py"],
         os.path.join(DC, "experiments"), None),
        ("flow battery", ["flow_promotion_battery.py"],
         os.path.join(DC, "experiments"), None),
        ("flow-election battery", ["flow_election_battery.py"],
         os.path.join(DC, "experiments"), None),
        ("verdict duel gate", ["verdict_port_gate.py", "ported"],
         os.path.join(DC, "experiments"), None),
        # the birth machinery (2026-07-17, road-trials amendment):
        # the first teacher, the instrument panel, the artifact claim
        ("inherit battery", ["test_inherit.py"],
         os.path.join(DC, "isaac_store"), None),
        ("road-trial battery", ["test_roadtrial.py"],
         os.path.join(DC, "isaac_store"), None),
        ("reproducibility check", ["reproducibility_check.py"],
         os.path.join(DC, "experiments"), None),
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

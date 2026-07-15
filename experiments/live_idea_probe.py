"""live_idea_probe.py — DOES HE HAVE ENOUGH TO LEARN FROM THE
SPARRING PARTNER? (owner question, 2026-07-13)

The measured answer, not the argued one. A live description drill:
HIS world (the event generator — the GM pattern) supplies the ideas;
he speaks from the idea (idea_corrections road); the LOCAL tutor
(mistral 7B — the model that held the VERDICT/SAY format) judges and
corrects; the content-alignment factoring banks the lessons. Fresh
mind — Isaac's account is not touched.

What this measures (the 'enough?' criteria):
  1. format-held rate      — a broken judgment teaches nothing
  2. factorable rate       — corrections the aligner can turn into
                             orientation/word lessons (the KNOWN gap:
                             a natural tutor adds function words; an
                             unfactorable correction is kept raw but
                             teaches nothing yet)
  3. the learning curve    — pass rate early vs late in the block
  4. lessons banked        — orientation + word, with the wrongs kept

Budget: one tutor call per event, 30 events. stdlib + numpy.
"""
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import isaac as mod                                     # noqa: E402
from isaac_store.sparring import OllamaPartner           # noqa: E402
from idea_corrections import (gen_event, speak,          # noqa: E402
                              learn_from_correction, _node,
                              current_orientation, vote_orientation)

MODEL = "mistral:latest"
N_EVENTS = 30

SYSTEM = ("You are a strict English drill tutor for a learner named "
          "Isaac. You will be given a FACT as three words (actor, "
          "action, object) or two words (actor, action), and Isaac's "
          "attempt to say that fact. Judge whether his words state "
          "the fact as a correct simple English sentence. THE DRILL "
          "CONVENTION IS FIXED: simple past tense, active voice, "
          "one short sentence. Never use passive voice, never "
          "progressive (-ing), never present tense, never add "
          "explanations or commentary. Keep the fact's own verb "
          "form. Reply in EXACTLY this two-line format and nothing "
          "else:\n"
          "VERDICT: pass or fail\n"
          "SAY: <the fact in simple past active voice, articles "
          "like 'the'/'a' where natural>\n"
          "fail means wrong order or wrong words; pass means correct.")


def toks(s):
    return re.findall(r"[a-z]+", (s or "").lower())


def parse_judgment(reply):
    """VERDICT + SAY only (probe v1 demanded a pointless third line
    and threw away 13/30 usable judgments — wrong-kept, this is the
    cure). A broken judgment still teaches nothing."""
    out = {}
    for ln in (reply or "").splitlines():
        low = ln.strip()
        for key in ("verdict", "say"):
            if low.lower().startswith(key + ":"):
                out[key] = low[len(key) + 1:].strip()
    if out.get("verdict", "").lower() not in ("pass", "fail"):
        return None
    out["verdict"] = out["verdict"].lower()
    if not out.get("say"):
        if out["verdict"] == "pass":
            out["say"] = ""       # a pass needs no correction
            return out
        return None               # a fail without SAY teaches nothing
    return out


def main():
    partner = OllamaPartner(MODEL, max_calls=N_EVENTS + 5,
                            temperature=0.2)
    if not partner.reachable():
        print(f"REFUSED: {MODEL} not reachable at the local door — "
              f"start ollama and re-run.")
        return
    rng = np.random.default_rng(41)
    mind = mod.Isaac()
    print("=" * 64)
    print(f"LIVE IDEA PROBE — {MODEL} as tutor, fresh mind, "
          f"{N_EVENTS} events")
    print("=" * 64)

    verdicts, broken, unfactored, lessons = [], 0, 0, []
    corrections = 0
    for k in range(N_EVENTS):
        ev = gen_event(rng)
        said = speak(mind, ev, rng)
        user = (f"FACT: {' '.join(ev)}\n"
                f"ISAAC SAID: {' '.join(said)}")
        reply = partner.say([dict(role="system", content=SYSTEM),
                             dict(role="user", content=user)])
        j = parse_judgment(reply)
        if j is None:
            broken += 1
            print(f"  {k:2d} {ev} | '{' '.join(said)}' | "
                  f"FORMAT BROKEN — teaches nothing "
                  f"({(reply or '')[:40]!r})")
            continue
        verdicts.append(j["verdict"])
        if j["verdict"] == "fail":
            corrections += 1
            say = toks(j["say"])
            kind = learn_from_correction(mind, ev, said, say)
            if kind == "unfactored":
                unfactored += 1
            else:
                lessons.append(kind)
            print(f"  {k:2d} {ev} | '{' '.join(said)}' | FAIL -> "
                  f"'{' '.join(say)}' [{kind}]")
        else:
            vote_orientation(mind, len(ev),
                             current_orientation(mind, len(ev)))
            print(f"  {k:2d} {ev} | '{' '.join(said)}' | pass")

    # learning curve + offline orientation check (slot-word ORDER in
    # his utterance vs the event; decoration reported separately —
    # the live tutor's articles are its own dialect, not gradable
    # against a script)
    half = len(verdicts) // 2
    early = sum(v == "pass" for v in verdicts[:half])
    late = sum(v == "pass" for v in verdicts[half:])
    rng2 = np.random.default_rng(93)
    ok = 0
    words = mind.net.get_meta("speak:words", "map") or {}
    back = {v: k for k, v in words.items()}
    for _ in range(40):
        ev = gen_event(rng2)
        out = [back.get(w, w) for w in speak(mind, ev, rng2)]
        ok += ([w for w in out if w in ev] == list(ev))
    deco = mind.net.get_meta("speak:decoration", "counts") or {}
    wrongs = mind.net.get_meta("speak:wrong", "record") or []

    print("\n" + "=" * 64)
    print("COMPLETE NUMBERS")
    n_j = len(verdicts)
    print(f"  format held: {n_j}/{n_j + broken} judgments "
          f"({broken} broken, skipped honestly)")
    print(f"  learning curve: pass {early}/{half} early -> "
          f"{late}/{len(verdicts) - half} late")
    print(f"  corrections: {corrections}; factorable "
          f"{corrections - unfactored}/{corrections}; lessons: "
          f"{lessons}")
    print(f"  orientation now: arity-3 {current_orientation(mind, 3)}"
          f", arity-2 {current_orientation(mind, 2)}")
    print(f"  offline: 40 unseen events, slot order correct: {ok}/40")
    print(f"  decoration learned live: {deco}")
    print(f"  wrongs kept with corrections: {len(wrongs)}")
    print("=" * 64)


if __name__ == "__main__":
    main()

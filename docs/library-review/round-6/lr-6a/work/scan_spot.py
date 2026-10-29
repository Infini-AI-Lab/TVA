"""lr-6a, round 6: spot probes of the text scan (`foreign_refs` in tools/tvr.py), unchanged since round 5
by the author's account. Each probe is a text and whether protocol §2 says the scan flags it. The
probe texts are assembled at run time from pieces, so that this file itself holds no path into a
sibling of TVA. `S` stands for a sibling folder name (not a project's name), `U` for a user name.

Usage: python3 -I scan_spot.py  (no arguments; prints to stdout)
"""
import os
import sys

TVA = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([os.pardir] * 5)))
sys.path.insert(0, os.path.join(TVA, "tools"))
import tvr  # noqa: E402

F, S, H = "fpv", "sibx", "~"
SL = "/"
PROBES = [
    # (id, text, flagged by §2)
    ("T1", H + SL + F + SL + S + SL + "a.py", True),
    ("T2", "$HOME" + SL + F + SL + S, True),
    ("T3", "${HOME:-}" + SL + F + SL + S, True),
    ("T4", H + SL + '"' + F + '"' + SL + S, True),
    ("T5", H + SL + F + SL + "{tva," + S + "}", True),
    ("T6", "file:" + SL * 3 + "home" + SL + "U" + SL + F + SL + S, True),
    ("T7", "see " + F + SL + S + SL + "x", True),
    ("T8", "cat >" + H + SL + "notes.txt", True),
    ("T9", "%7E" + SL + F + SL + S, True),
    ("T10", H + SL + F + SL + "tva" + SL + "docs", False),
    ("T11", H + SL + ".elan" + SL + "bin" + SL + "lake", False),
    ("T12", SL + "usr" + SL + "bin" + SL + "bwrap", False),
    ("T13", "round-<N>" + SL + "lr-6a" + SL, False),
    ("T14", SL + "var" + SL + "tmp" + SL + "x", True),
    ("T15", SL + "scratch" + SL + "U" + SL + "x", True),
]


def main():
    wrong = 0
    for pid, text, want in PROBES:
        got = tvr.foreign_refs(text, os.path.join(TVA, "docs"))
        ok = bool(got) == want
        wrong += not ok
        print(f"{pid}: {'flagged' if got else 'passes'}; §2 asks {'flagged' if want else 'passes'}: "
              f"{'as asked' if ok else 'NOT AS ASKED'}")
    print(f"{len(PROBES)} probes, {wrong} not as asked")


if __name__ == "__main__":
    main()

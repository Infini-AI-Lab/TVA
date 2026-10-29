"""The tool after round 10 (library issues L7, L40, L48, L50, L51, L52, L53, L54, L55): the packages made from
their upstream repositories with git free of the user's configuration (L7), interpolated strings and the
library's import closure (L40), item digests by sha256 of canonical texts (L48), the ranges and the
`@triton.jit` functions (L50), line numbering (L51), ledgers, certificates and generated pages (L52, L53),
gate staleness by contents (L53), the toolchain (L54), and `TVA.Rules` constants outside lean/TVA/Rules (L55).

Part U imports the tree's own tools/tvr.py and calls its functions on small inputs, and on a scratch copy
of the parts of the tree they read (tools/, docs/, lean/ without its build, kernels/vec_add_f32), with the
module's root pointed at the copy. Part N fetches from the packages' upstream repositories (it needs the
network) and runs `make_packages` into the scratch folder, once with a home directory whose git
configuration installs a smudge filter. Part L compiles two small kernels with `lean` (not `lake build`)
into the scratch folder and runs the tree's claim reader on them. The tree is never written, except for
out.txt. No GPU and no Triton run.

Usage: python3 -I evidence/tool-round10/run.py SCRATCH > evidence/tool-round10/out.txt
  SCRATCH  an empty folder outside TVA (about 300 MB for Part N); its path is never printed.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(TVA, "tools"))
import tvr  # noqa: E402

SCRATCH = os.path.realpath(sys.argv[1])
assert os.path.isdir(SCRATCH) and not os.listdir(SCRATCH), "SCRATCH must be an empty folder"
HOME = os.path.realpath(os.path.expanduser("~"))
REAL = {k: getattr(tvr, k) for k in ("ROOT", "KROOT", "LREVIEW", "LIBRARY_LEDGER")}


def mask(text):
    return str(text).replace(SCRATCH, "<scratch>").replace(TVA, "<tva>").replace(HOME, "~")


def show(label, value):
    print(f"- {label}: {mask(value)}")


def use_root(root):
    """Point the tool's module at a scratch root (or back at the tree, with None)."""
    if root is None:
        for k, v in REAL.items():
            setattr(tvr, k, v)
        return
    tvr.ROOT, tvr.KROOT = root, os.path.join(root, "kernels")
    tvr.LREVIEW = os.path.join(root, "docs", "library-review")
    tvr.LIBRARY_LEDGER = os.path.join(tvr.LREVIEW, "ledger.jsonl")


def scratch_root(name):
    """The parts of the tree the tool reads, copied to a fresh folder (no Lean build, no git metadata)."""
    root = os.path.join(SCRATCH, name)
    ign = shutil.ignore_patterns(".lake", "__pycache__", ".git", ".venv")
    for d in ("tools", "docs"):
        shutil.copytree(os.path.join(TVA, d), os.path.join(root, d), ignore=ign)
    shutil.copytree(os.path.join(TVA, "lean"), os.path.join(root, "lean"), ignore=ign)
    shutil.copytree(os.path.join(TVA, "kernels", "vec_add_f32"), os.path.join(root, "kernels", "vec_add_f32"), ignore=ign)
    return root


VEC = "vec_add_f32"


def part_u():
    print("# Part U: the tool's functions\n")

    print("## U1 (L40): string literals holding `{`, and the comment reader")
    cases = {
        "lr-7a H1": "def s : String := s!\"{'\"'} -- \" #eval 7\n",
        "lr-7b `--` in a nested string": "def s : String := s!\"{\"--\"}\"  #eval 7\n",
        "lr-7b `/-` in a nested string": "def s : String := s!\"a{\"/-\"}b\" #eval 7\n",
        "lr-7b a hidden `open`": "def s : String := s!\"{\"--\"}\"  open Nat in\ntheorem t : zero = 0 := rfl\n",
        "a plain string without `{` (control)": "def s : String := \"a -- b\" -- a comment #eval\n",
        "braces in a docstring and a comment (control)": "/-- `{x}` and \"{\" -/\ndef x := 1 -- s!\"{y}\"\n",
    }
    for name, text in cases.items():
        code = tvr._uncommented(text)
        show(f"{name}: lines refused", tvr.brace_string_problems(text) or "none")
        show(f"{name}: `#eval`/`open` left as code by the plain reading",
             bool(re.search(r"#eval|open Nat", code)))
    show("the library's code, every file: hygiene problems", tvr.library_hygiene_problems() or "none")
    print()

    print("## U2 (L40): every library file is reached from lean/TVA.lean")
    show("the tree", tvr.library_module_problems() or "every library file is reached")
    root = scratch_root("u2")
    open(os.path.join(root, "lean", "TVA", "Orphan.lean"), "w").write("theorem orphan : True := trivial\n")
    use_root(root)
    show("a copy with lean/TVA/Orphan.lean, which nothing imports", tvr.library_module_problems())
    use_root(None)
    print()

    print("## U3 (L50): the ranges hold every `@triton.jit` function")
    man = tvr.manifest(VEC)
    src = open(tvr.kpath(VEC, man["source"]["file"]), encoding="utf-8", newline="").read()
    mm = tvr.load_json(tvr.kpath(VEC, "model-map.json"))
    lean = " ".join(open(p, encoding="utf-8").read() for p in tvr.kernel_lean_files(VEC))
    show("jit functions of the source", tvr.jit_functions(src))
    show("the kernel as it is (ranges [[29, 54]])", tvr.map_problems(man, mm, src, lean))
    mm52 = {"entries": [e for e in mm["entries"] if 52 not in tvr.parse_lines(e["src"])]}
    for label, ranges in (("lr-7a M1: line 52 (output = x + y) unmapped, ranges [[29, 51], [53, 54]]", [[29, 51], [53, 54]]),
                          ("lr-7a M1: line 52 unmapped, ranges []", []),
                          ("line 52 unmapped, ranges [[29, 54]] (control)", [[29, 54]])):
        m = json.loads(json.dumps(man))
        m["source"]["ranges"] = ranges
        show(label, tvr.map_problems(m, mm52, src, lean))
    helper = ("import triton\nimport triton.language as tl\n\n@triton.jit\ndef helper(x):\n    return x + 1\n\n\n"
              "@triton.autotune(configs=[], key=[])\n@triton.jit\ndef kernel(p):\n    tl.store(p, helper(tl.load(p)))\n")
    hman = {"source": {"ranges": [[9, 12]], "unused": []}}
    hmm = {"entries": [{"src": [9, 10, 11], "drop": "decl"}, {"src": [12], "model": "x"}]}
    show("a helper outside the ranges", tvr.map_problems(hman, hmm, helper, "x")[0])
    hman["source"]["unused"] = ["helper"]
    show("the same with source.unused = [helper]", tvr.map_problems(hman, hmm, helper, "x"))
    hman["source"]["unused"] = ["nothing"]
    show("source.unused naming no jit function", tvr.map_problems(hman, hmm, helper, "x")[0])
    show("manifest_problems on source.unused = [1]", tvr.manifest_problems(VEC, {**man, "source": {**man["source"], "unused": [1]}}))
    print()

    print("## U4 (L51): line breaks other than a line feed")
    m2 = ["@triton.jit", "def k(x_ptr, c_ptr):", "    # a comment with a line separator", "    tl.atomic_add(c_ptr, 1)",
          "    x = tl.load(x_ptr)", "    tl.store(x_ptr, x)"]
    text = "\n".join(m2) + "\n"
    show("lr-7a M2: line 6 by str.splitlines (round 7)", repr(text.splitlines()[5]))
    show("line 6 by line feeds (round 10)", repr(text.split("\n")[5]))
    show("line 6 as Python's tokenizer and ast count it: the statement", repr(m2[5]))
    sman = {"source": {"ranges": [[1, 6]], "unused": []}}
    smm = {"entries": [{"src": [1, 2], "drop": "decl"}, {"src": [4, 5, 6], "model": "x"}]}
    show("the map check on that source", tvr.map_problems(sman, smm, text, "x")[0])
    for ch, name in (("\x0c", "form feed"), ("\r", "lone carriage return"), (" ", "U+2029")):
        show(f"the map check with a {name}", tvr.map_problems(sman, smm, text.replace(" ", ch), "x")[0][:1])
    show("the map check with CRLF line ends (allowed)", tvr.map_problems(sman, smm, text.replace(" ", " ").replace("\n", "\r\n"), "x"))
    root = scratch_root("u4")
    use_root(root)
    p = os.path.join(root, "kernels", VEC, "lean", "VecAdd.lean")
    open(p, "a").write("-- a page break\x0chere\n")
    show("kernel hygiene with a form feed in VecAdd.lean", [x for x in tvr.hygiene_problems(VEC) if "line break" in x])
    use_root(None)
    print()

    print("## U5 (L53): gate staleness by contents and by the gate's code")
    root = scratch_root("u5")
    use_root(root)
    kd = tvr.kpath(VEC)
    shutil.rmtree(os.path.join(kd, "review", "snapshots"), ignore_errors=True)
    gate = {"ok": True, "time": tvr.now(), "digest": "0" * 64, "items": {"model": "1" * 64},
            "files": tvr.kernel_files(VEC), "tool": tvr.tool_digest()}
    tvr.dump_json(os.path.join(kd, "review", "snapshots", "0" * 16 + ".json"),
                  {"items": {"model": {"digest": "1" * 64, "cone": {"file:docs/SEMANTICS.md": tvr._file_digest("docs/SEMANTICS.md"),
                                                                     "packages": tvr.packages_digest()}}}})
    show("S0 a gate recorded now", tvr.gate_staleness(VEC, gate) or "not stale")
    src_p = os.path.join(kd, "source", "01-vector-add.py")
    for label, path, add in (("S1 a comment added to the source", src_p, "# a comment\n"),
                             ("S1 a line added to the write-up", os.path.join(kd, "README.md"), "\nOne more line.\n")):
        old = open(path, "rb").read()
        open(path, "ab").write(add.encode())
        show(label, tvr.gate_staleness(VEC, gate))
        open(path, "wb").write(old)
    lp = os.path.join(kd, "lean", "VecAdd.lean")
    st = os.stat(lp)
    old = open(lp, "rb").read()
    open(lp, "wb").write(old.replace(b"BLOCK_SIZE", b"BLOCK_SIZE ", 1))
    os.utime(lp, ns=(st.st_atime_ns, st.st_mtime_ns))
    show("S1 VecAdd.lean changed, its old modification time put back", tvr.gate_staleness(VEC, gate))
    open(lp, "wb").write(old)
    open(os.path.join(kd, "lean", "Extra.lean"), "w").write("-- a new file\n")
    show("a Lean file added", tvr.gate_staleness(VEC, gate))
    os.remove(os.path.join(kd, "lean", "Extra.lean"))
    rd = os.path.join(root, "tools", "claim_reader.lean.in")
    old = open(rd).read()
    open(rd, "a").write("-- a change to the claim reader\n")
    show("a change to the gate's code (tools/claim_reader.lean.in)", tvr.gate_staleness(VEC, gate))
    open(rd, "w").write(old)
    show("the tree's shipped vec_add_f32 gate (from 2026-10-27)", "; ".join(
        (use_root(None), tvr.gate_staleness(VEC, tvr.load_json(tvr.kpath(VEC, "gate.json"))))[1]))
    print()

    print("## U6 (L52, L53): ledgers, certificates and generated pages")
    root = scratch_root("u6")
    led_lib = os.path.join(root, "docs", "library-review", "ledger.jsonl")
    if os.path.exists(led_lib):
        os.remove(led_lib)  # the copy starts with no library ledger, as before round 10
    use_root(root)
    led = f"kernels/{VEC}/review/ledger.jsonl"
    show("the kernel's ledger as shipped", tvr.ledger_problems(root, led, led) or "every recorded file as recorded")
    events = [json.loads(l) for l in open(os.path.join(root, led)) if l.strip()]
    recorded = sorted({p for e in events for p in e.get("files", {})})
    show("files the ledger records", recorded)
    fp = os.path.join(root, recorded[0])
    open(fp, "a").write("\n")
    show("a recorded file changed after recording", tvr.ledger_problems(root, led, led))
    open(fp, "w").write(open(os.path.join(TVA, recorded[0])).read())
    work = os.path.join(os.path.dirname(fp), "work")
    rdir = os.path.relpath(os.path.dirname(fp), root)
    with open(os.path.join(root, led), "a") as f:  # a filing recorded by round 10's tool names its folder
        f.write(json.dumps({"seq": len(events) + 1, "type": "note", "actor": "coordinator", "dir": rdir,
                            "files": tvr.archive_entry(*tvr.text_files(os.path.dirname(fp)))}) + "\n")
    show("a folder recorded with every file in it (as `tvr file` now records)", tvr.ledger_problems(root, led, led) or "consistent")
    os.makedirs(work, exist_ok=True)
    open(os.path.join(work, "late.txt"), "w").write("added after recording\n")
    show("a work file added to that folder after recording", tvr.ledger_problems(root, led, led))
    cert = {"kernel": VEC, "gate": "a" * 64, "items": {"model": "b" * 64}, "reviewers": [], "time": "t"}
    show("lr-7a G1: a shipped certificate that the ledger does not grant (regenerated: none)",
         tvr.certificate_mismatch(cert, None))
    show("a certificate regenerated the same but for its time", tvr.certificate_mismatch(cert, {**cert, "time": "u"}) or "the same")
    show("gate_mismatch: a shipped gate whose kernel files differ",
         tvr.gate_mismatch({"ok": True, "digest": "d", "items": {}, "files": {"a": "1"}, "tool": {}},
                           {"ok": True, "digest": "d", "items": {}, "files": {"a": "2"}, "tool": {}}))
    # library records: record round 7's folders in the copy, then change, add, leave a round out
    for who in ("author", "lr-7a", "lr-7b"):
        subprocess.run([sys.executable, "-I", "-c", "import sys; sys.argv=['tvr','record','library','7',sys.argv[1],'--no-lint'];"
                        "sys.path.insert(0,'tools'); sys.dont_write_bytecode=True; import tvr; sys.exit(tvr.main(sys.argv))", who],
                       cwd=root, capture_output=True, text=True)
    show("round 7 recorded in the copy: events", len(tvr.library_ledger(root)))
    again = subprocess.run([sys.executable, "-I", "tools/tvr.py", "record", "library", "7", "lr-7a", "--no-lint"],
                           cwd=root, capture_output=True, text=True)
    show("recording lr-7a a second time", (again.stderr or again.stdout).strip())
    rp = tvr.records_problems(root)
    show("records step: rounds 1-6 not recorded (first 2)", [x for x in rp if "not recorded" in x][:2])
    wf = os.path.join(root, "docs", "library-review", "round-7", "lr-7a", "work", "release.out")
    open(wf, "a").write("changed\n")
    open(os.path.join(root, "docs", "library-review", "round-7", "lr-7b", "work", "late.txt"), "w").write("added\n")
    show("a recorded work file changed, a work file added", [x for x in tvr.records_problems(root) if "round-7" in x])
    show("release_records lints only the last round's unrecorded folders", tvr.release_records(root))
    use_root(None)
    pages = {}
    for side, t in (("shipped", "2026-10-27T00:00:00Z"), ("regenerated", "2026-10-28T12:00:00Z")):
        kd = os.path.join(SCRATCH, "u6-" + side, "kernels", VEC)
        os.makedirs(kd)
        open(os.path.join(kd, "kernel.json"), "w").write("{}")
        tvr.dump_json(os.path.join(kd, "gate.json"), {"ok": True, "time": t})
        open(os.path.join(kd, "REVIEW.md"), "w").write(f"- Gate: **PASS**, digest `x` ({t}).\n- Certificate: **GRANTED**.\n")
        pages[side] = os.path.join(SCRATCH, "u6-" + side)
    show("generated pages that differ only in their gate's time", tvr.page_mismatch(pages["shipped"], pages["regenerated"]) or "the same")
    rv = os.path.join(pages["shipped"], "kernels", VEC, "REVIEW.md")
    open(rv, "w").write(open(rv).read().replace("GRANTED", "not yet"))
    show("a shipped REVIEW.md that says otherwise", tvr.page_mismatch(pages["shipped"], pages["regenerated"]))
    print()

    print("## U7 (L54): the toolchain")
    root = scratch_root("u7")
    use_root(root)
    tcf = os.path.join(root, "lean", "lean-toolchain")
    for tc in ("leanprover/lean4:v4.35.0", "someone/lean4:v4.35.0-patched", "leanprover/lean4:nightly", "leanprover/lean4:v4.35.0\nextra"):
        open(tcf, "w").write(tc + "\n")
        show(f"lean-toolchain {tc!r}", [x for x in tvr.lake_violations() if "lean-toolchain" in x] or "accepted")
    open(tcf, "w").write("leanprover/lean4:v4.35.0\n")
    home = os.path.join(SCRATCH, "u7-home")
    tdir = os.path.join(home, ".elan", "toolchains", "leanprover--lean4---v4.35.0")
    os.makedirs(os.path.join(home, ".elan", "bin"))
    os.makedirs(os.path.join(SCRATCH, "u7-elsewhere"))
    os.makedirs(os.path.dirname(tdir))
    os.symlink(os.path.join(SCRATCH, "u7-elsewhere"), tdir)  # as `elan toolchain link` makes it
    show("a toolchain folder that is a symbolic link", tvr.elan_parts(home)[1])
    os.remove(tdir)
    os.makedirs(tdir)
    show("a real toolchain folder", tvr.elan_parts(home)[1])
    use_root(None)
    show("the tree, with the user's toolchain", tvr.elan_parts(HOME)[1])
    print()

    print("## U8 (L55): a `TVA.Rules.*` constant declared outside lean/TVA/Rules")
    consts = {"TVA.Rules.Num.borrowed": {"name": "TVA.Rules.Num.borrowed", "module": "TVA.Lemmas.Num", "hash": "h"},
              "TVA.Rules.Compiler.Faithful": {"name": "TVA.Rules.Compiler.Faithful", "module": "TVA.Rules.Compiler", "hash": "h"},
              "TVA.Rules.Compiler.Faithful.eq_1": {"name": "TVA.Rules.Compiler.Faithful.eq_1", "module": "VecAdd", "hash": "h"}}
    surf = {"fields": ["num"], "field_docs": {"num": "[rule] a device fact"},
            "field_uses": {"num": {"consts": ["TVA.Rules.Num.borrowed"], "K": False, "P": True}}}
    show("kinds", tvr.kinds_problems(surf, consts, "Kernels.X"))
    show("rule pages in the digest of a cone holding it", sorted(k for k in tvr.library_files(["TVA.Rules.Num.borrowed"], consts) if "rules" in k))
    print()

    print("## U9 (L7): the manifest against Mathlib's own")
    man = tvr.load_json(os.path.join(TVA, "lean", "lake-manifest.json"))
    mrev = next(p["rev"] for p in man["packages"] if p["name"] == "mathlib")
    theirs = json.loads(subprocess.run(["git", "-C", os.path.join(TVA, "lean", ".lake", "packages", "mathlib"), "show",
                                        f"{mrev}:lake-manifest.json"], capture_output=True, text=True, env=tvr._git_env()).stdout)
    show("TVA's inherited packages against Mathlib's manifest at the pinned commit (read from the commit's blob)",
         tvr.inherited_problems(man, theirs) or "the same")
    m2 = json.loads(json.dumps(man))
    next(p for p in m2["packages"] if p["name"] == "batteries")["rev"] = "7d4e7b741a64" + "0" * 28
    show("lr-7a: batteries pinned at a local commit", tvr.inherited_problems(m2, theirs))
    m2 = json.loads(json.dumps(man))
    next(p for p in m2["packages"] if p["name"] == "batteries")["url"] = "https://example.invalid/someone/batteries"
    show("lr-7a P4: an inherited package from another URL", tvr.inherited_problems(m2, theirs))
    root = scratch_root("u9")
    use_root(root)
    mp = os.path.join(root, "lean", "lake-manifest.json")
    m2 = json.loads(json.dumps(man))
    m2["packagesDir"] = "../evidence/pk"
    json.dump(m2, open(mp, "w"))
    show("packagesDir ../evidence/pk", [x for x in tvr.lake_violations() if "packagesDir" in x])
    use_root(None)
    print()


def part_n():
    print("# Part N: the packages from their upstream repositories (network)\n")
    man = tvr.load_json(os.path.join(TVA, "lean", "lake-manifest.json"))
    m = next(p for p in man["packages"] if p["name"] == "mathlib")
    b = next(p for p in man["packages"] if p["name"] == "batteries")
    os.makedirs(os.path.join(SCRATCH, "n-anc"))
    show("N1 Mathlib's pin against the tag v4.35.0", tvr.upstream_problem(m["url"], m["inputRev"], m["rev"], os.path.join(SCRATCH, "n-anc", "1")) or "the tag's commit")
    show("N1 lr-7a's local Mathlib commit against the tag", tvr.upstream_problem(m["url"], m["inputRev"], "724ae5f0a9ab" + "0" * 28, os.path.join(SCRATCH, "n-anc", "2")))
    show("N1 an inputRev that names no ref (a commit hash)", tvr.upstream_problem(m["url"], m["rev"], m["rev"], os.path.join(SCRATCH, "n-anc", "3")))
    t = time.time()
    show("N1 batteries' pin against its branch main (reachable, not the head)", tvr.upstream_problem(b["url"], "main", b["rev"], os.path.join(SCRATCH, "n-anc", "4")) or "reachable from main")
    show("N1 aesop's pin against batteries' main (not reachable)", tvr.upstream_problem(b["url"], "main", next(p for p in man["packages"] if p["name"] == "aesop")["rev"], os.path.join(SCRATCH, "n-anc", "5")))
    print(f"  (the two branch checks took {time.time() - t:.0f} s)")

    # N2: lr-7b's route: the user's git configuration installs a smudge filter for *.lean
    home = os.path.join(SCRATCH, "n2-home")
    os.makedirs(home)
    open(os.path.join(home, ".gitattributes-global"), "w").write("*.lean filter=probe\n")
    script = os.path.join(home, "smudge.sh")
    open(script, "w").write("#!/bin/sh\necho '-- a line the commit does not have'\ncat\n")
    os.chmod(script, 0o755)
    open(os.path.join(home, ".gitconfig"), "w").write("[core]\n\tattributesFile = " + os.path.join(home, ".gitattributes-global") +
                                                      "\n[filter \"probe\"]\n\tsmudge = " + script + "\n\tclean = cat\n")
    saved = os.environ.get("HOME")
    os.environ["HOME"] = home
    lean = os.path.join(SCRATCH, "n2", "lean")
    os.makedirs(lean)
    os.makedirs(os.path.join(SCRATCH, "n2-scratch"))
    t = time.time()
    bad = tvr.make_packages(lean, os.path.join(SCRATCH, "n2-scratch"))
    os.environ["HOME"] = saved
    show("N2 make_packages with that home directory, the tree's manifest: problems", bad or "none")
    print(f"  (all {len(man['packages'])} packages fetched and checked in {time.time() - t:.0f} s)")
    cli = os.path.join(lean, ".lake", "packages", "Cli")
    if not os.path.isdir(cli):
        print("  (Cli was not made, so the rest of N2 does not run)")
        return
    first = sorted(f for f in os.listdir(cli) if f.endswith(".lean"))[0]
    show(f"N2 first line of Cli/{first}", repr(open(os.path.join(cli, first)).readline().rstrip("\n")))
    rev = next(p["rev"] for p in man["packages"] if p["name"] == "Cli")
    show("N2 Cli's checkout against its blobs", tvr.checkout_problems(cli, rev) or "every file is its blob")
    f = os.path.join(cli, first)
    open(f, "a").write("theorem borrowed : True := trivial\n")
    open(os.path.join(cli, "Extra.lean"), "w").write("-- not in the commit\n")
    show("N2 the same after a file is changed and one added", tvr.checkout_problems(cli, rev))
    # lr-7a's P2 route end to end: the manifest pins a Mathlib commit that is not the tag's
    root = scratch_root("n3")
    use_root(root)
    m2 = json.loads(json.dumps(man))
    next(p for p in m2["packages"] if p["name"] == "mathlib")["rev"] = "724ae5f0a9ab" + "0" * 28
    json.dump(m2, open(os.path.join(root, "lean", "lake-manifest.json"), "w"))
    os.makedirs(os.path.join(SCRATCH, "n3-lean"))
    os.makedirs(os.path.join(SCRATCH, "n3-scratch"))
    show("N3 make_packages with Mathlib pinned at a local commit", tvr.make_packages(os.path.join(SCRATCH, "n3-lean"), os.path.join(SCRATCH, "n3-scratch")))
    use_root(None)
    print()


DEMO = """import TVA

namespace Kernels.Demo
open TVA

/-- The model (a stand-in for a kernel's). -/
def model : Nat → Nat := fun {x} => {x} + {n}
/-- The launch (a stand-in). -/
def launch : Nat := 0
/-- The instances. -/
structure Inst where
  n : Nat
/-- None is covered. -/
def Inst.ok (_ : Inst) : Prop := False
def Pre (_ : Inst) (_ : Mem) : Prop := True
def Post (_ : Inst) (_ _ : Mem) : Prop := True
def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K
structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [kernel] a fact about the model. -/
  own : model 0 = model 0
theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K := by
{proof}
theorem consistent : Claim.Consistent Assumptions :=
  ⟨fun _ _ o => o = .hang, Platform.any, fun _ _ => ⟨.hang, rfl⟩, ⟨rfl⟩⟩
end Kernels.Demo
"""
PROOF_1 = "  intro K P h d hd\n  have := h.own\n  exact absurd hd id"
PROOF_2 = "  intro K P h d hd\n  have _u := h.own\n  exact absurd hd (fun h => h)"


def part_l():
    print("# Part L (L48): item digests from the claim reader, on two kernels whose old hashes collide\n")
    print("lr-7a's collision: `fun x => x + a` and `fun x => x + b` had the same 64-bit value hash. Kernel C is A")
    print("with a proof edit and a renamed bound variable; kernel D is A with `a + 1`.\n")
    a, b = 4536642005803190292, 3061831643823180776
    sides = {"A": (a, "x", PROOF_1), "B": (b, "x", PROOF_1), "C": (a, "y", PROOF_2), "D": (a + 1, "x", PROOF_1)}
    d = os.path.join(SCRATCH, "lean")
    os.makedirs(os.path.join(d, "olean"))
    lean_dir = os.path.join(TVA, "lean")
    pre = subprocess.run([tvr.LAKE, "env", "printenv", "LEAN_PATH"], cwd=lean_dir, capture_output=True, text=True).stdout.strip()
    env = dict(os.environ, LEAN_PATH=pre + os.pathsep + os.path.join(d, "olean"))
    digests = {}
    for s, (n, x, proof) in sides.items():
        f = os.path.join(d, f"Demo{s}.lean")
        open(f, "w").write(DEMO.replace("{n}", str(n)).replace("{x}", x).replace("{proof}", proof))
        r = subprocess.run([tvr.LAKE, "env", "lean", f"--root={d}", "-o", os.path.join(d, "olean", f"Demo{s}.olean"), f],
                           cwd=lean_dir, capture_output=True, text=True)
        if r.returncode:
            print(f"- {s}: does not compile: {mask(r.stdout + r.stderr)[:300]}")
            continue
        reader = os.path.join(d, f"reader{s}.lean")
        open(reader, "w").write(tvr.claim_reader_text(f"Demo{s}", "Kernels.Demo", set()))
        r = subprocess.run([tvr.LAKE, "env", "lean", reader], cwd=lean_dir, capture_output=True, text=True, env=env)
        js = [l for l in r.stdout.splitlines() if l.startswith("TVR_JSON ")]
        if r.returncode or len(js) != 1:
            print(f"- {s}: the reader fails: {mask(r.stdout + r.stderr)[:300]}")
            continue
        surf = json.loads(js[0][len("TVR_JSON "):])
        consts = tvr.const_table(surf)
        canon = next(c["canon"] for c in surf["consts"] if c and c["name"] == "Kernels.Demo.model")
        digests[s] = {k: v["digest"][:16] for k, v in tvr.item_digests(surf, consts, "0" * 64, "{}").items()}
        print(f"- {s}: model's canonical text: `{canon[:170]}…`")
        print(f"  model's hash {consts['Kernels.Demo.model']['hash'][:16]}; item digests {digests[s]}")
    if len(digests) == 4:
        for x, y in (("A", "B"), ("A", "C"), ("A", "D")):
            same = sorted(i for i in digests[x] if digests[x][i] == digests[y][i])
            print(f"- {x} against {y}: the same digest for {', '.join(same) or 'no item'}; different for "
                  f"{', '.join(sorted(set(digests[x]) - set(same))) or 'no item'}")
    print()


if __name__ == "__main__":
    print(f"tvr: tools/tvr.py sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'tvr.py'))[:16]}, "
          f"tools/claim_reader.lean.in sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'claim_reader.lean.in'))[:16]}; "
          f"{subprocess.run(['git', '--version'], capture_output=True, text=True).stdout.strip()}; Python {sys.version.split()[0]}\n")
    part_u()
    part_n()
    part_l()

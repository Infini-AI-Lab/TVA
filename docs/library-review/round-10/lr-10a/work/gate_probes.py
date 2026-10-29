"""lr-10a, round 10 (gate survey): probes of the round-10 tool, on scratch copies of the tree.

Imports the tree's tools/tvr.py and calls its functions on small inputs and on scratch copies of the parts
of the tree they read (tools/, docs/, lean/ without its build, kernels/vec_add_f32), with the module's root
pointed at the copy, as evidence/tool-round10/run.py does. The tree is never written. G7 needs the network
(it reads the refs of Mathlib's upstream repository).

  G1 (L52) a hand-written `accept` line in a kernel's ledger grants the certificate, and every comparison of
           the release check agrees with it
  G2 (L52) a library folder of the current round recorded with `tvr record ... --no-lint` is never linted
  G3 (L52) a recorded library report changed, with its sha256 edited in the library ledger
  G4 (L53) lean/lean-toolchain changed after a gate ran: is the gate stale?
  G5 (L50) `@triton.jit` functions spelled without a decorator that names `jit`, and a constexpr helper
  G6 (L50) the map check on a source whose kernel is made by `triton.jit(f)`
  G7 (L7)  `upstream_problem` on the head of a feature branch of Mathlib's repository
  G8 (L7)  the audit on a file in a folder of kernels/ that holds no kernel.json (run with the extra argument g8)
  G9 (L52) `tvr file` after a file an earlier filing recorded changed (run with the extra argument g9)

Usage: python3 -I work/gate_probes.py SCRATCH [g8|g9]   (SCRATCH: an empty folder; its path is never printed)
"""
import json
import os
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.realpath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(TVA, "tools"))
import tvr  # noqa: E402

SCRATCH = os.path.realpath(sys.argv[1])
assert os.path.isdir(SCRATCH) and not os.listdir(SCRATCH), "SCRATCH must be an empty folder"
HOME = os.path.realpath(os.path.expanduser("~"))
REAL = {k: getattr(tvr, k) for k in ("ROOT", "KROOT", "LREVIEW", "LIBRARY_LEDGER")}
VEC = "vec_add_f32"


def mask(text):
    return str(text).replace(SCRATCH, "<scratch>").replace(TVA, "<tva>").replace(HOME, "~")


def show(label, value):
    print(f"- {label}: {mask(value)}")


def use_root(root):
    if root is None:
        for k, v in REAL.items():
            setattr(tvr, k, v)
        return
    tvr.ROOT, tvr.KROOT = root, os.path.join(root, "kernels")
    tvr.LREVIEW = os.path.join(root, "docs", "library-review")
    tvr.LIBRARY_LEDGER = os.path.join(tvr.LREVIEW, "ledger.jsonl")


def scratch_root(name):
    root = os.path.join(SCRATCH, name)
    ign = shutil.ignore_patterns(".lake", "__pycache__", ".git", ".venv")
    for d in ("tools", "docs", "lean"):
        shutil.copytree(os.path.join(TVA, d), os.path.join(root, d), ignore=ign)
    shutil.copytree(os.path.join(TVA, "kernels", VEC), os.path.join(root, "kernels", VEC), ignore=ign)
    return root


def fresh_gate(root):
    """A passing gate recorded now in the copy (standing for the gate the sandbox recomputes), with its
    snapshot, as `tvr check` writes them: every kernel file and the gate's code by sha256, the library
    files of its items."""
    kd = tvr.kpath(VEC)
    items = {i: tvr.sha(i + " recomputed") for i in ("assumption:faithful", "conditions", "map", "model", "spec")}
    digest = tvr.sha("\n".join(f"{i} {d}" for i, d in sorted(items.items())))
    gate = {"protocol": "tvr-2", "kernel": VEC, "time": tvr.now(), "ok": True, "digest": digest, "checks": [],
            "items": items, "kinds": {"faithful": "rule"}, "files": tvr.kernel_files(VEC), "tool": tvr.tool_digest()}
    tvr.dump_json(os.path.join(kd, "gate.json"), gate)
    files = {"file:docs/SEMANTICS.md": tvr._file_digest("docs/SEMANTICS.md"), "packages": tvr.packages_digest()}
    tvr.dump_json(os.path.join(kd, "review", "snapshots", digest[:16] + ".json"),
                  {"items": {i: {"digest": d, "cone": dict(files)} for i, d in items.items()}})
    return gate


def g1():
    print("## G1 (L52): a hand-written accept line in a kernel's ledger")
    root = scratch_root("g1")
    use_root(root)
    gate = fresh_gate(root)
    S = tvr.state(VEC)
    show("the reviewers' last acceptances are at the old gate; certificate now", tvr.cert_state(VEC, S, gate)[0])
    led = os.path.join(tvr.kpath(VEC), "review", "ledger.jsonl")
    n = len(tvr.ledger(VEC))
    with open(led, "a") as f:  # one line, written by hand: no filing, no report, no `tvr file`
        f.write(json.dumps({"seq": n + 1, "time": tvr.now(), "type": "accept", "actor": "r-vecadd", "round": 2,
                            "items": dict(gate["items"]), "gate": gate["digest"]}, sort_keys=True) + "\n")
    S = tvr.state(VEC)
    show("after the line: certificate", tvr.cert_state(VEC, S, gate)[0])
    tvr.render_review(VEC)
    shipped = tvr._read_json(os.path.join(tvr.kpath(VEC), "review", "certificate.json"))
    show("certificate.json written by `tvr` from that ledger", "present" if shipped else "absent")
    # the release check regenerates the certificate from the same ledger and the recomputed gate
    tvr.render_review(VEC)
    regen = tvr._read_json(os.path.join(tvr.kpath(VEC), "review", "certificate.json"))
    show("release check: certificate_mismatch(shipped, regenerated)", tvr.certificate_mismatch(shipped, regen) or "the same")
    rel = f"kernels/{VEC}/review/ledger.jsonl"
    show("release check: ledger_problems", tvr.ledger_problems(root, rel, rel) or "none")
    rec = json.load(open(os.path.join(tvr.kpath(VEC), "review", "round-2", "r-vecadd", "filing.json")))
    show("the reviewer's recorded round-2 filing accepts", rec.get("accept"))
    show("REVIEW.md's status lines", [l for l in open(os.path.join(tvr.kpath(VEC), "REVIEW.md")).read().splitlines()
                                       if l.startswith("- Certificate") or l.startswith("- Recorded")])
    use_root(None)
    print()


def g2():
    print("## G2 (L52): `tvr record library ROUND WHO --no-lint` on the current round")
    root = scratch_root("g2")
    d = os.path.join(root, "docs", "library-review", "round-10", "lr-x")
    os.makedirs(d)
    open(os.path.join(d, "report.md"), "w").write("# Library review, round 10, by lr-x\n\nEverything is fine.\n")
    use_root(root)
    show("lint of that report", tvr.library_lint_problems(10, "lr-x")[:2])
    r = subprocess.run([sys.executable, "-I", "tools/tvr.py", "record", "library", "10", "lr-x", "--no-lint"],
                       cwd=root, capture_output=True, text=True)
    show("tvr record library 10 lr-x --no-lint", (r.stdout + r.stderr).strip())
    show("release check: the lint commands it runs for round 10 (release_records)",
         [c for c in tvr.release_records(root) if c[:3] == ["lint", "library", "10"]])
    show("release check: records_problems on round 10", [x for x in tvr.records_problems(root) if "round-10" in x] or "none")
    use_root(None)
    print()


def g3():
    print("## G3 (L52): a recorded library report changed, and its line in the library ledger edited")
    root = scratch_root("g3")
    use_root(root)
    p = os.path.join(root, "docs", "library-review", "round-7", "lr-7a", "report.md")
    old = open(p).read()
    open(p, "w").write(old.replace("- **L7** REOPEN", "- **L7** ACCEPT", 1))
    rel = "docs/library-review/round-7/lr-7a/report.md"
    show("records_problems after the change", [x for x in tvr.records_problems(root) if "round-7" in x])
    lp = os.path.join(root, "docs", "library-review", "ledger.jsonl")
    lines = open(lp).read().splitlines()
    out = []
    for l in lines:
        e = json.loads(l)
        if rel in e.get("files", {}):
            e["files"][rel] = tvr.file_sha(p)
        out.append(json.dumps(e, sort_keys=True, ensure_ascii=False))
    open(lp, "w").write("\n".join(out) + "\n")
    show("records_problems after the ledger line is edited too", [x for x in tvr.records_problems(root) if "round-7" in x] or "none")
    use_root(None)
    print()


def g4():
    print("## G4 (L53): lean/lean-toolchain changed after the gate ran")
    root = scratch_root("g4")
    use_root(root)
    gate = fresh_gate(root)
    show("a gate recorded now", tvr.gate_staleness(VEC, gate) or "not stale")
    tc = os.path.join(root, "lean", "lean-toolchain")
    open(tc, "w").write("leanprover/lean4:v4.36.0\n")
    show("lean/lean-toolchain now leanprover/lean4:v4.36.0 (the audit's lake check)",
         [x for x in tvr.lake_violations() if "toolchain" in x] or "accepted")
    show("the gate", tvr.gate_staleness(VEC, gate) or "not stale")
    show("the certificate's gate condition (cert_state's `full` needs ok and not stale)",
         gate["ok"] and not tvr.gate_staleness(VEC, gate))
    use_root(None)
    print()


HEAD = "import triton\nimport triton.language as tl\n"
VARIANTS = {
    "a decorator `@triton.jit` (control)": "@triton.jit\ndef k(p):\n    tl.store(p, 1)\n",
    "`triton.jit(f)` called, not a decorator": "def _k(p):\n    tl.store(p, 1)\n\n\nk = triton.jit(_k)\n",
    "an aliased decorator (`from triton import jit as J`)": "from triton import jit as J\n\n\n@J\ndef k(p):\n    tl.store(p, 1)\n",
    "a `@triton.constexpr_function` helper": ("@triton.constexpr_function\ndef width(n):\n    return 2 * n\n\n\n"
                                              "@triton.jit\ndef k(p, N: tl.constexpr):\n    tl.store(p + tl.arange(0, width(N)), 1)\n"),
}


def g5():
    print("## G5 (L50): which functions `jit_functions` finds")
    for label, body in VARIANTS.items():
        show(label, tvr.jit_functions(HEAD + body))
    print()


def g6():
    print("## G6 (L50): the map check when a helper is made by `triton.jit(_h)` and the ranges leave it out")
    src = HEAD + ("\n\ndef _h(x):\n    return x + 1\n\n\nh = triton.jit(_h)\n\n\n"
                  "@triton.jit\ndef k(p):\n    tl.store(p, h(tl.load(p)))\n")
    man = {"source": {"ranges": [[12, 14]], "unused": []}}
    mm = {"entries": [{"src": [12, 13], "drop": "decl"}, {"src": [14], "model": "x"}]}
    show("source lines", {i + 1: l for i, l in enumerate(src.split("\n")) if l})
    show("map_problems with ranges [[12, 14]] (the kernel `k` only; `_h`, which `k` calls, outside)",
         tvr.map_problems(man, mm, src, "x"))
    print()


def g7():
    print("## G7 (L7): a feature branch of Mathlib's repository as `inputRev`")
    url = "https://github.com/leanprover-community/mathlib4"
    r = tvr._git("ls-remote", "--heads", url)
    heads = {}
    for line in r.stdout.decode().splitlines():
        oid, _, name = line.partition("\t")
        heads[name.strip()[len("refs/heads/"):]] = oid.strip()
    show("branches of Mathlib's repository", len(heads))
    for b in ("feat/abel-sym-module", "ci-dev/marcelo/parsteps4-broken"):
        if b in heads:
            os.makedirs(os.path.join(SCRATCH, "g7", b.replace("/", "_")))
            show(f"upstream_problem(inputRev {b!r}, its head {heads[b][:12]})",
                 tvr.upstream_problem(url, b, heads[b], os.path.join(SCRATCH, "g7", b.replace("/", "_"))) or "None (accepted)")
    show("the audit's check of Mathlib's inputRev: only that it equals the lakefile's `rev`", "lake_violations")
    print()


if __name__ == "__main__" and len(sys.argv) == 2:
    print(f"tvr: tools/tvr.py sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'tvr.py'))[:16]}, "
          f"tools/claim_reader.lean.in sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'claim_reader.lean.in'))[:16]}\n")
    g1()
    g2()
    g3()
    g4()
    g5()
    g6()
    g7()


def g8():
    print("## G8 (L7, the audit's text scan): a file in a folder of kernels/ that has no kernel.json")
    root = os.path.join(SCRATCH, "g8")
    os.makedirs(root)
    ign = shutil.ignore_patterns(".lake", "__pycache__", ".git", ".venv")
    for d in sorted(os.listdir(TVA)):
        s = os.path.join(TVA, d)
        if d in (".git", ".lake", ".venv", "__pycache__"):
            continue
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(root, d), ignore=ign, symlinks=True)
        else:
            shutil.copy2(s, os.path.join(root, d))
    sibling = "~/" + "fp" + "v/" + "sibling/lemmas.lean"  # a path into a sibling of TVA, built here so this file holds none
    for where in ("kernels/notes/readme.md", "kernels/vec_add_f32/notes/readme.md"):
        os.makedirs(os.path.dirname(os.path.join(root, where)), exist_ok=True)
        open(os.path.join(root, where), "w").write(f"The proof follows {sibling}.\n")
    r = subprocess.run([sys.executable, "-I", "tools/tvr.py", "audit"], cwd=root, capture_output=True, text=True)
    out = (r.stdout + r.stderr).splitlines()
    for where in ("kernels/notes/readme.md", "kernels/vec_add_f32/notes/readme.md"):
        show(f"`tvr audit` names {where}", any(l.startswith(where) for l in out))
    show("the folders `kernels()` lists (a kernel.json each)", tvr.kernels())
    print()


def g9():
    print("## G9 (L52): a second filing in a round, after a file the first one recorded was changed")
    root = scratch_root("g9")
    use_root(root)
    fresh_gate(root)
    rd = os.path.join(tvr.kpath(VEC), "review", "round-2", "r-vecadd")
    open(os.path.join(rd, "report.md"), "a").write("\nA line added after recording.\n")
    tvr.dump_json(os.path.join(rd, "filing-2.json"), {"reviewer": "r-vecadd", "kernel": VEC, "round": 2, "findings": [],
                                                        "release": [], "reraise": [], "accept": [], "notes": ""})
    open(os.path.join(rd, "report-2.md"), "w").write(
        f"# Review of {VEC}, round 2, by r-vecadd\n\n## Verdict\n\nreject\nNothing accepted.\n\n## Findings\n\nNone.\n\n"
        "## Accepted\n\nNone.\n\n## Checked\n\n- the probe\n\n## Not checked\n\nNothing.\n")
    r = subprocess.run([sys.executable, "-I", "tools/tvr.py", "file", VEC, "r-vecadd"], cwd=root, capture_output=True, text=True)
    show("tvr file (second filing of round 2)", (r.stdout + r.stderr).strip()[:300])
    use_root(None)
    print()


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "g8":
    g8()
if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "g9":
    g9()

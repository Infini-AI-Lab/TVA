"""TVA library evidence: does the release check (`tvr release`, docs/REVIEW_PROTOCOL.md §2, library
issues L36, L7, L40, L41, L42) pass on what TVA ships, and fail on a dependency on anything outside what
TVA ships, Mathlib's pinned source, the Lean toolchain and the system?

Usage, from the TVA root:
  python3 -I evidence/release-check/run.py SCRATCH [R0 R1 R2 R3] > out

  SCRATCH  an empty scratch directory outside TVA and outside the home directory, with about 16 GB
           free per run (each run builds Mathlib from source in its sandbox, about 25 minutes).

Runs (all four by default), one at a time, each in its own empty sandbox under SCRATCH, deleted
after the run. After each run the script prints its time and a digest of the packages' build outputs
made in the sandbox (the `.olean` files Lean loads, and all build files), to show whether two builds
of Mathlib from the same pinned source give the same files.
  R0  the TVA tree as it is.
  R1  a copy of the tree (with its git metadata) with four probes:
      (a) `tvr audit` first reads a file in a fresh folder of the system's second temporary directory,
          at a path computed at run time (lr-6a's route: that directory was visible in round 6);
      (b) `tvr audit` first puts the site-packages folder of the Python that runs this script on
          `sys.path` and imports a module that only that Python has (lr-6a: `-I -S` holds only until
          the code extends `sys.path`);
      (c) a note that names a sibling of TVA, force-added under `docs/.lake/` (L41);
      (d) `kernels/vec_add_f32` brought up to date and gated in the copy, then a comment added to
          `lean/TVA/Lang/Ops.lean`, so the shipped gate.json is stale (L42).
      Outside the sandbox, before (c), the copy's audit passes with the probes in place.
  R2  a copy of the tree whose Lean packages are clones of the tree's, in which Mathlib's checkout holds
      a module file that its .gitignore ignores (`Mathlib/__pycache__/Lemmas.lean`, lr-6a's route), and
      `lean/TVA.lean` imports it. Outside the sandbox, TVA's import check accepts the import and the
      round-6 `packages` test (`git status --porcelain --untracked-files=all`) sees nothing.
  R3  a copy of the tree with `kernels/vec_add_f32` brought up to date and gated in the copy, so that
      its shipped gate.json is current (the positive control for R1 (d)).
The script prints each run's steps (PASS or FAIL and the first part of the detail), never a probe's path.
The tree is never written (except by redirecting the output).
"""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time

TVA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable
SIBLING = "fpv" + "/" + "sibling" + "/notes"  # a path into a sibling of TVA, built at run time
spec = importlib.util.spec_from_file_location("gate_digests", os.path.join(TVA, "evidence", "gate-digests", "run.py"))
gd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gd)


def tvr(root, *args):
    return subprocess.run([PY, "-I", "tools/tvr.py", *args], cwd=root, capture_output=True, text=True)


def copy(dst, build=True):
    """The tree with its git metadata; Mathlib's packages linked and the Lean build copied (as in
    evidence/gate-digests), or, with build=False, neither."""
    if build:
        gd.copy_tree(dst)
    else:
        shutil.copytree(TVA, dst, symlinks=True, ignore=shutil.ignore_patterns(".venv", ".lake", "__pycache__", ".git"))
    shutil.copytree(os.path.join(TVA, ".git"), os.path.join(dst, ".git"), symlinks=True)


def clean(detail):
    """A step's detail without paths or lists of path components (a Python traceback holds both)."""
    if "Traceback" in detail:
        errs = re.findall(r"\b([A-Z]\w*Error)\b", detail)
        return f"the tool stopped with {errs[-1] if errs else 'an exception'}"
    detail = re.sub(r"\[[^\]]*\]", "[…]", detail)
    return re.sub(r"(?<![\w.])/[^\s'\"|:,;)]+", "<path>", detail)


def build_digest(d):
    """The packages' build outputs in a sandbox `d`: how many `.olean` files (what Lean loads), and the
    sha256 of their relative paths and sha256s, in order; and the same over every build file."""
    out = {"olean": hashlib.sha256(), "all": hashlib.sha256()}
    n = {"olean": 0, "all": 0}
    home = os.path.join(d, "home")
    tops = sorted(dp for dp, dn, _ in os.walk(home) if dp.count(os.sep) - home.count(os.sep) <= 3 and "lean" in dn)
    for top in tops:  # the copy is at TVA's path inside the sandbox's home (`fpv/tva`, or a copy's name)
        pk = os.path.join(top, "lean", ".lake", "packages")
        for dirpath, dirnames, names in sorted(os.walk(pk)):
            dirnames.sort()
            if os.sep + ".lake" + os.sep + "build" not in dirpath + os.sep:
                continue
            for f in sorted(names):
                full = os.path.join(dirpath, f)
                h = hashlib.sha256(open(full, "rb").read()).hexdigest()
                line = (os.path.relpath(full, pk) + " " + h + "\n").encode()
                for kind in ("olean", "all") if ".olean" in f else ("all",):
                    out[kind].update(line)
                    n[kind] += 1
    return {k: (n[k], out[k].hexdigest()[:16]) for k in out}


def release(root, d):
    t0 = time.time()
    r = tvr(root, "release", "--dir", d, "--keep")  # kept for build_digest, then removed by main
    took = time.time() - t0
    try:
        rep = json.load(open(os.path.join(d, "release.json")))
    except OSError:
        print(f"  release check did not run: {clean(r.stdout + r.stderr)[:300]}")
        return None
    print(f"  release check: {'PASS' if rep['ok'] else 'FAIL'} ({rep['files']} shipped files, "
          f"{'allowlist sandbox (bwrap)' if rep['masked'] else 'HOME only'}, the tool on {clean(str(rep.get('python')))})")
    for s in rep["steps"]:
        print(f"    {'PASS' if s['ok'] else 'FAIL'}  {s['step']}: {clean(s['detail'])[:260]}")
    bd = build_digest(d)
    print(f"  {took / 60:.0f} minutes; the packages' build in the sandbox: {bd['olean'][0]} .olean files, digest "
          f"{bd['olean'][1]}; {bd['all'][0]} files in all, digest {bd['all'][1]}")
    return rep


def r0(scratch):
    print("## R0: the TVA tree as it is")
    release(TVA, os.path.join(scratch, "r0"))


def user_module():
    """A top-level package of this Python's site-packages that the system's Python cannot import."""
    site = sysconfig.get_paths()["purelib"]
    sysp = next((os.path.join("/usr/bin", n) for n in sorted(os.listdir("/usr/bin"), reverse=True)
                 if re.fullmatch(r"python3\.\d+", n)), "/usr/bin/python3")
    for n in sorted(os.listdir(site)):
        if n.isidentifier() and os.path.isfile(os.path.join(site, n, "__init__.py")) and \
                subprocess.run([sysp, "-I", "-S", "-c", f"import {n}"], capture_output=True).returncode != 0:
            return site, n
    return site, None


def r1(scratch):
    print("## R1: four probes (a file outside TVA, a module of the user's Python, a note in docs/.lake, a stale gate)")
    c = os.path.join(scratch, "copy1")
    copy(c)
    outside = tempfile.mkdtemp(prefix="tva-outside-", dir=os.sep + os.path.join("var", "tmp"))
    try:
        open(os.path.join(outside, "data.txt"), "w").write("probe\n")
        site, mod = user_module()
        pa = [p for p in outside.split(os.sep) if p]
        pb = [p for p in site.split(os.sep) if p]
        t = os.path.join(c, "tools", "tvr.py")
        s = open(t).read()
        anchor = "    if len(argv) >= 2 and argv[1] == \"audit\":\n"
        assert anchor in s
        probe = (anchor +
                 "        _probe = []\n"
                 "        try:\n"
                 f"            open(os.sep + os.sep.join({pa!r}) + os.sep + \"data.txt\").read()\n"
                 "        except OSError:\n"
                 "            _probe.append(\"(a) a file outside TVA\")\n"
                 "        try:\n"
                 f"            sys.path.insert(0, os.sep + os.sep.join({pb!r}))\n"
                 f"            import {mod}  # noqa\n"
                 "        except ImportError:\n"
                 "            _probe.append(\"(b) a module of the user's Python\")\n"
                 "        if _probe:\n"
                 "            print(\"probe failed: \" + \"; \".join(_probe))\n"
                 "            return 1\n")
        open(t, "w").write(s.replace(anchor, probe, 1))
        print(f"  probes (a) and (b) in the copy's audit ((b) imports `{mod}`); outside the sandbox: tvr audit "
              f"{'PASS' if tvr(c, 'audit').returncode == 0 else 'FAIL'}")
        os.makedirs(os.path.join(c, "docs", ".lake"))
        open(os.path.join(c, "docs", ".lake", "note.md"), "w").write(f"see ~/{SIBLING}\n")
        subprocess.run(["git", "-C", c, "add", "-f", "docs/.lake/note.md"], capture_output=True)
        print("  (c) docs/.lake/note.md force-added")
        gd.update_kernel(c)
        g = tvr(c, "check", "vec_add_f32")
        print(f"  (d) kernels/vec_add_f32 updated and gated in the copy: {g.stdout.strip().splitlines()[-1]}")
        open(os.path.join(c, "lean", "TVA", "Lang", "Ops.lean"), "a").write("\n-- a comment added after the gate\n")
        print("      then a comment added to lean/TVA/Lang/Ops.lean")
        release(c, os.path.join(scratch, "r1"))
    finally:
        shutil.rmtree(outside, ignore_errors=True)


def r2(scratch):
    print("## R2: Mathlib's checkout holds a module file its .gitignore ignores, and TVA imports it")
    c = os.path.join(scratch, "copy2")
    copy(c, build=False)
    man = json.load(open(os.path.join(TVA, "lean", "lake-manifest.json")))
    for p in man["packages"]:
        src = os.path.join(TVA, "lean", ".lake", "packages", p["name"])
        dst = os.path.join(c, "lean", ".lake", "packages", p["name"])
        subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", src, dst], check=True, capture_output=True)
        subprocess.run(["git", "-C", dst, "-c", "advice.detachedHead=false", "checkout", "-q", p["rev"]], check=True,
                       capture_output=True)
    ml = os.path.join(c, "lean", ".lake", "packages", "mathlib")
    os.makedirs(os.path.join(ml, "Mathlib", "__pycache__"))
    open(os.path.join(ml, "Mathlib", "__pycache__", "Lemmas.lean"), "w").write(
        "/-- a lemma from a file a fresh checkout of Mathlib does not have -/\ntheorem probeLemma : (2 : Nat) + 2 = 4 := rfl\n")
    root_mod = os.path.join(c, "lean", "TVA.lean")
    s = open(root_mod).read()
    open(root_mod, "w").write(s.replace("import TVA.Math\n", "import TVA.Math\nimport Mathlib.__pycache__.Lemmas\n", 1))
    ign = subprocess.run(["git", "-C", ml, "check-ignore", "-q", "Mathlib/__pycache__/Lemmas.lean"]).returncode == 0
    st = subprocess.run(["git", "-C", ml, "status", "--porcelain", "--untracked-files=all"], capture_output=True, text=True)
    spec2 = importlib.util.spec_from_file_location("tvr_copy", os.path.join(c, "tools", "tvr.py"))
    tc = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(tc)
    print(f"  outside the sandbox: the file is ignored by Mathlib's .gitignore: {ign}; the round-6 packages test sees "
          f"{len(st.stdout.splitlines())} line(s); TVA's import check: {tc.import_violations([root_mod], tc.LEAN_ROOTS) or 'no problem'}; "
          f"tvr audit {'PASS' if tvr(c, 'audit').returncode == 0 else 'FAIL'}")
    release(c, os.path.join(scratch, "r2"))


def r3(scratch):
    print("## R3: kernels/vec_add_f32 brought up to date and gated in the copy (its gate.json current)")
    c = os.path.join(scratch, "copy3")
    copy(c)
    gd.update_kernel(c)
    g = tvr(c, "check", "vec_add_f32")
    print(f"  outside the sandbox: {g.stdout.strip().splitlines()[-1]}")
    release(c, os.path.join(scratch, "r3"))


def main():
    scratch = os.path.realpath(sys.argv[1])
    runs = sys.argv[2:] or ["R0", "R1", "R2", "R3"]
    os.makedirs(scratch, exist_ok=True)
    for r in runs:
        {"R0": r0, "R1": r1, "R2": r2, "R3": r3}[r](scratch)
        for d in ("r0", "r1", "r2", "r3", "copy1", "copy2", "copy3"):
            shutil.rmtree(os.path.join(scratch, d), ignore_errors=True)  # each run needs about 16 GB
        print()


if __name__ == "__main__":
    main()

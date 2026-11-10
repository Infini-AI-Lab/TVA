"""L191 (round 42, author): the gate refuses a kernel whose module's root is not its own.

Usage: python3 -I same_root.py <tva> <empty folder outside TVA>. As lr-41a's work/same_root.py (round 41)
did, it copies what TVA ships and the local kernel vec_add_f32 into the folder (lean/.lake/packages linked,
lean/.lake/build copied), and runs `tvr check` there only, never in TVA's own kernels/:
  A. vec_add_aa: vec_add_f32's folder, records removed, a theorem that does not elaborate appended to its
     Lean file, same module `VecAdd`: both vec_add_aa and vec_add_f32 now fail at `manifest`;
  B. vec_add_aa with its module renamed `VecAddAa` (the file renamed too): `manifest` passes and `build`
     compiles its own file, so it fails (the appended theorem);
  C. modules `Aesop.VecAdd`, `std.VecAdd` and `TVA.VecAdd`: refused at `manifest` (a pinned package's root,
     the toolchain's `Std` ignoring case, TVA's);
  D. vec_add_aa removed: vec_add_f32 passes again.
It prints the folder as <dir> and the home directory as ~."""
import json
import os
import shutil
import subprocess
import sys

tva, d = os.path.realpath(sys.argv[1]), os.path.realpath(sys.argv[2])
assert not os.listdir(d), "the folder must be empty"
names = subprocess.run(["git", "-C", tva, "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                       capture_output=True, check=True).stdout.decode().split("\0")
for n in filter(None, names):
    src, dst = os.path.join(tva, n), os.path.join(d, n)
    if os.path.isfile(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
shutil.copytree(os.path.join(tva, "kernels", "vec_add_f32"), os.path.join(d, "kernels", "vec_add_f32"))
os.makedirs(os.path.join(d, "lean", ".lake"))
os.symlink(os.path.join(tva, "lean", ".lake", "packages"), os.path.join(d, "lean", ".lake", "packages"))
shutil.copytree(os.path.join(tva, "lean", ".lake", "build"), os.path.join(d, "lean", ".lake", "build"))
k = os.path.join(d, "kernels", "vec_add_aa")
shutil.copytree(os.path.join(d, "kernels", "vec_add_f32"), k)
shutil.rmtree(os.path.join(k, "review"))
os.makedirs(os.path.join(k, "review"))
for f in ("gate.json", "claim.md", "REVIEW.md"):
    if os.path.exists(os.path.join(k, f)):
        os.remove(os.path.join(k, f))
with open(os.path.join(k, "lean", "VecAdd.lean"), "a") as f:
    f.write("\ntheorem Kernels.VecAdd.broken : (1 : Nat) = 2 := rfl\n")
home = os.path.expanduser("~")


def show(title, cmd, keep=("manifest", "build", "gate")):
    r = subprocess.run(cmd, cwd=d, capture_output=True, text=True)
    out = [l for l in (r.stdout + r.stderr).splitlines() if l.split(" ", 1)[0] in ("PASS", "FAIL", "gate")
           and any(w in l for w in keep)]
    print(f"## {title}\n$ {' '.join(cmd[2:])}  (exit {r.returncode})")
    print("\n".join(out).replace(d, "<dir>").replace(home, "~")[:3000] + "\n")


def set_module(mod):
    p = os.path.join(k, "kernel.json")
    m = json.load(open(p))
    m["lean"]["module"] = mod
    json.dump(m, open(p, "w"), indent=2)


tvr = [sys.executable, "-I", "tools/tvr.py"]
show("A1: vec_add_aa, module VecAdd (vec_add_f32's root)", tvr + ["check", "vec_add_aa"])
show("A2: vec_add_f32, while vec_add_aa shares its root", tvr + ["check", "vec_add_f32"])
os.rename(os.path.join(k, "lean", "VecAdd.lean"), os.path.join(k, "lean", "VecAddAa.lean"))
set_module("VecAddAa")
show("B: vec_add_aa, module VecAddAa (its own root), the appended theorem", tvr + ["check", "vec_add_aa"])
for mod in ("Aesop.VecAdd", "std.VecAdd", "TVA.VecAdd"):
    set_module(mod)
    show(f"C: vec_add_aa, module {mod}", tvr + ["check", "vec_add_aa"])
shutil.rmtree(k)
show("D: vec_add_f32 alone", tvr + ["check", "vec_add_f32"])

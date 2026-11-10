"""NEW (gate): two local kernels whose modules share a root. `tvr check` builds with
lean/.lake/tvr/lakefile.toml, which declares one lean_lib per kernel folder; when two declare the same root
module, `lake build <module>` builds one file only (the library listed last), and the claim reader then
reads that build. So the other kernel's gate never compiles its own Lean.

Usage: python3 -I same_root.py <tva> <empty folder>. It copies what TVA ships and the local kernel
vec_add_f32 into the folder (lean/.lake/packages linked, lean/.lake/build copied, so nothing is rebuilt but
the kernels), makes vec_add_aa: vec_add_f32's folder with its records removed and one line appended to its
Lean file, a theorem that does not elaborate (`(1 : Nat) = 2 := rfl`), and runs `tvr check vec_add_aa`.
It prints the folder as <dir>."""
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


def show(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print(f"$ {' '.join(cmd[1:] if cmd[0] == sys.executable else cmd)}  (exit {r.returncode})".replace(os.path.expanduser("~"), "~"))
    print((r.stdout + r.stderr).replace(d, "<dir>").replace(os.path.expanduser("~"), "~").strip()[:4000] + "\n")


show([sys.executable, "-I", "tools/tvr.py", "check", "vec_add_aa"], d)
lake = os.path.join(os.path.expanduser("~"), ".elan", "bin", "lake")
# the same file, compiled on its own: it does not elaborate
show([lake, "env", "lean", os.path.join("..", "kernels", "vec_add_aa", "lean", "VecAdd.lean")], os.path.join(d, "lean"))
with open(os.path.join(d, "lean", ".lake", "tvr", "lakefile.toml")) as f:
    print("lean/.lake/tvr/lakefile.toml, its kernel libraries:\n" + f.read().split('name = "TVA"', 1)[1].strip())

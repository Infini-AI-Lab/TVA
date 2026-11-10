"""Round 41 (L186, L187): `tvr audit` on a copy of what TVA ships, without git metadata and in a fresh
git checkout, and on two planted defects there. Usage: python3 -I audit_copy.py <tva> <empty folder>.
Prints each case's exit code and output, with the copy's folder written as <dir>."""
import os
import subprocess
import sys

tva, d = os.path.realpath(sys.argv[1]), os.path.realpath(sys.argv[2])
assert not os.listdir(d), "the folder must be empty"
names = subprocess.run(["git", "-C", tva, "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                       capture_output=True, check=True).stdout.decode().split("\0")
for n in filter(None, names):
    src, dst = os.path.join(tva, n), os.path.join(d, n)
    if not os.path.lexists(src):
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(src):
        os.symlink(os.readlink(src), dst)
    else:
        with open(src, "rb") as f, open(dst, "wb") as g:
            g.write(f.read())


def audit(case):
    r = subprocess.run([sys.executable, "-I", "tools/tvr.py", "audit"], cwd=d, capture_output=True, text=True)
    out = (r.stdout + r.stderr).replace(d, "<dir>").strip()
    print(f"== {case}: exit {r.returncode}\n{out}\n")


audit("C1 the shipped files, no .git (a release copy)")
subprocess.run(["git", "init", "-q"], cwd=d, check=True)
subprocess.run(["git", "add", "-A"], cwd=d, check=True)
audit("C2 the same, a fresh git checkout")
os.makedirs(os.path.join(d, "kernels"), exist_ok=True)
with open(os.path.join(d, "kernels", "README.md"), "w") as f:
    f.write("local index\n")
with open(os.path.join(d, "docs", "LIBRARY_REVIEW.md"), "a") as f:
    f.write("\n[index](../kernels/README.md)\n")
audit("C3 a git-ignored kernels/README.md that exists, cited from a shipped page")
with open(os.path.join(d, "lean", "lakefile.toml"), "a") as f:
    f.write('\n[[lean_lib]]\nname = "x"\nsrcDir = "../kernels/x/lean"\nroots = ["X"]\n')
audit("C4 and a kernel library in the shipped lakefile")

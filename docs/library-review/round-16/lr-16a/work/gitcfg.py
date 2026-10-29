"""lr-16a, L7 end to end: the release check's packages step (`tvr.make_packages`, the step that runs git on the
network) run from inside a git repository whose own configuration redirects every GitHub URL to another host and
names a proxy that does not exist, with that repository holding the caller's folder, the temporary folder
(`TMPDIR`, where the tool makes git's empty home) and the folders git writes (the packages, the scratch of the
ref lookups). If any git the tool runs read that repository's configuration, the lookup of Mathlib's tag or a
fetch would go to the other host or the proxy and fail. Also `shipped_files` (git on TVA's own repository) from
there, and a plain git inside the repository as the control.

Usage: python3 -I gitcfg.py SCRATCH > gitcfg.out   (SCRATCH an empty folder outside TVA; it needs the network
and about 2 GB; its path is never printed)
"""
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))))
SCRATCH = os.path.realpath(sys.argv[1])
assert os.path.isdir(SCRATCH) and not os.listdir(SCRATCH), "SCRATCH must be an empty folder"
HOME = os.path.realpath(os.path.expanduser("~"))


def mask(text):
    return str(text).replace(SCRATCH, "<scratch>").replace(TVA, "<tva>").replace(HOME, "~")


def show(label, value):
    print(f"- {label}: {mask(value)}", flush=True)


repo = os.path.join(SCRATCH, "repo")
env = {"PATH": "/usr/bin:/bin", "HOME": SCRATCH, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
subprocess.run(["/usr/bin/git", "init", "-q", repo], check=True, env=env)
for k, v in (("url.https://example.invalid/.insteadOf", "https://github.com/"),
             ("http.proxy", "http://127.0.0.1:9"), ("https.proxy", "http://127.0.0.1:9")):
    subprocess.run(["/usr/bin/git", "-C", repo, "config", k, v], check=True, env=env)
for d in ("cwd/deeper", "tmp", "lean", "scratch"):
    os.makedirs(os.path.join(repo, d))
show("the repository's .git/config", open(os.path.join(repo, ".git", "config")).read().strip().replace("\n", " | "))
r = subprocess.run(["/usr/bin/git", "ls-remote", "https://github.com/leanprover-community/mathlib4", "refs/tags/v4.35.0"],
                   cwd=os.path.join(repo, "cwd", "deeper"), capture_output=True, text=True, env=env)
show("control: a plain git ls-remote of Mathlib's tag, run in a subfolder of it", (r.returncode, (r.stdout or r.stderr).strip()[:160]))

# The tool, with its temporary folder inside the repository and run from a subfolder of it.
os.environ["TMPDIR"] = os.path.join(repo, "tmp")
os.chdir(os.path.join(repo, "cwd", "deeper"))
sys.path.insert(0, os.path.join(TVA, "tools"))
import tvr  # noqa: E402

show("git's empty home (tvr._git_home) lies inside the repository", tvr._git_home().startswith(repo))
show("shipped_files (git -C <tva> ls-files) from there: files listed", len(tvr.shipped_files()))
t = time.time()
bad = tvr.make_packages(os.path.join(repo, "lean"), os.path.join(repo, "scratch"))
show("make_packages (the release check's packages step) into a folder of the repository", bad or "no problem: every package made "
     "afresh from its upstream repository, Mathlib's pin its tag's commit, every file its commit's blob")
show("packages made", sorted(os.listdir(os.path.join(repo, "lean", ".lake", "packages"))))
show("minutes", round((time.time() - t) / 60, 1))

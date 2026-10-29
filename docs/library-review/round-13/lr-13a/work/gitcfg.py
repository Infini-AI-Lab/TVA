"""lr-13a's probe of the release check's git (library issue L7): does the configuration of the git
repository the tool is run from decide where the upstream refs come from?

Usage: python3 -I docs/library-review/round-13/lr-13a/work/gitcfg.py SCRATCH > .../work/gitcfg.out
  SCRATCH  an empty folder outside TVA; its path is never printed. Needs the network.

`tvr release` is run from the TVA root, which is a git repository. The tool runs `/usr/bin/git` with no
user or system configuration (`_git_env`), but `git ls-remote` and `git clone`, which `upstream_problem`
runs without `-C`, read the configuration of the repository around the current folder (its `.git/config`).
Here a repository whose `.git/config` holds a `url.<base>.insteadOf` for github.com stands for TVA's: the
probe runs the tool's own `_git` and `upstream_problem` from inside it, and from a folder outside any
repository, on Mathlib's URL and pin. The tree is never written.
"""
import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))))
sys.path.insert(0, os.path.join(TVA, "tools"))
import tvr  # noqa: E402

SCRATCH = os.path.realpath(sys.argv[1])
assert os.path.isdir(SCRATCH) and not os.listdir(SCRATCH), "SCRATCH must be an empty folder"


def mask(text):
    return str(text).replace(SCRATCH, "<scratch>").replace(TVA, "<tva>").replace(os.path.expanduser("~"), "~")


def show(label, value):
    print(f"- {label}: {mask(value)}")


man = json.load(open(os.path.join(TVA, "lean", "lake-manifest.json")))
url, rev = man["packages"][0]["url"], man["packages"][0]["rev"]
plain = os.path.join(SCRATCH, "plain")
repo = os.path.join(SCRATCH, "repo")
os.makedirs(plain)
subprocess.run(["/usr/bin/git", "init", "-q", repo], check=True, env=tvr._git_env())
# the repository's own configuration: github.com is read from another host
subprocess.run(["/usr/bin/git", "-C", repo, "config", "url.https://example.invalid/.insteadOf", "https://github.com/"],
               check=True, env=tvr._git_env())
show("the repository's .git/config", open(os.path.join(repo, ".git", "config")).read().strip().replace("\n", " | "))
for where, cwd in (("outside any repository", plain), ("inside the repository", repo)):
    os.chdir(cwd)
    r = tvr._git("ls-remote", url, "refs/tags/v4.35.0")
    show(f"{where}: _git ls-remote {url} refs/tags/v4.35.0",
         (r.returncode, (r.stdout.decode() or r.stderr.decode()).strip()[:160]))
    os.makedirs(os.path.join(SCRATCH, "up-" + where[:6]))
    show(f"{where}: upstream_problem(Mathlib's URL, v4.35.0, its pin, tag)",
         tvr.upstream_problem(url, "v4.35.0", rev, os.path.join(SCRATCH, "up-" + where[:6]), "tag") or "accepted")

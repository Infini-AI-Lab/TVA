"""Does `make_packages` (tools/tvr.py, the release check's `packages` step, L7) give a package exactly its
pinned commit's files whatever the user's git configuration says? The step runs git outside the sandbox,
with the user's HOME, so git reads the user's global configuration (the `.gitconfig` of the home directory). This probe does not
touch the real one: it runs `tvr.make_packages` with HOME set to a scratch folder whose `.gitconfig`
names a global attributes file that gives `*.lean` a smudge filter (the mechanism git-lfs uses), and
with the manifest narrowed to one small package (`Cli`). Then it compares the checked-out files with the
commit's blobs (`git hash-object --no-filters` against `git ls-tree`).

Usage: python3 -I docs/library-review/round-7/lr-7b/work/git_config_probe.py <empty scratch folder outside the home directory>
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.abspath(os.path.join(HERE, *[os.pardir] * 5))
sys.path.insert(0, os.path.join(TVA, "tools"))
import tvr  # noqa: E402


def main():
    d = os.path.realpath(sys.argv[1])
    assert not os.listdir(d)
    real_load = tvr.load_json
    man = real_load(os.path.join(tvr.ROOT, "lean", "lake-manifest.json"))
    one = {**man, "packages": [p for p in man["packages"] if p["name"] == "Cli"]}
    tvr.load_json = lambda path, default=None: one if path.endswith("lake-manifest.json") else real_load(path, default)
    for case in ("plain", "smudge"):
        home = os.path.join(d, case, "home")
        lean = os.path.join(d, case, "lean")
        os.makedirs(home)
        os.makedirs(lean)
        if case == "smudge":
            with open(os.path.join(home, ".gitattributes-global"), "w") as f:
                f.write("*.lean filter=probe\n")
            script = os.path.join(home, "smudge.sh")
            with open(script, "w") as f:
                f.write("#!/bin/sh\necho '-- a line the commit does not have'\ncat\n")
            os.chmod(script, 0o755)
            with open(os.path.join(home, ".gitconfig"), "w") as f:
                f.write("[core]\n\tattributesFile = " + os.path.join(home, ".gitattributes-global") + "\n"
                        "[filter \"probe\"]\n\tsmudge = " + script + "\n\tclean = cat\n")
        os.environ["HOME"] = home
        bad = tvr.make_packages(lean)
        pkg = os.path.join(lean, ".lake", "packages", "Cli")
        tree = subprocess.run(["git", "-C", pkg, "ls-tree", "-r", "HEAD"], capture_output=True, text=True).stdout
        differ = []
        for line in tree.splitlines():
            meta, path = line.split("\t", 1)
            blob = meta.split()[2]
            if meta.split()[1] != "blob":
                continue
            got = subprocess.run(["git", "hash-object", "--no-filters", os.path.join(pkg, path)],
                                 capture_output=True, text=True).stdout.strip()
            if got != blob:
                differ.append(path)
        status = subprocess.run(["git", "-C", pkg, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        lean_files = [l.split("\t", 1)[1] for l in tree.splitlines() if l.endswith(".lean")]
        first = open(os.path.join(pkg, lean_files[0])).readline().rstrip("\n") if lean_files else ""
        print(f"{case}: make_packages problems: {bad or 'none'}; files that differ from the commit's blobs: "
              f"{len(differ)} of {len(tree.splitlines())}; `git status` clean: {not status}; "
              f"first line of {lean_files[0] if lean_files else '-'}: {first!r}")
    shutil.rmtree(os.path.join(d, "plain"))
    shutil.rmtree(os.path.join(d, "smudge"))


if __name__ == "__main__":
    main()

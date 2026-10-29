"""What the release check's sandbox shows (L7, L36): build the same bwrap command line as
`cmd_release` (tools/tvr.py), with the same allowlist, around an empty scratch home, and run a probe on
the system Python inside it. No Mathlib build, no copy of TVA: only the mounts are tested.

Usage: python3 -I docs/library-review/round-7/lr-7b/work/sandbox_probe.py <empty scratch folder outside the home directory>
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.abspath(os.path.join(HERE, *[os.pardir] * 5))
sys.path.insert(0, os.path.join(TVA, "tools"))
import tvr  # noqa: E402

PROBE = r'''
import os, socket, sys
def show(p):
    try:
        return sorted(os.listdir(p))
    except OSError as e:
        return f"absent ({e.__class__.__name__})"
home = os.environ.get("HOME")
print("root:", show("/"))
print("HOME:", home, show(home))
print("HOME/.elan:", show(os.path.join(home, ".elan")))
print("HOME/.elan/toolchains:", show(os.path.join(home, ".elan", "toolchains")))
for p in ("/var", "/opt", "/srv", "/mnt", "/run", "/sys", "/root", "/scratch", "/data", "/tmp", "/dev", "/dev/shm",
          os.path.join(home, ".triton"), os.path.join(home, ".cache"), os.path.join(home, ".config")):
    print(p, ":", show(p))
print("pids visible in /proc:", sorted(int(x) for x in os.listdir("/proc") if x.isdigit()))
print("environment:", sorted(os.environ))
print("sys.path:", sys.path)
print("sys.flags isolated, no_site:", sys.flags.isolated, sys.flags.no_site)
try:
    s = socket.create_connection(("1.1.1.1", 53), timeout=3); print("network: CONNECTED"); s.close()
except OSError as e:
    print("network: none (", e.__class__.__name__, ")")
# inside bwrap's user namespace root is unmapped (files show the overflow uid), so ownership is
# checked outside by the tool (`system_problems`); here, test that the system folders are read-only
for top in ("/usr", "/etc"):
    try:
        open(os.path.join(top, "lr7b-probe"), "w").close(); print(top, ": WRITABLE")
    except OSError as e:
        print(top, ": not writable (", e.__class__.__name__, ")")
try:
    open(os.path.join(home, ".elan", "settings.toml"), "a").close(); print("elan settings: WRITABLE")
except OSError as e:
    print("elan settings: read-only (", e.__class__.__name__, ")")
'''


def main():
    d = os.path.realpath(sys.argv[1])
    home = os.path.realpath(os.path.expanduser("~"))
    assert not d.startswith(home + os.sep) and not os.listdir(d)
    shome = os.path.join(d, "home")
    os.makedirs(shome)
    elan, tc = tvr.elan_parts(home)
    py = tvr.system_python()
    bwrap = shutil.which("bwrap")
    base = [bwrap, "--die-with-parent", "--unshare-net", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
            "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
    for x in tvr.SANDBOX_SYSTEM:
        p = "/" + x
        if os.path.islink(p):
            base += ["--symlink", os.readlink(p), p]
        elif os.path.isdir(p):
            base += ["--ro-bind", p, p]
    base += ["--bind", shome, home]
    for part in elan or []:
        base += ["--ro-bind", os.path.realpath(part), part]
    base += ["--clearenv", "--setenv", "HOME", home, "--setenv", "LANG", "C.UTF-8",
             "--setenv", "PATH", os.pathsep.join([os.path.join(home, ".elan", "bin"), "/usr/local/bin", "/usr/bin", "/bin"])]
    print(f"toolchain {tc}; system Python {py}; elan parts bound: {[os.path.relpath(p, home) for p in elan]}")
    print("toolchain folder is a link:", os.path.islink(elan[1]))
    r = subprocess.run(base + ["--chdir", home, py, "-I", "-S", "-c", PROBE], capture_output=True, text=True)
    out = (r.stdout + r.stderr).replace(home, "<home>").replace(d, "<scratch>")
    print(out)
    r = subprocess.run(base + ["--chdir", home, os.path.join(home, ".elan", "bin", "lean"), "--version"],
                       capture_output=True, text=True)
    print("lean --version inside:", (r.stdout + r.stderr).strip()[:200])
    shutil.rmtree(shome)


if __name__ == "__main__":
    main()

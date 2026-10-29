"""lr-20d, library review round 20 (survey area: rules).

Question: Compiler.Faithful's scope 2 says Triton's knobs are at their defaults, with no word on WHEN.
Scope 10 says "at any time before the launch ends"; scope 2 does not. The in-process cache key is the
specialisation and `str(options)` (`compute_cache_key`, `primitives/triton-3.8.0-jit.py:592-614`), and
`knobs.language.fp32_default` is read by the front end (`primitives/triton-3.8.0-core.py:2387`), not
through the options. Does a launch made while the knobs ARE at their defaults run code compiled earlier
in the same process while one was not?

Z1: one process sets `triton.knobs.language.fp32_default = "ieee"` (with `propagate_env` at its default
True, so the environment variable is set too), launches `tl.dot(a, I)` for 16x16 float32 blocks with
a = 1 + 2^-12, then sets the knob back to `None` and removes the environment variable (its default),
and launches again. Z2: the same with `propagate_env = False`. Control: a process that only launches
with the knob at its default. tf32 keeps 1.0, ieee 1 + 2^-12.

Run from the TVA root: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I <this file>
No path is printed.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

KSRC = '''import triton
import triton.language as tl


@triton.jit
def dotk(a_ptr, b_ptr, o_ptr):
    i = tl.arange(0, 16)
    a = tl.load(a_ptr + i[:, None] * 16 + i[None, :])
    b = tl.load(b_ptr + i[:, None] * 16 + i[None, :])
    tl.store(o_ptr + i[:, None] * 16 + i[None, :], tl.dot(a, b))
'''


def child(mode, kdir):
    import torch
    import triton
    sys.path.insert(0, kdir)
    import kmod_lr20d_knob as kmod
    a = torch.full((16, 16), 1.0 + 2.0 ** -12, device="cuda")
    b = torch.eye(16, device="cuda")

    def launch():
        o = torch.zeros(16, 16, device="cuda")
        kmod.dotk[(1,)](a, b, o)
        return o[0, 0].item().hex()

    out = []
    if mode in ("z1", "z2"):
        if mode == "z2":
            triton.knobs.propagate_env = False
        triton.knobs.language.fp32_default = "ieee"
        out.append(launch())
        triton.knobs.language.fp32_default = None
        os.environ.pop("TRITON_F32_DEFAULT", None)
        out.append(f"knob now {triton.knobs.language.fp32_default!r}, "
                   f"variable set: {'TRITON_F32_DEFAULT' in os.environ}")
    out.append(launch())
    return out


def run(mode, kdir):
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRITON_")}
    env["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="c-", dir=kdir)
    p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "child", mode, kdir],
                       env=env, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return "ERROR " + (p.stderr.strip().splitlines() or ["?"])[-1][:120]


def main():
    import torch
    import triton
    kdir = tempfile.mkdtemp(prefix="lr20d-knob-")
    with open(os.path.join(kdir, "kmod_lr20d_knob.py"), "w") as fh:
        fh.write(KSRC)
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"python {sys.version.split()[0]}")
    ctl = run("plain", kdir)
    print(f"control (knob at its default throughout): {ctl}")
    for mode, label in [("z1", "Z1 propagate_env True"), ("z2", "Z2 propagate_env False")]:
        r = run(mode, kdir)
        print(f"{label}: launch with the knob 'ieee' -> {r[0]}; {r[1]}; then a launch at the default -> {r[2]}"
              f"{' (the ieee code: DIFFERS from the control)' if r[2] != ctl[0] else ''}")
    shutil.rmtree(kdir, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        print("RESULT " + json.dumps(child(sys.argv[2], sys.argv[3])))
    else:
        main()

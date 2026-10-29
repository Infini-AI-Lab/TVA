"""The contiguity bound of a value built from a value that carries a `max_contiguous` hint.

Library evidence for library issue L58 (round 15). The analysis replaces the contiguity it infers for
a value whose defining operation carries a `tt.contiguity` attribute (a `tl.max_contiguous`, alone or
in the pair with `tl.multiple_of`) by the hinted value (`primitives/triton-3.8.0-AxisInfo.cpp`
1244-1259), and an operation on that value starts from it: `x + s` with `s` constant gets
`gcd(contiguity(x), constancy(s))` (242-279), a conversion or an expand keeps it (138-147, 607-668).
So `tl.arange(0, B) + s` has contiguity `B` only when the arange carries no contiguity hint; with
`tl.max_contiguous(tl.multiple_of(tl.arange(0, B), v), C)` it has `C`.

The loads below read `int8` elements through `x_ptr + y`, with `y = tl.multiple_of(a + s, 16)` and
one warp. Such a load's vector is at most `min(contiguity, divisibility)` bytes wide, so the PTX
shows the contiguity the analysis gave `y`: `ld.global.v4.b32` (16 bytes) for a contiguity of at
least 16, `ld.global.b32` (4 bytes) for 4.

  G1  a = tl.arange(0, 512)                                            (no hint on a)
  G2  a = tl.max_contiguous(tl.multiple_of(tl.arange(0, 512), 4), 4)   (the pair, C = 4 on a)
  G3  a = tl.max_contiguous(tl.multiple_of(tl.arange(0, 512), 8), 8)   (the pair, C = 8 on a)
  G4  a = tl.multiple_of(tl.arange(0, 512), 512)                       (multiple_of alone on a)
  G5  a = tl.max_contiguous(tl.multiple_of(tl.arange(0, 512), 4), 4).to(tl.int64)  (then converted)
  G6  a = tl.max_contiguous(tl.multiple_of(tl.arange(0, 512), 4), 4); y = s + a  (s on the left)

Each case runs in its own process with its own Triton cache, on GPU 0, with s = 16, and checks the
loaded values against the buffer. Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I
evidence/hint-hinted-bound/run.py (from TVA's root).
"""
import json, os, re, subprocess, sys, tempfile

import torch
import triton
import triton.language as tl


@triton.jit
def k(x_ptr, out_ptr, s, CASE: tl.constexpr):
    if CASE == 1:
        a = tl.arange(0, 512)
    elif CASE == 2:
        a = tl.max_contiguous(tl.multiple_of(tl.arange(0, 512), 4), 4)
    elif CASE == 3:
        a = tl.max_contiguous(tl.multiple_of(tl.arange(0, 512), 8), 8)
    elif CASE == 4:
        a = tl.multiple_of(tl.arange(0, 512), 512)
    elif CASE == 5:
        a = tl.max_contiguous(tl.multiple_of(tl.arange(0, 512), 4), 4).to(tl.int64)
    else:
        a = tl.max_contiguous(tl.multiple_of(tl.arange(0, 512), 4), 4)
    if CASE == 6:
        y = s + a
    else:
        y = a + s
    y = tl.multiple_of(y, 16)
    tl.store(out_ptr + tl.arange(0, 512), tl.load(x_ptr + y))


CASES = {
    "G1": "a = tl.arange(0, 512); y = tl.multiple_of(a + s, 16)",
    "G2": "a = tl.max_contiguous(tl.multiple_of(tl.arange(0, 512), 4), 4); y = tl.multiple_of(a + s, 16)",
    "G3": "a = tl.max_contiguous(tl.multiple_of(tl.arange(0, 512), 8), 8); y = tl.multiple_of(a + s, 16)",
    "G4": "a = tl.multiple_of(tl.arange(0, 512), 512); y = tl.multiple_of(a + s, 16)",
    "G5": "a = tl.max_contiguous(tl.multiple_of(tl.arange(0, 512), 4), 4).to(tl.int64); y = tl.multiple_of(a + s, 16)",
    "G6": "a = tl.max_contiguous(tl.multiple_of(tl.arange(0, 512), 4), 4); y = tl.multiple_of(s + a, 16)",
}


def run_case(name):
    x = (torch.arange(1024, device="cuda") % 127).to(torch.int8)
    out = torch.zeros(512, device="cuda", dtype=torch.int8)
    s = 16
    ck = k[(1,)](x, out, s, CASE=int(name[1]), num_warps=1)
    torch.cuda.synchronize()
    loads = sorted(set(re.findall(r"ld\.global(?:\.[a-z0-9]+)+", ck.asm["ptx"])))
    ok = torch.equal(out.cpu(), x.cpu()[s:s + 512])
    return {"loads": loads, "result": "values correct" if ok else "values WRONG"}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        try:
            res = run_case(sys.argv[2])
        except Exception as e:
            msg = [l for l in str(e).strip().splitlines() if l.strip()]
            res = {"loads": [], "result": "error: " + type(e).__name__ + ": " + (msg[-1] if msg else "")[:150]}
        print("RESULT " + json.dumps(res))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name, desc in CASES.items():
        with tempfile.TemporaryDirectory() as cache:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", TRITON_CACHE_DIR=cache)
            p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                               capture_output=True, text=True, env=env)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        res = json.loads(line[len("RESULT "):]) if line else {"loads": [], "result": "process failed"}
        print(f"{name}: {desc}\n  loads {res['loads']}; {res['result']}")


if __name__ == "__main__":
    main()

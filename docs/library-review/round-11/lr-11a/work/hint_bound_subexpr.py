"""lr-11a: is "the contiguity the analysis infers" (SEMANTICS, Compiler hints, L58) a lower bound when
a sub-expression carries a contiguity hint?

The library's bounds: `tl.arange(a, b)` has `b - a`, and `x + s` with a scalar `s` keeps `x`'s bound.
`tl.max_contiguous(tl.multiple_of(x, v), C)` (the pair) may give `C` below `x`'s own contiguity, and the
hint replaces the analysis' contiguity of `x` (AxisInfo.cpp:1244-1259), which `x + s` then inherits
(gcd, :242-279). So for `y = hinted_arange + s` the analysis infers `C`, not `b - a`.

Loads of int8 through `x_ptr + y` with `y = tl.multiple_of(y, 16)`: the load's vector width is
min(divisibility, contiguity) bytes (16 bytes: `ld.global.v4.b32`; 4 bytes: `ld.global.b32`), so the
PTX shows the contiguity the analysis gave `y`.

  B1  y = tl.arange(0, 512) + s; y = tl.multiple_of(y, 16)                            (s = 16)
  B2  a = tl.max_contiguous(tl.multiple_of(tl.arange(0, 512), 4), 4); y = a + s; y = tl.multiple_of(y, 16)
      the library: y's bound is 512 (an arange plus a scalar), so the model checks 16 | y[0] only;
      the analysis: contiguity 4, so the compiler believes 16 | y[4j] for every j (false for j odd...)
  B3  as B2 with s = 4 (the model's check 16 | y[0] false: a control of the hint's effect)
Each case runs in its own process on GPU 0, with one warp, and reports the loads and the result.
Run: CUDA_VISIBLE_DEVICES=0 <venv>/bin/python -I hint_bound_subexpr.py
"""
import json, os, re, subprocess, sys

import torch
import triton
import triton.language as tl


@triton.jit
def k(x_ptr, out_ptr, s, PAIR: tl.constexpr):
    a = tl.arange(0, 512)
    if PAIR:
        a = tl.max_contiguous(tl.multiple_of(a, 4), 4)
    y = a + s
    y = tl.multiple_of(y, 16)
    tl.store(out_ptr + tl.arange(0, 512), tl.load(x_ptr + y))


def run_case(name):
    x = (torch.arange(1024, device="cuda") % 127).to(torch.int8)
    out = torch.zeros(512, device="cuda", dtype=torch.int8)
    s = 4 if name == "B3" else 16
    ck = k[(1,)](x, out, s, PAIR=name != "B1", num_warps=1)
    loads = sorted(set(re.findall(r"ld\.global(?:\.[a-z0-9]+)+", ck.asm["ptx"])))
    try:
        torch.cuda.synchronize()
        ok = torch.equal(out.cpu(), x.cpu()[s:s + 512])
        res = "correct" if ok else "WRONG"
    except Exception as e:
        res = "error: " + str(e).strip().splitlines()[0][:150]
    return {"loads": loads, "result": res}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        try:
            res = run_case(sys.argv[2])
        except Exception as e:
            res = {"loads": [], "result": "error: " + type(e).__name__ + ": " + str(e).strip().splitlines()[0][:150]}
        print("RESULT " + json.dumps(res))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name in ["B1", "B2", "B3"]:
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                           capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        res = json.loads(line[7:]) if line else {"loads": [], "result": "process failed"}
        print(f"{name}: loads {res['loads']}; {res['result']}")


if __name__ == "__main__":
    main()

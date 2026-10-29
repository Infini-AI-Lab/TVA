"""A hint on one result of an operation that has several results.

Library evidence for library issue L60 (round 11). `set_attr` (`primitives/triton-3.8.0-ir.cc`
lines 411-425) puts a hint on the operation that defines the hinted value. When that operation has
several results and no visitor in the axis analysis, every result gets
`getPessimisticValueState` (`primitives/triton-3.8.0-AxisInfo.cpp` lines 1239-1242, 1318-1342),
which reads the hints off the defining operation: so a hint on one result would apply to all of
them. The results of a loop or a runtime `if` (`scf.for`, `scf.if`) are values of one operation
too, but the data-flow analysis gives them the values the loop or the branches yield. Which results
does the hint reach?

Each case loads index vectors of 64 `int32` from memory (so the analysis knows nothing of them):
`a = 4i` (every element a multiple of 4) and `b = 4i + 2` (none is). It builds results `r0`, `r1`
of one operation from them, so that `r0` holds `a` and `r1` holds `b`, hints one of them with
`tl.multiple_of(…, 4)`, and loads `x[w[i] + c]` for `c` in 0..3 through `w = r1` (or `w = r0`). With
one warp, each thread loads four consecutive `float32` when the compiler believes `w[i]` is a multiple
of 4: `ld.global.v4.b32`, which faults on `b`'s rows. Each case runs in its own process, on GPU 0.

  M1   r0, r1 = tl.split(z), z a loaded [64, 2] block of rows [4i, 4i + 2]
                                                          hint r0 (true), load through r1
  M2   as M1 without the hint (control)
  M3   r1, r0 = tl.max(tl.join(b, b), axis=1, return_indices=True)  (r0: the indices, all 0)
                                                          hint r0 (true), load through r1
  M4   r0, r1 = tl.reduce((tl.join(a, a), tl.join(b, b)), 1, <maximum of each>)
                                                          hint r0 (true), load through r1
  M5   r0, r1 = tl.associative_scan((a, b), 0, <the right operand of each>)
                                                          hint r0 (true), load through r1
  M6   s0, s1 = a helper declared noinline=True, given the loaded scalars 0 and 2 (only scalars may be
       passed to one); hint s0 (true), load through w = s1 + 4i
  M7   for _ in range(n): a += 4; b += 4  -> r0, r1 the values after the loop
                                                          hint r0 (true), load through r1
  M8   as M7, hint r1 (false), load through r1
  M9   a runtime if (on an argument 2) whose branches add 4 (or 8) to a and b -> r0, r1 after it
                                                          hint r0 (true), load through r1
  M10  as M9, hint r1 (false), load through r1
  M11  r1 = b + d, with d = 4 a runtime argument (one operation defines r1)
                                                          hint r1 (false), load through r1 (control)
  M12  r1 = an inlined helper's result (it returns b + d)  hint r1 (false), load through r1
  M13  r1 = tl.max(tl.join(b, b), axis=1)                  hint r1 (false), load through r1
  M14  r0, r1 = tl.split(z), z = tl.reshape(tl.arange(0, 128), (64, 2)) * 2 (rows [4i, 4i + 2],
       computed in the kernel)                            hint r0 (true), load through r1

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/hint-multi/run.py  (from TVA's root)
"""
import json, os, re, subprocess, sys

import torch
import triton
import triton.language as tl


@triton.jit
def max_each(a0, b0, a1, b1):
    return tl.maximum(a0, a1), tl.maximum(b0, b1)


@triton.jit
def right_each(a0, b0, a1, b1):
    return a1, b1


@triton.jit(noinline=True)
def pair(s0, s1):
    return s0, s1


@triton.jit
def add_d(v, d):
    return v + d


@triton.jit
def k(x_ptr, out_ptr, ia, ib, iz, n, flag, d, CASE: tl.constexpr):
    i = tl.arange(0, 64)
    a = tl.load(ia + i)
    b = tl.load(ib + i)
    HINT1: tl.constexpr = CASE == 8 or CASE == 10 or CASE == 11 or CASE == 12 or CASE == 13
    if CASE == 1 or CASE == 2:
        z = tl.load(iz + i[:, None] * 2 + tl.arange(0, 2)[None, :])
        r0, r1 = tl.split(z)
    elif CASE == 3:
        r1, r0 = tl.max(tl.join(b, b), axis=1, return_indices=True)
    elif CASE == 4:
        r0, r1 = tl.reduce((tl.join(a, a), tl.join(b, b)), 1, max_each)
    elif CASE == 5:
        r0, r1 = tl.associative_scan((a, b), 0, right_each)
    elif CASE == 6:
        s0, s1 = pair(tl.load(ia), tl.load(ib))
        s0 = tl.multiple_of(s0, 4)
        r0 = a
        r1 = s1 + 4 * i
    elif CASE == 7 or CASE == 8:
        for _ in range(n):
            a += 4
            b += 4
        r0 = a
        r1 = b
    elif CASE == 9 or CASE == 10:
        if flag > 0:
            a = a + 4
            b = b + 4
        else:
            a = a + 8
            b = b + 8
        r0 = a
        r1 = b
    elif CASE == 11:
        r0 = a
        r1 = b + d
    elif CASE == 12:
        r0 = a
        r1 = add_d(b, d)
    elif CASE == 14:
        z = tl.reshape(tl.arange(0, 128), (64, 2)) * 2
        r0, r1 = tl.split(z)
    else:
        r0 = a
        r1 = tl.max(tl.join(b, b), axis=1)
    if CASE != 2 and CASE != 6:
        if HINT1:
            r1 = tl.multiple_of(r1, 4)
        else:
            r0 = tl.multiple_of(r0, 4)
    c = tl.arange(0, 4)
    v = tl.load(x_ptr + r1[:, None] + c[None, :])
    tl.store(out_ptr + i[:, None] * 4 + c[None, :], v)


CASES = {
    "M1": "tl.split: hint r0 (true), load through r1",
    "M2": "tl.split: no hint (control)",
    "M3": "tl.max(..., return_indices=True): hint the indices (true), load through the values",
    "M4": "tuple tl.reduce: hint r0 (true), load through r1",
    "M5": "tuple tl.associative_scan: hint r0 (true), load through r1",
    "M6": "noinline helper returning two scalars: hint s0 (true), load through s1 + 4i",
    "M7": "loop results: hint r0 (true), load through r1",
    "M8": "loop results: hint r1 (false), load through r1",
    "M9": "runtime if results: hint r0 (true), load through r1",
    "M10": "runtime if results: hint r1 (false), load through r1",
    "M11": "r1 = b + d (one operation): hint r1 (false), load through r1 (control)",
    "M12": "r1 = an inlined helper's result: hint r1 (false), load through r1",
    "M13": "r1 = tl.max(..., axis=1) (an inlined library function): hint r1 (false), load through r1",
    "M14": "tl.split of a value computed in the kernel: hint r0 (true), load through r1",
}


def run_case(name):
    dev = "cuda"
    case = int(name[1:])
    x = torch.arange(1024, device=dev, dtype=torch.float32)
    out = torch.zeros(256, device=dev, dtype=torch.float32)
    ia = torch.arange(64, device=dev, dtype=torch.int32) * 4
    ib = ia + 2
    iz = torch.stack([ia, ib], dim=1).contiguous()
    n, flag, d = 3, 2, 4
    ck = k[(1,)](x, out, ia, ib, iz, n, flag, d, CASE=case, num_warps=1)
    w = ib.cpu() + {6: 0, 7: 4 * n, 8: 4 * n, 9: 4, 10: 4, 11: d, 12: d}.get(case, 0)
    ref = torch.tensor([float(w[r] + c) for r in range(64) for c in range(4)])
    loads = sorted(set(re.findall(r"ld\.global(?:\.[a-z0-9]+)+", ck.asm["ptx"])))
    try:
        torch.cuda.synchronize()
        got = out.cpu()
        result = "correct" if torch.equal(got, ref) else f"WRONG at {(got != ref).sum().item()} elements"
    except Exception as e:
        result = "error: " + str(e).strip().splitlines()[0][:150]
    def marked(stage):
        return [l.strip()[:120] for l in ck.asm[stage].splitlines()
                if "tt.divisibility = dense<4>" in l and "tt.func" not in l][:1]
    return {"loads": loads, "result": result, "front end": marked("source"), "after ttir": marked("ttir")}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        try:
            res = run_case(sys.argv[2])
        except Exception as e:
            res = {"loads": [], "front end": [], "after ttir": [],
                   "result": "error: " + type(e).__name__ + ": " + str(e).strip().splitlines()[0][:150]}
        print("RESULT " + json.dumps(res))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name, desc in CASES.items():
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                           capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line:
            res = json.loads(line[len("RESULT "):])
        else:
            err = [l for l in p.stderr.strip().splitlines() if l.strip()]
            res = {"loads": [], "front end": [], "after ttir": [],
                   "result": "process failed: " + (err[-1][:150] if err else "(no output)")}
        print(f"{name}: {desc}\n  loads {res['loads']}; {res['result']}\n"
              f"  the hint's operation, front end: {res['front end']}\n  after the TTIR passes: {res['after ttir']}")


if __name__ == "__main__":
    main()

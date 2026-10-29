"""Where does a compiler hint inside a loop body take effect?

Library evidence for library issue L1 (round 8). Each case compiles a small kernel with Triton
3.8.0, launches it on GPU 0 in its own process (a misaligned access ends the process's CUDA
context), and reports:
  - the global loads of the PTX (`ld.global.v4.b32` is a 16-byte vector load of four float32; a block of
    128 float32 with one warp gives each thread four consecutive elements, so a load is vectorised
    exactly when the compiler believes the address is 16-byte aligned);
  - the hint attributes in the front end's output (`asm['source']`, before any pass);
  - the result against the host's reference, or the error the launch raised.

Run: .venv/bin/python -I evidence/hint-loops/run.py  (from TVA's root; CUDA_VISIBLE_DEVICES=0)
"""
import json, os, re, subprocess, sys

import torch
import triton
import triton.language as tl


@triton.jit
def k_carried(x_ptr, out_ptr, n_iter, STEP: tl.constexpr, HINT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    p = x_ptr + offs                       # the value of `p` before the loop
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(0, n_iter):
        if HINT:
            tl.multiple_of(p, 16)          # a hint on the name the loop carries
        acc += tl.load(p)
        p += STEP
    tl.store(out_ptr + offs, acc)


@triton.jit
def k_while(x_ptr, out_ptr, n_iter, STEP: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    p = x_ptr + offs
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    i = 0
    while i < n_iter:
        tl.multiple_of(p, 16)
        acc += tl.load(p)
        p += STEP
        i += 1
    tl.store(out_ptr + offs, acc)


@triton.jit
def k_iv(x_ptr, out_ptr, n_iter, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(1, n_iter):
        i = tl.multiple_of(i, 16)          # a hint on the induction variable
        acc += tl.load(x_ptr + i + offs)
    tl.store(out_ptr + offs, acc)


@triton.jit
def k_arg(x_ptr, out_ptr, BLOCK: tl.constexpr):
    x_ptr = tl.multiple_of(x_ptr, 16)      # a hint on a kernel parameter
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs))


@triton.jit
def helper_load(p, BLOCK: tl.constexpr):
    p = tl.multiple_of(p, 16)              # a hint on a helper's parameter
    return tl.load(p + tl.arange(0, BLOCK))


@triton.jit
def k_helper(x_ptr, out_ptr, BLOCK: tl.constexpr):
    q = x_ptr + 1                          # 4 bytes past an aligned base
    tl.store(out_ptr + tl.arange(0, BLOCK), helper_load(q, BLOCK))


@triton.jit
def k_nested_both(x_ptr, out_ptr, n_outer, n_inner, S_IN: tl.constexpr, S_OUT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    p = x_ptr + offs                       # carried by both loops
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for o in range(0, n_outer):
        for i in range(0, n_inner):
            tl.multiple_of(p, 16)
            acc += tl.load(p)
            p += S_IN
        p += S_OUT
    tl.store(out_ptr + offs, acc)


@triton.jit
def k_nested_inner(x_ptr, out_ptr, n_outer, n_inner, S_IN: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for o in range(0, n_outer):
        p = x_ptr + o + offs               # defined by the outer body, carried by the inner loop
        for i in range(0, n_inner):
            tl.multiple_of(p, 16)
            acc += tl.load(p)
            p += S_IN
    tl.store(out_ptr + offs, acc)


BLOCK = 128   # with num_warps=1, each of the 32 threads holds 4 consecutive float32


HOST = torch.arange(1024, dtype=torch.float32)   # the host's copy of x


def ref_walk(x, starts):
    """Sum of the BLOCK-element loads at the given element offsets (exact: small integers)."""
    xs = HOST
    acc = torch.zeros(BLOCK, dtype=torch.float32)
    for s in starts:
        acc += xs[s:s + BLOCK]
    return acc.tolist()


CASES = {
    # name: (description, expectation)
    "C1": ("for: hint on carried p; p true before the loop; p += 1 element (later values 4-byte aligned)", "correct, no vector load"),
    "C2": ("for: the same with p += 4, but p false before the loop (base 4 bytes past alignment)", "fault: the check is on the pre-loop value"),
    "C3": ("for: p += 4, p true before the loop", "correct, vector loads"),
    "C4": ("for: as C2 without the hint", "correct (control)"),
    "C5": ("for: hint on the induction variable i (from 1, step 1)", "correct, no vector load"),
    "C6": ("hint on the kernel parameter x_ptr, x 4 bytes past alignment", "correct, no vector load"),
    "C7": ("hint on a helper's parameter, given a pointer 4 bytes past alignment", "correct, no vector load"),
    "C8": ("nested: p carried by both loops; true before the outer loop; p += 1 after each inner loop", "correct, no vector load"),
    "C9": ("nested: as C8 with p += 4, but p false before the outer loop", "fault: the check is on the value before the outer loop"),
    "C10": ("nested: p defined by the outer body (x + o), carried by the inner loop; one outer iteration (o = 0)", "correct, vector loads"),
    "C11": ("nested: as C10 with two outer iterations (o = 1 makes p false)", "fault: the check is on each outer iteration's value"),
    "C12": ("while: hint on carried p; true before the loop; p += 1", "correct, no vector load"),
    "C13": ("while: p += 4, false before the loop", "fault"),
}


def run_case(name):
    dev = "cuda"
    x = torch.arange(1024, device=dev, dtype=torch.float32)
    out = torch.zeros(BLOCK, device=dev, dtype=torch.float32)
    kw = dict(BLOCK=BLOCK, num_warps=1)
    xm = x[1:]                                 # 4 bytes past a 16-byte boundary
    n = 8
    if name in ("C1", "C2", "C3", "C4"):
        step = 1 if name == "C1" else 4
        base = xm if name in ("C2", "C4") else x
        hint = name != "C4"
        ck = k_carried[(1,)](base, out, n, STEP=step, HINT=hint, **kw)
        off0 = 1 if base is xm else 0
        ref = ref_walk(x, [off0 + i * step for i in range(n)])
    elif name in ("C12", "C13"):
        step = 1 if name == "C12" else 4
        base = x if name == "C12" else xm
        ck = k_while[(1,)](base, out, n, STEP=step, **kw)
        off0 = 0 if base is x else 1
        ref = ref_walk(x, [off0 + i * step for i in range(n)])
    elif name == "C5":
        ck = k_iv[(1,)](x, out, n, **kw)
        ref = ref_walk(x, [i for i in range(1, n)])
    elif name == "C6":
        ck = k_arg[(1,)](xm, out, **kw)
        ref = ref_walk(x, [1])
    elif name == "C7":
        ck = k_helper[(1,)](x, out, **kw)
        ref = ref_walk(x, [1])
    elif name in ("C8", "C9"):
        s_in, s_out = (4, 1) if name == "C8" else (4, 4)
        base = x if name == "C8" else xm
        ck = k_nested_both[(1,)](base, out, 3, 4, S_IN=s_in, S_OUT=s_out, **kw)
        off = 0 if base is x else 1
        starts = []
        for o in range(3):
            for i in range(4):
                starts.append(off)
                off += s_in
            off += s_out
        ref = ref_walk(x, starts)
    elif name in ("C10", "C11"):
        n_outer = 1 if name == "C10" else 2
        ck = k_nested_inner[(1,)](x, out, n_outer, 4, S_IN=4, **kw)
        ref = ref_walk(x, [o + 4 * i for o in range(n_outer) for i in range(4)])
    else:
        raise SystemExit(f"unknown case {name}")
    ptx = ck.asm["ptx"]
    loads = sorted(set(re.findall(r"ld\.global(?:\.[a-z0-9]+)+", ptx)))
    attrs = sorted(set(m.strip() for m in ck.asm["source"].splitlines()
                       if "tt.divisibility" in m and "tt.func" not in m))
    attrs = [re.sub(r"\s+loc\(.*\)$", "", a) for a in attrs]
    try:
        torch.cuda.synchronize()
        got = out.tolist()
        result = "correct" if got == ref else "WRONG"
    except Exception as e:  # the launch's error, reported to the host
        result = "error: " + str(e).splitlines()[0][:160]
    return {"loads": loads, "attrs": attrs, "result": result}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        name = sys.argv[2]
        try:
            res = run_case(name)
        except Exception as e:  # the launch's error, reported to the host
            res = {"loads": [], "attrs": [], "result": "error: " + str(e).splitlines()[0][:160]}
        print("RESULT " + json.dumps(res))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name, (desc, expect) in CASES.items():
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                           capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line is None:
            err = (p.stderr.strip().splitlines() or ["(no output)"])[-1]
            res = {"loads": [], "attrs": [], "result": "process failed: " + err[:160]}
        else:
            res = json.loads(line[len("RESULT "):])
        print(f"\n{name}: {desc}")
        print(f"  expected: {expect}")
        print(f"  result:   {res['result']}")
        print(f"  loads:    {', '.join(res['loads']) or '-'}")
        for a in res["attrs"]:
            print(f"  attr:     {a}")


if __name__ == "__main__":
    main()

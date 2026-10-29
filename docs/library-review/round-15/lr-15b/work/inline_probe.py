"""lr-15b: is every call of a helper inlined? A helper with a `return` inside a runtime `if`.

SEMANTICS "Compiler hints: which value" says every call of a `@triton.jit` function is inlined except
that of a helper declared `noinline=True`, so a hint on the value such a call returns has no effect
and the model drops it (TRANSCRIPTION §4 `no-op:hint`). A helper with a `return` inside a runtime
`if` is split into blocks (`cf.cond_br`, SEMANTICS "return"). Is its call inlined when the call sits
inside a `for` body or a runtime `if` (single-block regions), and is a hint on its result read?

  I1  call at the top level of the kernel             h_early(base + arange, n), then the hint pair
  I2  call inside a for (range) body                  the same
  I3  call inside a runtime if                        the same
  I0  control: a helper with one return, in a for body
  I5  control: the hint pair on base + arange directly, base = 16 (true): the loads it allows
Each case: is `tt.call` left in the TTGIR, the width of the loads in the PTX, and the loaded values
(base = 4, so the hint `multiple_of(…, 16)` is false; n = 0, so the helper returns its argument).
The hint pair is `tl.max_contiguous(tl.multiple_of(offs, 16), 16)`.
Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I <this file>
"""
import json, os, re, subprocess, sys, tempfile

import torch
import triton
import triton.language as tl


@triton.jit
def h_early(a, n):
    if n > 100:
        return a + 1
    return a


@triton.jit
def h_one(a, n):
    return a + n * 0


@triton.jit
def k(x_ptr, out_ptr, base, n, m, CASE: tl.constexpr):
    if CASE == 1:
        offs = h_early(base + tl.arange(0, 256), n)
        offs = tl.max_contiguous(tl.multiple_of(offs, 16), 16)
        tl.store(out_ptr + tl.arange(0, 256), tl.load(x_ptr + offs))
    elif CASE == 2:
        for it in range(0, m):
            offs = h_early(base + tl.arange(0, 256), n)
            offs = tl.max_contiguous(tl.multiple_of(offs, 16), 16)
            tl.store(out_ptr + tl.arange(0, 256), tl.load(x_ptr + offs))
    elif CASE == 3:
        if m > 0:
            offs = h_early(base + tl.arange(0, 256), n)
            offs = tl.max_contiguous(tl.multiple_of(offs, 16), 16)
            tl.store(out_ptr + tl.arange(0, 256), tl.load(x_ptr + offs))
    elif CASE == 5:
        offs = tl.max_contiguous(tl.multiple_of(base + tl.arange(0, 256), 16), 16)
        tl.store(out_ptr + tl.arange(0, 256), tl.load(x_ptr + offs))
    else:
        for it in range(0, m):
            offs = h_one(base + tl.arange(0, 256), n)
            offs = tl.max_contiguous(tl.multiple_of(offs, 16), 16)
            tl.store(out_ptr + tl.arange(0, 256), tl.load(x_ptr + offs))


CASES = {"I0": 0, "I1": 1, "I2": 2, "I3": 3, "I5": 5}


def run_case(name):
    x = torch.arange(1024, device="cuda", dtype=torch.int32).to(torch.int8)
    out = torch.zeros(256, device="cuda", dtype=torch.int8)
    base = 16 if name == "I5" else 4
    ck = k[(1,)](x, out, base, 0, 1, CASE=CASES[name], num_warps=1)
    torch.cuda.synchronize()
    ttgir = ck.asm["ttgir"]
    ptx = ck.asm["ptx"]
    calls = len(re.findall(r"tt\.call", ttgir))
    loads = sorted(set(re.findall(r"ld\.global(?:\.[a-z0-9]+)*", ptx)))
    expect = x[base:base + 256]
    ok = torch.equal(out, expect)
    bad = int((out != expect).sum().item())
    return f"tt.call in ttgir: {calls}; loads {loads}; values {'correct' if ok else f'WRONG in {bad} of 256'}"


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        try:
            res = run_case(sys.argv[2])
        except Exception as e:
            msg = [l for l in str(e).strip().splitlines() if l.strip()]
            res = "error: " + type(e).__name__ + ": " + (msg[-1].strip() if msg else "")[:200]
        print("RESULT " + json.dumps(res))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name in CASES:
        with tempfile.TemporaryDirectory() as cache:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", TRITON_CACHE_DIR=cache)
            p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                               capture_output=True, text=True, env=env)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line:
            res = json.loads(line[len("RESULT "):])
        else:
            err = [l for l in p.stderr.strip().splitlines() if l.strip()]
            res = f"process failed (exit {p.returncode}): " + (err[-1][:200] if err else "(no output)")
        print(f"{name}: {res}")


if __name__ == "__main__":
    main()

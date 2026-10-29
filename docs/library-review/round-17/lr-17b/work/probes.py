"""lr-17b, library review round 17: probes of the batch's statements and of the transcription area.

H: the value a helper returns when it has two reachable `return`s (SEMANTICS "Helpers are inlined",
   "`return`"). `code_generator.py` `decide_return_type` (556-585) gives the call the common type of
   every `return` the front end visited: one compile-time value when they are all the same one, and
   otherwise the tensor type of each (`to_tensor_type`), each `return` then converted to it
   (`cast_to`, 592-604). Does a helper whose runtime `if` returns one Python number and whose last
   `return` another give the caller a tensor?
  H1  _h_rt(c): `if c > 0: return -1` then `return -3`; the kernel stores `_h_rt(c) // 2`, c = 5
  H2  the same with c = -5
  H3  control: `if c > 0: return -1` then `return -1` (the same value), c = 5
  H4  `tl.static_assert(_h_rt(c) == -1)`: accepted only if the call is a compile-time value
  H5  _h_f(c): `if c > 0: return 0.1` then `return 0.2`; the kernel stores `x * _h_f(c)` for a
      bfloat16 block x (every finite bfloat16 pattern), against `x * 0.1` written directly (a
      Python number: built in bfloat16, the product in bfloat16) and against
      `(x.to(tl.float32) * 0.1).to(tl.bfloat16)` (the float32 product)
  H6  a conditional expression on a runtime condition (SEMANTICS "Order": "it means the runtime
      `if`"): the kernel stores `(-1 if c > 0 else -3) // 2`, c = 5 (`visit_IfExp`, 998-1030, makes
      each side `to_tensor`)
  H7  the same for H5's product: `x * (0.1 if c > 0 else 0.2)`, x every finite bfloat16

G: CUDA graph capture on another stream (L91's statement: "a call made while a CUDA stream capture
   is in progress also returns without an error and runs no program"). A warm-up compiles the
   kernel first; the output holds the sentinel -7 before the call.
  G0  control: the call made on the capturing stream (torch.cuda.graph's)
  G1  capture in progress (torch.cuda.graph, capture_error_mode "global", the default), the call
      made on another stream s2 (`with torch.cuda.stream(s2)`); after the capture, s2 synchronised,
      before any replay: did the programs run? Then the output is reset and the graph replayed
  G2  the same with capture_error_mode "thread_local"
  G3  the same with capture_error_mode "relaxed"

Triton 3.8.0, torch from TVA's venv, GPU 0 (an H200 NVL). Each case runs in its own process with
its own empty Triton cache. Run from this folder:
  CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I probes.py
"""
import json, os, subprocess, sys, tempfile

import torch
import triton
import triton.language as tl


@triton.jit
def _h_rt(c):
    if c > 0:
        return -1
    return -3


@triton.jit
def _h_same(c):
    if c > 0:
        return -1
    return -1


@triton.jit
def _h_f(c):
    if c > 0:
        return 0.1
    return 0.2


@triton.jit
def k_int(out_ptr, c, CASE: tl.constexpr):
    if CASE == 1:
        tl.store(out_ptr, _h_rt(c) // 2)
    elif CASE == 3:
        tl.store(out_ptr, _h_same(c) // 2)
    elif CASE == 6:
        tl.store(out_ptr, (-1 if c > 0 else -3) // 2)
    else:
        tl.static_assert(_h_rt(c) == -1)
        tl.store(out_ptr, 0)


@triton.jit
def k_f(x_ptr, out_ptr, c, n, MODE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    if MODE == 0:
        y = x * _h_f(c)
    elif MODE == 3:
        y = x * (0.1 if c > 0 else 0.2)
    elif MODE == 1:
        y = x * 0.1
    else:
        y = (x.to(tl.float32) * 0.1).to(tl.bfloat16)
    tl.store(out_ptr + offs, y, mask=m)


@triton.jit
def add1(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) + 1.0, mask=m)


CASES = {
    "H1": "_h_rt(c) // 2, c = 5 (returns -1 or -3 through a runtime if)",
    "H2": "_h_rt(c) // 2, c = -5",
    "H3": "control: _h_same(c) // 2 (returns -1 on both paths), c = 5",
    "H4": "tl.static_assert(_h_rt(c) == -1)",
    "H5": "x * _h_f(c) (returns 0.1 or 0.2 through a runtime if), x every finite bfloat16, c = 5",
    "H6": "(-1 if c > 0 else -3) // 2, c = 5 (a conditional expression on a runtime condition)",
    "H7": "x * (0.1 if c > 0 else 0.2), x every finite bfloat16, c = 5",
    "G0": "control: the call made on the capturing stream",
    "G1": "capture in progress (global mode), the call made on another stream",
    "G2": "capture in progress (thread_local mode), the call made on another stream",
    "G3": "capture in progress (relaxed mode), the call made on another stream",
}


def run_h(name):
    dev = "cuda"
    if name in ("H1", "H2", "H3", "H4", "H6"):
        out = torch.full((1,), 99, device=dev, dtype=torch.int32)
        c = -5 if name == "H2" else 5
        k_int[(1,)](out, c, CASE={"H1": 1, "H2": 1, "H3": 3, "H4": 4, "H6": 6}[name])
        torch.cuda.synchronize()
        return f"stored {int(out.item())}"
    bits = torch.arange(0, 2**16, dtype=torch.int32).to(torch.int16)
    x = bits.view(torch.bfloat16).to(dev)
    keep = torch.isfinite(x)
    x = x[keep].contiguous()
    n = x.numel()
    outs = []
    for mode in ((3 if name == "H7" else 0), 1, 2):
        o = torch.empty_like(x)
        k_f[(triton.cdiv(n, 1024),)](x, o, 5, n, MODE=mode, BLOCK=1024)
        outs.append(o.view(torch.int16))
    torch.cuda.synchronize()
    got, direct, f32 = outs
    what = "the conditional expression's product" if name == "H7" else "the helper's product"
    return (f"{n} inputs: {what} equal (bit for bit) to the float32 product on {int((got == f32).sum())}, "
            f"to the bfloat16 product of the Python number on {int((got == direct).sum())}; "
            f"the two products differ on {int((direct != f32).sum())}")


def run_g(name):
    dev = "cuda"
    n = 4096
    x = torch.ones(n, device=dev)
    o = torch.full((n,), -7.0, device=dev)
    add1[(n // 1024,)](x, o, n, BLOCK=1024)  # warm-up: compiles
    torch.cuda.synchronize()
    o.fill_(-7.0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s2 = torch.cuda.Stream()
    mode = {"G0": "global", "G1": "global", "G2": "thread_local", "G3": "relaxed"}[name]
    err = None
    try:
        with torch.cuda.graph(g, capture_error_mode=mode):
            if name == "G0":
                add1[(n // 1024,)](x, o, n, BLOCK=1024)
            else:
                with torch.cuda.stream(s2):
                    add1[(n // 1024,)](x, o, n, BLOCK=1024)
    except Exception as e:
        err = type(e).__name__ + ": " + str(e).strip().splitlines()[-1][:160]
    s2.synchronize()
    torch.cuda.synchronize()
    after_call = (float(o[0]), float(o[-1]))
    res = f"returned {'with ' + err if err else 'without an error'}; after the capture, before any replay: o[0], o[-1] = {after_call}"
    if err is None:
        o.fill_(-7.0)
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        res += f"; after resetting and one replay: {(float(o[0]), float(o[-1]))}"
    return res


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        name = sys.argv[2]
        try:
            res = run_h(name) if name.startswith("H") else run_g(name)
        except Exception as e:
            msg = [l for l in str(e).strip().splitlines() if l.strip()]
            res = "rejected on the host: " + type(e).__name__ + ": " + (msg[-1].strip() if msg else "")[:160]
        print("RESULT " + json.dumps(res))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name, desc in CASES.items():
        with tempfile.TemporaryDirectory() as cache:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", TRITON_CACHE_DIR=cache)
            p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                               capture_output=True, text=True, env=env)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line:
            res = json.loads(line[len("RESULT "):])
        else:
            err = [l for l in p.stderr.strip().splitlines() if l.strip()]
            res = "process failed: " + (err[-1][:160] if err else "(no output)")
        print(f"{name}: {desc}\n  {res}")


if __name__ == "__main__":
    main()

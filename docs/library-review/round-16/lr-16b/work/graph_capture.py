"""lr-16b: a Triton launch captured into a CUDA graph (torch.cuda.CUDAGraph).

Question: the library says that a launch that returns without an error has an outcome, even if no
program ran (TVA.Claim, compiler-faithful.md), and Compiler.Faithful relates that outcome to the
model. Under stream capture the call `k[grid](...)` returns without an error and runs no program;
the programs run at each `graph.replay()`, with the captured arguments, on the memory of the
replay. What does the library say a claim covers then?

G1  warm-up (compile) outside capture; capture one call; check the output right after capture
    (sentinel unchanged: the call returned, no program ran); replay; change the input on the
    host and replay again (the replay reads the memory of the replay, with the captured
    arguments).
G2  the same with the scalar argument changed between capture and replay on the host: the
    replay keeps the captured value.
G3  capture with no warm-up (the first call compiles while the stream captures).

Triton 3.8.0, torch from the TVA venv, GPU 0 (an H200 NVL). Run:
  CUDA_VISIBLE_DEVICES=0 python -I graph_capture.py
"""
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def add1(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(o_ptr + offs, x + 1.0, mask=m)


@triton.jit
def add2(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(o_ptr + offs, x + 2.0, mask=m)


def first(t):
    return float(t[0].item()), float(t[-1].item())


def main():
    dev = "cuda"
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    N, B = 4096, 256
    grid = (triton.cdiv(N, B),)

    # G1
    x = torch.full((N,), 1.0, device=dev)
    o = torch.full((N,), -7.0, device=dev)
    add1[grid](x, o, N, BLOCK=B)          # warm-up: compiles and runs once
    torch.cuda.synchronize()
    print(f"G1 warm-up launch: o[0], o[-1] = {first(o)} (x + 1)")
    o.fill_(-7.0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            ret = add1[grid](x, o, N, BLOCK=B)
    torch.cuda.synchronize()
    print(f"G1 captured call returned without an error ({type(ret).__name__}); right after capture "
          f"o[0], o[-1] = {first(o)} (the sentinel: no program ran)")
    g.replay()
    torch.cuda.synchronize()
    print(f"G1 after replay 1: o[0], o[-1] = {first(o)}")
    x.fill_(10.0)
    o.fill_(-7.0)
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    print(f"G1 x set to 10 on the host, replay 2: o[0], o[-1] = {first(o)} (the replay reads the memory of the replay)")

    # G2: the scalar argument is the captured one
    x2 = torch.full((N,), 1.0, device=dev)
    o2 = torch.full((N,), -7.0, device=dev)
    add2[grid](x2, o2, N, BLOCK=B)
    torch.cuda.synchronize()
    o2.fill_(-7.0)
    n_now = 16
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g2, stream=s):
            add2[grid](x2, o2, n_now, BLOCK=B)
    n_now = N  # a later host value; the graph keeps 16
    g2.replay()
    torch.cuda.synchronize()
    written = int((o2 != -7.0).sum().item())
    print(f"G2 captured with n = 16, host variable later {n_now}: replay wrote {written} elements")

    # G3: capture with no warm-up
    @triton.jit
    def add3(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) + 3.0, mask=m)

    o3 = torch.full((N,), -7.0, device=dev)
    g3 = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(s):
            with torch.cuda.graph(g3, stream=s):
                add3[grid](x2, o3, N, BLOCK=B)
        torch.cuda.synchronize()
        print(f"G3 capture with no warm-up: returned; right after capture o3[0] = {first(o3)[0]}")
        g3.replay()
        torch.cuda.synchronize()
        print(f"G3 after replay: o3[0] = {first(o3)[0]}")
    except Exception as e:  # noqa: BLE001
        print(f"G3 capture with no warm-up: {type(e).__name__}: {str(e).splitlines()[0][:200]}")


if __name__ == "__main__":
    sys.exit(main())

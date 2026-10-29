"""A hint on one result of an operation with two results (lr-8b, round 8).

`set_attr` (ir.cc:411-425) puts a hint on the operation that defines the value. `tt.split` has
two results and no AxisInfo visitor, so `AxisInfoAnalysis::visitOperation` sets both results
to `getPessimisticValueState`, which reads the hint off the defining operation
(AxisInfo.cpp:1234-1259, 1318-1340): a hint on one result is also a hint on the other.

z = [[4i, 4i + 2]] (64 rows); a, b = tl.split(z): a = 4i (a multiple of 4), b = 4i + 2 (not).
The kernel hints only a (true), then loads x[b[i] + c] for c in 0..3. Each case runs in its own
process (a misaligned access ends the CUDA context), GPU 0, one warp.
"""
import os, re, subprocess, sys
import torch
import triton
import triton.language as tl


@triton.jit
def k_split(x_ptr, out_ptr, HINT: tl.constexpr):
    z = tl.reshape(tl.arange(0, 128), (64, 2)) * 2      # rows [4i, 4i + 2]
    a, b = tl.split(z)                                  # a = 4i, b = 4i + 2
    if HINT == 1:
        a = tl.multiple_of(a, 4)                        # true of a
    if HINT == 2:
        b = tl.multiple_of(b, 2)                        # true of b (control: b's own hint)
    c = tl.arange(0, 4)
    v = tl.load(x_ptr + b[:, None] + c[None, :])
    tl.store(out_ptr + tl.arange(0, 64)[:, None] * 4 + c[None, :], v)
    tl.store(out_ptr + 256 + tl.arange(0, 64), a)


def run(h):
    x = torch.arange(512, device="cuda", dtype=torch.float32)
    out = torch.zeros(320, device="cuda", dtype=torch.float32)
    ck = k_split[(1,)](x, out, HINT=h, num_warps=1)
    loads = sorted(set(re.findall(r"ld\.global(?:\.[a-z0-9]+)+", ck.asm["ptx"])))
    ttir = ck.asm["ttir"]
    split_line = [l.strip() for l in ttir.splitlines() if "tt.split" in l]
    try:
        torch.cuda.synchronize()
        got = out.cpu()
        ref = torch.tensor([float(4 * i + 2 + c) for i in range(64) for c in range(4)] +
                           [float(4 * i) for i in range(64)])
        res = "correct" if torch.equal(got, ref) else f"WRONG at {(got != ref).sum().item()} elements"
    except Exception as e:
        res = "error: " + str(e).splitlines()[0][:120]
    print(f"HINT={h}: loads {loads}; {res}; ttir: {split_line[:1]}")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        run(int(sys.argv[1]))
    else:
        print("triton", triton.__version__, torch.cuda.get_device_name(0), torch.__version__)
        print("HINT=0 no hint; HINT=1 tl.multiple_of(a, 4) (true of a, false of b); HINT=2 tl.multiple_of(b, 2) (true of b)")
        for h in (0, 1, 2):
            p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), str(h)], capture_output=True, text=True)
            sys.stdout.write(p.stdout or (f"HINT={h} no output: " + p.stderr[-400:] + "\n"))

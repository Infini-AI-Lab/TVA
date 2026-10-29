"""The AxisInfo constructor's global divisibility (AxisInfo.h:31-40, 92-101), on the device (lr-8b).

A [16, 64] block of float16 pointers p = a + r[:, None] * S + c[None, :] (dimension 1 fastest).
`tl.multiple_of(p, [16, 1])` says nothing about dimension 1 under a "fastest dimension only"
reading; under the library's reading it promises that every pointer is 16-byte aligned (false
for every S). If the compiler uses dimension 0's divisibility only along dimension 0, the loads
along dimension 1 stay as without a hint (scalar for S = 36). Each case in its own process.
"""
import os, re, subprocess, sys
import torch
import triton
import triton.language as tl


@triton.jit
def k_rows(a_ptr, out_ptr, S, MODE: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
    r = tl.arange(0, BM)
    c = tl.arange(0, BK)
    p = a_ptr + r[:, None] * S + c[None, :]
    if MODE == 1:
        tl.multiple_of(p, [16, 1])
    v = tl.load(p)
    tl.store(out_ptr + r[:, None] * BK + c[None, :], v)


def run(mode, S):
    BM, BK = 16, 64
    a = torch.arange(BM * S + BK, device="cuda", dtype=torch.float32).to(torch.float16)
    out = torch.zeros(BM * BK, device="cuda", dtype=torch.float16)
    host = a.cpu()
    ref = torch.stack([host[r * S:r * S + BK] for r in range(BM)]).flatten()
    ck = k_rows[(1,)](a, out, S, MODE=mode, BM=BM, BK=BK, num_warps=1)
    loads = sorted(set(re.findall(r"ld\.global(?:\.[a-z0-9]+)+", ck.asm["ptx"])))
    try:
        torch.cuda.synchronize()
        res = "correct" if torch.equal(out.cpu(), ref) else "WRONG"
    except Exception as e:
        res = "error: " + str(e).splitlines()[0][:100]
    print(f"hint={'[16, 1]' if mode else 'none'} S={S}: loads {loads}; {res}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        run(int(sys.argv[1]), int(sys.argv[2]))
    else:
        print("triton", triton.__version__, torch.cuda.get_device_name(0))
        for mode in (0, 1):
            for S in (64, 36):
                p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), str(mode), str(S)],
                                   capture_output=True, text=True)
                sys.stdout.write(p.stdout or ("no output: " + p.stderr[-300:] + "\n"))

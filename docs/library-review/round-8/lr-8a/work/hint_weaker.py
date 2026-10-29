"""Are the library's hint readings (SEMANTICS "Compiler hints", L2) stronger than Triton relies on?

lr-8a, round 8. Each case runs once, in its own process, on GPU 0, with one warp. It prints the
global loads of the PTX and the result against the source's meaning with every hint dropped.
For each case: the library's reading (L2: `multiple_of` without `max_contiguous` promises every
element; `max_contiguous` promises the tensor's own contiguity, `ContigIs`) and a weaker reading
(`max_contiguous` with `multiple_of` on the same value: groups of C consecutive, `ContigAlong`, and
v divides each group's first element; `multiple_of` alone on `base + tl.arange(0, B)`: v divides
the elements at multiples of B, B being the contiguity the analysis gives an arange).

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I <this file>   (from TVA's root)
"""
import json, os, re, subprocess, sys

import torch
import triton
import triton.language as tl


@triton.jit
def k_1d(x_ptr, out_ptr, MODE: tl.constexpr, B: tl.constexpr):
    offs = tl.arange(0, B)
    p = x_ptr + offs
    if MODE == 1:
        p = tl.multiple_of(p, 16)                              # the common 1-D idiom
    if MODE == 2:
        offs = tl.max_contiguous(tl.multiple_of(offs, 16), 16)  # C = 16 below offs' own 128
        p = x_ptr + offs
    tl.store(out_ptr + tl.arange(0, B), tl.load(p))


@triton.jit
def k_rows(a_ptr, out_ptr, S, MODE: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
    r = tl.arange(0, BM)
    c = tl.arange(0, BK)
    p = a_ptr + r[:, None] * S + c[None, :]
    if MODE == 1:
        p = tl.max_contiguous(tl.multiple_of(p, [1, 16]), [1, 16])   # C = 16 below the rows' 64
    v = tl.load(p)
    tl.store(out_ptr + r[:, None] * BK + c[None, :], v)


@triton.jit
def k_rr(x_ptr, out_ptr, MODE: tl.constexpr):
    rr = tl.arange(8, 16) - 8          # [0..7]: inferred contiguity 8, divisibility 8
    if MODE == 1:
        rr = tl.max_contiguous(tl.multiple_of(rr, 1), 1)    # both hints: groups of 1, divisor 1
    if MODE == 2:
        rr = tl.max_contiguous(rr, 1)                       # as evidence/hint-rank2 C1
    c = tl.arange(0, 64)
    v = tl.load(x_ptr + rr[:, None] + c[None, :])
    tl.store(out_ptr + tl.arange(0, 8)[:, None] * 64 + c[None, :], v)


CASES = {
    "W1": ("1-D p = x + arange(128), multiple_of(p, 16), x 16-byte aligned", "false", "true"),
    "W2": ("the same, x 4 bytes past a 16-byte boundary", "false", "false"),
    "W3": ("1-D max_contiguous(multiple_of(offs, 16), 16) on offs = arange(128), x aligned", "false", "true"),
    "W4": ("the same, x 4 bytes past (the hint is on offs, which stays true)", "false", "true"),
    "W5": ("[16, 64] fp16 rows, max_contiguous(multiple_of(p, [1, 16]), [1, 16]), S = 64", "false", "true"),
    "W6": ("the same, S = 36", "false", "false"),
    "W7": ("rr = [0..7], max_contiguous(multiple_of(rr, 1), 1)", "false", "true"),
    "W8": ("rr = [0..7], max_contiguous(rr, 1) (hint-rank2 C1)", "false", "false (alone: needs rr's own contiguity)"),
}


def run_case(name):
    dev = "cuda"
    if name in ("W1", "W2", "W3", "W4"):
        mode = 1 if name in ("W1", "W2") else 2
        off = 1 if name in ("W2", "W4") else 0
        base = torch.arange(256, device=dev, dtype=torch.float32)
        x = base[off:]
        out = torch.zeros(128, device=dev, dtype=torch.float32)
        ref = x[:128].cpu()
        ck = k_1d[(1,)](x, out, MODE=mode, B=128, num_warps=1)
    elif name in ("W5", "W6"):
        S = 64 if name == "W5" else 36
        BM, BK = 16, 64
        a = torch.arange(BM * S + BK, device=dev, dtype=torch.float32).to(torch.float16)
        out = torch.zeros(BM * BK, device=dev, dtype=torch.float16)
        host = a.cpu()
        ref = torch.stack([host[r * S:r * S + BK] for r in range(BM)]).flatten()
        ck = k_rows[(1,)](a, out, S, MODE=1, BM=BM, BK=BK, num_warps=1)
    else:
        mode = 1 if name == "W7" else 2
        x = torch.arange(128, device=dev, dtype=torch.float32)
        out = torch.zeros(512, device=dev, dtype=torch.float32)
        host = x.cpu()
        ref = torch.stack([host[r:r + 64] for r in range(8)]).flatten()
        ck = k_rr[(1,)](x, out, MODE=mode, num_warps=1)
    ptx = ck.asm["ptx"]
    loads = sorted(set(re.findall(r"ld\.global(?:\.[a-z0-9]+)+", ptx)))
    try:
        torch.cuda.synchronize()
        got = out.cpu()
        result = "correct" if torch.equal(got, ref) else "WRONG"
    except Exception as e:
        result = "error: " + str(e).splitlines()[0][:120]
    return {"loads": loads, "result": result}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        try:
            res = run_case(sys.argv[2])
        except Exception as e:
            res = {"loads": [], "result": "error: " + str(e).splitlines()[0][:120]}
        print("RESULT " + json.dumps(res))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name, (desc, lib, weak) in CASES.items():
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                           capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        res = json.loads(line[len("RESULT "):]) if line else {"loads": [], "result": "process failed"}
        print(f"{name}: {desc}\n  library's reading: {lib}; weaker reading: {weak}\n"
              f"  result: {res['result']}; loads: {', '.join(res['loads']) or '-'}")


if __name__ == "__main__":
    main()

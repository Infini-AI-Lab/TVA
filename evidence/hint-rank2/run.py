"""What does a compiler hint promise, in terms of the hinted tensor's values?

Library evidence for library issue L2 (round 8). Each case compiles a small kernel with Triton
3.8.0 and launches it once on GPU 0, in its own process (a misaligned access ends the process's
CUDA context). It reports the global loads of the PTX, the result against the host's reference
(computed from the source's meaning, every hint dropped), and the launch's error if any.

Run: .venv/bin/python -I evidence/hint-rank2/run.py  (from TVA's root; CUDA_VISIBLE_DEVICES=0)
"""
import json, os, re, subprocess, sys

import torch
import triton
import triton.language as tl


@triton.jit
def k_rows(a_ptr, out_ptr, S, MODE: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
    # Tutorial 08's pattern, outside a loop: a [BM, BK] block of float16 rows `S` elements apart.
    r = tl.arange(0, BM)
    c = tl.arange(0, BK)
    p = a_ptr + r[:, None] * S + c[None, :]
    if MODE == 1:
        tl.multiple_of(p, [16, 16])                                   # tutorial 08's hint
    if MODE == 2:
        p = tl.max_contiguous(tl.multiple_of(p, [1, 16]), [1, BK])    # row starts 16-byte aligned
    v = tl.load(p)
    tl.store(out_ptr + r[:, None] * BK + c[None, :], v)


@triton.jit
def k_contig1(x_ptr, out_ptr, MODE: tl.constexpr):
    # rr = [0, 1, ..., 7] built so that the compiler infers its divisibility (8) and contiguity (8).
    rr = tl.arange(8, 16) - 8
    if MODE == 1:
        rr = tl.max_contiguous(rr, 1)       # true in the documented sense: groups of 1
    if MODE == 2:
        rr = tl.max_contiguous(rr, 8)       # rr's own contiguity
    c = tl.arange(0, 64)
    v = tl.load(x_ptr + rr[:, None] + c[None, :])
    tl.store(out_ptr + tl.arange(0, 8)[:, None] * 64 + c[None, :], v)


@triton.jit
def k_tile(x_ptr, out_ptr, N, BN: tl.constexpr, HINT: tl.constexpr):
    # Tutorial 09's pattern: offsets clamped to 0 past N, then hinted as one aligned run.
    offs = tl.program_id(0) * BN + tl.arange(0, BN)
    offs = tl.where(offs < N, offs, 0)
    if HINT:
        offs = tl.max_contiguous(tl.multiple_of(offs, BN), BN)
    tl.store(out_ptr + tl.arange(0, BN), tl.load(x_ptr + offs))


@triton.jit
def k_const(out_ptr, MODE: tl.constexpr, B: tl.constexpr):
    offs = tl.arange(0, B)
    if MODE == 1:
        y = tl.max_constancy(offs, 4)          # false: offs is not constant in groups of 4
    elif MODE == 2:
        y = tl.max_constancy(offs // 4, 4)     # true
    else:
        y = offs // 4
    tl.store(out_ptr + offs, y * 3 + 1)


CASES = {
    "R1": ("rank 2, tutorial 08's multiple_of(p, [16, 16]), S = 64 (rows 128 bytes apart)",
           "false (p[0, 1] = a + 2 bytes)"),
    "R2": ("the same, S = 36 (rows 72 bytes apart)", "false"),
    "R3": ("the same block, no hint, S = 64", "-"),
    "R4": ("max_contiguous(multiple_of(p, [1, 16]), [1, 64]), S = 64", "true"),
    "R5": ("the same hints, S = 36", "false (row 1 starts 72 bytes in)"),
    "R6": ("the same block, no hint, S = 36", "-"),
    "C1": ("max_contiguous(rr, 1) on rr = [0..7]: true for groups of 1, but rr's contiguity is 8", "false (not rr's contiguity)"),
    "C2": ("max_contiguous(rr, 8), rr's own contiguity", "true"),
    "C3": ("no hint on rr", "-"),
    "T1": ("tutorial 09's clamped offsets, full tile (N = 64, BN = 64)", "true"),
    "T2": ("the same, partial tile (N = 40): offsets [0..39, 0, ..., 0]", "false"),
    "T3": ("the same partial tile without the hints", "-"),
    "K1": ("max_constancy(offs, 4) on offs = [0..127]", "false"),
    "K2": ("max_constancy(offs // 4, 4)", "true"),
    "K3": ("offs // 4, no hint", "-"),
}


def run_case(name):
    dev = "cuda"
    if name.startswith("R"):
        S = 36 if name in ("R2", "R5", "R6") else 64
        mode = {"R1": 1, "R2": 1, "R3": 0, "R4": 2, "R5": 2, "R6": 0}[name]
        BM, BK = 16, 64
        a = torch.arange(BM * S + BK, device=dev, dtype=torch.float32).to(torch.float16)
        out = torch.zeros(BM * BK, device=dev, dtype=torch.float16)
        host = a.cpu()
        ref = torch.stack([host[r * S:r * S + BK] for r in range(BM)]).flatten()
        ck = k_rows[(1,)](a, out, S, MODE=mode, BM=BM, BK=BK, num_warps=1)
    elif name.startswith("C"):
        mode = {"C1": 1, "C2": 2, "C3": 0}[name]
        x = torch.arange(128, device=dev, dtype=torch.float32)
        out = torch.zeros(512, device=dev, dtype=torch.float32)
        host = x.cpu()
        ref = torch.stack([host[r:r + 64] for r in range(8)]).flatten()
        ck = k_contig1[(1,)](x, out, MODE=mode, num_warps=1)
    elif name.startswith("T"):
        N = 64 if name == "T1" else 40  # T2, T3
        BN = 64
        x = (torch.arange(256, device=dev, dtype=torch.float32) + 1000).to(torch.float16)
        out = torch.zeros(BN, device=dev, dtype=torch.float16)
        host = x.cpu()
        idx = [j if j < N else 0 for j in range(BN)]
        ref = host[idx]
        ck = k_tile[(1,)](x, out, N, BN=BN, HINT=name != "T3", num_warps=1)
    else:
        mode = {"K1": 1, "K2": 2, "K3": 0}[name]
        B = 128
        out = torch.zeros(B, device=dev, dtype=torch.int32)
        o = torch.arange(B, dtype=torch.int32)
        ref = (o if mode == 1 else o // 4) * 3 + 1
        ck = k_const[(1,)](out, MODE=mode, B=B, num_warps=1)
    ptx = ck.asm["ptx"]
    loads = sorted(set(re.findall(r"ld\.global(?:\.[a-z0-9]+)+", ptx)))
    try:
        torch.cuda.synchronize()
        got = out.cpu()
        if torch.equal(got, ref):
            result = "correct"
        else:
            bad = [i for i in range(got.numel()) if got[i].item() != ref[i].item()]
            result = (f"WRONG at {len(bad)} of {got.numel()} elements; first: index {bad[0]} "
                      f"got {got[bad[0]].item()} expected {ref[bad[0]].item()}")
            if name == "T2":
                result += "; wrong lanes: " + str(bad) + ", each holding x[1]: " + str(
                    all(got[j].item() == host[1].item() for j in bad))
    except Exception as e:  # the launch's error, reported to the host
        result = "error: " + str(e).splitlines()[0][:160]
    return {"loads": loads, "result": result}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        try:
            res = run_case(sys.argv[2])
        except Exception as e:
            res = {"loads": [], "result": "error: " + str(e).splitlines()[0][:160]}
        print("RESULT " + json.dumps(res))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name, (desc, reading) in CASES.items():
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                           capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line is None:
            err = (p.stderr.strip().splitlines() or ["(no output)"])[-1]
            res = {"loads": [], "result": "process failed: " + err[:160]}
        else:
            res = json.loads(line[len("RESULT "):])
        print(f"\n{name}: {desc}")
        print(f"  library's reading: {reading}")
        print(f"  result:  {res['result']}")
        print(f"  loads:   {', '.join(res['loads']) or '-'}")


if __name__ == "__main__":
    main()

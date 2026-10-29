"""lr-9a: three probes.

1. Which input of evidence/num-dot gives DotF32Ieee's largest ratio (the README attributes its
   0.499 to the absolute term on tiny inputs): every input kind of that script at K = 8, 16 x 16.
2. Does `tl.dot_scaled(…, "e2m1", …, "e2m1")` compile and run on sm_90 (Triton's emulation)?
3. Tutorial 10's `block_scaled_matmul_kernel` with each of its formats on the H200, with the
   tutorial's own configuration (exclusions.md: fp4 launches are rejected on the H200).

Run from this folder: CUDA_VISIBLE_DEVICES=0 python -I probes.py
"""
import importlib.util
import os
import sys
sys.dont_write_bytecode = True  # import nothing into a cache next to TVA's files
import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", ".."))


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(TVA, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@triton.jit
def fp4dot(a_ptr, sa_ptr, b_ptr, sb_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm = tl.arange(0, M); rn = tl.arange(0, N)
    rk = tl.arange(0, K // 2); rs = tl.arange(0, K // 32)
    a = tl.load(a_ptr + rm[:, None] * (K // 2) + rk[None, :])
    sa = tl.load(sa_ptr + rm[:, None] * (K // 32) + rs[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    sb = tl.load(sb_ptr + rn[:, None] * (K // 32) + rs[None, :])
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1"))


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}\n")
    R = load("numdot", os.path.join("evidence", "num-dot", "run.py"))
    torch.manual_seed(1)
    M = N = 16; K = 8
    print("## 1. DotF32Ieee, K = 8, 16 x 16, 1 warp: the largest ratio per input kind")
    print("| kind | largest ratio | error / (2^-24 mag) at that element | error / 2^-150 |")
    print("|---|---|---|---|")
    for kind in R.KINDS:
        a, b, c = R.gen(kind, M, N, K, "float32", "float32")
        o, path = R.run(a, b, c, 1, "ieee")
        ad, bd, cd = a.double(), b.double(), c.double()
        exact = ad @ bd + cd
        mag = ad.abs() @ bd.abs() + cd.abs()
        err = (o.double() - exact).abs()
        _, rel, ab = R.fact("float32", "float32", K)
        ratio = (err + (K + 1) * 2.0 ** -52 * mag) / (rel * mag + ab)
        i = int(ratio.argmax()); r, s = divmod(i, N)
        ru = err[r, s].item() / (2.0 ** -24 * mag[r, s].item()) if mag[r, s] > 0 else 0.0
        print(f"| {kind} | {ratio.max().item():.4f} | {ru:.3f} | {err[r, s].item() / 2.0 ** -150:.3g} |")
    print(f"\n(K - 1) / (2 (K + 1)) = {(K - 1) / (2 * (K + 1)):.4f}")
    print("\n## 2. tl.dot_scaled e2m1 by e2m1 on sm_90 (all elements 1.0, scales 1, K = 64)")
    a = torch.full((64, 32), 0x22, dtype=torch.uint8, device="cuda"); b = torch.full((32, 64), 0x22, dtype=torch.uint8, device="cuda")
    sa = torch.full((64, 2), 127, dtype=torch.uint8, device="cuda"); sb = torch.full((64, 2), 127, dtype=torch.uint8, device="cuda")
    o = torch.empty(64, 64, device="cuda")
    try:
        fp4dot[(1,)](a, sa, b, sb, o, 64, 64, 64, num_warps=4)
        torch.cuda.synchronize()
        print(f"compiled and ran: o[0, 0] = {o[0, 0].item()} (exact 64)")
    except Exception as e:
        print(f"rejected: {type(e).__name__}")
    print("\n## 3. Tutorial 10's block_scaled_matmul_kernel, its own configuration, M = N = 256, K = 512")
    t10 = load("t10", os.path.join("tutorials", "10-block-scaled-matmul.py"))
    torch.manual_seed(42)
    for fmt in ["mxfp8", "mixed", "mxfp4", "nvfp4"]:
        try:
            res = t10.initialize_block_scaled(256, 256, 512, fmt, compute_reference=True)
            a_desc, a_scale_desc, b_desc, b_scale_desc, rep_m, rep_n, rep_k, configs, reference = res[:9]
            out = t10.block_scaled_matmul(a_desc, a_scale_desc, b_desc, b_scale_desc, torch.float16, 256, 256, 512,
                                          rep_m, rep_n, rep_k, configs)
            torch.cuda.synchronize()
            print(f"- {fmt}: ran; max |reference - output| = {(reference - out.float()).abs().max().item():.4g}, "
                  f"max |reference| = {reference.abs().max().item():.4g}")
        except Exception as e:
            print(f"- {fmt}: rejected on the host: {type(e).__name__}: {str(e).splitlines()[0][:120]}")


if __name__ == "__main__":
    main()

"""GPU check of the prepared claim of `attn_bwd_pre_f16` (claims/README.md, check 4).

Runs the kernel's own host (`source/06-fused-attention.py`, imported by path): the test's launches through `attention(q, k, v, causal, sm_scale).backward(dout)` (the
forward, then `_attention.backward`, which launches `_attn_bwd_preprocess` at lines 600-607), and
the benchmark's sizes and the edges of `Inst.ok` by the same launch line (the grid
`(N_CTX // 128, BATCH * N_HEAD)`, `BLOCK_M = 128`, `HEAD_DIM`) on host-made `o`, `do` (the
benchmark's full tensors do not fit the 4 GB budget through the whole backward pass). For each
launch it computes `Post`'s reference `dotSum` and tolerance `tol HEAD_DIM (dotAbs)` in float64
exactly as `Claim.lean` states them, and reports the largest error / tolerance over every row, that
every row with finite inputs is finite, and the frame (`o`, `do` unchanged; in the direct launches
`delta` is a view of a larger buffer whose tail keeps a sentinel); and that wrong outputs (`delta`
rounded to `float16`, a column dropped or doubled, the wrong row of `do`) fail `Post`.

Run: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "06-fused-attention.py")
spec = importlib.util.spec_from_file_location("fused_attention_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = "cuda"
U, ETA = 2.0 ** -24, 2.0 ** -150


def tol(D, B):
    """`tol D B` of Claim.lean: errBound(3u, 4eta, ceil(log2 D), D-1, (1+u)B + D eta) + (uB + D eta)."""
    g = (1 + 3 * U) ** ((D - 1).bit_length())   # ceil(log2 D) for D >= 1
    A = (1 + U) * B + D * ETA
    return (g - 1) * A + 4 * ETA * (D - 1) * g + (U * B + D * ETA)


WORST = {"r": 0.0, "ok": True, "at": None}


def same(a, b):
    """Bitwise equality (a NaN equals itself)."""
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def check(name, o, do, delta, chunk=1 << 17):
    """`Post` on every row: `o`, `do` `[R, D]` float16, `delta` `[R]` float32 (in row chunks)."""
    D = o.shape[-1]
    O, DO, C = o.reshape(-1, D), do.reshape(-1, D), delta.reshape(-1)
    r_case, at = 0.0, None
    for a in range(0, C.numel(), chunk):
        o2, do2, c = O[a:a + chunk].detach().double(), DO[a:a + chunk].detach().double(), C[a:a + chunk].detach()
        finite_in = torch.isfinite(o2).all(1) & torch.isfinite(do2).all(1)
        prod = o2 * do2
        ref, B = prod.sum(1), prod.abs().sum(1)
        t = tol(D, B)
        if not bool(torch.isfinite(c[finite_in]).all()):
            print(f"  {name}: a row with finite inputs is not finite  FAILED")
            WORST["ok"] = False
        err = (c.double() - ref).abs()
        ratio = torch.where(finite_in, err / t, torch.zeros_like(err))
        r = float(ratio.max())
        if r > r_case:
            i = int(ratio.argmax())
            r_case, at = r, (name, float(err[i]), float(t[i]), float(B[i]), float(t[i] / max(B[i], 1e-300)))
    if r_case > WORST["r"]:
        WORST["r"], WORST["at"] = r_case, at
    print(f"{name}: rows {C.numel()}, worst err/tol {r_case:.3e}" + (f" (tol/B {at[4]:.2e})" if at else ""))
    if r_case >= 1:
        WORST["ok"] = False


class Recorder:
    """Stands for the module's `_attn_bwd_preprocess` while the host runs: `kernel[grid](...)`
    records the launch's tensors and runs the real JIT function's launch unchanged."""

    def __init__(self, real):
        self.real, self.calls = real, []

    def __getitem__(self, grid):
        launcher = self.real[grid]

        def run(o, do, delta, *a, **kw):
            snap = (o.clone(), do.clone())
            out = launcher(o, do, delta, *a, **kw)
            torch.cuda.synchronize()
            self.calls.append((o, do, delta, snap))
            return out

        return run


def host_case(Z, H, N, D, causal, gen_seed=20):
    """One launch of the test (lines 631-695): its inputs and its call of the host."""
    torch.manual_seed(gen_seed)
    dtype = torch.float16
    q = torch.empty((Z, H, N, D), dtype=dtype, device=DEV).normal_(mean=0.0, std=0.5).requires_grad_()
    k = torch.empty((Z, H, N, D), dtype=dtype, device=DEV).normal_(mean=0.0, std=0.5).requires_grad_()
    v = torch.empty((Z, H, N, D), dtype=dtype, device=DEV).normal_(mean=0.0, std=0.5).requires_grad_()
    dout = torch.randn_like(q)
    rec = Recorder(mod._attn_bwd_preprocess)
    mod._attn_bwd_preprocess = rec
    try:
        tri_out = mod.attention(q, k, v, causal, 0.5, False).half()
        tri_out.backward(dout)
    finally:
        mod._attn_bwd_preprocess = rec.real
    torch.cuda.synchronize()
    (o, do, delta, snap), = rec.calls
    if not (same(snap[0], o) and same(snap[1], do)):
        print("  FRAME VIOLATED (an input changed)")
        WORST["ok"] = False
    check(f"test Z{Z} H{H} N{N} D{D} causal={causal}", o, do, delta)
    del q, k, v, dout, tri_out, o, do, delta, snap, rec
    torch.cuda.empty_cache()


SENT = -7.0e30


def direct_case(name, Z, H, N, D, o=None, do=None, scale=1.0):
    """The launch of lines 600-607 on host-made `o`, `do`; `delta` a view of a longer buffer."""
    if o is None:
        o = (torch.randn((Z, H, N, D), device=DEV) * scale).to(torch.float16)
        do = (torch.randn((Z, H, N, D), device=DEV) * scale).to(torch.float16)
    snap = (o.clone(), do.clone())
    store = torch.full((Z * H * N + 1024,), SENT, dtype=torch.float32, device=DEV)
    delta = store[:Z * H * N].view(Z, H, N)
    mod._attn_bwd_preprocess[(N // 128, Z * H)](o, do, delta, Z, H, N, BLOCK_M=128, HEAD_DIM=D)
    torch.cuda.synchronize()
    if not (same(snap[0], o) and same(snap[1], do)):
        print(f"  {name}: FRAME VIOLATED (an input changed)")
        WORST["ok"] = False
    if not bool((store[Z * H * N:] == SENT).all()):
        print(f"  {name}: FRAME VIOLATED (a byte past delta changed)")
        WORST["ok"] = False
    check(name, o, do, delta)
    del o, do, snap, store, delta
    torch.cuda.empty_cache()


NEG_OK = True


def negatives(D, Z=1, H=4, N=1024):
    """Wrong outputs `Post` must reject, on the benchmark's input distribution: the kernel's
    `delta` rounded to `float16`, the sum without the last column or with it doubled (exact
    otherwise, then `float32`), and the sum against the next row of `do`. Each must exceed `tol` on
    some row; prints the fraction of rows rejected."""
    global NEG_OK
    o = torch.randn((Z, H, N, D), device=DEV).to(torch.float16)
    do = torch.randn((Z, H, N, D), device=DEV).to(torch.float16)
    delta = torch.empty((Z, H, N), dtype=torch.float32, device=DEV)
    mod._attn_bwd_preprocess[(N // 128, Z * H)](o, do, delta, Z, H, N, BLOCK_M=128, HEAD_DIM=D)
    torch.cuda.synchronize()
    O, DO = o.reshape(-1, D).double(), do.reshape(-1, D).double()
    prod = O * DO
    ref, B = prod.sum(1), prod.abs().sum(1)
    t = tol(D, B)
    wrong = {
        "delta rounded to float16": delta.reshape(-1).half().double(),
        "last column dropped": (ref - prod[:, -1]).float().double(),
        "last column doubled": (ref + prod[:, -1]).float().double(),
        "next row of do": (O * DO.roll(-1, 0)).sum(1).float().double(),
    }
    for k, w in wrong.items():
        r = (w - ref).abs() / t
        rej = bool((r > 1).any())
        NEG_OK = NEG_OK and rej
        print(f"negative D{D}: {k:26s} rows rejected {float((r > 1).double().mean()):.3f}: "
              f"{'rejected' if rej else 'ACCEPTED'}", flush=True)
    del o, do, delta


def main():
    # the test's launches (bwd, triton-fp16)
    for Z in (1, 4):
        for H in (2, 48):
            for N in (128, 1024, 4096):
                for D in (64, 128):
                    for causal in (False, True):
                        host_case(Z, H, N, D, causal)
    # the benchmark's sizes (BATCH 4, N_HEADS 32, N_CTX 2^10 .. 2^14), o and do standard normal
    for e in range(10, 15):
        for D in (64, 128):
            direct_case(f"bench N{2 ** e} D{D}", 4, 32, 2 ** e, D)
    # edges of Inst.ok: the other head dims the forward admits; Z*H = 1; N_CTX = 0 (no program)
    for D in (16, 32, 256):
        direct_case(f"edge D{D}", 2, 3, 1024, D)
    direct_case("edge ZH1", 1, 1, 128, 128)
    direct_case("edge N0", 2, 2, 0, 64)
    # edges of the values: the largest float16 magnitudes (room), cancellation, subnormals,
    # non-finite rows (Post says only that a value is stored there)
    direct_case("pre-big", 1, 4, 512, 256, scale=2.0 ** 13)
    o = torch.randn((1, 2, 256, 128), device=DEV).to(torch.float16)
    do = o.clone()
    do[..., 64:] = -do[..., :64]
    o[..., 64:] = o[..., :64]  # sum o*do = 0 exactly: the error must be within tol of 0
    direct_case("pre-cancel", 1, 2, 256, 128, o=o, do=do)
    o = (torch.randn((1, 2, 256, 64), device=DEV) * 2.0 ** -12).to(torch.float16)
    do = (torch.randn((1, 2, 256, 64), device=DEV) * 2.0 ** -12).to(torch.float16)
    direct_case("pre-subnormal", 1, 2, 256, 64, o=o, do=do)
    o = torch.randn((1, 2, 256, 64), device=DEV).to(torch.float16)
    do = torch.randn((1, 2, 256, 64), device=DEV).to(torch.float16)
    o[0, 0, 3, 5] = float("inf")
    do[0, 1, 7, 0] = float("nan")
    direct_case("pre-nonfinite", 1, 2, 256, 64, o=o, do=do)
    # wrong outputs Post must reject
    for D in (64, 128):
        negatives(D)
    print(f"WORST: err/tol {WORST['r']:.3e} (margin {1 / max(WORST['r'], 1e-300):.1f}x) at "
          f"(case, err, tol, B, tol/B) = {WORST['at']}; finiteness and frame "
          f"{'hold' if WORST['ok'] else 'FAIL'}; negatives {'all rejected' if NEG_OK else 'SOME ACCEPTED'}")
    sys.exit(0 if WORST["ok"] and NEG_OK else 1)


if __name__ == "__main__":
    main()

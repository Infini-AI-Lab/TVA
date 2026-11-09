"""GPU check of the prepared claim `Claims.ChunkBwdDvLocal` (claims/README.md, check 4).

Runs the kernel's own host, `chunk_bwd_dv_local` of `source/chunk_o.py` (imported by path), on
SETUP.md's launches (the host's test cases, the code base's benchmark shapes, gated DeltaNet's
default heads, a packed batch of 2^16 tokens with `int32` and `int64` tables, `B*HV = 65535`), each
without and with a gate, and on launches at the edges of `Inst.ok` and `Pre`, and checks `Post` as
it is stated, for every sequence `n`, token `p` of it, value head `hv` and column `e`:

* `dv[p, hv, e]` is finite and `|dv - ref| <= tol`, with `ref`, `Aex`, `kqMag`, `wErr`, `wUp`,
  `eA` and `tol` exactly as `Claim.lean` defines them, computed in `float64` from the host's
  inputs: tokens are grouped by chunk from `cu_seqlens` (or the rows) alone, never from
  `chunk_indices`; `float64`'s own rounding (`2^-52` relative per operation, over at most 256 + 64
  terms) is far below `tol`'s `2^-8` terms;
* `ValsOk` (`Pre`) holds on the inputs: finite values, `|do| <= 2^100`, the gate differences
  `<= 64`, `kqMag <= 2^100`, `sig*kqMag*(1 + wUp) <= 2^100`, and the per-output room `<= 2^120`;
* the host's tables satisfy `TablesOk`; the inputs are unchanged; and a direct launch with the
  host's own arguments and `dv` prefilled with a sentinel leaves every element outside the written
  set (`written`) unchanged.

It reports, for each case, the largest error / tolerance (`dv`, the only output), and runs negative
tests (outputs the check must reject: one element moved by twice its tolerance, the ungated output
against the gated specification, the output of chunk size 32 against chunk size 64, an unwritten
element changed). Finally it shows, in a child process, that a launch with `B*HV = 65536` fails
(the grid's `y` limit; README, Exclusions).

Usage: `CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import math
import pathlib
import subprocess

import torch
import triton

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "source" / "chunk_o.py"
spec = importlib.util.spec_from_file_location("chunk_o_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEVICE = triton.runtime.driver.active.get_active_torch_device()
U = 2.0**-24
LN2 = math.log(2.0)
RCP_LN2 = 1.4426950216
SENT = -12345.0  # a bfloat16 value no output takes here
F64 = torch.float64


def lift(x):
    """`Rules.Num.liftMag .bf16` of float64 copies of bfloat16 values."""
    a = x.abs()
    return torch.where(a == 0, a, a.clamp(min=2.0**-126))


def seqs(B, T, cu):
    """`(lo, hi)` of every sequence in the flat token axis (`nSeq`, `lo`, `hi`)."""
    if cu is None:
        return [(b * T, b * T + T) for b in range(B)]
    return [(cu[n], cu[n + 1]) for n in range(len(cu) - 1)]


def tables_ok(cu, ci, T, BT):
    """`TablesOk`, on the host's tables."""
    N = len(cu) - 1
    if not (cu[0] >= 0 and all(cu[i] <= cu[i + 1] for i in range(N)) and cu[N] <= T):
        return False
    rows = [tuple(r) for r in ci]
    if len(set(rows)) != len(rows):
        return False
    for n, c in rows:
        if not (0 <= n < N and c >= 0 and c * BT < cu[n + 1] - cu[n]):
            return False
    want = {(n, c) for n in range(N) for c in range((cu[n + 1] - cu[n] + BT - 1) // BT)}
    return want <= set(rows)


def post_check(q, k, do, g, scale, cu, BT, BK, dv):
    """`Post`'s value clause and `ValsOk`. Returns (worst |dv - ref| / tol, every element finite,
    ValsOk holds, the worst element's (|dv - ref|, tol))."""
    B, T, H, K = k.shape
    HV, V = do.shape[2], do.shape[3]
    G = HV // H
    NTOK = B * T
    qf, kf = q.reshape(NTOK, H, K), k.reshape(NTOK, H, K)
    df, of = do.reshape(NTOK, HV, V), dv.reshape(NTOK, HV, V)
    gf = None if g is None else g.reshape(NTOK, HV)
    starts, ends = [], []
    for lo, hi in seqs(B, T, cu):
        for c0 in range(lo, hi, BT):
            starts.append(c0)
            ends.append(min(c0 + BT, hi))
    if not starts:
        return 0.0, True, True, (0.0, 0.0)
    nK = (K + BK - 1) // BK
    epsA = (BK + 8 + 2 * nK) / 2.0**22
    sig = float(scale)
    nb = max(1, 2**25 // (HV * BT * max(BT, V, K) * 8))  # chunks per batch: each float64 block <= 32 MiB
    worst, wpair, finite, vals = 0.0, (0.0, 0.0), True, True
    ar = torch.arange(BT, device=DEVICE)
    for b0 in range(0, len(starts), nb):
        st = torch.tensor(starts[b0:b0 + nb], device=DEVICE)
        en = torch.tensor(ends[b0:b0 + nb], device=DEVICE)
        pos = st[:, None] + ar[None, :]                                   # [C, BT]
        valid = pos < en[:, None]
        pc = pos.clamp(max=NTOK - 1)
        z = torch.zeros((), dtype=F64, device=DEVICE)
        kk = torch.where(valid[..., None, None], kf[pc].to(F64), z).repeat_interleave(G, dim=2)
        qq = torch.where(valid[..., None, None], qf[pc].to(F64), z).repeat_interleave(G, dim=2)
        dd = torch.where(valid[..., None, None], df[pc].to(F64), z)     # [C, BT, HV, V]
        out = of[pc].to(F64)
        vals &= bool(torch.isfinite(kk).all() and torch.isfinite(qq).all() and torch.isfinite(dd).all())
        vals &= bool((dd.abs() <= 2.0**100).all())
        # the pairs p <= j of one chunk, both tokens of the sequence
        M = ((ar[:, None] <= ar[None, :])[None, None] & (valid[:, None, :, None] & valid[:, None, None, :])
             ).expand(-1, HV, -1, -1)
        kq = torch.einsum('cihx,cjhx->chij', kk, qq)
        kqm = torch.einsum('cihx,cjhx->chij', lift(kk), lift(qq))
        if g is not None:
            gg = torch.where(valid[..., None], gf[pc].to(F64), z).transpose(1, 2)  # [C, HV, BT]
            vals &= bool(torch.isfinite(gg).all())
            dg = gg[:, :, None, :] - gg[:, :, :, None]                    # (p, j): g[j] - g[p]
            vals &= bool((dg[M] <= 64).all())
            dg = torch.where(M, dg, z)
            w = torch.exp2(dg)
            ew = w * (torch.expm1(LN2 * (U * dg.abs() + 2.0**-149)) * (1 + 2.0**-22) + 2.0**-22) + 2.0**-126
        else:
            w = torch.ones_like(kq)
            ew = torch.zeros_like(kq)
        wu = w + ew
        vals &= bool((kqm[M] <= 2.0**100).all() and (sig * kqm * (1 + wu))[M].max() <= 2.0**100)
        Mf = M.to(F64)
        A = sig * kq * w * Mf
        eA = (sig * kqm * (epsA * wu + ew) + wu * (sig / 2.0**110 + 2.0**-140) + 2.0**-140) * Mf
        room = torch.einsum('chij,cjhv->cihv', sig * kqm * wu * Mf, lift(dd))
        vals &= bool((room <= 2.0**120).all())
        ref = torch.einsum('chij,cjhv->cihv', A, dd)
        B1 = torch.einsum('chij,cjhv->cihv', A.abs(), dd.abs())
        Be = torch.einsum('chij,cjhv->cihv', eA, dd.abs())
        Bl = torch.einsum('chij,cjhv->cihv', A.abs() + eA, lift(dd))
        Dl = torch.einsum('chij,cjhv->cihv', Mf, lift(dd))
        tol = (2.0**-8 * ref.abs() + (1 + 2.0**-8) * (2.0**-8 * B1 + (1 + 2.0**-8) * Be + (BT + 3) / 2.0**22 * Bl)
               + 2.0**-100 * (1 + Dl))
        vm = valid[..., None, None].expand_as(out)
        finite &= bool(torch.isfinite(out)[vm].all())
        err = (out - ref).abs()
        ratio = torch.where(vm, err / tol, z)
        i = int(ratio.argmax())
        r = ratio.flatten()[i].item()
        if r > worst:
            worst, wpair = r, (err.flatten()[i].item(), tol.flatten()[i].item())
        del kk, qq, dd, out, kq, kqm, A, eA, ref, B1, Be, Bl, Dl, tol, err, ratio, room, w, ew, wu
    return worst, finite, vals, wpair


def direct(q, k, do, g, scale, cu_t, ci_t, BT):
    """The host's launch (source lines 213-240) with `dv` prefilled with a sentinel."""
    B, T, H, K = k.shape
    HV, V = do.shape[2], do.shape[3]
    BK = min(max(triton.next_power_of_2(K), 16), 128)
    BV = min(max(triton.next_power_of_2(V), 16), 128)
    NT = triton.cdiv(T, BT) if cu_t is None else len(ci_t)
    dv = torch.full_like(do, SENT)
    mod.chunk_bwd_kernel_dv_local[(NT, B * HV)](
        q=q, k=k, g=g, g_gamma=None, A=None, do=do, dv=dv, cu_seqlens=cu_t, chunk_indices=ci_t,
        scale=scale, T=T, H=H, HV=HV, K=K, V=V, BT=BT, BK=BK, BV=BV, USE_G=g is not None,
        USE_G_GAMMA=False, USE_A=False, IS_VARLEN=cu_t is not None, num_warps=4, num_stages=3)
    torch.cuda.synchronize()
    return dv


def frame_ok(dv, B, T, cu):
    w = torch.zeros(B * T, dtype=torch.bool, device=dv.device)
    for lo, hi in seqs(B, T, cu):
        w[lo:hi] = True
    rest = dv.reshape(B * T, -1)[~w]
    return bool((rest == SENT).all().item())


gen = torch.Generator().manual_seed(7)


def randn(*shape, scale=1.0):
    return (torch.randn(shape, generator=gen) * scale).to(DEVICE, torch.bfloat16)


def chunk_gate(B, T, HV, BT, cu, per_token=None):
    """The chunk-local cumulative sum of per-token log decays, times `RCP_LN2` (gated DeltaNet's
    forward, `chunk_local_cumsum(g, chunk_size, scale=RCP_LN2)`), in `float32`."""
    x = torch.nn.functional.logsigmoid(torch.randn((B * T, HV), generator=gen)) if per_token is None else per_token
    x = x.double()
    out = torch.zeros_like(x)
    for lo, hi in seqs(B, T, cu):
        for c0 in range(lo, hi, BT):
            c1 = min(c0 + BT, hi)
            out[c0:c1] = x[c0:c1].cumsum(0) * RCP_LN2
    return out.reshape(B, T, HV).float().to(DEVICE)


results = []
worst_all = 0.0
DATA = {}


PEAK = [0.0, ""]


def check(name, B, T, H, HV, K, V, BT, cu=None, idx=torch.int64, gates=(False, True), qs=1.0,
          ks=1.0, ds=1.0, scale=None, gate_fn=None, host=True, keep=False, positive=False):
    global worst_all
    torch.cuda.reset_peak_memory_stats()
    if positive:
        q = (torch.rand((B, T, H, K), generator=gen) * qs).to(DEVICE, torch.bfloat16)
        k = (torch.rand((B, T, H, K), generator=gen) * ks).to(DEVICE, torch.bfloat16)
        do = (torch.rand((B, T, HV, V), generator=gen) * ds).to(DEVICE, torch.bfloat16)
    else:
        q, k, do = randn(B, T, H, K, scale=qs), randn(B, T, H, K, scale=ks), randn(B, T, HV, V, scale=ds)
    scale = K ** -0.5 if scale is None else scale
    BK = min(max(triton.next_power_of_2(K), 16), 128)
    cu_t = None if cu is None else torch.tensor(cu, dtype=idx, device=DEVICE)
    ci_t = None if cu is None else mod.prepare_chunk_indices(cu_t, BT)
    ok = True
    line = []
    if cu is not None:
        tok = tables_ok(cu, ci_t.cpu().tolist(), T, BT)
        ok &= tok
        line.append(f"TablesOk {tok}")
    before = [x.cpu() for x in (q, k, do)]  # kept on the host, to stay under 4 GB of device memory
    for G in gates:
        g = (gate_fn(B, T, HV, BT, cu) if gate_fn else chunk_gate(B, T, HV, BT, cu)) if G else None
        g_before = None if g is None else g.cpu()
        if host:
            dv = mod.chunk_bwd_dv_local(q, k, do, g=g, scale=scale, cu_seqlens=cu_t, chunk_size=BT)
            torch.cuda.synchronize()
            ok &= dv.dtype == torch.bfloat16 and dv.shape == do.shape
        else:
            dv = direct(q, k, do, g, scale, cu_t, ci_t, BT)
        w, f, v, (e, t) = post_check(q, k, do, g, scale, cu, BT, BK, dv)
        fr = True
        if not host:
            fr = frame_ok(dv, B, T, cu)
        ok &= w <= 1.0 and f and v and fr
        if g is not None:
            ok &= torch.equal(g.cpu().view(torch.int32), g_before.view(torch.int32))
        worst_all = max(worst_all, w)
        line.append(f"{'gate' if G else 'no gate'}: {w:.3f} (err {e:.2e}, tol {t:.2e})"
                    + ("" if v else " ValsOk FALSE") + ("" if fr else " FRAME BROKEN"))
        if keep:
            DATA[(name, G)] = (q, k, do, g, scale, cu, BT, BK, dv)
        del dv, g
    ok &= all(torch.equal(a.cpu().view(torch.int16), b.view(torch.int16)) for a, b in zip((q, k, do), before))
    results.append(ok)
    tab = '' if cu is None else f" {str(idx)[6:]} N{len(cu) - 1}"
    print(f"{'ok ' if ok else 'BAD'} {name} B{B} T{T} H{H} HV{HV} K{K} V{V} BT{BT}{tab}: " + "; ".join(line),
          flush=True)
    peak = torch.cuda.max_memory_allocated() / 2**30
    if peak > PEAK[0]:
        PEAK[0], PEAK[1] = peak, name
    if not keep:
        torch.cuda.empty_cache()


# The host's test cases (source lines 278-290).
for case in [(1, 63, 1, 1, 64, 64, 64, None), (2, 500, 4, 4, 60, 60, 64, None),
             (2, 1000, 2, 8, 128, 128, 64, None), (3, 1024, 4, 4, 100, 100, 64, None),
             (1, 64, 2, 4, 32, 32, 16, None), (1, 64, 2, 4, 32, 32, 32, None),
             (2, 1024, 6, 6, 256, 512, 64, None), (1, 2048, 16, 32, 128, 128, 64, None),
             (1, 15, 4, 4, 60, 60, 64, [0, 15]), (1, 1000, 4, 4, 64, 64, 64, [0, 256, 500, 1000]),
             (1, 2000, 4, 4, 100, 100, 64, [0, 15, 100, 300, 1200, 2000]),
             (1, 1000, 2, 8, 64, 64, 64, [0, 256, 500, 1000]),
             (1, 4096, 4, 8, 64, 64, 64, [0, 1, 63, 64, 65, 1000, 4096])]:
    check("host", *case, keep=case[1] == 4096 or case[1] == 2048)

# The code base's benchmark shapes (benchmarks/ops/registry.py:203-210), K = V = D.
for B, T, H, D in [(1, 8192, 96, 128), (2, 16384, 16, 128), (4, 2048, 16, 128), (4, 4096, 64, 128),
                   (8, 2048, 32, 256), (8, 1024, 8, 64)]:
    check("bench", B, T, H, H, D, D, 64)

# Gated DeltaNet's default heads at its default context; a packed batch of 2^16 tokens with
# Qwen3-Next's grouped value heads, int32 and int64 tables; the largest grid y (B*HV = 65535).
check("gdn default", 8, 2048, 6, 6, 256, 512, 64)
for idx in (torch.int32, torch.int64):
    check("train 2^16", 1, 65536, 16, 32, 128, 128, 64, [0, 1, 64, 65, 4096, 30000, 65536], idx=idx)
check("rows 65535", 4369, 16, 15, 15, 16, 16, 16)

# Edges of Inst.ok: head dims 1, 15, 17 (masked K and V tails), K = 256 with BK = 128 (two trips),
# V = 1000 (eight V trips), HV/H = 8, empty and short sequences, sequences not covering [0, T),
# a permuted chunk_indices, chunk sizes 16 and 32 on a packed batch.
check("edge K1 V1", 2, 100, 2, 2, 1, 1, 64)
check("edge K15 V17", 2, 300, 3, 6, 15, 17, 32)
check("edge K256 V1000", 1, 200, 2, 2, 256, 1000, 64)
check("edge HV/H 8", 2, 130, 1, 8, 64, 64, 64)
check("edge empty/short", 1, 5000, 2, 4, 64, 64, 64, [0, 0, 1, 17, 17, 64, 65, 129, 4096, 5000])
check("edge BT16 packed", 1, 3000, 2, 4, 128, 128, 16, [0, 7, 1500, 3000], idx=torch.int32)
check("edge BT32 packed", 1, 3000, 2, 4, 128, 128, 32, [0, 7, 1500, 3000], idx=torch.int32)
check("edge not covering [0,T)", 1, 300, 2, 2, 64, 64, 64, [5, 20, 20, 37, 200], host=False)
check("edge fixed direct", 3, 333, 2, 4, 64, 96, 64, None, host=False)
# a permuted chunk_indices (direct launch)
q, k, do = randn(1, 900, 2, 64), randn(1, 900, 2, 64), randn(1, 900, 4, 64)
cu = [0, 100, 450, 900]
cu_t = torch.tensor(cu, dtype=torch.int64, device=DEVICE)
ci_t = mod.prepare_chunk_indices(cu_t, 64).flip(0).contiguous()
g = chunk_gate(1, 900, 4, 64, cu)
dv = direct(q, k, do, g, 0.125, cu_t, ci_t, 64)
w, f, v, _ = post_check(q, k, do, g, 0.125, cu, 64, 64, dv)
ok = w <= 1 and f and v and frame_ok(dv, 1, 900, cu) and tables_ok(cu, ci_t.cpu().tolist(), 900, 64)
results.append(ok)
worst_all = max(worst_all, w)
print(f"{'ok ' if ok else 'BAD'} edge permuted chunk_indices: gate: {w:.3f}")

# Edges of Pre: large magnitudes, subnormal do, all-positive inputs (no cancellation), strong
# decays (gate differences far below -126), increasing gates up to +63 within a chunk, the scale's
# bounds 2^-32 and 2^32.
check("pre large", 1, 1000, 4, 4, 128, 128, 64, [0, 500, 1000], qs=2.0**20, ks=2.0**20, ds=2.0**40)
check("pre subnormal do", 2, 300, 2, 2, 64, 64, 64, ds=2.0**-130)
check("pre positive", 2, 512, 2, 4, 128, 128, 64, positive=True)
check("pre positive K256", 1, 256, 2, 2, 256, 512, 64, positive=True)
check("pre strong decay", 2, 1000, 2, 4, 128, 128, 64,
      gate_fn=lambda B, T, HV, BT, cu: chunk_gate(B, T, HV, BT, cu, per_token=-50.0 * torch.rand((B * T, HV), generator=gen)),
      gates=(True,))
check("pre rising gate", 2, 300, 2, 2, 64, 64, 64, qs=0.5, ks=0.5,
      gate_fn=lambda B, T, HV, BT, cu: chunk_gate(B, T, HV, BT, cu, per_token=torch.full((B * T, HV), 1.0 / RCP_LN2)),
      gates=(True,))
check("pre scale 2^-32", 1, 500, 2, 2, 128, 128, 64, scale=2.0**-32, qs=2.0**10, ks=2.0**10)
check("pre scale 2^32", 1, 500, 2, 2, 128, 128, 64, scale=2.0**32, qs=2.0**-10, ks=2.0**-10)

# Negative tests: outputs the check must reject.
neg = []
q, k, do, g, scale, cu, BT, BK, dv = DATA[("host", True)]  # the 4096-token packed case, gated
bad = dv.clone()
flat = bad.view(-1)
# one element moved by far more than its tolerance: |dv| * 2^-5 + 1
j = 12345
flat[j] = (flat[j].float() + (flat[j].float().abs() * 2.0**-5 + 1.0)).to(torch.bfloat16)
neg.append(("one element off", post_check(q, k, do, g, scale, cu, BT, BK, bad)[0] > 1))
dv0 = mod.chunk_bwd_dv_local(q, k, do, g=None, scale=scale, cu_seqlens=torch.tensor(cu, device=DEVICE), chunk_size=BT)
neg.append(("ungated output vs gated spec", post_check(q, k, do, g, scale, cu, BT, BK, dv0)[0] > 1))
dv32 = mod.chunk_bwd_dv_local(q, k, do, g=None, scale=scale, cu_seqlens=torch.tensor(cu, device=DEVICE), chunk_size=32)
neg.append(("chunk size 32 vs chunk size 64 spec", post_check(q, k, do, None, scale, cu, 64, BK, dv32)[0] > 1))
cu_t = torch.tensor([5, 20, 20, 37, 200], dtype=torch.int64, device=DEVICE)
qq, kk, dd = randn(1, 300, 2, 64), randn(1, 300, 2, 64), randn(1, 300, 2, 64)
dvd = direct(qq, kk, dd, None, 0.125, cu_t, mod.prepare_chunk_indices(cu_t, 64), 64)
dvd.view(-1)[0] = 1.0  # token 0 lies outside every sequence
neg.append(("unwritten element changed", not frame_ok(dvd, 1, 300, [5, 20, 20, 37, 200])))
for name, rejected in neg:
    results.append(rejected)
    print(f"{'ok ' if rejected else 'BAD'} negative: {name}: {'rejected' if rejected else 'NOT rejected'}")
del DATA

# The grid's y limit: B*HV = 65536 does not launch (a child process, so that a failed launch
# cannot affect this one).
child = f"""
import sys; sys.dont_write_bytecode = True
import importlib.util, torch
spec = importlib.util.spec_from_file_location('m', {str(SRC)!r}); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
q = torch.zeros((4096, 16, 1, 16), device='cuda', dtype=torch.bfloat16)
do = torch.zeros((4096, 16, 16, 16), device='cuda', dtype=torch.bfloat16)
try:
    dv = m.chunk_bwd_dv_local(q, q, do, scale=0.25, chunk_size=16); torch.cuda.synchronize(); print('LAUNCHED')
except Exception as e:
    print('FAILED', type(e).__name__, str(e).splitlines()[0][:120])
"""
r = subprocess.run([sys.executable, "-I", "-c", child], capture_output=True, text=True, timeout=600)
msg = (r.stdout.strip().splitlines() or [r.stderr.strip()[-200:]])[-1]
lim_ok = not msg.startswith("LAUNCHED")
results.append(lim_ok)
print(f"{'ok ' if lim_ok else 'BAD'} limit B*HV = 65536 (B 4096, HV 16): {msg}")

n_bad = results.count(False)
print(f"peak device memory {max(PEAK[0], torch.cuda.max_memory_allocated() / 2**30):.2f} GiB (largest case: {PEAK[1]})")
print(f"{len(results)} checks: worst error / tolerance {worst_all:.4f} (margin {1 / max(worst_all, 1e-300):.2f}x); "
      + ("every element within tol and finite, ValsOk and TablesOk on every input, every frame clause held, "
         "every negative test rejected, B*HV = 65536 does not launch" if n_bad == 0 else f"{n_bad} FAILED"))
sys.exit(1 if n_bad else 0)

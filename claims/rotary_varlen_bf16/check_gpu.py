"""GPU check of the prepared claim `Claims.RotaryVarlen` (Claim.lean).

Runs the kernel's own host, `rotary_embedding_fwd` of `source/rotary.py`
(imported by path), on the setup's launches (the host's 22 test cases with their own inputs, and the
other configurations `Claim.lean` names) and on launches at the edges of `Inst.ok`. For each launch
it computes `Post`'s reference and tolerance in float64 from the inputs, exactly as `Post` states
them:

    ref = a*cs - b*sn  (first channel of a pair)   or   a*cs + b*sn  (second)
    tol = 2^-8 * |ref| + 2^-22 * (|a*cs| + |b*sn|) + 2^-132

(`a` the channel's own value, `b` its partner's, `cs`, `sn` the table entries of its pair at the
token's position, its index in its sequence plus the sequence's offset: the one `int`, or its entry
of the tensor `seqlen_offsets`), and reports the largest |y - ref| / tol over the tokens whose
position is not negative (below 1 is a pass), that every such rotated value is finite, and the
frame: the rotated channels of the tokens at a negative position (left padding under a tensor of
offsets) keep their bytes, `y`'s channels from 2R on are what the host put there (`x`'s, bit for
bit), guard bytes around `y` are unchanged, and `x`, `cos`, `sin`, `cu_seqlens` and the offsets
tensor are unchanged.

Negative tests: outputs that `Post` must reject (the kernel run with the offsets ignored, with
`conjugate=True`, all-zero rotated channels, and the skipped tokens written) are checked the same
way and must fail.

To see the frame, `torch.zeros_like` is replaced during the host call by a function that returns a
view into a larger buffer filled with a NaN pattern (the kernel never reads `y`; the host then
copies `x[..., 2R:]` into it as usual), so a rotated channel the kernel failed to write would show
as a non-finite value and a write outside `y` as a changed guard.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "rotary.py")
spec = importlib.util.spec_from_file_location("rotary_src", SRC)
rot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rot)

DEV = torch.device("cuda")
SENT = 0x7FA5  # a bfloat16 NaN pattern
GUARD = 4096   # guard elements on each side of y


def sentinel_zeros_like(x, *args, **kwargs):
    n = x.numel()
    buf = torch.empty(n + 2 * GUARD, dtype=torch.int16, device=x.device)
    buf.fill_(SENT)
    buf = buf.view(torch.bfloat16)
    sentinel_zeros_like.buf = buf
    return buf[GUARD:GUARD + n].view(x.shape)


def run_host(x, cos, sin, off, cu, interleaved):
    real = torch.zeros_like
    torch.zeros_like = sentinel_zeros_like
    try:
        y = rot.rotary_embedding_fwd(x, cos, sin, seqlen_offsets=off, cu_seqlens=cu, interleaved=interleaved)
    finally:
        torch.zeros_like = real
    torch.cuda.synchronize()
    return y, sentinel_zeros_like.buf


def check(name, x, cos, sin, off, cu, interleaved, run=None, expect=True):
    """Runs the host (or `run`, a wrong output for a negative test) and checks Post."""
    B, T, H, D = x.shape
    TR, R = cos.shape
    x0, c0, s0 = x.clone(), cos.clone(), sin.clone()
    cu0 = None if cu is None else cu.clone()
    off0 = off.clone() if isinstance(off, torch.Tensor) else off
    y, buf = (run or run_host)(x, cos, sin, off, cu, interleaved)
    frame = True
    frame &= torch.equal(x.view(torch.int16), x0.view(torch.int16))
    frame &= torch.equal(cos.view(torch.int16), c0.view(torch.int16))
    frame &= torch.equal(sin.view(torch.int16), s0.view(torch.int16))
    if cu is not None:
        frame &= torch.equal(cu, cu0)
    if isinstance(off, torch.Tensor):
        frame &= torch.equal(off, off0)
    g = buf.view(torch.int16)
    frame &= bool((g[:GUARD] == SENT).all()) and bool((g[GUARD + x.numel():] == SENT).all())
    if 2 * R < D:
        frame &= torch.equal(y[..., 2 * R:].view(torch.int16), x[..., 2 * R:].view(torch.int16))
    # positions of the flat tokens (Post: index in the sequence plus the sequence's offset)
    if cu is None:
        seq = torch.arange(B, device=DEV).repeat_interleave(T)
        idx = torch.arange(T, device=DEV).repeat(B)
    else:
        cul = cu.long()
        lens = cul[1:] - cul[:-1]
        seq = torch.repeat_interleave(torch.arange(len(lens), device=DEV), lens)
        idx = torch.arange(T, device=DEV) - cul[seq]
    offv = off.long()[seq] if isinstance(off, torch.Tensor) else off
    pos = idx + offv
    live = pos >= 0
    # Post's second clause: a token at a negative position keeps its rotated channels' bytes
    yf16 = y.reshape(B * T, H, D).view(torch.int16)
    skipped = ~live
    if bool(skipped.any()):
        frame &= bool((yf16[skipped][..., :2 * R] == SENT).all())
    # channel maps
    c = torch.arange(2 * R, device=DEV)
    if interleaved:
        pair = c // 2
        partner = torch.where(c % 2 == 0, c + 1, c - 1)
        first = c % 2 == 0
    else:
        pair = torch.where(c < R, c, c - R)
        partner = torch.where(c < R, c + R, c - R)
        first = c < R
    xf = x.reshape(B * T, H, D)
    yf = y.reshape(B * T, H, D)
    worst, finite = 0.0, True
    step = max(1, (1 << 22) // (H * 2 * R))
    for t0 in range(0, B * T, step):
        t1 = min(B * T, t0 + step)
        lv = live[t0:t1]
        if not bool(lv.any()):
            continue
        p = pos[t0:t1][lv]
        a = xf[t0:t1][lv][:, :, :2 * R].double()
        b = xf[t0:t1][lv][:, :, partner].double()
        cs = cos[p][:, pair].double()[:, None, :]
        sn = sin[p][:, pair].double()[:, None, :]
        ac, bs = a * cs, b * sn
        ref = torch.where(first, ac - bs, ac + bs)
        tol = 2.0 ** -8 * ref.abs() + 2.0 ** -22 * (ac.abs() + bs.abs()) + 2.0 ** -132
        v = yf[t0:t1][lv][:, :, :2 * R].double()
        finite &= bool(torch.isfinite(v).all())
        worst = max(worst, ((v - ref).abs() / tol).max().item())
    post = finite and frame and worst < 1
    if not expect:
        ok = not post
        print(f"{'PASS' if ok else 'FAIL'} negative {name}: Post {'holds (not rejected)' if post else 'rejects it'} "
              f"(max err/tol {worst:.3g}, finite {finite}, frame {frame})")
        return ok, 0.0
    ok = post
    print(f"{'PASS' if ok else 'FAIL'} {name}: max err/tol {worst:.4f} (margin {1 - worst:.4f}), "
          f"finite {finite}, frame {frame}")
    return ok, worst


def tables(TR, rd):
    return rot.cos_sin_cache(TR, rd, DEV)


def main():
    torch.manual_seed(1)
    results = []
    gen = torch.Generator().manual_seed(0)

    def random_cu_seqlens(T, N, dtype):
        # the host test's own draw (source lines 315-318)
        cuts = torch.arange(1, T)[torch.randperm(T - 1, generator=gen)[:N - 1]]
        return torch.cat([torch.tensor([0]), cuts, torch.tensor([T])]).sort()[0].to(device=DEV, dtype=dtype)

    # 1. The host's 22 test cases (source lines 323-337), with the same inputs (same generator order).
    host = []
    for il in (False, True):
        host += [
            (2, 2048, 4, 128, 128, 0, None, il), (2, 4096, 4, 256, 256, 0, None, il),
            (1, 1, 1, 64, 64, 0, None, il), (3, 1000, 8, 64, 64, 7, None, il),
            (2, 256, 4, 128, 64, 0, None, il), (1, 2048, 4, 128, 128, 0, (4, torch.int64), il),
            (1, 4096, 4, 256, 256, 0, (4, torch.int32), il), (1, 777, 2, 64, 32, 0, (9, torch.int32), il),
            (1, 5, 2, 128, 128, 0, (5, torch.int64), il), (4, 8192, 32, 128, 128, 0, None, il),
            (1, 32768, 32, 128, 128, 0, (16, torch.int32), il),
        ]
    for B, T, H, D, rd, off, vl, il in host:
        cu = None if vl is None else random_cu_seqlens(T, vl[0], vl[1])
        cos, sin = tables(T + off, rd)
        x = torch.randn((B, T, H, D), generator=gen).to(device=DEV, dtype=torch.bfloat16)
        n = "padded" if vl is None else f"packed N={vl[0]} {vl[1]}".replace("torch.", "")
        results.append(check(f"host {'gptj' if il else 'neox'} B{B} T{T} H{H} D{D} rd{rd} off{off} {n}",
                             x, cos, sin, off, cu, il))
        del x, cos, sin, cu
        torch.cuda.empty_cache()

    def rnd(B, T, H, D):
        return torch.randn((B, T, H, D), device=DEV).to(torch.bfloat16)

    # 2. The other setup configurations Claim.lean names.
    for il in (False, True):
        cos, sin = tables(4096, 128)
        results.append(check(f"setup decode B8 T1 H32 D128 off4095 {'gptj' if il else 'neox'}",
                             rnd(8, 1, 32, 128), cos, sin, 4095, None, il))
    cos, sin = tables(2048, 64)
    results.append(check("setup fla config B4 T2048 H32 D64 neox", rnd(4, 2048, 32, 64), cos, sin, 0, None, False))
    cos, sin = tables(2048, 128)
    results.append(check("setup one key head B2 T2048 H1 D128 gptj", rnd(2, 2048, 1, 128), cos, sin, 0, None, True))
    for il in (False, True):
        cos, sin = tables(2048, 64)
        results.append(check(f"setup partial rd64 of D256 B2 T2048 H4 {'gptj' if il else 'neox'}",
                             rnd(2, 2048, 4, 256), cos, sin, 0, None, il))
    cos, sin = tables(4096, 128)
    cu = torch.tensor([0, 4096], device=DEV, dtype=torch.int64)
    results.append(check("setup one packed sequence T4096 H4 D128 int64 neox", rnd(1, 4096, 4, 128), cos, sin, 0, cu, False))

    # 3. Edges of Inst.ok.
    for il in (False, True):
        tag = "gptj" if il else "neox"
        cos, sin = tables(1, 64)
        results.append(check(f"edge B=8192 (largest batch, b_le) T1 H1 D64 {tag}", rnd(8192, 1, 1, 64), cos, sin, 0, None, il))
        cos, sin = tables(300, 2)
        results.append(check(f"edge R=1 (BD=2) D256 B2 T300 H3 {tag}", rnd(2, 300, 3, 256), cos, sin, 0, None, il))
        cos, sin = tables(500, 6)
        results.append(check(f"edge R=3 (BD=8 > 2R) D8 B3 T500 H2 {tag}", rnd(3, 500, 2, 8), cos, sin, 0, None, il))
        cos, sin = tables(17000 + (1 << 20), 64)
        results.append(check(f"edge offset 2^20, BT=128 (T=17000) B1 H2 D64 {tag}", rnd(1, 17000, 2, 64), cos, sin, 1 << 20, None, il))
        cos, sin = tables(1000, 256)
        cu = torch.tensor([0, 0, 1, 1, 1, 500, 500, 999, 1000, 1000], device=DEV, dtype=torch.int64)
        results.append(check(f"edge packed with empty sequences, D256 rd256 int64 {tag}", rnd(1, 1000, 2, 256), cos, sin, 0, cu, il))
        cu32 = torch.tensor([0] + list(range(1, 133)), device=DEV, dtype=torch.int32)
        cos, sin = tables(140, 128)
        results.append(check(f"edge 132 one-token sequences (BT=1), offset 7 int32 {tag}", rnd(1, 132, 4, 128), cos, sin, 7, cu32, il))
        # magnitudes at the bounds of Pre: |x| = 2^125, |cos| = |sin| = 1 (S up to 2^126)
        x = (torch.randint(0, 2, (2, 64, 2, 64), device=DEV) * 2 - 1).to(torch.float32) * 2.0 ** 125
        tab = (torch.randint(0, 2, (64, 32), device=DEV) * 2 - 1).to(torch.bfloat16)
        results.append(check(f"edge |x| = 2^125, |cos| = |sin| = 1 {tag}", x.to(torch.bfloat16), tab, tab.flip(0).contiguous(), 0, None, il))
        # subnormal and tiny magnitudes (the absolute floor)
        x = (torch.randn((2, 64, 2, 64), device=DEV) * 2.0 ** -128).to(torch.bfloat16)
        cos, sin = tables(64, 64)
        results.append(check(f"edge subnormal x {tag}", x, (cos.float() * 2.0 ** -10).to(torch.bfloat16), sin, 0, None, il))
        # cancellation: x0 = x1 and cos = sin, so the first channel's ref is 0
        x = rnd(2, 64, 2, 64)
        x = torch.cat([x[..., :32], x[..., :32]], -1) if not il else x[..., 0::2].repeat_interleave(2, -1).contiguous()
        cos, _ = tables(64, 64)
        results.append(check(f"edge cancellation {tag}", x, cos, cos.clone(), 0, None, il))

        # Pre constrains only what the kernel reads: NaN in x's channels from 2R on and in the
        # table rows outside [off, off + T)
        x = rnd(2, 300, 2, 128)
        x[..., 64:] = float("nan")
        cos, sin = tables(400, 64)
        cos, sin = cos.clone(), sin.clone()
        cos[:50] = float("nan"); sin[:50] = float("nan"); cos[350:] = float("inf"); sin[350:] = float("nan")
        results.append(check(f"edge NaN outside what is read (x channels >= 2R, table rows) {tag}", x, cos, sin, 50, None, il))

    # 4. A tensor of offsets (IS_SEQLEN_OFFSETS_TENSOR = True): the setup's launches and edges.
    def offs(vals, dt=torch.int32):
        return torch.tensor(vals, device=DEV, dtype=dt)
    for il in (False, True):
        tag = "gptj" if il else "neox"
        cos, sin = tables(2048, 64)
        # batched generation's prefill under left padding: offset -pad (0 for the longest prompt)
        results.append(check(f"setup tensor offsets, prefill under left padding B4 T512 H32 D64 int32 {tag}",
                             rnd(4, 512, 32, 64), cos, sin, offs([0, -37, -200, -511]), None, il))
        # the decoding step: the cache's length minus the padding
        results.append(check(f"setup tensor offsets, decoding B8 T1 H32 D64 int32 {tag}",
                             rnd(8, 1, 32, 64), cos, sin, offs([600, 563, 400, 89, 0, 1, 599, 2047]), None, il))
    cos, sin = tables(4096, 128)
    results.append(check("setup tensor offsets int64 B2 T2048 H4 D128 neox",
                         rnd(2, 2048, 4, 128), cos, sin, offs([0, 2048], torch.int64), None, False))
    cos, sin = tables(1024, 32)
    results.append(check("setup tensor offsets int64, packed N=3 int32 T777 H2 D64 rd32 neox",
                         rnd(1, 777, 2, 64), cos, sin, offs([5, -50, 547], torch.int64),
                         offs([0, 100, 300, 777]), False))
    for il in (False, True):
        tag = "gptj" if il else "neox"
        cos, sin = tables(17000, 64)
        results.append(check(f"edge tensor offsets, BT=128 (T=17000), pads across chunks, whole row padded {tag}",
                             rnd(3, 17000, 2, 64), cos, sin, offs([-300, -16999, -17000]), None, il))
        cos, sin = tables(64, 64)
        results.append(check(f"edge tensor offsets -2^31 (every token skipped) and 64 - T int32 {tag}",
                             rnd(2, 40, 2, 64), cos, sin, offs([-2 ** 31, 24]), None, il))
        results.append(check(f"edge tensor offsets int64 -2^31 and -1, packed with empty sequences {tag}",
                             rnd(1, 50, 2, 64), cos, sin, offs([-2 ** 31, -1, 3, 0], torch.int64),
                             offs([0, 20, 20, 50, 50], torch.int64), il))
        cos, sin = tables(300, 128)
        results.append(check(f"edge tensor offsets, 132 one-token sequences (BT=1) int32 {tag}",
                             rnd(1, 132, 4, 128), cos, sin, offs([(7 * i) % 301 - 1 for i in range(132)]),
                             offs(list(range(133))), il))
    # the host's torch.zeros_like leaves the skipped tokens' rotated channels 0
    cos, sin = tables(2048, 64)
    x = rnd(4, 512, 2, 64)
    y = rot.rotary_embedding_fwd(x, cos, sin, seqlen_offsets=offs([0, -37, -200, -511]), interleaved=False)
    zero = all(bool((y[b, :p] == 0).all()) for b, p in enumerate([0, 37, 200, 511]))
    print(f"NOTE the host's y at the left-padding tokens is 0: {zero}")
    results.append((zero, 0.0))

    # 5. Negative tests: wrong outputs Post must reject.
    xo = rnd(4, 512, 32, 64)
    to = offs([0, -37, -200, -511])
    cos, sin = tables(2048, 64)

    def ignore_offsets(x, cos, sin, off, cu, il):
        return run_host(x, cos, sin, 0, cu, il)

    def conjugate(x, cos, sin, off, cu, il):
        real = torch.zeros_like
        torch.zeros_like = sentinel_zeros_like
        try:
            y = rot.rotary_embedding_fwd(x, cos, sin, seqlen_offsets=off, cu_seqlens=cu, interleaved=il, conjugate=True)
        finally:
            torch.zeros_like = real
        return y, sentinel_zeros_like.buf

    def all_zero(x, cos, sin, off, cu, il):
        y, buf = run_host(x, cos, sin, off, cu, il)
        y[..., :2 * cos.shape[1]] = 0
        return y, buf

    def write_skipped(x, cos, sin, off, cu, il):
        y, buf = run_host(x, cos, sin, off, cu, il)
        for b, p in enumerate((-off).clamp(min=0).tolist()):
            y[b, :p] = x[b, :p]
        return y, buf

    for nm, fn in [("offsets ignored (position = index)", ignore_offsets), ("conjugate=True (sin negated)", conjugate),
                   ("all-zero rotated channels", all_zero), ("left-padding tokens written (x copied)", write_skipped)]:
        results.append(check(nm, xo, cos, sin, to, None, False, run=fn, expect=False))
    cos, sin = tables(4096, 128)
    results.append(check("int offset 1 instead of 0", rnd(2, 256, 4, 128), cos, sin, 0, None, True,
                         run=lambda x, c, s, o, cu, il: run_host(x, c, s, 1, cu, il), expect=False))

    # 6. Beyond Inst.ok (not counted): B = 65535 (above b_le, the largest grid axis 1) still runs;
    # the kernel limit (excluded by Inst.ok.grid_b): a padded batch of 65536 fails to launch.
    cos, sin = tables(1, 64)
    for il in (False, True):
        ok, _ = check(f"beyond Inst.ok (b_le) B=65535 (largest grid axis 1) T1 H1 D64 {'gptj' if il else 'neox'}",
                      rnd(65535, 1, 1, 64), cos, sin, 0, None, il)
        print(f"NOTE B=65535 (outside Inst.ok: b_le, not counted): {'passes' if ok else 'fails'}")
    try:
        run_host(rnd(65536, 1, 1, 64), cos, sin, 0, None, False)
        print("NOTE B=65536: launched (unexpected)")
    except RuntimeError as e:
        print(f"NOTE B=65536 (outside Inst.ok, a kernel limit): the launch fails: {str(e)[:80]}")

    worst = max(w for _, w in results)
    print(f"{sum(ok for ok, _ in results)}/{len(results)} launches pass; worst err/tol {worst:.4f} (margin {1 - worst:.4f})")
    sys.exit(0 if all(ok for ok, _ in results) else 1)


if __name__ == "__main__":
    main()

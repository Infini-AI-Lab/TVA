"""GPU check of the claim for `kv_cache_write_bf16` (README.md).

Runs the kernel's own host (`source/reshape_and_cache_flash.py`) on the
setup's launches (the host's test cases, both of vLLM's cache layouts, with the same generator) and
on launches at the edges of `Inst.ok` and `Pre`, and checks `Post` exactly as `Claim.lean` states
it, bit for bit:

* for every token `t` with a slot `s >= 0`, every head `h` and component `e`, element
  `(s // block_size) * block_stride + (s % block_size) * page_stride + h * head_stride + e` of the
  key cache's storage (from its first element, with the key cache's strides) holds element
  `t * key.stride(0) + h * head_size + e` of `key`'s storage, and likewise for the value cache;
* the frame: every other byte of the caches' allocation is unchanged, and the inputs too; around
  the allocation, guard regions are unchanged.

The caches and inputs hold random 16-bit patterns (NaN payloads included), so the comparison is on
the bits; exact, no tolerance.

With an fp8 cache (`kv_cache_dtype = "fp8"`; the caches carved from a random `uint8` allocation,
which the host views as `float8e4nv`, as vLLM's are) it checks `Stored`:

* `Mode.fp8`, `bfloat16` inputs: each written element is one of the saturating roundings to
  nearest even (`cvtSat`: clamp to `[-448, 448]`, then torch's round-to-nearest-even cast) of
  the two `float32` values at the ends of `[q - e, q + e]`, `q = x / k_scale` (`v_scale` for the
  value cache), `e = 2^-22 |q| + 2^-148`; `cvtSat` is monotone and the interval is far narrower
  than the fp8 spacing, so these are all the values `Stored` allows. Elements whose `x` is not
  finite are not constrained (beyond holding a byte). The inputs are `randn` rows with some NaN,
  infinities, values beyond `448 * scale` (saturation) and tiny values (fp8 subnormals and zero);
  the scales are vLLM's test's `key.amax() / 64` and others. Negative tests: wrong caches `Post`
  must reject (the scale left out, the scale multiplied, the values rounded toward zero, the value
  cache divided by `k_scale`).
* `Mode.fp8in`, `float8e4nv` inputs (random bytes, NaN patterns included): bit for bit.

Run with `CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""

import importlib.util
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "reshape_and_cache_flash.py")
sys.dont_write_bytecode = True  # leave no __pycache__ in the kernel's folder
spec = importlib.util.spec_from_file_location("kv_cache_write_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = torch.device("cuda")
GUARD = 4096


def views_of(layout, nb, bs, nh, hs, pool):
    """The caches as vLLM allocates them, carved from `pool` (an `int16` buffer) after `GUARD`
    elements; returns the two `bfloat16` views and the allocation's element range."""
    size = nb * 2 * bs * nh * hs
    flat = pool[GUARD:GUARD + size].view(torch.bfloat16)
    if layout == "flash":
        kv = flat.view(nb, 2, bs, nh, hs)
        return kv[:, 0], kv[:, 1], size
    kv = flat.view(nb, nh, bs, 2 * hs)
    k, v = kv.transpose(1, 2).split(hs, dim=-1)
    return k, v, size


def expected_pool(pool0, key, value, kc, vc, slots, base_ptr):
    """`Post` on the whole pool (`int16` bits, on the host), from the host's inputs: the pool before the launch
    with each non-padding token's row written at its slot, at the addresses the caches' strides
    give (the key cache's strides for both, as `Inst` states)."""
    exp = pool0.clone()
    nb, bs, nh, hs = kc.shape
    s0, s1, s2, s3 = kc.stride()
    assert vc.stride() == kc.stride() and s3 == 1
    koff = (kc.data_ptr() - base_ptr) // 2
    voff = (vc.data_ptr() - base_ptr) // 2
    T = slots.numel()
    sl = slots.cpu().long()
    keep = (sl >= 0)
    assert bool((sl < nb * bs).all())
    nonneg = sl[keep]
    assert nonneg.unique().numel() == nonneg.numel()  # Pre: distinct non-negative slots
    t = torch.arange(T)[keep]
    blk = (nonneg // bs)
    row = (nonneg % bs)
    h = torch.arange(nh)
    e = torch.arange(hs)
    dst = (blk * s0 + row * s1)[:, None, None] + (h * s2)[None, :, None] + e[None, None, :]
    # key's storage: element t * stride(0) + h * hs + e
    kst = key.stride(0)
    vst = value.stride(0)
    kstore = torch.as_strided(key, (T, nh, hs), (kst, hs, 1)).contiguous().view(torch.int16).cpu()
    vstore = torch.as_strided(value, (T, nh, hs), (vst, hs, 1)).contiguous().view(torch.int16).cpu()
    exp_c = exp
    exp_c[GUARD + koff + dst.reshape(-1)] = kstore[t].reshape(-1)
    exp_c[GUARD + voff + dst.reshape(-1)] = vstore[t].reshape(-1)
    return exp_c


results = []


def run(name, layout, T, nh, hs, bs, nb, num_pad, gen, key_stride_heads=3, direct_tile=None):
    """One launch through the host, caches carved from a random-filled pool with guards; key and
    value slices of a fused `[T, key_stride_heads, nh, hs]` projection."""
    size = nb * 2 * bs * nh * hs
    pool = torch.randint(-2 ** 15, 2 ** 15, (GUARD + size + GUARD,), generator=gen, dtype=torch.int16).to(DEV)
    pool0 = pool.cpu()  # the initial bits, kept on the host (device memory stays below 4 GB)
    kc, vc, _ = views_of(layout, nb, bs, nh, hs, pool)
    base_ptr = pool.data_ptr() + 2 * GUARD
    slots = torch.randperm(nb * bs, generator=gen)[:T]
    pad = torch.randperm(T, generator=gen)[:num_pad]
    slots[pad] = -1
    if num_pad and T > 3:
        slots[pad[: max(1, num_pad // 4)]] = -7  # other negative values: padding too
    qkv = torch.randint(-2 ** 15, 2 ** 15, (T, key_stride_heads, nh, hs), generator=gen,
                        dtype=torch.int16).view(torch.bfloat16).to(DEV)
    key, value = qkv[:, key_stride_heads - 2], qkv[:, key_stride_heads - 1]
    qkv0 = qkv.clone()
    slot_mapping = slots.to(DEV, torch.long)
    sm0 = slot_mapping.clone()
    k_scale = torch.tensor(1.0, dtype=torch.float32, device=DEV)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=DEV)
    mod.reshape_and_cache_flash(key, value, kc, vc, slot_mapping, "auto", k_scale, v_scale)
    exp = expected_pool(pool0, key, value, kc, vc, slot_mapping, base_ptr)
    got = pool.cpu()
    ok = torch.equal(got[GUARD:GUARD + size], exp[GUARD:GUARD + size])
    frame = (torch.equal(got[:GUARD], exp[:GUARD]) and torch.equal(got[GUARD + size:], exp[GUARD + size:])
             and torch.equal(qkv.view(torch.int16), qkv0.view(torch.int16)) and torch.equal(slot_mapping, sm0))
    n_written = int((slots >= 0).sum()) * 2 * nh * hs
    results.append((name, ok, frame, n_written))
    print(f"{name}: {layout}, {T} tokens ({int((slots < 0).sum())} padding), {nh} x {hs}, blocks {nb} x {bs}: "
          f"caches bit-exact {ok} ({n_written} elements written, {size - n_written} kept), "
          f"inputs and guards unchanged {frame}")


def views_of8(layout, nb, bs, nh, hs, pool):
    """As `views_of`, from a `uint8` pool: the caches as `uint8` views (vLLM's fp8 allocation)."""
    size = nb * 2 * bs * nh * hs
    flat = pool[GUARD:GUARD + size]
    if layout == "flash":
        kv = flat.view(nb, 2, bs, nh, hs)
        return kv[:, 0], kv[:, 1], size
    kv = flat.view(nb, nh, bs, 2 * hs)
    k, v = kv.transpose(1, 2).split(hs, dim=-1)
    return k, v, size


def dst_index(kc, slots, nh, hs, base_ptr, esize):
    """For every non-padding token: its index and the pool elements its key and value rows go to
    (from the key cache's strides, `Inst`); and the value cache's offset."""
    nb, bs = kc.shape[0], kc.shape[1]
    s0, s1, s2, _ = kc.stride()
    sl = slots.cpu().long()
    keep = sl >= 0
    assert bool((sl < nb * bs).all())
    nonneg = sl[keep]
    assert nonneg.unique().numel() == nonneg.numel()
    t = torch.arange(sl.numel())[keep]
    dst = ((nonneg // bs) * s0 + (nonneg % bs) * s1)[:, None, None] + \
        (torch.arange(nh) * s2)[None, :, None] + torch.arange(hs)[None, None, :]
    koff = (kc.data_ptr() - base_ptr) // esize
    return t, dst, koff


def cvt_sat(r):
    """`FloatV.cvtSat .rn` into float8e4nv of float32 values (finite): clamp, then round to
    nearest even (torch's cast), as bits."""
    return r.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)


def allowed8(x, s):
    """The fp8 bits `Stored` allows for finite bfloat16 values `x` (float64) with a finite scale
    `s != 0`, `|x / s| <= 2^126`: cvtSat of the float32 ends of [q - e, q + e]. Returns
    (bits at the lower end, bits at the upper end, q)."""
    q = x / s
    e = 2.0 ** -22 * q.abs() + 2.0 ** -148
    lo, hi = (q - e), (q + e)
    rlo = lo.float()
    rlo = torch.where(rlo.double() < lo, torch.nextafter(rlo, torch.full_like(rlo, float("inf"))), rlo)
    rhi = hi.float()
    rhi = torch.where(rhi.double() > hi, torch.nextafter(rhi, torch.full_like(rhi, float("-inf"))), rhi)
    return cvt_sat(rlo), cvt_sat(rhi), q


def toward_zero8(q):
    """The fp8 value next to q toward zero (a rounding toward zero), as bits; a wrong result."""
    b = cvt_sat(q.float())
    v = b.view(torch.float8_e4m3fn).double()
    mag = b & 0x7F
    down = (v.abs() > q.abs()) & (mag > 0)
    return torch.where(down, b - 1, b)


def post8(got_k, got_v, x_k, x_v, ks, vs):
    """`Post`'s element clauses for `Mode.fp8` on the written elements (flat): each finite input's
    element is one of the allowed fp8 values. Returns (holds, number checked, and over the
    elements with `2^-6 <= |q| <= 448` (fp8's normal range) and `|q| < 2^-6` the median and worst
    of the allowed deviation from the clamped quotient over |out|, `max |c - clamp(q)| / |out|`)."""
    ok = True
    n = 0
    ratios = []
    for got, x, s in ((got_k, x_k, ks), (got_v, x_v, vs)):
        xd = x.double()
        fin = torch.isfinite(xd) & math.isfinite(s) & (s != 0)
        fin = fin & ((xd / s).abs() <= 2.0 ** 126)
        a, b, q = allowed8(xd[fin], s)
        g = got[fin]
        ok = ok and bool(((g == a) | (g == b)).all())
        n += int(fin.sum())
        va = a.view(torch.float8_e4m3fn).double()
        vb = b.view(torch.float8_e4m3fn).double()
        qc = q.clamp(-448.0, 448.0)
        tol = torch.maximum((va - qc).abs(), (vb - qc).abs())
        gv = g.view(torch.float8_e4m3fn).double()
        nz = (gv != 0) & (q.abs() <= 448.0)
        ratios.append(((tol[nz] / gv[nz].abs()), (q.abs()[nz] >= 2.0 ** -6)))
    r = torch.cat([x for x, _ in ratios])
    big = torch.cat([y for _, y in ratios])

    def med_max(v):
        return (float(v.median()), float(v.max())) if v.numel() else (0.0, 0.0)
    return ok, n, med_max(r[big]) + med_max(r[~big])


def run8(name, layout, T, nh, hs, bs, nb, num_pad, gen, mode="fp8", scales=None, specials=True,
         negatives=False):
    """`specials`: NaN, infinities, values 10^4 times larger (beyond 448 * scale) and 10^4 times
    smaller (fp8 subnormals and zero) among the `randn` inputs."""
    """One launch through the host with an fp8 cache (`kv_cache_dtype = "fp8"`)."""
    size = nb * 2 * bs * nh * hs
    pool = torch.randint(0, 256, (GUARD + size + GUARD,), generator=gen, dtype=torch.uint8).to(DEV)
    pool0 = pool.cpu()
    kc, vc, _ = views_of8(layout, nb, bs, nh, hs, pool)
    base_ptr = pool.data_ptr() + GUARD
    slots = torch.randperm(nb * bs, generator=gen)[:T]
    pad = torch.randperm(T, generator=gen)[:num_pad]
    slots[pad] = -1
    if mode == "fp8":
        qkv = torch.randn((T, 3, nh, hs), generator=gen)
        if scales is None:   # vLLM's test: key.amax() / 64, value.amax() / 64 (in bfloat16)
            b = qkv.to(torch.bfloat16)
            scales = (float((b[:, 1].amax() / 64.0).float()), float((b[:, 2].amax() / 64.0).float()))
        if specials:
            flat = qkv.view(-1)
            k = flat.numel()
            idx = torch.randperm(k, generator=gen)
            flat[idx[: k // 200]] = float("nan")
            flat[idx[k // 200: k // 100]] = float("inf")
            flat[idx[k // 100: k // 50]] *= -1e4          # beyond 448 * scale: saturates
            flat[idx[k // 50: k // 25]] *= 1e-4           # fp8 subnormals and zero
        qkv = qkv.to(DEV, torch.bfloat16)
    else:
        qkv = torch.randint(0, 256, (T, 3, nh, hs), generator=gen, dtype=torch.uint8).to(DEV).view(
            torch.float8_e4m3fn)
    _, key, value = qkv.unbind(dim=1)
    qkv0 = qkv.view(torch.uint8).clone()
    slot_mapping = slots.to(DEV, torch.long)
    sm0 = slot_mapping.clone()
    if scales is None:
        scales = (1.0, 1.0)
    k_scale = torch.tensor(scales[0], dtype=torch.float32, device=DEV)
    v_scale = torch.tensor(scales[1], dtype=torch.float32, device=DEV)
    ks, vs = float(k_scale), float(v_scale)
    mod.reshape_and_cache_flash(key, value, kc, vc, slot_mapping, "fp8", k_scale, v_scale)
    got = pool.cpu()
    t, dst, koff = dst_index(kc, slot_mapping, nh, hs, base_ptr, 1)
    voff = (vc.data_ptr() - base_ptr)
    kidx = (GUARD + koff + dst).reshape(-1)
    vidx = (GUARD + voff + dst).reshape(-1)
    keyc = key.cpu()[t].reshape(-1)
    valc = value.cpu()[t].reshape(-1)
    written = torch.zeros(got.numel(), dtype=torch.bool)
    written[kidx] = True
    written[vidx] = True
    frame = (torch.equal(got[~written], pool0[~written])
             and torch.equal(qkv.view(torch.uint8), qkv0) and torch.equal(slot_mapping, sm0)
             and float(k_scale) == ks and float(v_scale) == vs)
    stats = None
    if mode == "fp8":
        ok, n, stats = post8(got[kidx], got[vidx], keyc, valc, ks, vs)
    else:
        ok = torch.equal(got[kidx], keyc.view(torch.uint8)) and torch.equal(got[vidx], valc.view(torch.uint8))
        n = 2 * kidx.numel()
    results.append((name, ok, frame, 2 * kidx.numel()))
    extra = (f"; tol/|out| median {stats[0]:.3g}, worst {stats[1]:.3g} in fp8's normal range "
             f"(median {stats[2]:.3g}, worst {stats[3]:.3g} below it)") if stats else ""
    print(f"{name}: {mode} {layout}, {T} tokens ({int((slots < 0).sum())} padding), {nh} x {hs}, "
          f"blocks {nb} x {bs}, scales {ks:.4g} {vs:.4g}: Post {ok} ({n} elements constrained of "
          f"{2 * kidx.numel()} written), inputs, scales and the rest of the allocation unchanged {frame}{extra}")
    if negatives and mode == "fp8":
        xk, xv = keyc.double(), valc.double()
        fk = torch.isfinite(xk)
        fv = torch.isfinite(xv)
        cases = [
            ("the scale left out", cvt_sat(torch.where(fk, xk, 0.0).float()),
             cvt_sat(torch.where(fv, xv, 0.0).float())),
            ("the scale multiplied", cvt_sat(torch.where(fk, xk * ks, 0.0).float()),
             cvt_sat(torch.where(fv, xv * vs, 0.0).float())),
            ("rounded toward zero", toward_zero8(torch.where(fk, xk / ks, 0.0)),
             toward_zero8(torch.where(fv, xv / vs, 0.0))),
            ("the value cache divided by k_scale", got[kidx],
             cvt_sat(torch.where(fv, xv / ks, 0.0).float())),
        ]
        for cname, bk, bv in cases:
            bad, _, _ = post8(bk, bv, keyc, valc, ks, vs)
            print(f"  negative, {cname}: Post {'rejects' if not bad else 'ACCEPTS'}")
            results.append(("negative " + cname, not bad, True, 0))


if __name__ == "__main__":
    # The setup: the host's test cases (source lines 344-353), both layouts.
    gen = torch.Generator().manual_seed(0)
    for T, nh, hs, bs, nb, layout, num_pad in [
        (21, 8, 64, 16, 512, "flash", 0),
        (21, 8, 80, 8, 512, "flash", 0),
        (21, 8, 256, 32, 5000, "flash", 0),
        (1, 8, 128, 16, 64, "flash", 0),
        (1, 8, 128, 16, 64, "triton_attn", 1),
        (256, 8, 128, 16, 64, "triton_attn", 64),
        (300, 4, 96, 64, 20, "triton_attn", 17),
        (4096, 32, 128, 16, 300, "flash", 100),
    ]:
        run(f"setup {T} tokens {nh}x{hs} block {bs}", layout, T, nh, hs, bs, nb, num_pad, gen)
    # Edges: two tiles (n = 4096 > 2048) in the triton layout; n not a power of two (80 * 5);
    # one head of 1; block size 1; every slot used; only padding; the largest model shape (32 heads
    # of 256: n = 8192, four tiles); a single block; key and value of a fused kv (stride 2n).
    g = torch.Generator().manual_seed(1)
    run("edge two tiles triton", "triton_attn", 300, 32, 128, 16, 40, 30, g)
    run("edge n = 400", "flash", 77, 5, 80, 8, 30, 5, g)
    run("edge 1 head of 1", "triton_attn", 50, 1, 1, 4, 20, 3, g)
    run("edge block size 1", "flash", 64, 2, 64, 1, 64, 0, g)
    run("edge every slot used", "triton_attn", 256, 8, 64, 16, 16, 0, g)
    run("edge only padding", "flash", 40, 8, 64, 16, 8, 40, g)
    run("edge n = 8192, four tiles", "flash", 64, 32, 256, 16, 16, 7, g)
    run("edge n = 8192 triton", "triton_attn", 64, 32, 256, 16, 16, 7, g)
    run("edge one block", "flash", 16, 4, 32, 16, 1, 2, g)
    run("edge token stride 2n (key, value of a fused kv)", "triton_attn", 33, 8, 128, 16, 8, 1, g, key_stride_heads=2)
    run("edge 4096 tokens triton", "triton_attn", 4096, 8, 128, 16, 512, 300, g)

    # The fp8 cache (`kv_cache_dtype = "fp8"`): vLLM's test cases (test_cache.py:181, implementation
    # "triton", NHD, the "tensor" scales key.amax() / 64), both layouts, then other scales; and
    # float8e4nv inputs (the `is_fp8()` side).
    g8 = torch.Generator().manual_seed(2)
    for T, nh, hs, bs, nb, layout, num_pad in [
        (21, 8, 64, 16, 512, "flash", 0),
        (21, 8, 80, 8, 512, "flash", 0),
        (21, 8, 256, 32, 5000, "flash", 0),
        (256, 8, 128, 16, 64, "triton_attn", 64),
        (4096, 32, 128, 16, 300, "flash", 100),
        (4096, 8, 128, 16, 512, "triton_attn", 300),
    ]:
        run8(f"fp8 {T} tokens {nh}x{hs} block {bs}", layout, T, nh, hs, bs, nb, num_pad, g8,
             specials=False)
    run8("fp8 test shape with NaN, inf, saturating and tiny values", "flash", 21, 8, 256, 32, 5000, 0, g8,
         negatives=True)
    run8("fp8 serving-like 4096 tokens, specials", "triton_attn", 4096, 8, 128, 16, 512, 300, g8,
         negatives=True)
    run8("fp8 scales 1 and 1", "flash", 300, 8, 128, 16, 64, 20, g8, scales=(1.0, 1.0))
    run8("fp8 scales 0.0137 and -3.5", "triton_attn", 300, 4, 96, 64, 20, 17, g8, scales=(0.0137, -3.5))
    run8("fp8 tiny scale 1e-30 (q above 448 saturates)", "flash", 64, 4, 64, 16, 16, 3, g8,
         scales=(1e-30, 2.0 ** -126))
    run8("fp8 scale 0 and inf (unconstrained)", "flash", 32, 4, 64, 16, 8, 2, g8,
         scales=(0.0, float("inf")))
    run8("fp8in e4m3 inputs, flash", "flash", 4096, 8, 128, 16, 512, 100, g8, mode="fp8in")
    run8("fp8in e4m3 inputs, triton", "triton_attn", 256, 8, 128, 16, 64, 64, g8, mode="fp8in")

    bad = [r for r in results if not (r[1] and r[2])]
    launches = [r for r in results if not r[0].startswith("negative")]
    negs = len(results) - len(launches)
    print(f"{len(launches)} launches, {sum(r[3] for r in launches)} written elements checked (and every "
          f"other element of each allocation): bit-exact with a bfloat16 cache and with float8e4nv "
          f"inputs, one of the allowed fp8 values (`Stored`) with bfloat16 inputs and an fp8 cache; "
          f"frame held on every launch; {negs} negative tests rejected" if not bad else f"FAILED: {bad}")
    sys.exit(1 if bad else 0)

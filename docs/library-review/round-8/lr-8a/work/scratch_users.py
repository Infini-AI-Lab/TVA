"""Which kernels need the launch's global scratch (Compiler.Faithful scope point 5, L5)?

lr-8a, round 8. Scope point 5 names the scratch of descriptors built in the kernel. Does any other
construct the pinned tutorials use make Triton 3.8.0 ask for global scratch on sm_90? Each kernel
is compiled with `warmup` (no launch) and its `metadata.global_scratch_size` printed.
Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I <this file>   (from TVA's root)
"""
import torch
import triton
import triton.language as tl


@triton.jit
def k_dot(a, b, c, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    r = tl.arange(0, BM)
    n = tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros([BM, BN], tl.float32)
    for i in range(0, 4):
        x = tl.load(a + r[:, None] * BK + k[None, :])
        y = tl.load(b + k[:, None] * BN + n[None, :])
        acc = tl.dot(x, y, acc)
    tl.store(c + r[:, None] * BN + n[None, :], acc)


@triton.jit
def k_misc(x, y, h, B: tl.constexpr):
    o = tl.arange(0, B)
    v = tl.load(x + o)
    s = tl.sort(v)
    cs = tl.cumsum(v, 0)
    am = tl.argmax(v, 0)
    hist = tl.histogram(o % 16, 16)
    g = tl.gather(v, (o * 7) % B, 0)
    tl.store(y + o, s + cs + g + am)
    tl.store(h + tl.arange(0, 16), hist)
    tl.atomic_add(h + o % 16, 1)


@triton.jit
def k_ws(a, b, c, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    r = tl.arange(0, BM)
    n = tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros([BM, BN], tl.float32)
    for i in tl.range(0, K, BK, num_stages=3):
        x = tl.load(a + r[:, None] * K + (i + k)[None, :])
        y = tl.load(b + (i + k)[:, None] * BN + n[None, :])
        acc = tl.dot(x, y, acc)
    tl.store(c + r[:, None] * BN + n[None, :], acc)


@triton.jit
def k_scaled(a, sa, b, sb, c, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    r = tl.arange(0, BM)
    n = tl.arange(0, BN)
    k = tl.arange(0, BK)
    x = tl.load(a + r[:, None] * BK + k[None, :])
    y = tl.load(b + k[:, None] * BN + n[None, :])
    xs = tl.load(sa + r[:, None] * (BK // 32) + tl.arange(0, BK // 32)[None, :])
    ys = tl.load(sb + n[:, None] * (BK // 32) + tl.arange(0, BK // 32)[None, :])
    acc = tl.dot_scaled(x, xs, "e4m3", y, ys, "e4m3")
    tl.store(c + r[:, None] * BN + n[None, :], acc)


def main():
    dev = "cuda"
    BM = BN = 64
    BK = 64
    a = torch.randn(BM * BK, device=dev).to(torch.float16)
    b = torch.randn(BK * BN, device=dev).to(torch.float16)
    c = torch.empty(BM * BN, device=dev, dtype=torch.float32)
    out = []
    ck = k_dot.warmup(a, b, c, BM=BM, BN=BN, BK=BK, grid=(1,))
    out.append(("tl.dot fp16 loop", ck.metadata.global_scratch_size))
    x = torch.randn(128, device=dev)
    y = torch.empty(128, device=dev)
    h = torch.zeros(128, device=dev, dtype=torch.int32)
    ck = k_misc.warmup(x, y, h, B=128, grid=(1,))
    out.append(("sort, cumsum, argmax, histogram, gather, atomic_add", ck.metadata.global_scratch_size))
    ck = k_ws.warmup(a, b, c, 256, BM=BM, BN=BN, BK=BK, grid=(1,))
    out.append(("tl.range num_stages=3 with tl.dot", ck.metadata.global_scratch_size))
    a8 = torch.randn(BM * BK, device=dev).to(torch.float8_e4m3fn)
    b8 = torch.randn(BK * BN, device=dev).to(torch.float8_e4m3fn)
    sa = torch.full((BM * BK // 32,), 127, device=dev, dtype=torch.uint8)
    sb = torch.full((BN * BK // 32,), 127, device=dev, dtype=torch.uint8)
    try:
        ck = k_scaled.warmup(a8, sa, b8, sb, c, BM=BM, BN=BN, BK=BK, grid=(1,))
        out.append(("tl.dot_scaled e4m3 (sm_90 emulation)", ck.metadata.global_scratch_size))
    except Exception as e:
        out.append(("tl.dot_scaled e4m3", "compile error: " + str(e).splitlines()[0][:100]))
    for name, size in out:
        print(f"{name}: global_scratch_size = {size}")
    print("triton", triton.__version__, torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()

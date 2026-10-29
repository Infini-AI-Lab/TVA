"""lr-11b, round 11: does a launch write global memory outside its buffers, for kernels of other
shapes than evidence/launch-frame's? (Library issue L73, `Compiler.Faithful`'s "Memory" frame, and
scope 5: "only descriptors built in the kernel need" the global scratch.)

Every buffer (inputs, outputs, and the allocator's scratch when the kernel asks for one) is carved
out of one sentinel-filled pool with 4096-byte guards. After the launch the script counts the pool
bytes that changed outside the buffers the kernel's program writes (the scratch excepted), and prints
`global_scratch_size` and `profile_scratch_size` from the compiled kernel's metadata, and the
allocator's requests. Each case runs in its own process, on GPU 0, with sentinels 0xA5 and 0x5A.

  W1  host TensorDescriptor loads, tl.dot in a tl.range(warp_specialize=True) loop, descriptor store
  W2  pointer loads, tl.dot in a tl.range(warp_specialize=True) loop
  W3  tl.gather along an axis
  W4  tl.dot_scaled, mxfp8 (e4m3) x mxfp8 with e8m0 scales (emulated on sm_90)
  W5  tl.device_print of a block
  W6  tl.reduce with a custom combine, tl.associative_scan, tl.argmax
  W7  tl.join, tl.split, tl.permute, tl.trans, tl.reshape
  W8  a 128 x 256 x 64 tl.dot loop with num_stages=4 (pipelined through shared memory)
  W9  a persistent tl.range(flatten=True) loop over tiles with descriptors built in the kernel
  W10 tl.atomic_cas and tl.atomic_xchg, scalar and block
  W11 float16 tl.atomic_add on a block with a mask that ends inside a pair (packed f16x2 candidate)
  W12 tl.cumsum on a 2-D block along each axis, tl.sort (descending, last axis), tl.flip

Run (from TVA's root): CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I <this file>
"""
import json, os, subprocess, sys

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

GUARD = 4096


@triton.jit
def w1(a_desc, b_desc, c_desc, KT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in tl.range(KT, warp_specialize=True):
        a = a_desc.load([0, k * BK])
        b = b_desc.load([0, k * BK])
        acc = tl.dot(a, b.T, acc)
    c_desc.store([0, 0], acc.to(tl.float16))


@triton.jit
def w2(a_ptr, b_ptr, c_ptr, KT: tl.constexpr, B: tl.constexpr):
    i = tl.arange(0, B)
    acc = tl.zeros((B, B), dtype=tl.float32)
    for k in tl.range(KT, warp_specialize=True):
        a = tl.load(a_ptr + i[:, None] * (B * KT) + k * B + i[None, :])
        b = tl.load(b_ptr + (k * B + i[:, None]) * B + i[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + i[:, None] * B + i[None, :], acc)


@triton.jit
def w3(x_ptr, idx_ptr, o_ptr):
    r = tl.arange(0, 32)
    c = tl.arange(0, 64)
    x = tl.load(x_ptr + r[:, None] * 64 + c[None, :])
    idx = tl.load(idx_ptr + r[:, None] * 64 + c[None, :])
    tl.store(o_ptr + r[:, None] * 64 + c[None, :], tl.gather(x, idx, 1))


@triton.jit
def w4(a_ptr, as_ptr, b_ptr, bs_ptr, c_ptr):
    m = tl.arange(0, 64)
    k = tl.arange(0, 64)
    ks = tl.arange(0, 2)
    a = tl.load(a_ptr + m[:, None] * 64 + k[None, :])
    b = tl.load(b_ptr + k[:, None] * 64 + m[None, :])
    sa = tl.load(as_ptr + m[:, None] * 2 + ks[None, :])
    sb = tl.load(bs_ptr + m[:, None] * 2 + ks[None, :])
    c = tl.dot_scaled(a, sa, "e4m3", b, sb, "e4m3")
    tl.store(c_ptr + m[:, None] * 64 + m[None, :], c)


@triton.jit
def w5(x_ptr, o_ptr):
    i = tl.arange(0, 32)
    x = tl.load(x_ptr + i)
    tl.device_print("x", x)
    tl.store(o_ptr + i, x + 1)


@triton.jit
def _comb(a, b):
    return a * 2 + b


@triton.jit
def w6(x_ptr, r_ptr, s_ptr, a_ptr):
    i = tl.arange(0, 128)
    x = tl.load(x_ptr + i)
    tl.store(r_ptr, tl.reduce(x, 0, _comb))
    tl.store(s_ptr + i, tl.associative_scan(x, 0, _comb))
    tl.store(a_ptr, tl.argmax(x, 0))


@triton.jit
def w7(x_ptr, y_ptr, o1_ptr, o2_ptr, o3_ptr):
    r = tl.arange(0, 16)
    c = tl.arange(0, 32)
    x = tl.load(x_ptr + r[:, None] * 32 + c[None, :])
    y = tl.load(y_ptr + r[:, None] * 32 + c[None, :])
    j = tl.join(x, y)                      # [16, 32, 2]
    p = tl.permute(j, (2, 0, 1))           # [2, 16, 32]
    a, b = tl.split(tl.permute(p, (1, 2, 0)))
    t = tl.trans(a + b)                    # [32, 16]
    tl.store(o1_ptr + c[:, None] * 16 + r[None, :], t)
    q = tl.reshape(x, (32, 16))
    tl.store(o2_ptr + c[:, None] * 16 + r[None, :], q)
    tl.store(o3_ptr + tl.arange(0, 1024), tl.reshape(p, (1024,)))


@triton.jit
def w8(a_ptr, b_ptr, c_ptr, KT: tl.constexpr):
    m = tl.arange(0, 128)
    n = tl.arange(0, 256)
    kk = tl.arange(0, 64)
    acc = tl.zeros((128, 256), dtype=tl.float32)
    for k in tl.range(KT, num_stages=4):
        a = tl.load(a_ptr + m[:, None] * (64 * KT) + k * 64 + kk[None, :])
        b = tl.load(b_ptr + (k * 64 + kk[:, None]) * 256 + n[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + m[:, None] * 256 + n[None, :], acc)


@triton.jit
def w9(x_ptr, y_ptr, M, N, NT, NUM: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    xd = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    yd = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    tiles_n = N // BN
    for t in tl.range(tl.program_id(0), NT, NUM, flatten=True):
        pm = t // tiles_n
        pn = t % tiles_n
        yd.store([pm * BM, pn * BN], xd.load([pm * BM, pn * BN]) * 2)


@triton.jit
def w10(c_ptr, x_ptr, o_ptr):
    pid = tl.program_id(0)
    old = tl.atomic_cas(c_ptr + pid, 0, pid + 1)
    tl.store(o_ptr + pid, old)
    i = tl.arange(0, 16)
    tl.atomic_xchg(x_ptr + pid * 16 + i, i + pid)


@triton.jit
def w11(c_ptr, n):
    i = tl.arange(0, 64)
    tl.atomic_add(c_ptr + i, tl.full([64], 1.0, tl.float16), mask=i < n)


@triton.jit
def w12(x_ptr, o1_ptr, o2_ptr, o3_ptr, o4_ptr):
    r = tl.arange(0, 16)
    c = tl.arange(0, 32)
    x = tl.load(x_ptr + r[:, None] * 32 + c[None, :])
    off = r[:, None] * 32 + c[None, :]
    tl.store(o1_ptr + off, tl.cumsum(x, 0))
    tl.store(o2_ptr + off, tl.cumsum(x, 1))
    tl.store(o3_ptr + off, tl.sort(x, 1, descending=True))
    tl.store(o4_ptr + off, tl.flip(x, 1))


class Pool:
    def __init__(self, nbytes, sentinel):
        self.mem = torch.full((nbytes,), sentinel, dtype=torch.uint8, device="cuda")
        self.off = GUARD
        self.spans = {}

    def take(self, name, numel, dtype):
        size = numel * torch.empty((), dtype=dtype).element_size()
        start = (self.off + 255) // 256 * 256
        self.spans[name] = (start, start + size)
        self.off = start + size + GUARD
        return self.mem[start:start + size].view(dtype)


def run(name, sentinel):
    pool = Pool(1 << 24, sentinel)
    requests = []

    def alloc(size, align, stream):
        requests.append(size)
        return pool.take("scratch%d" % len(requests), size, torch.int8)

    triton.set_allocator(alloc)
    g = torch.Generator().manual_seed(1)
    if name == "W1":
        BM, BN, BK, KT = 64, 64, 64, 4
        a = pool.take("a", BM * BK * KT, torch.float16); b = pool.take("b", BN * BK * KT, torch.float16)
        c = pool.take("c", BM * BN, torch.float16)
        a.copy_(torch.randint(-2, 3, (BM * BK * KT,), generator=g).half()); b.copy_(torch.randint(-2, 3, (BN * BK * KT,), generator=g).half())
        ad = TensorDescriptor(a.view(BM, BK * KT), [BM, BK * KT], [BK * KT, 1], [BM, BK])
        bd = TensorDescriptor(b.view(BN, BK * KT), [BN, BK * KT], [BK * KT, 1], [BN, BK])
        cd = TensorDescriptor(c.view(BM, BN), [BM, BN], [BN, 1], [BM, BN])
        before = pool.mem.clone()
        ck = w1[(1,)](ad, bd, cd, KT=KT, BM=BM, BN=BN, BK=BK)
        torch.cuda.synchronize()
        ref = (a.view(BM, BK * KT).float() @ b.view(BN, BK * KT).float().T).half()
        ok = torch.equal(c.view(BM, BN).cpu(), ref.cpu()); allowed = ["c"]
    elif name == "W2":
        B, KT = 64, 4
        a = pool.take("a", B * B * KT, torch.float16); b = pool.take("b", B * KT * B, torch.float16)
        c = pool.take("c", B * B, torch.float32)
        a.copy_(torch.randint(-2, 3, (B * B * KT,), generator=g).half()); b.copy_(torch.randint(-2, 3, (B * KT * B,), generator=g).half())
        before = pool.mem.clone()
        ck = w2[(1,)](a, b, c, KT=KT, B=B)
        torch.cuda.synchronize()
        ok = torch.equal(c.view(B, B).cpu(), (a.view(B, B * KT).float() @ b.view(B * KT, B).float()).cpu()); allowed = ["c"]
    elif name == "W3":
        x = pool.take("x", 32 * 64, torch.float32); idx = pool.take("idx", 32 * 64, torch.int32)
        o = pool.take("o", 32 * 64, torch.float32)
        x.copy_(torch.randn(32 * 64, generator=g)); idx.copy_(torch.randint(0, 64, (32 * 64,), generator=g, dtype=torch.int32))
        before = pool.mem.clone()
        ck = w3[(1,)](x, idx, o)
        torch.cuda.synchronize()
        ok = torch.equal(o.view(32, 64).cpu(), torch.gather(x.view(32, 64).cpu(), 1, idx.view(32, 64).cpu().long())); allowed = ["o"]
    elif name == "W4":
        a = pool.take("a", 64 * 64, torch.uint8); b = pool.take("b", 64 * 64, torch.uint8)
        sa = pool.take("sa", 64 * 2, torch.uint8); sb = pool.take("sb", 64 * 2, torch.uint8)
        c = pool.take("c", 64 * 64, torch.float32)
        a.copy_(torch.randn(64 * 64, generator=g).to(torch.float8_e4m3fn).view(torch.uint8))
        b.copy_(torch.randn(64 * 64, generator=g).to(torch.float8_e4m3fn).view(torch.uint8))
        sa.fill_(127); sb.fill_(127)
        before = pool.mem.clone()
        ck = w4[(1,)](a.view(torch.float8_e4m3fn), sa, b.view(torch.float8_e4m3fn), sb, c)
        torch.cuda.synchronize()
        ref = a.view(torch.float8_e4m3fn).float().view(64, 64) @ b.view(torch.float8_e4m3fn).float().view(64, 64)
        ok = bool(torch.allclose(c.view(64, 64), ref, rtol=1e-5, atol=1e-5)); allowed = ["c"]
    elif name == "W5":
        x = pool.take("x", 32, torch.float32); o = pool.take("o", 32, torch.float32)
        x.copy_(torch.arange(32, dtype=torch.float32))
        before = pool.mem.clone()
        ck = w5[(1,)](x, o)
        torch.cuda.synchronize()
        ok = torch.equal(o.cpu(), torch.arange(32, dtype=torch.float32) + 1); allowed = ["o"]
    elif name == "W6":
        x = pool.take("x", 128, torch.int32); r = pool.take("r", 1, torch.int32)
        s = pool.take("s", 128, torch.int32); am = pool.take("am", 1, torch.int32)
        x.copy_(torch.randint(0, 3, (128,), generator=g, dtype=torch.int32))
        before = pool.mem.clone()
        ck = w6[(1,)](x, r, s, am)
        torch.cuda.synchronize()
        ok = int(am.item()) == int(torch.argmax(x.cpu()).item()); allowed = ["r", "s", "am"]
    elif name == "W7":
        x = pool.take("x", 512, torch.float32); y = pool.take("y", 512, torch.float32)
        o1 = pool.take("o1", 512, torch.float32); o2 = pool.take("o2", 512, torch.float32)
        o3 = pool.take("o3", 1024, torch.float32)
        x.copy_(torch.randn(512, generator=g)); y.copy_(torch.randn(512, generator=g))
        before = pool.mem.clone()
        ck = w7[(1,)](x, y, o1, o2, o3)
        torch.cuda.synchronize()
        ok = torch.equal(o1.view(32, 16).cpu(), (x + y).view(16, 32).T.cpu()); allowed = ["o1", "o2", "o3"]
    elif name == "W8":
        KT = 2
        a = pool.take("a", 128 * 64 * KT, torch.float16); b = pool.take("b", 64 * KT * 256, torch.float16)
        c = pool.take("c", 128 * 256, torch.float32)
        a.copy_(torch.randint(-2, 3, (128 * 64 * KT,), generator=g).half()); b.copy_(torch.randint(-2, 3, (64 * KT * 256,), generator=g).half())
        before = pool.mem.clone()
        ck = w8[(1,)](a, b, c, KT=KT)
        torch.cuda.synchronize()
        ok = torch.equal(c.view(128, 256).cpu(), (a.view(128, 64 * KT).float() @ b.view(64 * KT, 256).float()).cpu()); allowed = ["c"]
    elif name == "W9":
        M = N = 128
        x = pool.take("x", M * N, torch.float32); y = pool.take("y", M * N, torch.float32)
        x.copy_(torch.arange(M * N, dtype=torch.float32))
        before = pool.mem.clone()
        ck = w9[(4,)](x, y, M, N, (M // 32) * (N // 32), NUM=4, BM=32, BN=32)
        torch.cuda.synchronize()
        ok = torch.equal(y.cpu(), torch.arange(M * N, dtype=torch.float32) * 2); allowed = ["y"]
    elif name == "W10":
        cc = pool.take("cc", 8, torch.int32); xx = pool.take("xx", 128, torch.int32); oo = pool.take("oo", 8, torch.int32)
        cc.zero_(); xx.zero_()
        before = pool.mem.clone()
        ck = w10[(8,)](cc, xx, oo)
        torch.cuda.synchronize()
        ok = cc.cpu().tolist() == list(range(1, 9)) and oo.cpu().tolist() == [0] * 8; allowed = ["cc", "xx", "oo"]
    elif name == "W11":
        n = 37
        cbuf = pool.take("c", n, torch.float16)
        cbuf.fill_(-0.0)
        # the element just past the buffer (in the guard) holds -0.0 too: a packed add of 0 would make it +0.0
        s0, s1 = pool.spans["c"]
        pool.mem[s1:s1 + 2] = torch.tensor([-0.0], dtype=torch.float16).view(torch.uint8).cuda()
        before = pool.mem.clone()
        ck = w11[(1,)](cbuf, n)
        torch.cuda.synchronize()
        ok = cbuf.cpu().tolist() == [1.0] * n; allowed = ["c"]
    else:
        x = pool.take("x", 512, torch.int32)
        o = [pool.take("o%d" % i, 512, torch.int32) for i in range(4)]
        x.copy_(torch.randint(-50, 50, (512,), generator=g, dtype=torch.int32))
        before = pool.mem.clone()
        ck = w12[(1,)](x, *o)
        torch.cuda.synchronize()
        xv = x.view(16, 32).cpu()
        ok = torch.equal(o[0].view(16, 32).cpu(), torch.cumsum(xv, 0).int()) and torch.equal(o[3].view(16, 32).cpu(), torch.flip(xv, [1]))
        allowed = ["o0", "o1", "o2", "o3"]
    allowed += [k for k in pool.spans if k.startswith("scratch")]
    diff = (pool.mem != before).nonzero().flatten().cpu().tolist()
    outside = [b for b in diff if not any(pool.spans[n][0] <= b < pool.spans[n][1] for n in allowed)]
    return {"correct": bool(ok), "changed": len(diff), "changed outside the written buffers and scratch": len(outside),
            "global_scratch_size": ck.metadata.global_scratch_size,
            "profile_scratch_size": ck.metadata.profile_scratch_size, "allocator requests": requests}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        res = {}
        for s in (0xA5, 0x5A):
            try:
                res[hex(s)] = run(sys.argv[2], s)
            except Exception as e:
                msg = [l for l in str(e).strip().splitlines() if l.strip()]
                res[hex(s)] = "error: " + type(e).__name__ + ": " + (msg[0] if msg else "")[:200]
        print("RESULT " + json.dumps(res))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name in ["W%d" % i for i in range(1, 13)]:
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                           capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        res = json.loads(line[len("RESULT "):]) if line else "process failed: " + p.stderr.strip()[-300:]
        print(name + ":")
        if isinstance(res, dict):
            for k, v in res.items():
                print("  sentinel %s: %s" % (k, v))
        else:
            print("  " + res)


if __name__ == "__main__":
    main()

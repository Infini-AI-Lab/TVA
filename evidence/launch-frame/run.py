"""Which global memory does a launch write besides the buffers its program writes?

Library evidence for library issue L73 (round 11). The launcher passes a kernel two buffers of its
own besides the arguments: the global scratch (`metadata.global_scratch_size` bytes per program, from
the allocator `triton.set_allocator` sets) and the profile scratch (`metadata.profile_scratch_size`,
for instrumentation) (`primitives/triton-3.8.0-nvidia-driver.py` lines 280-333). Every buffer here
(inputs, outputs, the allocator's scratch) is carved out of one pool filled with a sentinel byte,
with guard regions of 4096 bytes between them; after the launch, the script compares every byte of
the pool with its contents just before the launch, outside the buffers the kernel's program writes
(and, for K6, the allocator's scratch). It runs each kernel with
sentinels 0xA5 and 0x5A (so a byte that happens to equal a sentinel is still seen), checks the
results against the host, and prints both scratch sizes from the compiled kernel's metadata. Each
kernel runs in its own process, on GPU 0.

  K1  vector add, n = 1000 with BLOCK = 128 (a partial last tile, masked)
  K2  a row softmax (tl.max, tl.exp, tl.sum), 64 rows of 100 in blocks of 128 (masked)
  K3  a tl.dot of float16 64 x 64 x 64 blocks, float32 out, a K loop with tl.range(num_stages=3)
  K4  integer and float tl.atomic_add into 16 counters from 64 programs
  K5  tl.sort, tl.cumsum and tl.histogram of a block
  K6  a copy through descriptors built in the kernel (tl.make_tensor_descriptor); the allocator's
      scratch is a region of the pool, which may change, and nothing else outside the output

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/launch-frame/run.py  (from TVA's root)
"""
import json, os, subprocess, sys

import torch
import triton
import triton.language as tl

GUARD = 4096


@triton.jit
def k_add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)


@triton.jit
def k_softmax(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    m = c < n
    x = tl.load(x_ptr + r * n + c, mask=m, other=-float("inf"))
    e = tl.exp(x - tl.max(x, axis=0))
    tl.store(o_ptr + r * n + c, e / tl.sum(e, axis=0), mask=m)


@triton.jit
def k_dot(a_ptr, b_ptr, c_ptr, B: tl.constexpr, KT: tl.constexpr):
    i = tl.arange(0, B)
    acc = tl.zeros([B, B], dtype=tl.float32)
    for k in tl.range(0, KT, num_stages=3):
        a = tl.load(a_ptr + i[:, None] * (B * KT) + k * B + i[None, :])
        b = tl.load(b_ptr + (k * B + i[:, None]) * B + i[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + i[:, None] * B + i[None, :], acc)


@triton.jit
def k_atomic(ci_ptr, cf_ptr):
    j = tl.arange(0, 16)
    tl.atomic_add(ci_ptr + j, tl.program_id(0) + j)
    tl.atomic_add(cf_ptr + j, 1.0)


@triton.jit
def k_sortscan(x_ptr, s_ptr, c_ptr, h_ptr):
    i = tl.arange(0, 128)
    x = tl.load(x_ptr + i)
    tl.store(s_ptr + i, tl.sort(x))
    tl.store(c_ptr + i, tl.cumsum(x, 0))
    tl.store(h_ptr + tl.arange(0, 16), tl.histogram(x % 16, 16))


@triton.jit
def k_desc(x_ptr, y_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    xd = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    yd = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    yd.store([pm * BM, pn * BN], xd.load([pm * BM, pn * BN]) + 1)


class Pool:
    """Buffers carved out of one sentinel-filled byte pool, each with a guard after it."""

    def __init__(self, nbytes, sentinel):
        self.mem = torch.full((nbytes,), sentinel, dtype=torch.uint8, device="cuda")
        self.sentinel = sentinel
        self.off = GUARD
        self.spans = {}

    def take(self, name, numel, dtype):
        size = numel * torch.empty((), dtype=dtype).element_size()
        start = (self.off + 255) // 256 * 256
        self.spans[name] = (start, start + size)
        self.off = start + size + GUARD
        return self.mem[start:start + size].view(dtype)


def run_kernel(name, sentinel):
    pool = Pool(1 << 22, sentinel)
    meta = {}
    if name == "K1":
        n = 1000
        x = pool.take("x", n, torch.float32); y = pool.take("y", n, torch.float32)
        o = pool.take("o", n, torch.float32)
        x.copy_(torch.arange(n, dtype=torch.float32)); y.fill_(1.0)
        before = pool.mem.clone()
        ck = k_add[(8,)](x, y, o, n, BLOCK=128)
        torch.cuda.synchronize()
        ok = torch.equal(o.cpu(), torch.arange(n, dtype=torch.float32) + 1)
        allowed = ["o"]
    elif name == "K2":
        n = 100
        x = pool.take("x", 64 * n, torch.float32); o = pool.take("o", 64 * n, torch.float32)
        x.copy_(torch.randn(64 * n))
        before = pool.mem.clone()
        ck = k_softmax[(64,)](x, o, n, BLOCK=128)
        torch.cuda.synchronize()
        ok = torch.allclose(o.view(64, n).cpu(), torch.softmax(x.view(64, n).cpu(), 1), atol=1e-6)
        allowed = ["o"]
    elif name == "K3":
        B, KT = 64, 4
        a = pool.take("a", B * B * KT, torch.float16); b = pool.take("b", B * KT * B, torch.float16)
        c = pool.take("c", B * B, torch.float32)
        a.copy_(torch.randint(-3, 4, (B * B * KT,)).half()); b.copy_(torch.randint(-3, 4, (B * KT * B,)).half())
        before = pool.mem.clone()
        ck = k_dot[(1,)](a, b, c, B=B, KT=KT)
        torch.cuda.synchronize()
        ok = torch.equal(c.view(B, B).cpu(), (a.view(B, B * KT).float() @ b.view(B * KT, B).float()).cpu())
        allowed = ["c"]
    elif name == "K4":
        ci = pool.take("ci", 16, torch.int32); cf = pool.take("cf", 16, torch.float32)
        ci.zero_(); cf.zero_()
        before = pool.mem.clone()
        ck = k_atomic[(64,)](ci, cf)
        torch.cuda.synchronize()
        ok = ci.cpu().tolist() == [sum(range(64)) + 64 * j for j in range(16)] and cf.cpu().tolist() == [64.0] * 16
        allowed = ["ci", "cf"]
    elif name == "K5":
        x = pool.take("x", 128, torch.int32); s = pool.take("s", 128, torch.int32)
        c = pool.take("c", 128, torch.int32); h = pool.take("h", 16, torch.int32)
        x.copy_(torch.randint(0, 1000, (128,), dtype=torch.int32))
        before = pool.mem.clone()
        ck = k_sortscan[(1,)](x, s, c, h)
        torch.cuda.synchronize()
        xh = x.cpu()
        ok = (torch.equal(s.cpu(), torch.sort(xh).values) and torch.equal(c.cpu(), torch.cumsum(xh, 0).int())
              and torch.equal(h.cpu(), torch.bincount(xh % 16, minlength=16).int()))
        allowed = ["s", "c", "h"]
    else:
        M = N = 128
        x = pool.take("x", M * N, torch.float32); y = pool.take("y", M * N, torch.float32)
        x.copy_(torch.arange(M * N, dtype=torch.float32))
        requests = []

        def alloc(size, align, stream):
            requests.append(size)
            return pool.take("scratch", size, torch.int8)

        triton.set_allocator(alloc)
        before = pool.mem.clone()
        ck = k_desc[(4, 4)](x, y, M, N, BM=32, BN=32)
        torch.cuda.synchronize()
        ok = torch.equal(y.cpu(), torch.arange(M * N, dtype=torch.float32) + 1)
        allowed = ["y", "scratch"]
        meta["scratch requests"] = requests
    # bytes that differ from the pool before the launch, outside the allowed buffers
    diff = (pool.mem != before).nonzero().flatten().cpu().tolist()
    outside = [b for b in diff if not any(pool.spans[n][0] <= b < pool.spans[n][1] for n in allowed)]
    meta.update({"correct": bool(ok), "bytes changed by the launch": len(diff),
                 "changed outside the written buffers": len(outside),
                 "global_scratch_size": ck.metadata.global_scratch_size,
                 "profile_scratch_size": ck.metadata.profile_scratch_size})
    return meta


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        res = {}
        for sentinel in (0xA5, 0x5A):
            try:
                res[hex(sentinel)] = run_kernel(sys.argv[2], sentinel)
            except Exception as e:
                res[hex(sentinel)] = "error: " + type(e).__name__ + ": " + str(e).strip().splitlines()[0][:150]
        print("RESULT " + json.dumps(res))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name in ("K1", "K2", "K3", "K4", "K5", "K6"):
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                           capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        res = json.loads(line[len("RESULT "):]) if line else "process failed: " + p.stderr.strip()[-200:]
        print(f"{name}:")
        if isinstance(res, dict):
            for k, v in res.items():
                print(f"  sentinel {k}: {v}")
        else:
            print("  " + res)


if __name__ == "__main__":
    main()

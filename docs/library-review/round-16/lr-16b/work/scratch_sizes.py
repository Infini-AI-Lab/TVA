"""lr-16b: which constructs make the launcher pass a global scratch buffer (Compiler.Faithful, "Memory").

The rule's frame says that a valid, well-behaved launch in scope writes no byte other than those
`init` gives, "except the scratch of the descriptors the kernel builds", since the launcher's only
other buffers are that scratch and instrumentation's (`nvidia-driver.py:280-333`: it allocates
`global_scratch_size` bytes per program from the allocator whenever the compiled kernel asks for
any). evidence/launch-frame measured 0 for six kernels. This sweep compiles more constructs of the
supported subset on the H200 and prints each compiled kernel's `global_scratch_size` and
`profile_scratch_size`: a non-zero size for a kernel that builds no descriptor would make the
frame's exception incomplete.

Triton 3.8.0, GPU 0 (an H200 NVL). Run: CUDA_VISIBLE_DEVICES=0 python -I scratch_sizes.py
"""
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def alloc(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


triton.set_allocator(alloc)


@triton.jit
def k_gather(x_ptr, i_ptr, o_ptr):
    r = tl.arange(0, 16)
    x = tl.load(x_ptr + r[:, None] * 16 + r[None, :])
    idx = tl.load(i_ptr + r[:, None] * 16 + r[None, :])
    tl.store(o_ptr + r[:, None] * 16 + r[None, :], tl.gather(x, idx, 1))


@triton.jit
def k_ws(a_ptr, b_ptr, c_ptr, K: tl.constexpr):
    r = tl.arange(0, 64)
    acc = tl.zeros((64, 64), tl.float32)
    for k in tl.range(0, K, 64, warp_specialize=True):
        a = tl.load(a_ptr + r[:, None] * K + k + r[None, :])
        b = tl.load(b_ptr + (k + r[:, None]) * 64 + r[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + r[:, None] * 64 + r[None, :], acc)


@triton.jit
def k_hostdesc(d_in, d_out):
    x = d_in.load([0, 0])
    d_out.store([0, 0], x)


@triton.jit
def k_hostdesc_atomic(d_in, d_out):
    x = d_in.load([0, 0])
    d_out.atomic_add([0, 0], x)


@triton.jit
def k_blockptr(x_ptr, o_ptr, n):
    p = tl.make_block_ptr(x_ptr, (n, n), (n, 1), (0, 0), (32, 32), (1, 0))
    q = tl.make_block_ptr(o_ptr, (n, n), (n, 1), (0, 0), (32, 32), (1, 0))
    tl.store(q, tl.load(p, boundary_check=(0, 1), padding_option="zero"), boundary_check=(0, 1))


@triton.jit
def k_scan(x_ptr, o_ptr):
    r = tl.arange(0, 256)
    x = tl.load(x_ptr + r)
    tl.store(o_ptr + r, tl.cumsum(x, 0, reverse=True) + tl.argmax(x, 0))


@triton.jit
def k_rand(o_ptr, seed):
    r = tl.arange(0, 256)
    tl.store(o_ptr + r, tl.rand(seed, r))


@triton.jit
def k_shapes(x_ptr, o_ptr):
    r = tl.arange(0, 32)
    x = tl.load(x_ptr + r[:, None] * 32 + r[None, :])
    y = tl.reshape(tl.permute(tl.reshape(x, (4, 8, 32)), (2, 1, 0)), (32, 32))
    a, b = tl.split(tl.reshape(y, (32, 16, 2)))
    tl.store(o_ptr + r[:, None] * 16 + tl.arange(0, 16)[None, :], a + b)


@triton.jit
def k_dot8(a_ptr, b_ptr, c_ptr):
    r = tl.arange(0, 64)
    a = tl.load(a_ptr + r[:, None] * 64 + r[None, :])
    b = tl.load(b_ptr + r[:, None] * 64 + r[None, :])
    tl.store(c_ptr + r[:, None] * 64 + r[None, :], tl.dot(a, b))


@triton.jit
def k_dot_scaled(a_ptr, b_ptr, sa_ptr, sb_ptr, c_ptr):
    r = tl.arange(0, 128)
    a = tl.load(a_ptr + r[:, None] * 128 + r[None, :])
    b = tl.load(b_ptr + r[:, None] * 128 + r[None, :])
    s = tl.arange(0, 4)
    sa = tl.load(sa_ptr + r[:, None] * 4 + s[None, :])
    sb = tl.load(sb_ptr + r[:, None] * 4 + s[None, :])
    tl.store(c_ptr + r[:, None] * 128 + r[None, :], tl.dot_scaled(a, sa, "e4m3", b.T, sb, "e4m3"))


@triton.jit
def k_devdesc(x_ptr, o_ptr, n):
    d = tl.make_tensor_descriptor(x_ptr, [n, n], [n, 1], [16, 16])
    e = tl.make_tensor_descriptor(o_ptr, [n, n], [n, 1], [16, 16])
    e.store([0, 0], d.load([0, 0]))


def sizes(name, ck):
    md = ck.metadata
    print(f"- {name}: global_scratch_size {md.global_scratch_size}, profile_scratch_size {md.profile_scratch_size}")


def main():
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    dev = "cuda"
    f32 = lambda *s: torch.randn(*s, device=dev)  # noqa: E731
    x = f32(16, 16)
    idx = torch.randint(0, 16, (16, 16), device=dev, dtype=torch.int32)
    sizes("tl.gather", k_gather[(1,)](x, idx, torch.empty_like(x)))
    a = f32(64, 256).half()
    b = f32(256, 64).half()
    try:
        sizes("tl.range(warp_specialize=True) with tl.dot", k_ws[(1,)](a, b, torch.empty(64, 64, device=dev), K=256))
    except Exception as e:  # noqa: BLE001
        print(f"- tl.range(warp_specialize=True): {type(e).__name__}: {str(e).splitlines()[0][:160]}")
    t = f32(64, 64)
    o = torch.zeros(64, 64, device=dev)
    di = TensorDescriptor.from_tensor(t, [16, 16])
    do = TensorDescriptor.from_tensor(o, [16, 16])
    sizes("host TensorDescriptor load and store", k_hostdesc[(1,)](di, do))
    sizes("host TensorDescriptor atomic_add", k_hostdesc_atomic[(1,)](di, do))
    sizes("tl.make_block_ptr with boundary_check", k_blockptr[(1,)](f32(40, 40), torch.empty(40, 40, device=dev), 40))
    sizes("tl.cumsum(reverse=True) and tl.argmax", k_scan[(1,)](f32(256), torch.empty(256, device=dev)))
    sizes("tl.rand", k_rand[(1,)](torch.empty(256, device=dev), 7))
    sizes("tl.reshape, tl.permute, tl.split", k_shapes[(1,)](f32(32, 32), torch.empty(32, 16, device=dev)))
    a8 = f32(64, 64).to(torch.float8_e4m3fn)
    b8 = f32(64, 64).to(torch.float8_e4m3fn)
    sizes("tl.dot of float8e4nv", k_dot8[(1,)](a8, b8, torch.empty(64, 64, device=dev)))
    sa = torch.full((128, 4), 127, dtype=torch.uint8, device=dev)
    try:
        sizes("tl.dot_scaled e4m3", k_dot_scaled[(1,)](f32(128, 128).to(torch.float8_e4m3fn),
                                                       f32(128, 128).to(torch.float8_e4m3fn), sa, sa,
                                                       torch.empty(128, 128, device=dev)))
    except Exception as e:  # noqa: BLE001
        print(f"- tl.dot_scaled: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
    sizes("tl.make_tensor_descriptor (two, the control)", k_devdesc[(1,)](f32(64, 64), torch.empty(64, 64, device=dev), 64))
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()

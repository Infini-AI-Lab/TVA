"""In-kernel descriptors at the edges the library's `tl.make_tensor_descriptor` admits (L5).

lr-8a, round 8. `TVA.tl.make_tensor_descriptor` accepts every non-negative shape entry and
stride, and `TensorDesc.Valid` asks no lower bound of them, so in the model a zero dimension makes
every lane out of bounds (loads give the padding, stores write nothing) and a zero leading stride
reads one row repeatedly. What does the device do? Each case in its own process, on GPU 0.

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I <this file>   (from TVA's root)
"""
import json, os, subprocess, sys

import torch
import triton
import triton.language as tl

BM = BN = 32


@triton.jit
def k_copy(x_ptr, y_ptr, M, N, SX, BM: tl.constexpr, BN: tl.constexpr):
    # x through a descriptor of shape [M, N], strides [SX, 1]; y written through plain pointers
    xd = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[SX, 1], block_shape=[BM, BN])
    v = xd.load([0, 0])
    r = tl.arange(0, BM)
    c = tl.arange(0, BN)
    tl.store(y_ptr + r[:, None] * BN + c[None, :], v)


@triton.jit
def k_store(y_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    yd = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    yd.store([0, 0], tl.full([BM, BN], 7.0, tl.float32))


@triton.jit
def k_build_only(x_ptr, y_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    xd = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    if M > 0:                      # a runtime branch no program takes when M = 0
        v = xd.load([0, 0])
        tl.store(y_ptr + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :], v)
    tl.store(y_ptr, 5.0)


@triton.jit
def k_host(xd, y_ptr, BM: tl.constexpr, BN: tl.constexpr):
    v = xd.load([0, 0])
    tl.store(y_ptr + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :], v)


CASES = {
    "E1": "load, shape [32, 32], stride 32 (control)",
    "E2": "load, shape [0, 32]: model: every lane the padding 0",
    "E3": "load, shape [32, 0]: model: every lane the padding 0",
    "E4": "load, shape [32, 32], leading stride 0: model: row 0 in every row",
    "E5": "store, shape [0, 32]: model: nothing written",
    "E6": "host-built TensorDescriptor, shape [0, 32], load: model: every lane the padding 0",
    "E7": "in-kernel, shape [0, 32], built but not accessed (runtime branch not taken): model: y[0] = 5",
    "E8": "load, shape [1, 32], leading stride 2^40 elements (only row 0 in bounds): model: row 0, then padding",
}


def run_case(name):
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
    x = torch.arange(32 * 32, dtype=torch.float32, device="cuda") + 1
    y = torch.full((BM * BN,), -1.0, device="cuda")
    if name == "E6":
        from triton.tools.tensor_descriptor import TensorDescriptor
        xd = TensorDescriptor(x, shape=[0, 32], strides=[32, 1], block_shape=[BM, BN])
        k_host[(1,)](xd, y, BM=BM, BN=BN)
        torch.cuda.synchronize()
        ok = bool((y == 0).all())
        return {"result": "as the model" if ok else "DIFFERS: " + str(y[:4].tolist())}
    if name == "E7":
        k_build_only[(1,)](x, y, 0, 32, BM=BM, BN=BN)
        torch.cuda.synchronize()
        ok = y[0].item() == 5.0 and bool((y[1:] == -1).all())
        return {"result": "as the model" if ok else "DIFFERS: " + str(y[:4].tolist())}
    if name == "E5":
        k_store[(1,)](y, 0, 32, BM=BM, BN=BN)
        torch.cuda.synchronize()
        ok = bool((y == -1).all())
        return {"result": "as the model (nothing written)" if ok else "DIFFERS: " + str(y[:4].tolist())}
    M, N, SX = {"E1": (32, 32, 32), "E2": (0, 32, 32), "E3": (32, 0, 32), "E4": (32, 32, 0),
                "E8": (1, 32, 2 ** 40)}[name]
    k_copy[(1,)](x, y, M, N, SX, BM=BM, BN=BN)
    torch.cuda.synchronize()
    xh = x.cpu().view(32, 32)
    if name == "E1":
        ref = xh.flatten()
    elif name in ("E2", "E3"):
        ref = torch.zeros(BM * BN)
    elif name == "E8":
        ref = torch.cat([xh[0], torch.zeros(31 * 32)])
    else:
        ref = xh[0].repeat(32)
    got = y.cpu()
    ok = torch.equal(got, ref)
    return {"result": "as the model" if ok else "DIFFERS: first values " + str(got[:4].tolist()) +
            ", row 1 " + str(got[32:36].tolist())}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        try:
            res = run_case(sys.argv[2])
        except Exception as e:
            res = {"result": "error: " + type(e).__name__ + ": " + str(e).splitlines()[0][:140]}
        print("RESULT " + json.dumps(res))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name, desc in CASES.items():
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                           capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line:
            res = json.loads(line[len("RESULT "):])
        else:
            err = [l for l in p.stderr.strip().splitlines() if l.strip()]
            res = {"result": "process failed: " + (err[-1][:140] if err else "(no output)")}
        print(f"{name}: {desc}\n  {res['result']}")


if __name__ == "__main__":
    main()

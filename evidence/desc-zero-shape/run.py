"""Accesses through a tensor descriptor with a zero shape entry.

Library evidence for library issue L5 (round 11). `tl.make_tensor_descriptor` accepts a shape entry
of 0 at compile time and at run time (the shape is a runtime argument). In TVA's model before
round 11 such a descriptor was valid: every lane is out of bounds, so a load gives the padding, a
store or a reduction writes nothing, a gather gives unspecified values. What does the device do?
Each case runs in its own process (a fault ends the process's CUDA context), on GPU 0:

  Z1   load, in-kernel descriptor of shape [32, 32] (control)
  Z2   load, shape [0, 32]
  Z3   load, shape [32, 0]
  Z4   load, shape [1, 32] (the smallest positive entry; only row 0 in bounds)
  Z5   store, shape [0, 32]
  Z6   store, shape [32, 0]
  Z7   atomic_add (a TMA reduction), shape [0, 32]
  Z8   gather of 8 rows, shape [0, 32]
  Z9   scatter of 8 rows, shape [0, 32]
  Z10  load, rank 1, shape [0]
  Z11  load, shape [0, 32], at offsets [32, 0] (a block that lies outside even a positive shape)
  Z12  built with shape [0, 32] but never accessed (the access is in a runtime branch no program
       takes); the kernel stores a marker through a pointer
  Z13  a host-built descriptor (TensorDescriptor) of shape [0, 32]
  Z14  gather of 8 rows, in-kernel descriptor of shape [32, 32] (control for Z8)
  Z15  scatter of 8 rows, in-kernel descriptor of shape [32, 32] (control for Z9)
  Z16  gather of 8 rows through a host-built descriptor of shape [32, 32]

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/desc-zero-shape/run.py  (from TVA's root)
"""
import json, os, subprocess, sys

import torch
import triton
import triton.language as tl

BM = BN = 32


@triton.jit
def k_load(x_ptr, y_ptr, M, N, OM, ON, BM: tl.constexpr, BN: tl.constexpr):
    xd = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[32, 1], block_shape=[BM, BN])
    v = xd.load([OM, ON])
    r = tl.arange(0, BM)
    c = tl.arange(0, BN)
    tl.store(y_ptr + r[:, None] * BN + c[None, :], v)


@triton.jit
def k_load1(x_ptr, y_ptr, N, BN: tl.constexpr):
    xd = tl.make_tensor_descriptor(x_ptr, shape=[N], strides=[1], block_shape=[BN])
    tl.store(y_ptr + tl.arange(0, BN), xd.load([0]))


@triton.jit
def k_store(y_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    yd = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[32, 1], block_shape=[BM, BN])
    yd.store([0, 0], tl.full([BM, BN], 7.0, tl.float32))


@triton.jit
def k_atomic(y_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    yd = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[32, 1], block_shape=[BM, BN])
    yd.atomic_add([0, 0], tl.full([BM, BN], 7.0, tl.float32))


@triton.jit
def k_gather(x_ptr, y_ptr, M, N, BN: tl.constexpr):
    xd = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[32, 1], block_shape=[1, BN])
    rows = tl.arange(0, 8)
    v = xd.gather(rows, 0)
    tl.store(y_ptr + rows[:, None] * BN + tl.arange(0, BN)[None, :], v)


@triton.jit
def k_scatter(y_ptr, M, N, BN: tl.constexpr):
    yd = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[32, 1], block_shape=[1, BN])
    rows = tl.arange(0, 8)
    yd.scatter(tl.full([8, BN], 7.0, tl.float32), rows, 0)


@triton.jit
def k_build_only(x_ptr, y_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    xd = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[32, 1], block_shape=[BM, BN])
    if M > 0:                          # no program takes this branch when M = 0
        v = xd.load([0, 0])
        tl.store(y_ptr + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :], v)
    tl.store(y_ptr, 5.0)


@triton.jit
def k_host_gather(xd, y_ptr, BN: tl.constexpr):
    rows = tl.arange(0, 8)
    v = xd.gather(rows, 0)
    tl.store(y_ptr + rows[:, None] * BN + tl.arange(0, BN)[None, :], v)


@triton.jit
def k_host(xd, y_ptr, BM: tl.constexpr, BN: tl.constexpr):
    v = xd.load([0, 0])
    tl.store(y_ptr + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :], v)


CASES = {
    "Z1": "load, shape [32, 32] (control); model: the block",
    "Z2": "load, shape [0, 32]; model before round 11: every lane the padding 0",
    "Z3": "load, shape [32, 0]; model before round 11: every lane the padding 0",
    "Z4": "load, shape [1, 32]; model: row 0, then the padding 0",
    "Z5": "store, shape [0, 32]; model before round 11: nothing written",
    "Z6": "store, shape [32, 0]; model before round 11: nothing written",
    "Z7": "atomic_add, shape [0, 32]; model before round 11: nothing written",
    "Z8": "gather of 8 rows, shape [0, 32]; model before round 11: unspecified values",
    "Z9": "scatter of 8 rows, shape [0, 32]; model: undefined (every lane out of bounds)",
    "Z10": "load, rank 1, shape [0]; model before round 11: every lane the padding 0",
    "Z11": "load, shape [0, 32], offsets [32, 0]; model before round 11: every lane the padding 0",
    "Z12": "shape [0, 32], built but never accessed; model: y[0] = 5, the rest unchanged",
    "Z13": "host-built TensorDescriptor, shape [0, 32]",
    "Z14": "gather of 8 rows, shape [32, 32] (control for Z8); model: rows 0-7",
    "Z15": "scatter of 8 rows, shape [32, 32] (control for Z9); model: rows 0-7 written with 7",
    "Z16": "gather of 8 rows, host-built descriptor of shape [32, 32]; model: rows 0-7",
}


def run_case(name):
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
    x = torch.arange(32 * 32, dtype=torch.float32, device="cuda") + 1
    y = torch.full((BM * BN,), -1.0, device="cuda")
    xh = x.cpu().view(32, 32)

    def compare(ref):
        torch.cuda.synchronize()
        got = y.cpu()
        if torch.equal(got, ref):
            return "as the model"
        return "differs from the model: first values " + str(got[:4].tolist())

    if name in ("Z1", "Z2", "Z3", "Z4", "Z11"):
        M, N, OM, ON = {"Z1": (32, 32, 0, 0), "Z2": (0, 32, 0, 0), "Z3": (32, 0, 0, 0),
                        "Z4": (1, 32, 0, 0), "Z11": (0, 32, 32, 0)}[name]
        k_load[(1,)](x, y, M, N, OM, ON, BM=BM, BN=BN)
        if name == "Z1":
            ref = xh.flatten()
        elif name == "Z4":
            ref = torch.cat([xh[0], torch.zeros(31 * 32)])
        else:
            ref = torch.zeros(BM * BN)
        return compare(ref)
    if name in ("Z5", "Z6"):
        M, N = (0, 32) if name == "Z5" else (32, 0)
        k_store[(1,)](y, M, N, BM=BM, BN=BN)
        return compare(torch.full((BM * BN,), -1.0))
    if name == "Z7":
        k_atomic[(1,)](y, 0, 32, BM=BM, BN=BN)
        return compare(torch.full((BM * BN,), -1.0))
    if name == "Z8":
        k_gather[(1,)](x, y, 0, 32, BN=BN)
        torch.cuda.synchronize()
        return "ran without an error (values: " + str(y[:4].tolist()) + ")"
    if name == "Z9":
        k_scatter[(1,)](y, 0, 32, BN=BN)
        torch.cuda.synchronize()
        return "ran without an error"
    if name == "Z14":
        k_gather[(1,)](x, y, 32, 32, BN=BN)
        return compare(torch.cat([xh[:8].flatten(), torch.full((24 * 32,), -1.0)]))
    if name == "Z15":
        k_scatter[(1,)](y, 32, 32, BN=BN)
        return compare(torch.cat([torch.full((8 * 32,), 7.0), torch.full((24 * 32,), -1.0)]))
    if name == "Z16":
        from triton.tools.tensor_descriptor import TensorDescriptor
        xd = TensorDescriptor(x, shape=[32, 32], strides=[32, 1], block_shape=[1, BN])
        k_host_gather[(1,)](xd, y, BN=BN)
        return compare(torch.cat([xh[:8].flatten(), torch.full((24 * 32,), -1.0)]))
    if name == "Z10":
        k_load1[(1,)](x, y, 0, BN=BN)
        ref = torch.cat([torch.zeros(BN), torch.full((BM * BN - BN,), -1.0)])
        return compare(ref)
    if name == "Z12":
        k_build_only[(1,)](x, y, 0, 32, BM=BM, BN=BN)
        ref = torch.full((BM * BN,), -1.0)
        ref[0] = 5.0
        return compare(ref)
    if name == "Z13":
        from triton.tools.tensor_descriptor import TensorDescriptor
        xd = TensorDescriptor(x, shape=[0, 32], strides=[32, 1], block_shape=[BM, BN])
        k_host[(1,)](xd, y, BM=BM, BN=BN)
        return compare(torch.zeros(BM * BN))
    raise SystemExit("unknown case " + name)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        try:
            res = run_case(sys.argv[2])
        except Exception as e:
            lines = str(e).strip().splitlines() or [""]
            res = "error: " + type(e).__name__ + ": " + lines[0][:150]
            why = [l.split("error   :", 1)[1].strip() for l in lines if "error   :" in l]
            if why:  # ptxas's own reason
                res += " (ptxas: " + why[0][:150] + ")"
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
            res = "process failed: " + (err[-1][:150] if err else "(no output)")
        print(f"{name}: {desc}\n  {res}")


if __name__ == "__main__":
    main()

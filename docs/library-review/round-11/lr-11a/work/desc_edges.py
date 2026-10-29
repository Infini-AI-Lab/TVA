"""lr-11a: in-kernel tensor descriptors whose values TensorDesc.Valid admits (after L5), at edges
the round-11 evidence does not try. Each case in its own process (a fault ends the CUDA context).

  D1  shape [32, 32], strides [0, 1]          leading stride 0 (Valid: 0 * 4 % 16 = 0); model: every row is row 0
  D2  shape [1, 32],  strides [2^38, 1]       leading stride 2^40 bytes; only row 0 in bounds; model: row 0, then padding
  D3  shape [1, 32],  strides [2^37, 1]       leading stride 2^39 bytes (control for D2)
  D4  shape [32, 32], strides [4, 1]          rows overlap (16-byte stride); model: row r starts at element 4r
  D5  shape [32, 3],  strides [32, 1]         innermost extent 12 bytes; model: columns 0-2, then padding
  D6  block [512, 16], shape [512, 16]        a block dimension above 256 (TMA's box limit)
  D7  shape [2^31 - 1, 32], strides [32, 1]   a huge extent, block at [0, 0]; model: rows 0-31
  D8  shape [32, 32], strides [2^28, 1], offsets [0,0]: rows far apart (2^30 bytes), all rows inside: model reads 32 rows 1 GiB apart (needs memory) -- skipped
  D9  rank 3, shape [1, 32, 32], strides [0, 32, 1]  leading stride 0 on a size-1 dimension
Run: CUDA_VISIBLE_DEVICES=0 <venv>/bin/python -I desc_edges.py
"""
import json, os, subprocess, sys

import torch
import triton
import triton.language as tl


@triton.jit
def k2(x_ptr, y_ptr, M, N, S0, BM: tl.constexpr, BN: tl.constexpr):
    xd = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[S0, 1], block_shape=[BM, BN])
    v = xd.load([0, 0])
    tl.store(y_ptr + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :], v)


@triton.jit
def k3(x_ptr, y_ptr, S0, BN: tl.constexpr):
    xd = tl.make_tensor_descriptor(x_ptr, shape=[1, 32, 32], strides=[S0, 32, 1], block_shape=[1, 32, BN])
    v = xd.load([0, 0, 0])
    tl.store(y_ptr + tl.arange(0, 32)[:, None] * BN + tl.arange(0, BN)[None, :], v.reshape(32, BN))


def run_case(name):
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
    x = torch.arange(512 * 32, dtype=torch.float32, device="cuda") + 1
    xh = x.cpu()
    if name == "D9":
        y = torch.full((32 * 32,), -1.0, device="cuda")
        k3[(1,)](x, y, 0, BN=32)
        torch.cuda.synchronize()
        ref = xh[:1024]
        return "as the model" if torch.equal(y.cpu(), ref) else "differs: " + str(y.cpu()[:4].tolist())
    BM, BN = (512, 16) if name == "D6" else (32, 32)
    y = torch.full((BM * BN,), -1.0, device="cuda")
    M, N, S0 = {"D1": (32, 32, 0), "D2": (1, 32, 2 ** 38), "D3": (1, 32, 2 ** 37), "D4": (32, 32, 4),
                "D5": (32, 3, 32), "D6": (512, 16, 16), "D7": (2 ** 31 - 1, 32, 32)}[name]
    k2[(1,)](x, y, M, N, S0, BM=BM, BN=BN)
    torch.cuda.synchronize()
    got = y.cpu().view(BM, BN)
    ref = torch.zeros(BM, BN)
    for r in range(BM):
        for c in range(BN):
            if r < M and c < N:
                ref[r, c] = xh[r * S0 + c]
    if torch.equal(got, ref):
        return "as the model"
    bad = (got != ref).nonzero()
    return f"differs from the model in {bad.shape[0]} of {BM*BN} lanes; first {bad[0].tolist()}: got {got[tuple(bad[0])].item()} want {ref[tuple(bad[0])].item()}"


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        try:
            res = run_case(sys.argv[2])
        except Exception as e:
            lines = str(e).strip().splitlines() or [""]
            res = "error: " + type(e).__name__ + ": " + lines[0][:200]
        print("RESULT " + json.dumps(res))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name in ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D9"]:
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                           capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        res = json.loads(line[7:]) if line else "process failed: " + (p.stderr.strip().splitlines() or ["?"])[-1][:200]
        print(f"{name}: {res}")


if __name__ == "__main__":
    main()

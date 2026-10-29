"""lr-11b, round 11: descriptors built in the kernel with strides `TensorDesc.Valid` allows (library
issue L5: an access through a valid descriptor must not fault, `Compiler.Faithful`'s "never faults").

`TensorDesc.Valid` (lean/TVA/Lang/Desc.lean) asks of the strides only that the last is 1 and that
every other one is a multiple of 16 bytes (`strides : List Nat`, so 0 is allowed and there is no
upper bound). TMA hardware documents global strides as multiples of 16 bytes below 2^40. Each case
builds a float32 descriptor with `tl.make_tensor_descriptor` and loads one block at [0, 0] into a
plain store; the model (`tl.desc_load`) gives element (r, c) = x[r * stride + c] when (r, c) is in the
shape and the padding 0 otherwise.

  S1  shape [4, 32], strides [0, 1]: every row is x[0:32]
  S2  shape [4, 32], strides [4, 1] (16 bytes): rows overlap, row r is x[4r : 4r+32]
  S3  shape [1, 32], strides [2^38, 1] (2^40 bytes), block [2, 32]: row 0 is x[0:32], row 1 padding
  S4  shape [1, 32], strides [2^36, 1] (2^38 bytes), block [2, 32]: as S3 (control below 2^40)
  S5  shape [4, 32], strides [32, 1] (control)
  S6  store through shape [4, 32], strides [0, 1] would write one row four times: a conflict in the
      model (not run)

Run (from TVA's root): CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I <this file>
"""
import json, os, subprocess, sys

import torch
import triton
import triton.language as tl


@triton.jit
def k(x_ptr, y_ptr, S0, R: tl.constexpr, BR: tl.constexpr):
    d = tl.make_tensor_descriptor(x_ptr, shape=[R, 32], strides=[S0, 1], block_shape=[BR, 32])
    v = d.load([0, 0])
    r = tl.arange(0, BR)
    c = tl.arange(0, 32)
    tl.store(y_ptr + r[:, None] * 32 + c[None, :], v)


CASES = {"S1": (4, 0, 4), "S2": (4, 4, 4), "S3": (1, 2 ** 38, 2), "S4": (1, 2 ** 36, 2), "S5": (4, 32, 4)}


def run(name):
    R, S0, BR = CASES[name]
    triton.set_allocator(lambda size, align, stream: torch.empty(size, device="cuda", dtype=torch.int8))
    x = torch.arange(4 * 32 + 64, device="cuda", dtype=torch.float32)
    y = torch.full((BR * 32,), -1.0, device="cuda")
    k[(1,)](x, y, S0, R=R, BR=BR)
    torch.cuda.synchronize()
    xh, yh = x.cpu(), y.view(BR, 32).cpu()
    model = torch.zeros(BR, 32)
    for r in range(BR):
        for c in range(32):
            if r < R:
                model[r, c] = xh[r * S0 + c] if r * S0 + c < xh.numel() else float("nan")
    return "as the model" if torch.equal(yh, model) else "differs: rows " + str(yh[:, :4].tolist())


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        try:
            res = run(sys.argv[2])
        except Exception as e:
            msg = [l for l in str(e).strip().splitlines() if l.strip()]
            res = "error: " + type(e).__name__ + ": " + (msg[0] if msg else "")[:160]
        print("RESULT " + json.dumps(res))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name, (R, S0, BR) in CASES.items():
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                           capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        res = json.loads(line[len("RESULT "):]) if line else "process failed: " + p.stderr.strip()[-160:]
        print(f"{name}: shape [{R}, 32], strides [{S0}, 1], block [{BR}, 32]: {res}")


if __name__ == "__main__":
    main()

"""A host `TensorDescriptor` with `round_f32_to_tf32=True`: what `desc.load` returns.

Library evidence for library issue L76 (round 15). Triton 3.8.0's host descriptor has a field
`round_f32_to_tf32` (default `False`, `float32` tensors only); the NVIDIA launcher then builds the TMA
descriptor with the element type `TMA_TF32` (`primitives/triton-3.8.0-nvidia-driver.py:231-232`), and
`tl.dot`'s documentation advises it to round `float32` inputs to tf32
(`primitives/triton-3.8.0-core.py:2364-2367`). The library's `tl.desc_load` returns the stored values.

  T1  desc.load of a float32 block through a host descriptor with round_f32_to_tf32=False, stored back
      through a pointer
  T2  the same with round_f32_to_tf32=True

The buffer holds values with low mantissa bits set (1 + 2^-23, 1 + 2^-13 + 2^-14, 1 + 2^-11, ...).
Each case runs in its own process with its own Triton cache, on GPU 0.
Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/desc-tf32/run.py (from TVA's root).
"""
import json, os, struct, subprocess, sys, tempfile

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def k(desc, out_ptr, BM: tl.constexpr, BN: tl.constexpr):
    v = desc.load([0, 0])
    offs = tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
    tl.store(out_ptr + offs, v)


VALUES = [1 + 2**-23, 1 + 2**-13 + 2**-14, 1 + 2**-11, 1 + 2**-10 + 2**-11, -3.0000002, 0.1, 1.0, 2**-130]


def bits(f):
    return f"0x{struct.unpack('<I', struct.pack('<f', f))[0]:08x}"


def run_case(name):
    BM, BN = 16, 16
    src = torch.ones(BM, BN, device="cuda", dtype=torch.float32)
    src.view(-1)[:len(VALUES)] = torch.tensor(VALUES, dtype=torch.float32)
    out = torch.zeros(BM, BN, device="cuda", dtype=torch.float32)
    desc = TensorDescriptor.from_tensor(src, [BM, BN], round_f32_to_tf32=(name == "T2"))
    k[(1,)](desc, out, BM, BN)
    torch.cuda.synchronize()
    got = out.view(-1)[:len(VALUES)].tolist()
    stored = src.view(-1)[:len(VALUES)].tolist()
    return {"stored": [bits(v) for v in stored], "loaded": [bits(v) for v in got],
            "equal": [a == b for a, b in zip(stored, got)]}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        try:
            res = run_case(sys.argv[2])
        except Exception as e:
            msg = [l for l in str(e).strip().splitlines() if l.strip()]
            res = {"error": type(e).__name__ + ": " + (msg[-1] if msg else "")[:150]}
        print("RESULT " + json.dumps(res))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name, desc in (("T1", "round_f32_to_tf32=False"), ("T2", "round_f32_to_tf32=True")):
        with tempfile.TemporaryDirectory() as cache:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", TRITON_CACHE_DIR=cache)
            p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                               capture_output=True, text=True, env=env)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        res = json.loads(line[len("RESULT "):]) if line else {"error": "process failed"}
        print(f"{name}: {desc}")
        if "error" in res:
            print("  " + res["error"])
            continue
        for s, l, e in zip(res["stored"], res["loaded"], res["equal"]):
            print(f"  stored {s}  loaded {l}  {'same' if e else 'DIFFERENT'}")


if __name__ == "__main__":
    main()

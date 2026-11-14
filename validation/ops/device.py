#!/usr/bin/env python3
"""Run the single-operation cases on the GPU: one Triton kernel per operation.

  device.py CASES.json DEVICE.json [ID_PREFIX ...]

For each case of CASES.json (cases.py), loads its inputs from bit patterns, runs a kernel whose
only operation is the case's (`z = x + y`, `x // y`, `x.to(D)`, `tl.sum(x)`, `tl.dot(a, b) + c`,
…), and writes the output bit patterns to DEVICE.json under the case's id, with the stack
(`_meta`: GPU, driver, torch, triton). Ops.lean then compares them with the model.

Needs torch and triton on a CUDA device (the evidence's stack: H200, Triton 3.8.0). Not run in
the repository's CI. Each kernel is written to a module file, so that triton.jit can read its
source."""
import importlib.util, json, os, subprocess, sys, tempfile
import numpy as np
import torch, triton
import triton.language as tl

TORCH = {"i1": torch.bool, "i8": torch.int8, "i16": torch.int16, "i32": torch.int32, "i64": torch.int64,
         "u8": torch.uint8, "u32": torch.uint32, "f16": torch.float16, "bf16": torch.bfloat16,
         "f32": torch.float32, "f64": torch.float64, "f8e5": torch.float8_e5m2,
         "f8e4nv": torch.float8_e4m3fn}
TL = {"i1": "tl.int1", "i8": "tl.int8", "i16": "tl.int16", "i32": "tl.int32", "i64": "tl.int64",
      "u8": "tl.uint8", "u32": "tl.uint32", "f16": "tl.float16", "bf16": "tl.bfloat16",
      "f32": "tl.float32", "f64": "tl.float64", "f8e5": "tl.float8e5", "f8e4nv": "tl.float8e4nv"}
STORE = {1: (np.uint8, torch.uint8), 8: (np.uint8, torch.uint8), 16: (np.uint16, torch.int16),
         32: (np.uint32, torch.int32), 64: (np.uint64, torch.int64)}

def width(d): return {"i1": 1, "f8e5": 8, "f8e4nv": 8, "f16": 16, "bf16": 16, "f32": 32, "f64": 64}.get(d) or int(d[1:])

def to_tensor(bits, d):
    npu, tint = STORE[width(d)]
    raw = np.array(bits, dtype=np.uint64).astype(npu)
    t = torch.from_numpy(raw.view(np.dtype(str(tint).replace("torch.", "")))).cuda()
    return t.bool() if d == "i1" else t.view(TORCH[d])

def to_bits(t, d):
    if d == "i1": return [int(v) for v in t.to(torch.uint8).cpu().tolist()]
    npu, tint = STORE[width(d)]
    return [int(v) for v in t.view(tint).cpu().numpy().view(npu).tolist()]

ELEM = {"add": "x + y", "sub": "x - y", "mul": "x * y", "div": "x / y", "div_rn": "tl.div_rn(x, y)",
        "fma": "tl.fma(x, y, w)", "maximum": "tl.maximum(x, y)", "minimum": "tl.minimum(x, y)",
        "neg": "-x", "abs": "tl.abs(x)", "sqrt_rn": "tl.sqrt_rn(x)", "exp": "tl.exp(x)",
        "exp2": "tl.exp2(x)", "log": "tl.log(x)", "log2": "tl.log2(x)", "sqrt": "tl.sqrt(x)",
        "rsqrt": "tl.rsqrt(x)", "idiv": "x // y", "irem": "x % y", "shl": "x << y",
        "ashr": "x >> y", "lshr": "x >> y"}
ARITY = {"fma": 3, "neg": 1, "abs": 1, "sqrt_rn": 1, "exp": 1, "exp2": 1, "log": 1, "log2": 1,
         "sqrt": 1, "rsqrt": 1, "convert": 1}

def source(case):
    op, dts = case["op"], case["dtypes"]
    if op in ELEM or op == "convert":
        k = ARITY.get(op, 2)
        expr = f"x.to({TL[dts[1]]})" if op == "convert" else ELEM[op]
        names = ["x", "y", "w"][:k]
        loads = "\n".join(f"    {v} = tl.load({v}_ptr + o, mask=m)" for v in names)
        return f"""
@triton.jit
def kernel({", ".join(v + "_ptr" for v in names)}, z_ptr, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
{loads}
    tl.store(z_ptr + o, {expr}, mask=m)
"""
    if op == "sum":
        return """
@triton.jit
def kernel(x_ptr, z_ptr, N: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, N))
    tl.store(z_ptr, tl.sum(x, axis=0))
"""
    if op.startswith("dot"):
        prec = op.rsplit("_", 1)[1]
        call = (f"tl.dot(a, b, c, input_precision='{prec}', out_dtype={TL[dts[2]]})" if op.startswith("dot_ieee") or op.startswith("dot_tf32")
                else f"tl.dot(a, b, input_precision='{prec}', out_dtype={TL[dts[2]]}) + c")
        return f"""
@triton.jit
def kernel(a_ptr, b_ptr, c_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm = tl.arange(0, M); rn = tl.arange(0, N); rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    c = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    tl.store(z_ptr + rm[:, None] * N + rn[None, :], {call})
"""
    raise ValueError(op)

def load_kernel(src, tmp, name):
    path = os.path.join(tmp, name + ".py")
    with open(path, "w") as f:
        f.write("import triton\nimport triton.language as tl\n" + src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.kernel

def run(case, kernel):
    op, dts, dims, ins = case["op"], case["dtypes"], case["dims"], case["inputs"]
    nw = case.get("num_warps", 4)
    if op.startswith("dot"):
        M, N, K = dims
        a, b, c = (to_tensor(x, d) for x, d in zip(ins, dts))
        z = torch.empty(M * N, dtype=TORCH[dts[2]], device="cuda")
        kernel[(1,)](a, b, c, z, M, N, K, num_warps=nw)
        return to_bits(z, dts[2])
    if op == "sum":
        x = to_tensor(ins[0], dts[0]); z = torch.empty(1, dtype=TORCH[dts[0]], device="cuda")
        kernel[(1,)](x, z, dims[0], num_warps=nw)
        return to_bits(z, dts[0])
    n = dims[0]; out_d = dts[-1]
    xs = [to_tensor(x, dts[0]) for x in ins]
    z = torch.empty(n, dtype=TORCH[out_d], device="cuda")
    kernel[(triton.cdiv(n, 1024),)](*xs, z, n, BLOCK=1024, num_warps=nw)
    return to_bits(z, out_d)

def meta():
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return {"gpu": smi, "torch": torch.__version__, "triton": triton.__version__,
            "cuda": torch.version.cuda}

def main():
    cases = json.load(open(sys.argv[1])); out = sys.argv[2]; prefixes = sys.argv[3:]
    res = {"_meta": meta()}
    with tempfile.TemporaryDirectory() as tmp:
        for i, c in enumerate(cases):
            if prefixes and not any(c["id"].startswith(p) for p in prefixes): continue
            try:
                res[c["id"]] = run(c, load_kernel(source(c), tmp, f"k{i}"))
                print(f"{c['id']}: ok")
            except Exception as e:  # a kernel Triton rejects is reported, not fatal
                res[c["id"] + ".error"] = repr(e)
                print(f"{c['id']}: {e!r}")
    json.dump(res, open(out, "w"))

if __name__ == "__main__":
    main()

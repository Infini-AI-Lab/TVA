"""TVA library evidence (library issue L18): two of the environment variables Triton 3.8.0 reads,
`TRITON_F32_DEFAULT` and `TRITON_DEFAULT_FP_FUSION`, and what Triton does with them unset.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/env-knobs/run.py
Output: evidence/env-knobs/out.txt (README.md says what it shows).

Each case runs in a fresh subprocess whose environment has every `TRITON_*` variable removed, a
new empty `TRITON_CACHE_DIR`, and then only the variable under test set (or none). The child
prints what it compiled and computed.

Part F32: `tl.dot(a, b)` of float32 16x16 blocks, without `input_precision` (and, as controls,
with `allow_tf32=False` and with an explicit `input_precision="ieee"`). Printed: the
`inputPrecision` attribute of `tt.dot` in the TTIR (the printer omits it at its default value,
`IEEE`: `DefaultValuedAttr<..., InputPrecision::IEEE>`, `primitives/triton-3.8.0-TritonOps.td:654`,
`:663`), and the result for operands chosen so that the three precisions differ: every element of
`a` is `1 + 2^-12` (not a tf32 value), `b` is all ones, so the exact result is `16 + 2^-8` in every
element; tf32 inputs give `16`.

Part FUSION: `y = a * b + c` in float32 with the default `enable_fp_fusion` (and, as a control, an
explicit `enable_fp_fusion=True` at the launch). Printed: the number of `fma.rn.f32`, `mul.rn.f32`
and `add.rn.f32` in the PTX, and `y` for `a = b = 1 + 2^-13`, `c = -(1 + 2^-12)`: fused, `2^-26`;
unfused (the product rounded first), `0`.
"""
import os
import subprocess
import sys
import tempfile
import time

CASES = [
    ("f32", None), ("f32", "ieee"), ("f32", "tf32"), ("f32", "tf32x3"), ("f32", "bogus"),
    ("fusion", None), ("fusion", "0"), ("fusion", "1"),
]
VAR = {"f32": "TRITON_F32_DEFAULT", "fusion": "TRITON_DEFAULT_FP_FUSION"}


def child(part):
    import re

    import torch
    import triton
    import triton.language as tl
    from triton import knobs

    print(f"  knobs.language.fp32_default={knobs.language.fp32_default!r}, "
          f"knobs.language.default_fp_fusion={knobs.language.default_fp_fusion!r}")

    @triton.jit
    def dot(a_ptr, b_ptr, c_ptr, B: tl.constexpr, MODE: tl.constexpr):
        r = tl.arange(0, B)
        t = r[:, None] * B + r[None, :]
        a = tl.load(a_ptr + t)
        b = tl.load(b_ptr + t)
        if MODE == 0:
            c = tl.dot(a, b)
        elif MODE == 1:
            c = tl.dot(a, b, allow_tf32=False)
        else:
            c = tl.dot(a, b, input_precision="ieee")
        tl.store(c_ptr + t, c)

    @triton.jit
    def fma(a_ptr, b_ptr, c_ptr, y_ptr, B: tl.constexpr):
        o = tl.arange(0, B)
        tl.store(y_ptr + o, tl.load(a_ptr + o) * tl.load(b_ptr + o) + tl.load(c_ptr + o))

    if part == "f32":
        B = 16
        a = torch.full((B, B), 1 + 2 ** -12, device="cuda", dtype=torch.float32)
        b = torch.ones((B, B), device="cuda", dtype=torch.float32)
        for mode, label in ((0, "tl.dot(a, b)"), (1, "tl.dot(a, b, allow_tf32=False)"),
                            (2, 'tl.dot(a, b, input_precision="ieee")')):
            c = torch.zeros((B, B), device="cuda", dtype=torch.float32)
            try:
                k = dot[(1,)](a, b, c, B, mode)
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                msg = str(e).strip().splitlines()
                print(f"  {label:37}: rejected on the host, {type(e).__name__}: {msg[-1] if msg else ''}")
                continue
            attrs = re.findall(r"tt\.dot[^\n]*", k.asm["ttir"])
            prec = re.findall(r"inputPrecision = (\w+)", attrs[0]) if attrs else []
            vals = sorted({float(v) for v in c.flatten().tolist()})
            print(f"  {label:37}: TTIR tt.dot inputPrecision = {prec[0] if prec else '(omitted, so ieee)'}; "
                  f"result values {vals} (16 + 2^-8 = {16 + 2 ** -8})")
    else:
        B = 128
        a = torch.full((B,), 1 + 2 ** -13, device="cuda", dtype=torch.float32)
        b = a.clone()
        c = torch.full((B,), -(1 + 2 ** -12), device="cuda", dtype=torch.float32)
        for label, kw in (("default enable_fp_fusion", {}), ("enable_fp_fusion=True", {"enable_fp_fusion": True})):
            y = torch.zeros(B, device="cuda", dtype=torch.float32)
            k = fma[(1,)](a, b, c, y, B, **kw)
            torch.cuda.synchronize()
            ptx = k.asm["ptx"]
            vals = sorted({float(v) for v in y.tolist()})
            print(f"  {label:25}: compiled enable_fp_fusion={k.metadata.enable_fp_fusion}; PTX fma.rn.f32 "
                  f"{ptx.count('fma.rn.f32')}, mul.rn.f32 {ptx.count('mul.rn.f32')}, add.rn.f32 "
                  f"{ptx.count('add.rn.f32')}; y values {vals} (fused 2^-26 = {2 ** -26}, unfused 0.0)")


def main():
    import torch
    import triton
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}, {time.strftime('%Y-%m-%d')}", flush=True)
    for part, val in CASES:
        env = {k: v for k, v in os.environ.items() if not k.startswith("TRITON_")}
        with tempfile.TemporaryDirectory() as cache:
            env["TRITON_CACHE_DIR"] = cache
            if val is not None:
                env[VAR[part]] = val
            print(f"\n== {VAR[part]} {'unset' if val is None else '= ' + repr(val)} ==", flush=True)
            r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--child", part], env=env,
                               capture_output=True, text=True)
            print(r.stdout, end="", flush=True)
            if r.returncode != 0:
                tail = [ln for ln in r.stderr.strip().splitlines() if ln.strip()]
                print(f"  child failed (exit {r.returncode}): {tail[-1] if tail else ''}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
    else:
        main()

"""What type and value does a kernel receive for each kind of Python scalar argument?

Library evidence for library issue L29 (round 8). Each case compiles and launches small kernels
with Triton 3.8.0 on GPU 0, in its own process, and reports:
  - the parameter's type and attributes in the front end's output (`asm['source']`, the
    `tt.func` signature), or `absent` when the argument became a compile-time constant;
  - the value the kernel received, read back exactly (integers stored into an `int64` buffer,
    floats bitcast to an integer of their width), against a host reference;
  - for a rejected argument, the exception the launch raised on the host.

Run: .venv/bin/python -I evidence/scalar-args/run.py  (from TVA's root; CUDA_VISIBLE_DEVICES=0)
"""
import json, math, os, re, struct, subprocess, sys

import numpy as np
import torch
import triton
import triton.language as tl

DEV = "cuda"


# ---------------------------------------------------------------- kernels

@triton.jit
def k_int(out, n):
    tl.store(out + 0, n)                    # the value, widened to int64 by the store
    tl.store(out + 1, n + n)                # arithmetic in the parameter's own type
    tl.store(out + 2, (n - 2) // 2)         # Python floors; Triton's integer // truncates
    tl.store(out + 3, n * 65536 * 65536)    # exact in Python; wraps in int32


@triton.jit(do_not_specialize=["n"])
def k_int_dns(out, n):
    tl.store(out + 0, n)
    tl.store(out + 1, n + n)
    tl.store(out + 2, (n - 2) // 2)
    tl.store(out + 3, n * 65536 * 65536)


@triton.jit(do_not_specialize_on_alignment=["n"])
def k_int_dnsa(out, n):
    tl.store(out + 0, n)


@triton.jit
def k_int_ann(out, n: tl.int32):
    tl.store(out + 0, n)
    tl.store(out + 1, n + n)
    tl.store(out + 2, (n - 2) // 2)
    tl.store(out + 3, n * 65536 * 65536)


@triton.jit
def k_int_pyann(out, n: int):
    tl.store(out + 0, n)


@triton.jit
def k_bool(out, b):
    tl.store(out + 0, b.to(tl.int32))


@triton.jit
def k_bool_ann(out, b: tl.int1):
    tl.store(out + 0, b.to(tl.int32))


@triton.jit(do_not_specialize=["i"])
def k_f32(out, i, x):
    tl.store(out + i, x.to(tl.int32, bitcast=True))


@triton.jit(do_not_specialize=["i"])
def k_f16(out, i, x: tl.float16):
    tl.store(out + i, x.to(tl.int16, bitcast=True))


@triton.jit(do_not_specialize=["i"])
def k_bf16(out, i, x: tl.bfloat16):
    tl.store(out + i, x.to(tl.int16, bitcast=True))


@triton.jit(do_not_specialize=["i"])
def k_f64(out, i, x: tl.float64):
    tl.store(out + i, x.to(tl.int64, bitcast=True))


@triton.jit(do_not_specialize=["i"])
def k_f32_ann(out, i, x: tl.float32):
    tl.store(out + i, x.to(tl.int32, bitcast=True))


@triton.jit
def k_i64_ann(out, n: tl.int64):
    tl.store(out, n)


@triton.jit
def k_u32_ann(out, n: tl.uint32):
    tl.store(out, n)


@triton.jit
def k_i8_ann(out, n: tl.int8):
    tl.store(out, n)


@triton.jit
def k_any(out, x):
    tl.store(out, x)


@triton.jit
def k_anyf(out, x):
    tl.store(out, x.to(tl.float64))


@triton.jit
def k_none(out, x):
    if x is None:
        tl.store(out, 1)
    else:
        tl.store(out, 2)


# ---------------------------------------------------------------- helpers

def sig(ck, name):
    """The parameter's type in the JIT's signature (`ck.src.signature`: Triton's front-end type,
    with signedness), then its type and attributes in the front end's IR (`tt.func`; MLIR integers
    are signless, so `u32` shows as `i32`), or `absent` when it became a compile-time constant."""
    fe = ck.src.signature.get(name, "?")
    src = ck.asm["source"]
    head = src[src.index("tt.func"):]
    head = head[:head.index("{\n")] if "{\n" in head else head
    m = re.search(r"%" + name + r": ([^\s,)]+)(\s*\{[^}]*\})?", head)
    if not m:
        return f"{fe}, absent from the IR (compile-time constant)"
    return f"{fe} (IR {m.group(1)}{' ' + m.group(2).strip() if m.group(2) else ''})"


def err_text(e):
    lines = [l.strip() for l in str(e).splitlines() if l.strip()]
    return f"{type(e).__name__}: {(lines[0] if lines else '')[:160]}"


def i64buf(n=4, fill=-777):
    return torch.full((n,), fill, device=DEV, dtype=torch.int64)


def as_u64(v):
    return v & (2**64 - 1)


def f32_bits(d):
    return struct.unpack("<I", struct.pack("<f", np.float32(d)))[0] if not math.isnan(d) else \
        int(np.array([d], dtype=np.float64).astype(np.float32).view(np.uint32)[0])


def np_f32_bits(ds):
    with np.errstate(over="ignore", invalid="ignore"):
        return np.array(ds, dtype=np.float64).astype(np.float32).view(np.uint32).astype(np.int64)


def np_f16_bits(ds):
    with np.errstate(over="ignore", invalid="ignore"):
        return np.array(ds, dtype=np.float64).astype(np.float16).view(np.uint16).astype(np.int64)


def edge_doubles_f32():
    F = float(np.finfo(np.float32).max)
    return [
        ("1+2^-24 (tie, to even: 1)", 1 + 2**-24), ("1+3*2^-24 (tie, to even: up)", 1 + 3 * 2**-24),
        ("1+2^-24+2^-50 (above the tie)", 1 + 2**-24 + 2**-50), ("0.1", 0.1), ("1/3", 1 / 3),
        ("FLT_MAX", F), ("FLT_MAX+2^103 (tie above FLT_MAX)", F + 2**103),
        ("FLT_MAX+2^103-2^80 (below the tie)", F + 2**103 - 2**80), ("3.5e38", 3.5e38),
        ("1e300", 1e300), ("-1e300", -1e300), ("2^-126 (min normal)", 2**-126),
        ("2^-149 (min subnormal)", 2**-149), ("2^-150 (tie, to even: 0)", 2**-150),
        ("3*2^-151 (rounds to 2^-149)", 3 * 2**-151), ("1e-46", 1e-46), ("-1e-46", -1e-46),
        ("1.5*2^-140 (subnormal, exact)", 1.5 * 2**-140), ("-0.0", -0.0), ("inf", math.inf),
        ("-inf", -math.inf), ("nan", math.nan), ("-nan", -math.nan),
    ]


def random_doubles(rng, k, lo, hi):
    """k doubles with uniformly random signs, exponents in [lo, hi] and 52 random mantissa bits."""
    sign = rng.integers(0, 2, k)
    exp = rng.integers(lo, hi + 1, k)
    mant = rng.integers(0, 2**52, k, dtype=np.uint64)
    bits = (sign.astype(np.uint64) << np.uint64(63)) | ((exp + 1023).astype(np.uint64) << np.uint64(52)) | mant
    return bits.view(np.float64).tolist()


def launch_floats(kern, ds, bits_dtype):
    out = torch.full((len(ds),), -1, device=DEV, dtype=bits_dtype)
    errs = {}
    ck = None
    for i, d in enumerate(ds):
        try:
            ck = kern[(1,)](out, i, d)
        except Exception as e:
            errs[i] = err_text(e)
    torch.cuda.synchronize()
    return out.cpu().numpy().astype(np.int64), errs, ck


def fmt_bits(b, w):
    return f"0x{int(b) & ((1 << w) - 1):0{w // 4}x}"


# ---------------------------------------------------------------- cases

def case_S1():
    lines = []
    for label, kern, n in (("unannotated, n=1", k_int, 1), ("unannotated, n=3", k_int, 3),
                           ("annotated n: tl.int32, n=1", k_int_ann, 1),
                           ("do_not_specialize=['n'], n=1", k_int_dns, 1)):
        out = i64buf()
        ck = kern[(1,)](out, n)
        torch.cuda.synchronize()
        v = out.tolist()
        lines.append(f"{label}: parameter {sig(ck, 'n')}; n={v[0]}, n+n={v[1]}, (n-2)//2={v[2]}, n*65536*65536={v[3]}")
    return lines


def case_S2():
    lines = []
    vals = [0, 2, 16, 24, 2**31 - 1, 2**31, -2**31, -2**31 - 1, 2**30, 2**63 - 1, 2**63, 2**64 - 1,
            -2**63, 2**64, -2**63 - 1, 2**70]
    for n in vals:
        out = i64buf()
        try:
            ck = k_int[(1,)](out, n)
            torch.cuda.synchronize()
        except Exception as e:
            torch.cuda.synchronize()
            lines.append(f"n={n}: rejected on the host: {err_text(e)}; buffer untouched: {out.tolist() == [-777] * 4}")
            continue
        s = sig(ck, "n")
        got = out.tolist()
        u = s.startswith("u64")   # stored bit-for-bit into the int64 buffer; read back as unsigned
        v0 = as_u64(got[0]) if u else got[0]
        lines.append(f"n={n}: parameter {s}; received {v0} (exact: {v0 == n}); n+n = {as_u64(got[1]) if u else got[1]}")
    for label, kern, n in (("do_not_specialize_on_alignment=['n'], n=32", k_int_dnsa, 32),
                           ("do_not_specialize=['n'], n=32", k_int_dns, 32),
                           ("annotated n: tl.int32, n=32", k_int_ann, 32)):
        out = i64buf()
        ck = kern[(1,)](out, n)
        torch.cuda.synchronize()
        lines.append(f"{label}: parameter {sig(ck, 'n')}; received {out[0].item()}")
    return lines


def case_S3():
    lines = []
    for b in (True, False):
        out = i64buf()
        ck = k_bool[(1,)](out, b)
        torch.cuda.synchronize()
        lines.append(f"b={b}: parameter {sig(ck, 'b')}; received {out[0].item()}")
    for b in (True, 1, 2, 0):
        out = i64buf()
        try:
            ck = k_bool_ann[(1,)](out, b)
            torch.cuda.synchronize()
            lines.append(f"annotated b: tl.int1, b={b!r}: parameter {sig(ck, 'b')}; received {out[0].item()}")
        except Exception as e:
            lines.append(f"annotated b: tl.int1, b={b!r}: rejected on the host: {err_text(e)}")
    return lines


def case_S4():
    lines = []
    edges = edge_doubles_f32()
    got, errs, ck = launch_floats(k_f32, [d for _, d in edges], torch.int32)
    ref = np_f32_bits([d for _, d in edges])
    lines.append(f"parameter {sig(ck, 'x')}")
    for (label, d), g, r in zip(edges, got, ref):
        lines.append(f"{label}: received {fmt_bits(g, 32)} = {np.array([g], dtype=np.int64).astype(np.uint32).view(np.float32)[0]!r}; "
                     f"round-to-nearest-even of the double {fmt_bits(r, 32)}; same: {int(g) & 0xffffffff == int(r)}")
    rng = np.random.default_rng(20261028)
    ds = random_doubles(rng, 4000, -160, 135)
    got, errs, _ = launch_floats(k_f32, ds, torch.int32)
    ref = np_f32_bits(ds)
    same = int(((got & 0xffffffff) == ref).sum())
    lines.append(f"random doubles (exponents -160..135, 4000): received == nearest-even float32: {same} of {len(ds)}; host errors: {len(errs)}")
    return lines


def case_S5_f16():
    lines = []
    edges = [("1+2^-11 (tie, to even: 1)", 1 + 2**-11), ("1+3*2^-11 (tie, to even: up)", 1 + 3 * 2**-11),
             ("1+2^-11+2^-40 (above the tie; rounding through float32 first would give 1)", 1 + 2**-11 + 2**-40),
             ("0.1", 0.1), ("65504 (max)", 65504.0), ("65519.99 (below the overflow tie)", 65519.99),
             ("65520 (rounds to inf)", 65520.0), ("70000", 70000.0), ("2^-24 (min subnormal)", 2**-24),
             ("2^-25 (tie, to even: 0)", 2**-25), ("1e-8", 1e-8), ("-0.0", -0.0), ("inf", math.inf),
             ("-inf", -math.inf), ("nan", math.nan), ("int 3", 3)]
    got, errs, ck = launch_floats(k_f16, [d for _, d in edges], torch.int16)
    ref = np_f16_bits([float(d) for _, d in edges])
    lines.append(f"parameter {sig(ck, 'x')}")
    for i, ((label, d), g, r) in enumerate(zip(edges, got, ref)):
        if i in errs:
            lines.append(f"{label}: rejected on the host: {errs[i]}")
        else:
            lines.append(f"{label}: received {fmt_bits(g, 16)}; nearest-even float16 of the double {fmt_bits(r, 16)}; same: {int(g) & 0xffff == int(r)}")
    rng = np.random.default_rng(16)
    ds = random_doubles(rng, 4000, -26, 15)
    got, errs, _ = launch_floats(k_f16, ds, torch.int16)
    ref = np_f16_bits(ds)
    ok = [i for i in range(len(ds)) if i not in errs]
    same = sum(1 for i in ok if (int(got[i]) & 0xffff) == int(ref[i]))
    lines.append(f"random doubles (exponents -26..15, 4000): {len(errs)} rejected on the host (all round to +-inf: "
                 f"{all(abs(np.float16(ds[i])) == np.inf for i in errs)}); of the rest, received == nearest-even float16 of the double: {same} of {len(ok)}")
    return lines


def bf16_trunc_bits(ds):
    return (np_f32_bits(ds) >> 16) & 0xffff


def bf16_rne_bits(ds):
    t = torch.tensor(ds, dtype=torch.float64).to(torch.float32).to(torch.bfloat16)
    return t.view(torch.int16).numpy().astype(np.int64) & 0xffff


def case_S5_bf16():
    lines = []
    F = float(np.finfo(np.float32).max)
    edges = [("1+2^-8 (tie)", 1 + 2**-8), ("1+3*2^-8 (tie)", 1 + 3 * 2**-8), ("1+2^-8+2^-12 (above the tie)", 1 + 2**-8 + 2**-12),
             ("-(1+2^-8+2^-12)", -(1 + 2**-8 + 2**-12)), ("0.1", 0.1), ("FLT_MAX", F), ("3.5e38", 3.5e38),
             ("1e-40 (float32 subnormal)", 1e-40), ("-0.0", -0.0), ("inf", math.inf), ("nan", math.nan)]
    got, errs, ck = launch_floats(k_bf16, [d for _, d in edges], torch.int16)
    tr = bf16_trunc_bits([d for _, d in edges])
    rn = bf16_rne_bits([d for _, d in edges])
    lines.append(f"parameter {sig(ck, 'x')}")
    for (label, d), g, t, r in zip(edges, got, tr, rn):
        rs = "NaN" if math.isnan(d) else fmt_bits(r, 16)
        lines.append(f"{label}: received {fmt_bits(g, 16)}; top 16 bits of the nearest float32 {fmt_bits(t, 16)}; "
                     f"nearest-even bfloat16 {rs}; truncation: {int(g) & 0xffff == int(t)}")
    rng = np.random.default_rng(1616)
    ds = random_doubles(rng, 4000, -130, 127)
    got, errs, _ = launch_floats(k_bf16, ds, torch.int16)
    tr = bf16_trunc_bits(ds)
    rn = bf16_rne_bits(ds)
    g = got & 0xffff
    lines.append(f"random doubles (exponents -130..127, 4000): received == truncation of the nearest float32: {int((g == tr).sum())} of {len(ds)}; "
                 f"== nearest-even bfloat16: {int((g == rn).sum())}; host errors: {len(errs)}")
    return lines


def case_S5_other():
    lines = []
    rng = np.random.default_rng(64)
    ds = random_doubles(rng, 2000, -1022, 1023) + [-0.0, math.inf, -math.inf, math.nan, 5e-324]
    got, errs, ck = launch_floats(k_f64, ds, torch.int64)
    ref = np.array(ds, dtype=np.float64).view(np.int64)
    lines.append(f"x: tl.float64: parameter {sig(ck, 'x')}; received == the double's bits: {int((got == ref).sum())} of {len(ds)}")
    got, errs, ck = launch_floats(k_f32_ann, [3, 2**24 + 1, 0.1], torch.int32)
    lines.append(f"x: tl.float32 given the ints 3 and 2^24+1 and 0.1: parameter {sig(ck, 'x')}; received "
                 f"{[float(np.array([v], dtype=np.int64).astype(np.uint32).view(np.float32)[0]) for v in got]}")
    for label, kern, vals in (("n: tl.int64", k_i64_ann, [5, -1, 2**40, 2**63 - 1, 2**63]),
                              ("n: tl.uint32", k_u32_ann, [5, 2**32 - 1, 2**32, 2**32 + 7, -1, 2**64]),
                              ("n: tl.int8", k_i8_ann, [5, 127, 128, 300, -129, 2**40, 2**63]),
                              ("n: tl.int32", k_int_ann, [2**31, 2**32 + 5, -2**31 - 1]),
                              ("n: int", k_int_pyann, [7, 2**31, 2**40])):
        for v in vals:
            out = i64buf()
            try:
                ck = kern[(1,)](out, v)
                torch.cuda.synchronize()
                lines.append(f"{label}, n={v}: parameter {sig(ck, 'n')}; received {out[0].item()}")
            except Exception as e:
                torch.cuda.synchronize()
                lines.append(f"{label}, n={v}: rejected on the host: {err_text(e)}")
    return lines


def case_S6():
    lines = []
    for label, x in (("Python int 2", 2), ("Python float 2.0", 2.0), ("Python bool True", True),
                     ("numpy.float64(0.1)", np.float64(0.1)), ("numpy.float32(0.1)", np.float32(0.1)),
                     ("numpy.int64(5)", np.int64(5)), ("numpy.int32(5)", np.int32(5)), ("numpy.bool_(True)", np.bool_(True)),
                     ("0-d CUDA tensor torch.tensor(5)", torch.tensor(5, device=DEV)),
                     ("0-d CPU tensor torch.tensor(5)", torch.tensor(5))):
        isf = isinstance(x, (float, np.floating))
        out = torch.full((1,), -777, device=DEV, dtype=torch.float64) if isf else i64buf()
        try:
            ck = (k_anyf if isf else k_any)[(1,)](out, x)
            torch.cuda.synchronize()
            r = out[0].item()
            extra = ""
            if isinstance(x, torch.Tensor):
                extra = f" (== the tensor's address: {r == x.data_ptr()})"
            lines.append(f"{label}: parameter {sig(ck, 'x')}; stored {r}{extra}")
        except Exception as e:
            torch.cuda.synchronize()
            lines.append(f"{label}: rejected on the host: {err_text(e)}")
    for x in (None, 0):
        out = i64buf()
        ck = k_none[(1,)](out, x)
        torch.cuda.synchronize()
        lines.append(f"x={x!r} in `if x is None`: parameter {sig(ck, 'x')}; branch {'then' if out[0].item() == 1 else 'else'}")
    return lines


CASES = {
    "S1": ("the int 1 is a compile-time constant unless annotated or not specialised", case_S1),
    "S2": ("ints: i32, i64, u64 by value; out of range; divisibility by 16", case_S2),
    "S3": ("bool", case_S3),
    "S4": ("Python float -> float32 parameter", case_S4),
    "S5a": ("annotated float16 parameter", case_S5_f16),
    "S5b": ("annotated bfloat16 parameter", case_S5_bf16),
    "S5c": ("annotated float64, float32, int64, uint32, int8, int32 and int parameters", case_S5_other),
    "S6": ("numpy scalars, 0-d tensors, None", case_S6),
}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        try:
            lines = CASES[sys.argv[2]][1]()
        except Exception as e:
            lines = [f"error: {err_text(e)}"]
        print("RESULT " + json.dumps(lines))
        return
    import shutil, tempfile
    print(f"Triton {triton.__version__}, torch {torch.__version__}, numpy {np.__version__}, {torch.cuda.get_device_name(0)}")
    cache = tempfile.mkdtemp()
    for name, (desc, _) in CASES.items():
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                           capture_output=True, text=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES="0", TRITON_CACHE_DIR=cache))
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line is None:
            err = (p.stderr.strip().splitlines() or ["(no output)"])[-1]
            lines = ["process failed: " + err[:200]]
        else:
            lines = json.loads(line[len("RESULT "):])
        print(f"\n{name}: {desc}")
        for l in lines:
            print(f"  {l}")
    shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    main()

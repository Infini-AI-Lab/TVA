"""Front-end forms of round 32: a pointer minus an integer, the predicates of a dtype, which float
conversions take a rounding mode, and a block's `.type` where a dtype goes.

Library evidence for library issues L151, L152, L147 and L148 (round 32). Sources, pinned in
`primitives/`: `triton-3.8.0-semantic.py` 230-276 (`add`, `sub`: `ptr - offset` is
`add(ptr, minus(offset))`), 478-485 (`minus`: `0 - x` in `x`'s type), 811-916 (`cast`), 1205-1243 (`store`
converts the value with `cast`); `triton-3.8.0-core.py` 449-524 (`dtype.is_fp8()` … `is_int()`), 1130-1135
(`tensor.to`).

  S1  `p - i` for a pointer argument `p` and an integer `i` (a runtime `int32` argument, a loaded `int32`,
      `int64`, `uint32` scalar, a Python literal, an `int32` block, a block of pointers minus a block): the
      address it gives, read with `.to(tl.int64)`, against `p.addr - i * size` (signed) and
      `p.addr + ((2^w - i) mod 2^w) * size` (unsigned), on edge values (`i = 0, 1, -3, 7, 2^31 - 1, -2^31`;
      `uint32` `1`, `2^31`); the TTIR of `p - i` (`arith.subi` from `0`, then `tt.addptr`)
  S2  forms Triton should reject: an integer minus a pointer, a pointer minus a pointer, `p - 1.0`
  S3  a load and a store through `p + k - 1` (the vLLM form, `cu_num_draft_tokens_ptr + req_idx - 1`)
  T1  each predicate of a dtype (`is_fp8`, `is_fp8e4nv`, `is_fp8e5`, `is_fp16`, `is_bf16`, `is_fp32`,
      `is_fp64`, `is_floating`, `is_standard_floating`, `is_int`, `is_int_signed`, `is_int_unsigned`,
      `is_bool`, `is_int1`, `is_int8` … `is_uint64`, `is_ptr`) on a loaded block of each of 13 dtypes, as
      a compile-time value: in `tl.static_assert`, in a compile-time `if` and in `a if c else b`
  T2  the vLLM form `x if x.dtype.is_fp8() else x / s` stored to an fp8 pointer and to a bfloat16 one:
      the stored bits, against the saturating rounding to nearest even of the `float32` value
  R1  `fp_downcast_rounding` "rtne" and "rtz" on `bfloat16 -> float16` and `float16 -> bfloat16` (the
      same width): accepted?; on `float32 -> float16` and `float32 -> bfloat16`: accepted (control)
  Y1  a block's `.type` and a scalar's `.type` where a dtype goes: `block.to(y.type)` (a block `y` of
      another shape), `scalar.to(x.type)` (a block `x`), `tl.zeros([16], dtype=x.type)`,
      `tl.zeros([16], dtype=s.type)` (a scalar `s`), `block.to(x.type.element_ty)`: accepted?, and the
      dtype of the result

Each case runs in its own process with its own empty Triton cache. Paths in messages are not printed.
Run from TVA's root: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/front-end-round32/run.py
"""
import importlib.util, os, re, subprocess, sys, tempfile

import torch
import triton
import triton.language as tl


def ttir(ck):
    t = ck.asm["ttir"]
    t = re.sub(r" loc\(#loc\d*\)", "", t)
    t = re.sub(r"loc\([^)]*\)", "", t)
    return "\n".join(l for l in t.splitlines() if not l.startswith("#loc"))


def why(e):
    chain, c = [], e
    while c is not None:
        chain.append(str(c))
        c = c.__cause__ or c.__context__
    lines = [l.strip() for l in "\n".join(chain).splitlines() if l.strip()]
    pick = [l for l in lines if ("Error" in l or "error" in l or "expected" in l or "cannot" in l
                                 or "unsupported" in l or "should" in l) and not l.startswith("at ")]
    msg = (pick or lines or [type(e).__name__])[-1]
    msg = re.sub(r"(/[^\s:'\"]+)+", "<path>", msg)
    return f"{type(e).__name__}: {msg[:150]}"


def load_module(src, name, d):
    path = os.path.join(d, name + ".py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------- S

@triton.jit
def k_s1_arg(p, o, i):
    tl.store(o, (p - i).to(tl.int64))


@triton.jit
def k_s1_load(p, o, ip):
    i = tl.load(ip)
    tl.store(o, (p - i).to(tl.int64))


@triton.jit
def k_s1_lit(p, o):
    tl.store(o, (p - 3).to(tl.int64))
    tl.store(o + 1, (p - 2**40).to(tl.int64))


@triton.jit
def k_s1_block(p, o, ip):
    r = tl.arange(0, 8)
    i = tl.load(ip + r)
    tl.store(o + r, (p - i).to(tl.int64))
    q = p + r * 10
    tl.store(o + 8 + r, (q - i).to(tl.int64))


def wneg(v, w=32):
    """-v in a signed type of w bits, wrapping (-2^(w-1) is its own negation)."""
    return -v if v != -2**(w - 1) else v


def s1():
    out = []
    o = torch.zeros(16, dtype=torch.int64, device="cuda")
    for dt, tdt, size in ((torch.float32, None, 4), (torch.float16, None, 2), (torch.float64, None, 8)):
        x = torch.zeros(4, dtype=dt, device="cuda")
        base = x.data_ptr()
        bad = 0
        for i in (0, 1, -3, 7, 2**31 - 1, -2**31):
            k = k_s1_arg[(1,)](x, o, i)
            torch.cuda.synchronize()
            if int(o[0]) != base + wneg(i) * size:
                bad += 1
        out.append(f"{str(dt).split('.')[1]} argument i in (0, 1, -3, 7, 2^31-1, -2^31): {bad} of 6 differ from "
                   f"p + (-i)*{size}, -i wrapping in int32")
    ir = ttir(k)
    out.append("TTIR of p - i: " + ("arith.subi from 0 then tt.addptr" if re.search(r"arith.subi %c0_i32", ir)
                                    and "tt.addptr" in ir else "OTHER"))
    x = torch.zeros(4, dtype=torch.float32, device="cuda")
    base = x.data_ptr()
    res = []
    for dt, vals, w, signed in ((torch.int32, (5, -3, 2**31 - 1, -2**31), 32, True),
                                (torch.int64, (5, -3, 2**40), 64, True),
                                (torch.uint32, (1, 2**31), 32, False)):
        bad = 0
        for v in vals:
            ip = torch.tensor([v if (signed or v < 2**31) else v], dtype=torch.int64, device="cuda").to(dt) \
                if dt != torch.uint32 else torch.tensor([v], dtype=torch.int64, device="cuda").to(torch.uint32)
            k_s1_load[(1,)](x, o, ip)
            torch.cuda.synchronize()
            if signed:
                exp = base + wneg(v, w) * 4
            else:
                exp = base + ((2**w - v) % 2**w) * 4
            if int(o[0]) != exp:
                bad += 1
        res.append(f"loaded {str(dt).split('.')[1]} {vals}: {bad} of {len(vals)} differ")
    out.append("; ".join(res) + " (signed: p - i*4, -2^31 wraps to itself; uint32: p + ((2^32 - i) mod 2^32)*4)")
    k_s1_lit[(1,)](x, o)
    torch.cuda.synchronize()
    out.append(f"literals: p - 3 {'= p - 12' if int(o[0]) == base - 12 else 'DIFFERS'}, "
               f"p - 2**40 {'= p - 2^40*4' if int(o[1]) == base - 2**40 * 4 else 'DIFFERS'}")
    iv = torch.tensor([0, 1, -1, 5, -7, 2**31 - 1, -2**31, 3], dtype=torch.int32, device="cuda")
    k_s1_block[(1,)](x, o, iv)
    torch.cuda.synchronize()
    e1 = [base + wneg(v) * 4 for v in iv.tolist()]
    e2 = [base + r * 40 + wneg(v) * 4 for r, v in enumerate(iv.tolist())]
    d1 = sum(int(a) != b for a, b in zip(o[:8].tolist(), e1))
    d2 = sum(int(a) != b for a, b in zip(o[8:].tolist(), e2))
    out.append(f"blocks: p - offs {d1} of 8 differ, ptrs - offs {d2} of 8 differ")
    return "S1: " + "; ".join(out)


S2_SRC = '''
import triton
import triton.language as tl

@triton.jit
def kern(p, q, o, i):
    tl.store(o, ({E}).to(tl.int64))
'''


def s2():
    out = []
    with tempfile.TemporaryDirectory() as d:
        for n, e in enumerate(("i - p", "p - q", "p - 1.0", "1 - p")):
            m = load_module(S2_SRC.replace("{E}", e), f"s2_{n}", d)
            x = torch.zeros(4, dtype=torch.float32, device="cuda")
            o = torch.zeros(1, dtype=torch.int64, device="cuda")
            try:
                m.kern[(1,)](x, x, o, 5)
                torch.cuda.synchronize()
                out.append(f"`{e}`: accepted")
            except Exception as ex:
                out.append(f"`{e}`: rejected ({why(ex)})")
    return "S2: " + "; ".join(out)


@triton.jit
def k_s3(cu, out, req):
    start = tl.load(cu + req - 1)
    tl.store(out + req - 1, start + 100)


def s3():
    cu = torch.tensor([10, 20, 30, 40], dtype=torch.int32, device="cuda")
    out = torch.zeros(4, dtype=torch.int32, device="cuda")
    for req in (1, 2, 3):
        k_s3[(1,)](cu, out, req)
    torch.cuda.synchronize()
    return f"S3: cu + req - 1 for req = 1, 2, 3: out = {out.tolist()} (expected [110, 120, 130, 0])"


# ------------------------------------------------------------------------------------------- T

PRED = ["is_fp8", "is_fp8e4nv", "is_fp8e5", "is_fp16", "is_bf16", "is_fp32", "is_fp64", "is_floating",
        "is_standard_floating", "is_int", "is_int_signed", "is_int_unsigned", "is_bool", "is_int1", "is_int8",
        "is_int16", "is_int32", "is_int64", "is_uint8", "is_uint16", "is_uint32", "is_uint64", "is_ptr"]

T1_SRC = '''
import triton
import triton.language as tl

@triton.jit
def kern(x_ptr, o_ptr):
    x = tl.load(x_ptr + tl.arange(0, 16))
    c: tl.constexpr = x.dtype.{P}()
    tl.static_assert(x.dtype.{P}() == {V})
    if x.dtype.{P}():
        r = tl.full([16], 1, tl.int32)
    else:
        r = tl.full([16], 0, tl.int32)
    s = tl.full([16], 2, tl.int32) if x.dtype.{P}() else tl.full([16], 3, tl.int32)
    tl.store(o_ptr + tl.arange(0, 16), r * 10 + s)
'''

T1_DT = [("float8e4nv", torch.float8_e4m3fn), ("float8e5", torch.float8_e5m2), ("float16", torch.float16),
         ("bfloat16", torch.bfloat16), ("float32", torch.float32), ("float64", torch.float64),
         ("int8", torch.int8), ("int16", torch.int16), ("int32", torch.int32), ("int64", torch.int64),
         ("uint8", torch.uint8), ("bool", torch.bool), ("uint32", torch.uint32)]

TRUTH = {
    "is_fp8": {"float8e4nv", "float8e5"}, "is_fp8e4nv": {"float8e4nv"}, "is_fp8e5": {"float8e5"},
    "is_fp16": {"float16"}, "is_bf16": {"bfloat16"}, "is_fp32": {"float32"}, "is_fp64": {"float64"},
    "is_floating": {"float8e4nv", "float8e5", "float16", "bfloat16", "float32", "float64"},
    "is_standard_floating": {"float16", "bfloat16", "float32", "float64"},
    "is_int": {"int8", "int16", "int32", "int64", "uint8", "bool", "uint32"},
    "is_int_signed": {"int8", "int16", "int32", "int64"},
    "is_int_unsigned": {"uint8", "bool", "uint32"}, "is_bool": {"bool"}, "is_int1": {"bool"},
    "is_int8": {"int8"}, "is_int16": {"int16"}, "is_int32": {"int32"}, "is_int64": {"int64"},
    "is_uint8": {"uint8"}, "is_uint16": set(), "is_uint32": {"uint32"}, "is_uint64": set(), "is_ptr": set(),
}


def t1():
    out, bad, n = [], 0, 0
    with tempfile.TemporaryDirectory() as d:
        for pi, p in enumerate(PRED):
            for di, (name, tdt) in enumerate(T1_DT):
                exp = name in TRUTH[p]
                m = load_module(T1_SRC.replace("{P}", p).replace("{V}", str(exp)), f"t1_{pi}_{di}", d)
                x = torch.zeros(16, dtype=torch.float32, device="cuda").to(tdt) if tdt != torch.uint32 else \
                    torch.zeros(16, dtype=torch.int64, device="cuda").to(torch.uint32)
                o = torch.zeros(16, dtype=torch.int32, device="cuda")
                n += 1
                try:
                    m.kern[(1,)](x, o)
                    torch.cuda.synchronize()
                    got = int(o[0])
                    if got != (12 if exp else 3):
                        bad += 1
                        out.append(f"{p} on {name}: stored {got}")
                except Exception as ex:
                    bad += 1
                    out.append(f"{p} on {name}: {why(ex)}")
    return (f"T1: {len(PRED)} predicates x {len(T1_DT)} dtypes ({n} kernels): {bad} differ from the dtype's name "
            f"(static_assert, compile-time if, conditional expression)" + ("; " + "; ".join(out[:8]) if out else ""))


@triton.jit
def k_t2(x_ptr, s_ptr, o_ptr):
    r = tl.arange(0, 1024)
    x = tl.load(x_ptr + r)
    y = x if x.dtype.is_fp8() else x / tl.load(s_ptr)
    tl.store(o_ptr + r, y)


def sat_rn(v32, dt):
    # torch's float32 -> fp8 is round to nearest even; saturate finite and infinite values to +-max first
    mx = torch.finfo(dt).max
    v = torch.where(torch.isnan(v32), v32, v32.clamp(-mx, mx))
    return v.to(dt)


def t2():
    g = torch.Generator(device="cuda").manual_seed(5)
    out = []
    xb = (torch.randn(1024, device="cuda", generator=g) * 300).to(torch.bfloat16)
    xb[:4] = torch.tensor([1e6, -1e6, 0.0, -0.0], dtype=torch.bfloat16)
    s = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    for odt, oname in ((torch.float8_e4m3fn, "float8e4nv"), (torch.float8_e5m2, "float8e5")):
        o = torch.zeros(1024, dtype=odt, device="cuda")
        k_t2[(1,)](xb, s, o)
        torch.cuda.synchronize()
        ref = sat_rn((xb.float() / s).float(), odt)
        diff = int((o.view(torch.uint8) != ref.view(torch.uint8)).sum())
        out.append(f"bfloat16 x / s stored to {oname}: {diff} of 1024 bits differ from the saturating rounding to "
                   f"nearest even of the float32 quotient (1e6 stored as {float(o[0].float())})")
    x8 = (torch.randn(1024, device="cuda", generator=g) * 4).to(torch.float8_e4m3fn)
    o = torch.zeros(1024, dtype=torch.float8_e4m3fn, device="cuda")
    k_t2[(1,)](x8, s, o)
    torch.cuda.synchronize()
    out.append(f"float8e4nv x stored to float8e4nv (the is_fp8 branch): {int((o.view(torch.uint8) != x8.view(torch.uint8)).sum())} of 1024 differ")
    return "T2: " + "; ".join(out)


# ------------------------------------------------------------------------------------------- R, Y

R1_SRC = '''
import triton
import triton.language as tl

@triton.jit
def kern(x_ptr, o_ptr):
    x = tl.load(x_ptr + tl.arange(0, 16))
    tl.store(o_ptr + tl.arange(0, 16), x.to({D}, fp_downcast_rounding="{R}"))
'''


def r1():
    out = []
    with tempfile.TemporaryDirectory() as d:
        for n, (src, dst, sdt, ddt) in enumerate((("bfloat16", "tl.float16", torch.bfloat16, torch.float16),
                                                  ("float16", "tl.bfloat16", torch.float16, torch.bfloat16),
                                                  ("float32", "tl.float16", torch.float32, torch.float16),
                                                  ("float32", "tl.bfloat16", torch.float32, torch.bfloat16))):
            for r in ("rtne", "rtz"):
                m = load_module(R1_SRC.replace("{D}", dst).replace("{R}", r), f"r1_{n}_{r}", d)
                x = torch.ones(16, dtype=sdt, device="cuda")
                o = torch.zeros(16, dtype=ddt, device="cuda")
                try:
                    m.kern[(1,)](x, o)
                    torch.cuda.synchronize()
                    out.append(f"{src}->{dst[3:]} {r}: accepted")
                except Exception as ex:
                    out.append(f"{src}->{dst[3:]} {r}: rejected ({why(ex)})")
    return "R1: " + "; ".join(out)


Y1_SRC = '''
import triton
import triton.language as tl

@triton.jit
def kern(x_ptr, y_ptr, o_ptr, s):
    x = tl.load(x_ptr + tl.arange(0, 16))
    y = tl.load(y_ptr + tl.arange(0, 32))
    v = tl.load(x_ptr)
    z = tl.load(y_ptr + tl.arange(0, 16))
    r = {E}
    tl.static_print("DTYPE", r.dtype)
    tl.store(o_ptr + tl.arange(0, 16), tl.zeros([16], dtype=tl.float32) + r.to(tl.float32))
'''


def y1():
    out = []
    with tempfile.TemporaryDirectory() as d:
        for n, e in enumerate(("x.to(y.type)", "v.to(x.type)", "v.to(y.type)", "v.to(z.type)", "tl.zeros([16], dtype=x.type)",
                               "tl.zeros([16], dtype=v.type)", "x.to(y.type.element_ty)", "x.to(v.type)",
                               "tl.full([16], 1, x.type)")):
            m = load_module(Y1_SRC.replace("{E}", e), f"y1_{n}", d)
            x = torch.full((16,), 1.5, dtype=torch.float32, device="cuda")
            y = torch.zeros(32, dtype=torch.int32, device="cuda")
            o = torch.zeros(16, dtype=torch.float32, device="cuda")
            try:
                k = m.kern[(1,)](x, y, o, 3)
                torch.cuda.synchronize()
                out.append(f"`{e}`: accepted, o[0] = {float(o[0])}")
            except Exception as ex:
                out.append(f"`{e}`: rejected ({why(ex)})")
    return "Y1 (x a float32 block of 16, y an int32 block of 32, z an int32 block of 16, v a float32 scalar): " + "; ".join(out)


CASES = {"S1": s1, "S2": s2, "S3": s3, "T1": t1, "T2": t2, "R1": r1, "Y1": y1}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(CASES[sys.argv[1]]())
        sys.exit(0)
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    for c in CASES:
        with tempfile.TemporaryDirectory() as cache:
            env = dict(os.environ, TRITON_CACHE_DIR=cache)
            r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), c], env=env,
                               capture_output=True, text=True)
            last = [re.sub(r"(/[^\s:'\"]+)+", "<path>", l) for l in r.stderr.strip().splitlines()[-3:]]
            mine = [l for l in r.stdout.splitlines() if l.startswith(c + ":") or l.startswith(c + " ")]
            print(mine[-1] if mine else f"{c}: failed: {last}", flush=True)
    print("Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/front-end-round32/run.py` from TVA's root.")

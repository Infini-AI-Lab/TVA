"""Front-end forms of round 31: a dtype as a `tl.constexpr` argument, `tl.broadcast_to`, a pointer's
`.type.element_ty` and a pointer as an `if` condition, `tl.cast` (with `fp_downcast_rounding=` and
`bitcast=`).

Library evidence for library issues L139, L140, L142 and L143 (round 31). Sources, pinned in
`primitives/`: `triton-3.8.0-core.py` 667-705 (`pointer_type.element_ty`), 707-756 (`block_type`),
884-890 (`tensor.type`, `tensor.dtype`), 1130-1135 (`tensor.to` is `cast`), 2049-2066 (`broadcast_to`),
2322-2345 (`cast`); `triton-3.8.0-semantic.py` 629-634 (`splat`), 711-725 (`broadcast_impl_shape`),
793-809 (`bitcast`), 811-916 (`cast`); `triton-3.8.0-code-generator.py` 967-996 (`visit_If`),
1535-1544 (`get_Attribute`); `triton-3.8.0-specialize.cc` 515-560, `triton-3.8.0-jit.py` 396-410 (how a
launch passes a `tl.constexpr` argument and a dtype).

  D1  a dtype `D` passed to a `tl.constexpr` parameter, used as `.to(D)`, `tl.cast(x, D)`,
      `tl.zeros(..., dtype=D)`, `tl.full(..., D)`, `d = D` then `.to(d)`, `tl.static_assert(y.dtype == D)`
      and `if D == tl.float32:` -- against the same kernel with the literal dtype written in each place:
      the TTIR, and the stored values
  D2  `tl.dot(a, b, acc, out_dtype=D)` with `acc = tl.zeros(..., dtype=D)`: as D1
  D3  `D` given by keyword and by position: the same TTIR
  D4  one process, the same kernel launched with `D = tl.int32` then `tl.float32`: each launch's own values
  D5  a `tl` dtype passed to a parameter without annotation; a `torch` dtype to a `tl.constexpr`
      parameter used as a dtype: accepted?
  B1  `tl.broadcast_to(x, shape)`, `x.broadcast_to(shape)` and `x.broadcast_to(*shape)` of rank-2 and
      rank-3 blocks with size-1 dimensions, against torch's `expand`
  B2  a scalar (a runtime argument; a loaded scalar) broadcast to a block
  B3  shapes Triton should reject: a non-1 dimension that differs, a rank change, a non-power-of-2 target
  P1  `p.type.element_ty` of a pointer argument of each dtype, against `p.dtype.element_ty`
  P2  `.type.element_ty` of a block (its `.dtype`), of a block of pointers (the pointer type), of an
      `int32` scalar argument (rejected?)
  P3  `if p:` with a pointer argument: a tensor (non-zero address), an empty tensor (address 0), `None`
  C1  `tl.cast(x, d, r, b)`, `x.to(d, r, b)` and `x.cast(d, r, b)`: the same TTIR, over 24 conversions
  C2  `tl.cast` of a Python number, and of a scalar argument specialised to the constant `1`
  C3  `fp_downcast_rounding=` on conversions that are not float narrowings: accepted?
  C4  `bitcast=True`: every pattern of 8- and 16-bit pairs, 2^24 random and edge patterns of 32- and
      64-bit pairs, against the bits; different widths rejected?

Each case runs in its own process with its own empty Triton cache. Paths in messages are not printed.
Run from TVA's root: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/front-end-round31/run.py
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
    inner = [l.strip() for l in chain[-1].splitlines() if l.strip()]
    if chain[-1].startswith("at ") and len(chain) == 1:
        inner = [l.strip() for l in chain[-1].splitlines()[1:] if l.strip()][-1:]
    msg = (inner or [type(e).__name__])[0]
    msg = re.sub(r"(/[^\s:'\"]+)+", "<path>", msg)
    return f"{type(e).__name__}: {msg[:160]}"


def load_module(src, name, d):
    path = os.path.join(d, name + ".py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------------------- D

D1_SRC = '''
import triton
import triton.language as tl

@triton.jit
def kern(x_ptr, o_ptr, D: tl.constexpr):
    offs = tl.arange(0, 16)
    x = tl.load(x_ptr + offs)
    y = x.to({D})
    tl.static_assert(y.dtype == {D})
    z = tl.zeros([16], dtype={D})
    f = tl.full([16], 3, {D})
    d = {D}
    w = tl.cast(x, {D})
    if {D} == tl.float32:
        f = f + 1
    r = (y + z + f + w + (x * 2).to(d)).to(tl.float64)
    tl.store(o_ptr + offs, r)
'''

D2_SRC = '''
import triton
import triton.language as tl

@triton.jit
def kern(a_ptr, b_ptr, o_ptr, D: tl.constexpr):
    r = tl.arange(0, 32)
    a = tl.load(a_ptr + r[:, None] * 32 + r[None, :])
    b = tl.load(b_ptr + r[:, None] * 32 + r[None, :])
    acc = tl.zeros([32, 32], dtype={D})
    acc = tl.dot(a, b, acc, out_dtype={D})
    tl.store(o_ptr + r[:, None] * 32 + r[None, :], acc.to(tl.float64))
'''

DTYPES = [("int8", tl.int8, torch.int8), ("int32", tl.int32, torch.int32), ("int64", tl.int64, torch.int64),
          ("float16", tl.float16, torch.float16), ("bfloat16", tl.bfloat16, torch.bfloat16),
          ("float32", tl.float32, torch.float32), ("float64", tl.float64, torch.float64)]


def d1():
    out = []
    with tempfile.TemporaryDirectory() as d:
        mc = load_module(D1_SRC.replace("{D}", "D"), "d1_cx", d)
        for name, tdt, _ in DTYPES:
            ml = load_module(D1_SRC.replace("{D}", "tl." + name), "d1_lit_" + name, d)
            x = (torch.arange(16, device="cuda", dtype=torch.float32) - 5.5)
            o1 = torch.empty(16, dtype=torch.float64, device="cuda")
            o2 = torch.empty_like(o1)
            k1 = mc.kern[(1,)](x, o1, tdt)
            k2 = ml.kern[(1,)](x, o2, tdt)
            same_ir = ttir(k1) == ttir(k2)
            same_v = bool(torch.equal(o1.view(torch.int64), o2.view(torch.int64)))
            out.append(f"{name}: TTIR {'same' if same_ir else 'DIFFERENT'}, values {'same' if same_v else 'DIFFERENT'}"
                       f" (o[0..2] = {o1[:3].tolist()})")
    return "D1: dtype constexpr vs literal: " + "; ".join(out)


def d2():
    out = []
    with tempfile.TemporaryDirectory() as d:
        mc = load_module(D2_SRC.replace("{D}", "D"), "d2_cx", d)
        for opn, otd, accn, acct in (("int8", torch.int8, "int32", tl.int32), ("float16", torch.float16, "float32", tl.float32),
                                     ("float16", torch.float16, "float16", tl.float16)):
            ml = load_module(D2_SRC.replace("{D}", "tl." + accn), f"d2_lit_{opn}_{accn}", d)
            g = torch.Generator(device="cuda").manual_seed(31)
            if otd == torch.int8:
                a = torch.randint(-128, 128, (32, 32), dtype=torch.int8, device="cuda", generator=g)
                b = torch.randint(-128, 128, (32, 32), dtype=torch.int8, device="cuda", generator=g)
            else:
                a = torch.randn(32, 32, device="cuda", generator=g).to(otd)
                b = torch.randn(32, 32, device="cuda", generator=g).to(otd)
            o1 = torch.empty(32, 32, dtype=torch.float64, device="cuda")
            o2 = torch.empty_like(o1)
            k1 = mc.kern[(1,)](a, b, o1, acct)
            k2 = ml.kern[(1,)](a, b, o2, acct)
            same_ir = ttir(k1) == ttir(k2)
            same_v = bool(torch.equal(o1, o2))
            out.append(f"{opn} into {accn}: TTIR {'same' if same_ir else 'DIFFERENT'}, values {'same' if same_v else 'DIFFERENT'}")
    return "D2: tl.dot out_dtype=D, tl.zeros(dtype=D): " + "; ".join(out)


@triton.jit
def k_d3(x_ptr, o_ptr, A: tl.constexpr, D: tl.constexpr):
    offs = tl.arange(0, 16)
    tl.store(o_ptr + offs, (tl.load(x_ptr + offs).to(D) + A).to(tl.float64))


def d3():
    x = torch.arange(16, device="cuda", dtype=torch.float32) * 0.75
    o1 = torch.empty(16, dtype=torch.float64, device="cuda")
    o2 = torch.empty_like(o1)
    k1 = k_d3[(1,)](x, o1, 1, tl.int16)
    k2 = k_d3[(1,)](x, o2, D=tl.int16, A=1)
    return (f"D3: D=tl.int16 by position and by keyword: TTIR {'same' if ttir(k1) == ttir(k2) else 'DIFFERENT'}, "
            f"values {'same' if torch.equal(o1, o2) else 'DIFFERENT'} ({o1[:4].tolist()})")


@triton.jit
def k_d4(x_ptr, o_ptr, D: tl.constexpr):
    offs = tl.arange(0, 16)
    tl.store(o_ptr + offs, tl.load(x_ptr + offs).to(D).to(tl.float64))


def d4():
    res = []
    x = torch.full((16,), 1.5, device="cuda")
    for name, tdt in (("int32", tl.int32), ("float32", tl.float32), ("int32", tl.int32)):
        o = torch.empty(16, dtype=torch.float64, device="cuda")
        k_d4[(1,)](x, o, tdt)
        res.append(f"D={name}: {o[0].item()}")
    return "D4: one process, x.to(D) of x = 1.5: " + "; ".join(res)


@triton.jit
def k_d5a(o_ptr, D):
    tl.store(o_ptr + tl.arange(0, 16), tl.zeros([16], dtype=tl.float32))


@triton.jit
def k_d5b(x_ptr, o_ptr, D: tl.constexpr):
    offs = tl.arange(0, 16)
    tl.store(o_ptr + offs, tl.load(x_ptr + offs).to(D).to(tl.float32))


def d5():
    out = []
    o = torch.empty(16, dtype=torch.float32, device="cuda")
    try:
        k_d5a[(1,)](o, tl.int32)
        torch.cuda.synchronize()
        out.append("tl.int32 to a parameter without annotation: accepted")
    except Exception as e:
        out.append(f"tl.int32 to a parameter without annotation: rejected ({why(e)})")
    try:
        k_d5b[(1,)](o, o, torch.int32)
        torch.cuda.synchronize()
        out.append("torch.int32 to a tl.constexpr parameter used in .to(D): accepted")
    except Exception as e:
        out.append(f"torch.int32 to a tl.constexpr parameter used in .to(D): rejected ({why(e)})")
    return "D5: " + "; ".join(out)


# ---------------------------------------------------------------------------------------------- B

@triton.jit
def k_b2(x_ptr, o_ptr, S0: tl.constexpr, S1: tl.constexpr, T0: tl.constexpr, T1: tl.constexpr,
         FORM: tl.constexpr):
    r0 = tl.arange(0, S0)
    r1 = tl.arange(0, S1)
    x = tl.load(x_ptr + r0[:, None] * S1 + r1[None, :])
    if FORM == 0:
        y = tl.broadcast_to(x, (T0, T1))
    elif FORM == 1:
        y = x.broadcast_to((T0, T1))
    else:
        y = x.broadcast_to(T0, T1)
    q0 = tl.arange(0, T0)
    q1 = tl.arange(0, T1)
    tl.store(o_ptr + q0[:, None] * T1 + q1[None, :], y)


@triton.jit
def k_b3(x_ptr, o_ptr, S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
         T0: tl.constexpr, T1: tl.constexpr, T2: tl.constexpr):
    r0 = tl.arange(0, S0)
    r1 = tl.arange(0, S1)
    r2 = tl.arange(0, S2)
    x = tl.load(x_ptr + (r0[:, None] * S1 + r1[None, :])[:, :, None] * S2 + r2[None, None, :])
    y = x.broadcast_to((T0, T1, T2))
    q0 = tl.arange(0, T0)
    q1 = tl.arange(0, T1)
    q2 = tl.arange(0, T2)
    tl.store(o_ptr + (q0[:, None] * T1 + q1[None, :])[:, :, None] * T2 + q2[None, None, :], y)


def b1():
    out, bad = [], 0
    g = torch.Generator(device="cuda").manual_seed(310)
    for (s0, s1, t0, t1) in ((1, 8, 4, 8), (4, 1, 4, 8), (1, 1, 4, 8), (4, 8, 4, 8), (1, 1, 1, 1), (16, 1, 16, 1), (1, 1, 16, 1)):
        x = torch.randn(s0, s1, device="cuda", generator=g)
        for form in (0, 1, 2):
            o = torch.empty(t0, t1, device="cuda")
            k_b2[(1,)](x, o, s0, s1, t0, t1, form)
            ok = torch.equal(o, x.expand(t0, t1))
            bad += not ok
        out.append(f"[{s0},{s1}]->[{t0},{t1}]")
    for (s, t) in (((1, 4, 1), (2, 4, 8)), ((2, 1, 8), (2, 4, 8)), ((1, 1, 1), (2, 4, 8))):
        x = torch.randn(*s, device="cuda", generator=g)
        o = torch.empty(*t, device="cuda")
        k_b3[(1,)](x, o, *s, *t)
        bad += not torch.equal(o, x.expand(*t))
        out.append(f"{list(s)}->{list(t)}")
    return f"B1: {len(out)} shape pairs ({', '.join(out)}), rank-2 in 3 forms: {bad} differ from torch expand"


@triton.jit
def k_b4(x_ptr, o_ptr, s, LOADED: tl.constexpr):
    if LOADED:
        v = tl.load(x_ptr)
    else:
        v = s
    y = tl.broadcast_to(v, (4, 8))
    tl.static_assert(y.shape[0] == 4 and y.shape[1] == 8)
    q0 = tl.arange(0, 4)
    q1 = tl.arange(0, 8)
    tl.store(o_ptr + q0[:, None] * 8 + q1[None, :], y)


def b2():
    x = torch.tensor([2.5], device="cuda")
    o = torch.empty(4, 8, device="cuda")
    k_b4[(1,)](x, o, 7.25, True)
    a = torch.equal(o, torch.full((4, 8), 2.5, device="cuda"))
    k_b4[(1,)](x, o, 7.25, False)
    b = torch.equal(o, torch.full((4, 8), 7.25, device="cuda"))
    return f"B2: a loaded scalar broadcast to [4, 8]: {'every element the scalar' if a else 'DIFFERENT'}; a runtime float argument: {'every element the scalar' if b else 'DIFFERENT'}"


@triton.jit
def k_b5(x_ptr, o_ptr, CASE: tl.constexpr):
    r = tl.arange(0, 4)
    x = tl.load(x_ptr + r)
    if CASE == 0:
        y = x[:, None].broadcast_to((8, 1))
        tl.store(o_ptr + tl.arange(0, 8)[:, None], y)
    elif CASE == 1:
        y = x.broadcast_to((4, 4))
        tl.store(o_ptr + tl.arange(0, 4)[:, None] * 4 + tl.arange(0, 4)[None, :], y)
    elif CASE == 2:
        y = x[:, None].broadcast_to((4,))
        tl.store(o_ptr + r, y)
    elif CASE == 3:
        y = x[:, None].broadcast_to((4, 3))
        tl.store(o_ptr + r, tl.sum(y, axis=1))
    elif CASE == 4:
        y = x[:, None].broadcast_to((2, 1))
        tl.store(o_ptr + tl.arange(0, 2)[:, None], y)
    else:
        y = tl.broadcast_to(1.0, (4,))
        tl.store(o_ptr + r, y)


def b3():
    names = {0: "[4,1]->[8,1]", 1: "[4]->[4,4] (rank change)", 2: "[4,1]->[4] (rank change)",
             3: "[4,1]->[4,3] (not a power of 2)", 4: "[4,1]->[2,1]", 5: "tl.broadcast_to(1.0, (4,)) (a Python number)"}
    out = []
    for c in range(6):
        x = torch.ones(16, device="cuda")
        o = torch.zeros(64, device="cuda")
        try:
            k_b5[(1,)](x, o, c)
            torch.cuda.synchronize()
            out.append(f"{names[c]}: accepted")
        except Exception as e:
            out.append(f"{names[c]}: rejected ({why(e)})")
    return "B3: " + "; ".join(out)


# ---------------------------------------------------------------------------------------------- P

@triton.jit
def k_p1(x_ptr, c_ptr, n):
    tl.static_assert(c_ptr.type.element_ty == c_ptr.dtype.element_ty)
    tl.static_assert(c_ptr.type == c_ptr.dtype)
    offs = tl.arange(0, 16)
    x = tl.load(x_ptr + offs)
    tl.store(c_ptr + offs, x.to(c_ptr.type.element_ty), mask=offs < n)


def p1():
    out, bad = [], 0
    x = torch.linspace(-3, 70, 16, device="cuda", dtype=torch.float32)
    for td in (torch.float32, torch.float16, torch.bfloat16, torch.float64, torch.int8, torch.int32, torch.int64, torch.uint8):
        c = torch.zeros(16, dtype=td, device="cuda")
        try:
            k_p1[(1,)](x, c, 16)
            torch.cuda.synchronize()
            exp = x.to(td) if not td in (torch.int8, torch.int32, torch.int64, torch.uint8) else None
            if exp is not None:
                bad += not torch.equal(c, exp)
            out.append(f"{str(td).split('.')[1]}: accepted")
        except Exception as e:
            out.append(f"{str(td).split('.')[1]}: rejected ({why(e)})")
    return f"P1: c_ptr.type.element_ty == c_ptr.dtype.element_ty and c_ptr.type == c_ptr.dtype (static_assert), store of x.to(c_ptr.type.element_ty): {'; '.join(out)}; float stores differing from torch's .to: {bad}"


@triton.jit
def k_p2(x_ptr, o_ptr, n, CASE: tl.constexpr):
    offs = tl.arange(0, 16)
    if CASE == 0:
        x = tl.load(x_ptr + offs)
        tl.static_assert(x.type.element_ty == x.dtype)
        tl.static_assert(x.type.element_ty == tl.float32)
        tl.store(o_ptr + offs, x)
    elif CASE == 1:
        ptrs = x_ptr + offs
        tl.static_assert(ptrs.type.element_ty == ptrs.dtype)
        tl.static_assert(ptrs.type.element_ty == x_ptr.type)
        tl.static_assert(ptrs.type.element_ty != tl.float32)
        tl.store(o_ptr + offs, tl.load(ptrs))
    else:
        tl.static_assert(n.type.element_ty == tl.int32)
        tl.store(o_ptr + offs, tl.load(x_ptr + offs))


def p2():
    out = []
    for c, name in ((0, "a float32 block: .type.element_ty == .dtype == tl.float32"),
                    (1, "a block of pointers: .type.element_ty == .dtype == the pointer type, != tl.float32"),
                    (2, "an int32 scalar argument's .type.element_ty")):
        x = torch.ones(16, device="cuda")
        o = torch.zeros(16, device="cuda")
        try:
            k_p2[(1,)](x, o, 100, c)
            torch.cuda.synchronize()
            out.append(f"{name}: accepted")
        except Exception as e:
            out.append(f"{name}: rejected ({why(e)})")
    return "P2: " + "; ".join(out)


@triton.jit
def k_p3(o_ptr, b_ptr):
    tl.store(o_ptr, 1)
    if b_ptr:
        tl.store(o_ptr + 1, 2)


@triton.jit
def k_p3n(o_ptr, b_ptr):
    tl.store(o_ptr, 1)
    if b_ptr:
        tl.static_assert(False, "visited")
        tl.store(o_ptr + 1, 2)


def p3():
    out = []
    o = torch.zeros(2, dtype=torch.int32, device="cuda")
    t = torch.ones(4, device="cuda")
    ck = k_p3[(1,)](o, t)
    ir = ttir(ck)
    out.append(f"a tensor (address non-zero): the branch {'ran' if o[1].item() == 2 else 'did not run'} (o = {o.tolist()})")
    e = torch.empty(0, device="cuda")
    o.zero_()
    k_p3[(1,)](o, e)
    out.append(f"an empty tensor (data_ptr {e.data_ptr()}): the branch {'ran' if o[1].item() == 2 else 'did not run'} (o = {o.tolist()})")
    o.zero_()
    k_p3n[(1,)](o, None)
    out.append(f"None, the branch holding tl.static_assert(False): no error, o = {o.tolist()}")
    has = ("tt.ptr_to_int" in ir) and ("arith.cmpi ne" in ir) and ("scf.if" in ir)
    out.append(f"the tensor's TTIR {'has' if has else 'LACKS'} tt.ptr_to_int to i64, arith.cmpi ne with 0 and scf.if")
    return "P3: if b_ptr: " + "; ".join(out)


# ---------------------------------------------------------------------------------------------- C

@triton.jit
def k_c1(x_ptr, o_ptr, D: tl.constexpr, R: tl.constexpr, BIT: tl.constexpr, FORM: tl.constexpr):
    offs = tl.arange(0, 256)
    x = tl.load(x_ptr + offs)
    if FORM == 0:
        y = tl.cast(x, dtype=D, fp_downcast_rounding=R, bitcast=BIT)
    elif FORM == 1:
        y = x.to(D, fp_downcast_rounding=R, bitcast=BIT)
    else:
        y = x.cast(D, fp_downcast_rounding=R, bitcast=BIT)
    tl.store(o_ptr + offs, y)


C1_CASES = [
    (torch.float32, tl.float16, None, False), (torch.float32, tl.float16, "rtne", False),
    (torch.float32, tl.float16, "rtz", False), (torch.float32, tl.bfloat16, "rtz", False),
    (torch.float32, tl.bfloat16, None, False), (torch.float16, tl.float32, None, False),
    (torch.bfloat16, tl.float16, None, False), (torch.float64, tl.float32, None, False),
    (torch.float32, tl.int32, None, False), (torch.float32, tl.uint8, None, False),
    (torch.int32, tl.float32, None, False), (torch.int32, tl.int64, None, False),
    (torch.int64, tl.int32, None, False), (torch.int8, tl.uint32, None, False),
    (torch.int32, tl.int1, None, False), (torch.float32, tl.float8e4nv, None, False),
    (torch.float32, tl.float8e5, "rtz", False), (torch.float32, tl.float32, "rtz", False),
    (torch.float32, tl.int32, None, True), (torch.float16, tl.int16, None, True),
    (torch.bfloat16, tl.float16, None, True), (torch.float64, tl.uint64, None, True),
    (torch.float32, tl.float16, "rtz", True), (torch.int8, tl.float8e4nv, None, True),
]

TORCH_OF = {tl.float16: torch.float16, tl.bfloat16: torch.bfloat16, tl.float32: torch.float32,
            tl.float64: torch.float64, tl.int1: torch.bool, tl.int8: torch.int8, tl.uint8: torch.uint8,
            tl.int16: torch.int16, tl.uint16: torch.uint16, tl.int32: torch.int32, tl.uint32: torch.uint32, tl.int64: torch.int64,
            tl.uint64: torch.uint64, tl.float8e4nv: torch.float8_e4m3fn, tl.float8e5: torch.float8_e5m2}


def c1():
    out, diff = [], 0
    g = torch.Generator(device="cuda").manual_seed(311)
    for (sd, d, r, bit) in C1_CASES:
        if sd.is_floating_point:
            x = (torch.randn(256, device="cuda", generator=g) * 50).to(sd)
        else:
            x = torch.randint(-100, 100, (256,), device="cuda", generator=g).to(sd)
        irs, vals = [], []
        for form in (0, 1, 2):
            o = torch.zeros(256, dtype=TORCH_OF[d], device="cuda")
            try:
                ck = k_c1[(1,)](x, o, d, r, bit, form)
                irs.append(ttir(ck))
            except Exception as e:
                irs.append("rejected: " + why(e))
            vals.append(o.clone())
        same = irs[0] == irs[1] == irs[2]
        if irs[0].startswith("rejected"):
            same = same and True
            out.append(f"{str(sd).split('.')[1]}->{d}{',' + r if r else ''}{',bitcast' if bit else ''}: all three {irs[0] if same else 'NOT ALL rejected'}")
            diff += not same
            continue
        diff += not same
        out.append(f"{str(sd).split('.')[1]}->{d}{',' + r if r else ''}{',bitcast' if bit else ''}: {'same' if same else 'DIFFERENT'}")
    return f"C1: TTIR of tl.cast / .to / .cast, {len(C1_CASES)} conversions, {diff} differ: " + "; ".join(out)


@triton.jit
def k_c2(o_ptr, o64_ptr, s, CASE: tl.constexpr):
    if CASE == 0:
        y = tl.cast(s, dtype=tl.int64)
        tl.static_assert(y.dtype == tl.int64)
        tl.store(o64_ptr, y)
    elif CASE == 1:
        tl.store(o64_ptr, tl.cast(1, tl.int64))
    elif CASE == 2:
        tl.store(o_ptr, tl.cast(2**40 + 5, tl.int32))
    elif CASE == 3:
        tl.store(o_ptr, tl.cast(1.75, tl.int32))
    else:
        tl.store(o_ptr, tl.cast(-2.5, tl.int32))


def c2():
    o = torch.zeros(1, dtype=torch.int32, device="cuda")
    o64 = torch.zeros(1, dtype=torch.int64, device="cuda")
    r = []
    for s in (1, 7, 2**40):
        k_c2[(1,)](o, o64, s, 0)
        r.append(f"tl.cast(s, dtype=tl.int64) with s={s}: {o64.item()}")
    k_c2[(1,)](o, o64, 0, 1)
    r.append(f"tl.cast(1, tl.int64): {o64.item()}")
    k_c2[(1,)](o, o64, 0, 2)
    r.append(f"tl.cast(2**40 + 5, tl.int32): {o.item()}")
    k_c2[(1,)](o, o64, 0, 3)
    r.append(f"tl.cast(1.75, tl.int32): {o.item()}")
    k_c2[(1,)](o, o64, 0, 4)
    r.append(f"tl.cast(-2.5, tl.int32): {o.item()}")
    return "C2: " + "; ".join(r)


def c3():
    out = []
    for (sd, d, rr) in ((torch.float16, tl.float32, "rtne"), (torch.float32, tl.float64, "rtz"),
                        (torch.int32, tl.float32, "rtne"), (torch.float32, tl.int32, "rtz"),
                        (torch.float32, tl.float32, "rtz"), (torch.float16, tl.float16, "rtne"),
                        (torch.float32, tl.float16, "rtne"), (torch.float32, tl.float16, "rd")):
        x = torch.ones(256, dtype=sd, device="cuda")
        o = torch.zeros(256, dtype=TORCH_OF[d], device="cuda")
        name = f"{str(sd).split('.')[1]}->{d},{rr}"
        try:
            k_c1[(1,)](x, o, d, rr, False, 0)
            torch.cuda.synchronize()
            out.append(f"{name}: accepted")
        except Exception as e:
            out.append(f"{name}: rejected ({why(e)})")
    return "C3: fp_downcast_rounding=: " + "; ".join(out)


@triton.jit
def k_c4(x_ptr, o_ptr, n, D: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m).to(D, bitcast=True), mask=m)


def c4():
    out, bad = [], 0
    g = torch.Generator(device="cuda").manual_seed(312)
    pairs8 = [(torch.int8, tl.uint8), (torch.uint8, tl.int8), (torch.uint8, tl.float8e4nv), (torch.uint8, tl.float8e5),
              (torch.float8_e4m3fn, tl.uint8), (torch.float8_e5m2, tl.int8), (torch.float8_e4m3fn, tl.float8e5)]
    pairs16 = [(torch.float16, tl.int16), (torch.int16, tl.float16), (torch.bfloat16, tl.uint16),
               (torch.uint16, tl.bfloat16), (torch.float16, tl.bfloat16), (torch.bfloat16, tl.float16)]
    pairs32 = [(torch.float32, tl.int32), (torch.int32, tl.float32), (torch.float32, tl.uint32), (torch.uint32, tl.float32)]
    pairs64 = [(torch.float64, tl.int64), (torch.int64, tl.float64), (torch.float64, tl.uint64)]
    ibits = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    for group, w in ((pairs8, 1), (pairs16, 2), (pairs32, 4), (pairs64, 8)):
        if w <= 2:
            bits = torch.arange(0, 2 ** (8 * w), device="cuda", dtype=torch.int64)
            bits = bits.to(torch.int32).to(ibits[w]) if w == 2 else bits.to(torch.uint8)
        elif w == 4:
            bits = torch.randint(-2**31, 2**31 - 1, (2**24,), dtype=torch.int32, device="cuda", generator=g)
            edges = torch.tensor([0, -2**31, 0x7f800000, -8388608, 0x7fc00000, 0x7f800001, 0x7fffffff, -1, 1, 0x00800000],
                                 dtype=torch.int64).to(torch.int32).to("cuda")
            bits = torch.cat([bits, edges])
        else:
            bits = torch.randint(-2**63, 2**63 - 1, (2**24,), dtype=torch.int64, device="cuda", generator=g)
            edges = torch.tensor([0, -2**63, 0x7ff0000000000000, 0x7ff8000000000000, 0x7ff0000000000001, -1, 1],
                                 dtype=torch.int64, device="cuda")
            bits = torch.cat([bits, edges])
        for (sd, d) in group:
            x = bits.view(sd)
            o = torch.empty(x.numel(), dtype=TORCH_OF[d], device="cuda")
            k_c4[(triton.cdiv(x.numel(), 1024),)](x, o, x.numel(), d, BLOCK=1024)
            dd = int((o.view(ibits[w]) != bits.view(ibits[w])).sum().item())
            bad += dd
            out.append(f"{str(sd).split('.')[1]}->{d} ({x.numel()}): {dd}")
    rej = []
    for (sd, d) in ((torch.float32, tl.int16), (torch.int8, tl.float16), (torch.float64, tl.float32)):
        x = torch.ones(16, dtype=sd, device="cuda")
        o = torch.zeros(16, dtype=TORCH_OF[d], device="cuda")
        try:
            k_c4[(1,)](x, o, 16, d, BLOCK=16)
            torch.cuda.synchronize()
            rej.append(f"{str(sd).split('.')[1]}->{d}: accepted")
        except Exception as e:
            rej.append(f"{str(sd).split('.')[1]}->{d}: rejected ({why(e)})")
    return (f"C4: bitcast=True, patterns whose stored bits differ from the input's: {'; '.join(out)}; total {bad}. "
            f"Different widths: {'; '.join(rej)}")


CASES = {"D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5, "B1": b1, "B2": b2, "B3": b3,
         "P1": p1, "P2": p2, "P3": p3, "C1": c1, "C2": c2, "C3": c3, "C4": c4}

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
            print(mine[-1] if mine else f"{c}: failed: {last}")
    print("Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/front-end-round31/run.py` from TVA's root.")

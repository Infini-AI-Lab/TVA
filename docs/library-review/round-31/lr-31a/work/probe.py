"""lr-31a probes for round 31 (L139-L143). Run from TVA's root:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-31/lr-31a/work/probe.py [all [<case> ...]]

  E1  `if p:` on a pointer argument that is an empty view of a non-empty tensor (data_ptr non-zero),
      against `torch.empty(0)` (data_ptr 0)
  R1  `fp_downcast_rounding=` on conversions the page does not list: bfloat16 <-> float16 (same width),
      float64 -> float32 / bfloat16 / float16 with "rtz", float16 -> float8e5 with "rtz", an invalid
      mode to the same dtype, `tl.cast` of Python numbers with `bitcast=True`
  T1  integer `tl.dot`: `b` given transposed (`tl.trans`, `.T`), pipelined (`num_stages=3`), at the
      edge of the room, both instruction paths
  K1  a dtype `tl.constexpr` argument: `tl.float8e4nv`, `tl.float8e5`, `tl.uint8`, `tl.int1` in `.to(D)`
      and `tl.zeros(dtype=D)`, against the literal (the fp8 ones add in fp8, outside the subset: rejected)
  Y1  survey: a block's `.type` used where a dtype goes (`.to`, `tl.zeros`)
  F1  survey: `float64` to `float16` / `bfloat16` / `float32`, one rounding? (double-rounding traps)
"""
import os, re, subprocess, sys, tempfile

import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))


def short(e):
    s = str(e).strip().splitlines()
    s = [l.strip() for l in s if l.strip()]
    msg = s[-1] if s else type(e).__name__
    c = e.__cause__ or e.__context__
    while c is not None:
        cs = [l.strip() for l in str(c).splitlines() if l.strip()]
        if cs:
            msg = cs[-1]
        c = c.__cause__ or c.__context__
    msg = re.sub(r"/[^\s:]+", "<path>", msg)
    return f"{type(e).__name__}: {msg[:200]}"


# ---------------------------------------------------------------- E1
@triton.jit
def k_ifp(o_ptr, b_ptr):
    tl.store(o_ptr, 1)
    if b_ptr:
        tl.store(o_ptr + 1, 2)


def case_E1():
    base = torch.arange(16, device="cuda", dtype=torch.float32)
    cases = {
        "torch.empty(0)": torch.empty(0, device="cuda"),
        "base[:0]": base[:0],
        "base[5:5]": base[5:5],
        "torch.empty(0) then .view": torch.zeros(0, 4, device="cuda"),
    }
    out = []
    for name, t in cases.items():
        o = torch.zeros(2, dtype=torch.int32, device="cuda")
        k_ifp[(1,)](o, t)
        torch.cuda.synchronize()
        out.append(f"{name}: numel {t.numel()}, data_ptr {'0' if t.data_ptr() == 0 else 'non-zero'}, "
                   f"branch {'ran' if o[1].item() == 2 else 'did not run'}")
    print("E1: " + "; ".join(out))


# ---------------------------------------------------------------- R1
def make_cast(src, dst, rnd, bit):
    @triton.jit
    def k(x_ptr, o_ptr, SRC: tl.constexpr, DST: tl.constexpr, R: tl.constexpr, BIT: tl.constexpr):
        offs = tl.arange(0, 16)
        x = tl.load(x_ptr + offs)
        y = tl.cast(x, DST, fp_downcast_rounding=R, bitcast=BIT)
        tl.store(o_ptr + offs, y)
    return k


TD = {tl.float64: torch.float64, tl.float32: torch.float32, tl.float16: torch.float16,
      tl.bfloat16: torch.bfloat16, tl.float8e5: torch.float8_e5m2, tl.float8e4nv: torch.float8_e4m3fn,
      tl.int32: torch.int32, tl.int16: torch.int16}


def run_cast(src, dst, rnd, bit=False):
    k = make_cast(src, dst, rnd, bit)
    x = (torch.randn(16, device="cuda", dtype=torch.float32) * 3).to(TD[src])
    o = torch.empty(16, device="cuda", dtype=TD[dst])
    try:
        k[(1,)](x, o, src, dst, rnd, bit)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        return "rejected (" + short(e) + ")"
    if dst in (tl.float32, tl.float16, tl.bfloat16, tl.float64) and src in (tl.float64, tl.float32,
                                                                           tl.bfloat16, tl.float16):
        xd = x.double()
        if rnd == "rtz":
            # every result must not exceed |x| in magnitude and be the largest such value of dst
            ok = bool(((o.double().abs() <= xd.abs()) | xd.isnan()).all())
            return f"accepted ({'toward zero' if ok else 'NOT toward zero'})"
        ref = x.to(TD[dst]) if src != tl.bfloat16 or dst != tl.float16 else x.float().to(torch.float16)
        same = bool((o.view(torch.int16 if TD[dst].itemsize == 2 else torch.int32 if TD[dst].itemsize == 4
                             else torch.int64) == ref.view(torch.int16 if TD[dst].itemsize == 2 else torch.int32
                                                           if TD[dst].itemsize == 4 else torch.int64)).all())
        return f"accepted ({'= torch' if same else 'differs from torch'})"
    return "accepted"


@triton.jit
def k_pybit(o_ptr):
    tl.store(o_ptr, tl.cast(1.0, tl.int32, bitcast=True))
    tl.store(o_ptr + 1, tl.cast(True, tl.int32))
    tl.store(o_ptr + 2, tl.cast(-1, tl.uint32, bitcast=True).to(tl.int32, bitcast=True))


def case_R1():
    out = []
    for (s, d, r) in [(tl.bfloat16, tl.float16, "rtne"), (tl.bfloat16, tl.float16, "rtz"),
                      (tl.float16, tl.bfloat16, "rtne"), (tl.bfloat16, tl.float16, None),
                      (tl.float64, tl.float32, "rtz"), (tl.float64, tl.bfloat16, "rtz"),
                      (tl.float64, tl.float16, "rtz"), (tl.float64, tl.float32, "rtne"),
                      (tl.float16, tl.float8e5, "rtz"), (tl.float16, tl.float8e5, "rtne"),
                      (tl.float8e5, tl.float8e4nv, None), (tl.float32, tl.float32, "rd"),
                      (tl.float32, tl.float32, "rtz")]:
        out.append(f"{s}->{d},{r}: {run_cast(s, d, r)}")
    o = torch.zeros(3, dtype=torch.int32, device="cuda")
    try:
        k_pybit[(1,)](o)
        torch.cuda.synchronize()
        out.append(f"tl.cast(1.0, int32, bitcast=True) = {hex(o[0].item())}; tl.cast(True, int32) = {o[1].item()};"
                   f" tl.cast(-1, uint32, bitcast=True) as int32 = {o[2].item()}")
    except Exception as e:  # noqa: BLE001
        out.append("python-number casts: rejected (" + short(e) + ")")
    print("R1: " + "; ".join(out))


# ---------------------------------------------------------------- T1
@triton.jit
def k_dot(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
          TRANS: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    for k in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None] * K + (k + rk)[None, :])
        if TRANS == 1:
            # b stored as [N, K]
            bt = tl.load(b_ptr + rn[:, None] * K + (k + rk)[None, :])
            acc = tl.dot(a, tl.trans(bt), acc)
        elif TRANS == 2:
            bt = tl.load(b_ptr + rn[:, None] * K + (k + rk)[None, :])
            acc += tl.dot(a, bt.T)
        else:
            b = tl.load(b_ptr + (k + rk)[:, None] * N + rn[None, :])
            acc = tl.dot(a, b, acc, out_dtype=tl.int32)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)


def case_T1():
    torch.manual_seed(31)
    out = []
    tot = 0
    bad = 0
    for (BM, BN, BK, nw, ns) in [(128, 128, 64, 8, 3), (64, 64, 128, 4, 4), (16, 32, 64, 2, 3),
                                 (64, 256, 32, 4, 2)]:
        for trans in (0, 1, 2):
            M, N, K = 2 * BM, 2 * BN, 1024
            a = torch.randint(-128, 128, (M, K), device="cuda", dtype=torch.int8)
            b = torch.randint(-128, 128, (K, N), device="cuda", dtype=torch.int8)
            a[0, :] = -128
            b[:, 0] = -128
            a[1, :] = 127
            room = (2**31 - 1) - (a.double().abs() @ b.double().abs())
            acc0 = ((torch.rand(M, N, device="cuda", dtype=torch.float64) * 2 - 1) * room).floor()
            acc0[0, 0] = room[0, 0]  # at the edge of the room
            acc = acc0.to(torch.int32).contiguous()
            exact = acc0 + a.double() @ b.double()
            bb = b.t().contiguous() if trans else b
            k_dot[(M // BM, N // BN)](a, bb, acc, M, N, K, BM=BM, BN=BN, BK=BK, TRANS=trans,
                                      num_warps=nw, num_stages=ns)
            torch.cuda.synchronize()
            bad += int((acc.double() != exact).sum().item())
            tot += M * N
        out.append(f"({BM},{BN},{BK},{nw},{ns})")
    print(f"T1: configs {' '.join(out)} x 3 forms (b, tl.trans(bT), bT.T with +=), {tot} outputs: {bad} differ; "
          f"edge output [0,0] at |acc|+sum|ab| = 2^31-1 included")


# ---------------------------------------------------------------- K1
def make_k1(lit):
    src = f'''
import triton
import triton.language as tl
@triton.jit
def kp(x_ptr, o_ptr, D: tl.constexpr):
    offs = tl.arange(0, 16)
    x = tl.load(x_ptr + offs)
    z = tl.zeros([16], dtype=D)
    tl.store(o_ptr + offs, (x.to(D) + z).to(tl.float32))
@triton.jit
def kl(x_ptr, o_ptr, D: tl.constexpr):
    offs = tl.arange(0, 16)
    x = tl.load(x_ptr + offs)
    z = tl.zeros([16], dtype={lit})
    tl.store(o_ptr + offs, (x.to({lit}) + z).to(tl.float32))
'''
    d = tempfile.mkdtemp()
    p = os.path.join(d, f"k1_{lit.replace('.', '_')}.py")
    with open(p, "w") as f:
        f.write(src)
    import importlib.util
    spec = importlib.util.spec_from_file_location(os.path.basename(p)[:-3], p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def case_K1():
    out = []
    x = torch.linspace(-3.3, 3.3, 16, device="cuda")
    for lit, D in [("tl.float8e4nv", tl.float8e4nv), ("tl.float8e5", tl.float8e5), ("tl.uint8", tl.uint8),
                   ("tl.int1", tl.int1)]:
        m = make_k1(lit)
        o1 = torch.empty(16, device="cuda")
        o2 = torch.empty(16, device="cuda")
        r = []
        for k, o in ((m.kp, o1), (m.kl, o2)):
            try:
                k[(1,)](x, o, D)
                torch.cuda.synchronize()
                r.append("ok")
            except Exception as e:  # noqa: BLE001
                r.append("rejected (" + short(e) + ")")
        same = r == ["ok", "ok"] and bool((o1.view(torch.int32) == o2.view(torch.int32)).all())
        out.append(f"{lit}: param {r[0]}, literal {r[1]}, values {'same' if same else 'differ or n/a'}")
    print("K1: " + "; ".join(out))


# ---------------------------------------------------------------- F1 (survey: "Float conversions")
@triton.jit
def k_trunc64(x_ptr, o16_ptr, ob_ptr, o32_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    tl.store(o16_ptr + offs, x.to(tl.float16))
    tl.store(ob_ptr + offs, x.to(tl.bfloat16))
    tl.store(o32_ptr + offs, x.to(tl.float32))


def rne(v, frac_bits, emin):
    """Round the double v (finite, normal range of the target) to `frac_bits` fraction bits, RNE, exactly."""
    from fractions import Fraction
    import math
    q = Fraction(v)
    if q == 0:
        return 0.0
    s = -1 if q < 0 else 1
    q = abs(q)
    e = math.floor(math.log2(q))
    while Fraction(2) ** e > q:
        e -= 1
    while Fraction(2) ** (e + 1) <= q:
        e += 1
    e = max(e, emin)
    ulp = Fraction(2) ** (e - frac_bits)
    n = q / ulp
    f = n.numerator // n.denominator
    r = n - f
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and f % 2 == 1):
        f += 1
    return float(s * f * ulp)


def case_F1():
    vals = []
    for k in range(1, 7):
        vals += [1 + 2**-8 + 2**-30 * k, 1 + 2**-11 + 2**-40 * k, -(1 + 2**-8 + 2**-29 * k),
                 1 + 2**-24 + 2**-50 * k, 3 * (1 + 2**-8 + 2**-31 * k), 1.5 * (1 + 2**-11 + 2**-37 * k)]
    torch.manual_seed(5)
    rnd = (torch.randn(4096, dtype=torch.float64) * 4).tolist()
    vals += rnd
    n0 = len(vals)
    N = 1
    while N < len(vals):
        N *= 2
    vals += [0.0] * (N - len(vals))
    x = torch.tensor(vals, dtype=torch.float64, device="cuda")
    o16 = torch.empty(N, dtype=torch.float16, device="cuda")
    ob = torch.empty(N, dtype=torch.bfloat16, device="cuda")
    o32 = torch.empty(N, dtype=torch.float32, device="cuda")
    k_trunc64[(1,)](x, o16, ob, o32, N=N)
    torch.cuda.synchronize()
    bad16 = sum(1 for v, r in zip(vals, o16.double().tolist()) if rne(v, 10, -14) != r)
    badb = sum(1 for v, r in zip(vals, ob.double().tolist()) if rne(v, 7, -126) != r)
    bad32 = sum(1 for v, r in zip(vals, o32.double().tolist()) if rne(v, 23, -126) != r)
    ptx = ""
    try:
        ck = k_trunc64.warmup(x, o16, ob, o32, N=N, grid=(1,))
        ptx = " ".join(sorted(set(re.findall(r"cvt\.[a-z0-9.]+", ck.asm["ptx"]))))
    except Exception as e:  # noqa: BLE001
        ptx = "n/a (" + short(e) + ")"
    print(f"F1: float64 -> float16 / bfloat16 / float32, {n0} values (36 double-rounding traps, "
          f"4096 random; padded with zeros to {N}): differ from one RNE rounding: {bad16} / {badb} / {bad32}; PTX conversions: {ptx}")


# ---------------------------------------------------------------- Y1 (survey: a block's `.type` used as a dtype)
@triton.jit
def k_bt(x_ptr, i_ptr, o_ptr, F: tl.constexpr):
    offs = tl.arange(0, 16)
    x = tl.load(x_ptr + offs)          # float16 block
    i = tl.load(i_ptr + offs)          # int32 block
    if F == 0:
        y = i.to(x.type)               # a block type as `.to`'s dtype
    elif F == 1:
        y = tl.load(i_ptr).to(x.type) + tl.zeros([16], dtype=tl.float16)  # a scalar to a block type
    else:
        y = tl.zeros([16], dtype=x.type)  # a block type as `tl.zeros`'s dtype
    tl.store(o_ptr + offs, y)


def case_Y1():
    x = torch.ones(16, device="cuda", dtype=torch.float16)
    i = torch.arange(16, device="cuda", dtype=torch.int32) + 1
    out = []
    for F, name in ((0, "i.to(x.type)"), (1, "scalar.to(x.type)"), (2, "tl.zeros([16], dtype=x.type)")):
        o = torch.full((16,), -1.0, device="cuda", dtype=torch.float16)
        try:
            k_bt[(1,)](x, i, o, F)
            torch.cuda.synchronize()
            out.append(f"{name}: accepted, o[0..2] = {o[:3].tolist()}")
        except Exception as e:  # noqa: BLE001
            out.append(f"{name}: rejected ({short(e)})")
    print("Y1: " + "; ".join(out))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "all":
        globals()["case_" + sys.argv[1]]()
    else:
        print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
        for c in (sys.argv[2:] or ["E1", "R1", "T1", "K1", "F1", "Y1"]):
            env = dict(os.environ)
            env["TRITON_CACHE_DIR"] = tempfile.mkdtemp()
            r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), c], env=env,
                               capture_output=True, text=True, timeout=1200)
            # keep the case's own lines only (a rejected module's IR dump carries source paths)
            sys.stdout.write("".join(l for l in r.stdout.splitlines(True) if l.startswith(c + ": ")))
            if r.returncode:
                print(f"{c}: exit {r.returncode}: {r.stderr.strip().splitlines()[-1][:300] if r.stderr.strip() else ''}")

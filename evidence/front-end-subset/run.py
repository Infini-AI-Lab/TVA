"""The supported subset of the front end: facts the round-15 batch states, and forms it excludes.

Library evidence for library issues L80, L83, L84, L85, L86, L87 and L88 (round 15). Sources:
`primitives/triton-3.8.0-code-generator.py` (the front end), `triton-3.8.0-semantic.py` (`where`,
`load`, the promotions) and `triton-3.8.0-standard.py` (`cumsum`).

Each case is compiled and run on GPU 0 in its own process, with its own empty Triton cache; a case the
front end refuses is reported with its error ("rejected on the host"). The kernels are small: one
program unless the case says otherwise.

  W  tl.where with a Python number as a value (L83; semantic.py 1638-1652, 175-210, 68-116)
  W1   tl.where(keep, x, 0.0), x int32, keep an int32 mask: the result's dtype, and its values
  W2   tl.where(keep, x, 0), x int32 (tutorial 09's form)
  W3   tl.where(m, 0, -1.0e6), both values Python numbers (tutorial 06's form)
  W4   tl.where(keep, x, 0.0), x bfloat16                W5  the same, x float16
  C  annotated assignments (L84; code_generator.py 694-760)
  C1   c: tl.constexpr = -7; c // 2                       (one statement, a new name)
  C2   c = 1, then c: tl.constexpr = -7                   (the name is already defined)
  C3   c: tl.constexpr = -7; c += 0; c // 2               (a later plain assignment)
  C4   BLOCK = BLOCK * 1 (a tl.constexpr parameter); tl.arange(0, BLOCK)
  C5   bare  e: tl.constexpr ; then e = 1e-7; x * e, x bfloat16   (excluded form)
  C6   a: tl.int64 = 2**31 - 1; a + 1                     (another annotation; excluded form)
  C7   BM: tl.constexpr = x.shape[0]; tl.arange(0, BM)    (tutorial 06's form)
  C8   P: tl.constexpr = A == 1 and B == 2; if P: ... else: tl.static_assert(False)   (tutorial 10)
  C9   for j in tl.static_range(2): b: tl.constexpr = 4  (the second copy defines b again)
  R  one-argument range (L85; code_generator.py 1204-1245)
  R1   for i in range(tl.atomic_add(c, 1) * 0), 8 programs: atomics run
  R2   range(0, tl.atomic_add(c, 1) * 0)                  R3  tl.range(tl.atomic_add(c, 1) * 0)
  R4   for i in range(tl.atomic_add(c, 1)), one program, c = 5: trips, and c after
  D  helper defaults and parameter annotations (L80; code_generator.py 1358-1365, 628-692)
  D1   h(x) with def h(x, e=1e-7): return x * e, x bfloat16       D2  h(x, 1e-7)
  D3   e = 1e-7; h(x, e)                                  D4  def h(x, e: tl.constexpr = 1e-7)
  P1   def g(n: tl.constexpr): return n // 2; a = -7; g(a)       P2  g(-7)
  L  names around a loop (code_generator.py 1204-1339, 1126-1185, 465-503, 121-140)
  L1   for i in range(0, n): t = i * 2; then t after the loop
  L2   for i in range(0, n): s += i; then i after the loop (n = 4)
  L3   s = 0; for i in range(0, n): s += i; then s after the loop (control)
  L4   while j < n: u = j; j += 1; then u after the loop
  Z  the dry visit of a loop body (code_generator.py 465-503)
  Z1   for i in range(0, n): tl.atomic_add(c, 1), n = 0, 8 programs   Z2  the same with n = 1
  U  a tuple assignment with fewer names than elements (code_generator.py 714-725)
  U1   a, b = 1, 2, 3
  O  tl.load's other (L86; semantic.py 1043-1045)
  O1   masked-off lanes of a float16 load with other = 1 + 2**-11 + 2**-30: the bits
  S  tl.cumsum's dtype= (L87; standard.py 334-343, 270-281)
  S1   cumsum(x, dtype=tl.bfloat16), x bfloat16 [1, 2**-8, 2**-8, 2**-8]   (default: in float32)
  S2   cumsum(x, dtype=tl.int8), x int8 [100, 100, 100, 100]               (default: in int32)
  S3   cumsum(x, dtype=tl.float32), x float16 [2048, 1, 1, 1]              (default: in float16)

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/front-end-subset/run.py  (from TVA's root;
the evidence used `.venv` of the TVA checkout, Triton 3.8.0)
"""
import json, os, struct, subprocess, sys, tempfile

import torch
import triton
import triton.language as tl


# --- W: tl.where -------------------------------------------------------------------------------

@triton.jit
def k_where(x_ptr, keep_ptr, out_ptr, CASE: tl.constexpr):
    i = tl.arange(0, 4)
    keep = tl.load(keep_ptr + i)
    if CASE == 1:
        x = tl.load(x_ptr + i)
        y = tl.where(keep, x, 0.0)
        tl.static_assert(y.dtype == tl.float32)
    elif CASE == 2:
        x = tl.load(x_ptr + i)
        y = tl.where(keep, x, 0)
        tl.static_assert(y.dtype == tl.int32)
    else:
        y = tl.where(keep != 0, 0, -1.0e6)
        tl.static_assert(y.dtype == tl.float32)
    tl.store(out_ptr + i, y)


@triton.jit
def k_where_f(x_ptr, keep_ptr, out_ptr, CASE: tl.constexpr):
    i = tl.arange(0, 4)
    keep = tl.load(keep_ptr + i)
    x = tl.load(x_ptr + i)
    y = tl.where(keep, x, 0.0)
    if CASE == 4:
        tl.static_assert(y.dtype == tl.bfloat16)
    else:
        tl.static_assert(y.dtype == tl.float16)
    tl.store(out_ptr + i, y.to(tl.float32))


# --- C: annotations ----------------------------------------------------------------------------

@triton.jit
def k_ann_int(out_ptr, CASE: tl.constexpr):
    if CASE == 1:
        c: tl.constexpr = -7
        q = c // 2
    elif CASE == 2:
        c = 1
        c: tl.constexpr = -7
        q = c // 2
    elif CASE == 3:
        c: tl.constexpr = -7
        c += 0
        q = c // 2
    else:
        a: tl.int64 = 2**31 - 1
        q = a + 1
    tl.store(out_ptr, q)


@triton.jit
def k_ann_block(out_ptr, BLOCK: tl.constexpr, CASE: tl.constexpr):
    if CASE == 4:
        BLOCK = BLOCK * 1
        tl.store(out_ptr + tl.arange(0, BLOCK), 1)
    else:
        x = tl.zeros([BLOCK], dtype=tl.int32)
        BM: tl.constexpr = x.shape[0]
        tl.store(out_ptr + tl.arange(0, BM), 1)


@triton.jit
def k_ann_bare(x_ptr, out_ptr, CASE: tl.constexpr):
    i = tl.arange(0, 4)
    x = tl.load(x_ptr + i)
    if CASE == 5:
        e: tl.constexpr
        e = 1e-7
        y = x * e
    else:
        e = 1e-7
        y = x * e
    tl.store(out_ptr + i, y.to(tl.float32))


@triton.jit
def k_ann_static(out_ptr):
    for j in tl.static_range(2):
        b: tl.constexpr = 4
        tl.store(out_ptr + j, b)


@triton.jit
def k_ann_bool(out_ptr, A: tl.constexpr, B: tl.constexpr):
    P: tl.constexpr = A == 1 and B == 2
    if P:
        tl.store(out_ptr, 1)
    else:
        tl.static_assert(False)


# --- R: range with one argument ----------------------------------------------------------------

@triton.jit
def k_range(c_ptr, out_ptr, CASE: tl.constexpr):
    s = 0
    if CASE == 1:
        for i in range(tl.atomic_add(c_ptr, 1) * 0):
            s += 1
    elif CASE == 2:
        for i in range(0, tl.atomic_add(c_ptr, 1) * 0):
            s += 1
    elif CASE == 3:
        for i in tl.range(tl.atomic_add(c_ptr, 1) * 0):
            s += 1
    else:
        for i in range(tl.atomic_add(c_ptr, 1)):
            s += 1
    tl.store(out_ptr + tl.program_id(0), s)


# --- D, P: helpers ----------------------------------------------------------------------------

@triton.jit
def h_default(x, e=1e-7):
    return x * e


@triton.jit
def h_default_ann(x, e: tl.constexpr = 1e-7):
    return x * e


@triton.jit
def g_half(n: tl.constexpr):
    return n // 2


@triton.jit
def k_helper(x_ptr, out_ptr, CASE: tl.constexpr):
    i = tl.arange(0, 4)
    x = tl.load(x_ptr + i)
    if CASE == 1:
        y = h_default(x)
    elif CASE == 2:
        y = h_default(x, 1e-7)
    elif CASE == 3:
        e = 1e-7
        y = h_default(x, e)
    else:
        y = h_default_ann(x)
    tl.store(out_ptr + i, y.to(tl.float32))


@triton.jit
def k_param_ann(out_ptr, CASE: tl.constexpr):
    if CASE == 1:
        a = -7
        q = g_half(a)
    else:
        q = g_half(-7)
    tl.store(out_ptr, q)


# --- L: names around a loop -------------------------------------------------------------------

@triton.jit
def k_loop(out_ptr, n, CASE: tl.constexpr):
    if CASE == 1:
        for i in range(0, n):
            t = i * 2
        tl.store(out_ptr, t)
    elif CASE == 2:
        s = 0
        for i in range(0, n):
            s += i
        tl.store(out_ptr, i)
        tl.store(out_ptr + 1, s)
    elif CASE == 3:
        s = 0
        for i in range(0, n):
            s += i
        tl.store(out_ptr, s)
    else:
        j = 0
        while j < n:
            u = j
            j += 1
        tl.store(out_ptr, u)


@triton.jit
def k_dry(c_ptr, n):
    for i in range(0, n):
        tl.atomic_add(c_ptr, 1)


@triton.jit
def k_tuple(out_ptr):
    a, b = 1, 2, 3
    tl.store(out_ptr, a)
    tl.store(out_ptr + 1, b)


# --- O: other of tl.load ----------------------------------------------------------------------

@triton.jit
def k_other(x_ptr, out_ptr):
    i = tl.arange(0, 4)
    v = tl.load(x_ptr + i, mask=i < 1, other=1 + 2**-11 + 2**-30)
    tl.store(out_ptr + i, v)


# --- S: cumsum's dtype= -----------------------------------------------------------------------

@triton.jit
def k_cumsum(x_ptr, out_ptr, CASE: tl.constexpr):
    i = tl.arange(0, 4)
    x = tl.load(x_ptr + i)
    if CASE == 1:
        y = tl.cumsum(x, 0, dtype=tl.bfloat16)
        tl.static_assert(y.dtype == tl.bfloat16)
    elif CASE == 2:
        y = tl.cumsum(x, 0)
        tl.static_assert(y.dtype == tl.float32)
    elif CASE == 3:
        y = tl.cumsum(x, 0, dtype=tl.int8)
        tl.static_assert(y.dtype == tl.int8)
    elif CASE == 4:
        y = tl.cumsum(x, 0)
        tl.static_assert(y.dtype == tl.int32)
    elif CASE == 5:
        y = tl.cumsum(x, 0, dtype=tl.float32)
        tl.static_assert(y.dtype == tl.float32)
    else:
        y = tl.cumsum(x, 0)
        tl.static_assert(y.dtype == tl.float16)
    tl.store(out_ptr + i, y.to(tl.float32))


CASES = {
    "W1": "tl.where(keep, x, 0.0), x int32 [16777217, 3, -5, 7], keep [1, 0, 1, 1]: dtype float32 (static_assert); stored to float32",
    "W1i": "W1's result stored into an int32 buffer",
    "W2": "tl.where(keep, x, 0), x int32: dtype int32 (static_assert)",
    "W3": "tl.where(m, 0, -1.0e6), both values Python numbers: dtype float32 (static_assert)",
    "W4": "tl.where(keep, x, 0.0), x bfloat16 [1.5, 2, 3, 4]: dtype bfloat16 (static_assert)",
    "W5": "tl.where(keep, x, 0.0), x float16 [1.5, 2, 3, 4]: dtype float16 (static_assert)",
    "C1": "c: tl.constexpr = -7; c // 2",
    "C2": "c = 1; then c: tl.constexpr = -7",
    "C3": "c: tl.constexpr = -7; c += 0; c // 2",
    "C4": "BLOCK = BLOCK * 1 (BLOCK a tl.constexpr parameter); tl.arange(0, BLOCK)",
    "C5": "bare e: tl.constexpr; e = 1e-7; x * e, x bfloat16 [1, 2, 3, 1000]",
    "C5c": "e = 1e-7; x * e, x bfloat16 (control: no annotation)",
    "C6": "a: tl.int64 = 2**31 - 1; a + 1",
    "C7": "BM: tl.constexpr = x.shape[0]; tl.arange(0, BM)",
    "C8": "P: tl.constexpr = A == 1 and B == 2 (A = 1, B = 2); if P: store 1 else: tl.static_assert(False)",
    "C9": "for j in tl.static_range(2): b: tl.constexpr = 4; store b",
    "R1": "for i in range(tl.atomic_add(c, 1) * 0), 8 programs: atomics run",
    "R2": "for i in range(0, tl.atomic_add(c, 1) * 0), 8 programs: atomics run",
    "R3": "for i in tl.range(tl.atomic_add(c, 1) * 0), 8 programs: atomics run",
    "R4": "for i in range(tl.atomic_add(c, 1)), one program, c = 5 before: trips; c after",
    "D1": "h(x) with def h(x, e=1e-7): x * e, x bfloat16 [1, 2, 3, 1000]",
    "D2": "h(x, 1e-7)",
    "D3": "e = 1e-7; h(x, e)",
    "D4": "h(x) with def h(x, e: tl.constexpr = 1e-7)",
    "P1": "def g(n: tl.constexpr): n // 2; a = -7; g(a)",
    "P2": "g(-7)",
    "L1": "for i in range(0, n): t = i * 2; store t after the loop (n = 4)",
    "L2": "s = 0; for i in range(0, n): s += i; store i after the loop (n = 4)",
    "L3": "s = 0; for i in range(0, n): s += i; store s after the loop (n = 4)",
    "L4": "j = 0; while j < n: u = j; j += 1; store u after the loop (n = 4)",
    "Z1": "for i in range(0, n): tl.atomic_add(c, 1), n = 0, 8 programs: atomics run",
    "Z2": "the same with n = 1",
    "U1": "a, b = 1, 2, 3; store a and b",
    "O1": "tl.load(p + i, mask=i < 1, other=1 + 2**-11 + 2**-30), p float16 [2, 2, 2, 2]: the bits",
    "S1": "tl.cumsum(x, 0, dtype=tl.bfloat16), x bfloat16 [1, 2**-8, 2**-8, 2**-8]",
    "S1d": "tl.cumsum(x, 0), the same x (default)",
    "S2": "tl.cumsum(x, 0, dtype=tl.int8), x int8 [100, 100, 100, 100]",
    "S2d": "tl.cumsum(x, 0), the same x (default)",
    "S3": "tl.cumsum(x, 0, dtype=tl.float32), x float16 [2048, 1, 1, 1]",
    "S3d": "tl.cumsum(x, 0), the same x (default)",
}


def bits16(t):
    return [f"0x{int(v) & 0xffff:04x}" for v in t.view(torch.int16).tolist()]


def run_case(name):
    dev = "cuda"
    keep = torch.tensor([1, 0, 1, 1], device=dev, dtype=torch.int32)
    if name in ("W1", "W1i", "W2", "W3"):
        x = torch.tensor([16777217, 3, -5, 7], device=dev, dtype=torch.int32)
        if name == "W1i" or name == "W2":
            out = torch.zeros(4, device=dev, dtype=torch.int32)
        else:
            out = torch.zeros(4, device=dev, dtype=torch.float32)
        k_where[(1,)](x, keep, out, CASE={"W1": 1, "W1i": 1, "W2": 2, "W3": 3}[name])
        torch.cuda.synchronize()
        return "the static_assert held; stored " + str(out.tolist())
    if name in ("W4", "W5"):
        dt = torch.bfloat16 if name == "W4" else torch.float16
        x = torch.tensor([1.5, 2.0, 3.0, 4.0], device=dev, dtype=dt)
        out = torch.zeros(4, device=dev, dtype=torch.float32)
        k_where_f[(1,)](x, keep, out, CASE=4 if name == "W4" else 5)
        torch.cuda.synchronize()
        return "the static_assert held; stored " + str(out.tolist())
    if name in ("C1", "C2", "C3", "C6"):
        out = torch.zeros(1, device=dev, dtype=torch.int32)
        k_ann_int[(1,)](out, CASE={"C1": 1, "C2": 2, "C3": 3, "C6": 6}[name])
        torch.cuda.synchronize()
        return "stored " + str(out.item())
    if name in ("C4", "C7"):
        out = torch.zeros(8, device=dev, dtype=torch.int32)
        k_ann_block[(1,)](out, BLOCK=8, CASE=4 if name == "C4" else 7)
        torch.cuda.synchronize()
        return "compiled and ran; stored " + str(out.tolist())
    if name in ("C5", "C5c"):
        x = torch.tensor([1.0, 2.0, 3.0, 1000.0], device=dev, dtype=torch.bfloat16)
        out = torch.zeros(4, device=dev, dtype=torch.float32)
        k_ann_bare[(1,)](x, out, CASE=5 if name == "C5" else 0)
        torch.cuda.synchronize()
        return "stored " + str(out.tolist())
    if name == "C9":
        out = torch.zeros(2, device=dev, dtype=torch.int32)
        k_ann_static[(1,)](out)
        torch.cuda.synchronize()
        return "compiled and ran; stored " + str(out.tolist())
    if name in ("Z1", "Z2"):
        c = torch.zeros(1, device=dev, dtype=torch.int32)
        k_dry[(8,)](c, 0 if name == "Z1" else 1)
        torch.cuda.synchronize()
        return f"atomics: {c.item()} in 8 programs"
    if name == "U1":
        out = torch.full((2,), -99, device=dev, dtype=torch.int32)
        k_tuple[(1,)](out)
        torch.cuda.synchronize()
        return "compiled and ran; stored " + str(out.tolist())
    if name == "C8":
        out = torch.zeros(1, device=dev, dtype=torch.int32)
        k_ann_bool[(1,)](out, A=1, B=2)
        torch.cuda.synchronize()
        return "compiled and ran; stored " + str(out.item())
    if name in ("R1", "R2", "R3"):
        c = torch.zeros(1, device=dev, dtype=torch.int32)
        out = torch.full((8,), -1, device=dev, dtype=torch.int32)
        k_range[(8,)](c, out, CASE=int(name[1]))
        torch.cuda.synchronize()
        return f"atomics: {c.item()} in 8 programs; trips {out.tolist()}"
    if name == "R4":
        c = torch.full((1,), 5, device=dev, dtype=torch.int32)
        out = torch.full((1,), -1, device=dev, dtype=torch.int32)
        k_range[(1,)](c, out, CASE=4)
        torch.cuda.synchronize()
        return f"trips {out.item()}; c after: {c.item()}"
    if name in ("D1", "D2", "D3", "D4"):
        x = torch.tensor([1.0, 2.0, 3.0, 1000.0], device=dev, dtype=torch.bfloat16)
        out = torch.zeros(4, device=dev, dtype=torch.float32)
        k_helper[(1,)](x, out, CASE=int(name[1]))
        torch.cuda.synchronize()
        return "stored " + str(out.tolist())
    if name in ("P1", "P2"):
        out = torch.zeros(1, device=dev, dtype=torch.int32)
        k_param_ann[(1,)](out, CASE=int(name[1]))
        torch.cuda.synchronize()
        return "stored " + str(out.item())
    if name in ("L1", "L2", "L3", "L4"):
        out = torch.full((2,), -99, device=dev, dtype=torch.int32)
        k_loop[(1,)](out, 4, CASE=int(name[1]))
        torch.cuda.synchronize()
        return "compiled and ran; stored " + str(out.tolist())
    if name == "O1":
        x = torch.full((4,), 2.0, device=dev, dtype=torch.float16)
        out = torch.zeros(4, device=dev, dtype=torch.float16)
        k_other[(1,)](x, out)
        torch.cuda.synchronize()
        c = 1 + 2**-11 + 2**-30
        direct = struct.unpack("<H", struct.pack("<e", c))[0]
        via32 = struct.unpack("<H", struct.pack("<e", struct.unpack("<f", struct.pack("<f", c))[0]))[0]
        return (f"stored bits {bits16(out)}; the float16 nearest to the double: 0x{direct:04x}; "
                f"the float16 nearest to its float32 (a tie, to even): 0x{via32:04x}")
    if name.startswith("S"):
        if name.startswith("S1"):
            x = torch.tensor([1.0, 2**-8, 2**-8, 2**-8], device=dev, dtype=torch.bfloat16)
            case = 1 if name == "S1" else 2
        elif name.startswith("S2"):
            x = torch.tensor([100, 100, 100, 100], device=dev, dtype=torch.int8)
            case = 3 if name == "S2" else 4
        else:
            x = torch.tensor([2048.0, 1.0, 1.0, 1.0], device=dev, dtype=torch.float16)
            case = 5 if name == "S3" else 6
        out = torch.zeros(4, device=dev, dtype=torch.float32)
        k_cumsum[(1,)](x, out, CASE=case)
        torch.cuda.synchronize()
        return "the static_assert on the dtype held; stored " + str(out.tolist())
    raise SystemExit("unknown case " + name)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        try:
            res = run_case(sys.argv[2])
        except Exception as e:
            msg = [l for l in str(e).strip().splitlines() if l.strip()]
            res = "rejected on the host: " + type(e).__name__ + ": " + (msg[-1].strip() if msg else "")[:160]
        print("RESULT " + json.dumps(res))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name, desc in CASES.items():
        with tempfile.TemporaryDirectory() as cache:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", TRITON_CACHE_DIR=cache)
            p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                               capture_output=True, text=True, env=env)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line:
            res = json.loads(line[len("RESULT "):])
        else:
            err = [l for l in p.stderr.strip().splitlines() if l.strip()]
            res = "process failed: " + (err[-1][:160] if err else "(no output)")
        print(f"{name}: {desc}\n  {res}")


if __name__ == "__main__":
    main()

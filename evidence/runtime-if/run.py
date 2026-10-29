"""What does a runtime `if`, `elif`, conditional expression, `and`/`or` or early `return` do?

Library evidence for library issue L28 (round 8). Each case compiles and launches small kernels
with Triton 3.8.0 on GPU 0, in its own process (a fault ends the process's CUDA context), and
reports:
  - per program, what ran: stores into a buffer filled with a sentinel, and atomic counters;
  - markers of the front end's output (`asm['source']`, before any pass): `scf.if`, `cf.cond_br`,
    the comparison a condition became;
  - for a rejected kernel, the exception's type, its message, and whether anything ran (the
    sentinel buffer is checked after the error).

Run: .venv/bin/python -I evidence/runtime-if/run.py  (from TVA's root; CUDA_VISIBLE_DEVICES=0)
"""
import json, os, re, subprocess, sys, warnings

import torch
import triton
import triton.language as tl

DEV = "cuda"
SENT = -777
NPROG = 64


# ---------------------------------------------------------------- kernels

@triton.jit
def r1(out, cnt):
    pid = tl.program_id(0)
    if pid % 3 == 0:
        tl.store(out + pid, 1000 + pid)
        tl.atomic_add(cnt + 0, 1)
    else:
        tl.store(out + pid, 2000 + pid)
        tl.atomic_add(cnt + 1, 1)


@triton.jit(do_not_specialize=["flag"])
def r2a(cnt, flag):
    if flag == 7:
        tl.atomic_add(cnt, 1)


@triton.jit
def r2b(x, cnt):
    pid = tl.program_id(0)
    v = tl.load(x + pid)
    if v > 0:
        tl.atomic_add(cnt, 1)


@triton.jit(do_not_specialize=["flag"])
def r2c(x, flag, far):
    if flag == 7:
        tl.store(x + far, 1)        # 2**40 elements past x: faults if it runs


@triton.jit(do_not_specialize=["n"])
def r3_int(out, n):
    pid = tl.program_id(0)
    if n:
        tl.store(out + pid, 1)
    else:
        tl.store(out + pid, 0)


@triton.jit
def r3_float(out, x):
    pid = tl.program_id(0)
    if x:
        tl.store(out + pid, 1)
    else:
        tl.store(out + pid, 0)


@triton.jit
def r4_block4(x, out):
    v = tl.load(x + tl.arange(0, 4))
    if v > 0:
        tl.store(out, 1)


@triton.jit
def r4_block1(x, out):
    v = tl.load(x + tl.arange(0, 1))
    if v > 0:
        tl.store(out, 1)
    else:
        tl.store(out, 0)


@triton.jit
def r5_only_then(out, n):
    if n > 0:
        y = n + 1
    tl.store(out, y)


@triton.jit
def r5_livein_dtype(out, n):
    y = n + 1
    if n > 0:
        y = 1.5
    tl.store(out, y)


@triton.jit
def r5_both_dtype(out, n):
    if n > 0:
        y = n + 1
    else:
        y = 1.5
    tl.store(out, y)


@triton.jit
def r5_both_shape(out, n):
    if n > 0:
        y = tl.zeros([4], dtype=tl.int32)
    else:
        y = tl.zeros([8], dtype=tl.int32)
    tl.store(out + tl.arange(0, 4), tl.sum(y))


@triton.jit
def r5_merge(out):
    pid = tl.program_id(0)
    y = pid * 10            # live in, changed in one branch only
    if pid % 2 == 0:
        y = pid * 100
        z = 1
    else:
        z = 2               # defined in both branches, same type
    tl.store(out + pid, y + z)


@triton.jit
def r6_return_then(out, cnt):
    pid = tl.program_id(0)
    if pid % 2 == 1:
        return
    tl.store(out + pid, 1)
    tl.atomic_add(cnt, 1)


@triton.jit
def r6_return_else(cnt):
    pid = tl.program_id(0)
    if pid < 10:
        tl.atomic_add(cnt + 0, 1)
    else:
        tl.atomic_add(cnt + 1, 1)
        return
    tl.atomic_add(cnt + 2, 1)


@triton.jit
def r6_return_top(cnt):
    tl.atomic_add(cnt + 0, 1)
    return
    tl.atomic_add(cnt + 1, 1)


@triton.jit
def r6_return_value(out):
    pid = tl.program_id(0)
    tl.store(out + pid, 1)
    return pid


@triton.jit
def r7_nested_if(cnt):
    pid = tl.program_id(0)
    if pid < 32:
        tl.atomic_add(cnt + 0, 1)
        if pid % 2 == 0:
            return
        tl.atomic_add(cnt + 1, 1)
    tl.atomic_add(cnt + 2, 1)


@triton.jit
def r7_in_for(cnt, n):
    pid = tl.program_id(0)
    for i in range(0, n):
        if pid == i:
            return
        tl.atomic_add(cnt, 1)


@triton.jit
def r7_in_while(cnt, n):
    pid = tl.program_id(0)
    i = 0
    while i < n:
        if pid == i:
            return
        tl.atomic_add(cnt, 1)
        i += 1


@triton.jit
def r7_bare_in_for(cnt, n):
    for i in range(0, n):
        tl.atomic_add(cnt, 1)
        return


@triton.jit
def r7_static_range(cnt):
    pid = tl.program_id(0)
    for i in tl.static_range(4):
        if pid == i:
            return
        tl.atomic_add(cnt + i, 1)


@triton.jit
def helper_early(x):
    if x > 5:
        return x * 10
    return x + 1


@triton.jit
def helper_loop_return(x, n):
    for i in range(0, n):
        if x == i:
            return i
    return -1


@triton.jit
def r7_helper(out):
    pid = tl.program_id(0)
    tl.store(out + pid, helper_early(pid))


@triton.jit
def r7_helper_in_for(out, n):
    pid = tl.program_id(0)
    acc = 0
    for i in range(0, n):
        acc += helper_early(pid + i)
    tl.store(out + pid, acc)


@triton.jit
def r7_helper_loop_return(out, n):
    pid = tl.program_id(0)
    tl.store(out + pid, helper_loop_return(pid, n))


@triton.jit
def r8_ifexp(a, b, out):
    pid = tl.program_id(0)
    v = tl.atomic_add(a, 1) if pid % 2 == 0 else tl.atomic_add(b, 1)
    tl.store(out + pid, v)


@triton.jit
def r8_ifexp_block(x, out):
    offs = tl.arange(0, 4)
    v = tl.load(x + offs)
    w = v if offs < 2 else -v
    tl.store(out + offs, w)


@triton.jit
def r8_ifexp_types(out, n):
    y = 1 if n > 0 else 2.0
    tl.store(out, y)


@triton.jit
def r9_elif(out, c):
    pid = tl.program_id(0)
    if pid < 2:
        tl.store(out + pid, 1)
    elif tl.atomic_add(c, 1) < 3:      # evaluated only by programs with pid >= 2
        tl.store(out + pid, 2)
    elif pid % 2 == 0:
        tl.store(out + pid, 3)
    else:
        tl.store(out + pid, 4)


@triton.jit
def r9_and(a, b, cnt):
    if (tl.atomic_add(a, 1) < 0) and (tl.atomic_add(b, 1) >= 0):
        tl.atomic_add(cnt, 1)


@triton.jit
def r9_or(a, b, cnt):
    if (tl.atomic_add(a, 1) >= 0) or (tl.atomic_add(b, 1) >= 0):
        tl.atomic_add(cnt, 1)


@triton.jit
def r9_and_const(a, cnt, FLAG: tl.constexpr):
    if (tl.atomic_add(a, 1) >= 0) and FLAG:
        tl.atomic_add(cnt, 1)


@triton.jit(do_not_specialize=["n", "m"])
def r9_and_int(out, n, m):
    if n and m:
        tl.store(out, 1)


@triton.jit(do_not_specialize=["n"])
def r9_not_int(out, n):
    if not n:
        tl.store(out, 1)


@triton.jit
def r9_not_bool(out):
    pid = tl.program_id(0)
    if not (pid % 2 == 0):
        tl.store(out + pid, 1)
    else:
        tl.store(out + pid, 0)


# ---------------------------------------------------------------- helpers

def buf(n, dtype=torch.int32, fill=SENT):
    return torch.full((n,), fill, device=DEV, dtype=dtype)


def zeros(n):
    return torch.zeros(n, device=DEV, dtype=torch.int32)


MARKERS = [("scf.if", r"\bscf\.if\b"), ("cf.cond_br", r"\bcf\.cond_br\b"),
           ("cmpi ne", r"arith\.cmpi ne"), ("cmpf une", r"arith\.cmpf une"),
           ("atomic_rmw", r"tt\.atomic_rmw"), ("tt.call", r"tt\.call\b")]


def markers(ck):
    src = ck.asm["source"]
    out = []
    for name, pat in MARKERS:
        k = len(re.findall(pat, src))
        if k:
            out.append(f"{name} x{k}")
    return ", ".join(out) or "-"


def err_text(e):
    """The cause of an error. A CompilationError's text is a source excerpt, then the cause on its
    last line; a nested one (an error in a helper) is unwrapped to the innermost. Other errors: the
    first line."""
    from triton.compiler.errors import CompilationError
    while isinstance(e, CompilationError) and isinstance(e.__cause__, CompilationError):
        e = e.__cause__
    lines = [l.strip() for l in str(e).splitlines() if l.strip()]
    if not lines:
        return type(e).__name__
    if isinstance(e, CompilationError):
        tail = [l for l in lines if not l.startswith("^")]
        return f"{type(e).__name__}: {tail[-1][:200]}"
    return f"{type(e).__name__}: {lines[0][:200]}"


def with_stderr(fn):
    """Run fn with the process's file descriptor 2 sent to a file; return (result, MLIR diagnostics)."""
    import tempfile
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile(mode="w+") as f:
        os.dup2(f.fileno(), 2)
        try:
            res = fn()
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
        f.seek(0)
        diag = [l.strip().replace(os.path.abspath(__file__), "evidence/runtime-if/run.py")
                for l in f.read().splitlines() if "error:" in l]
    return res, diag


def rejected(launch, watch):
    """Run `launch`; on an exception, report it, the compiler's diagnostics, and whether the
    watched buffers changed."""
    def go():
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            try:
                launch()
                torch.cuda.synchronize()
                return "accepted (no error)", ws
            except Exception as e:
                torch.cuda.synchronize()
                untouched = all(bool((w == SENT).all()) for w in watch)
                return f"rejected on the host: {err_text(e)}; buffers untouched: {untouched}", ws
    (res, ws), diag = with_stderr(go)
    if diag:
        res += "; compiler diagnostic: " + " | ".join(d[:200] for d in diag[:2])
    return res, ws


def warns(ws):
    return [str(w.message)[:160] for w in ws if "deprecated" in str(w.message) or "conditional" in str(w.message)]


# ---------------------------------------------------------------- cases

def case_R1():
    out, cnt = buf(NPROG), zeros(2)
    ck = r1[(NPROG,)](out, cnt)
    torch.cuda.synchronize()
    exp = [1000 + p if p % 3 == 0 else 2000 + p for p in range(NPROG)]
    ok = out.tolist() == exp
    return [f"IR: {markers(ck)}",
            f"stores as the taken branch only: {ok}; then-count {cnt[0].item()} (expected 22), else-count {cnt[1].item()} (expected 42)"]


def case_R2():
    lines = []
    for flag in (0, 7):
        cnt = zeros(1)
        ck = r2a[(NPROG,)](cnt, flag)
        torch.cuda.synchronize()
        lines.append(f"flag={flag}: atomic count {cnt.item()} (expected {NPROG if flag == 7 else 0}); IR: {markers(ck)}")
    x = torch.tensor([1 if p in (3, 9, 10, 40, 63) else -p for p in range(NPROG)], device=DEV, dtype=torch.int32)
    cnt = zeros(1)
    r2b[(NPROG,)](x, cnt)
    torch.cuda.synchronize()
    lines.append(f"loaded condition: atomic count {cnt.item()} (expected 5)")
    x = zeros(16)
    r2c[(NPROG,)](x, 0, 2**40)
    torch.cuda.synchronize()
    lines.append("untaken branch stores 2**40 elements out of bounds: no fault (flag=0)")
    return lines


def case_R2_fault():
    x = zeros(16)
    r2c[(NPROG,)](x, 7, 2**40)
    torch.cuda.synchronize()
    return ["taken branch with the same store: no fault (unexpected)"]


def case_R3():
    lines = []
    for n in (0, 2, 256, -1, 2**32, -2**40):
        out = buf(4)
        ck = r3_int[(4,)](out, n)
        torch.cuda.synchronize()
        sig = re.search(r"%n: (\w+)", ck.asm["source"]).group(1)
        lines.append(f"int n={n} ({sig}): taken={out.tolist() == [1] * 4}; IR: {markers(ck)}")
    for x in (0.0, -0.0, float("nan"), 0.5, 1e-45, float("-inf")):
        out = buf(4)
        ck = r3_float[(4,)](out, x)
        torch.cuda.synchronize()
        lines.append(f"float x={x!r}: taken={out.tolist() == [1] * 4}; IR: {markers(ck)}")
    return lines


def case_R4():
    lines = []
    x = torch.tensor([1, 2, 3, 4], device=DEV, dtype=torch.int32)
    out = buf(1)
    res, ws = rejected(lambda: r4_block4[(1,)](x, out), [out])
    lines.append("block of 4 as the condition: " + res)
    for v in (5, -5):
        x = torch.tensor([v], device=DEV, dtype=torch.int32)
        out = buf(1)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            ck = r4_block1[(1,)](x, out)
            torch.cuda.synchronize()
        lines.append(f"block of 1 ({v}): branch {'then' if out.item() == 1 else 'else'}; warnings at compile time: {warns(ws) or '(none: same kernel, already compiled)'}; IR: {markers(ck)}")
    return lines


def case_R5():
    lines = []
    for name, k in (("name assigned only in the then branch, used after", r5_only_then),
                    ("live-in name redefined with another dtype in one branch", r5_livein_dtype),
                    ("name defined in both branches with different dtypes", r5_both_dtype),
                    ("name defined in both branches with different shapes", r5_both_shape)):
        out = buf(4) if k is r5_both_shape else buf(1, torch.float32)
        res, _ = rejected(lambda: k[(1,)](out, 3), [out])
        lines.append(f"{name}: {res}")
    out = buf(NPROG)
    ck = r5_merge[(NPROG,)](out)
    torch.cuda.synchronize()
    exp = [p * 100 + 1 if p % 2 == 0 else p * 10 + 2 for p in range(NPROG)]
    lines.append(f"live-in changed in one branch, new name in both: merged correctly: {out.tolist() == exp}; IR: {markers(ck)}")
    return lines


def case_R6():
    lines = []
    out, cnt = buf(NPROG), zeros(1)
    ck = r6_return_then[(NPROG,)](out, cnt)
    torch.cuda.synchronize()
    exp = [1 if p % 2 == 0 else SENT for p in range(NPROG)]
    lines.append(f"return in then: programs that return ran nothing after it: {out.tolist() == exp}; count {cnt.item()} (expected 32); IR: {markers(ck)}")
    cnt = zeros(3)
    ck = r6_return_else[(NPROG,)](cnt)
    torch.cuda.synchronize()
    lines.append(f"return in else: counts {cnt.tolist()} (expected [10, 54, 10]); IR: {markers(ck)}")
    cnt = zeros(2)
    ck = r6_return_top[(NPROG,)](cnt)
    torch.cuda.synchronize()
    lines.append(f"return at top level: counts {cnt.tolist()} (expected [64, 0]); IR: {markers(ck)}")
    out = buf(NPROG)
    res, _ = rejected(lambda: r6_return_value[(NPROG,)](out), [out])
    lines.append("kernel returning a value: " + res)
    return lines


def case_R7():
    lines = []
    cnt = zeros(3)
    ck = r7_nested_if[(NPROG,)](cnt)
    torch.cuda.synchronize()
    lines.append(f"return in an if nested in a runtime if: counts {cnt.tolist()} (expected [32, 16, 48]); IR: {markers(ck)}")
    for name, k in (("return in a runtime if in a for loop", r7_in_for),
                    ("return in a runtime if in a while loop", r7_in_while),
                    ("return directly in a for body", r7_bare_in_for)):
        cnt = buf(1)
        res, _ = rejected(lambda: k[(NPROG,)](cnt, 4), [cnt])
        lines.append(f"{name}: {res}")
    cnt = zeros(4)
    ck = r7_static_range[(8,)](cnt)
    torch.cuda.synchronize()
    lines.append(f"return in a runtime if in tl.static_range: counts {cnt.tolist()} (expected [7, 6, 5, 4]); IR: {markers(ck)}")
    out = buf(NPROG)
    ck = r7_helper[(NPROG,)](out)
    torch.cuda.synchronize()
    exp = [p * 10 if p > 5 else p + 1 for p in range(NPROG)]
    lines.append(f"helper with an early return, called at top level: correct {out.tolist() == exp}; IR: {markers(ck)}")
    out = buf(NPROG)
    res, _ = rejected(lambda: r7_helper_in_for[(NPROG,)](out, 3), [out])
    if res.startswith("accepted"):
        exp = [sum((p + i) * 10 if p + i > 5 else p + i + 1 for i in range(3)) for p in range(NPROG)]
        ck = r7_helper_in_for[(NPROG,)](out, 3)
        torch.cuda.synchronize()
        calls = len(re.findall(r"tt\.call\b", ck.asm["ttir"]))
        res += f"; correct {out.tolist() == exp}; IR: {markers(ck)}; tt.call after the passes (ttir): {calls}"
    lines.append(f"the same helper called in a for loop: {res}")
    out = buf(NPROG)
    res, _ = rejected(lambda: r7_helper_loop_return[(NPROG,)](out, 4), [out])
    lines.append(f"helper with a return in a runtime if in its own for loop: {res}")
    return lines


def case_R8():
    lines = []
    a, b, out = zeros(1), zeros(1), buf(NPROG)
    ck = r8_ifexp[(NPROG,)](a, b, out)
    torch.cuda.synchronize()
    lines.append(f"x if c else y with atomics on both sides: a={a.item()}, b={b.item()} (expected 32, 32); IR: {markers(ck)}")
    x = torch.tensor([1, 2, 3, 4], device=DEV, dtype=torch.int32)
    out = buf(4)
    res, _ = rejected(lambda: r8_ifexp_block[(1,)](x, out), [out])
    lines.append(f"x if c else y with a block condition: {res}")
    out = buf(1, torch.float32)
    res, _ = rejected(lambda: r8_ifexp_types[(1,)](out, 3), [out])
    lines.append(f"x if c else y with an int32 and an fp32 side: {res}")
    return lines


def case_R9():
    lines = []
    out, c = buf(NPROG), zeros(1)
    ck = r9_elif[(NPROG,)](out, c)
    torch.cuda.synchronize()
    o = out.tolist()
    lines.append(f"elif chain: programs 0,1 -> {o[:2]}; elif condition's atomic ran {c.item()} times (expected 62); "
                 f"values 2: {o.count(2)} (expected 3), 3+4: {o.count(3) + o.count(4)} (expected 59); IR: {markers(ck)}")
    for name, k, exp in (("and", r9_and, 0), ("or", r9_or, NPROG)):
        a, b, cnt = zeros(1), zeros(1), zeros(1)
        ck = k[(NPROG,)](a, b, cnt)
        torch.cuda.synchronize()
        lines.append(f"`{name}` of two runtime conditions with an atomic in each: a={a.item()}, b={b.item()}, "
                     f"branch taken {cnt.item()} (expected {exp}); IR: {markers(ck)}")
    a, cnt = zeros(1), zeros(1)
    ck = r9_and_const[(NPROG,)](a, cnt, FLAG=False)
    torch.cuda.synchronize()
    lines.append(f"`<runtime with atomic> and False`: atomic ran {a.item()}, branch taken {cnt.item()}; IR: {markers(ck)}")
    out = buf(1)
    res, _ = rejected(lambda: r9_and_int[(1,)](out, 2, 3), [out])
    lines.append(f"`n and m` on int32 scalars: {res}")
    out = buf(1)
    res, _ = rejected(lambda: r9_not_int[(1,)](out, 2), [out])
    lines.append(f"`not n` on an int32 scalar: {res}")
    out = buf(NPROG)
    ck = r9_not_bool[(NPROG,)](out)
    torch.cuda.synchronize()
    lines.append(f"`not c` on an int1 scalar: correct {out.tolist() == [p % 2 for p in range(NPROG)]}; IR: {markers(ck)}")
    return lines


CASES = {
    "R1": ("runtime if/else on an int1 scalar, stores and atomics in both branches", case_R1),
    "R2": ("an atomic or an out-of-bounds store only in the untaken branch", case_R2),
    "R2f": ("control for R2: the out-of-bounds store in the taken branch", case_R2_fault),
    "R3": ("if on an int32/int64 or fp32 runtime scalar", case_R3),
    "R4": ("if on a block (tensor) condition", case_R4),
    "R5": ("names defined in one or both branches, with different types", case_R5),
    "R6": ("early return in a runtime if at the kernel's top level", case_R6),
    "R7": ("return nested in ifs, in loops, in static_range, and in helpers", case_R7),
    "R8": ("conditional expression x if c else y with a runtime condition", case_R8),
    "R9": ("elif chains; and/or/not on runtime scalars", case_R9),
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
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    cache = tempfile.mkdtemp()     # a fresh cache: every kernel is compiled, so compile-time warnings show
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

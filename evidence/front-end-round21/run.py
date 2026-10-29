"""TVA library evidence (library issues L102, L111): a helper's `return`s of equal compile-time values that are not
the same value, and arithmetic computed in an fp8 dtype.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/front-end-round21/run.py
Output: evidence/front-end-round21/out.txt (README.md says what it shows).

Each kernel is written to a file of its own, in a temporary folder, and imported from it (`@triton.jit` reads its
function's source file); the process uses a fresh Triton cache directory. No path is printed.

T1-T4 (L102) a helper with a runtime `if c > 0:` and two `return`s of compile-time values that Python's `==` finds
    equal, called with the runtime `int32` `c = 5` (the `if` taken) and `c = -5` (not taken; never `1`, which
    Triton passes as a compile-time constant):
    T1 `return (1, 2)`, then `return (1.0, 2)`; the kernel stores `x * th(c)[0]` for the `int32`
       `x = 2^24 + 1` (the integer product `16777217` for `1`, the `float32` product `16777216` for `1.0`);
    T2 the same with the two `return`s in the other order;
    T3 `return (1, 2)`, then `a, b = th(c)` and `x * a`;
    T4 `return 0.0`, then `return -0.0`; the kernel stores `2` when the f-string `f"{h(c)}"` is `"-0.0"`, else
       `1` (a compile-time `if`).
F (L111) every form below on an fp8 tensor `x` of each fp8 dtype (`tl.float8e5`, `tl.float8e4nv`) loaded by bit cast
    from all 256 patterns, the result stored by bit cast: what is stored, or the host's rejection. The forms are
    arithmetic whose computation dtype is the fp8 dtype (a Python number of a kind no higher than the tensor's
    takes the tensor's dtype), unary `-`, a comparison, and arithmetic on two fp8 constants.
"""
import importlib.util
import os
import sys
import tempfile

HELPERS = {
    "T1": ("""@triton.jit
def th(c):
    if c > 0:
        return (1, 2)
    return (1.0, 2)


@triton.jit
def k(o_ptr, x_ptr, c):
    i = tl.arange(0, 1)
    tl.store(o_ptr + i, tl.load(x_ptr + i) * th(c)[0])
"""),
    "T2": ("""@triton.jit
def th(c):
    if c > 0:
        return (1.0, 2)
    return (1, 2)


@triton.jit
def k(o_ptr, x_ptr, c):
    i = tl.arange(0, 1)
    tl.store(o_ptr + i, tl.load(x_ptr + i) * th(c)[0])
"""),
    "T3": ("""@triton.jit
def th(c):
    if c > 0:
        return (1, 2)
    return (1.0, 2)


@triton.jit
def k(o_ptr, x_ptr, c):
    i = tl.arange(0, 1)
    a, b = th(c)
    tl.store(o_ptr + i, tl.load(x_ptr + i) * a)
"""),
    "T4": ("""@triton.jit
def h(c):
    if c > 0:
        return 0.0
    return -0.0


@triton.jit
def k(o_ptr, x_ptr, c):
    i = tl.arange(0, 1)
    if f"{h(c)}" == "-0.0":
        tl.store(o_ptr + i, tl.full([1], 2, tl.int32))
    else:
        tl.store(o_ptr + i, tl.full([1], 1, tl.int32))
"""),
}
PYTHON_READING = {"T1": {5: 16777217, -5: 16777216}, "T2": {5: 16777216, -5: 16777217},
                  "T3": {5: 16777217, -5: 16777216}, "T4": {5: 1, -5: 2}}

FP8_FORMS = ["x * 1.0", "1.0 * x", "x * 1", "1 * x", "x / 1.0", "x / 1", "x - 0.0", "x - 0",
             "x + 0.0", "x * 2.0", "x * -1.0", "x / 2.0", "x + x", "x - x", "x * x", "-x",
             "tl.where(x < x, 1, 0)",
             "tl.full([256], 1.5, DT) * tl.full([256], 2.0, DT)", "tl.full([256], 1.5, DT) + 0.5",
             "tl.full([256], 1.5, DT) - tl.full([256], 0.5, DT)"]


def load(tmp, name, body):
    path = os.path.join(tmp, name + ".py")
    with open(path, "w") as fh:
        fh.write("import triton\nimport triton.language as tl\n\n\n" + body)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    tmp = tempfile.mkdtemp(prefix="tva-r21-fe-")
    os.environ["TRITON_CACHE_DIR"] = os.path.join(tmp, "cache")
    import torch
    import triton
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"python {sys.version.split()[0]}")

    print("\nT. a helper's returns of equal compile-time values (runtime c)")
    wrong = 0
    for case, body in HELPERS.items():
        mod = load(tmp, f"fe_r21_{case.lower()}", body)
        got = {}
        for c in (5, -5):
            x = torch.full((1,), 2 ** 24 + 1, dtype=torch.int32, device="cuda")
            o = torch.zeros(1, dtype=torch.int32, device="cuda")
            mod.k[(1,)](o, x, c)
            torch.cuda.synchronize()
            got[c] = int(o.item())
        diff = [c for c in (5, -5) if got[c] != PYTHON_READING[case][c]]
        wrong += len(diff)
        print(f"{case}: c = 5 -> {got[5]}, c = -5 -> {got[-5]}; the value of the return taken gives "
              f"{PYTHON_READING[case][5]}, {PYTHON_READING[case][-5]}; differ: {diff or 'none'}")
    print(f"T launches that differ from the value of the return taken: {wrong}")

    print("\nF. arithmetic computed in an fp8 dtype (all 256 patterns of x)")
    x = torch.arange(256, dtype=torch.int32, device="cuda").to(torch.uint8)
    ran = 0
    for dt in ("tl.float8e5", "tl.float8e4nv"):
        for j, form in enumerate(FP8_FORMS):
            body = (f"@triton.jit\ndef k(x_ptr, o_ptr):\n    i = tl.arange(0, 256)\n"
                    f"    x = tl.load(x_ptr + i).to({dt}, bitcast=True)\n"
                    f"    y = {form.replace('DT', dt)}\n")
            if "where" in form:
                body += "    tl.store(o_ptr + i, y.to(tl.uint8))\n"
            else:
                body += "    tl.store(o_ptr + i, y.to(tl.uint8, bitcast=True))\n"
            mod = load(tmp, f"fe_r21_f8_{dt.split('.')[1]}_{j}", body)
            o = torch.full((256,), 0x55, dtype=torch.uint8, device="cuda")
            try:
                mod.k[(1,)](x, o)
                torch.cuda.synchronize()
            except Exception as ex:
                lines = str(ex).strip().splitlines()
                print(f"{dt:14s} {form:52s} rejected on the host ({type(ex).__name__}: {(lines[-1] if lines else '')[:60]})")
                continue
            ran += 1
            got = o.tolist()
            same = sum(1 for a, b in zip(range(256), got) if a == b)
            vals = sorted(set(got))
            print(f"{dt:14s} {form:52s} RAN: {same}/256 patterns stored unchanged; "
                  f"stored values {['%02x' % v for v in vals[:4]]}{' …' if len(vals) > 4 else ''}")
    print(f"F forms that compiled and ran: {ran} of {2 * len(FP8_FORMS)}")


if __name__ == "__main__":
    main()

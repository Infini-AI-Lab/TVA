"""lr-20f, round 20, L103: does a fresh Triton cache directory meet Compiler.Faithful's scope 11, as
REVIEWER_GUIDE item 11 and the rule page's checklist item 11 say ("A fresh cache directory, or values
Triton keys apart, meets it")? One process; the claimed launch is `C = 1.0` (the product is float32, stored back into an int32 output:
16777217 * 1.0 rounds to 16777216), after an earlier launch with `C = 1` (the integer product keeps
16777217). Case A: one fresh cache directory for the whole process. Case B: a new, empty cache
directory set just before the claimed launch. Control: `C = 1.0` alone in a fresh process.
Run from the TVA root: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I <this file>"""
import os, sys, subprocess, tempfile


def child(case):
    os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="lr20f-cache-")
    import torch, triton, triton.language as tl

    @triton.jit
    def scale(x_ptr, o_ptr, C: tl.constexpr):
        i = tl.arange(0, 1)
        tl.store(o_ptr + i, tl.load(x_ptr + i) * C)

    x = torch.tensor([16777217], dtype=torch.int32, device="cuda")
    o = torch.zeros(1, dtype=torch.int32, device="cuda")
    if case != "control":
        scale[(1,)](x, o, C=1)
        torch.cuda.synchronize()
        if case == "B":
            os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="lr20f-cache-")
    o.zero_()
    scale[(1,)](x, o, C=1.0)
    torch.cuda.synchronize()
    print("RESULT", repr(o.item()))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        child(sys.argv[1])
    else:
        import triton, torch
        print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, Python {sys.version.split()[0]}")
        for case in ("control", "A", "B"):
            r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), case], capture_output=True, text=True)
            res = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
            print(case, res[0] if res else r.stderr[-400:])
        print("the claimed launch C = 1.0 should store 16777216 (x is an int32 tensor, C a float: the product "
              "is float32, stored into int32); 16777217 means it ran the code compiled for C = 1")

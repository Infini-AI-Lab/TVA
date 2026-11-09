"""TVA library evidence (library issue L158): a `torch.bool` tensor argument.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/bool-args/run.py
Output: evidence/bool-args/out.txt (README.md says what it shows).

A `torch.bool` tensor whose bytes are 0, 1, 2, 127, 128 and 255 (written through a `uint8` view of the
same storage) is passed to kernels that (1) print the argument's type at compile time, (2) load it,
masked and unmasked, and store the loaded `int1` values converted to `int32`, (3) store `int1` values
(comparisons) into a `torch.bool` output, and (4) store `int32` values 0, 1, 2, 255, 256 and 257 into a
`torch.bool` output. The output bytes are read back through a `uint8` view.
"""
import torch
import triton
import triton.language as tl

DEV = "cuda"
OUT = []


def out(s=""):
    print(s, flush=True)
    OUT.append(s)


@triton.jit
def k_load(b_ptr, o_ptr, o2_ptr, N: tl.constexpr):
    tl.static_print("b_ptr.dtype:", b_ptr.dtype, "element_ty:", b_ptr.dtype.element_ty)
    j = tl.arange(0, N)
    x = tl.load(b_ptr + j)
    tl.static_print("loaded dtype:", x.dtype)
    tl.store(o_ptr + j, x.to(tl.int32))
    y = tl.load(b_ptr + j, mask=j < 3, other=5)
    tl.store(o2_ptr + j, y.to(tl.int32))


@triton.jit
def k_store_cmp(v_ptr, o_ptr, N: tl.constexpr):
    j = tl.arange(0, N)
    v = tl.load(v_ptr + j)
    tl.store(o_ptr + j, v > 1)


@triton.jit
def k_store_int(v_ptr, o_ptr, N: tl.constexpr):
    j = tl.arange(0, N)
    tl.store(o_ptr + j, tl.load(v_ptr + j))


def main():
    out(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    pat = [0, 1, 2, 127, 128, 255, 0, 1]
    raw = torch.tensor(pat, dtype=torch.uint8, device=DEV)
    b = raw.view(torch.bool)
    o = torch.full((8,), -1, dtype=torch.int32, device=DEV)
    o2 = torch.full((8,), -1, dtype=torch.int32, device=DEV)
    h = k_load[(1,)](b, o, o2, 8)
    torch.cuda.synchronize()
    sig = [ln for ln in h.asm["ttir"].splitlines() if "tt.func" in ln][0].strip()
    out(f"bytes in the bool tensor: {pat}")
    out(f"TTIR signature: {sig}")
    out(f"tl.load, unmasked, .to(int32): {o.tolist()}")
    out(f"tl.load, mask j < 3, other=5, .to(int32): {o2.tolist()}")
    ptx = h.asm["ptx"]
    lds = sorted({ln.split()[0] for ln in ptx.splitlines() if ln.strip().startswith("ld.global")})
    out(f"PTX loads: {lds}")
    v = torch.tensor([0, 1, 2, 3, -1, 5, 1, 2], dtype=torch.int32, device=DEV)
    ob = torch.full((8,), 7, dtype=torch.uint8, device=DEV)
    k_store_cmp[(1,)](v, ob.view(torch.bool), 8)
    torch.cuda.synchronize()
    out(f"tl.store(bool_ptr, v > 1) with v = {v.tolist()}: bytes {ob.tolist()}")
    w = torch.tensor([0, 1, 2, 255, 256, 257, -1, 512], dtype=torch.int32, device=DEV)
    ob2 = torch.full((8,), 7, dtype=torch.uint8, device=DEV)
    try:
        k_store_int[(1,)](w, ob2.view(torch.bool), 8)
        torch.cuda.synchronize()
        out(f"tl.store(bool_ptr, int32 values {w.tolist()}): bytes {ob2.tolist()}")
    except Exception as e:  # noqa: BLE001
        out(f"tl.store(bool_ptr, int32 values): rejected ({type(e).__name__}: {str(e).splitlines()[0][:120]})")
    with open(__file__.rsplit("/", 1)[0] + "/out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()

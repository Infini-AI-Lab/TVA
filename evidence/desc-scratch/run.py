"""In-kernel tensor descriptors and the launch's global scratch memory.

Library evidence for library issue L5 (round 8). A kernel that calls `tl.make_tensor_descriptor`
gets its descriptors from memory the launcher asks the user's allocator for (`triton.set_allocator`).
Each case runs in its own process (the allocator is per process, and a case may corrupt memory):

  D1  no allocator set: what the launch raises, and whether any program ran;
  D2  a recording allocator: the size, alignment and stream of each request, against the compiled
      kernel's `global_scratch_size` / `global_scratch_align`, the grid and the descriptor count;
      one request per launch, also when the compiled kernel is reused;
  D3  which bytes the launch writes: the scratch sits inside a larger sentinel-filled buffer, the
      inputs and the output carry guard regions; two sentinels, so a byte that happens to equal
      one sentinel is still seen;
  D4  an allocator that returns memory overlapping an input buffer;
  D5  descriptors built on the host (`TensorDescriptor` arguments): no scratch, no allocator.

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/desc-scratch/run.py  (from TVA's root)
"""
import json, os, subprocess, sys

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

M = N = 128
BM = BN = 32


@triton.jit
def k_one(x_ptr, y_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    # one descriptor (the output); the input is read through pointers
    pm, pn = tl.program_id(0), tl.program_id(1)
    r = pm * BM + tl.arange(0, BM)
    c = pn * BN + tl.arange(0, BN)
    v = tl.load(x_ptr + r[:, None] * N + c[None, :])
    yd = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    yd.store([pm * BM, pn * BN], v + 1)


@triton.jit
def k_two(x_ptr, y_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    xd = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    yd = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    v = xd.load([pm * BM, pn * BN])
    yd.store([pm * BM, pn * BN], v + 1)


@triton.jit
def k_three(x_ptr, y_ptr, z_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    xd = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    yd = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    zd = tl.make_tensor_descriptor(z_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    v = xd.load([pm * BM, pn * BN])
    yd.store([pm * BM, pn * BN], v + 1)
    zd.store([pm * BM, pn * BN], v + 2)


@triton.jit
def k_host(xd, yd, BM: tl.constexpr, BN: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    v = xd.load([pm * BM, pn * BN])
    yd.store([pm * BM, pn * BN], v + 1)


GUARD = 4096  # bytes of guard before and after each buffer


def guarded(nbytes, fill):
    """A uint8 buffer of nbytes with GUARD bytes of `fill` on each side, and its inner view."""
    big = torch.full((nbytes + 2 * GUARD,), fill, dtype=torch.uint8, device="cuda")
    return big, big[GUARD:GUARD + nbytes]


def inputs(fill):
    xb, xv = guarded(M * N * 4, fill)
    x = xv.view(torch.float32).view(M, N)
    x.copy_(torch.arange(M * N, dtype=torch.float32, device="cuda").view(M, N))
    yb, yv = guarded(M * N * 4, fill)
    zb, zv = guarded(M * N * 4, fill)
    return xb, x, yb, yv.view(torch.float32).view(M, N), zb, zv.view(torch.float32).view(M, N)


def launch(name, grid, x, y, z):
    if name == "one":
        return k_one[grid](x, y, M, N, BM=BM, BN=BN)
    if name == "two":
        return k_two[grid](x, y, M, N, BM=BM, BN=BN)
    return k_three[grid](x, y, z, M, N, BM=BM, BN=BN)


def case_D1():
    xb, x, yb, y, zb, z = inputs(0xA5)
    before = yb.clone()
    try:
        launch("two", (M // BM, N // BN), x, y, z)
        torch.cuda.synchronize()
        return {"raised": None}
    except Exception as e:
        torch.cuda.synchronize()
        return {"raised": f"{type(e).__name__}: {e}", "output bytes changed": int((yb != before).sum())}


def case_D2():
    calls = []

    def alloc(size, alignment, stream):
        calls.append({"size": size, "alignment": alignment, "stream": stream})
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc)
    rows = []
    stream = torch.cuda.current_stream().cuda_stream
    for name in ("one", "two", "three"):
        for grid in ((1, 1), (4, 4), (2, 3), (4, 4)):
            xb, x, yb, y, zb, z = inputs(0)
            n0 = len(calls)
            ck = launch(name, grid, x, y, z)
            torch.cuda.synchronize()
            ok = torch.equal(y[:grid[0] * BM, :grid[1] * BN], x[:grid[0] * BM, :grid[1] * BN] + 1)
            got = calls[n0:]
            rows.append({
                "kernel": name, "grid": grid, "requests": len(got),
                "size": [c["size"] for c in got], "alignment": [c["alignment"] for c in got],
                "stream is the launch's": all(c["stream"] == stream for c in got),
                "global_scratch_size": ck.metadata.global_scratch_size,
                "global_scratch_align": ck.metadata.global_scratch_align,
                "programs x scratch_size": grid[0] * grid[1] * ck.metadata.global_scratch_size,
                "result correct": ok,
            })
    # a side stream: the allocator is told that stream
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        xb, x, yb, y, zb, z = inputs(0)
        n0 = len(calls)
        launch("two", (2, 2), x, y, z)
    torch.cuda.synchronize()
    side = {"side stream": s.cuda_stream, "allocator saw": [c["stream"] for c in calls[n0:]]}
    # two CTAs per program (num_ctas=2): the request is programs x num_ctas x scratch_size
    xb, x, yb, y, zb, z = inputs(0)
    n0 = len(calls)
    ck = k_two[(4, 4)](x, y, M, N, BM=BM, BN=BN, num_ctas=2)
    torch.cuda.synchronize()
    rows.append({"kernel": "two, num_ctas=2", "grid": (4, 4), "requests": len(calls) - n0,
                 "size": [c["size"] for c in calls[n0:]], "global_scratch_size": ck.metadata.global_scratch_size,
                 "programs x num_ctas x scratch_size": 16 * 2 * ck.metadata.global_scratch_size,
                 "result correct": torch.equal(y, x + 1)})
    # warmup compiles without launching: no request
    n0 = len(calls)
    k_two.warmup(x, y, M, N, BM=BM, BN=BN, grid=(1, 1))
    return {"rows": rows, "side": side, "requests during warmup": len(calls) - n0}


def case_D3(name):
    res = {}
    for fill in (0xA5, 0x5A):
        regions = []

        def alloc(size, alignment, stream):
            big, inner = guarded(size, fill)
            regions.append((big, inner, size))
            return inner

        triton.set_allocator(alloc)
        xb, x, yb, y, zb, z = inputs(fill)
        xb0, zb0 = xb.clone(), zb.clone()
        grid = (4, 4)
        ck = launch(name, grid, x, y, z)
        torch.cuda.synchronize()
        big, inner, size = regions[-1]
        per = ck.metadata.global_scratch_size
        changed = (inner != fill).view(grid[0] * grid[1], per)
        y_ok = torch.equal(y, x + 1)
        res[hex(fill)] = {
            "scratch bytes": size, "per program": per,
            "changed per program (min, max)": (int(changed.sum(1).min()), int(changed.sum(1).max())),
            "changed bytes by offset within a program (first, last)": (
                int(changed.any(0).nonzero().min()), int(changed.any(0).nonzero().max())),
            "guard around scratch unchanged": bool((big[:GUARD] == fill).all() and (big[GUARD + size:] == fill).all()),
            "input x (with guards) unchanged": bool(torch.equal(xb, xb0)),
            "output y correct": y_ok,
            "guards around y unchanged": bool((yb[:GUARD] == fill).all() and (yb[-GUARD:] == fill).all()),
            "z (with guards) unchanged" if name != "three" else "z correct":
                bool(torch.equal(zb, zb0)) if name != "three" else bool(torch.equal(z, x + 2)),
            "_mask": changed.any(0).tolist(),
        }
    a, b = res["0xa5"].pop("_mask"), res["0x5a"].pop("_mask")
    res["bytes written within a program's scratch (union of both sentinels)"] = sum(
        1 for i in range(len(a)) if a[i] or b[i])
    return res


def case_D4():
    xb, x, yb, y, zb, z = inputs(0)
    x_before = x.clone()

    def alloc(size, alignment, stream):
        # adversarial: hand out the first bytes of the input x as the scratch
        return x.view(torch.int8).view(-1)[:size]

    triton.set_allocator(alloc)
    launch("two", (4, 4), x, y, z)
    torch.cuda.synchronize()
    dx = (x != x_before)
    dy = (y != x_before + 1)
    return {"x elements overwritten": int(dx.sum()), "y elements wrong": int(dy.sum()),
            "first overwritten x indices": dx.view(-1).nonzero().view(-1)[:4].tolist()}


def case_D5():
    x = torch.arange(M * N, dtype=torch.float32, device="cuda").view(M, N)
    y = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    xd = TensorDescriptor(x, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    yd = TensorDescriptor(y, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    ck = k_host[(M // BM, N // BN)](xd, yd, BM=BM, BN=BN)  # no allocator set
    torch.cuda.synchronize()
    return {"global_scratch_size": ck.metadata.global_scratch_size, "result correct": torch.equal(y, x + 1)}


CASES = {
    "D1": ("two descriptors in the kernel, no allocator set", case_D1),
    "D2": ("recording allocator; kernels with 1, 2, 3 descriptors; several grids", case_D2),
    "D3-one": ("bytes written, one descriptor", lambda: case_D3("one")),
    "D3-two": ("bytes written, two descriptors", lambda: case_D3("two")),
    "D3-three": ("bytes written, three descriptors", lambda: case_D3("three")),
    "D4": ("allocator returns the first bytes of the input x", case_D4),
    "D5": ("descriptors built on the host, no allocator", case_D5),
}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        try:
            res = CASES[sys.argv[2]][1]()
        except Exception as e:
            res = {"error": f"{type(e).__name__}: " + str(e).splitlines()[0][:200]}
        print("RESULT " + json.dumps(res, default=str))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    print(f"M = N = {M}, blocks {BM} x {BN} float32")
    for name, (desc, _) in CASES.items():
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                           capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        print(f"\n{name}: {desc}")
        if line is None:
            print("  process failed: " + ((p.stderr.strip().splitlines() or ["(no output)"])[-1])[:200])
            continue
        res = json.loads(line[len("RESULT "):])
        if name == "D2" and "rows" in res:
            for r in res["rows"]:
                print("  " + json.dumps(r))
            print("  " + json.dumps(res["side"]))
            print(f"  requests during warmup: {res['requests during warmup']}")
        else:
            for k, v in res.items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

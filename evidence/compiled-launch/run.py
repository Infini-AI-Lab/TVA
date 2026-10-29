"""Launching a precompiled `CompiledKernel`, and `launch_metadata=`.

Library evidence for library issue L47 (round 8). Tutorial 02 (`tutorials/02-fused-softmax.py`
lines 143-174) compiles its kernel with `softmax_kernel.warmup(...)` and launches the returned
`CompiledKernel` directly with `kernel[(num_programs, 1, 1)](...)`. That path skips the JIT's
argument binding and specialisation. Each case runs in its own process:

  K1  tutorial 02's own pattern (its kernel, copied verbatim, and its two calls): the result, the
      specialisation the warm-up compiled, what the constexpr positions of the launch do, a launch
      without them, a 1-element grid;
  K2  launches of a compiled kernel with arguments outside the specialisation it was compiled for;
  K3  `@triton.jit(launch_metadata=f)`: when `f` runs, and whether it changes the launch.

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/compiled-launch/run.py  (from TVA's root)
"""
import json, os, re, resource, subprocess, sys

import torch
import triton
import triton.language as tl
from triton.runtime import driver
from triton import knobs


# tutorials/02-fused-softmax.py lines 84-109, verbatim (comments dropped)
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def t02_compile(x, y):
    """Tutorial 02's `softmax` up to the launch (lines 125-171), on NVIDIA."""
    properties = driver.active.utils.get_device_properties(torch.cuda.current_device())
    NUM_SM, NUM_REGS, SIZE_SMEM, WARP_SIZE = (properties["multiprocessor_count"], properties["max_num_regs"],
                                               properties["max_shared_mem"], properties["warpSize"])
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200000 else 2
    kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1, ))
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = min(NUM_SM * occupancy, n_rows)
    return kernel, num_programs, BLOCK_SIZE, num_stages


def sig_of(ck):
    """The specialisation a compiled kernel was built for: signature, constexprs, attributes."""
    return {"signature": {k: v for k, v in ck.src.signature.items()},
            "constexprs": {str(k): str(v) for k, v in ck.src.constants.items()},
            "attrs": {str(k): str(v) for k, v in ck.src.attrs.items()}}


def case_K1():
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device="cuda")      # the tutorial's unit-test shape
    y = torch.empty_like(x)
    kernel, num_programs, BLOCK_SIZE, num_stages = t02_compile(x, y)
    ref = torch.softmax(x, axis=1)
    out = {"compiled for": sig_of(kernel), "num_programs": num_programs}
    # the tutorial's launch (line 174)
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), x.shape[0], x.shape[1], BLOCK_SIZE, num_stages)
    torch.cuda.synchronize()
    out["tutorial launch: allclose to torch.softmax"] = torch.allclose(y, ref)
    y0 = y.clone()
    # other values in the constexpr positions
    y.zero_()
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), x.shape[0], x.shape[1], 1, 99)
    torch.cuda.synchronize()
    out["constexpr positions given 1 and 99: result identical"] = torch.equal(y, y0)
    y.zero_()
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), x.shape[0], x.shape[1], None, "anything")
    torch.cuda.synchronize()
    out["constexpr positions given None and a string: result identical"] = torch.equal(y, y0)
    # without the constexpr positions
    try:
        y.zero_()
        kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), x.shape[0], x.shape[1])
        torch.cuda.synchronize()
        out["without the constexpr positions"] = "ran; result identical: %s" % torch.equal(y, y0)
    except Exception as e:
        out["without the constexpr positions"] = f"{type(e).__name__}: {str(e).splitlines()[0][:150]}"
    try:
        kernel[(num_programs,)](y, x, x.stride(0), y.stride(0), x.shape[0], x.shape[1], BLOCK_SIZE, num_stages)
        out["grid (num_programs,)"] = "ran"
    except Exception as e:
        out["grid (num_programs,)"] = f"{type(e).__name__}: {str(e).splitlines()[0][:150]}"
    return out


@triton.jit
def k_copy(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m), mask=m)


@triton.jit
def k_stride(x_ptr, y_ptr, n, s, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs * s, mask=m), mask=m)


BLOCK = 128
KW = dict(BLOCK=BLOCK, num_warps=1)   # each thread holds 4 consecutive float32


def vec(ck):
    return sorted(set(re.findall(r"(?:ld|st)\.global(?:\.[a-z0-9]+)+", ck.asm["ptx"])))


def case_K2a():
    """compiled with 16-byte aligned pointers, launched with x 4 bytes past alignment"""
    x = torch.arange(4096, dtype=torch.float32, device="cuda")
    y = torch.zeros(1024, dtype=torch.float32, device="cuda")
    ck = k_copy.warmup(x, y, 1024, grid=(8,), **KW)
    res = {"compiled for": sig_of(ck), "memory ops": vec(ck)}
    xm = x[1:]
    k_copy[(8,)](xm, y, 1024, **KW)                  # through the JIT: recompiled for xm
    torch.cuda.synchronize()
    res["JIT launch with x[1:]: correct"] = torch.equal(y, xm[:1024])
    y.zero_()
    try:
        ck[(8, 1, 1)](xm, y, 1024, BLOCK)
        torch.cuda.synchronize()
        res["compiled kernel with x[1:]"] = "correct" if torch.equal(y, xm[:1024]) else "WRONG"
    except Exception as e:
        res["compiled kernel with x[1:]"] = "error: " + str(e).splitlines()[0][:150]
    return res


def case_K2b():
    """compiled with n = 1024 (divisible by 16), launched with n = 1001"""
    x = torch.arange(1024, dtype=torch.float32, device="cuda")
    y = torch.full((1024,), -1.0, dtype=torch.float32, device="cuda")
    ck = k_copy.warmup(x, y, 1024, grid=(8,), **KW)
    res = {"compiled for": sig_of(ck), "memory ops": vec(ck)}
    k_copy[(8,)](x, y, 1001, **KW)
    torch.cuda.synchronize()
    res["JIT launch with n=1001: elements written"] = int((y != -1).sum())
    y.fill_(-1)
    ck[(8, 1, 1)](x, y, 1001, BLOCK)
    torch.cuda.synchronize()
    res["compiled kernel with n=1001: elements written"] = int((y != -1).sum())
    res["compiled kernel with n=1001: indices >= 1001 written"] = (y[1001:] != -1).nonzero().view(-1).add(1001).tolist()
    return res


def case_K2c():
    """compiled with s = 1 (a constexpr), launched with s = 3"""
    x = torch.arange(4096, dtype=torch.float32, device="cuda")
    y = torch.zeros(1024, dtype=torch.float32, device="cuda")
    ck = k_stride.warmup(x, y, 1024, 1, grid=(8,), **KW)
    res = {"compiled for": sig_of(ck)}
    ck[(8, 1, 1)](x, y, 1024, 3, BLOCK)
    torch.cuda.synchronize()
    res["compiled kernel with s=3: equals x[3*i]"] = torch.equal(y, x[0:3072:3])
    res["compiled kernel with s=3: equals x[i] (s taken as 1)"] = torch.equal(y, x[:1024])
    return res


def case_K2d():
    """compiled with s = 3 (an int32), launched with s = 1"""
    x = torch.arange(4096, dtype=torch.float32, device="cuda")
    y = torch.zeros(1024, dtype=torch.float32, device="cuda")
    ck = k_stride.warmup(x, y, 1024, 3, grid=(8,), **KW)
    res = {"compiled for": sig_of(ck)}
    ck[(8, 1, 1)](x, y, 1024, 1, BLOCK)
    torch.cuda.synchronize()
    res["compiled kernel with s=1: equals x[i]"] = torch.equal(y, x[:1024])
    return res


def case_K2e():
    """compiled with n = 128 (int32), launched with n = 2^31 + 5 and n = 2^32 + 5"""
    x = torch.arange(128, dtype=torch.float32, device="cuda")
    y = torch.full((128,), -1.0, dtype=torch.float32, device="cuda")
    ck = k_copy.warmup(x, y, 128, grid=(1,), **KW)
    res = {"compiled for": sig_of(ck)}
    for n in (2**31 + 5, 2**32 + 5):
        y.fill_(-1)
        try:
            ck[(1, 1, 1)](x, y, n, BLOCK)
            torch.cuda.synchronize()
            res[f"compiled kernel with n={n}: elements written"] = int((y != -1).sum())
        except Exception as e:
            res[f"compiled kernel with n={n}"] = "error: " + str(e).splitlines()[0][:150]
        y.fill_(-1)
        k_copy[(1,)](x, y, n, **KW)
        torch.cuda.synchronize()
        res[f"JIT launch with n={n}: elements written"] = int((y != -1).sum())
    res["JIT signatures compiled"] = sorted({str(dict(v.src.signature)) for d in k_copy.device_caches.values()
                                             for v in d[0].values()})
    return res


def case_K2f():
    """compiled with float32 x, launched with a float16 x of the same byte size"""
    xf = torch.arange(1024, dtype=torch.float32, device="cuda")
    y = torch.zeros(1024, dtype=torch.float32, device="cuda")
    ck = k_copy.warmup(xf, y, 1024, grid=(8,), **KW)
    xh = torch.randn(2048, device="cuda").to(torch.float16)
    ck[(8, 1, 1)](xh, y, 1024, BLOCK)
    torch.cuda.synchronize()
    return {"compiled for": sig_of(ck),
            "compiled kernel with float16 x: y equals x's bytes read as float32": torch.equal(
                y.view(torch.int32), xh.view(torch.int32))}


def case_K2g():
    """too few arguments: two of the three runtime ones"""
    x = torch.arange(1024, dtype=torch.float32, device="cuda")
    y = torch.zeros(1024, dtype=torch.float32, device="cuda")
    ck = k_copy.warmup(x, y, 1024, grid=(8,), **KW)
    ck[(8, 1, 1)](x, y)
    torch.cuda.synchronize()
    return {"two arguments": "ran"}


def case_K2h():
    """a host (CPU) tensor where the kernel was compiled for a device tensor"""
    x = torch.arange(1024, dtype=torch.float32, device="cuda")
    y = torch.zeros(1024, dtype=torch.float32, device="cuda")
    ck = k_copy.warmup(x, y, 1024, grid=(8,), **KW)
    try:
        ck[(8, 1, 1)](x.cpu(), y, 1024, BLOCK)
        torch.cuda.synchronize()
        return {"host tensor for x": "ran", "y written": bool((y != 0).any())}
    except Exception as e:
        return {"host tensor for x": f"{type(e).__name__}: {str(e).splitlines()[0][:150]}",
                "y untouched": bool((y == 0).all())}


META_CALLS = []


def meta_fn(grid, metadata, args):
    META_CALLS.append(sorted(args))
    return {"name": "probe", "flops": 7}


@triton.jit(launch_metadata=meta_fn)
def k_meta(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m) * 2, mask=m)


def case_K3():
    x = torch.arange(1024, dtype=torch.float32, device="cuda")
    y = torch.zeros(1024, dtype=torch.float32, device="cuda")
    res = {"knobs.runtime.launch_enter_hook": type(knobs.runtime.launch_enter_hook).__name__,
           "launch_enter_hook is None": knobs.runtime.launch_enter_hook is None}

    def run(label, compiled=None):
        y.zero_()
        n0 = len(META_CALLS)
        try:
            if compiled is None:
                k_meta[(8,)](x, y, 1024, **KW)
            else:
                compiled[(8, 1, 1)](x, y, 1024, BLOCK)
            torch.cuda.synchronize()
            res[label] = {"launch_metadata calls": len(META_CALLS) - n0, "result correct": torch.equal(y, 2 * x)}
        except Exception as e:
            res[label] = {"launch_metadata calls": len(META_CALLS) - n0, "raised": f"{type(e).__name__}: {e}",
                          "y untouched": bool((y == 0).all())}

    run("no hook added (JIT)")
    ck = k_meta.warmup(x, y, 1024, grid=(8,), **KW)
    run("no hook added (compiled kernel)", ck)
    seen = []
    quiet = lambda md: seen.append(type(md).__name__)
    knobs.runtime.launch_enter_hook.add(quiet)
    run("a hook that does not read the metadata (JIT)")
    knobs.runtime.launch_enter_hook.remove(quiet)
    reader = lambda md: seen.append(md.get())
    knobs.runtime.launch_enter_hook.add(reader)
    run("a hook that reads the metadata (JIT)")
    run("a hook that reads the metadata (compiled kernel)", ck)
    res["what the reading hook saw"] = [{k: str(v) for k, v in d.items() if k in ("name", "flops")}
                                        for d in seen if isinstance(d, dict)][:1]

    def bad(grid, metadata, args):
        raise ValueError("launch_metadata failed")
    k_meta.launch_metadata = bad
    run("a hook that reads the metadata, launch_metadata raises (JIT)")
    knobs.runtime.launch_enter_hook.remove(reader)
    return res


CASES = {
    "K1": "tutorial 02: warmup, then the CompiledKernel launched with the tutorial's arguments",
    "K2a": "compiled for 16-byte aligned x; launched with x 4 bytes past alignment",
    "K2b": "compiled for n = 1024 (divisible by 16); launched with n = 1001",
    "K2c": "compiled for s = 1 (a constexpr); launched with s = 3",
    "K2d": "compiled for s = 3 (int32); launched with s = 1",
    "K2e": "compiled for an int32 n; launched with n >= 2^31",
    "K2f": "compiled for float32 x; launched with float16 x",
    "K2g": "compiled kernel launched with too few arguments",
    "K2h": "compiled kernel launched with a host tensor",
    "K3": "launch_metadata= and launch hooks",
}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        try:
            res = globals()["case_" + sys.argv[2]]()
        except Exception as e:
            msg = re.sub(r"\S*/(Objects/\S+)", r"\1", str(e).splitlines()[0])  # drop the interpreter's build path
            res = {"error": f"{type(e).__name__}: " + msg[:200]}
        print("RESULT " + json.dumps(res, default=str))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")

    def no_core():  # a case that crashes the interpreter must not leave a core file in TVA's tree
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    for name, desc in CASES.items():
        if name == "K2g":   # the outcome varies between processes: tally five
            tally = {}
            for _ in range(5):
                p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                                   capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"),
                                   preexec_fn=no_core)
                line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
                if line is not None:
                    out = json.loads(line[len("RESULT "):])
                    key = out.get("error") or json.dumps(out)
                else:
                    key = f"process killed by signal {-p.returncode}" if p.returncode < 0 else f"exit code {p.returncode}"
                tally[key] = tally.get(key, 0) + 1
            print(f"\n{name}: {desc}")
            for k, v in tally.items():
                print(f"  {v} of 5 processes: {k}")
            continue
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                           capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"),
                                   preexec_fn=no_core)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        print(f"\n{name}: {desc}")
        if line is None:
            rc = p.returncode
            how = f"killed by signal {-rc}" if rc < 0 else f"exit code {rc}"
            print(f"  process failed ({how}): " + ((p.stderr.strip().splitlines() or ["(no output)"])[-1])[:200])
            continue
        for k, v in json.loads(line[len("RESULT "):]).items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

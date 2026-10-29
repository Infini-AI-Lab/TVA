"""What `tl.multiple_of` alone, and `tl.max_contiguous` with it, promise on the shapes kernels hint.

Library evidence for library issue L58 (round 11). Triton 3.8.0's axis analysis
(`primitives/triton-3.8.0-AxisInfo.cpp`) gives each value a contiguity, a divisibility and a
constancy per dimension. A hint replaces one of them on the operation that defines the hinted value
(`visitOperation`, lines 1232-1259): `multiple_of` the divisibility, `max_contiguous` the contiguity.
`multiple_of` alone therefore pairs its value with the contiguity the analysis infers for that value:
`tl.arange(0, B)` has contiguity `B` (lines 169-183), and a sum or difference with an operand that is
constant along the dimension keeps it (lines 242-279; a scalar is splat with constancy the whole
shape, lines 553-575; a size-1 dimension that is broadcast has constancy the whole dimension, lines
642-668; integer casts keep it, lines 138-147, 1180-1185).

Each case loads a block of 128 `float32` (one warp: each thread holds four consecutive elements, so a
load is vectorised, `ld.global.v4.b32`, exactly when the compiler believes each thread's first
address is 16-byte aligned) through `x_ptr + offs`, with `x_ptr` 16-byte aligned, and reports the
PTX loads and the result against the host, or the launch's error. Each case runs in its own process
(a misaligned access ends the CUDA context), on GPU 0. "true" and "false" are the library's reading
after round 11: `FirstsDiv offs k C v` with `C` the contiguity stated for the shape (for the pair,
`ContigAlong offs k C` and `FirstsDiv offs k C v` with the hinted `C`).

  H1/H2    offs = s + tl.arange(0, 128); multiple_of(offs, 4)                    s = 4 / s = 1
  H3/H4    offs = tl.arange(0, 128) + s                                          s = 4 / s = 2
  H5/H6    offs = tl.arange(0, 128) - s                                          s = -8 / s = -3
  H7/H8    offs = tl.load(s_ptr) + tl.arange(0, 128) (a loaded scalar)           s = 8 / s = 1
  H9/H10   p = tl.multiple_of(x_ptr + s + tl.arange(0, 128), 16) (bytes)         s = 4 / s = 1
  H11/H12  offs = (s + tl.arange(0, 128)).to(tl.int64); multiple_of(offs, 4)     s = 4 / s = 1
  H13/H14  2-D: offs = (tl.arange(0, 4) * S)[:, None] + tl.arange(0, 32)[None, :];
           multiple_of(offs, [1, 4])                                             S = 36 / S = 34
  H15/H16  2-D: offs = tl.load(r_ptr + tl.arange(0, 4))[:, None] + tl.arange(0, 32)[None, :];
           multiple_of(offs, [1, 4]) (a loaded column, broadcast along axis 1)  rows [0, 36, 72, 108] / [0, 34, 68, 102]
  H17      pair below the value's own contiguity:
           max_contiguous(multiple_of(s + tl.arange(0, 128), 4), 4)              s = 4 (true)
  H18      the same with max_contiguous(…, 8)                                    s = 4 (true)
  H19      the pair (4, 4) with s = 2                                            (false)
  H20      no hint, s = 4 (control: s is not a multiple of 16, so the launcher does not mark it)
  H21      2-D, no hint, S = 36 (control)

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/hint-contig/run.py  (from TVA's root)
"""
import json, os, re, subprocess, sys

import torch
import triton
import triton.language as tl


@triton.jit
def k_add_left(x_ptr, out_ptr, s, HINT: tl.constexpr):
    offs = s + tl.arange(0, 128)
    if HINT == 1:
        offs = tl.multiple_of(offs, 4)
    if HINT == 2:
        offs = tl.max_contiguous(tl.multiple_of(offs, 4), 4)
    if HINT == 3:
        offs = tl.max_contiguous(tl.multiple_of(offs, 4), 8)
    tl.store(out_ptr + tl.arange(0, 128), tl.load(x_ptr + offs))


@triton.jit
def k_add_right(x_ptr, out_ptr, s):
    offs = tl.arange(0, 128) + s
    offs = tl.multiple_of(offs, 4)
    tl.store(out_ptr + tl.arange(0, 128), tl.load(x_ptr + offs))


@triton.jit
def k_sub(x_ptr, out_ptr, s):
    offs = tl.arange(0, 128) - s
    offs = tl.multiple_of(offs, 4)
    tl.store(out_ptr + tl.arange(0, 128), tl.load(x_ptr + offs))


@triton.jit
def k_loaded(x_ptr, out_ptr, s_ptr):
    offs = tl.load(s_ptr) + tl.arange(0, 128)
    offs = tl.multiple_of(offs, 4)
    tl.store(out_ptr + tl.arange(0, 128), tl.load(x_ptr + offs))


@triton.jit
def k_ptr(x_ptr, out_ptr, s):
    p = tl.multiple_of(x_ptr + s + tl.arange(0, 128), 16)
    tl.store(out_ptr + tl.arange(0, 128), tl.load(p))


@triton.jit
def k_cast(x_ptr, out_ptr, s):
    offs = (s + tl.arange(0, 128)).to(tl.int64)
    offs = tl.multiple_of(offs, 4)
    tl.store(out_ptr + tl.arange(0, 128), tl.load(x_ptr + offs))


@triton.jit
def k_2d(x_ptr, out_ptr, S, HINT: tl.constexpr):
    offs = (tl.arange(0, 4) * S)[:, None] + tl.arange(0, 32)[None, :]
    if HINT:
        offs = tl.multiple_of(offs, [1, 4])
    o = tl.arange(0, 4)[:, None] * 32 + tl.arange(0, 32)[None, :]
    tl.store(out_ptr + o, tl.load(x_ptr + offs))


@triton.jit
def k_2d_loaded(x_ptr, out_ptr, r_ptr):
    offs = tl.load(r_ptr + tl.arange(0, 4))[:, None] + tl.arange(0, 32)[None, :]
    offs = tl.multiple_of(offs, [1, 4])
    o = tl.arange(0, 4)[:, None] * 32 + tl.arange(0, 32)[None, :]
    tl.store(out_ptr + o, tl.load(x_ptr + offs))


CASES = {
    "H1": ("s + arange, multiple_of 4, s = 4", "true"),
    "H2": ("s + arange, multiple_of 4, s = 1", "false"),
    "H3": ("arange + s, multiple_of 4, s = 4", "true"),
    "H4": ("arange + s, multiple_of 4, s = 2", "false"),
    "H5": ("arange - s, multiple_of 4, s = -8", "true"),
    "H6": ("arange - s, multiple_of 4, s = -3", "false"),
    "H7": ("loaded scalar + arange, multiple_of 4, s = 8", "true"),
    "H8": ("loaded scalar + arange, multiple_of 4, s = 1", "false"),
    "H9": ("pointer x + s + arange, multiple_of 16 bytes, s = 4", "true"),
    "H10": ("pointer x + s + arange, multiple_of 16 bytes, s = 1", "false"),
    "H11": ("(s + arange).to(int64), multiple_of 4, s = 4", "true"),
    "H12": ("(s + arange).to(int64), multiple_of 4, s = 1", "false"),
    "H13": ("2-D rows of stride S, multiple_of [1, 4], S = 36", "true"),
    "H14": ("2-D rows of stride S, multiple_of [1, 4], S = 34", "false"),
    "H15": ("2-D loaded row starts [0, 36, 72, 108], multiple_of [1, 4]", "true"),
    "H16": ("2-D loaded row starts [0, 34, 68, 102], multiple_of [1, 4]", "false"),
    "H17": ("pair max_contiguous(multiple_of(s + arange, 4), 4), s = 4", "true"),
    "H18": ("pair max_contiguous(multiple_of(s + arange, 4), 8), s = 4", "true"),
    "H19": ("pair max_contiguous(multiple_of(s + arange, 4), 4), s = 2", "false"),
    "H20": ("s + arange, no hint, s = 4 (control)", "-"),
    "H21": ("2-D rows of stride S, no hint, S = 36 (control)", "-"),
}


def run_case(name):
    dev = "cuda"
    x = torch.arange(1024, device=dev, dtype=torch.float32)
    out = torch.zeros(128, device=dev, dtype=torch.float32)
    xs = torch.arange(1024, dtype=torch.float32)
    kw = dict(num_warps=1)
    if name in ("H1", "H2", "H17", "H18", "H19", "H20"):
        s = {"H1": 4, "H2": 1, "H17": 4, "H18": 4, "H19": 2, "H20": 4}[name]
        hint = {"H17": 2, "H18": 3, "H19": 2, "H20": 0}.get(name, 1)
        ck = k_add_left[(1,)](x, out, s, HINT=hint, **kw)
        ref = xs[s:s + 128]
    elif name in ("H3", "H4"):
        s = 4 if name == "H3" else 2
        ck = k_add_right[(1,)](x, out, s, **kw)
        ref = xs[s:s + 128]
    elif name in ("H5", "H6"):
        s = -8 if name == "H5" else -3
        ck = k_sub[(1,)](x, out, s, **kw)
        ref = xs[-s:-s + 128]
    elif name in ("H7", "H8"):
        s = 8 if name == "H7" else 1
        sp = torch.tensor([s], device=dev, dtype=torch.int32)
        ck = k_loaded[(1,)](x, out, sp, **kw)
        ref = xs[s:s + 128]
    elif name in ("H9", "H10"):
        s = 4 if name == "H9" else 1
        ck = k_ptr[(1,)](x, out, s, **kw)
        ref = xs[s:s + 128]
    elif name in ("H11", "H12"):
        s = 4 if name == "H11" else 1
        ck = k_cast[(1,)](x, out, s, **kw)
        ref = xs[s:s + 128]
    elif name in ("H13", "H14", "H21"):
        S = 34 if name == "H14" else 36
        ck = k_2d[(1,)](x, out, S, HINT=name != "H21", **kw)
        ref = torch.cat([xs[r * S:r * S + 32] for r in range(4)])
    elif name in ("H15", "H16"):
        rows = [0, 36, 72, 108] if name == "H15" else [0, 34, 68, 102]
        rp = torch.tensor(rows, device=dev, dtype=torch.int32)
        ck = k_2d_loaded[(1,)](x, out, rp, **kw)
        ref = torch.cat([xs[r:r + 32] for r in rows])
    else:
        raise SystemExit("unknown case " + name)
    loads = sorted(set(re.findall(r"ld\.global(?:\.[a-z0-9]+)+", ck.asm["ptx"])))
    try:
        torch.cuda.synchronize()
        got = out.cpu()
        result = "correct" if torch.equal(got, ref) else f"WRONG at {(got != ref).sum().item()} elements"
    except Exception as e:  # the launch's error, reported to the host
        result = "error: " + str(e).strip().splitlines()[0][:150]
    return {"loads": loads, "result": result}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        try:
            res = run_case(sys.argv[2])
        except Exception as e:
            res = {"loads": [], "result": "error: " + type(e).__name__ + ": " + str(e).strip().splitlines()[0][:150]}
        print("RESULT " + json.dumps(res))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name, (desc, truth) in CASES.items():
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                           capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line:
            res = json.loads(line[len("RESULT "):])
        else:
            err = [l for l in p.stderr.strip().splitlines() if l.strip()]
            res = {"loads": [], "result": "process failed: " + (err[-1][:150] if err else "(no output)")}
        print(f"{name}: {desc}; hint {truth}\n  loads {res['loads']}; {res['result']}")


if __name__ == "__main__":
    main()

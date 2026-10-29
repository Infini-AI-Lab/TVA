"""The contiguity bound of a value whose max_contiguous mark the compiler may not read.

Library evidence for library issue L58 (round 17). SEMANTICS.md ("Compiler hints", "The contiguity
the analysis infers") gives a value that carries a `max_contiguous` hint the hinted `C` as its lower
bound. This checks which values the compiler gives that `C`, and which keep the bound of the list
without the hint. Sources: `primitives/triton-3.8.0-AxisInfo.cpp` 1232-1259 (`visitOperation` reads
the hint attributes of every operation it visits) and 1309-1341 (`getPessimisticValueState`, for an operation no visitor handles).

Each case loads 512 `int8` elements through `x_ptr + y`, with `y = tl.multiple_of(r + s, 16)`,
`s = 16`, one warp. The width of that load is at most the contiguity the analysis gives `y` (and
its divisibility, 16 here), so the PTX shows it: 16 bytes (`ld.global.v4.b32`), 4 bytes
(`ld.global.b32`) or 1 byte (`ld.global.b8`).

  N0  r = tl.load(idx + i), no hint (control: the analysis' own contiguity of a load, 1)
  N1  r = tl.load(idx + i) + step, then the pair hint (16, 16) on r: an add defines r
  N2  r carried by a for loop (r = r + step in the body), the pair hint (16, 16) after the loop
  N3  r merged by a runtime if (if n > 0: r = r + step), the pair hint (16, 16) after the if
  N4  r = the result of an inlined helper that returns tl.load(idx + i), the pair (16, 16) on it
  N5  r = the first result of tl.split of a loaded [512, 2] block, the pair (16, 16) on it
  N6  r = tl.arange(0, 512), the pair hint (4, 4) on r (control: the hinted 4 replaces 512)
  N7  r = the result of an inlined helper that returns tl.arange(0, 512), the pair (4, 4) on it
  N8  r = the result of an inlined helper that returns the pair (4, 4) on tl.arange(0, 512)
  N9  r = tl.load(idx + i) + 0, then the pair (16, 16) on r: an add the canonicalizer folds away
  N10 r = tl.load(idx + i) * 1, then the pair (16, 16) on r: likewise a multiplication

In N0-N5, N9 and N10 `r` holds 0..511 from memory, so its own contiguity is 1 for the analysis; in
N6-N8 it is the arange. Each case runs in its own process with its own empty Triton cache, and checks the loaded
values against the reference.

Run from TVA's root: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/hint-unread-round17/run.py
"""
import json, os, re, subprocess, sys, tempfile

import torch
import triton
import triton.language as tl


@triton.jit
def _ret_load(idx_ptr):
    return tl.load(idx_ptr + tl.arange(0, 512))


@triton.jit
def _ret_arange():
    return tl.arange(0, 512)


@triton.jit
def _ret_hinted_arange():
    return tl.max_contiguous(tl.multiple_of(tl.arange(0, 512), 4), 4)


@triton.jit
def k(x_ptr, idx_ptr, z_ptr, out_ptr, s, step, n, CASE: tl.constexpr):
    i = tl.arange(0, 512)
    if CASE == 0:
        r = tl.load(idx_ptr + i)
    elif CASE == 1:
        r = tl.load(idx_ptr + i) + step
        r = tl.max_contiguous(tl.multiple_of(r, 16), 16)
    elif CASE == 2:
        r = tl.load(idx_ptr + i)
        for t in range(0, n):
            r = r + step
        r = tl.max_contiguous(tl.multiple_of(r, 16), 16)
    elif CASE == 3:
        r = tl.load(idx_ptr + i)
        if n > 0:
            r = r + step
        r = tl.max_contiguous(tl.multiple_of(r, 16), 16)
    elif CASE == 4:
        r = _ret_load(idx_ptr)
        r = tl.max_contiguous(tl.multiple_of(r, 16), 16)
    elif CASE == 5:
        z = tl.load(z_ptr + i[:, None] * 2 + tl.arange(0, 2)[None, :])
        r, r1 = tl.split(z)
        r = tl.max_contiguous(tl.multiple_of(r, 16), 16)
    elif CASE == 6:
        r = tl.arange(0, 512)
        r = tl.max_contiguous(tl.multiple_of(r, 4), 4)
    elif CASE == 7:
        r = _ret_arange()
        r = tl.max_contiguous(tl.multiple_of(r, 4), 4)
    elif CASE == 8:
        r = _ret_hinted_arange()
    elif CASE == 9:
        r = tl.load(idx_ptr + i) + 0
        r = tl.max_contiguous(tl.multiple_of(r, 16), 16)
    else:
        r = tl.load(idx_ptr + i) * 1
        r = tl.max_contiguous(tl.multiple_of(r, 16), 16)
    y = tl.multiple_of(r + s, 16)
    tl.store(out_ptr + i, tl.load(x_ptr + y))


CASES = {
    "N0": "r = load, no hint (control)",
    "N1": "r = load + step, pair (16, 16) on r",
    "N2": "r carried by a for loop, pair (16, 16) after the loop",
    "N3": "r merged by a runtime if, pair (16, 16) after the if",
    "N4": "r = inlined helper's result (a load), pair (16, 16) on it",
    "N5": "r = tl.split(loaded [512, 2])[0], pair (16, 16) on it",
    "N6": "r = tl.arange(0, 512), pair (4, 4) on r (control)",
    "N7": "r = inlined helper's result (tl.arange(0, 512)), pair (4, 4) on it",
    "N8": "r = inlined helper returning the pair (4, 4) on tl.arange(0, 512)",
    "N9": "r = load + 0, pair (16, 16) on r",
    "N10": "r = load * 1, pair (16, 16) on r",
}


def run_case(name):
    dev = "cuda"
    x = (torch.arange(1024, device=dev) % 127).to(torch.int8)
    idx = torch.arange(512, device=dev, dtype=torch.int32)
    z = torch.stack([idx, idx + 1000], dim=1).contiguous()
    out = torch.zeros(512, device=dev, dtype=torch.int8)
    ck = k[(1,)](x, idx, z, out, 16, 0, 2, CASE=int(name[1:]), num_warps=1)
    torch.cuda.synchronize()
    loads = sorted(set(re.findall(r"ld\.global(?:\.[a-z0-9]+)+", ck.asm["ptx"])))
    ok = torch.equal(out.cpu(), x.cpu()[16:16 + 512])
    return {"loads": loads, "values": "correct" if ok else "WRONG"}


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

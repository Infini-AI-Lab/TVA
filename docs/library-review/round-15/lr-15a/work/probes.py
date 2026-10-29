"""lr-15a, round 15: probes of the batch's statements and of the semantics area.

B  The contiguity bound of a value built from one that "carries a max_contiguous hint" (L58; SEMANTICS
   "Compiler hints", "The contiguity the analysis infers": "a value that carries a max_contiguous
   hint, alone or in the pair (its defining operation is marked), whatever operation defines it: the
   hinted C"). Each case loads 512 int8 elements through x_ptr + y, y = tl.multiple_of(r + s, 16),
   with r pair-hinted (C = 16, v = 16), s = 16 and one warp. By the library's bound, y's contiguity
   is 16, so the compiler would believe groups of 16 and use 16-byte loads (ld.global.v4.b32). The
   PTX shows the contiguity the compiler actually used. r holds 0..511 at run time, from memory (its
   inferred contiguity is 1).
  B0  r = tl.load(idx + i) + step, no hint on r (control: the compiler's own contiguity, 1)
  B1  r = tl.load(idx + i) + step (an add defines r; control: the compiler reads the mark)
  B2  r carried by a for loop (r = r + step in the body), the pair hint after the loop
  B3  r merged by a runtime if (if n > 0: r = r + step), the pair hint after the if
  B4  r returned by an inlined helper (the pair hint on the call's result)
  B5  r = the first result of tl.split of a loaded [512, 2] block, the pair hint on it
A  name: tl.constexpr = e in a for loop body (L84: "on a name the function has not defined before
   it ... binds name to e's value as it is")
  A1  for i in range(0, n): c: tl.constexpr = -7; acc += c // 2   (n = 3): Python's -4 each trip?
  A2  if n > 0: c: tl.constexpr = -7 else: c: tl.constexpr = -9; then c // 2 after the if
      (SEMANTICS "A runtime if": a name new to the if that both branches define is merged)
  A3  the same with c used inside each branch only (control)
K  tl.cdiv on a runtime int32 (used 23 times by the in-scope tutorial kernels; no stated meaning)
  K1  tl.cdiv(n, 128) with n = 2**31 - 1 (int32) and n = 1000, stored as int32
  K2  tl.cdiv(n, 4) with n = -5 (int32): Triton's (n + 3) // 4 truncates (0), Python's ceil is -1

Each case runs in its own process with its own empty Triton cache, on GPU 0.
Run from this folder: CUDA_VISIBLE_DEVICES=0 <TVA>/.venv/bin/python -I probes.py

Reading probes.out (Triton 3.8.0, H200 NVL, 2026-10-28): in the B cases the int32 loads of idx (and
z in B5) are ld.global.v4.b32 (v2.b32); the int8 load through y is ld.global.v4.b32 only in B1, and
ld.global.b8 (one byte, contiguity 1) in B0 and in B2-B5: the compiler did not use the hinted 16.
"""
import json, os, re, subprocess, sys, tempfile

import torch
import triton
import triton.language as tl


@triton.jit
def _h_load(idx_ptr):
    return tl.load(idx_ptr + tl.arange(0, 512))


@triton.jit
def k_bound(x_ptr, idx_ptr, z_ptr, out_ptr, s, step, n, CASE: tl.constexpr):
    i = tl.arange(0, 512)
    if CASE == 0:
        r = tl.load(idx_ptr + i) + step
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
        r = _h_load(idx_ptr)
        r = tl.max_contiguous(tl.multiple_of(r, 16), 16)
    else:
        z = tl.load(z_ptr + i[:, None] * 2 + tl.arange(0, 2)[None, :])
        r, r1 = tl.split(z)
        r = tl.max_contiguous(tl.multiple_of(r, 16), 16)
    y = tl.multiple_of(r + s, 16)
    tl.store(out_ptr + i, tl.load(x_ptr + y))


@triton.jit
def k_annloop(out_ptr, n):
    acc = 0
    for t in range(0, n):
        c: tl.constexpr = -7
        acc += c // 2
    tl.store(out_ptr, acc)


@triton.jit
def k_annif(out_ptr, n, USE_AFTER: tl.constexpr):
    if n > 0:
        c: tl.constexpr = -7
        if not USE_AFTER:
            tl.store(out_ptr, c // 2)
    else:
        c: tl.constexpr = -9
        if not USE_AFTER:
            tl.store(out_ptr, c // 2)
    if USE_AFTER:
        tl.store(out_ptr, c // 2)


@triton.jit
def k_cdiv(n_ptr, out_ptr, D: tl.constexpr):
    n = tl.load(n_ptr)
    tl.store(out_ptr, tl.cdiv(n, D))


CASES = {
    "B0": "r = load + step, no hint (control)",
    "B1": "r = load + step (control), pair hint on r",
    "B2": "r carried by a for loop, pair hint after the loop",
    "B3": "r merged by a runtime if, pair hint after the if",
    "B4": "r returned by an inlined helper, pair hint on it",
    "B5": "r = tl.split(loaded [512, 2])[0], pair hint on it",
    "A1": "for t in range(0, 3): c: tl.constexpr = -7; acc += c // 2",
    "A2": "runtime if, both branches c: tl.constexpr = ..., c used after the if",
    "A3": "the same, c used inside each branch (control)",
    "K1": "tl.cdiv(n, 128), n int32 = 2**31 - 1 and 1000",
    "K2": "tl.cdiv(n, 4), n int32 = -5",
}


def run_case(name):
    dev = "cuda"
    if name[0] == "B":
        x = (torch.arange(1024, device=dev) % 127).to(torch.int8)
        idx = torch.arange(512, device=dev, dtype=torch.int32)
        z = torch.stack([idx, idx + 1000], dim=1).contiguous()
        out = torch.zeros(512, device=dev, dtype=torch.int8)
        ck = k_bound[(1,)](x, idx, z, out, 16, 0, 2, CASE=int(name[1]), num_warps=1)
        torch.cuda.synchronize()
        ptx = ck.asm["ptx"]
        loads = sorted(set(re.findall(r"ld\.global(?:\.[a-z0-9]+)+", ptx)))
        ok = torch.equal(out.cpu(), x.cpu()[16:16 + 512])
        return {"loads": loads, "values": "correct" if ok else "WRONG"}
    if name == "A1":
        out = torch.zeros(1, device=dev, dtype=torch.int32)
        k_annloop[(1,)](out, 3)
        torch.cuda.synchronize()
        return "stored " + str(out.item())
    if name in ("A2", "A3"):
        out = torch.zeros(1, device=dev, dtype=torch.int32)
        k_annif[(1,)](out, 2, USE_AFTER=(name == "A2"))
        torch.cuda.synchronize()
        return "stored " + str(out.item())
    if name in ("K1", "K2"):
        vals = [2**31 - 1, 1000] if name == "K1" else [-5]
        D = 128 if name == "K1" else 4
        res = []
        for v in vals:
            nb = torch.tensor([v], device=dev, dtype=torch.int32)
            out = torch.zeros(1, device=dev, dtype=torch.int32)
            k_cdiv[(1,)](nb, out, D=D)
            torch.cuda.synchronize()
            res.append(f"n={v}: stored {out.item()} (Python ceil: {-(-v // D)})")
        return "; ".join(res)
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

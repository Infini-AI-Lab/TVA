"""lr-9a: do the float32 math facts hold with enable_fp_fusion=False?

enable_fp_fusion is one of the launch options Compiler.Faithful admits (exclusions.md). With it
False, LLVM does not contract and ptxas gets --fmad=false, so libdevice's polynomials (tl.log,
tl.math.log2) may compile to different code than the default the evidence measured.

For every float32 bit pattern, run the six functions with the default and with
enable_fp_fusion=False, compare the results bit for bit, and, where they differ, check the
fact's bound (same method as evidence/num-math-f32). Also print the PTX/SASS instruction sets.

Run from the TVA root: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-9/lr-9a/work/math_nofusion.py
"""
import re
import sys
import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def funcs(x_ptr, e_ptr, e2_ptr, l_ptr, l2_ptr, s_ptr, rs_ptr, n, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m).to(tl.float32, bitcast=True)
    tl.store(e_ptr + offs, tl.exp(x), mask=m)
    tl.store(e2_ptr + offs, tl.math.exp2(x), mask=m)
    tl.store(l_ptr + offs, tl.log(x), mask=m)
    tl.store(l2_ptr + offs, tl.math.log2(x), mask=m)
    tl.store(s_ptr + offs, tl.sqrt(x), mask=m)
    tl.store(rs_ptr + offs, tl.math.rsqrt(x), mask=m)


NAMES = ["exp", "exp2", "log", "log2", "sqrt", "rsqrt"]
MAXV = float(np.finfo(np.float32).max)
N126 = 2.0 ** -126


def reference(name, x):
    with np.errstate(all="ignore"):
        return {"exp": np.exp, "exp2": np.exp2, "log": np.log, "log2": np.log2, "sqrt": np.sqrt,
                "rsqrt": lambda v: 1.0 / np.sqrt(v)}[name](x)


def bound_terms(name, x):
    fin = np.isfinite(x)
    if name == "exp":
        return fin, 2.0 ** -22 + 2.0 ** -23 * np.abs(x), 2.0 ** -140
    if name == "exp2":
        return fin, np.full_like(x, 2.0 ** -22), 2.0 ** -126
    if name in ("log", "log2"):
        return fin & (x > 0), np.full_like(x, 2.0 ** -22), 2.0 ** -149
    if name == "sqrt":
        return fin & ((x == 0) | (x >= N126)), np.full_like(x, 2.0 ** -22), 2.0 ** -149
    return fin & (x >= N126), np.full_like(x, 2.0 ** -22), 2.0 ** -149


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    CH = 1 << 26
    st = {n: dict(diff=0, checked=0, viol=0, nonfin=0, ratio=0.0, at=None) for n in NAMES}
    hd = hn = None
    for c in range(64):
        lo = c * CH
        b64 = torch.arange(lo, lo + CH, dtype=torch.int64, device="cuda")
        bits = torch.where(b64 >= 2 ** 31, b64 - 2 ** 32, b64).to(torch.int32)
        o_def = [torch.empty(CH, dtype=torch.float32, device="cuda") for _ in NAMES]
        o_nf = [torch.empty(CH, dtype=torch.float32, device="cuda") for _ in NAMES]
        hd = funcs[(CH // 1024,)](bits, *o_def, CH, B=1024)
        hn = funcs[(CH // 1024,)](bits, *o_nf, CH, B=1024, enable_fp_fusion=False)
        x = None
        for name, od, on in zip(NAMES, o_def, o_nf):
            d = od.view(torch.int32) != on.view(torch.int32)
            # NaN payloads are not compared
            d &= ~(torch.isnan(od) & torch.isnan(on))
            nd = int(d.sum())
            st[name]["diff"] += nd
            if nd == 0:
                continue
            if x is None:
                x = bits.view(torch.float32).cpu().numpy().astype(np.float64)
            r = on.cpu().numpy().astype(np.float64)
            ref = reference(name, x)
            dom, rel, ab = bound_terms(name, x)
            aref = np.abs(ref)
            room = dom & np.isfinite(ref) & ((1 + rel) * aref + ab <= MAXV)
            s = st[name]
            s["checked"] += int(room.sum())
            s["nonfin"] += int((room & ~np.isfinite(r)).sum())
            with np.errstate(all="ignore"):
                err = np.abs(r - ref)
                lhs = err * (1 + 2.0 ** -52) + 2.0 ** -51 * aref
                bound = rel * aref + ab
                ratio = np.where(room & np.isfinite(r), lhs / bound, 0.0)
            s["viol"] += int((room & np.isfinite(r) & (lhs > bound)).sum())
            i = int(np.argmax(ratio))
            if ratio[i] > s["ratio"]:
                s["ratio"], s["at"] = float(ratio[i]), float(x[i])
        print(f"chunk {c} done", file=sys.stderr, flush=True)
    print("\n| function | results differing from the default (all 2^32 inputs) | inputs checked against the fact (with fusion off) | outside | non-finite | largest error / bound |")
    print("|---|---|---|---|---|---|")
    for n in NAMES:
        s = st[n]
        print(f"| {n} | {s['diff']} | {s['checked'] if s['diff'] else '(same bits as the default)'} | {s['viol']} | {s['nonfin']} | {s['ratio']:.4f} ({s['at']!r}) |")
    for label, h in (("default", hd), ("enable_fp_fusion=False", hn)):
        ptx = h.asm["ptx"]
        ops = sorted(set(re.findall(r"^\s*((?:fma|mul|add|sub|ex2|lg2|sqrt|rsqrt)(?:\.[a-z0-9]+)*)\s", ptx, re.M)))
        sass = h.asm.get("sass", "") if hasattr(h.asm, "get") else ""
        print(f"\nPTX float ops, {label}: " + ", ".join(ops))
        if sass:
            sops = sorted(set(re.findall(r"\b(FFMA|FMUL|FADD|MUFU\.[A-Z0-9]+)\b", sass)))
            print(f"SASS float ops, {label}: " + ", ".join(sops) + f"; FFMA count {len(re.findall(r'FFMA', sass))}")


if __name__ == "__main__":
    main()

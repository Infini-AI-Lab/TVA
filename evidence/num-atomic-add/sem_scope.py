"""TVA library evidence (library issues L34, L101): float32 pointer `atomic_add` in every `sem` and `scope`.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/num-atomic-add/sem_scope.py
Output: evidence/num-atomic-add/sem_scope.out (README.md, "Every `sem` and `scope`").

`AtomicAddF32` (docs/rules/num.md) states `P.atomicAdd .binary32` for every `sem` and `scope` of a
float32 `tl.atomic_add`; run.py runs only `sem="acq_rel"` and `"relaxed"` at the default scope
`"gpu"`. This script runs the pointer atomic in all 12 combinations of `sem` (`acquire`, `release`,
`acq_rel`, `relaxed`) and `scope` (`gpu`, `cta`, `sys`), every combination on the same inputs: one draw
per kind of run.py's five (its `pairs`, from its seed; `2^24` lanes per kind, each adding its own `v` to
its own cell). It checks each result against the fact's bound (run.py's `check`) and, bit for bit,
against the `acq_rel`/`gpu` result on those inputs. It also lists the `atom` instruction of each.
"""
import importlib.util
import os
import re
import sys

sys.dont_write_bytecode = True   # run.py is imported below; write no __pycache__ into the folder
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("atomic_run", os.path.join(HERE, "run.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


@triton.jit
def ptr_add(c_ptr, v_ptr, n, B: tl.constexpr, SEM: tl.constexpr, SCOPE: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    v = tl.load(v_ptr + offs, mask=m)
    tl.atomic_add(c_ptr + offs, v, mask=m, sem=SEM, scope=SCOPE)


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}\n")
    torch.manual_seed(20261028)
    n, B = 1 << 24, 1024
    sems = ["acquire", "release", "acq_rel", "relaxed"]
    scopes = ["gpu", "cta", "sys"]
    stats = {(s, c): dict(k=0, viol=0, ratio=0.0, diff=0, rn=True) for s in sems for c in scopes}
    insn = {}
    for kind in ["random", "subnormal", "cancel", "scale", "edge"]:
        o, v = R.pairs(kind, n)
        ref = o.clone()
        ptr_add[(n // B,)](ref, v, n, B=B, SEM="acq_rel", SCOPE="gpu")
        for s in sems:
            for c in scopes:
                cell = o.clone()
                h = ptr_add[(n // B,)](cell, v, n, B=B, SEM=s, SCOPE=c)
                torch.cuda.synchronize()
                insn[(s, c)] = sorted(set(re.findall(r"\batom\.[a-z0-9.:_]+", h.asm["ptx"])))
                vi, r, k, rn, _, _ = R.check(o, v, cell)
                st = stats[(s, c)]
                st["k"] += k
                st["viol"] += vi
                st["ratio"] = max(st["ratio"], r)
                st["rn"] = st["rn"] and rn
                st["diff"] += int((cell.view(torch.int32) != ref.view(torch.int32)).sum())
    print("| sem | scope | instruction | pairs checked | outside the bound | largest error / bound "
          "| normal cases rounded to nearest | bits differing from acq_rel, gpu |")
    print("|---|---|---|---|---|---|---|---|")
    tv = td = 0
    for s in sems:
        for c in scopes:
            st = stats[(s, c)]
            tv += st["viol"]
            td += st["diff"]
            print(f"| {s} | {c} | {', '.join(f'`{i}`' for i in insn[(s, c)])} | {st['k']} | {st['viol']} "
                  f"| {st['ratio']:.4f} | {st['rn']} | {st['diff']} |")
    print(f"\nall 12: pairs outside the bound {tv}; results differing from acq_rel, gpu {td}")


if __name__ == "__main__":
    main()

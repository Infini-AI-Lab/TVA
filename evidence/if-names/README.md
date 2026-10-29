# Names after a runtime `if`

**Question.** `docs/SEMANTICS.md` ("A runtime `if`") said that after a runtime `if` a name has the
taken branch's value when it was defined before the `if` or in both branches, "otherwise the kernel
is rejected on the host". Is a name that only one branch defines rejected even when nothing after
the `if` uses it? Library issue L61 (round 11; raised by lr-8a,
`docs/library-review/round-8/lr-8a/work/if_names.py`; see also `evidence/runtime-if` R5).

**Source.** `primitives/triton-3.8.0-code-generator.py` lines 834-892 (`visit_then_else_blocks`):
after both branches, the names defined before the `if` that a branch changed are merged (each
branch's value must have the name's type), and so are the names new to the `if` that both branches
define (with one type); a name new to the `if` that only one branch defines is not merged, so it is
undefined after the `if` (lines 893-966 build the merge, `cf` or `scf.if`).

**Method.** `run.py` launches 4 programs per case with a runtime condition `flag > 0`, for
`flag = 2` and `-2` (not `1`, which the launcher would make a compile-time constant), each case in
its own process, on GPU 0:

- N1: `y` defined in the then branch only and used there only.
- N2: `y` defined in the then branch only and used after the `if`.
- N3: `y` defined before, reassigned with the same dtype in the then branch only.
- N4: `y` defined before as `int32`, reassigned to a float in both branches.
- N5: `y` new, defined in both branches with one dtype.
- N6: `y` new, `int32` in one branch and `float32` in the other.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/if-names/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`output.txt`).

| case | result |
|---|---|
| N1 | ran: `flag = 2` stores `[100, …, 103]`, `flag = -2` stores nothing |
| N2 | rejected on the host: `NameError('y is not defined')` |
| N3 | ran: `[11, …, 14]` and `[1, …, 4]` |
| N4 | rejected on the host: "initial value for `y` is of type int32[], but the then block redefines it as fp32[]" |
| N5 | ran: `[100, …, 103]` and `[200, …, 203]` |
| N6 | rejected on the host: "Mismatched type for y between then block (int32) and else block (fp32)" |

**Conclusions**, used by `docs/SEMANTICS.md` ("A runtime `if`"):

1. A name new to the `if` that only one branch defines belongs to that branch: it may be used there
   (N1), and a use after the `if` is rejected on the host (N2). It is not rejected otherwise.
2. A name defined before the `if` keeps its dtype in every branch that assigns it, also when both do
   (N4 rejected, N3 merged), and a name new to the `if` that both branches define has one dtype in
   both (N6 rejected, N5 merged).

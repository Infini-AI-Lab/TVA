# The contiguity bound of a value built from a value with a contiguity hint

**Question.** `docs/SEMANTICS.md` ("Compiler hints") reads `tl.multiple_of(x, v)` alone as a
promise about the first element of each group of `C`, with `C` a lower bound of the contiguity
Triton's analysis infers for `x`, from a list (`tl.arange(a, b)` has `b - a`, `x + s` with `s`
constant keeps `x`'s bound, …). Is the list a lower bound when a sub-expression carries a
`max_contiguous` hint, alone or in the pair with `multiple_of`, with a `C` below its own
contiguity? Library issue L58 (round 15; reopened by lr-11a, whose
`docs/library-review/round-11/lr-11a/work/hint_bound_subexpr.py` showed it).

**Sources** (Triton 3.8.0).
- `primitives/triton-3.8.0-AxisInfo.cpp` lines 1232-1259 (`visitOperation`): after a visitor computes
  an operation's contiguity, divisibility and constancy, a `tt.contiguity`, `tt.divisibility` or
  `tt.constancy` attribute on the operation replaces the computed value, whatever the operation.
  `tl.max_contiguous` sets `tt.contiguity` and `tl.multiple_of` sets `tt.divisibility` on the
  operation that defines the hinted value (`primitives/triton-3.8.0-ir.cc` lines 411-425).
- lines 242-279 (`AddSubOpAxisInfoVisitor`): `x + y` gets
  `max(gcd(constancy(x), contiguity(y)), gcd(contiguity(x), constancy(y)))`, so `x + s` with `s`
  constant starts from `x`'s contiguity, the hinted one when `x` carries a hint; a splatted scalar
  is constant along every dimension (553-575). Lines 138-147: a conversion keeps its operand's
  contiguity, and 607-668: an expand and a broadcast keep it along the dimensions they keep.

**Method.** `run.py` loads 512 `int8` elements through `x_ptr + y`, with
`y = tl.multiple_of(a + s, 16)`, `s = 16` and one warp, for six definitions of `a`. The vector of
such a load is at most `min(contiguity, divisibility)` bytes wide, so the PTX shows the contiguity
the analysis gave `y`: 16 bytes (`ld.global.v4.b32`), 8 (`v2.b32`) or 4 (`b32`). Each case runs in
its own process, with its own Triton cache, and checks the loaded values.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/hint-hinted-bound/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`output.txt`).

| case | `a` | loads | values |
|---|---|---|---|
| G1 | `tl.arange(0, 512)` | 16 bytes | correct |
| G2 | `tl.max_contiguous(tl.multiple_of(tl.arange(0, 512), 4), 4)` | 4 bytes | correct |
| G3 | the same with `8`, `8` | 8 bytes | correct |
| G4 | `tl.multiple_of(tl.arange(0, 512), 512)` (no contiguity hint) | 16 bytes | correct |
| G5 | G2's `a`, then `.to(tl.int64)` | 4 bytes | correct |
| G6 | G2's `a`, with `y = tl.multiple_of(s + a, 16)` | 4 bytes | correct |

**Conclusions**, used by `docs/SEMANTICS.md` and `TVA.Lang.Ops` ("Compiler hints": the contiguity
the analysis infers):

1. A value whose defining operation carries a `max_contiguous` hint, alone or in the pair, has the
   hinted `C` as its contiguity in the analysis, and operations on it start from that `C`: `a + s`,
   `s + a` and a conversion of `a` had contiguity `4` (G2, G5, G6) or `8` (G3), not the arange's
   `512`. The list's bound for `tl.arange(0, B) + s` is `B` only for an arange without a contiguity
   hint (G1); a `multiple_of` alone does not change it (G4).
2. So the library's lower bounds take the hinted `C` for a value that carries a `max_contiguous`,
   whatever operation defines it, and apply the list to the values without one. With that, the
   model's `FirstsDiv x k C v` with such a bound covers what the compiler believes. No wrong value
   arose here: the vectoriser takes the smaller of divisibility and contiguity, so the narrower
   contiguity only narrowed the loads.

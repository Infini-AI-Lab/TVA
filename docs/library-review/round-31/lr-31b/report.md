# Library review, round 31, by lr-31b

## Verdicts

- **L139** ACCEPT — SEMANTICS.md ("Parameters and arguments", "A dtype as a `tl.constexpr` argument") states what `jit.py:396-410` and `specialize.cc:515-560` do: a `tl.constexpr` argument is passed as it is, so the parameter is the dtype at each use, and a dtype given to a parameter without annotation has no specialisation handler. My re-run of evidence/front-end-round31 D1–D5 matched its `output.txt` line for line. A kernel reviewer can apply the rule as written: the model writes the dtype.
- **L140** ACCEPT — The entry and `Shape.bcastOk` match `semantic.broadcast_impl_shape` (a scalar is splat, the same rank is required, and each dimension must be equal or `1` in the source) and `validate_block_shape`. `Tensor.broadcastTo`, `get_broadcastTo`, `broadcastTo_cols` and `broadcastTo_rows` check, use only the standard axioms, and the default proof works for literal shapes (work/Check.lean). My re-run of B1–B3 matched.
- **L141** REOPEN — `DotI8` itself holds. My re-run of evidence/num-dot-int8 matched its output. My own tests found 0 wrong outputs out of 1745920 with room (work/dot_int8_extra.txt X1–X4). They covered `b` given `[N, K]` and transposed, `a` transposed, operands narrowed in registers, masked `K` tails, `num_stages` 1 and 4, `wgmma` `n8` and `n256`, and the edge of the room with alternating signs at `K = 65536`. X6 confirms that `acc += tl.dot(a, b)` is fused (mma.sync saturates), as `tl.dot_add` allows. But one statement of the fix is false. SEMANTICS.md (`tl.dot`: "an accumulator must be `int32`, and `out_dtype`, when given with it, `tl.int32`, or the kernel is rejected on the host") and rules/num.md ("The integer `tl.dot`": "an accumulator other than `int32` … rejected on the host") both say an accumulator other than `int32` is rejected. A `uint32` accumulator is accepted, with `out_dtype` omitted or `tl.uint32`. The IR's integers are signless, so it passes the verifier. The result is an `int32` whose value is the accumulator's bits read as `int32`, plus the sum: 256/256 elements, and `3000000000 + 9` gave `-1294967287`, which is `< 0` in the kernel (work/dot_int8_uacc.txt U1–U2, work/dot_int8_extra.txt X5). A prover who trusts the page would call such a launch rejected, so the claim would cover nothing there. Fix: state the `uint32` accumulator, as `tl.bitcast .u32 .i32` of the accumulator followed by the dot with cfg `⟨.i8, .i8, .i32, p, 0⟩`, with evidence. Or put it outside the subset in exclusions.md. Then correct the two sentences and the evidence (I5, conclusion 1).
- **L142** ACCEPT — The `.type` entry matches `core.py` (`tensor.type`; `block_type.element_ty` is the scalar type; a `dtype` has no `element_ty`). The pointer condition matches `semantic.cast` to `int1` (`ptrtoint` to `i64`, then `!= 0`) and the launcher's null pointer for an empty tensor. My re-run of P1–P3 matched.
- **L143** ACCEPT — The entry matches `semantic.cast` and `semantic.bitcast`: the same dtype returns `x` before the rounding mode is read; a rounding mode is allowed only on a float narrowing; `bitcast` needs the same width, and a pointer goes through `cast`. `tl.bitcast` and `tl.bitcast_self` check, with the standard axioms only. My re-run of C1–C4 matched.

## New issues

None.

## Re-ran

- evidence/num-dot-int8 (I1–I6) on an H200 NVL: matched `output.txt` line for line (work/rerun-num-dot-int8.txt).
- evidence/front-end-round31 (D1–D5, B1–B3, P1–P3, C1–C4): matched line for line (work/rerun-front-end-round31.txt).
- Tests I added: work/dot_int8_extra.py (X1–X6), work/dot_int8_uacc.py (U1–U2), and work/Check.lean (axioms and the default proof of `broadcastTo`).

## Not checked

- `lake build` of the whole library (not run here; the author reports it green). `tvr release` was not run (the coordinator runs it for the round).
- Lean proofs other than by `#print axioms` and the small examples in work/Check.lean.

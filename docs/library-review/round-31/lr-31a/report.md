# Library review, round 31, by lr-31a

## Verdicts

- **L139** ACCEPT — "A dtype as a `tl.constexpr` argument" agrees with `specialize.cc:515-560` and `jit.py:396-410` (a `tl.constexpr` argument is passed as it is; a dtype to any other parameter has no handler). The re-run of evidence/front-end-round31 D1–D5 matched `output.txt` line for line. My `work/probe.py` K1 adds `tl.uint8` and `tl.int1` as the parameter in `.to(D)` and `tl.zeros(dtype=D)`: the same values as the literal dtype. The two rejections are stated, and the subset ("Parameters and arguments") names the dtype argument. No Lean constant is involved, so "None" is the right reach.
- **L140** ACCEPT — `Shape.bcastOk` is `broadcast_impl_shape`'s check (`semantic.py:711-725`): same rank, and each dimension equal or `1`. `Tensor.broadcastTo` reads `0` on the size-1 dimensions, as `tt.broadcast` does. A scalar is `splat`. The page states the power-of-two and `2^20` limits of `validate_block_shape`. The re-run of B1–B3 was identical. `work/Probe.lean` checks usability: the default `by decide` proof works for literal shapes; for a `tl.constexpr` size the prover gives `bcastOk [m, 1] [m, n]` by `simp`, and `broadcastTo_cols` then applies; a rank change and a differing dimension are refuted by `decide`. The new constants are listed.
- **L141** ACCEPT — `DotI8` is `DotAcc` with `rel = abs = 0` at `⟨.i8, .i8, .i32, p, 0⟩`, and `DType.bound .i32 = 2^31 - 1`, so the room is the stated one. `Platform.sm90ref.dotI8` uses only the standard axioms (`work/Probe.lean`). `semantic.py:1483-1486` confirms that only `int8` is allowed and the result is `int32` whatever `out_dtype` says. The re-run of evidence/num-dot-int8 was identical to `output.txt`. My T1 adds `tl.trans(bT)` and `bT.T` operands, `acc += tl.dot(…)`, `num_stages` 2–4 and `wgmma` widths up to `n256`, with an output exactly at `|acc| + Σ|a·b| = 2^31 - 1`: 0 of 448512 outputs differ. The fact is listed alone in "Facts the library states", as it should be (`Platform → Prop`).
- **L142** ACCEPT — `tensor.type`/`.dtype` (`core.py:884-890`), `pointer_type.element_ty` and `block_type.element_ty` agree with the page. `if p:` is `ptr_to_int` then `!= 0` (`semantic.py:900-906`, `visit_If`). The re-run of P1–P3 was identical. My E1 confirms the pitfall "an empty tensor's address is `0`" for torch 2.11 also for empty views of a non-empty tensor (`base[:0]`, `base[5:5]`): `data_ptr()` is `0`, and the branch is not taken.
- **L143** ACCEPT — `core.cast` and `semantic.bitcast`/`cast` (`semantic.py:793-916`) match the entry. Casting to the same dtype returns its input before the rounding is parsed, so it holds "whatever `r`": my R1 accepted `"rd"` on `float32 → float32`. `tl.bitcast` is `ofBits ∘ toBits`, and `DType.bitwidth` is `primitive_bitwidth`. In `work/Probe.lean`, the default width proof works and the `f32 → i32 → f32` round trip is provable. The re-run of C1–C4 was identical. R1 also showed that `tl.cast` of Python numbers with `bitcast=True` reads their own type's bits (`1.0` gives `0x3f800000`).

## New issues

- **NEW** wording — The `x.to(d)` / `tl.cast` entry of `docs/SEMANTICS.md` says a `fp_downcast_rounding` is rejected "on a conversion that is not a float narrowing", but it does not say what a narrowing is. Triton's test is a float target with fewer bits (`semantic.py:824-832`), so `bfloat16 ↔ float16` (a rounding conversion of equal width) is rejected with `"rtne"` as well as with `"rtz"` (`work/probe.py` R1). Fix: state "to a float dtype of fewer bits", and add `bfloat16 ↔ float16` to the examples.
- **NEW** wording — "A tensor's `.type`" (`docs/SEMANTICS.md`, Values) does not say that a block's `.type` is not a dtype of `tl`. L139's entry allows a dtype "wherever this page admits a dtype of `tl`", but a block type works only in `blk.to(x.type)`: `scalar.to(x.type)` and `tl.zeros(…, dtype=x.type)` are rejected on the host (`work/probe.py` Y1). Fix: state that only a scalar's `.type` (its dtype) and a block type's `.element_ty` may be used where a dtype is admitted.

## Re-ran

- `evidence/front-end-round31/run.py`: identical to `output.txt` (`work/rerun-front-end-round31.txt`).
- `evidence/num-dot-int8/run.py`: identical to `output.txt` (`work/rerun-num-dot-int8.txt`).
- `work/probe.py` (E1, R1, T1, K1, F1, Y1; output `work/probe-output.txt`), on an H200 NVL, GPU 0, Triton 3.8.0, 2026-11-08. F1 (survey) found the `float64 → float16/bfloat16/float32` conversions single-rounded on 36 double-rounding traps and 4096 random values (`cvt.rn.*.f64`), as "Float conversions" states.
- `work/Probe.lean` with `lake env lean`: no errors; axioms `propext`, `Classical.choice`, `Quot.sound`.

## Not checked

- `tvr release` (run by the coordinator for the round).
- `DotI8` with 3-D (batched) operands, and with `int8` operands from a tensor descriptor.

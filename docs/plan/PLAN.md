# Library plan (v2)

Goal (user, 2026-10-26): "re-design, re-organize, enrich our library to describe and formalize
triton programs … let's not focus on the concrete kernel proof, let's work on the foundation
library and lemma construction."

## Done (2026-10-26)

- v1 archived outside this repository; nothing here depends on it.
- Language: scalars, tensors, memory, platform, `Prog`, the `tl.*` API, including atomics
  (`add`, `xchg`, `cas`, `max`, `min`, `and`, `or`, `xor`), the fusion forms (`add`, `mul_add`,
  `addc`) and tree reductions.
- Semantics: `Step`, `Launch` (SC interleaving, inter-program races), `WellBehaved`,
  `Outcomes`; no barrier semantics (user decision).
- Claim and rule: `Meets`, `Rules.Compiler.Faithful`.
- Proof kit:
  - `Safe` with `bind`/`mono`/`transfer`/`acc`;
  - per-op rules;
  - the 1-D block kit;
  - integer no-overflow lemmas;
  - `independent` and `meets_independent`.

  All of it is proved, with only the standard axioms.
- Numerics: `roundRat_rn_err`; the error lemmas `FloatV.add/mul/fma/divRn/cvt_err`;
  `Rules.Num` cards (totality and accuracy of `P`).
- Reductions: `tl.sum` (IEEE nodes) vs `tl.sumc` (fusable); `Tree.eval_err`, `sum_err_covers`.
- Generic buffer accesses (`Access`): `Buf.Addrs`, `addrs_of_get`, `safe_load_buf`,
  `safe_store_buf`; the 1-D block kit rebased on it.
- Shared atomics: `SafeA`, `shared`, the scalar atomic rule, integer accumulation
  (`replay_add`, `sum_proj`).
- Library review (2026-10-26) soundness batch: NaN payloads, signed zeros, fusion with
  overflow, `acq_rel`-only race exemption, compiler hints as checks, typed loop bounds,
  `dot_add`, integer `//`/`%`/shifts, `debug`.
- Tensor descriptors (TMA): `desc.load`/`store` (zero or NaN padding), `desc.atomic_*`,
  `desc.gather`/`scatter`, with rules (`safe_atomic_buf` for owned elements).
- Block pointers (`make_block_ptr`, `advance`, `boundary_check`, `padding_option`) as Triton's
  frontend lowers them, with rules.
- `tl.dot_scaled` (platform relation over unpacked fp4/fp8/bf16/fp16 bits and `e8m0` or
  float scales, group `16` or `32`; `DotScaledTotal`/`DotScaledAcc`).
- Review questions Q1, Q2, Q8, Q12, Q14 settled from the pinned 3.8.0 compiler sources
  (review-questions S5–S9): reverse scans narrowed to the lowered order, `tl.dot_scaled_add`
  and `tl.sum_bcast_mul` for the combine pass, float scales for `dot_scaled`.
- Q6 settled (S10): `/` accuracy rules `DivFull` (fp32 `div.full.f32`) and `DivRn` (fp64),
  with `DivRn.divAcc` and the satisfiability lemma `DivFull.of_divRn`.
- Second library review (review-questions S11, S12):
  - launch validity and consistency witnesses (`Launch.Valid`, `Claim.Consistent`,
    `exists_total_faithful`, `Platform.ref`);
  - no synchronises-with between programs;
  - conversions (`cvt` modes, `cvtFrom8`, `itof` double rounding);
  - shifts (`ashr`/`lshr`), `RangeOk`;
  - descriptor block shapes;
  - `dot_scaled` dtypes and a per-output `DotScaledAcc`;
  - `MathAcc` guards and `MathAt`;
  - `maxAll`/`max0`;
  - Philox `randint`/`rand`.
- Witnesses for every numeric rule (`Platform.ref`, the most accurate platform): `MathAcc` with
  no absolute term in the normal range (`Format.roundRat_rn_err_normal`), a new generic
  `tl.dot` rule `Rules.Num.DotAcc` (float and integer accumulators), and correctly rounded
  `tl.dot`/`tl.dot_scaled` witnesses (`dotAcc`, `dotAcc_int`, `dotScaledAcc`).
- `tl.join`/`split`/`interleave`/`reshape` (exact; `interleave` proved equal to Triton's
  definition) and `reshape(can_reorder=True)`.
- Scans (`associative_scan`, `cumsum`, `cumprod`) with exact integer prefix sums and a float
  error bound along any leaf list (`sum_err_list`); `tl.sort` with a proof that a sorted
  result exists for integers and floats; `tl.histogram`.
- Compiler hints for any rank and for pointer tensors.
- Library test: `VecAdd`, end to end.
- Docs: DESIGN, SEMANTICS, LIBRARY, PROVER_GUIDE, REVIEWER_GUIDE, the rule page, exclusions,
  review questions.

## Now (user, 2026-10-27): the library, the protocol and the interface

User: "let's stop our work proving kernels. let's strengthen our library, protocol and
interface." The three kernels (`kernels/`) stay as they are: round 1 is recorded, and their
gates show what they need from the library.

1. **Library batch 1**: the issues L1–L6 of [library-review/ISSUES.md](../library-review/ISSUES.md).
   - Hints in loop bodies (L1) and on rank ≥ 2 tensors (L2).
   - Rejected launches outside the claim (L3).
   - Device facts for sm_90 with Triton 3.8.0, on TVA's own evidence (L4): `exp2`, `log2`, `/`
     in `float32`, and `tl.dot` from `float16`/`bfloat16` into `float32`.
   - Descriptors built in the kernel (L5).
   - `tl.static_assert` (L6).

   Then an independent library reviewer checks the batch ([LIBRARY_REVIEW.md](../LIBRARY_REVIEW.md)).
2. **Device-fact pages**: one per target, listing each fact's operation, format, lowering,
   constants and evidence. Kernels name their target in `kernel.json`, and the gate checks that
   each cited fact is the target's.
3. **Transcription rules with ids**: each rule of [TRANSCRIPTION.md](../TRANSCRIPTION.md) gets
   an id, and each model-map entry names the rule it applies. A kernel reviewer then checks
   each entry against one rule.
4. **Library review in the tool**: `tvr` commands for library issues and library rounds, the
   same interface as kernel review (a ledger, filings, a generated status page).
5. **Coverage audit of the library**: every `tl.*` construct Triton 3.8.0 documents has a
   meaning and a transcription rule, or is in [exclusions.md](../exclusions.md).

## Next, in order

1. **Rest of the numeric calculus.**
   - An `AddRel` bound. This needs a half-ulp lemma for every rational that rounds to a finite
     value, including values just above `maxVal`.
2. **A structural `tl.dot` card**: exact products and a summation error bound over all orders,
   per input precision, from which the constants of `Rules.Num.DotAcc` follow.
3. **2-D tiles.** The `Block` kit for `offs_m[:, None] * stride + offs_n[None, :]` with a 2-D
   mask, plus `tl.dot` loops over `K` (the GEMM pattern). Also the row-wise pattern of softmax
   and layernorm (`tl.sum1`, `tl.max1` over a row tile).
4. **More shared-atomic kit.**
   - A tensor-atomic rule (`SafeA` for `atomicWith` over distinct lanes, in any order).
   - Float `atomic_add` accumulation bounds over all orders, from `Rules.Num.AtomicAddAcc`.
   - `max`/`min` accumulation.
   - A progress rule (`Rules.Progress`) for programs that wait on each other. It needs a
     meaning for `tl.debug_barrier` first: a release publishes only what the barrier before
     it covers (S11). Tutorial 05's lock is the test case.
   - Footprints that depend on the values atomics return, for counters that hand out work.
5. **Shared relaxed reductions**: a semantics for elements that several programs reduce into
   with relaxed atomics or TMA reductions (per-element orders), so that split-K kernels with
   `desc.atomic_add` are covered (Q10).
6. ~~**Tooling**: a checker for the bundle shape.~~ Done: the gate of `tools/tvr.py`
   ([REVIEW_PROTOCOL.md](../REVIEW_PROTOCOL.md) §8).
7. **A reviewer pass over the semantics** ([review-questions.md](../review-questions.md)).

Kernel proofs beyond library tests wait until (1)–(4) exist.

# r-fa, round 1: report

(Recorded by the coordinator, verbatim: the reviewer's final message. From round 2 on, reviewers
write this file themselves.)

**Verdict: not accepted yet.** I filed 5 blocking findings, F1–F5. The filing is in
`bundles/flash_attn_fwd_f16/review/inbox/r-fa-1.json` and `tvr` recorded it. Two assumptions are
false for the real device, the spec says nothing about O at some covered sizes, and some covered
instances may not compile.

**Accepted:** model, map, conditions, dotTotal, dotAcc, exp2Total, exp2NegInf, log2Total,
divTotal, divFull. I checked the model line by line against the source and against Triton 3.8.0
sm_90 compiles of this exact file in the user's cache (`~/.triton/cache/*/_attn_fwd.*`). Every
operation, constant, fusable/IEEE choice, the grid and the descriptors match.

**Blocking findings:**
- **F1, ASSUMPTION_FALSE, `exp2Acc`:** `tl.math.exp2` does not compile to `ex2.approx.f32`.
  - It goes through libdevice `__nv_exp2f`, and with Triton's default `enable_reflect_ftz=True`
    the PTX is `ex2.approx.ftz.f32` (18 of these, 0 of the non-ftz form).
  - That instruction flushes results below 2^-126 to +0. At x = −127 the error is 2^-127, far
    above the assumed `2^-22·2^x + 2^-149`, and the kernel does reach such inputs.
  - Fix: set `absE = 2^-126` (another project's `Ex2FtzCard`) and correct the docstrings. The error
    constants barely change.
- **F2, ASSUMPTION_UNCLEAR, `log2Acc`:** `tl.math.log2` compiles to libdevice `__nv_log2f`, a
  polynomial, not `lg2.approx`. The Lean bound is still true because CUDA documents 1 ulp for
  it. Only the stated lowering and the cited evidence are wrong.
- **F3, SPEC_TOO_WEAK, spec:** `Bnd.Ok` admits realistic instances where `errO > V`, so Post
  does not constrain O there (an all-zero O passes).
  - At HD=128, BN=128, R=8, N=131072 (a 128K context) with sm_scale = 1/√128: errO = 4.60
    against V = 4.
  - For HD=128, R=4, errO/V grows from 2% at 1K to 16% at 64K and 31% at 128K.
  - At the edge of `Bnd.Ok`, errO is about 3V.
  - Fix: add a meaningfulness condition to `Bnd.Ok` or tighten the accumulation bound.
- **F4, EVIDENCE_REQUEST, `faithful`:** HD=256 with BN=128 needs at least 256 KB of shared
  memory for the double-buffered k and v tiles. That exceeds the 227 KB sm_90 limit for every
  config the tutorial offers. A cached compile at BM=BN=HD=128 uses 163,872 B, which confirms two
  buffers each. HD=256 with BM=128 at 4 warps needs 256 accumulator registers per thread. Fix:
  compile logs, or narrow `Inst.ok`.
- **F5, ASSUMPTION_FALSE, `faithful`:** on the tutorial's default Hopper path
  (`warp_specialize=True`), the kernel builds its descriptors itself. It writes 512 B per
  program to global scratch, which the model never touches. So Faithful and Post's "nothing else
  changes" clause are false as written. The best fix is a library statement that launch scratch
  is outside the memory compared; the alternative is narrowing the bundle to host-side
  descriptors.

**Non-blocking:**
- F6 (DOC): the `Oref` docstring should say c is the device's rounded scale, not exactly
  `sm_scale·log₂e`.
- F7 (QUESTION): `dotAcc` rests on another project's bf16 wgmma card, which has about a 4× margin. fp16 is
  not measured.
- F8 (DOC): the drop reason given for `tl.static_assert` (map line 190) does not fit it.

**Not checked:**
- The `Numeric.lean` derivations and the consistency witness (Lean checks those).
- MLIR's pattern priority in its source; I relied on the cached compile output instead.
- Whether the HD=256 instances actually compile.
- fp16 wgmma accuracy.

My script for the error constants is `work/err.py`.

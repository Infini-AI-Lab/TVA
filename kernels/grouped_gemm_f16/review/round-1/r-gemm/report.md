# r-gemm, round 1: report

(Recorded by the coordinator, verbatim: the reviewer's final message. From round 2 on, reviewers
write this file themselves.)

**Verdict: not accepted.** The filing (`bundles/grouped_gemm_f16/review/inbox/r-gemm-1.json`)
has 3 blocking and 3 non-blocking findings. `tvr` refuses every acceptance while a `lib:` finding
is open, so I accepted nothing. Without it I would have accepted `dotTotal`, `dotAcc` and `spec`.

**Blocking**
- **F1 MODEL_MISMATCH `map:148`.** The two `tl.multiple_of(a_ptrs/b_ptrs, [16, 16])` hints are
  not no-ops, so dropping them as `no-op:block-arg-hint` is wrong.
  - Triton's frontend runs the loop body once as a dry run (`_find_carries`,
    code-generator.py:465-473) while the names still hold their pre-loop values.
  - So the hint sets `tt.divisibility` on the pre-loop `tt.addptr` (ir.cc:413). The attribute
    survives later passes, AxisInfo reads it, and the loop-carried pointers inherit it.
  - The compiled loads are therefore 16-byte vector or async copies. Without the hint they
    would be 2-byte loads.
  - Counterexample inside `Inst.ok`/`Pre`: config (64, 64, 32), M = N = 64, K = 32,
    **lda = 36**. Row 1 of A starts at byte 72 ≡ 8 (mod 16), so the device faults on a
    misaligned access, and `faithful` fails on that instance.
  - Fix: write the hints as checks at the pre-loop definitions, and add `16 ∣ a.base, b.base`
    and `8 ∣ lda, ldb` to the conditions. The host's normal use meets these.
- **F2 LIBRARY_SEMANTICS `lib:docs/SEMANTICS.md`.** The library rule "a hint on a loop-carried
  variable is a no-op" misses the dry run. The same rule appears in REVIEWER_GUIDE, Ops.lean
  and the protocol's drop list.
  - Once the hint counts, the library's reading of `[16, 16]` is false along dim 0
    (contiguity 1, so every element would have to be 16-byte aligned).
  - Under that reading tutorial 08 is undefined behaviour on every input, although 3.8.0 only
    uses the hint's alignment along the fastest axis.
  - The library must choose between a reading at the level of what the compiler uses and
    declaring such kernels out of scope. F1's fix depends on that choice.
- **F3 CONDITION_WRONG `item:conditions`.** `Inst.ok` covers block sizes that do not compile,
  so `faithful` is asserted for launches that fail.
  - BK < 16 fails: `tl.dot` needs K ≥ 16 for 16-bit operands (nvidia compiler.py:19-33).
  - Very large tiles fail: BM = BN = BK = 512 needs at least 512 KiB of shared memory, far
    above sm_90's 227 KiB.
  - Fix: restrict to the four autotune block configs, or require BK ≥ 16 with upper bounds
    backed by a compile log.

**Non-blocking**
- **F4 DOC (write-up).** At the largest covered K the error bound is about 0.65·S. From
  K ≈ 12k (BK = 32), one dropped k-step of nonnegative data stays inside the bound. Wrong
  kernels are still rejected on smaller instances.
- **F5 DOC `spec`.** `Post`'s frame leaves C's padding columns (ldc > N) unprotected, and C
  disjointness is stated per buffer, not per cell. The host never produces either case.
- **F6 QUESTION `dotAcc`.** The only tensor-core accuracy data (another project) is for bf16, not fp16.
  The constant has about 7x margin over that data, but an fp16 run, including fp16 subnormal
  inputs, is missing.

**Checked:** every model line against the source, the launch and its block shapes, `#print`
with full names (all `TVA.*`), and `#print axioms` (only `propext`, `Classical.choice`,
`Quot.sound`). I computed the bound at the covered sizes with `work/errc.py`.

**Not checked:** I did not run Triton, so I have no compiled IR or PTX showing the 16-byte
copies; the F1 chain is read from source. I did not read the proof files.

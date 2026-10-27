# r-vecadd, round 1: report

(Recorded by the coordinator, verbatim: the reviewer's final message. From round 2 on, reviewers
write this file themselves.)

Verdict: I accepted 4 of the 5 items: `model`, `map`, `conditions` and `spec`. I did not accept
`assumption:faithful`, because one blocking evidence request on it is open. My filing is
recorded in `bundles/vec_add_f32/review/inbox/r-vecadd-1.json`. No certificate yet.

**F1 (EVIDENCE_REQUEST, blocking, `item:assumption:faithful`)**
- `Inst.ok` covers every power-of-two `BLOCK` up to `2^20`, but the host only ever uses 1024.
  With the default `num_warps=4`, `BLOCK=2^20` means each thread holds 8192 f32 values per
  tensor, plus pointers and masks. That likely spills heavily to local memory, so compiling
  (ptxas) or launching could fail.
- A failure counts as a `fault`, which makes `Faithful` false for that instance.
- Because `faithful` is stated for all covered instances, one failing instance makes
  `Assumptions K P` false for the real device. `main` then says nothing about any instance,
  BLOCK=1024 included.
- Resolve by either:
  - a Triton 3.8.0 compile log (ptxas -v: registers, spills, local bytes) plus a successful run
    at `BLOCK=2^20`; or
  - narrowing the block bound to what is shown to compile, and listing that narrowing in
    `vec-add.md`.

**F2 (DOC, non-blocking, `decl:TVAExamples.VecAdd.Inst.ok`)**
- The bound `n + BLOCK ≤ 2^31` is stricter than int32 needs. Since `BLOCK` divides `2^31`, the
  offsets fit for every `n < 2^31`; for example `n = 2^31−1` with `BLOCK = 1024` works, and Lean
  confirms it is excluded.
- So the docstring's reason ("offsets do not overflow") is wrong, and the write-up's
  "Narrowings: None" is not exact. Fix by relaxing to `n < 2^31`, or by rewording the docstring
  and listing the narrowing.

**Checked and found correct:**
- Every name in the model resolves to the library (printed with full names), and the bundle
  defines no local instances.
- `main` and `consistent` use only the three standard axioms.
- The launch: grid is cdiv, `debug=false`, `numCtas=1`, and `[[BLOCK]]` is the source's only
  block shape.
- Every source line matches its model line: int32 arithmetic, signed compare, sign-extended
  offsets, no `other` on the loads, and IEEE `tl.add` since no product feeds the `+`.
- `FloatV.add` gave the right IEEE results on spot checks (`#eval`): ties-to-even, subnormals,
  signed zeros, overflow, inf−inf.
- Triton does not flush subnormals on NVIDIA.
- Edge cases: Triton treats `n=1` as a constant, which gives the same compare, and its launcher
  skips `n=0` (empty grid).
- `Post` fixes every output up to NaN payload, and leaves every byte outside `out` unchanged.

**Not checked:** I could not compile or run the kernel on a GPU, so whether `BLOCK=2^20` really
compiles and launches is the open question in F1.

# Exclusions

These Triton features are hard to give meaning to, rarely used, or both. A model may not use
them; a kernel that needs one is out of scope until the library covers it.

| Feature | Why excluded | What a model does instead |
|---|---|---|
| `tl.debug_barrier` semantics | the language has no threads; user: "we do not need debug semantics here" | treat the call as a no-op |
| `tl.device_print`, `tl.device_assert`, `tl.static_print` (and Python's `print` and `assert`) as debugging aids | no effect on global memory without `debug` | drop the call when its arguments are pure, otherwise transcribe the argument operations ([TRANSCRIPTION.md](TRANSCRIPTION.md) §10) |
| `tl.inline_asm_elementwise` | arbitrary PTX | out of scope |
| `float8e4b15`, AMD fp8 formats (`float8e4b8`, `float8e5b16`) | not used on the target GPUs | out of scope |
| pointers to pointers (`tl.pointer_type` of pointers) | rare; needs a typed pointer memory | out of scope |
| `range` with a non-positive step | rare; not modelled yet (constant negative steps follow Python's semantics, `code_generator.py:1246-1251`) | `tl.forRange` is undefined behaviour there |
| `range` with unsigned bounds at or above `2^(w−1)` | `scf.for` compares signed | `tl.forRange` is undefined behaviour there (`tl.RangeOk`) |
| descriptor reductions (`desc.atomic_*`) into elements other programs also reduce into | TMA reductions are relaxed: they race in the semantics (an SC interleaving would hide their reorderings) | out of scope; covered when one program owns the elements ([review-questions.md](review-questions.md) Q10) |
| pointer and descriptor accesses to the same bytes | proxy ordering not yet confirmed ([review-questions.md](review-questions.md) Q11) | out of scope |
| 3-D (batched) `tl.dot_scaled` | only the 2-D form is in the library | ask for the operation in the library |
| `tl.histogram` with an enabled value outside `[0, bins)` | undocumented | undefined behaviour in `tl.histogram`: the prover shows values are in range |
| `desc.scatter` with an out-of-bounds lane | undocumented | undefined behaviour in `tl.desc_scatter` |
| programs that *wait* on each other (locks, spin loops, cooperative grid sync) | no progress rule yet | planned ([plan/PLAN.md](plan/PLAN.md)); accumulation through atomics is covered by `TVA.shared` |
| data passed between programs through plain memory behind a flag or counter ("the last program reduces the partial tiles", semaphore split-K, a lock) | a scalar atomic is issued by one thread with no barrier before it, so its release does not publish the other threads' writes; the semantics has no synchronises-with, so such launches race | out of scope |
| a counter that hands out work (`tile = atomic_add(cnt, 1)`) | `TVA.shared` needs footprints fixed in advance | out of scope until footprints may depend on the values the atomics return |
| `num_ctas > 1` | a program is a cluster of CTAs; `cta`-scope atomics are atomic only within one CTA | `Launch.Valid` requires `num_ctas = 1` |
| the 64-bit Philox (`uint64` offsets as counters) and `tl.randn` | not modelled | `tl.randint`/`tl.rand` cover the 32-bit generator |
| `cta`-scope atomics used across programs | atomicity only within a CTA | they race; the launch is not well behaved |
| lanes that hold LLVM poison (`x.to(int)` out of range, a shift by the width or more) but whose values are never used | poison is read conservatively as undefined behaviour | the model is undefined there; a kernel that converts unspecified masked-off lanes needs `other` or a `tl.where` before the conversion |
| relaxed / acquire-only / release-only atomics on memory other programs touch | an SC interleaving would hide their reorderings | they race unless ordered ([review-questions.md](review-questions.md) Q10) |
| `debug=True` (`TRITON_DEBUG`) | asserts and overflow checks trap | `Compiler.Faithful` covers `debug = false` only |
| aggregate classes (classes made with Triton's `_aggregate`, `triton.language.core`) | their operators, subscripts and methods are `@triton.jit` functions Triton inlines, so an operator on them may run any operation; no transcription rule covers them | out of scope; an expression with an aggregate operand is not pure ([TRANSCRIPTION.md](TRANSCRIPTION.md) `[pure]`) |
| programmatic dependent launch (`launch_pdl=True`) | the kernel may start before the previous kernel on the stream has finished writing, so its programs may read memory other than the claim's `init` | `Launch.Valid` requires `pdl = false`; `gdc_wait` and `gdc_launch_dependents` are then no-ops (`no-op:gdc`) |
| a Triton environment variable that changes compilation or launch (`TRITON_F32_DEFAULT`, `TRITON_DEFAULT_FP_FUSION`, `TRITON_INTERPRET`, …) | it changes defaults the model fixes, or replaces the compiler's output | `Compiler.Faithful` covers Triton's environment at its defaults only ([rules/compiler-faithful.md](rules/compiler-faithful.md)) |

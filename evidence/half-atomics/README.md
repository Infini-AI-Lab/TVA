# 16-bit atomics on sm_90 (library issue L172)

Library evidence (not a kernel's), for library issue L172 (round 38, from library review lr-37b's
`work/half_atomics.py` and `work/sass_dump.py`, round 37). Triton 3.8.0 (TVA's `.venv`, torch
2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-11-08. Scripts: [run.py](run.py), output
[out.txt](out.txt) (under a minute; identical to lr-37b's output); [sass_dump.py](sass_dump.py), run
with `float16` and `bfloat16`, outputs [sass_float16.out.txt](sass_float16.out.txt) (identical to
lr-37b's) and [sass_bfloat16.out.txt](sass_bfloat16.out.txt). The SASS of a 16-bit `atomic_cas` is
in [evidence/atomic-dtypes](../atomic-dtypes/README.md) (`adv.py`, part S).

**Question.** docs/review-questions.md Q7 said that two contiguous `float16` lanes may become one
32-bit `f16x2` atomic and that the pre-sm_90 `bfloat16` atomic is a 32-bit compare-and-swap loop.
What do `float16` and `bfloat16` `atomic_add` compile to on the target, and does any form give a
wrong value, alone or next to another program's plain stores to the neighbouring halves?

**The source.** `nvidia-LoadStoreOpToLLVM.cpp:636-646`: `supportsVectorized` is true for a float
add of `float16`, `bfloat16` or `float32` where the target supports vector atomics (sm_90);
`729-740`: the vector width is the pointer's contiguity (limited by the mask's alignment), and only
where vector atomics are not supported are two `float16` lanes packed into one `f16x2` atomic.

**Method.**
- `sass_dump.py`: the SASS of a 64-lane `atomic_add` (one lane per thread, result unused).
- `run.py`, P: a 1024-lane `float16` or `bfloat16` `atomic_add` (8 contiguous lanes per thread),
  unmasked and with a mask that disables odd lanes, with the result unused and stored: the PTX and
  SASS atomics, the new values against one IEEE addition per lane (the exact sum in `float64`,
  rounded once), the returned values against the old ones.
- `run.py`, N: 64 programs; programs 0 to 31 add `1` (`sem="relaxed"`, result unused) `R` times to
  each even half of a 2048-half buffer (one lane per half, so each is its own compare-and-swap
  loop); programs 32 to 63 store plain values `R` times into the odd halves, each odd half written
  by one program. The model has no race (other bytes). Each even half must be `32·R` (`R = 64` for
  `float16`, `8` for `bfloat16`, so every partial sum is exact) and each odd half the last value
  stored. 20 launches per format.

**Result.**
- A lane on its own (one per thread, or masked) is `atom.global.gpu.acq_rel.add.noftz.f16`
  (`.bf16`), which `ptxas` compiles to a loop: load the aligned 32-bit word (`LOP3 … 0xfffffffd`),
  add into its half (`HADD2`, `HADD2.BF16_V2`), merge (`PRMT`), `ATOMG.E.CAS.STRONG.GPU`, retry
  while the word changed. The neighbouring half is written back with the value read.
- Unmasked, 8 contiguous lanes per thread: `atom.global.gpu.acq_rel.add.noftz.v8.f16` (`.v8.bf16`),
  SASS `ATOMG.E.ADD.F16` (`.BF16`).
- P: no lane's new value differs from one IEEE addition (0 of 1024 in each of the 8 cases), and the
  returned value is the old one on every enabled lane.
- N: 0 of 20480 even halves and 0 of 20480 odd halves wrong, per format.

**Conclusions.** On sm_90 a `float16` or `bfloat16` atomic add is, per lane, a 32-bit
compare-and-swap loop on the word holding the half, or, for contiguous lanes one thread holds, one
vector atomic; no `f16x2` packing. Neither form gave a wrong value, also with another program's
plain stores into the neighbouring halves. The model's 2-byte `Event.Strong` events are not the
device's accesses, so this remains part of the compiler trust (review-questions.md Q7); the test
supports, but does not prove, it.

# Float `atomic_add` on `float32` (`AtomicAddF32`)

Library evidence (not a kernel's), for library issues L34 and L101. Triton 3.8.0 (TVA's `.venv`,
torch 2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-10-28. Scripts: [run.py](run.py),
[sem_scope.py](sem_scope.py) (round 20); outputs: [out.txt](out.txt),
[sem_scope.out](sem_scope.out).

**Question.** `P.atomicAdd .binary32` gives the new value of a `float32` cell after a
`tl.atomic_add` (pointer atomics) or a descriptor's `atomic_add` (the TMA reduction). The library
states `AtomicAddF32` ([rules/num.md](../../docs/rules/num.md), "Float `atomic_add`"): for a
finite old value `o` and operand `v` with `|o + v| ≤ maxVal`, the new value is finite and within
`2^-23·|o + v| + 2^-124` of `o + v`. Does the device meet it, on both instructions, and how close
does it come to the tight bound `2^-24·|o + v| + 2^-125` (rounding to nearest with subnormals
flushed)?

**What Triton compiles them to** (the PTX in `out.txt`):
- pointer `tl.atomic_add` with `sem="acq_rel"` (the default) and `"relaxed"`:
  `atom.global.gpu.acq_rel.add.v4.f32` and `atom.global.gpu.relaxed.add.v4.f32` (four lanes per
  instruction, each its own cell);
- descriptor `atomic_add`: `cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.bulk_group`;
- a scalar `tl.atomic_add(p, 0.0, sem="relaxed")`: a strong load at the atomic's scope and order,
  `ld.global.gpu.relaxed.b32` (Triton's `LoadAcquireOp`,
  `primitives/triton-3.8.0-nvidia-LoadStoreOpToLLVM.cpp:648-687, 762-806`), not a plain `ld`; no
  store, so memory keeps its value (`-0.0` stays `-0.0`). (An earlier version said "a plain
  `ld`"; library issue L66.)

**Method.** `2^24` lanes per run, each adding its own `v` to its own cell `o` (distinct
addresses, so the result of each lane is one `P.atomicAdd` step), for five kinds of inputs: random
bit patterns, subnormals, near-cancellations (`v = -o·(1 + k·2^-23)`), random magnitudes over
the whole range, and sums near `maxVal`; the pointer atomic with both orders and the descriptor
atomic each get fresh inputs. The reference `o + v` is computed in `float64`; its rounding error,
at most `2^-53·|o + v|`, is added to the measured error (`2^-52·|o + v|` is added). Every pair of
finite `o`, `v` with `|o + v| ≤ maxVal` is checked: the new value finite, and within the bound.
The script also checks, where `o`, `v` and `o + v` are normal and the `float64` sum is exact,
that the new value is the `float32` rounding to nearest of `o + v`; and it runs eight special
pairs (subnormal operands and results, signed zeros, overflow) on both instructions.

**Result** (`out.txt`).

| Instruction | Pairs checked (all kinds) | Outside the fact | Largest error / fact's bound | Outside the tight bound | Largest error / tight bound | Normal cases rounded to nearest |
|---|---|---|---|---|---|---|
| pointer, `acq_rel` | 74989473 | 0 | 0.500 | 0 | 1.000 (half an ulp) | yes |
| pointer, `relaxed` | 74990764 | 0 | 0.500 | 0 | 1.000 | yes |
| descriptor (TMA) | 74991374 | 0 | 0.500 | 0 | 1.000 | yes |

**Every `sem` and `scope`** ([sem_scope.py](sem_scope.py), [sem_scope.out](sem_scope.out); round
20, library issue L101). `AtomicAddF32` is stated for every `sem` and `scope` of the pointer
atomic, while `run.py` runs `acq_rel` and `relaxed` at the default scope `gpu`. `sem_scope.py` runs
all 12 combinations of `sem` (`acquire`, `release`, `acq_rel`, `relaxed`) and `scope` (`gpu`,
`cta`, `sys`) on run.py's five kinds of input (`2^24` lanes each, 74,991,578 pairs checked per
combination). Each compiles to `atom.global.<scope>.<sem>.add.v4.f32`; in all 12, no pair is
outside the fact, the largest error is `0.5000` of its bound, the normal cases round to nearest,
and every result is bit-identical to `acq_rel` at `gpu` on the same inputs
(`results differing from acq_rel, gpu 0`).

Special pairs:

| `o` | `v` | pointer | descriptor |
|---|---|---|---|
| `2^-149` | `0` | `0` | `2^-149` |
| `0` | `2^-140` | `0` | `2^-140` |
| `2^-126` | `-2^-127` | `2^-126` | `2^-127` |
| `2^-126` | `2^-149` | `2^-126` | `2^-126 + 2^-149` |
| `-0.0` | `+0.0` | `+0.0` | `+0.0` |
| `-0.0` | `-0.0` | `-0.0` | `-0.0` |
| `1` | `-1` | `+0.0` | `+0.0` |
| `3·10^38` | `3·10^38` | `+inf` | `+inf` |

**Conclusions.**
1. `AtomicAddF32` holds on every pair tested, on both instructions and, for the pointer atomic,
   in all 12 combinations of `sem` and `scope`, with a factor `2` of margin: no error above
   `0.500` of its bound. The fact covers them all: the order and scope change which instruction
   variant runs, not its value (bit-identical results).
2. The pointer atomic rounds to nearest and flushes subnormal operands and results to zero (the
   special pairs); the TMA reduction rounds to nearest and keeps subnormals (its errors on the
   subnormal and cancellation inputs are `0`). Both stay within the tight bound
   `2^-24·|o + v| + 2^-125`, which they reach by construction: `1.000` is a rounding of half an ulp
   just above a power of two. The fact doubles both constants rather than state a bound the
   device meets with no margin.
3. The scalar `atomic_add(p, 0.0)` with `relaxed` order leaves memory unchanged, as the fact
   allows (`n = o` is within any bound when `v = ±0`).
4. Overflow (`|o + v| > maxVal`) gives `inf`; the fact says nothing there.

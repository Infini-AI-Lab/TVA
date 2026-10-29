# Conflicting global accesses within one program

Library evidence (not a kernel's), for library issue L15. Triton 3.8.0 (`~/fpv/tva/.venv`, torch
2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-10-27. Script: [run.py](run.py); output:
[out.txt](out.txt). It is TVA's own reproduction, after `issue/stale_readback.py` and the round-2
review's probes (`docs/library-review/round-2/lr-2b/work/seq_readback.py`,
`loop_carried_mem.py`).

**Question.** Does one Triton program see its own earlier global accesses in source order? A
program is many threads, and nothing in the source orders one thread's access to an address
before another thread's access to the same address in the same program.

**Method.** Each case counts results that a program run in source order could not give.
- Part A: each program stores its block to `buf` (a sentinel before), then loads it back and
  stores what it read: reversed (`buf[B-1-j]`, written by another thread), reversed with
  `tl.debug_barrier()` between, at the same offsets (each thread its own writes); and in place
  `buf[j] = buf[B-1-j] + 1` (write after read), and two stores, the second reversed (write after
  write). 8192 programs of 1024 elements (4 warps) and 4096 of 4096 (8 warps).
- Part B: a loop whose iteration `t` loads row `t` and stores it plus one into row `t + 1` (same
  offsets): `range` and `tl.range(..., num_stages=3)` at `num_stages` 1 and 3, the latter also with
  a `tl.debug_barrier()` after the store; and with the row going through `tl.dot` (identity) at
  `num_stages` 1, 3 (the default) and 4.
- Part C: two tensor atomics on the same addresses from different lanes, `atomic_xchg(p + j, 1)`
  then `atomic_xchg(p + (B-1-j), 2)`, and `atomic_add(q + j, 1)` then
  `o2 = atomic_add(q + (B-1-j), 1)`; 10 launches each.

**Result.**

| Case | Results not possible in source order |
|---|---|
| A, reversed read-back | 41.4% of elements (B = 1024), 23.0% (B = 4096), all of them the sentinel |
| A, reversed read-back with `tl.debug_barrier()` | 0 |
| A, read-back at the same offsets | 0 |
| A, in-place reversed copy (write after read) | 0.2% (B = 1024), 1.8% (B = 4096) |
| A, second store reversed (write after write) | 0 observed (in source order the second store wins; nothing guarantees it) |
| B, `range`, `num_stages` 1 or 3 | 0 |
| B, `tl.range(..., num_stages=3)`, kernel `num_stages` 1 or 3 | 96.9% of elements stale |
| B, the same with `tl.debug_barrier()` after the store | 0 |
| B, row through `tl.dot`, `num_stages` 1 | 0 (no `cp.async` in the PTX) |
| B, row through `tl.dot`, `num_stages` 3 (default) or 4 | 96.9% stale (`cp.async` in the PTX) |
| C, `atomic_xchg` pair | 0.33% (B = 1024) and 0.76% (B = 4096) of addresses end with the first operation's value |
| C, `atomic_add` pair | 0.34% and 0.73% of the second operation's old values show it ran first; every counter ends at 2 |

**Conclusions.**
1. A program that writes global memory and then reads, or writes again, the same bytes from other
   threads, or in a loop that Triton pipelines, does not behave as its source read in order. This
   is why the library's rule (L15) exists: two global accesses of one program that touch the same
   bytes, at least one of them a write, unless both are atomics on the same bytes, are a conflict, and a model with a
   conflict anywhere in a program is not well behaved, like a race between programs; no claim
   covers it.
2. The cases that came out right (a barrier, the same offsets, `num_stages = 1`, no pipelining) are
   not promised by Triton's documentation, and the rule promises no ordering for them:
   `tl.debug_barrier()` keeps no meaning in the model, and `num_stages` stays dropped.
3. Two atomics of one program on the same address are each atomic (every counter ends at 2), but
   their order is not the source order (Part C). So the semantics must allow every order of a
   program's atomics, for returned old values as for `xchg`.

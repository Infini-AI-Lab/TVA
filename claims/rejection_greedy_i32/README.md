# Prepared claim: `rejection_greedy_i32`

Kernel: `source/rejection_greedy.py`, vLLM v0.33.0's
`rejection_greedy_sample_kernel` (lines 67-122), launched by its host `rejection_greedy_sample`
(lines 125-178). Claim: `Claim.lean`, namespace `Claims.RejectionGreedy`; the statement to prove is
`Statement : ∀ K P, Assumptions K P → Spec K`. The model map is `model-map.json`.

One program per request: request `r` walks its `n_r` packed draft tokens; at each position it
writes the target's argmax token, and stops after the first position whose draft differs from it
(that position's target token is still written); if every draft matched it writes the bonus token
at column `n_r`; a request whose `is_greedy` flag is false returns at once and writes nothing. The
claim covers the launches SETUP.md asks for: `SYNTHETIC_MODE = False`, with `is_greedy = None`
(an all-greedy batch) or an `is_greedy` mask (a mixed batch). The model writes the addresses in
the source's order (`(ptr + start_idx) + pos`, `(ptr + req_idx * (max_spec_len + 1)) + pos`,
`(ptr + req_idx) - 1` with the library's pointer minus integer), the mask's element as a scalar
`int1` load, and `.to(tl.int32)` as `tl.convert .i64 .i32` (`docs/SEMANTICS.md`'s meaning of
`x.to`).

## Specification

`Spec K := Claim.Meets Inst.ok Pre Post K`: on every covered instance and every initial memory
satisfying `Pre`, every outcome of the launch is a normal end whose memory satisfies `Post`.

`Post d init fin`, with `start_r = 0` for `r = 0` and `cu[r-1]` otherwise, `n_r = cu[r] - start_r`,
`t(p)` the `int32` truncation (`Int.bmod · 2^32`) of `target_argmax[p]`, and `k_r` the number of
leading positions `p < n_r` with `draft[start_r + p] = t(start_r + p)` (`matchLen`, `n_r` when all
match), and `greedy_r` true with `is_greedy = None` and otherwise the mask's byte `r` non-zero
(`greedyAt`, the `int1` load's meaning):

- for each request `r < B` and column `j < S + 1`, output element `r * (S + 1) + j` is
  - its initial value if `greedy_r` is false (the program returned);
  - `t(start_r + j)` if `j < min(k_r + 1, n_r)` (every column up to and including the first
    mismatch);
  - `bonus[r]` if `j = n_r` and `k_r = n_r` (every draft matched);
  - its initial value otherwise (the host's `-1`);
- every byte outside the output's first `B * (S + 1)` elements is unchanged (the inputs, any
  tail of the output buffer, and every other byte the launch is given).

The reference is computed from the host's inputs (`cu`, the drafts, the targets, the bonus
tokens, the mask and the initial output), with SETUP.md's packing (request `r`'s drafts at
`[cu[r-1], cu[r])`). Exact: integers only, no tolerance. The truncation `t` is what the kernel's
`.to(tl.int32)` computes; on vLLM's inputs (token ids in `[0, vocab_size)`) it is the identity, so
`Post` is then exactly the host's `reference` (the greedy walk on the full `int64` values).

## Conditions

`Inst`: `B = batch_size` (the grid `(B,)`), `S = max_spec_len` (a runtime `int32`,
`do_not_specialize`), and the buffers `out` (`[B, S+1]` `int32`), `cu` (`[B]` `int32`), `draft`
(`[num_tokens]` `int32`), `target` (`[num_tokens]` `int64`), `bonus` (`[B, 1]` `int32`), and
`greedy`: `none` for `is_greedy = None`, or the mask (`[B]` `bool`, one byte per element).
`uniform_probs` (`None`, or the `float32` tensor vLLM passes on a mixed batch) is not read with
`SYNTHETIC_MODE = False` and is not part of `Inst`.

`Inst.ok`:
- `1 ≤ B < 2^31`: at least one request (the host's batch is never empty), and a valid grid;
- `B * (S + 1) ≤ 2^31`: the `int32` offsets `req_idx * (max_spec_len + 1) + pos` do not wrap
  (a limit of the kernel, see Exclusions); it also gives `S < 2^31`, so `max_spec_len` is an
  `int32`;
- `B ≤ cu.len`, `B ≤ bonus.len`, `B * (S + 1) ≤ out.len`, and `B ≤ greedy.len` for a mask;
- the output shares no byte with any input (the host allocates it fresh). The inputs may overlap.

Not required: `S + 1 < 2^31` (at `B = 1`, `S = 2^31 - 1` the wrapped `max_spec_len + 1` is
multiplied by `req_idx = 0`, and the kernel is right), `num_tokens < 2^31` (every index is below an
`int32` `cu` value), `cu.len = B`, `bonus.len = B`, `target.len = draft.len`. The `cu` values are
read from memory, not fixed in `Inst`.

`Pre d init`: the five buffers, and the mask when given, are valid; `cu` holds the inclusive running sums of the draft
counts (`cu[0] ≥ 0`, nondecreasing), its last entry at most the lengths of `draft` and `target`;
every `n_r ≤ S` (the host passes `max_r n_r`), for every request, greedy or not (vLLM's
`cu_num_draft_tokens` counts every request). Draft, target, bonus and mask values are arbitrary
(`-1` drafts, any `int64` target, any mask byte, any initial output).

`launch d = Launch.grid1 B [] (fusion := true)`: the host's grid, no block shape, Triton's default
`enable_fp_fusion`; `num_warps = 4`, `num_stages = 3` change no result; `debug = false`,
`num_ctas = 1`, no `launch_pdl`.

`Compiler.Faithful`'s scope, part (A), checked against the source: (4) the launch passes only
`num_warps` and `num_stages`, no `@triton.heuristics`, no `@triton.autotune`; (5) no descriptor is
built in the kernel; (6) no `CompiledKernel` launch (the host launches the `@triton.jit` function
directly); (8) no host `TensorDescriptor`. Part (B), the environment (no other writer
of the claim's bytes, Triton's knobs at their defaults, no hook, the H200 and the x86-64 CPython
3.13 host, Triton as installed, the kernel new), is assumed and not checked.

## Assumptions

One `[rule]` field, `faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)`.
The kernel uses no approximate operation, so no `Rules.Num` fact; `consistent` uses
`Platform.any`. No `[kernel]` field.

## Coverage

Each launch of SETUP.md (the host's test cases; each random batch runs with `is_greedy = None` and
with the test's random mask; `S` and `N` of the random batches are what the test's generator draws)
has an instance in `Claim.lean` with `Inst.ok` proved (`mk B S N true` places a mask of `B` bytes):

| Launch | `B`, `S`, `num_tokens` | Instance |
|---|---|---|
| vLLM "perfect match", "early mismatch" (`[[1, 2, 3]]`) | 1, 3, 3 | `setup_vllm_one_req` |
| vLLM `[[1, 2], [3]]` | 2, 2, 3 | `setup_vllm_two_req` |
| vLLM "single token" | 1, 1, 1 | `setup_vllm_single` |
| vLLM "no drafts" | 1, 0, 0 | `setup_vllm_no_drafts` |
| vLLM `[[1, 2, 3], [4, 5, 6]]` | 2, 3, 6 | `setup_vllm_multi_mismatch` |
| random batch 1, k 1 | 1, 1, 1 | `setup_b1_k1` |
| random batch 8, k 3 | 8, 3, 24 | `setup_b8_k3` |
| random batch 64, k 5 (0 to k drafts) | 64, 5, 134 | `setup_b64_k5` |
| random batch 256, k 4 (0 to k drafts, padded drafts) | 256, 4, 500 | `setup_b256_k4` |
| random batch 1024, k 8 (0 to k drafts) | 1024, 8, 4084 | `setup_b1024_k8` |
| random batch 37, k 128 (0 to k drafts) | 37, 123, 2340 | `setup_b37_k128` |
| random batch 16, k 2, accept 1.0 | 16, 2, 32 | `setup_b16_k2_all` |
| random batch 16, k 2, accept 0.0 (0 to k drafts) | 16, 2, 17 | `setup_b16_k2_none` |
| each random batch above with its mask | as above | `setup_<batch>_mask` (8 instances) |
| serving: `MAX_SPEC_LEN = 128`, 1024 requests | 1024, 128, 131072 | `setup_serving_max` |
| serving, mixed batch (`is_greedy = temperature == 0`) | 1024, 128, 131072 | `setup_serving_max_mask` |
| edge of `Inst.ok`: `B * (S + 1) = 2^31` | `2^24`, 127, 0 | `edge_offsets` |

`Inst.ok` puts no bound on the values (any vocabulary, any `int64` target, `-1` drafts, any mask),
and none on `B` or `S` beyond the offset limit, so every serving size SETUP.md names (batches up to
a thousand or more, 1 to 128 drafts, vocabularies 32000 to 151936), all-greedy or mixed, is
covered. Synthetic mode is not part of the setup.

## Tolerances

None: `Post` is exact (integers only).

## Exclusions

- **`B * (S + 1) > 2^31`: a limit of the kernel.** The output offset
  `req_idx * (max_spec_len + 1) + pos` is `int32` arithmetic and wraps once a written slot's
  offset reaches `2^31`; the store then goes below the output buffer. Every request writes its
  column 0, so at `S = 128` every batch of 16,647,162 requests or more goes wrong (row 16,647,161
  starts at offset `2^31 + 121`), and so does a batch of 16,647,161 whose last request writes
  column 8 or more (row 16,647,160 starts at `2^31 - 8`); at `S = 1`, `2^30 + 1` requests. The condition
  also leaves out the few launches just above `2^31` whose last rows write no slot past `2^31`.
  Each needs an output of at least 8 GiB; real use (batches of at most a few thousand requests,
  `S ≤ 128`) is below `2^22` and never reaches it.
- **An empty batch (`B = 0`)** and **an output overlapping an input** are outside real use (the
  host's batch is never empty; it allocates the output fresh).

## Library requests

None. The `is_greedy` load is a `torch.bool` argument: SEMANTICS.md states that such an argument is a `*i1` pointer, one byte per element, and that loading an `*i1` element reads the byte and tests it `≠ 0`.

## Checks

- **Builds**: `Claim.lean` builds with no error,
  no warning, no `sorry`, no `axiom`; `consistent`, `satisfiable` and every `setup_*_ok` use only
  `propext`, `Classical.choice`, `Quot.sound`.
- **Not vacuous**: `consistent` (`Platform.any`, `exists_total_faithful`); `satisfiable` (vLLM's
  "no drafts" case on the all-zero memory).
- **Coverage**: the 24 instances above, `Inst.ok` proved for each.
- **GPU** (`check_gpu.py`, H200, Triton 3.8.0): 34 launches, the 22 setup launches through the
  host (the random batches with and without their mask) plus edges (`S = 128` all accepted; `S`
  above every `n_r` with zero-draft requests; `int64` targets with high bits, where the kernel
  compares the truncation; `S = 0` with 70000 requests; `2^21` requests with `int32` extreme
  drafts; a 1024-request `S = 128` mixed batch with `is_greedy = temperature == 0` and vLLM's
  `uniform_probs`; masks all true, all false and of bytes `0..255`; three direct launches with the
  host's launch line on an output carved from a random-filled pool with guards, one with a mask).
  19,327,627 output cells: every cell exactly as `Post` states (error 0, no tolerance), every input
  unchanged, every guard unchanged; the host-launched cases also equal the host's own `reference`.
  Negative tests: 123 of 123 wrong outputs rejected (the mask ignored, a dropped bonus token, an
  all-zero output, a shifted packing, one written column missing). The offset edge
  `B * (S + 1) = 2^31` needs an 8 GiB output and is not run.

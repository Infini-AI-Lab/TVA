# Prepared claim: `kv_indices_i32`

Kernel: `source/kv_indices.py`, SGLang's
`create_flashinfer_kv_indices_triton` (lines 69-150), launched by `create_kv_indices` (lines
153-186). Claim: `Claim.lean`, namespace `Claims.KvIndices`; the statement to prove is
`Statement : ∀ K P, Assumptions K P → Spec K`. The model map is `model-map.json`.

One program per request `b` copies the `seq_lens[b]` entries of row `req_pool_indices[b]` of the
`int32` table `req_to_token` from position `s[b]` (`kv_start_idx[b]`, or `0` without it) to
`kv_indices[kv_indptr[b] :]`, in blocks of 512. The claim covers the configuration SETUP.md asks
for: `ENTRY_PAGE_SIZE = 1`, `TOKEN_BLOCK_PARALLEL = False`, `token_mapping = None`, grid `(bs,)`,
`kv_start_idx` `None` or a tensor (FlashInfer's sliding windows), with `int64` or `int32`
`req_pool_indices`, `seq_lens` and `kv_start_idx` (independently).

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.KvIndices.main : Statement` (`Statement`: `Claim.lean`
line 275). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 182), and every initial memory `init` that `Pre` accepts (line 232), every outcome of
the launch is a normal end (no fault, no hang) whose final memory `fin` satisfies `Post` (line 254).

**What `Post` asks** (line 254), two parts. The initial values `rpi d init b` (line 153),
`slen d init b` (line 156), `ind d init b` (line 159) and `sidx d init b` (line 162) are
`req_pool_indices[b]`, `seq_lens[b]`, `kv_indptr[b]` and the start `kv_start_idx[b]` (`0` when
`hasStart = false`), read through `valOf` (line 147) as integers.
1. for every request `b < bs` and every `j` with `j < slen d init b`: element
   `rpi·max_context_len + sidx + j` of `req_to_token` reads in `init` as some `v : I32`, and element
   `ind + j` of `kv_indices` reads `v` in `fin`. It is exact: an `int32` copy, no tolerance;
2. the frame: every byte outside `kv_indices.elems (kvOut d init)` is unchanged; `kvOut`
   (line 245) is the set of output elements `ind d init b + p` with `b < bs`, `p < slen d init b`.

**What you are given** (`Assumptions`, line 266): one field, `faithful` (`Compiler.Faithful`): the
device's outcomes `K d` are outcomes of the model's launch, when the launch is valid, has
`debug = false`, and the model's launch is well behaved from `init` (no out-of-bounds access, no race).
No numeric fact: the kernel moves integers.

**The kernel, as the model runs it** (`create_flashinfer_kv_indices_triton` line 41, `model`
line 216, `launch` line 225): a 1-D grid of `bs` programs. Program `b` loads four scalars
(`tl.load1`): `req_pool_indices[b]` (converted to `int64`), `kv_indptr[b]`, `seq_lens[b]` (converted
to `int32`) and, when `kv_start_idx` is `some p` and `p.addr ≠ 0`, `kv_start_idx[b]` (converted to
`int32`). It computes `num_loop = (kv_end - kv_start + 511) / 512` (`tl.idiv`, `int32`) and runs
`tl.forRange 0 num_loop 1`: trip `i` loads the masked block of 512 table entries
`rpi·stride + kv_start + 512·i + [0, 512)` (mask `offset < kv_end - kv_start`, in `int64`) and stores
it to `kv_indices` at `ind + 512·i + [0, 512)` under the same mask. Programs write disjoint ranges of
`kv_indices` (from `kv_indptr`'s running sums) and use no atomics: the route is
`TVA.meets_independent`, with `safe_forRange` and a per-trip write footprint `Wt i` for the loop
(PROVER_GUIDE §4). The footprints depend on `init` (the rows and offsets are loaded values).

**What the proof must establish.**
- each launch is valid and has `debug = false` (`Launch.valid_grid1`: fewer than `2^31` programs
  from `grid_lt`, and the block `[512]` Triton accepts);
- each program is `Safe`: the scalar loads are in bounds (`plen`, `rplen`, `llen`, `stlen`; the
  buffers valid from `Pre`), and the branch on `kv_start_idx` takes the load (`start_nz`); the
  conversions are exact and the `int32` arithmetic gives `kv_end - kv_start = slen` and
  `num_loop = ceil(slen / 512)` without a bad lane (`ReqOk`, line 170: `0 ≤ slen`,
  `slen + 511 < 2^31`, `0 ≤ sidx < 2^31`; `CtxOk`, line 178: `slen ≤ 2^22` and `sidx ≤ 2^22`, so
  `kv_end = sidx + slen ≤ 2^23` does not wrap either), `tl.RangeOk` holds, and `i·512` does not wrap;
- in each trip, the enabled lanes' table addresses lie in `req_to_token` (`ReqOk`'s last part)
  and their output addresses in `kv_indices` (`Pre`: `0 ≤ ind b` and `ind (b + 1) ≤ kv_indices.len`,
  with `ind (b + 1) = ind b + slen b`); the trips' write footprints are disjoint;
- the programs' write footprints are disjoint (the ranges `[ind b, ind b + slen b)` from the running
  sums), and no program reads what any program writes (`disj`: the output shares no byte with an
  input);
- the stored values give part 1 (the loaded entry, stored unchanged), and the bytes no program
  writes give part 2.

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable`, `mk_ok` and the
`setup_*_ok` theorems; `bytes_disjoint_of_le` (line 284) can be reused.

## Specification

`Spec K := Claim.Meets Inst.ok Pre Post K`. `Post d init fin`, with `rpi[b]`, `len[b]`, `ind[b]`,
`s[b]` the initial values of `req_pool_indices[b]`, `seq_lens[b]`, `kv_indptr[b]` and
`kv_start_idx[b]` (`s[b] = 0` without `kv_start_idx`; `sidx`):

- for every request `b < bs` and `j < len[b]`, element `ind[b] + j` of `kv_indices` holds the
  initial `int32` element `rpi[b] * max_context_len + s[b] + j` of the table's storage, that is
  `req_to_token[rpi[b], s[b] + j]`;
- every byte outside those elements is unchanged.

The reference is the host's `reference` (the rows' slices concatenated, at the offsets the host
builds), computed from the host's inputs. Exact (an `int32` copy), no tolerance.

## Conditions

`Inst`: `max_context_len` (the table's row length, the launch's `req_to_token_ptr_stride`), `bs`,
the index dtypes `dP`, `dL`, `dS`, `hasStart` (`kv_start_idx` given), and the six buffers (storage
of the contiguous tensors; `kv_start_idx`'s is unused without it).

`Inst.ok`:
- `dP`, `dL`, `dS` each `int64` (SGLang's serving buffers) or `int32` (the code base's test);
- with `kv_start_idx`: it has `bs` elements and a non-zero address (line 117 tests the pointer:
  a CUDA tensor of `bs ≥ 1` elements never has the null data pointer; an empty tensor's, `0`,
  would skip the branch), and it shares no byte with the output;
- `bs < 2^31` (a valid grid); `max_context_len < 2^31` (the stride is an `int32` argument);
- `max_context_len ≤ 2^22` (`mcl_le`: the row length is a context length);
- `bs ≤ 8192` (`bs_le`: the number of requests is a batch size);
- `kv_indptr` has `bs + 1` elements, the index buffers `bs`;
- the output shares no byte with an input (`torch.empty`).

`bs = 0` is covered (an empty grid runs nothing). The number of table rows and the row index are
not bounded separately: `Pre`'s in-storage condition is what the kernel needs.

`Pre d init`: the buffers valid (`kv_start_idx` when given); for every request (`ReqOk`):
`0 ≤ len[b]`, `len[b] + 511 < 2^31` (a limit of the kernel, Exclusions), `0 ≤ s[b] < 2^31` (its
`.to(tl.int32)` is exact; real starts are `seq_lens - paged_kernel_lens ≥ 0`), and the row's
`len[b]` entries from `s[b]` in the table's storage
(`0 ≤ rpi[b] * max_context_len + s[b] + j < req_to_token.numel()`); `len[b] ≤ 2^22` and
`s[b] ≤ 2^22` (`CtxOk`: the request's length and the prefix before its window are context
lengths); `kv_indptr` the
running sums `ind[b + 1] = ind[b] + len[b]`, all in `[0, kv_indices.len]` (the host's start is
`0`; any nonnegative start is covered). Any table entries; rows may be shared between requests.

`launch d = Launch.grid1 bs [[512]] (fusion := true)`; `num_warps = 4`, `num_stages = 3` change no
result; `num_ctas = 1`, no `launch_pdl`, `debug = false`.

`Compiler.Faithful`'s scope, part (A), checked against the source: (4) the launch passes only the
kernel's parameters, `num_warps` and `num_stages`; no `@triton.heuristics` or `@triton.autotune`;
(5) no descriptor; (6) no `CompiledKernel`; (8) no host `TensorDescriptor`. The `int64` index buffers come from SGLang's schedule batch
(`srt/managers/schedule_batch.py:2545, 2786`). Part (B), the environment, is assumed and not checked.

## Assumptions

One `[rule]` field, `faithful` (`Compiler.Faithful` as the library states it). No approximate
operation, no `Rules.Num` fact (`Platform.any`), no `[kernel]` field.

## Coverage

The host's test cases (source lines 203-211) and FlashInfer's `kv_start_idx` launches have
instances with `Inst.ok` proved (`Inst.ok` does not depend on the lengths and starts, which are
`Pre`'s):

| Launch | table, batch, dtypes | Instance |
|---|---|---|
| pool 4096 × 4096, 1 request | `int32` | `setup_pool4096_bs1_i32` |
| pool 4096 × 4096, 37 requests | `int32` | `setup_pool4096_bs37_i32` |
| pool 4096 × 4096, 1786 requests | `int64` | `setup_pool4096_bs1786_i64` |
| pool 513 × 8192, lengths 1, 511, 512, 513, 1025, 8192 | `int64` | `setup_pool513_lens` |
| pool 9 × 131072, lengths 131072, 1, 70000, 4097, 0, 33, 99999, 512 | `int64` | `setup_pool9_long` |
| pool 2049 × 32768, 256 requests | `int64` | `setup_pool2049_bs256` |
| pool 300 × 4096, 100 sliding windows with `kv_start_idx` | `int64` | `setup_pool300_window` |
| `int64` rows with `int32` lengths (the host accepts it) | 64 × 1024, 16 | `setup_mixed_dtypes` |
| serving: 1M-token rows | 4 × `2^20`, 4 | `setup_context_1m` |
| FlashInfer sliding-window decode (window 4096, `kv_start_idx`) | 64 × 131072, 64, `int64` | `setup_swa_decode` |
| `int32` lengths and `kv_start_idx`, `int64` rows | 128 × 8192, 32 | `setup_start_i32` |

Every batch size, any length from 0 to the row (and beyond it, within the table), not a multiple
of 512 or one, and contexts to `2^22` tokens are covered.

## Tolerances

None: `Post` is exact.

## Exclusions

- **Dimensions that are not multiples of 128 or exceed 2^17, contexts longer than 2^22 tokens, and
  batches of more than 8192 requests** — out of scope: the claim covers real model shapes only. The
  fields: `max_context_len` (`mcl_le`), each `seq_lens[b]` and `kv_start_idx[b]` (`Pre`'s
  `CtxOk`), all at most `2^22`, and the batch size `bs` (`bs_le`), at most `8192` (`SETUP.md`'s
  launches: at most `1786` requests). The claim has no model dimension; the batch's total length (`kv_indptr[bs]`, the sum of the
  requests' contexts) is not bounded by `2^22`.
- **`seq_lens[b] + 511 ≥ 2^31`: a limit of the kernel.** `tl.cdiv(kv_end - kv_start, 512)` is
  `(x + 511) // 512` in `int32`; at `x ≥ 2^31 - 511` the addition wraps, `num_loop` is negative,
  and the program copies nothing. The first launch that hits it has one request of
  `2^31 - 511` tokens (a table row of 8 GiB); real contexts (128K to a few million tokens) never
  reach it.
- **A batch whose total length reaches `2^31`**: the host's `int32` `kv_indptr` wraps (`Pre`'s
  running sums are `int32` values), a limit of the host's format (an 8 GiB output); never reached.
- **`max_context_len ≥ 2^31`** (the stride would be an `int64` argument) and **`bs ≥ 2^31`** (the
  grid): outside real use.
- **An output that overlaps an input**: outside real use (fresh allocation).
- **`kv_start_idx` an empty tensor (address `0`), or a start outside `[0, 2^31)`**: outside real
  use (`kv_start_idx` has one element per request; starts are positions in a row).
- `ENTRY_PAGE_SIZE > 1`, `TOKEN_BLOCK_PARALLEL` and `token_mapping` (`pack_read_stream`'s
  translating and block-parallel launches) are not part of the setup (`SETUP.md`).

## Modelling decisions

- Line 120 is `kv_end += …`, written `kv_end + …`.
- `blk = 0` and `num_blk = 1` are `int32` scalar tensors (a name assigned a Python number), named
  and passed to `tl.forRange`.
- The addresses are in the source's order: the `int32` scalars `kv_start` and `kv_indices_offset`
  are added to the pointer directly.
- Line 117's `if kv_start_idx:` is a `match` on the `Option` pointer: `none` (the compile-time
  `None`) visits no branch; `some p` is the runtime `if` on `p.addr ≠ 0`, whose branch returns the
  new `kv_start` and `kv_end` (the names it assigns) to the merge.

## Library requests

None.

## Checks

- **Builds**: `lake env lean ../Claim.lean`: no error or warning, no
  `sorry`, no `axiom`; the theorems use only `propext`, `Classical.choice`, `Quot.sound`.
- **Not vacuous**: `consistent` (`Platform.any`); `satisfiable` (pool 4096 × 4096, 37 requests,
  `int64`, with `kv_start_idx`, all lengths and starts `0`, on the all-zero memory).
- **Coverage**: the 11 instances above (`setup_*_ok`).
- **GPU** (`check_gpu.py`, H200, Triton 3.8.0): 24 launches: the 7 setup cases through the host
  (same generator, the last with `kv_start_idx`); FlashInfer's `kv_start_idx` launches (sliding-
  window decode with a 4096-token window over contexts to 131072, all-zero starts, the ragged
  prefill's window over the prefix, `int32` lengths and starts with either row dtype, windows past
  the end of their row); edges: mixed index dtypes both ways with lengths 0, 1, 511, 512, 513,
  1023, 1024; one row shared by several requests; `max_context_len = 1` (the stride a
  compile-time constant) and `16`; all lengths 0; `2^20`-token rows; 8192 requests (`bs_le`); three direct
  launches with the host's launch line, `kv_indptr` starting at 0, 1234 and 99 (the last with
  `kv_start_idx`), into an output longer than needed inside a random-filled guarded pool.
  14,167,929 written elements: every one exact as `Post` states, every other output element,
  every input and every guard unchanged. Negative tests: 33 wrong outputs (the start ignored, the
  start off by one, each request's last block of 512 dropped) all fail `Post`.

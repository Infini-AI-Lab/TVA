# `kv_indices_i32`

**The kernel.** SGLang's `create_flashinfer_kv_indices_triton` (`source/kv_indices.py`, kernel
lines 69–150; host `create_kv_indices` lines 153–186). A 1-D grid of `bs` programs, one per
request; program `b` reads the request's table row `req_pool_indices[b]`, its length
`seq_lens[b]`, its output offset `kv_indptr[b]` and, when given, its first position
`kv_start_idx[b]`, then copies the row's entries `s[b] … s[b] + seq_lens[b] − 1` of the `int32`
table `req_to_token` to `kv_indices[kv_indptr[b] …]`, in blocks of `512`. The launches covered have
`ENTRY_PAGE_SIZE = 1`, `TOKEN_BLOCK_PARALLEL = False`, `token_mapping = None`, and `kv_start_idx`
either `None` or a tensor. The claim is `Claims.KvIndices.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and:
- for every request `b < bs` and position `j < seq_lens[b]`, element `kv_indptr[b] + j` of
  `kv_indices` holds the initial `int32` value of `req_to_token[req_pool_indices[b], s[b] + j]`
  (element `req_pool_indices[b]·max_context_len + s[b] + j` of the contiguous table), with `s[b]`
  `kv_start_idx[b]`, or `0` without it;
- every byte outside those elements of `kv_indices` is unchanged.

It is exact: no float operation, no tolerance, no device fact.

**The conditions** (`Inst.ok`, `Pre`):
- `req_pool_indices`, `seq_lens` and `kv_start_idx` are `int64` or `int32`;
- `bs < 2^31`, `bs ≤ 8192`; `max_context_len < 2^31`, `max_context_len ≤ 2^22`;
- `kv_indptr` has `bs + 1` elements, the index buffers `bs`; a given `kv_start_idx` is not at
  address `0`; all buffers are valid in the initial memory;
- `kv_indices` shares no byte with any input;
- every request's length is nonnegative with `seq_lens[b] + 511 < 2^31`, its start is in
  `[0, 2^31)`, both are at most `2^22`, and its `seq_lens[b]` entries from the start lie in the
  table's storage (`ReqOk`, `CtxOk`);
- `kv_indptr` holds the exclusive running sums of the lengths from a nonnegative start, each within
  the output's length.

Any `int32` table entries are covered.

**The launches covered** (`SETUP.md`): the host's test cases and FlashInfer's `kv_start_idx`
launches (`setup_*_ok`): pools `4096 × 4096` (1, 37 and 1786 requests), `513 × 8192`,
`9 × 131072`, `2049 × 32768`, `300 × 4096` with sliding windows, mixed index dtypes, 1M-token rows,
sliding-window decode, and `int32` lengths with `kv_start_idx`. `mk_ok` covers the host's layout for
every batch and table; the lengths and starts are `Pre`'s.

## Exclusions

- **Lengths or starts above `2^22`, `max_context_len > 2^22`, `bs > 8192`**: the claim covers real
  model shapes only. The batch's total length is not bounded by `2^22`.
- **`seq_lens[b] + 511 ≥ 2^31`**: a kernel limit; the `int32` `(x + 511) // 512` wraps and the
  program copies nothing.
- **A batch whose total length reaches `2^31`**: the host's `int32` `kv_indptr` wraps.
- **`max_context_len ≥ 2^31`, `bs ≥ 2^31`**: outside real use.
- **An output overlapping an input**: outside real use (fresh allocation).
- **`kv_start_idx` at address `0`, or a start outside `[0, 2^31)`**: outside real use.
- **`ENTRY_PAGE_SIZE > 1`, `TOKEN_BLOCK_PARALLEL`, `token_mapping`**: not part of the setup.

`check_gpu.py` (one H200) checks `Post` on the setup's launches.

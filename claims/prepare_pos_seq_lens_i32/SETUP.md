# Setup of `prepare_pos_seq_lens_i32`

From vLLM v0.33.0 (commit `4ba787ec99`), `vllm/v1/worker/gpu/input_batch.py:357`
`_prepare_pos_seq_lens_kernel`, launched by `prepare_pos_seq_lens` (`input_batch.py:393`), whose
one caller is the GPU model runner's input preparation (`vllm/v1/worker/gpu/model_runner.py:1499`).

## The source

`source/prepare_pos_seq_lens.py` holds:
- `_prepare_pos_seq_lens_kernel`, the kernel, with its decorator `@triton.jit`
  (`input_batch.py:356-390`): one program per request stores the request's `seq_lens` entry and
  the positions of its tokens, and one more program stores `0` to the unused `seq_lens` entries;
- `prepare_pos_seq_lens`, the host (`input_batch.py:393-412`);
- `reference` (exact, on the CPU) and `make_batch`, test helpers;
- the test under `if __name__ == "__main__":`.

Changes from the code base: one comment line of the kernel (`input_batch.py:369`, above the
zeroing loop) and the two comment lines above the host's launch (`input_batch.py:401-402`) are
removed; the code is unchanged. vLLM's `from vllm.triton_utils import tl, triton` is
`import triton` and `import triton.language as tl`. The module docstring lists the changes.

## The config

No autotuning and no heuristics. The host's launch: grid `(num_reqs + 1,)` with
`num_reqs = idx_mapping.shape[0]`, `max_num_reqs = seq_lens.shape[0]` (a runtime integer;
Triton makes the value `1` a compile-time constant, which gives the kernel the same values), and
`BLOCK_SIZE = 1024`; launch options Triton's defaults (`num_warps = 4`, `num_stages = 3`).

## The launches

The model runner's call (`model_runner.py:1400-1505`) passes vLLM's persistent buffers:
- `idx_mapping`: `[num_reqs]`, `int64`, each request's slot in the request states
  (`model_runner.py:1406-1408`, `async_tensor_h2d(..., dtype=torch.int64)`);
- `query_start_loc`: `query_start_loc[: num_reqs_padded + 1]` of the `int32` buffer
  `[max_num_reqs + 1]` (`input_batch.py:32-34`): `0`, then the running sums of the scheduled
  tokens, and the entries past `num_reqs` set to the total (`model_runner.py:1460-1468`; the
  adaptive verification path rebuilds it the same way,
  `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:462-468`);
- `num_computed_tokens`: `req_states.num_computed_tokens.gpu`, `[max_num_reqs]`, `int32`
  (`vllm/v1/worker/gpu/states.py:61-63`), indexed by slot; each value at most the model's
  context length;
- `pos`: `input_buffers.positions`, `[max_num_tokens]`, `int64` (`input_batch.py:30`);
- `seq_lens`: `input_buffers.seq_lens`, `[max_num_reqs]`, `int32` (`input_batch.py:35`).

Sizes: `num_reqs ≤ max_num_reqs = max_num_seqs` (vLLM's default 256 or 1024, up to thousands),
`max_num_tokens = max_num_batched_tokens` (typically 2048 to 16384, up to millions for long
prefills), contexts up to `max_model_len` (to 128K tokens and beyond). The host's own test
(source lines 142-165):

| case | `num_reqs` | `max_num_reqs` | `max_num_tokens` | query lengths | computed tokens |
|---|---|---|---|---|---|
| decode | 8 | 256 | 8192 | 1 | `< 4096` |
| prefill | 7 | 16 | 8192 | 1, 1023, 1024, 1025, 3000, 0, 17 | `< 2048` |
| full batch | 128 | 128 | 512 | 2 | `< 1000` |
| one request | 1 | 1 | 16 | 5 | `< 100` |
| long padding | 3 | 4096 | 64 | 3, 1, 2 | `< 100` |
| spec decode | 64 | 1024 | 8192 | 4 | `< 32768` |
| serving | 1024 | 1024 | 8192 | 1000 × 1, 24 × 299 | `< 131072` |
| long prefill | 1 | 256 | `2^20` | `2^20` | `< 2^21` |

and the large serving sizes `setup_serving_max`: 8192 requests, `max_num_reqs = 8192`,
`2^22` positions. Each is a `setup_*_ok` theorem of `Claim.lean`.

## Checked on the H200

`check_gpu.py` (Triton 3.8.0) runs 19 launches, every one with both outputs carved from
sentinel-filled pools and initialised with random values: the 9 setup launches through the host
(the 8 test cases, and 8192 requests with 4,190,310 tokens and computed tokens up to `2^22`),
edges through the host (no request, with `max_num_reqs` 300 and 1; 50 empty queries; computed
tokens `2^22` with `2^22` tokens; `max_num_reqs = 2^24`, 16384 trips of the zeroing loop; slots
shared by several requests), and four direct launches with the host's launch line
(`query_start_loc` starting at 777 with buffers longer than the batch; `max_num_reqs` 2500, 7, 1
and 0, the last three below `num_reqs`). All 26,272,462 output cells are exactly as `Post`
states; every other output element, every input and every guard is unchanged; 91 of 91 wrong
outputs (`seq_lens` off by one, positions ignoring the computed tokens, the slot ignored,
padding entries left unzeroed, a token dropped, a position off by one) fail `Post`.

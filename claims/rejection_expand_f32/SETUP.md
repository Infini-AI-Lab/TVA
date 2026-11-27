# Setup of `rejection_expand_f32`

From vLLM v0.33.0 (commit `4ba787ec99`), `vllm/v1/sample/rejection_sampler.py:1023` `expand_kernel`,
launched by `expand_batch_to_tokens` (`rejection_sampler.py:648`).

## The source

`source/rejection_expand.py` holds:
- `MAX_SPEC_LEN = 128` (`rejection_sampler.py:47`), verbatim;
- `expand_batch_to_tokens`, the host (`rejection_sampler.py:648-686`), verbatim: it allocates the
  output with `x.new_empty(num_tokens)` and calls `_expand`;
- `expand_kernel`, the kernel, verbatim with its comment and decorator
  `@triton.jit(do_not_specialize=["replace_from", "replace_to"])` (`rejection_sampler.py:1021-1042`):
  one program per request stores the request's value, replaced by `replace_to` where it equals
  `replace_from`, at the request's positions of the output, with one masked store of
  `MAX_NUM_TOKENS` lanes;
- `_expand`, the launch (`rejection_sampler.py:1057-1067`);
- `reference` (exact, on the CPU) and the test under `if __name__ == "__main__":`.

Changes from the code base, all in the host: `_expand` launches the kernel directly instead of
through vLLM's warm-up dispatcher (`@triton_kernel_dispatcher_with_warmup`, with the warm-up inputs
`_expand_warmup_inputs`, `rejection_sampler.py:1045-1054`; the dispatcher's launch is
`vllm/model_executor/warmup/jit_warmup_triton_helper.py:585-605` and `:315-358`). The dispatcher
passes the same arguments by name, with the grid `(input.shape[0],)` and
`MAX_NUM_TOKENS=MAX_SPEC_LEN` of the `DispatchSpec` (`rejection_sampler.py:1067`) and Triton's
default launch options, which the source writes out (`num_warps=4`, `num_stages=3`). The module
constant `GREEDY_TEMPERATURE = 0` (`rejection_sampler.py:44`) is a local of the test, its only
user here. `reference` and the test are new.

## The config

No autotuning and no heuristics in the source. The launch `_expand` makes: grid `(batch_size,)`,
`MAX_NUM_TOKENS = MAX_SPEC_LEN = 128`; launch options Triton's defaults (`num_warps=4`,
`num_stages=3`). `replace_from` and `replace_to` are runtime scalars (`do_not_specialize`), so
`int32` arguments whatever their value in `[-2^31, 2^31)`.

## The launches

vLLM launches the kernel only through `expand_batch_to_tokens`, whose only callers are the three
calls of `apply_sampling_constraints` (`rejection_sampler.py:589-645`, calls at 617-641), made by
`RejectionSampler.forward` (`rejection_sampler.py:201-205`) on every step whose batch is not all
greedy:
- `temperature = expand_batch_to_tokens(sampling_metadata.temperature, cu_num_draft_tokens,
  num_tokens, replace_from=GREEDY_TEMPERATURE, replace_to=1)`: `float32`, `0 → 1` (a greedy
  request's temperature `0` becomes `1` before `logits.div_`);
- `top_k = expand_batch_to_tokens(sampling_metadata.top_k, cu_num_draft_tokens, num_tokens)` when
  some request sets top-k: `int32`, `0 → 0` (the defaults);
- `top_p = expand_batch_to_tokens(sampling_metadata.top_p, cu_num_draft_tokens, num_tokens)` when
  some request sets top-p: `float32`, `0 → 0`.

With shapes as variables:
- `batch_size ≥ 1` requests (the grid), request `r` with `n_r ≥ 0` draft tokens,
  `max_r n_r = max_spec_len ≤ MAX_SPEC_LEN = 128` (asserted by `RejectionSampler.forward`,
  `rejection_sampler.py:157`);
- `x`: `[batch_size]`, contiguous: `temperature` and `top_p` are `float32`, `top_k` is `int32`
  (`vllm/v1/worker/gpu_input_batch.py:194-212`, sliced `[:num_reqs]`); any values (vLLM's
  temperatures are `≥ 0`, its top-p in `(0, 1]`, its top-k `≥ 0`, but the claim covers every value);
- `cu_num_tokens = cu_num_draft_tokens`: `[batch_size]`, `int32`, the inclusive prefix sums of the
  `n_r` (`vllm/v1/worker/gpu_model_runner.py:2629-2639`);
- `num_tokens = logits.shape[0]`, the number of draft tokens `Σ n_r` (the target logits are
  gathered at the draft positions, `rejection_sampler.py:186-188`);
- the output `expanded_x = x.new_empty(num_tokens)`: `[num_tokens]`, the dtype of `x`;
- the source's own test (source lines 160-206): the docstring's example (`x = [a, b, c]`,
  `cu_num_tokens = [2, 5, 6]`) and seven random batches of 1 to 1024 requests with 0 to 128 tokens
  each, each run with the three calls;
- serving sizes: batches up to the maximum number of sequences (hundreds to a thousand requests),
  `num_speculative_tokens` typically 1 to 8 (EAGLE, MTP, n-gram), at most `MAX_SPEC_LEN = 128`.

The warm-up inputs `_expand_warmup_inputs` (`float32`, `int32` or `int64` tensors) are compiled by
vLLM's warm-up, which is not a launch the claim covers; no caller passes an `int64` tensor.

## Checked on the H200

`check_gpu.py` (Triton 3.8.0) runs 48 launches: the 25 setup launches through the host (the
docstring's example, the seven random batches with the three calls, and 1024 requests of 128 tokens
with the three calls) and edges (8192 requests with counts `0` to `128`, zero-token requests first;
`float32` values with NaNs of several payloads and signs, infinities, signed zeros, subnormals and
the neighbours of `replace_from`; `replace_from`/`replace_to` at `1`, at the `int32` extremes and
at `2^24 + 1`, which `float32` rounds; one request of 128 tokens, one of none; three guarded direct
launches on outputs longer than the number of tokens), and one launch beyond `Inst.ok` (70000
requests). All 5,096,617 output elements are bit for bit as `Post` states, every input, output tail
and guard is unchanged, and the host-launched cases also equal the host's `reference`; 140 of 140
wrong outputs (the replacement ignored, a bitwise comparison in place of the float one, `replace_to`
left unconverted, a shifted packing, a request's last token missing, an all-zero output) fail
`Post`.

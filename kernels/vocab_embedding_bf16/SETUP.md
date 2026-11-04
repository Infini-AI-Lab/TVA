# Setup of `vocab_embedding_bf16`

Set up by the coordinator on 2026-11-04 for the sandbox's test set (sandbox/TESTSET.md), from
SGLang v0.5.23 (commit `297e54fd10`),
`python/sglang/kernels/ops/embeddings/vocab_parallel_embedding.py:9`
`_vocab_parallel_embedding_kernel`.

## The source

`source/vocab_parallel_embedding.py` holds:
- `_vocab_parallel_embedding_kernel`, the `@triton.jit` kernel, copied verbatim: one program per
  (token, block of `BLOCK_H` columns) loads the token id, maps it to the shard's local row (the
  original vocabulary, or the added vocabulary after the padding), and stores that row's columns,
  or zeros when the id is outside the shard;
- `vocab_parallel_embedding`, the host wrapper (`vocab_parallel_embedding.py:56`), copied
  verbatim but for one added launch option;
- `shard_indices` and `reference`, test helpers (a tensor-parallel rank's vocabulary ranges, as
  `VocabParallelEmbedding` computes them, and the lookup in torch);
- the test under `if __name__ == "__main__":`.

The vocabulary ranges, `hidden_dim` and the weight's row stride are compile-time values of the
kernel (`tl.constexpr`), one compiled kernel per embedding layer.

## The config

No decorator was removed: the kernel has no `@triton.autotune` or `@triton.heuristics`. The
configuration is the wrapper's own: grid `(n_tokens, cdiv(hidden_dim, BLOCK_H))` with
`BLOCK_H = min(1024, next_power_of_2(hidden_dim))`, `num_warps=8`, and no launch for an empty
input. The launch option `num_stages=3` is added: the wrapper passes none, and it is Triton
3.8.0's default. No flags.

## The launches

The claim should cover the wrapper's launches as the embedding layer makes them
(`srt/layers/vocab_parallel_embedding.py:449-496`: tensor-parallel size above 1, unquantized
weights), with shapes as variables:
- `bfloat16` weight and output; `int64` token ids (SGLang's `input_ids`,
  `srt/model_executor/cuda_graph_buffer_registry.py:633`), contiguous, of any shape;
- the weight `[num_embeddings_per_partition, hidden_dim]`, unit column stride, any row stride
  (the code base's test passes a column slice of a wider buffer,
  `test/registered/kernels/ops/embeddings/test_vocab_parallel_embedding.py:80-94`);
- `hidden_dim` from the code base's tests (7, 128, 257, 6144) to model hidden sizes (2048 to
  8192), not necessarily a power of two;
- the ranges as a rank's shard: `org_vocab_start_index ≤ org_vocab_end_index`, the shard's
  padding, and the added range after the original vocabulary (LoRA's extra vocabulary), so that
  every id in either range maps to a row of the weight (`srt/layers/vocab_parallel_embedding.py:
  273-298, 353-388`): vocabularies such as 128256 (Llama 3), 129280 (DeepSeek-V3) and 151936
  (Qwen), padded to a multiple of 64, split over 2 to 8 ranks;
- token ids of any `int64` value: ids outside the shard, negative ids and ids past the
  vocabulary give zero rows (the code base's test passes `-100`, `-1`, `2**30` and the `int64`
  extremes).

## Checked on the H200

On the coordinator's H200 run (Triton 3.8.0), the test passes, the output equal to the reference
bit for bit in every case: vocabularies 128256 (TP 8, hidden 4096, 8192 ids), 129280 (TP 8,
hidden 7168, 1 id), 151936 (TP 4, hidden 2048, ids of shape `(3, 77)`), 152064 (TP 2, hidden
5120, `int32` ids), 32000 plus 256 added (TP 2, hidden 257), 50257 plus 3 added (TP 8, hidden
6144), 1000 (TP 2, hidden 7, `int32` ids) and an empty input; each with ids outside the shard,
`-1`, the vocabulary size and `2**30`, and with `int64` ids the `int64` extremes.

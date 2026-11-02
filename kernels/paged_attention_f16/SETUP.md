# Setup of `paged_attention_f16`

Set up by the coordinator on 2026-11-01, before any prover started (protocol §3, "Setting up a
kernel folder"). This kernel is not from a tutorial: the user asked for new kernels ("please write
and prove new kernels. 1. paged_attention. (indirect memory access)."), and the coordinator wrote
`source/paged_attention.py` for this folder.

## The source

`source/paged_attention.py` holds:
- `paged_attention_kernel`, the `@triton.jit` kernel: decode-phase paged attention, one program
  per (sequence, query head);
- `paged_attention`, the host wrapper that launches it;
- `make_cache` and `reference`, test helpers (a random paged cache, and attention in `float64`);
- the test under `if __name__ == "__main__":`.

The kernel reads each sequence's keys and values through its block table: page
`block_tables[s, j]` holds the sequence's tokens `j·PAGE … j·PAGE + PAGE − 1`. The page offset is
computed in `int64`, because a cache can hold more than `2^31` elements.

## The config

No autotuning. The wrapper launches with grid `(num_seqs, num_heads)` and `num_warps=4`, and passes
`QUERIES_PER_KV = num_heads // num_kv_heads`, `PAGE_SIZE` (the cache's dimension 1) and `HEAD_DIM`
(the query's last dimension) as compile-time values.

## The launches

The claim should cover the wrapper's launches as the test makes them, with shapes as variables:
- `float16` `q`, caches and output; `int32` block tables and sequence lengths;
- the test's head counts, head dims, page sizes and sequence lengths: (32, 8, 128, 16), (8, 8, 64,
  16) and (16, 2, 128, 32), with lengths from 1 to 8191;
- each sequence's pages distinct and inside the cache.

## Checked on the H200

On the coordinator's H200 run (Triton 3.8.0), the test passes: max abs error against the `float64`
reference 4.9e-4, 4.9e-4 and 2.4e-4 for the three configurations. A separate run with a cache of
2,148,532,224 elements (more than `2^31`), reading pages past element `2^31`, gave 2.4e-4.

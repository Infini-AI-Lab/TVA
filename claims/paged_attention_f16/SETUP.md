# Setup of `paged_attention_f16`

A classic kernel written for TVA (not from a tutorial): decode-phase paged attention,
`paged_attention_kernel` (`source/paged_attention.py:30-72`), launched by its host wrapper
`paged_attention` (`source/paged_attention.py:75-97`).

## The source

`source/paged_attention.py` holds:
- `paged_attention_kernel`, the `@triton.jit` kernel: one program per (sequence, query head); it
  reads each sequence's keys and values through its block table (page `block_tables[s, j]` holds the
  sequence's tokens `j·PAGE … j·PAGE + PAGE − 1`), computes the page offset in `int64` (a cache can
  hold more than `2^31` elements), and runs an online softmax over the pages in `float32`;
- `paged_attention`, the host wrapper that launches it;
- `make_cache` and `reference`, test helpers (a random paged cache with distinct random pages per
  sequence, and attention in `float64`);
- the test under `if __name__ == "__main__":`.

There is no upstream file: the source is as written.

## The config

Launched directly: no autotuning. The wrapper launches with grid `(num_seqs, num_heads)` and
`num_warps=4`, and passes `QUERIES_PER_KV = num_heads // num_kv_heads`, `PAGE_SIZE` (the cache's
dimension 1) and `HEAD_DIM` (the query's last dimension) as compile-time values, and
`scale = head_dim ** -0.5`.

## The launches

The wrapper's launches as the test makes them, with shapes as variables:
- `float16` `q` (`[num_seqs, num_heads, head_dim]`), `k_cache`, `v_cache` (`[num_pages, PAGE,
  num_kv_heads, head_dim]`, the same strides) and output; `int32` `block_tables`
  (`[num_seqs, max_pages]`) and `seq_lens` (`[num_seqs]`, contiguous); last strides `1` (asserted);
- the test's head counts, head dims, page sizes and sequence lengths: `(32, 8, 128, 16)`,
  `(8, 8, 64, 16)` and `(16, 2, 128, 32)`, with lengths from `1` to `8191`
  (`[1, 15, 16, 17, 1000, 4096, 333]`, `[5, 64, 2049]`, `[31, 32, 33, 8191]`), `num_pages` the
  sequences' pages plus `50`;
- the wrapper's other shapes: head dims and page sizes any power of two up to `256` (pages of
  one token included), up to `2^16` pages per sequence, long decode sequences (32K to 64K tokens)
  included;
- each sequence's pages inside the cache (`make_cache` makes them distinct; the claim allows
  shared pages too), the cache of any size, more than `2^31` elements included.

## Checked on the H200

`check_gpu.py` (Triton 3.8.0) runs the test's three configurations from its generator and launches
at the edges of the claim (head dims and pages from `1` to `256`, one KV head, 200 sequences,
shared pages, a strided `q`, sequences of `2^16` tokens with pages of 16 and of 1, sequences of 32K
to 64K tokens, empty sequences, sharp attention, values near `2^15` and near `2^-20`). Against
`Post`'s exact reference, every covered element is finite and within its tolerance: worst error /
tolerance `0.88` on the test's launches and `0.997` overall (an output near `10^-7`, where the bound
is the output's `float16` rounding, `2^-25`); the inputs are unchanged and the bytes past `out`
keep their sentinel. `Post` rejects the wrong outputs it tries (all zeros, the last or first page
dropped, the neighbouring key/value head, the scores without `scale`). A cache of more than `2^31`
elements does not fit the check's 4 GB budget; the claim covers it (`setup_bigcache_ok`).

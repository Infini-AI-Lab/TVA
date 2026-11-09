# Setup of `seeded_dropout_f32`

Triton 3.8.0 tutorial 04, "Low-Memory Dropout" (`python/tutorials/04-low-memory-dropout.py`), kernel
`_seeded_dropout` and its host `seeded_dropout`.

## The source

`source/04-low-memory-dropout.py` is the tutorial file, unchanged: the `@triton.jit` kernel
`_seeded_dropout` (lines 110–131), the host `seeded_dropout` (lines 134–140) and the file's
demonstration of it (lines 143–155). The file also holds the baseline kernel `_dropout` (lines
44–63), its host `dropout` (lines 66–72) and its demonstration (lines 76–86), which read a keep mask from memory;
they are not part of this setup. The kernel has no `@triton.autotune` and no `@triton.heuristics`.

## The config

Launched directly: `seeded_dropout(x, p, seed)` allocates `output = torch.empty_like(x)` (it asserts
`x` contiguous) and launches `_seeded_dropout[grid](x, output, n_elements, p, seed,
BLOCK_SIZE=1024)` on the grid `(cdiv(n_elements, 1024),)`, with `n_elements = x.numel()`, the
Python `float` `p` and the Python `int` `seed` as runtime scalars, and no launch option (Triton's
defaults: `num_warps = 4`, `num_stages = 3`, `enable_fp_fusion = True`, `debug = False`).
`tl.rand` uses its default 10 Philox rounds.

## The launches

The claim must cover every launch the file makes through `seeded_dropout`, and every host launch
with the sizes, probabilities and seeds below:
- the file (lines 143–147): `x = torch.randn(10)` (`float32`), `p = 0.5`, the seed `123` twice
  (the same output both times) and the seed `512`;
- `x` any contiguous `float32` vector of `n` elements, `0 ≤ n < 2^31`, any values; `output` fresh;
- `p` any Python `float` whose `float32` lies in `[0, 1]` (a dropout probability; `0`, `-0.0`
  and `1` included);
- `seed` any Python `int` Triton accepts as an argument, `-2^63 ≤ seed < 2^64` (an `int32`,
  `int64` or `uint64` by its value; `1` is a compile-time constant);
- `BLOCK_SIZE = 1024`; the claim also covers direct launches with any power-of-two `BLOCK_SIZE` up
  to `2^20`.

## Checked on the H200

`check_gpu.py` (Triton 3.8.0, one H200): its numpy Philox 4x32-10 matches the three Random123
known-answer vectors, the claim's `rbits`, `rint`, `uRef` and `keep` evaluated in Lean (5 seeds × 4
indices), and `tl.randint4x` and `tl.rand` on the GPU bit for bit (16 seeds covering `int32`,
`int64`, `uint64` and the constant `1`, 2^20 offsets each). On the file's three launches and on 37
more (sizes `0, 1, 2, 1023, 1024, 1025, 4097, 98432, 98433, 2^20, 2^20 + 3, 2^22, 3000001, 2^24,
2^26, 100000`; `p` from `0`, `-0.0` and `1e-50` (`float32` `0`) through `2^-149` and `1 - 2^-24` to
`1 - 2^-26` (`float32` `1`) and `1`, among them `0.1, 0.2, 0.25, 1/3, 0.5, 0.6, 0.7,
0.9, 0.999`; every seed class; `randn` values, random bit patterns and special values: NaN,
infinities, signed zeros, subnormals, quotients beyond `2^126`), and on direct launches with
`BLOCK_SIZE` `1, 16, 128, 1024, 4096, 65536` into sentinel-guarded outputs (a launch at `BLOCK_SIZE = 2^20`, the
top of the covered range, did not finish compiling within 25 minutes and is not run): every dropped element is `+0`
bit for bit; every kept element with a finite `x` and `|x/(1-p)| ≤ 2^126` is within the `/` bound,
worst error 0.500 of it (`p = 0.6`), and 94803836 of the 101027006 kept-element checks (all launches together) equal the
IEEE quotient bit for bit (all of them at `p = 0.5`); `x` is unchanged and no guard element is written;
the file's two seed-`123` launches agree. Nine negative tests (altered outputs, wrong seed, offset
or round count, a write outside `output`) are all rejected: 76 of 76 cases pass.

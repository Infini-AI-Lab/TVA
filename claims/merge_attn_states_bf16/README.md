# `merge_attn_states_bf16`

**The kernel.** `merge_attn_states_kernel` of vLLM (`source/merge_attn_states.py`, kernel lines
65–216; host `merge_attn_states` lines 221–299). A grid of `num_tokens × num_heads` programs; the
program of a token with context (`t < prefill_tokens_with_context`) merges the prefix and suffix
attention outputs of one head by their log-sum-exp values (the larger lse subtracted, `tl.exp`,
`/`, `tl.log` for the merged lse) and stores the `bfloat16` row and, with `return_lse`, the merged
lse; the program of any other token copies the suffix's row and lse. The covered paths are
`OUTPUT_LSE` either way and `USE_FP8 = False` (`SETUP.md`). The claim is
`Claims.MergeAttnStates.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
head `h`:
- for a merged token `t`, with `p`, `s` the two initial lse values (`+∞` read as `−∞`, an empty
  side): if both are `−∞`, the merged lse (when `output_lse` is passed) is exactly `−∞` and every
  output element exactly `+0`; otherwise the merged lse is finite and within
  `2^-16·(1 + |refLse|)` of `refLse = log(e^p + e^s)`, and each output element `j < head_size` is
  finite and within `2^-8·|refOut| + 2^-13·max(|pv|, |sv|) + 2^-133` of
  `refOut = (pv·e^(p−m) + sv·e^(s−m)) / (e^(p−m) + e^(s−m))`, exact in the reals, `pv`, `sv` the two
  initial `bfloat16` inputs;
- for a copied token, the suffix's lse (when `output_lse` is passed) and its output row, bit for
  bit;
- every byte outside the written elements is unchanged.

**The conditions** (`Inst.ok`, `Pre`):
- `PADDED_HEAD_SIZE` is a power of two, at most `2^20`, and at least `HEAD_SIZE`;
- `prefill_tokens_with_context ≤ num_tokens ≤ 2^22`; `num_heads < 2^16`, `num_tokens < 2^31`,
  `num_tokens·num_heads < 2^31`;
- the layouts the kernel indexes: `prefix_output` and `suffix_output` share one head stride, each
  output-like tensor's token stride is `num_heads` times its head stride, `output`'s head stride is
  at least `head_size`, and `output_lse`, when passed, is row-major in one order or the other;
- the strides are below `2^31` and the kernel's `int32` products do not wrap;
- each buffer holds every element the launch touches (the prefix's only for the merged tokens),
  and the outputs share no byte with each other or with the four inputs (the inputs may overlap);
- the buffers are valid, the merged tokens' lse values are not NaN, and, unless both sides are
  empty, their two output rows are finite. Nothing is required of the copied tokens' values.

**The launches covered** (`SETUP.md`): the host's test cases (source lines 335–345:
`setup_t1_ok` … `setup_t4096_ok`, `setup_t300_ok`, `setup_largest_ok`) with `output_lse`, and the
same calls without it (`setup_nolse_*_ok`); `canon_ok` covers the host's layouts at every shape the
conditions allow.

**Device facts** the tolerances rest on (all in `Device P`): `Device.ExpF32` and
`Device.ExpF32NegInf` (`tl.exp` on `float32`, lines 163–164), `Device.LogF32` (`tl.log`, line
168), `Device.DivFull` (`/` on `float32`, lines 197–198), and the totality facts for `exp`, `log`
and `/`.

**Not covered.**
- `int32` offsets of `2^31` or more (a kernel limit: an output of `2^31` elements or more), and
  strides of `2^31` or more;
- a token stride other than `num_heads · head_stride` (a kernel limit: the kernel takes no token
  stride), for example a slice of the heads of a wider tensor;
- NaN lse values or non-finite rows on a merged token, outside real use;
- outputs overlapping the inputs (vLLM's in-place merge), the fp8 output (`USE_FP8 = True`), and
  contexts longer than `2^22` tokens.

`check_gpu.py` (one H200) checks `Post` on the host's test launches, with and without
`return_lse`, and on edge cases.

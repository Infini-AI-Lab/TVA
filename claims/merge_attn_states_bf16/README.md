# Claim: `merge_attn_states_bf16`

Kernel: `source/merge_attn_states.py`, `merge_attn_states_kernel`
(lines 65–216, the kernel lines 61–216; the host lines are 59 and 221–393). Claim:
[`Claim.lean`](Claim.lean), namespace `Claims.MergeAttnStates`, `Statement := ∀ K P, Assumptions K P → Spec K`.
Model map: [`model-map.json`](model-map.json). GPU check: [`check_gpu.py`](check_gpu.py).

The covered paths are the ones `SETUP.md` asks for: `OUTPUT_LSE` either value (`Inst.outLse`:
the host's `return_lse`, an `output_lse` tensor or `None`) and `USE_FP8 = False` (a `bfloat16`
output, no `output_scale`).

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.MergeAttnStates.main : Statement` (`Statement`:
`Claim.lean` line 447). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 237), and every initial memory `init` that `Pre` accepts (line 331), every outcome of
the launch is a normal end (no fault, no hang) whose final memory `fin` satisfies `Post` (line 413).

**What `Post` asks** (line 413), three parts:
1. for every merged `(t, h)`, `t < ptwc`, `h < H`: `MergeOk d init fin t h` (line 385). With `p`,
   `s` the initial `prefix_lse`, `suffix_lse` elements (`rdF`, `plIdx`, `slIdx`) and `corr`
   (line 321) mapping `+∞` to `-∞`:
   - when `d.outLse`, element `olIdx d t h` of `output_lse` reads as some `cl`; if
     `bothEmpty p s` (line 325) then `cl = negInfC`; otherwise `cl` is finite and
     `|cl.toReal - refLse (corr p) (corr s)| ≤ lseTol * (1 + |refLse (corr p) (corr s)|)`;
   - for every `j < head_size`, element `oIdx d t h j` of `output` reads as some `w`; if
     `bothEmpty p s` then `w = zeroB` (`+0`, line 377); otherwise, with `pv`, `sv` the initial
     `prefix_output`, `suffix_output` elements (`poIdx`, `soIdx`), `w` is finite and
     `|w.toReal - refOut …| ≤ bf16Round * |refOut …| + outTol * (|pv.toReal| + |sv.toReal|) + absTol`;
   - `refMax`, `side`, `refSum`, `refLse`, `refOut` (lines 346–361) are exact, in the reals, from
     `init`; the constants `lseTol`, `bf16Round`, `outTol`, `absTol` are lines 365–374;
2. for every copied `(t, h)`, `ptwc ≤ t < T`, `h < H`: `CopyOk d init fin t h` (line 405): when
   `d.outLse` the `output_lse` element is the initial `suffix_lse` element, and every output
   element `j < head_size` is the initial `suffix_output` element, bit for bit. No condition on
   the values;
3. the frame: every byte outside `writeSet d` (line 233: `output.elems (outSet d)`, and
   `output_lse.elems (outLseSet d)` when `d.outLse`) is unchanged.

**What you are given** (`Assumptions`, line 422):
- `faithful` (`Compiler.Faithful`): the device's outcomes are the model's, for a valid launch with
  `debug = false` that is well behaved from `init`;
- `exp_total`, `log_total` (`Rules.Num.MathTotal`): `tl.exp` (source lines 163–164) and `tl.log`
  (source line 168) on `float32` have a result for every input, NaN included;
- `exp_acc` (`Rules.Num.ExpF32`): on `[-X, X]`, within `(2^-22 + 2^-23·X)·e^x + 2^-140`;
- `exp_ninf` (`Rules.Num.ExpF32NegInf`, a `MathAt`): `tl.exp(-∞) = +0`;
- `log_acc` (`Rules.Num.LogF32`): within `2^-22·|ln x| + 2^-149`;
- `div_total`, `div_full` (`Rules.Num.DivTotal`, `Rules.Num.DivFull`): `/` (source lines 197–198) has a
  result, within `2^-22` relative and `2^-148` absolute.

**The kernel, as the model runs it** (`merge_attn_states_kernel` line 51, `model` line 300,
`launch` line 310): a 2-D grid of `T × H` programs, one per `(t, h)`. The `int32` test
`t < ptwc` picks the branch. Copy branch: load `suffix_lse[h, t]`, store it to `output_lse` when
passed, load the masked `[PADDED_HEAD_SIZE]` suffix row and store it to `output`. Merge branch:
two scalar loads, `corr`, `tl.maximum`, two `tl.sub`, two `tl.exp`, `tl.add`; when `output_lse` is
passed, `tl.log`, `tl.add`, a `where` and a scalar store; then two masked row loads, two `/`, the
widening, two products, `tl.addc`, a `where` to `+0`, the rounding to `bfloat16` and the masked row
store. No loop, no reduction, no dot, no atomic. Programs write disjoint elements and only read
the four inputs: the route is `TVA.meets_independent` (PROVER_GUIDE §4).

**What the proof must establish.**
- each launch is valid and has `debug = false`: `Launch.Valid` directly (2-D grid): `T_lt`,
  `H_lt`, `grid_lt`, and the block `[padded]` from `padded_pow2`, `padded_le`;
- each program is `Safe`: the `int32` products do not wrap (`out_i32`, `in_i32`, `lse_i32`,
  `strides_lt`); the kernel's offsets `t·H·phs + h·phs + j` and `t·H·ohs + h·ohs + j` are the
  reference's `poIdx`, `soIdx`, `oIdx` (`pots_eq`, `sots_eq`, `shs_eq`, `ots_eq`); every access is
  inside its buffer (the six `*_len` fields; the prefix's only for `t < ptwc`); `safe_load1`,
  `safe_store1`, `safe_load`, `safe_store`, `safe_maximum`, `safe_sub`, `safe_math`, `safe_add`,
  `safe_div`, `safe_cvt`, `safe_mul`, `safe_addc` line by line;
- the write footprints are disjoint across programs (`ohs_ge` with `ots_eq` for `output`,
  `olse_layout` for `output_lse`), and no program reads what another writes (`output_disj`,
  `output_lse_in_disj`, `output_lse_disj`);
- the value part of part 1: in the both-empty case the `where`s select the constants; otherwise
  `exp_acc`, `exp_ninf`, `log_acc`, `div_full`, `FloatV.sub_err`, `add_err`, `mul_err`,
  `cvt_err`, and `AddRel` for `tl.addc`, under the finiteness `Pre` gives. `## Tolerances` gives
  the derivation. Part 2 is exact;
- the bytes no program writes give part 3.

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable`, `canon_ok`
(line 507) and the `setup_*_ok` theorems; `bytes_disj` (line 495) can be reused.

## Specification

One program per `(t, h)` of the grid `(num_tokens, num_heads)`. `Post` states:

* **Merged tokens** (`t < prefill_tokens_with_context`, `MergeOk`). With `p = prefix_lse[h, t]`,
  `s = suffix_lse[h, t]` (initial values) and `corr` the source's reading of them (`+∞` is `−∞`):
  * both sides empty (`corr p = corr s = −∞`): `output_lse[h, t]` (when passed) is exactly `−∞`
    and every `output[t, h, j]`, `j < head_size`, is exactly `+0`;
  * otherwise, with `m = refMax` (the larger non-empty side), `side x = e^(x−m)` (`0` for an empty
    side), `refSum = side p + side s`:
    * `output_lse[h, t]` (when `output_lse` is passed) is finite and within
      `2^-16 · (1 + |refLse|)` of `refLse = log(refSum) + m`;
    * each `output[t, h, j]` is finite and within
      `2^-8 · |refOut| + 2^-14 · (|pv| + |sv|) + 2^-133` of
      `refOut = (pv · side p + sv · side s) / refSum`, with `pv = prefix_output[t, h, j]`,
      `sv = suffix_output[t, h, j]` (initial values).
* **Copied tokens** (`t ≥ prefill_tokens_with_context`, `CopyOk`): `output_lse[h, t]` (when
  passed) and every `output[t, h, j]` are the suffix's, bit for bit.
* **Frame**: every byte outside the written elements (`writeSet`: `output[t, h, j < head_size]`,
  and `output_lse[h, t]` when `output_lse` is passed) is unchanged.

The reference reads every input by the tensor's own strides (`poIdx`, `soIdx`, `plIdx`, `slIdx`
use `prefix_output.stride(0/1)`, `suffix_output.stride(0/1)`, the lse tensors' strides), and the
outputs by theirs (`oIdx`, `olIdx`), not by the kernel's `token_idx * num_heads * head_stride`;
`Inst.ok` states the layout the kernel assumes (`pots_eq`, `sots_eq`, `ots_eq`, `shs_eq`).

## Conditions

`Inst`: `outLse` (`return_lse`), `num_tokens` (`T`), `num_heads` (`H`), `head_size`, `padded`
(`PADDED_HEAD_SIZE`), `prefill_tokens_with_context` (`ptwc`, `None` read as `T`), each tensor's
own strides and the six buffers (`output_lse`'s only used when `outLse`; without it the host
passes `None` and the strides `0`, `0`, and the model's `output_lse` is `none`). `Inst.ok`:

* `padded` a power of two, `≤ 2^20`, `≥ head_size`; `ptwc ≤ T`;
* the grid: `T < 2^31`, `H < 2^16`, `T·H < 2^31` (`Launch.Valid`);
* the layout the kernel indexes (SETUP.md): one head stride for both inputs, each output-like
  tensor's token stride `H` times its head stride; `head_size ≤ output.stride(1)`; `output_lse`,
  when passed, row-major in one order or the other (the host's fresh tensors);
* the eight strides passed are `int32` arguments (`< 2^31`);
* **the kernel's `int32` products do not wrap** (`out_i32`, `in_i32`, `lse_i32`): a limit of the
  kernel, see Exclusions;
* each buffer holds the last element the launch touches (the prefix's for the merged tokens only;
  `output_lse`'s only when passed);
* the outputs share no byte with each other or with the inputs (the host allocates them).

`Pre`: the buffers are `Valid` (`output_lse` when passed); for every merged `(t, h)`, both lse values are not NaN
(finite of any magnitude, `+∞` or `−∞`), and unless both sides are empty, the two `bfloat16` rows
are finite (any magnitude, subnormals included). Nothing is asked of the copied tokens' values,
nor of the output rows of a both-empty pair.

The launch: grid `(T, H, 1)`, block `[PADDED_HEAD_SIZE]`, `fusion = true`, `debug = false`. It is
within part (A) of `Compiler.Faithful`'s scope: only `num_warps`/`num_stages` defaults, no
heuristics or autotuning, no descriptors, a `JITFunction` launch. The claim
relies on part (B) (the environment) as the rule states it.

## Assumptions

`[rule]` fields, each the library's fact for the model's operation (`docs/rules/num.md`):
`faithful` (`Compiler.Faithful`), `exp_total` (`MathTotal P .exp .binary32`), `exp_acc`
(`ExpF32`), `exp_ninf` (`ExpF32NegInf`), `log_total` (`MathTotal P .log .binary32`), `log_acc`
(`LogF32`), `div_total` (`DivTotal P .binary32`), `div_full` (`DivFull`). `consistent` uses
`Platform.sm90ref`.

## Coverage

`canon T H hs padded ptwc pad lseT outLse` is the host's launch for those sizes (inputs
`[T, H, hs+pad]` sliced to `hs`, lse `[H, T]` or a transpose of `[T, H]`, a fresh contiguous
`output`, and a fresh contiguous `output_lse` when `outLse`, otherwise `None`).
`canon_ok` proves `Inst.ok` for every `T ≤ 2^16`, `H ≤ 64`, `hs ≤ padded ≤ 256` (`padded` a power
of two), `pad ≤ 64`, `ptwc ≤ T`, either lse layout, with or without `output_lse`: every
combination of the code base's sizes (tokens 256 … 4096, heads 4 … 64, head sizes 32 … 256,
`ptwc` `None` or `128`) and of the host's test. The host's nine test cases are `setup_t1`,
`setup_t6`, `setup_t256`, `setup_t613`, `setup_t512`, `setup_t1024`, `setup_t1536`,
`setup_t4096`, `setup_t300` (each `_ok`, with `output_lse`), and the code base's largest shape is
`setup_largest` (4096 tokens, 64 heads, head size 256, `ptwc = 128`). Without `output_lse` (the
backends' calls, and the host's test's second run of each case): `setup_nolse_t613`,
`setup_nolse_t1024`, `setup_nolse_t300`, `setup_nolse_largest`.

## Tolerances

The constants are `lseTol = 2^-16`, `bf16Round = 2^-8`, `outTol = 2^-14`, `absTol = 2^-133`, on
every non-NaN lse and every finite `bfloat16` input; the both-empty output is exact (`+0`). Why they
hold, from the library's facts (`u = 2^-24`):

* `x = fl(side − m) ≤ 0` (IEEE `−`): `x = (side − m)(1+ε)`, `|ε| ≤ u`, or `−∞` when it overflows
  (then `tl.exp` gives `+0` by `ExpF32NegInf`, against a reference weight `e^(x−m) < e^(-2^127)`);
  an empty side gives `−∞ − m = −∞`, so `+0` exactly. The larger side gives `x = 0` exactly.
* `tl.exp(x)` (`ExpF32`, instantiated at `X = |x|`): within `(2^-22 + 2^-23|x|)·e^x + 2^-140`; the
  room condition holds since `e^x ≤ 1`. With the argument's rounding (`e^(xε)`, `|xε| ≤ u|x|`), each
  weight is within `(2^-22 + 1.5·2^-23·|x|)·e^x + 2^-140`, and `|x|·e^x ≤ 1/e`: so both weights are
  within `2^-21` of `side` relative to `refSum ≥ 1`, plus `2^-140`, for every `|x|` (no bound on the
  lse is needed).
* `out_se` (IEEE `+`, operands in `[0, 1+2^-21]`): `refSum·(1 ± 2^-20.5)`, in `[1 − 2^-20, 2 + 2^-19]`.
* `tl.log` (`LogF32`, `x > 0` finite): `log(out_se) ± (2^-22·0.7 + 2^-149)`, so
  `log(refSum) ± 2^-19.9`; then `+ m` (IEEE): `|cl − refLse| ≤ 2^-19.9 + u·|cl|`, and `cl` is finite
  (at `m` near the `float32` maximum, `m + 0.7` rounds to `m`). This is `≤ 2^-19·(1 + |refLse|)`,
  inside `2^-16·(1 + |refLse|)` with a factor 8 to spare.
* `p_scale`, `s_scale` (`DivFull`, quotient `≤ 1 + 2^-19 ≤ 2^126`): within `2^-22·q + 2^-148` of
  the computed quotient, so each scale is within `2^-19.5` (relative to `1`) of `side / refSum`,
  plus `2^-139`.
* The products (exact widening of `bfloat16`, then IEEE `·`) and `tl.addc` (IEEE, or with one
  product contracted, `AddRel`): each within `2u` of its exact value, plus `2^-149`. For every
  finite `bfloat16` input (up to `(2 − 2^-7)·2^127`) nothing overflows `float32`: `|y| ≤
  max(|pv|, |sv|)·(1 + 2^-18) < 2^128`. So the `float32` result `y` satisfies
  `|y − refOut| ≤ 2^-18·(|pv| + |sv|) + 2^-138`.
* The store's rounding to `bfloat16` (`cvt`, nearest even, `u_bf16 = 2^-8`, `η = 2^-134`):
  `|w − y| ≤ 2^-8·|y| + 2^-134`, and `w` is finite (`|y|` stays below the rounding threshold to
  `∞`, `(2 − 2^-8)·2^127`). Hence `|w − refOut| ≤ 2^-8·|refOut| + (1+2^-8)·2^-18·(|pv|+|sv|) +
  2^-134 + 2^-137 ≤ bf16Round·|refOut| + outTol·(|pv|+|sv|) + absTol`.
* Both empty: `max_lse = −∞`, the two `tl.where`s select the constants: `cl = −∞`, and
  `out = +0` in `float32`, whose conversion is the `bfloat16` `+0`.
* Without `output_lse` (`OUTPUT_LSE = False`) the output's computation is the same operations on
  the same values (lines 168–177 only compute and store the lse), so the output bound is the
  same; nothing is stored to an lse.

Meaningful: the output bound is one `bfloat16` rounding (`2^-8`) plus `2^-14` of the inputs'
magnitudes; the H200 run reaches `0.978` of it (the rounding itself), and the lse bound is about
`1.5·10^-5` at `|lse| ≈ 1`.

## Exclusions

* **`int32` offsets (a limit of the kernel).** The kernel forms
  `token_idx * num_heads * head_stride` and `head_idx * head_stride` (output and inputs), and
  `head_idx * lse_head_stride`, `token_idx * lse_token_stride`, in `int32`; they wrap beyond `2^31`,
  and the kernel then reads and writes the wrong elements. `Inst.ok` keeps
  `(T−1)·H·head_stride < 2^31` and the like (`out_i32`, `in_i32`, `lse_i32`). The first host launch
  that hits it has `(T−1)·H·head_stride ≥ 2^31`, for example `T = 131,073` tokens with 64 heads of
  head size 256 (an output of 4.3 GB), or `T = 524,289` with 32 heads of 128. The code base's
  shapes stop at 4096 tokens (`2^26` elements); a vLLM chunked-prefill merge handles one chunk of
  at most `max_num_batched_tokens` tokens, so ordinary configurations do not reach it, but the
  wrapper does not prevent it (any output of `2^31` elements or more, 4 GiB of `bfloat16`).
* **Token stride other than `num_heads · head_stride` (a limit of the kernel).** The kernel takes
  no token stride: an input or output whose token stride differs is indexed wrongly. That happens
  for a slice of the heads of a wider tensor. It also happens with `num_heads = 1`: PyTorch then
  reports the tensor contiguous whatever its `stride(1)` is (for example `(64, 14400, 1)` from a
  `torch.where`), so the token stride need not be `stride(1)`, and the kernel reads the wrong rows
  (error / tolerance about 252 on such a launch). `SETUP.md`'s layouts have it, and `Inst.ok`
  states it (`pots_eq`, `sots_eq`, `ots_eq`).
* **Strides of `2^31` or more** are passed as `int64`, which is another model; such a tensor has
  more than `2^31` elements per head, outside real use.
* **The grid** (`H < 2^16`, `T·H < 2^31`): launches beyond CUDA's and Triton's limits do not run.
* **NaN lse values** on a merged token, and **non-finite `bfloat16` rows** on a merged pair whose
  sides are not both empty: outside real use (the backends give finite lse or `±∞`, and finite
  outputs, `0` for an empty side); the kernel's result there is NaN.
* **Outputs overlapping inputs, the in-place merge** (outside TVA). vLLM's MLA prefill merges a
  continuation chunk in place (`vllm/model_executor/layers/attention/mla_attention.py:2882`:
  `output=prefix_output`, `output_lse=prefix_lse`): each program reads `prefix_output[t, h]` and
  `prefix_lse[h, t]` and then writes the same bytes. A program reading then writing the same bytes
  is out of scope (the library gives no meaning to a program that reads and then writes the same bytes), so the claim does not cover that call; `Inst.ok` keeps the outputs
  disjoint from the inputs (`output_disj`, `output_lse_in_disj`), as the host's fresh outputs are.
* **The fp8 output (`USE_FP8 = True`, an `output_scale`)**: not part of the setup (`SETUP.md`).
* **Where the kernel runs.** vLLM's dispatcher (`vllm/v1/attention/ops/merge_attn_states.py:153-166`)
  sends `bfloat16` inputs whose head size is divisible by `8` to its CUDA op on CUDA platforms, so
  this Triton kernel runs there for the other head sizes, and on the platforms without that op.
  The claim covers the kernel on every launch above, whichever path the dispatcher takes.

## Library requests

None.

## Checks

* **Builds**: `lake env lean ../claims/merge_attn_states_bf16/Claim.lean` from `lean/`: no errors, no warnings,
  no `sorry`, no `axiom`.
* **Not vacuous**: `consistent` (`Platform.sm90ref`) and `satisfiable` (`setup_t1` with every
  byte `0`) are proved.
* **Covers the setup**: `canon_ok` and the fourteen `setup_*_ok` above.
* **GPU** (`check_gpu.py`, H200, Triton 3.8.0), each launch with and without `output_lse`: the
  host's nine test cases, the code base's largest shape, four `Inst.ok` edge launches (one token, one head of size
  1; 3 tokens, 2 heads of size 1 padded by 1; head size 3 padded by 5; 64 heads of 256 padded by 64) and four `Pre` edges (lse of every bit pattern but NaN, huge,
  subnormal and infinite; `bfloat16` of every finite pattern; exact cancellation; both lse at the
  `float32` maximum; any bits, NaN included, in the copied tokens and in both-empty rows). Worst
  output error / tolerance `0.978` (margin `1.02×`, the bound being one `bfloat16` rounding), worst
  lse error / tolerance `6.7·10^-3` (margin `149×`); both-empty lse `−∞` and output `+0`, the copied
  tokens bit for bit, and every input storage unchanged, on every case. Seven wrong outputs on
  the host's case (4096 tokens, 32 heads of 128, `ptwc = 128`) are rejected: all-zero merged
  output (error / tolerance `252`), the suffix's output alone (`1.6·10^4`), the prefix's output
  alone (`1.6·10^4`), an unweighted average
  (`8.2·10^3`), swapped weights (`1.6·10^4`), the lse's log term dropped (`4.5·10^4`), zeroed
  copied tokens (exactness fails).

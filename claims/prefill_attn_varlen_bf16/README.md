# `prefill_attn_varlen_bf16`: the prepared claim

Kernel: `source/prefill_attention.py`, `_fwd_kernel` (lines
51–184; the kernel lines are 47–48 and 51–184, the host lines 45 and 187–293). Claim:
[`Claim.lean`](Claim.lean), namespace `Claims.PrefillAttnVarlen`,
`Statement := ∀ K P, Assumptions K P → Spec K`. Model map: [`model-map.json`](model-map.json). GPU
check: [`check_gpu.py`](check_gpu.py).

The covered paths are `SETUP.md`'s: `IS_CAUSAL = True` and `IS_CAUSAL = False` (`Inst.causal`),
`bfloat16` `q`, `k`, `v`, `o`, `int32` `b_start_loc`, `b_seq_len`, on an H200
(`BLOCK_M = BLOCK_N = 128`).

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.PrefillAttnVarlen.main : Statement` (`Statement`:
`Claim.lean` line 523). Unfolded (`Spec`, line 495, is `Claim.Meets Inst.ok Pre Post K`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 270), and every initial memory `init` that `Pre` accepts (line 418), every outcome of
the launch is a normal end (no fault, no hang) whose final memory `fin` satisfies `Post` (line 487).

**What `Post` asks** (line 487), two parts:
1. for every `b < batch`, `p < lenOf d init b` (line 361), `h < H`, `e < Lk`: element
   `oIdx d (startOf d init b + p) h e` (lines 257, 359) of `o` reads in `fin` (`rdB`, line 340) as
   a `bfloat16` `c` that is finite, with
   `|c.toReal − attnRef| ≤ tolRel d.BD (trips (lastKey d init b p)) (Xrow d init b h p) · attnAbs + tolAbs (vSum)`
   (all at `d init b h p e`). `attnRef` (line 427) is the exact softmax-weighted mean of `v`, in the
   reals, over the keys `j ≤ lastKey` (line 387: `p` when `causal`, `lenOf − 1` when not), with the
   exact scores `score` (line 375) from `init`; `attnAbs` (line 433) is the same mean of `|v|`,
   `vSum` (line 439) the plain sum of `|v|`. `trips` is line 444, `Xrow` (line 391) the largest
   `scoreMag` (line 381) over the attended keys. The tolerance is `tolRel` (line 472, built from
   `dlt`, `rho`, `gam`, `errE`, lines 453–466) and `tolAbs` (line 475). Every element of every
   sequence is asked; there is no further guard;
2. the frame: every byte outside `d.o.elems (outElems d init)` (line 478) is unchanged.

**What you are given** (`Assumptions`, line 499), eight fields:
- `faithful` (`Compiler.Faithful`): the device's outcomes `K d` are outcomes of the model's launch,
  when the launch is valid, has `debug = false`, and is well behaved from `init`;
- `dot` (`Num.DotBF16`): `tl.dot` of `bfloat16` into `float32` is within `relDot16 K = (K+2)·2^-22`
  of `|acc| + Σ liftMag a·liftMag b`, plus `(K+2)·2^-126` (source lines 132, 171);
- `dot_total` (`Num.DotTotal P cfgBF16`): that `tl.dot` has a result for all operands;
- `exp_acc` (`Num.ExpF32`): `tl.exp` on `[-X, X]` is within `(2^-22 + 2^-23·X)·e^x + 2^-140`
  (source lines 149, 153, 154);
- `exp_ninf` (`Num.ExpF32NegInf`): `tl.exp(−∞) = +0` (masked lanes; the first trip's `alpha`);
- `exp_total` (`Num.MathTotal P .exp .binary32`): `tl.exp` has a result for every input;
- `div_full` (`Num.DivFull`): `float32` `/` is within `2^-22·|x/y| + 2^-148` for
  `|x/y| ≤ 2^126` (source lines 158, 161);
- `div_total` (`Num.DivTotal P .binary32`): `/` has a result for every pair of inputs.

**The kernel, as the model runs it** (`fwd_kernel` line 58, `model` line 321, `launch` line 334):
a 3-D grid `(batch, H, nBlocks d)` (`nBlocks`, line 267: `cdiv(maxLen, 128)`). Program
`(b, h, m)` loads `b_seq_len[b]` and `b_start_loc[b]` (`tl.load1`), takes the key/value head
`h / kvg d` (`tl.idiv`; `kvh`, line 264), and loads the `[128, BD]` tile of `q` at rows
`128·m + [0, 128)` of sequence `b`, masked to rows below `len_b` and columns below `Lk`. The loop
(`tl.forRange`, line 126) runs `0 … block_mask·end_n` in steps of 128: no trip when
`128·m ≥ len_b`, else `m + 1` trips when causal and `cdiv(len_b, 128)` when not; for every row
`p < len_b` of the block that is `trips (lastKey d init b p)`. Each trip loads a `[BD, 128]` tile of
`k` and a `[128, BD]` tile of `v` (masked keys below `len_b`, columns below `Lk`), computes the
scores (`tl.dot_add`, line 141, then `tl.mul_add` by `sm_scale` with the `0`/`−∞` mask), the row
maximum (`tl.max1`), `p = tl.exp(qk − m_ij)`, `tl.sum1`, the online-softmax update of `m_i`, `l_i`
(`tl.maximum`, `tl.exp`, `tl.addc`, `tl.div`, `tl.mul`), rescales `acc`, rounds `p` to `bfloat16`
(`tl.cvt`) and adds `tl.dot_add p v` (line 190). After the loop it rounds `acc` to `bfloat16` and
stores it (line 201) to the same rows of `o`, head `h`. Programs read only the five inputs, write
disjoint parts of `o`, and use no atomics: the route is `TVA.meets_independent` (PROVER_GUIDE §4).
The loop rule is `safe_forRange` with an invariant on `(m_i, l_i, acc)`; the body writes nothing
(per-trip footprint `∅`, `iterW_empty`).

**What the proof must establish.**
- each launch has `debug = false` (the default in `launch`) and is `Launch.Valid`: `batch_lt`,
  `H_lt`, `nBlocks d < 2^16` (from `maxLen_le`), `grid_lt`, the default `numCtas` and `pdl`, and the
  block shapes `TritonOk` (`bd_pow2`, `bd_le`). `Launch.valid_grid1` is for 1-D grids only; prove
  the seven parts directly;
- each program is `Safe`: the scalar loads in bounds (`bsl_len`, `bsq_len`, `Valid` from `Pre`;
  `safe_load1_buf`) with `SeqOk` (line 397) giving `0 ≤ start`, `len ≤ maxLen`,
  `start + len ≤ total`; the `tl.idiv` divisor `kvg d ≥ 1` (`kv_pos`, `kv_dvd`), and `kvh d h < HKV`;
  every enabled lane's `int32` offset is the exact element index, below `2^31` and within its
  buffer (`strides_lt`, `q_i32` … `o_i32`, `q_len` … `o_len`); the store writes distinct elements
  (`o_layout`); `tl.exp`, `/`, `tl.dot` have results (`exp_total`, `div_total`, `dot_total`);
- footprints: each program writes only its rows of `o` at head `h`; those are disjoint across
  programs by `o_layout` and `SeqOk`'s packing (no two non-empty sequences share a token), and no
  program reads `o` (`disj`), which gives both `hdisj` and `hRW`;
- the value part of `Post`: a per-row loop invariant for the online softmax (README `## Tolerances`,
  steps 3–5), from `ValsOk` (line 408: finite inputs, `|v| ≤ 2^120`, `scoreMag ≤ 2^12` on every pair
  of the sequence, masked or not), `dot` (one step: `DotLift.dotAdd_step`), `exp_acc`
  (`MathAcc.mono`), `exp_ninf`, `div_full`, `sum_err_covers`, `FloatV.mul_err`, `FloatV.cvt_err`,
  `FloatV.AddRel.err`; then the store's rounding and the assembly into `tolRel·attnAbs + tolAbs`
  (steps 6–7);
- the frame: the union of the write footprints is `d.o.elems (outElems d init)`, and the bytes no
  program writes keep their `init` value.

**Already in `Claim.lean`** (do not prove again): `consistent` (line 527), `satisfiable` (line
1031), `canon_ok` (line 583), `fusedQKV_ok` (line 832) and the `setup_*_ok` theorems;
`bytes_disj` (line 565) can be reused.

## Specification

Sequence `b` is the tokens `start_b … start_b + len_b − 1` (`b_start_loc[b]`, `b_seq_len[b]`, read
from the initial memory) of `q`, `k`, `v`. Every tensor is read by its own strides (`qIdx`,
`kIdx`, `vIdx`, `oIdx`), and query head `h` reads key/value head `h // (num_heads //
num_kv_heads)` (the host reference's `repeat_interleave`). For query `p < len_b` of sequence `b`,
head `h`, with `σ = sm_scale` (the double), the query attends the keys `j ≤ L`, `L = lastKey`:
`L = p` when causal, `L = len_b − 1` (every key of the sequence) when not:

* `score_j = σ·Σ_{e<Lk} q[start_b+p, h, e]·k[start_b+j, g, e]` for the keys `j ≤ L`;
* `attnRef[e] = Σ_{j≤L} e^{score_j}·v[start_b+j, g, e] / Σ_{j≤L} e^{score_j}` (exact softmax);
* `attnAbs[e]` the same weighted mean of `|v|` (`≥ |attnRef|`), `vSum[e] = Σ_{j≤L} |v_j[e]|`;
* `Xrow = max_{j≤L} σ·Σ_e liftMag(q_e)·liftMag(k_je)` (`Rules.Num.liftMag .bf16`, a subnormal
  counted at `2^-126`; `Xrow ≥ max|score_j|`), and `T = L // 128 + 1` (the trips its program
  makes: `p // 128 + 1` when causal, `cdiv(len_b, 128)` when not).

`Post`: every `o[start_b + p, h, e]` (`b < batch`, `p < len_b`, `h < num_heads`, `e < Lk`) is a
finite `bfloat16` `c` with

`|c − attnRef| ≤ tolRel(BLOCK_DMODEL, T, Xrow)·attnAbs + 2^-80·(1 + vSum)`,

and every byte outside those elements is unchanged. With `D = BLOCK_DMODEL`:

* `δ = (D + 8)·2^-22·X + 2^-40` (each computed score's error),
* `ρ = exp(2^-8 + (T + 8)·2^-18 + (X + δ)·2^-17) − 1` (each key's coefficient and the normaliser),
* `γ = exp(T·136·2^-22) − 1` (the second `tl.dot` over `T` trips),
* `E = e^{2δ}·(ρ + γ·(1 + ρ))/(1 − ρ) + (e^{2δ} − 1)`, `tolRel = 2^-8 + (1 + 2^-8)·E`.

| `D` | `T` (tokens) | `X` | `tolRel` |
|---|---|---|---|
| 128 | 1 (≤ 128) | 8 | `8.5·10^-3` |
| 128 | 16 (≤ 2048) | 8 | `9.1·10^-3` |
| 128 | 64 (≤ 8192) | 16 | `1.1·10^-2` |
| 128 | 1024 (≤ 2^17) | 16 | `4.8·10^-2` |
| 128 | 1 | 256 | `2.7·10^-2` |
| 256 | 1024 | 4096 (the cap) | `0.81` |
| 128 | 43 (5476, not causal) | 8 | `1.0·10^-2` |
| 128 | 512 (65536, not causal) | 8 | `2.8·10^-2` |

On the host's `randn` inputs `Xrow` is about `0.64·Lk·σ` (`≈ 7` at head dim 128). The floor
`≈ 2·2^-8` is the two `bfloat16` roundings the kernel makes (`p.to(bfloat16)` and the store); the
GPU check reaches `0.86` of the bound on an adversarial input and `0.80` on the host's. Against
the output itself, on `randn` inputs, `tol / |attnRef|` has median `0.37` (90th percentile `2.1`)
at 64/8 heads of 128, causal, prompts of 8192 and 4096, and median `0.54` (90th percentile `2.9`)
at Qwen2.5-VL's 16 heads of 80, not causal, one image of 5476 patches: the bound is relative to
`attnAbs`, which a mean over many keys of random sign makes much larger than `|attnRef|`.

## Conditions

`Inst`: `batch`, `total` (`q.shape[0]`), `H`, `HKV`, `Lk`, `BD` (`BLOCK_DMODEL`), `maxLen`
(`max_input_len`), `sm_scale` (the double), each tensor's two outer strides, and the six storages.
`Inst.ok`:

* the grid `(batch, H, cdiv(maxLen, 128))` within CUDA's limits, fewer than `2^31` programs;
* `1 ≤ HKV`, `HKV ∣ H` (SETUP.md); `1 ≤ Lk ≤ BD ≤ 256`, `BD` a power of two;
* `maxLen ≤ 2^17` (see Exclusions);
* `total ≤ 2^22`: the packed batch's tokens (`total_tokens`) are a context length, at most `2^22`
  tokens (real model shapes, see Exclusions);
* `batch ≤ 8192`: the number of sequences (`b_seq_len.shape[0]`) is a batch size, at most `8192`
  (real model shapes, see Exclusions);
* `sm_scale` finite in `[2^-64, 2^64]` (the default is `1/sqrt(Lq) ∈ [1/16, 1]`);
* the eight strides are `int32` arguments (`< 2^31`);
* **every element index of `q`, `k`, `v`, `o` is below `2^31`** (`q_i32`, …): a limit of the
  kernel, see Exclusions;
* each storage holds its tensor; `b_start_loc`, `b_seq_len` hold `batch` elements;
* `o` row-major in one order or the other (`empty_like(q)`), and disjoint from the five inputs.

`Pre`: the six buffers `Valid`; `SeqOk`: each `b_start_loc[b]`, `b_seq_len[b]` with
`0 ≤ start`, `0 ≤ len ≤ maxLen`, `start + len ≤ total`, and no two sequences sharing a token
(packed); `ValsOk`: in every sequence, every `q`, `k`, `v` component finite, `|v| ≤ 2^120`, and
every query–key pair of the sequence (causal-masked pairs included, since the kernel computes
their scores before masking) with `σ·Σ liftMag q·liftMag k ≤ 2^12`. `Inst.causal` is the host's
`is_causal`, either value.

The launch: grid `(batch, H, cdiv(maxLen, 128))`, the source's block shapes, `fusion = true`,
`debug = false`, `num_ctas = 1`, no `launch_pdl`; only `num_warps`/`num_stages` among the other
options, no heuristics or autotuning, a `JITFunction` launch, no descriptors: within part (A) of `Compiler.Faithful`'s scope. The claim relies on part (B) (the
environment) as the rule states it.

## Assumptions

`[rule]` fields (`docs/rules/num.md`'s table): `faithful` (`Compiler.Faithful`), `dot`
(`DotBF16`), `dot_total` (`DotTotal P cfgBF16`), `exp_acc` (`ExpF32`), `exp_ninf`
(`ExpF32NegInf`), `exp_total` (`MathTotal P .exp .binary32`), `div_full` (`DivFull`), `div_total`
(`DivTotal P .binary32`). `consistent` uses `Platform.sm90ref`.

## Model notes

`alpha * l_i + beta * l_ij` is `tl.addc` of two products; `qk *= sm_scale` and
`qk += tl.where(…)` are one `tl.mul_add`, whose addend is the select of the branch the
compile-time `if IS_CAUSAL:` visits (the non-causal `[1, BLOCK_N]` select written at the broadcast
shape `[BLOCK_M, BLOCK_N]`); `end_n` is the side of the compile-time conditional expression
`not IS_CAUSAL` picks; both dots are `tl.dot_add`; `block_start_loc` is in the source's order
(`BLOCK_M * start_m`).

## Coverage

`canon batch total H HKV Lk BD maxLen sm causal` is the host's launch (contiguous `q`, `k`, `v`,
`o = empty_like(q)`). `canon_ok` proves `Inst.ok` for every `batch ≤ 8192`, `total ≤ 2^17`,
`1 ≤ H ≤ 64`, `HKV ∣ H`, `1 ≤ Lk ≤ BD ≤ 256` (`BD` a power of two), `maxLen ≤ 2^17`, every
`sm_scale` in range and both values of `causal`. The host's eight test cases (causal) are `setup_h128`, `setup_h96`, `setup_h80`,
`setup_h13`, `setup_llama` (32/8, 128, lengths 1 … 1000), `setup_qwen` (28/4, lengths up to
2048), `setup_vision` (16/16, 80), `setup_h64` (8/1, 64), each `_ok` with the host's
`1/sqrt(Lq)` (the doubles `sm128`, …, `sm64`); SETUP.md's model shapes at their largest are
`setup_large` (64 heads, 8 KV heads, head dim 128, two prompts of 8192 tokens). SETUP.md's strided inputs are `setup_fused`: `q`, `k`, `v` views of one fused
`[total, (H + 2·Hkv)·D]` `qkv` tensor (token stride 6144, Llama-3-8B's 32/8 heads, the host's
lengths), with `o = empty_like(q)` contiguous.

The vision encoders' launches (`is_causal=False`): `fusedQKV batch total H HKV Lk BD maxLen sm
causal` is the encoder's launch on the views of its fused `qkv` projection (token stride
`(H + 2·HKV)·Lk`) with `o = empty_like(q)` contiguous; `fusedQKV_ok` proves `Inst.ok` for every
`batch ≤ 8192`, `total ≤ 2^22`, `1 ≤ H ≤ 64`, `HKV ∣ H`, head dims up to 256, `maxLen ≤ 2^17` and a
`qkv` of at most `2^31` elements. SETUP.md's shapes: `setup_vit_full` (Qwen2.5-VL, 16 heads of 80, one image
of 5476 patches), `setup_vit_window` (the same image in 100 windows of at most 64 patches),
`setup_vit_max` (one image of 65536 patches, `SGLANG_IMAGE_MAX_PIXELS`), `setup_vit_max_tp8` (the
same, 2 heads per rank), `setup_vit_video` (16 frame pairs of 3072 patches), `setup_vit_qwen3`
(Qwen3-VL, 16 heads of 72), `setup_vit_mimo` (MiMo-VL, 32 heads of 64 on 8 KV heads), and
`setup_vit_scale1` (`sm_scale = 1.0` on contiguous inputs, as Gemma 4's tower passes).

## Tolerances

Derived from the library's facts (`u = 2^-24`, `docs/rules/num.md`). Fix a row (`b`, `h`, `p`),
`L = lastKey`, `X = Xrow`, `N = 128`, `D = BLOCK_DMODEL`; the row's trips are `t = 0 … T−1`
(`start_n = 128·t`), and every key a trip enables is attended except, when causal, the keys above
`p`; each trip has at least one attended key (key `start_n ≤ L`: `start_n ≤ 128·start_m ≤ p` when
causal, `start_n < len_b` when not), so every trip's row maximum is finite. The two modes differ
only in the set of attended keys, `j ≤ L`, which the steps below sum over.

1. **Scores.** `qk₁ = tl.dot_add(q, k, 0)`: `DotBF16` (`relDot16 D = (D+2)·2^-22` of
   `L = Σ liftMag q·liftMag k = scoreMag/σ`, plus `(D+2)·2^-126`; room since `L ≤ 2^76`), or the
   dot followed by `tl.addc` with `+0` (`AddRel.err`: `3u` of the operands' magnitudes, plus `4η`,
   more; either form within `DotLoop.stepRel relDot16 D = (D+2)·2^-22 + 3u·(1 + (D+2)·2^-22)`
   of `L`, `DotLift.dotAdd_step`). `σ32 = hostF32(σ)`: `u`. `tl.mul_add(qk₁, σ32, mask)` (either
   mode's mask): at an attended lane `fl(qk₁·σ32) + 0` or `fma(qk₁, σ32, 0)`, within `3u` of
   `|qk₁·σ32|` plus `4η` (`FusedErr.mulAdd_err`; in fact one rounding); at a masked lane `−∞`
   exactly (the product is finite: `scoreMag ≤ 2^12` on every pair). In units of `2^-22`
   (`u = 0.25`): `D + 2 + 0.75 + 0.25 + 0.75 = D + 3.75`, and the products of these small factors
   add less than `0.01`. So every attended `qk_j = score_j + ε_j`, `|ε_j| ≤ ((D+4)·2^-22)·X +
   2^-52 ≤ δ` (`δ`'s factor is `D+8`; the absolute part is the dot's `(D+2)·2^-126` times `σ32 ≤
   2^64`, below `2^-53`).
2. **Effect of the scores.** With `ŵ_j ∝ e^{score_j}` and `ŵ'_j ∝ e^{qk_j}` (normalised over
   `j ≤ L`), `ŵ'_j/ŵ_j ∈ [e^{-2δ}, e^{2δ}]`, so `|Σŵ'v − attnRef| ≤ (e^{2δ}−1)·attnAbs` and
   `A' = Σŵ'|v| ≤ e^{2δ}·attnAbs`.
3. **The online softmax, exactly.** With `m_t` the trip's row maximum and `M_t` the running one,
   the products of the kernel's factors telescope: key `j` of trip `t` reaches the final `acc`
   with the coefficient `c_j = p_j·(β_t/L_t)·Π_{t'>t}(L_{t'−1}/L_{t'})·α_{t'}` (the same computed
   `L`'s cancel), equal to `e^{qk_j − M_{T−1}}/L_{T−1}` in exact arithmetic, and
   `L_{T−1} = Σ_j e^{qk_j − M_{T−1}}`; so `c_j = ŵ'_j` without rounding.
4. **The coefficients' errors** (`1+κ_j` for `c_j`, `1+λ_j` for each key's share of `L`), each a
   product of factors `1 ± r`:
   * `tl.exp` (`ExpF32` at `X' = |x|`): `2^-22 + 2^-23·|x|` relative, `2^-140` absolute; the
     argument's IEEE `−`: `e^{±u|x|}`. For `p_j` and `β_t`, `|x| ≤ 2(X + δ)`; for the `α`'s, the
     arguments `M_{t'} − M_{t'−1} ≥ 0` sum to at most `2(X+δ)` (the running maximum only grows),
     so all `α`'s together give `T·2^-22 + 1.5·2^-23·2(X+δ)`. The first trip's `α = exp(−∞) = +0`
     (`ExpF32NegInf`).
   * `tl.sum1` of `p` (128 nonnegative lanes, any tree; `sum_err_covers`): `127u`.
   * `tl.mul`s (`β·l`, `α·L`, `p·p_scale`, `(L/L')·α`, `acc·acc_scale`): `u` each; `tl.addc`
     (`α·L + β·l`, nonnegative operands): `3u`; `DivFull` (`β/L`, `L/L'`, quotients `≤ 2^20`, since
     `L_t ≥ 1/2`: each trip's `β` or `α` is `exp(0)`): `2^-22` and `2^-148`.
   * `p.to(bfloat16)`: `2^-8` relative, `2^-134` absolute.

   Summed: per trip at most `18u`, otherwise `2^-8 + 152u`, and `18·2^-23·(X+δ)` from the growing
   `tl.exp` error, so `K + Λ ≤ ρ` (`K = max|κ_j|`, `Λ = max|λ_j|`): the proof needs
   `18u·T + 152u ≤ (T+8)·2^-18` and `18·2^-23 ≤ 2^-17`, that is `18·T + 152 ≤ 64·(T+8)` and
   `18 ≤ 64`, true for every `T ≥ 0` (the right sides are at least `3.37×` the left, the least at
   `T = 0`, and `3.55×`); `|Π(1±r) − 1| ≤ e^{Σr} − 1`. Then `|c_j/ŵ'_j − 1| ≤ (K+Λ)/(1−Λ) ≤ ρ/(1−ρ)`, and
   `A_c = Σ c_j|v_j| ≤ (1+ρ)/(1−ρ)·A'`.
5. **The second `tl.dot`** (`acc₂ = tl.dot_add(p₂, v, acc₁)`): `DotBF16` with `K = N = 128`:
   `relDot16 128 = 130·2^-22` of `|acc₁| + Σ liftMag p₂·liftMag v` (room: everything is below
   `3·2^120`), or the dot then `tl.addc` with `acc₁` (`AddRel.err`: `3u` more, `130.75·2^-22` in all,
   `DotLoop.stepRel relDot16 128`). Carried
   to the end by the later `acc_scale`s, trip `t`'s error is at most `132·2^-22` of
   `B + (earlier errors)`, `B ≈ A_c` the final-scaled `Σ c_j|v_j|`; by induction the total is at
   most `(e^{T·132·2^-22} − 1)·B ≤ γ·B` (`136` for the margin and the `acc·acc_scale` roundings).
6. **Assembly.** `|acc − attnRef| ≤ γ·A_c + ρ/(1−ρ)·A' + (e^{2δ}−1)·attnAbs + abs ≤
   E·attnAbs + abs`. The store: `|c − acc| ≤ 2^-8·|acc| + 2^-134`, `|acc| ≤ attnAbs·(1+E) + abs`,
   so `|c − attnRef| ≤ (2^-8 + (1+2^-8)·E)·attnAbs + (1+2^-8)·abs + 2^-134`, and `c` is finite
   (`|acc| < 2^122`).
7. **Absolute terms.** Each key's coefficient has an absolute error below `2^-133` (the
   `bfloat16` subnormal step, `tl.exp`'s `2^-140`, `DivFull`'s `2^-148`), carried by later
   scalings of at most `2^9·T`; the dots' lifts of subnormals add `2^-126` per key or `p₂`, and
   `(N+2)·2^-126` per trip: in all `abs ≤ 2^-123·T·vSum + 2^-109·T² + 2^-134 ≤ 2^-80·(1+vSum)`
   for `T ≤ 1024` (`T ≤ cdiv(2^17, 128)` in either mode).

The bound is provable from `DotBF16`, `ExpF32`, `ExpF32NegInf`, `DivFull` and the library's IEEE
rounding lemmas (`FloatV.mul_err`, `add_err`, `sub_err`, `cvt_err`, `AddRel.err`, `mulAdd_err`,
`sum_err_covers`, `DotLift.dotAdd_step`); `ρ < 1/2` on every covered row (`T ≤ 1024`, `X ≤ 2^12`: `ρ ≤ 0.04`).
Meaningful: `tolRel ≈ 0.85·10^-2` of the weighted mean of `|v|` at the host's sizes, about
`1.16×` the kernel's worst error on an adversarial input; it grows with `T` because the library's
`tl.dot` fact charges `130·2^-22` of the accumulator per key block.

## Exclusions

* **`int32` element offsets (a limit of the kernel).** The kernel computes
  `(start + offs_m)·stride_qbs + cur_head·stride_qh + offs_d` (and the same for `k`, `v`, `o`) in
  `int32`; beyond `2^31` it wraps and the kernel reads and writes the wrong elements. `Inst.ok`
  keeps every element index of the four tensors below `2^31` (`q_i32`, `k_i32`, `v_i32`,
  `o_i32`). The first host launch that hits it has `total_tokens·num_heads·head_dim ≥ 2^31`
  (contiguous tensors), for example `524,289` tokens at 32 heads of 128 (a 4 GiB `q`); SETUP.md's
  prompts of thousands of tokens are far below, but a packed batch of very long prompts (half a
  million tokens at that shape) reaches it, and the wrapper does not check.
* **`max_input_len > 2^17`.** The kernel is correct there (up to the `int32` limit), but the
  library's `tl.dot` fact (`130·2^-22` per key block on the accumulator) makes the bound grow with
  the number of trips: `tolRel ≈ 0.05` at `2^17` tokens, about `0.09` at `2^18`. SETUP.md asks
  for prompts of 1 to thousands of tokens; `2^17` is far beyond. Not a kernel limit.
* **Contexts longer than `2^22` tokens, batches of more than `8192` (out of scope).** Dimensions
  that are not multiples of 128 or exceed `2^17`, contexts longer than `2^22` tokens, and batches of
  more than `8192` requests: the claim covers real model shapes only. The fields are `total ≤ 2^22`
  (`total_le`, the packed batch's tokens) and `batch ≤ 8192` (`batch_le`, the sequences: requests,
  or a vision encoder's images, frame pairs or windows); each sequence is already within
  `maxLen ≤ 2^17`. The kernel has no model dimension (`Lk` is a head dimension, `H`, `HKV` head
  counts). Every launch of `SETUP.md` meets them (`total ≤ 2^16`, at most `100` sequences).
* **Head dims of 8 or less** (`BLOCK_DMODEL < 16`) are rejected on the host (`tl.dot` needs
  `K ≥ 16`; `check_gpu.py` confirms it at `Lk = 1` and `8`): no outcome. They stay in `Inst.ok`
  (the claim is about the model there too). Head dims above 256 are outside real use (SETUP.md:
  13 to 128).
* **Input values.** Non-finite `q`, `k`, `v` in a sequence (a NaN or `∞` anywhere in a sequence
  can reach a row through `0·NaN` or a masked score), `|v| > 2^120` (the `float32` accumulator can
  overflow near `2^127`: a limit of the kernel at that magnitude), and query–key pairs with
  `σ·Σ liftMag q·liftMag k > 2^12` (real scores are far below; beyond it the bound would be
  vacuous): outside real use.
* **`sm_scale` outside `[2^-64, 2^64]`**, negative or zero: outside real use.
* **Overlapping sequences**, `o` overlapping the inputs: the host's packed sequences and fresh
  `o` (two sequences sharing a token would make two programs write one element).
* **The grid** beyond CUDA's limits (`num_heads ≥ 2^16`, `2^31` programs): no launch.
* **`int64` `b_start_loc`/`b_seq_len`** (the code base's unit test), other dtypes: not part of
  SETUP.md's launches.

## Library requests

None.

## Checks

* **Builds**: `Claim.lean` builds with no errors or
  warnings, no `sorry`, no `axiom` (`#print axioms`: `propext`, `Classical.choice`, `Quot.sound`).
* **Not vacuous**: `consistent` (`Platform.sm90ref`); `satisfiable` (one sequence of one token,
  head dim 16, zero inputs, `b_seq_len[0] = 1`).
* **Covers the setup**: `canon_ok`, `fusedQKV_ok` and the eighteen `setup_*_ok` above.
* **GPU** (`check_gpu.py`, H200, Triton 3.8.0), each launch causal and not unless named: the
  host's eight test cases; a fused-`qkv` strided causal launch (`setup_fused`'s shape); SETUP.md's
  largest causal model shape (64/8 heads, head dim 128, prompts of 8192 and 4096); the vision
  encoders' non-causal launches on fused-`qkv` views (Qwen2.5-VL's 16 heads of 80 on one image of
  5476 patches, on its 100 windows, on 16 video frame pairs of 3072 patches, and 2 heads on one
  image of 65536 patches; Qwen3-VL's 16 heads of 72; MiMo-VL's 32 heads of 64 on 8 KV heads), and
  `sm_scale = 1.0` on unit-RMS inputs; `Inst.ok` edges (head dims 9 and 256, 64 heads on one KV
  head, gaps and empty sequences between sequences, `sm_scale = 0.5`, one prompt of `2^17`
  tokens); `Pre` edges (scores scaled 36× to `Xrow` in the hundreds, `|v| ≈ 10^30`,
  `|v| ≈ 10^-37` (subnormal `bfloat16`), and uniform attention with `v` constant, where every
  `bfloat16` rounding of the weights pushes the same way). Worst error / tolerance `0.859`
  (margin `1.16×`, the adversarial uniform causal case; `0.80` on the host's own tests, `0.76` on
  the non-causal launches); largest `tolRel` met `4.7·10^-2` (the `2^17`-token prompt); every
  written element finite, every input unchanged, and every element of `o` outside the sequences
  keeps its initial bits. Head dims 1 and 8 are rejected on the host as expected.
* **Negative tests**: on the strided causal launch and four non-causal ones (`vit-full`,
  `vit-window`, `vit-mimo`, `vit-scale1`), `Post` rejects an all-zero output, the kernel's output
  in the other mode (causal for a non-causal launch and back), the output without the last key
  block of each row, without the first key, and (grouped heads) with each query head on the
  wrong key/value head: the smallest worst error / tolerance among them is `2.6` (the first key
  left out of 5476 keys).

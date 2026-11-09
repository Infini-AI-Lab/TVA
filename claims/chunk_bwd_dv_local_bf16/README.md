# `chunk_bwd_dv_local_bf16`: the prepared claim

fla's `chunk_bwd_kernel_dv_local` (`source/chunk_o.py`, kernel lines 74–149 with the helper `exp2`
70–71; host `chunk_bwd_dv_local` 191–241). Claim: [`Claim.lean`](Claim.lean), namespace
`Claims.ChunkBwdDvLocal`, `Statement := ∀ K P, Assumptions K P → Spec K`. Model map:
[`model-map.json`](model-map.json). GPU check: [`check_gpu.py`](check_gpu.py).

The covered path is `SETUP.md`'s: `bfloat16` `q`, `k`, `do`, `dv`, an optional `float32` gate `g`
(`USE_G` either way), `USE_G_GAMMA = USE_A = False`, fixed-length rows or a packed
variable-length batch (`IS_VARLEN` either way, `int32` or `int64` tables), on an H200
(`BK`, `BV` the host's `min(max(next_power_of_2(·), 16), 128)`).

## Specification

Sequences (`nSeq`, `lo`, `hi`) are the rows `[b·T, b·T + T)` of a fixed-length batch, or
`[cu_seqlens[n], cu_seqlens[n+1])` of a variable-length one, in the flat token axis; token `p`'s
chunk ends at `chunkEnd = min(chunkStart + BT, hi)`, chunks of `BT` tokens counted from the
sequence's start. For value head `hv` the query/key head is `h = hv // (HV // H)` (`hk`). With
`σ` the host's double `scale` (`sig`), exactly:

* `kq(p, j) = Σ_{c<K} k[p, h, c]·q[j, h, c]`, and `kqMag(p, j) = Σ_c liftMag k·liftMag q`
  (`Rules.Num.liftMag .bf16`: a subnormal counted at `2^-126`; `≥ |kq|`);
* `wt(p, j) = 2^{g[j, hv] − g[p, hv]}` with a gate, `1` without (`dg` the exponent);
* `Aex(p, j) = σ·kq·wt`, the entry of the causal-masked matrix;
* `ref(p, hv, e) = Σ_{j=p}^{chunkEnd−1} Aex(p, j)·do[j, hv, e]`.

The reference is computed from `cu_seqlens` (or the rows), `q`, `k`, `g` and `do` alone, never
from `chunk_indices` or the kernel's indexing.

`Post`: every `dv[p, hv, e]` (`p` a token of a sequence, `hv < HV`, `e < V`) is a finite
`bfloat16` `c` with `|c − ref| ≤ tol`, and every byte outside those elements is unchanged.
Over `j ∈ [p, chunkEnd)`, with `u = 2^-24`, `nK = cdiv(K, BK)`:

* `wErr = 2^{dg}·(2^{u·|dg| + 2^-149}·(1 + 2^-22) − 1) + 2^-126` with a gate, `0` without; `wUp = wt + wErr`;
* `εA = (BK + 8 + 2·nK)·2^-22`;
* `eA = σ·kqMag·(εA·wUp + wErr) + wUp·(σ·2^-110 + 2^-140) + 2^-140`;
* `B1 = Σ_j |Aex|·|do_j|`, `Be = Σ_j eA·|do_j|`, `Bl = Σ_j (|Aex| + eA)·liftMag do_j`,
  `Dl = Σ_j liftMag do_j`;
* `tol = 2^-8·|ref| + (1 + 2^-8)·(2^-8·B1 + (1 + 2^-8)·Be + (BT + 3)·2^-22·Bl) + 2^-100·(1 + Dl)`.

The two `2^-8` terms are the kernel's two `bfloat16` roundings, of each entry of `A` (line 140) and
of the stored result (line 149); `Be` (the `float32` errors of the entries, about `3·10^-5` of
`σ·kqMag` at `K = 128`) and the `Bl` term (the second `tl.dot`) are two to three orders smaller.
On the host's `randn` inputs `tol` is about `(2^-8)·(|ref| + B1)`; the worst error on the H200 is
`0.973` of it.

## Conditions

`Inst`: `varlen`, `useG`, `B`, `T`, `H`, `HV`, `K`, `V`, `BT`, `BK`, `BV`, `scale` (double), the
storages `q`, `k`, `dO` (the source's `do`), `dv`, `g`, and for variable length the tables' dtype,
`N`, `cu_seqlens` (`cuVal`), `NT`, `chunk_indices` (`ciVal`).

`Inst.ok`:
* `BT ∈ {16, 32, 64}`; `BK`, `BV` powers of two in `[16, 128]`;
* `1 ≤ K ≤ 256`, `1 ≤ V ≤ 2^16`, `HV·V < 2^31` (every constexpr of the kernel an `int32`
  constant, see Exclusions); `H ≥ 1` divides `HV ≥ 1`; `B ≥ 1`, and `B = 1` on a variable-length
  launch;
* `T < 2^31` (the `int32` argument);
* the grid: `NT < 2^31`, **`B·HV < 2^16`** (a limit of the kernel, see Exclusions), fewer than
  `2^31` programs;
* `scale` a finite double in `[2^-32, 2^32]`;
* the buffers hold the contiguous tensors; `dv` (a fresh `empty_like(do)`) shares no byte with
  `q`, `k`, `do`, `g` and the tables;
* variable length: `int32` or `int64` tables, `N < 2^31`, the tables' lengths.

`Pre`: the buffers valid; on a variable-length launch the tables hold the instance's contents and
are `TablesOk` (`cu_seqlens` nonnegative, nondecreasing and at most `T`; each row of
`chunk_indices` a distinct (sequence, chunk) whose chunk starts inside its sequence; every such
chunk has a row: what `prepare_chunk_indices` builds, in any row order); `ValsOk`: in every
sequence, every `q`, `k`, `do` component finite, `|do| ≤ 2^100`, every `g` finite (with a gate);
every pair `p ≤ j` of one chunk with `g[j] − g[p] ≤ 64` (with a gate), `kqMag ≤ 2^100` and
`σ·kqMag·(1 + wUp) ≤ 2^100`; every output with `Σ_j σ·kqMag·wUp·liftMag do ≤ 2^120`.

The launch: grid `(NT, B·HV, 1)`, the source's block shapes, `fusion = true`, `debug = false`,
`num_ctas = 1`, no `launch_pdl`; `num_warps=4`, `num_stages=3` only, no heuristics or autotuning
(removed from the source), a `JITFunction` launch, no descriptors: within
part (A) of `Compiler.Faithful`'s scope. The claim relies on part (B) (the environment) as the rule
states it.

## Assumptions

`[rule]` fields (`docs/rules/num.md`'s table): `faithful` (`Compiler.Faithful`), `dot`
(`DotBF16`), `dot_total` (`DotTotal P cfgBF16`), `exp2` (`Exp2F32`), `exp2_ninf`
(`Exp2F32NegInf`), `exp2_total` (`MathTotal P .exp2 .binary32`). `consistent` uses
`Platform.sm90ref`.

## Model notes

`b_A += tl.dot(b_k, b_q) * scale` is `tl.dot` with a zero accumulator then `tl.mul_add` (the
product by `scale` feeds the addition; the dot itself feeds a `*`, so it is not the combine pass's
`dot_add`); `b_g[None, :] - b_g[:, None]` is an IEEE `tl.sub` (no product feeds it) and the gate's
`*` an IEEE `tl.mul`; every element offset is `int64` (`i_t`, `i_bh` converted to `int64` at line 99, `bos`,
`o_t`), except the `int32` `o_k` (below `K + BK ≤ 2^9`), `o_v` (below `V + BV ≤ 2^16 + 2^7`) and `i_n`; the varlen branch's two loads at
line 102 both read the program's own `i_t`.

## Coverage

`canon vl G B T H HV K V BT BK BV sc dt N NT` is the host's launch for those sizes (contiguous
tensors, `dv = empty_like(do)`); `canon_ok` proves `Inst.ok` for every size within the bounds
above with tensors of at most `2^38` elements. The `setup_*_ok` theorems, each for both values of
the gate flag `G`:
* the host's test cases (source lines 278–290): `setup_t63`, `setup_t500`, `setup_t1000`,
  `setup_t1024`, `setup_bt16`, `setup_bt32`, `setup_gdn`, `setup_qwen` (fixed length) and
  `setup_var1`, `setup_var3`, `setup_var5`, `setup_var3g`, `setup_var6` (variable length, `int64`
  tables, `NT` as `prepare_chunk_indices` gives it: 1, 16, 35, 16, 68);
* the benchmark shapes `setup_b1` … `setup_b6`;
* gated DeltaNet's default layer, `setup_gdn_default`;
* the `2^16`-token packed batch, `setup_train` (`int32` and `int64` tables, 1027 chunks);
* the largest grid `y`, `setup_rows` (`B·HV = 65535`).

`setup_var3` carries its real tables (`cu_seqlens = [0, 256, 500, 1000]` and the 16 rows
`prepare_chunk_indices` builds), and `setup_var3_tables` proves them `TablesOk`. The other
instances leave the tables' contents open; `TablesOk` and `ValsOk` are about contents, so
`check_gpu.py` checks them on the host's inputs (they hold on every one).

## Tolerances

From the library's facts (`docs/rules/num.md`): `DotBF16` (`relDot16 K = (K+2)·2^-22`,
`absDot16 K = (K+2)·2^-126`, relative to `|acc| + Σ liftMag a·liftMag b`), `Exp2F32` (`2^-22`
relative, `2^-126` absolute), `Exp2F32NegInf`, and the IEEE rounding lemmas (`float32`: `u = 2^-24`,
`η = 2^-150`; `bfloat16`: `2^-8`, `2^-134`). Fix a valid pair `p ≤ j` of one chunk, value head
`hv`, `L = kqMag(p, j)`, `S = kq(p, j)`.

1. **The `K` loop** (lines 128–136, `nK ≤ 16` trips, each a fresh `tl.dot` of `BK` columns then
   `tl.mul_add` with `σ₃₂ = hostF32(σ)`, `|σ₃₂ − σ| ≤ u·σ` since `σ ∈ [2^-32, 2^32]`). Trip `t`'s
   dot is within `(BK+2)·2^-22·L_t + (BK+2)·2^-126` of its exact sum (`L_t` its columns' share of
   `L`; masked `K` tails load `+0`, of `liftMag 0`). The multiply-add, fused or not, adds at most
   `u·|D_t·σ₃₂| + u·|partial sum| + 2η`; summed over the trips, the partial sums are at most
   `σ₃₂·L·(1 + ε)`, so the computed `P` has
   `|P − σ₃₂·S| ≤ σ₃₂·L·((BK+2)·2^-22 + (nK+1)·u)·(1 + 2^-20) + nK·(σ₃₂·(BK+2)·2^-126 + 2^-148)`.
   Room: `L ≤ 2^100` and `σ·L ≤ 2^100` (`ValsOk`).
2. **The gate** (line 138). The IEEE `fl(g[j] − g[p])` is within `u·|dg| + 2^-149` of `dg` (an
   overflow to `−∞` gives `exp2 = +0`, `Exp2F32NegInf`, and then `2^{dg} < 2^-126`); `Exp2F32` on
   it: the computed `G` has `|G − 2^{dg}| ≤ 2^{dg}·(2^{u·|dg| + 2^-149}·(1 + 2^-22) − 1) + 2^-126 =
   wErr` and `G ≤ wUp`; finite since `dg ≤ 64`. Without a gate `G` is absent (`wt = 1`,
   `wErr = 0`).
3. **The entry before rounding.** `Ā = fl(P·G)` (or `P`): `|Ā − σ·S·wt| ≤ |P − σ₃₂S|·G +
   |σ₃₂ − σ|·|S|·G + σ·|S|·|G − wt| + u·|P·G| + η`. The coefficient of `σ·L·wUp` is at most
   `(BK + 2)·2^-22 + (nK + 3)·u + 2^-34 ≤ εA`: the cross terms (`εK·u`, `(1 + 2^-20)` factors) are
   at most `130·2^-42 < 2^-34` at `BK = 128`, and `εA`'s `(6 + 2nK)·2^-22` beyond `(BK+2)·2^-22`
   is more than six times `(nK + 3)·u + 2^-34`, the `σ·|S|·wErr` term is at most `σ·L·wErr`, and the absolute terms
   at most `wUp·(σ·2^-115 + 2^-144) + 2^-149`: in all `|Ā − Aex| ≤ eA`. Room: `σ·L·wUp ≤ 2^100`.
4. **Line 140.** Masked lanes (`p > j`, or a token past the sequence) are `+0` exactly (`tl.where`
   discards any `NaN` or `∞` there). A valid entry rounds to `bfloat16`:
   `|Â − Ā| ≤ 2^-8·|Ā| + 2^-134`, so `|Â − Aex| ≤ 2^-8·|Aex| + (1 + 2^-8)·eA + 2^-134`.
5. **Line 148** (`tl.dot`, `K = BT`, zero accumulator): within `(BT+2)·2^-22·Σ_j liftMag Â·liftMag do
   + (BT+2)·2^-126` of `Σ_j Â·do`; `liftMag Â ≤ |Â| + 2^-126 ≤ (1 + 2^-8)·(|Aex| + eA) + 2^-125`,
   and `(BT + 2)·(1 + 2^-8) ≤ BT + 3`. Room: `Σ_j liftMag Â·liftMag do ≤ 3·2^120 + 2^94 < 2^127`
   (`ValsOk`'s per-output bound, `|do| ≤ 2^100`, `eA ≤ (1 + εA)·σ·L·wUp + 2^-13`).
6. **Assembly.** With `x` the `float32` result,
   `|x − ref| ≤ E = 2^-8·B1 + (1 + 2^-8)·Be + (BT+3)·2^-22·Bl + abs`, `abs ≤ 2^-134·Dl +
   (BT+2)·2^-147·Dl + (BT+2)·2^-126`. The store: `|c − x| ≤ 2^-8·|x| + 2^-134` and
   `|x| ≤ |ref| + E`, so `|c − ref| ≤ 2^-8·|ref| + (1 + 2^-8)·E + 2^-134 ≤ tol`
   (`(1 + 2^-8)·abs + 2^-134 ≤ 2^-100·(1 + Dl)`); `c` is finite (`|x| < 2^126`).

The bound is provable from `DotBF16`, `Exp2F32`, `Exp2F32NegInf` and the library's IEEE rounding
lemmas (`FloatV.mul_err`, `add_err`, `sub_err`, `cvt_err`, `AddRel.err`; `tl.mul_add`'s two
alternatives, the fused `FloatV.fma` or the two IEEE operations). It is meaningful: its leading terms are the two `bfloat16` roundings themselves, so
on random inputs it is about `1.03×` the kernel's worst error (`check_gpu.py`: worst
error / tolerance `0.973`), and a single wrong element, a missing gate or a wrong chunk size is
rejected.

## Exclusions

* **`B·HV ≥ 65536` on fixed-length batches (a limit of the kernel).** The wrapper puts `B·HV` on the
  grid's `y` axis, which CUDA limits to `65535`; the first failing launch has `B·HV = 65536`
  (`check_gpu.py`: `B = 4096`, `HV = 16`, `Triton Error [CUDA]: invalid argument`; also, for
  instance, 2048 rows at 32 value heads). `Inst.ok.grid1_lt` keeps `B·HV < 2^16`. SETUP.md's model
  and benchmark launches have at most `4·64 = 256` (its edge launch has `65535`); variable-length
  batches (`B = 1`) never reach it.
* **More than `2^31` programs or chunks** (`NT·B·HV ≥ 2^31`): CUDA's and the launcher's limit
  (`Launch.Valid`); real use is far below (the `2^16`-token batch has `1027·32` programs).
* **`int32` offsets: none.** Every element offset is `int64` (lines 99, 106, 108–113, 120–121,
  131–132, 145–146), so the kernel addresses tensors of more than `2^31` elements correctly; only
  `o_k`, `o_v` (below `2^9`) and the sequence index `i_n` (`N < 2^31`) are `int32`. `check_gpu.py`
  does not run a tensor of `2^31` elements (4 GiB in `bfloat16`, above the check's memory budget);
  `T ≥ 2^31` makes `T` an `int64` argument, another specialisation: outside real use.
* **Gate values.** With a gate rising by more than about 128 within a chunk the kernel's
  `exp2(g[j] − g[p])` overflows to `∞` and a valid entry becomes `∞` or `NaN` (`0·∞`): a limit of
  the kernel at those values; `ValsOk` keeps `g[j] − g[p] ≤ 64`. Real use never reaches it: fla's
  `g` is the chunk-local cumulative sum of nonpositive log decays, nonincreasing within a chunk up
  to `tl.cumsum`'s `float32` rounding (any rise is about `u·|g|`, far below 64).
* **Input magnitudes.** Non-finite inputs in a sequence, `|do| > 2^100`, `kqMag > 2^100`, and
  products beyond the room bounds (`2^100`, `2^120`): `float32` overflows near `2^128` (a limit of
  the kernel at those magnitudes); real gradients and normalised queries and keys are far below.
  Outside real use.
* **`scale` outside `[2^-32, 2^32]`**, zero or negative: outside real use (the default is
  `K^-0.5 ∈ [1/16, 1]`). An `int` `scale` (`1`, as some of fla's tests pass it) is a compile-time
  constant, another specialisation: not part of the setup.
* **`V > 2^16` or `HV·V ≥ 2^31`**: outside real use (SETUP.md: `V ≤ 1000`, `HV·V ≤ 12288`). At
  `V ≥ 2^31` the kernel is not correct (a limit of the kernel): Triton builds the constexpr `V` as a
  `uint32` constant, so line 144's `o_v < V` is an unsigned comparison the model's signed one does
  not match (at `V = 2^31` every lane is masked off and nothing is stored), and beyond `2^31` the
  `int32` `i_v * BV` (line 143) wraps.
* **`K > 256`**: outside real use (the chunked forward asserts `K ≤ 256`). Chunk sizes other than
  16, 32, 64: not part of the setup.
* **`USE_G_GAMMA`, `USE_A`** (`g_gamma` or `A` given), `float16`/`float32` inputs: not part of
  SETUP.md's launches (no caller in fla v0.6.0 passes `g_gamma` or `A` to this host).
* **Tables other than `TablesOk`** (overlapping or out-of-range rows would make two programs write
  one element, or a program read outside its sequence), `dv` overlapping the inputs: outside the
  host's use.

## Library requests

None.

## Checks

* **Builds**: `Claim.lean` builds with no errors or warnings, no `sorry`, no `axiom`.
* **Not vacuous**: `consistent` (`Platform.sm90ref`); `satisfiable` (one row of one token, one
  head, head dims 16, chunk size 16, no gate, every byte `0`).
* **Covers the setup**: `canon_ok` and the `setup_*_ok` above (each for both gate flags; the
  training batch for both table dtypes); `setup_var3_tables` (`TablesOk` on real tables).
* **GPU** (`check_gpu.py`, H200, Triton 3.8.0, peak device memory 2.17 GiB): 46 checks pass: every
  SETUP.md launch without and with a gate, `Inst.ok` and `Pre` edges (SETUP.md, "Checked on the
  H200"); worst error / tolerance `0.973` (margin `1.03×`); every written element finite, `ValsOk`
  and `TablesOk` on every input, the inputs unchanged, every element of `dv` outside the sequences
  unchanged; four negative tests rejected; `B·HV = 65536` does not launch.

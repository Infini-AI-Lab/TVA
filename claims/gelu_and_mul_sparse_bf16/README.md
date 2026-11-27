# `gelu_and_mul_sparse_bf16`

**The kernel.** vLLM's `_gelu_and_mul_sparse_kernel` (`source/activation.py`, kernel lines 56–92;
host `_gelu_and_mul_sparse_triton_op` lines 113–134), the activation of Gemma 3n's sparse MLP
layers (`GeluAndMulSparse`), on `bfloat16` inputs. A 1-D grid of `n_rows` programs; program `t`
handles row `t` of `x = [gate | up]` (`2·d` elements) in one block of `BLOCK_SIZE =
next_power_of_2(d)` lanes: it computes the mean and the biased standard deviation of `gate` (two
`tl.sum`s and `tl.sqrt`), the cutoff `mean + std·std_multiplier`, rounding the mean, the standard
deviation, the scaled standard deviation and the cutoff to `bfloat16` as PyTorch's native
implementation does, the sparse gate `max(gate − cutoff, 0)` (rounded to `bfloat16`), its
tanh-approximate GELU through `2·sigmoid(2·inner) − 1` (rounded to `bfloat16`), and stores its
product with `up` into a fresh `[n_rows, d]` output. The claim is `Claims.GeluAndMulSparse.claim`
in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race). For every
row `t < n_rows` and column `j < d`, with `g j = gate[t, j]` and `u j = up[t, j]` read from the
initial memory, `M`, `A`, `Sd` the row's exact mean `Σ g/d`, mean magnitude `Σ |g|/d` and
standard deviation `sqrt(Σ (g − M)²/d)`, `m` the host's double and the exact cutoff
`cRef = M + m·Sd`:
- `output[t, j]` is a finite `bfloat16` within
  `yTol j = 2^-5·|yRef j| + 2·|u j|·cutTol + 2^-132` of
  `yRef j = geluRef (max (g j − cRef) 0) · u j`, where
  `geluRef x = x/2·(1 + tanh(0.7978845608028654·(x + 0.044715·x³)))` (the source's constants) and
  `cutTol = 2^-6·(|M| + |m|·Sd) + 2^-17·(1 + |m|)·A + 2^-130`;
- where `g j ≤ cRef − cutTol`, `output[t, j]` is a zero (`+0` or `−0`);
- every byte outside the `n_rows·d` elements of `output` is unchanged.

Near the cutoff the kernel's comparison `sparse_gate < 0` follows an inexact mean and standard
deviation, so a gate within `cutTol` of `cRef` may give `0` or a small positive value: the first
item covers both (its `2·|u|·cutTol` term), the second says that a gate clearly below the cutoff
gives exactly `0`.

**The conditions** (`Inst.ok`, `Pre`):
- `1 ≤ d ≤ 32768` (the host's `_MAX_INTERMEDIATE_SIZE`), `BLOCK_SIZE` its next power of two;
- `1 ≤ n_rows < 2^31` (the host launches nothing for an empty output; CUDA's grid limit);
- the multiplier a finite double with `|m| ≤ 64`;
- `x` holds `n_rows·2d` elements and `output` `n_rows·d`, both valid in the initial memory, the
  output sharing no byte with `x`;
- every element of `x` a finite `bfloat16` with `|v| ≤ 2^32`;
- every row's gate not nearly constant: `2^-18·A + 2^-62 ≤ Sd` (`RowOk`). Every row that is not
  constant and whose largest `|g|` is at least `2^-44` meets it: two distinct `bfloat16` values
  differ by at least `2^-9` of the larger magnitude, so such a row has `Sd ≥ 2^-17·max|g|` for
  `d ≤ 2^15`.

**The launches covered** (`SETUP.md`, `setup_*_ok`): Gemma 3n's sparse MLP shards with the model's
multiplier `1.6448535919189453`: E4B `d` = 16384 (one token and 8192 tokens), 8192, 4096, 2048
(8192 tokens), E2B `d` = 8192 (one token and 8192 tokens) and 1024 (8192 tokens); the source's
test shapes `(1, 8192)`, `(7, 16384)`, `(33, 4096)`, `(5, 100)`; and, with vLLM's input
generator's multiplier `1.6448536269514722`, 8192 × 16384 and the shapes of vLLM's tests
`(64, 13824)`, `(64, 32768)`, `(32, 8192)`, `(5, 31)`, `(5, 128)`: eighteen in all. `mkInst_ok`
covers the host's tensors for every `1 ≤ d ≤ 32768`, `1 ≤ n_rows < 2^31` with `n_rows·d ≤ 2^37`,
and every finite multiplier with `|m| ≤ 64`, which includes the rest of vLLM's tests' `bfloat16`
shapes.

**Device facts** the tolerances rest on (all in `Device P`): `Device.SumDepth` (the tree of a
float `tl.sum` over `BLOCK_SIZE` positions has depth at most `⌈log₂ BLOCK_SIZE⌉`),
`Device.DivFull` (`/` on `float32`: relative `2^-22`, absolute `2^-148`, quotients up to
`2^126`), `Device.SqrtF32` (`tl.sqrt` on `float32`, on `0` and inputs `≥ 2^-126`:
`2^-22·√x + 2^-149`), `Device.ExpF32` (`tl.exp` on `float32` on `[-X, X]`:
`(2^-22 + 2^-23·X)·e^x + 2^-140`, with room), and the totality facts for `/`, `sqrt` and `exp`.

## Tolerances

Units: `u = 2^-24` (one `float32` rounding: `u·|q| + 2^-150`), `v = 2^-8` (one `bfloat16`
rounding: `v·|q| + 2^-134`); the widenings `bfloat16 → float32` are exact. `L = log₂ BLOCK_SIZE
≤ 15`. Write `c_k` for the kernel's cutoff (a `bfloat16` value) and `s_k` for its sparse gate.

1. **Mean** (line 75): the IEEE sum along a tree of depth `L` (`Device.SumDepth`) is within
   `((1 + u)^L − 1)·Σ|g| + 2^-150·(2^L − 1)·(1 + u)^L` of `Σ g`, and `/ d` (the exact `float32`
   constant `d`) adds `DivFull`: `|mean32 − M| ≤ (L + 5)·u·A + 2^-134`.
2. **Standard deviation** (lines 76–77, 80): each lane's fusable subtraction is within
   `3u·(|g j| + |mean32|) + 2^-148` of `g j − mean32`; the squares and their fusable sum along a
   tree of depth `L` add `(1 + 3u)^L − 1 ≤ 45.01u` relative (the squares are nonnegative), `/ d`
   and `tl.sqrt` add `DivFull` and `SqrtF32`. Since `Σ (g − mean32)² = Σ (g − M)² +
   d·(M − mean32)²`, the triangle inequality in `ℓ²` bounds the computed norm against `√d·Sd` by
   `√d·|M − mean32|` plus the norm of the lane errors, `≤ 3u·√d·(rms g + |mean32|)` with
   `rms g ≤ A + Sd`. So `std32` is within `(L + 12)·u·A + 40·u·Sd` of `Sd`, an absolute bound:
   the kernel takes a square root, not its inverse, and needs no condition number. `RowOk` keeps
   the computed variance at least `2^-126` (in `SqrtF32`'s domain): the lane errors are far below
   `Sd`.
3. **Cutoff** (lines 79–82): four `bfloat16` roundings, of `mean32`, of `std32`, of the product
   `std·m32` and of the IEEE sum `mean + scaled_std`; `m32 = float32(m)` is within `u·|m|`. To
   first order `|c_k − cRef| ≤ 2^-7·|M| + 3·2^-8·|m|·Sd`, and the `float32` terms of steps 1–2,
   carried through (also multiplied by `|m| ≤ 64`), stay within `2^-17·(1 + |m|)·A` and the slack
   of `2^-6` against `2^-7` and `3·2^-8`; the absolute terms within `2^-131`. So
   `|c_k − cRef| ≤ 7/8·cutTol − 2^-131`.
4. **Sparse gate** (lines 83–84): `s_k` is `bfloat16(float32(g − c_k))`, or `0` where that is
   negative; it is within `(2^-8 + 2^-20)·max(g − c_k, 0) + 2^-131` of `max(g − c_k, 0)`, which
   is within `|c_k − cRef|` of `sRef = max(g − cRef, 0)` (`max(·, 0)` is 1-Lipschitz).
5. **GELU in `float32`** (lines 86–90): with `1 + tanh z = 2·σ(2z)` (`σ(x) = 1/(1 + e^-x)`),
   `geluRef s = s·σ(X)`, `X = 2·0.7978845608028654·(s + 0.044715·s³)`. The kernel's constants are
   within `u` relative of the decimals; the IEEE products, `tl.mul_add` and the product by the
   constant give the argument of `tl.sigmoid` within `2^-20·X + 2^-120` of `X`; `tl.exp`
   (`ExpF32` at `X = |x|`, with room since `−X_k ≤ 1`), the IEEE `1 + e` and `DivFull` give the
   logistic within `2^-18` of `σ(X)` (`σ` is `1/4`-Lipschitz, and `X·σ'(X)` is bounded); the
   fusable `2σ − 1`, the IEEE `1 + tanh` and the product by `0.5·s` give `act` within
   `2^-15·geluRef s_k + 2^-145` of `geluRef s_k` (`σ ≥ 1/2`). Then the `bfloat16` rounding of
   `act`, the `float32` product with `u`, and the `bfloat16` rounding of the store: within
   `(2^-7 + 2^-12)·|geluRef(s_k)·u| + 2^-133·(1 + |u|)` of `geluRef(s_k)·u`.
6. **Back to the reference.** `geluRef` is 2-Lipschitz on `[0, ∞)`: its derivative is
   `σ(X) + s·X'·σ(X)(1 − σ(X))` with `s·X' ≤ 3X` and `3X·σ(X)(1 − σ(X)) ≤ 3X/(e^X + 2) ≤ 1`
   (`e^X ≥ 1 + X + X²/2`), and it is nonnegative. Also `geluRef s ≥ s/2`. With step 4,
   `|geluRef s_k − geluRef sRef| ≤ 2·((2^-8 + 2^-20)·(sRef + |c_k − cRef|) + |c_k − cRef| +
   2^-131)`, and `2·(2^-8 + 2^-20)·sRef ≤ 4·(2^-8 + 2^-20)·geluRef sRef`. Collecting with step 5:
   relative `2^-7 + 2^-12 + 4.1·2^-8 ≈ 0.024 ≤ 2^-5`, the cutoff term
   `2·1.004·|u|·|c_k − cRef| ≤ 2·|u|·(7/8·cutTol)`, absolute `2^-133 ≤ 2^-132`, and the terms in
   `|u|` (`2^-130·|u|` and `2^-133·|u|`) within the rest of `2·|u|·cutTol` (`cutTol ≥ 2^-130`).
   This is `yTol`.
7. **The zero clause.** If `g ≤ cRef − cutTol`, then `g − c_k ≤ −cutTol/8 − 2^-131` (step 3);
   the `float32` difference and its `bfloat16` rounding stay negative (their errors are far below
   `2^-131`), the `tl.where` gives `+0`, every product of `+0` with a finite value is a zero, and
   so is the stored value.

On the H200 (`check_gpu.py`) the worst error is `0.342` of `yTol` (`0.26` at Gemma 3n's prefill
shapes). At Gemma 3n's E4B shape with an `N(0, 1)` gate, `cutTol` is about `2.6·10^-2·Sd` and,
on elements with `sRef ≥ 1`, `yTol` is about `7.9%` of `|yRef|` (median): the tolerance is set
by the four `bfloat16` roundings of the cutoff, which move every surviving output by up to
`|u|·|c_k − cRef|`. It does not tell the tanh GELU from the exact (erf) GELU, or from the tanh
GELU without its cube term (both within `0.30` of `yTol` there).

## Exclusions

- **Nearly constant rows** (`RowOk` fails; constant rows, an all-zero row): a library gap. The
  kernel's variance may then be a subnormal `float32` (the library's `DivFull` allows `0 / d` to
  be one, and a constant row's mean need not be exact), where the library's `tl.sqrt` fact says
  nothing (`tl.sqrt` flushes a subnormal input; on the H200 such rows give `std = 0`).
- **`|x| > 2^32`**: outside real use (bfloat16 MLP activations are far smaller); from about
  `2^43` the kernel's `float32` cube `0.044715·s³` overflows (its result then is still the GELU,
  `s`, but the claim's proof does not follow infinities through `tl.sigmoid`).
- **`|m| > 64`**: outside real use (`Normal(0, 1).icdf` of a `float32` sparsity in `(0, 1)` lies
  within `±40`).
- **Non-finite inputs**: there is no reference value.
- **`n_rows ≥ 2^31`**: CUDA's grid limit, a kernel limit.
- **`float16` and `float32` inputs**: other compiled kernels (the pointer's dtype is part of the
  specialisation), not part of `SETUP.md`.

`check_gpu.py` (one H200) checks `Post` on the setup's launches, vLLM's tests' `bfloat16` shapes,
and edge launches.

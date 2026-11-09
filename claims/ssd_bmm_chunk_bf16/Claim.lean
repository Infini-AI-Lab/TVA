import TVA

/-!
# `ssd_bmm_chunk_bf16`: the prepared claim

Source: `source/ssd_bmm.py`. The kernel is `_bmm_chunk_fwd_kernel` (lines 80–190); it reads the
module-level names `triton` and `tl` (lines 57–58). The host `bmm_chunk_fwd` (lines 193–282)
launches it as Mamba-2's chunked scan does (`CB = _bmm_chunk_fwd(C, B, chunk_size,
seq_idx=seq_idx, output_dtype=torch.float32)`): `a = C`, `b = B`, both `[1, seqlen, ngroups,
dstate]` `bfloat16`; `seq_idx` `[1, seqlen]` `int32`; a fresh `float32` `out` `[1, nchunks,
ngroups, chunk_size, chunk_size]`; `IS_CAUSAL = False`, `HAS_SEQ_IDX = True`,
`dot_dtype = tl.bfloat16`; `BLOCK_SIZE_M = 64`, `BLOCK_SIZE_N = 128`, `BLOCK_SIZE_K = 32`,
`num_stages = 4`, `num_warps = 4` (`SETUP.md`); the grid `(cdiv(chunk_size, 64) ·
cdiv(chunk_size, 128), 1, nchunks · ngroups)`.

* The model: `bmm_chunk_fwd_kernel`, `model`, `launch`.
* The conditions: `Inst`, `Inst.ok`, `Pre`.
* The specification: `Post`, `Spec`: every element of `out`, either exactly `+0.0` (a pair of
  positions of different requests, or past the end of the sequence) or the exact dot product of
  the two tokens' rows within the tolerance `tol`, and nothing else written.
* The assumptions: `Assumptions`.
* The checks: `consistent`, `satisfiable`, `tol_rel_lt`, and the covered launches `setup_*_ok`.

The claim to prove is `Statement`.
-/

namespace Claims.SsdBmm

open TVA

/-! ## The model -/

/-- The `tl.dot` configuration of line 161: `bfloat16` operands (`dot_dtype = tl.bfloat16`), the
default `float32` accumulator, the default `input_precision` (`tf32`; no effect on 16-bit
operands) and the default `max_num_imprecise_acc` (`0`: not an fp8 dot). -/
def cfgBF16 : DotCfg :=
  { a := .bf16, b := .bf16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }

/-- The `float32` `+0.0`: `other=0.0` (lines 153, 159, before its conversion to `bfloat16`) and
the `tl.where` value `0.0` beside the `float32` `acc` (line 179). -/
def zf32 : F32 := FloatV.litF32 (FloatV.ofPat 0)

/-- An `int32` scalar sign-extended to `int64` (`tl.convert`, exact): the promotion of an `int32`
operand of an operator whose other operand is `int64`. -/
def wide (x : I32) : Prog I64 := do
  return (← tl.convert .i32 .i64 (Tensor.scalar x)).item

/-- A runtime integer argument as Triton types it by its value (SEMANTICS.md, "Scalar
arguments"): an `int32` scalar for a value in `[-2^31, 2^31)`, an `int64` one for a value in
`[2^31, 2^63)`. The three batch strides `stride_a_batch`, `stride_b_batch`, `stride_out_batch`
are of either class on covered launches (`Inst.ok.aB_lt`, `bB_lt`, `oC_lt`). -/
abbrev IntArg := I32 ⊕ I64

/-- The class of a non-negative argument `n < 2^63`: `int32` below `2^31`, `int64` from `2^31`. -/
def intArg (n : Nat) : IntArg := if n < 2 ^ 31 then .inl (IntV.ofNat n) else .inr (IntV.ofNat n)

/-- `pid_b * s` (lines 126, 131, 183) for the `int32` `pid_b` and a batch stride `s`, as the `int64`
operand of the `+` with the `int64` terms that follow it: for an `int32` `s` the wrapping `int32`
product, sign-extended by the `+`; for an `int64` `s`, `pid_b` sign-extended (the promotion to
the other operand's `int64`), then the wrapping `int64` product. -/
def batchOff (pid_b : I32) : IntArg → Prog I64
  | .inl s => wide (pid_b * s)
  | .inr s => do return (← wide pid_b) * s

/-- `_bmm_chunk_fwd_kernel` (source lines 80–190) on the launches the host makes. `a_ptr`,
`b_ptr` are `*bfloat16`, `out_ptr` `*float32`, `seq_idx_ptr` `*int32`. `seqlen`, `chunk_size`,
`K`, `ngroups` and twelve of the fifteen strides are `int32` scalar arguments (`Inst.ok` keeps
each below `2^31`); the three batch strides `stride_a_batch`, `stride_b_batch`, `stride_out_batch`
are `int32` or `int64` arguments by their values (`IntArg`; the kernel only multiplies each by
`pid_b`, `batchOff`). An integer argument equal to `1` is the compile-time constant `1`, and each of its uses in
the kernel, an operand of `+`, `-`, `*`, `//` with an `int32` or `int64` tensor, a comparison, a
`min`, the bound of `tl.cdiv` whose result feeds `range`, builds the constant in the computation's
dtype with the same value, or gives the Python value the `int32` computation gives: see `model`.
`BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K` (`tl.constexpr`) are Lean parameters.

`IS_CAUSAL = False`, `HAS_SEQ_IDX = True` and `dot_dtype = tl.bfloat16` on every covered launch
(`SETUP.md`): the compile-time `if IS_CAUSAL:` (line 122) visits nothing, the two
`if HAS_SEQ_IDX:` (lines 135, 167) visit their bodies, and `.to(dot_dtype)` is a conversion to
`bfloat16` (a dtype `tl.constexpr` holds the dtype and means at each use what the dtype written in
its place means: SEMANTICS.md, "A dtype as a `tl.constexpr` argument"); the model has no
parameters for them (the model map drops line 122–124 as `config`). -/
def bmm_chunk_fwd_kernel (a_ptr b_ptr : Ptr .bf16) (out_ptr : Ptr .f32) (seq_idx_ptr : Ptr .i32)
    (seqlen chunk_size K ngroups : I32)
    (stride_a_batch : IntArg) (stride_a_seqlen stride_a_head stride_ak : I32)
    (stride_b_batch : IntArg) (stride_b_seqlen stride_b_head stride_bk : I32)
    (stride_out_batch : IntArg) (stride_out_chunk stride_out_head stride_outm stride_outn : I32)
    (stride_seq_idx_batch stride_seq_idx_seqlen : I32)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) : Kernel where
  body ctx := do
    -- line 115: axis 1 of the grid, an `int32`
    let pid_b := tl.program_id ctx 1
    -- line 116: axis 2 of the grid, sign-extended to `int64` (`.to(tl.int64)`, exact)
    let pid_ch : I64 := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id ctx 2))).item
    -- line 117: `int64 // int32`: `ngroups` sign-extended to `int64`, then the truncating `//`
    -- (`tl.idiv`, undefined for a zero divisor)
    let pid_c : I64 := (← tl.idiv (Tensor.scalar pid_ch) (Tensor.scalar (← wide ngroups))).item
    -- line 118: `ngroups` sign-extended again; the wrapping `int64` product and difference
    let pid_h : I64 := pid_ch - pid_c * (← wide ngroups)
    -- line 119: `tl.cdiv(chunk_size, BLOCK_SIZE_N)` is `(chunk_size + (BLOCK_SIZE_N - 1)) //
    -- BLOCK_SIZE_N`: the constexprs computed by Python, built in `int32`; the wrapping `int32`
    -- `+`, then `tl.idiv`
    let num_pid_n : I32 := (← tl.idiv
      (Tensor.scalar (chunk_size + (IntV.ofNat (BLOCK_SIZE_N - 1) : I32)))
      (Tensor.scalar (IntV.ofNat BLOCK_SIZE_N : I32))).item
    -- line 120: axis 0 of the grid, `int32` `//`
    let pid_m : I32 := (← tl.idiv (Tensor.scalar (tl.program_id ctx 0))
      (Tensor.scalar num_pid_n)).item
    -- line 121: axis 0 again, `int32` `%` (`tl.irem`)
    let pid_n : I32 := (← tl.irem (Tensor.scalar (tl.program_id ctx 0))
      (Tensor.scalar num_pid_n)).item
    -- lines 122–124: `if IS_CAUSAL:` is a compile-time condition, `False`: not visited
    -- lines 125–129: `a_ptr += …`: `pid_b * stride_a_batch` is an `int32` product (wrapping)
    -- sign-extended for the first `+`, or an `int64` one for an `int64` stride (`batchOff`);
    -- `pid_c * chunk_size * stride_a_seqlen` and `pid_h * stride_a_head` are `int64` (the
    -- `int32` operands sign-extended); the pointer `+`
    let a_ptr := a_ptr + ((← batchOff pid_b stride_a_batch) +
      pid_c * (← wide chunk_size) * (← wide stride_a_seqlen) + pid_h * (← wide stride_a_head))
    -- lines 130–134: the same for `b_ptr`
    let b_ptr := b_ptr + ((← batchOff pid_b stride_b_batch) +
      pid_c * (← wide chunk_size) * (← wide stride_b_seqlen) + pid_h * (← wide stride_b_head))
    -- lines 135–138: `if HAS_SEQ_IDX:` taken; as lines 125–129
    let seq_idx_ptr := seq_idx_ptr + ((← wide (pid_b * stride_seq_idx_batch)) +
      pid_c * (← wide chunk_size) * (← wide stride_seq_idx_seqlen))
    -- line 140: `int32` products and sums (wrapping)
    let offs_m : Tensor [BLOCK_SIZE_M] I32 :=
      pid_m * (IntV.ofNat BLOCK_SIZE_M : I32) + tl.arange 0 BLOCK_SIZE_M
    -- line 141
    let offs_n : Tensor [BLOCK_SIZE_N] I32 :=
      pid_n * (IntV.ofNat BLOCK_SIZE_N : I32) + tl.arange 0 BLOCK_SIZE_N
    -- line 142
    let offs_k : Tensor [BLOCK_SIZE_K] I32 := tl.arange 0 BLOCK_SIZE_K
    -- line 143: `int32` products and sum (wrapping), broadcast to `[BLOCK_SIZE_M, BLOCK_SIZE_K]`;
    -- the pointer `+` on each lane
    let a_ptrs : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_K] (Ptr .bf16) :=
      a_ptr + (offs_m.col BLOCK_SIZE_K * stride_a_seqlen + offs_k.row BLOCK_SIZE_M * stride_ak)
    -- line 144: the same, `[BLOCK_SIZE_K, BLOCK_SIZE_N]`
    let b_ptrs : Tensor [BLOCK_SIZE_K, BLOCK_SIZE_N] (Ptr .bf16) :=
      b_ptr + (offs_k.col BLOCK_SIZE_N * stride_bk + offs_n.row BLOCK_SIZE_K * stride_b_seqlen)
    -- line 145: `seqlen - pid_c * chunk_size` is `int64` (`seqlen` and `chunk_size` sign-extended);
    -- Python's `min` with a tensor operand is `tl.minimum`, which sign-extends `chunk_size` to the
    -- other operand's `int64`: the exact `tl.iminimum`
    let chunk_size_limit : I64 := (tl.iminimum (Tensor.scalar (← wide chunk_size))
      (Tensor.scalar ((← wide seqlen) - pid_c * (← wide chunk_size)))).item
    -- line 147: `float32` `+0.0` in every element
    let acc := tl.zeros [BLOCK_SIZE_M, BLOCK_SIZE_N] .f32
    -- line 148: `for k in range(0, tl.cdiv(K, BLOCK_SIZE_K))`: the bound
    -- `(K + (BLOCK_SIZE_K - 1)) // BLOCK_SIZE_K` in `int32`, evaluated once before the loop (a
    -- two-argument `range`); the `int32` bounds `0`, the step `1` and the induction variable
    -- `k`; the loop carries `a_ptrs`, `b_ptrs`, `acc` (assigned in the body, defined before it)
    let (_a_ptrs, _b_ptrs, acc) ← tl.forRange (IntV.ofInt 0 : I32)
        (← tl.idiv (Tensor.scalar (K + (IntV.ofNat (BLOCK_SIZE_K - 1) : I32)))
          (Tensor.scalar (IntV.ofNat BLOCK_SIZE_K : I32))).item
        (IntV.ofInt 1 : I32) (a_ptrs, b_ptrs, acc)
        fun k (a_ptrs, b_ptrs, acc) => do
      -- lines 149–154: the mask `(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < K -
      -- k * BLOCK_SIZE_K)`: the first comparison with the `int64` scalar sign-extends `offs_m`; the
      -- second is `int32` (wrapping product and difference); `&` on `int1`. `other=0.0`: the
      -- `float32` `+0.0`, converted to the pointer's dtype `bfloat16` (`tl.convert`).
      -- `.to(dot_dtype)` is `.to(tl.bfloat16)` of a `bfloat16` tensor: the identity
      let a ← tl.load a_ptrs
        (some (((← tl.convert .i32 .i64 (offs_m.col BLOCK_SIZE_K)).ltS chunk_size_limit).and
          ((offs_k.row BLOCK_SIZE_M).ltS (K - k * (IntV.ofNat BLOCK_SIZE_K : I32)))))
        (some (← tl.convert .f32 .bf16 (tl.full [BLOCK_SIZE_M, BLOCK_SIZE_K] zf32)))
      -- lines 155–160: the same for `b`, `[BLOCK_SIZE_K, BLOCK_SIZE_N]`
      let b ← tl.load b_ptrs
        (some (((offs_k.col BLOCK_SIZE_N).ltS (K - k * (IntV.ofNat BLOCK_SIZE_K : I32))).and
          ((← tl.convert .i32 .i64 (offs_n.row BLOCK_SIZE_K)).ltS chunk_size_limit)))
        (some (← tl.convert .f32 .bf16 (tl.full [BLOCK_SIZE_K, BLOCK_SIZE_N] zf32)))
      -- line 161: `acc += tl.dot(a, b)` is `acc = acc + tl.dot(a, b)`, a source `c + tl.dot(a, b)`
      -- whose dot has the default zero accumulator: the combine pass may rewrite it to
      -- `tl.dot(a, b, acc)` (`tl.dot_add`)
      let acc ← tl.dot_add cfgBF16 a b acc
      -- lines 162–163: `x += y` is `x = x + y`: the constexpr `BLOCK_SIZE_K` built in `int32` as
      -- an operand of `*` with the `int32` scalar stride (wrapping), added to every pointer
      let a_ptrs := a_ptrs +
        (bc ((IntV.ofNat BLOCK_SIZE_K : I32) * stride_ak) : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_K] I32)
      let b_ptrs := b_ptrs +
        (bc ((IntV.ofNat BLOCK_SIZE_K : I32) * stride_bk) : Tensor [BLOCK_SIZE_K, BLOCK_SIZE_N] I32)
      return (a_ptrs, b_ptrs, acc)
    -- lines 165–166: `offs_m`, `offs_n` computed again (the same values)
    let offs_m : Tensor [BLOCK_SIZE_M] I32 :=
      pid_m * (IntV.ofNat BLOCK_SIZE_M : I32) + tl.arange 0 BLOCK_SIZE_M
    let offs_n : Tensor [BLOCK_SIZE_N] I32 :=
      pid_n * (IntV.ofNat BLOCK_SIZE_N : I32) + tl.arange 0 BLOCK_SIZE_N
    -- line 167: `if HAS_SEQ_IDX:` taken
    -- line 168: as line 145
    let chunk_size_limit : I64 := (tl.iminimum (Tensor.scalar (← wide chunk_size))
      (Tensor.scalar ((← wide seqlen) - pid_c * (← wide chunk_size)))).item
    -- lines 169–173: `offs_m * stride_seq_idx_seqlen` in `int32` (wrapping), the pointer `+`;
    -- the mask sign-extends `offs_m` for the comparison with the `int64` scalar; `other=-1`: the
    -- Python `-1`, an `int32` tensor, already the pointer's dtype
    let seq_idx_m : Tensor [BLOCK_SIZE_M] I32 ←
      tl.load (seq_idx_ptr + offs_m * stride_seq_idx_seqlen)
        (some ((← tl.convert .i32 .i64 offs_m).ltS chunk_size_limit))
        (some (tl.full [BLOCK_SIZE_M] (IntV.ofInt (-1 : Int) : I32)))
    -- lines 174–178: the same for `offs_n`, `other=-2`
    let seq_idx_n : Tensor [BLOCK_SIZE_N] I32 ←
      tl.load (seq_idx_ptr + offs_n * stride_seq_idx_seqlen)
        (some ((← tl.convert .i32 .i64 offs_n).ltS chunk_size_limit))
        (some (tl.full [BLOCK_SIZE_N] (IntV.ofInt (-2 : Int) : I32)))
    -- line 179: `seq_idx_m[:, None] == seq_idx_n[None, :]` on `int32` (`int1`); `tl.where` in the
    -- dtype of `acc`, `float32`, with the constant `0.0` built in it
    let acc := tl.where ((seq_idx_m.col BLOCK_SIZE_N).eq (seq_idx_n.row BLOCK_SIZE_M)) acc
      (tl.full [BLOCK_SIZE_M, BLOCK_SIZE_N] zf32)
    -- line 180: `acc.to(out_ptr.dtype.element_ty)`: `out` is `float32`, so `.to(tl.float32)` of a
    -- `float32` tensor, the identity
    let out := acc
    -- lines 182–184: as lines 125–129 (`pid_b * stride_out_batch` by `batchOff`;
    -- `pid_c * stride_out_chunk`, `pid_h * stride_out_head` `int64`)
    let out_ptr := out_ptr + ((← batchOff pid_b stride_out_batch) +
      pid_c * (← wide stride_out_chunk) + pid_h * (← wide stride_out_head))
    -- line 185: `int32` products and sum (wrapping), the pointer `+` on each lane
    let out_ptrs : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N] (Ptr .f32) :=
      out_ptr + (stride_outm * offs_m.col BLOCK_SIZE_N + offs_n.row BLOCK_SIZE_M * stride_outn)
    -- lines 186–190: the mask `(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size)`,
    -- `int32` comparisons, `&` on `int1`; `out` is already `float32`, the pointer's dtype
    tl.store out_ptrs out
      (some ((offs_m.col BLOCK_SIZE_N).ltS chunk_size |>.and ((offs_n.row BLOCK_SIZE_M).ltS chunk_size)))

/-! ## The conditions -/

/-- What the host chooses: one launch of `bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx,
output_dtype=torch.float32)` with `a = C`, `b = B` (`[1, seqlen, ngroups, dstate]`, `bfloat16`,
any strides: contiguous, or the views of a packed `xBC` tensor), `seq_idx` `[1, seqlen]` `int32`
contiguous, and the fresh contiguous `out`. The buffers are given by their storage (`Buf`). -/
structure Inst where
  /-- `seqlen` (`a.shape[1]`): the packed prefill tokens. -/
  T : Nat
  /-- `ngroups` (`a.shape[2]`). -/
  G : Nat
  /-- `K = dstate` (`a.shape[3]`). -/
  K : Nat
  /-- `chunk_size`. -/
  C : Nat
  /-- `a.stride(0)`, `a.stride(1)`, `a.stride(2)`, `a.stride(3)`, in elements. -/
  aB : Nat
  aS : Nat
  aH : Nat
  aK : Nat
  /-- `b.stride(0)`, …, `b.stride(3)`, in elements. -/
  bB : Nat
  bS : Nat
  bH : Nat
  bK : Nat
  /-- The storage of `a` (`C`), `bfloat16`: element `(0, t, g, k)` is element
  `t·aS + g·aH + k·aK` of it. -/
  a : Buf .bf16
  /-- The storage of `b` (`B`), `bfloat16`. -/
  b : Buf .bf16
  /-- The storage of `seq_idx`, `int32`: element `(0, t)` is element `t`. -/
  seqIdx : Buf .i32
  /-- The storage of `out`, `float32` (allocated by the host). -/
  out : Buf .f32

/-- `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K` (the host's line 239). -/
def BM : Nat := 64

def BN : Nat := 128

def BK : Nat := 32

/-- `nchunks = ceil(seqlen / chunk_size)` (line 217). -/
def nch (d : Inst) : Nat := (d.T + d.C - 1) / d.C

/-- `out`'s strides (a contiguous `[1, nchunks, ngroups, chunk_size, chunk_size]`):
`stride(0) = nchunks·ngroups·C²`, `stride(1) = ngroups·C²`, `stride(2) = C²`, `stride(-2) = C`,
`stride(-1) = 1`. -/
def oB (d : Inst) : Nat := nch d * (d.G * (d.C * d.C))

def oC (d : Inst) : Nat := d.G * (d.C * d.C)

def oH (d : Inst) : Nat := d.C * d.C

/-- The grid's `x`: `cdiv(chunk_size, 64) · cdiv(chunk_size, 128)` (line 241). -/
def gx (d : Inst) : Nat := (d.C + BM - 1) / BM * ((d.C + BN - 1) / BN)

/-- The grid's `z`: `nchunks · ngroups` (line 243). -/
def gz (d : Inst) : Nat := nch d * d.G

/-- Element `(0, t, g, k)` of `a`, of `b`. -/
def aIdx (d : Inst) (t g k : Nat) : Nat := t * d.aS + g * d.aH + k * d.aK

def bIdx (d : Inst) (t g k : Nat) : Nat := t * d.bS + g * d.bH + k * d.bK

/-- Element `(0, c, g, m, n)` of `out`. -/
def oIdx (d : Inst) (c g m n : Nat) : Nat :=
  c * oC d + g * oH d + m * d.C + n

/-- The instances covered (README, Conditions). -/
structure Inst.ok (d : Inst) : Prop where
  /-- At least one token, one group, one state element. -/
  T_pos : 0 < d.T
  G_pos : 0 < d.G
  K_pos : 0 < d.K
  /-- `chunk_size` is a power of two (the code base asserts it, `ssd_combined.py:77`), at least `2`. -/
  C_pow2 : ∃ j, d.C = 2 ^ j
  C_ge : 2 ≤ d.C
  /-- `dstate ≤ 2^10`: the tolerance stays below `2.7·10^-4` of the products' magnitudes
  (`tol_rel_lt`); every Mamba-2 model's `dstate` is at most `256` (README, Exclusions). -/
  K_le : d.K ≤ 2 ^ 10
  /-- The grid's `z = nchunks · ngroups` is within CUDA's limit `2^16 − 1` (a kernel limit:
  README, Exclusions); with it `ngroups < 2^31`. -/
  gz_lt : gz d < 2 ^ 16
  /-- The `int32` arguments: `seqlen` (and `seq_idx.stride(0) = seqlen`); the strides of `a` and
  `b` other than `stride(0)`; `out.stride(1) = ngroups·C²` (the largest of `out`'s strides after
  `stride(0)`, and it bounds `chunk_size`, `C·C` and the store offsets `m·C + n < C²`). The batch
  strides `a.stride(0)`, `b.stride(0)` are `int32` or `int64` arguments (below `2^63`); so is
  `out.stride(0) = nchunks·oC`, below `2^47` by `gz_lt` and `oC_lt`. -/
  T_lt : d.T < 2 ^ 31
  aB_lt : d.aB < 2 ^ 63
  aS_lt : d.aS < 2 ^ 31
  aH_lt : d.aH < 2 ^ 31
  aK_lt : d.aK < 2 ^ 31
  bB_lt : d.bB < 2 ^ 63
  bS_lt : d.bS < 2 ^ 31
  bH_lt : d.bH < 2 ^ 31
  bK_lt : d.bK < 2 ^ 31
  oC_lt : oC d < 2 ^ 31
  /-- The `int32` lane offsets of lines 143–144 do not wrap on the lanes the loads enable (rows
  and columns below `chunk_size`, `k`-lanes below `32`), nor does the step `32·stride` of lines
  162–163. -/
  a_fit : (d.C - 1) * d.aS + (BK - 1) * d.aK < 2 ^ 31
  b_fit : (BK - 1) * d.bK + (d.C - 1) * d.bS < 2 ^ 31
  aK_step : BK * d.aK < 2 ^ 31
  bK_step : BK * d.bK < 2 ^ 31
  /-- Every element of `a`, `b`, `seq_idx`, `out` lies in its storage. -/
  a_len : ∀ t g k, t < d.T → g < d.G → k < d.K → aIdx d t g k < d.a.len
  b_len : ∀ t g k, t < d.T → g < d.G → k < d.K → bIdx d t g k < d.b.len
  seq_len : d.T ≤ d.seqIdx.len
  out_len : oB d ≤ d.out.len
  /-- `out` shares no byte with the inputs (the host allocates it, lines 220–228). `a` and `b`
  may overlap each other (both are only read; the Mamba-2 mixer's `C` and `B` are views of one
  tensor). -/
  out_a : Disjoint d.out.bytes d.a.bytes
  out_b : Disjoint d.out.bytes d.b.bytes
  out_s : Disjoint d.out.bytes d.seqIdx.bytes

/-- The model on an instance: the launch of lines 245–281 with `HAS_SEQ_IDX = True`, the
instance's strides, `out`'s contiguous strides and `seq_idx`'s `(seqlen, 1)`, and the block sizes
`64, 128, 32`. The three batch strides are `int32` or `int64` arguments by their values
(`intArg`). An integer argument equal to `1` is the compile-time constant `1`; each use of it
gives the value of the model's `IntV.ofNat 1`: as an operand of `*`, `+`, `-` or `//` with an
`int32` or `int64` tensor, a comparison or a `min`, it is built in that dtype; `tl.cdiv(1, B)` with
a constexpr `B` is Python's `1`, the value of the `int32` computation, held by `num_pid_n` (an
`int32` scalar tensor) or used as the `range` bound. -/
def model (d : Inst) : Kernel :=
  bmm_chunk_fwd_kernel (d.a.ptr 0) (d.b.ptr 0) (d.out.ptr 0) (d.seqIdx.ptr 0)
    (IntV.ofNat d.T) (IntV.ofNat d.C) (IntV.ofNat d.K) (IntV.ofNat d.G)
    (intArg d.aB) (IntV.ofNat d.aS) (IntV.ofNat d.aH) (IntV.ofNat d.aK)
    (intArg d.bB) (IntV.ofNat d.bS) (IntV.ofNat d.bH) (IntV.ofNat d.bK)
    (intArg (oB d)) (IntV.ofNat (oC d)) (IntV.ofNat (oH d)) (IntV.ofNat d.C) (IntV.ofNat 1)
    (IntV.ofNat d.T) (IntV.ofNat 1)
    BM BN BK

/-- The host's launch (lines 239–281): the grid `(cdiv(C, 64)·cdiv(C, 128), 1, nchunks·ngroups)`;
the source's block shapes (`tl.arange`s, the loaded tiles, the accumulator); `enable_fp_fusion`
at Triton's default (`true`); `num_ctas = 1`, no `launch_pdl`, `debug = false`. `num_warps = 4`
and `num_stages = 4` are performance-only. -/
def launch (d : Inst) : Launch :=
  { grid := ![gx d, 1, gz d], fusion := true,
    blocks := [[BM], [BN], [BK], [BM, BK], [BK, BN], [BM, BN]] }

/-- The `int32` value at element `i` of an `int32` buffer in memory `m` (`0` when absent). -/
def val32 (m : Mem) (b : Buf .i32) (i : Nat) : Int :=
  match m.read .i32 (b.ptr (i : Int)).addr with
  | some v => v.toInt
  | none => 0

/-- The inputs covered: the four buffers are valid (aligned, inside `[0, 2^64)`, every byte
present), and every `seq_idx` entry is a non-negative `int32` (real use: the request of each
token, `0, 1, …`). Any `bfloat16` values in `a` and `b`, NaN and infinities included. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.a.Valid init ∧ d.b.Valid init ∧ d.seqIdx.Valid init ∧ d.out.Valid init ∧
    ∀ t, t < d.T → ∃ v, init.read .i32 (d.seqIdx.ptr (t : Int)).addr = some v ∧ 0 ≤ v.toInt

/-! ## The specification -/

/-- The number of `tl.dot` steps of the loop (line 148): `cdiv(K, 32)`. -/
def steps (K : Nat) : Nat := (K + BK - 1) / BK

/-- The per-step relative constant: `DotLift.dotAdd_step`'s for one `bfloat16` `tl.dot` of `32`
products, either form of `tl.dot_add`: `relDot16 32 = 34·2^-22` plus the room of a separate
`float32` addition, `3·2^-24·(1 + relDot16 32)`. -/
def rho : ℚ := DotLoop.stepRel Rules.Num.relDot16 32

/-- The per-step absolute constant: `absDot16 32·(1 + 3·2^-24) + 4·2^-150`. -/
def alpha : ℚ := DotLoop.stepAbs Rules.Num.absDot16 32

/-- The exact dot product of row values `va` and column values `vb` over `k < K`. -/
def dotSum (va vb : Nat → BF16) (K : Nat) : ℚ := ∑ k ∈ Finset.range K, (va k).toRat * (vb k).toRat

/-- The sum of the products' lifted magnitudes, `Σ liftMag (va k)·liftMag (vb k)` over `k < K`
(`Rules.Num.liftMag .bf16 q` is `|q|` for zero and normal values, `2^-126` for a nonzero
subnormal one). -/
def dotMag (va vb : Nat → BF16) (K : Nat) : ℚ :=
  ∑ k ∈ Finset.range K, Rules.Num.liftMag .bf16 (va k).toRat * Rules.Num.liftMag .bf16 (vb k).toRat

/-- Product `e` of the reduction: `va e · vb e` for `e < K`, and `0` from `K` on (the lanes the
masks of lines 151–152 and 157–158 disable, loaded as `+0`). -/
def prodV (va vb : Nat → BF16) (K e : Nat) : ℚ := if e < K then (va e).toRat * (vb e).toRat else 0

/-- The lifted magnitude of product `e`, `0` from `K` on. -/
def liftV (va vb : Nat → BF16) (K e : Nat) : ℚ :=
  if e < K then Rules.Num.liftMag .bf16 (va e).toRat * Rules.Num.liftMag .bf16 (vb e).toRat else 0

/-- The exact partial sum before step `t`: `p e` for `e < 32·t`. -/
def partSum (p : Nat → ℚ) (t : Nat) : ℚ := ∑ e ∈ Finset.range (BK * t), p e

/-- The lifted magnitude of step `t`: `Σ q e` for `32·t ≤ e < 32·t + 32`. -/
def stepMag (q : Nat → ℚ) (t : Nat) : ℚ := ∑ k ∈ Finset.range BK, q (BK * t + k)

/-- **The error bound after `t` steps** of the loop, from the exact partial sums: `E 0 = 0` and
`E (t + 1) = (1 + ρ)·E t + ρ·(|S t| + L t) + α`, with `S t = partSum p t` and `L t = stepMag q t`.
Step `t` (`acc += tl.dot(a, b)`, line 161) leaves, by `Rules.Num.DotBF16` through
`DotLift.dotAdd_step`, a value within `ρ·(|acc| + L t) + α` of `acc + c t` (`c t` the step's exact
sum); with `acc` within `E t` of `S t`, `|acc| ≤ |S t| + E t`. -/
def errAt (p q : Nat → ℚ) : Nat → ℚ
  | 0 => 0
  | t + 1 => (1 + rho) * errAt p q t + rho * (|partSum p t| + stepMag q t) + alpha

/-- **The tolerance of `Post`** for row values `va`, column values `vb` and `K`: the bound after
the loop's `steps K` steps. -/
def tol (va vb : Nat → BF16) (K : Nat) : ℚ := errAt (prodV va vb K) (liftV va vb K) (steps K)

/-- The relative factor `(1 + ρ)^T − 1` on covered instances (`K ≤ 2^10`, `Inst.ok.K_le`): below
`2.7·10^-4`. `tol ≤ ((1 + ρ)^T − 1)·dotMag + T·(1 + ρ)^T·α` (README, Tolerances). -/
theorem tol_rel_lt {K : Nat} (hK : K ≤ 2 ^ 10) :
    (1 + rho) ^ steps K - 1 < 27 / 100000 := by
  have hT : steps K ≤ 32 := by unfold steps BK; omega
  have hr : rho = 34 / 2 ^ 22 + 3 * Math.Float.Format.binary32.u * (1 + 34 / 2 ^ 22) := by
    unfold rho DotLoop.stepRel Rules.Num.relDot16
    norm_num
  have h1 : (1 : ℚ) ≤ 1 + rho := by rw [hr, Math.Float.Format.binary32_u]; norm_num
  have h2 := pow_le_pow_right₀ h1 hT
  have h3 : (1 + rho) ^ 32 < 1 + 27 / 100000 := by
    rw [hr, Math.Float.Format.binary32_u]; norm_num
  linarith

/-- The two positions `(m, n)` of chunk `c` are tokens of the sequence (`tm = c·C + m < seqlen`,
`tn = c·C + n < seqlen`) of the same request: `seq_idx[0, tm] = seq_idx[0, tn]`. -/
def sameReq (d : Inst) (init : Mem) (c m n : Nat) : Bool :=
  decide (c * d.C + m < d.T) && decide (c * d.C + n < d.T) &&
    decide (val32 init d.seqIdx (c * d.C + m) = val32 init d.seqIdx (c * d.C + n))

/-- What the launch leaves in element `(0, c, g, m, n)` of `out`: a `float32` value `v`; if the
positions are tokens of one request (`sameReq`), then whenever row `tm` of `a` and row `tn` of
`b` (group `g`) hold finite values `va k`, `vb k` (`k < K`) and the products' lifted magnitudes
leave room in `float32` (`dotMag ≤ 2^120`), `v` is finite and within `tol` of the exact sum
`Σ_k va k · vb k`; otherwise `v` is the `float32` `+0.0`, for every input. -/
def OutElem (d : Inst) (init fin : Mem) (c g m n : Nat) : Prop :=
  ∃ v : F32, fin.read .f32 (d.out.ptr (oIdx d c g m n : Int)).addr = some v ∧
    if sameReq d init c m n then
      ∀ va vb : Nat → BF16,
        (∀ k, k < d.K → init.read .bf16 (d.a.ptr (aIdx d (c * d.C + m) g k : Int)).addr = some (va k) ∧
          (va k).isFinite = true) →
        (∀ k, k < d.K → init.read .bf16 (d.b.ptr (bIdx d (c * d.C + n) g k : Int)).addr = some (vb k) ∧
          (vb k).isFinite = true) →
        dotMag va vb d.K ≤ 2 ^ 120 →
        v.isFinite = true ∧ |v.toRat - dotSum va vb d.K| ≤ tol va vb d.K
    else
      FloatV.NaNEq (FloatV.ofPat 0) v

/-- What the launch leaves: every element of `out` (`nchunks × ngroups × C × C`) as `OutElem`
says; every byte outside `out`'s `nchunks·ngroups·C²` elements unchanged (`a`, `b`, `seq_idx`). -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ c g m n, c < nch d → g < d.G → m < d.C → n < d.C → OutElem d init fin c g m n) ∧
    ∀ z, z ∉ d.out.elems {j | j < oB d} → fin z = init z

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## The assumptions -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on every covered instance, when `launch d` has `debug = false`
  and is valid, every outcome of the compiled kernel launched as `launch d`, from a memory from
  which the model's launch is well behaved, is one of the outcomes of the model's launch. The
  rule covers a launch within its scope: part (A), the kernel and its launches, which the host's
  launches meet (README, Conditions), and part (B), the environment, which the claim relies on. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)
  /-- [rule] `Num.DotBF16`: `tl.dot` of `bfloat16` operands into `float32` (`maxImpreciseAcc =
  0`, any `input_precision`, the model's line 161; `Rules.Num.DotLiftAcc`): each output element
  whose row of `a`, column of `b` and accumulator have values is finite and within
  `relDot16 K·L + absDot16 K` of `acc + Σ a·b`, with `L = |acc| + Σ liftMag a·liftMag b` (a
  subnormal operand counted at the smallest normal `2^-126`), when that leaves room below the
  `float32` range. -/
  dot : Rules.Num.DotBF16 P
  /-- [rule] `Num.DotTotal`: the model's `tl.dot` (line 161) always has a result. -/
  dot_total : Rules.Num.DotTotal P cfgBF16

/-! ## The claim -/

/-- The claim to prove: under the assumptions, the launch meets the specification on every
covered instance and input. Not proved here. -/
def Statement : Prop := ∀ K P, Assumptions K P → Spec K

/-! ## Not vacuous: the assumptions are consistent -/

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, htot, ⟨fun d _ => hf d, Platform.sm90ref.dotBF16,
    Platform.ref.dotTotal _⟩⟩

/-! ## Coverage: the launches `SETUP.md` names

`mk T G K C cd` is the instance the host builds for `seqlen = T`, `ngroups = G`, `dstate = K`,
`chunk_size = C`, with `C` and `B` of seqlen stride `cd`: `cd = G·K` for contiguous tensors (the
code base's tests), `cd = conv_dim` for the Mamba-2 mixer's views of its packed `xBC` tensor
(strides `(T·cd, cd, K, 1)`). The four buffers sit at the aligned addresses `k·2^36`
(`k = 1 … 4`); the claim covers any addresses `Inst.ok` allows, and these only witness the sizes. -/

/-- The host's instance for `(T, G, K, C, cd)`. -/
def mk (T G K C cd : Nat) : Inst where
  T := T
  G := G
  K := K
  C := C
  aB := T * cd
  aS := cd
  aH := K
  aK := 1
  bB := T * cd
  bS := cd
  bH := K
  bK := 1
  a := ⟨1 * 68719476736, T * cd⟩
  b := ⟨2 * 68719476736, T * cd⟩
  seqIdx := ⟨3 * 68719476736, T⟩
  out := ⟨4 * 68719476736, (T + C - 1) / C * (G * (C * C))⟩

/-- `t·X + x < T·X` for `t < T`, `x < X`. -/
theorem row_lt {t T X x : Nat} (ht : t < T) (hx : x < X) : t * X + x < T * X := by
  have h : (t + 1) * X ≤ T * X := Nat.mul_le_mul_right X ht
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- Two buffers whose byte ranges are in order share no byte. -/
theorem bytes_disjoint {d₁ d₂ : DType} {x : Buf d₁} {y : Buf d₂}
    (h : x.base + x.len * d₁.bytes ≤ y.base ∨ y.base + y.len * d₂.bytes ≤ x.base) :
    Disjoint x.bytes y.bytes := by
  rw [Set.disjoint_left]
  intro z hx hy
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

/-- The host's instance is covered when its sizes meet `Inst.ok`'s numeric bounds, `G·K ≤ cd`,
and the buffers fit the address layout of `mk`. -/
theorem mk_ok {T G K C cd j : Nat} (hC : C = 2 ^ j)
    (h : 0 < T ∧ 0 < G ∧ 0 < K ∧ 2 ≤ C ∧ K ≤ 2 ^ 10 ∧ (T + C - 1) / C * G < 2 ^ 16 ∧
      T < 2 ^ 31 ∧ T * cd < 2 ^ 63 ∧ cd < 2 ^ 31 ∧ K < 2 ^ 31 ∧
      G * (C * C) < 2 ^ 31 ∧
      (C - 1) * cd + 31 < 2 ^ 31 ∧ G * K ≤ cd ∧
      T * cd * 2 ≤ 68719476736 ∧ T * 4 ≤ 68719476736 ∧
      (T + C - 1) / C * (G * (C * C)) * 4 ≤ 68719476736) :
    (mk T G K C cd).ok := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, hA, hS, hO⟩ := h
  have hlen : ∀ t g k, t < T → g < G → k < K → t * cd + g * K + k * 1 < T * cd := by
    intro t g k ht hg hk
    have h1' : g * K + k < G * K := row_lt hg hk
    have h2' : t * cd + (g * K + k) < T * cd := row_lt ht (by omega)
    omega
  refine
    { T_pos := h1, G_pos := h2, K_pos := h3, C_pow2 := ⟨j, hC⟩, C_ge := h4, K_le := h5,
      gz_lt := h6, T_lt := h7, aB_lt := h8, aS_lt := h9, aH_lt := h10, aK_lt := by simp only [mk]; norm_num,
      bB_lt := h8, bS_lt := h9, bH_lt := h10, bK_lt := by simp only [mk]; norm_num, oC_lt := h11,
      a_fit := by simp only [mk, BK]; omega, b_fit := by simp only [mk, BK]; omega,
      aK_step := by simp only [mk, BK]; norm_num, bK_step := by simp only [mk, BK]; norm_num,
      a_len := hlen, b_len := hlen, seq_len := le_refl _, out_len := le_refl _,
      out_a := ?_, out_b := ?_, out_s := ?_ }
  all_goals
    refine bytes_disjoint ?_
    simp only [mk, DType.bytes]
    omega

/-- SGLang's single-example test (`test_mamba_ssm_ssd.py:228-286`): 112 tokens, chunks of 16,
`ngroups = n_heads ∈ {3, 32}`, `dstate = d_head ∈ {5, 128}`, one request, contiguous. -/
def setup_single_g3_k5 : Inst := mk 112 3 5 16 15
theorem setup_single_g3_k5_ok : setup_single_g3_k5.ok := mk_ok (j := 4) rfl (by norm_num)

def setup_single_g32_k128 : Inst := mk 112 32 128 16 4096
theorem setup_single_g32_k128_ok : setup_single_g32_k128.ok := mk_ok (j := 4) rfl (by norm_num)

/-- The single-example test's other sizes outside CI (`test_mamba_ssm_ssd.py:219-222`: 128 tokens
in chunks of 32, `ngroups` up to 32, `dstate` up to 128), and the intermediate-states test
(`test_mamba_ssm_ssd.py:632-692`: `bfloat16`, 128 tokens in chunks of 32 or 256 in chunks of 64,
`ngroups ∈ {4, 16}`, `dstate ∈ {32, 64}`). -/
def setup_single_g11_k19_c32 : Inst := mk 128 11 19 32 209
theorem setup_single_g11_k19_c32_ok : setup_single_g11_k19_c32.ok := mk_ok (j := 5) rfl (by norm_num)

def setup_inter_g16_k64_c64 : Inst := mk 256 16 64 64 1024
theorem setup_inter_g16_k64_c64_ok : setup_inter_g16_k64_c64.ok := mk_ok (j := 6) rfl (by norm_num)

/-- The continuous-batch test (`test_mamba_ssm_ssd.py:289-394`): 64 tokens in chunks of 8 or 256,
768 tokens in chunks of 128; `ngroups ∈ {4, 13}`, `dstate ∈ {5, 32}`. -/
def setup_cont_c8_g4_k5 : Inst := mk 64 4 5 8 20
theorem setup_cont_c8_g4_k5_ok : setup_cont_c8_g4_k5.ok := mk_ok (j := 3) rfl (by norm_num)

def setup_cont_c256_g13_k32 : Inst := mk 64 13 32 256 416
theorem setup_cont_c256_g13_k32_ok : setup_cont_c256_g13_k32.ok := mk_ok (j := 8) rfl (by norm_num)

def setup_cont_c128_g13_k32 : Inst := mk 768 13 32 128 416
theorem setup_cont_c128_g13_k32_ok : setup_cont_c128_g13_k32.ok := mk_ok (j := 7) rfl (by norm_num)

/-- The prefill-chunking test (`test_mamba_ssm_ssd.py:397-529`): `ngroups = 16`, `dstate = 64`,
requests `(270, 88, 212, 203)` (773 tokens), chunks of 8 and 256. -/
def setup_chunking_c8 : Inst := mk 773 16 64 8 1024
theorem setup_chunking_c8_ok : setup_chunking_c8.ok := mk_ok (j := 3) rfl (by norm_num)

def setup_chunking_c256 : Inst := mk 773 16 64 256 1024
theorem setup_chunking_c256_ok : setup_chunking_c256.ok := mk_ok (j := 8) rfl (by norm_num)

/-- The Mamba-2 mixer at a 32768-token prefill: Codestral-Mamba / Nemotron-H-like layers at TP 1
(`ngroups = 8`, `dstate = 128`, `intermediate = 8192`, `conv_dim = 10240`), chunks of 256 and 128. -/
def setup_mamba2_g8_c256 : Inst := mk 32768 8 128 256 10240
theorem setup_mamba2_g8_c256_ok : setup_mamba2_g8_c256.ok := mk_ok (j := 8) rfl (by norm_num)

def setup_mamba2_g8_c128 : Inst := mk 32768 8 128 128 10240
theorem setup_mamba2_g8_c128_ok : setup_mamba2_g8_c128.ok := mk_ok (j := 7) rfl (by norm_num)

/-- One group per rank (`ngroups = 1`, or groups split across tensor-parallel ranks),
`dstate = 128` (Granite-4.0-H, Mamba-2 1-group layers) and `dstate = 256` (Falcon-H1), chunks of
256, 32768 tokens. -/
def setup_g1_k128 : Inst := mk 32768 1 128 256 4352
theorem setup_g1_k128_ok : setup_g1_k128.ok := mk_ok (j := 8) rfl (by norm_num)

def setup_g1_k256 : Inst := mk 32768 1 256 256 4608
theorem setup_g1_k256_ok : setup_g1_k256.ok := mk_ok (j := 8) rfl (by norm_num)

/-- A long prefill with chunked prefill disabled, through the Mamba-2 mixer (`ngroups = 8`,
`dstate = 128`, `conv_dim = 10240`, chunks of 256): 262,144 tokens, where `a.stride(0) =
b.stride(0) = seqlen·conv_dim ≈ 2.7·10^9` are `int64` arguments (`out.stride(0) = 2^29` is
`int32`); and 2,096,896 tokens (8191 chunks, the largest grid `z` below `2^16`), where
`out.stride(0) = nchunks·ngroups·C² ≈ 4.3·10^9` is `int64` too. -/
def setup_long_g8_c256 : Inst := mk 262144 8 128 256 10240
theorem setup_long_g8_c256_ok : setup_long_g8_c256.ok := mk_ok (j := 8) rfl (by norm_num)

def setup_longest_g8_c256 : Inst := mk 2096896 8 128 256 10240
theorem setup_longest_g8_c256_ok : setup_longest_g8_c256.ok := mk_ok (j := 8) rfl (by norm_num)

/-! ## Not vacuous: a covered instance and memory -/

/-- A memory in which every byte is present (and `0`). -/
def zeroMem : Mem := fun _ => some 0

/-- Every aligned `int32` read of `zeroMem` gives `0`. -/
theorem zero_i32 {a : Int} (ha : Mem.Aligned .i32 a) : zeroMem.read .i32 a = some (IntV.ofNat 0) := by
  have hb : zeroMem.readBytes a 4 = some [0, 0, 0, 0] := by simp [Mem.readBytes, zeroMem]
  unfold Mem.read
  rw [show DType.i32.bytes = 4 from rfl]
  simp only [ha, ↓reduceIte, hb, Option.map_some]
  rfl

theorem valid_zero {dt : DType} (b : Buf dt) (hb : 0 ≤ b.base)
    (h2 : b.base + b.len * dt.bytes ≤ 2 ^ 64) (ha : b.base % dt.bytes = 0) : b.Valid zeroMem :=
  ⟨hb, h2, ha, fun _ _ => by simp [zeroMem]⟩

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  refine ⟨setup_single_g3_k5, zeroMem, setup_single_g3_k5_ok, ?_⟩
  refine ⟨valid_zero _ (by decide) (by decide) (by decide), valid_zero _ (by decide) (by decide) (by decide),
    valid_zero _ (by decide) (by decide) (by decide), valid_zero _ (by decide) (by decide) (by decide), ?_⟩
  intro t ht
  refine ⟨IntV.ofNat 0, zero_i32 ?_, ?_⟩
  · have ht' : t < 112 := ht
    show 0 ≤ _ ∧ _ ∧ _
    simp only [setup_single_g3_k5, mk, Buf.ptr, DType.bytes]
    omega
  · rw [IntV.I32.toInt_ofNat (by norm_num)]
    norm_num

end Claims.SsdBmm

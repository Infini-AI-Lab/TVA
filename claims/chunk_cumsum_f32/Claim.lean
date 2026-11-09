import TVA

/-!
# `chunk_cumsum_f32`: the prepared claim for fla's `chunk_local_cumsum_scalar_kernel`

Source: `source/chunk_cumsum.py`. The kernel is
`chunk_local_cumsum_scalar_kernel` (lines 80–120), which calls the `@triton.jit` helper
`unflatten_program_id` (lines 70–77); it reads the module-level names `triton` and `tl` (lines
66–67). The entry point `chunk_local_cumsum` (lines 209–247) launches it through
`chunk_local_cumsum_scalar` (lines 162–206) on a `float32` gate `s` (`[B, T, H]`) and a fresh
`float32` output `o`, over fixed-length rows (`cu_seqlens is None`) or a packed variable-length
batch (`B = 1`, `cu_seqlens` and `chunk_indices` of one dtype, `int32` or `int64`); one program per
(chunk, batch row, head).

The claim (`Statement`): on every covered instance, every launch that runs ends normally; for
every sequence (a row of a fixed-length batch, or a sequence `[cu_seqlens[n], cu_seqlens[n+1])`),
every position `p` in it and every head `h`, the output element `o[p, h]` holds a `float32` value
which, when its inputs are finite and the magnitudes leave room (`InRange`), is finite and within
`tol` of `σ · Σ s[q, h]` over `q` from the start of `p`'s chunk to `p` (forward), or from `p` to
the chunk's end (reverse), chunks of `BT` positions counted from the sequence's start (`σ` the
host's double scale, or `1`); every other byte is unchanged.

* The model: `chunk_local_cumsum_scalar_kernel`, `model`, `launch`.
* The conditions: `Inst`, `Inst.ok`, `Pre` (the tables as `prepare_chunk_indices` builds them).
* The specification: `nSeq`, `lo`, `hi`, `chunkStart`, `rng`, `refSum`, `absSum`, `InRange`,
  `tol`, `Post`, `Spec`.
* The assumptions: one `[rule]` field, `Compiler.Faithful` (IEEE `+` and `*` only).
* Proved here: `consistent`, `satisfiable`, and the coverage of the host's launches. `Statement`
  is the claim to prove.
-/

namespace Claims.ChunkCumsum

open TVA TVA.Math.Float

/-! ## The model -/

set_option linter.unusedVariables false in
/-- `chunk_local_cumsum_scalar_kernel` (source lines 80–120) with `unflatten_program_id` (lines
70–77) inlined at line 96 (`Y` its default `None`: the compile-time `if Y is None` takes the
`pid % X, pid // X` branch). The pointers `s`, `o` are `*fp32`; `cu_seqlens` and `chunk_indices`
are pointers of the table dtype `idxD` (`int32` or `int64`; `None` on a fixed-length launch, never
read); `scale` is the runtime `float32` scalar (`FloatV.hostF32` of the Python `float`; `None`, never
read, when `HAS_SCALE` is false); `T` is the runtime `int32` (`do_not_specialize=['T']`: never a
compile-time constant). `B`, `H`, `BT` and the flags are `tl.constexpr`, Lean parameters; the
flags' branches are compile-time `if`s, written as Lean `if`s. `USE_GRAPH` is `False` on every
covered launch: `USE_GRAPH and i_n < 0` is the compile-time `False`, lines 100–101 not visited. -/
def chunk_local_cumsum_scalar_kernel (s_ptr o_ptr : Ptr .f32) (idxD : DType)
    (ci_ptr cu_ptr : Ptr idxD) (scale : F32) (T : I32) (B H BT : Nat)
    (REVERSE HAS_SCALE IS_VARLEN : Bool) : Kernel where
  body ctx := do
    -- line 96 (lines 73 and 75): `pid = tl.program_id(0).to(tl.int64)`, then `pid % X, pid // X`
    -- with the compile-time `X = B * H` built in `int64`
    let pid := tl.program_id ctx 0
    let pid := (← tl.convert .i32 .i64 (Tensor.scalar pid)).item
    let i_bh := (← tl.irem (Tensor.scalar pid) (Tensor.scalar (IntV.ofNat (B * H) : I64))).item
    let i_t := (← tl.idiv (Tensor.scalar pid) (Tensor.scalar (IntV.ofNat (B * H) : I64))).item
    -- line 97: `int64` `//` and `%` by the constexpr `H`
    let i_b := (← tl.idiv (Tensor.scalar i_bh) (Tensor.scalar (IntV.ofNat H : I64))).item
    let i_h := (← tl.irem (Tensor.scalar i_bh) (Tensor.scalar (IntV.ofNat H : I64))).item
    -- lines 98–106: the compile-time `if IS_VARLEN:`; each branch gives the chunk index `i_t`,
    -- `bos` and the value line 109 compares with (`T`, as an `int64`)
    let (i_t, bos, T) ← if IS_VARLEN then do
        -- line 99: `chunk_indices[i_t, 0]`, `.to(tl.int32)` (of an `int32` value the identity)
        let i_n := (← tl.convert idxD .i32 (Tensor.scalar
          (← tl.load1 (ci_ptr + i_t * (IntV.ofNat 2 : I64))))).item
        -- line 102: `chunk_indices[i_t, 1]`, `.to(tl.int64)`
        let i_t := (← tl.convert idxD .i64 (Tensor.scalar
          (← tl.load1 (ci_ptr + i_t * (IntV.ofNat 2 : I64) + (IntV.ofNat 1 : I32))))).item
        -- line 103: the sequence's bounds, `.to(tl.int64)`
        let bos := (← tl.convert idxD .i64 (Tensor.scalar (← tl.load1 (cu_ptr + i_n)))).item
        let eos := (← tl.convert idxD .i64
          (Tensor.scalar (← tl.load1 (cu_ptr + i_n + (IntV.ofNat 1 : I32))))).item
        -- line 104: `T = eos - bos`, the wrapping `int64` subtraction
        pure (i_t, bos, eos - bos)
      else do
        -- line 106: `bos, eos = i_b * T, i_b * T + T`: the `int32` `T` promoted to `int64`;
        -- `eos` is not used on this path, and `T` stays the argument, which line 109 promotes to
        -- `int64` the same way
        let T64 := (← tl.convert .i32 .i64 (Tensor.scalar T)).item
        pure (i_t, i_b * T64, T64)
    -- line 108: the `int64` `i_t * BT` (the constexpr built in `int64`) plus `tl.arange(0, BT)`
    -- (`int32`, promoted to `int64`)
    let o_t := bc (i_t * (IntV.ofNat BT : I64)) + (← tl.convert .i32 .i64 (tl.arange 0 BT))
    -- line 109: the signed `int64` comparison
    let m_t := o_t.ltS T
    -- lines 110–111: `int64` products and sums, the pointers advanced in elements
    let p_s := s_ptr + bos * (IntV.ofNat H : I64) + i_h + o_t * (IntV.ofNat H : I64)
    let p_o := o_ptr + bos * (IntV.ofNat H : I64) + i_h + o_t * (IntV.ofNat H : I64)
    -- line 113: the masked load, `other=0.0` the `float32` `+0`; `.to(tl.float32)` is no conversion
    let b_s ← tl.load p_s (some m_t) (some (bc (FloatV.litF32 (FloatV.ofPat 0))))
    -- lines 114–117: `tl.cumsum(b_s, axis=0, reverse=REVERSE)` (the two calls differ only in
    -- `reverse`); IEEE additions, no product feeds the scan
    let b_o ← tl.cumsum b_s 0 REVERSE
    -- lines 118–119: `b_o *= scale`, the IEEE `float32` product with the runtime scalar
    let b_o ← if HAS_SCALE then tl.mul b_o (bc scale) else pure b_o
    -- line 120: the masked store; `.to(p_o.dtype.element_ty)` of `float32` is no conversion
    tl.store p_o b_o (some m_t)

/-! ## The conditions -/

/-- What the host chooses: one launch through `chunk_local_cumsum(g, chunk_size, reverse, scale,
cu_seqlens, output_dtype=torch.float, chunk_indices, use_graph=False)` on a 3-D `g`, with the
contents of its input tensors. -/
structure Inst where
  /-- `IS_VARLEN = cu_seqlens is not None`. -/
  varlen : Bool
  /-- `B = g.shape[0]` (`1` on a variable-length launch). -/
  B : Nat
  /-- `T = g.shape[1]`. -/
  T : Nat
  /-- `H = g.shape[2]`. -/
  H : Nat
  /-- `BT = chunk_size`. -/
  BT : Nat
  /-- `REVERSE = reverse`. -/
  reverse : Bool
  /-- `HAS_SCALE = scale is not None`. -/
  hasScale : Bool
  /-- `scale` (the Python `float`, a double; unused when `hasScale` is false). -/
  scale : F64
  /-- `s = g`, contiguous `[B, T, H]` `float32`: element `(b, t, h)` is `(b·T + t)·H + h`. -/
  s : Buf .f32
  /-- Its contents. -/
  sv : Nat → F32
  /-- `o`, fresh, contiguous `[B, T, H]` `float32`. -/
  o : Buf .f32
  /-- Variable length: the tables' dtype (`cu_seqlens`'s; `chunk_indices` carries it). -/
  idxD : DType
  /-- Variable length: the number of sequences `N` (`cu_seqlens` has `N + 1` entries). -/
  N : Nat
  /-- Variable length: `cu_seqlens` and its entries. -/
  cu : Buf idxD
  cuVal : Nat → idxD.Val
  /-- Variable length: the number of rows `NT = len(chunk_indices)`. -/
  NT : Nat
  /-- Variable length: `chunk_indices` (`[NT, 2]`, contiguous) and its entries. -/
  ci : Buf idxD
  ciVal : Nat → idxD.Val

/-- An integer table entry (`int32`/`int64`) as an integer. -/
def entryInt : (d : DType) → d.Val → Int
  | .i32, x => x.toInt
  | .i64, x => x.toInt
  | _, _ => 0

/-- `cu_seqlens[i]`. -/
def cuAt (d : Inst) (i : Nat) : Int := entryInt d.idxD (d.cuVal i)

/-- `chunk_indices[r, 0]` (the sequence) and `chunk_indices[r, 1]` (the chunk within it). -/
def seqAt (d : Inst) (r : Nat) : Int := entryInt d.idxD (d.ciVal (2 * r))
def chkAt (d : Inst) (r : Nat) : Int := entryInt d.idxD (d.ciVal (2 * r + 1))

/-- The grid size: `NT·B·H` with `NT = len(chunk_indices)` (variable length) or
`triton.cdiv(T, BT)` (fixed length). -/
def gridSize (d : Inst) : Nat :=
  (if d.varlen then d.NT else (d.T + d.BT - 1) / d.BT) * d.B * d.H

/-- The instances covered. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `BT` a power of two (the entry point asserts it; `tl.arange`) of at most `2^20`. -/
  bt_pow2 : ∃ e, d.BT = 2 ^ e
  bt_le : d.BT ≤ 2 ^ 20
  /-- At least one batch row and one head. -/
  B_pos : 0 < d.B
  H_pos : 0 < d.H
  /-- `B·H < 2^31`: the compile-time divisor `B * H` of line 96 is an `int32`-range constant
  (real use has at most a few thousand batch rows times heads). -/
  BH_lt : d.B * d.H < 2 ^ 31
  /-- `T` is an `int32` (`do_not_specialize`: a runtime `int32` below `2^31`). -/
  T_lt : d.T < 2 ^ 31
  /-- Variable length: `B = 1` (the entry point asserts it). -/
  varlen_B : d.varlen = true → d.B = 1
  /-- The grid has fewer than `2^31` programs (CUDA's limit: README, Exclusions). -/
  grid_lt : gridSize d < 2 ^ 31
  /-- The buffers hold the tensors. -/
  s_len : d.B * d.T * d.H ≤ d.s.len
  o_len : d.B * d.T * d.H ≤ d.o.len
  /-- `o` is a fresh allocation: it shares no byte with `s`. -/
  o_s : Disjoint d.o.bytes d.s.bytes
  /-- Variable length: `int32` or `int64` tables, `N < 2^31` (a sequence index is an `int32`
  after line 99's `.to(tl.int32)`), the tables' lengths, and `o` sharing no byte with them. -/
  idx_ok : d.varlen = true → d.idxD = .i32 ∨ d.idxD = .i64
  N_lt : d.varlen = true → d.N < 2 ^ 31
  cu_len : d.varlen = true → d.N + 1 ≤ d.cu.len
  ci_len : d.varlen = true → 2 * d.NT ≤ d.ci.len
  o_cu : d.varlen = true → Disjoint d.o.bytes d.cu.bytes
  o_ci : d.varlen = true → Disjoint d.o.bytes d.ci.bytes

/-- The tables of a variable-length launch, as `prepare_chunk_indices` builds them (or the caller
passes the same table): `cu_seqlens` nonnegative, nondecreasing and at most `T`; each row of
`chunk_indices` names a sequence `n < N` and a chunk `c ≥ 0` that starts inside it
(`c·BT < cu_seqlens[n+1] − cu_seqlens[n]`); no two rows are the same; and every such chunk has a
row. -/
def TablesOk (d : Inst) : Prop :=
  0 ≤ cuAt d 0 ∧ (∀ i, i < d.N → cuAt d i ≤ cuAt d (i + 1)) ∧ cuAt d d.N ≤ d.T ∧
  (∀ r, r < d.NT → 0 ≤ seqAt d r ∧ seqAt d r < d.N ∧ 0 ≤ chkAt d r ∧
    chkAt d r * d.BT < cuAt d ((seqAt d r).toNat + 1) - cuAt d (seqAt d r).toNat) ∧
  (∀ r r', r < d.NT → r' < d.NT → r ≠ r' → (seqAt d r, chkAt d r) ≠ (seqAt d r', chkAt d r')) ∧
  (∀ n c : Nat, n < d.N → (c * d.BT : Int) < cuAt d (n + 1) - cuAt d n →
    ∃ r, r < d.NT ∧ seqAt d r = n ∧ chkAt d r = c)

/-- The inputs covered: the buffers are valid and hold the instance's contents; on a
variable-length launch, the tables are `TablesOk`. Every `float32` value of `s` is covered. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.s.Valid init ∧ d.o.Valid init ∧ d.s.Holds init d.sv ∧
    (d.varlen = true → d.cu.Valid init ∧ d.ci.Valid init ∧ d.cu.Holds init d.cuVal ∧
      d.ci.Holds init d.ciVal ∧ TablesOk d)

/-- The scale's `float32` value, as the kernel receives it (`FloatV.hostF32` of the double). -/
def sigma32 (d : Inst) : F32 := FloatV.hostF32 d.scale

/-- The model on an instance (`USE_GRAPH = False`). -/
def model (d : Inst) : Kernel :=
  chunk_local_cumsum_scalar_kernel (d.s.ptr 0) (d.o.ptr 0) d.idxD (d.ci.ptr 0) (d.cu.ptr 0)
    (sigma32 d) (IntV.ofNat d.T) d.B d.H d.BT d.reverse d.hasScale d.varlen

/-- The host's launch (lines 188–205): the grid `(NT·B·H,)`, the block shape `[BT]`;
`enable_fp_fusion = true` (the default), `debug = false`, `num_ctas = 1`, no `launch_pdl`.
`num_warps = 2` is performance-only; `num_stages = 3` changes no outcome of a model without a
conflict inside a program. -/
def launch (d : Inst) : Launch := Launch.grid1 (gridSize d) [[d.BT]] (fusion := true)

/-! ## The specification -/

/-- The number of sequences: the rows of a fixed-length batch, or `N`. -/
def nSeq (d : Inst) : Nat := if d.varlen then d.N else d.B

/-- Sequence `n`'s positions are `[lo n, hi n)` of the flat time axis (`b·T + t` for row `b`). -/
def lo (d : Inst) (n : Nat) : Nat := if d.varlen then (cuAt d n).toNat else n * d.T
def hi (d : Inst) (n : Nat) : Nat := if d.varlen then (cuAt d (n + 1)).toNat else n * d.T + d.T

/-- The start of position `p`'s chunk, chunks of `BT` counted from the sequence's start. -/
def chunkStart (d : Inst) (n p : Nat) : Nat := lo d n + (p - lo d n) / d.BT * d.BT

/-- The positions summed at `p`: from the chunk's start to `p` (forward), or from `p` to the
chunk's end `min(start + BT, hi n)` (reverse). -/
def rng (d : Inst) (n p : Nat) : Finset Nat :=
  if d.reverse then Finset.Ico p (min (chunkStart d n p + d.BT) (hi d n))
  else Finset.Icc (chunkStart d n p) p

/-- `s[p, h]` (element `p·H + h`). -/
def sAt (d : Inst) (p h : Nat) : F32 := d.sv (p * d.H + h)

/-- The exact chunk-local cumulative sum at `(p, h)`, and the sum of its terms' magnitudes. -/
def refSum (d : Inst) (n p h : Nat) : ℚ := ∑ q ∈ rng d n p, (sAt d q h).toRat
def absSum (d : Inst) (n p h : Nat) : ℚ := ∑ q ∈ rng d n p, |(sAt d q h).toRat|

/-- The additions the scan does along position `p`'s leaves, less one: the scan of a program
combines `j + 1` lanes at lane `j` forward, `BT − j` lanes reverse (the lanes past the sequence's
end hold `+0`), `j = p − chunkStart`. -/
def depth (d : Inst) (n p : Nat) : Nat :=
  if d.reverse then chunkStart d n p + d.BT - 1 - p else p - chunkStart d n p

/-- The scale: the host's double and its `float32` value, or `1` when there is none. -/
def sig (d : Inst) : ℚ := if d.hasScale then d.scale.toRat else 1
def sig32 (d : Inst) : ℚ := if d.hasScale then (sigma32 d).toRat else 1

/-- One rounding of `float32`: `2^-24` relative, `2^-150` absolute (`Format.binary32`). -/
def u32 : ℚ := 2 ^ (-24 : ℤ)
def eta32 : ℚ := 2 ^ (-150 : ℤ)

theorem u32_eq : u32 = Format.binary32.u := by unfold u32; rw [Format.binary32_u]
theorem eta32_eq : eta32 = Format.binary32.eta := by unfold eta32; rw [Format.binary32_eta]

/-- The element is in range: its terms are finite, `Σ|s| ≤ 2^126`, and with a scale the double
and its `float32` value are finite and `|σ₃₂|·Σ|s| ≤ 2^126` (`float32` overflows near `2^128`). -/
def InRange (d : Inst) (n p h : Nat) : Prop :=
  (∀ q ∈ rng d n p, (sAt d q h).isFinite = true) ∧ absSum d n p h ≤ 2 ^ 126 ∧
    (d.hasScale = true → d.scale.isFinite = true ∧ (sigma32 d).isFinite = true ∧
      |(sigma32 d).toRat| * absSum d n p h ≤ 2 ^ 126)

/-- **The tolerance**, from the depth `m`, `S = refSum`, `A = absSum`: the scan's error
`E = Tree.errBound u η m m A` (a tree of `m + 1` leaves, of depth and node count at most `m`,
each node an IEEE addition; `sum_err_list`); with a scale, the `float32` product's rounding
`Em = u·|σ₃₂|·(|S| + E) + η`, the scan's error scaled, `|σ₃₂|·E`, and the scale's own conversion
`|σ − σ₃₂|·|S|` (the reference uses the host's double). -/
def tol (d : Inst) (n p h : Nat) : ℚ :=
  let E := Tree.errBound u32 eta32 (depth d n p) (depth d n p) (absSum d n p h)
  let S := refSum d n p h
  if d.hasScale then
    |sig32 d| * E + (u32 * (|sig32 d| * (|S| + E)) + eta32) + |sig d - sig32 d| * |S|
  else E

/-- The output elements the launch writes: every position of every sequence, every head. -/
def written (d : Inst) : Set Nat :=
  {e | ∃ n p h, n < nSeq d ∧ lo d n ≤ p ∧ p < hi d n ∧ h < d.H ∧ e = p * d.H + h}

/-- What the launch leaves: for every sequence `n`, position `p` in it and head `h`, `o[p, h]`
holds a `float32` value which, when the element is `InRange`, is finite and within `tol` of
`σ · refSum`; every byte outside the written elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ n p h, n < nSeq d → lo d n ≤ p → p < hi d n → h < d.H → ∃ v : F32,
      fin.read .f32 (d.o.ptr (p * d.H + h)).addr = some v ∧
      (InRange d n p h → v.isFinite = true ∧
        |v.toRat - sig d * refSum d n p h| ≤ tol d n p h)) ∧
    ∀ z, z ∉ d.o.elems (written d) → fin z = init z

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## The assumptions -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on every covered instance, when `launch d` has `debug = false`
  and is valid, every outcome of the compiled kernel launched as `launch d`, from a memory from
  which the model's launch is well behaved, is one of the outcomes of the model's launch. The rule
  covers a launch within its scope: part (A), the kernel and its launches, which the host's
  launches meet (README, Conditions), and part (B), the environment, which the claim relies on. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)

/-- **The claim to prove.** -/
def Statement : Prop := ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K

/-! ## Not vacuous -/

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.any model launch
  exact ⟨K, Platform.any, htot, ⟨fun d _ => hf d⟩⟩

/-- An instance of the host's launch with `s` at `2^32`, `o` at `2^36`, `cu_seqlens` at `2^40`,
`chunk_indices` at `2^41`, zero contents: `vl` variable length, `B`, `T`, `H`, `BT = 2^e`, the
flags, the scale's double pattern, the tables' dtype, `N` sequences and `NT` rows. -/
def mk (vl : Bool) (B T H e : Nat) (rev hs : Bool) (sbits : Nat) (dt : DType) (N NT : Nat) :
    Inst where
  varlen := vl
  B := B
  T := T
  H := H
  BT := 2 ^ e
  reverse := rev
  hasScale := hs
  scale := FloatV.ofPat sbits
  s := ⟨2 ^ 32, B * T * H⟩
  sv := fun _ => FloatV.ofPat 0
  o := ⟨2 ^ 36, B * T * H⟩
  idxD := dt
  N := N
  cu := ⟨2 ^ 40, N + 1⟩
  cuVal := fun _ => dt.ofBits 0
  NT := NT
  ci := ⟨2 ^ 41, 2 * NT⟩
  ciVal := fun _ => dt.ofBits 0

theorem bytes_disj {d d' : DType} (b : Buf d) (c : Buf d')
    (h : b.base + b.len * d.bytes ≤ c.base ∨ c.base + c.len * d'.bytes ≤ b.base) :
    Disjoint b.bytes c.bytes := by
  rw [Set.disjoint_left]
  intro x hx hy
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

theorem mk_ok {vl : Bool} {B T H e : Nat} {rev hs : Bool} {sbits : Nat} {dt : DType} {N NT : Nat}
    (he : e ≤ 20) (hB : 0 < B) (hH : 0 < H) (hBH : B * H < 2 ^ 31) (hT : T < 2 ^ 31) (hvB : vl = true → B = 1)
    (hg : gridSize (mk vl B T H e rev hs sbits dt N NT) < 2 ^ 31)
    (hsz : B * T * H * 4 ≤ 2 ^ 35) (hdt : dt = .i32 ∨ dt = .i64) (hN : N < 2 ^ 31)
    (hNsz : (N + 1) * 8 ≤ 2 ^ 39) (hNT : 2 * NT * 8 ≤ 2 ^ 39) :
    (mk vl B T H e rev hs sbits dt N NT).ok := by
  have hbytes : dt.bytes ≤ 8 := by rcases hdt with h | h <;> subst h <;> decide
  have h1 : ((B * T * H : Nat) : Int) * 4 ≤ 2 ^ 35 := by exact_mod_cast hsz
  have h2 : ((N + 1 : Nat) : Int) * dt.bytes ≤ 2 ^ 39 := by
    have : (N + 1) * dt.bytes ≤ (N + 1) * 8 := Nat.mul_le_mul_left _ hbytes
    exact_mod_cast le_trans this hNsz
  have h3 : ((2 * NT : Nat) : Int) * dt.bytes ≤ 2 ^ 39 := by
    have : 2 * NT * dt.bytes ≤ 2 * NT * 8 := Nat.mul_le_mul_left _ hbytes
    exact_mod_cast le_trans this hNT
  refine ⟨⟨e, rfl⟩, Nat.pow_le_pow_right (by norm_num) he, hB, hH, hBH, hT, hvB, hg, le_refl _,
    le_refl _, ?_, fun _ => hdt, fun _ => hN, fun _ => le_refl _, fun _ => le_refl _, fun _ => ?_,
    fun _ => ?_⟩
  · exact bytes_disj _ _ (Or.inr (by simp only [mk, DType.bytes]; push_cast at h1 ⊢; omega))
  · exact bytes_disj _ _ (Or.inl (by simp only [mk, DType.bytes]; push_cast at h1 ⊢; omega))
  · exact bytes_disj _ _ (Or.inl (by simp only [mk, DType.bytes]; push_cast at h1 h3 ⊢; omega))

/-- A memory in which every byte is present (and `0`). -/
def zeroMem : Mem := fun _ => some 0

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  refine ⟨mk false 1 1 1 4 false false 0 .i32 0 0, zeroMem,
    mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide)
      (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num), ?_⟩
  have hb4 : (DType.f32.bytes : Int) = 4 := rfl
  refine ⟨⟨?_, ?_, ?_, ?_⟩, ⟨?_, ?_, ?_, ?_⟩, ?_, ?_⟩
  · simp [mk]
  · simp only [mk, hb4]; norm_num
  · simp only [mk, hb4]; decide
  · intro x _; simp [zeroMem]
  · simp [mk]
  · simp only [mk, hb4]; norm_num
  · simp only [mk, hb4]; decide
  · intro x _; simp [zeroMem]
  · intro i hi
    have hi0 : i = 0 := by simp only [mk] at hi; omega
    subst hi0
    simp only [mk, Mem.read, Mem.Aligned, Buf.ptr, hb4, Nat.cast_zero]
    norm_num
    exact ⟨[0, 0, 0, 0], rfl, rfl⟩
  · intro hc; simp [mk] at hc

/-! ## Coverage of the host's launches

Every launch of the host's test (source lines 267–300): each fixed-length `(B, T, H, chunk_size)`
and each variable-length `(T, H, chunk_size, cu_seqlens)` case, forward and reverse, without a
scale and with `RCP_LN2 = 1.4426950216` (pattern `4609176139934333942`); the variable-length
cases with `int32` tables (the test's) and `int64` tables (`SETUP.md`), `NT` the number of chunks
`prepare_chunk_indices` gives. The contents (and so `TablesOk`) are not part of `Inst.ok`; the
host's tables are checked against `TablesOk` on the GPU (`check_gpu.py`). Names:
`setup_<fixed|varlen>_<shape>_<fwd|rev>_<noscale|scale>[_<i32|i64>]`. -/

def setup_fixed_1x63x1_bt16_fwd_noscale : Inst := mk false 1 63 1 4 false false 0 .i32 0 0
theorem setup_fixed_1x63x1_bt16_fwd_noscale_ok : setup_fixed_1x63x1_bt16_fwd_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_fixed_1x63x1_bt16_fwd_scale : Inst := mk false 1 63 1 4 false true 4609176139934333942 .i32 0 0
theorem setup_fixed_1x63x1_bt16_fwd_scale_ok : setup_fixed_1x63x1_bt16_fwd_scale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_fixed_1x63x1_bt16_rev_noscale : Inst := mk false 1 63 1 4 true false 0 .i32 0 0
theorem setup_fixed_1x63x1_bt16_rev_noscale_ok : setup_fixed_1x63x1_bt16_rev_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_fixed_1x63x1_bt16_rev_scale : Inst := mk false 1 63 1 4 true true 4609176139934333942 .i32 0 0
theorem setup_fixed_1x63x1_bt16_rev_scale_ok : setup_fixed_1x63x1_bt16_rev_scale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_fixed_2x500x4_bt32_fwd_noscale : Inst := mk false 2 500 4 5 false false 0 .i32 0 0
theorem setup_fixed_2x500x4_bt32_fwd_noscale_ok : setup_fixed_2x500x4_bt32_fwd_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_fixed_2x500x4_bt32_fwd_scale : Inst := mk false 2 500 4 5 false true 4609176139934333942 .i32 0 0
theorem setup_fixed_2x500x4_bt32_fwd_scale_ok : setup_fixed_2x500x4_bt32_fwd_scale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_fixed_2x500x4_bt32_rev_noscale : Inst := mk false 2 500 4 5 true false 0 .i32 0 0
theorem setup_fixed_2x500x4_bt32_rev_noscale_ok : setup_fixed_2x500x4_bt32_rev_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_fixed_2x500x4_bt32_rev_scale : Inst := mk false 2 500 4 5 true true 4609176139934333942 .i32 0 0
theorem setup_fixed_2x500x4_bt32_rev_scale_ok : setup_fixed_2x500x4_bt32_rev_scale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_fixed_2x1000x5_bt64_fwd_noscale : Inst := mk false 2 1000 5 6 false false 0 .i32 0 0
theorem setup_fixed_2x1000x5_bt64_fwd_noscale_ok : setup_fixed_2x1000x5_bt64_fwd_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_fixed_2x1000x5_bt64_fwd_scale : Inst := mk false 2 1000 5 6 false true 4609176139934333942 .i32 0 0
theorem setup_fixed_2x1000x5_bt64_fwd_scale_ok : setup_fixed_2x1000x5_bt64_fwd_scale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_fixed_2x1000x5_bt64_rev_noscale : Inst := mk false 2 1000 5 6 true false 0 .i32 0 0
theorem setup_fixed_2x1000x5_bt64_rev_noscale_ok : setup_fixed_2x1000x5_bt64_rev_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_fixed_2x1000x5_bt64_rev_scale : Inst := mk false 2 1000 5 6 true true 4609176139934333942 .i32 0 0
theorem setup_fixed_2x1000x5_bt64_rev_scale_ok : setup_fixed_2x1000x5_bt64_rev_scale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_fixed_3x1024x6_bt64_fwd_noscale : Inst := mk false 3 1024 6 6 false false 0 .i32 0 0
theorem setup_fixed_3x1024x6_bt64_fwd_noscale_ok : setup_fixed_3x1024x6_bt64_fwd_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_fixed_3x1024x6_bt64_fwd_scale : Inst := mk false 3 1024 6 6 false true 4609176139934333942 .i32 0 0
theorem setup_fixed_3x1024x6_bt64_fwd_scale_ok : setup_fixed_3x1024x6_bt64_fwd_scale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_fixed_3x1024x6_bt64_rev_noscale : Inst := mk false 3 1024 6 6 true false 0 .i32 0 0
theorem setup_fixed_3x1024x6_bt64_rev_noscale_ok : setup_fixed_3x1024x6_bt64_rev_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_fixed_3x1024x6_bt64_rev_scale : Inst := mk false 3 1024 6 6 true true 4609176139934333942 .i32 0 0
theorem setup_fixed_3x1024x6_bt64_rev_scale_ok : setup_fixed_3x1024x6_bt64_rev_scale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_fixed_4x2048x8_bt128_fwd_noscale : Inst := mk false 4 2048 8 7 false false 0 .i32 0 0
theorem setup_fixed_4x2048x8_bt128_fwd_noscale_ok : setup_fixed_4x2048x8_bt128_fwd_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_fixed_4x2048x8_bt128_fwd_scale : Inst := mk false 4 2048 8 7 false true 4609176139934333942 .i32 0 0
theorem setup_fixed_4x2048x8_bt128_fwd_scale_ok : setup_fixed_4x2048x8_bt128_fwd_scale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_fixed_4x2048x8_bt128_rev_noscale : Inst := mk false 4 2048 8 7 true false 0 .i32 0 0
theorem setup_fixed_4x2048x8_bt128_rev_noscale_ok : setup_fixed_4x2048x8_bt128_rev_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_fixed_4x2048x8_bt128_rev_scale : Inst := mk false 4 2048 8 7 true true 4609176139934333942 .i32 0 0
theorem setup_fixed_4x2048x8_bt128_rev_scale_ok : setup_fixed_4x2048x8_bt128_rev_scale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_15x2_bt16_n1_fwd_noscale_i32 : Inst := mk true 1 15 2 4 false false 0 .i32 1 1
theorem setup_varlen_15x2_bt16_n1_fwd_noscale_i32_ok : setup_varlen_15x2_bt16_n1_fwd_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_15x2_bt16_n1_fwd_noscale_i64 : Inst := mk true 1 15 2 4 false false 0 .i64 1 1
theorem setup_varlen_15x2_bt16_n1_fwd_noscale_i64_ok : setup_varlen_15x2_bt16_n1_fwd_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_15x2_bt16_n1_fwd_scale_i32 : Inst := mk true 1 15 2 4 false true 4609176139934333942 .i32 1 1
theorem setup_varlen_15x2_bt16_n1_fwd_scale_i32_ok : setup_varlen_15x2_bt16_n1_fwd_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_15x2_bt16_n1_fwd_scale_i64 : Inst := mk true 1 15 2 4 false true 4609176139934333942 .i64 1 1
theorem setup_varlen_15x2_bt16_n1_fwd_scale_i64_ok : setup_varlen_15x2_bt16_n1_fwd_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_15x2_bt16_n1_rev_noscale_i32 : Inst := mk true 1 15 2 4 true false 0 .i32 1 1
theorem setup_varlen_15x2_bt16_n1_rev_noscale_i32_ok : setup_varlen_15x2_bt16_n1_rev_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_15x2_bt16_n1_rev_noscale_i64 : Inst := mk true 1 15 2 4 true false 0 .i64 1 1
theorem setup_varlen_15x2_bt16_n1_rev_noscale_i64_ok : setup_varlen_15x2_bt16_n1_rev_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_15x2_bt16_n1_rev_scale_i32 : Inst := mk true 1 15 2 4 true true 4609176139934333942 .i32 1 1
theorem setup_varlen_15x2_bt16_n1_rev_scale_i32_ok : setup_varlen_15x2_bt16_n1_rev_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_15x2_bt16_n1_rev_scale_i64 : Inst := mk true 1 15 2 4 true true 4609176139934333942 .i64 1 1
theorem setup_varlen_15x2_bt16_n1_rev_scale_i64_ok : setup_varlen_15x2_bt16_n1_rev_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_15x2_bt32_n1_fwd_noscale_i32 : Inst := mk true 1 15 2 5 false false 0 .i32 1 1
theorem setup_varlen_15x2_bt32_n1_fwd_noscale_i32_ok : setup_varlen_15x2_bt32_n1_fwd_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_15x2_bt32_n1_fwd_noscale_i64 : Inst := mk true 1 15 2 5 false false 0 .i64 1 1
theorem setup_varlen_15x2_bt32_n1_fwd_noscale_i64_ok : setup_varlen_15x2_bt32_n1_fwd_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_15x2_bt32_n1_fwd_scale_i32 : Inst := mk true 1 15 2 5 false true 4609176139934333942 .i32 1 1
theorem setup_varlen_15x2_bt32_n1_fwd_scale_i32_ok : setup_varlen_15x2_bt32_n1_fwd_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_15x2_bt32_n1_fwd_scale_i64 : Inst := mk true 1 15 2 5 false true 4609176139934333942 .i64 1 1
theorem setup_varlen_15x2_bt32_n1_fwd_scale_i64_ok : setup_varlen_15x2_bt32_n1_fwd_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_15x2_bt32_n1_rev_noscale_i32 : Inst := mk true 1 15 2 5 true false 0 .i32 1 1
theorem setup_varlen_15x2_bt32_n1_rev_noscale_i32_ok : setup_varlen_15x2_bt32_n1_rev_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_15x2_bt32_n1_rev_noscale_i64 : Inst := mk true 1 15 2 5 true false 0 .i64 1 1
theorem setup_varlen_15x2_bt32_n1_rev_noscale_i64_ok : setup_varlen_15x2_bt32_n1_rev_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_15x2_bt32_n1_rev_scale_i32 : Inst := mk true 1 15 2 5 true true 4609176139934333942 .i32 1 1
theorem setup_varlen_15x2_bt32_n1_rev_scale_i32_ok : setup_varlen_15x2_bt32_n1_rev_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_15x2_bt32_n1_rev_scale_i64 : Inst := mk true 1 15 2 5 true true 4609176139934333942 .i64 1 1
theorem setup_varlen_15x2_bt32_n1_rev_scale_i64_ok : setup_varlen_15x2_bt32_n1_rev_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_1000x3_bt64_n3_fwd_noscale_i32 : Inst := mk true 1 1000 3 6 false false 0 .i32 3 16
theorem setup_varlen_1000x3_bt64_n3_fwd_noscale_i32_ok : setup_varlen_1000x3_bt64_n3_fwd_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_1000x3_bt64_n3_fwd_noscale_i64 : Inst := mk true 1 1000 3 6 false false 0 .i64 3 16
theorem setup_varlen_1000x3_bt64_n3_fwd_noscale_i64_ok : setup_varlen_1000x3_bt64_n3_fwd_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_1000x3_bt64_n3_fwd_scale_i32 : Inst := mk true 1 1000 3 6 false true 4609176139934333942 .i32 3 16
theorem setup_varlen_1000x3_bt64_n3_fwd_scale_i32_ok : setup_varlen_1000x3_bt64_n3_fwd_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_1000x3_bt64_n3_fwd_scale_i64 : Inst := mk true 1 1000 3 6 false true 4609176139934333942 .i64 3 16
theorem setup_varlen_1000x3_bt64_n3_fwd_scale_i64_ok : setup_varlen_1000x3_bt64_n3_fwd_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_1000x3_bt64_n3_rev_noscale_i32 : Inst := mk true 1 1000 3 6 true false 0 .i32 3 16
theorem setup_varlen_1000x3_bt64_n3_rev_noscale_i32_ok : setup_varlen_1000x3_bt64_n3_rev_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_1000x3_bt64_n3_rev_noscale_i64 : Inst := mk true 1 1000 3 6 true false 0 .i64 3 16
theorem setup_varlen_1000x3_bt64_n3_rev_noscale_i64_ok : setup_varlen_1000x3_bt64_n3_rev_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_1000x3_bt64_n3_rev_scale_i32 : Inst := mk true 1 1000 3 6 true true 4609176139934333942 .i32 3 16
theorem setup_varlen_1000x3_bt64_n3_rev_scale_i32_ok : setup_varlen_1000x3_bt64_n3_rev_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_1000x3_bt64_n3_rev_scale_i64 : Inst := mk true 1 1000 3 6 true true 4609176139934333942 .i64 3 16
theorem setup_varlen_1000x3_bt64_n3_rev_scale_i64_ok : setup_varlen_1000x3_bt64_n3_rev_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_2000x4_bt64_n5_fwd_noscale_i32 : Inst := mk true 1 2000 4 6 false false 0 .i32 5 35
theorem setup_varlen_2000x4_bt64_n5_fwd_noscale_i32_ok : setup_varlen_2000x4_bt64_n5_fwd_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_2000x4_bt64_n5_fwd_noscale_i64 : Inst := mk true 1 2000 4 6 false false 0 .i64 5 35
theorem setup_varlen_2000x4_bt64_n5_fwd_noscale_i64_ok : setup_varlen_2000x4_bt64_n5_fwd_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_2000x4_bt64_n5_fwd_scale_i32 : Inst := mk true 1 2000 4 6 false true 4609176139934333942 .i32 5 35
theorem setup_varlen_2000x4_bt64_n5_fwd_scale_i32_ok : setup_varlen_2000x4_bt64_n5_fwd_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_2000x4_bt64_n5_fwd_scale_i64 : Inst := mk true 1 2000 4 6 false true 4609176139934333942 .i64 5 35
theorem setup_varlen_2000x4_bt64_n5_fwd_scale_i64_ok : setup_varlen_2000x4_bt64_n5_fwd_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_2000x4_bt64_n5_rev_noscale_i32 : Inst := mk true 1 2000 4 6 true false 0 .i32 5 35
theorem setup_varlen_2000x4_bt64_n5_rev_noscale_i32_ok : setup_varlen_2000x4_bt64_n5_rev_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_2000x4_bt64_n5_rev_noscale_i64 : Inst := mk true 1 2000 4 6 true false 0 .i64 5 35
theorem setup_varlen_2000x4_bt64_n5_rev_noscale_i64_ok : setup_varlen_2000x4_bt64_n5_rev_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_2000x4_bt64_n5_rev_scale_i32 : Inst := mk true 1 2000 4 6 true true 4609176139934333942 .i32 5 35
theorem setup_varlen_2000x4_bt64_n5_rev_scale_i32_ok : setup_varlen_2000x4_bt64_n5_rev_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_2000x4_bt64_n5_rev_scale_i64 : Inst := mk true 1 2000 4 6 true true 4609176139934333942 .i64 5 35
theorem setup_varlen_2000x4_bt64_n5_rev_scale_i64_ok : setup_varlen_2000x4_bt64_n5_rev_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_2048x4_bt128_n5_fwd_noscale_i32 : Inst := mk true 1 2048 4 7 false false 0 .i32 5 19
theorem setup_varlen_2048x4_bt128_n5_fwd_noscale_i32_ok : setup_varlen_2048x4_bt128_n5_fwd_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_2048x4_bt128_n5_fwd_noscale_i64 : Inst := mk true 1 2048 4 7 false false 0 .i64 5 19
theorem setup_varlen_2048x4_bt128_n5_fwd_noscale_i64_ok : setup_varlen_2048x4_bt128_n5_fwd_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_2048x4_bt128_n5_fwd_scale_i32 : Inst := mk true 1 2048 4 7 false true 4609176139934333942 .i32 5 19
theorem setup_varlen_2048x4_bt128_n5_fwd_scale_i32_ok : setup_varlen_2048x4_bt128_n5_fwd_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_2048x4_bt128_n5_fwd_scale_i64 : Inst := mk true 1 2048 4 7 false true 4609176139934333942 .i64 5 19
theorem setup_varlen_2048x4_bt128_n5_fwd_scale_i64_ok : setup_varlen_2048x4_bt128_n5_fwd_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_2048x4_bt128_n5_rev_noscale_i32 : Inst := mk true 1 2048 4 7 true false 0 .i32 5 19
theorem setup_varlen_2048x4_bt128_n5_rev_noscale_i32_ok : setup_varlen_2048x4_bt128_n5_rev_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_2048x4_bt128_n5_rev_noscale_i64 : Inst := mk true 1 2048 4 7 true false 0 .i64 5 19
theorem setup_varlen_2048x4_bt128_n5_rev_noscale_i64_ok : setup_varlen_2048x4_bt128_n5_rev_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_2048x4_bt128_n5_rev_scale_i32 : Inst := mk true 1 2048 4 7 true true 4609176139934333942 .i32 5 19
theorem setup_varlen_2048x4_bt128_n5_rev_scale_i32_ok : setup_varlen_2048x4_bt128_n5_rev_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_2048x4_bt128_n5_rev_scale_i64 : Inst := mk true 1 2048 4 7 true true 4609176139934333942 .i64 5 19
theorem setup_varlen_2048x4_bt128_n5_rev_scale_i64_ok : setup_varlen_2048x4_bt128_n5_rev_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_1x32_bt64_n1_fwd_noscale_i32 : Inst := mk true 1 1 32 6 false false 0 .i32 1 1
theorem setup_varlen_1x32_bt64_n1_fwd_noscale_i32_ok : setup_varlen_1x32_bt64_n1_fwd_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_1x32_bt64_n1_fwd_noscale_i64 : Inst := mk true 1 1 32 6 false false 0 .i64 1 1
theorem setup_varlen_1x32_bt64_n1_fwd_noscale_i64_ok : setup_varlen_1x32_bt64_n1_fwd_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_1x32_bt64_n1_fwd_scale_i32 : Inst := mk true 1 1 32 6 false true 4609176139934333942 .i32 1 1
theorem setup_varlen_1x32_bt64_n1_fwd_scale_i32_ok : setup_varlen_1x32_bt64_n1_fwd_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_1x32_bt64_n1_fwd_scale_i64 : Inst := mk true 1 1 32 6 false true 4609176139934333942 .i64 1 1
theorem setup_varlen_1x32_bt64_n1_fwd_scale_i64_ok : setup_varlen_1x32_bt64_n1_fwd_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_1x32_bt64_n1_rev_noscale_i32 : Inst := mk true 1 1 32 6 true false 0 .i32 1 1
theorem setup_varlen_1x32_bt64_n1_rev_noscale_i32_ok : setup_varlen_1x32_bt64_n1_rev_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_1x32_bt64_n1_rev_noscale_i64 : Inst := mk true 1 1 32 6 true false 0 .i64 1 1
theorem setup_varlen_1x32_bt64_n1_rev_noscale_i64_ok : setup_varlen_1x32_bt64_n1_rev_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_1x32_bt64_n1_rev_scale_i32 : Inst := mk true 1 1 32 6 true true 4609176139934333942 .i32 1 1
theorem setup_varlen_1x32_bt64_n1_rev_scale_i32_ok : setup_varlen_1x32_bt64_n1_rev_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_1x32_bt64_n1_rev_scale_i64 : Inst := mk true 1 1 32 6 true true 4609176139934333942 .i64 1 1
theorem setup_varlen_1x32_bt64_n1_rev_scale_i64_ok : setup_varlen_1x32_bt64_n1_rev_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_65536x32_bt64_n6_fwd_noscale_i32 : Inst := mk true 1 65536 32 6 false false 0 .i32 6 1027
theorem setup_varlen_65536x32_bt64_n6_fwd_noscale_i32_ok : setup_varlen_65536x32_bt64_n6_fwd_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_65536x32_bt64_n6_fwd_noscale_i64 : Inst := mk true 1 65536 32 6 false false 0 .i64 6 1027
theorem setup_varlen_65536x32_bt64_n6_fwd_noscale_i64_ok : setup_varlen_65536x32_bt64_n6_fwd_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_65536x32_bt64_n6_fwd_scale_i32 : Inst := mk true 1 65536 32 6 false true 4609176139934333942 .i32 6 1027
theorem setup_varlen_65536x32_bt64_n6_fwd_scale_i32_ok : setup_varlen_65536x32_bt64_n6_fwd_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_65536x32_bt64_n6_fwd_scale_i64 : Inst := mk true 1 65536 32 6 false true 4609176139934333942 .i64 6 1027
theorem setup_varlen_65536x32_bt64_n6_fwd_scale_i64_ok : setup_varlen_65536x32_bt64_n6_fwd_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_65536x32_bt64_n6_rev_noscale_i32 : Inst := mk true 1 65536 32 6 true false 0 .i32 6 1027
theorem setup_varlen_65536x32_bt64_n6_rev_noscale_i32_ok : setup_varlen_65536x32_bt64_n6_rev_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_65536x32_bt64_n6_rev_noscale_i64 : Inst := mk true 1 65536 32 6 true false 0 .i64 6 1027
theorem setup_varlen_65536x32_bt64_n6_rev_noscale_i64_ok : setup_varlen_65536x32_bt64_n6_rev_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_65536x32_bt64_n6_rev_scale_i32 : Inst := mk true 1 65536 32 6 true true 4609176139934333942 .i32 6 1027
theorem setup_varlen_65536x32_bt64_n6_rev_scale_i32_ok : setup_varlen_65536x32_bt64_n6_rev_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
def setup_varlen_65536x32_bt64_n6_rev_scale_i64 : Inst := mk true 1 65536 32 6 true true 4609176139934333942 .i64 6 1027
theorem setup_varlen_65536x32_bt64_n6_rev_scale_i64_ok : setup_varlen_65536x32_bt64_n6_rev_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by simp) (by decide) (by norm_num) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num)

end Claims.ChunkCumsum

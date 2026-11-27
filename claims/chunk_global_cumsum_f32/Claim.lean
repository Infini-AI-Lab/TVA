import TVA

/-!
# `chunk_global_cumsum_f32`: global cumulative sum, fla

The kernel is `chunk_global_cumsum_scalar_kernel` of `source/chunk_global_cumsum.py` (lines
60–101); it reads the module-level names `triton` and `tl` (lines 56–57). The entry point
`chunk_global_cumsum` (lines 142–173) launches it through `chunk_global_cumsum_scalar` (lines
104–139) on a `float32` gate `s` (`[B, T, H]`) and a fresh `float32` output `z`, over fixed-length
rows (`cu_seqlens is None`) or a packed variable-length batch (`B = 1`, `cu_seqlens` `int32` or
`int64`); one program per (sequence, head), which walks the sequence in blocks of `BT` positions
(the last block first when reversed) and carries the running sum of the blocks it has done.
`SETUP.md` lists the launches.

What is claimed: every covered launch ends normally; for every sequence (a row of a fixed-length
batch, or a sequence `[cu_seqlens[n], cu_seqlens[n+1])`), every position `p` in it and every head
`h`, the output element `z[p, h]` holds a `float32` value which, when its inputs are finite and the
magnitudes leave room (`InRange`), is finite and within `tol` of `σ · Σ s[q, h]` over `q` from the
sequence's start to `p` (forward), or from `p` to the sequence's end (reverse) (`σ` the host's
double scale, or `1`); every other byte is unchanged.

The conditions: `BT` is a power of two at most `2^20`; `B, H ≥ 1`, `H ≤ 2^17`, `T < 2^31`,
`T ≤ 2^22`, `B ≤ 8192`; `s` and `z` hold `B·T·H` elements and `z` shares no byte with `s`. On a
variable-length launch: `B = 1`, an `int32` or `int64` table of `N + 1` entries, `N ≤ 8192`
sequences, the table disjoint from `z`, and its contents nonnegative, nondecreasing and at most
`T` (`TablesOk`). Every `float32` gate is covered.

The float operations are `tl.cumsum`'s and `tl.sum`'s IEEE additions, two IEEE `+` and an IEEE
`*`; the tolerance rests on one device fact, that `tl.sum`'s tree has depth at most `⌈log₂ BT⌉`
(`Device.SumDepth`).
-/

namespace Claims.ChunkGlobalCumsum

open TVA

/-! ## The model -/

set_option linter.unusedVariables false in
/-- `chunk_global_cumsum_scalar_kernel` (source lines 60–101). The pointers `s`, `o` are `*fp32`;
`cu_seqlens` is a pointer of the table dtype `idxD` (`int32` or `int64`; `None` on a fixed-length
launch, never read); `scale` is the runtime `float32` scalar (`FloatV.hostF32` of the Python
`float`; `None`, never read, when `HAS_SCALE` is false); `T` is the runtime `int32`
(`do_not_specialize=['T']`: never a compile-time constant). `B`, `H`, `BT` and the flags are
`tl.constexpr`, Lean parameters; the flags' branches are compile-time `if`s, written as Lean
`if`s. -/
def chunk_global_cumsum_scalar_kernel (s_ptr o_ptr : Ptr .f32) (idxD : DType) (cu_ptr : Ptr idxD)
    (scale : F32) (T : I32) (B H BT : Nat) (REVERSE HAS_SCALE IS_VARLEN : Bool) : Kernel where
  body ctx := do
    -- line 74: `tl.program_id(0).to(tl.int64)`
    let i_nh := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id ctx 0))).item
    -- line 75: `int64` `//` and `%` by the constexpr `H` (built in `int64`)
    let i_n := (← tl.idiv (Tensor.scalar i_nh) (Tensor.scalar (IntV.ofNat H : I64))).item
    let i_h := (← tl.irem (Tensor.scalar i_nh) (Tensor.scalar (IntV.ofNat H : I64))).item
    -- lines 76–79: the compile-time `if IS_VARLEN:`
    let (bos, eos) ← if IS_VARLEN then do
        -- line 77: the sequence's bounds `cu_seqlens[i_n]`, `cu_seqlens[i_n + 1]`, `.to(tl.int64)`
        let bos := (← tl.convert idxD .i64 (Tensor.scalar (← tl.load1 (cu_ptr + i_n)))).item
        let eos := (← tl.convert idxD .i64
          (Tensor.scalar (← tl.load1 (cu_ptr + i_n + (IntV.ofNat 1 : I32))))).item
        pure (bos, eos)
      else do
        -- line 79: `bos, eos = i_n * T, i_n * T + T`: the `int32` `T` promoted to `int64`
        let T64 := (← tl.convert .i32 .i64 (Tensor.scalar T)).item
        pure (i_n * T64, i_n * T64 + T64)
    -- line 80: `T = eos - bos`, the wrapping `int64` subtraction (it replaces the argument)
    let T := eos - bos
    -- line 82: `tl.zeros([], dtype=tl.float32)`, the `float32` `+0`
    let b_z : F32 := (tl.zeros [] .f32).item
    -- line 83: `tl.cdiv(T, BT)`, Triton's `(T + (BT - 1)) // BT` with `BT - 1` a compile-time value
    -- built in `int64`: the wrapping addition, then the truncating `int64` `//`
    let NT := (← tl.idiv (Tensor.scalar (T + (IntV.ofNat (BT - 1) : I64)))
      (Tensor.scalar (IntV.ofNat BT : I64))).item
    -- line 84: `for i_c in range(NT)` over the `int64` `NT` (a plain name), the `int64` variable
    -- `i_c`; the loop carries `b_z`, the only name defined before it that the body assigns
    let _b_z ← tl.forRange (IntV.ofInt 0 : I64) NT (IntV.ofInt 1 : I64) b_z fun i_c b_z => do
      -- line 85: `NT - 1 - i_c if REVERSE else i_c`, a compile-time choice; the `1` is built in
      -- `int64`, the subtractions wrap
      let i_t := if REVERSE then NT - IntV.ofInt 1 - i_c else i_c
      -- line 86: the `int64` `i_t * BT` (the constexpr built in `int64`) plus `tl.arange(0, BT)`
      -- (`int32`, promoted to `int64`)
      let o_t := bc (i_t * (IntV.ofNat BT : I64)) + (← tl.convert .i32 .i64 (tl.arange 0 BT))
      -- line 87: the signed `int64` comparison
      let m_t := o_t.ltS T
      -- lines 88–89: `int64` products and sums, the pointers advanced in elements
      let p_s := s_ptr + bos * (IntV.ofNat H : I64) + i_h + o_t * (IntV.ofNat H : I64)
      let p_o := o_ptr + bos * (IntV.ofNat H : I64) + i_h + o_t * (IntV.ofNat H : I64)
      -- line 90: the masked load, `other=0.0` the `float32` `+0`; `.to(tl.float32)` is no
      -- conversion
      let b_s ← tl.load p_s (some m_t) (some (bc (FloatV.litF32 (FloatV.ofPat 0))))
      -- lines 91–94: `tl.cumsum(b_s, axis=0, reverse=REVERSE)` (the two calls differ only in
      -- `reverse`); IEEE additions, no product feeds the scan
      let b_o ← tl.cumsum b_s 0 REVERSE
      -- line 95: `tl.sum(b_s, 0)` of the 1-D block, a scalar; IEEE additions
      let b_ss ← tl.sum b_s
      -- line 96: `b_o += b_z`, the IEEE `+` with the scalar broadcast
      let b_o ← tl.add b_o (bc b_z)
      -- lines 97–98: a runtime `if` on the `int64` comparison `i_c >= 0` (the Python `0` an
      -- `int32` scalar, sign-extended); `b_z += b_ss`, the IEEE `+` of two scalars
      let b_z ← if IntV.le (IntV.ofInt 0 : I64) i_c then do
          pure (← tl.add (Tensor.scalar b_z) (Tensor.scalar b_ss)).item
        else pure b_z
      -- lines 99–100: `b_o *= scale`, the IEEE `float32` product with the runtime scalar
      let b_o ← if HAS_SCALE then tl.mul b_o (bc scale) else pure b_o
      -- line 101: the masked store; `.to(p_o.dtype.element_ty)` of `float32` is no conversion
      tl.store p_o b_o (some m_t)
      pure b_z
    pure ()

/-! ## The conditions -/

/-- What the host chooses: one launch through `chunk_global_cumsum(s, reverse, cu_seqlens, scale,
output_dtype=torch.float)` on a 3-D `s`, with the contents of its input tensors; `BT` is the
block size of the configuration (`256` in the source's host, any entry of the removed autotune
list in fla's). -/
structure Inst where
  /-- `IS_VARLEN = cu_seqlens is not None`. -/
  varlen : Bool
  /-- `B = s.shape[0]` (`1` on a variable-length launch). -/
  B : Nat
  /-- `T = s.shape[1]`. -/
  T : Nat
  /-- `H = s.shape[2]`. -/
  H : Nat
  /-- `BT`, the configuration's block size. -/
  BT : Nat
  /-- `REVERSE = reverse`. -/
  reverse : Bool
  /-- `HAS_SCALE = scale is not None`. -/
  hasScale : Bool
  /-- `scale` (the Python `float`, a double; unused when `hasScale` is false). -/
  scale : F64
  /-- `s`, contiguous `[B, T, H]` `float32`: element `(b, t, h)` is `(b·T + t)·H + h`. -/
  s : Buf .f32
  /-- Its contents. -/
  sv : Nat → F32
  /-- `o = z`, fresh, contiguous `[B, T, H]` `float32`. -/
  o : Buf .f32
  /-- Variable length: the dtype of `cu_seqlens`. -/
  idxD : DType
  /-- Variable length: the number of sequences `N = len(cu_seqlens) - 1`. -/
  N : Nat
  /-- Variable length: `cu_seqlens` and its entries. -/
  cu : Buf idxD
  cuVal : Nat → idxD.Val

/-- An integer table entry (`int32`/`int64`) as an integer. -/
def entryInt : (d : DType) → d.Val → Int
  | .i32, x => x.toInt
  | .i64, x => x.toInt
  | _, _ => 0

/-- `cu_seqlens[i]`. -/
def cuAt (d : Inst) (i : Nat) : Int := entryInt d.idxD (d.cuVal i)

/-- The number of sequences: the rows of a fixed-length batch, or `N`. -/
def nSeq (d : Inst) : Nat := if d.varlen then d.N else d.B

/-- The grid size: `N·H` (line 123), `N = B` on a fixed-length launch. -/
def gridSize (d : Inst) : Nat := nSeq d * d.H

/-- The instances covered. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `BT` a power of two (`tl.arange`) of at most `2^20`. -/
  bt_pow2 : ∃ e, d.BT = 2 ^ e
  bt_le : d.BT ≤ 2 ^ 20
  /-- At least one batch row and one head. -/
  B_pos : 0 < d.B
  H_pos : 0 < d.H
  /-- `H`, the heads, is a model dimension: at most `2^17`. -/
  H_le : d.H ≤ 2 ^ 17
  /-- `T` is an `int32` (`do_not_specialize`: a runtime `int32` below `2^31`). -/
  T_lt : d.T < 2 ^ 31
  /-- `T` is a context length (the tokens of a row, or of the packed batch): at most `2^22`. -/
  T_le : d.T ≤ 2 ^ 22
  /-- `B`, the batch rows, is a batch size: at most `8192`. -/
  B_le : d.B ≤ 8192
  /-- Variable length: `B = 1` (the entry point asserts it). -/
  varlen_B : d.varlen = true → d.B = 1
  /-- The buffers hold the tensors. -/
  s_len : d.B * d.T * d.H ≤ d.s.len
  o_len : d.B * d.T * d.H ≤ d.o.len
  /-- `o` is a fresh allocation: it shares no byte with `s`. -/
  o_s : Disjoint d.o.bytes d.s.bytes
  /-- Variable length: an `int32` or `int64` table, at most `8192` sequences (a batch size), the
  table's length, and `o` sharing no byte with it. -/
  idx_ok : d.varlen = true → d.idxD = .i32 ∨ d.idxD = .i64
  N_le : d.varlen = true → d.N ≤ 8192
  cu_len : d.varlen = true → d.N + 1 ≤ d.cu.len
  o_cu : d.varlen = true → Disjoint d.o.bytes d.cu.bytes

/-- The table of a variable-length launch: `cu_seqlens` nonnegative, nondecreasing and at most
`T`. -/
def TablesOk (d : Inst) : Prop :=
  0 ≤ cuAt d 0 ∧ (∀ i, i < d.N → cuAt d i ≤ cuAt d (i + 1)) ∧ cuAt d d.N ≤ d.T

/-- The inputs covered: the buffers are valid and hold the instance's contents; on a
variable-length launch, the table is `TablesOk`. Every `float32` value of `s` is covered. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.s.Valid init ∧ d.o.Valid init ∧ d.s.Holds init d.sv ∧
    (d.varlen = true → d.cu.Valid init ∧ d.cu.Holds init d.cuVal ∧ TablesOk d)

/-- The scale's `float32` value, as the kernel receives it (`FloatV.hostF32` of the double). -/
def sigma32 (d : Inst) : F32 := FloatV.hostF32 d.scale

/-- The model on an instance. -/
def model (d : Inst) : Kernel :=
  chunk_global_cumsum_scalar_kernel (d.s.ptr 0) (d.o.ptr 0) d.idxD (d.cu.ptr 0) (sigma32 d)
    (IntV.ofNat d.T) d.B d.H d.BT d.reverse d.hasScale d.varlen

/-- The host's launch (lines 123–138): the grid `(N·H,)`, the block shape `[BT]`;
`enable_fp_fusion = true` (the default), `debug = false`, `num_ctas = 1`, no `launch_pdl`.
`num_warps = 8` is performance-only; `num_stages = 3` changes no outcome of a model without a
conflict inside a program. -/
def launch (d : Inst) : Launch := Launch.grid1 (gridSize d) [[d.BT]] (fusion := true)

/-! ## The specification -/

/-- Sequence `n`'s positions are `[lo n, hi n)` of the flat time axis (`b·T + t` for row `b`). -/
def lo (d : Inst) (n : Nat) : Nat := if d.varlen then (cuAt d n).toNat else n * d.T
def hi (d : Inst) (n : Nat) : Nat := if d.varlen then (cuAt d (n + 1)).toNat else n * d.T + d.T

/-- The positions summed at `p`: from the sequence's start to `p` (forward), or from `p` to the
sequence's end (reverse). -/
def rng (d : Inst) (n p : Nat) : Finset Nat :=
  if d.reverse then Finset.Ico p (hi d n) else Finset.Icc (lo d n) p

/-- `s[p, h]` (element `p·H + h`). -/
def sAt (d : Inst) (p h : Nat) : F32 := d.sv (p * d.H + h)

/-- The exact cumulative sum at `(p, h)`, and the sum of its terms' magnitudes. -/
def refSum (d : Inst) (n p h : Nat) : ℚ := ∑ q ∈ rng d n p, (sAt d q h).toRat
def absSum (d : Inst) (n p h : Nat) : ℚ := ∑ q ∈ rng d n p, |(sAt d q h).toRat|

/-- The number of blocks of `BT` positions of sequence `n`, `cdiv(hi − lo, BT)`, and the block
holding position `p`, counted from the sequence's start. -/
def nBlk (d : Inst) (n : Nat) : Nat := (hi d n - lo d n + d.BT - 1) / d.BT
def blk (d : Inst) (n p : Nat) : Nat := (p - lo d n) / d.BT

/-- The start of position `p`'s block. -/
def blkStart (d : Inst) (n p : Nat) : Nat := lo d n + blk d n p * d.BT

/-- The additions the scan does along position `p`'s leaves within its block, at most: the scan
of a block combines `j + 1` lanes at lane `j` forward, `BT − j` lanes reverse (the lanes past the
sequence's end hold `+0`), `j = p − blkStart`. -/
def laneDepth (d : Inst) (n p : Nat) : Nat :=
  if d.reverse then blkStart d n p + d.BT - 1 - p else p - blkStart d n p

/-- The blocks the program finishes before `p`'s: the earlier ones forward, the later ones
reverse. -/
def before (d : Inst) (n p : Nat) : Nat :=
  if d.reverse then nBlk d n - 1 - blk d n p else blk d n p

/-- The additions along the deepest path to the running sum the program adds at `p`'s block: none
before the first block (it is `+0`); after `k ≥ 1` blocks, `k` additions into it, the first after a
`tl.sum` of depth at most `⌈log₂ BT⌉` (`Device.SumDepth`). -/
def carryDepth (d : Inst) (n p : Nat) : Nat :=
  if before d n p = 0 then 0 else Nat.clog 2 d.BT + before d n p

/-- The additions along the deepest path to the output at `p`: the deeper of the block's scan and
the running sum, then the one `b_o + b_z`. -/
def depth (d : Inst) (n p : Nat) : Nat := max (laneDepth d n p) (carryDepth d n p) + 1

/-- The scale: the host's double and its `float32` value, or `1` when there is none. -/
def sig (d : Inst) : ℚ := if d.hasScale then d.scale.toRat else 1
def sig32 (d : Inst) : ℚ := if d.hasScale then (sigma32 d).toRat else 1

/-- The element is in range: its terms are finite, `Σ|s| ≤ 2^126`, and with a scale the double
and its `float32` value are finite and `|σ₃₂|·Σ|s| ≤ 2^126` (`float32` overflows near `2^128`). -/
def InRange (d : Inst) (n p h : Nat) : Prop :=
  (∀ q ∈ rng d n p, (sAt d q h).isFinite = true) ∧ absSum d n p h ≤ 2 ^ 126 ∧
    (d.hasScale = true → d.scale.isFinite = true ∧ (sigma32 d).isFinite = true ∧
      |(sigma32 d).toRat| * absSum d n p h ≤ 2 ^ 126)

/-- **The tolerance**: `(depth + 1)·2^-23` relative to `M = |σ₃₂|·A` (`A = absSum`; `depth`
additions along the deepest path, and the scale's product), the scale's own conversion
`|σ − σ₃₂|·A` (the reference uses the host's double; `0` without a scale), and an absolute floor
`2^-125·(1 + |σ₃₂|)` (`float32`'s subnormal roundings over at most `2^23` additions, scaled).
Without a scale it is `(depth + 1)·2^-23·A + 2^-124`. -/
def tol (d : Inst) (n p h : Nat) : ℚ :=
  ((depth d n p + 1 : ℚ) * 2 ^ (-23 : ℤ)) * (|sig32 d| * absSum d n p h)
    + |sig d - sig32 d| * absSum d n p h + 2 ^ (-125 : ℤ) * (1 + |sig32 d|)

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

/-! ## The claim -/

/-- **The claim**: every covered launch of `chunk_global_cumsum_scalar_kernel` ends normally with
`Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## Not vacuous -/

/-- An instance of the host's launch with `s` at `2^32`, `o` at `2^36`, `cu_seqlens` at `2^40`,
zero contents: `vl` variable length, `B`, `T`, `H`, `BT = 2^e`, the flags, the scale's double
pattern, the table's dtype and `N` sequences. -/
def mk (vl : Bool) (B T H e : Nat) (rev hs : Bool) (sbits : Nat) (dt : DType) (N : Nat) :
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

theorem bytes_disj {d d' : DType} (b : Buf d) (c : Buf d')
    (h : b.base + b.len * d.bytes ≤ c.base ∨ c.base + c.len * d'.bytes ≤ b.base) :
    Disjoint b.bytes c.bytes := by
  rw [Set.disjoint_left]
  intro x hx hy
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

theorem mk_ok {vl : Bool} {B T H e : Nat} {rev hs : Bool} {sbits : Nat} {dt : DType} {N : Nat}
    (he : e ≤ 20) (hB : 0 < B) (hH : 0 < H) (hH17 : H ≤ 2 ^ 17) (hT : T < 2 ^ 31)
    (hT22 : T ≤ 2 ^ 22) (hB13 : B ≤ 8192) (hvB : vl = true → B = 1)
    (hsz : B * T * H * 4 ≤ 2 ^ 35) (hdt : dt = .i32 ∨ dt = .i64) (hN13 : N ≤ 8192) :
    (mk vl B T H e rev hs sbits dt N).ok := by
  have hbytes : dt.bytes ≤ 8 := by rcases hdt with h | h <;> subst h <;> decide
  have h1 : ((B * T * H : Nat) : Int) * 4 ≤ 2 ^ 35 := by exact_mod_cast hsz
  have h2 : ((N + 1 : Nat) : Int) * dt.bytes ≤ 2 ^ 39 := by
    have : (N + 1) * dt.bytes ≤ (N + 1) * 8 := Nat.mul_le_mul_left _ hbytes
    have h3 : (N + 1) * 8 ≤ 2 ^ 39 := by omega
    exact_mod_cast le_trans this h3
  refine ⟨⟨e, rfl⟩, Nat.pow_le_pow_right (by norm_num) he, hB, hH, hH17, hT, hT22, hB13, hvB,
    le_refl _, le_refl _, ?_, fun _ => hdt, fun _ => hN13, fun _ => le_refl _, fun _ => ?_⟩
  · exact bytes_disj _ _ (Or.inr (by simp only [mk, DType.bytes]; push_cast at h1 ⊢; omega))
  · exact bytes_disj _ _ (Or.inl (by simp only [mk, DType.bytes]; push_cast at h1 ⊢; omega))

/-- A memory in which every byte is present (and `0`). -/
def zeroMem : Mem := fun _ => some 0

theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨mk false 1 1 1 8 false false 0 .i32 0, zeroMem,
    mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num), ?_⟩
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

Every launch of the host's test (source lines 192–224): each fixed-length `(B, T, H)` and each
variable-length `(T, H, cu_seqlens)` case, forward and reverse, without a scale and with
`RCP_LN2 = 1.4426950216` (pattern `4609176139934333942`), with the host's `BT = 256`; the
variable-length cases with `int32` tables (the test's) and `int64` tables (`SETUP.md`). The
contents (and so `TablesOk`) are not part of `Inst.ok`; the host's tables are checked against
`TablesOk` on the GPU (`check_gpu.py`). Names:
`setup_<fixed|varlen>_<shape>_<fwd|rev>_<noscale|scale>[_<i32|i64>]`. -/

def setup_fixed_1x63x1_fwd_noscale : Inst := mk false 1 63 1 8 false false 0 .i32 0
theorem setup_fixed_1x63x1_fwd_noscale_ok : setup_fixed_1x63x1_fwd_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_fixed_1x63x1_fwd_scale : Inst := mk false 1 63 1 8 false true 4609176139934333942 .i32 0
theorem setup_fixed_1x63x1_fwd_scale_ok : setup_fixed_1x63x1_fwd_scale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_fixed_1x63x1_rev_noscale : Inst := mk false 1 63 1 8 true false 0 .i32 0
theorem setup_fixed_1x63x1_rev_noscale_ok : setup_fixed_1x63x1_rev_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_fixed_1x63x1_rev_scale : Inst := mk false 1 63 1 8 true true 4609176139934333942 .i32 0
theorem setup_fixed_1x63x1_rev_scale_ok : setup_fixed_1x63x1_rev_scale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_fixed_2x500x4_fwd_noscale : Inst := mk false 2 500 4 8 false false 0 .i32 0
theorem setup_fixed_2x500x4_fwd_noscale_ok : setup_fixed_2x500x4_fwd_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_fixed_2x500x4_fwd_scale : Inst := mk false 2 500 4 8 false true 4609176139934333942 .i32 0
theorem setup_fixed_2x500x4_fwd_scale_ok : setup_fixed_2x500x4_fwd_scale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_fixed_2x500x4_rev_noscale : Inst := mk false 2 500 4 8 true false 0 .i32 0
theorem setup_fixed_2x500x4_rev_noscale_ok : setup_fixed_2x500x4_rev_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_fixed_2x500x4_rev_scale : Inst := mk false 2 500 4 8 true true 4609176139934333942 .i32 0
theorem setup_fixed_2x500x4_rev_scale_ok : setup_fixed_2x500x4_rev_scale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_fixed_2x1000x5_fwd_noscale : Inst := mk false 2 1000 5 8 false false 0 .i32 0
theorem setup_fixed_2x1000x5_fwd_noscale_ok : setup_fixed_2x1000x5_fwd_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_fixed_2x1000x5_fwd_scale : Inst := mk false 2 1000 5 8 false true 4609176139934333942 .i32 0
theorem setup_fixed_2x1000x5_fwd_scale_ok : setup_fixed_2x1000x5_fwd_scale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_fixed_2x1000x5_rev_noscale : Inst := mk false 2 1000 5 8 true false 0 .i32 0
theorem setup_fixed_2x1000x5_rev_noscale_ok : setup_fixed_2x1000x5_rev_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_fixed_2x1000x5_rev_scale : Inst := mk false 2 1000 5 8 true true 4609176139934333942 .i32 0
theorem setup_fixed_2x1000x5_rev_scale_ok : setup_fixed_2x1000x5_rev_scale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_fixed_3x1024x6_fwd_noscale : Inst := mk false 3 1024 6 8 false false 0 .i32 0
theorem setup_fixed_3x1024x6_fwd_noscale_ok : setup_fixed_3x1024x6_fwd_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_fixed_3x1024x6_fwd_scale : Inst := mk false 3 1024 6 8 false true 4609176139934333942 .i32 0
theorem setup_fixed_3x1024x6_fwd_scale_ok : setup_fixed_3x1024x6_fwd_scale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_fixed_3x1024x6_rev_noscale : Inst := mk false 3 1024 6 8 true false 0 .i32 0
theorem setup_fixed_3x1024x6_rev_noscale_ok : setup_fixed_3x1024x6_rev_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_fixed_3x1024x6_rev_scale : Inst := mk false 3 1024 6 8 true true 4609176139934333942 .i32 0
theorem setup_fixed_3x1024x6_rev_scale_ok : setup_fixed_3x1024x6_rev_scale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_fixed_4x2048x8_fwd_noscale : Inst := mk false 4 2048 8 8 false false 0 .i32 0
theorem setup_fixed_4x2048x8_fwd_noscale_ok : setup_fixed_4x2048x8_fwd_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_fixed_4x2048x8_fwd_scale : Inst := mk false 4 2048 8 8 false true 4609176139934333942 .i32 0
theorem setup_fixed_4x2048x8_fwd_scale_ok : setup_fixed_4x2048x8_fwd_scale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_fixed_4x2048x8_rev_noscale : Inst := mk false 4 2048 8 8 true false 0 .i32 0
theorem setup_fixed_4x2048x8_rev_noscale_ok : setup_fixed_4x2048x8_rev_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_fixed_4x2048x8_rev_scale : Inst := mk false 4 2048 8 8 true true 4609176139934333942 .i32 0
theorem setup_fixed_4x2048x8_rev_scale_ok : setup_fixed_4x2048x8_rev_scale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_varlen_15x2_n1_fwd_noscale_i32 : Inst := mk true 1 15 2 8 false false 0 .i32 1
theorem setup_varlen_15x2_n1_fwd_noscale_i32_ok : setup_varlen_15x2_n1_fwd_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_varlen_15x2_n1_fwd_scale_i32 : Inst := mk true 1 15 2 8 false true 4609176139934333942 .i32 1
theorem setup_varlen_15x2_n1_fwd_scale_i32_ok : setup_varlen_15x2_n1_fwd_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_varlen_15x2_n1_rev_noscale_i32 : Inst := mk true 1 15 2 8 true false 0 .i32 1
theorem setup_varlen_15x2_n1_rev_noscale_i32_ok : setup_varlen_15x2_n1_rev_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_varlen_15x2_n1_rev_scale_i32 : Inst := mk true 1 15 2 8 true true 4609176139934333942 .i32 1
theorem setup_varlen_15x2_n1_rev_scale_i32_ok : setup_varlen_15x2_n1_rev_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_varlen_15x2_n1_fwd_noscale_i64 : Inst := mk true 1 15 2 8 false false 0 .i64 1
theorem setup_varlen_15x2_n1_fwd_noscale_i64_ok : setup_varlen_15x2_n1_fwd_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inr rfl) (by norm_num)
def setup_varlen_15x2_n1_fwd_scale_i64 : Inst := mk true 1 15 2 8 false true 4609176139934333942 .i64 1
theorem setup_varlen_15x2_n1_fwd_scale_i64_ok : setup_varlen_15x2_n1_fwd_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inr rfl) (by norm_num)
def setup_varlen_15x2_n1_rev_noscale_i64 : Inst := mk true 1 15 2 8 true false 0 .i64 1
theorem setup_varlen_15x2_n1_rev_noscale_i64_ok : setup_varlen_15x2_n1_rev_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inr rfl) (by norm_num)
def setup_varlen_15x2_n1_rev_scale_i64 : Inst := mk true 1 15 2 8 true true 4609176139934333942 .i64 1
theorem setup_varlen_15x2_n1_rev_scale_i64_ok : setup_varlen_15x2_n1_rev_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inr rfl) (by norm_num)
def setup_varlen_1000x3_n3_fwd_noscale_i32 : Inst := mk true 1 1000 3 8 false false 0 .i32 3
theorem setup_varlen_1000x3_n3_fwd_noscale_i32_ok : setup_varlen_1000x3_n3_fwd_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_varlen_1000x3_n3_fwd_scale_i32 : Inst := mk true 1 1000 3 8 false true 4609176139934333942 .i32 3
theorem setup_varlen_1000x3_n3_fwd_scale_i32_ok : setup_varlen_1000x3_n3_fwd_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_varlen_1000x3_n3_rev_noscale_i32 : Inst := mk true 1 1000 3 8 true false 0 .i32 3
theorem setup_varlen_1000x3_n3_rev_noscale_i32_ok : setup_varlen_1000x3_n3_rev_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_varlen_1000x3_n3_rev_scale_i32 : Inst := mk true 1 1000 3 8 true true 4609176139934333942 .i32 3
theorem setup_varlen_1000x3_n3_rev_scale_i32_ok : setup_varlen_1000x3_n3_rev_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_varlen_1000x3_n3_fwd_noscale_i64 : Inst := mk true 1 1000 3 8 false false 0 .i64 3
theorem setup_varlen_1000x3_n3_fwd_noscale_i64_ok : setup_varlen_1000x3_n3_fwd_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inr rfl) (by norm_num)
def setup_varlen_1000x3_n3_fwd_scale_i64 : Inst := mk true 1 1000 3 8 false true 4609176139934333942 .i64 3
theorem setup_varlen_1000x3_n3_fwd_scale_i64_ok : setup_varlen_1000x3_n3_fwd_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inr rfl) (by norm_num)
def setup_varlen_1000x3_n3_rev_noscale_i64 : Inst := mk true 1 1000 3 8 true false 0 .i64 3
theorem setup_varlen_1000x3_n3_rev_noscale_i64_ok : setup_varlen_1000x3_n3_rev_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inr rfl) (by norm_num)
def setup_varlen_1000x3_n3_rev_scale_i64 : Inst := mk true 1 1000 3 8 true true 4609176139934333942 .i64 3
theorem setup_varlen_1000x3_n3_rev_scale_i64_ok : setup_varlen_1000x3_n3_rev_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inr rfl) (by norm_num)
def setup_varlen_2000x4_n5_fwd_noscale_i32 : Inst := mk true 1 2000 4 8 false false 0 .i32 5
theorem setup_varlen_2000x4_n5_fwd_noscale_i32_ok : setup_varlen_2000x4_n5_fwd_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_varlen_2000x4_n5_fwd_scale_i32 : Inst := mk true 1 2000 4 8 false true 4609176139934333942 .i32 5
theorem setup_varlen_2000x4_n5_fwd_scale_i32_ok : setup_varlen_2000x4_n5_fwd_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_varlen_2000x4_n5_rev_noscale_i32 : Inst := mk true 1 2000 4 8 true false 0 .i32 5
theorem setup_varlen_2000x4_n5_rev_noscale_i32_ok : setup_varlen_2000x4_n5_rev_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_varlen_2000x4_n5_rev_scale_i32 : Inst := mk true 1 2000 4 8 true true 4609176139934333942 .i32 5
theorem setup_varlen_2000x4_n5_rev_scale_i32_ok : setup_varlen_2000x4_n5_rev_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_varlen_2000x4_n5_fwd_noscale_i64 : Inst := mk true 1 2000 4 8 false false 0 .i64 5
theorem setup_varlen_2000x4_n5_fwd_noscale_i64_ok : setup_varlen_2000x4_n5_fwd_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inr rfl) (by norm_num)
def setup_varlen_2000x4_n5_fwd_scale_i64 : Inst := mk true 1 2000 4 8 false true 4609176139934333942 .i64 5
theorem setup_varlen_2000x4_n5_fwd_scale_i64_ok : setup_varlen_2000x4_n5_fwd_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inr rfl) (by norm_num)
def setup_varlen_2000x4_n5_rev_noscale_i64 : Inst := mk true 1 2000 4 8 true false 0 .i64 5
theorem setup_varlen_2000x4_n5_rev_noscale_i64_ok : setup_varlen_2000x4_n5_rev_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inr rfl) (by norm_num)
def setup_varlen_2000x4_n5_rev_scale_i64 : Inst := mk true 1 2000 4 8 true true 4609176139934333942 .i64 5
theorem setup_varlen_2000x4_n5_rev_scale_i64_ok : setup_varlen_2000x4_n5_rev_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inr rfl) (by norm_num)
def setup_varlen_2048x4_n5_fwd_noscale_i32 : Inst := mk true 1 2048 4 8 false false 0 .i32 5
theorem setup_varlen_2048x4_n5_fwd_noscale_i32_ok : setup_varlen_2048x4_n5_fwd_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_varlen_2048x4_n5_fwd_scale_i32 : Inst := mk true 1 2048 4 8 false true 4609176139934333942 .i32 5
theorem setup_varlen_2048x4_n5_fwd_scale_i32_ok : setup_varlen_2048x4_n5_fwd_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_varlen_2048x4_n5_rev_noscale_i32 : Inst := mk true 1 2048 4 8 true false 0 .i32 5
theorem setup_varlen_2048x4_n5_rev_noscale_i32_ok : setup_varlen_2048x4_n5_rev_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_varlen_2048x4_n5_rev_scale_i32 : Inst := mk true 1 2048 4 8 true true 4609176139934333942 .i32 5
theorem setup_varlen_2048x4_n5_rev_scale_i32_ok : setup_varlen_2048x4_n5_rev_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_varlen_2048x4_n5_fwd_noscale_i64 : Inst := mk true 1 2048 4 8 false false 0 .i64 5
theorem setup_varlen_2048x4_n5_fwd_noscale_i64_ok : setup_varlen_2048x4_n5_fwd_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inr rfl) (by norm_num)
def setup_varlen_2048x4_n5_fwd_scale_i64 : Inst := mk true 1 2048 4 8 false true 4609176139934333942 .i64 5
theorem setup_varlen_2048x4_n5_fwd_scale_i64_ok : setup_varlen_2048x4_n5_fwd_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inr rfl) (by norm_num)
def setup_varlen_2048x4_n5_rev_noscale_i64 : Inst := mk true 1 2048 4 8 true false 0 .i64 5
theorem setup_varlen_2048x4_n5_rev_noscale_i64_ok : setup_varlen_2048x4_n5_rev_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inr rfl) (by norm_num)
def setup_varlen_2048x4_n5_rev_scale_i64 : Inst := mk true 1 2048 4 8 true true 4609176139934333942 .i64 5
theorem setup_varlen_2048x4_n5_rev_scale_i64_ok : setup_varlen_2048x4_n5_rev_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inr rfl) (by norm_num)
def setup_varlen_1x32_n1_fwd_noscale_i32 : Inst := mk true 1 1 32 8 false false 0 .i32 1
theorem setup_varlen_1x32_n1_fwd_noscale_i32_ok : setup_varlen_1x32_n1_fwd_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_varlen_1x32_n1_fwd_scale_i32 : Inst := mk true 1 1 32 8 false true 4609176139934333942 .i32 1
theorem setup_varlen_1x32_n1_fwd_scale_i32_ok : setup_varlen_1x32_n1_fwd_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_varlen_1x32_n1_rev_noscale_i32 : Inst := mk true 1 1 32 8 true false 0 .i32 1
theorem setup_varlen_1x32_n1_rev_noscale_i32_ok : setup_varlen_1x32_n1_rev_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_varlen_1x32_n1_rev_scale_i32 : Inst := mk true 1 1 32 8 true true 4609176139934333942 .i32 1
theorem setup_varlen_1x32_n1_rev_scale_i32_ok : setup_varlen_1x32_n1_rev_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_varlen_1x32_n1_fwd_noscale_i64 : Inst := mk true 1 1 32 8 false false 0 .i64 1
theorem setup_varlen_1x32_n1_fwd_noscale_i64_ok : setup_varlen_1x32_n1_fwd_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inr rfl) (by norm_num)
def setup_varlen_1x32_n1_fwd_scale_i64 : Inst := mk true 1 1 32 8 false true 4609176139934333942 .i64 1
theorem setup_varlen_1x32_n1_fwd_scale_i64_ok : setup_varlen_1x32_n1_fwd_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inr rfl) (by norm_num)
def setup_varlen_1x32_n1_rev_noscale_i64 : Inst := mk true 1 1 32 8 true false 0 .i64 1
theorem setup_varlen_1x32_n1_rev_noscale_i64_ok : setup_varlen_1x32_n1_rev_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inr rfl) (by norm_num)
def setup_varlen_1x32_n1_rev_scale_i64 : Inst := mk true 1 1 32 8 true true 4609176139934333942 .i64 1
theorem setup_varlen_1x32_n1_rev_scale_i64_ok : setup_varlen_1x32_n1_rev_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inr rfl) (by norm_num)
def setup_varlen_65536x32_n6_fwd_noscale_i32 : Inst := mk true 1 65536 32 8 false false 0 .i32 6
theorem setup_varlen_65536x32_n6_fwd_noscale_i32_ok : setup_varlen_65536x32_n6_fwd_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_varlen_65536x32_n6_fwd_scale_i32 : Inst := mk true 1 65536 32 8 false true 4609176139934333942 .i32 6
theorem setup_varlen_65536x32_n6_fwd_scale_i32_ok : setup_varlen_65536x32_n6_fwd_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_varlen_65536x32_n6_rev_noscale_i32 : Inst := mk true 1 65536 32 8 true false 0 .i32 6
theorem setup_varlen_65536x32_n6_rev_noscale_i32_ok : setup_varlen_65536x32_n6_rev_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_varlen_65536x32_n6_rev_scale_i32 : Inst := mk true 1 65536 32 8 true true 4609176139934333942 .i32 6
theorem setup_varlen_65536x32_n6_rev_scale_i32_ok : setup_varlen_65536x32_n6_rev_scale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_varlen_65536x32_n6_fwd_noscale_i64 : Inst := mk true 1 65536 32 8 false false 0 .i64 6
theorem setup_varlen_65536x32_n6_fwd_noscale_i64_ok : setup_varlen_65536x32_n6_fwd_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inr rfl) (by norm_num)
def setup_varlen_65536x32_n6_fwd_scale_i64 : Inst := mk true 1 65536 32 8 false true 4609176139934333942 .i64 6
theorem setup_varlen_65536x32_n6_fwd_scale_i64_ok : setup_varlen_65536x32_n6_fwd_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inr rfl) (by norm_num)
def setup_varlen_65536x32_n6_rev_noscale_i64 : Inst := mk true 1 65536 32 8 true false 0 .i64 6
theorem setup_varlen_65536x32_n6_rev_noscale_i64_ok : setup_varlen_65536x32_n6_rev_noscale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inr rfl) (by norm_num)
def setup_varlen_65536x32_n6_rev_scale_i64 : Inst := mk true 1 65536 32 8 true true 4609176139934333942 .i64 6
theorem setup_varlen_65536x32_n6_rev_scale_i64_ok : setup_varlen_65536x32_n6_rev_scale_i64.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inr rfl) (by norm_num)

/-! Two launches of fla's own tests of this kernel (`tests/ops/utils/test_cumsum.py`, `float32`
cases the host's test does not have): `2050` sequences of `16` tokens with `32` heads, and the
reversed sum over `2050` fixed rows of `64` tokens with `32` heads. -/

def setup_fla_varlen_32800x32_n2050_fwd_noscale_i32 : Inst := mk true 1 32800 32 8 false false 0 .i32 2050
theorem setup_fla_varlen_32800x32_n2050_fwd_noscale_i32_ok : setup_fla_varlen_32800x32_n2050_fwd_noscale_i32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)
def setup_fla_fixed_2050x64x32_rev_noscale : Inst := mk false 2050 64 32 8 true false 0 .i32 0
theorem setup_fla_fixed_2050x64x32_rev_noscale_ok : setup_fla_fixed_2050x64x32_rev_noscale.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by simp) (by norm_num) (Or.inl rfl) (by norm_num)

end Claims.ChunkGlobalCumsum

import TVA

/-!
# `rotary_varlen_bf16`: the prepared claim

Source: `source/rotary.py`. The kernel is `rotary_embedding_kernel`
(lines 63–154); it reads the module-level names `triton` and `tl` (lines 59–60). The host
`rotary_embedding_fwd` (lines 200–272) launches it on the `bfloat16` tensors `x`, `cos`, `sin` and
a fresh `y`, over a padded batch (`cu_seqlens = None`) or packed variable-length sequences
(`cu_seqlens` and `chunk_indices`, `int32` or `int64`), interleaved (GPT-J) or not (GPT-NeoX), with
`seqlen_offsets` one integer or a tensor of one offset per sequence (`int32` or `int64`, negative
under left padding), and `conjugate = False` (the forward of fla's `RotaryEmbedding`).

The claim (`Statement`): on every covered instance and input, every launch ends normally, every
rotated channel (`< 2 R`) of every head of every token whose position (its index in its sequence
plus the sequence's offset) is not negative is a finite `bfloat16` within
`2^-8·|ref| + 2^-22·(|x0·cos| + |x1·sin|) + 2^-132` of the exact rotation `ref` of the initial
`bfloat16` values at that position, the rotated channels of a token at a negative position keep
their bytes, and every other byte is unchanged.

This file holds the model (`rotary_embedding_kernel`, `model`, `launch`), the conditions (`Inst`,
`Inst.ok`, `Pre`), the specification (`Post`, `Spec`), the assumptions (`Assumptions`), the claim
(`Statement`), and the proofs that it is not vacuous (`consistent`, `satisfiable`) and that it covers
the setup's launches (`setup_*_ok`). It does not prove `Statement`.
-/

namespace Claims.RotaryVarlen

open TVA TVA.Math.Float

/-! ## The model -/

-- `B` is a constexpr parameter the body never reads (as in the source)
set_option linter.unusedVariables false in
/-- `rotary_embedding_kernel` (source lines 63–154). `x`, `cos`, `sin`, `y` are `*bf16` pointers;
`cu_seqlens` and `chunk_indices` are pointers of one integer dtype `sD` (`int32` or `int64` on
covered instances; the host builds `chunk_indices` with `.to(cu_seqlens)`). They are `None` (a
compile-time constant) on the padded path, where the model's `IS_VARLEN = false` branch never uses
them. The argument `seq_offsets` is one of two kinds, and `IS_SEQLEN_OFFSETS_TENSOR`
(`isinstance(seqlen_offsets, torch.Tensor)`, line 265) says which: an `int` (`False`; below `2^31`:
a runtime `int32`, or at `1` the compile-time constant `1`, whose only use `o_t + seq_offsets` gives
the same `int64` value), written `seq_offsets : I32`; or a tensor of `N` offsets (`True`), a pointer
of the integer dtype `oD` (`int32` or `int64` on covered instances), written `seq_offsets_t`. Each
path reads only its own one; the other Lean argument is unused. `T` (`do_not_specialize=['T']`: a
runtime `int32` even at `1`) is `I32`. `B H D R TR BT BD` are the `tl.constexpr` parameters, Lean
parameters; `IS_SEQLEN_OFFSETS_TENSOR`, `IS_VARLEN` and `INTERLEAVED` are `Bool` parameters.
`CONJUGATE = False` on every covered launch, so its branches (lines 120–121, 148–149) are not
visited and not in the model. -/
def rotary_embedding_kernel {sD oD : DType} (x cos sin y : Ptr .bf16) (cu_seqlens chunk_indices : Ptr sD)
    (seq_offsets_t : Ptr oD) (seq_offsets T : I32) (B H D R TR BT BD : Nat)
    (IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED : Bool) : Kernel where
  body ctx := do
    -- line 85: axes 0 and 1 converted to `int64` (`.to(tl.int64)`, sign extension), axis 2 `int32`
    let i_t0 : I64 := IntV.cast (tl.program_id ctx 0)
    let i_b : I64 := IntV.cast (tl.program_id ctx 1)
    let i_h : I32 := tl.program_id ctx 2
    -- lines 87–96: the compile-time `if IS_VARLEN`; the front end visits the taken branch only.
    -- Each branch gives the program's chunk `i_t`, the head's base pointers, the sequence
    -- length `T` (line 90 rebinds `T` to the `int64` `eos - bos`; the padded `T` is the `int32`
    -- argument, sign-extended where an `int64` operand meets it) and the packed path's `int32`
    -- sequence index `i_n` (line 105 reads `i_n`: this one when packed, `i_b` when padded).
    let st ← (if IS_VARLEN then (do
        -- line 88: `i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)` (`i_t * 2`: the
        -- constant `2` in `int64`, the tensor's dtype), then
        -- `i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)` (`+ 1`: the pointer plus the
        -- `int32` constant `1`)
        let v_n ← tl.load1 (chunk_indices + i_t0 * (IntV.ofNat 2 : I64))
        let i_n : I32 := (← tl.convert sD .i32 (Tensor.scalar v_n)).item
        let v_t ← tl.load1 (chunk_indices + i_t0 * (IntV.ofNat 2 : I64) + (IntV.ofNat 1 : I32))
        let i_t : I64 := (← tl.convert sD .i64 (Tensor.scalar v_t)).item
        -- line 89: `bos = tl.load(cu_seqlens + i_n).to(tl.int64)`,
        -- `eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)`
        let v_b ← tl.load1 (cu_seqlens + i_n)
        let bos : I64 := (← tl.convert sD .i64 (Tensor.scalar v_b)).item
        let v_e ← tl.load1 (cu_seqlens + i_n + (IntV.ofNat 1 : I32))
        let eos : I64 := (← tl.convert sD .i64 (Tensor.scalar v_e)).item
        -- line 90: `T = eos - bos` (`int64`)
        let T_n : I64 := eos - bos
        -- lines 91–92: `x + bos * H*D + i_h * D`: `(bos * H) * D` in `int64` (the constexprs
        -- built in the tensor's dtype), then `i_h * D` in `int32`, each added to the pointer
        let x_h : Ptr .bf16 := x + bos * (IntV.ofNat H : I64) * (IntV.ofNat D : I64) + i_h * (IntV.ofNat D : I32)
        let y_h : Ptr .bf16 := y + bos * (IntV.ofNat H : I64) * (IntV.ofNat D : I64) + i_h * (IntV.ofNat D : I32)
        pure (i_t, x_h, y_h, T_n, i_n))
      else (do
        -- line 94: `i_n = i_b`, the `int64` program id, which line 105 reads as `i_b` (the
        -- tuple's last component is the packed path's and unused here)
        -- lines 95–96: `x + i_n * T*H*D + i_h * D`: `((i_n * T) * H) * D` in `int64` (`T`
        -- sign-extended), then `i_h * D` in `int32`
        let x_h : Ptr .bf16 := x + i_b * (IntV.cast T : I64) * (IntV.ofNat H : I64) * (IntV.ofNat D : I64) + i_h * (IntV.ofNat D : I32)
        let y_h : Ptr .bf16 := y + i_b * (IntV.cast T : I64) * (IntV.ofNat H : I64) * (IntV.ofNat D : I64) + i_h * (IntV.ofNat D : I32)
        pure (i_t0, x_h, y_h, (IntV.cast T : I64), (IntV.ofNat 0 : I32))) :
          Prog (I64 × Ptr .bf16 × Ptr .bf16 × I64 × I32))
    let (i_t, x_h, y_h, T_n, i_n) := st
    -- lines 98–99: `if i_t * BT >= T: return`, a runtime `if` on `int64` values (signed)
    if IntV.le T_n (i_t * (IntV.ofNat BT : I64)) then
      return ()
    -- line 101: `o_t = i_t * BT + tl.arange(0, BT)`: the `int32` range converted to `int64`
    let o_t : Tensor [BT] I64 := i_t * (IntV.ofNat BT : I64) + ((tl.arange 0 BT).castInt : Tensor [BT] I64)
    -- lines 102–105: the compile-time `if not IS_SEQLEN_OFFSETS_TENSOR`
    let o_cs : Tensor [BT] I64 ← (if !IS_SEQLEN_OFFSETS_TENSOR then
        -- line 103: `o_cs = o_t + seq_offsets` (the `int32` offset sign-extended)
        pure (o_t + (IntV.cast seq_offsets : I64))
      else (do
        -- line 105: `o_cs = o_t + tl.load(seq_offsets + i_n)`: a scalar load of the offsets'
        -- dtype `oD` at the sequence's index (`int32` `i_n` packed, `int64` `i_b` padded),
        -- promoted to `int64` for the sum with the `int64` `o_t` (`tl.convert`, sign extension;
        -- the identity for `int64` offsets)
        let v_o ← tl.load1 (if IS_VARLEN then seq_offsets_t + i_n else seq_offsets_t + i_b)
        let off_n : I64 := (← tl.convert oD .i64 (Tensor.scalar v_o)).item
        pure (o_t + off_n)) : Prog (Tensor [BT] I64))
    -- line 106: `m_t = (o_t >= 0) & (o_t < T) & (o_cs >= 0) & (o_cs < TR)`, left to right; each
    -- Python integer is an `int32` tensor converted to `int64`, and `a >= 0` is `not (a < 0)`
    let m_t : Tensor [BT] Bool :=
      (((o_t.ltS (IntV.ofInt 0 : I64)).not.and (o_t.ltS T_n)).and
        (o_cs.ltS (IntV.ofInt 0 : I64)).not).and (o_cs.ltS (IntV.ofNat TR : I64))
    if !INTERLEAVED then (do
      -- line 110: `o_r = tl.arange(0, BD // 2)`
      let o_r : Tensor [BD / 2] I32 := tl.arange 0 (BD / 2)
      -- line 111: `p_x = x + o_t[:, None] * H*D + o_r[None, :]`: `((o_t * H) * D)` in `int64`
      -- added to the pointer, then the `int32` `o_r` (the broadcast to `[BT, BD // 2]` written at
      -- once with `col`/`row`)
      let p_x : Tensor [BT, BD / 2] (Ptr .bf16) :=
        x_h + o_t.col (BD / 2) * (IntV.ofNat H : I64) * (IntV.ofNat D : I64) + o_r.row BT
      -- lines 112–113: `cos + (o_cs[:, None] * R + o_r[None, :])`: the `int64` sum (`o_r`
      -- sign-extended), then the pointer
      let p_cos : Tensor [BT, BD / 2] (Ptr .bf16) :=
        cos + (o_cs.col (BD / 2) * (IntV.ofNat R : I64) + ((o_r.row BT).castInt : Tensor [BT, BD / 2] I64))
      let p_sin : Tensor [BT, BD / 2] (Ptr .bf16) :=
        sin + (o_cs.col (BD / 2) * (IntV.ofNat R : I64) + ((o_r.row BT).castInt : Tensor [BT, BD / 2] I64))
      -- line 114: `mask = m_t[:, None] & (o_r < R)[None, :]`
      let mask : Tensor [BT, BD / 2] Bool :=
        (m_t.col (BD / 2)).and ((o_r.ltS (IntV.ofNat R : I32)).row BT)
      -- lines 116–119: `other=1.0` / `other=0.0` are `float32` tensors converted to the pointer's
      -- `bfloat16` (`1.0`, `+0.0`); each load's `.to(tl.float32)` is the exact widening
      let one_bf : Tensor [BT, BD / 2] BF16 ←
        tl.convert .f32 .bf16 (tl.full [BT, BD / 2] (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)))
      let l_cos ← tl.load p_cos (some mask) (some one_bf)
      let b_cos : Tensor [BT, BD / 2] F32 ← tl.cvt l_cos
      let zero_bf : Tensor [BT, BD / 2] BF16 ←
        tl.convert .f32 .bf16 (tl.full [BT, BD / 2] (FloatV.litF32 (FloatV.ofPat 0)))
      let l_sin ← tl.load p_sin (some mask) (some zero_bf)
      let b_sin : Tensor [BT, BD / 2] F32 ← tl.cvt l_sin
      let l_x0 ← tl.load p_x (some mask) (some zero_bf)
      let b_x0 : Tensor [BT, BD / 2] F32 ← tl.cvt l_x0
      -- line 119: `p_x + R` (the `int32` constant `R`)
      let l_x1 ← tl.load (p_x + (bc (IntV.ofNat R : I32) : Tensor [BT, BD / 2] I32)) (some mask) (some zero_bf)
      let b_x1 : Tensor [BT, BD / 2] F32 ← tl.cvt l_x1
      -- line 122: `b_x0 * b_cos - b_x1 * b_sin`: products feed the subtraction (`tl.subc`: IEEE,
      -- or one product contracted into an `fma`)
      let b_o0 ← tl.subc (← tl.mul b_x0 b_cos) (← tl.mul b_x1 b_sin)
      -- line 123: `b_x0 * b_sin + b_x1 * b_cos` (`tl.addc`, as above)
      let b_o1 ← tl.addc (← tl.mul b_x0 b_sin) (← tl.mul b_x1 b_cos)
      -- line 124: `p_y = y + (o_t[:, None] * H*D + o_r[None, :])`: the `int64` sum, then the pointer
      let p_y : Tensor [BT, BD / 2] (Ptr .bf16) :=
        y_h + (o_t.col (BD / 2) * (IntV.ofNat H : I64) * (IntV.ofNat D : I64) +
          ((o_r.row BT).castInt : Tensor [BT, BD / 2] I64))
      -- line 125: the `float32` value converted to `y`'s `bfloat16` (round to nearest even)
      let s_o0 : Tensor [BT, BD / 2] BF16 ← tl.cvt b_o0
      tl.store p_y s_o0 (some mask)
      -- line 126: `p_y + R`
      let s_o1 : Tensor [BT, BD / 2] BF16 ← tl.cvt b_o1
      tl.store (p_y + (bc (IntV.ofNat R : I32) : Tensor [BT, BD / 2] I32)) s_o1 (some mask))
    else (do
      -- line 134: `o_d = tl.arange(0, BD)`
      let o_d : Tensor [BD] I32 := tl.arange 0 BD
      -- line 136: `o_d + ((o_d + 1) % 2) * 2 - 1`, `int32`: `(o_d + (((o_d + 1) % 2) * 2)) - 1`
      -- (`%`: `tl.irem`, truncating; the constants in `int32`)
      let r_swap ← tl.irem (o_d + (IntV.ofNat 1 : I32)) (bc (IntV.ofNat 2 : I32))
      let o_d_swap : Tensor [BD] I32 := o_d + r_swap * (IntV.ofNat 2 : I32) - bc (IntV.ofNat 1 : I32)
      -- line 137: `tl.arange(0, BD) // 2` (`tl.idiv`, truncating)
      let o_d_repeat ← tl.idiv (tl.arange 0 BD) (bc (IntV.ofNat 2 : I32))
      -- lines 138–139: `x + o_t[:, None] * H*D + o_d[None, :]` and `… + o_d_swap[None, :]`
      let p_x0 : Tensor [BT, BD] (Ptr .bf16) :=
        x_h + o_t.col BD * (IntV.ofNat H : I64) * (IntV.ofNat D : I64) + o_d.row BT
      let p_x1 : Tensor [BT, BD] (Ptr .bf16) :=
        x_h + o_t.col BD * (IntV.ofNat H : I64) * (IntV.ofNat D : I64) + o_d_swap.row BT
      -- lines 140–141: `cos + (o_cs[:, None] * R + o_d_repeat[None, :])` (the `int64` sum)
      let p_cos : Tensor [BT, BD] (Ptr .bf16) :=
        cos + (o_cs.col BD * (IntV.ofNat R : I64) + ((o_d_repeat.row BT).castInt : Tensor [BT, BD] I64))
      let p_sin : Tensor [BT, BD] (Ptr .bf16) :=
        sin + (o_cs.col BD * (IntV.ofNat R : I64) + ((o_d_repeat.row BT).castInt : Tensor [BT, BD] I64))
      -- line 142: `mask = m_t[:, None] & (o_d_repeat < R)[None, :]`
      let mask : Tensor [BT, BD] Bool :=
        (m_t.col BD).and ((o_d_repeat.ltS (IntV.ofNat R : I32)).row BT)
      -- lines 144–147: as lines 116–119
      let one_bf : Tensor [BT, BD] BF16 ←
        tl.convert .f32 .bf16 (tl.full [BT, BD] (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)))
      let l_cos ← tl.load p_cos (some mask) (some one_bf)
      let b_cos : Tensor [BT, BD] F32 ← tl.cvt l_cos
      let zero_bf : Tensor [BT, BD] BF16 ←
        tl.convert .f32 .bf16 (tl.full [BT, BD] (FloatV.litF32 (FloatV.ofPat 0)))
      let l_sin ← tl.load p_sin (some mask) (some zero_bf)
      let b_sin : Tensor [BT, BD] F32 ← tl.cvt l_sin
      let l_x0 ← tl.load p_x0 (some mask) (some zero_bf)
      let b_x0 : Tensor [BT, BD] F32 ← tl.cvt l_x0
      let l_x1 ← tl.load p_x1 (some mask) (some zero_bf)
      let b_x1 : Tensor [BT, BD] F32 ← tl.cvt l_x1
      -- lines 150–151: the IEEE products
      let b_o0 ← tl.mul b_x0 b_cos
      let b_o1 ← tl.mul b_x1 b_sin
      -- line 152: `tl.where(o_d[None, :] % 2 == 0, b_o0 - b_o1, b_o0 + b_o1)`, evaluated left to
      -- right: the condition (`tl.irem`, then `== 0`, already `int1`), then the difference and
      -- the sum, each fed by products (`tl.subc`, `tl.addc`)
      let r_par ← tl.irem o_d (bc (IntV.ofNat 2 : I32))
      let even : Tensor [BT, BD] Bool := (Tensor.eq r_par (bc (IntV.ofInt 0 : I32))).row BT
      let b_dif ← tl.subc b_o0 b_o1
      let b_sum ← tl.addc b_o0 b_o1
      let b_y : Tensor [BT, BD] F32 := tl.where even b_dif b_sum
      -- line 153: `p_y = y + (o_t[:, None] * H*D + o_d[None, :])`
      let p_y : Tensor [BT, BD] (Ptr .bf16) :=
        y_h + (o_t.col BD * (IntV.ofNat H : I64) * (IntV.ofNat D : I64) +
          ((o_d.row BT).castInt : Tensor [BT, BD] I64))
      -- line 154: `b_y` converted to `bfloat16`
      let s_y : Tensor [BT, BD] BF16 ← tl.cvt b_y
      tl.store p_y s_y (some mask))

/-! ## The conditions -/

/-- One launch of `rotary_embedding_kernel` as `rotary_embedding_fwd` makes it. -/
structure Inst where
  /-- The dtype of `cu_seqlens` and `chunk_indices` (any value on the padded path, where both are
  `None`). -/
  idx : DType
  /-- `B = x.shape[0]`, the grid's axis 1 (`1` for a packed batch). -/
  B : Nat
  /-- `T = x.shape[1]`, the runtime scalar `T` (the packed batch's total tokens). -/
  T : Nat
  /-- `H = x.shape[2]`, the grid's axis 2. -/
  H : Nat
  /-- `D = x.shape[3]`. -/
  D : Nat
  /-- `R = cos.shape[1]`, the number of rotated pairs (`rotary_dim = 2 R`). -/
  R : Nat
  /-- `TR = cos.shape[0]`, the positions in the tables. -/
  TR : Nat
  /-- `BT`: tokens per program. -/
  BT : Nat
  /-- `BD = triton.next_power_of_2(2 R)`. -/
  BD : Nat
  /-- `N`, the packed sequences (`cu_seqlens.shape[0] - 1`; unused when padded). -/
  N : Nat
  /-- `NT = len(chunk_indices)`, the grid's axis 0 when packed (unused when padded). -/
  NT : Nat
  /-- `seqlen_offsets` when it is one `int` for every sequence (unused when `offT`). -/
  off : Nat
  /-- `IS_SEQLEN_OFFSETS_TENSOR = isinstance(seqlen_offsets, torch.Tensor)`: one offset per
  sequence, read from the tensor `offs`. -/
  offT : Bool
  /-- The dtype of the tensor `seqlen_offsets` (any value when `offT = false`). -/
  offD : DType
  /-- `IS_VARLEN = cu_seqlens is not None`. -/
  isVarlen : Bool
  /-- `INTERLEAVED` (GPT-J `true`, GPT-NeoX `false`). -/
  interleaved : Bool
  /-- `x`: contiguous `bfloat16` `[B, T, H, D]`. -/
  x : Buf .bf16
  /-- `cos`: contiguous `bfloat16` `[TR, R]`. -/
  cos : Buf .bf16
  /-- `sin`: contiguous `bfloat16` `[TR, R]`. -/
  sin : Buf .bf16
  /-- `y`: the fresh output, of `x`'s shape and dtype. -/
  y : Buf .bf16
  /-- `cu_seqlens`: `[N + 1]` (an empty buffer when padded). -/
  cu : Buf idx
  /-- `chunk_indices`: `[NT, 2]` (an empty buffer when padded). -/
  ci : Buf idx
  /-- `seqlen_offsets` as a tensor: `[N]` (`[B]` when padded); an empty buffer when `offT = false`. -/
  offs : Buf offD

/-- The tokens of `x` (`B · T`; `T` when packed, with `B = 1`). -/
def nTok (d : Inst) : Nat := if d.isVarlen then d.T else d.B * d.T

/-- The sequences: `B` padded, `N` packed (the host's `N`, line 213). -/
def nSeq (d : Inst) : Nat := if d.isVarlen then d.N else d.B

/-- The grid's axis 0 (source lines 245, 247): `len(chunk_indices)` packed, `cdiv(T, BT)` padded. -/
def g0 (d : Inst) : Nat := if d.isVarlen then d.NT else (d.T + d.BT - 1) / d.BT

/-- The instances covered. Every field is either the host's own requirement, a bound outside the
setup's real use, or a limit of the kernel (README, Exclusions). -/
structure Inst.ok (d : Inst) : Prop where
  /-- `cu_seqlens` (and `chunk_indices`) are `int32` or `int64` (SETUP.md). -/
  idx_ok : d.idx = .i32 ∨ d.idx = .i64
  /-- `BT` is a power of two (`triton.next_power_of_2`; `tl.arange`). -/
  bt_pow2 : ∃ e, d.BT = 2 ^ e
  /-- `BD` is a power of two (`triton.next_power_of_2`). -/
  bd_pow2 : ∃ e, d.BD = 2 ^ e
  /-- The tile has at most `2^20` elements (Triton's largest block; the host's are `≤ 128·256`). -/
  block_le : d.BT * d.BD ≤ 2 ^ 20
  /-- `R ≥ 1` (SETUP.md). -/
  r_pos : 1 ≤ d.R
  /-- `2 R ≤ BD` (`BD = next_power_of_2(2 R)`). -/
  r_bd : 2 * d.R ≤ d.BD
  /-- `2 R ≤ D` (the host's assert, line 218). -/
  r_d : 2 * d.R ≤ d.D
  /-- `D ≤ 256` (the host's assert, line 217). -/
  d_le : d.D ≤ 256
  /-- `T < 2^31`: `T` is an `int32` argument (beyond: a `[B, 2^31, H, D]` tensor, outside real use). -/
  t_lt : d.T < 2 ^ 31
  /-- `seqlen_offsets < 2^31`: an `int32` argument (beyond: tables of `2^31` positions). -/
  off_lt : d.off < 2 ^ 31
  /-- `seqlen_offsets + T ≤ TR` for an `int` offset (the host's assert, line 234; a tensor of
  offsets is bounded in `Pre.offs_vals`, the host's `torch._assert_async`, lines 228–232). -/
  off_tr : d.offT = false → d.off + d.T ≤ d.TR
  /-- The tables' element indices stay far inside `int64` (tables below `2^63` bytes). -/
  tr_lt : d.TR * d.R < 2 ^ 62
  /-- `x`'s element indices stay far inside `int64`. -/
  size_lt : nTok d * d.H * d.D < 2 ^ 62
  /-- `x` holds `[B, T, H, D]`. -/
  x_len : nTok d * d.H * d.D ≤ d.x.len
  /-- `y` holds `[B, T, H, D]`. -/
  y_len : nTok d * d.H * d.D ≤ d.y.len
  /-- `cos` holds `[TR, R]`. -/
  cos_len : d.TR * d.R ≤ d.cos.len
  /-- `sin` holds `[TR, R]`. -/
  sin_len : d.TR * d.R ≤ d.sin.len
  /-- A packed batch is `[1, T, H, D]` (SETUP.md). -/
  varlen_b : d.isVarlen = true → d.B = 1
  /-- `cu_seqlens` holds `N + 1` entries. -/
  cu_len : d.isVarlen = true → d.N + 1 ≤ d.cu.len
  /-- `chunk_indices` holds `[NT, 2]`. -/
  ci_len : d.isVarlen = true → 2 * d.NT ≤ d.ci.len
  /-- **Kernel limit**: fewer than `2^31` sequences (line 88 truncates the sequence index to
  `int32`). Not reached by real use (README, Exclusions). -/
  n_lt : d.isVarlen = true → d.N < 2 ^ 31
  /-- The grid's axis 0 is below CUDA's `2^31`. -/
  grid_t : g0 d < 2 ^ 31
  /-- **Kernel limit**: the grid's axis 1 is `B`, and CUDA allows at most `65535`: a padded
  batch of `65536` or more sequences fails to launch (README, Exclusions). -/
  grid_b : d.B < 2 ^ 16
  /-- **Kernel limit**: the grid's axis 2 is `H`, at most `65535` heads (README, Exclusions). -/
  grid_h : d.H < 2 ^ 16
  /-- Fewer than `2^31` programs (Triton's launcher; README, Exclusions). -/
  grid_size : g0 d * d.B * d.H < 2 ^ 31
  /-- `y` is fresh (`torch.zeros_like(x)`): it shares no byte with an input. -/
  y_x : Disjoint d.y.bytes d.x.bytes
  y_cos : Disjoint d.y.bytes d.cos.bytes
  y_sin : Disjoint d.y.bytes d.sin.bytes
  y_cu : Disjoint d.y.bytes d.cu.bytes
  y_ci : Disjoint d.y.bytes d.ci.bytes
  /-- A padded batch's `T ≤ TR` (the host's assert, line 220). -/
  tr_t : d.isVarlen = false → d.T ≤ d.TR
  /-- The tensor `seqlen_offsets` is `int32` or `int64` (the host's assert, line 227). -/
  offD_ok : d.offT = true → d.offD = .i32 ∨ d.offD = .i64
  /-- The tensor `seqlen_offsets` holds one offset per sequence (the host's assert, line 226). -/
  offs_len : d.offT = true → nSeq d ≤ d.offs.len
  y_offs : Disjoint d.y.bytes d.offs.bytes

/-- The model on an instance. -/
def model (d : Inst) : Kernel :=
  rotary_embedding_kernel (sD := d.idx) (oD := d.offD) (d.x.ptr 0) (d.cos.ptr 0) (d.sin.ptr 0)
    (d.y.ptr 0) (d.cu.ptr 0) (d.ci.ptr 0) (d.offs.ptr 0) (IntV.ofNat d.off) (IntV.ofNat d.T)
    d.B d.H d.D d.R d.TR d.BT d.BD d.offT d.isVarlen d.interleaved

/-- The source's block shapes: `[BT]` (`o_t`, `o_cs`, `m_t`), `[BT, 1]` (`o_t[:, None]`), and
`[BD // 2]`, `[1, BD // 2]`, `[BT, BD // 2]` (not interleaved) or `[BD]`, `[1, BD]`, `[BT, BD]`
(interleaved). -/
def blockShapes (d : Inst) : List Shape :=
  if d.interleaved then [[d.BT], [d.BT, 1], [d.BD], [1, d.BD], [d.BT, d.BD]]
  else [[d.BT], [d.BT, 1], [d.BD / 2], [1, d.BD / 2], [d.BT, d.BD / 2]]

/-- The host's launch (lines 247–271): the grid `(NT, B, H)`; `enable_fp_fusion` at Triton's
default (`true`); `num_warps = 4`, `num_stages = 2` change no meaning; `num_ctas = 1`, no
`launch_pdl`, `debug = false`. -/
def launch (d : Inst) : Launch :=
  { grid := ![g0 d, d.B, d.H], fusion := true, blocks := blockShapes d }

/-- The integer an index element holds (`int32` or `int64`). -/
def idxVal : (d : DType) → d.Val → Int
  | .i32, v => v.toInt
  | .i64, v => v.toInt
  | _, _ => 0

/-- Element `j` of an index buffer, as an integer, when present. -/
def idxAt {dt : DType} (b : Buf dt) (m : Mem) (j : Nat) : Option Int :=
  (m.read dt (b.ptr (j : Int)).addr).map (idxVal dt)

/-- Element `j` of a `bfloat16` buffer. -/
def bfAt (b : Buf .bf16) (m : Mem) (j : Nat) : Option BF16 := m.read .bf16 (b.ptr (j : Int)).addr

/-- `cu_seqlens[n]` (as a natural number). -/
def cuN (d : Inst) (m : Mem) (n : Nat) : Nat := ((idxAt d.cu m n).getD 0).toNat

/-- The length of sequence `n`, `cu_seqlens[n + 1] - cu_seqlens[n]`. -/
def seqLen (d : Inst) (m : Mem) (n : Nat) : Nat := cuN d m (n + 1) - cuN d m n

/-- `chunk_indices[i, 0]`, the sequence of program `i`. -/
def chSeq (d : Inst) (m : Mem) (i : Nat) : Nat := ((idxAt d.ci m (2 * i)).getD 0).toNat

/-- `chunk_indices[i, 1]`, the chunk of program `i` in its sequence. -/
def chChunk (d : Inst) (m : Mem) (i : Nat) : Nat := ((idxAt d.ci m (2 * i + 1)).getD 0).toNat

/-- Sequence `n`'s offset: `seqlen_offsets[n]` from the tensor (`offT`), else the one `int`. -/
def offAt (d : Inst) (m : Mem) (n : Nat) : Int :=
  if d.offT then (idxAt d.offs m n).getD 0 else (d.off : Int)

/-- Sequence `n`'s length: `cu_seqlens[n + 1] - cu_seqlens[n]` packed, `T` padded. -/
def lenAt (d : Inst) (m : Mem) (n : Nat) : Nat := if d.isVarlen then seqLen d m n else d.T

/-- Sequence `n`'s first flat token: `cu_seqlens[n]` packed, `n · T` padded. -/
def bosAt (d : Inst) (m : Mem) (n : Nat) : Nat := if d.isVarlen then cuN d m n else n * d.T

/-- The packed batch's metadata, as fla passes it: `cu_seqlens` non-decreasing from `0` to `T`,
and `chunk_indices` listing every (sequence, chunk of `BT` tokens) pair exactly once, as
`prepare_chunk_indices` builds it (lines 189–197; any order). -/
structure SeqOk (d : Inst) (init : Mem) : Prop where
  cuv : d.cu.Valid init
  civ : d.ci.Valid init
  cu_some : ∀ n, n ≤ d.N → ∃ v, idxAt d.cu init n = some v ∧ 0 ≤ v
  cu_zero : cuN d init 0 = 0
  cu_last : cuN d init d.N = d.T
  cu_mono : ∀ n, n < d.N → cuN d init n ≤ cuN d init (n + 1)
  ci_some : ∀ i, i < 2 * d.NT → ∃ v, idxAt d.ci init i = some v ∧ 0 ≤ v
  ci_ok : ∀ i, i < d.NT → chSeq d init i < d.N ∧ chChunk d init i * d.BT < seqLen d init (chSeq d init i)
  ci_all : ∀ n j, n < d.N → j * d.BT < seqLen d init n →
    ∃ i, i < d.NT ∧ chSeq d init i = n ∧ chChunk d init i = j
  ci_once : ∀ i i', i < d.NT → i' < d.NT → chSeq d init i = chSeq d init i' →
    chChunk d init i = chChunk d init i' → i = i'

/-- An activation the claim covers: finite, `|v| ≤ 2^126` (so that a rotated value stays inside
`bfloat16`'s range). -/
def XOk (v : BF16) : Prop := v.isFinite = true ∧ |v.toRat| ≤ 2 ^ 126

/-- A table entry the claim covers: a finite cosine or sine, `|v| ≤ 1`. -/
def TabOk (v : BF16) : Prop := v.isFinite = true ∧ |v.toRat| ≤ 1

/-- The inputs covered. -/
structure Pre (d : Inst) (init : Mem) : Prop where
  xv : d.x.Valid init
  yv : d.y.Valid init
  cosv : d.cos.Valid init
  sinv : d.sin.Valid init
  /-- The rotated channels (`c < 2 R`) of every token and head of `x`; the channels from `2 R` on,
  which the kernel never reads, may hold anything. -/
  x_vals : ∀ j, j < nTok d * d.H * d.D → j % d.D < 2 * d.R → ∃ v, bfAt d.x init j = some v ∧ XOk v
  /-- The table rows the launch reads: token `l` of sequence `n` reads position
  `l + offAt n` when it is not negative; other rows may hold anything. -/
  cos_vals : ∀ n l k, n < nSeq d → l < lenAt d init n → 0 ≤ (l : Int) + offAt d init n → k < d.R →
    ∃ v, bfAt d.cos init (((l : Int) + offAt d init n).toNat * d.R + k) = some v ∧ TabOk v
  sin_vals : ∀ n l k, n < nSeq d → l < lenAt d init n → 0 ≤ (l : Int) + offAt d init n → k < d.R →
    ∃ v, bfAt d.sin init (((l : Int) + offAt d init n).toNat * d.R + k) = some v ∧ TabOk v
  seqs : d.isVarlen = true → SeqOk d init
  /-- A tensor of offsets: valid, and every sequence's offset `v` (negative under left padding)
  with `len + v ≤ TR` (the host's `torch._assert_async`, lines 228–232) and `v ≥ -2^31` (beyond:
  more than `2^31` padding tokens, outside real use). -/
  offs_vals : d.offT = true → d.offs.Valid init ∧ ∀ n, n < nSeq d →
    ∃ v, idxAt d.offs init n = some v ∧ -2 ^ 31 ≤ v ∧ (lenAt d init n : Int) + v ≤ d.TR

/-! ## The specification -/

/-- Element `(t, h, c)` of a contiguous `[·, H, D]` tensor, `t` the flat token (`b · T + l`). -/
def tokIdx (d : Inst) (t h c : Nat) : Nat := (t * d.H + h) * d.D + c

/-- The pair of rotated channel `c < 2 R`: `c // 2` (interleaved) or `c mod R`. -/
def pairOf (d : Inst) (c : Nat) : Nat :=
  if d.interleaved then c / 2 else if c < d.R then c else c - d.R

/-- The other channel of `c`'s pair: `c ± 1` by parity (interleaved), `c ± R` (not). -/
def partner (d : Inst) (c : Nat) : Nat :=
  if d.interleaved then (if c % 2 = 0 then c + 1 else c - 1)
  else if c < d.R then c + d.R else c - d.R

/-- Whether channel `c` is the first of its pair (`x0·cos − x1·sin`) or the second
(`x1·cos + x0·sin`). -/
def isFirst (d : Inst) (c : Nat) : Bool :=
  if d.interleaved then c % 2 = 0 else c < d.R

/-- The exact rotation of channel `c`: `a·cs − b·sn` (first) or `a·cs + b·sn` (second), with `a`
the channel's own value, `b` its partner's, `cs`, `sn` the table entries of its pair. -/
def rotRef (first : Bool) (a b cs sn : ℚ) : ℚ := if first then a * cs - b * sn else a * cs + b * sn

/-- The tolerance: one `bfloat16` rounding of the result (`2^-8·|ref|`), the `float32`
arithmetic before it (`2^-22·(|a·cs| + |b·sn|)`), and the subnormal floors (`2^-132`). -/
def tol (first : Bool) (a b cs sn : ℚ) : ℚ :=
  2 ^ (-8 : ℤ) * |rotRef first a b cs sn| + 2 ^ (-22 : ℤ) * (|a * cs| + |b * sn|) + 2 ^ (-132 : ℤ)

/-- Rotated channel `c` of flat token `t`, head `h`, at table position `pos`: the final `y` holds a
finite `bfloat16` within `tol` of the exact rotation of the initial `x`, `cos`, `sin`. -/
def CellOk (d : Inst) (init fin : Mem) (t h c pos : Nat) : Prop :=
  ∃ a b cs sn v : BF16,
    bfAt d.x init (tokIdx d t h c) = some a ∧
    bfAt d.x init (tokIdx d t h (partner d c)) = some b ∧
    bfAt d.cos init (pos * d.R + pairOf d c) = some cs ∧
    bfAt d.sin init (pos * d.R + pairOf d c) = some sn ∧
    bfAt d.y fin (tokIdx d t h c) = some v ∧
    v.isFinite = true ∧
    |v.toRat - rotRef (isFirst d c) a.toRat b.toRat cs.toRat sn.toRat| ≤
      tol (isFirst d c) a.toRat b.toRat cs.toRat sn.toRat

/-- The elements of `y` the launch may write: the rotated channels of every token and head. -/
def outSet (d : Inst) : Set Nat :=
  {k | ∃ t h c, t < nTok d ∧ h < d.H ∧ c < 2 * d.R ∧ k = tokIdx d t h c}

/-- What the launch leaves. Token `l` of sequence `n` (padded: batch entry `n`) is flat token
`bosAt n + l` (`n·T + l` padded, `cu_seqlens[n] + l` packed) at position `p = l + offAt n`. When
`p ≥ 0`, every rotated channel of every head of the token meets `CellOk` at `p`; when `p < 0` (a
left-padding token under a tensor of offsets), the kernel skips it and its rotated channels keep
their bytes (the host's `torch.zeros_like` made them `0`). Every byte outside the rotated
channels of `y` is unchanged (`y`'s channels from `2 R` on hold what the host put there). -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ n l h c, n < nSeq d → l < lenAt d init n → h < d.H → c < 2 * d.R →
      (0 ≤ (l : Int) + offAt d init n →
        CellOk d init fin (bosAt d init n + l) h c ((l : Int) + offAt d init n).toNat) ∧
      ((l : Int) + offAt d init n < 0 →
        ∀ z, z ∈ d.y.elems {tokIdx d (bosAt d init n + l) h c} → fin z = init z)) ∧
  ∀ z, z ∉ d.y.elems (outSet d) → fin z = init z

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## The assumptions -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on every covered instance, when `launch d` has `debug = false`
  and is valid, every outcome of the compiled kernel launched as `launch d`, from a memory from
  which the model's launch is well behaved, is one of the outcomes of the model's launch. The
  kernel's float operations (`tl.mul`, `tl.addc`, `tl.subc`, `tl.cvt`, `tl.convert`) are exact up
  to NaN payload in the library, so this is the only assumption. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)

/-- The claim to prove. -/
def Statement : Prop := ∀ K P, Assumptions K P → Spec K

/-! ## Not vacuous -/

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.any model launch
  exact ⟨K, Platform.any, htot, ⟨fun d _ => hf d⟩⟩

/-- Two buffers whose byte ranges do not meet are disjoint. -/
theorem bytes_disj {d₁ d₂ : DType} {b : Buf d₁} {c : Buf d₂}
    (h : c.base + c.len * d₂.bytes ≤ b.base ∨ b.base + b.len * d₁.bytes ≤ c.base) :
    Disjoint b.bytes c.bytes := by
  rw [Set.disjoint_left]
  intro z hb hc
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hb hc
  omega

/-- A memory in which every byte is present and `0`. -/
def zeroMem : Mem := fun _ => some 0

theorem zeroMem_valid {dt : DType} {b : Buf dt} (h1 : 0 ≤ b.base)
    (h2 : b.base + b.len * dt.bytes ≤ 2 ^ 64) (h3 : b.base % dt.bytes = 0) : b.Valid zeroMem :=
  ⟨h1, h2, h3, fun _ _ => by simp [zeroMem]⟩

theorem zeroMem_read_bf16 {a : Int} (ha : Mem.Aligned .bf16 a) :
    zeroMem.read .bf16 a = some (FloatV.ofPat 0 : BF16) := by
  have hb : zeroMem.readBytes a DType.bf16.bytes = some [0, 0] := by
    show zeroMem.readBytes a 2 = some [0, 0]
    simp [Mem.readBytes, zeroMem]
  unfold Mem.read
  simp only [ha, ↓reduceIte, hb, Option.map_some]
  rfl

theorem zero_toRat : (FloatV.ofPat 0 : BF16).toRat = 0 := by decide +kernel

theorem zero_xok : XOk (FloatV.ofPat 0 : BF16) :=
  ⟨by decide +kernel, by rw [zero_toRat]; norm_num⟩

theorem zero_tabok : TabOk (FloatV.ofPat 0 : BF16) :=
  ⟨by decide +kernel, by rw [zero_toRat]; norm_num⟩

theorem zeroMem_bfAt {b : Buf .bf16} (hv : b.Valid zeroMem) {j : Nat} (hj : j < b.len) :
    bfAt b zeroMem j = some (FloatV.ofPat 0 : BF16) :=
  zeroMem_read_bf16 (Buf.aligned b hv hj)

/-! ## Coverage of the setup -/

/-- the host test case (source lines 323-337): `B = 2`, `T = 2048`, `H = 4`, `D = 128`, rotary_dim `128`, offset `0`, padded, GPT-NeoX; `BT = 16`, `BD = 128` (the host's formulas, 132 SMs). -/
def setup_neox_padded_b2_t2048_h4_d128_r128 : Inst where
  idx := .i32
  B := 2
  T := 2048
  H := 4
  D := 128
  R := 64
  TR := 2048
  BT := 16
  BD := 128
  N := 2
  NT := 128
  off := 0
  isVarlen := false
  interleaved := false
  x := ⟨1099511627776, 2097152⟩
  cos := ⟨2199023255552, 131072⟩
  sin := ⟨3298534883328, 131072⟩
  y := ⟨4398046511104, 2097152⟩
  cu := ⟨5497558138880, 0⟩
  ci := ⟨6597069766656, 0⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_neox_padded_b2_t2048_h4_d128_r128_ok : setup_neox_padded_b2_t2048_h4_d128_r128.ok :=
  ⟨Or.inl rfl,
    ⟨4, rfl⟩,
    ⟨7, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- the host test case (source lines 323-337): `B = 2`, `T = 4096`, `H = 4`, `D = 256`, rotary_dim `256`, offset `0`, padded, GPT-NeoX; `BT = 32`, `BD = 256` (the host's formulas, 132 SMs). -/
def setup_neox_padded_b2_t4096_h4_d256_r256 : Inst where
  idx := .i32
  B := 2
  T := 4096
  H := 4
  D := 256
  R := 128
  TR := 4096
  BT := 32
  BD := 256
  N := 2
  NT := 128
  off := 0
  isVarlen := false
  interleaved := false
  x := ⟨1099511627776, 8388608⟩
  cos := ⟨2199023255552, 524288⟩
  sin := ⟨3298534883328, 524288⟩
  y := ⟨4398046511104, 8388608⟩
  cu := ⟨5497558138880, 0⟩
  ci := ⟨6597069766656, 0⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_neox_padded_b2_t4096_h4_d256_r256_ok : setup_neox_padded_b2_t4096_h4_d256_r256.ok :=
  ⟨Or.inl rfl,
    ⟨5, rfl⟩,
    ⟨8, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- the host test case (source lines 323-337): `B = 1`, `T = 1`, `H = 1`, `D = 64`, rotary_dim `64`, offset `0`, padded, GPT-NeoX; `BT = 1`, `BD = 64` (the host's formulas, 132 SMs). -/
def setup_neox_padded_b1_t1_h1_d64_r64 : Inst where
  idx := .i32
  B := 1
  T := 1
  H := 1
  D := 64
  R := 32
  TR := 1
  BT := 1
  BD := 64
  N := 1
  NT := 1
  off := 0
  isVarlen := false
  interleaved := false
  x := ⟨1099511627776, 64⟩
  cos := ⟨2199023255552, 32⟩
  sin := ⟨3298534883328, 32⟩
  y := ⟨4398046511104, 64⟩
  cu := ⟨5497558138880, 0⟩
  ci := ⟨6597069766656, 0⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_neox_padded_b1_t1_h1_d64_r64_ok : setup_neox_padded_b1_t1_h1_d64_r64.ok :=
  ⟨Or.inl rfl,
    ⟨0, rfl⟩,
    ⟨6, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- the host test case (source lines 323-337): `B = 3`, `T = 1000`, `H = 8`, `D = 64`, rotary_dim `64`, offset `7`, padded, GPT-NeoX; `BT = 8`, `BD = 64` (the host's formulas, 132 SMs). -/
def setup_neox_padded_b3_t1000_h8_d64_r64_off7 : Inst where
  idx := .i32
  B := 3
  T := 1000
  H := 8
  D := 64
  R := 32
  TR := 1007
  BT := 8
  BD := 64
  N := 3
  NT := 125
  off := 7
  isVarlen := false
  interleaved := false
  x := ⟨1099511627776, 1536000⟩
  cos := ⟨2199023255552, 32224⟩
  sin := ⟨3298534883328, 32224⟩
  y := ⟨4398046511104, 1536000⟩
  cu := ⟨5497558138880, 0⟩
  ci := ⟨6597069766656, 0⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_neox_padded_b3_t1000_h8_d64_r64_off7_ok : setup_neox_padded_b3_t1000_h8_d64_r64_off7.ok :=
  ⟨Or.inl rfl,
    ⟨3, rfl⟩,
    ⟨6, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- the host test case (source lines 323-337): `B = 2`, `T = 256`, `H = 4`, `D = 128`, rotary_dim `64`, offset `0`, padded, GPT-NeoX; `BT = 2`, `BD = 64` (the host's formulas, 132 SMs). -/
def setup_neox_padded_b2_t256_h4_d128_r64 : Inst where
  idx := .i32
  B := 2
  T := 256
  H := 4
  D := 128
  R := 32
  TR := 256
  BT := 2
  BD := 64
  N := 2
  NT := 128
  off := 0
  isVarlen := false
  interleaved := false
  x := ⟨1099511627776, 262144⟩
  cos := ⟨2199023255552, 8192⟩
  sin := ⟨3298534883328, 8192⟩
  y := ⟨4398046511104, 262144⟩
  cu := ⟨5497558138880, 0⟩
  ci := ⟨6597069766656, 0⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_neox_padded_b2_t256_h4_d128_r64_ok : setup_neox_padded_b2_t256_h4_d128_r64.ok :=
  ⟨Or.inl rfl,
    ⟨1, rfl⟩,
    ⟨6, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- the host test case (source lines 323-337): `B = 1`, `T = 2048`, `H = 4`, `D = 128`, rotary_dim `128`, offset `0`, packed, `N = 4` sequences (`cu_seqlens` i64), `NT = 130` chunks, GPT-NeoX; `BT = 16`, `BD = 128` (the host's formulas, 132 SMs). -/
def setup_neox_packed_b1_t2048_h4_d128_r128_n4_i64 : Inst where
  idx := .i64
  B := 1
  T := 2048
  H := 4
  D := 128
  R := 64
  TR := 2048
  BT := 16
  BD := 128
  N := 4
  NT := 130
  off := 0
  isVarlen := true
  interleaved := false
  x := ⟨1099511627776, 1048576⟩
  cos := ⟨2199023255552, 131072⟩
  sin := ⟨3298534883328, 131072⟩
  y := ⟨4398046511104, 1048576⟩
  cu := ⟨5497558138880, 5⟩
  ci := ⟨6597069766656, 260⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_neox_packed_b1_t2048_h4_d128_r128_n4_i64_ok : setup_neox_packed_b1_t2048_h4_d128_r128_n4_i64.ok :=
  ⟨Or.inr rfl,
    ⟨4, rfl⟩,
    ⟨7, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- the host test case (source lines 323-337): `B = 1`, `T = 4096`, `H = 4`, `D = 256`, rotary_dim `256`, offset `0`, packed, `N = 4` sequences (`cu_seqlens` i32), `NT = 130` chunks, GPT-NeoX; `BT = 32`, `BD = 256` (the host's formulas, 132 SMs). -/
def setup_neox_packed_b1_t4096_h4_d256_r256_n4_i32 : Inst where
  idx := .i32
  B := 1
  T := 4096
  H := 4
  D := 256
  R := 128
  TR := 4096
  BT := 32
  BD := 256
  N := 4
  NT := 130
  off := 0
  isVarlen := true
  interleaved := false
  x := ⟨1099511627776, 4194304⟩
  cos := ⟨2199023255552, 524288⟩
  sin := ⟨3298534883328, 524288⟩
  y := ⟨4398046511104, 4194304⟩
  cu := ⟨5497558138880, 5⟩
  ci := ⟨6597069766656, 260⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_neox_packed_b1_t4096_h4_d256_r256_n4_i32_ok : setup_neox_packed_b1_t4096_h4_d256_r256_n4_i32.ok :=
  ⟨Or.inl rfl,
    ⟨5, rfl⟩,
    ⟨8, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- the host test case (source lines 323-337): `B = 1`, `T = 777`, `H = 2`, `D = 64`, rotary_dim `32`, offset `0`, packed, `N = 9` sequences (`cu_seqlens` i32), `NT = 101` chunks, GPT-NeoX; `BT = 8`, `BD = 32` (the host's formulas, 132 SMs). -/
def setup_neox_packed_b1_t777_h2_d64_r32_n9_i32 : Inst where
  idx := .i32
  B := 1
  T := 777
  H := 2
  D := 64
  R := 16
  TR := 777
  BT := 8
  BD := 32
  N := 9
  NT := 101
  off := 0
  isVarlen := true
  interleaved := false
  x := ⟨1099511627776, 99456⟩
  cos := ⟨2199023255552, 12432⟩
  sin := ⟨3298534883328, 12432⟩
  y := ⟨4398046511104, 99456⟩
  cu := ⟨5497558138880, 10⟩
  ci := ⟨6597069766656, 202⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_neox_packed_b1_t777_h2_d64_r32_n9_i32_ok : setup_neox_packed_b1_t777_h2_d64_r32_n9_i32.ok :=
  ⟨Or.inl rfl,
    ⟨3, rfl⟩,
    ⟨5, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- the host test case (source lines 323-337): `B = 1`, `T = 5`, `H = 2`, `D = 128`, rotary_dim `128`, offset `0`, packed, `N = 5` sequences (`cu_seqlens` i64), `NT = 5` chunks, GPT-NeoX; `BT = 1`, `BD = 128` (the host's formulas, 132 SMs). -/
def setup_neox_packed_b1_t5_h2_d128_r128_n5_i64 : Inst where
  idx := .i64
  B := 1
  T := 5
  H := 2
  D := 128
  R := 64
  TR := 5
  BT := 1
  BD := 128
  N := 5
  NT := 5
  off := 0
  isVarlen := true
  interleaved := false
  x := ⟨1099511627776, 1280⟩
  cos := ⟨2199023255552, 320⟩
  sin := ⟨3298534883328, 320⟩
  y := ⟨4398046511104, 1280⟩
  cu := ⟨5497558138880, 6⟩
  ci := ⟨6597069766656, 10⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_neox_packed_b1_t5_h2_d128_r128_n5_i64_ok : setup_neox_packed_b1_t5_h2_d128_r128_n5_i64.ok :=
  ⟨Or.inr rfl,
    ⟨0, rfl⟩,
    ⟨7, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- the host test case (source lines 323-337): `B = 4`, `T = 8192`, `H = 32`, `D = 128`, rotary_dim `128`, offset `0`, padded, GPT-NeoX; `BT = 64`, `BD = 128` (the host's formulas, 132 SMs). -/
def setup_neox_padded_b4_t8192_h32_d128_r128 : Inst where
  idx := .i32
  B := 4
  T := 8192
  H := 32
  D := 128
  R := 64
  TR := 8192
  BT := 64
  BD := 128
  N := 4
  NT := 128
  off := 0
  isVarlen := false
  interleaved := false
  x := ⟨1099511627776, 134217728⟩
  cos := ⟨2199023255552, 524288⟩
  sin := ⟨3298534883328, 524288⟩
  y := ⟨4398046511104, 134217728⟩
  cu := ⟨5497558138880, 0⟩
  ci := ⟨6597069766656, 0⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_neox_padded_b4_t8192_h32_d128_r128_ok : setup_neox_padded_b4_t8192_h32_d128_r128.ok :=
  ⟨Or.inl rfl,
    ⟨6, rfl⟩,
    ⟨7, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- the host test case (source lines 323-337): `B = 1`, `T = 32768`, `H = 32`, `D = 128`, rotary_dim `128`, offset `0`, packed, `N = 16` sequences (`cu_seqlens` i32), `NT = 263` chunks, GPT-NeoX; `BT = 128`, `BD = 128` (the host's formulas, 132 SMs). -/
def setup_neox_packed_b1_t32768_h32_d128_r128_n16_i32 : Inst where
  idx := .i32
  B := 1
  T := 32768
  H := 32
  D := 128
  R := 64
  TR := 32768
  BT := 128
  BD := 128
  N := 16
  NT := 263
  off := 0
  isVarlen := true
  interleaved := false
  x := ⟨1099511627776, 134217728⟩
  cos := ⟨2199023255552, 2097152⟩
  sin := ⟨3298534883328, 2097152⟩
  y := ⟨4398046511104, 134217728⟩
  cu := ⟨5497558138880, 17⟩
  ci := ⟨6597069766656, 526⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_neox_packed_b1_t32768_h32_d128_r128_n16_i32_ok : setup_neox_packed_b1_t32768_h32_d128_r128_n16_i32.ok :=
  ⟨Or.inl rfl,
    ⟨7, rfl⟩,
    ⟨7, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- the host test case (source lines 323-337): `B = 2`, `T = 2048`, `H = 4`, `D = 128`, rotary_dim `128`, offset `0`, padded, GPT-J; `BT = 16`, `BD = 128` (the host's formulas, 132 SMs). -/
def setup_gptj_padded_b2_t2048_h4_d128_r128 : Inst where
  idx := .i32
  B := 2
  T := 2048
  H := 4
  D := 128
  R := 64
  TR := 2048
  BT := 16
  BD := 128
  N := 2
  NT := 128
  off := 0
  isVarlen := false
  interleaved := true
  x := ⟨1099511627776, 2097152⟩
  cos := ⟨2199023255552, 131072⟩
  sin := ⟨3298534883328, 131072⟩
  y := ⟨4398046511104, 2097152⟩
  cu := ⟨5497558138880, 0⟩
  ci := ⟨6597069766656, 0⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_gptj_padded_b2_t2048_h4_d128_r128_ok : setup_gptj_padded_b2_t2048_h4_d128_r128.ok :=
  ⟨Or.inl rfl,
    ⟨4, rfl⟩,
    ⟨7, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- the host test case (source lines 323-337): `B = 2`, `T = 4096`, `H = 4`, `D = 256`, rotary_dim `256`, offset `0`, padded, GPT-J; `BT = 32`, `BD = 256` (the host's formulas, 132 SMs). -/
def setup_gptj_padded_b2_t4096_h4_d256_r256 : Inst where
  idx := .i32
  B := 2
  T := 4096
  H := 4
  D := 256
  R := 128
  TR := 4096
  BT := 32
  BD := 256
  N := 2
  NT := 128
  off := 0
  isVarlen := false
  interleaved := true
  x := ⟨1099511627776, 8388608⟩
  cos := ⟨2199023255552, 524288⟩
  sin := ⟨3298534883328, 524288⟩
  y := ⟨4398046511104, 8388608⟩
  cu := ⟨5497558138880, 0⟩
  ci := ⟨6597069766656, 0⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_gptj_padded_b2_t4096_h4_d256_r256_ok : setup_gptj_padded_b2_t4096_h4_d256_r256.ok :=
  ⟨Or.inl rfl,
    ⟨5, rfl⟩,
    ⟨8, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- the host test case (source lines 323-337): `B = 1`, `T = 1`, `H = 1`, `D = 64`, rotary_dim `64`, offset `0`, padded, GPT-J; `BT = 1`, `BD = 64` (the host's formulas, 132 SMs). -/
def setup_gptj_padded_b1_t1_h1_d64_r64 : Inst where
  idx := .i32
  B := 1
  T := 1
  H := 1
  D := 64
  R := 32
  TR := 1
  BT := 1
  BD := 64
  N := 1
  NT := 1
  off := 0
  isVarlen := false
  interleaved := true
  x := ⟨1099511627776, 64⟩
  cos := ⟨2199023255552, 32⟩
  sin := ⟨3298534883328, 32⟩
  y := ⟨4398046511104, 64⟩
  cu := ⟨5497558138880, 0⟩
  ci := ⟨6597069766656, 0⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_gptj_padded_b1_t1_h1_d64_r64_ok : setup_gptj_padded_b1_t1_h1_d64_r64.ok :=
  ⟨Or.inl rfl,
    ⟨0, rfl⟩,
    ⟨6, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- the host test case (source lines 323-337): `B = 3`, `T = 1000`, `H = 8`, `D = 64`, rotary_dim `64`, offset `7`, padded, GPT-J; `BT = 8`, `BD = 64` (the host's formulas, 132 SMs). -/
def setup_gptj_padded_b3_t1000_h8_d64_r64_off7 : Inst where
  idx := .i32
  B := 3
  T := 1000
  H := 8
  D := 64
  R := 32
  TR := 1007
  BT := 8
  BD := 64
  N := 3
  NT := 125
  off := 7
  isVarlen := false
  interleaved := true
  x := ⟨1099511627776, 1536000⟩
  cos := ⟨2199023255552, 32224⟩
  sin := ⟨3298534883328, 32224⟩
  y := ⟨4398046511104, 1536000⟩
  cu := ⟨5497558138880, 0⟩
  ci := ⟨6597069766656, 0⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_gptj_padded_b3_t1000_h8_d64_r64_off7_ok : setup_gptj_padded_b3_t1000_h8_d64_r64_off7.ok :=
  ⟨Or.inl rfl,
    ⟨3, rfl⟩,
    ⟨6, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- the host test case (source lines 323-337): `B = 2`, `T = 256`, `H = 4`, `D = 128`, rotary_dim `64`, offset `0`, padded, GPT-J; `BT = 2`, `BD = 64` (the host's formulas, 132 SMs). -/
def setup_gptj_padded_b2_t256_h4_d128_r64 : Inst where
  idx := .i32
  B := 2
  T := 256
  H := 4
  D := 128
  R := 32
  TR := 256
  BT := 2
  BD := 64
  N := 2
  NT := 128
  off := 0
  isVarlen := false
  interleaved := true
  x := ⟨1099511627776, 262144⟩
  cos := ⟨2199023255552, 8192⟩
  sin := ⟨3298534883328, 8192⟩
  y := ⟨4398046511104, 262144⟩
  cu := ⟨5497558138880, 0⟩
  ci := ⟨6597069766656, 0⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_gptj_padded_b2_t256_h4_d128_r64_ok : setup_gptj_padded_b2_t256_h4_d128_r64.ok :=
  ⟨Or.inl rfl,
    ⟨1, rfl⟩,
    ⟨6, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- the host test case (source lines 323-337): `B = 1`, `T = 2048`, `H = 4`, `D = 128`, rotary_dim `128`, offset `0`, packed, `N = 4` sequences (`cu_seqlens` i64), `NT = 130` chunks, GPT-J; `BT = 16`, `BD = 128` (the host's formulas, 132 SMs). -/
def setup_gptj_packed_b1_t2048_h4_d128_r128_n4_i64 : Inst where
  idx := .i64
  B := 1
  T := 2048
  H := 4
  D := 128
  R := 64
  TR := 2048
  BT := 16
  BD := 128
  N := 4
  NT := 130
  off := 0
  isVarlen := true
  interleaved := true
  x := ⟨1099511627776, 1048576⟩
  cos := ⟨2199023255552, 131072⟩
  sin := ⟨3298534883328, 131072⟩
  y := ⟨4398046511104, 1048576⟩
  cu := ⟨5497558138880, 5⟩
  ci := ⟨6597069766656, 260⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_gptj_packed_b1_t2048_h4_d128_r128_n4_i64_ok : setup_gptj_packed_b1_t2048_h4_d128_r128_n4_i64.ok :=
  ⟨Or.inr rfl,
    ⟨4, rfl⟩,
    ⟨7, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- the host test case (source lines 323-337): `B = 1`, `T = 4096`, `H = 4`, `D = 256`, rotary_dim `256`, offset `0`, packed, `N = 4` sequences (`cu_seqlens` i32), `NT = 129` chunks, GPT-J; `BT = 32`, `BD = 256` (the host's formulas, 132 SMs). -/
def setup_gptj_packed_b1_t4096_h4_d256_r256_n4_i32 : Inst where
  idx := .i32
  B := 1
  T := 4096
  H := 4
  D := 256
  R := 128
  TR := 4096
  BT := 32
  BD := 256
  N := 4
  NT := 129
  off := 0
  isVarlen := true
  interleaved := true
  x := ⟨1099511627776, 4194304⟩
  cos := ⟨2199023255552, 524288⟩
  sin := ⟨3298534883328, 524288⟩
  y := ⟨4398046511104, 4194304⟩
  cu := ⟨5497558138880, 5⟩
  ci := ⟨6597069766656, 258⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_gptj_packed_b1_t4096_h4_d256_r256_n4_i32_ok : setup_gptj_packed_b1_t4096_h4_d256_r256_n4_i32.ok :=
  ⟨Or.inl rfl,
    ⟨5, rfl⟩,
    ⟨8, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- the host test case (source lines 323-337): `B = 1`, `T = 777`, `H = 2`, `D = 64`, rotary_dim `32`, offset `0`, packed, `N = 9` sequences (`cu_seqlens` i32), `NT = 103` chunks, GPT-J; `BT = 8`, `BD = 32` (the host's formulas, 132 SMs). -/
def setup_gptj_packed_b1_t777_h2_d64_r32_n9_i32 : Inst where
  idx := .i32
  B := 1
  T := 777
  H := 2
  D := 64
  R := 16
  TR := 777
  BT := 8
  BD := 32
  N := 9
  NT := 103
  off := 0
  isVarlen := true
  interleaved := true
  x := ⟨1099511627776, 99456⟩
  cos := ⟨2199023255552, 12432⟩
  sin := ⟨3298534883328, 12432⟩
  y := ⟨4398046511104, 99456⟩
  cu := ⟨5497558138880, 10⟩
  ci := ⟨6597069766656, 206⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_gptj_packed_b1_t777_h2_d64_r32_n9_i32_ok : setup_gptj_packed_b1_t777_h2_d64_r32_n9_i32.ok :=
  ⟨Or.inl rfl,
    ⟨3, rfl⟩,
    ⟨5, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- the host test case (source lines 323-337): `B = 1`, `T = 5`, `H = 2`, `D = 128`, rotary_dim `128`, offset `0`, packed, `N = 5` sequences (`cu_seqlens` i64), `NT = 5` chunks, GPT-J; `BT = 1`, `BD = 128` (the host's formulas, 132 SMs). -/
def setup_gptj_packed_b1_t5_h2_d128_r128_n5_i64 : Inst where
  idx := .i64
  B := 1
  T := 5
  H := 2
  D := 128
  R := 64
  TR := 5
  BT := 1
  BD := 128
  N := 5
  NT := 5
  off := 0
  isVarlen := true
  interleaved := true
  x := ⟨1099511627776, 1280⟩
  cos := ⟨2199023255552, 320⟩
  sin := ⟨3298534883328, 320⟩
  y := ⟨4398046511104, 1280⟩
  cu := ⟨5497558138880, 6⟩
  ci := ⟨6597069766656, 10⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_gptj_packed_b1_t5_h2_d128_r128_n5_i64_ok : setup_gptj_packed_b1_t5_h2_d128_r128_n5_i64.ok :=
  ⟨Or.inr rfl,
    ⟨0, rfl⟩,
    ⟨7, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- the host test case (source lines 323-337): `B = 4`, `T = 8192`, `H = 32`, `D = 128`, rotary_dim `128`, offset `0`, padded, GPT-J; `BT = 64`, `BD = 128` (the host's formulas, 132 SMs). -/
def setup_gptj_padded_b4_t8192_h32_d128_r128 : Inst where
  idx := .i32
  B := 4
  T := 8192
  H := 32
  D := 128
  R := 64
  TR := 8192
  BT := 64
  BD := 128
  N := 4
  NT := 128
  off := 0
  isVarlen := false
  interleaved := true
  x := ⟨1099511627776, 134217728⟩
  cos := ⟨2199023255552, 524288⟩
  sin := ⟨3298534883328, 524288⟩
  y := ⟨4398046511104, 134217728⟩
  cu := ⟨5497558138880, 0⟩
  ci := ⟨6597069766656, 0⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_gptj_padded_b4_t8192_h32_d128_r128_ok : setup_gptj_padded_b4_t8192_h32_d128_r128.ok :=
  ⟨Or.inl rfl,
    ⟨6, rfl⟩,
    ⟨7, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- the host test case (source lines 323-337): `B = 1`, `T = 32768`, `H = 32`, `D = 128`, rotary_dim `128`, offset `0`, packed, `N = 16` sequences (`cu_seqlens` i32), `NT = 262` chunks, GPT-J; `BT = 128`, `BD = 128` (the host's formulas, 132 SMs). -/
def setup_gptj_packed_b1_t32768_h32_d128_r128_n16_i32 : Inst where
  idx := .i32
  B := 1
  T := 32768
  H := 32
  D := 128
  R := 64
  TR := 32768
  BT := 128
  BD := 128
  N := 16
  NT := 262
  off := 0
  isVarlen := true
  interleaved := true
  x := ⟨1099511627776, 134217728⟩
  cos := ⟨2199023255552, 2097152⟩
  sin := ⟨3298534883328, 2097152⟩
  y := ⟨4398046511104, 134217728⟩
  cu := ⟨5497558138880, 17⟩
  ci := ⟨6597069766656, 524⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_gptj_packed_b1_t32768_h32_d128_r128_n16_i32_ok : setup_gptj_packed_b1_t32768_h32_d128_r128_n16_i32.ok :=
  ⟨Or.inl rfl,
    ⟨7, rfl⟩,
    ⟨7, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- decoding: one token per sequence at the cache length 4095 (fla/layers/attn.py:113-115): `B = 8`, `T = 1`, `H = 32`, `D = 128`, rotary_dim `128`, offset `4095`, padded, GPT-NeoX; `BT = 1`, `BD = 128` (the host's formulas, 132 SMs). -/
def setup_neox_padded_b8_t1_h32_d128_r128_off4095 : Inst where
  idx := .i32
  B := 8
  T := 1
  H := 32
  D := 128
  R := 64
  TR := 4096
  BT := 1
  BD := 128
  N := 8
  NT := 1
  off := 4095
  isVarlen := false
  interleaved := false
  x := ⟨1099511627776, 32768⟩
  cos := ⟨2199023255552, 262144⟩
  sin := ⟨3298534883328, 262144⟩
  y := ⟨4398046511104, 32768⟩
  cu := ⟨5497558138880, 0⟩
  ci := ⟨6597069766656, 0⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_neox_padded_b8_t1_h32_d128_r128_off4095_ok : setup_neox_padded_b8_t1_h32_d128_r128_off4095.ok :=
  ⟨Or.inl rfl,
    ⟨0, rfl⟩,
    ⟨7, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- decoding, GPT-J: `B = 8`, `T = 1`, `H = 32`, `D = 128`, rotary_dim `128`, offset `4095`, padded, GPT-J; `BT = 1`, `BD = 128` (the host's formulas, 132 SMs). -/
def setup_gptj_padded_b8_t1_h32_d128_r128_off4095 : Inst where
  idx := .i32
  B := 8
  T := 1
  H := 32
  D := 128
  R := 64
  TR := 4096
  BT := 1
  BD := 128
  N := 8
  NT := 1
  off := 4095
  isVarlen := false
  interleaved := true
  x := ⟨1099511627776, 32768⟩
  cos := ⟨2199023255552, 262144⟩
  sin := ⟨3298534883328, 262144⟩
  y := ⟨4398046511104, 32768⟩
  cu := ⟨5497558138880, 0⟩
  ci := ⟨6597069766656, 0⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_gptj_padded_b8_t1_h32_d128_r128_off4095_ok : setup_gptj_padded_b8_t1_h32_d128_r128_off4095.ok :=
  ⟨Or.inl rfl,
    ⟨0, rfl⟩,
    ⟨7, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- fla's Transformer config: hidden 2048, 32 heads, head dim 64: `B = 4`, `T = 2048`, `H = 32`, `D = 64`, rotary_dim `64`, offset `0`, padded, GPT-NeoX; `BT = 16`, `BD = 64` (the host's formulas, 132 SMs). -/
def setup_neox_padded_b4_t2048_h32_d64_r64 : Inst where
  idx := .i32
  B := 4
  T := 2048
  H := 32
  D := 64
  R := 32
  TR := 2048
  BT := 16
  BD := 64
  N := 4
  NT := 128
  off := 0
  isVarlen := false
  interleaved := false
  x := ⟨1099511627776, 16777216⟩
  cos := ⟨2199023255552, 65536⟩
  sin := ⟨3298534883328, 65536⟩
  y := ⟨4398046511104, 16777216⟩
  cu := ⟨5497558138880, 0⟩
  ci := ⟨6597069766656, 0⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_neox_padded_b4_t2048_h32_d64_r64_ok : setup_neox_padded_b4_t2048_h32_d64_r64.ok :=
  ⟨Or.inl rfl,
    ⟨4, rfl⟩,
    ⟨6, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- one key head (tests: 1 key head): `B = 2`, `T = 2048`, `H = 1`, `D = 128`, rotary_dim `128`, offset `0`, padded, GPT-J; `BT = 16`, `BD = 128` (the host's formulas, 132 SMs). -/
def setup_gptj_padded_b2_t2048_h1_d128_r128 : Inst where
  idx := .i32
  B := 2
  T := 2048
  H := 1
  D := 128
  R := 64
  TR := 2048
  BT := 16
  BD := 128
  N := 2
  NT := 128
  off := 0
  isVarlen := false
  interleaved := true
  x := ⟨1099511627776, 524288⟩
  cos := ⟨2199023255552, 131072⟩
  sin := ⟨3298534883328, 131072⟩
  y := ⟨4398046511104, 524288⟩
  cu := ⟨5497558138880, 0⟩
  ci := ⟨6597069766656, 0⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_gptj_padded_b2_t2048_h1_d128_r128_ok : setup_gptj_padded_b2_t2048_h1_d128_r128.ok :=
  ⟨Or.inl rfl,
    ⟨4, rfl⟩,
    ⟨7, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- partial rotary, rotary_dim 64 of D 256 (tests/modules/test_rotary.py:165-172): `B = 2`, `T = 2048`, `H = 4`, `D = 256`, rotary_dim `64`, offset `0`, padded, GPT-NeoX; `BT = 16`, `BD = 64` (the host's formulas, 132 SMs). -/
def setup_neox_padded_b2_t2048_h4_d256_r64 : Inst where
  idx := .i32
  B := 2
  T := 2048
  H := 4
  D := 256
  R := 32
  TR := 2048
  BT := 16
  BD := 64
  N := 2
  NT := 128
  off := 0
  isVarlen := false
  interleaved := false
  x := ⟨1099511627776, 4194304⟩
  cos := ⟨2199023255552, 65536⟩
  sin := ⟨3298534883328, 65536⟩
  y := ⟨4398046511104, 4194304⟩
  cu := ⟨5497558138880, 0⟩
  ci := ⟨6597069766656, 0⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_neox_padded_b2_t2048_h4_d256_r64_ok : setup_neox_padded_b2_t2048_h4_d256_r64.ok :=
  ⟨Or.inl rfl,
    ⟨4, rfl⟩,
    ⟨6, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- partial rotary, GPT-J: `B = 2`, `T = 2048`, `H = 4`, `D = 256`, rotary_dim `64`, offset `0`, padded, GPT-J; `BT = 16`, `BD = 64` (the host's formulas, 132 SMs). -/
def setup_gptj_padded_b2_t2048_h4_d256_r64 : Inst where
  idx := .i32
  B := 2
  T := 2048
  H := 4
  D := 256
  R := 32
  TR := 2048
  BT := 16
  BD := 64
  N := 2
  NT := 128
  off := 0
  isVarlen := false
  interleaved := true
  x := ⟨1099511627776, 4194304⟩
  cos := ⟨2199023255552, 65536⟩
  sin := ⟨3298534883328, 65536⟩
  y := ⟨4398046511104, 4194304⟩
  cu := ⟨5497558138880, 0⟩
  ci := ⟨6597069766656, 0⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_gptj_padded_b2_t2048_h4_d256_r64_ok : setup_gptj_padded_b2_t2048_h4_d256_r64.ok :=
  ⟨Or.inl rfl,
    ⟨4, rfl⟩,
    ⟨6, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩

/-- one packed sequence, int64 cu_seqlens: `B = 1`, `T = 4096`, `H = 4`, `D = 128`, rotary_dim `128`, offset `0`, packed, `N = 1` sequences (`cu_seqlens` i64), `NT = 128` chunks, GPT-NeoX; `BT = 32`, `BD = 128` (the host's formulas, 132 SMs). -/
def setup_neox_packed_b1_t4096_h4_d128_r128_n1_i64 : Inst where
  idx := .i64
  B := 1
  T := 4096
  H := 4
  D := 128
  R := 64
  TR := 4096
  BT := 32
  BD := 128
  N := 1
  NT := 128
  off := 0
  isVarlen := true
  interleaved := false
  x := ⟨1099511627776, 2097152⟩
  cos := ⟨2199023255552, 262144⟩
  sin := ⟨3298534883328, 262144⟩
  y := ⟨4398046511104, 2097152⟩
  cu := ⟨5497558138880, 2⟩
  ci := ⟨6597069766656, 256⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

theorem setup_neox_packed_b1_t4096_h4_d128_r128_n1_i64_ok : setup_neox_packed_b1_t4096_h4_d128_r128_n1_i64.ok :=
  ⟨Or.inr rfl,
    ⟨5, rfl⟩,
    ⟨7, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun h => absurd h (by decide),
    by decide,
    bytes_disj (by decide)⟩
/-- batched generation's prefill under left padding (`fla/layers/attn.py:117-119`): `B = 4`, `T = 512`, `H = 32`, `D = 64`, rotary_dim `64`, a tensor of `int32` offsets (`-pad`, negative), `TR = 2048`, padded, GPT-NeoX; `BT = 4`, `BD = 64` (the host's formulas, 132 SMs). -/
def setup_neox_padded_b4_t512_h32_d64_r64_offt_i32 : Inst where
  idx := .i32
  B := 4
  T := 512
  H := 32
  D := 64
  R := 32
  TR := 2048
  BT := 4
  BD := 64
  N := 4
  NT := 128
  off := 0
  isVarlen := false
  interleaved := false
  x := ⟨1099511627776, 4194304⟩
  cos := ⟨2199023255552, 65536⟩
  sin := ⟨3298534883328, 65536⟩
  y := ⟨4398046511104, 4194304⟩
  cu := ⟨5497558138880, 0⟩
  ci := ⟨6597069766656, 0⟩
  offT := true
  offD := .i32
  offs := ⟨7696581394432, 4⟩

theorem setup_neox_padded_b4_t512_h32_d64_r64_offt_i32_ok : setup_neox_padded_b4_t512_h32_d64_r64_offt_i32.ok :=
  ⟨Or.inl rfl,
    ⟨2, rfl⟩,
    ⟨6, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun _ => Or.inl rfl,
    by decide,
    bytes_disj (by decide)⟩

/-- batched generation's decoding step under left padding (`fla/layers/attn.py:117-119`): `B = 8`, `T = 1`, `H = 32`, `D = 64`, rotary_dim `64`, a tensor of `int32` offsets (the cache's length minus the padding), `TR = 2048`, padded, GPT-NeoX; `BT = 1`, `BD = 64` (the host's formulas, 132 SMs). -/
def setup_neox_padded_b8_t1_h32_d64_r64_offt_i32 : Inst where
  idx := .i32
  B := 8
  T := 1
  H := 32
  D := 64
  R := 32
  TR := 2048
  BT := 1
  BD := 64
  N := 8
  NT := 1
  off := 0
  isVarlen := false
  interleaved := false
  x := ⟨1099511627776, 16384⟩
  cos := ⟨2199023255552, 65536⟩
  sin := ⟨3298534883328, 65536⟩
  y := ⟨4398046511104, 16384⟩
  cu := ⟨5497558138880, 0⟩
  ci := ⟨6597069766656, 0⟩
  offT := true
  offD := .i32
  offs := ⟨7696581394432, 8⟩

theorem setup_neox_padded_b8_t1_h32_d64_r64_offt_i32_ok : setup_neox_padded_b8_t1_h32_d64_r64_offt_i32.ok :=
  ⟨Or.inl rfl,
    ⟨0, rfl⟩,
    ⟨6, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun _ => Or.inl rfl,
    by decide,
    bytes_disj (by decide)⟩

/-- batched generation's prefill under left padding (`fla/layers/attn.py:117-119`): `B = 4`, `T = 512`, `H = 32`, `D = 64`, rotary_dim `64`, a tensor of `int32` offsets (`-pad`, negative), `TR = 2048`, padded, GPT-J; `BT = 4`, `BD = 64` (the host's formulas, 132 SMs). -/
def setup_gptj_padded_b4_t512_h32_d64_r64_offt_i32 : Inst where
  idx := .i32
  B := 4
  T := 512
  H := 32
  D := 64
  R := 32
  TR := 2048
  BT := 4
  BD := 64
  N := 4
  NT := 128
  off := 0
  isVarlen := false
  interleaved := true
  x := ⟨1099511627776, 4194304⟩
  cos := ⟨2199023255552, 65536⟩
  sin := ⟨3298534883328, 65536⟩
  y := ⟨4398046511104, 4194304⟩
  cu := ⟨5497558138880, 0⟩
  ci := ⟨6597069766656, 0⟩
  offT := true
  offD := .i32
  offs := ⟨7696581394432, 4⟩

theorem setup_gptj_padded_b4_t512_h32_d64_r64_offt_i32_ok : setup_gptj_padded_b4_t512_h32_d64_r64_offt_i32.ok :=
  ⟨Or.inl rfl,
    ⟨2, rfl⟩,
    ⟨6, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun _ => Or.inl rfl,
    by decide,
    bytes_disj (by decide)⟩

/-- batched generation's decoding step under left padding (`fla/layers/attn.py:117-119`): `B = 8`, `T = 1`, `H = 32`, `D = 64`, rotary_dim `64`, a tensor of `int32` offsets (the cache's length minus the padding), `TR = 2048`, padded, GPT-J; `BT = 1`, `BD = 64` (the host's formulas, 132 SMs). -/
def setup_gptj_padded_b8_t1_h32_d64_r64_offt_i32 : Inst where
  idx := .i32
  B := 8
  T := 1
  H := 32
  D := 64
  R := 32
  TR := 2048
  BT := 1
  BD := 64
  N := 8
  NT := 1
  off := 0
  isVarlen := false
  interleaved := true
  x := ⟨1099511627776, 16384⟩
  cos := ⟨2199023255552, 65536⟩
  sin := ⟨3298534883328, 65536⟩
  y := ⟨4398046511104, 16384⟩
  cu := ⟨5497558138880, 0⟩
  ci := ⟨6597069766656, 0⟩
  offT := true
  offD := .i32
  offs := ⟨7696581394432, 8⟩

theorem setup_gptj_padded_b8_t1_h32_d64_r64_offt_i32_ok : setup_gptj_padded_b8_t1_h32_d64_r64_offt_i32.ok :=
  ⟨Or.inl rfl,
    ⟨0, rfl⟩,
    ⟨6, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun _ => Or.inl rfl,
    by decide,
    bytes_disj (by decide)⟩

/-- a tensor of `int64` offsets (the host's other dtype, line 227): `B = 2`, `T = 2048`, `H = 4`, `D = 128`, rotary_dim `128`, `TR = 4096`, padded, GPT-NeoX; `BT = 16`, `BD = 128` (the host's formulas, 132 SMs). -/
def setup_neox_padded_b2_t2048_h4_d128_r128_offt_i64 : Inst where
  idx := .i32
  B := 2
  T := 2048
  H := 4
  D := 128
  R := 64
  TR := 4096
  BT := 16
  BD := 128
  N := 2
  NT := 128
  off := 0
  isVarlen := false
  interleaved := false
  x := ⟨1099511627776, 2097152⟩
  cos := ⟨2199023255552, 262144⟩
  sin := ⟨3298534883328, 262144⟩
  y := ⟨4398046511104, 2097152⟩
  cu := ⟨5497558138880, 0⟩
  ci := ⟨6597069766656, 0⟩
  offT := true
  offD := .i64
  offs := ⟨7696581394432, 2⟩

theorem setup_neox_padded_b2_t2048_h4_d128_r128_offt_i64_ok : setup_neox_padded_b2_t2048_h4_d128_r128_offt_i64.ok :=
  ⟨Or.inl rfl,
    ⟨4, rfl⟩,
    ⟨7, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun _ => Or.inr rfl,
    by decide,
    bytes_disj (by decide)⟩

/-- packed sequences with a tensor of `int64` offsets: `T = 777`, `H = 2`, `D = 64`, rotary_dim `32`, `cu_seqlens = [0, 100, 300, 777]` `int32`, `TR = 1024`, GPT-NeoX; `BT = 8`, `BD = 32` (the host's formulas, 132 SMs). -/
def setup_neox_packed_b1_t777_h2_d64_r32_n3_i32_offt_i64 : Inst where
  idx := .i32
  B := 1
  T := 777
  H := 2
  D := 64
  R := 16
  TR := 1024
  BT := 8
  BD := 32
  N := 3
  NT := 98
  off := 0
  isVarlen := true
  interleaved := false
  x := ⟨1099511627776, 99456⟩
  cos := ⟨2199023255552, 16384⟩
  sin := ⟨3298534883328, 16384⟩
  y := ⟨4398046511104, 99456⟩
  cu := ⟨5497558138880, 4⟩
  ci := ⟨6597069766656, 196⟩
  offT := true
  offD := .i64
  offs := ⟨7696581394432, 3⟩

theorem setup_neox_packed_b1_t777_h2_d64_r32_n3_i32_offt_i64_ok : setup_neox_packed_b1_t777_h2_d64_r32_n3_i32_offt_i64.ok :=
  ⟨Or.inl rfl,
    ⟨3, rfl⟩,
    ⟨5, rfl⟩,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    by decide,
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    bytes_disj (by decide),
    by decide,
    fun _ => Or.inr rfl,
    by decide,
    bytes_disj (by decide)⟩

/-! ## Satisfiable -/

/-- Some covered instance and input exist: the host's one-token case with every byte `0`. -/
theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  have hx : setup_neox_padded_b1_t1_h1_d64_r64.x.Valid zeroMem :=
    zeroMem_valid (by norm_num [setup_neox_padded_b1_t1_h1_d64_r64])
      (by norm_num [setup_neox_padded_b1_t1_h1_d64_r64, DType.bytes])
      (by norm_num [setup_neox_padded_b1_t1_h1_d64_r64, DType.bytes])
  have hy : setup_neox_padded_b1_t1_h1_d64_r64.y.Valid zeroMem :=
    zeroMem_valid (by norm_num [setup_neox_padded_b1_t1_h1_d64_r64])
      (by norm_num [setup_neox_padded_b1_t1_h1_d64_r64, DType.bytes])
      (by norm_num [setup_neox_padded_b1_t1_h1_d64_r64, DType.bytes])
  have hc : setup_neox_padded_b1_t1_h1_d64_r64.cos.Valid zeroMem :=
    zeroMem_valid (by norm_num [setup_neox_padded_b1_t1_h1_d64_r64])
      (by norm_num [setup_neox_padded_b1_t1_h1_d64_r64, DType.bytes])
      (by norm_num [setup_neox_padded_b1_t1_h1_d64_r64, DType.bytes])
  have hs : setup_neox_padded_b1_t1_h1_d64_r64.sin.Valid zeroMem :=
    zeroMem_valid (by norm_num [setup_neox_padded_b1_t1_h1_d64_r64])
      (by norm_num [setup_neox_padded_b1_t1_h1_d64_r64, DType.bytes])
      (by norm_num [setup_neox_padded_b1_t1_h1_d64_r64, DType.bytes])
  refine ⟨setup_neox_padded_b1_t1_h1_d64_r64, zeroMem, setup_neox_padded_b1_t1_h1_d64_r64_ok,
    ⟨hx, hy, hc, hs, ?_, ?_, ?_, fun h => absurd h (by decide), fun h => absurd h (by decide)⟩⟩
  · intro j hj _
    exact ⟨_, zeroMem_bfAt hx (lt_of_lt_of_le hj setup_neox_padded_b1_t1_h1_d64_r64_ok.x_len), zero_xok⟩
  · intro n l k hn hl _ hk
    have h : ((l : Int) + offAt setup_neox_padded_b1_t1_h1_d64_r64 zeroMem n).toNat *
        setup_neox_padded_b1_t1_h1_d64_r64.R + k < setup_neox_padded_b1_t1_h1_d64_r64.cos.len := by
      simp only [setup_neox_padded_b1_t1_h1_d64_r64, offAt, lenAt, nSeq] at hn hl hk ⊢; simp at hl ⊢; omega
    exact ⟨_, zeroMem_bfAt hc h, zero_tabok⟩
  · intro n l k hn hl _ hk
    have h : ((l : Int) + offAt setup_neox_padded_b1_t1_h1_d64_r64 zeroMem n).toNat *
        setup_neox_padded_b1_t1_h1_d64_r64.R + k < setup_neox_padded_b1_t1_h1_d64_r64.sin.len := by
      simp only [setup_neox_padded_b1_t1_h1_d64_r64, offAt, lenAt, nSeq] at hn hl hk ⊢; simp at hl ⊢; omega
    exact ⟨_, zeroMem_bfAt hs h, zero_tabok⟩

end Claims.RotaryVarlen

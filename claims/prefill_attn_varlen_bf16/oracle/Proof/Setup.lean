import Claim
import TVA.Proof
import Proof.NumBase
import Proof.NumStep

section
/-!
# Program footprints and post-condition

Program `i` of the grid `(batch, num_heads, cdiv(max_input_len, 128))` is the triple
`(b, h, sm)` = `(i % batch, i / batch % H, i / (batch·H))`: sequence `b`, query head `h`, query
block `sm`. It reads the inputs and writes the `128 × Lk` tile of `o` of its block, cut at the
sequence's length (`tile`); `Q` says each of its output elements is within the claim's tolerance of
the exact attention.
-/

namespace Claims.PrefillAttnVarlen

open TVA

/-! ## The grid -/

/-- The sequence of program `i`. -/
def bOf (d : Inst) (i : Nat) : Nat := i % d.batch
/-- The query head of program `i`. -/
def hOf (d : Inst) (i : Nat) : Nat := i / d.batch % d.H
/-- The query block of program `i`. -/
def mOf (d : Inst) (i : Nat) : Nat := i / (d.batch * d.H)

/-- The launch has one program per batch, head and query block. -/
theorem launch_size (d : Inst) : (launch d).size = d.batch * d.H * nBlocks d := rfl

/-- Program axis 0 of program `i` is its batch index. -/
theorem launch_pid0 (d : Inst) (i : Nat) : ((launch d).ctx i).pid 0 = bOf d i := rfl
/-- Program axis 1 of program `i` is its head index. -/
theorem launch_pid1 (d : Inst) (i : Nat) : ((launch d).ctx i).pid 1 = hOf d i := rfl
/-- Program axis 2 of program `i` is its query block. -/
theorem launch_pid2 (d : Inst) (i : Nat) : ((launch d).ctx i).pid 2 = mOf d i := rfl

/-- `BLOCK = 128`. -/
theorem BLOCK_eq : BLOCK = 128 := rfl

/-- For a valid instance the number of query blocks is at most `2 ^ 10`. -/
theorem nBlocks_le {d : Inst} (hd : d.ok) : nBlocks d ≤ 2 ^ 10 := by
  have := hd.maxLen_le
  simp only [nBlocks, BLOCK]
  omega

private theorem tritonOk1 {x : Nat} (hx : ∃ e, x = 2 ^ e) (h : x ≤ 2 ^ 20) :
    Shape.TritonOk [x] := by
  refine ⟨?_, by simpa using h⟩
  intro k hk
  simp only [List.mem_singleton] at hk
  subst hk; exact hx

private theorem tritonOk2 {x y : Nat} (hx : ∃ e, x = 2 ^ e) (hy : ∃ e, y = 2 ^ e)
    (h : x * y ≤ 2 ^ 20) : Shape.TritonOk [x, y] := by
  refine ⟨?_, by simpa using h⟩
  intro k hk
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
  rcases hk with rfl | rfl
  exacts [hx, hy]

/-- The shapes of the launch are valid Triton block shapes. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  obtain ⟨a, ha⟩ := hd.bd_pow2
  have hbd := hd.bd_le
  have hnb := nBlocks_le hd
  refine ⟨by simpa [launch] using hd.batch_lt, by simpa [launch] using hd.H_lt, ?_, ?_,
    rfl, rfl, rfl, ?_⟩
  · simp only [launch]; show nBlocks d < 2 ^ 16; omega
  · rw [launch_size]; exact hd.grid_lt
  · have h128 : ∃ e, BLOCK = 2 ^ e := ⟨7, by norm_num [BLOCK]⟩
    have h1 : ∃ e, (1 : Nat) = 2 ^ e := ⟨0, by norm_num⟩
    have hbdp : ∃ e, d.BD = 2 ^ e := ⟨a, ha⟩
    have hB : BLOCK = 128 := rfl
    have hBD : 1 ≤ d.BD := by rw [ha]; exact Nat.one_le_two_pow
    intro s hs
    simp only [launch, List.mem_cons, List.not_mem_nil, or_false] at hs
    rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · exact tritonOk1 h128 (by rw [hB]; norm_num)
    · exact tritonOk1 hbdp (by omega)
    · exact tritonOk2 h128 hbdp (by rw [hB]; omega)
    · exact tritonOk2 hbdp h128 (by rw [hB]; omega)
    · exact tritonOk2 h128 h128 (by rw [hB]; norm_num)
    · exact tritonOk2 h128 h1 (by rw [hB]; norm_num)
    · exact tritonOk2 h1 h128 (by rw [hB]; norm_num)
    · exact tritonOk2 h1 hbdp (by omega)
    · exact tritonOk2 hbdp h1 (by omega)

/-! ## Footprints -/

/-- Program `i` reads (parts of) the five inputs. -/
def Rf (d : Inst) : Set Int :=
  d.q.bytes ∪ d.k.bytes ∪ d.v.bytes ∪ d.b_start_loc.bytes ∪ d.b_seq_len.bytes

/-- The elements of `o` of the rows `sm·128 + r` (`r < 128`, below the sequence's length) of
sequence `b`, head `h`, all `Lk` components. -/
def tile (d : Inst) (init : Mem) (b h sm : Nat) : Set Nat :=
  {x | ∃ r e, r < BLOCK ∧ e < d.Lk ∧ sm * BLOCK + r < lenOf d init b ∧
    x = oIdx d (startOf d init b + (sm * BLOCK + r)) h e}

/-- Program `i` writes its tile of `o`. -/
def Wf (d : Inst) (init : Mem) (i : Nat) : Set Int :=
  d.o.elems (tile d init (bOf d i) (hOf d i) (mOf d i))

/-- **What a program leaves**: each element of its tile of `o` holds a finite `bfloat16` within
the claim's tolerance of the exact attention. -/
def Q (d : Inst) (init : Mem) (i : Nat) (m : Mem) : Prop :=
  ∀ r e, r < BLOCK → e < d.Lk → mOf d i * BLOCK + r < lenOf d init (bOf d i) →
    ∃ c : BF16, rdB m d.o (oIdx d (startOf d init (bOf d i) + (mOf d i * BLOCK + r)) (hOf d i) e) =
        some c ∧ c.isFinite = true ∧
      |c.toReal - attnRef d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r) e| ≤
        tolRel d.BD (trips (lastKey d init (bOf d i) (mOf d i * BLOCK + r)))
            (Xrow d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r)) *
          attnAbs d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r) e +
        tolAbs (vSum d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r) e)

end Claims.PrefillAttnVarlen

end

section
/-!
# The kernel body, cut into named pieces

`restProg` is the body after the five scalar steps; `afterQ` is what follows the load of `q`;
`loopBody` is one trip of the loop (lines 122-174 of the source) and `postProg` the epilogue (the
store). The text is the model's, unchanged: `body_eq` is `rfl`. Walking a piece is then a lemma
about a name rather than a copy of the model.
-/

namespace Claims.PrefillAttnVarlen

open TVA

/-- One trip of the loop: lines 122-174, for the carried state `st3 = (m_i, l_i, acc)`. -/
def loopBody (_Q K V _Out : Ptr .bf16) (sm_scale : F32)
    (_B_Start_Loc _B_Seqlen : Ptr .i32)
    (stride_qbs stride_qh stride_kbs stride_kh stride_vbs stride_vh _stride_obs _stride_oh : I32)
    (_kv_group_num BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) (IS_CAUSAL : Bool) (Lk : Nat)
    (_cur_batch cur_head start_m cur_kv_head cur_batch_seq_len cur_batch_in_all_start_index : I32)
    (q : Tensor [BLOCK_M, BLOCK_DMODEL] BF16) (start_n : I32)
    (st3 : Tensor [BLOCK_M] F32 × Tensor [BLOCK_M] F32 × Tensor [BLOCK_M, BLOCK_DMODEL] F32) :
    Prog (Tensor [BLOCK_M] F32 × Tensor [BLOCK_M] F32 × Tensor [BLOCK_M, BLOCK_DMODEL] F32) := do
    let _block_start_loc := (IntV.ofNat BLOCK_M : I32) * start_m
    -- lines 87–89
    let offs_n : Tensor [BLOCK_N] I32 := tl.arange 0 BLOCK_N
    let offs_d : Tensor [BLOCK_DMODEL] I32 := tl.arange 0 BLOCK_DMODEL
    let offs_m : Tensor [BLOCK_M] I32 :=
      start_m * (IntV.ofNat BLOCK_M : I32) + tl.arange 0 BLOCK_M
    -- lines 90–94: the `int32` products and sums in the source's order, each term broadcast to
    -- the final shape (`x[:, None]` is `.col`, `x[None, :]` is `.row`, a scalar `bc`)
    let _off_q : Tensor [BLOCK_M, BLOCK_DMODEL] I32 :=
      (cur_batch_in_all_start_index + offs_m.col BLOCK_DMODEL) * stride_qbs +
        bc (cur_head * stride_qh) + offs_d.row BLOCK_M
    -- lines 95–96
    let off_k : Tensor [BLOCK_DMODEL, BLOCK_N] I32 :=
      offs_n.row BLOCK_DMODEL * stride_kbs +
        bc (cur_kv_head * stride_kh) + offs_d.col BLOCK_N
    let off_v : Tensor [BLOCK_N, BLOCK_DMODEL] I32 :=
      offs_n.col BLOCK_DMODEL * stride_vbs +
        bc (cur_kv_head * stride_vh) + offs_d.row BLOCK_N
    -- line 98: the constexpr `Lk` built in `int32`, signed comparison
    let mask_d : Tensor [BLOCK_DMODEL] Bool := offs_d.ltS (IntV.ofNat Lk : I32)
    let k_ptrs := K + off_k
    let v_ptrs := V + off_v
    let (m_i, l_i, acc) := st3
    -- line 122: a hint on the loop's induction variable: no effect (not modelled)
    -- lines 124–128: the scalar `int32` offset broadcast; `other=0.0` converted
    let k : Tensor [BLOCK_DMODEL, BLOCK_N] BF16 ←
      tl.load (k_ptrs + (bc ((cur_batch_in_all_start_index + start_n) *
          stride_kbs) : Tensor [BLOCK_DMODEL, BLOCK_N] I32))
        (some ((((start_n + offs_n).row BLOCK_DMODEL).ltS cur_batch_seq_len).and
          (mask_d.col BLOCK_N)))
        (some (← tl.convert .f32 .bf16 (tl.full [BLOCK_DMODEL, BLOCK_N] zf32)))
    -- line 131
    let qk₀ := tl.zeros [BLOCK_M, BLOCK_N] .f32
    -- line 132: `qk += tl.dot(q, k)`: a zero default accumulator, the product used once:
    -- the combine pass's `tl.dot_add`
    let qk₁ : Tensor [BLOCK_M, BLOCK_N] F32 ← tl.dot_add cfgBF16 q k qk₀
    -- lines 133–145: `qk *= sm_scale` feeds `qk += tl.where(cond, 0, float("-inf"))` of the
    -- branch the compile-time `if IS_CAUSAL:` (line 135) visits: one fusable multiply-add;
    -- the `tl.where` values built in `float32`. Lines 136–141: the key mask and the causal
    -- mask; lines 143–145: the key mask alone, a `[1, BLOCK_N]` select that the `+=`
    -- broadcasts to `[BLOCK_M, BLOCK_N]` (written at the broadcast shape: the same values)
    let qk : Tensor [BLOCK_M, BLOCK_N] F32 ←
      tl.mul_add qk₁ (bc sm_scale)
        (if IS_CAUSAL then
          tl.where
            ((((start_n + offs_n).row BLOCK_M).ltS cur_batch_seq_len).and
              (Tensor.le ((start_n + offs_n).row BLOCK_M) (offs_m.col BLOCK_N)))
            (bc zf32) (bc ninf32)
        else
          tl.where (((start_n + offs_n).row BLOCK_M).ltS cur_batch_seq_len)
            (bc zf32) (bc ninf32))
    -- line 148: the row maximum
    let m_ij : Tensor [BLOCK_M] F32 ← tl.max1 true qk
    -- line 149: the IEEE `-` (no product feeds it), then `tl.exp`
    let p : Tensor [BLOCK_M, BLOCK_N] F32 ← tl.exp (← tl.sub qk (m_ij.col BLOCK_N))
    -- line 150: `p` is not a product: the IEEE `tl.sum1`
    let l_ij : Tensor [BLOCK_M] F32 ← tl.sum1 p
    -- line 152
    let m_i_new : Tensor [BLOCK_M] F32 ← tl.maximum m_i m_ij
    -- line 153: the IEEE `-`, then `tl.exp`
    let alpha : Tensor [BLOCK_M] F32 ← tl.exp (← tl.sub m_i m_i_new)
    -- line 154
    let beta : Tensor [BLOCK_M] F32 ← tl.exp (← tl.sub m_ij m_i_new)
    -- line 155: `alpha * l_i + beta * l_ij`: a product on each side: `tl.addc`
    let l_i_new : Tensor [BLOCK_M] F32 ←
      tl.addc (← tl.mul alpha l_i) (← tl.mul beta l_ij)
    -- line 158: the platform's `float32` division
    let p_scale : Tensor [BLOCK_M] F32 ← tl.div beta l_i_new
    -- line 159: the IEEE `*`
    let p₁ : Tensor [BLOCK_M, BLOCK_N] F32 ← tl.mul p (p_scale.col BLOCK_N)
    -- line 161: `(l_i / l_i_new) * alpha`
    let acc_scale : Tensor [BLOCK_M] F32 ← tl.mul (← tl.div l_i l_i_new) alpha
    -- line 162: the IEEE `*`
    let acc₁ : Tensor [BLOCK_M, BLOCK_DMODEL] F32 ← tl.mul acc (acc_scale.col BLOCK_DMODEL)
    -- lines 164–168
    let v : Tensor [BLOCK_N, BLOCK_DMODEL] BF16 ←
      tl.load (v_ptrs + (bc ((cur_batch_in_all_start_index + start_n) *
          stride_vbs) : Tensor [BLOCK_N, BLOCK_DMODEL] I32))
        (some ((((start_n + offs_n).col BLOCK_DMODEL).ltS cur_batch_seq_len).and
          (mask_d.row BLOCK_N)))
        (some (← tl.convert .f32 .bf16 (tl.full [BLOCK_N, BLOCK_DMODEL] zf32)))
    -- line 170: `p.to(v.dtype)`, one rounding to nearest even
    let p₂ : Tensor [BLOCK_M, BLOCK_N] BF16 ← tl.cvt p₁
    -- line 171: `acc += tl.dot(p, v)`: `tl.dot_add`, the accumulator `acc₁`
    let acc₂ : Tensor [BLOCK_M, BLOCK_DMODEL] F32 ← tl.dot_add cfgBF16 p₂ v acc₁
    -- lines 173–174
    pure (m_i_new, l_i_new, acc₂)

/-- The epilogue: the store of the output tile. -/
def postProg (_Q _K _V Out : Ptr .bf16) (_sm_scale : F32)
    (_B_Start_Loc _B_Seqlen : Ptr .i32)
    (stride_qbs stride_qh stride_kbs stride_kh stride_vbs stride_vh stride_obs stride_oh : I32)
    (_kv_group_num BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) (_IS_CAUSAL : Bool) (Lk : Nat)
    (_cur_batch cur_head start_m cur_kv_head cur_batch_seq_len cur_batch_in_all_start_index : I32)
    (acc : Tensor [BLOCK_M, BLOCK_DMODEL] F32) : Prog Unit := do
    let _block_start_loc := (IntV.ofNat BLOCK_M : I32) * start_m
    -- lines 87–89
    let offs_n : Tensor [BLOCK_N] I32 := tl.arange 0 BLOCK_N
    let offs_d : Tensor [BLOCK_DMODEL] I32 := tl.arange 0 BLOCK_DMODEL
    let offs_m : Tensor [BLOCK_M] I32 :=
      start_m * (IntV.ofNat BLOCK_M : I32) + tl.arange 0 BLOCK_M
    -- lines 90–94: the `int32` products and sums in the source's order, each term broadcast to
    -- the final shape (`x[:, None]` is `.col`, `x[None, :]` is `.row`, a scalar `bc`)
    let _off_q : Tensor [BLOCK_M, BLOCK_DMODEL] I32 :=
      (cur_batch_in_all_start_index + offs_m.col BLOCK_DMODEL) * stride_qbs +
        bc (cur_head * stride_qh) + offs_d.row BLOCK_M
    -- lines 95–96
    let _off_k : Tensor [BLOCK_DMODEL, BLOCK_N] I32 :=
      offs_n.row BLOCK_DMODEL * stride_kbs +
        bc (cur_kv_head * stride_kh) + offs_d.col BLOCK_N
    let _off_v : Tensor [BLOCK_N, BLOCK_DMODEL] I32 :=
      offs_n.col BLOCK_DMODEL * stride_vbs +
        bc (cur_kv_head * stride_vh) + offs_d.row BLOCK_N
    -- line 98: the constexpr `Lk` built in `int32`, signed comparison
    let mask_d : Tensor [BLOCK_DMODEL] Bool := offs_d.ltS (IntV.ofNat Lk : I32)
    let off_o : Tensor [BLOCK_M, BLOCK_DMODEL] I32 :=
      (cur_batch_in_all_start_index + offs_m.col BLOCK_DMODEL) * stride_obs +
        bc (cur_head * stride_oh) + offs_d.row BLOCK_M
    -- line 181
    let out_ptrs := Out + off_o
    -- lines 182–184: the store's conversion to the pointer's `bfloat16` (nearest even)
    let o16 : Tensor [BLOCK_M, BLOCK_DMODEL] BF16 ← tl.cvt acc
    tl.store out_ptrs o16
      (some (((offs_m.col BLOCK_DMODEL).ltS cur_batch_seq_len).and (mask_d.row BLOCK_M)))

/-- Everything after the load of `q`: the loop and the epilogue. -/
def afterQ (Q K V Out : Ptr .bf16) (sm_scale : F32)
    (B_Start_Loc B_Seqlen : Ptr .i32)
    (stride_qbs stride_qh stride_kbs stride_kh stride_vbs stride_vh stride_obs stride_oh : I32)
    (kv_group_num BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) (IS_CAUSAL : Bool) (Lk : Nat)
    (cur_batch cur_head start_m cur_kv_head cur_batch_seq_len cur_batch_in_all_start_index : I32)
    (q : Tensor [BLOCK_M, BLOCK_DMODEL] BF16) : Prog Unit := do
    let block_start_loc := (IntV.ofNat BLOCK_M : I32) * start_m
    -- lines 87–89
    let offs_n : Tensor [BLOCK_N] I32 := tl.arange 0 BLOCK_N
    let offs_d : Tensor [BLOCK_DMODEL] I32 := tl.arange 0 BLOCK_DMODEL
    let offs_m : Tensor [BLOCK_M] I32 :=
      start_m * (IntV.ofNat BLOCK_M : I32) + tl.arange 0 BLOCK_M
    -- lines 90–94: the `int32` products and sums in the source's order, each term broadcast to
    -- the final shape (`x[:, None]` is `.col`, `x[None, :]` is `.row`, a scalar `bc`)
    let _off_q : Tensor [BLOCK_M, BLOCK_DMODEL] I32 :=
      (cur_batch_in_all_start_index + offs_m.col BLOCK_DMODEL) * stride_qbs +
        bc (cur_head * stride_qh) + offs_d.row BLOCK_M
    -- lines 95–96
    let off_k : Tensor [BLOCK_DMODEL, BLOCK_N] I32 :=
      offs_n.row BLOCK_DMODEL * stride_kbs +
        bc (cur_kv_head * stride_kh) + offs_d.col BLOCK_N
    let off_v : Tensor [BLOCK_N, BLOCK_DMODEL] I32 :=
      offs_n.col BLOCK_DMODEL * stride_vbs +
        bc (cur_kv_head * stride_vh) + offs_d.row BLOCK_N
    -- line 98: the constexpr `Lk` built in `int32`, signed comparison
    let _mask_d : Tensor [BLOCK_DMODEL] Bool := offs_d.ltS (IntV.ofNat Lk : I32)
    let _k_ptrs := K + off_k
    let _v_ptrs := V + off_v
    -- line 110: `tl.zeros(...) - float("inf")`: the Python number built in `float32`, IEEE `-`
    let m_i : Tensor [BLOCK_M] F32 ← tl.sub (tl.zeros [BLOCK_M] .f32) (bc pinf32)
    -- line 111
    let l_i := tl.zeros [BLOCK_M] .f32
    -- line 112
    let acc := tl.zeros [BLOCK_M, BLOCK_DMODEL] .f32
    -- line 114: the Python `1` and `0` built in `int32`; a scalar `int32` tensor
    let block_mask : Tensor [] I32 :=
      tl.where (Tensor.lt (Tensor.scalar block_start_loc) (Tensor.scalar cur_batch_seq_len))
        (bc (IntV.ofInt 1 : I32)) (bc (IntV.ofInt 0 : I32))
    -- lines 116–120: a conditional expression on the constexpr `not IS_CAUSAL`: only the side it
    -- picks is evaluated: `cur_batch_seq_len`, or `tl.minimum((start_m + 1) * BLOCK_M,
    -- cur_batch_seq_len)`, an exact `int32` minimum
    let end_n : I32 :=
      if !IS_CAUSAL then cur_batch_seq_len
      else
        (tl.iminimum (Tensor.scalar ((start_m + (IntV.ofInt 1 : I32)) * (IntV.ofNat BLOCK_M : I32)))
          (Tensor.scalar cur_batch_seq_len)).item
    -- line 121: `range(0, block_mask * end_n, BLOCK_N)`, the bound the `int32` product,
    -- evaluated once; the loop carries `m_i`, `l_i`, `acc`
    let (_, _, acc) ← tl.forRange (IntV.ofInt 0 : I32)
      ((block_mask * Tensor.scalar end_n).item) (IntV.ofNat BLOCK_N : I32) (m_i, l_i, acc)
      fun start_n st3 => loopBody Q K V Out sm_scale B_Start_Loc B_Seqlen stride_qbs stride_qh stride_kbs stride_kh stride_vbs stride_vh stride_obs stride_oh kv_group_num BLOCK_M BLOCK_DMODEL BLOCK_N IS_CAUSAL Lk cur_batch cur_head start_m cur_kv_head cur_batch_seq_len cur_batch_in_all_start_index q start_n st3
    postProg Q K V Out sm_scale B_Start_Loc B_Seqlen stride_qbs stride_qh stride_kbs stride_kh stride_vbs stride_vh stride_obs stride_oh kv_group_num BLOCK_M BLOCK_DMODEL BLOCK_N IS_CAUSAL Lk cur_batch cur_head start_m cur_kv_head cur_batch_seq_len cur_batch_in_all_start_index acc

/-- Everything after the five scalar steps: the load of `q`, then `afterQ`. -/
def restProg (Q K V Out : Ptr .bf16) (sm_scale : F32)
    (B_Start_Loc B_Seqlen : Ptr .i32)
    (stride_qbs stride_qh stride_kbs stride_kh stride_vbs stride_vh stride_obs stride_oh : I32)
    (kv_group_num BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) (IS_CAUSAL : Bool) (Lk : Nat)
    (cur_batch cur_head start_m cur_kv_head cur_batch_seq_len cur_batch_in_all_start_index : I32) : Prog Unit := do
    let _block_start_loc := (IntV.ofNat BLOCK_M : I32) * start_m
    -- lines 87–89
    let offs_n : Tensor [BLOCK_N] I32 := tl.arange 0 BLOCK_N
    let offs_d : Tensor [BLOCK_DMODEL] I32 := tl.arange 0 BLOCK_DMODEL
    let offs_m : Tensor [BLOCK_M] I32 :=
      start_m * (IntV.ofNat BLOCK_M : I32) + tl.arange 0 BLOCK_M
    -- lines 90–94: the `int32` products and sums in the source's order, each term broadcast to
    -- the final shape (`x[:, None]` is `.col`, `x[None, :]` is `.row`, a scalar `bc`)
    let off_q : Tensor [BLOCK_M, BLOCK_DMODEL] I32 :=
      (cur_batch_in_all_start_index + offs_m.col BLOCK_DMODEL) * stride_qbs +
        bc (cur_head * stride_qh) + offs_d.row BLOCK_M
    -- lines 95–96
    let _off_k : Tensor [BLOCK_DMODEL, BLOCK_N] I32 :=
      offs_n.row BLOCK_DMODEL * stride_kbs +
        bc (cur_kv_head * stride_kh) + offs_d.col BLOCK_N
    let _off_v : Tensor [BLOCK_N, BLOCK_DMODEL] I32 :=
      offs_n.col BLOCK_DMODEL * stride_vbs +
        bc (cur_kv_head * stride_vh) + offs_d.row BLOCK_N
    -- line 98: the constexpr `Lk` built in `int32`, signed comparison
    let mask_d : Tensor [BLOCK_DMODEL] Bool := offs_d.ltS (IntV.ofNat Lk : I32)
    let q : Tensor [BLOCK_M, BLOCK_DMODEL] BF16 ←
      tl.load (Q + off_q)
        (some (((offs_m.col BLOCK_DMODEL).ltS cur_batch_seq_len).and (mask_d.row BLOCK_M)))
        (some (← tl.convert .f32 .bf16 (tl.full [BLOCK_M, BLOCK_DMODEL] zf32)))
    afterQ Q K V Out sm_scale B_Start_Loc B_Seqlen stride_qbs stride_qh stride_kbs stride_kh stride_vbs stride_vh stride_obs stride_oh kv_group_num BLOCK_M BLOCK_DMODEL BLOCK_N IS_CAUSAL Lk cur_batch cur_head start_m cur_kv_head cur_batch_seq_len cur_batch_in_all_start_index q

/-- The model's body is the five scalar steps followed by `restProg`. -/
theorem body_eq (Q K V Out : Ptr .bf16) (sm_scale : F32)
    (B_Start_Loc B_Seqlen : Ptr .i32)
    (stride_qbs stride_qh stride_kbs stride_kh stride_vbs stride_vh stride_obs stride_oh : I32)
    (kv_group_num BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) (IS_CAUSAL : Bool) (Lk : Nat) (ctx : Ctx) :
    (fwd_kernel Q K V Out sm_scale B_Start_Loc B_Seqlen stride_qbs stride_qh stride_kbs stride_kh stride_vbs stride_vh stride_obs stride_oh kv_group_num BLOCK_M BLOCK_DMODEL BLOCK_N IS_CAUSAL Lk).body ctx = (do
      let cur_batch := tl.program_id ctx 0
      -- line 76: axis 1
      let cur_head := tl.program_id ctx 1
      -- line 77: axis 2
      let start_m := tl.program_id ctx 2
      -- line 79: `int32` `//` by the constexpr built in `int32` (truncating; the divisor is at
      -- least `1` on a covered instance)
      let cur_kv_head :=
        (← tl.idiv (Tensor.scalar cur_head) (Tensor.scalar (IntV.ofNat kv_group_num : I32))).item
      -- line 81: the `*i32` pointer advanced by `cur_batch`; one `int32` element
      let cur_batch_seq_len ← tl.load1 (B_Seqlen + cur_batch)
      -- line 82
      let cur_batch_in_all_start_index ← tl.load1 (B_Start_Loc + cur_batch)
      restProg Q K V Out sm_scale B_Start_Loc B_Seqlen stride_qbs stride_qh stride_kbs stride_kh stride_vbs stride_vh stride_obs stride_oh kv_group_num BLOCK_M BLOCK_DMODEL BLOCK_N IS_CAUSAL Lk cur_batch cur_head start_m cur_kv_head cur_batch_seq_len cur_batch_in_all_start_index) := rfl

/-! ## The pieces on an instance -/

/-- `loopBody` for the instance `d`. -/
abbrev loopBodyD (d : Inst) := loopBody (d.q.ptr 0) (d.k.ptr 0) (d.v.ptr 0) (d.o.ptr 0) (FloatV.hostF32 d.sm_scale)
        (d.b_start_loc.ptr 0) (d.b_seq_len.ptr 0)
        (IntV.ofNat d.sqt) (IntV.ofNat d.sqh) (IntV.ofNat d.skt) (IntV.ofNat d.skh)
        (IntV.ofNat d.svt) (IntV.ofNat d.svh) (IntV.ofNat d.sot) (IntV.ofNat d.soh)
        (kvg d) BLOCK d.BD BLOCK d.causal d.Lk

/-- `postProg` for the instance `d`. -/
abbrev postProgD (d : Inst) := postProg (d.q.ptr 0) (d.k.ptr 0) (d.v.ptr 0) (d.o.ptr 0) (FloatV.hostF32 d.sm_scale)
        (d.b_start_loc.ptr 0) (d.b_seq_len.ptr 0)
        (IntV.ofNat d.sqt) (IntV.ofNat d.sqh) (IntV.ofNat d.skt) (IntV.ofNat d.skh)
        (IntV.ofNat d.svt) (IntV.ofNat d.svh) (IntV.ofNat d.sot) (IntV.ofNat d.soh)
        (kvg d) BLOCK d.BD BLOCK d.causal d.Lk

/-- `afterQ` for the instance `d`. -/
abbrev afterQD (d : Inst) := afterQ (d.q.ptr 0) (d.k.ptr 0) (d.v.ptr 0) (d.o.ptr 0) (FloatV.hostF32 d.sm_scale)
        (d.b_start_loc.ptr 0) (d.b_seq_len.ptr 0)
        (IntV.ofNat d.sqt) (IntV.ofNat d.sqh) (IntV.ofNat d.skt) (IntV.ofNat d.skh)
        (IntV.ofNat d.svt) (IntV.ofNat d.svh) (IntV.ofNat d.sot) (IntV.ofNat d.soh)
        (kvg d) BLOCK d.BD BLOCK d.causal d.Lk

/-- `restProg` for the instance `d`. -/
abbrev restProgD (d : Inst) := restProg (d.q.ptr 0) (d.k.ptr 0) (d.v.ptr 0) (d.o.ptr 0) (FloatV.hostF32 d.sm_scale)
        (d.b_start_loc.ptr 0) (d.b_seq_len.ptr 0)
        (IntV.ofNat d.sqt) (IntV.ofNat d.sqh) (IntV.ofNat d.skt) (IntV.ofNat d.skh)
        (IntV.ofNat d.svt) (IntV.ofNat d.svh) (IntV.ofNat d.sot) (IntV.ofNat d.soh)
        (kvg d) BLOCK d.BD BLOCK d.causal d.Lk

end Claims.PrefillAttnVarlen

end

section
/-!
# The loop's state, for one query row

`RowSpec` is the exact data of a row (the last key it attends, the exact scores, the value
columns). `RowState` is what the loop keeps true after `t` trips of the float values `m` (running
maximum), `l` (running sum) and the accumulator `a` (one `float32` per component): before the first
trip they are `-∞`, `+0`, `+0`; afterwards they are finite, and within the running bounds of
`Proof/Num*` of the exact sums at the computed scores `sh`.
-/

namespace Claims.PrefillAttnVarlen

open TVA

/-- The exact data of one row: the last key `L` it attends, its exact scores `sc j`, and its value
columns `vv e j` (component `e` of key `j`; `0` for the padded components). -/
structure RowSpec (BD : ℕ) where
  L : ℕ
  sc : ℕ → ℝ
  vv : Fin BD → ℕ → ℝ

/-- **The row's state after `t` trips**, with score error `δ`, score bound `Y` and value sums `V e`
(`m` the running maximum, `l` the running sum, `a e` the accumulator of component `e`). -/
def RowState {BD : ℕ} (R : RowSpec BD) (δ Y : ℝ) (V : Fin BD → ℝ) (t : ℕ)
    (m l : F32) (a : Fin BD → F32) : Prop :=
  (t = 0 ∧ m = FloatV.negInf ∧ l = FloatV.ofPat 0 ∧ ∀ e, a e = FloatV.ofPat 0) ∨
  (0 < t ∧ ∃ sh : ℕ → ℝ, Num.RowInv R.L t R.sc sh δ Y m.toReal l.toReal ∧
    m.isFinite = true ∧ l.isFinite = true ∧
    ∀ e, (a e).isFinite = true ∧
      Num.Close (l.toReal * (a e).toReal) (Num.Asum sh (R.vv e) (Num.keysN R.L t) m.toReal)
        (Num.Aabs sh (R.vv e) (Num.keysN R.L t) m.toReal) (Num.omA Y t m.toReal) (Num.bA Y)
        (Num.ηA * (1 + V e)) t)

end Claims.PrefillAttnVarlen

end

section
/-!
# The exact data of a row, for the loop's invariant

`rowSpec` is the row of query `p` (sequence `b`, head `h`): the last key it attends, its exact
scores and its value columns (zero beyond `Lk`). `scoreErr X` is the bound on a computed score's
error when the row's score bound is `X`, and `rowY X = X + scoreErr X` bounds the computed scores.
-/

namespace Claims.PrefillAttnVarlen

open TVA

/-- The exact data of the row of query `p` of sequence `b`, head `h`. -/
noncomputable def rowSpec (d : Inst) (m : Mem) (b h p : Nat) : RowSpec d.BD where
  L := lastKey d m b p
  sc := fun j => score d m b h p j
  vv := fun e j => if e.val < d.Lk then ((vv d m b h j e.val : ℚ) : ℝ) else 0

/-- The error bound of a computed score when the row's score bound is `X`:
`(BD + 8)·2^-22·X + 2^-40`. -/
noncomputable def scoreErr (BD : Nat) (X : ℝ) : ℝ := ((BD : ℝ) + 8) * 2 ^ (-22 : ℤ) * X + 2 ^ (-40 : ℤ)

/-- The bound on the computed scores of a row whose score bound is `X`. -/
noncomputable def rowY (BD : Nat) (X : ℝ) : ℝ := X + scoreErr BD X

/-- The value sums of a row: `Σ_j |v_j[e]|` for `e < Lk`, `0` beyond. -/
noncomputable def rowV (d : Inst) (m : Mem) (b h p : Nat) : Fin d.BD → ℝ :=
  fun e => if e.val < d.Lk then vSum d m b h p e.val else 0

end Claims.PrefillAttnVarlen

end

section
/-!
# The loop: invariant and the two lemmas that walk its body and its epilogue

`LoopInv` says what the carried state `(m_i, l_i, acc)` is after `t` trips, row by row, for the
rows of the program's block that lie inside the sequence (`RowState`); the memory is untouched.
`LoopStep.lean` walks one trip and `Post.lean` the epilogue after the last trip.
-/

namespace Claims.PrefillAttnVarlen

open TVA

/-- The scalars of a program, as the walk meets them: the sequence `b`, head `h` and query block
`smv` of the grid index, the key/value head, and the sequence's length and start (all `int32`). -/
structure Scal (d : Inst) (init : Mem) (i : Nat) (cb ch sm kvh len st : I32) : Prop where
  cb : cb.toInt = bOf d i
  ch : ch.toInt = hOf d i
  sm : sm.toInt = mOf d i
  kvh : kvh.toInt = (hOf d i : Int) / kvg d
  len : len.toInt = lenOf d init (bOf d i)
  st : st.toInt = startOf d init (bOf d i)

/-- Row `r` of the program's block lies inside its sequence. -/
@[reducible] def validRow (d : Inst) (init : Mem) (i : Nat) (r : Nat) : Prop :=
  mOf d i * BLOCK + r < lenOf d init (bOf d i)

/-- The number of loop trips of program `i`: none when its block starts beyond the sequence, else
`cdiv(end_n, 128)` with `end_n` the sequence's length, or `min((sm + 1) * 128, len)` when causal. -/
def nTrips (d : Inst) (init : Mem) (i : Nat) : Nat :=
  if mOf d i * BLOCK < lenOf d init (bOf d i) then
    ((if d.causal then min ((mOf d i + 1) * BLOCK) (lenOf d init (bOf d i))
      else lenOf d init (bOf d i)) + BLOCK - 1) / BLOCK
  else 0

/-- The loaded `q` tile: lane `(r, e)` holds `q[start + sm*128 + r, h, e]` inside the sequence and
the head dimension, and `+0` outside. -/
def QFacts (d : Inst) (init : Mem) (i : Nat) (q : Tensor [BLOCK, d.BD] BF16) : Prop :=
  ∀ (r : Fin BLOCK) (e : Fin d.BD), DType.val? .bf16 (q.get2 r e) =
    some (if mOf d i * BLOCK + r < lenOf d init (bOf d i) ∧ e.val < d.Lk then
      qv d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r) e else 0)

/-- **The loop's invariant** after `t` trips. -/
def LoopInv (d : Inst) (init : Mem) (i : Nat) (t : Nat)
    (st3 : Tensor [BLOCK] F32 × Tensor [BLOCK] F32 × Tensor [BLOCK, d.BD] F32) (m : Mem) : Prop :=
  m = init ∧ ∀ r : Fin BLOCK, validRow d init i r →
    RowState (rowSpec d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r))
      (scoreErr d.BD (Xrow d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r)))
      (rowY d.BD (Xrow d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r)))
      (rowV d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r)) t
      (st3.1.get1 r) (st3.2.1.get1 r) (fun e => st3.2.2.get2 r e)

end Claims.PrefillAttnVarlen

end

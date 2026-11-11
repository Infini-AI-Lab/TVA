import Claim
import Proof.Defs
import Proof.Loop
import Proof.Num
import Proof.Ptr
import TVA.Proof

/-!
# SafeProg

The programs' results give `Post` (`post_of_locals`), and scalar conversions the walk uses.
-/

namespace Claims.SsdChunkState

open TVA

set_option maxHeartbeats 4000000

/-!
# `ssd_chunk_state_bf16`: one program is safe

The program's prelude computes the five buffer pointers, the tile offsets and the loop bound; the
loop loads the tiles and accumulates `float32` products; the last line stores the tile of `states`.
Nothing wraps, every load is inside the buffers and the store is inside the tile, so each step is
`Safe` with the read footprint `R` and the write footprint `W`.
-/

set_option maxHeartbeats 800000


open TVA

/-! ## The program index and the integer literals -/

/-- `program_id(0)` of program `i` as an `int32`. -/
theorem pid0_eq (d : Inst) (i : Fin (launch d).size) :
    tl.program_id ((launch d).ctx i) 0 = (IntV.ofNat (qOf d i) : I32) := by
  show (IntV.ofNat (((launch d).ctx i).pid 0) : I32) = _
  rw [launch_pid0]
  rfl

/-- `program_id(1)` of program `i` as an `int32`. -/
theorem pid1_eq (d : Inst) (i : Fin (launch d).size) :
    tl.program_id ((launch d).ctx i) 1 = (IntV.ofNat (chunkOf d i) : I32) := by
  show (IntV.ofNat (((launch d).ctx i).pid 1) : I32) = _
  rw [launch_pid1]
  rfl

/-- `program_id(2)` of program `i` as an `int32`. -/
theorem pid2_eq (d : Inst) (i : Fin (launch d).size) :
    tl.program_id ((launch d).ctx i) 2 = (IntV.ofNat (headOf d i) : I32) := by
  show (IntV.ofNat (((launch d).ctx i).pid 2) : I32) = _
  rw [launch_pid2]
  rfl

/-! ### `IntV` arithmetic on nonnegative values -/

/-- `int32` sum of two naturals, without wrap. -/
theorem I32.toInt_add_nat {a b : Nat} (h : a + b < 2 ^ 31) :
    ((IntV.ofNat a + IntV.ofNat b : I32)).toInt = (a : Int) + b :=
  IntV.toInt_add (IntV.I32.toInt_ofNat (by omega)) (IntV.I32.toInt_ofNat (by omega))
    (by simp only [IntV.SFits]; norm_num at h ⊢; omega)

/-- `int64` sum of nonnegative values, without wrap. -/
theorem I64.toInt_add_nat {a b : I64} (ha : 0 ≤ a.toInt) (hb : 0 ≤ b.toInt)
    (h : a.toInt + b.toInt < 2 ^ 63) : (a + b).toInt = a.toInt + b.toInt :=
  IntV.toInt_add rfl rfl (by simp only [IntV.SFits]; norm_num at h ⊢; omega)

/-- `int64` product of nonnegative values, without wrap. -/
theorem I64.toInt_mul_nat {a b : I64} (ha : 0 ≤ a.toInt) (hb : 0 ≤ b.toInt)
    (h : a.toInt * b.toInt < 2 ^ 63) : (a * b).toInt = a.toInt * b.toInt :=
  IntV.toInt_mul rfl rfl (by
    simp only [IntV.SFits]
    norm_num at h ⊢
    exact ⟨by nlinarith, h⟩)

/-- `int64` difference of nonnegative values, without wrap. -/
theorem I64.toInt_sub_nat {a b : I64} (_ha : 0 ≤ a.toInt) (_hb : 0 ≤ b.toInt)
    (hab : b.toInt ≤ a.toInt) (h : a.toInt - b.toInt < 2 ^ 63) :
    (a - b).toInt = a.toInt - b.toInt :=
  IntV.toInt_sub rfl rfl (by simp only [IntV.SFits]; norm_num at h ⊢; omega)

/-! ### Scalar conversions -/

/-- Converting an `int32` scalar to `int64` keeps its value. -/
theorem safe_conv64 {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {z : I32}
    {k : Tensor [] I64 → Prog α}
    (hk : ∀ r : Tensor [] I64, r.item.toInt = z.toInt → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.convert DType.i32 DType.i64 (Tensor.scalar z) >>= k) := by
  refine safe_convert (d := .i32) (d' := .i64) (fun _ => ⟨IntV.ofInt z.toInt, rfl⟩) ?_
  intro r hr
  refine hk r ?_
  have h1 : r.item = (IntV.ofInt z.toInt : I64) := hr ()
  rw [h1]
  refine IntV.toInt_ofInt (by norm_num) ?_
  have h := IntV.sfits_toInt z
  simp only [IntV.SFits] at h ⊢
  norm_num at h ⊢
  constructor <;> omega


/-! ## Post -/

/-!
# `ssd_chunk_state_bf16`: the programs' results give `Post`

Each `(c, h, p, n)` is owned by the program `progOf d c h p n`, whose `Q` holds its tile; the
frame conditions move that value from the program's memory to the launch's final memory.
-/

set_option maxHeartbeats 800000


open TVA

/-- The programs' results give `Post`: each element is read back from the program that owns it, and every byte outside `states` is unchanged. -/
theorem post_of_locals {d : Inst} (hd : d.ok) {init : Mem} (_hpre : Pre d init) {fin : Mem}
    (hloc : ∀ i, ∃ m, Q d init i m ∧ ∀ x ∈ R d init i ∪ W d init i, fin x = m x)
    (hfr : ∀ x, (∀ i, x ∉ W d init i) → fin x = init x) : Post d init fin := by
  constructor
  · intro c h p n hc hh hp hn
    have hi : progOf d c h p n < (launch d).size := progOf_lt hd hc hh hp hn
    set i : Fin (launch d).size := ⟨progOf d c h p n, hi⟩ with hi_def
    obtain ⟨m, hQ, hframe⟩ := hloc i
    have hchc : chunkOf d (↑i) = c := chunkOf_progOf hd hp hn hc
    have hchh : headOf d (↑i) = h := headOf_progOf hd hp hn hc
    have hps : p ∈ pSet d (↑i) := mem_pSet_progOf hd hp hn
    have hns : n ∈ nSet d (↑i) := mem_nSet_progOf hd hp hn
    obtain ⟨v, hvread, hvprop⟩ := hQ p n hps hns
    have hidx : stIdx d (chunkOf d (↑i)) (headOf d (↑i)) p n = stIdx d c h p n := by
      rw [hchc, hchh]
    rw [hidx] at hvread
    rw [hchc, hchh] at hvprop
    refine ⟨v, ?_, hvprop⟩
    have hj : stIdx d c h p n ∈ stSet d (↑i) := by
      refine ⟨p, hps, n, hns, ?_⟩
      exact hidx.symm
    have hspan : Mem.span .f32 (d.states.ptr (stIdx d c h p n : Int)).addr
        ⊆ W d init i := Buf.span_sub_elems d.states hj
    have hfin : ∀ x ∈ Mem.span .f32 (d.states.ptr (stIdx d c h p n : Int)).addr,
        fin x = m x := fun x hx => hframe x (Or.inr (hspan hx))
    rw [Mem.read_congr (d := .f32) hfin]
    exact hvread
  · intro z hz
    apply hfr
    intro i hzi
    exact hz (Buf.elems_mono d.states (stSet_sub hd i.isLt) hzi)


/-! ## Tail -/

/-!
# From the loop invariant to the claim's value bound

At the end of the accumulation loop the invariant `Inv` says that the accumulator element
`(a, n)` is within `DotLoop.bound` of `∑ Sf` and that the operands fed to the dot products are
the claim's `xV` and a `b`-side value within `errB` of `bV * wRef` on the chunk's tokens (and
`0` elsewhere).  This file turns that into the claim's per-element bound `|acc - sRef| ≤ tol`
by way of `Num.value_bound`.
-/

open TVA

set_option maxHeartbeats 800000

/-- Flattening a double sum over `range T` and `Fin BK` into one over `range (T·BK)`. -/
private theorem flatSum (xh : Nat → Fin BM → ℚ) (bh : Nat → Fin BN → ℚ) (a : Fin BM) (n : Fin BN) :
    ∀ (T : Nat),
      (∑ t' ∈ Finset.range T, ∑ j : Fin BK,
          (xh (t' * BK + (j : Nat)) a : ℝ) * (bh (t' * BK + (j : Nat)) n : ℝ))
      = ∑ s ∈ Finset.range (T * BK), (xh s a : ℝ) * (bh s n : ℝ) := by
  intro T
  induction T with
  | zero => simp
  | succ T ih =>
    rw [Finset.sum_range_succ, ih, Nat.succ_mul, Finset.sum_range_add]
    rw [Fin.sum_univ_eq_sum_range
      (fun x => (xh (T * BK + x) a : ℝ) * (bh (T * BK + x) n : ℝ)) BK]

/-- **The accumulator's value bound.**  For an enabled lane of the tile, the accumulator's
element is within `tol` of the chunk state. -/
theorem acc_bound {d : Inst} {init : Mem} {i : Fin (launch d).size} (hd : d.ok)
    (hV : Vals init d)
    (xh : Nat → Fin BM → ℚ) (bh : Nat → Fin BN → ℚ)
    (hx : ∀ (s : Nat) (a : Fin BM), s < steps d (chunkOf d i) * BK →
      ((chunkOf d i * d.chunk + s ≤ endTok d (chunkOf d i) ∧
          pmOf d i * BM + (a : Nat) < d.headdim →
        (xh s a : ℝ) = xV init d (chunkOf d i * d.chunk + s) (headOf d i)
          (pmOf d i * BM + (a : Nat))) ∧
       (¬(chunkOf d i * d.chunk + s ≤ endTok d (chunkOf d i) ∧
          pmOf d i * BM + (a : Nat) < d.headdim) → xh s a = 0)))
    (hb : ∀ (s : Nat) (n : Fin BN), s < steps d (chunkOf d i) * BK →
      (((chunkOf d i * d.chunk + s) ∈ toks d (chunkOf d i) ∧
          pnOf d i * BN + (n : Nat) < d.dstate →
        |(bh s n : ℝ) - bV init d (chunkOf d i * d.chunk + s) (headOf d i)
            (pnOf d i * BN + (n : Nat)) *
            wRef init d (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + s)|
          ≤ errB init d (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + s)
            (pnOf d i * BN + (n : Nat))) ∧
       (¬((chunkOf d i * d.chunk + s) ∈ toks d (chunkOf d i) ∧
          pnOf d i * BN + (n : Nat) < d.dstate) → bh s n = 0)))
    {a : Fin BM} {n : Fin BN}
    (hp : pmOf d i * BM + (a : Nat) < d.headdim)
    (hn : pnOf d i * BN + (n : Nat) < d.dstate)
    {acc : DType.f32.Val}
    (hacc : |((acc.toRat : ℚ) : ℝ) -
        ((∑ t' ∈ Finset.range (steps d (chunkOf d i)), Sf xh bh a n t' : ℚ) : ℝ)| ≤
      ((DotLoop.bound rho alp (steps d (chunkOf d i))
        (∑ t' ∈ Finset.range (steps d (chunkOf d i)), Af xh bh a n t') : ℚ) : ℝ)) :
    |acc.toReal -
        sRef init d (chunkOf d i) (headOf d i) (pmOf d i * BM + (a : Nat))
          (pnOf d i * BN + (n : Nat))| ≤
      tol init d (chunkOf d i) (headOf d i) (pmOf d i * BM + (a : Nat))
        (pnOf d i * BN + (n : Nat)) := by
  have hc : chunkOf d i < nc d := chunkOf_lt hd i
  have hh : headOf d i < d.nheads := headOf_lt hd i.isLt
  have hBK : 0 < BK := by simp only [BK]; norm_num
  -- the loop's range covers the chunk's tokens
  have hSBK : endTok d (chunkOf d i) + 1 - chunkOf d i * d.chunk ≤ steps d (chunkOf d i) * BK := by
    unfold steps
    have hdm := Nat.div_add_mod (endTok d (chunkOf d i) + 1 - chunkOf d i * d.chunk + BK - 1) BK
    have hmod := Nat.mod_lt (endTok d (chunkOf d i) + 1 - chunkOf d i * d.chunk + BK - 1) hBK
    simp only [BK] at hdm hmod ⊢
    omega
  -- cast the rational sums to real sums
  have hcast : ((∑ t' ∈ Finset.range (steps d (chunkOf d i)), Sf xh bh a n t' : ℚ) : ℝ)
      = ∑ t' ∈ Finset.range (steps d (chunkOf d i)), ∑ j : Fin BK,
          (xh (t' * BK + (j : Nat)) a : ℝ) * (bh (t' * BK + (j : Nat)) n : ℝ) := by
    rw [Rat.cast_sum]
    apply Finset.sum_congr rfl
    intro t' _
    rw [Sf, Rat.cast_sum]
    apply Finset.sum_congr rfl
    intro j _
    rw [Rat.cast_mul]
  -- the tokens of `toks c` are exactly the loop's positions
  have hmemIcc : ∀ t ∈ toks d (chunkOf d i),
      chunkOf d i * d.chunk ≤ t ∧ t ≤ endTok d (chunkOf d i) := by
    intro t ht
    exact Finset.mem_Icc.mp (Finset.mem_filter.mp ht).1
  have htoks_lt : ∀ t ∈ toks d (chunkOf d i),
      t - chunkOf d i * d.chunk < steps d (chunkOf d i) * BK := by
    intro t ht
    obtain ⟨h1, h2⟩ := hmemIcc t ht
    omega
  -- the sum over the loop's positions is the sum over the chunk's tokens
  have hsum2 :
      (∑ s ∈ Finset.range (steps d (chunkOf d i) * BK), (xh s a : ℝ) * (bh s n : ℝ))
      = ∑ t ∈ toks d (chunkOf d i), xV init d t (headOf d i) (pmOf d i * BM + (a : Nat)) *
          (bh (t - chunkOf d i * d.chunk) n : ℝ) := by
    have hterm : ∀ s ∈ Finset.range (steps d (chunkOf d i) * BK),
        (xh s a : ℝ) * (bh s n : ℝ)
        = if chunkOf d i * d.chunk + s ∈ toks d (chunkOf d i)
            then xV init d (chunkOf d i * d.chunk + s) (headOf d i)
                (pmOf d i * BM + (a : Nat)) * (bh s n : ℝ)
            else 0 := by
      intro s hs
      have hslt : s < steps d (chunkOf d i) * BK := Finset.mem_range.mp hs
      have hx' := hx s a hslt
      have hb' := hb s n hslt
      by_cases hmem : chunkOf d i * d.chunk + s ∈ toks d (chunkOf d i)
      · rw [ite_eq_left hmem]
        have hle : chunkOf d i * d.chunk + s ≤ endTok d (chunkOf d i) := (hmemIcc _ hmem).2
        rw [hx'.1 ⟨hle, hp⟩]
      · rw [ite_eq_right hmem]
        have hzero : bh s n = 0 := hb'.2 (fun hcond => hmem hcond.1)
        rw [hzero]
        simp
    rw [Finset.sum_congr rfl hterm, ← Finset.sum_filter]
    refine Finset.sum_bij' (fun s _ => chunkOf d i * d.chunk + s)
      (fun t _ => t - chunkOf d i * d.chunk) ?_ ?_ ?_ ?_ ?_
    · intro s hs
      exact (Finset.mem_filter.mp hs).2
    · intro t ht
      refine Finset.mem_filter.mpr ⟨Finset.mem_range.mpr (htoks_lt t ht), ?_⟩
      rw [Nat.add_sub_cancel' (hmemIcc t ht).1]
      exact ht
    · intro s _
      exact Nat.add_sub_cancel_left _ _
    · intro t ht
      exact Nat.add_sub_cancel' (hmemIcc t ht).1
    · intro s _
      rw [Nat.add_sub_cancel_left]
  have hmain := value_bound (d := d) hd (m₀ := init) hV
    (c := chunkOf d i) (h := headOf d i)
    (p := pmOf d i * BM + (a : Nat)) (n := pnOf d i * BN + (n : Nat))
    (aT := ((acc.toRat : ℚ) : ℝ))
    (bh := fun t => bh (t - chunkOf d i * d.chunk) n)
    (S := Sf xh bh a n) (A := Af xh bh a n)
    hc hh hp hn
    (by rw [hcast, flatSum xh bh a n]; exact hsum2)
    (aSum_le_dotMag hd (le_refl (steps d (chunkOf d i))) xh bh hx hb hV a n)
    (by
      intro t ht
      obtain ⟨h1, h2⟩ := hmemIcc t ht
      have hslt : t - chunkOf d i * d.chunk < steps d (chunkOf d i) * BK := htoks_lt t ht
      have hb' := hb (t - chunkOf d i * d.chunk) n hslt
      have hmem : chunkOf d i * d.chunk + (t - chunkOf d i * d.chunk) ∈ toks d (chunkOf d i) := by
        rw [Nat.add_sub_cancel' h1]
        exact ht
      have hb2 := hb'.1 ⟨hmem, hn⟩
      rwa [Nat.add_sub_cancel' h1] at hb2)
    hacc
  simpa only [TVA.FloatV.toReal] using hmain


/-! ## TailStore -/

/-!
# The program after the accumulation loop

The last two statements of the kernel convert two `int32` scalars to `int64` (the `states`
strides) and then store the accumulator tile.  This file turns the loop invariant at the end of
the loop into the `Safe` proof of that tail.
-/

open TVA

set_option maxHeartbeats 800000

/-- **The tail of the kernel.**  Starting from the loop invariant at the end of the loop, the
final store is safe and leaves the claim's `Q`. -/
theorem safe_tail {E : Env} {d : Inst} {init : Mem} {i : Fin (launch d).size}
    (hd : d.ok) (hvs : d.states.Valid init)
    {hi : I64} {r1 r2 r3 : Tensor [] I64} {r5 r6 : Tensor [] I32}
    (hlim : hi.toInt = ((endTok d (chunkOf d i) + 1 - chunkOf d i * d.chunk : Nat) : Int))
    (hz0 : (IntV.ofInt 0 : I64).toInt = 0)
    (hbk : (IntV.ofNat BK : I64).toInt = 64)
    (H3 : r3.item.toInt = (chunkOf d i : Int))
    (H5 : r5.item.toInt = (pmOf d i : Int))
    (H6 : r6.item.toInt = (pnOf d i : Int))
    (hpb : (r1.item - r3.item * r2.item).toInt = 0)
    {st : LoopSt d} {m₁ : Mem}
    (hinv : Inv d init i
      (tl.tripCount (IntV.ofInt 0 : I64).toInt hi.toInt (IntV.ofNat BK : I64).toInt) st m₁) :
    Safe E (R d init i) (W d init i) (fun _ m => Q d init i m) m₁
      (tl.convert DType.i32 DType.i64 (Tensor.scalar (IntV.ofNat (sSc d))) >>= fun l1 =>
        tl.convert DType.i32 DType.i64
            (Tensor.scalar (tl.program_id ((launch d).ctx i) 2 * IntV.ofNat (sSh d))) >>= fun l2 =>
          tl.store
            (d.states.ptr 0 +
                ((r1.item - r3.item * r2.item) * (IntV.ofNat (sSb d) : I64) +
                    r3.item * l1.item + l2.item) +
              ((r5.item * (IntV.ofNat BM : I32) + tl.arange 0 BM).col BN *
                  (IntV.ofNat d.dstate : I32) +
                (r6.item * (IntV.ofNat BN : I32) + tl.arange 0 BN).row BM *
                  (IntV.ofNat 1 : I32)))
            st.2.2.2.2.2
            (some
              ((((r5.item * (IntV.ofNat BM : I32) + tl.arange 0 BM).col BN).ltS
                    (IntV.ofNat d.headdim : I32)).and
                (((r6.item * (IntV.ofNat BN : I32) + tl.arange 0 BN).row BM).ltS
                  (IntV.ofNat d.dstate : I32))))) := by
  have hc : chunkOf d i < nc d := chunkOf_lt hd i
  have hhh : headOf d i < d.nheads := headOf_lt hd i.isLt
  have hh31 : headOf d i < 2 ^ 31 := by
    have := hd.grid_z
    omega
  -- the loop's trip count is the number of `BK`-tiles of the chunk
  have htrip : tl.tripCount (IntV.ofInt 0 : I64).toInt hi.toInt (IntV.ofNat BK : I64).toInt
      = steps d (chunkOf d i) := by
    rw [hz0, hlim, hbk, show (64 : Int) = (BK : Int) by norm_num [BK]]
    have hge : chunkOf d i * d.chunk ≤ endTok d (chunkOf d i) + 1 := by
      have hlt := c_chunk_lt hd hc
      unfold endTok
      have h5 : chunkOf d i * d.chunk < min ((chunkOf d i + 1) * d.chunk) d.seqlen := by
        apply lt_min
        · have h6 : (chunkOf d i + 1) * d.chunk = chunkOf d i * d.chunk + d.chunk := by ring
          have := chunk_pos hd
          omega
        · exact hlt
      omega
    have hcast : ((endTok d (chunkOf d i) + 1 - chunkOf d i * d.chunk : Nat) : Int)
        = (↑(endTok d (chunkOf d i)) + 1 - ↑(chunkOf d i) * ↑d.chunk) := by
      rw [Nat.cast_sub hge, Nat.cast_add, Nat.cast_one, Nat.cast_mul]
    rw [hcast]
    exact tripCount_eq hd hc
  rw [htrip] at hinv
  obtain ⟨hminit, xh, bh, hp1, hp2, hp3, hp4, hp5, hx, hb, hacc⟩ := hinv
  rw [hminit]
  -- the two `int32`→`int64` conversions of the `states` strides
  refine safe_conv64 (E := E) (R := R d init i) (W := W d init i)
    (Q := fun _ m => Q d init i m) (m := init) (z := (IntV.ofNat (sSc d) : I32)) ?_
  intro l1 hl1
  have HS1 : l1.item.toInt = (sSc d : Int) := by
    rw [hl1, IntV.I32.toInt_ofNat hd.st_lt]
  refine safe_conv64 (E := E) (R := R d init i) (W := W d init i)
    (Q := fun _ m => Q d init i m) (m := init)
    (z := (tl.program_id ((launch d).ctx i) 2 * IntV.ofNat (sSh d) : I32)) ?_
  intro l2 hl2
  have hSSh31 : sSh d < 2 ^ 31 := by
    have hp := hd.heads_pos
    have hs := hd.st_lt
    unfold sSc at hs
    unfold sSh
    have h1 : d.headdim * d.dstate ≤ d.nheads * (d.headdim * d.dstate) := by
      calc d.headdim * d.dstate = 1 * (d.headdim * d.dstate) := (Nat.one_mul _).symm
        _ ≤ d.nheads * (d.headdim * d.dstate) := Nat.mul_le_mul_right _ hp
    omega
  have hHS2lt : headOf d i * sSh d < 2 ^ 31 := by
    have hs := hd.st_lt
    unfold sSc at hs
    unfold sSh
    have h1 : headOf d i * (d.headdim * d.dstate)
        ≤ (d.nheads - 1) * (d.headdim * d.dstate) := Nat.mul_le_mul_right _ (by omega)
    have h2 : (d.nheads - 1) * (d.headdim * d.dstate) < d.nheads * (d.headdim * d.dstate) :=
      Nat.mul_lt_mul_of_pos_right (by have := hd.heads_pos; omega)
        (Nat.mul_pos hd.headdim_pos hd.dstate_pos)
    omega
  have HS2 : l2.item.toInt = (headOf d i * sSh d : Int) := by
    rw [hl2, IntV.I32.mul_nat (by rw [pid2_eq]; exact IntV.I32.toInt_ofNat hh31)
      (IntV.I32.toInt_ofNat hSSh31) hHS2lt]
    push_cast
    ring
  exact store_states (E := E) hd hvs HS1 HS2 hpb H3 H5 H6 st.2.2.2.2.2
    (fun hV a n ha hn => (hacc hV).1 a n)
    (fun hV a n ha hn => acc_bound hd hV xh bh hx (hb hV) ha hn ((hacc hV).2 a n))


end Claims.SsdChunkState

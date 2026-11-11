import TVA.Proof
import Claim
import Proof.Pid
import Proof.Foot
import Proof.Defs
import Proof.Tol
import Proof.Store
import Proof.Inv

/-!
# One trip of the `K` loop

`trip_safe`: `tripBody` (lines 333-377) keeps `InInv`: it loads the two operand tiles (masked past
`K`, `other = +0`), runs `tl.dot_add` (either form), and advances the pointers; each accumulator
lane moves from the exact partial sum of trip `t` to that of trip `t + 1` by `trip_step`.
-/

namespace Claims.FusedMoe

open TVA

/-- Converting the `float32` zero tile to `bfloat16` has results, and each is `+0`. -/
theorem conv_zero {P : Platform} {s : Shape} :
    (∀ i, ∃ r, ConvRel P .f32 .bf16 ((tl.full s (FloatV.litF32 (FloatV.ofPat 0))).get i) r) ∧
    ∀ r : Tensor s BF16, (∀ i, ConvRel P .f32 .bf16 ((tl.full s (FloatV.litF32 (FloatV.ofPat 0))).get i)
      (r.get i)) → ∀ i, r.get i = FloatV.ofPat 0 := by
  have hc : (FloatV.cvt (FloatV.ofPat 0 : F32) : BF16) = FloatV.ofPat 0 :=
    FloatV.cvt_ofPat_zero Format.binary32_regular
  refine ⟨fun i => ⟨_, ConvRel.cvt_f32_bf16 _⟩, fun r hr i => ?_⟩
  have h := ConvRel.f32_bf16_iff.mp (hr i)
  simp only [tl.full, Tensor.get_full, FloatV.litF32_ofPat_zero, hc] at h
  exact FloatV.NaNEq.eq_of_isFinite h (FloatV.isFinite_zero_bf16)

/-- A `bfloat16` lane that holds `+0` has value `0`. -/
theorem val_zero_bf16 {x : BF16} (h : x = FloatV.ofPat 0) : DType.val? .bf16 x = some 0 := by
  rw [h]; exact DType.val?_zero_bf16

/-- `t * BK < K` on a trip of the loop. -/
theorem trip_lt {d : Inst} (hd : d.ok) {t : Nat} (ht : t < steps d.BK d.K) : t * d.BK < d.K := by
  have hBK := (cfg_facts hd).2.2.1
  have h1 : t + 1 ≤ (d.K + d.BK - 1) / d.BK := ht
  rw [Nat.le_div_iff_mul_le (by omega)] at h1
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

/-- **One trip of the loop keeps the invariant.** -/
theorem trip_safe {P : Platform} {d : Inst} (hd : d.ok) (hP : Device P) {init : Mem}
    (hpre : Pre d init) {pmN pnN e : Nat} (he : e < d.E)
    (hact : pmN * d.BM + d.BM ≤ ntpOf d init) {tm : Tensor [d.BM] Bool}
    (hm : ∀ l : Fin d.BM, tm.get1 l = decide (entryOf d init (pmN * d.BM + l) < (nv d : Int)))
    {t : Nat} (ht : t < steps d.BK d.K) {k : I32} (hk : k.toInt = (t : Int)) {st : St d} {m1 : Mem}
    (hI : InInv d init pmN pnN e t st m1) :
    Safe ((launch d).env P) (Rfp d) (∅ : Set Int) (fun st' m2 => InInv d init pmN pnN e (t + 1) st' m2)
      m1 (tripBody d tm k st) := by
  obtain ⟨A, B, C⟩ := st
  obtain ⟨hm1, hpA, hpB, hacc⟩ := hI
  subst hm1
  unfold tripBody
  dsimp only
  have hBK := (cfg_facts hd).2.2.1
  have hBM := (cfg_facts hd).1
  have hBN := (cfg_facts hd).2.1
  have hKle := hd.K_le
  have htK := trip_lt hd ht
  have hKpos := hd.K_pos
  have hBKcases := (by omega : d.BK = 64 ∨ d.BK = 128 ∨ d.BK = 256)
  have hrem : (IntV.ofNat d.K - k * IntV.ofNat d.BK : I32).toInt = ((d.K - t * d.BK : Nat) : Int) := by
    have h1 : (k * (IntV.ofNat d.BK : I32)).toInt = ((t * d.BK : Nat) : Int) := by
      rw [IntV.toInt_mul rfl rfl (by rw [hk, i32_ofNat (by omega)]; exact sfits_mul_nat (by omega)),
        hk, i32_ofNat (by omega)]
      push_cast; ring
    rw [IntV.toInt_sub rfl rfl (by rw [i32_ofNat (by omega), h1]; exact s32 (by omega) (by omega)),
      i32_ofNat (by omega), h1]
    omega
  have hAmask : ∀ (l : Fin d.BM) (kk : Fin d.BK),
      ((tm.col d.BK).and (((tl.arange 0 d.BK).row d.BM).ltS (IntV.ofNat d.K - k * IntV.ofNat d.BK))).get2 l kk
        = (decide (entryOf d m1 (pmN * d.BM + l) < (nv d : Int)) && decide ((kk : Nat) < d.K - t * d.BK)) := by
    intro l kk
    show (tm.get1 l && IntV.lt ((tl.arange 0 d.BK).get1 kk) (IntV.ofNat d.K - k * IntV.ofNat d.BK)) = _
    rw [hm l, IntV.lt_eq, IntV.I32.toInt_tl_arange (n := d.BK) (by omega) kk, hrem]
    congr 1
    exact decide_eq_decide.mpr (by omega)
  have hBmask : ∀ (kk : Fin d.BK) (j : Fin d.BN),
      (((tl.arange 0 d.BK).col d.BN).ltS (IntV.ofNat d.K - k * IntV.ofNat d.BK)).get2 kk j
        = decide ((kk : Nat) < d.K - t * d.BK) := by
    intro kk j
    show IntV.lt ((tl.arange 0 d.BK).get1 kk) (IntV.ofNat d.K - k * IntV.ofNat d.BK) = _
    rw [IntV.lt_eq, IntV.I32.toInt_tl_arange (n := d.BK) (by omega) kk, hrem]
    exact decide_eq_decide.mpr (by omega)
  refine safe_convert (conv_zero (P := P)).1 fun za hza => ?_
  have hza0 := conv_zero.2 za hza
  have hrowlt : ∀ l : Fin d.BM, entryOf d m1 (pmN * d.BM + l) < (nv d : Int) →
      arow d m1 pmN l < d.M := by
    intro l hl
    have h0 := hpre.s1 (pmN * d.BM + l) (by have := l.isLt; omega)
    have h2 : (entryOf d m1 (pmN * d.BM + l)).toNat < nv d := by omega
    unfold arow
    apply Nat.div_lt_of_lt_mul
    rw [Nat.mul_comm d.top_k d.M]
    exact h2
  refine safe_load_buf d.a
    (g := fun idx : Idx [d.BM, d.BK] => arow d m1 pmN idx.1 * d.K + (t * d.BK + (idx.2.1 : Nat)))
    (en := fun idx : Idx [d.BM, d.BK] => decide (entryOf d m1 (pmN * d.BM + idx.1) < (nv d : Int)) &&
      decide ((idx.2.1 : Nat) < d.K - t * d.BK))
    (Buf.addrs_of_get d.a (fun ⟨l, kk, ⟨⟩⟩ => hAmask l kk) (fun ⟨l, kk, ⟨⟩⟩ _ => hpA l kk)) ?_
    hpre.va ?_ fun av hav hav' => ?_
  · rintro ⟨l, kk, ⟨⟩⟩ hen
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hen
    exact hd.a_len (arow d m1 pmN l) (t * d.BK + kk) (hrowlt l hen.1) (by have := hen.2; omega)
  · intro x hx
    refine Rfp_a d (d.a.elems_sub_bytes ?_ hx)
    rintro j ⟨⟨l, kk, ⟨⟩⟩, hen, rfl⟩
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hen
    exact hd.a_len (arow d m1 pmN l) (t * d.BK + kk) (hrowlt l hen.1) (by have := hen.2; omega)
  refine safe_convert (conv_zero (P := P)).1 fun zb hzb => ?_
  have hzb0 := conv_zero.2 zb hzb
  have hcolb : ∀ j : Fin d.BN, bcol d pnN j < d.N := fun j =>
    Nat.mod_lt _ hd.N_pos
  refine safe_load_buf d.b
    (g := fun idx : Idx [d.BK, d.BN] => e * d.stride_be + bcol d pnN idx.2.1 * d.stride_bn + (t * d.BK + (idx.1 : Nat)))
    (en := fun idx : Idx [d.BK, d.BN] => decide ((idx.1 : Nat) < d.K - t * d.BK))
    (Buf.addrs_of_get d.b (fun ⟨kk, j, ⟨⟩⟩ => hBmask kk j) (fun ⟨kk, j, ⟨⟩⟩ _ => hpB kk j)) ?_
    hpre.vb ?_ fun bv hbv hbv' => ?_
  · rintro ⟨kk, j, ⟨⟩⟩ hen
    simp only [decide_eq_true_eq] at hen
    exact hd.b_len e (bcol d pnN j) (t * d.BK + kk) he (hcolb j) (by omega) 
  · intro x hx
    refine Rfp_b d (d.b.elems_sub_bytes ?_ hx)
    rintro j ⟨⟨kk, jj, ⟨⟩⟩, hen, rfl⟩
    simp only [decide_eq_true_eq] at hen
    exact hd.b_len e (bcol d pnN jj) (t * d.BK + kk) he (hcolb jj) (by omega)
  refine safe_dot_add (hex := hP.dotTotal { a := DType.bf16, b := DType.bf16, acc := DType.f32 } av bv C) fun r hr => safe_ret ?_
  refine ⟨rfl, ?_, ?_, ?_⟩
  · intro l kk
    show (A.get2 l kk).offset ((IntV.ofNat d.BK * IntV.ofNat 1 : I32)).toInt = _
    rw [hpA l kk, Buf.ptr_offset, IntV.toInt_mul rfl rfl (by
      rw [i32_ofNat (by omega), i32_ofNat (by norm_num)]; exact sfits_mul_nat (by omega)),
      i32_ofNat (by omega), i32_ofNat (by norm_num)]
    congr 1
    push_cast; ring
  · intro kk j
    show (B.get2 kk j).offset ((IntV.ofNat d.BK * IntV.ofNat 1 : I32)).toInt = _
    rw [hpB kk j, Buf.ptr_offset, IntV.toInt_mul rfl rfl (by
      rw [i32_ofNat (by omega), i32_ofNat (by norm_num)]; exact sfits_mul_nat (by omega)),
      i32_ofNat (by omega), i32_ofNat (by norm_num)]
    congr 1
    push_cast; ring
  · intro l j hv va vb ha hb hmag
    obtain ⟨v0, hv0, hprev⟩ := hacc l j hv va vb ha hb hmag
    refine trip_step hP.dotBF16Chunk (a := av) (b := bv) (z := Tensor.full (DType.f32.ofBits 0))
      (c := C) (r := r) hBKcases hd.K_le15 ht l j va vb hmag ?_ ?_ DType.val?_zero_f32 hr hv0 hprev
    · intro kk
      by_cases hk' : t * d.BK + (kk : Nat) < d.K
      · have hen : (decide (entryOf d m1 (pmN * d.BM + l) < (nv d : Int)) &&
            decide ((kk : Nat) < d.K - t * d.BK)) = true := by
          simp only [Bool.and_eq_true, decide_eq_true_eq]; exact ⟨hv, by omega⟩
        have h1 := hav (l, kk, ()) hen
        have h2 := (ha (t * d.BK + kk) hk').1
        have h3 : av.get2 l kk = va (t * d.BK + kk) := by
          have := h1.symm.trans h2
          exact Option.some.inj this
        rw [ite_eq_left hk', h3]
        exact DType.val?_bf16.mpr ⟨(ha _ hk').2, rfl⟩
      · have hen : (decide (entryOf d m1 (pmN * d.BM + l) < (nv d : Int)) &&
            decide ((kk : Nat) < d.K - t * d.BK)) = false := by
          simp only [Bool.and_eq_false_imp, decide_eq_true_eq, decide_eq_false_iff_not]; intro _; omega
        have h1 := hav' (l, kk, ()) hen za rfl
        rw [ite_eq_right hk']
        exact val_zero_bf16 ((h1.trans (hza0 _)))
    · intro kk
      by_cases hk' : t * d.BK + (kk : Nat) < d.K
      · have hen : decide ((kk : Nat) < d.K - t * d.BK) = true := by
          simp only [decide_eq_true_eq]; omega
        have h1 := hbv (kk, j, ()) hen
        have h2 := (hb (t * d.BK + kk) hk').1
        have h3 : bv.get2 kk j = vb (t * d.BK + kk) := by
          have := h1.symm.trans h2
          exact Option.some.inj this
        rw [ite_eq_left hk', h3]
        exact DType.val?_bf16.mpr ⟨(hb _ hk').2, rfl⟩
      · have hen : decide ((kk : Nat) < d.K - t * d.BK) = false := by
          simp only [decide_eq_false_iff_not]; omega
        have h1 := hbv' (kk, j, ()) hen zb rfl
        rw [ite_eq_right hk']
        exact val_zero_bf16 ((h1.trans (hzb0 _)))

import Proof.Setup
import Proof.Reduction
import Proof.Epilogue

/-!
# `tma_persistent_matmul_f16`: The inner loop over `K`: its invariant and its safety.
-/

namespace Claims.TmaPersistentMatmul

open TVA

/-!
## The accumulator after the inner loop

The inner loop keeps, for every in-bounds lane of the tile, the accumulator finite and within the
error recursion `E` of the exact partial sum. This file states that invariant, its base case, its
step (one `tl.dot`, via `DotLift.dotAdd_step`), and reads the loaded operands back as the values of
`a` and `b` in the initial memory.
-/


/-- The inner-loop invariant of tile `T` after `q` steps of the reduction. -/
noncomputable def AccInv (d : Inst) (init : Mem) (T q : Nat)
    (acc : Tensor [128, 256] F32) : Prop :=
  ∀ (r : Fin 128) (n : Fin 256),
    128 * pidM d T + r.val < d.M → 256 * pidN d T + n.val < d.N →
    RowFin d init (128 * pidM d T + r.val) → ColFin d init (256 * pidN d T + n.val) →
    (acc.get2 r n).isFinite = true ∧
      |(acc.get2 r n).toRat
          - pSum d init (128 * pidM d T + r.val) (256 * pidN d T + n.val) q|
        ≤ E d init (128 * pidM d T + r.val) (256 * pidN d T + n.val) q

/-- Every element of the zero accumulator is `+0`. -/
theorem zeros_get2 (r : Fin 128) (n : Fin 256) :
    (tl.zeros [128, 256] .f32).get2 r n = (FloatV.ofPat 0 : F32) := rfl

/-- The zero accumulator satisfies the invariant at `q = 0`. -/
theorem AccInv_zero (d : Inst) (init : Mem) (T : Nat) :
    AccInv d init T 0 (tl.zeros [128, 256] .f32) := by
  intro r n _ _ _ _
  have hz := zeros_get2 r n
  have hfin : (FloatV.ofPat 0 : F32).isFinite = true := by
    rw [FloatV.isFinite_ofPat (f := Format.binary32) (by norm_num : 0 < 2 ^ 32)]
    exact Format.isFinite_zero Format.binary32_regular
  have hrat : (FloatV.ofPat 0 : F32).toRat = 0 := by
    rw [FloatV.toRat_ofPat (f := Format.binary32) (by norm_num : 0 < 2 ^ 32)]
    exact Format.toRat_of_isZero Format.isZero_zero
  refine ⟨?_, ?_⟩
  · rw [hz]; exact hfin
  · rw [hz, hrat]
    have hp : pSum d init (128 * pidM d T + r.val) (256 * pidN d T + n.val) 0 = 0 := by
      simp [pSum, partSum]
    rw [hp, E.eq_1]
    simp

/-- The room condition `DotLift.dotAdd_step` asks for, in the form the accumulator meets it. -/
theorem dot_room {d : Inst} (hd : d.ok) {init : Mem} {I J q : Nat}
    (hI : RowFin d init I) (hJ : ColFin d init J) (hq : q < steps d.K) {c : ℚ}
    (hc : |c| ≤ E d init I J q + |pSum d init I J q|) :
    2 * ((1 + Device.relDot16 64)
          * (|c| + ∑ k : Fin 64,
              Device.liftMag .f16 (aGet d init I (64 * q + k.val)).toRat
                * Device.liftMag .f16 (bGet d init J (64 * q + k.val)).toRat)
        + Device.absDot16 64 + 2 * Format.binary32.eta)
      ≤ Format.binary32.maxVal := by
  have hKdvd : 128 ∣ d.K := Nat.dvd_of_mod_eq_zero hd.K_dim.1
  have hK : 64 * (q + 1) ≤ d.K := by
    have h1 : q + 1 ≤ steps d.K := hq
    have h2 := Nat.mul_le_mul_left 64 h1
    have h3 : 64 * steps d.K = d.K := steps_mul hKdvd
    omega
  have hl0 : 0 ≤ ∑ k : Fin 64,
      Device.liftMag .f16 (aGet d init I (64 * q + k.val)).toRat
        * Device.liftMag .f16 (bGet d init J (64 * q + k.val)).toRat :=
    Finset.sum_nonneg fun k _ => mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
  have hl : (∑ k : Fin 64,
      Device.liftMag .f16 (aGet d init I (64 * q + k.val)).toRat
        * Device.liftMag .f16 (bGet d init J (64 * q + k.val)).toRat) ≤ 2 ^ (38 : ℤ) := by
    rw [← lSum_eq hK]
    exact lSum_le hd init I J q hI hJ hq
  have hx : |c| ≤ 2 ^ (46 : ℤ) := by
    have h2 : |pSum d init I J q| ≤ 2 ^ (45 : ℤ) :=
      pSum_abs_le hd init I J q hI hJ (le_of_lt hq)
    have h3 : E d init I J q ≤ 2 ^ (38 : ℤ) :=
      (E_le_dTol_q hd init I J q (le_of_lt hq)).trans (dTol_le hd init I J hI hJ)
    have h4 : (2 : ℚ) ^ (38 : ℤ) + 2 ^ (45 : ℤ) ≤ 2 ^ (46 : ℤ) := by norm_num
    linarith
  exact room_of_le hx hl0 hl

/-- One `tl.dot` step of the reduction, in one lane. -/
theorem dot_step_lane {d : Inst} (hd : d.ok) {P : Platform} {pr : InputPrecision}
    (hacc : Device.DotLiftAcc P ⟨.f16, .f16, .f32, pr, 0⟩
      Device.relDot16 Device.absDot16)
    {init : Mem} {I J q : Nat} (hI : RowFin d init I) (hJ : ColFin d init J)
    (hq : q < steps d.K)
    {acc r : Tensor [128, 256] F32} {a : Tensor [128, 64] F16} {b : Tensor [64, 256] F16}
    (i : Fin 128) (n : Fin 256)
    (hva : ∀ k : Fin 64,
      DType.val? .f16 (a.get2 i k) = some (aGet d init I (64 * q + k.val)).toRat)
    (hvb : ∀ k : Fin 64,
      DType.val? .f16 (b.get2 k n) = some (bGet d init J (64 * q + k.val)).toRat)
    (hc : (acc.get2 i n).isFinite = true)
    (hce : |(acc.get2 i n).toRat - pSum d init I J q| ≤ E d init I J q)
    (hdot : P.dot ⟨.f16, .f16, .f32, pr, 0⟩ a b acc r) :
    (r.get2 i n).isFinite = true ∧
      |(r.get2 i n).toRat - pSum d init I J (q + 1)| ≤ E d init I J (q + 1) := by
  have hKdvd : 128 ∣ d.K := Nat.dvd_of_mod_eq_zero hd.K_dim.1
  have hK : 64 * (q + 1) ≤ d.K := by
    have h1 : q + 1 ≤ steps d.K := hq
    have h2 := Nat.mul_le_mul_left 64 h1
    have h3 : 64 * steps d.K = d.K := steps_mul hKdvd
    omega
  have hcabs : |(acc.get2 i n).toRat| ≤ E d init I J q + |pSum d init I J q| := by
    have h1 : |(acc.get2 i n).toRat|
        = |((acc.get2 i n).toRat - pSum d init I J q) + pSum d init I J q| := by
      rw [sub_add_cancel]
    rw [h1]
    linarith [abs_add_le ((acc.get2 i n).toRat - pSum d init I J q) (pSum d init I J q)]
  have hroom := dot_room hd hI hJ hq hcabs
  obtain ⟨hfin, herr⟩ :=
    DotLift.dotAdd_step (P := P) (fusion := true) (mi := 0) (pr := pr)
      (rel := Device.relDot16) (abs := Device.absDot16) hacc
      (by unfold Device.relDot16; positivity)
      (by unfold Device.absDot16; positivity)
      (Or.inl hdot) i n (z := tl.zeros [128, 256] .f32) (DType.val?_zero_f32)
      (va := fun k : Fin 64 => (aGet d init I (64 * q + k.val)).toRat)
      (vb := fun k : Fin 64 => (bGet d init J (64 * q + k.val)).toRat)
      hva hvb hc hroom
  refine ⟨hfin, ?_⟩
  have hsum : (∑ k : Fin 64,
        (aGet d init I (64 * q + k.val)).toRat * (bGet d init J (64 * q + k.val)).toRat)
      = pSum d init I J (q + 1) - pSum d init I J q := by
    rw [pSum_succ hK]; ring
  have hstep : |(r.get2 i n).toRat
        - ((acc.get2 i n).toRat + (pSum d init I J (q + 1) - pSum d init I J q))|
      ≤ rho * (|(acc.get2 i n).toRat| + lSum d init I J q) + alpha := by
    rw [lSum_eq hK, ← hsum]
    exact herr
  have hred := red_arith (p0 := pSum d init I J q) (p1 := pSum d init I J (q + 1))
    (l := lSum d init I J q) (c := (acc.get2 i n).toRat) (r := (r.get2 i n).toRat)
    (e := E d init I J q) hstep hce
  rw [E.eq_2]
  exact hred

/-- One `tl.dot` step of the reduction, on the whole tile. -/
theorem AccInv_step {d : Inst} (hd : d.ok) {P : Platform} {pr : InputPrecision}
    (hacc : Device.DotLiftAcc P ⟨.f16, .f16, .f32, pr, 0⟩
      Device.relDot16 Device.absDot16)
    {init : Mem} {T q : Nat} (hq : q < steps d.K)
    {acc r : Tensor [128, 256] F32} (hinv : AccInv d init T q acc)
    {a : Tensor [128, 64] F16} {b : Tensor [64, 256] F16}
    (hva : ∀ (i : Fin 128) (k : Fin 64), 128 * pidM d T + i.val < d.M →
      RowFin d init (128 * pidM d T + i.val) →
      DType.val? .f16 (a.get2 i k)
        = some (aGet d init (128 * pidM d T + i.val) (64 * q + k.val)).toRat)
    (hvb : ∀ (k : Fin 64) (n : Fin 256), 256 * pidN d T + n.val < d.N →
      ColFin d init (256 * pidN d T + n.val) →
      DType.val? .f16 (b.get2 k n)
        = some (bGet d init (256 * pidN d T + n.val) (64 * q + k.val)).toRat)
    (hdot : P.dot ⟨.f16, .f16, .f32, pr, 0⟩ a b acc r) :
    AccInv d init T (q + 1) r := by
  intro i n hiM hnN hRi hCn
  obtain ⟨hc, hce⟩ := hinv i n hiM hnN hRi hCn
  exact dot_step_lane hd hacc hRi hCn hq i n
    (fun k => hva i k hiM hRi) (fun k => hvb k n hnN hCn) hc hce hdot

/-! ### Reading the loaded operands back -/

/-- A read of `a` in the covered footprint gives the same bytes in the current memory as in the
initial one. -/
theorem aGet_read {d : Inst} {init m : Mem} {I kk : Nat} (hI : I < d.M) (hk : kk < d.K)
    (hR : ∀ x ∈ d.a.elems (aIdx d), m x = init x) :
    m.read .f16 (d.a.ptr (↑(I * d.lda + kk))).addr
      = init.read .f16 (d.a.ptr (↑(I * d.lda + kk))).addr := by
  apply Mem.read_congr
  intro x hx
  exact hR x (Buf.span_sub_elems d.a (S := aIdx d) (j := I * d.lda + kk)
    ⟨I, kk, hI, hk, rfl⟩ hx)

/-- A finite loaded element of `a` has the value of the corresponding initial element. -/
theorem aVal {d : Inst} {init m : Mem} {I kk : Nat} (hI : I < d.M) (hk : kk < d.K)
    (hR : ∀ x ∈ d.a.elems (aIdx d), m x = init x) {v : F16}
    (hread : m.read .f16 (d.a.ptr (↑(I * d.lda + kk))).addr = some v)
    (hfin : v.isFinite = true) :
    DType.val? .f16 v = some (aGet d init I kk).toRat := by
  have h1 := aGet_read (d := d) (init := init) (m := m) (I := I) (kk := kk) hI hk hR
  rw [h1] at hread
  have hget : aGet d init I kk = v := by
    simp only [aGet, hread, Option.getD_some]
  rw [hget]
  simp only [DType.val?]
  rw [ite_eq_left hfin]

/-- A read of `b` in the covered footprint gives the same bytes in the current memory as in the
initial one. -/
theorem bGet_read {d : Inst} {init m : Mem} {J kk : Nat} (hJ : J < d.N) (hk : kk < d.K)
    (hR : ∀ x ∈ d.b.elems (bIdx d), m x = init x) :
    m.read .f16 (d.b.ptr (↑(J * d.ldb + kk))).addr
      = init.read .f16 (d.b.ptr (↑(J * d.ldb + kk))).addr := by
  apply Mem.read_congr
  intro x hx
  exact hR x (Buf.span_sub_elems d.b (S := bIdx d) (j := J * d.ldb + kk)
    ⟨J, kk, hJ, hk, rfl⟩ hx)

/-- A finite loaded element of `b` has the value of the corresponding initial element. -/
theorem bVal {d : Inst} {init m : Mem} {J kk : Nat} (hJ : J < d.N) (hk : kk < d.K)
    (hR : ∀ x ∈ d.b.elems (bIdx d), m x = init x) {v : F16}
    (hread : m.read .f16 (d.b.ptr (↑(J * d.ldb + kk))).addr = some v)
    (hfin : v.isFinite = true) :
    DType.val? .f16 v = some (bGet d init J kk).toRat := by
  have h1 := bGet_read (d := d) (init := init) (m := m) (J := J) (kk := kk) hJ hk hR
  rw [h1] at hread
  have hget : bGet d init J kk = v := by
    simp only [bGet, hread, Option.getD_some]
  rw [hget]
  simp only [DType.val?]
  rw [ite_eq_left hfin]

/-! ### The end of the reduction -/

/-- After all `steps K` inner steps the partial sum is the whole dot product. -/
theorem pSum_steps {d : Inst} (hd : d.ok) (init : Mem) (I J : Nat) :
    pSum d init I J (steps d.K) = dSum d init I J := by
  have hKdvd : 128 ∣ d.K := Nat.dvd_of_mod_eq_zero hd.K_dim.1
  have hKm : 64 * steps d.K = d.K := steps_mul hKdvd
  simp only [pSum, partSum, dSum, dotSum, prodV]
  rw [hKm]
  exact Finset.sum_congr rfl fun e he => by
    have heK : e < d.K := Finset.mem_range.mp he
    rw [ite_eq_left heK]

/-- The invariant at the end of the reduction gives the lane specification. -/
theorem AccInv_final {d : Inst} (hd : d.ok) {init : Mem} {T : Nat}
    {acc : Tensor [128, 256] F32} (hinv : AccInv d init T (steps d.K) acc)
    (r : Fin 128) (n : Fin 256) (hrM : 128 * pidM d T + r.val < d.M)
    (hnN : 256 * pidN d T + n.val < d.N)
    (hR : RowFin d init (128 * pidM d T + r.val))
    (hC : ColFin d init (256 * pidN d T + n.val)) :
    (acc.get2 r n).isFinite = true ∧
      |(acc.get2 r n).toRat
          - dSum d init (128 * pidM d T + r.val) (256 * pidN d T + n.val)|
        ≤ dTol d init (128 * pidM d T + r.val) (256 * pidN d T + n.val) := by
  obtain ⟨hfin, herr⟩ := hinv r n hrM hnN hR hC
  refine ⟨hfin, ?_⟩
  rw [← pSum_steps hd init]
  exact herr.trans (E_le_dTol_q hd init (128 * pidM d T + r.val) (256 * pidN d T + n.val)
    (steps d.K) (le_refl _))

/-!
## The inner reduction loop is `Safe`

One step of the inner loop loads the two operand blocks of a tile with the descriptors, runs one
`tl.dot`, and leaves the accumulator satisfying `AccInv` one step further. The inner loop itself is
a `tl.forRange` with an empty write footprint: it only reads `a` and `b`, and its lengths are the
tile count and `steps K`.
-/


set_option maxHeartbeats 1000000 in
/-- One step of the inner loop: load the two operand blocks and run one `tl.dot`. -/
theorem safe_step {d : Inst} (hd : d.ok) {E : Env} {pr : InputPrecision}
    (hacc : Device.DotLiftAcc E.P ⟨.f16, .f16, .f32, pr, 0⟩
      Device.relDot16 Device.absDot16)
    (htot : Device.DotTotal E.P ⟨.f16, .f16, .f32, pr, 0⟩)
    {init : Mem} {T q : Nat} {m : Mem} {acc : Tensor [128, 256] F32}
    {ki offs_am offs_bn : I32}
    (hq : q < steps d.K) (hki : ki.toInt = (q : Int))
    (hvaM : offs_am.toInt = ((128 * pidM d T : Nat) : Int))
    (hvbN : offs_bn.toInt = ((256 * pidN d T : Nat) : Int))
    (hinv : AccInv d init T q acc)
    (hvA : d.a.Valid m) (hvB : d.b.Valid m)
    (hRA : ∀ x ∈ d.a.elems (aIdx d), m x = init x)
    (hRB : ∀ x ∈ d.b.elems (bIdx d), m x = init x) :
    Safe E (R d) ∅ (fun a m' => AccInv d init T (q + 1) a ∧ m' = m) m
      (do
        let offs_k := ki * (IntV.ofNat 64 : I32)
        let a ← tl.desc_load (bs := [128, 64]) (aDesc d) [offs_am, offs_k]
        let b ← tl.desc_load (bs := [256, 64]) (bDesc d) [offs_bn, offs_k]
        tl.dot { a := .f16, b := .f16, acc := .f32, precision := pr, maxImpreciseAcc := 0 }
          a b.trans acc) := by
  have hKdvd : 128 ∣ d.K := Nat.dvd_of_mod_eq_zero hd.K_dim.1
  have hKm : 64 * steps d.K = d.K := steps_mul hKdvd
  have hq128 : q < 128 := lt_of_lt_of_le hq (steps_le hd.K_le)
  have hok : (ki * (IntV.ofNat 64 : I32)).toInt = ((64 * q : Nat) : Int) := by
    have h64 : ((IntV.ofNat 64 : I32)).toInt = ((64 : Nat) : Int) :=
      IntV.I32.toInt_ofNat (by norm_num)
    have hm := IntV.I32.mul_nat (a := ki) (b := (IntV.ofNat 64 : I32)) (x := q) (y := 64)
      hki h64 (by omega)
    rw [hm]
    push_cast
    ring
  refine safe_desc_load (b := d.a) (bs := [128, 64]) (D := aDesc d)
    (offsets := [offs_am, ki * (IntV.ofNat 64 : I32)]) rfl ?_ ?_ hvA ?_ ?_
  · exact aDesc_access hd offs_am.toInt (ki * (IntV.ofNat 64 : I32)).toInt
      (by rw [hok]; push_cast; omega)
  · intro i hinb
    exact aDesc_len hd offs_am.toInt (ki * (IntV.ofNat 64 : I32)).toInt i hinb
  · exact (d.a.elems_mono (aDesc_img_subset d offs_am.toInt (ki * (IntV.ofNat 64 : I32)).toInt))
      |>.trans Set.subset_union_left
  · intro v1 hr1 _hp1
    refine safe_desc_load (b := d.b) (bs := [256, 64]) (D := bDesc d)
      (offsets := [offs_bn, ki * (IntV.ofNat 64 : I32)]) rfl ?_ ?_ hvB ?_ ?_
    · exact bDesc_access hd offs_bn.toInt (ki * (IntV.ofNat 64 : I32)).toInt
        (by rw [hok]; push_cast; omega)
    · intro i hinb
      exact bDesc_len hd offs_bn.toInt (ki * (IntV.ofNat 64 : I32)).toInt i hinb
    · exact (d.b.elems_mono (bDesc_img_subset d offs_bn.toInt (ki * (IntV.ofNat 64 : I32)).toInt))
        |>.trans Set.subset_union_right
    · intro v2 hr2 _hp2
      have hva : ∀ (i : Fin 128) (k : Fin 64), 128 * pidM d T + i.val < d.M →
          RowFin d init (128 * pidM d T + i.val) →
          DType.val? .f16 (v1.get2 i k)
            = some (aGet d init (128 * pidM d T + i.val) (64 * q + k.val)).toRat := by
        intro i k hiM hRow
        let idx : Idx [128, 64] := (i, (k, ()))
        have hidx0 : (idx.coord 0 : Int) = (i.val : Int) := by simp [idx, Idx.coord]
        have hidx1 : (idx.coord 1 : Int) = (k.val : Int) := by simp [idx, Idx.coord]
        have h0 : 0 ≤ offs_am.toInt + (i.val : Int) := by
          rw [hvaM]; push_cast; omega
        have h1 : offs_am.toInt + (i.val : Int) < (d.M : Int) := by
          rw [hvaM]; exact_mod_cast hiM
        have h2 : 0 ≤ (ki * (IntV.ofNat 64 : I32)).toInt + (k.val : Int) := by
          rw [hok]; push_cast; omega
        have h3 : (ki * (IntV.ofNat 64 : I32)).toInt + (k.val : Int) < (d.K : Int) := by
          rw [hok]; push_cast; omega
        have hrow : (offs_am.toInt + (idx.coord 0 : Int))
            = ((128 * pidM d T + i.val : Nat) : Int) := by
          rw [hidx0, hvaM]; push_cast; ring
        have hcol : ((ki * (IntV.ofNat 64 : I32)).toInt + (idx.coord 1 : Int))
            = ((64 * q + k.val : Nat) : Int) := by
          rw [hidx1, hok]; push_cast; ring
        have hinb : (aDesc d).inbAt (TensorDesc.coord (bs := [128, 64])
            [offs_am.toInt, (ki * (IntV.ofNat 64 : I32)).toInt] idx) = true := by
          refine (aDesc_inbAt d offs_am.toInt (ki * (IntV.ofNat 64 : I32)).toInt idx).mpr ?_
          refine ⟨?_, ?_, ?_, ?_⟩
          · rw [hidx0]; exact h0
          · rw [hidx0]; exact h1
          · rw [hidx1]; exact h2
          · rw [hidx1]; exact h3
        have helem : ((aDesc d).elemAt (TensorDesc.coord (bs := [128, 64])
            [offs_am.toInt, (ki * (IntV.ofNat 64 : I32)).toInt] idx)).toNat
            = (128 * pidM d T + i.val) * d.lda + (64 * q + k.val) := by
          rw [elemAt_toNat_two (bs := [128, 64]) (A := d.M) (B := d.K) (s := d.lda)
            (aDesc d) (by simp [aDesc]) (by simp [aDesc])
            _ _ idx (by rw [hidx0]; exact h0) (by rw [hidx1]; exact h2)]
          rw [hrow, hcol, Int.toNat_natCast, Int.toNat_natCast]
        have hread := hr1 idx hinb
        change m.read .f16 (d.a.ptr ↑((aDesc d).elemAt (TensorDesc.coord (bs := [128, 64])
          [offs_am.toInt, (ki * (IntV.ofNat 64 : I32)).toInt] idx)).toNat).addr = _ at hread
        rw [helem] at hread
        have hread2 : m.read .f16
            (d.a.ptr (↑((128 * pidM d T + i.val) * d.lda + (64 * q + k.val)))).addr
            = some (v1.get2 i k) := hread
        have hvaleq : v1.get2 i k
            = aGet d init (128 * pidM d T + i.val) (64 * q + k.val) := by
          have h1 := aGet_read (d := d) (init := init) (m := m)
            (I := 128 * pidM d T + i.val) (kk := 64 * q + k.val) hiM (by omega) hRA
          rw [h1] at hread2
          simp only [aGet, hread2, Option.getD_some]
        exact aVal (d := d) (init := init) (m := m) (I := 128 * pidM d T + i.val)
          (kk := 64 * q + k.val) hiM (by omega) hRA hread2
          (by rw [hvaleq]; exact hRow (64 * q + k.val) (by omega))
      have hvb : ∀ (k : Fin 64) (n : Fin 256), 256 * pidN d T + n.val < d.N →
          ColFin d init (256 * pidN d T + n.val) →
          DType.val? .f16 (v2.trans.get2 k n)
            = some (bGet d init (256 * pidN d T + n.val) (64 * q + k.val)).toRat := by
        intro k n hnN hCol
        simp only [Tensor.get2_trans]
        let idx : Idx [256, 64] := (n, (k, ()))
        have hidx0 : (idx.coord 0 : Int) = (n.val : Int) := by simp [idx, Idx.coord]
        have hidx1 : (idx.coord 1 : Int) = (k.val : Int) := by simp [idx, Idx.coord]
        have h0 : 0 ≤ offs_bn.toInt + (n.val : Int) := by
          rw [hvbN]; push_cast; omega
        have h1 : offs_bn.toInt + (n.val : Int) < (d.N : Int) := by
          rw [hvbN]; exact_mod_cast hnN
        have h2 : 0 ≤ (ki * (IntV.ofNat 64 : I32)).toInt + (k.val : Int) := by
          rw [hok]; push_cast; omega
        have h3 : (ki * (IntV.ofNat 64 : I32)).toInt + (k.val : Int) < (d.K : Int) := by
          rw [hok]; push_cast; omega
        have hrow : (offs_bn.toInt + (idx.coord 0 : Int))
            = ((256 * pidN d T + n.val : Nat) : Int) := by
          rw [hidx0, hvbN]; push_cast; ring
        have hcol : ((ki * (IntV.ofNat 64 : I32)).toInt + (idx.coord 1 : Int))
            = ((64 * q + k.val : Nat) : Int) := by
          rw [hidx1, hok]; push_cast; ring
        have hinb : (bDesc d).inbAt (TensorDesc.coord (bs := [256, 64])
            [offs_bn.toInt, (ki * (IntV.ofNat 64 : I32)).toInt] idx) = true := by
          refine (bDesc_inbAt d offs_bn.toInt (ki * (IntV.ofNat 64 : I32)).toInt idx).mpr ?_
          refine ⟨?_, ?_, ?_, ?_⟩
          · rw [hidx0]; exact h0
          · rw [hidx0]; exact h1
          · rw [hidx1]; exact h2
          · rw [hidx1]; exact h3
        have helem : ((bDesc d).elemAt (TensorDesc.coord (bs := [256, 64])
            [offs_bn.toInt, (ki * (IntV.ofNat 64 : I32)).toInt] idx)).toNat
            = (256 * pidN d T + n.val) * d.ldb + (64 * q + k.val) := by
          rw [elemAt_toNat_two (bs := [256, 64]) (A := d.N) (B := d.K) (s := d.ldb)
            (bDesc d) (by simp [bDesc]) (by simp [bDesc])
            _ _ idx (by rw [hidx0]; exact h0) (by rw [hidx1]; exact h2)]
          rw [hrow, hcol, Int.toNat_natCast, Int.toNat_natCast]
        have hread := hr2 idx hinb
        change m.read .f16 (d.b.ptr ↑((bDesc d).elemAt (TensorDesc.coord (bs := [256, 64])
          [offs_bn.toInt, (ki * (IntV.ofNat 64 : I32)).toInt] idx)).toNat).addr = _ at hread
        rw [helem] at hread
        have hread2 : m.read .f16
            (d.b.ptr (↑((256 * pidN d T + n.val) * d.ldb + (64 * q + k.val)))).addr
            = some (v2.get2 n k) := hread
        have hvbeq : v2.get2 n k
            = bGet d init (256 * pidN d T + n.val) (64 * q + k.val) := by
          have h1 := bGet_read (d := d) (init := init) (m := m)
            (J := 256 * pidN d T + n.val) (kk := 64 * q + k.val) hnN (by omega) hRB
          rw [h1] at hread2
          simp only [bGet, hread2, Option.getD_some]
        exact bVal (d := d) (init := init) (m := m) (J := 256 * pidN d T + n.val)
          (kk := 64 * q + k.val) hnN (by omega) hRB hread2
          (by rw [hvbeq]; exact hCol (64 * q + k.val) (by omega))
      have hstep : Safe E (R d) ∅ (fun a m' => AccInv d init T (q + 1) a ∧ m' = m) m
          ((tl.dot { a := .f16, b := .f16, acc := .f32, precision := pr, maxImpreciseAcc := 0 }
            v1 v2.trans acc) >>= Prog.ret) := by
        refine safe_dot (k := Prog.ret) (hex := htot v1 v2.trans acc) ?_
        intro r hdot
        exact .ret ⟨AccInv_step hd hacc hq hinv hva hvb hdot, rfl⟩
      simpa only [Prog.bind_eq, Prog.bind_ret] using hstep

set_option maxHeartbeats 1000000 in
/-- The inner `tl.forRange` reduction loop is safe: it only reads, and after it the accumulator
satisfies `AccInv` at `steps K`. -/
theorem safe_inner {d : Inst} (hd : d.ok) {E : Env}
    (hacc : Device.DotLiftAcc E.P ⟨.f16, .f16, .f32, .tf32, 0⟩
      Device.relDot16 Device.absDot16)
    (htot : Device.DotTotal E.P ⟨.f16, .f16, .f32, .tf32, 0⟩)
    {init : Mem} {T : Nat} {m : Mem} {acc : Tensor [128, 256] F32}
    {kt offs_am offs_bn : I32}
    (hkt : kt.toInt = (steps d.K : Int))
    (hvaM : offs_am.toInt = ((128 * pidM d T : Nat) : Int))
    (hvbN : offs_bn.toInt = ((256 * pidN d T : Nat) : Int))
    (hinv : AccInv d init T 0 acc)
    (hvA : d.a.Valid m) (hvB : d.b.Valid m)
    (hRA : ∀ x ∈ d.a.elems (aIdx d), m x = init x)
    (hRB : ∀ x ∈ d.b.elems (bIdx d), m x = init x) :
    Safe E (R d) ∅ (fun a m' => AccInv d init T (steps d.K) a ∧ m' = m) m
      (tl.forRange (IntV.ofInt 0 : I32) kt (IntV.ofInt 1 : I32) acc
        (fun ki acc => do
          let offs_k := ki * (IntV.ofNat 64 : I32)
          let a ← tl.desc_load (bs := [128, 64]) (aDesc d) [offs_am, offs_k]
          let b ← tl.desc_load (bs := [256, 64]) (bDesc d) [offs_bn, offs_k]
          tl.dot { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }
            a b.trans acc)) := by
  have h0 : (IntV.ofInt 0 : I32).toInt = 0 :=
    IntV.I32.toInt_ofInt (by norm_num)
  have h1 : (IntV.ofInt 1 : I32).toInt = 1 :=
    IntV.I32.toInt_ofInt (by norm_num)
  have hKle : steps d.K ≤ 2 ^ 13 := (steps_le hd.K_le).trans (by norm_num)
  have hKint : (steps d.K : Int) ≤ 2 ^ 31 - 1 := by
    have h1 : (steps d.K : Int) ≤ 2 ^ 13 := by exact_mod_cast hKle
    have h2 : (2 : Int) ^ 13 ≤ 2 ^ 31 - 1 := by norm_num
    omega
  have htc : tl.tripCount (IntV.ofInt 0 : I32).toInt kt.toInt
      (IntV.ofInt 1 : I32).toInt = steps d.K := by
    rw [h0, h1, hkt]
    simp only [tl.tripCount]
    rw [show ((steps d.K : Int) - 0 + 1 - 1) = (steps d.K : Int) by ring]
    simp
  have hrange : tl.RangeOk (IntV.ofInt 0 : I32) kt (IntV.ofInt 1 : I32) := by
    unfold tl.RangeOk
    rw [h0, h1, hkt]
    have hint : IntV.maxInt true 32 = 2 ^ 31 - 1 := by
      norm_num [IntV.maxInt]
    rw [hint]
    refine ⟨by norm_num, by omega, by omega, by omega⟩
  have hmain := safe_forRange (E := E) (R := R d) (m := m)
    (Q := fun a m' => AccInv d init T (steps d.K) a ∧ m' = m)
    (lo := (IntV.ofInt 0 : I32)) (hi := kt)
    (step := (IntV.ofInt 1 : I32))
    (init := acc)
    (body := fun ki acc => do
      let offs_k := ki * (IntV.ofNat 64 : I32)
      let a ← tl.desc_load (bs := [128, 64]) (aDesc d) [offs_am, offs_k]
      let b ← tl.desc_load (bs := [256, 64]) (bDesc d) [offs_bn, offs_k]
      tl.dot { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }
        a b.trans acc)
    (k := Prog.ret) hrange (fun t st m₁ => AccInv d init T t st ∧ m₁ = m)
    (fun _ => ∅) (Wk := ∅) ?_ ?_ ?_ ?_ ?_
  simpa only [iterW_empty, Set.empty_union, Prog.bind_eq, Prog.bind_ret, htc] using hmain
  · intro t u _ _; simp
  · rw [iterW_empty]; simp
  · exact ⟨hinv, rfl⟩
  · intro t ht st m₁ hI
    obtain ⟨hinv', hm⟩ := hI
    subst hm
    rw [htc] at ht
    have hki : (IntV.ofInt ((IntV.ofInt 0 : I32).toInt + ↑t * (IntV.ofInt 1 : I32).toInt)
        : I32).toInt = (t : Int) := by
      rw [h0, h1]
      rw [show ((0 : Int) + ↑t * 1) = (t : Int) by ring]
      exact IntV.I32.toInt_ofInt (by omega)
    exact safe_step hd hacc htot ht hki hvaM hvbN hinv' hvA hvB hRA hRB
  · intro st m₁ hI
    rw [htc] at hI
    obtain ⟨hinv', hm⟩ := hI
    subst hm
    exact .ret ⟨hinv', rfl⟩

end Claims.TmaPersistentMatmul

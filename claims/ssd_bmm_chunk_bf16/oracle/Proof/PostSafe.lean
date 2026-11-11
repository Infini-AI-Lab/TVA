import Claim
import TVA.Proof
import Proof.Step

/-!
# The store after the loop

After the loop the kernel reloads the two `seq_idx` slices of its tile, forms the request mask
`seq_idx_m == seq_idx_n`, and stores the tile into `out` (`acc` where the mask holds, `+0`
elsewhere). The turn leaves the specification's `Q`: every enabled lane of the tile holds the
rounded dot product of its request, or `+0` when the request differs.
-/

open TVA
namespace Claims.SsdBmm

set_option maxHeartbeats 800000

/-! ## Small helpers on the store's value -/

/-- `a == b` is `true` exactly when the values agree. -/
theorem intv_eq_true_iff {w : Nat} (a b : IntV true w) :
    (IntV.eq a b = true ↔ a.toInt = b.toInt) := by
  rw [IntV.eq_eq]; simp

/-- `a == b` is `false` exactly when the values differ. -/
theorem intv_eq_false_iff {w : Nat} (a b : IntV true w) :
    (IntV.eq a b = false ↔ a.toInt ≠ b.toInt) := by
  constructor
  · intro h hEq
    have h1 : IntV.eq a b = true := (intv_eq_true_iff a b).mpr hEq
    rw [h1] at h
    exact Bool.noConfusion h
  · intro h
    cases hb : IntV.eq a b with
    | false => rfl
    | true => exact absurd ((intv_eq_true_iff a b).mp hb) h

/-- `out.stride(2) = C²` is an `int32`. -/
theorem oH_lt_31 {d : Inst} (hd : d.ok) : oH d < 2 ^ 31 := by
  have h1 : oH d ≤ oC d := by
    unfold oH oC
    exact Nat.le_mul_of_pos_left _ hd.G_pos
  exact lt_of_le_of_lt h1 hd.oC_lt

/-- Lane `(m, n)` of `tl.where c x y` picks `x` or `y` by the lane of `c`. -/
theorem where_get_lane {BM BN : Nat} (c : Tensor [BM, BN] Bool) (x y : Tensor [BM, BN] F32)
    (m : Fin BM) (n : Fin BN) :
    (tl.where c x y).get (m, n, ()) =
      if c.get (m, n, ()) then x.get (m, n, ()) else y.get (m, n, ()) := rfl

/-- Lane `(m, n)` of `x[:, None] == y[None, :]` compares the lanes `x m` and `y n`. -/
theorem eq_col_row_get {BM BN : Nat} (x : Tensor [BM] I32) (y : Tensor [BN] I32)
    (m : Fin BM) (n : Fin BN) :
    ((x.col BN).eq (y.row BM)).get (m, n, ()) = IntV.eq (x.get (m, ())) (y.get (n, ())) := rfl

/-- Where the condition holds, `tl.where` takes the first value. -/
theorem where_get_true {BM BN : Nat} (c : Tensor [BM, BN] Bool) (x y : Tensor [BM, BN] F32)
    (m : Fin BM) (n : Fin BN) (h : c.get (m, n, ()) = true) :
    (tl.where c x y).get (m, n, ()) = x.get (m, n, ()) := by
  rw [where_get_lane]
  exact ite_eq_left h

/-- Where the condition fails, `tl.where` takes the second value. -/
theorem where_get_false {BM BN : Nat} (c : Tensor [BM, BN] Bool) (x y : Tensor [BM, BN] F32)
    (m : Fin BM) (n : Fin BN) (h : c.get (m, n, ()) = false) :
    (tl.where c x y).get (m, n, ()) = y.get (m, n, ()) := by
  rw [where_get_lane]
  exact ite_eq_right (by rw [h]; exact Bool.false_ne_true)

/-- The store after the loop: from the invariant after all `steps d.K` turns, the program leaves
`Q`. -/
theorem post_safe {P : Platform} {d : Inst} (hd : d.ok) (i : Fin (launch d).size)
    {st : StT} {m₁ init : Mem} (hpre : Pre d init) (hinv : inv d init i (steps d.K) st m₁) :
    Safe ((launch d).env P) (R d init i) (W d init i) (fun _ m => Q d init i m) m₁
      (do
        let l1 ← wide (IntV.ofNat d.C)
        let l2 ← wide (IntV.ofNat d.T)
        let l3 ← wide (IntV.ofNat d.C)
        let c1 ←
          tl.convert .i32 .i64
            ((Tensor.scalar (IntV.ofInt (((i : Nat) % gx d / ((d.C + (BN - 1)) / BN) : ℕ) : ℤ) : I32)).item *
                  (IntV.ofNat BM : I32) + tl.arange 0 BM)
        let seq_idx_m ←
          tl.load
            (d.seqIdx.ptr 0 +
                ((IntV.ofInt (0 : ℤ) : I64) +
                    (Tensor.scalar (IntV.ofInt (((i : Nat) / gx d / d.G : ℕ) : ℤ) : I64)).item *
                      IntV.ofInt (d.C : ℤ) * IntV.ofInt (1 : ℤ)) +
              ((Tensor.scalar (IntV.ofInt (((i : Nat) % gx d / ((d.C + (BN - 1)) / BN) : ℕ) : ℤ) : I32)).item *
                      (IntV.ofNat BM : I32) + tl.arange 0 BM) *
                (IntV.ofNat (1 : Nat) : I32))
            (some
              (c1.ltS
                (tl.iminimum (Tensor.scalar l1)
                    (Tensor.scalar
                      (l2 - (Tensor.scalar (IntV.ofInt (((i : Nat) / gx d / d.G : ℕ) : ℤ) : I64)).item * l3))).item))
            (some (tl.full [BM] (IntV.ofInt (-1 : ℤ) : I32)))
        let c2 ←
          tl.convert .i32 .i64
            ((Tensor.scalar (IntV.ofInt (((i : Nat) % gx d % ((d.C + (BN - 1)) / BN) : ℕ) : ℤ) : I32)).item *
                  (IntV.ofNat BN : I32) + tl.arange 0 BN)
        let seq_idx_n ←
          tl.load
            (d.seqIdx.ptr 0 +
                ((IntV.ofInt (0 : ℤ) : I64) +
                    (Tensor.scalar (IntV.ofInt (((i : Nat) / gx d / d.G : ℕ) : ℤ) : I64)).item *
                      IntV.ofInt (d.C : ℤ) * IntV.ofInt (1 : ℤ)) +
              ((Tensor.scalar (IntV.ofInt (((i : Nat) % gx d % ((d.C + (BN - 1)) / BN) : ℕ) : ℤ) : I32)).item *
                      (IntV.ofNat BN : I32) + tl.arange 0 BN) *
                (IntV.ofNat (1 : Nat) : I32))
            (some
              (c2.ltS
                (tl.iminimum (Tensor.scalar l1)
                    (Tensor.scalar
                      (l2 - (Tensor.scalar (IntV.ofInt (((i : Nat) / gx d / d.G : ℕ) : ℤ) : I64)).item * l3))).item))
            (some (tl.full [BN] (IntV.ofInt (-2 : ℤ) : I32)))
        let l4 ← batchOff (IntV.ofNat 0 : I32) (intArg (oB d))
        let l5 ← wide (IntV.ofNat (oC d))
        let l6 ← wide (IntV.ofNat (oH d))
        tl.store
            (d.out.ptr 0 +
                (l4 + (Tensor.scalar (IntV.ofInt (((i : Nat) / gx d / d.G : ℕ) : ℤ) : I64)).item * l5 +
                  IntV.ofInt (↑(hOf d i) : ℤ) * l6) +
              ((IntV.ofNat d.C : I32) *
                  ((Tensor.scalar (IntV.ofInt (((i : Nat) % gx d / ((d.C + (BN - 1)) / BN) : ℕ) : ℤ) : I32)).item *
                        (IntV.ofNat BM : I32) + tl.arange 0 BM).col
                    BN +
                ((Tensor.scalar (IntV.ofInt (((i : Nat) % gx d % ((d.C + (BN - 1)) / BN) : ℕ) : ℤ) : I32)).item *
                        (IntV.ofNat BN : I32) + tl.arange 0 BN).row
                    BM *
                  (IntV.ofNat (1 : Nat) : I32) :
                Tensor [BM, BN] I32))
            (tl.where ((seq_idx_m.col BN).eq (seq_idx_n.row BM)) st.2.2 (tl.full [BM, BN] zf32))
            (some
              (((((Tensor.scalar (IntV.ofInt (((i : Nat) % gx d / ((d.C + (BN - 1)) / BN) : ℕ) : ℤ) : I32)).item *
                            (IntV.ofNat BM : I32) + tl.arange 0 BM).col
                        BN).ltS
                    (IntV.ofNat d.C : I32)).and
                ((((Tensor.scalar (IntV.ofInt (((i : Nat) % gx d % ((d.C + (BN - 1)) / BN) : ℕ) : ℤ) : I32)).item *
                          (IntV.ofNat BN : I32) + tl.arange 0 BN).row
                      BM).ltS
                  (IntV.ofNat d.C : I32))))) := by
  have hmm : m₁ = init := hinv.1
  subst m₁
  have hC31 : d.C < 2 ^ 31 := C_lt_31 hd
  have hoH31 : oH d < 2 ^ 31 := oH_lt_31 hd
  -- the three `wide`s of the scalar arithmetic
  refine safe_wide_nat d.C hC31 ?_
  refine safe_wide_nat d.T hd.T_lt ?_
  refine safe_wide_nat d.C hC31 ?_
  -- the `seq_idx` load of the `m` rows
  refine safe_convert (d := .i32) (d' := .i64) (x := offsM d i) ?_ ?_
  · intro j
    exact ⟨_, ConvRel.iff_cast_i32_i64.mpr rfl⟩
  · intro c1 hc1
    have hc1' : c1 = conv64 (offsM d i) := Tensor.ext (funext fun j => (ConvRel.iff_cast_i32_i64.mp (hc1 j)).trans (IntV.cast_eq_ofInt _))
    rw [hc1']
    refine safe_load_buf d.seqIdx (ptrs := seqPtrsM d i) (mask := some (maskSeqM d i))
      (g := fun idx => rowA d i idx.1.val) (en := fun idx => enM d i idx.1.val) ?_ ?_
      hpre.2.2.1 ?_ ?_
    · refine Buf.addrs_of_get d.seqIdx (mask := maskSeqM d i) ?_ ?_
      · intro idx
        exact maskSeqM_get hd i idx.1
      · intro idx hen
        exact seqPtrsM_get hd i idx.1
    · intro idx hen
      exact lt_of_lt_of_le (rowA_lt_of_chunk hd i (enM_row hen)) hd.seq_len
    · refine le_trans (Buf.elems_sub_bytes d.seqIdx ?_) ?_
      · rintro j ⟨idx, hen, rfl⟩
        exact lt_of_lt_of_le (rowA_lt_of_chunk hd i (enM_row hen)) hd.seq_len
      · intro x hx
        exact Or.inr hx
    · intro vm hreadM hotherM
      -- the `seq_idx` load of the `n` columns
      refine safe_convert (d := .i32) (d' := .i64) (x := offsN d i) ?_ ?_
      · intro j
        exact ⟨_, ConvRel.iff_cast_i32_i64.mpr rfl⟩
      · intro c2 hc2
        have hc2' : c2 = conv64 (offsN d i) := Tensor.ext (funext fun j => (ConvRel.iff_cast_i32_i64.mp (hc2 j)).trans (IntV.cast_eq_ofInt _))
        rw [hc2']
        refine safe_load_buf d.seqIdx (ptrs := seqPtrsN d i) (mask := some (maskSeqN d i))
          (g := fun idx => colB d i idx.1.val) (en := fun idx => enN d i idx.1.val) ?_ ?_
          hpre.2.2.1 ?_ ?_
        · refine Buf.addrs_of_get d.seqIdx (mask := maskSeqN d i) ?_ ?_
          · intro idx
            exact maskSeqN_get hd i idx.1
          · intro idx hen
            exact seqPtrsN_get hd i idx.1
        · intro idx hen
          exact lt_of_lt_of_le (colB_lt_of_chunk hd i (enN_col hen)) hd.seq_len
        · refine le_trans (Buf.elems_sub_bytes d.seqIdx ?_) ?_
          · rintro j ⟨idx, hen, rfl⟩
            exact lt_of_lt_of_le (colB_lt_of_chunk hd i (enN_col hen)) hd.seq_len
          · intro x hx
            exact Or.inr hx
        · intro vn hreadN hotherN
          -- the three `wide`s of the output offsets
          refine safe_batchOff_zero (oB d) ?_
          refine safe_wide_nat (oC d) hd.oC_lt ?_
          refine safe_wide_nat (oH d) hoH31 ?_
          -- the store of the tile
          refine safe_last ?_
          refine safe_store_buf d.out (ptrs := outPtrs d i) (mask := some (maskStore d i))
            (vals := (tl.where ((vm.col BN).eq (vn.row BM)) st.2.2 (tl.full [BM, BN] zf32)))
            (g := fun idx => oIdx d (cOf d i) (hOf d i) (mOf d i * BM + idx.1.val)
              (nOf d i * BN + idx.2.1.val))
            (en := fun idx => enS d i idx.1.val idx.2.1.val) ?_ ?_ ?_ hpre.2.2.2.1 ?_ ?_
          · refine Buf.addrs_of_get d.out (mask := maskStore d i) ?_ ?_
            · intro idx
              exact maskStore_get hd i idx.1 idx.2.1
            · rintro ⟨mll, nn, u⟩ hen
              exact outPtrs_get hd i mll nn (enS_row hen) (enS_col hen)
          · rintro ⟨mll, nn, u⟩ hen
            exact lt_of_lt_of_le
              (oIdx_lt_oB (cOf_lt i) (hOf_lt hd i) (enS_row hen) (enS_col hen)) hd.out_len
          · rintro ⟨m1, n1, u1⟩ ⟨m2, n2, u2⟩ h1 h2 heq
            obtain ⟨-, -, hmeq, hneq⟩ :=
              oIdx_inj hd (hOf_lt hd i) (hOf_lt hd i) (enS_row h1) (enS_col h1)
                (enS_row h2) (enS_col h2) heq
            have hm' : m1 = m2 := by
              refine Fin.ext ?_
              have h : (((m1, n1, u1) : Idx [BM, BN]).1).val =
                  (((m2, n2, u2) : Idx [BM, BN]).1).val := Nat.add_left_cancel hmeq
              simpa using h
            have hn' : n1 = n2 := by
              refine Fin.ext ?_
              have h : (((m1, n1, u1) : Idx [BM, BN]).2.1).val =
                  (((m2, n2, u2) : Idx [BM, BN]).2.1).val := Nat.add_left_cancel hneq
              simpa using h
            subst hm'; subst hn'
            exact Prod.ext rfl (Prod.ext rfl (by cases u1; cases u2; rfl))
          · refine Buf.elems_mono d.out ?_
            rintro j ⟨idx, hen, rfl⟩
            exact ⟨idx.1.val, idx.2.1.val, idx.1.isLt, idx.2.1.isLt,
              enS_row hen, enS_col hen, rfl⟩
          · intro m' hread hun hpres
            refine safe_ret ?_
            intro m n hm hn hM hN
            have hval : (⟨m, hm⟩ : Fin BM).val = m := rfl
            have hvaln : (⟨n, hn⟩ : Fin BN).val = n := rfl
            have hrowM : cOf d i * d.C + (mOf d i * BM + m) =
                rowA d i (⟨m, hm⟩ : Fin BM).val := by
              unfold rowA
              rw [hval]
              omega
            have hcolN : cOf d i * d.C + (nOf d i * BN + n) =
                colB d i (⟨n, hn⟩ : Fin BN).val := by
              unfold colB
              rw [hvaln]
              omega
            have hidx : (fun idx : Idx [BM, BN] => enS d i idx.1.val idx.2.1.val)
                (⟨m, hm⟩, ⟨n, hn⟩, ()) = true := by
              show enS d i m n = true
              unfold enS
              simp only [Bool.and_eq_true, decide_eq_true_eq]
              exact ⟨hM, hN⟩
            have hstr := hread (⟨m, hm⟩, ⟨n, hn⟩, ()) hidx
            have hvmOf : enM d i m = true →
                (vm.get (⟨m, hm⟩, ())).toInt =
                  val32 init d.seqIdx (cOf d i * d.C + (mOf d i * BM + m)) := by
              intro hMe
              have hr := hreadM (⟨m, hm⟩, ()) hMe
              rw [hrowM]
              unfold val32
              rw [hr]
            have hvnOf : enN d i n = true →
                (vn.get (⟨n, hn⟩, ())).toInt =
                  val32 init d.seqIdx (cOf d i * d.C + (nOf d i * BN + n)) := by
              intro hNe
              have hr := hreadN (⟨n, hn⟩, ()) hNe
              rw [hcolN]
              unfold val32
              rw [hr]
            have hotherM' : enM d i m ≠ true → (vm.get (⟨m, hm⟩, ())).toInt = -1 := by
              intro hMe
              have hMe' : enM d i m = false := by
                cases hb : enM d i m with
                | false => rfl
                | true => exact absurd hb hMe
              have h := hotherM (⟨m, hm⟩, ()) hMe'
                (tl.full [BM] (IntV.ofInt (-1 : ℤ) : I32)) rfl
              rw [h]
              show ((IntV.ofInt (-1 : ℤ) : I32)).toInt = -1
              exact IntV.toInt_ofInt (w := 32) (by norm_num) (by norm_num [IntV.SFits])
            have hotherN' : enN d i n ≠ true → (vn.get (⟨n, hn⟩, ())).toInt = -2 := by
              intro hNe
              have hNe' : enN d i n = false := by
                cases hb : enN d i n with
                | false => rfl
                | true => exact absurd hb hNe
              have h := hotherN (⟨n, hn⟩, ()) hNe'
                (tl.full [BN] (IntV.ofInt (-2 : ℤ) : I32)) rfl
              rw [h]
              show ((IntV.ofInt (-2 : ℤ) : I32)).toInt = -2
              exact IntV.toInt_ofInt (w := 32) (by norm_num) (by norm_num [IntV.SFits])
            have hnonnegM : enM d i m = true → 0 ≤ (vm.get (⟨m, hm⟩, ())).toInt := by
              intro hMe
              obtain ⟨w, hw, hw0⟩ := hpre.2.2.2.2 (rowA d i (⟨m, hm⟩ : Fin BM).val)
                (rowA_lt_of_chunk hd i (enM_row hMe))
              have hrm := hreadM (⟨m, hm⟩, ()) hMe
              rw [hrm] at hw
              injection hw with hw'
              rw [hw']
              exact hw0
            have hnonnegN : enN d i n = true → 0 ≤ (vn.get (⟨n, hn⟩, ())).toInt := by
              intro hNe
              obtain ⟨w, hw, hw0⟩ := hpre.2.2.2.2 (colB d i (⟨n, hn⟩ : Fin BN).val)
                (colB_lt_of_chunk hd i (enN_col hNe))
              have hrn := hreadN (⟨n, hn⟩, ()) hNe
              rw [hrn] at hw
              injection hw with hw'
              rw [hw']
              exact hw0
            refine ⟨_, hstr, ?_⟩
            split_ifs with hsame
            · have hs' := hsame
              unfold sameReq at hs'
              simp only [Bool.and_eq_true, decide_eq_true_eq] at hs'
              obtain ⟨⟨hMT, hNT⟩, hvalEq⟩ := hs'
              have hMe : enM d i m = true := by
                unfold enM chunkNat
                simp only [decide_eq_true_eq]
                exact lt_min hM (by omega)
              have hNe : enN d i n = true := by
                unfold enN chunkNat
                simp only [decide_eq_true_eq]
                exact lt_min hN (by omega)
              have heq : IntV.eq (vm.get (⟨m, hm⟩, ())) (vn.get (⟨n, hn⟩, ())) = true := by
                rw [intv_eq_true_iff, hvmOf hMe, hvnOf hNe]
                exact hvalEq
              have hvg : (tl.where ((vm.col BN).eq (vn.row BM)) st.2.2
                    (tl.full [BM, BN] zf32)).get (⟨m, hm⟩, ⟨n, hn⟩, ()) =
                  st.2.2.get2 ⟨m, hm⟩ ⟨n, hn⟩ := by
                have hc : ((vm.col BN).eq (vn.row BM)).get (⟨m, hm⟩, ⟨n, hn⟩, ()) = true := by
                  rw [eq_col_row_get]
                  exact heq
                exact where_get_true ((vm.col BN).eq (vn.row BM)) st.2.2 (tl.full [BM, BN] zf32)
                  ⟨m, hm⟩ ⟨n, hn⟩ hc
              rw [hvg]
              intro va vb hva hvb hmag
              have hva' : ∀ k, k < d.K →
                  init.read .bf16
                    (d.a.ptr (aIdx d (rowA d i (⟨m, hm⟩ : Fin BM).val) (hOf d i) k : ℤ)).addr =
                      some (va k) ∧ (va k).isFinite = true := by
                intro k hk
                rw [← hrowM]
                exact hva k hk
              have hvb' : ∀ k, k < d.K →
                  init.read .bf16
                    (d.b.ptr (bIdx d (colB d i (⟨n, hn⟩ : Fin BN).val) (hOf d i) k : ℤ)).addr =
                      some (vb k) ∧ (vb k).isFinite = true := by
                intro k hk
                rw [← hcolN]
                exact hvb k hk
              have hnum := hinv.2.2.2 ⟨m, hm⟩ ⟨n, hn⟩ hMe hNe va vb hva' hvb'
                (by simpa only [min_steps] using hmag)
              rw [min_steps] at hnum
              obtain ⟨hfin, hbnd, -⟩ := hnum
              exact ⟨hfin, le_trans hbnd (bound_le_tol (T := steps d.K) (steps_le hd.K_le)
                (dotMag_nonneg va vb d.K))⟩
            · have htoIntNe : (vm.get (⟨m, hm⟩, ())).toInt ≠ (vn.get (⟨n, hn⟩, ())).toInt := by
                intro hEq
                by_cases hMe : enM d i m = true
                · by_cases hNe : enN d i n = true
                  · have hMT : cOf d i * d.C + (mOf d i * BM + m) < d.T := by
                      rw [hrowM]
                      exact rowA_lt_of_chunk hd i (enM_row hMe)
                    have hNT : cOf d i * d.C + (nOf d i * BN + n) < d.T := by
                      rw [hcolN]
                      exact colB_lt_of_chunk hd i (enN_col hNe)
                    have hvalEq : val32 init d.seqIdx (cOf d i * d.C + (mOf d i * BM + m)) =
                        val32 init d.seqIdx (cOf d i * d.C + (nOf d i * BN + n)) := by
                      rw [← hvmOf hMe, ← hvnOf hNe]
                      exact hEq
                    have hsr : sameReq d init (cOf d i) (mOf d i * BM + m)
                        (nOf d i * BN + n) = true := by
                      unfold sameReq
                      simp only [Bool.and_eq_true, decide_eq_true_eq]
                      exact ⟨⟨hMT, hNT⟩, hvalEq⟩
                    exact hsame hsr
                  · have h1 := hnonnegM hMe
                    have h2 := hotherN' hNe
                    omega
                · have h1 := hotherM' hMe
                  by_cases hNe : enN d i n = true
                  · have h2 := hnonnegN hNe
                    omega
                  · have h2 := hotherN' hNe
                    omega
              have heqf : IntV.eq (vm.get (⟨m, hm⟩, ())) (vn.get (⟨n, hn⟩, ())) = false :=
                (intv_eq_false_iff _ _).mpr htoIntNe
              have hvg : (tl.where ((vm.col BN).eq (vn.row BM)) st.2.2
                    (tl.full [BM, BN] zf32)).get (⟨m, hm⟩, ⟨n, hn⟩, ()) = zf32 := by
                have hc : ((vm.col BN).eq (vn.row BM)).get (⟨m, hm⟩, ⟨n, hn⟩, ()) = false := by
                  rw [eq_col_row_get]
                  exact heqf
                exact where_get_false ((vm.col BN).eq (vn.row BM)) st.2.2 (tl.full [BM, BN] zf32)
                  ⟨m, hm⟩ ⟨n, hn⟩ hc
              rw [hvg]
              exact Or.inl zf32_eq.symm

end Claims.SsdBmm




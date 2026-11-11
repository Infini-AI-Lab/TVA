import Claim
import TVA.Proof
import Proof.Basic
import Proof.Arith
import Proof.Off
import Proof.Foot
import Proof.Lane
import Proof.Tol
import Proof.Body

/-!
# The `K`-loop of one program: its trips keep the invariant, the final accumulator is rounded to `float16` and stored, and the store's footprint is the program's tile of `c`.
-/

open TVA

namespace Claims.BlockScaledMxfp8

/-! ## The descriptor's offsets -/

theorem cDesc_offsets {d : Inst} {i : Fin (launch d).size} {am bn : I32}
    (ham : am.toInt = (((i : Nat) % gridM d) * 128 : Nat))
    (hbn : bn.toInt = (((i : Nat) / gridM d) * 256 : Nat)) :
    (([am, bn] : List I32).map IntV.toInt) = laneOffs d i := by
  simp only [laneOffs, List.map_cons, List.map_nil, ham, hbn]

/-- The coordinates of lane `l` of the store are the tile's row and column. -/
theorem cDesc_coord_offs {d : Inst} {i : Fin (launch d).size} {am bn : I32}
    (ham : am.toInt = (((i : Nat) % gridM d) * 128 : Nat))
    (hbn : bn.toInt = (((i : Nat) / gridM d) * 256 : Nat)) (l : Idx [128, 256]) :
    TensorDesc.coord (([am, bn] : List I32).map IntV.toInt) l = laneCoord d i l := by
  funext k
  simp only [laneCoord, laneOffs, List.map_cons, List.map_nil, ham, hbn]

/-- A lane of the store is in bounds exactly when `laneMask` says so. -/
theorem cDesc_inbAt_offs {d : Inst} {i : Fin (launch d).size} {am bn : I32}
    (ham : am.toInt = (((i : Nat) % gridM d) * 128 : Nat))
    (hbn : bn.toInt = (((i : Nat) / gridM d) * 256 : Nat)) (l : Idx [128, 256]) :
    (cDesc d).inbAt (TensorDesc.coord (([am, bn] : List I32).map IntV.toInt) l) =
      laneMask d i l := by
  rw [cDesc_coord_offs ham hbn, cDesc_laneMask]

/-- The element of `c` that a lane of the store addresses is `laneIdx`. -/
theorem cDesc_elemAt_offs {d : Inst} {i : Fin (launch d).size} {am bn : I32}
    (ham : am.toInt = (((i : Nat) % gridM d) * 128 : Nat))
    (hbn : bn.toInt = (((i : Nat) / gridM d) * 256 : Nat)) (l : Idx [128, 256]) :
    ((cDesc d).elemAt (TensorDesc.coord (([am, bn] : List I32).map IntV.toInt) l)).toNat =
      laneIdx d i l := by
  rw [cDesc_coord_offs ham hbn, elemAt_cDesc, Int.toNat_natCast]

/-- An in-bounds lane of the store addresses an element below the length of `c`. -/
theorem cDesc_hin {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {am bn : I32}
    (ham : am.toInt = (((i : Nat) % gridM d) * 128 : Nat))
    (hbn : bn.toInt = (((i : Nat) / gridM d) * 256 : Nat)) (l : Idx [128, 256])
    (hl : laneMask d i l = true) :
    ((cDesc d).elemAt (TensorDesc.coord (([am, bn] : List I32).map IntV.toInt) l)).toNat <
      d.c.len := by
  rw [cDesc_elemAt_offs ham hbn]
  obtain ⟨hrow, hcol⟩ := laneMask_bounds hl
  have h := off_mul_add_lt_mul hrow hcol
  have hlen : d.M * d.N ≤ d.c.len := hd.c_len
  unfold laneIdx
  omega

/-- The elements of `c` a store at the offsets `[am, bn]` touches. -/
def storeSet (d : Inst) (am bn : I32) : Set Nat :=
  Buf.img (fun l : Idx [128, 256] =>
      ((cDesc d).elemAt (TensorDesc.coord (([am, bn] : List I32).map IntV.toInt) l)).toNat)
    (fun l : Idx [128, 256] =>
      (cDesc d).inbAt (TensorDesc.coord (([am, bn] : List I32).map IntV.toInt) l))

/-- The bytes of `c` a store at the offsets `[am, bn]` touches. -/
def storeBytes (d : Inst) (am bn : I32) : Set Int := d.c.elems (storeSet d am bn)

/-- The elements the store writes are the tile's `storeSet`. -/
theorem storeSet_eq {d : Inst} {i : Fin (launch d).size} {am bn : I32}
    (ham : am.toInt = (((i : Nat) % gridM d) * 128 : Nat))
    (hbn : bn.toInt = (((i : Nat) / gridM d) * 256 : Nat)) :
    storeSet d am bn = cImg d i := by
  unfold storeSet cImg
  simp only [cDesc_elemAt_offs ham hbn, cDesc_inbAt_offs ham hbn]

/-- The bytes the store writes are the footprint `W` of the program. -/
theorem storeBytes_eq {d : Inst} {i : Fin (launch d).size} {am bn : I32}
    (ham : am.toInt = (((i : Nat) % gridM d) * 128 : Nat))
    (hbn : bn.toInt = (((i : Nat) / gridM d) * 256 : Nat)) :
    storeBytes d am bn = W d i := by
  unfold storeBytes W
  rw [storeSet_eq ham hbn]

/-! ## The initial accumulator -/

theorem zeros_get (l : Idx [128, 256]) :
    (tl.zeros [128, 256] .f32).get l = (FloatV.ofPat 0 : F32) := rfl

/-- Every element of the initial accumulator is finite. -/
theorem zeros_isFinite (l : Idx [128, 256]) :
    ((tl.zeros [128, 256] .f32).get l).isFinite = true := by
  rw [zeros_get l, FloatV.isFinite_ofPat (Nat.two_pow_pos _)]
  exact Format.isFinite_zero Format.binary32_regular

/-- Every element of the initial accumulator is `0`. -/
theorem zeros_toRat (l : Idx [128, 256]) :
    ((tl.zeros [128, 256] .f32).get l).toRat = 0 := by
  rw [zeros_get l, FloatV.toRat_ofPat (Nat.two_pow_pos _)]
  exact Format.toRat_of_isZero (by decide)

/-- The partial sum of `0` steps is `0`. -/
theorem partSum_zero (p : Nat → ℚ) : partSum p 0 = 0 := by simp [partSum]

/-- The tolerance after `0` steps is `0`. -/
theorem tol_zero (p : Nat → ℚ) : tol p 0 = 0 := rfl

/-! ## The epilogue -/

theorem epilogue_safe (P : Platform) {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    (i : Fin (launch d).size) {acc : Tensor [128, 256] F32} {am bn : I32} {m : Mem}
    (hm : m = init)
    (hacc : ∀ l : Idx [128, 256], laneMask d i l = true →
      (acc.get l).isFinite = true ∧
        |(acc.get l).toRat - partSum (prods d init (laneRow d i l) (laneCol d i l)) (d.K / 128)| ≤
          tol (prods d init (laneRow d i l) (laneCol d i l)) (d.K / 128))
    (ham : am.toInt = (((i : Nat) % gridM d) * 128 : Nat))
    (hbn : bn.toInt = (((i : Nat) / gridM d) * 256 : Nat)) :
    Safe ((launch d).env P) (R d i) (W d i) (fun _ m => Q d init i m) m
      (do
        let c ← tl.cvt acc
        tl.desc_store (cDesc d) [am, bn] c) := by
  rw [hm]
  refine safe_cvt (g := .binary16) (rnd := .rn)
    ⟨by simp [Format.Std], by simp [Format.Std], Or.inl rfl⟩ ?_
  intro r hr
  have hinj : ∀ l l' : Idx [128, 256],
      (cDesc d).inbAt (TensorDesc.coord (([am, bn] : List I32).map IntV.toInt) l) = true →
      (cDesc d).inbAt (TensorDesc.coord (([am, bn] : List I32).map IntV.toInt) l') = true →
      ((cDesc d).elemAt (TensorDesc.coord (([am, bn] : List I32).map IntV.toInt) l)).toNat =
        ((cDesc d).elemAt (TensorDesc.coord (([am, bn] : List I32).map IntV.toInt) l')).toNat →
      l = l' := by
    intro l l' hl hl' he
    rw [cDesc_inbAt_offs ham hbn] at hl
    rw [cDesc_inbAt_offs ham hbn] at hl'
    rw [cDesc_elemAt_offs ham hbn, cDesc_elemAt_offs ham hbn] at he
    exact laneIdx_inj hd hl hl' he
  have hin : ∀ l : Idx [128, 256],
      (cDesc d).inbAt (TensorDesc.coord (([am, bn] : List I32).map IntV.toInt) l) = true →
      ((cDesc d).elemAt (TensorDesc.coord (([am, bn] : List I32).map IntV.toInt) l)).toNat <
        d.c.len := by
    intro l hl
    rw [cDesc_inbAt_offs ham hbn] at hl
    exact cDesc_hin hd i ham hbn l hl
  have hW' : d.c.elems (Buf.img
      (fun l : Idx [128, 256] =>
        ((cDesc d).elemAt (TensorDesc.coord (([am, bn] : List I32).map IntV.toInt) l)).toNat)
      (fun l : Idx [128, 256] =>
        (cDesc d).inbAt (TensorDesc.coord (([am, bn] : List I32).map IntV.toInt) l))) ⊆ W d i := by
    show storeBytes d am bn ⊆ W d i
    rw [storeBytes_eq ham hbn]
  have hstore : Safe ((launch d).env P) (R d i) (W d i)
      (fun _ m => Q d init i m) init
      (tl.desc_store (cDesc d) [am, bn] r) := by
    simpa only [Prog.bind_eq, Prog.bind_ret] using
      (safe_desc_store (b := d.c) (bs := [128, 256]) (D := cDesc d) (offsets := [am, bn])
        (vals := r) (k := Prog.ret) (Q := fun _ m => Q d init i m) (E := (launch d).env P)
        (R := R d i) rfl
        (by rw [cDesc_offsets ham hbn]; exact cDesc_access hd i)
        hin hinj hpre.2.2.2.2.1 hW'
        (fun m' hread _hout _hnone => Safe.ret (by
          intro l hl
          have hreadl : m'.read .f16 (d.c.ptr ((laneIdx d i l : Nat) : Int)).addr =
              some (r.get l) := by
            have h := hread l (by
              show (cDesc d).inbAt (TensorDesc.coord (([am, bn] : List I32).map IntV.toInt) l) = true
              rw [cDesc_inbAt_offs ham hbn]; exact hl)
            simpa only [cDesc_elemAt_offs ham hbn] using h
          have hv : |(acc.get l).toRat - ref d init (laneRow d i l) (laneCol d i l)| ≤
              tol (prods d init (laneRow d i l) (laneCol d i l)) (d.K / 128) := by
            have h1 : ref d init (laneRow d i l) (laneCol d i l) =
                partSum (prods d init (laneRow d i l) (laneCol d i l)) (d.K / 128) := by
              simp only [ref]
              exact (partSum_ref hd
                (prods d init (laneRow d i l) (laneCol d i l))).symm
            rw [h1]
            exact (hacc l hl).2
          have hcvt : FloatV.NaNEq (FloatV.cvt (acc.get l)) (r.get l) := by
            simpa only [FloatV.cvtWith_rn] using hr l
          refine ⟨r.get l, by simpa only [laneIdx] using hreadl, acc.get l, (hacc l hl).1,
            acc_ref_bound hd (hacc l hl).2, hcvt, ?_⟩
          intro hroom
          have hroom' : |ref d init (laneRow d i l) (laneCol d i l)| +
              tol (prods d init (laneRow d i l) (laneCol d i l)) (d.K / 128) ≤
              Format.binary16.maxVal := by
            simpa only [tolOut] using hroom
          exact cvt_bound (prods d init (laneRow d i l) (laneCol d i l))
            (T := d.K / 128) (divK_le hd) (hacc l hl).1 hv hcvt hroom')))
  exact hstore

/-! ## The loop -/

theorem loop_safe (P : Platform) {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    (i : Fin (launch d).size) (hdot : Device.DotScaledE4m3 P)
    (hdotm : Device.DotScaledE4m3E2m1 P) (htot : Device.DotScaledTotal P cfgMx)
    (htotm : Device.DotScaledTotal P cfgMixed) {am bn sm sn : I32}
    (ham : am.toInt = (((i : Nat) % gridM d) * 128 : Nat))
    (hbn : bn.toInt = (((i : Nat) / gridM d) * 256 : Nat))
    (hsm : sm.toInt = (((i : Nat) % gridM d : Nat) : Int))
    (hsn : sn.toInt = ((2 * ((i : Nat) / gridM d) : Nat) : Int)) :
    Safe ((launch d).env P) (R d i) (W d i) (fun _ m => Q d init i m) init
      (do
        let (accumulator, _offs_k_a, _offs_k_b, _offs_scale_k) ←
          tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat ((d.K + (128 - 1)) / 128) : I32)
            (IntV.ofInt 1 : I32)
            (tl.zeros [128, 256] .f32, (IntV.ofInt 0 : I32), (IntV.ofInt 0 : I32),
              (IntV.ofInt 0 : I32))
            fun _k (accumulator, offs_k_a, offs_k_b, offs_scale_k) => do
              let a ← tl.desc_load (bs := [128, 128 / 1]) (aDesc d) [am, offs_k_a]
              let b ← tl.desc_load (bs := [256, 128 / epbB d.mixed]) (bDesc d) [bn, offs_k_b]
              let scale_a ← tl.desc_load (bs := [1, 1, 1, 2, 256]) (aScaleDesc d)
                [(IntV.ofInt 0 : I32), sm, offs_scale_k, (IntV.ofInt 0 : I32), (IntV.ofInt 0 : I32)]
              let scale_b ← tl.desc_load (bs := [1, 2, 1, 2, 256]) (bScaleDesc d)
                [(IntV.ofInt 0 : I32), sn, offs_scale_k, (IntV.ofInt 0 : I32), (IntV.ofInt 0 : I32)]
              let scale_a := ((scale_a.reshape [1, 1, 32, 4, 4]
                (by simp only [List.prod_cons, List.prod_nil]; ring)).permute [0, 3, 2, 1, 4]).reshape
                  [128, 128 / 32] rfl
              let scale_b := ((scale_b.reshape [2, 1, 32, 4, 4]
                (by simp only [List.prod_cons, List.prod_nil]; ring)).permute [0, 3, 2, 1, 4]).reshape
                  [256, 128 / 32] rfl
              let accumulator ←
                if 1 = 1 ∧ epbB d.mixed = 2 then
                  tl.dot_scaled cfgMixed 128 256 128 a (some scale_a) b.trans (some scale_b) accumulator
                else if 1 = 2 ∧ epbB d.mixed = 2 then
                  tl.dot_scaled cfgFp4 128 256 128 a (some scale_a) b.trans (some scale_b) accumulator
                else
                  tl.dot_scaled cfgMx 128 256 128 a (some scale_a) b.trans (some scale_b) accumulator
              let offs_k_a := offs_k_a + (IntV.ofNat (128 / 1) : I32)
              let offs_k_b := offs_k_b + (IntV.ofNat (128 / epbB d.mixed) : I32)
              let offs_scale_k := offs_scale_k + (IntV.ofNat 1 : I32)
              return (accumulator, offs_k_a, offs_k_b, offs_scale_k)
        let c ← tl.cvt accumulator
        tl.desc_store (cDesc d) [am, bn] c) := by
  have h := safe_forRange (σ := LoopState) (lo := (IntV.ofInt 0 : I32))
    (hi := (IntV.ofNat ((d.K + (128 - 1)) / 128) : I32)) (step := (IntV.ofInt 1 : I32))
    (init := ((tl.zeros [128, 256] .f32, (IntV.ofInt 0 : I32), (IntV.ofInt 0 : I32),
      (IntV.ofInt 0 : I32)) : LoopState))
    (body := fun (_k : I32) (st : LoopState) => do
        let accumulator := st.1
        let offs_k_a := st.2.1
        let offs_k_b := st.2.2.1
        let offs_scale_k := st.2.2.2
        let a ← tl.desc_load (bs := [128, 128 / 1]) (aDesc d) [am, offs_k_a]
        let b ← tl.desc_load (bs := [256, 128 / epbB d.mixed]) (bDesc d) [bn, offs_k_b]
        let scale_a ← tl.desc_load (bs := [1, 1, 1, 2, 256]) (aScaleDesc d)
          [(IntV.ofInt 0 : I32), sm, offs_scale_k, (IntV.ofInt 0 : I32), (IntV.ofInt 0 : I32)]
        let scale_b ← tl.desc_load (bs := [1, 2, 1, 2, 256]) (bScaleDesc d)
          [(IntV.ofInt 0 : I32), sn, offs_scale_k, (IntV.ofInt 0 : I32), (IntV.ofInt 0 : I32)]
        let scale_a := ((scale_a.reshape [1, 1, 32, 4, 4]
          (by simp only [List.prod_cons, List.prod_nil]; ring)).permute [0, 3, 2, 1, 4]).reshape
            [128, 128 / 32] rfl
        let scale_b := ((scale_b.reshape [2, 1, 32, 4, 4]
          (by simp only [List.prod_cons, List.prod_nil]; ring)).permute [0, 3, 2, 1, 4]).reshape
            [256, 128 / 32] rfl
        let accumulator ←
          if 1 = 1 ∧ epbB d.mixed = 2 then
            tl.dot_scaled cfgMixed 128 256 128 a (some scale_a) b.trans (some scale_b) accumulator
          else if 1 = 2 ∧ epbB d.mixed = 2 then
            tl.dot_scaled cfgFp4 128 256 128 a (some scale_a) b.trans (some scale_b) accumulator
          else
            tl.dot_scaled cfgMx 128 256 128 a (some scale_a) b.trans (some scale_b) accumulator
        let offs_k_a := offs_k_a + (IntV.ofNat (128 / 1) : I32)
        let offs_k_b := offs_k_b + (IntV.ofNat (128 / epbB d.mixed) : I32)
        let offs_scale_k := offs_scale_k + (IntV.ofNat 1 : I32)
        return (accumulator, offs_k_a, offs_k_b, offs_scale_k))
    (k := fun (st : LoopState) => do
        let c ← tl.cvt st.1
        tl.desc_store (cDesc d) [am, bn] c)
    (hrange := rangeOk hd) (I := fun t st m => Inv d init i t st m) (Wt := fun _ => ∅)
    (Wk := W d i) (E := (launch d).env P) (R := R d i) (Q := fun _ m => Q d init i m)
    (m := init)
    (by intro t u _ _; exact Set.disjoint_empty ∅) ?_ ?_ ?_ ?_
  · have hset : iterW (fun _ : Nat => (∅ : Set Int))
        (tl.tripCount (IntV.ofInt 0 : I32).toInt
          (IntV.ofNat ((d.K + (128 - 1)) / 128) : I32).toInt
          (IntV.ofInt 1 : I32).toInt) ∪ W d i = W d i := by
      rw [tripCount_eq hd, iterW_empty, Set.empty_union]
    rw [hset] at h
    exact h
  · rw [tripCount_eq hd, iterW_empty]
    exact Set.empty_disjoint _
  · refine ⟨rfl, ?_, ?_, ?_, ?_⟩
    · intro l hl
      refine ⟨zeros_isFinite l, ?_⟩
      rw [zeros_toRat l, partSum_zero, tol_zero]
      simp
    · rw [ofInt_toInt (by norm_num)]
      simp
    · rw [ofInt_toInt (by norm_num)]
      simp
    · rw [ofInt_toInt (by norm_num)]
      simp
  · intro t ht st m1 hI
    rcases st with ⟨acc, ka, kb, ks⟩
    rw [tripCount_eq hd] at ht
    exact body_safe P hd hpre i hdot hdotm htot htotm ht hI.1 hI.2.1 ham hbn hsm hsn
      hI.2.2.1 hI.2.2.2.1 hI.2.2.2.2
  · intro st m1 hI
    rcases st with ⟨acc, ka, kb, ks⟩
    rw [tripCount_eq hd] at hI
    exact epilogue_safe P hd hpre i hI.1 hI.2.1 ham hbn

end Claims.BlockScaledMxfp8

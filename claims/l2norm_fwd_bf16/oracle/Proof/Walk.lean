import Proof.Defs
import Proof.Lanes
import Proof.Leaf16
import Proof.Leaf32

/-!
# `l2norm_fwd_bf16`: the `Safe` walk of one program

`walk16`/`walk32` handle the two shapes of the output buffer (`y` a `bf16` tile, converted at the
end, or an `f32` tile stored as it is): one `tl.*` operation at a time, from the load of the tile
of `x` to the two stores of `y` and `rstd`. `safe_program` peels the index arithmetic
(`program_id`, `arange`, the `T64` conversion) and hands the rest to `walk16`/`walk32`.
-/

set_option maxHeartbeats 1000000

namespace Claims.L2NormFwd

open TVA
open TVA TVA.Format

/-- The row block of program `p` ends within the offset range the `int64` addresses can hold. -/
theorem p_bound {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) :
    p * d.BT + d.BT ≤ 2 ^ 51 := by
  have hg : grid d < 2 ^ 31 := hd.grid_lt
  have h31 : p + 1 ≤ 2 ^ 31 := by omega
  have hBT := bt_le hd
  have h1 : p * d.BT + d.BT = (p + 1) * d.BT := by ring
  rw [h1]
  calc (p + 1) * d.BT ≤ 2 ^ 31 * 2 ^ 20 := Nat.mul_le_mul h31 hBT
    _ = 2 ^ 51 := by norm_num

/-- The `Safe` walk when the output is `bf16`: the tile of `x` is loaded and converted to `f32`,
squared, summed, `eps` added, `sqrt`, reciprocal, multiplied back and rounded to `bf16`. -/
theorem walk16 {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {yb : Buf .bf16}
    (hy : d.y = .inl yb) {p : Nat} (hp : p * d.BT + d.BT ≤ 2 ^ 51)
    {o_t : Tensor [d.BT] I64} {T64 : I64}
    (hot : ∀ i : Idx [d.BT], IntV.toInt (o_t.get i) = ((p * d.BT + i.1.val : Nat) : Int))
    (hT64 : IntV.toInt T64 = (d.T : Int))
    {P : Platform} {R W : Set Int} {Q : Unit → Mem → Prop}
    (hR : d.x.elems (tileIdx d p) ⊆ R)
    (hW1 : yb.elems (tileIdx d p) ⊆ W)
    (hW2 : d.rstd.elems (tileRow d p) ⊆ W \ yb.elems (tileIdx d p))
    (hdisjY : Disjoint (yb.elems (tileIdx d p)) (d.rstd.elems (tileRow d p)))
    (hQ : ∀ m : Mem, Qn d init p m → Q () m)
    (hsqrt_total : Device.MathTotal P .sqrt .binary32) (hsqrt_acc : Device.SqrtF32 P)
    (hdiv_total : Device.DivTotal P .binary32) (hdiv_full : Device.DivFull P) :
    Safe ((launch d).env P) R W Q init
      (tl.load (pX d o_t) (some (mX d o_t T64)) (some (bc (FloatV.litBF16 (FloatV.ofPat 0)))) >>= fun b_x16 =>
       tl.cvt b_x16 >>= fun b_x =>
       tl.mul b_x b_x >>= fun b_xx =>
       tl.sum1c b_xx >>= fun s1 =>
       tl.addc s1 (bc (FloatV.hostF32 d.eps)) >>= fun s2 =>
       tl.sqrt s2 >>= fun sq =>
       tl.div (bc (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000))) sq >>= fun b_rstd =>
       tl.mul b_x (b_rstd.col d.BD) >>= fun b_y =>
       (tl.cvt b_y >>= fun b_y16 =>
         (tl.store (pY16 yb d o_t) b_y16 (some (mX d o_t T64)) >>= fun _ =>
          tl.store (pR d o_t) b_rstd (some (mT d o_t T64))))) := by
  have hD31 : d.D < 2 ^ 31 := by have := hd.d_le; omega
  have hBD31 : d.BD < 2 ^ 31 := by have := bd_le20 hd; omega
  have hrow := hrow_of_hot hp hot
  have hvY : yb.Valid init := by simpa only [yValid, hy] using hpre.2.1
  refine safe_load_buf (b := d.x) (g := gx d p) (en := enx d p)
    (addrs_x hD31 hd.d_le hBD31 hrow hot hT64)
    (fun i hi => gx_lt_xLen hd hi) hpre.1 ?_ ?_
  · rw [img_x hd]
    exact hR
  intro b_x16 hx16 hz'
  have hz : ∀ j : Idx [d.BT, d.BD], enx d p j = false →
      b_x16.get j = FloatV.litBF16 (FloatV.ofPat 0) := by
    intro j hj
    have h := hz' j hj (bc (FloatV.litBF16 (FloatV.ofPat 0))) rfl
    simpa only [bc, Tensor.get_full] using h
  refine safe_cvt (g := .binary32) (rnd := .rn)
    ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩ ?_
  intro b_x hcvt
  refine safe_mul (x := b_x) (y := b_x) ?_
  intro b_xx hmul
  refine safe_sum1c (N := d.BD) (bd_pos hd) ?_
  intro s1 htree
  refine safe_addc (x := s1) (y := bc (FloatV.hostF32 d.eps)) ?_
  intro s2 haddc
  refine safe_math (fn := .sqrt) (f := .binary32) (x := s2) (hex := fun _ => hsqrt_total _) ?_
  intro sq hsqrt
  refine safe_div (f := .binary32) (x := bc (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)))
    (y := sq) (hex := fun i => hdiv_total _ _) ?_
  intro b_rstd hdiv
  refine safe_mul (x := b_x) (y := b_rstd.col d.BD) ?_
  intro b_y hmul2
  refine safe_cvt (f := .binary32) (g := .bfloat16) (rnd := .rn) (x := b_y)
    ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl), Or.inl rfl⟩ ?_
  intro b_y16 hcvt16
  refine safe_store_buf (b := yb) (g := gx d p) (en := enx d p)
    (addrs_y16 hD31 hd.d_le hBD31 hrow hot hT64)
    (fun i hi => by simpa only [yLen, hy] using gx_lt_yLen hd hi)
    (fun i j hi hj h => gx_inj hd hi hj h) hvY ?_ ?_
  · rw [img_x hd]
    exact hW1
  intro m1 hyread hfr hpres
  refine safe_last ?_
  refine safe_store_buf (b := d.rstd) (g := gT d p) (en := enT d p)
    (addrs_rstd hot hT64)
    (fun i hi => gT_lt_rLen hd hi)
    (fun i j hi hj h => gT_inj hi hj h)
    (Buf.valid_of_agree hpre.2.2.1 (fun x hx => hpres x (hpre.2.2.1.2.2.2 x hx))) ?_ ?_
  · rw [img_T, img_x hd]
    exact hW2
  intro m2 hrread hfr2 hpres2
  have hyread2 : ∀ j, enx d p j = true →
      m2.read .bf16 (yb.ptr (gx d p j)).addr = some (b_y16.get j) := by
    intro j hj
    have hnot : ∀ x ∈ Mem.span .bf16 (yb.ptr (gx d p j)).addr,
        x ∉ d.rstd.elems (Buf.img (gT d p) (enT d p)) := by
      intro x hx hm
      rw [img_T] at hm
      have hgi : gx d p j ∈ tileIdx d p := by
        rw [← img_x hd]
        exact ⟨j, hj, rfl⟩
      exact (Set.disjoint_left.mp hdisjY) (Buf.span_sub_elems yb hgi hx) hm
    rw [Mem.read_congr (m₁ := m2) (m₂ := m1) (d := .bf16)
      (a := (yb.ptr (gx d p j)).addr) (fun x hx => hfr2 x (hnot x hx))]
    exact hyread j hj
  exact safe_ret (hQ m2 (qn16 hd hpre hy p m2 P hsqrt_acc hdiv_full b_x16 b_x b_xx s1 s2 sq b_rstd
    b_y b_y16 hx16 hz hcvt hmul (fun r => (htree r).imp fun t h => ⟨h.1, h.2.2⟩) haddc hsqrt hdiv hmul2 hcvt16 hyread2 hrread))

/-- The `Safe` walk when the output is `f32`: same as `walk16`, without the final conversion. -/
theorem walk32 {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {yc : Buf .f32}
    (hy : d.y = .inr yc) {p : Nat} (hp : p * d.BT + d.BT ≤ 2 ^ 51)
    {o_t : Tensor [d.BT] I64} {T64 : I64}
    (hot : ∀ i : Idx [d.BT], IntV.toInt (o_t.get i) = ((p * d.BT + i.1.val : Nat) : Int))
    (hT64 : IntV.toInt T64 = (d.T : Int))
    {P : Platform} {R W : Set Int} {Q : Unit → Mem → Prop}
    (hR : d.x.elems (tileIdx d p) ⊆ R)
    (hW1 : yc.elems (tileIdx d p) ⊆ W)
    (hW2 : d.rstd.elems (tileRow d p) ⊆ W \ yc.elems (tileIdx d p))
    (hdisjY : Disjoint (yc.elems (tileIdx d p)) (d.rstd.elems (tileRow d p)))
    (hQ : ∀ m : Mem, Qn d init p m → Q () m)
    (hsqrt_total : Device.MathTotal P .sqrt .binary32) (hsqrt_acc : Device.SqrtF32 P)
    (hdiv_total : Device.DivTotal P .binary32) (hdiv_full : Device.DivFull P) :
    Safe ((launch d).env P) R W Q init
      (tl.load (pX d o_t) (some (mX d o_t T64)) (some (bc (FloatV.litBF16 (FloatV.ofPat 0)))) >>= fun b_x16 =>
       tl.cvt b_x16 >>= fun b_x =>
       tl.mul b_x b_x >>= fun b_xx =>
       tl.sum1c b_xx >>= fun s1 =>
       tl.addc s1 (bc (FloatV.hostF32 d.eps)) >>= fun s2 =>
       tl.sqrt s2 >>= fun sq =>
       tl.div (bc (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000))) sq >>= fun b_rstd =>
       tl.mul b_x (b_rstd.col d.BD) >>= fun b_y =>
       (tl.store (pY32 yc d o_t) b_y (some (mX d o_t T64)) >>= fun _ =>
        tl.store (pR d o_t) b_rstd (some (mT d o_t T64)))) := by
  have hD31 : d.D < 2 ^ 31 := by have := hd.d_le; omega
  have hBD31 : d.BD < 2 ^ 31 := by have := bd_le20 hd; omega
  have hrow := hrow_of_hot hp hot
  have hvY : yc.Valid init := by simpa only [yValid, hy] using hpre.2.1
  refine safe_load_buf (b := d.x) (g := gx d p) (en := enx d p)
    (addrs_x hD31 hd.d_le hBD31 hrow hot hT64)
    (fun i hi => gx_lt_xLen hd hi) hpre.1 ?_ ?_
  · rw [img_x hd]
    exact hR
  intro b_x16 hx16 hz'
  have hz : ∀ j : Idx [d.BT, d.BD], enx d p j = false →
      b_x16.get j = FloatV.litBF16 (FloatV.ofPat 0) := by
    intro j hj
    have h := hz' j hj (bc (FloatV.litBF16 (FloatV.ofPat 0))) rfl
    simpa only [bc, Tensor.get_full] using h
  refine safe_cvt (g := .binary32) (rnd := .rn)
    ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩ ?_
  intro b_x hcvt
  refine safe_mul (x := b_x) (y := b_x) ?_
  intro b_xx hmul
  refine safe_sum1c (N := d.BD) (bd_pos hd) ?_
  intro s1 htree
  refine safe_addc (x := s1) (y := bc (FloatV.hostF32 d.eps)) ?_
  intro s2 haddc
  refine safe_math (fn := .sqrt) (f := .binary32) (x := s2) (hex := fun _ => hsqrt_total _) ?_
  intro sq hsqrt
  refine safe_div (f := .binary32) (x := bc (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)))
    (y := sq) (hex := fun i => hdiv_total _ _) ?_
  intro b_rstd hdiv
  refine safe_mul (x := b_x) (y := b_rstd.col d.BD) ?_
  intro b_y hmul2
  refine safe_store_buf (b := yc) (g := gx d p) (en := enx d p)
    (addrs_y32 hD31 hd.d_le hBD31 hrow hot hT64)
    (fun i hi => by simpa only [yLen, hy] using gx_lt_yLen hd hi)
    (fun i j hi hj h => gx_inj hd hi hj h) hvY ?_ ?_
  · rw [img_x hd]
    exact hW1
  intro m1 hyread hfr hpres
  refine safe_last ?_
  refine safe_store_buf (b := d.rstd) (g := gT d p) (en := enT d p)
    (addrs_rstd hot hT64)
    (fun i hi => gT_lt_rLen hd hi)
    (fun i j hi hj h => gT_inj hi hj h)
    (Buf.valid_of_agree hpre.2.2.1 (fun x hx => hpres x (hpre.2.2.1.2.2.2 x hx))) ?_ ?_
  · rw [img_T, img_x hd]
    exact hW2
  intro m2 hrread hfr2 hpres2
  have hyread2 : ∀ j, enx d p j = true →
      m2.read .f32 (yc.ptr (gx d p j)).addr = some (b_y.get j) := by
    intro j hj
    have hnot : ∀ x ∈ Mem.span .f32 (yc.ptr (gx d p j)).addr,
        x ∉ d.rstd.elems (Buf.img (gT d p) (enT d p)) := by
      intro x hx hm
      rw [img_T] at hm
      have hgi : gx d p j ∈ tileIdx d p := by
        rw [← img_x hd]
        exact ⟨j, hj, rfl⟩
      exact (Set.disjoint_left.mp hdisjY) (Buf.span_sub_elems yc hgi hx) hm
    rw [Mem.read_congr (m₁ := m2) (m₂ := m1) (d := .f32)
      (a := (yc.ptr (gx d p j)).addr) (fun x hx => hfr2 x (hnot x hx))]
    exact hyread j hj
  exact safe_ret (hQ m2 (qn32 hd hpre hy p m2 P hsqrt_acc hdiv_full b_x16 b_x b_xx s1 s2 sq b_rstd
    b_y hx16 hz hcvt hmul (fun r => (htree r).imp fun t h => ⟨h.1, h.2.2⟩) haddc hsqrt hdiv hmul2 hyread2 hrread))

/-- `safe_program`: peel the index arithmetic of the kernel (`program_id`, `arange`, the `T64`
conversion) and hand the remaining tile computation to `walk16`/`walk32`. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (P : Platform)
    (hsqrt_total : Device.MathTotal P .sqrt .binary32) (hsqrt_acc : Device.SqrtF32 P)
    (hdiv_total : Device.DivTotal P .binary32) (hdiv_full : Device.DivFull P)
    (i : Fin (launch d).size) :
    Safe ((launch d).env P) (R d init i) (W d init i) (fun _ m => Q d init i m) init
      ((model d).body ((launch d).ctx i)) := by
  have hlt : i.val < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  have hp : i.val * d.BT + d.BT ≤ 2 ^ 51 := p_bound hd hlt
  have hi_lt : i.val < 2 ^ 31 := by
    have hg : grid d < 2 ^ 31 := hd.grid_lt
    omega
  have hBT31 : d.BT < 2 ^ 31 := by have := bt_le hd; omega
  cases hy : d.y with
  | inl yb =>
    unfold model l2norm_fwd_kernel
    dsimp only
    by_cases hT : d.T < 2 ^ 31
    · rw [ite_eq_left_of_eq_true _ _ (eq_true hT)]
      simp only [yPtr, hy]
      have hxi : tileIdx d i.val ⊆ rowSet d := tileIdx_subset_rowSet hd
      have hri : tileRow d i.val ⊆ rSet d := tileRow_subset_rSet
      have hdisjY : Disjoint (yb.elems (tileIdx d i.val)) (d.rstd.elems (tileRow d i.val)) := by
        have h := hd.r_y.symm.mono (yElems_mono hxi) (Buf.elems_mono _ hri)
        simpa only [yElems, hy] using h
      have hW1 : yb.elems (tileIdx d i.val) ⊆ W d init i := by
        simp only [W, yElems, hy]
        exact Set.subset_union_left
      have hW2 : d.rstd.elems (tileRow d i.val) ⊆ W d init i \ yb.elems (tileIdx d i.val) := by
        intro x hx
        refine ⟨?_, ?_⟩
        · simp only [W, yElems, hy]
          exact Set.mem_union_right _ hx
        · intro hxy
          exact Set.disjoint_left.mp hdisjY hxy hx
      refine safe_convert (d := .i32) (d' := .i64)
        (x := Tensor.scalar (tl.program_id ((launch d).ctx i) 0)) (hex := fun _ => ⟨_, rfl⟩) ?_
      intro r hr
      refine safe_convert (d := .i32) (d' := .i64) (x := tl.arange 0 d.BT)
        (hex := fun _ => ⟨_, rfl⟩) ?_
      intro o_a hoa
      refine safe_convert (d := .i32) (d' := .i64) (x := Tensor.scalar (IntV.ofNat d.T))
        (hex := fun _ => ⟨_, rfl⟩) ?_
      intro r1 hr1
      simp only [Prog.pure_eq]
      have hi32 : IntV.toInt (tl.program_id ((launch d).ctx i) 0) = (i.val : Int) := by
        simp only [tl.program_id, launch_pid]
        exact IntV.I32.toInt_ofNat hi_lt
      have hT32 : IntV.toInt (IntV.ofNat d.T : I32) = (d.T : Int) := IntV.I32.toInt_ofNat hT
      have hi_t : IntV.toInt r.item = (i.val : Int) := by
        have h1 : r.item = IntV.ofInt (IntV.toInt (tl.program_id ((launch d).ctx i) 0)) := hr ()
        rw [h1, hi32]
        exact IntV.toInt_ofInt (by norm_num) ⟨by omega, by omega⟩
      have hoa' : ∀ j : Idx [d.BT], IntV.toInt (o_a.get j) = (j.1.val : Int) := by
        intro j
        have h1 : o_a.get j = IntV.ofInt (IntV.toInt ((tl.arange 0 d.BT).get j)) := hoa j
        have h2 : IntV.toInt ((tl.arange 0 d.BT).get j) = (j.1.val : Int) :=
          arange_toInt d.BT hBT31 j.1
        rw [h1, h2]
        exact IntV.toInt_ofInt (by norm_num) ⟨by omega, by omega⟩
      have hT64 : IntV.toInt r1.item = (d.T : Int) := by
        have h1 : r1.item = IntV.ofInt (IntV.toInt ((Tensor.scalar (IntV.ofNat d.T)).get ())) :=
          hr1 ()
        have h2 : IntV.toInt ((Tensor.scalar (IntV.ofNat d.T)).get ()) = (d.T : Int) := hT32
        rw [h1, h2]
        exact IntV.toInt_ofInt (by norm_num) ⟨by omega, by omega⟩
      exact walk16 hd hpre hy hp (get_o_t (p := i.val) hi_t hoa' hp) hT64
        (fun _ hx => hx) hW1 hW2 hdisjY (fun m h => h) hsqrt_total hsqrt_acc hdiv_total hdiv_full
    · rw [ite_eq_right_of_eq_false _ _ (eq_false hT)]
      simp only [yPtr, hy]
      have hxi : tileIdx d i.val ⊆ rowSet d := tileIdx_subset_rowSet hd
      have hri : tileRow d i.val ⊆ rSet d := tileRow_subset_rSet
      have hdisjY : Disjoint (yb.elems (tileIdx d i.val)) (d.rstd.elems (tileRow d i.val)) := by
        have h := hd.r_y.symm.mono (yElems_mono hxi) (Buf.elems_mono _ hri)
        simpa only [yElems, hy] using h
      have hW1 : yb.elems (tileIdx d i.val) ⊆ W d init i := by
        simp only [W, yElems, hy]
        exact Set.subset_union_left
      have hW2 : d.rstd.elems (tileRow d i.val) ⊆ W d init i \ yb.elems (tileIdx d i.val) := by
        intro x hx
        refine ⟨?_, ?_⟩
        · simp only [W, yElems, hy]
          exact Set.mem_union_right _ hx
        · intro hxy
          exact Set.disjoint_left.mp hdisjY hxy hx
      refine safe_convert (d := .i32) (d' := .i64)
        (x := Tensor.scalar (tl.program_id ((launch d).ctx i) 0)) (hex := fun _ => ⟨_, rfl⟩) ?_
      intro r hr
      refine safe_convert (d := .i32) (d' := .i64) (x := tl.arange 0 d.BT)
        (hex := fun _ => ⟨_, rfl⟩) ?_
      intro o_a hoa
      simp only [Prog.pure_eq]
      have hi32 : IntV.toInt (tl.program_id ((launch d).ctx i) 0) = (i.val : Int) := by
        simp only [tl.program_id, launch_pid]
        exact IntV.I32.toInt_ofNat hi_lt
      have hi_t : IntV.toInt r.item = (i.val : Int) := by
        have h1 : r.item = IntV.ofInt (IntV.toInt (tl.program_id ((launch d).ctx i) 0)) := hr ()
        rw [h1, hi32]
        exact IntV.toInt_ofInt (by norm_num) ⟨by omega, by omega⟩
      have hoa' : ∀ j : Idx [d.BT], IntV.toInt (o_a.get j) = (j.1.val : Int) := by
        intro j
        have h1 : o_a.get j = IntV.ofInt (IntV.toInt ((tl.arange 0 d.BT).get j)) := hoa j
        have h2 : IntV.toInt ((tl.arange 0 d.BT).get j) = (j.1.val : Int) :=
          arange_toInt d.BT hBT31 j.1
        rw [h1, h2]
        exact IntV.toInt_ofInt (by norm_num) ⟨by omega, by omega⟩
      have hT64 : IntV.toInt (IntV.ofNat d.T : I64) = (d.T : Int) := by
        refine IntV.toInt_ofNat ?_
        have := t_lt hd
        omega
      exact walk16 hd hpre hy hp (get_o_t (p := i.val) hi_t hoa' hp) hT64
        (fun _ hx => hx) hW1 hW2 hdisjY (fun m h => h) hsqrt_total hsqrt_acc hdiv_total hdiv_full
  | inr yc =>
    unfold model l2norm_fwd_kernel
    dsimp only
    by_cases hT : d.T < 2 ^ 31
    · rw [ite_eq_left_of_eq_true _ _ (eq_true hT)]
      simp only [yPtr, hy]
      have hxi : tileIdx d i.val ⊆ rowSet d := tileIdx_subset_rowSet hd
      have hri : tileRow d i.val ⊆ rSet d := tileRow_subset_rSet
      have hdisjY : Disjoint (yc.elems (tileIdx d i.val)) (d.rstd.elems (tileRow d i.val)) := by
        have h := hd.r_y.symm.mono (yElems_mono hxi) (Buf.elems_mono _ hri)
        simpa only [yElems, hy] using h
      have hW1 : yc.elems (tileIdx d i.val) ⊆ W d init i := by
        simp only [W, yElems, hy]
        exact Set.subset_union_left
      have hW2 : d.rstd.elems (tileRow d i.val) ⊆ W d init i \ yc.elems (tileIdx d i.val) := by
        intro x hx
        refine ⟨?_, ?_⟩
        · simp only [W, yElems, hy]
          exact Set.mem_union_right _ hx
        · intro hxy
          exact Set.disjoint_left.mp hdisjY hxy hx
      refine safe_convert (d := .i32) (d' := .i64)
        (x := Tensor.scalar (tl.program_id ((launch d).ctx i) 0)) (hex := fun _ => ⟨_, rfl⟩) ?_
      intro r hr
      refine safe_convert (d := .i32) (d' := .i64) (x := tl.arange 0 d.BT)
        (hex := fun _ => ⟨_, rfl⟩) ?_
      intro o_a hoa
      refine safe_convert (d := .i32) (d' := .i64) (x := Tensor.scalar (IntV.ofNat d.T))
        (hex := fun _ => ⟨_, rfl⟩) ?_
      intro r1 hr1
      simp only [Prog.pure_eq]
      have hi32 : IntV.toInt (tl.program_id ((launch d).ctx i) 0) = (i.val : Int) := by
        simp only [tl.program_id, launch_pid]
        exact IntV.I32.toInt_ofNat hi_lt
      have hT32 : IntV.toInt (IntV.ofNat d.T : I32) = (d.T : Int) := IntV.I32.toInt_ofNat hT
      have hi_t : IntV.toInt r.item = (i.val : Int) := by
        have h1 : r.item = IntV.ofInt (IntV.toInt (tl.program_id ((launch d).ctx i) 0)) := hr ()
        rw [h1, hi32]
        exact IntV.toInt_ofInt (by norm_num) ⟨by omega, by omega⟩
      have hoa' : ∀ j : Idx [d.BT], IntV.toInt (o_a.get j) = (j.1.val : Int) := by
        intro j
        have h1 : o_a.get j = IntV.ofInt (IntV.toInt ((tl.arange 0 d.BT).get j)) := hoa j
        have h2 : IntV.toInt ((tl.arange 0 d.BT).get j) = (j.1.val : Int) :=
          arange_toInt d.BT hBT31 j.1
        rw [h1, h2]
        exact IntV.toInt_ofInt (by norm_num) ⟨by omega, by omega⟩
      have hT64 : IntV.toInt r1.item = (d.T : Int) := by
        have h1 : r1.item = IntV.ofInt (IntV.toInt ((Tensor.scalar (IntV.ofNat d.T)).get ())) :=
          hr1 ()
        have h2 : IntV.toInt ((Tensor.scalar (IntV.ofNat d.T)).get ()) = (d.T : Int) := hT32
        rw [h1, h2]
        exact IntV.toInt_ofInt (by norm_num) ⟨by omega, by omega⟩
      exact walk32 hd hpre hy hp (get_o_t (p := i.val) hi_t hoa' hp) hT64
        (fun _ hx => hx) hW1 hW2 hdisjY (fun m h => h) hsqrt_total hsqrt_acc hdiv_total hdiv_full
    · rw [ite_eq_right_of_eq_false _ _ (eq_false hT)]
      simp only [yPtr, hy]
      have hxi : tileIdx d i.val ⊆ rowSet d := tileIdx_subset_rowSet hd
      have hri : tileRow d i.val ⊆ rSet d := tileRow_subset_rSet
      have hdisjY : Disjoint (yc.elems (tileIdx d i.val)) (d.rstd.elems (tileRow d i.val)) := by
        have h := hd.r_y.symm.mono (yElems_mono hxi) (Buf.elems_mono _ hri)
        simpa only [yElems, hy] using h
      have hW1 : yc.elems (tileIdx d i.val) ⊆ W d init i := by
        simp only [W, yElems, hy]
        exact Set.subset_union_left
      have hW2 : d.rstd.elems (tileRow d i.val) ⊆ W d init i \ yc.elems (tileIdx d i.val) := by
        intro x hx
        refine ⟨?_, ?_⟩
        · simp only [W, yElems, hy]
          exact Set.mem_union_right _ hx
        · intro hxy
          exact Set.disjoint_left.mp hdisjY hxy hx
      refine safe_convert (d := .i32) (d' := .i64)
        (x := Tensor.scalar (tl.program_id ((launch d).ctx i) 0)) (hex := fun _ => ⟨_, rfl⟩) ?_
      intro r hr
      refine safe_convert (d := .i32) (d' := .i64) (x := tl.arange 0 d.BT)
        (hex := fun _ => ⟨_, rfl⟩) ?_
      intro o_a hoa
      simp only [Prog.pure_eq]
      have hi32 : IntV.toInt (tl.program_id ((launch d).ctx i) 0) = (i.val : Int) := by
        simp only [tl.program_id, launch_pid]
        exact IntV.I32.toInt_ofNat hi_lt
      have hi_t : IntV.toInt r.item = (i.val : Int) := by
        have h1 : r.item = IntV.ofInt (IntV.toInt (tl.program_id ((launch d).ctx i) 0)) := hr ()
        rw [h1, hi32]
        exact IntV.toInt_ofInt (by norm_num) ⟨by omega, by omega⟩
      have hoa' : ∀ j : Idx [d.BT], IntV.toInt (o_a.get j) = (j.1.val : Int) := by
        intro j
        have h1 : o_a.get j = IntV.ofInt (IntV.toInt ((tl.arange 0 d.BT).get j)) := hoa j
        have h2 : IntV.toInt ((tl.arange 0 d.BT).get j) = (j.1.val : Int) :=
          arange_toInt d.BT hBT31 j.1
        rw [h1, h2]
        exact IntV.toInt_ofInt (by norm_num) ⟨by omega, by omega⟩
      have hT64 : IntV.toInt (IntV.ofNat d.T : I64) = (d.T : Int) := by
        refine IntV.toInt_ofNat ?_
        have := t_lt hd
        omega
      exact walk32 hd hpre hy hp (get_o_t (p := i.val) hi_t hoa' hp) hT64
        (fun _ hx => hx) hW1 hW2 hdisjY (fun m h => h) hsqrt_total hsqrt_acc hdiv_total hdiv_full

end Claims.L2NormFwd

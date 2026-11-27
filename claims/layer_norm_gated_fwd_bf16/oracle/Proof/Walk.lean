import Proof.Defs
import Proof.Lanes
import Proof.LeafQ

/-!
# `layer_norm_gated_fwd_bf16`: the `Safe` walk of one program

`walk`: one `tl.*` operation at a time, from the load of the tile of `x` to the store of `y`: the
row statistics and the `rstd` store, the loads of `weight` and `g` (from the memory after the
`rstd` store, which has not touched them), the products, `tl.sigmoid`'s four steps and the store
of `y`. `safe_program` peels the index arithmetic (`program_id`, `arange`, the `T64` conversion)
and hands the rest to `walk`.
-/

set_option maxHeartbeats 4000000

namespace Claims.LayerNormGatedFwd

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

/-- A read of a buffer element the `rstd` store did not touch gives the value it had before. -/
theorem read_keep {dt : DType} {b : Buf dt} {S : Set Int} {m m' : Mem} {k : Nat} {v : dt.Val}
    (hfr : ∀ x, x ∉ S → m' x = m x) (hk : Disjoint (b.elems {k}) S)
    (h : m'.read dt (b.ptr (k : Int)).addr = some v) : m.read dt (b.ptr (k : Int)).addr = some v := by
  rw [Mem.read_congr (m₁ := m) (m₂ := m') (d := dt) (a := (b.ptr (k : Int)).addr)
    (fun x hx => (hfr x (fun hS => Set.disjoint_left.mp hk
      (Buf.span_sub_elems b (Set.mem_singleton k) hx) hS)).symm)]
  exact h

/-- **The walk of one program.** -/
theorem walk {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {p : Nat}
    (hp : p * d.BT + d.BT ≤ 2 ^ 51) {o_t : Tensor [d.BT] I64} {T64 : I64}
    (hot : ∀ i : Idx [d.BT], IntV.toInt (o_t.get i) = ((p * d.BT + i.1.val : Nat) : Int))
    (hT64 : IntV.toInt T64 = (d.T : Int))
    {P : Platform} {R W : Set Int} {Q : Unit → Mem → Prop}
    (hRx : d.x.elems (tileIdx d p) ⊆ R) (hRg : d.g.elems (tileIdx d p) ⊆ R)
    (hRw : d.w.elems (vecSet d) ⊆ R)
    (hW1 : d.rstd.elems (tileRow d p) ⊆ W)
    (hW2 : d.y.elems (tileIdx d p) ⊆ W \ d.rstd.elems (tileRow d p))
    (hQ : ∀ m : Mem, Qn d init p m → Q () m)
    (hsqrt_total : Device.MathTotal P .sqrt .binary32)
    (hexp_total : Device.MathTotal P .exp .binary32) (hsqrt_acc : Device.SqrtF32 P)
    (hdiv_total : Device.DivTotal P .binary32) (hdiv_full : Device.DivFull P)
    (hexp : Device.ExpF32 P) (hovf : Device.ExpF32Overflow P) (hinf : Device.DivF32InfDivisor P) :
    Safe ((launch d).env P) R W Q init
      (tl.load (pT d.x d o_t) (some (mX d o_t T64)) (some (bc (FloatV.litBF16 (FloatV.ofPat 0))))
        >>= fun b_x16 =>
       tl.cvt b_x16 >>= fun (b_x : Tensor [d.BT, d.BD] F32) =>
       tl.mul (tl.where (((tl.arange 0 d.BD).ltS (IntV.ofNat d.D : I32)).row d.BT) b_x
          (bc (FloatV.litF32 (FloatV.ofPat 0))))
        (tl.where (((tl.arange 0 d.BD).ltS (IntV.ofNat d.D : I32)).row d.BT) b_x
          (bc (FloatV.litF32 (FloatV.ofPat 0)))) >>= fun b_xx =>
       tl.sum1c b_xx >>= fun s1 =>
       tl.div s1 (bc (FloatV.litF32 (FloatV.ofRat (d.D : ℚ)))) >>= fun b_var =>
       tl.addc b_var (bc (FloatV.hostF32 d.eps)) >>= fun s2 =>
       tl.sqrt s2 >>= fun sq =>
       tl.div (bc (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000))) sq >>= fun b_rstd =>
       tl.store (pR d o_t) b_rstd (some (mT d o_t T64)) >>= fun _ =>
       tl.load (d.w.ptr 0 + tl.arange 0 d.BD) (some ((tl.arange 0 d.BD).ltS (IntV.ofNat d.D : I32)))
        >>= fun b_w16 =>
       tl.cvt b_w16 >>= fun (b_w : Tensor [d.BD] F32) =>
       tl.mul b_x (b_rstd.col d.BD) >>= fun b_x_hat =>
       tl.mul b_x_hat (b_w.row d.BT) >>= fun b_y0 =>
       tl.load (pT d.g d o_t) (some (mX d o_t T64)) (some (bc (FloatV.litBF16 (FloatV.ofPat 0))))
        >>= fun b_g16 =>
       tl.cvt b_g16 >>= fun (b_g : Tensor [d.BT, d.BD] F32) =>
       if d.swish = true then
          (tl.mul b_y0 b_g >>= fun b_yg => tl.sigmoid b_g >>= fun b_s => tl.mul b_yg b_s >>= fun b_y =>
           tl.cvt b_y >>= fun (b_y16 : Tensor [d.BT, d.BD] BF16) =>
           tl.store (pT d.y d o_t) b_y16 (some (mX d o_t T64)))
        else
          (tl.sigmoid b_g >>= fun b_s => tl.mul b_y0 b_s >>= fun b_y =>
           tl.cvt b_y >>= fun (b_y16 : Tensor [d.BT, d.BD] BF16) =>
           tl.store (pT d.y d o_t) b_y16 (some (mX d o_t T64)))) := by
  have hD31 : d.D < 2 ^ 31 := by have := hd.d_le; omega
  have hBD31 : d.BD < 2 ^ 31 := by have := hd.bd_le; omega
  have hrow := hrow_of_hot hp hot
  -- disjointness of the `rstd` rows from the inputs, and of `y` from `rstd`
  have hRin : d.rstd.elems (tileRow d p) ⊆ outBytes d :=
    fun x hx => Or.inr (Buf.elems_mono _ tileRow_subset_rowSet hx)
  have hdisjW : ∀ k ∈ vecSet d, Disjoint (d.w.elems {k}) (d.rstd.elems (Buf.img (gT d p) (enT d p))) := by
    intro k hk
    rw [img_T]
    refine (hd.out_in.mono hRin (fun x hx => Or.inr (Buf.elems_mono _ ?_ hx))).symm
    exact Set.singleton_subset_iff.mpr hk
  have hdisjG : ∀ k ∈ matSet d, Disjoint (d.g.elems {k}) (d.rstd.elems (Buf.img (gT d p) (enT d p))) := by
    intro k hk
    rw [img_T]
    refine (hd.out_in.mono hRin (fun x hx => Or.inl (Or.inr (Buf.elems_mono _ ?_ hx)))).symm
    exact Set.singleton_subset_iff.mpr hk
  -- the load of `x`
  refine safe_load_buf (b := d.x) (g := gx d p) (en := enx d p)
    (addrs_t hD31 hd.d_le hBD31 hrow hot hT64)
    (fun i hi => gx_lt_len hd hd.x_len hi) hpre.1 (by rw [img_x hd]; exact hRx) ?_
  intro b_x16 hx16 hz'
  have hz : ∀ j : Idx [d.BT, d.BD], enx d p j = false →
      b_x16.get j = FloatV.litBF16 (FloatV.ofPat 0) := by
    intro j hj
    have h := hz' j hj (bc (FloatV.litBF16 (FloatV.ofPat 0))) rfl
    simpa only [bc, Tensor.get_full] using h
  refine safe_cvt (g := .binary32) (rnd := .rn)
    ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩ ?_
  intro b_x hcvt
  refine safe_mul ?_
  intro b_xx hmul
  refine safe_sum1c (N := d.BD) (bd_pos hd) ?_
  intro s1 htree
  refine safe_div (f := .binary32) (hex := fun i => hdiv_total _ _) ?_
  intro b_var hdv
  refine safe_addc ?_
  intro s2 haddc
  refine safe_math (fn := .sqrt) (f := .binary32) (hex := fun _ => hsqrt_total _) ?_
  intro sq hsqrt
  refine safe_div (f := .binary32) (hex := fun i => hdiv_total _ _) ?_
  intro b_rstd hdiv
  -- the store of `rstd`
  refine safe_store_buf (b := d.rstd) (g := gT d p) (en := enT d p) (addrs_rstd hot hT64)
    (fun i hi => gT_lt_rLen hd hi) (fun i j hi hj h => gT_inj hi hj h) hpre.2.2.2.2.1
    (by rw [img_T]; exact hW1) ?_
  intro m1 hrread1 hfr1 hpres1
  -- the load of `weight`
  refine safe_load_buf (b := d.w) (g := gW) (en := enW d) (addrs_w hD31 hBD31)
    (fun i hi => by
      have : i.1.val < d.D := by simpa only [enW, decide_eq_true_eq] using hi
      simp only [gW]; have := hd.w_len; omega)
    (Buf.valid_of_agree hpre.2.2.1 (fun x hx => hpres1 x (hpre.2.2.1.2.2.2 x hx)))
    (by rw [img_w hd]; exact hRw) ?_
  intro b_w16 hw16' _
  have hw16 : ∀ j : Idx [d.BD], enW d j = true →
      init.read .bf16 (d.w.ptr (gW j)).addr = some (b_w16.get j) := by
    intro j hj
    refine read_keep hfr1 (hdisjW _ ?_) (hw16' j hj)
    show j.1.val < d.D
    simpa only [enW, decide_eq_true_eq] using hj
  refine safe_cvt (g := .binary32) (rnd := .rn)
    ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩ ?_
  intro b_w hcvtw
  refine safe_mul ?_
  intro b_x_hat hmA
  refine safe_mul ?_
  intro b_y0 hmB
  -- the load of `g`
  refine safe_load_buf (b := d.g) (g := gx d p) (en := enx d p)
    (addrs_t hD31 hd.d_le hBD31 hrow hot hT64)
    (fun i hi => gx_lt_len hd hd.g_len hi)
    (Buf.valid_of_agree hpre.2.1 (fun x hx => hpres1 x (hpre.2.1.2.2.2 x hx)))
    (by rw [img_x hd]; exact hRg) ?_
  intro b_g16 hg16' _
  have hg16 : ∀ j : Idx [d.BT, d.BD], enx d p j = true →
      init.read .bf16 (d.g.ptr (gx d p j)).addr = some (b_g16.get j) :=
    fun j hj => read_keep hfr1 (hdisjG _ (gx_lt_matSet hd hj)) (hg16' j hj)
  refine safe_cvt (g := .binary32) (rnd := .rn)
    ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩ ?_
  intro b_g hcvtg
  -- the tail after the gate: the rounding, the store of `y`, and `Qn`
  have tail : ∀ b_y : Tensor [d.BT, d.BD] F32,
      (∀ j, GateOut d.swish P (b_y0.get j) (b_g.get j) (b_y.get j)) →
      Safe ((launch d).env P) R (W \ d.rstd.elems (Buf.img (gT d p) (enT d p))) Q m1
        (tl.cvt b_y >>= fun (b_y16 : Tensor [d.BT, d.BD] BF16) =>
          tl.store (pT d.y d o_t) b_y16 (some (mX d o_t T64))) := by
    intro b_y hgate
    refine safe_cvt (f := .binary32) (g := .bfloat16) (rnd := .rn) (x := b_y)
      ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl), Or.inl rfl⟩ ?_
    intro b_y16 hcvt16
    refine safe_last ?_
    refine safe_store_buf (b := d.y) (g := gx d p) (en := enx d p)
      (addrs_t hD31 hd.d_le hBD31 hrow hot hT64)
      (fun i hi => gx_lt_len hd hd.y_len hi)
      (fun i j hi hj h => gx_inj hd hi hj h)
      (Buf.valid_of_agree hpre.2.2.2.1 (fun x hx => hpres1 x (hpre.2.2.2.1.2.2.2 x hx)))
      (by rw [img_x hd, img_T]; exact hW2) ?_
    intro m2 hyread hfr2 _
    have hrread : ∀ j : Idx [d.BT], enT d p j = true →
        m2.read .f32 (d.rstd.ptr (gT d p j)).addr = some (b_rstd.get j) := by
      intro j hj
      have hdj : Disjoint (d.rstd.elems {gT d p j}) (d.y.elems (Buf.img (gx d p) (enx d p))) := by
        rw [img_x hd]
        refine (hd.y_rstd.mono (Buf.elems_mono _ (tileIdx_subset_matSet hd))
          (Buf.elems_mono _ ?_)).symm
        exact Set.singleton_subset_iff.mpr (gT_lt_rowSet hj)
      have h := hrread1 j hj
      rw [Mem.read_congr (m₁ := m2) (m₂ := m1) (d := .f32)
        (a := (d.rstd.ptr (gT d p j : Int)).addr)
        (fun x hx => hfr2 x (fun hS => Set.disjoint_left.mp hdj
          (Buf.span_sub_elems d.rstd (Set.mem_singleton _) hx) hS))]
      exact h
    exact safe_ret (hQ m2 (qn hd hpre p m2 P hsqrt_acc hdiv_full hexp hovf hinf b_x16 b_x b_xx s1
      b_var s2 sq b_rstd b_w16 b_w b_x_hat b_y0 b_g16 b_g b_y b_y16 hx16 hz hcvt hmul
      (fun r => (htree r).imp fun t h => ⟨h.1, h.2.2⟩) hdv haddc hsqrt hdiv hw16 hcvtw hmA hmB
      hg16 hcvtg hgate hcvt16 hyread hrread))
  -- the gate
  cases hsw : d.swish with
  | true =>
    simp only [hsw, ↓reduceIte, tl.sigmoid, bind_assoc]
    refine safe_mul ?_
    intro b_yg hyg
    refine safe_neg ?_
    intro nx hnx
    refine safe_math (fn := .exp) (f := .binary32) (hex := fun _ => hexp_total _) ?_
    intro ex hex
    refine safe_add ?_
    intro den hden
    refine safe_div (f := .binary32) (hex := fun i => hdiv_total _ _) ?_
    intro sg hsg
    refine safe_mul ?_
    intro b_y hy
    refine tail b_y (fun j => ?_)
    rw [hsw]
    exact ⟨nx.get j, ex.get j, den.get j, sg.get j, hnx j, hex j, hden j, hsg j,
      by rw [if_pos rfl]; exact ⟨b_yg.get j, hyg j, hy j⟩⟩
  | false =>
    simp only [hsw, Bool.false_eq_true, ↓reduceIte, tl.sigmoid, bind_assoc]
    refine safe_neg ?_
    intro nx hnx
    refine safe_math (fn := .exp) (f := .binary32) (hex := fun _ => hexp_total _) ?_
    intro ex hex
    refine safe_add ?_
    intro den hden
    refine safe_div (f := .binary32) (hex := fun i => hdiv_total _ _) ?_
    intro sg hsg
    refine safe_mul ?_
    intro b_y hy
    refine tail b_y (fun j => ?_)
    rw [hsw]
    exact ⟨nx.get j, ex.get j, den.get j, sg.get j, hnx j, hex j, hden j, hsg j,
      by rw [if_neg (by decide)]; exact hy j⟩

/-- **Each program is `Safe`**: the index arithmetic, then `walk`. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (P : Platform)
    (hsqrt_total : Device.MathTotal P .sqrt .binary32)
    (hexp_total : Device.MathTotal P .exp .binary32) (hsqrt_acc : Device.SqrtF32 P)
    (hdiv_total : Device.DivTotal P .binary32) (hdiv_full : Device.DivFull P)
    (hexp : Device.ExpF32 P) (hovf : Device.ExpF32Overflow P) (hinf : Device.DivF32InfDivisor P)
    (i : Fin (launch d).size) :
    Safe ((launch d).env P) (R d init i) (W d init i) (fun _ m => Q d init i m) init
      ((model d).body ((launch d).ctx i)) := by
  have hlt : i.val < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  have hp : i.val * d.BT + d.BT ≤ 2 ^ 51 := p_bound hd hlt
  have hi_lt : i.val < 2 ^ 31 := by
    have hg : grid d < 2 ^ 31 := hd.grid_lt
    omega
  have hBT31 : d.BT < 2 ^ 31 := by have := bt_le hd; omega
  have hdisjY : Disjoint (d.y.elems (tileIdx d i.val)) (d.rstd.elems (tileRow d i.val)) :=
    hd.y_rstd.mono (Buf.elems_mono _ (tileIdx_subset_matSet hd))
      (Buf.elems_mono _ tileRow_subset_rowSet)
  have hRx : d.x.elems (tileIdx d i.val) ⊆ R d init i := fun x hx => Or.inl (Or.inl hx)
  have hRg : d.g.elems (tileIdx d i.val) ⊆ R d init i := fun x hx => Or.inl (Or.inr hx)
  have hRw : d.w.elems (vecSet d) ⊆ R d init i := fun x hx => Or.inr hx
  have hW1 : d.rstd.elems (tileRow d i.val) ⊆ W d init i := fun x hx => Or.inr hx
  have hW2 : d.y.elems (tileIdx d i.val) ⊆ W d init i \ d.rstd.elems (tileRow d i.val) :=
    fun x hx => ⟨Or.inl hx, fun hr => Set.disjoint_left.mp hdisjY hx hr⟩
  have hi32 : IntV.toInt (tl.program_id ((launch d).ctx i) 0) = (i.val : Int) := by
    simp only [tl.program_id, launch_pid]
    exact IntV.I32.toInt_ofNat hi_lt
  unfold model layer_norm_gated_fwd_kernel
  dsimp only
  refine safe_convert (d := .i32) (d' := .i64)
    (x := Tensor.scalar (tl.program_id ((launch d).ctx i) 0)) (hex := fun _ => ⟨_, rfl⟩) ?_
  intro r hr
  refine safe_convert (d := .i32) (d' := .i64) (x := tl.arange 0 d.BT)
    (hex := fun _ => ⟨_, rfl⟩) ?_
  intro o_a hoa
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
  have hot := get_o_t (p := i.val) hi_t hoa' hp
  by_cases hT : d.T < 2 ^ 31
  · rw [ite_eq_left_of_eq_true _ _ (eq_true hT)]
    dsimp only
    simp only [pure_bind]
    refine safe_convert (d := .i32) (d' := .i64) (x := Tensor.scalar (IntV.ofNat d.T))
      (hex := fun _ => ⟨_, rfl⟩) ?_
    intro r1 hr1
    have hT32 : IntV.toInt (IntV.ofNat d.T : I32) = (d.T : Int) := IntV.I32.toInt_ofNat hT
    have hT64 : IntV.toInt r1.item = (d.T : Int) := by
      have h1 : r1.item = IntV.ofInt (IntV.toInt ((Tensor.scalar (IntV.ofNat d.T)).get ())) :=
        hr1 ()
      have h2 : IntV.toInt ((Tensor.scalar (IntV.ofNat d.T)).get ()) = (d.T : Int) := hT32
      rw [h1, h2]
      exact IntV.toInt_ofInt (by norm_num) ⟨by omega, by omega⟩
    exact walk hd hpre hp hot hT64 hRx hRg hRw hW1 hW2 (fun m h => h) hsqrt_total hexp_total
      hsqrt_acc hdiv_total hdiv_full hexp hovf hinf
  · rw [ite_eq_right_of_eq_false _ _ (eq_false hT)]
    dsimp only
    simp only [pure_bind]
    have hT64 : IntV.toInt (IntV.ofNat d.T : I64) = (d.T : Int) := by
      refine IntV.toInt_ofNat ?_
      have := t_lt hd
      omega
    exact walk hd hpre hp hot hT64 hRx hRg hRw hW1 hW2 (fun m h => h) hsqrt_total hexp_total
      hsqrt_acc hdiv_total hdiv_full hexp hovf hinf

end Claims.LayerNormGatedFwd

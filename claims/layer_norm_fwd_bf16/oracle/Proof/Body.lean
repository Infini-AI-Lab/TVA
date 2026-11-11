import Proof.Setup
import Proof.LeafAux
import Proof.RowVals
import Proof.YPath
import Proof.LaneVal
import Proof.TolMean
import Proof.TolRstd
import Proof.ValConsts
import Proof.TolY
import Proof.YBound
import Proof.YProd

namespace Claims.LayerNormFwd

open TVA
open Format


/-! The walk of the kernel: `safe_program` steps through the body of one program line by line, one
rule per `tl.*` operation, and shows that the program is safe within its footprints and leaves each
of its rows `Q`: the residual, the mean, `rstd` and the outputs within the tolerances proved in the
`Tol*` and `Y*` modules. -/

/-! ## The kernel body, as a `do` block -/

/-- The kernel body as a `do` block over the instance, without the `model` wrapper. -/
theorem body_eq (d : Inst) (c : Ctx) :
    (model d).body c =
      (do
        let i_t := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id c 0))).item
        let o_t : Tensor [32] I64 :=
          i_t * (IntV.ofNat 32 : I64) + (← tl.convert .i32 .i64 (tl.arange 0 32))
        let o_g ← tl.irem o_t (bc (IntV.ofNat d.G : I64))
        let o_d : Tensor [d.BD] I32 := tl.arange 0 d.BD
        let m_d := o_d.ltS (IntV.ofNat d.D : I32)
        let m_t := o_t.ltS (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat d.T))).item
        let m_x := (m_t.col d.BD).and (m_d.row 32)
        let p_x := (d.x.ptr 0) + o_t.col d.BD * (IntV.ofNat d.D : I64) + o_d.row 32
        let b_x0 : Tensor [32, d.BD] F32 ← tl.cvt (← tl.load p_x (some m_x)
          (some (← tl.convert .f32 .bf16 (tl.full [32, d.BD] (FloatV.litF32 (FloatV.ofPat 0))))))
        let b_x : Tensor [32, d.BD] F32 ← if d.hasRes then do
            let p_res := (d.res.ptr 0) + o_t.col d.BD * (IntV.ofNat d.D : I64) +
              o_d.row 32
            tl.add b_x0 (← tl.cvt (← tl.load p_res (some m_x)
              (some (← tl.convert .f32 .bf16 (tl.full [32, d.BD] (FloatV.litF32 (FloatV.ofPat 0)))))))
          else pure b_x0
        if d.hasRes then do
          let p_res_out := (d.res_out.ptr 0) + o_t.col d.BD * (IntV.ofNat d.D : I64) +
            o_d.row 32
          let b_x16 : Tensor [32, d.BD] BF16 ← tl.cvt b_x
          tl.store p_res_out b_x16 (some m_x)
        let b_mean : Tensor [32] F32 ←
          tl.div (← tl.sum1 b_x) (bc (FloatV.litF32 (FloatV.ofRat (d.D : ℚ))))
        let p_mean := (d.mean.ptr 0) + o_t
        tl.store p_mean b_mean (some m_t)
        let b_xbar := tl.where (m_d.row 32) (← tl.subc b_x (b_mean.col d.BD))
          (bc (FloatV.litF32 (FloatV.ofPat 0)))
        let b_var : Tensor [32] F32 ←
          tl.div (← tl.sum1c (← tl.mul b_xbar b_xbar)) (bc (FloatV.litF32 (FloatV.ofRat (d.D : ℚ))))
        let b_rstd : Tensor [32] F32 ←
          tl.div (bc (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)))
            (← tl.sqrt (← tl.addc b_var (bc (FloatV.hostF32 d.eps))))
        let p_rstd := (d.rstd.ptr 0) + o_t
        tl.store p_rstd b_rstd (some m_t)
        let p_w := (d.w.ptr 0) + o_g.col d.BD * (IntV.ofNat d.D : I64) + o_d.row 32
        let b_w : Tensor [32, d.BD] F32 ← tl.cvt (← tl.load p_w (some (m_d.row 32)))
        let p_b := (d.b.ptr 0) + o_g.col d.BD * (IntV.ofNat d.D : I64) + o_d.row 32
        let b_b : Tensor [32, d.BD] F32 ← tl.cvt (← tl.load p_b (some (m_d.row 32)))
        let b_x_hat : Tensor [32, d.BD] F32 ←
          tl.mul (← tl.subc b_x (b_mean.col d.BD)) (b_rstd.col d.BD)
        let b_y0 : Tensor [32, d.BD] F32 ← tl.mul b_x_hat b_w
        let b_y : Tensor [32, d.BD] F32 ← tl.addc b_y0 b_b
        let p_y := (d.y.ptr 0) + o_t.col d.BD * (IntV.ofNat d.D : I64) + o_d.row 32
        let b_y16 : Tensor [32, d.BD] BF16 ← tl.cvt b_y
        tl.store p_y b_y16 (some m_x)) := rfl

/-! ## The launch -/

/-- The grid has `⌈T / 32⌉` programs. -/
theorem launch_size (d : Inst) : (launch d).size = (d.T + 31) / 32 := Launch.size_grid1 _ _ _

/-- In the launch, the program id of instance `i` along axis 0 is `i`. -/
theorem launch_pid (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = i := by
  have hi : (i : Nat) < (d.T + 31) / 32 := by rw [← launch_size d]; exact i.isLt
  simp [Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi]

/-- `tl.program_id` on instance `i` of the launch returns `i` as an integer value. -/
theorem program_id_eq (d : Inst) (i : Fin (launch d).size) :
    tl.program_id ((launch d).ctx i) 0 = IntV.ofNat (i : Nat) := by
  simp [tl.program_id, launch_pid d i]

/-- The grid is below `2^31` programs. -/
theorem grid_lt {d : Inst} (hd : d.ok) : (d.T + 31) / 32 < 2 ^ 31 := by
  have hT := hd.t_lt
  rw [Nat.div_lt_iff_lt_mul (by norm_num : 0 < 32)]
  omega

/-- For an accepted instance the block width `BD` is positive. -/
theorem bd_pos {d : Inst} (hd : d.ok) : 0 < d.BD := by
  obtain ⟨e, he⟩ := hd.bd_pow2
  rw [he]
  exact Nat.two_pow_pos e

/-- For an accepted instance the block width `BD` is at most 1024. -/
theorem bd_le {d : Inst} (hd : d.ok) : d.BD ≤ 1024 := by
  have h := hd.bd_lt
  have h2 := hd.d_le
  omega

/-- The shape `[32]` satisfies the Triton shape constraints. -/
theorem tritonOk32 : Shape.TritonOk ([32] : Shape) := by
  refine ⟨fun k hk => ?_, by norm_num⟩
  simp only [List.mem_singleton] at hk
  subst hk
  exact ⟨5, rfl⟩

/-- The block shape `[BD]` is one Triton accepts. -/
theorem tritonOk_bd {d : Inst} (hd : d.ok) : Shape.TritonOk ([d.BD] : Shape) := by
  refine ⟨fun k hk => ?_, ?_⟩
  · simp only [List.mem_singleton] at hk
    subst hk
    exact hd.bd_pow2
  · have h := bd_le hd
    simp only [List.prod_cons, List.prod_nil, Nat.mul_one]
    omega

/-- The shape `[32, BD]` satisfies the Triton shape constraints for an accepted instance. -/
theorem tritonOk_32_bd {d : Inst} (hd : d.ok) : Shape.TritonOk ([32, d.BD] : Shape) := by
  refine ⟨fun k hk => ?_, by
    have h := bd_le hd
    simp only [List.prod_cons, List.prod_nil, Nat.mul_one]
    omega⟩
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
  rcases hk with rfl | rfl
  · exact ⟨5, rfl⟩
  · exact hd.bd_pow2

/-- The shape `[32, 1]` satisfies the Triton shape constraints. -/
theorem tritonOk_32_1 : Shape.TritonOk ([32, 1] : Shape) := by
  refine ⟨fun k hk => ?_, by norm_num⟩
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
  rcases hk with rfl | rfl
  · exact ⟨5, rfl⟩
  · exact ⟨0, rfl⟩

/-- The shape `[1, BD]` satisfies the Triton shape constraints for an accepted instance. -/
theorem tritonOk_1_bd {d : Inst} (hd : d.ok) : Shape.TritonOk ([1, d.BD] : Shape) := by
  refine ⟨fun k hk => ?_, by
    have h := bd_le hd
    simp only [List.prod_cons, List.prod_nil, Nat.mul_one]
    omega⟩
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
  rcases hk with rfl | rfl
  · exact ⟨0, rfl⟩
  · exact hd.bd_pow2

/-- The launch is valid: a one-dimensional grid under `2^31` whose block shapes Triton accepts. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 (grid_lt hd) fun s hs => ?_
  simp only [List.mem_cons, List.not_mem_nil,
    or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl | rfl
  · exact tritonOk32
  · exact tritonOk_bd hd
  · exact tritonOk_32_1
  · exact tritonOk_1_bd hd
  · exact tritonOk_32_bd hd

/-! ## The proof -/

set_option maxHeartbeats 1000000 in
/-- **One program is safe.** Program `i` run alone from `init` cannot fault, hang or race, stays inside `R` and `W`, and ends in a memory satisfying `Q`: the walk of the kernel body, line by line, with the device facts for `/`, `sqrt` and the depth of `tl.sum`. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (P : Platform)
    (hdivt : Device.DivTotal P .binary32) (hsqrt_t : Device.MathTotal P .sqrt .binary32)
    (hsqrt : Device.SqrtF32 P) (hdivf : Device.DivFull P) (hsumD : Device.SumDepth P)
    (i : Fin (launch d).size) :
    Safe ((launch d).env P) (R d i) (W d i) (fun _ m => Q d init i m) init
      ((model d).body ((launch d).ctx i)) := by
  obtain ⟨hvx, hvres, hvw, hvb, hvy, hvresout, hvmean, hvrstd, hInOk, hWB⟩ := hpre
  have hp : (i : Nat) < 2 ^ 31 := by
    have h1 : (i : Nat) < (launch d).size := i.isLt
    have h2 : (launch d).size = (d.T + 31) / 32 := launch_size d
    have h3 : (d.T + 31) / 32 < 2 ^ 31 := grid_lt hd
    omega
  have hT31 : d.T < 2 ^ 31 := hd.t_lt
  have hD31 : d.D < 2 ^ 31 := by have := hd.d_le; omega
  have hBD31 : d.BD < 2 ^ 31 := by have := bd_le hd; omega
  have hGpos : 0 < d.G := hd.g_pos
  rw [body_eq d ((launch d).ctx i)]
  cases hres : d.hasRes
  case true =>
      simp only [ite_true]
      refine safe_convert (d := .i32) (d' := .i64) (fun j => ⟨_, rfl⟩) ?_
      intro t1 h1
      refine safe_convert (d := .i32) (d' := .i64) (fun j => ⟨_, rfl⟩) ?_
      intro ar har
      have hit : t1.item = IntV.ofInt (i : Int) := by
        have h := h1 ()
        have h2 : (Tensor.scalar (tl.program_id ((launch d).ctx ↑i) 0)).get () =
            tl.program_id ((launch d).ctx ↑i) 0 := rfl
        rw [h2, program_id_eq d i] at h
        exact it_eq hp h
      have hot : ∀ r : Fin 32,
          ((t1.item * (IntV.ofNat 32 : I64) + ar).get1 r).toInt = 32 * (i : Int) + (r : Int) :=
        fun r => oT_toInt hp hit (ar_eq r (har (r, ())))
      have hG2 : d.G < 2 ^ 62 := by have := hd.g_le; omega
      refine safe_irem ?_ ?_
      · intro j
        obtain ⟨r, u⟩ := j
        cases u
        have ha : 0 ≤ ((t1.item * (IntV.ofNat 32 : I64) + ar).get1 r).toInt := by
          rw [hot r]
          have h1 : (0 : Int) ≤ (i : Int) := Int.natCast_nonneg _
          have h2 : (0 : Int) ≤ (r : Int) := Int.natCast_nonneg _
          omega
        have hb : 0 < ((bc (IntV.ofNat d.G : I64) : Tensor [32] I64).get1 r).toInt := by
          have hgf : ((Tensor.full (IntV.ofNat d.G : I64) : Tensor [32] I64).get1 r) =
              (IntV.ofNat d.G : I64) := rfl
          rw [hgf, toInt_ofNat64' hG2]
          exact_mod_cast hGpos
        obtain ⟨q, hq, _⟩ := IntV.rem?_of_nonneg ha hb
        exact ⟨q, hq⟩
      intro og hog
      refine safe_convert (d := .i32) (d' := .i64) (fun j => ⟨_, rfl⟩) ?_
      intro t4 h4
      have hT : t4.item = IntV.ofInt (d.T : Int) := by
        have h := h4 ()
        have h2 : (Tensor.scalar (IntV.ofNat d.T : I32) : Tensor [] I32).get () =
            (IntV.ofNat d.T : I32) := rfl
        rw [h2] at h
        have h3 := conv_i32_i64 h
        rw [toInt_ofNat32' hT31] at h3
        exact h3
      have hT' : IntV.toInt t4.item = (d.T : Int) := by
        rw [hT, toInt_ofInt64' (by omega) (by omega)]
      have hod : ∀ j : Fin d.BD, IntV.toInt ((tl.arange 0 d.BD).get1 j) = (j : Int) := by
        intro j
        have h2 : (tl.arange 0 d.BD).get1 j = IntV.ofInt (0 + (j : Nat)) := rfl
        rw [h2, toInt_ofInt32' (by omega) (by omega)]
        omega
      refine safe_convert (d := .f32) (d' := .bf16)
        (fun j => ⟨_, Or.inl ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl),
          FloatV.NaNEq.refl _⟩⟩) ?_
      intro z hz
      refine safe_load_buf (b := d.x)
        (g := fun (ix : Idx [32, d.BD]) => idx d (32 * (i : Nat) + (ix.1 : Nat)) (ix.2.1 : Nat))
        (en := fun (ix : Idx [32, d.BD]) =>
          decide (32 * (i : Nat) + (ix.1 : Nat) < d.T ∧ (ix.2.1 : Nat) < d.D)) ?_ ?_ ?_ ?_ ?_
      · exact mat_addrs hd i d.x hp hot hod (fun r => mT_get hT31 r (hot r) hT')
          (fun j => mD_get hD31 j (hod j))
      · exact mat_hin hd i (B := d.x) hd.x_len
      · exact hvx
      · exact (Buf.elems_mono d.x (img_mat_sub_yRows i)).trans (x_elems_sub_R d i)
      · intro v hld hoth
        refine safe_cvt ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩ ?_
        intro bx0 hbx0
        refine safe_convert (d := .f32) (d' := .bf16)
          (fun j => ⟨_, Or.inl ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl),
            FloatV.NaNEq.refl _⟩⟩) ?_
        intro z2 hz2
        refine safe_load_buf (b := d.res)
          (g := fun (ix : Idx [32, d.BD]) => idx d (32 * (i : Nat) + (ix.1 : Nat)) (ix.2.1 : Nat))
          (en := fun (ix : Idx [32, d.BD]) =>
            decide (32 * (i : Nat) + (ix.1 : Nat) < d.T ∧ (ix.2.1 : Nat) < d.D)) ?_ ?_ ?_ ?_ ?_
        · exact mat_addrs hd i d.res hp hot hod (fun r => mT_get hT31 r (hot r) hT')
            (fun j => mD_get hD31 j (hod j))
        · exact mat_hin hd i (B := d.res) (hd.res_len hres)
        · exact hvres hres
        · exact (Buf.elems_mono d.res (img_mat_sub_resRows i hres)).trans (res_elems_sub_R d i)
        · intro v2 hld2r hoth2
          refine safe_cvt ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩ ?_
          intro br hbr
          refine safe_add ?_
          intro bx hbx
          refine safe_cvt ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl), Or.inl rfl⟩ ?_
          intro bx16 hbx16
          refine safe_store_buf (b := d.res_out) (W := W d i)
            (g := fun (ix : Idx [32, d.BD]) => idx d (32 * (i : Nat) + (ix.1 : Nat)) (ix.2.1 : Nat))
            (en := fun (ix : Idx [32, d.BD]) =>
              decide (32 * (i : Nat) + (ix.1 : Nat) < d.T ∧ (ix.2.1 : Nat) < d.D)) ?_ ?_ ?_ ?_ ?_ ?_
          · exact mat_addrs hd i d.res_out hp hot hod (fun r => mT_get hT31 r (hot r) hT')
              (fun j => mD_get hD31 j (hod j))
          · exact mat_hin hd i (B := d.res_out) (hd.res_out_len hres)
          · exact mat_img_inj (d := d) i
          · exact hvresout hres
          · exact (Buf.elems_mono d.res_out (img_mat_sub_resRows i hres)).trans
              (res_out_elems_sub_W d i)
          · intro m2 hld2 hfr2 hpr2
            have hp2 : ∀ x, init x ≠ none → m2 x ≠ none := hpr2
            have hvmean2 : d.mean.Valid m2 := valid_of_present hvmean hp2
            refine safe_sum1 (bd_pos hd) ?_
            intro bsum hsum
            refine safe_div (hex := fun j => hdivt _ _) ?_
            intro bmean hbmean
            refine safe_store_buf (b := d.mean)
              (g := fun (ix : Idx [32]) => 32 * (i : Nat) + (ix.1 : Nat))
              (en := fun (ix : Idx [32]) => decide (32 * (i : Nat) + (ix.1 : Nat) < d.T)) ?_ ?_ ?_ ?_ ?_ ?_
            · exact row_addrs d.mean (i : Nat) hp hot (fun r => mT_get hT31 r (hot r) hT')
            · exact row_hin hd i (B := d.mean) hd.mean_len
            · exact row_img_inj (d := d) (i : Nat)
            · exact hvmean2
            · exact mean_elems_sub_W_sdiff hd (i : Nat)
                (img_mat_sub_resRows (i : Nat) hres) (img_row_sub_mRows (i : Nat))
            · intro m3 hld3 hfr3 hpr3
              have hp3 : ∀ x, init x ≠ none → m3 x ≠ none := fun x hx => hpr3 x (hp2 x hx)
              have hvrstd3 : d.rstd.Valid m3 := valid_of_present hvrstd hp3
              refine safe_subc ?_
              intro bxh hbxh
              refine safe_mul ?_
              intro bsq1 hbsq1
              refine safe_sum1c (bd_pos hd) ?_
              intro bvar hbvar
              refine safe_div (hex := fun j => hdivt _ _) ?_
              intro bva hbva
              refine safe_addc ?_
              intro be hbe
              refine safe_math (fn := .sqrt) (hex := fun j => hsqrt_t _) ?_
              intro bsq hbsq
              refine safe_div (hex := fun j => hdivt _ _) ?_
              intro brstd hbrstd
              refine safe_store_buf (b := d.rstd)
                (g := fun (ix : Idx [32]) => 32 * (i : Nat) + (ix.1 : Nat))
                (en := fun (ix : Idx [32]) => decide (32 * (i : Nat) + (ix.1 : Nat) < d.T))
                ?_ ?_ ?_ ?_ ?_ ?_
              · exact row_addrs d.rstd (i : Nat) hp hot (fun r => mT_get hT31 r (hot r) hT')
              · exact row_hin hd i (B := d.rstd) hd.rstd_len
              · exact row_img_inj (d := d) (i : Nat)
              · exact hvrstd3
              · exact rstd_elems_sub_W_sdiff hd (i : Nat)
                  (img_mat_sub_resRows (i : Nat) hres) (img_row_sub_mRows (i : Nat))
                  (img_row_sub_mRows (i : Nat))
              · intro m4 hld4 hfr4 hpr4
                have hp4 : ∀ x, init x ≠ none → m4 x ≠ none := fun x hx => hpr4 x (hp3 x hx)
                have hvw4 : d.w.Valid m4 := valid_of_present hvw hp4
                have hvb4 : d.b.Valid m4 := valid_of_present hvb hp4
                have hvy4 : d.y.Valid m4 := valid_of_present hvy hp4
                refine safe_load_buf (b := d.w)
                  (g := fun (ix : Idx [32, d.BD]) =>
                    gidx d (32 * (i : Nat) + (ix.1 : Nat)) (ix.2.1 : Nat))
                  (en := fun (ix : Idx [32, d.BD]) => decide ((ix.2.1 : Nat) < d.D)) ?_ ?_ ?_ ?_ ?_
                · exact vec_addrs hd i d.w hp
                    (fun r => oG_toInt hGpos hG2 (hot r) (hog (r, ()))) hod
                    (fun j => mD_get hD31 j (hod j))
                · exact vec_hin hd i (B := d.w) hd.w_len
                · exact hvw4
                · exact (Buf.elems_mono d.w (img_vec_sub_wRows (i : Nat))).trans
                    (w_elems_sub_R d i)
                · intro vw hlw hothw
                  refine safe_cvt ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)),
                    Or.inl rfl⟩ ?_
                  intro bw hbw
                  refine safe_load_buf (b := d.b)
                    (g := fun (ix : Idx [32, d.BD]) =>
                      gidx d (32 * (i : Nat) + (ix.1 : Nat)) (ix.2.1 : Nat))
                    (en := fun (ix : Idx [32, d.BD]) => decide ((ix.2.1 : Nat) < d.D)) ?_ ?_ ?_ ?_ ?_
                  · exact vec_addrs hd i d.b hp
                      (fun r => oG_toInt hGpos hG2 (hot r) (hog (r, ()))) hod
                      (fun j => mD_get hD31 j (hod j))
                  · exact vec_hin hd i (B := d.b) hd.b_len
                  · exact hvb4
                  · exact (Buf.elems_mono d.b (img_vec_sub_wRows (i : Nat))).trans
                      (b_elems_sub_R d i)
                  · intro vb hlb hothb
                    refine safe_cvt ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)),
                      Or.inl rfl⟩ ?_
                    intro bb hbb
                    refine safe_subc ?_
                    intro bx2 hbx2
                    refine safe_mul ?_
                    intro bhat hhat
                    refine safe_mul ?_
                    intro by0 hy0
                    refine safe_addc ?_
                    intro byy hyy
                    refine safe_cvt ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl),
                      Or.inl rfl⟩ ?_
                    intro by16 hy16
                    refine safe_last ?_
                    refine safe_store_buf (b := d.y)
                      (g := fun (ix : Idx [32, d.BD]) =>
                        idx d (32 * (i : Nat) + (ix.1 : Nat)) (ix.2.1 : Nat))
                      (en := fun (ix : Idx [32, d.BD]) =>
                        decide (32 * (i : Nat) + (ix.1 : Nat) < d.T ∧ (ix.2.1 : Nat) < d.D))
                      ?_ ?_ ?_ ?_ ?_ ?_
                    · exact mat_addrs hd i d.y hp hot hod (fun r => mT_get hT31 r (hot r) hT')
                        (fun j => mD_get hD31 j (hod j))
                    · exact mat_hin hd i (B := d.y) hd.y_len
                    · exact mat_img_inj (d := d) i
                    · exact hvy4
                    · exact y_elems_sub_W_sdiff hd (i : Nat)
                        (img_mat_sub_resRows (i : Nat) hres) (img_row_sub_mRows (i : Nat))
                        (img_row_sub_mRows (i : Nat)) (img_mat_sub_yRows (i : Nat))
                    · intro mF hldF hfrF hprF
                      have hpF : ∀ x, init x ≠ none → mF x ≠ none := fun x hx => hprF x (hp4 x hx)
                      have hvresoutF : d.res_out.Valid mF := valid_of_present (hvresout hres) hpF
                      have hvmeanF : d.mean.Valid mF := valid_of_present hvmean hpF
                      have hvrstdF : d.rstd.Valid mF := valid_of_present hvrstd hpF
                      have hvyF : d.y.Valid mF := valid_of_present hvy hpF
                      refine safe_ret ?_
                      intro t ht htp
                      let r : Fin 32 := ⟨t % 32, Nat.mod_lt _ (by norm_num)⟩
                      have hr : (r : Nat) = t % 32 := rfl
                      have h32 : 32 * (i : Nat) + (r : Nat) = t := by
                        rw [hr, ← htp]; exact Nat.div_add_mod t 32
                      let Sx : Set Nat := Buf.img
                        (fun ix : Idx [32, d.BD] =>
                          idx d (32 * (i : Nat) + (ix.1 : Nat)) (ix.2.1 : Nat))
                        (fun ix : Idx [32, d.BD] =>
                          decide (32 * (i : Nat) + (ix.1 : Nat) < d.T ∧ (ix.2.1 : Nat) < d.D))
                      let Sr : Set Nat := Buf.img
                        (fun ix : Idx [32] => 32 * (i : Nat) + (ix.1 : Nat))
                        (fun ix : Idx [32] => decide (32 * (i : Nat) + (ix.1 : Nat) < d.T))
                      have hmemm : t ∈ Sr := mem_img_row htp ht
                      have dYM : Disjoint (d.mean.elems Sr) (d.y.elems Sx) :=
                        elems_disjoint_mono hd.y_mean.symm
                          ((img_row_sub_mRows (i : Nat)).trans (mRows_subset_rowSet d (i : Nat)))
                          ((img_mat_sub_yRows (i : Nat)).trans (yRows_subset_matSet d (i : Nat)))
                      have dMR : Disjoint (d.mean.elems Sr) (d.rstd.elems Sr) :=
                        elems_disjoint_mono hd.mean_rstd
                          ((img_row_sub_mRows (i : Nat)).trans (mRows_subset_rowSet d (i : Nat)))
                          ((img_row_sub_mRows (i : Nat)).trans (mRows_subset_rowSet d (i : Nat)))
                      have hrdm0 : mF.read .f32 (d.mean.ptr (t : Int)).addr
                          = m4.read .f32 (d.mean.ptr (t : Int)).addr :=
                        read_frame (B := d.mean) (S := Sr) hmemm (C := d.y) (T := Sx) dYM hfrF
                      have hrdm1 : m4.read .f32 (d.mean.ptr (t : Int)).addr
                          = m3.read .f32 (d.mean.ptr (t : Int)).addr :=
                        read_frame (B := d.mean) (S := Sr) hmemm (C := d.rstd) (T := Sr) dMR hfr4
                      have hrdm2 : m3.read .f32 (d.mean.ptr (t : Int)).addr
                          = some (bmean.get (r, ())) :=
                        row_read htp ht hld3
                      have dYR : Disjoint (d.rstd.elems Sr) (d.y.elems Sx) :=
                        elems_disjoint_mono hd.y_rstd.symm
                          ((img_row_sub_mRows (i : Nat)).trans (mRows_subset_rowSet d (i : Nat)))
                          ((img_mat_sub_yRows (i : Nat)).trans (yRows_subset_matSet d (i : Nat)))
                      have hrd_r0 : mF.read .f32 (d.rstd.ptr (t : Int)).addr
                          = m4.read .f32 (d.rstd.ptr (t : Int)).addr :=
                        read_frame (B := d.rstd) (S := Sr) hmemm (C := d.y) (T := Sx) dYR hfrF
                      have hrd_r1 : m4.read .f32 (d.rstd.ptr (t : Int)).addr
                          = some (brstd.get (r, ())) :=
                        row_read htp ht hld4
                      have hsum57 : ∀ j, j < d.D →
                          |(xv d init t j).toRat| + |(rv d init t j).toRat| ≤ (2 : ℚ) ^ 57 := by
                        intro j hj
                        have h2 : (2 : ℚ) ^ 57 = 2 ^ 56 + 2 ^ 56 := by norm_num
                        linarith [(hInOk t ht j hj).1.2, ((hInOk t ht j hj).2 hres).2]
                      have hbdd : ∀ j, j < d.D → |sR d init t j| ≤ (2 : ℝ) ^ 57 := by
                        intro j hj
                        have h := add_cvt_abs_le (a := xv d init t j) (b := rv d init t j)
                          (hInOk t ht j hj).1.1 ((hInOk t ht j hj).2 hres).1 (hsum57 j hj)
                        unfold sR sv FloatV.toReal
                        rw [ite_eq_left hres]
                        exact_mod_cast h
                      have hdec_all : ∀ j, j < d.BD → ¬ j < d.D →
                          decide (32 * (i : Nat) + (r : Nat) < d.T ∧ j < d.D) = false := by
                        intro j hjBD hj
                        have h32 : 32 * (i : Nat) + (r : Nat) = t := by
                          rw [hr, ← htp]; exact Nat.div_add_mod t 32
                        rw [h32]
                        exact decide_eq_false (fun hc => hj hc.2)
                      have hzero_lane : ∀ (zz : Tensor [32, d.BD] BF16)
                          (hzz : ∀ ix : Idx [32, d.BD],
                            ConvRel ((launch d).env P).P .f32 .bf16
                              ((tl.full [32, d.BD] (FloatV.ofPat (0 : ℕ)).litF32).get ix)
                              (zz.get ix)) (ix : Idx [32, d.BD]),
                          (zz.get ix).toReal = 0 ∧ (zz.get ix).isFinite = true := by
                        intro zz hzz ix
                        refine ConvRel.f32_bf16_zero (P := ((launch d).env P).P) ?_
                        simpa only [tl.full, Tensor.full, FloatV.litF32_ofPat_zero] using hzz ix
                      have hfin : ∀ j, j < d.BD →
                          (if h : j < d.BD then bx.get (r, ⟨j, h⟩, ()) else default).isFinite
                            = true := by
                        intro j hjBD
                        rw [dite_eq_left hjBD]
                        by_cases hj : j < d.D
                        · have hxvj : xv d init t j = v.get (r, ⟨j, hjBD⟩, ()) := by
                            have h : v.get2 r ⟨j, hjBD⟩ = xv d init t j := by
                              simpa only [xv, Tensor.get2] using
                                (rval_of_read (mat_read htp ht hj hjBD hld)).symm
                            simpa only [Tensor.get2] using h.symm
                          have hrvj : rv d init t j = v2.get (r, ⟨j, hjBD⟩, ()) := by
                            have h : v2.get2 r ⟨j, hjBD⟩ = rv d init t j := by
                              simpa only [rv, Tensor.get2] using
                                (rval_of_read (mat_read htp ht hj hjBD hld2r)).symm
                            simpa only [Tensor.get2] using h.symm
                          have ha : (v.get (r, ⟨j, hjBD⟩, ())).isFinite = true := by
                            rw [← hxvj]; exact (hInOk t ht j hj).1.1
                          have hb : (v2.get (r, ⟨j, hjBD⟩, ())).isFinite = true := by
                            rw [← hrvj]; exact ((hInOk t ht j hj).2 hres).1
                          have hab : |(v.get (r, ⟨j, hjBD⟩, ())).toRat|
                              + |(v2.get (r, ⟨j, hjBD⟩, ())).toRat| ≤ (2 : ℚ) ^ 57 := by
                            rw [← hxvj, ← hrvj]; exact hsum57 j hj
                          exact (lane_sum (a := v.get (r, ⟨j, hjBD⟩, ()))
                            (b := v2.get (r, ⟨j, hjBD⟩, ()))
                            (bx0 := bx0.get (r, ⟨j, hjBD⟩, ()))
                            (br := br.get (r, ⟨j, hjBD⟩, ()))
                            (bx := bx.get (r, ⟨j, hjBD⟩, ())) ha hb hab
                            (hbx0 (r, ⟨j, hjBD⟩, ())) (hbr (r, ⟨j, hjBD⟩, ()))
                            (hbx (r, ⟨j, hjBD⟩, ()))).2.1
                        · obtain ⟨hzr, hzf⟩ := hzero_lane z hz (r, ⟨j, hjBD⟩, ())
                          obtain ⟨hz2r, hz2f⟩ := hzero_lane z2 hz2 (r, ⟨j, hjBD⟩, ())
                          have hdec := hdec_all j hjBD hj
                          exact (lane_sum_zero
                            (a := z.get (r, ⟨j, hjBD⟩, ()))
                            (b := z2.get (r, ⟨j, hjBD⟩, ()))
                            (bx0 := bx0.get (r, ⟨j, hjBD⟩, ()))
                            (br := br.get (r, ⟨j, hjBD⟩, ()))
                            (bx := bx.get (r, ⟨j, hjBD⟩, ())) hzf hz2f
                            (FloatV.toRat_eq_zero_of_toReal hzr)
                            (FloatV.toRat_eq_zero_of_toReal hz2r)
                            (by rw [← hoth (r, ⟨j, hjBD⟩, ()) hdec z rfl]
                                exact hbx0 (r, ⟨j, hjBD⟩, ()))
                            (by rw [← hoth2 (r, ⟨j, hjBD⟩, ()) hdec z2 rfl]
                                exact hbr (r, ⟨j, hjBD⟩, ()))
                            (hbx (r, ⟨j, hjBD⟩, ()))).1
                      have hval : ∀ j, j < d.D →
                          (if h : j < d.BD then bx.get (r, ⟨j, h⟩, ()) else default).toReal
                            = sR d init t j := by
                        intro j hj
                        have hjBD : j < d.BD := lt_of_lt_of_le hj hd.d_le_bd
                        rw [dite_eq_left hjBD]
                        have hxvj : xv d init t j = v.get (r, ⟨j, hjBD⟩, ()) := by
                          have h : v.get2 r ⟨j, hjBD⟩ = xv d init t j := by
                            simpa only [xv, Tensor.get2] using
                              (rval_of_read (mat_read htp ht hj hjBD hld)).symm
                          simpa only [Tensor.get2] using h.symm
                        have hrvj : rv d init t j = v2.get (r, ⟨j, hjBD⟩, ()) := by
                          have h : v2.get2 r ⟨j, hjBD⟩ = rv d init t j := by
                            simpa only [rv, Tensor.get2] using
                              (rval_of_read (mat_read htp ht hj hjBD hld2r)).symm
                          simpa only [Tensor.get2] using h.symm
                        have ha : (v.get (r, ⟨j, hjBD⟩, ())).isFinite = true := by
                          rw [← hxvj]; exact (hInOk t ht j hj).1.1
                        have hb : (v2.get (r, ⟨j, hjBD⟩, ())).isFinite = true := by
                          rw [← hrvj]; exact ((hInOk t ht j hj).2 hres).1
                        have hab : |(v.get (r, ⟨j, hjBD⟩, ())).toRat|
                            + |(v2.get (r, ⟨j, hjBD⟩, ())).toRat| ≤ (2 : ℚ) ^ 57 := by
                          rw [← hxvj, ← hrvj]; exact hsum57 j hj
                        have hbl := (lane_sum (a := v.get (r, ⟨j, hjBD⟩, ()))
                          (b := v2.get (r, ⟨j, hjBD⟩, ()))
                          (bx0 := bx0.get (r, ⟨j, hjBD⟩, ()))
                          (br := br.get (r, ⟨j, hjBD⟩, ()))
                          (bx := bx.get (r, ⟨j, hjBD⟩, ())) ha hb hab
                          (hbx0 (r, ⟨j, hjBD⟩, ())) (hbr (r, ⟨j, hjBD⟩, ()))
                          (hbx (r, ⟨j, hjBD⟩, ()))).1
                        rw [hbl, ← hxvj, ← hrvj]
                        unfold sR sv
                        rw [ite_eq_left hres]
                      have hzero : ∀ j, j < d.BD → d.D ≤ j →
                          (if h : j < d.BD then bx.get (r, ⟨j, h⟩, ()) else default).toReal
                            = 0 := by
                        intro j hjBD hDj
                        rw [dite_eq_left hjBD]
                        obtain ⟨hzr, hzf⟩ := hzero_lane z hz (r, ⟨j, hjBD⟩, ())
                        obtain ⟨hz2r, hz2f⟩ := hzero_lane z2 hz2 (r, ⟨j, hjBD⟩, ())
                        have hdec := hdec_all j hjBD (by omega)
                        exact (lane_sum_zero
                          (a := z.get (r, ⟨j, hjBD⟩, ()))
                          (b := z2.get (r, ⟨j, hjBD⟩, ()))
                          (bx0 := bx0.get (r, ⟨j, hjBD⟩, ()))
                          (br := br.get (r, ⟨j, hjBD⟩, ()))
                          (bx := bx.get (r, ⟨j, hjBD⟩, ())) hzf hz2f
                            (FloatV.toRat_eq_zero_of_toReal hzr)
                            (FloatV.toRat_eq_zero_of_toReal hz2r)
                          (by rw [← hoth (r, ⟨j, hjBD⟩, ()) hdec z rfl]
                              exact hbx0 (r, ⟨j, hjBD⟩, ()))
                          (by rw [← hoth2 (r, ⟨j, hjBD⟩, ()) hdec z2 rfl]
                              exact hbr (r, ⟨j, hjBD⟩, ()))
                          (hbx (r, ⟨j, hjBD⟩, ()))).2
                      have hcover : ∃ tr : Tree, Tree.Covers d.BD tr ∧
                          tr.depth ≤ Nat.clog 2 d.BD ∧
                            Tree.Eval (AddRel false)
                              (fun j => if h : j < d.BD then bx.get (r, ⟨j, h⟩, ()) else default)
                              tr (bsum.get (r, ())) := by
                        obtain ⟨tr, hcov, hdep, hev⟩ := hsum r
                        exact ⟨tr, hcov, hdep.trans (max_le (hsumD _) le_rfl), hev⟩
                      have hdvR : ((bc (FloatV.litF32 (FloatV.ofRat (d.D : ℚ))) : Tensor [32] F32).get
                          (r, ())).toReal = (d.D : ℝ) := by
                        show FloatV.toReal (FloatV.litF32 (FloatV.ofRat (d.D : ℚ))) = (d.D : ℝ)
                        exact dv_toReal hd
                      have hdvF : ((bc (FloatV.litF32 (FloatV.ofRat (d.D : ℚ))) : Tensor [32] F32).get
                          (r, ())).isFinite = true := by
                        show FloatV.isFinite (FloatV.litF32 (FloatV.ofRat (d.D : ℚ))) = true
                        exact dv_isFinite hd
                      have hm := mean_tol (d := d) (init := init) (t := t) (P := P) hd hdivf
                        (bx := fun j => if h : j < d.BD then bx.get (r, ⟨j, h⟩, ()) else default)
                        (dv := ((bc (FloatV.litF32 (FloatV.ofRat (d.D : ℚ))) : Tensor [32] F32).get (r, ())))
                        (bsum := bsum.get (r, ())) (bmean := bmean.get (r, ()))
                        hdvR hdvF hfin hval hzero hbdd hcover (hbmean (r, ()))
                      have hbmfin : (bmean.get (r, ())).isFinite = true := hm.1
                      have hbm58 : |(bmean.get (r, ())).toReal| ≤ (2 : ℝ) ^ 58 := by
                        have h2 := rstd_tol_meanTol_le (d := d) hd hbdd
                        have h3 := rstd_tol_absMean_le (d := d) (init := init) (t := t) hbdd hd.d_pos
                        have h4 := abs_meanRef_le_absMean (d := d) (init := init) (t := t) hd.d_pos
                        nlinarith [abs_nonneg (meanRef d init t), hm.2,
                          abs_sub_abs_le_abs_sub (bmean.get (r, ())).toReal (meanRef d init t)]
                      have hbm58q : |(bmean.get (r, ())).toRat| ≤ (2 : ℚ) ^ 58 := by
                        have h := hbm58
                        unfold FloatV.toReal at h
                        exact_mod_cast h
                      let x : Nat → F32 := fun j =>
                        if h : j < d.BD then
                          (tl.where (((tl.arange 0 d.BD).ltS (IntV.ofNat d.D)).row 32) bxh
                            (bc (FloatV.ofPat (0 : ℕ)).litF32)).get2 r ⟨j, h⟩
                        else default
                      let sq : Nat → F32 := fun j =>
                        if h : j < d.BD then bsq1.get2 r ⟨j, h⟩ else default
                      have hx_def : ∀ j (hj : j < d.BD), x j =
                          (tl.where (((tl.arange 0 d.BD).ltS (IntV.ofNat d.D)).row 32) bxh
                            (bc (FloatV.ofPat (0 : ℕ)).litF32)).get2 r ⟨j, hj⟩ := by
                        intro j hj
                        dsimp only [x]
                        rw [dite_eq_left hj]
                      have hsq_def : ∀ j (hj : j < d.BD), sq j = bsq1.get2 r ⟨j, hj⟩ := by
                        intro j hj
                        dsimp only [sq]
                        rw [dite_eq_left hj]
                      have hmask_t : ∀ j (hj : j < d.BD), j < d.D →
                          (((tl.arange 0 d.BD).ltS (IntV.ofNat d.D)).row 32).get2 r ⟨j, hj⟩
                            = true := by
                        intro j hj hjD
                        rw [mask_lane hD31 hBD31 r ⟨j, hj⟩]
                        exact decide_eq_true hjD
                      have hmask_f : ∀ j (hj : j < d.BD), ¬ j < d.D →
                          (((tl.arange 0 d.BD).ltS (IntV.ofNat d.D)).row 32).get2 r ⟨j, hj⟩
                            = false := by
                        intro j hj hjD
                        rw [mask_lane hD31 hBD31 r ⟨j, hj⟩]
                        exact decide_eq_false_iff_not.mpr hjD
                      have hwhere_t : ∀ j (hj : j < d.BD), j < d.D →
                          (tl.where (((tl.arange 0 d.BD).ltS (IntV.ofNat d.D)).row 32) bxh
                            (bc (FloatV.ofPat (0 : ℕ)).litF32)).get2 r ⟨j, hj⟩
                            = bxh.get2 r ⟨j, hj⟩ := by
                        intro j hj hjD
                        rw [where_lane
                          (c := ((tl.arange 0 d.BD).ltS (IntV.ofNat d.D)).row 32)
                          (x := bxh) (y := (bc (FloatV.ofPat (0 : ℕ)).litF32)) r ⟨j, hj⟩,
                          hmask_t j hj hjD]
                        exact ite_eq_left rfl
                      have hwhere_f : ∀ j (hj : j < d.BD), ¬ j < d.D →
                          (tl.where (((tl.arange 0 d.BD).ltS (IntV.ofNat d.D)).row 32) bxh
                            (bc (FloatV.ofPat (0 : ℕ)).litF32)).get2 r ⟨j, hj⟩
                            = (bc (FloatV.ofPat (0 : ℕ)).litF32).get2 r ⟨j, hj⟩ := by
                        intro j hj hjD
                        rw [where_lane
                          (c := ((tl.arange 0 d.BD).ltS (IntV.ofNat d.D)).row 32)
                          (x := bxh) (y := (bc (FloatV.ofPat (0 : ℕ)).litF32)) r ⟨j, hj⟩,
                          hmask_f j hj hjD]
                        exact ite_eq_right (by simp)
                      have hbx57 : ∀ j (hj : j < d.D) (hjBD : j < d.BD),
                          |(bx.get (r, ⟨j, hjBD⟩, ())).toRat| ≤ (2 : ℚ) ^ 57 := by
                        intro j hj hjBD
                        have hb := hval j hj
                        rw [dite_eq_left hjBD] at hb
                        have h1 : |(bx.get (r, ⟨j, hjBD⟩, ())).toReal| ≤ (2 : ℝ) ^ 57 := by
                          rw [hb]
                          exact hbdd j hj
                        unfold FloatV.toReal at h1
                        exact_mod_cast h1
                      have hbmn58 : ∀ j (hjBD : j < d.BD),
                          |(((bmean.col d.BD).get (r, ⟨j, hjBD⟩, ())).neg).toRat|
                            ≤ (2 : ℚ) ^ 58 := by
                        intro j hjBD
                        simpa only [FloatV.toRat_neg, abs_neg, Tensor.get_col] using hbm58q
                      have hdevD : ∀ j (hj : j < d.D) (hjBD : j < d.BD),
                          AddRel ((launch d).env P).fusion (bx.get (r, ⟨j, hjBD⟩, ()))
                            (((bmean.col d.BD).get (r, ⟨j, hjBD⟩, ())).neg)
                            (bxh.get (r, ⟨j, hjBD⟩, ())) := by
                        intro j hj hjBD
                        exact hbxh (r, ⟨j, hjBD⟩, ())
                      have hfinD : ∀ j, j < d.D →
                          (if h : j < d.BD then bx.get (r, ⟨j, h⟩, ()) else default).isFinite
                            = true :=
                        fun j hj => hfin j (lt_of_lt_of_le hj hd.d_le_bd)
                      have hxfin : ∀ j, j < d.BD → (x j).isFinite = true := by
                        intro j hj
                        rw [hx_def j hj]
                        by_cases hjD : j < d.D
                        · rw [hwhere_t j hj hjD]
                          have hfinb : (bx.get (r, ⟨j, hj⟩, ())).isFinite = true := by
                            have h := hfin j hj
                            rw [dite_eq_left hj] at h
                            exact h
                          have hnegfin : (((bmean.col d.BD).get (r, ⟨j, hj⟩, ())).neg).isFinite
                              = true := by
                            rw [FloatV.isFinite_neg]
                            simpa only [Tensor.get_col] using hbmfin
                          have hM : |(bx.get (r, ⟨j, hj⟩, ())).toRat|
                              + |(((bmean.col d.BD).get (r, ⟨j, hj⟩, ())).neg).toRat|
                              ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
                            have hroom : (2 : ℚ) ^ 57 + 2 ^ 58
                                ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
                              rw [Format.binary32_maxVal, Format.binary32_eta]; norm_num
                            linarith [hbx57 j hjD hj, hbmn58 j hj]
                          exact (FloatV.AddRel.err Format.binary32_regular (by decide)
                            (by rw [Format.binary32_u]; norm_num) (hdevD j hjD hj) hfinb hnegfin
                            hM).1
                        · rw [hwhere_f j hj hjD]
                          exact bc_zero_isFinite r ⟨j, hj⟩
                      have hdev : ∀ j, j < d.D → AddRel true
                          ((fun j => if h : j < d.BD then bx.get (r, ⟨j, h⟩, ()) else default) j)
                          (FloatV.neg (bmean.get (r, ()))) (x j) := by
                        intro j hj
                        have hjBD : j < d.BD := lt_of_lt_of_le hj hd.d_le_bd
                        have hbapp : ((fun j => if h : j < d.BD then bx.get (r, ⟨j, h⟩, ())
                            else default) j)
                            = (if h : j < d.BD then bx.get (r, ⟨j, h⟩, ()) else default) := rfl
                        rw [hbapp, dite_eq_left hjBD, hx_def j hjBD, hwhere_t j hjBD hj]
                        exact AddRel.of_true (hbxh (r, ⟨j, hjBD⟩, ()))
                      have hzero_x : ∀ j, j < d.BD → d.D ≤ j → (x j).toReal = 0 := by
                        intro j hj hjD
                        rw [hx_def j hj, hwhere_f j hj (by omega)]
                        exact bc_zero_toReal r ⟨j, hj⟩
                      have hxbd : ∀ j, j < d.BD → |(x j).toReal| ≤ 2 ^ 58 + 2 ^ 40 := by
                        intro j hj
                        exact (rstd_tol_x_bd (d := d) hd
                          (bx := fun j => if h : j < d.BD then bx.get (r, ⟨j, h⟩, ()) else default)
                          (bmean := bmean.get (r, ())) (x := x)
                          hbmfin hm.2 hfinD hval hbdd hxfin hdev hzero_x j hj).2
                      have hmulfin : ∀ j, j < d.BD →
                          (FloatV.mul (x j) (x j)).isFinite = true := by
                        intro j hj
                        have hxb : |(x j).toRat| ≤ (2 : ℚ) ^ 58 + 2 ^ 40 := by
                          have h := hxbd j hj
                          unfold FloatV.toReal at h
                          exact_mod_cast h
                        have hmax : |(x j).toRat * (x j).toRat| ≤ Format.binary32.maxVal := by
                          rw [abs_mul]
                          have h2 : |(x j).toRat| * |(x j).toRat|
                              ≤ ((2 : ℚ) ^ 58 + 2 ^ 40) * ((2 : ℚ) ^ 58 + 2 ^ 40) :=
                            mul_le_mul hxb hxb (abs_nonneg _) (by positivity)
                          have h3 : ((2 : ℚ) ^ 58 + 2 ^ 40) * ((2 : ℚ) ^ 58 + 2 ^ 40)
                              ≤ Format.binary32.maxVal := by
                            rw [Format.binary32_maxVal]; norm_num
                          linarith
                        exact (FloatV.mul_err Format.binary32_regular (hxfin j hj) (hxfin j hj)
                          hmax).1
                      have hsq_fl : ∀ j (hj : j < d.BD), sq j = FloatV.mul (x j) (x j) := by
                        intro j hj
                        have hfm : (FloatV.mul (x j) (x j)).isFinite = true := hmulfin j hj
                        rw [hx_def j hj] at hfm
                        rw [hsq_def j hj, hx_def j hj]
                        exact naneq_eq_of_isFinite (hbsq1 (r, ⟨j, hj⟩, ())) hfm
                      have hfsq : ∀ j, j < d.BD → (sq j).isFinite = true := by
                        intro j hj
                        rw [hsq_fl j hj]
                        exact hmulfin j hj
                      have hsq : ∀ j, j < d.BD → FloatV.NaNEq (FloatV.mul (x j) (x j)) (sq j) := by
                        intro j hj
                        rw [hsq_fl j hj]
                        exact FloatV.NaNEq.refl _
                      have hcover_x : ∃ tr : Tree, Tree.Covers d.BD tr ∧
                          tr.depth ≤ Nat.clog 2 d.BD ∧
                            Tree.Eval (AddRel true) sq tr (bvar.get1 r) := by
                        obtain ⟨tr, hcov, hdep, hev⟩ := hbvar r
                        exact ⟨tr, hcov, hdep.trans (max_le (hsumD _) le_rfl),
                          Tree.Eval.mono (fun a b c h => AddRel.of_true h) hev⟩
                      have hrt := rstd_tol (d := d) (init := init) (t := t) (P := P)
                        (hdivf := hdivf) (hsqrt := hsqrt)
                        (bx := fun j => if h : j < d.BD then bx.get (r, ⟨j, h⟩, ()) else default)
                        (bmean := bmean.get (r, ()))
                        (dv := ((bc (FloatV.litF32 (FloatV.ofRat (d.D : ℚ))) : Tensor [32] F32).get
                          (r, ())))
                        (epsV := ((bc (FloatV.hostF32 d.eps) : Tensor [32] F32).get (r, ())))
                        (one := FloatV.litF32 (FloatV.ofPat (0x3FF0000000000000 : ℕ)))
                        (x := x) (sq := sq) (bvar := bvar.get1 r) (bva := bva.get (r, ()))
                        (be := be.get (r, ())) (bsq := bsq.get (r, ())) (brstd := brstd.get (r, ()))
                        hd hdvR hdvF one_toReal one_isFinite hbmfin hm.2 hfinD hval hbdd hxfin hdev
                        hzero_x hfsq hsq hcover_x (hbva (r, ())) (AddRel.of_true (hbe (r, ())))
                        (epsV_isFinite hd) (epsV_ge hd) (epsV_le hd) (epsV_err hd)
                        (hbsq (r, ())) (hbrstd (r, ()))
                      refine ⟨?_, ?_, ?_, ?_⟩
                      · -- (a) the residual store
                        intro _ j hj
                        have hjBD : j < d.BD := lt_of_lt_of_le hj hd.d_le_bd
                        let jj : Fin d.BD := ⟨j, hjBD⟩
                        have hmem : idx d t j ∈ Sx :=
                          mem_img_mat htp ht hj hjBD
                        have dY_resout : Disjoint (d.res_out.elems Sx) (d.y.elems Sx) :=
                          (elems_disjoint_mono hd.y_res_out
                            ((img_mat_sub_yRows (i : Nat)).trans
                              (yRows_subset_matSet d (i : Nat)))
                            ((img_mat_sub_resRows (i : Nat) hres).trans
                              (resRows_subset_resSet d (i : Nat)))).symm
                        have dR_rstd : Disjoint (d.res_out.elems Sx) (d.rstd.elems Sr) :=
                          elems_disjoint_mono hd.res_out_rstd
                            ((img_mat_sub_resRows (i : Nat) hres).trans
                              (resRows_subset_resSet d (i : Nat)))
                            ((img_row_sub_mRows (i : Nat)).trans
                              (mRows_subset_rowSet d (i : Nat)))
                        have dR_mean : Disjoint (d.res_out.elems Sx) (d.mean.elems Sr) :=
                          elems_disjoint_mono hd.res_out_mean
                            ((img_mat_sub_resRows (i : Nat) hres).trans
                              (resRows_subset_resSet d (i : Nat)))
                            ((img_row_sub_mRows (i : Nat)).trans
                              (mRows_subset_rowSet d (i : Nat)))
                        have hrd0 : mF.read .bf16 (d.res_out.ptr (idx d t j : Int)).addr
                            = m4.read .bf16 (d.res_out.ptr (idx d t j : Int)).addr :=
                          read_frame (B := d.res_out) (S := Sx) hmem (C := d.y) (T := Sx)
                            dY_resout hfrF
                        have hrd1 : m4.read .bf16 (d.res_out.ptr (idx d t j : Int)).addr
                            = m3.read .bf16 (d.res_out.ptr (idx d t j : Int)).addr :=
                          read_frame (B := d.res_out) (S := Sx) hmem (C := d.rstd) (T := Sr)
                            dR_rstd hfr4
                        have hrd2 : m3.read .bf16 (d.res_out.ptr (idx d t j : Int)).addr
                            = m2.read .bf16 (d.res_out.ptr (idx d t j : Int)).addr :=
                          read_frame (B := d.res_out) (S := Sx) hmem (C := d.mean) (T := Sr)
                            dR_mean hfr3
                        have hrd3 : m2.read .bf16 (d.res_out.ptr (idx d t j : Int)).addr
                            = some (bx16.get2 r jj) :=
                          mat_read htp ht hj hjBD hld2
                        have hxv : xv d init t j = v.get (r, jj, ()) := by
                          have h : v.get (r, jj, ()) = xv d init t j := by
                            simpa only [xv, Tensor.get2] using
                              (rval_of_read (mat_read htp ht hj hjBD hld)).symm
                          exact h.symm
                        have hrv : rv d init t j = v2.get (r, jj, ()) := by
                          have h : v2.get (r, jj, ()) = rv d init t j := by
                            simpa only [rv, Tensor.get2] using
                              (rval_of_read (mat_read htp ht hj hjBD hld2r)).symm
                          exact h.symm
                        have hxvok := (hInOk t ht j hj).1
                        have hrvok := (hInOk t ht j hj).2 hres
                        have hsum57 : |(xv d init t j).toRat| + |(rv d init t j).toRat|
                            ≤ (2 : ℚ) ^ 57 := by
                          have h2 : (2 : ℚ) ^ 57 = 2 ^ 56 + 2 ^ 56 := by norm_num
                          linarith [hxvok.2, hrvok.2]
                        have hbx0l : bx0.get (r, jj, ()) = FloatV.cvt (xv d init t j) := by
                          have hvfin : (v.get (r, jj, ())).isFinite = true := by
                            rw [← hxv]; exact hxvok.1
                          have hcvt : (FloatV.cvtWith .rn (v.get (r, jj, ())) : F32).isFinite = true := by
                            rw [FloatV.cvtWith_rn]; exact (cvt_bf16_f32_spec hvfin).1
                          have h := naneq_eq_of_isFinite (hbx0 (r, jj, ())) hcvt
                          rw [h, FloatV.cvtWith_rn, ← hxv]
                        have hbrl : br.get (r, jj, ()) = FloatV.cvt (rv d init t j) := by
                          have hvfin : (v2.get (r, jj, ())).isFinite = true := by
                            rw [← hrv]; exact hrvok.1
                          have hcvt : (FloatV.cvtWith .rn (v2.get (r, jj, ())) : F32).isFinite = true := by
                            rw [FloatV.cvtWith_rn]; exact (cvt_bf16_f32_spec hvfin).1
                          have h := naneq_eq_of_isFinite (hbr (r, jj, ())) hcvt
                          rw [h, FloatV.cvtWith_rn, ← hrv]
                        have hbxl : bx.get (r, jj, ())
                            = FloatV.add (FloatV.cvt (xv d init t j)) (FloatV.cvt (rv d init t j)) := by
                          have hfin : (FloatV.add (bx0.get (r, jj, ())) (br.get (r, jj, ()))).isFinite = true := by
                            rw [hbx0l, hbrl]
                            exact add_cvt_isFinite hxvok.1 hrvok.1 hsum57
                          have h := naneq_eq_of_isFinite (hbx (r, jj, ())) hfin
                          rw [h, hbx0l, hbrl]
                        have hbxfin : (bx.get (r, jj, ())).isFinite = true := by
                          rw [hbxl]; exact add_cvt_isFinite hxvok.1 hrvok.1 hsum57
                        have hbx57 : |(bx.get (r, jj, ())).toRat| ≤ (2 : ℚ) ^ 57 := by
                          rw [hbxl]; exact add_cvt_abs_le hxvok.1 hrvok.1 hsum57
                        have hbxl16 : bx16.get (r, jj, ()) = FloatV.cvt (bx.get (r, jj, ())) := by
                          have hc : (FloatV.cvtWith .rn (bx.get (r, jj, ())) : BF16).isFinite = true := by
                            rw [FloatV.cvtWith_rn]
                            exact cvt_f32_bf16_fin hbxfin hbx57
                          have h := naneq_eq_of_isFinite (hbx16 (r, jj, ())) hc
                          rw [h, FloatV.cvtWith_rn]
                        have hsv : bx16.get (r, jj, ()) = FloatV.cvt (sv d init t j) := by
                          rw [hbxl16, hbxl, sv, ite_eq_left hres]
                        calc mF.read .bf16 (d.res_out.ptr (idx d t j : Int)).addr
                            = m4.read .bf16 (d.res_out.ptr (idx d t j : Int)).addr := hrd0
                          _ = m3.read .bf16 (d.res_out.ptr (idx d t j : Int)).addr := hrd1
                          _ = m2.read .bf16 (d.res_out.ptr (idx d t j : Int)).addr := hrd2
                          _ = some (bx16.get (r, jj, ())) := hrd3
                          _ = some (FloatV.cvt (sv d init t j)) := by rw [hsv]
                      · -- (b) the row's mean
                        exact ⟨bmean.get (r, ()), by rw [hrdm0, hrdm1]; exact hrdm2, hm.1, hm.2⟩
                      · -- (c) the inverse standard deviation
                        refine ⟨brstd.get (r, ()), ?_, hrt.1, hrt.2⟩
                        rw [hrd_r0, hrd_r1]
                      · -- (d) the output element
                        intro j hj
                        have hjBD : j < d.BD := lt_of_lt_of_le hj hd.d_le_bd
                        let jj : Fin d.BD := ⟨j, hjBD⟩
                        have hwmem : gidx d (32 * (i : Nat) + (r : Nat)) j ∈ wRows d i :=
                          ⟨(r : Nat), r.isLt, j, hj, rfl⟩
                        have hysub : d.y.elems Sx ⊆ W d i :=
                          (d.y.elems_mono (img_mat_sub_yRows (i : Nat))).trans
                            (y_elems_sub_W d i)
                        have hrstdsub : d.rstd.elems Sr ⊆ W d i :=
                          (d.rstd.elems_mono (img_row_sub_mRows (i : Nat))).trans
                            (rstd_elems_sub_W d i)
                        have hmeansub : d.mean.elems Sr ⊆ W d i :=
                          (d.mean.elems_mono (img_row_sub_mRows (i : Nat))).trans
                            (mean_elems_sub_W d i)
                        have hressub : d.res_out.elems Sx ⊆ W d i :=
                          (d.res_out.elems_mono (img_mat_sub_resRows (i : Nat) hres)).trans
                            (res_out_elems_sub_W d i)
                        have dYw : Disjoint (d.w.elems (wRows d i)) (d.y.elems Sx) :=
                          disj_sub (R_disjoint_W hd i i) (w_elems_sub_R d i) hysub
                        have dB_y : Disjoint (d.b.elems (wRows d i)) (d.y.elems Sx) :=
                          disj_sub (R_disjoint_W hd i i) (b_elems_sub_R d i) hysub
                        have dW_rstd : Disjoint (d.w.elems (wRows d i)) (d.rstd.elems Sr) :=
                          disj_sub (R_disjoint_W hd i i) (w_elems_sub_R d i) hrstdsub
                        have dW_mean : Disjoint (d.w.elems (wRows d i)) (d.mean.elems Sr) :=
                          disj_sub (R_disjoint_W hd i i) (w_elems_sub_R d i) hmeansub
                        have dW_res : Disjoint (d.w.elems (wRows d i)) (d.res_out.elems Sx) :=
                          disj_sub (R_disjoint_W hd i i) (w_elems_sub_R d i) hressub
                        have dB_rstd : Disjoint (d.b.elems (wRows d i)) (d.rstd.elems Sr) :=
                          disj_sub (R_disjoint_W hd i i) (b_elems_sub_R d i) hrstdsub
                        have dB_mean : Disjoint (d.b.elems (wRows d i)) (d.mean.elems Sr) :=
                          disj_sub (R_disjoint_W hd i i) (b_elems_sub_R d i) hmeansub
                        have dB_res : Disjoint (d.b.elems (wRows d i)) (d.res_out.elems Sx) :=
                          disj_sub (R_disjoint_W hd i i) (b_elems_sub_R d i) hressub
                        have hw0 : mF.read .bf16
                            (d.w.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr
                            = m4.read .bf16
                            (d.w.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr :=
                          read_frame (B := d.w) (S := wRows d i) hwmem (C := d.y) (T := Sx)
                            dYw hfrF
                        have hw1 : m4.read .bf16
                            (d.w.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr
                            = m3.read .bf16
                            (d.w.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr :=
                          read_frame (B := d.w) (S := wRows d i) hwmem (C := d.rstd) (T := Sr)
                            dW_rstd hfr4
                        have hw2 : m3.read .bf16
                            (d.w.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr
                            = m2.read .bf16
                            (d.w.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr :=
                          read_frame (B := d.w) (S := wRows d i) hwmem (C := d.mean) (T := Sr)
                            dW_mean hfr3
                        have hw3 : m2.read .bf16
                            (d.w.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr
                            = init.read .bf16
                            (d.w.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr :=
                          read_frame (B := d.w) (S := wRows d i) hwmem (C := d.res_out) (T := Sx)
                            dW_res hfr2
                        have hw4 : m4.read .bf16
                            (d.w.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr
                            = some (vw.get (r, jj, ())) :=
                          hlw (r, jj, ()) (by simpa [jj] using hj)
                        have hwread : init.read .bf16 (d.w.ptr (↑(gidx d t j) : Int)).addr
                            = some (vw.get (r, jj, ())) := by
                          rw [← h32]
                          exact hw3.symm.trans (hw2.symm.trans (hw1.symm.trans hw4))
                        have hwv : vw.get (r, jj, ()) = wv d init t j := by
                          have h : wv d init t j = vw.get (r, jj, ()) := by
                            simpa only [wv] using rval_of_read (p := d.w) hwread
                          exact h.symm
                        have hwmemt : gidx d t j ∈ wRows d i := by
                          rw [← h32]
                          exact hwmem
                        have hwok : InOkW (wv d init t j) :=
                          (hWB (gidx d t j) (wRows_subset_vecSet hd i hwmemt)).1
                        have hwvfin : (wv d init t j).isFinite = true := hwok.1
                        have hbwl : bw.get (r, jj, ()) = FloatV.cvt (wv d init t j) := by
                          have hvfin : (vw.get (r, jj, ())).isFinite = true := by
                            rw [hwv]; exact hwvfin
                          have hcvt : (FloatV.cvtWith .rn (vw.get (r, jj, ())) : F32).isFinite
                              = true := by
                            rw [FloatV.cvtWith_rn]; exact (cvt_bf16_f32_spec hvfin).1
                          have h := naneq_eq_of_isFinite (hbw (r, jj, ())) hcvt
                          rw [h, FloatV.cvtWith_rn, hwv]
                        have hbwfin : (bw.get (r, jj, ())).isFinite = true := by
                          rw [hbwl]; exact (cvt_bf16_f32_spec hwvfin).1
                        have hbwv : (bw.get (r, jj, ())).toReal = (wv d init t j).toReal := by
                          rw [hbwl]
                          unfold FloatV.toReal
                          exact_mod_cast (cvt_bf16_f32_spec hwvfin).2
                        have hbwbd : |(bw.get (r, jj, ())).toReal| ≤ (2 : ℝ) ^ 16 := by
                          rw [hbwv]
                          have h := hwok.2
                          unfold FloatV.toReal
                          exact_mod_cast h
                        have hb0 : mF.read .bf16
                            (d.b.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr
                            = m4.read .bf16
                            (d.b.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr :=
                          read_frame (B := d.b) (S := wRows d i) hwmem (C := d.y) (T := Sx)
                            dB_y hfrF
                        have hb1 : m4.read .bf16
                            (d.b.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr
                            = m3.read .bf16
                            (d.b.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr :=
                          read_frame (B := d.b) (S := wRows d i) hwmem (C := d.rstd) (T := Sr)
                            dB_rstd hfr4
                        have hb2 : m3.read .bf16
                            (d.b.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr
                            = m2.read .bf16
                            (d.b.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr :=
                          read_frame (B := d.b) (S := wRows d i) hwmem (C := d.mean) (T := Sr)
                            dB_mean hfr3
                        have hb3 : m2.read .bf16
                            (d.b.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr
                            = init.read .bf16
                            (d.b.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr :=
                          read_frame (B := d.b) (S := wRows d i) hwmem (C := d.res_out) (T := Sx)
                            dB_res hfr2
                        have hb4 : m4.read .bf16
                            (d.b.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr
                            = some (vb.get (r, jj, ())) :=
                          hlb (r, jj, ()) (by simpa [jj] using hj)
                        have hbread : init.read .bf16 (d.b.ptr (↑(gidx d t j) : Int)).addr
                            = some (vb.get (r, jj, ())) := by
                          rw [← h32]
                          exact hb3.symm.trans (hb2.symm.trans (hb1.symm.trans hb4))
                        have hbv : vb.get (r, jj, ()) = bv d init t j := by
                          have h : bv d init t j = vb.get (r, jj, ()) := by
                            simpa only [bv] using rval_of_read (p := d.b) hbread
                          exact h.symm
                        have hbok : InOkW (bv d init t j) :=
                          (hWB (gidx d t j) (wRows_subset_vecSet hd i hwmemt)).2
                        have hbvfin : (bv d init t j).isFinite = true := hbok.1
                        have hbbl : bb.get (r, jj, ()) = FloatV.cvt (bv d init t j) := by
                          have hvfin : (vb.get (r, jj, ())).isFinite = true := by
                            rw [hbv]; exact hbvfin
                          have hcvt : (FloatV.cvtWith .rn (vb.get (r, jj, ())) : F32).isFinite
                              = true := by
                            rw [FloatV.cvtWith_rn]; exact (cvt_bf16_f32_spec hvfin).1
                          have h := naneq_eq_of_isFinite (hbb (r, jj, ())) hcvt
                          rw [h, FloatV.cvtWith_rn, hbv]
                        have hbbfin : (bb.get (r, jj, ())).isFinite = true := by
                          rw [hbbl]; exact (cvt_bf16_f32_spec hbvfin).1
                        have hbbv : (bb.get (r, jj, ())).toReal = (bv d init t j).toReal := by
                          rw [hbbl]
                          unfold FloatV.toReal
                          exact_mod_cast (cvt_bf16_f32_spec hbvfin).2
                        have hbbbd : |(bb.get (r, jj, ())).toReal| ≤ (2 : ℝ) ^ 16 := by
                          rw [hbbv]
                          have h := hbok.2
                          unfold FloatV.toReal
                          exact_mod_cast h
                        have hfinb : (bx.get (r, jj, ())).isFinite = true := by
                          have h := hfin j hjBD
                          rw [dite_eq_left hjBD] at h
                          exact h
                        have hnegfin : (((bmean.col d.BD).get (r, jj, ())).neg).isFinite
                            = true := by
                          rw [FloatV.isFinite_neg]
                          simpa only [Tensor.get_col] using hbmfin
                        have hbxj57 : |(bx.get (r, jj, ())).toRat| ≤ (2 : ℚ) ^ 57 :=
                          hbx57 j hj hjBD
                        have hbmnj58 : |(((bmean.col d.BD).get (r, jj, ())).neg).toRat|
                            ≤ (2 : ℚ) ^ 58 := hbmn58 j hjBD
                        have hM : |(bx.get (r, jj, ())).toRat|
                            + |(((bmean.col d.BD).get (r, jj, ())).neg).toRat|
                            ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
                          have hroom : (2 : ℚ) ^ 57 + 2 ^ 58
                              ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
                            rw [Format.binary32_maxVal, Format.binary32_eta]; norm_num
                          linarith [hbxj57, hbmnj58]
                        have hbx2fin : (bx2.get (r, jj, ())).isFinite = true :=
                          (FloatV.AddRel.err Format.binary32_regular (by decide)
                            (by rw [Format.binary32_u]; norm_num) (hbx2 (r, jj, ())) hfinb hnegfin
                            hM).1
                        have hdevD2 : AddRel true (bx.get (r, jj, ()))
                            (FloatV.neg (bmean.get (r, ()))) (bx2.get (r, jj, ())) := by
                          have h := hbx2 (r, jj, ())
                          simp only [Tensor.get_col] at h
                          exact AddRel.of_true h
                        have hvalj : (bx.get (r, jj, ())).toReal = sR d init t j := by
                          have h := hval j hj
                          rw [dite_eq_left hjBD] at h
                          exact h
                        have hhat1 : FloatV.NaNEq (FloatV.mul (bx2.get (r, jj, ()))
                            (brstd.get (r, ()))) (bhat.get (r, jj, ())) := by
                          have h := hhat (r, jj, ())
                          simpa only [Tensor.get_col] using h
                        obtain ⟨hbhatfin, hby0fin, hbhatbd, hby0bd⟩ :=
                          y_hat_by0_bd (d := d) hd (init := init) (t := t) (j := j) hj
                            (bx0 := bx.get (r, jj, ())) (bmean := bmean.get (r, ()))
                            (bx := bx2.get (r, jj, ())) (brstd := brstd.get (r, ()))
                            (bw := bw.get (r, jj, ())) (bhat := bhat.get (r, jj, ()))
                            (by0 := by0.get (r, jj, ()))
                            hvalj hfinb (hbdd j hj) hbmfin hm.2 hrt.1 hrt.2 hdevD2 hbx2fin
                            hhat1 hbwfin hbwbd (hy0 (r, jj, ()))
                        have hyy1 : AddRel true (by0.get (r, jj, ())) (bb.get (r, jj, ()))
                            (byy.get (r, jj, ())) := AddRel.of_true (hyy (r, jj, ()))
                        have hyt := y_tol (d := d) hd (init := init) (t := t) (j := j) hj
                          (bx0 := bx.get (r, jj, ())) (bmean := bmean.get (r, ()))
                          (bx := bx2.get (r, jj, ())) (brstd := brstd.get (r, ()))
                          (bw := bw.get (r, jj, ())) (bb := bb.get (r, jj, ()))
                          (bhat := bhat.get (r, jj, ())) (by0 := by0.get (r, jj, ()))
                          (byy := byy.get (r, jj, ())) (by16 := by16.get (r, jj, ()))
                          hvalj hfinb (hbdd j hj) hbmfin hm.2 hrt.1 hrt.2 hdevD2 hbx2fin
                          hhat1 hbhatfin hbhatbd (hy0 (r, jj, ())) hby0fin hby0bd
                          hbwfin hbwv hbwbd hbbfin hbbv hbbbd hyy1 (hy16 (r, jj, ()))
                        exact ⟨by16.get (r, jj, ()), mat_read htp ht hj hjBD hldF, hyt.1, hyt.2⟩
  case false =>
      simp only [Bool.false_eq_true, ite_false]
      refine safe_convert (d := .i32) (d' := .i64) (fun j => ⟨_, rfl⟩) ?_
      intro t1 h1
      refine safe_convert (d := .i32) (d' := .i64) (fun j => ⟨_, rfl⟩) ?_
      intro ar har
      have hit : t1.item = IntV.ofInt (i : Int) := by
        have h := h1 ()
        have h2 : (Tensor.scalar (tl.program_id ((launch d).ctx ↑i) 0)).get () =
            tl.program_id ((launch d).ctx ↑i) 0 := rfl
        rw [h2, program_id_eq d i] at h
        exact it_eq hp h
      have hot : ∀ r : Fin 32,
          ((t1.item * (IntV.ofNat 32 : I64) + ar).get1 r).toInt = 32 * (i : Int) + (r : Int) :=
        fun r => oT_toInt hp hit (ar_eq r (har (r, ())))
      have hG2 : d.G < 2 ^ 62 := by have := hd.g_le; omega
      refine safe_irem ?_ ?_
      · intro j
        obtain ⟨r, u⟩ := j
        cases u
        have ha : 0 ≤ ((t1.item * (IntV.ofNat 32 : I64) + ar).get1 r).toInt := by
          rw [hot r]
          have h1 : (0 : Int) ≤ (i : Int) := Int.natCast_nonneg _
          have h2 : (0 : Int) ≤ (r : Int) := Int.natCast_nonneg _
          omega
        have hb : 0 < ((bc (IntV.ofNat d.G : I64) : Tensor [32] I64).get1 r).toInt := by
          have hgf : ((Tensor.full (IntV.ofNat d.G : I64) : Tensor [32] I64).get1 r) =
              (IntV.ofNat d.G : I64) := rfl
          rw [hgf, toInt_ofNat64' hG2]
          exact_mod_cast hGpos
        obtain ⟨q, hq, _⟩ := IntV.rem?_of_nonneg ha hb
        exact ⟨q, hq⟩
      intro og hog
      refine safe_convert (d := .i32) (d' := .i64) (fun j => ⟨_, rfl⟩) ?_
      intro t4 h4
      have hT : t4.item = IntV.ofInt (d.T : Int) := by
        have h := h4 ()
        have h2 : (Tensor.scalar (IntV.ofNat d.T : I32) : Tensor [] I32).get () =
            (IntV.ofNat d.T : I32) := rfl
        rw [h2] at h
        have h3 := conv_i32_i64 h
        rw [toInt_ofNat32' hT31] at h3
        exact h3
      have hT' : IntV.toInt t4.item = (d.T : Int) := by
        rw [hT, toInt_ofInt64' (by omega) (by omega)]
      have hod : ∀ j : Fin d.BD, IntV.toInt ((tl.arange 0 d.BD).get1 j) = (j : Int) := by
        intro j
        have h2 : (tl.arange 0 d.BD).get1 j = IntV.ofInt (0 + (j : Nat)) := rfl
        rw [h2, toInt_ofInt32' (by omega) (by omega)]
        omega
      refine safe_convert (d := .f32) (d' := .bf16)
        (fun j => ⟨_, Or.inl ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl),
          FloatV.NaNEq.refl _⟩⟩) ?_
      intro z hz
      refine safe_load_buf (b := d.x)
        (g := fun (ix : Idx [32, d.BD]) => idx d (32 * (i : Nat) + (ix.1 : Nat)) (ix.2.1 : Nat))
        (en := fun (ix : Idx [32, d.BD]) =>
          decide (32 * (i : Nat) + (ix.1 : Nat) < d.T ∧ (ix.2.1 : Nat) < d.D)) ?_ ?_ ?_ ?_ ?_
      · exact mat_addrs hd i d.x hp hot hod (fun r => mT_get hT31 r (hot r) hT')
          (fun j => mD_get hD31 j (hod j))
      · exact mat_hin hd i (B := d.x) hd.x_len
      · exact hvx
      · exact (Buf.elems_mono d.x (img_mat_sub_yRows i)).trans (x_elems_sub_R d i)
      · intro v hld hoth
        refine safe_cvt ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩ ?_
        intro bx0 hbx0
        refine safe_sum1 (bd_pos hd) ?_
        intro bsum hsum
        refine safe_div (hex := fun j => hdivt _ _) ?_
        intro bmean hbmean
        refine safe_store_buf (b := d.mean)
          (ptrs := d.mean.ptr 0 + (t1.item * IntV.ofNat 32 + ar)) (vals := bmean)
          (mask := some ((t1.item * IntV.ofNat 32 + ar).ltS t4.item))
          (g := fun (ix : Idx [32]) => 32 * (i : Nat) + (ix.1 : Nat))
          (en := fun (ix : Idx [32]) => decide (32 * (i : Nat) + (ix.1 : Nat) < d.T)) ?_ ?_ ?_ ?_ ?_ ?_
        · exact row_addrs d.mean (i : Nat) hp hot (fun r => mT_get hT31 r (hot r) hT')
        · exact row_hin hd i (B := d.mean) hd.mean_len
        · exact row_img_inj (d := d) (i : Nat)
        · exact hvmean
        · exact (Buf.elems_mono d.mean (img_row_sub_mRows (i : Nat))).trans (mean_elems_sub_W d (i : Nat))
        · intro m2 hld2 hfr2 hpr2
          have hp2 : ∀ x, init x ≠ none → m2 x ≠ none := hpr2
          have hvrstd2 : d.rstd.Valid m2 := valid_of_present hvrstd hp2
          refine safe_subc ?_
          intro bxh hbxh
          refine safe_mul ?_
          intro bsq1 hbsq1
          refine safe_sum1c (bd_pos hd) ?_
          intro bvar hbvar
          refine safe_div (hex := fun j => hdivt _ _) ?_
          intro bva hbva
          refine safe_addc ?_
          intro be hbe
          refine safe_math (fn := .sqrt) (hex := fun j => hsqrt_t _) ?_
          intro bsq hbsq
          refine safe_div (hex := fun j => hdivt _ _) ?_
          intro brstd hbrstd
          refine safe_store_buf (b := d.rstd)
            (g := fun (ix : Idx [32]) => 32 * (i : Nat) + (ix.1 : Nat))
            (en := fun (ix : Idx [32]) => decide (32 * (i : Nat) + (ix.1 : Nat) < d.T)) ?_ ?_ ?_ ?_ ?_ ?_
          · exact row_addrs d.rstd (i : Nat) hp hot (fun r => mT_get hT31 r (hot r) hT')
          · exact row_hin hd i (B := d.rstd) hd.rstd_len
          · exact row_img_inj (d := d) (i : Nat)
          · exact hvrstd2
          · exact rstd_elems_sub_W_sdiff0 hd (i : Nat)
              (img_row_sub_mRows (i : Nat)) (img_row_sub_mRows (i : Nat))
          · intro m3 hld3 hfr3 hpr3
            have hp3 : ∀ x, init x ≠ none → m3 x ≠ none := fun x hx => hpr3 x (hp2 x hx)
            have hvw3 : d.w.Valid m3 := valid_of_present hvw hp3
            have hvb3 : d.b.Valid m3 := valid_of_present hvb hp3
            have hvy3 : d.y.Valid m3 := valid_of_present hvy hp3
            refine safe_load_buf (b := d.w)
              (g := fun (ix : Idx [32, d.BD]) =>
                gidx d (32 * (i : Nat) + (ix.1 : Nat)) (ix.2.1 : Nat))
              (en := fun (ix : Idx [32, d.BD]) => decide ((ix.2.1 : Nat) < d.D)) ?_ ?_ ?_ ?_ ?_
            · exact vec_addrs hd i d.w hp
                (fun r => oG_toInt hGpos hG2 (hot r) (hog (r, ()))) hod
                (fun j => mD_get hD31 j (hod j))
            · exact vec_hin hd i (B := d.w) hd.w_len
            · exact hvw3
            · exact (Buf.elems_mono d.w (img_vec_sub_wRows (i : Nat))).trans
                (w_elems_sub_R d i)
            · intro vw hlw hothw
              refine safe_cvt ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)),
                Or.inl rfl⟩ ?_
              intro bw hbw
              refine safe_load_buf (b := d.b)
                (g := fun (ix : Idx [32, d.BD]) =>
                  gidx d (32 * (i : Nat) + (ix.1 : Nat)) (ix.2.1 : Nat))
                (en := fun (ix : Idx [32, d.BD]) => decide ((ix.2.1 : Nat) < d.D)) ?_ ?_ ?_ ?_ ?_
              · exact vec_addrs hd i d.b hp
                  (fun r => oG_toInt hGpos hG2 (hot r) (hog (r, ()))) hod
                  (fun j => mD_get hD31 j (hod j))
              · exact vec_hin hd i (B := d.b) hd.b_len
              · exact hvb3
              · exact (Buf.elems_mono d.b (img_vec_sub_wRows (i : Nat))).trans
                  (b_elems_sub_R d i)
              · intro vb hlb hothb
                refine safe_cvt ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)),
                  Or.inl rfl⟩ ?_
                intro bb hbb
                refine safe_subc ?_
                intro bx2 hbx2
                refine safe_mul ?_
                intro bhat hhat
                refine safe_mul ?_
                intro by0 hy0
                refine safe_addc ?_
                intro byy hyy
                refine safe_cvt ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl),
                  Or.inl rfl⟩ ?_
                intro by16 hy16
                refine safe_last ?_
                refine safe_store_buf (b := d.y)
                  (g := fun (ix : Idx [32, d.BD]) =>
                    idx d (32 * (i : Nat) + (ix.1 : Nat)) (ix.2.1 : Nat))
                  (en := fun (ix : Idx [32, d.BD]) =>
                    decide (32 * (i : Nat) + (ix.1 : Nat) < d.T ∧ (ix.2.1 : Nat) < d.D))
                  ?_ ?_ ?_ ?_ ?_ ?_
                · exact mat_addrs hd i d.y hp hot hod (fun r => mT_get hT31 r (hot r) hT')
                    (fun j => mD_get hD31 j (hod j))
                · exact mat_hin hd i (B := d.y) hd.y_len
                · exact mat_img_inj (d := d) i
                · exact hvy3
                · exact y_elems_sub_W_sdiff0 hd (i : Nat)
                    (img_row_sub_mRows (i : Nat)) (img_row_sub_mRows (i : Nat))
                    (img_mat_sub_yRows (i : Nat))
                · intro mF hldF hfrF hprF
                  have hpF : ∀ x, init x ≠ none → mF x ≠ none := fun x hx => hprF x (hp3 x hx)
                  have hvmeanF : d.mean.Valid mF := valid_of_present hvmean hpF
                  have hvrstdF : d.rstd.Valid mF := valid_of_present hvrstd hpF
                  have hvyF : d.y.Valid mF := valid_of_present hvy hpF
                  refine safe_ret ?_
                  intro t ht htp
                  let r : Fin 32 := ⟨t % 32, Nat.mod_lt _ (by norm_num)⟩
                  have hr : (r : Nat) = t % 32 := rfl
                  have h32 : 32 * (i : Nat) + (r : Nat) = t := by
                    rw [hr, ← htp]; exact Nat.div_add_mod t 32
                  let Sx : Set Nat := Buf.img
                    (fun ix : Idx [32, d.BD] =>
                      idx d (32 * (i : Nat) + (ix.1 : Nat)) (ix.2.1 : Nat))
                    (fun ix : Idx [32, d.BD] =>
                      decide (32 * (i : Nat) + (ix.1 : Nat) < d.T ∧ (ix.2.1 : Nat) < d.D))
                  let Sr : Set Nat := Buf.img
                    (fun ix : Idx [32] => 32 * (i : Nat) + (ix.1 : Nat))
                    (fun ix : Idx [32] => decide (32 * (i : Nat) + (ix.1 : Nat) < d.T))
                  have hmemm : t ∈ Sr := mem_img_row htp ht
                  have dYM : Disjoint (d.mean.elems Sr) (d.y.elems Sx) :=
                    elems_disjoint_mono hd.y_mean.symm
                      ((img_row_sub_mRows (i : Nat)).trans (mRows_subset_rowSet d (i : Nat)))
                      ((img_mat_sub_yRows (i : Nat)).trans (yRows_subset_matSet d (i : Nat)))
                  have dMR : Disjoint (d.mean.elems Sr) (d.rstd.elems Sr) :=
                    elems_disjoint_mono hd.mean_rstd
                      ((img_row_sub_mRows (i : Nat)).trans (mRows_subset_rowSet d (i : Nat)))
                      ((img_row_sub_mRows (i : Nat)).trans (mRows_subset_rowSet d (i : Nat)))
                  have hrdm0 : mF.read .f32 (d.mean.ptr (t : Int)).addr
                      = m3.read .f32 (d.mean.ptr (t : Int)).addr :=
                    read_frame (B := d.mean) (S := Sr) hmemm (C := d.y) (T := Sx) dYM hfrF
                  have hrdm1 : m3.read .f32 (d.mean.ptr (t : Int)).addr
                      = m2.read .f32 (d.mean.ptr (t : Int)).addr :=
                    read_frame (B := d.mean) (S := Sr) hmemm (C := d.rstd) (T := Sr) dMR hfr3
                  have hrdm2 : m2.read .f32 (d.mean.ptr (t : Int)).addr
                      = some (bmean.get (r, ())) :=
                    row_read htp ht hld2
                  have dYR : Disjoint (d.rstd.elems Sr) (d.y.elems Sx) :=
                    elems_disjoint_mono hd.y_rstd.symm
                      ((img_row_sub_mRows (i : Nat)).trans (mRows_subset_rowSet d (i : Nat)))
                      ((img_mat_sub_yRows (i : Nat)).trans (yRows_subset_matSet d (i : Nat)))
                  have hrd_r0 : mF.read .f32 (d.rstd.ptr (t : Int)).addr
                      = m3.read .f32 (d.rstd.ptr (t : Int)).addr :=
                    read_frame (B := d.rstd) (S := Sr) hmemm (C := d.y) (T := Sx) dYR hfrF
                  have hrd_r1 : m3.read .f32 (d.rstd.ptr (t : Int)).addr
                      = some (brstd.get (r, ())) :=
                    row_read htp ht hld3
                  have hxvjv : ∀ j (hj : j < d.D) (hjBD : j < d.BD),
                      xv d init t j = v.get (r, ⟨j, hjBD⟩, ()) := by
                    intro j hj hjBD
                    have h : v.get2 r ⟨j, hjBD⟩ = xv d init t j := by
                      simpa only [xv, Tensor.get2] using
                        (rval_of_read (mat_read htp ht hj hjBD hld)).symm
                    simpa only [Tensor.get2] using h.symm
                  have hbdd : ∀ j, j < d.D → |sR d init t j| ≤ (2 : ℝ) ^ 57 := by
                    intro j hj
                    have hok := (hInOk t ht j hj).1
                    have h1 : |(xv d init t j).toReal| ≤ (2 : ℝ) ^ 56 := by
                      unfold FloatV.toReal
                      exact_mod_cast hok.2
                    unfold sR sv
                    rw [ite_eq_right (by simp [hres]), cvt_toReal hok.1]
                    have h2 : (2 : ℝ) ^ (56 : ℕ) ≤ (2 : ℝ) ^ 57 := by norm_num
                    linarith
                  have hdec_all : ∀ j, j < d.BD → ¬ j < d.D →
                      decide (32 * (i : Nat) + (r : Nat) < d.T ∧ j < d.D) = false := by
                    intro j hjBD hj
                    have h32 : 32 * (i : Nat) + (r : Nat) = t := by
                      rw [hr, ← htp]; exact Nat.div_add_mod t 32
                    rw [h32]
                    exact decide_eq_false (fun hc => hj hc.2)
                  have hzero_lane : ∀ (zz : Tensor [32, d.BD] BF16)
                      (hzz : ∀ ix : Idx [32, d.BD],
                        ConvRel ((launch d).env P).P .f32 .bf16
                          ((tl.full [32, d.BD] (FloatV.ofPat (0 : ℕ)).litF32).get ix)
                          (zz.get ix)) (ix : Idx [32, d.BD]),
                      (zz.get ix).toReal = 0 ∧ (zz.get ix).isFinite = true := by
                    intro zz hzz ix
                    refine ConvRel.f32_bf16_zero (P := ((launch d).env P).P) ?_
                    simpa only [tl.full, Tensor.full, FloatV.litF32_ofPat_zero] using hzz ix
                  have hfin : ∀ j, j < d.BD →
                      (if h : j < d.BD then bx0.get (r, ⟨j, h⟩, ()) else default).isFinite
                        = true := by
                    intro j hjBD
                    rw [dite_eq_left hjBD]
                    by_cases hj : j < d.D
                    · rw [naneq_cvt_eq (hbx0 (r, ⟨j, hjBD⟩, ())) (by
                          rw [← hxvjv j hj hjBD]; exact (hInOk t ht j hj).1.1)]
                      exact cvt_isFinite (by
                        rw [← hxvjv j hj hjBD]; exact (hInOk t ht j hj).1.1)
                    · obtain ⟨hzr, hzf⟩ := hzero_lane z hz (r, ⟨j, hjBD⟩, ())
                      have hdec := hdec_all j hjBD hj
                      have hz0 : v.get (r, ⟨j, hjBD⟩, ()) = z.get (r, ⟨j, hjBD⟩, ()) :=
                        hoth (r, ⟨j, hjBD⟩, ()) hdec z rfl
                      have hvfin : (v.get (r, ⟨j, hjBD⟩, ())).isFinite = true := by
                        rw [hz0]; exact hzf
                      rw [naneq_cvt_eq (hbx0 (r, ⟨j, hjBD⟩, ())) hvfin, hz0]
                      exact cvt_isFinite hzf
                  have hval : ∀ j, j < d.D →
                      (if h : j < d.BD then bx0.get (r, ⟨j, h⟩, ()) else default).toReal
                        = sR d init t j := by
                    intro j hj
                    have hjBD : j < d.BD := lt_of_lt_of_le hj hd.d_le_bd
                    rw [dite_eq_left hjBD]
                    have hf : (v.get (r, ⟨j, hjBD⟩, ())).isFinite = true := by
                      rw [← hxvjv j hj hjBD]; exact (hInOk t ht j hj).1.1
                    rw [naneq_cvt_eq (hbx0 (r, ⟨j, hjBD⟩, ())) hf, cvt_toReal hf,
                      ← hxvjv j hj hjBD]
                    unfold sR sv
                    rw [ite_eq_right (by simp [hres]), cvt_toReal ((hInOk t ht j hj).1.1)]
                  have hzero : ∀ j, j < d.BD → d.D ≤ j →
                      (if h : j < d.BD then bx0.get (r, ⟨j, h⟩, ()) else default).toReal
                        = 0 := by
                    intro j hjBD hDj
                    rw [dite_eq_left hjBD]
                    obtain ⟨hzr, hzf⟩ := hzero_lane z hz (r, ⟨j, hjBD⟩, ())
                    have hdec := hdec_all j hjBD (by omega)
                    have hz0 : v.get (r, ⟨j, hjBD⟩, ()) = z.get (r, ⟨j, hjBD⟩, ()) :=
                      hoth (r, ⟨j, hjBD⟩, ()) hdec z rfl
                    have hvfin : (v.get (r, ⟨j, hjBD⟩, ())).isFinite = true := by
                      rw [hz0]; exact hzf
                    rw [naneq_cvt_eq (hbx0 (r, ⟨j, hjBD⟩, ())) hvfin, hz0]
                    exact cvt_zero_toReal hzf hzr
                  have hcover : ∃ tr : Tree, Tree.Covers d.BD tr ∧
                      tr.depth ≤ Nat.clog 2 d.BD ∧
                        Tree.Eval (AddRel false)
                          (fun j => if h : j < d.BD then bx0.get (r, ⟨j, h⟩, ()) else default)
                          tr (bsum.get (r, ())) := by
                    obtain ⟨tr, hcov, hdep, hev⟩ := hsum r
                    exact ⟨tr, hcov, hdep.trans (max_le (hsumD _) le_rfl), hev⟩
                  have hdvR : ((bc (FloatV.litF32 (FloatV.ofRat (d.D : ℚ))) : Tensor [32] F32).get
                      (r, ())).toReal = (d.D : ℝ) := by
                    show FloatV.toReal (FloatV.litF32 (FloatV.ofRat (d.D : ℚ))) = (d.D : ℝ)
                    exact dv_toReal hd
                  have hdvF : ((bc (FloatV.litF32 (FloatV.ofRat (d.D : ℚ))) : Tensor [32] F32).get
                      (r, ())).isFinite = true := by
                    show FloatV.isFinite (FloatV.litF32 (FloatV.ofRat (d.D : ℚ))) = true
                    exact dv_isFinite hd
                  have hm := mean_tol (d := d) (init := init) (t := t) (P := P) hd hdivf
                    (bx := fun j => if h : j < d.BD then bx0.get (r, ⟨j, h⟩, ()) else default)
                    (dv := ((bc (FloatV.litF32 (FloatV.ofRat (d.D : ℚ))) : Tensor [32] F32).get (r, ())))
                    (bsum := bsum.get (r, ())) (bmean := bmean.get (r, ()))
                    hdvR hdvF hfin hval hzero hbdd hcover (hbmean (r, ()))
                  have hbmfin : (bmean.get (r, ())).isFinite = true := hm.1
                  have hbm58 : |(bmean.get (r, ())).toReal| ≤ (2 : ℝ) ^ 58 := by
                    have h2 := rstd_tol_meanTol_le (d := d) hd hbdd
                    have h3 := rstd_tol_absMean_le (d := d) (init := init) (t := t) hbdd hd.d_pos
                    have h4 := abs_meanRef_le_absMean (d := d) (init := init) (t := t) hd.d_pos
                    nlinarith [abs_nonneg (meanRef d init t), hm.2,
                      abs_sub_abs_le_abs_sub (bmean.get (r, ())).toReal (meanRef d init t)]
                  have hbm58q : |(bmean.get (r, ())).toRat| ≤ (2 : ℚ) ^ 58 := by
                    have h := hbm58
                    unfold FloatV.toReal at h
                    exact_mod_cast h
                  let x : Nat → F32 := fun j =>
                    if h : j < d.BD then
                      (tl.where (((tl.arange 0 d.BD).ltS (IntV.ofNat d.D)).row 32) bxh
                        (bc (FloatV.ofPat (0 : ℕ)).litF32)).get2 r ⟨j, h⟩
                    else default
                  let sq : Nat → F32 := fun j =>
                    if h : j < d.BD then bsq1.get2 r ⟨j, h⟩ else default
                  have hx_def : ∀ j (hj : j < d.BD), x j =
                      (tl.where (((tl.arange 0 d.BD).ltS (IntV.ofNat d.D)).row 32) bxh
                        (bc (FloatV.ofPat (0 : ℕ)).litF32)).get2 r ⟨j, hj⟩ := by
                    intro j hj
                    dsimp only [x]
                    rw [dite_eq_left hj]
                  have hsq_def : ∀ j (hj : j < d.BD), sq j = bsq1.get2 r ⟨j, hj⟩ := by
                    intro j hj
                    dsimp only [sq]
                    rw [dite_eq_left hj]
                  have hmask_t : ∀ j (hj : j < d.BD), j < d.D →
                      (((tl.arange 0 d.BD).ltS (IntV.ofNat d.D)).row 32).get2 r ⟨j, hj⟩
                        = true := by
                    intro j hj hjD
                    rw [mask_lane hD31 hBD31 r ⟨j, hj⟩]
                    exact decide_eq_true hjD
                  have hmask_f : ∀ j (hj : j < d.BD), ¬ j < d.D →
                      (((tl.arange 0 d.BD).ltS (IntV.ofNat d.D)).row 32).get2 r ⟨j, hj⟩
                        = false := by
                    intro j hj hjD
                    rw [mask_lane hD31 hBD31 r ⟨j, hj⟩]
                    exact decide_eq_false_iff_not.mpr hjD
                  have hwhere_t : ∀ j (hj : j < d.BD), j < d.D →
                      (tl.where (((tl.arange 0 d.BD).ltS (IntV.ofNat d.D)).row 32) bxh
                        (bc (FloatV.ofPat (0 : ℕ)).litF32)).get2 r ⟨j, hj⟩
                        = bxh.get2 r ⟨j, hj⟩ := by
                    intro j hj hjD
                    rw [where_lane
                      (c := ((tl.arange 0 d.BD).ltS (IntV.ofNat d.D)).row 32)
                      (x := bxh) (y := (bc (FloatV.ofPat (0 : ℕ)).litF32)) r ⟨j, hj⟩,
                      hmask_t j hj hjD]
                    exact ite_eq_left rfl
                  have hwhere_f : ∀ j (hj : j < d.BD), ¬ j < d.D →
                      (tl.where (((tl.arange 0 d.BD).ltS (IntV.ofNat d.D)).row 32) bxh
                        (bc (FloatV.ofPat (0 : ℕ)).litF32)).get2 r ⟨j, hj⟩
                        = (bc (FloatV.ofPat (0 : ℕ)).litF32).get2 r ⟨j, hj⟩ := by
                    intro j hj hjD
                    rw [where_lane
                      (c := ((tl.arange 0 d.BD).ltS (IntV.ofNat d.D)).row 32)
                      (x := bxh) (y := (bc (FloatV.ofPat (0 : ℕ)).litF32)) r ⟨j, hj⟩,
                      hmask_f j hj hjD]
                    exact ite_eq_right (by simp)
                  have hbx57 : ∀ j (hj : j < d.D) (hjBD : j < d.BD),
                      |(bx0.get (r, ⟨j, hjBD⟩, ())).toRat| ≤ (2 : ℚ) ^ 57 := by
                    intro j hj hjBD
                    have hb := hval j hj
                    rw [dite_eq_left hjBD] at hb
                    have h1 : |(bx0.get (r, ⟨j, hjBD⟩, ())).toReal| ≤ (2 : ℝ) ^ 57 := by
                      rw [hb]
                      exact hbdd j hj
                    unfold FloatV.toReal at h1
                    exact_mod_cast h1
                  have hbmn58 : ∀ j (hjBD : j < d.BD),
                      |(((bmean.col d.BD).get (r, ⟨j, hjBD⟩, ())).neg).toRat|
                        ≤ (2 : ℚ) ^ 58 := by
                    intro j hjBD
                    simpa only [FloatV.toRat_neg, abs_neg, Tensor.get_col] using hbm58q
                  have hdevD : ∀ j (hj : j < d.D) (hjBD : j < d.BD),
                      AddRel ((launch d).env P).fusion (bx0.get (r, ⟨j, hjBD⟩, ()))
                        (((bmean.col d.BD).get (r, ⟨j, hjBD⟩, ())).neg)
                        (bxh.get (r, ⟨j, hjBD⟩, ())) := by
                    intro j hj hjBD
                    exact hbxh (r, ⟨j, hjBD⟩, ())
                  have hfinD : ∀ j, j < d.D →
                      (if h : j < d.BD then bx0.get (r, ⟨j, h⟩, ()) else default).isFinite
                        = true :=
                    fun j hj => hfin j (lt_of_lt_of_le hj hd.d_le_bd)
                  have hxfin : ∀ j, j < d.BD → (x j).isFinite = true := by
                    intro j hj
                    rw [hx_def j hj]
                    by_cases hjD : j < d.D
                    · rw [hwhere_t j hj hjD]
                      have hfinb : (bx0.get (r, ⟨j, hj⟩, ())).isFinite = true := by
                        have h := hfin j hj
                        rw [dite_eq_left hj] at h
                        exact h
                      have hnegfin : (((bmean.col d.BD).get (r, ⟨j, hj⟩, ())).neg).isFinite
                          = true := by
                        rw [FloatV.isFinite_neg]
                        simpa only [Tensor.get_col] using hbmfin
                      have hM : |(bx0.get (r, ⟨j, hj⟩, ())).toRat|
                          + |(((bmean.col d.BD).get (r, ⟨j, hj⟩, ())).neg).toRat|
                          ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
                        have hroom : (2 : ℚ) ^ 57 + 2 ^ 58
                            ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
                          rw [Format.binary32_maxVal, Format.binary32_eta]; norm_num
                        linarith [hbx57 j hjD hj, hbmn58 j hj]
                      exact (FloatV.AddRel.err Format.binary32_regular (by decide)
                        (by rw [Format.binary32_u]; norm_num) (hdevD j hjD hj) hfinb hnegfin
                        hM).1
                    · rw [hwhere_f j hj hjD]
                      exact bc_zero_isFinite r ⟨j, hj⟩
                  have hdev : ∀ j, j < d.D → AddRel true
                      ((fun j => if h : j < d.BD then bx0.get (r, ⟨j, h⟩, ()) else default) j)
                      (FloatV.neg (bmean.get (r, ()))) (x j) := by
                    intro j hj
                    have hjBD : j < d.BD := lt_of_lt_of_le hj hd.d_le_bd
                    have hbapp : ((fun j => if h : j < d.BD then bx0.get (r, ⟨j, h⟩, ())
                        else default) j)
                        = (if h : j < d.BD then bx0.get (r, ⟨j, h⟩, ()) else default) := rfl
                    rw [hbapp, dite_eq_left hjBD, hx_def j hjBD, hwhere_t j hjBD hj]
                    exact AddRel.of_true (hbxh (r, ⟨j, hjBD⟩, ()))
                  have hzero_x : ∀ j, j < d.BD → d.D ≤ j → (x j).toReal = 0 := by
                    intro j hj hjD
                    rw [hx_def j hj, hwhere_f j hj (by omega)]
                    exact bc_zero_toReal r ⟨j, hj⟩
                  have hxbd : ∀ j, j < d.BD → |(x j).toReal| ≤ 2 ^ 58 + 2 ^ 40 := by
                    intro j hj
                    exact (rstd_tol_x_bd (d := d) hd
                      (bx := fun j => if h : j < d.BD then bx0.get (r, ⟨j, h⟩, ()) else default)
                      (bmean := bmean.get (r, ())) (x := x)
                      hbmfin hm.2 hfinD hval hbdd hxfin hdev hzero_x j hj).2
                  have hmulfin : ∀ j, j < d.BD →
                      (FloatV.mul (x j) (x j)).isFinite = true := by
                    intro j hj
                    have hxb : |(x j).toRat| ≤ (2 : ℚ) ^ 58 + 2 ^ 40 := by
                      have h := hxbd j hj
                      unfold FloatV.toReal at h
                      exact_mod_cast h
                    have hmax : |(x j).toRat * (x j).toRat| ≤ Format.binary32.maxVal := by
                      rw [abs_mul]
                      have h2 : |(x j).toRat| * |(x j).toRat|
                          ≤ ((2 : ℚ) ^ 58 + 2 ^ 40) * ((2 : ℚ) ^ 58 + 2 ^ 40) :=
                        mul_le_mul hxb hxb (abs_nonneg _) (by positivity)
                      have h3 : ((2 : ℚ) ^ 58 + 2 ^ 40) * ((2 : ℚ) ^ 58 + 2 ^ 40)
                          ≤ Format.binary32.maxVal := by
                        rw [Format.binary32_maxVal]; norm_num
                      linarith
                    exact (FloatV.mul_err Format.binary32_regular (hxfin j hj) (hxfin j hj)
                      hmax).1
                  have hsq_fl : ∀ j (hj : j < d.BD), sq j = FloatV.mul (x j) (x j) := by
                    intro j hj
                    have hfm : (FloatV.mul (x j) (x j)).isFinite = true := hmulfin j hj
                    rw [hx_def j hj] at hfm
                    rw [hsq_def j hj, hx_def j hj]
                    exact naneq_eq_of_isFinite (hbsq1 (r, ⟨j, hj⟩, ())) hfm
                  have hfsq : ∀ j, j < d.BD → (sq j).isFinite = true := by
                    intro j hj
                    rw [hsq_fl j hj]
                    exact hmulfin j hj
                  have hsq : ∀ j, j < d.BD → FloatV.NaNEq (FloatV.mul (x j) (x j)) (sq j) := by
                    intro j hj
                    rw [hsq_fl j hj]
                    exact FloatV.NaNEq.refl _
                  have hcover_x : ∃ tr : Tree, Tree.Covers d.BD tr ∧
                      tr.depth ≤ Nat.clog 2 d.BD ∧
                        Tree.Eval (AddRel true) sq tr (bvar.get1 r) := by
                    obtain ⟨tr, hcov, hdep, hev⟩ := hbvar r
                    exact ⟨tr, hcov, hdep.trans (max_le (hsumD _) le_rfl),
                      Tree.Eval.mono (fun a b c h => AddRel.of_true h) hev⟩
                  have hrt := rstd_tol (d := d) (init := init) (t := t) (P := P)
                    (hdivf := hdivf) (hsqrt := hsqrt)
                    (bx := fun j => if h : j < d.BD then bx0.get (r, ⟨j, h⟩, ()) else default)
                    (bmean := bmean.get (r, ()))
                    (dv := ((bc (FloatV.litF32 (FloatV.ofRat (d.D : ℚ))) : Tensor [32] F32).get
                      (r, ())))
                    (epsV := ((bc (FloatV.hostF32 d.eps) : Tensor [32] F32).get (r, ())))
                    (one := FloatV.litF32 (FloatV.ofPat (0x3FF0000000000000 : ℕ)))
                    (x := x) (sq := sq) (bvar := bvar.get1 r) (bva := bva.get (r, ()))
                    (be := be.get (r, ())) (bsq := bsq.get (r, ())) (brstd := brstd.get (r, ()))
                    hd hdvR hdvF one_toReal one_isFinite hbmfin hm.2 hfinD hval hbdd hxfin hdev
                    hzero_x hfsq hsq hcover_x (hbva (r, ())) (AddRel.of_true (hbe (r, ())))
                    (epsV_isFinite hd) (epsV_ge hd) (epsV_le hd) (epsV_err hd)
                    (hbsq (r, ())) (hbrstd (r, ()))
                  refine ⟨?_, ?_, ?_, ?_⟩
                  · -- (a) the residual store
                    intro hcon
                    rw [hres] at hcon
                    exact absurd hcon (by simp)
                  · -- (b) the mean
                    exact ⟨bmean.get (r, ()), by rw [hrdm0, hrdm1]; exact hrdm2, hm.1, hm.2⟩
                  · -- (c) the reciprocal standard deviation
                    refine ⟨brstd.get (r, ()), ?_, hrt.1, hrt.2⟩
                    rw [hrd_r0, hrd_r1]
                  · -- (d) the output element
                    intro j hj
                    have hjBD : j < d.BD := lt_of_lt_of_le hj hd.d_le_bd
                    let jj : Fin d.BD := ⟨j, hjBD⟩
                    have hwmem : gidx d (32 * (i : Nat) + (r : Nat)) j ∈ wRows d i :=
                      ⟨(r : Nat), r.isLt, j, hj, rfl⟩
                    have hrstdsub : d.rstd.elems Sr ⊆ W d i :=
                      (d.rstd.elems_mono (img_row_sub_mRows (i : Nat))).trans
                        (rstd_elems_sub_W d i)
                    have hmeansub : d.mean.elems Sr ⊆ W d i :=
                      (d.mean.elems_mono (img_row_sub_mRows (i : Nat))).trans
                        (mean_elems_sub_W d i)
                    have dW_rstd : Disjoint (d.w.elems (wRows d i)) (d.rstd.elems Sr) :=
                      disj_sub (R_disjoint_W hd i i) (w_elems_sub_R d i) hrstdsub
                    have dW_mean : Disjoint (d.w.elems (wRows d i)) (d.mean.elems Sr) :=
                      disj_sub (R_disjoint_W hd i i) (w_elems_sub_R d i) hmeansub
                    have dB_rstd : Disjoint (d.b.elems (wRows d i)) (d.rstd.elems Sr) :=
                      disj_sub (R_disjoint_W hd i i) (b_elems_sub_R d i) hrstdsub
                    have dB_mean : Disjoint (d.b.elems (wRows d i)) (d.mean.elems Sr) :=
                      disj_sub (R_disjoint_W hd i i) (b_elems_sub_R d i) hmeansub
                    have hwR : m3.read .bf16
                        (d.w.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr
                        = m2.read .bf16
                        (d.w.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr :=
                      read_frame (B := d.w) (S := wRows d i) hwmem (C := d.rstd) (T := Sr)
                        dW_rstd hfr3
                    have hwM : m2.read .bf16
                        (d.w.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr
                        = init.read .bf16
                        (d.w.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr :=
                      read_frame (B := d.w) (S := wRows d i) hwmem (C := d.mean) (T := Sr)
                        dW_mean hfr2
                    have hwL : m3.read .bf16
                        (d.w.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr
                        = some (vw.get (r, jj, ())) :=
                      hlw (r, jj, ()) (by simpa [jj] using hj)
                    have hwread : init.read .bf16 (d.w.ptr (↑(gidx d t j) : Int)).addr
                        = some (vw.get (r, jj, ())) := by
                      rw [← h32]
                      exact hwM.symm.trans (hwR.symm.trans hwL)
                    have hbR : m3.read .bf16
                        (d.b.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr
                        = m2.read .bf16
                        (d.b.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr :=
                      read_frame (B := d.b) (S := wRows d i) hwmem (C := d.rstd) (T := Sr)
                        dB_rstd hfr3
                    have hbM : m2.read .bf16
                        (d.b.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr
                        = init.read .bf16
                        (d.b.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr :=
                      read_frame (B := d.b) (S := wRows d i) hwmem (C := d.mean) (T := Sr)
                        dB_mean hfr2
                    have hbL : m3.read .bf16
                        (d.b.ptr (↑(gidx d (32 * (i : Nat) + (r : Nat)) (jj : Nat)) : Int)).addr
                        = some (vb.get (r, jj, ())) :=
                      hlb (r, jj, ()) (by simpa [jj] using hj)
                    have hbread : init.read .bf16 (d.b.ptr (↑(gidx d t j) : Int)).addr
                        = some (vb.get (r, jj, ())) := by
                      rw [← h32]
                      exact hbM.symm.trans (hbR.symm.trans hbL)
                    have hwv : vw.get (r, jj, ()) = wv d init t j := by
                      have h : wv d init t j = vw.get (r, jj, ()) := by
                        simpa only [wv] using rval_of_read (p := d.w) hwread
                      exact h.symm
                    have hwmemt : gidx d t j ∈ wRows d i := by
                      rw [← h32]
                      exact hwmem
                    have hwok : InOkW (wv d init t j) :=
                      (hWB (gidx d t j) (wRows_subset_vecSet hd i hwmemt)).1
                    have hwvfin : (wv d init t j).isFinite = true := hwok.1
                    have hbwl : bw.get (r, jj, ()) = FloatV.cvt (wv d init t j) := by
                      have hvfin : (vw.get (r, jj, ())).isFinite = true := by
                        rw [hwv]; exact hwvfin
                      have hcvt : (FloatV.cvtWith .rn (vw.get (r, jj, ())) : F32).isFinite
                          = true := by
                        rw [FloatV.cvtWith_rn]; exact (cvt_bf16_f32_spec hvfin).1
                      have h := naneq_eq_of_isFinite (hbw (r, jj, ())) hcvt
                      rw [h, FloatV.cvtWith_rn, hwv]
                    have hbwfin : (bw.get (r, jj, ())).isFinite = true := by
                      rw [hbwl]; exact (cvt_bf16_f32_spec hwvfin).1
                    have hbwv : (bw.get (r, jj, ())).toReal = (wv d init t j).toReal := by
                      rw [hbwl]
                      unfold FloatV.toReal
                      exact_mod_cast (cvt_bf16_f32_spec hwvfin).2
                    have hbwbd : |(bw.get (r, jj, ())).toReal| ≤ (2 : ℝ) ^ 16 := by
                      rw [hbwv]
                      have h := hwok.2
                      unfold FloatV.toReal
                      exact_mod_cast h
                    have hbv : vb.get (r, jj, ()) = bv d init t j := by
                      have h : bv d init t j = vb.get (r, jj, ()) := by
                        simpa only [bv] using rval_of_read (p := d.b) hbread
                      exact h.symm
                    have hbok : InOkW (bv d init t j) :=
                      (hWB (gidx d t j) (wRows_subset_vecSet hd i hwmemt)).2
                    have hbvfin : (bv d init t j).isFinite = true := hbok.1
                    have hbbl : bb.get (r, jj, ()) = FloatV.cvt (bv d init t j) := by
                      have hvfin : (vb.get (r, jj, ())).isFinite = true := by
                        rw [hbv]; exact hbvfin
                      have hcvt : (FloatV.cvtWith .rn (vb.get (r, jj, ())) : F32).isFinite
                          = true := by
                        rw [FloatV.cvtWith_rn]; exact (cvt_bf16_f32_spec hvfin).1
                      have h := naneq_eq_of_isFinite (hbb (r, jj, ())) hcvt
                      rw [h, FloatV.cvtWith_rn, hbv]
                    have hbbfin : (bb.get (r, jj, ())).isFinite = true := by
                      rw [hbbl]; exact (cvt_bf16_f32_spec hbvfin).1
                    have hbbv : (bb.get (r, jj, ())).toReal = (bv d init t j).toReal := by
                      rw [hbbl]
                      unfold FloatV.toReal
                      exact_mod_cast (cvt_bf16_f32_spec hbvfin).2
                    have hbbbd : |(bb.get (r, jj, ())).toReal| ≤ (2 : ℝ) ^ 16 := by
                      rw [hbbv]
                      have h := hbok.2
                      unfold FloatV.toReal
                      exact_mod_cast h
                    have hfinb : (bx0.get (r, jj, ())).isFinite = true := by
                      have h := hfin j hjBD
                      rw [dite_eq_left hjBD] at h
                      exact h
                    have hnegfin : (((bmean.col d.BD).get (r, jj, ())).neg).isFinite
                        = true := by
                      rw [FloatV.isFinite_neg]
                      simpa only [Tensor.get_col] using hbmfin
                    have hbxj57 : |(bx0.get (r, jj, ())).toRat| ≤ (2 : ℚ) ^ 57 :=
                      hbx57 j hj hjBD
                    have hbmnj58 : |(((bmean.col d.BD).get (r, jj, ())).neg).toRat|
                        ≤ (2 : ℚ) ^ 58 := hbmn58 j hjBD
                    have hM : |(bx0.get (r, jj, ())).toRat|
                        + |(((bmean.col d.BD).get (r, jj, ())).neg).toRat|
                        ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
                      have hroom : (2 : ℚ) ^ 57 + 2 ^ 58
                          ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
                        rw [Format.binary32_maxVal, Format.binary32_eta]; norm_num
                      linarith [hbxj57, hbmnj58]
                    have hbx2fin : (bx2.get (r, jj, ())).isFinite = true :=
                      (FloatV.AddRel.err Format.binary32_regular (by decide)
                        (by rw [Format.binary32_u]; norm_num) (hbx2 (r, jj, ())) hfinb hnegfin
                        hM).1
                    have hdevD2 : AddRel true (bx0.get (r, jj, ()))
                        (FloatV.neg (bmean.get (r, ()))) (bx2.get (r, jj, ())) := by
                      have h := hbx2 (r, jj, ())
                      simp only [Tensor.get_col] at h
                      exact AddRel.of_true h
                    have hvalj : (bx0.get (r, jj, ())).toReal = sR d init t j := by
                      have h := hval j hj
                      rw [dite_eq_left hjBD] at h
                      exact h
                    have hhat1 : FloatV.NaNEq (FloatV.mul (bx2.get (r, jj, ()))
                        (brstd.get (r, ()))) (bhat.get (r, jj, ())) := by
                      have h := hhat (r, jj, ())
                      simpa only [Tensor.get_col] using h
                    obtain ⟨hbhatfin, hby0fin, hbhatbd, hby0bd⟩ :=
                      y_hat_by0_bd (d := d) hd (init := init) (t := t) (j := j) hj
                        (bx0 := bx0.get (r, jj, ())) (bmean := bmean.get (r, ()))
                        (bx := bx2.get (r, jj, ())) (brstd := brstd.get (r, ()))
                        (bw := bw.get (r, jj, ())) (bhat := bhat.get (r, jj, ()))
                        (by0 := by0.get (r, jj, ()))
                        hvalj hfinb (hbdd j hj) hbmfin hm.2 hrt.1 hrt.2 hdevD2 hbx2fin
                        hhat1 hbwfin hbwbd (hy0 (r, jj, ()))
                    have hyy1 : AddRel true (by0.get (r, jj, ())) (bb.get (r, jj, ()))
                        (byy.get (r, jj, ())) := AddRel.of_true (hyy (r, jj, ()))
                    have hyt := y_tol (d := d) hd (init := init) (t := t) (j := j) hj
                      (bx0 := bx0.get (r, jj, ())) (bmean := bmean.get (r, ()))
                      (bx := bx2.get (r, jj, ())) (brstd := brstd.get (r, ()))
                      (bw := bw.get (r, jj, ())) (bb := bb.get (r, jj, ()))
                      (bhat := bhat.get (r, jj, ())) (by0 := by0.get (r, jj, ()))
                      (byy := byy.get (r, jj, ())) (by16 := by16.get (r, jj, ()))
                      hvalj hfinb (hbdd j hj) hbmfin hm.2 hrt.1 hrt.2 hdevD2 hbx2fin
                      hhat1 hbhatfin hbhatbd (hy0 (r, jj, ())) hby0fin hby0bd
                      hbwfin hbwv hbwbd hbbfin hbbv hbbbd hyy1 (hy16 (r, jj, ()))
                    exact ⟨by16.get (r, jj, ()), mat_read htp ht hj hjBD hldF, hyt.1, hyt.2⟩

end Claims.LayerNormFwd

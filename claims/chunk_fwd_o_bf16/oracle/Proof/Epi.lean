import Claim
import TVA.Proof
import Proof.EpiLane
import Proof.Lanes

/-!
# The epilogue

`safe_epilogue` (lines 171–181): the causal mask, the scores' rounding, the dot with the values,
the scale, the sum and the store. Each output lane's value comes from `epi_lane`.
-/

open TVA
namespace Claims.ChunkFwdO

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
variable {d : Inst} {g : Geo} {r : Nat} {init : Mem}

/-- The causal mask (line 171) in terms of the local rows. -/
theorem mA_get {q k v h o : Ptr .bf16} {o_t : Tensor [BT] I64} {T : I64} {o_v : Tensor [d.bk] I64}
    (hl : TailLanes d g q k v h o o_t T o_v) (i k : Fin BT) :
    ((Tensor.le (o_t.row BT) (o_t.col BT)).and (((o_t.ltS T).col BT).and ((o_t.ltS T).row BT))).get2 i k
      = (decide ((k : Nat) ≤ i) && (decide (g.rowOk i) && decide (g.rowOk k))) := by
  show (IntV.le (o_t.get1 k) (o_t.get1 i) && ((o_t.ltS T).get1 i && (o_t.ltS T).get1 k)) = _
  rw [IntV.le_eq, hl.ot, hl.ot, mt_get hl, mt_get hl]
  simp only [Nat.cast_le, Geo.rowOk]
  congr 2
  simp

/-- After the mask, `b_A` still follows the gated scores, and masked lanes are `+0`. -/
theorem awhere {q k v h o : Ptr .bf16} {o_t : Tensor [BT] I64} {T : I64} {o_v : Tensor [d.bk] I64}
    (hl : TailLanes d g q k v h o o_t T o_v) {bA : Tensor [BT, BT] F32} (hA : AGate d g r bA) :
    AGate d g r (tl.where ((Tensor.le (o_t.row BT) (o_t.col BT)).and
      (((o_t.ltS T).col BT).and ((o_t.ltS T).row BT))) bA (bc zf32)) ∧
    ∀ i k : Fin BT, ¬ k ≤ i → (tl.where ((Tensor.le (o_t.row BT) (o_t.col BT)).and
      (((o_t.ltS T).col BT).and ((o_t.ltS T).row BT))) bA (bc zf32)).get2 i k = FloatV.ofPat 0 := by
  constructor
  · intro i i' hrow hle hr
    have hrow' : g.rowOk i' := by
      have : (i' : Nat) ≤ i := hle
      unfold Geo.rowOk at *; omega
    have := hA i i' hrow hle hr
    have hm : ((Tensor.le (o_t.row BT) (o_t.col BT)).and
        (((o_t.ltS T).col BT).and ((o_t.ltS T).row BT))).get2 i i' = true := by
      rw [mA_get hl]; simp [hle, hrow, hrow']
    have hw : (tl.where ((Tensor.le (o_t.row BT) (o_t.col BT)).and
        (((o_t.ltS T).col BT).and ((o_t.ltS T).row BT))) bA (bc zf32)).get2 i i' = bA.get2 i i' := by
      have := Tensor.get_where ((Tensor.le (o_t.row BT) (o_t.col BT)).and
        (((o_t.ltS T).col BT).and ((o_t.ltS T).row BT))) bA (bc zf32) (i, i', ())
      exact this.trans (by rw [show ((Tensor.le (o_t.row BT) (o_t.col BT)).and
        (((o_t.ltS T).col BT).and ((o_t.ltS T).row BT))).get (i, i', ()) = true from hm]; rfl)
    rw [hw]; exact this
  · intro i k hk
    have hm : ((Tensor.le (o_t.row BT) (o_t.col BT)).and
        (((o_t.ltS T).col BT).and ((o_t.ltS T).row BT))).get2 i k = false := by
      rw [mA_get hl]; simp [hk]
    have := Tensor.get_where ((Tensor.le (o_t.row BT) (o_t.col BT)).and
        (((o_t.ltS T).col BT).and ((o_t.ltS T).row BT))) bA (bc zf32) (i, k, ())
    refine this.trans ?_
    rw [show ((Tensor.le (o_t.row BT) (o_t.col BT)).and
        (((o_t.ltS T).col BT).and ((o_t.ltS T).row BT))).get (i, k, ()) = false from hm]
    simp [zf32_eq]
    rfl

/-- **The epilogue** (lines 171–181). -/
theorem safe_epilogue (hP : Device E.P) (hd : d.ok) (hpre : Pre d init) (hg : Geo.Ok d g r)
    {q k v h o : Ptr .bf16} {o_t : Tensor [BT] I64} {T : I64} {o_v : Tensor [d.bk] I64}
    (hl : TailLanes d g q k v h o o_t T o_v) (hm : m = init)
    {bo : Tensor [BT, d.bk] F32} {bA : Tensor [BT, BT] F32}
    (hOG : OGate d g r bo) (hAG : AGate d g r bA) :
    Safe E (Rs d) (Wg d g) (fun _ m' => Qg d g r m') m
      (epilogue d v o o_t (o_t.ltS T) o_v bo bA) := by
  obtain ⟨hAG2, hzero⟩ := awhere hl hAG
  simp only [epilogue, Prog.bind_eq]
  have hV0 := hd.V_pos
  have hTn : 1 ≤ g.Tn := by have := hg.start_lt; omega
  have hqh' : g.ih < d.HV := hg.ih_lt
  have hHV := HV_lt16 hd
  have hBT := BT_le hd
  have hVV : d.HV * d.V ≤ 2 ^ 25 := by
    have := hd.V_le
    calc d.HV * d.V ≤ 2 ^ 16 * 512 := Nat.mul_le_mul (by omega) (by omega)
      _ = 2 ^ 25 := by norm_num
  have hHVV63 : d.HV * d.V < 2 ^ (64 - 1) := lt_of_le_of_lt hVV (by norm_num)
  have hTn35 : g.Tn ≤ 2 ^ 35 := le_trans (by have := hg.end_le; omega) hBT
  have hfit : (g.Tn - 1) * (d.HV * d.V) < 2 ^ (64 - 1) := by
    have h1 : (g.Tn - 1) * (d.HV * d.V) ≤ 2 ^ 35 * 2 ^ 25 :=
      Nat.mul_le_mul (by omega : g.Tn - 1 ≤ 2 ^ 35) hVV
    exact lt_of_le_of_lt h1 (by norm_num)
  have hV63 : d.V < 2 ^ (64 - 1) := by have := hd.V_le; omega
  have hlen : g.Tn = 0 ∨ d.V = 0 ∨ (g.bos * d.HV + g.ih) * d.V + (g.Tn - 1) * (d.HV * d.V) +
      (d.V - 1) * 1 < d.v.len := Or.inr (Or.inr (by
    rw [Nat.mul_one]; exact slab_last_lt hTn hqh' hV0 hg.end_le hd.v_len))
  refine safe_load_tileV d.v d.vv (hm ▸ hpre.v_holds) hm (ρ := fun i : Fin BT => g.it * BT + i)
    (κ := fun c : Fin d.bk => g.iv * d.bk + c) (M := g.Tn) (N := d.V)
    (e := (g.bos * d.HV + g.ih) * d.V) (s0 := d.HV * d.V) (s1 := 1) ?_
    (maskAt_colrow (a := o_t.ltS T) (b := o_v.ltS (IntV.ofNat d.V : I64)) (mt_get hl) ?_) ?_
    (hm ▸ hpre.v_valid) ?_ fun bv hbv => ?_
  · -- pointers
    refine Tile2.ptr_lane_seq_of_on d.v hl.hv
      (fun i j hi => Tile2.colMul_toInt_on hl.ot (IntV.toInt_ofNat (w := 64) hHVV63) hfit i j hi)
      (fun i j _ => Tile2.row_toInt (κ := fun c : Fin d.bk => g.iv * d.bk + c) hl.ov i j)
  · intro c
    rw [Tensor.get1_ltS, IntV.lt_eq, hl.ov c, IntV.toInt_ofNat (w := 64) hV63]
    simp only [Nat.cast_lt]
  · exact hlen
  · exact (Buf.tileAt_sub_bytes d.v hlen).trans (fun x hx => Or.inr (Or.inr (Or.inl hx)))
  have hstd32 : Format.Std .binary32 := Or.inr (Or.inr (Or.inl rfl))
  have hstdbf : Format.Std .bfloat16 := Or.inr (Or.inl rfl)
  refine safe_mul fun x1 hx1 => ?_
  refine safe_cvt ⟨hstd32, hstdbf, Or.inl rfl⟩ fun a16 ha16 => ?_
  refine safe_dot (hP.dotTotal cfgBF16 _ _ _) fun Dt hDt => ?_
  refine safe_mul fun x2 hx2 => ?_
  refine safe_addc fun xs hxs => ?_
  refine safe_cvt ⟨hstd32, hstdbf, Or.inl rfl⟩ fun o16 ho16 => ?_
  refine safe_last ?_
  have hHVV : d.V * 1 ≤ d.HV * d.V := by
    have := hd.HV_pos
    calc d.V * 1 = 1 * d.V := by ring
      _ ≤ d.HV * d.V := Nat.mul_le_mul_right _ this
  have hlenO : g.Tn = 0 ∨ d.V = 0 ∨ (g.bos * d.HV + g.ih) * d.V + (g.Tn - 1) * (d.HV * d.V) +
      (d.V - 1) * 1 < d.o.len := Or.inr (Or.inr (by
    rw [Nat.mul_one]; exact slab_last_lt hTn hqh' hV0 hg.end_le hd.o_len))
  refine safe_store_tileAt d.o (ρ := fun i : Fin BT => g.it * BT + i)
    (κ := fun c : Fin d.bk => g.iv * d.bk + c) (M := g.Tn) (N := d.V)
    (e := (g.bos * d.HV + g.ih) * d.V) (s0 := d.HV * d.V) (s1 := 1) ?_
    (maskAt_colrow (a := o_t.ltS T) (b := o_v.ltS (IntV.ofNat d.V : I64)) (mt_get hl) ?_) hlenO
    (Tile2.inj_rowMajor (by norm_num) hHVV) ?_ ?_ (hm ▸ hpre.o_valid) ?_ ?_
  · refine Tile2.ptr_lane_seq_of_on d.o hl.ho
      (fun i j hi => Tile2.colMul_toInt_on hl.ot (IntV.toInt_ofNat (w := 64) hHVV63) hfit i j hi)
      (fun i j _ => Tile2.row_toInt (κ := fun c : Fin d.bk => g.iv * d.bk + c) hl.ov i j)
  · intro c
    rw [Tensor.get1_ltS, IntV.lt_eq, hl.ov c, IntV.toInt_ofNat (w := 64) hV63]
    simp only [Nat.cast_lt]
  · intro i i' _ _ h; exact Fin.ext (by have := h; omega)
  · intro c c' _ _ h; exact Fin.ext (by have := h; omega)
  · refine d.o.elems_mono ?_
    rintro x ⟨i, c, hi, hc, rfl⟩
    refine ⟨i, c, i.isLt, c.isLt, hi, hc, ?_⟩
    unfold oIdx Geo.lo
    ring
  · intro m' hread hframe hpres
    refine safe_ret ?_
    intro i c hi hc hrow hcol
    refine ⟨o16.get2 ⟨i, hi⟩ ⟨c, hc⟩, ?_, fun hin => ?_⟩
    · have := hread ⟨i, hi⟩ ⟨c, hc⟩ hrow hcol
      have hidx : (g.bos * d.HV + g.ih) * d.V + (g.it * BT + i) * (d.HV * d.V) +
          (g.iv * d.bk + c) * 1 = oIdx d (g.lo + i) g.ih (g.iv * d.bk + c) := by
        unfold oIdx Geo.lo; ring
      rw [hidx] at this; exact this
    · have hbv' : ∀ k : Fin BT, bv.get2 k ⟨c, hc⟩ = vTile d g k c := by
        intro k
        rw [hbv k ⟨c, hc⟩]
        have hidx : (g.bos * d.HV + g.ih) * d.V + (g.it * BT + k) * (d.HV * d.V) +
            (g.iv * d.bk + c) * 1 = ((g.lo + k) * d.HV + g.ih) * d.V + (g.iv * d.bk + c) := by
          unfold Geo.lo; ring
        unfold vTile Geo.rowOk Geo.colOk
        rw [hidx]
      exact epi_lane hP hd hg (i := ⟨i, hi⟩) (c := ⟨c, hc⟩) hrow hcol hin hOG hAG2
        (hzero ⟨i, hi⟩) (fun k => ha16 (⟨i, hi⟩, k, ())) hbv' hDt
        (hx1 (⟨i, hi⟩, ⟨c, hc⟩, ())) (hx2 (⟨i, hi⟩, ⟨c, hc⟩, ())) (hxs (⟨i, hi⟩, ⟨c, hc⟩, ()))
        (ho16 (⟨i, hi⟩, ⟨c, hc⟩, ()))

end Claims.ChunkFwdO

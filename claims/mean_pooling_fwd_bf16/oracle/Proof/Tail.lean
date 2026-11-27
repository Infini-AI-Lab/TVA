import Claim
import TVA.Proof
import Proof.Basic
import Proof.Geo
import Proof.Lanes
import Proof.Value

/-!
# After the scalars: offsets, mask, load, sum, division, store

`tail_safe` walks the body from line 108 on, given the six `int64` scalars (`i_d`, `i_h`, `i_tg`,
`i_t`, `bos`, `T`) with their values: the offsets and the mask, the masked tile load
(`safe_load_tileAt`), and then `rest_safe`: the widening, the column sums, the divisor, the
division, the rounding and the store; each stored column is bounded by `lane_value`.
-/

namespace Claims.MeanPoolingFwd

open TVA

/-- `+0` of `float32` converted to `bfloat16` (the load's `other`) is `+0`. -/
theorem pad_zero {P : Platform} {r : BF16}
    (h : ConvRel P .f32 .bf16 (FloatV.litF32 (FloatV.ofPat 0)) r) : r = FloatV.ofPat 0 := by
  have hz : (FloatV.litF32 (FloatV.ofPat 0 : F64) : F32) = FloatV.ofPat 0 := by
    have h0 : (FloatV.ofPat 0 : F64).isZero = true := by decide
    simp only [FloatV.litF32, h0, ite_true]
  rw [hz] at h
  have hne : (Format.binary32 : Format) ≠ Format.bfloat16 := by decide
  simp only [ConvRel, ofFloatRel, FloatRel, hne, ite_false] at h
  rcases h with ⟨-, -, h⟩ | ⟨-, hg, -⟩ | ⟨hf, -, -⟩
  · rw [FloatV.cvt_ofPat_zero Format.binary32_regular] at h
    exact h.eq (by decide)
  · exact absurd hg (by decide)
  · exact absurd hf (by decide)

/-- **The rest of the body after the load**: the widening, the column sums, the divisor `L`
(of value `Lof`), the division, the rounding to `bfloat16` and the masked store (columns `kap l`,
pointers into `o` at `eO + kap l`) are safe and end with `Q`. -/
theorem rest_safe {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {E : Env}
    (hP : Device E.P) (hT : d.varlen = true → TablesOk d) {i : Nat} (hi : i < gx d * gy d)
    {v : Tensor [d.BT, d.BD] BF16}
    (hread : ∀ (r : Fin d.BT) (l : Fin d.BD), rho d i r < lenS d i → kap d i l < d.D →
      v.get2 r l = xAt d (start d (outRow d i) + r) (hOf d i) (kap d i l))
    (hpad : ∀ (r : Fin d.BT) (l : Fin d.BD), lenS d i ≤ rho d i r → v.get2 r l = FloatV.ofPat 0)
    {L : I64} (hL : L.toInt = (Lof d i : Int))
    {o_d : Tensor [d.BD] I64} (ho_d : ∀ l : Fin d.BD, (o_d.get1 l).toInt = ((kap d i l : Nat) : Int))
    {p_o : Tensor [d.BD] (Ptr .bf16)}
    (hp_o : ∀ l : Fin d.BD, p_o.get1 l = d.o.ptr ((eO d i + kap d i l : Nat) : Int)) :
    Safe E (R d i) (W d i) (fun _ m => Q d i m) init
      (do
        let b_x : Tensor [d.BT, d.BD] F32 ← tl.cvt v
        let b_s ← tl.sum0 b_x
        let L32 : Tensor [] F32 ← tl.itof (Tensor.scalar L)
        let b_o ← tl.div b_s (bc L32.item)
        let out : Tensor [d.BD] BF16 ← tl.cvt b_o
        tl.store p_o out (some (o_d.ltS (IntV.ofNat d.D : I64)))) := by
  obtain ⟨hL1, hLB⟩ := Lof_bounds hd hT hi
  have hov : d.o.Valid init := hpre.2.1
  refine safe_cvt (g := .binary32)
    (hok := ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩) ?_
  intro bx hbx
  refine safe_sum0 (bt_pos hd) ?_
  intro bs hbs
  refine safe_itof Format.binary32_std ?_
  intro L32 hL32
  have hL32' : tl.ItofRel .binary32 L.toInt (L32.get ()) := hL32 ()
  rw [hL] at hL32'
  obtain ⟨hL32f, hL32v⟩ := itof_exact (L := Lof d i) (by have := bt_le hd; omega) hL32'
  refine safe_div (fun l => hP.divTotal _ _ _) ?_
  intro bo hbo
  refine safe_cvt (g := .bfloat16)
    (hok := ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl), Or.inl rfl⟩) ?_
  intro out hout
  refine safe_last ?_
  -- the store
  have hmask : ∀ l : Idx [d.BD], (o_d.ltS (IntV.ofNat d.D : I64)).get l = decide (kap d i l.1 < d.D) := by
    intro l
    show (o_d.ltS (IntV.ofNat d.D : I64)).get1 l.1 = decide (kap d i l.1 < d.D)
    rw [Tensor.get1_ltS, IntV.lt_eq, i64_ofNat (by have := hd.D_le; omega), ho_d l.1]
    simp only [Nat.cast_lt]
  have hA : d.o.Addrs (tl.addrs p_o (some (o_d.ltS (IntV.ofNat d.D : I64))))
      (fun l => eO d i + kap d i l.1) (fun l => decide (kap d i l.1 < d.D)) :=
    Buf.addrs_of_get d.o hmask fun l _ => hp_o l.1
  refine safe_store_buf d.o hA ?_ ?_ hov ?_ ?_
  · intro l hl
    have h1 := eO_last hd hi
    have h2 := hd.o_len
    simp only [decide_eq_true_eq] at hl
    omega
  · intro l l' _ _ h
    have : (l.1 : Nat) = l'.1 := by unfold kap at h; omega
    exact Prod.ext (Fin.ext this) rfl
  · intro z hz
    refine d.o.elems_mono ?_ hz
    rintro e ⟨l, hl, rfl⟩
    exact ⟨l.1, of_decide_eq_true hl, rfl⟩
  · intro m' hm' _ _
    refine safe_ret ?_
    intro l hl
    refine ⟨out.get (l, ()), ?_, ?_⟩
    · have := hm' (l, ()) (decide_eq_true hl)
      simpa [eO, kap] using this
    · intro hin
      obtain ⟨t, hcov, hdepB, ht⟩ := hbs l
      have hdep : t.depth ≤ Nat.log 2 d.BT := by
        have := hP.sumDepth d.BT
        rw [← clog_bt hd]
        exact hdepB.trans (max_le this le_rfl)
      refine lane_value hd hT hi hP (kap d i l) ?_ ?_ hcov hdep ht hL32f (by rw [hL32v])
        (hbo (l, ())) (hout (l, ())) hin
      · intro r hr
        have hrB : r < d.BT := by omega
        rw [dite_eq_left hrB]
        have h1 := hbx (⟨r, hrB⟩, l, ())
        have hv := hread ⟨r, hrB⟩ l (by unfold rho; have := (rho_lt_iff i ⟨r, hrB⟩).mpr hr; exact this)
          hl
        rw [FloatV.cvtWith_rn] at h1
        show FloatV.NaNEq _ (bx.get (⟨r, hrB⟩, l, ()))
        rw [← hv]
        exact h1
      · intro r hr hrB
        rw [dite_eq_left hrB]
        have h1 := hbx (⟨r, hrB⟩, l, ())
        have hv := hpad ⟨r, hrB⟩ l
          (Nat.le_of_not_lt ((rho_lt_iff i ⟨r, hrB⟩).not.mpr (Nat.not_lt.mpr hr)))
        rw [FloatV.cvtWith_rn] at h1
        show FloatV.NaNEq _ (bx.get (⟨r, hrB⟩, l, ()))
        have h2 : v.get (⟨r, hrB⟩, l, ()) = FloatV.ofPat 0 := hv
        rw [h2, FloatV.cvt_ofPat_zero Format.bfloat16_regular] at h1
        exact h1

end Claims.MeanPoolingFwd

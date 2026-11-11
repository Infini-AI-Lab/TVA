import Claim
import TVA.Proof
import Proof.Geometry

/-!
# From the programs' results to `Post`.
-/

set_option maxRecDepth 8000

namespace Claims.PersistentMatmul

open TVA

/-- **The programs' results give `Post`.** -/
theorem post_of_tiles {d : Inst} (hd : d.ok) {init fin : Mem}
    (hloc : ∀ i : Fin (launch d).size, ∃ m, Qd d init i m ∧
      ∀ x ∈ Rd d init i ∪ Wr d init i, fin x = m x)
    (hfr : ∀ x, (∀ i : Fin (launch d).size, x ∉ Wr d init i) → fin x = init x) :
    Post d init fin := by
  refine ⟨fun i j hi hj => ?_, fun z hz => ?_⟩
  · -- the element `(i, j)` is in the tile of one trip of one program
    have hi' : i < d.M := hi
    have hj' : j < d.N := hj
    obtain ⟨t, ht, hmem⟩ := exists_tile hd hi' hj'
    obtain ⟨p, t', hp, htc, hpt⟩ := exists_trip hd ht
    rw [← hpt] at ht hmem
    have hp' : p < (launch d).size := by
      rw [launch_size]; exact hp
    obtain ⟨m, hQ, hagree⟩ := hloc ⟨p, hp'⟩
    obtain ⟨a, b, ha, hb, he⟩ := hmem
    have hrow : pm d (p + t' * d.NUM_SMS) * 128 + a < d.M := by
      have := M_eq hd; have := pm_lt hd ht; omega
    have hcol : pn d (p + t' * d.NUM_SMS) * 256 + b < d.N := by
      have := N_eq hd; have := pn_lt hd ht; omega
    obtain ⟨h1, h2⟩ := hd.c_inj i j _ _ hi hj hrow hcol he
    have hspec := hQ t' htc a b ha hb
    rw [← h1, ← h2] at hspec
    obtain ⟨c, hc, hrest⟩ := hspec
    refine ⟨c, ?_, hrest⟩
    have hcong : fin.read .f16 (d.c.ptr ↑((pm d (p + t' * d.NUM_SMS) * 128 + a) * d.stride_cm +
        (pn d (p + t' * d.NUM_SMS) * 256 + b) * d.stride_cn)).addr =
        m.read .f16 (d.c.ptr ↑((pm d (p + t' * d.NUM_SMS) * 128 + a) * d.stride_cm +
          (pn d (p + t' * d.NUM_SMS) * 256 + b) * d.stride_cn)).addr :=
      Mem.read_congr fun x hx => hagree x (Or.inr ⟨t', htc,
        Buf.span_sub_elems (b := d.c) (S := tileOf d (p + t' * d.NUM_SMS))
          (hj := ⟨a, b, ha, hb, rfl⟩) hx⟩)
    rw [he]
    exact hcong.trans (by simpa only [he] using hc)
  · exact hfr z fun i => fun hzi => hz (Wr_sub_cIdx hd hzi)

end Claims.PersistentMatmul

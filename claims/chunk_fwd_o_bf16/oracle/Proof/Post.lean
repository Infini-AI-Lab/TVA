import Claim
import TVA.Proof
import Proof.Foot

/-!
# The claim's `Post` from the programs' results

`post_of_Q`: if every program's final memory satisfies `Qg` on its tile, and bytes no program
writes are unchanged, then `Post` holds.
-/

open TVA
namespace Claims.ChunkFwdO

/-- A column lies in a value block. -/
theorem div_lt_nVB {d : Inst} (hd : d.ok) {e : Nat} (he : e < d.V) : e / d.bk < nVB d := by
  have hb := bk_pos hd
  unfold nVB
  have h1 : e / d.bk * d.bk ≤ e := Nat.div_mul_le_self _ _
  have h2 : (e / d.bk).succ * d.bk = e / d.bk * d.bk + d.bk := Nat.succ_mul _ _
  exact (Nat.le_div_iff_mul_le hb).mpr (by omega)

/-- **`Post`** from the programs' post-conditions. -/
theorem post_of_Q {d : Inst} (hd : d.ok) (ht : d.varlen = true → TablesOk d) {init : Mem}
    {fin : Mem}
    (hloc : ∀ i : Nat, i < grid0 d * (d.B * d.HV) → ∃ m : Mem, Qg d (geoOf d i) (geoOf d i).itg m ∧
      ∀ x ∈ Rs d ∪ Wg d (geoOf d i), fin x = m x)
    (hfr : ∀ x, (∀ i : Nat, i < grid0 d * (d.B * d.HV) → x ∉ Wg d (geoOf d i)) → fin x = init x) :
    Post d init fin := by
  constructor
  · intro r t j e hr ht1 ht2 hj he
    have hb := bk_pos hd
    have hv : e / d.bk < nVB d := div_lt_nVB hd he
    obtain ⟨hlt, -, -⟩ := progOf_lt hd hr hj hv
    obtain ⟨hiv, hih, hitg⟩ := geoOf_progOf hd hr hj hv
    set i := progOf d r j (e / d.bk) with hi
    obtain ⟨m, hQ, hagree⟩ := hloc i hlt
    have hg := geoOf_ok hd ht hlt
    set g := geoOf d i with hgdef
    rw [hitg] at hg hQ
    have hlo := hg.lo_eq
    have hhi := hg.hi_eq
    have hk : t - g.lo < BT := by
      simp only [BT] at *; omega
    have hrow : g.rowOk (t - g.lo) := by
      unfold Geo.rowOk; unfold Geo.lo at *; simp only [BT] at *; omega
    have hc : e % d.bk < d.bk := Nat.mod_lt _ hb
    have hcol : g.colOk d (e % d.bk) := by
      unfold Geo.colOk; rw [hiv]; have := Nat.div_add_mod' e d.bk; omega
    obtain ⟨cv, hread, hrange⟩ := hQ (t - g.lo) (e % d.bk) hk hc hrow hcol
    have ht' : g.lo + (t - g.lo) = t := by omega
    have he' : g.iv * d.bk + e % d.bk = e := by
      rw [hiv]; exact Nat.div_add_mod' e d.bk
    rw [ht', hih, he'] at hread
    rw [ht', hih, he'] at hrange
    refine ⟨cv, ?_, hrange⟩
    rw [← hread]
    refine Mem.read_congr fun x hx => hagree x (Or.inr ?_)
    refine d.o.span_sub_elems (S := tileS d g) ?_ hx
    exact ⟨t - g.lo, e % d.bk, hk, hc, hrow, hcol, by rw [ht', hih, he']⟩
  · intro z hz
    refine hfr z fun i hi hzW => hz ?_
    have hg := geoOf_ok hd ht hi
    exact d.o.elems_mono (tileS_written hg) hzW

end Claims.ChunkFwdO

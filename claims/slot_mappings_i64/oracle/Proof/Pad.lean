import Proof.Store

/-!
# `slot_mappings_i64`: the padding program

Program `(g, R)` (lines 102-106) loads `query_start_loc[R]` and stores `-1` at the elements
`(g, t)` for `query_start_loc[R] ≤ t < max_num_tokens`.
-/

namespace Claims.SlotMappings

open TVA

/-- The padding program's body (lines 102-106), in group `g`, at `batch_idx = b` (the last row,
`b = R`). -/
def padProg (d : Inst) (g b : Nat) : Prog Unit := do
  let actual_num_tokens ← tl.load1 (d.query_start_loc.ptr 0 + (IntV.ofNat b : I32))
  let _ ← tl.forRange actual_num_tokens (IntV.ofNat d.N : I32) (IntV.ofNat 1024 : I32) ()
    fun i _ => do
      let offset : Tensor [1024] I32 := i + tl.arange 0 1024
      tl.store ((d.slot_mappings.ptr 0 + (IntV.ofNat g : I32) * (IntV.ofNat d.S : I32)) + offset)
        (Tensor.castInt (tl.full [1024] (IntV.ofInt (-1) : I32)))
        (some (offset.ltS (IntV.ofNat d.N : I32)))
  return ()

/-- The stored `-1`, widened, is `PAD`. -/
theorem pad_cast : (IntV.cast (IntV.ofInt (-1) : I32) : I64) = PAD :=
  IntV.ext_toInt (by rw [toInt_cast64, toInt_PAD]; exact IntV.I32.toInt_ofInt (by norm_num))

/-- The initial memory is a memory of `Inv`. -/
theorem inv_init {d : Inst} {init : Mem} (hpre : Pre d init) : Inv d init init :=
  ⟨fun _ _ => rfl, hpre.2.2.2.2.2.2.2.2.1⟩

/-- The induction variable of trip `t` of a loop from `L ≥ 0` below `H ≤ 2^22`: the token
`L + 1024 t`. -/
theorem trip_var {lo : I32} {L H : Int} (hlo : lo.toInt = L) (hL : 0 ≤ L) (hH : H ≤ 2 ^ 22)
    {t : Nat} (ht : L + 1024 * t < H) :
    (IntV.ofInt (lo.toInt + (t : Int) * (IntV.ofNat 1024 : I32).toInt) : I32).toInt =
      ((L + 1024 * t).toNat : Nat) := by
  rw [toInt_1024, hlo, Int.toNat_of_nonneg (by omega)]
  rw [IntV.I32.toInt_ofInt (by omega)]
  ring

/-- **The padding program** is safe: it writes `-1` over its range. -/
theorem safe_padProg {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    {g : Nat} (hg : g < d.G) :
    Safe E (Rall d) (Wp d init g d.R) (fun _ m => Qp d init g d.R m) init (padProg d g d.R) := by
  have hR := hd.R_le
  have hN := hd.N_le
  have hNi : (d.N : Int) ≤ 2 ^ 22 := by exact_mod_cast hN
  have hp : d.query_start_loc.ptr 0 + (IntV.ofNat d.R : I32) = d.query_start_loc.ptr d.R :=
    ptr_zero_add _ (toInt_ofNat32 (by omega))
  have hlen : d.R < d.query_start_loc.len := by have := hd.qsl_len; omega
  obtain ⟨v, hv⟩ := Buf.read_some d.query_start_loc hpre.2.1 hlen
  have hq : v.toInt = qsl d init d.R := (rd32_of hv).symm
  obtain ⟨hL, -⟩ := qsl_range hpre (le_refl d.R)
  simp only [padProg]
  rw [hp]
  refine safe_load1 hv (fun z hz => ?_) ?_
  · exact rall_qsl (span_bytes _ hlen hz)
  have hW : Wp d init g d.R = d.slot_mappings.elems (tok d g (qsl d init d.R) d.N) := by
    simp only [Wp, hiB, lt_irrefl, ite_false]
  rw [hW]
  refine Safe.mono (safe_tokLoop hd hg hq (toInt_ofNat32 (by omega)) hL (le_refl _)
    (fun _ => PAD) (fun t ht m hm => ?_) (inv_init hpre)) ?_ (le_refl _)
  · -- one trip
    have hX := trip_var hq hL hNi ht
    set X := (qsl d init d.R + 1024 * t).toNat with hXdef
    have hXc : (X : Int) = qsl d init d.R + 1024 * t := Int.toNat_of_nonneg (by omega)
    have htw : tripW d g (qsl d init d.R) d.N t =
        d.slot_mappings.elems (tok d g X (min (d.N : Int) (X + 1024))) := by
      simp only [tripW, hXc]
    rw [htw]
    refine safe_last ?_
    refine safe_store_row hd hg (le_refl _) hX (toInt_ofNat32 (by omega)) (by omega) _
      (le_refl _) hm.2 fun m' hread hfr hpres => Safe.ret ⟨fun x h1 h2 => ?_, hfr, hpres⟩
    rw [← hXc] at h1 h2
    rw [lt_min_iff] at h2
    have hl : x - X < 1024 := by omega
    have h := hread (⟨x - X, hl⟩, ()) (by push_cast; omega)
    simp only [tl.full] at h
    rw [show X + (x - X) = x by omega] at h
    exact h
  · intro _ m hm t h1 h2
    simp only [hiB, lt_irrefl, ite_false] at h2
    rw [hm t h1 h2]
    simp only [val, lt_irrefl, ite_false]

end Claims.SlotMappings

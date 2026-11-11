import Proof.Basic

/-!
# `kv_indices_i32`: The postcondition from the programs' results.
-/

namespace Claims.KvIndices

open TVA

/-- The programs' windows give `Post`: each output element is the table entry, and every other byte is unchanged. -/
theorem post_of_aux {d : Inst} (_hd : d.ok) {init : Mem} (hpre : Pre d init) {fin : Mem}
    (hloc : ∀ i : Fin (launch d).size,
      ∃ m, Q d init (i : Nat) m ∧ ∀ x ∈ R d init (i : Nat) ∪ W d init (i : Nat), fin x = m x)
    (hfr : ∀ x, (∀ i : Fin (launch d).size, x ∉ W d init (i : Nat)) → fin x = init x) :
    Post d init fin := by
  rcases hpre with ⟨hrt, hrp, hsl, hki, hko, hst, hreq, hctx, hsum, hind⟩
  constructor
  · intro b hb j hj
    have hbs : b < (launch d).size := by rw [launch_size]; exact hb
    obtain ⟨m, hQ, hag⟩ := hloc ⟨b, hbs⟩
    have hQb : Q d init b m := hQ
    have hagb : ∀ x ∈ R d init b ∪ W d init b, fin x = m x := hag
    set K : Int := rpi d init b * (d.max_context_len : Int) + sidx d init b + (j : Int) with hK
    have hReq := hreq b hb
    have hK0 : 0 ≤ K := by rw [hK]; exact (hReq.2.2.2.2 j hj).1
    have hKlen : K < (d.req_to_token.len : Int) := by rw [hK]; exact (hReq.2.2.2.2 j hj).2
    have hcast : (K.toNat : Int) = K := Int.toNat_of_nonneg hK0
    have hkn : K.toNat < d.req_to_token.len := by
      have h2 : (K.toNat : Int) < (d.req_to_token.len : Int) := by rw [hcast]; exact hKlen
      omega
    obtain ⟨v, hv⟩ := Buf.read_some d.req_to_token hrt (i := K.toNat) hkn
    have hvK : init.read .i32 (d.req_to_token.ptr K).addr = some v := by
      simpa only [hcast] using hv
    have hind0 : 0 ≤ ind d init b := (hind b (le_of_lt hb)).1
    have hj0 : (0 : Int) ≤ (j : Int) := Int.natCast_nonneg j
    have hsum0 : 0 ≤ ind d init b + (j : Int) := by linarith
    have hjeq : (((ind d init b + (j : Int)).toNat : Nat) : Int) = ind d init b + (j : Int) :=
      Int.toNat_of_nonneg hsum0
    have hfin : fin.read .i32 (d.kv_indices.ptr (ind d init b + (j : Int))).addr =
        m.read .i32 (d.kv_indices.ptr (ind d init b + (j : Int))).addr := by
      apply Mem.read_congr
      intro x hx
      refine hagb x (Or.inr ?_)
      exact ⟨(ind d init b + (j : Int)).toNat, ⟨j, hj, hjeq⟩, by simpa only [hjeq] using hx⟩
    refine ⟨v, hvK, ?_⟩
    rw [hfin, hQb j hj]
    simpa only [hK] using hvK
  · intro z hz
    refine hfr z ?_
    intro i hzw
    have hib : (i : Nat) < d.bs := Nat.lt_of_lt_of_eq i.isLt (launch_size d)
    refine hz (d.kv_indices.elems_mono (S := outElems d init (i : Nat))
      (T := kvOut d init) ?_ hzw)
    intro k hk
    obtain ⟨p, hp, hke⟩ := hk
    exact ⟨(i : Nat), hib, p, hp, hke⟩

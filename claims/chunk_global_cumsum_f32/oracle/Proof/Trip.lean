import Claim
import TVA.Proof
import Proof.Basic
import Proof.Geo
import Proof.Num
import Proof.Trees
import Proof.Lane
import Proof.LaneGood
import Proof.TripLanes

/-!
# One trip of the loop

`safe_trip` walks `tripBody` line by line: the offsets, the mask and pointers (`TripLanes`), the
masked load (each lane is the leaf value `X` of its position, `leaf_X`), the scan, the
`tl.sum`, the two additions, the scale, and the store (`store_step`), keeping `Inv`.
-/

namespace Claims.ChunkGlobalCumsum

open TVA

variable {d : Inst} {i : Nat}

/-- A trip writes bytes of `o`. -/
theorem Wt_sub_o (hd : d.ok) (hT : d.varlen = true → TablesOk d) (hi : i < gridSize d) (u : Nat) :
    Wt d i u ⊆ d.o.bytes := by
  refine Buf.elems_sub_bytes d.o ?_
  rintro e ⟨l, hl, hen, rfl⟩
  have h1 := elem_lt hd hT hi (q := loOf d i + bt d i u * d.BT + l)
    (by have := lo_add_len hT hi; omega)
  have h2 := hd.o_len
  omega

/-- What the trips wrote is not in `s` or `cu_seqlens`. -/
theorem iterW_disj_R (hd : d.ok) (hT : d.varlen = true → TablesOk d) (hi : i < gridSize d)
    (t : Nat) {x : Int} (hx : x ∈ R d) : x ∉ iterW (Wt d i) t := by
  rintro ⟨u, -, hu⟩
  have ho := Wt_sub_o hd hT hi u hu
  unfold R at hx
  rcases hx with hx | hx
  · exact Set.disjoint_left.mp hd.o_s ho hx
  · split at hx
    · rename_i hv; exact Set.disjoint_left.mp (hd.o_cu hv) ho hx
    · exact hx

/-- **The loaded lanes are the leaf values** of their positions. -/
theorem leaf_X (hd : d.ok) {init : Mem} (hpre : Pre d init) (hT : d.varlen = true → TablesOk d)
    (hi : i < gridSize d) {t : Nat} {m : Mem}
    (hfr : ∀ x, x ∉ iterW (Wt d i) t → m x = init x) {v : Tensor [d.BT] F32}
    (hread : ∀ j, enE d i t j = true →
      m.read .f32 (d.s.ptr (gE d i t j : Int)).addr = some (v.get j))
    (hdis : ∀ j, enE d i t j = false → ∀ o : Tensor [d.BT] F32,
      some (bc (FloatV.litF32 (FloatV.ofPat 0))) = some o → v.get j = o.get j)
    {j : Nat} (hj : j < d.BT) : v.get (⟨j, hj⟩, ()) = X d i (j + base d i t) := by
  by_cases he : enE d i t (⟨j, hj⟩, ()) = true
  · have hen : bt d i t * d.BT + j < lenOf d i := by unfold enE at he; simpa using he
    have hq : base d i t + j < hiOf d i := lane_hi hT hi hen
    have hlt : gE d i t (⟨j, hj⟩, ()) < d.s.len := by
      have h1 := elem_lt hd hT hi hq
      have h2 := hd.s_len
      unfold gE; simp only; omega
    have h1 := hread _ he
    have h2 : m.read .f32 (d.s.ptr (gE d i t (⟨j, hj⟩, ()) : Int)).addr =
        init.read .f32 (d.s.ptr (gE d i t (⟨j, hj⟩, ()) : Int)).addr := by
      refine Mem.read_congr (fun x hx => hfr x (iterW_disj_R hd hT hi t ?_))
      exact Or.inl (d.s.span_sub hlt hx)
    have h3 := hpre.2.2.1 _ hlt
    rw [h2, h3] at h1
    rw [← Option.some.inj h1]
    have hin : loOf d i ≤ j + base d i t ∧ j + base d i t < hiOf d i := by
      unfold base at *; constructor <;> omega
    rw [X_in hin]
    unfold sAt gE
    simp only
    rw [Nat.add_comm j]
  · have hef : enE d i t (⟨j, hj⟩, ()) = false := by simpa using he
    rw [hdis _ hef _ rfl]
    simp only [bc, litF32_zero]
    have hen : ¬ bt d i t * d.BT + j < lenOf d i := by unfold enE at hef; simpa using hef
    have := lo_add_len hT hi
    rw [X_out (by unfold base; omega)]
    rfl

/-- **The store** of a trip's values, all good on the enabled lanes, keeps `Inv` at `t + 1`. -/
theorem store_step (hd : d.ok) {init : Mem} (hpre : Pre d init) {E : Env}
    (hT : d.varlen = true → TablesOk d) (hi : i < gridSize d) {t : Nat} (ht : t < nbOf d i)
    {m : Mem} (hfr : ∀ x, x ∉ iterW (Wt d i) t → m x = init x)
    (hpres : ∀ x, init x ≠ none → m x ≠ none) (hdone : Done d i t m) {z' : F32}
    (hz' : ZTree d i (t + 1) z') {p_o : Tensor [d.BT] (Ptr .f32)} {m_t : Tensor [d.BT] Bool}
    (hmask : ∀ j, m_t.get j = enE d i t j)
    (hp : ∀ j, enE d i t j = true → p_o.get j = d.o.ptr (gE d i t j : Int))
    {vals : Tensor [d.BT] F32}
    (hval : ∀ l (hl : l < d.BT), bt d i t * d.BT + l < lenOf d i →
      Good d (seqOf d i) (base d i t + l) (headOf d i) (vals.get (⟨l, hl⟩, ()))) :
    Safe E (R d) (Wt d i t) (fun z'' m' => Inv d init i (t + 1) z'' m') m
      (tl.store p_o vals (some m_t) >>= fun _ => pure z') := by
  have hov : d.o.Valid m :=
    Buf.valid_of_agree hpre.2.1 (fun x hx => hpres x (hpre.2.1.2.2.2 x hx))
  have hin : ∀ j, enE d i t j = true → gE d i t j < d.o.len := by
    intro j hj
    have hen : bt d i t * d.BT + j.1.val < lenOf d i := by unfold enE at hj; simpa using hj
    have h1 := elem_lt hd hT hi (lane_hi hT hi hen)
    have h2 := hd.o_len
    unfold gE; omega
  refine safe_store_buf d.o (Buf.addrs_of_get d.o hmask hp) hin
    (fun x y _ _ h => gE_inj hd h) hov (d.o.elems_mono img_sub) ?_
  intro m' hwrite hframe hpres'
  refine safe_ret ⟨?_, ?_, hz', ?_⟩
  · intro x hx
    have hx1 : x ∉ d.o.elems (Buf.img (gE d i t) (enE d i t)) := by
      intro h
      exact hx ⟨t, by omega, d.o.elems_mono img_sub h⟩
    rw [hframe x hx1]
    exact hfr x (fun ⟨u, hu, hxu⟩ => hx ⟨u, by omega, hxu⟩)
  · intro x hx; exact hpres' x (hpres x hx)
  · intro u hu l hl hen
    rcases Nat.lt_or_ge u t with hut | hut
    · obtain ⟨v, hv, hg⟩ := hdone u hut l hl hen
      refine ⟨v, ?_, hg⟩
      rw [← hv]
      refine Mem.read_congr (fun x hx => hframe x ?_)
      rintro ⟨e, ⟨j, hj, rfl⟩, hxe⟩
      -- two elements of different blocks
      have hjen : bt d i t * d.BT + j.1.val < lenOf d i := by unfold enE at hj; simpa using hj
      have hne : bt d i u ≠ bt d i t := by unfold bt; split <;> omega
      have hH := hd.H_pos
      have hneq : (base d i u + l) * d.H + headOf d i ≠ gE d i t j := by
        intro heq
        unfold gE at heq
        have h1 : (base d i u + l) * d.H = (base d i t + j.1.val) * d.H := by omega
        have h2 := Nat.eq_of_mul_eq_mul_right hH h1
        unfold base at h2
        have hj' := j.1.isLt
        have : bt d i u * d.BT + l = bt d i t * d.BT + j.1.val := by omega
        have k1 : (bt d i u * d.BT + l) / d.BT = bt d i u := by
          rw [Nat.add_comm, Nat.add_mul_div_right _ _ (bt_pos hd), Nat.div_eq_of_lt hl,
            Nat.zero_add]
        have k2 : (bt d i t * d.BT + j.1.val) / d.BT = bt d i t := by
          rw [Nat.add_comm, Nat.add_mul_div_right _ _ (bt_pos hd), Nat.div_eq_of_lt hj',
            Nat.zero_add]
        rw [this, k2] at k1
        exact hne k1.symm
      simp only [Mem.span, Set.mem_ofPred_eq, Buf.ptr_addr] at hx hxe
      have hb4 : (DType.f32.bytes : Int) = 4 := rfl
      rw [hb4] at hx hxe
      push_cast at hx hxe
      have : ((base d i u + l) * d.H + headOf d i : Int) ≠ (gE d i t j : Int) := by
        exact_mod_cast hneq
      omega
    · have hut' : u = t := by omega
      subst hut'
      refine ⟨vals.get (⟨l, hl⟩, ()), ?_, hval l hl hen⟩
      exact hwrite (⟨l, hl⟩, ()) (by unfold enE; simpa using hen)

end Claims.ChunkGlobalCumsum

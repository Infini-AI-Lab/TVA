import Proof.Lane

/-!
# `slot_mappings_i64`: one trip of a request's loop

Trip `t` of program `(g, b)` (lines 122-154) covers the tokens `X, …, X + 1023` (`X = qsl b +
1024 t`) below `qsl (b + 1)`: it loads their positions, computes the local positions
(`safe_cpSec`), loads the block numbers of the lanes that need one, and stores each token's slot.
-/

namespace Claims.SlotMappings

open TVA

/-- The body of the loop of lines 121-154, after the scalars of lines 108-120: the block table
`bt_ptr`, its row stride, the block size `kv`, the flags, the request's row `req` and
`is_real_req`, the end of its tokens `end_idx`; `i` is the loop variable. -/
def tripProg (d : Inst) (g : Nat) (bt_ptr : Ptr .i32) (stride : I64) (kv : I32)
    (en sharded : Bool) (req : I64) (real : Bool) (end_idx i : I32) : Prog Unit := do
  let offset : Tensor [1024] I32 := i + tl.arange 0 1024
  let positions : Tensor [1024] I64 ← tl.load (d.positions.ptr 0 + offset)
    (some (offset.ltS end_idx)) (some (Tensor.castInt (tl.full [1024] (IntV.ofInt 0 : I32))))
  let bs64 : I64 := IntV.cast kv
  let (local_positions, is_local) ←
    cpSec d.CP_SIZE d.CP_INTERLEAVE kv bs64 sharded (IntV.ofNat d.cp_rank) positions
  let block_indices := tl.where (bc en)
    (← tl.idiv local_positions (bc bs64)) (bc (IntV.ofInt 0 : I64))
  let block_offsets ← tl.irem local_positions (bc bs64)
  let block_numbers : Tensor [1024] I32 ← tl.load
    (bt_ptr + req * stride + block_indices)
    (some (is_local.and (bc real))) (some (tl.full [1024] (IntV.ofInt 0 : I32)))
  let slot_ids : Tensor [1024] I64 :=
    Tensor.castInt (block_numbers * bc kv) + block_offsets
  let slot_ids := if d.CP_SIZE ≠ 1 then tl.where is_local slot_ids (bc PAD) else slot_ids
  let slot_ids := tl.where (bc (en && real)) slot_ids (bc PAD)
  tl.store ((d.slot_mappings.ptr 0 + (IntV.ofNat g : I32) * (IntV.ofNat d.S : I32)) + offset)
    slot_ids (some (offset.ltS end_idx))

/-- An input buffer is valid in a memory of `Inv`. -/
theorem valid_inv {d : Inst} {init m : Mem} (hm : Inv d init m) {dt : DType} {b : Buf dt}
    (hv : b.Valid init) (hdisj : Disjoint d.slot_mappings.bytes b.bytes) : b.Valid m :=
  Buf.valid_of_agree hv fun x hx => by
    rw [hm.1 x fun h => Set.disjoint_left.mp hdisj h hx]; exact hv.2.2.2 x hx

/-- A real request's row offset `idx * stride` fits `int64`, and is below the table's length. -/
theorem row_fits {d : Inst} {init : Mem} (hpre : Pre d init) {g : Nat} (hg : g < d.G)
    {b : Nat} (hb : b < d.R) (hidx : 0 ≤ idx d init b) (hlt : qsl d init b < qsl d init (b + 1)) :
    0 ≤ idx d init b * (d.bstride g : Int) ∧
      idx d init b * (d.bstride g : Int) < ((d.bt g).len : Int) ∧
      ((d.bt g).len : Int) ≤ 2 ^ 62 := by
  have hrow := hpre.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 b hb hidx hlt g hg
  have hv := (hpre.2.2.2.2.2.2.2.2.2.1 g hg).2.1
  have hb0 := (hpre.2.2.2.2.2.2.2.2.2.1 g hg).1
  simp only [DType.bytes] at hv
  refine ⟨mul_nonneg hidx (by positivity), hrow, ?_⟩
  push_cast at hv
  omega

/-- **One trip of a request's loop.** -/
theorem safe_tripProg {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    {g b : Nat} (hg : g < d.G) (hb : b < d.R) {kv : I32} (hkv : kv.toInt = d.bsz g)
    {en sharded : Bool} (hen : en = d.en g) (hsh : d.CP_SIZE ≠ 1 → sharded = d.sh g)
    {stride : I64} (hstride : stride.toInt = d.bstride g) {req : I64}
    (hreq : req.toInt = idx d init b) {real : Bool} (hreal : real = decide (0 ≤ idx d init b))
    {end_idx : I32} (hend : end_idx.toInt = qsl d init (b + 1)) {i : I32} {X : Nat}
    (hi : i.toInt = X) (hL : qsl d init b ≤ X) (hXH : (X : Int) < qsl d init (b + 1))
    {m : Mem} (hm : Inv d init m) {P : Ptr .i32} (hP : real = true → P = (d.bt g).ptr 0) :
    Safe E (Rall d) (d.slot_mappings.elems (tok d g X (min (qsl d init (b + 1)) (X + 1024))))
      (fun _ m' => (∀ x : Nat, X ≤ x → (x : Int) < min (qsl d init (b + 1)) (X + 1024) →
          m'.read .i64 (d.slot_mappings.ptr ((g * d.S + x : Nat) : Int)).addr =
            some (IntV.ofInt (slot d init g b x))) ∧
        (∀ z, z ∉ d.slot_mappings.elems (tok d g X (min (qsl d init (b + 1)) (X + 1024))) →
          m' z = m z) ∧ (∀ z, m z ≠ none → m' z ≠ none))
      m (tripProg d g P stride kv en sharded req real end_idx i) := by
  set H := qsl d init (b + 1) with hHdef
  have hN := hd.N_le
  have hHN : H ≤ d.N := (qsl_range hpre (show b + 1 ≤ d.R by omega)).2
  have hHP : H ≤ d.positions.len :=
    le_trans (qsl_mono hpre (show b + 1 ≤ d.R by omega) (le_refl _)) hpre.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hX : X ≤ 2 ^ 22 := by
    have : (d.N : Int) ≤ 2 ^ 22 := by exact_mod_cast hN
    omega
  have hLH : qsl d init b < H := lt_of_le_of_lt hL hXH
  have hB := hd.bsz_pos g hg
  have hbs : (IntV.cast kv : I64).toInt = (d.bsz g : Int) := by rw [toInt_cast64, hkv]
  have hbs0 : (0 : Int) < (IntV.cast kv : I64).toInt := by rw [hbs]; exact_mod_cast hB
  have hpv := valid_inv hm hpre.2.2.1 hd.disj_pos
  have hbtv := valid_inv hm (hpre.2.2.2.2.2.2.2.2.2.1 g hg) (hd.disj_bt g hg)
  simp only [tripProg]
  -- line 123: the positions
  let gP : Idx [1024] → Nat := fun l => X + (l.1 : Nat)
  let enP : Idx [1024] → Bool := fun l => decide ((X : Int) + (l.1 : Nat) < H)
  have hmaskP : ∀ l, ((i + tl.arange 0 1024).ltS end_idx).get l = enP l := by
    intro l; rw [Tensor.get_ltS, IntV.lt_eq, offset_lane hi hX, hend]
  have hptrP : ∀ l, enP l = true →
      (d.positions.ptr 0 + (i + tl.arange 0 1024)).get l = d.positions.ptr (gP l : Int) := by
    intro l _
    rw [Tensor.get_ptrAdd, Buf.ptr_offset, offset_lane hi hX]
    simp only [gP]; push_cast; ring_nf
  have hinP : ∀ l, enP l = true → gP l < d.positions.len := by
    intro l he
    have := of_decide_eq_true he
    have : ((gP l : Nat) : Int) < d.positions.len := by simp only [gP]; push_cast; omega
    exact_mod_cast this
  have hRP : d.positions.elems (Buf.img gP enP) ⊆ Rall d := by
    rintro z ⟨j, ⟨l, he, rfl⟩, hz⟩
    exact rall_pos (span_bytes _ (hinP l he) hz)
  refine safe_load_buf d.positions (Buf.addrs_of_get _ hmaskP hptrP) hinP hpv hRP
    fun pv hpvr hpvo => ?_
  have hposv : ∀ l, (pv.get l).toInt =
      if (X : Int) + (l.1 : Nat) < H then pos d init (X + (l.1 : Nat)) else 0 := by
    intro l
    by_cases he : (X : Int) + (l.1 : Nat) < H
    · rw [ite_eq_left he]
      have h := hpvr l (decide_eq_true he)
      rw [read_inv hm hd.disj_pos (hinP l (decide_eq_true he))] at h
      exact (rd64_of h).symm
    · rw [ite_eq_right he]
      have h := hpvo l (by simp only [enP, he, decide_false]) _ rfl
      rw [h, Tensor.get_castInt, tl.full, Tensor.get_full, toInt_cast64]
      exact IntV.I32.toInt_ofInt (by norm_num)
  -- lines 125-140
  refine safe_cpSec hd hg hkv hsh (toInt_ofNat32 hd.rank_lt) pv fun lp il hlane => ?_
  -- lines 142-143
  refine safe_idiv (fun l => div?_some _ (by rw [Tensor.get_full]; exact hbs0)) fun q hq => ?_
  refine safe_irem (fun l => rem?_some _ (by rw [Tensor.get_full]; exact hbs0)) fun bo hbo => ?_
  -- the lanes' positions, local positions and offsets
  have hpos0 : d.en g = true → 0 ≤ idx d init b → ∀ l, 0 ≤ (pv.get l).toInt := by
    intro he hidx l
    rw [hposv l]
    split_ifs with hin
    · have hx : qsl d init b ≤ ((X + (l.1 : Nat) : Nat) : Int) := by push_cast; omega
      have hx2 : ((X + (l.1 : Nat) : Nat) : Int) < qsl d init (b + 1) := by push_cast; omega
      exact (hpre.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2 g hg he b hb hidx _ hx hx2).1
    · exact le_refl _
  have hqv : ∀ l, 0 ≤ (lp.get l).toInt → (q.get l).toInt = (lp.get l).toInt / (d.bsz g : Int) := by
    intro l h
    rw [IntV.toInt_of_div? (hq l) h (by rw [Tensor.get_full]; exact hbs0), Tensor.get_full, hbs]
  have hbov : ∀ l, 0 ≤ (lp.get l).toInt →
      (bo.get l).toInt = (lp.get l).toInt % (d.bsz g : Int) := by
    intro l h
    rw [IntV.toInt_of_rem? (hbo l) h (by rw [Tensor.get_full]; exact hbs0), Tensor.get_full, hbs]
  -- lines 144-148: the block numbers
  set bi : Tensor [1024] I64 := tl.where (bc en) q (bc (IntV.ofInt 0 : I64)) with hbi
  have hbiv : ∀ l, (bi.get l).toInt = if en = true then (q.get l).toInt else 0 := by
    intro l
    simp only [hbi, tl.where, Tensor.get_where, Tensor.get_full]
    cases en
    · simp only [Bool.false_eq_true, ite_false]; exact IntV.I64.toInt_ofInt (x := 0) (by norm_num)
    · simp only [ite_true]
  let gB : Idx [1024] → Nat := fun l => (idx d init b * (d.bstride g : Int) + (bi.get l).toInt).toNat
  let enB : Idx [1024] → Bool := fun l => il.get l && real
  have hreal' : ∀ l, enB l = true → 0 ≤ idx d init b := by
    intro l he
    simp only [enB, Bool.and_eq_true, hreal, decide_eq_true_eq] at he
    exact he.2
  have hrow : 0 ≤ idx d init b → 0 ≤ idx d init b * (d.bstride g : Int) ∧
      idx d init b * (d.bstride g : Int) < ((d.bt g).len : Int) ∧ ((d.bt g).len : Int) ≤ 2 ^ 62 :=
    fun h => row_fits hpre hg hb h hLH
  -- the element a lane of a real request reads, and that it lies in the table
  have hinB' : ∀ l, enB l = true → 0 ≤ idx d init b * (d.bstride g : Int) + (bi.get l).toInt ∧
      idx d init b * (d.bstride g : Int) + (bi.get l).toInt < ((d.bt g).len : Int) := by
    intro l he
    have hidx := hreal' l he
    obtain ⟨r0, r1, -⟩ := hrow hidx
    rw [hbiv l]
    split_ifs with hen'
    · have heng : d.en g = true := by rw [← hen]; exact hen'
      have hp := hpos0 heng hidx l
      obtain ⟨hil, hlpv⟩ := hlane l hp
      obtain ⟨c0, c1⟩ := cpPos_range d g hB hd.cp_pos hd.il_pos hp
      have hlp0 : 0 ≤ (lp.get l).toInt := by rw [hlpv]; exact c0
      rw [hqv l hlp0, hlpv]
      by_cases hin : (X : Int) + (l.1 : Nat) < H
      · have hposl : (pv.get l).toInt = pos d init (X + (l.1 : Nat)) := by
          rw [hposv l, ite_eq_left hin]
        have hloc : cpLocal d g (pos d init (X + (l.1 : Nat))) := by
          have : il.get l = true := by simp only [enB, Bool.and_eq_true] at he; exact he.1
          rw [hil, hposl] at this
          exact of_decide_eq_true this
        have hx : qsl d init b ≤ ((X + (l.1 : Nat) : Nat) : Int) := by push_cast; omega
        have hx2 : ((X + (l.1 : Nat) : Nat) : Int) < qsl d init (b + 1) := by push_cast; omega
        have htok := (hpre.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2 g hg heng b hb hidx _ hx hx2).2
          ⟨heng, hidx, hloc⟩
        rw [hposl]
        have := Int.ediv_nonneg c0 (show (0 : Int) ≤ d.bsz g by positivity)
        refine ⟨by rw [← hposl]; positivity, ?_⟩
        exact htok.1
      · have hposl : (pv.get l).toInt = 0 := by rw [hposv l, ite_eq_right hin]
        rw [hposl] at c0 c1 ⊢
        have : cpPos d g 0 = 0 := le_antisymm c1 c0
        rw [this]; simp only [Int.zero_ediv, add_zero]
        exact ⟨r0, r1⟩
    · simp only [add_zero]; exact ⟨r0, r1⟩
  have hinB : ∀ l, enB l = true → gB l < (d.bt g).len := by
    intro l he
    obtain ⟨h0, h1⟩ := hinB' l he
    have : ((gB l : Nat) : Int) < (d.bt g).len := by simp only [gB]; rw [Int.toNat_of_nonneg h0]; exact h1
    exact_mod_cast this
  have hmaskB : ∀ l, (il.and (bc real)).get l = enB l := by
    intro l; simp only [Tensor.get_and, Tensor.get_full, enB]
  have hptrB : ∀ l, enB l = true →
      (P + req * stride + bi).get l = (d.bt g).ptr (gB l : Int) := by
    intro l he
    rw [hP (by simp only [enB, Bool.and_eq_true] at he; exact he.2)]
    obtain ⟨h0, -⟩ := hinB' l he
    obtain ⟨r0, r1, r2⟩ := hrow (hreal' l he)
    have hmul : (req * stride).toInt = idx d init b * (d.bstride g : Int) :=
      IntV.toInt_mul hreq hstride (IntV.I64.sfits (by omega) (by omega))
    rw [Tensor.get_ptrAdd, Ptr.add_int, Buf.ptr_offset, Buf.ptr_offset, hmul]
    simp only [gB]; rw [Int.toNat_of_nonneg h0]; ring_nf
  have hRB : (d.bt g).elems (Buf.img gB enB) ⊆ Rall d := by
    rintro z ⟨j, ⟨l, he, rfl⟩, hz⟩
    exact rall_bt hg (span_bytes _ (hinB l he) hz)
  refine safe_load_buf (d.bt g) (Buf.addrs_of_get _ hmaskB hptrB) hinB hbtv hRB
    fun bn hbnr hbno => ?_
  -- lines 149-154: the slots and the store
  refine safe_last ?_
  refine safe_store_row hd hg hHN hi hend hXH _ (le_refl _) hm.2
    fun m' hread hfr hpres => Safe.ret ⟨fun x h1 h2 => ?_, hfr, hpres⟩
  rw [lt_min_iff] at h2
  have hl : x - X < 1024 := by omega
  have h := hread (⟨x - X, hl⟩, ()) (by push_cast; omega)
  rw [show X + (x - X) = x by omega] at h
  rw [h]
  congr 1
  -- lane `x - X`'s value
  set l : Idx [1024] := (⟨x - X, hl⟩, ()) with hldef
  have hxl : X + (l.1 : Nat) = x := by simp only [hldef]; omega
  have hin : (X : Int) + (l.1 : Nat) < H := by have : ((X + (l.1 : Nat) : Nat) : Int) = x := by rw [hxl]
                                               push_cast at this; omega
  have hposl : (pv.get l).toInt = pos d init x := by rw [hposv l, ite_eq_left hin, hxl]
  have hx1 : qsl d init b ≤ (x : Int) := by omega
  have hx2 : (x : Int) < qsl d init (b + 1) := by omega
  simp only [tl.where, Tensor.get_where, Tensor.get_full, slot]
  by_cases hs : HasSlot d init g b x
  · obtain ⟨heng, hidx, hloc⟩ := hs
    have hen' : en = true := by rw [hen]; exact heng
    have hreal1 : real = true := by rw [hreal]; exact decide_eq_true hidx
    have hp := hpos0 heng hidx l
    obtain ⟨hil, hlpv⟩ := hlane l hp
    rw [hposl] at hil hlpv hp
    have hil1 : il.get l = true := by rw [hil]; exact decide_eq_true hloc
    have htok := (hpre.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2 g hg heng b hb hidx _ hx1 hx2).2
      ⟨heng, hidx, hloc⟩
    obtain ⟨-, e0, e1⟩ := htok
    obtain ⟨c0, -⟩ := cpPos_range d g hB hd.cp_pos hd.il_pos hp
    have hlp0 : 0 ≤ (lp.get l).toInt := by rw [hlpv]; exact c0
    have henB : enB l = true := by simp only [enB, hil1, hreal1, Bool.and_self]
    have hgB : gB l = (btIdx d init g b x).toNat := by
      simp only [gB, btIdx, hbiv l, hen', ite_true, hqv l hlp0, hlpv]
    have hbnv : (bn.get l).toInt = rd32 init (d.bt g) (btIdx d init g b x).toNat := by
      have hr := hbnr l henB
      rw [read_inv hm (hd.disj_bt g hg) (hinB l henB)] at hr
      rw [← hgB]; exact (rd32_of hr).symm
    set e := rd32 init (d.bt g) (btIdx d init g b x).toNat
    have hBi : (0 : Int) < d.bsz g := by exact_mod_cast hB
    have hmod0 : 0 ≤ cpPos d g (pos d init x) % (d.bsz g : Int) := Int.emod_nonneg _ hBi.ne'
    have hmod1 : cpPos d g (pos d init x) % (d.bsz g : Int) < d.bsz g := Int.emod_lt_of_pos _ hBi
    have hB31 : (d.bsz g : Int) < 2 ^ 31 := by
      have := hd.bsz_cp g hg
      have : d.bsz g ≤ d.bsz g * d.CP_SIZE := Nat.le_mul_of_pos_right _ hd.cp_pos
      have : d.bsz g < 2 ^ 31 := by omega
      exact_mod_cast this
    have hprod : (bn.get l * kv).toInt = e * (d.bsz g : Int) :=
      IntV.toInt_mul hbnv hkv (IntV.I32.sfits e0 e1)
    have hsum : ((IntV.cast (bn.get l * kv) : I64) + bo.get l).toInt =
        e * (d.bsz g : Int) + cpPos d g (pos d init x) % (d.bsz g : Int) := by
      refine IntV.toInt_add (by rw [toInt_cast64, hprod]) (by rw [hbov l hlp0, hlpv])
        (IntV.I64.sfits (by omega) (by omega))
    have hif : (if HasSlot d init g b x then e * (d.bsz g : Int) +
        cpPos d g (pos d init x) % (d.bsz g : Int) else -1) =
        e * (d.bsz g : Int) + cpPos d g (pos d init x) % (d.bsz g : Int) :=
      ite_eq_left ⟨heng, hidx, hloc⟩
    rw [hif]
    simp only [hen', hreal1, Bool.and_self, ite_true]
    apply IntV.ext_toInt
    rw [IntV.I64.toInt_ofInt (by omega)]
    split_ifs
    · simp only [Tensor.get_where, hil1, ite_true, Tensor.get_add, Tensor.get_castInt,
        Tensor.get_mul, Tensor.get_full]
      exact hsum
    · simp only [Tensor.get_add, Tensor.get_castInt, Tensor.get_mul, Tensor.get_full]
      exact hsum
  · have hif : (if HasSlot d init g b x then rd32 init (d.bt g) (btIdx d init g b x).toNat *
        (d.bsz g : Int) + cpPos d g (pos d init x) % (d.bsz g : Int) else -1) = -1 := ite_eq_right hs
    rw [hif]
    by_cases hER : en = true ∧ real = true
    · obtain ⟨hen', hreal1⟩ := hER
      have heng : d.en g = true := by rw [← hen]; exact hen'
      have hidx : 0 ≤ idx d init b := by rw [hreal] at hreal1; exact of_decide_eq_true hreal1
      have hnloc : ¬ cpLocal d g (pos d init x) := fun h => hs ⟨heng, hidx, h⟩
      have hC : d.CP_SIZE ≠ 1 := fun h => hnloc (cpLocal_of d g _ (Or.inl h))
      have hp := hpos0 heng hidx l
      obtain ⟨hil, -⟩ := hlane l hp
      rw [hposl] at hil
      have hil0 : il.get l = false := by rw [hil]; exact decide_eq_false hnloc
      simp only [hen', hreal1, Bool.and_self, ite_true, hC, ne_eq, not_false_eq_true,
        Tensor.get_where, Tensor.get_full, hil0, Bool.false_eq_true, ite_false]
      rfl
    · have : (en && real) = false := by
        cases en <;> cases real <;> first | rfl | exact absurd ⟨rfl, rfl⟩ hER
      rw [this]
      simp only [Bool.false_eq_true, ite_false]
      rfl

end Claims.SlotMappings

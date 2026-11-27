import Proof.Prog

/-!
# `token_bitmask_f32`: the block of one work item

Once `block_offset = j * 4096` and `batch_id = r` (the row of the `k`-th processed row) are known,
the loop body loads the block's `128` bitmask words, expands them to `4096` bits and stores `-∞`
where the token is below `vocab_size` and its bit is `0` (`safe_tail`).
-/

namespace Claims.TokenBitmask

open TVA

/-! ## Bits -/

/-- `(y & 1) == 0` is the negation of bit `0` of `y`. -/
theorem and_one_eq_zero (y : BitVec 32) :
    (y &&& BitVec.ofInt 32 1 == BitVec.ofInt 32 0) = !y.getLsbD 0 := by
  have h1 : BitVec.ofInt 32 1 = 1#32 := rfl
  have h0 : BitVec.ofInt 32 0 = 0#32 := rfl
  rw [h1, h0, Bool.eq_iff_iff]
  simp only [beq_iff_eq, Bool.not_eq_true']
  constructor
  · intro h
    have := congrArg (fun z => z.getLsbD 0) h
    simpa using this
  · intro h
    ext i hi
    simp only [BitVec.getElem_and, BitVec.getElem_one, BitVec.getElem_zero]
    by_cases hi0 : i = 0
    · subst hi0
      simp only [decide_true, Bool.and_true]
      rw [← BitVec.getLsbD_eq_getElem]; exact h
    · simp [hi0]

/-- `((x >> j) & 1) == 0`, the arithmetic shift by `j < 32`, is the negation of bit `j` of `x`. -/
theorem bit_eq (x : I32) {j : Nat} (hj : j < 32) :
    (IntV.and (⟨x.bits.sshiftRight j⟩ : I32) (IntV.ofInt 1)).eq (IntV.ofInt 0) =
      !x.bits.getLsbD j := by
  have := and_one_eq_zero (x.bits.sshiftRight j)
  simp only [IntV.eq, IntV.and, IntV.ofInt] at *
  rw [this, BitVec.getLsbD_sshiftRight]
  simp [hj]

/-- Lane `b` of `tl.arange(0, 32)` has bit pattern `b`. -/
theorem arange32_bits (b : Fin 32) : ((tl.arange 0 32 : Tensor [32] I32).get1 b).bits.toNat = b := by
  have hb := b.isLt
  simp only [tl.arange, Tensor.get1_arange, IntV.ofInt, BitVec.toNat_ofInt]
  omega

/-- Lane `i` of the reshape of a `[128, 32]` tensor to `[4096]` is element `(i / 32, i % 32)`. -/
theorem get_reshape4096 {α : Type} (x : Tensor [128, 32] α) (i : Fin 4096) :
    (x.reshape [4096] (by decide)).get (i, ()) =
      x.get2 ⟨i / 32, by omega⟩ ⟨i % 32, by omega⟩ := by
  unfold Tensor.reshape Tensor.get2
  apply congrArg x.get
  refine Prod.ext (Fin.ext ?_) (Prod.ext (Fin.ext ?_) rfl)
  · show (Idx.pos [4096] (i, ())) / 32 = i / 32
    simp [Idx.pos]
  · show ((Idx.pos [4096] (i, ())) % 32) / 1 = i % 32
    simp [Idx.pos]

/-- `-float("inf")` is the `float32` `-∞`. -/
theorem negInfC_eq : negInfC = (FloatV.negInf : F32) := by decide

/-! ## The block -/

/-- **The block of one work item is safe.** Item `p + t * NUM_SMS` is row `k`, block `j`;
`block_offset` has value `j * 4096` and `batch_id` the row `r` of the `k`-th processed row. The
block writes its cells (`itemElems`) and extends the invariant by one trip. -/
theorem safe_tail {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    {p t k j r : Nat} (hk : k < d.num_rows) (hj : j < nb d) (hr : (r : Int) = rowOf d init k)
    (hitem : k * nb d + j = p + t * d.NUM_SMS)
    {bo bid : I32} (hbo : bo.toInt = ((j * 4096 : Nat) : Int)) (hbid : bid.toInt = (r : Int))
    {m : Mem} (hI : Inv d init p t m) :
    Safe E (R d) (d.logits.elems (itemElems d init (p + t * d.NUM_SMS)))
      (fun _ m' => Inv d init p (t + 1) m') m (tailProg d bo bid) := by
  obtain ⟨hr0, hrL, hrB⟩ := rowOf_bounds hd hpre hk
  have hrL' : r < d.logits_rows := by rw [← hr] at hrL; exact_mod_cast hrL
  have hrB' : r < d.bitmask_rows := by rw [← hr] at hrB; exact_mod_cast hrB
  have hV := hd.vocab_lt
  have hj4096 : j * 4096 + 4096 ≤ d.vocab_size + 4095 := by unfold nb at hj; omega
  obtain ⟨hIold, hIfr, hIval⟩ := hI
  have hbm_agree : ∀ z ∈ d.bitmask.bytes, m z = init z :=
    fun z hz => hIfr z (fun hz' => Set.disjoint_left.mp hd.disj_bitmask hz' hz)
  have hbmV : d.bitmask.Valid m :=
    Buf.valid_of_agree hpre.2.1 (fun x hx => by rw [hbm_agree x hx]; exact hpre.2.1.2.2.2 x hx)
  simp only [tailProg]
  -- line 109: the block's token offsets
  have hoff : ∀ i : Fin 4096,
      ((bo + ar4096).get (i, ())).toInt = ((j * 4096 + i : Nat) : Int) :=
    fun i => IntV.I32.toInt_sadd_arange (n := 4096) hbo (by omega) i
  -- line 110: `block_offset // 32`, then the word offsets
  have h32 : (IntV.ofNat 32 : I32).toInt = 32 := IntV.I32.toInt_ofNat (by norm_num)
  refine safe_idiv1 (x := bo) (y := (IntV.ofNat 32 : I32)) (by rw [hbo]; positivity)
    (by rw [h32]; norm_num) fun q hq => ?_
  have hq' : q.item.toInt = ((j * 128 : Nat) : Int) := by
    rw [hq, hbo, h32]; push_cast; omega
  have hbof : ∀ i : Fin 128,
      ((q.item + ar128).get (i, ())).toInt = ((j * 128 + i : Nat) : Int) :=
    fun i => IntV.I32.toInt_sadd_arange (n := 128) hq' (by omega) i
  -- lines 113-116: the load of the block's words of row `r`
  have hsB : (IntV.ofNat d.bitmask_strides : I32).toInt = (d.bitmask_strides : Int) :=
    IntV.I32.toInt_ofNat hd.bs_lt
  have hbfit := hd.bitmask_fit
  have hblen := hd.bitmask_len
  have hrsB1 : r * d.bitmask_strides + d.bitmask_strides ≤ d.bitmask_rows * d.bitmask_strides := by
    rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hrB'
  have hrsB : r * d.bitmask_strides < 2 ^ 31 := by
    rcases Nat.eq_zero_or_pos d.bitmask_strides with h0 | h0
    · rw [h0, Nat.mul_zero]; norm_num
    · omega
  have hmulB : (bid * (IntV.ofNat d.bitmask_strides : I32)).toInt =
      ((r * d.bitmask_strides : Nat) : Int) := IntV.I32.mul_nat hbid hsB hrsB
  let gL : Idx [128] → Nat := fun i => r * d.bitmask_strides + (j * 128 + i.1.val)
  let enL : Idx [128] → Bool := fun i => decide (j * 128 + i.1.val < d.bitmask_strides)
  have hmaskL : ∀ i, ((q.item + ar128).ltS (IntV.ofNat d.bitmask_strides : I32)).get i =
      enL i := by
    rintro ⟨i, ⟨⟩⟩
    erw [Tensor.get_ltS, IntV.lt_eq, hbof i, hsB]
    simp only [enL, Nat.cast_lt]
  have hptrL : ∀ i, enL i = true →
      (d.bitmask.ptr 0 + bid * (IntV.ofNat d.bitmask_strides : I32) + (q.item + ar128)).get i
        = d.bitmask.ptr (gL i : Int) := by
    rintro ⟨i, ⟨⟩⟩ _
    erw [Tensor.get_ptrAdd]
    simp only [Ptr.add_int, Buf.ptr_offset, zero_add]
    erw [hmulB, hbof i]
    simp only [gL]
    push_cast
    ring_nf
  have hinL : ∀ i, enL i = true → gL i < d.bitmask.len := by
    rintro ⟨i, ⟨⟩⟩ he
    simp only [enL, decide_eq_true_eq] at he
    simp only [gL]
    omega
  refine safe_load_buf d.bitmask (Buf.addrs_of_get d.bitmask hmaskL hptrL) hinL hbmV
    (fun z hz => Or.inl (Buf.elems_sub_bytes d.bitmask
      (by rintro e ⟨i, he, rfl⟩; exact hinL i he) hz)) fun v hv _ => ?_
  -- line 117: the shifts, `& 1`, `== 0`
  refine safe_ashr (x := v.col 32) (n := (tl.arange 0 32 : Tensor [32] I32).row 128) ?_
    fun sh hsh => ?_
  · rintro ⟨a, b, ⟨⟩⟩
    show ((tl.arange 0 32 : Tensor [32] I32).get1 b).bits.toNat < 32
    rw [arange32_bits]
    exact b.isLt
  have hshv : ∀ (a : Fin 128) (b : Fin 32),
      sh.get (a, b, ()) = ⟨(v.get (a, ())).bits.sshiftRight b⟩ := by
    intro a b
    have h := hsh (a, b, ())
    have hb : ((tl.arange 0 32 : Tensor [32] I32).get1 b).bits.toNat = b := arange32_bits b
    simp only [Tensor.get_col, Tensor.get_row] at h
    simp only [Tensor.get1] at hb
    rw [hb] at h
    simp only [IntV.ashr?, b.isLt, ↓reduceIte] at h
    exact (Option.some.inj h).symm
  -- line 118: the reshape; lane `i` is bit `i % 32` of word `i / 32`
  have hbit : ∀ i : Fin 4096,
      (((Tensor.map2 IntV.and sh (bc (IntV.ofInt 1 : I32))).eq (bc (IntV.ofInt 0 : I32))).reshape
        [4096] (by decide)).get (i, ()) =
        !(v.get (⟨i / 32, by omega⟩, ())).bits.getLsbD (i % 32) := by
    have hbit2 : ∀ ii : Idx [128, 32],
        ((Tensor.map2 IntV.and sh (bc (IntV.ofInt 1 : I32))).eq (bc (IntV.ofInt 0 : I32))).get ii =
          !(v.get (ii.1, ())).bits.getLsbD ii.2.1 := by
      intro ii
      rw [Tensor.get_eq, Tensor.get_map2, bc, Tensor.get_full, Tensor.get_full]
      obtain ⟨a, b, ⟨⟩⟩ := ii
      rw [hshv a b]
      exact bit_eq _ b.isLt
    intro i
    rw [get_reshape4096]
    exact hbit2 (⟨i / 32, by omega⟩, ⟨i % 32, by omega⟩, ())
  -- the bits the kernel reads are the initial bitmask's
  have hbitInit : ∀ i : Fin 4096, j * 4096 + i < d.vocab_size →
      bitOf d init (rowOf d init k) (j * 4096 + i) =
        (v.get (⟨i / 32, by omega⟩, ())).bits.getLsbD (i % 32) := by
    intro i hc
    have hw : j * 128 + i / 32 < d.bitmask_strides := by have := hd.vocab_bs; omega
    have hen : enL (⟨i / 32, by omega⟩, ()) = true := by simp only [enL]; exact decide_eq_true hw
    have hrd := hv _ hen
    have hgL : (gL (⟨i / 32, by omega⟩, ()) : Int) =
        rowOf d init k * (d.bitmask_strides : Int) + (((j * 4096 + i) / 32 : Nat) : Int) := by
      simp only [gL]
      rw [← hr]
      have : (j * 4096 + i) / 32 = j * 128 + i / 32 := by omega
      rw [this]
      push_cast
      ring
    have hmi : m.read .i32 (d.bitmask.ptr (gL (⟨i / 32, by omega⟩, ()) : Int)).addr =
        init.read .i32 (d.bitmask.ptr (gL (⟨i / 32, by omega⟩, ()) : Int)).addr :=
      Mem.read_congr fun x hx => hbm_agree x (Buf.span_sub d.bitmask (hinL _ hen) hx)
    rw [hmi, hgL] at hrd
    unfold bitOf
    rw [hrd]
    simp only
    congr 1
    omega
  -- lines 120-124: the store of `-∞` to the block's cells of row `r`
  have hsL : (IntV.ofNat d.logits_strides : I32).toInt = (d.logits_strides : Int) :=
    IntV.I32.toInt_ofNat hd.ls_lt
  have hlfit := hd.logits_fit
  have hllen := hd.logits_len
  have hvls := hd.vocab_ls
  have hrsL1 : r * d.logits_strides + d.logits_strides ≤ d.logits_rows * d.logits_strides := by
    rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hrL'
  have hrsL : r * d.logits_strides < 2 ^ 31 := by
    rcases Nat.eq_zero_or_pos d.logits_strides with h0 | h0
    · rw [h0, Nat.mul_zero]; norm_num
    · omega
  have hmulL : (bid * (IntV.ofNat d.logits_strides : I32)).toInt =
      ((r * d.logits_strides : Nat) : Int) := IntV.I32.mul_nat hbid hsL hrsL
  have hVv : (IntV.ofNat d.vocab_size : I32).toInt = (d.vocab_size : Int) :=
    IntV.I32.toInt_ofNat (by omega)
  let gS : Idx [4096] → Nat := fun i => r * d.logits_strides + (j * 4096 + i.1.val)
  let enS : Idx [4096] → Bool := fun i => decide (j * 4096 + i.1.val < d.vocab_size) &&
    !bitOf d init (rowOf d init k) (j * 4096 + i.1.val)
  have hmaskS : ∀ i, (((bo + ar4096).ltS (IntV.ofNat d.vocab_size : I32)).and
      ((((Tensor.map2 IntV.and sh (bc (IntV.ofInt 1 : I32))).eq
        (bc (IntV.ofInt 0 : I32))).reshape [4096] (by decide)))).get i = enS i := by
    rintro ⟨i, ⟨⟩⟩
    erw [Tensor.get_and, Tensor.get_ltS, IntV.lt_eq, hoff i, hVv, hbit i]
    simp only [enS]
    by_cases hc : j * 4096 + i.val < d.vocab_size
    · rw [hbitInit i hc]
      have hc' : ((j * 4096 + i : Nat) : Int) < d.vocab_size := by exact_mod_cast hc
      rw [decide_eq_true hc', decide_eq_true hc]
    · have hc' : ¬ ((j * 4096 + i : Nat) : Int) < d.vocab_size := by exact_mod_cast hc
      rw [decide_eq_false hc', decide_eq_false hc, Bool.false_and, Bool.false_and]
  have hptrS : ∀ i, enS i = true →
      (d.logits.ptr 0 + bid * (IntV.ofNat d.logits_strides : I32) + (bo + ar4096)).get i
        = d.logits.ptr (gS i : Int) := by
    rintro ⟨i, ⟨⟩⟩ _
    erw [Tensor.get_ptrAdd]
    simp only [Ptr.add_int, Buf.ptr_offset, zero_add]
    erw [hmulL, hoff i]
    simp only [gS]
    push_cast
    ring_nf
  have henS : ∀ i, enS i = true → j * 4096 + i.1.val < d.vocab_size ∧
      bitOf d init (rowOf d init k) (j * 4096 + i.1.val) = false := by
    intro i he
    simp only [enS, Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_true'] at he
    exact he
  have hinS : ∀ i, enS i = true → gS i < d.logits.len := by
    intro i he
    have hc := (henS i he).1
    simp only [gS]
    omega
  have hinj : ∀ i i', enS i = true → enS i' = true → gS i = gS i' → i = i' := by
    rintro ⟨i, ⟨⟩⟩ ⟨i', ⟨⟩⟩ _ _ hg
    simp only [gS] at hg
    have : i = i' := Fin.ext (by omega)
    rw [this]
  have hgS : ∀ i, enS i = true → (gS i : Int) = cell d init k (j * 4096 + i.1.val) := by
    intro i _
    simp only [gS, cell]
    rw [← hr]
    push_cast
    ring
  have himg : Buf.img gS enS ⊆ itemElems d init (p + t * d.NUM_SMS) := by
    rintro e ⟨i, he, rfl⟩
    obtain ⟨hc, hb⟩ := henS i he
    refine ⟨k, j * 4096 + i.1.val, ⟨hk, hc, hb⟩, ?_, hgS i he⟩
    unfold itemOf
    have : (j * 4096 + i.1.val) / 4096 = j := by omega
    rw [this]
    exact hitem
  have himgLen : ∀ e ∈ Buf.img gS enS, e < d.logits.len := by
    rintro e ⟨i, he, rfl⟩
    exact hinS i he
  refine safe_last ?_
  refine safe_store_buf d.logits (Buf.addrs_of_get d.logits hmaskS hptrS) hinS hinj hIval
    (Buf.elems_mono d.logits himg) fun m' hrd hfr hpres => safe_ret ?_
  refine ⟨?_, ?_, ?_⟩
  · intro k' c' hM hmod hlt
    by_cases hold : itemOf d k' c' < p + t * d.NUM_SMS
    · -- an earlier trip's cell: not written here
      have hcb := cell_bounds hd hpre hM
      have hne : ∀ x ∈ Mem.span .f32 (d.logits.ptr (cell d init k' c')).addr,
          x ∉ d.logits.elems (Buf.img gS enS) := by
        rintro x hx ⟨e, he, hxe⟩
        by_cases heq : (e : Int) = cell d init k' c'
        · obtain ⟨k'', c'', hM'', hi'', he''⟩ := himg he
          obtain ⟨rfl, rfl⟩ := cell_inj hd hpre hM'' hM (he''.symm.trans heq)
          omega
        · exact Set.disjoint_left.mp (d.logits.span_disjoint heq) hxe hx
      rw [Mem.read_congr fun x hx => hfr x (hne x hx)]
      exact hIold k' c' hM hmod hold
    · -- this trip's cell
      push Not at hold
      have hw := item_eq_of_mod hmod hold (by rw [Nat.add_mul, Nat.one_mul] at *; omega)
      obtain ⟨hk1, hc1⟩ := itemOf_eq hM.2.1 hw
      obtain ⟨hk2, hj2⟩ := divmod_of_eq hj hitem
      have hkk : k' = k := by rw [hk1, hk2]
      subst hkk
      have hcj : c' / 4096 = j := by rw [hc1, hj2]
      have hl : c' % 4096 < 4096 := Nat.mod_lt _ (by norm_num)
      have hc'eq : j * 4096 + c' % 4096 = c' := by omega
      have hen : enS (⟨c' % 4096, hl⟩, ()) = true := by
        simp only [enS, hc'eq, hM.2.1, decide_true, Bool.true_and, hM.2.2, Bool.not_false]
      have h := hrd _ hen
      rw [hgS _ hen] at h
      simp only [hc'eq] at h
      rw [h]
      exact congrArg some negInfC_eq
  · intro z hz
    rw [hfr z (fun hz' => hz (Buf.elems_sub_bytes d.logits himgLen hz'))]
    exact hIfr z hz
  · exact Buf.valid_of_agree hIval (fun x hx => hpres x (hIval.2.2.2 x hx))

end Claims.TokenBitmask

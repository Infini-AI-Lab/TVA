import Proof.Lanes

/-!
# The program, and its max loop

The model's body with its three loops' trips as definitions (`maxTrip`, `sumTrip`, `gatherTrip`,
`body_eq`), the row's footprints (`rowIn`, `rowIds`, `rowOut`), the row's values as the initial
memory holds them (`xsOf`, `tidOf`), the `int32` loops' trip counts (`trips_eq`, `range_ok`), and
one trip of the max loop (`safe_maxTrip`), which keeps `MaxInv`: every logit seen so far is at
most the carried max, which is `-∞` or one of them.
-/
namespace Claims.TopkLogSoftmax

open TVA

/-! ## The trips -/

/-- One trip of the max loop (source lines 47–49). -/
def maxTrip (row_ptr : Ptr .f32) (vocab_size : I32) (i : I32) (max_val : F32) : Prog F32 := do
  let block := i + tl.arange 0 1024
  let logits ← tl.load (row_ptr + block) (some (block.ltS vocab_size)) (some (bc negInfC))
  tl.maxAll true (← tl.maximum logits (bc max_val))

/-- One trip of the sum loop (source lines 54–60). -/
def sumTrip (row_ptr : Ptr .f32) (vocab_size : I32) (max_val : F32) (i : I32) (se : F32) :
    Prog F32 := do
  let block := i + tl.arange 0 1024
  let logits ← tl.load (row_ptr + block) (some (block.ltS vocab_size)) (some (bc zeroC))
  let e ← tl.exp (← tl.sub logits (bc max_val))
  let e := tl.where (block.ltS vocab_size) e (bc zeroC)
  return (← tl.add (bc se) (bc (← tl.sum e))).item

/-- One trip of the gather loop (source lines 64–72). -/
def gatherTrip (output_ptr row_ptr : Ptr .f32) (topk_ids_ptr : Ptr .i64) (req_idx : I64)
    (topk_ids_stride topk : I32) (TOPK_BLOCK_SIZE : Nat) (max_val lse : F32) (j : I32) (_u : Unit) :
    Prog Unit := do
  let k_offset := j + tl.arange 0 TOPK_BLOCK_SIZE
  let k_mask := k_offset.ltS topk
  let topk_ids ← tl.load
    (topk_ids_ptr + req_idx * (← tl.convert .i32 .i64 (Tensor.scalar topk_ids_stride)).item
      + k_offset) (some k_mask) (some (bc (IntV.ofInt 0 : I64)))
  let logits ← tl.load (row_ptr + topk_ids) (some k_mask)
  let o ← tl.sub (← tl.sub logits (bc max_val)) (bc lse)
  tl.store (output_ptr + req_idx * (← tl.convert .i32 .i64 (Tensor.scalar topk)).item
    + k_offset) o (some k_mask)

/-- The model's body, with the trips named. -/
theorem body_eq (d : Inst) (c : Ctx) :
    (model d).body c = (do
      let req_idx := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id c 0))).item
      let row_ptr := d.logits.ptr 0 + req_idx *
        (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat d.logits_stride : I32))).item
      let max_val ← tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat d.vocab : I32)
        (IntV.ofNat 1024 : I32) negInfC (maxTrip row_ptr (IntV.ofNat d.vocab))
      let se ← tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat d.vocab : I32)
        (IntV.ofNat 1024 : I32) zeroC (sumTrip row_ptr (IntV.ofNat d.vocab) max_val)
      let lse := (← tl.log (bc se)).item
      tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat d.topk : I32)
        (IntV.ofNat d.TOPK_BLOCK_SIZE : I32) ()
        (gatherTrip (d.out.ptr 0) row_ptr (d.ids.ptr 0) req_idx (IntV.ofNat d.ids_stride)
          (IntV.ofNat d.topk) d.TOPK_BLOCK_SIZE max_val lse) : Prog Unit) := rfl

/-! ## The row's footprints and values -/

/-- The logits of row `r`. -/
def rowIn (d : Inst) (r : Nat) : Set Nat := {e | ∃ k, k < d.vocab ∧ e = r * d.logits_stride + k}

/-- The token ids of row `r`. -/
def rowIds (d : Inst) (r : Nat) : Set Nat := {e | ∃ j, j < d.topk ∧ e = r * d.ids_stride + j}

/-- The output elements of row `r`. -/
def rowOut (d : Inst) (r : Nat) : Set Nat := {e | ∃ j, j < d.topk ∧ e = r * d.topk + j}

/-- Logit `k` of row `r` as memory `m` holds it. -/
noncomputable def xsOf (d : Inst) (m : Mem) (r k : Nat) : F32 :=
  (m.read .f32 (d.logits.ptr (inIdx d r k)).addr).getD default

/-- Token id `j` of row `r` as memory `m` holds it. -/
noncomputable def tidOf (d : Inst) (m : Mem) (r j : Nat) : I64 :=
  (m.read .i64 (d.ids.ptr (idIdx d r j)).addr).getD default

/-- A logit read is `xsOf`. -/
lemma xsOf_of_read {d : Inst} {m : Mem} {r k : Nat} {v : F32}
    (h : m.read .f32 (d.logits.ptr (inIdx d r k)).addr = some v) : xsOf d m r k = v := by
  unfold xsOf; rw [h]; rfl

/-- A token id read is `tidOf`. -/
lemma tidOf_of_read {d : Inst} {m : Mem} {r j : Nat} {v : I64}
    (h : m.read .i64 (d.ids.ptr (idIdx d r j)).addr = some v) : tidOf d m r j = v := by
  unfold tidOf; rw [h]; rfl

/-- The logits of a row of the launch are read in a valid memory. -/
lemma read_xs {d : Inst} (hd : d.ok) {m : Mem} (hv : d.logits.Valid m) {r k : Nat}
    (hr : r < d.batch) (hk : k < d.vocab) :
    m.read .f32 (d.logits.ptr (inIdx d r k)).addr = some (xsOf d m r k) := by
  obtain ⟨v, hv'⟩ := Buf.read_some d.logits hv (i := inIdx d r k)
    (by have := hd.logits_len r hr; simp only [inIdx]; omega)
  rw [hv', xsOf_of_read hv']

/-- The token ids of a row of the launch are read in a valid memory. -/
lemma read_tid {d : Inst} (hd : d.ok) {m : Mem} (hv : d.ids.Valid m) {r j : Nat}
    (hr : r < d.batch) (hj : j < d.topk) :
    m.read .i64 (d.ids.ptr (idIdx d r j)).addr = some (tidOf d m r j) := by
  obtain ⟨v, hv'⟩ := Buf.read_some d.ids hv (hd.ids_len r j hr hj)
  rw [hv', tidOf_of_read hv']

/-! ## The `int32` loops -/

/-- The trip count of `range(0, n, B)`. -/
lemma trips_eq {n B : Nat} (hB : 0 < B) (h : n + B ≤ 2 ^ 31) (hB31 : B < 2 ^ 31) :
    tl.tripCount ((IntV.ofInt 0 : I32).toInt) ((IntV.ofNat n : I32).toInt)
      ((IntV.ofNat B : I32).toInt) = (n + B - 1) / B := by
  have h0 : (IntV.ofInt 0 : I32).toInt = 0 := by decide
  rw [IntV.I32.toInt_ofNat (n := n) (by omega), IntV.I32.toInt_ofNat (n := B) hB31, h0]
  unfold tl.tripCount
  have h1 : ((n : Int) - 0 + (B : Int) - 1) = ((n + B - 1 : Nat) : Int) := by omega
  rw [h1, ← Int.natCast_ediv, Int.toNat_natCast]

/-- The bounds of `range(0, n, B)` are within `tl.RangeOk` when `n + B ≤ 2^31`. -/
lemma range_ok {n B : Nat} (hB : 0 < B) (h : n + B ≤ 2 ^ 31) (hB31 : B < 2 ^ 31) :
    tl.RangeOk (IntV.ofInt 0 : I32) (IntV.ofNat n : I32) (IntV.ofNat B : I32) := by
  have h0 : (IntV.ofInt 0 : I32).toInt = 0 := by decide
  unfold tl.RangeOk
  rw [IntV.I32.toInt_ofNat (n := n) (by omega), IntV.I32.toInt_ofNat (n := B) hB31, h0]
  simp only [IntV.maxInt]
  norm_num
  omega

/-- The induction variable of trip `t` of `range(0, n, B)` has value `t·B`. -/
lemma iv_eq {t B : Nat} (h : t * B < 2 ^ 31) (hB : B < 2 ^ 31) :
    (IntV.ofInt ((IntV.ofInt 0 : I32).toInt + (t : Int) * (IntV.ofNat B : I32).toInt) : I32).toInt =
      ((t * B : Nat) : Int) := by
  have h0 : (IntV.ofInt 0 : I32).toInt = 0 := by decide
  rw [h0, IntV.I32.toInt_ofNat hB, zero_add]
  have h2 : ((t * B : Nat) : Int) < 2 ^ 31 := by exact_mod_cast h
  have h3 : (0 : Int) ≤ ((t * B : Nat) : Int) := by positivity
  push_cast at h2 h3 ⊢
  rw [IntV.I32.toInt_ofInt ⟨by linarith, h2⟩]

/-! ## The max loop -/

/-- After `t` trips of the max loop, on a `RowOk` row: the carried max is not NaN, is at least
every logit below `t·1024`, and is `-∞` or one of the row's logits. -/
def MaxInv (xs : Nat → F32) (V t : Nat) (mv : F32) : Prop :=
  RowOk xs V → mv.isNaN = false ∧ (∀ k, k < V → k < t * 1024 → (xs k).toRat ≤ mv.toRat) ∧
    (mv = FloatV.negInf ∨ ∃ k, k < V ∧ mv = xs k)

/-- The kernel's `float("-inf")` is the `float32` `-∞`. -/
lemma negInfC_eq : negInfC = FloatV.negInf := by decide

/-- Before the loop. -/
lemma maxInv_zero (xs : Nat → F32) (V : Nat) : MaxInv xs V 0 negInfC := by
  intro _
  refine ⟨by rw [negInfC_eq]; exact FloatV.negInf_isNaN b32_ieee, fun k _ hk => by omega,
    Or.inl negInfC_eq⟩

/-- A logit of a `RowOk` row is not NaN. -/
lemma row_nan {xs : Nat → F32} {V k : Nat} (h : RowOk xs V) (hk : k < V) :
    (xs k).isNaN = false := by
  rcases h.1 k hk with hf | hn
  · exact (FloatV.isFinite_iff.mp hf).1
  · rw [hn]; exact FloatV.negInf_isNaN b32_ieee

set_option maxHeartbeats 800000 in
/-- **One trip of the max loop** reads block `t` of row `p` and keeps `MaxInv`. -/
lemma safe_maxTrip {d : Inst} (hd : d.ok) {E : Env} {init : Mem} (hv : d.logits.Valid init)
    {p t : Nat} (hp : p < d.batch) (ht : t * 1024 < d.vocab) {c : I32}
    (hc : c.toInt = ((t * 1024 : Nat) : Int)) {mv : F32}
    (hI : MaxInv (xsOf d init p) d.vocab t mv) {Rp : Set Int}
    (hR : d.logits.elems (rowIn d p) ⊆ Rp) :
    Safe E Rp ∅ (fun st m₂ => m₂ = init ∧ MaxInv (xsOf d init p) d.vocab (t + 1) st) init
      (maxTrip (d.logits.ptr ((p * d.logits_stride : Nat) : Int)) (IntV.ofNat d.vocab) c mv) := by
  have hV19 := hd.vocab_le
  have hfit : t * 1024 + 1024 ≤ 2 ^ 31 := by omega
  have hV31 : d.vocab < 2 ^ 31 := by omega
  unfold maxTrip
  refine safe_load_buf d.logits (blk_addrs d.logits (p * d.logits_stride) hc hfit hV31)
    ?_ hv ?_ fun v hv1 hv2 => ?_
  · intro i hi
    have := hd.logits_len p hp
    have := of_decide_eq_true hi
    omega
  · exact (Buf.elems_mono d.logits blk_img).trans hR
  refine safe_maximum fun m hm => ?_
  refine safe_last (safe_maxAll (by norm_num) true fun mv' hmv' => safe_ret ⟨rfl, ?_⟩)
  have hbcget : ∀ i : Idx [1024 - 0], (bc mv : Tensor [1024 - 0] F32).get i = mv := fun _ => rfl
  simp only [hbcget] at hm
  intro hrow
  obtain ⟨hmvn, hle, hmem⟩ := hI hrow
  set xs := xsOf d init p with hxs
  -- the lanes of the load
  have hlane : ∀ (l : Nat) (hl : l < 1024),
      v.get1 ⟨l, by simpa using hl⟩ =
        if t * 1024 + l < d.vocab then xs (t * 1024 + l) else FloatV.negInf := by
    intro l hl
    by_cases hk : t * 1024 + l < d.vocab
    · rw [if_pos hk]
      have h1 := hv1 (⟨l, by simpa using hl⟩, ()) (decide_eq_true hk)
      exact (xsOf_of_read (d := d) (r := p) (k := t * 1024 + l) h1).symm
    · rw [if_neg hk]
      have h2 := hv2 (⟨l, by simpa using hl⟩, ()) (decide_eq_false hk) _ rfl
      exact h2.trans negInfC_eq
  have hvnan : ∀ (l : Nat) (hl : l < 1024), (v.get1 ⟨l, by simpa using hl⟩).isNaN = false := by
    intro l hl
    rw [hlane l hl]
    split_ifs with hk
    · exact row_nan hrow hk
    · exact FloatV.negInf_isNaN b32_ieee
  have hmnan : ∀ (l : Nat) (hl : l < 1024), (m.get1 ⟨l, by simpa using hl⟩).isNaN = false := by
    intro l hl
    exact FloatV.isNaN_of_mem (hvnan l hl) hmvn
      ((hm (⟨l, by simpa using hl⟩, ())).mem (hvnan l hl) hmvn)
  obtain ⟨tr, hcov, hev⟩ := hmv'
  have hleaf : ∀ j ∈ tr.leaves, j < 1024 := fun j hj => by
    have := List.mem_range.mp (hcov.subset hj); simpa using this
  have hxnan : ∀ j ∈ tr.leaves,
      ((fun j => if h : j < 1024 - 0 then m.get1 ⟨j, h⟩ else default) j).isNaN = false := by
    intro j hj
    have hj' := hleaf j hj
    simp only [dif_pos (show j < 1024 - 0 by simpa using hj')]
    exact hmnan j hj'
  obtain ⟨⟨j0, hj0, hmv'eq⟩, hall⟩ := Tree.eval_maxRel hev hxnan
  have hj0' := hleaf j0 hj0
  simp only [dif_pos (show j0 < 1024 - 0 by simpa using hj0')] at hmv'eq
  have hmv'nan : mv'.isNaN = false := by rw [hmv'eq]; exact hmnan j0 hj0'
  -- every lane's result is at most `mv'`
  have hm_le : ∀ (l : Nat) (hl : l < 1024),
      (m.get1 ⟨l, by simpa using hl⟩).toRat ≤ mv'.toRat := by
    intro l hl
    have hmemb : l ∈ tr.leaves := hcov.symm.subset (List.mem_range.mpr (by simpa using hl))
    have h := hall l hmemb
    simp only [ite_true, dif_pos (show l < 1024 - 0 by simpa using hl)] at h
    exact (FloatV.le_iff (hmnan l hl) hmv'nan).mp h
  have hlane_le : ∀ (l : Nat) (hl : l < 1024),
      (v.get1 ⟨l, by simpa using hl⟩).toRat ≤ (m.get1 ⟨l, by simpa using hl⟩).toRat ∧
        mv.toRat ≤ (m.get1 ⟨l, by simpa using hl⟩).toRat := by
    intro l hl
    obtain ⟨h1, h2⟩ := (hm (⟨l, by simpa using hl⟩, ())).le (hvnan l hl) hmvn
    exact ⟨(FloatV.le_iff (hvnan l hl) (hmnan l hl)).mp h1,
      (FloatV.le_iff hmvn (hmnan l hl)).mp h2⟩
  refine ⟨hmv'nan, fun k hk hkt => ?_, ?_⟩
  · by_cases hkt' : k < t * 1024
    · have h1 := hle k hk hkt'
      have h2 := (hlane_le 0 (by norm_num)).2
      have h3 := hm_le 0 (by norm_num)
      linarith
    · obtain ⟨l, rfl⟩ : ∃ l, k = t * 1024 + l := ⟨k - t * 1024, by omega⟩
      have hl : l < 1024 := by
        have : (t + 1) * 1024 = t * 1024 + 1024 := by ring
        omega
      have h1 := (hlane_le l hl).1
      have h2 := hm_le l hl
      rw [hlane l hl, if_pos hk] at h1
      linarith
  · have hmem0 := (hm (⟨j0, by simpa using hj0'⟩, ())).mem (hvnan j0 hj0') hmvn
    rcases hmem0 with h | h
    · have h' : mv' = v.get1 ⟨j0, by simpa using hj0'⟩ := hmv'eq.trans h
      rw [hlane j0 hj0'] at h'
      split_ifs at h' with hk
      · exact Or.inr ⟨_, hk, h'⟩
      · exact Or.inl h'
    · have h' : m.get1 ⟨j0, by simpa using hj0'⟩ = mv := h
      rw [hmv'eq, h']
      exact hmem

end Claims.TopkLogSoftmax

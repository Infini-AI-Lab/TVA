import Proof.Trip

/-!
# `compute_position_i64`: one program

Program `b` reads its `extend_prefix_lens[b]` and the lengths `0 … b`, sums the lengths before
its own (`safe_sum`), writes its positions with the store loop (`Proof/Trip.lean`) and its start
(`safe_tail`). `safe_program` walks the whole body.
-/

namespace Claims.ComputePosition

open TVA

/-! ## The body, in parts -/

/-- One trip of the summing loop (source line 81). -/
def sumBody (lens : Ptr .i32) (i cs : I64) : Prog I64 := do
  let x ← tl.load1 (lens + i)
  let x64 := (← tl.convert .i32 .i64 (Tensor.scalar x)).item
  pure (cs + x64)

/-- The program after the summing loop (source lines 83-91). -/
def tailProg (d : Inst) (pid : I64) (prefix_len seq_len : I32) (cs : I64) : Prog Unit := do
  let num_loop := (← tl.idiv (Tensor.scalar (seq_len + (IntV.ofNat 511 : I32)))
    (Tensor.scalar (IntV.ofNat 512 : I32))).item
  let _ ← tl.forRange (IntV.ofInt 0 : I32) num_loop (IntV.ofInt 1 : I32) () fun i _ =>
    tripBody (d.positions.ptr 0) prefix_len seq_len cs i
  let start_loc := (← tl.convert .i64 .i32 (Tensor.scalar cs)).item
  tl.store1 (d.extend_start_loc.ptr 0 + pid) start_loc

/-- The program after `prefix_len` is known (source lines 76-91). -/
def afterPrefix (d : Inst) (pid : I64) (prefix_len : I32) : Prog Unit := do
  let seq_len ← tl.load1 (d.extend_seq_lens.ptr 0 + pid)
  let c0 := (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofInt 0 : I32))).item
  let cs ← tl.forRange (IntV.ofInt 0 : I64) pid (IntV.ofInt 1 : I64) c0
    (sumBody (d.extend_seq_lens.ptr 0))
  tailProg d pid prefix_len seq_len cs

theorem body_eq (d : Inst) (ctx : Ctx) : (model d).body ctx = (do
    let pid := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id ctx 0))).item
    let prefix_len : I32 ← if d.hasPrefix then tl.load1 (d.extend_prefix_lens.ptr 0 + pid)
      else pure (IntV.ofInt 0)
    afterPrefix d pid prefix_len) := rfl

/-! ## Constants -/

theorem i64_zero : (IntV.ofInt 0 : I64).toInt = 0 := IntV.I64.toInt_ofInt (by norm_num)
theorem i64_one : (IntV.ofInt 1 : I64).toInt = 1 := IntV.I64.toInt_ofInt (by norm_num)
theorem i32_zero : (IntV.ofInt 0 : I32).toInt = 0 := IntV.I32.toInt_ofInt (by norm_num)
theorem i32_one : (IntV.ofInt 1 : I32).toInt = 1 := IntV.I32.toInt_ofInt (by norm_num)

/-! ## The summing loop -/

/-- **The summing loop** (source lines 79-81) of program `b` ends with `cumsum_start = start(b)`,
reading only the lengths `0 … b - 1`, and leaves the memory alone. -/
theorem safe_sum {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env) {b : Nat}
    (hb : b < d.bs) (pid : I64) (hpid : pid.toInt = (b : Int)) (c0 : I64) (hc0 : c0.toInt = 0)
    {Q' : Unit → Mem → Prop} {k : I64 → Prog Unit}
    (hk : ∀ cs : I64, cs.toInt = start d init b → Safe E (R d b) (W d init b) Q' init (k cs)) :
    Safe E (R d b) (W d init b) Q' init
      (tl.forRange (IntV.ofInt 0 : I64) pid (IntV.ofInt 1 : I64) c0
        (sumBody (d.extend_seq_lens.ptr 0)) >>= k) := by
  have hir : IntV.maxInt true 64 = 2 ^ 63 - 1 := by norm_num [IntV.maxInt]
  have hg := hd.grid_lt
  have hrange : tl.RangeOk (IntV.ofInt 0 : I64) pid (IntV.ofInt 1 : I64) := by
    refine ⟨?_, ?_, ?_, ?_⟩ <;> simp only [hir, i64_zero, i64_one, hpid] <;> omega
  have hN : tl.tripCount (IntV.ofInt 0 : I64).toInt pid.toInt (IntV.ofInt 1 : I64).toInt = b := by
    simp only [tl.tripCount, i64_zero, i64_one, hpid]
    omega
  have hmain : Safe E (R d b) (iterW (fun _ => (∅ : Set Int))
      (tl.tripCount (IntV.ofInt 0 : I64).toInt pid.toInt (IntV.ofInt 1 : I64).toInt) ∪ W d init b)
      Q' init (tl.forRange (IntV.ofInt 0 : I64) pid (IntV.ofInt 1 : I64) c0
        (sumBody (d.extend_seq_lens.ptr 0)) >>= k) := by
    refine safe_forRange hrange (fun t st m => m = init ∧ st.toInt = start d init t) (fun _ => ∅)
      (fun _ _ _ _ => Set.disjoint_empty _) (by rw [iterW_empty]; exact Set.empty_disjoint _)
      ⟨rfl, by rw [hc0, start_zero]⟩ ?_ ?_
    · intro t ht st m₁ ⟨hm, hst⟩
      rw [hm]
      rw [hN] at ht
      have hi : (IntV.ofInt ((IntV.ofInt 0 : I64).toInt + (t : Int) * (IntV.ofInt 1 : I64).toInt) :
          I64).toInt = (t : Int) := by
        rw [i64_zero, i64_one]
        have := IntV.I64.toInt_ofInt (x := (0 : Int) + (t : Int) * 1) (by omega)
        simpa using this
      simp only [sumBody]
      have htl : t < d.extend_seq_lens.len := by have := hd.lens_len; omega
      obtain ⟨v0, hv0⟩ := Buf.read_some d.extend_seq_lens hpre.2.2.1 htl
      refine safe_load1 (v0 := v0) (by rw [ptr_zero_add _ _ hi]; exact hv0) ?_ ?_
      · rw [ptr_zero_add _ _ hi]
        exact fun x hx => Or.inl ⟨t, by show t ≤ b; omega, hx⟩
      refine safe_convert (fun _ => ⟨_, (ConvRel.iff_cast_i32_i64).2 rfl⟩) fun r hr => ?_
      refine safe_ret ⟨rfl, ?_⟩
      have hr' : r.item.toInt = len d init t := by
        have h := ConvRel.toInt_i32_i64 (hr ())
        rw [show r.get () = r.item from rfl] at h
        rw [h, len, valOf_of_read hv0]
        rfl
      have h1 : 0 ≤ start d init (t + 1) := start_nonneg hpre (by omega)
      have h2 : start d init (t + 1) ≤ start d init b := start_mono hpre (by omega) (le_of_lt hb)
      have h3 := (hpre.2.2.2.2.1 b hb).2.2.2
      rw [start_succ] at h1 h2 ⊢
      exact IntV.toInt_add hst hr' (IntV.I64.sfits (by linarith) (by linarith))
    · intro st m₁ ⟨hm, hst⟩
      rw [hm]
      rw [hN] at hst
      exact hk st hst
  rwa [iterW_empty, Set.empty_union] at hmain

/-! ## The store loop and the start -/

/-- The trip count `(s + 511) / 512` of a nonnegative length `s` covers `s`. -/
theorem lt_trip_mul {s : Int} (h0 : 0 ≤ s) : s ≤ (((s + 511) / 512).toNat : Int) * 512 := by
  have hq0 : 0 ≤ (s + 511) / 512 := Int.ediv_nonneg (by linarith) (by norm_num)
  rw [Int.toNat_of_nonneg hq0]
  omega

/-- **The rest of program `b`** after the summing loop (source lines 83-91): the store loop, then
the start. -/
theorem safe_tail {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env) {b : Nat}
    (hb : b < d.bs) (pid : I64) (hpid : pid.toInt = (b : Int)) (prefix_len seq_len : I32)
    (hp : prefix_len.toInt = pre d init b) (hs : seq_len.toInt = len d init b) (cs : I64)
    (hcs : cs.toInt = start d init b) :
    Safe E (R d b) (W d init b) (fun _ m => Q d init b m) init
      (tailProg d pid prefix_len seq_len cs) := by
  obtain ⟨hl0, hl1, -, hst⟩ := hpre.2.2.2.2.1 b hb
  have hs0 : 0 ≤ start d init b := start_nonneg hpre (le_of_lt hb)
  simp only [tailProg]
  have h511 : (IntV.ofNat 511 : I32).toInt = 511 := IntV.I32.toInt_ofNat (by norm_num)
  have h512 : (IntV.ofNat 512 : I32).toInt = 512 := IntV.I32.toInt_ofNat (by norm_num)
  have hnum : (seq_len + (IntV.ofNat 511 : I32)).toInt = len d init b + 511 :=
    IntV.toInt_add hs h511 (IntV.I32.sfits (by linarith) (by linarith))
  refine safe_idiv1 (by rw [hnum]; linarith) (by rw [h512]; norm_num) fun n hn => ?_
  rw [hnum, h512] at hn
  have hq0 : 0 ≤ (len d init b + 511) / 512 := Int.ediv_nonneg (by linarith) (by norm_num)
  have hir : IntV.maxInt true 32 = 2 ^ 31 - 1 := by norm_num [IntV.maxInt]
  have hrange : tl.RangeOk (IntV.ofInt 0 : I32) n.item (IntV.ofInt 1 : I32) := by
    refine ⟨?_, ?_, ?_, ?_⟩ <;> simp only [hir, i32_zero, i32_one, hn] <;> omega
  have hN : tl.tripCount (IntV.ofInt 0 : I32).toInt n.item.toInt (IntV.ofInt 1 : I32).toInt =
      ((len d init b + 511) / 512).toNat := by
    simp only [tl.tripCount, i32_zero, i32_one, hn, sub_zero, add_sub_cancel_right, Int.ediv_one]
  let Wt : Nat → Set Int := fun t => d.positions.elems (tripElems d init b t)
  have hWsub : ∀ N, iterW Wt N ⊆ d.positions.elems (posElems d init b) := by
    rintro N x ⟨t, -, hx⟩
    exact d.positions.elems_mono (tripElems_sub d init b t) hx
  have hposB : d.positions.elems (posElems d init b) ⊆ d.positions.bytes :=
    d.positions.elems_sub_bytes (posElems_len hpre hb)
  have hslB : d.extend_start_loc.elems {b} ⊆ d.extend_start_loc.bytes :=
    d.extend_start_loc.elems_sub_bytes (by
      intro j hj
      rw [Set.mem_singleton_iff] at hj
      subst hj
      exact lt_of_lt_of_le hb hd.start_len)
  refine Safe.mono (Q := fun _ m => Q d init b m)
    (W := iterW Wt (tl.tripCount (IntV.ofInt 0 : I32).toInt n.item.toInt
      (IntV.ofInt 1 : I32).toInt) ∪ d.extend_start_loc.elems {b}) ?_ (fun _ _ h => h)
    (Set.union_subset_union (hWsub _) (le_refl _))
  refine safe_forRange hrange (fun t _ m => TripI d init b t m) Wt ?_ ?_
    ⟨fun j _ hj => by have := Nat.zero_le j; push_cast at hj; omega, hpre.1, hpre.2.1⟩ ?_ ?_
  · intro t u htu _
    exact d.positions.elems_disjoint (tripElems_disjoint htu)
  · exact Set.disjoint_of_subset (hWsub _ |>.trans hposB) hslB hd.disj.1
  · intro t ht st m₁ hI
    rw [hN] at ht
    have ht' : (t : Int) < (len d init b + 511) / 512 := by
      have : (t : Int) < (((len d init b + 511) / 512).toNat : Int) := by exact_mod_cast ht
      rwa [Int.toNat_of_nonneg hq0] at this
    have ht512 : (t : Int) * 512 + 512 ≤ 2 ^ 31 := by omega
    have hi : (IntV.ofInt ((IntV.ofInt 0 : I32).toInt + (t : Int) * (IntV.ofInt 1 : I32).toInt) :
        I32).toInt = (t : Int) := by
      rw [i32_zero, i32_one]
      have := IntV.I32.toInt_ofInt (x := (0 : Int) + (t : Int) * 1) (by omega)
      simpa using this
    exact safe_trip hpre E hb prefix_len seq_len cs hp hs hcs t _ hi ht512 m₁ hI
  · intro st m₁ hI
    rw [hN] at hI
    obtain ⟨hQ, hVp, hVs⟩ := hI
    refine safe_convert (fun _ => ⟨_, rfl⟩) fun r hr => ?_
    have hr' : r.item.toInt = start d init b := by
      have h := ConvRel.toInt_i64_i32 (x := cs) (hr ())
        (by rw [hcs]; exact IntV.I32.sfits (by linarith) hst)
      rw [show r.get () = r.item from rfl] at h
      rw [h, hcs]
    rw [ptr_zero_add d.extend_start_loc pid hpid]
    refine safe_last ?_
    have hbl : b < d.extend_start_loc.len := lt_of_lt_of_le hb hd.start_len
    refine safe_store1 (Buf.writable d.extend_start_loc hVs hbl)
      (Buf.span_sub_elems d.extend_start_loc (by simp)) fun m' hread hframe => safe_ret ?_
    refine ⟨fun j hj _ => ?_, ⟨r.item, hread, hr'⟩⟩
    obtain ⟨v, hv, hvv⟩ := hQ j hj (lt_of_lt_of_le hj (lt_trip_mul hl0))
    refine ⟨v, ?_, hvv⟩
    have hjn : start d init b + (j : Int) = (((start d init b + (j : Int)).toNat : Nat) : Int) :=
      (Int.toNat_of_nonneg (by have := Nat.zero_le j; omega)).symm
    have hjin : (start d init b + (j : Int)).toNat ∈ posElems d init b :=
      ⟨j, hj, by rw [← hjn]⟩
    rw [Mem.read_congr (m₂ := m₁) fun x hx => hframe x ?_]
    · exact hv
    · intro hx'
      rw [hjn] at hx
      exact Set.disjoint_left.mp hd.disj.1
        (hposB (Buf.span_sub_elems d.positions hjin hx)) (hslB (Buf.span_sub_elems _ (by simp) hx'))

/-! ## The whole body -/

/-- The program once `prefix_len` is known (source lines 76-91). -/
theorem safe_after {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env) {b : Nat}
    (hb : b < d.bs) (pid : I64) (hpid : pid.toInt = (b : Int)) (prefix_len : I32)
    (hp : prefix_len.toInt = pre d init b) :
    Safe E (R d b) (W d init b) (fun _ m => Q d init b m) init
      (afterPrefix d pid prefix_len) := by
  simp only [afterPrefix]
  have hbl : b < d.extend_seq_lens.len := lt_of_lt_of_le hb hd.lens_len
  obtain ⟨v0, hv0⟩ := Buf.read_some d.extend_seq_lens hpre.2.2.1 hbl
  refine safe_load1 (v0 := v0) (by rw [ptr_zero_add _ _ hpid]; exact hv0) ?_ ?_
  · rw [ptr_zero_add _ _ hpid]
    exact fun x hx => Or.inl ⟨b, le_refl b, hx⟩
  have hs : v0.toInt = len d init b := by rw [len, valOf_of_read hv0]
  refine safe_convert (fun _ => ⟨_, (ConvRel.iff_cast_i32_i64).2 rfl⟩) fun c0 hc0 => ?_
  have hc0' : c0.item.toInt = 0 := by
    have h := ConvRel.toInt_i32_i64 (hc0 ())
    rw [show c0.get () = c0.item from rfl] at h
    rw [h]
    exact i32_zero
  exact safe_sum hd hpre E hb pid hpid c0.item hc0' fun cs hcs =>
    safe_tail hd hpre E hb pid hpid prefix_len v0 hp hs cs hcs

/-- **One program** of the grid is safe from `init`: it reads its `extend_prefix_lens[b]` and the
lengths up to its own, writes its positions and its start, and ends with `Q`. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    (i : Fin (launch d).size) :
    Safe E (R d (i : Nat)) (W d init (i : Nat)) (fun _ m => Q d init (i : Nat) m) init
      ((model d).body ((launch d).ctx i)) := by
  rw [body_eq]
  have hb : (i : Nat) < d.bs := lt_of_lt_of_eq i.isLt (launch_size d)
  have hb31 : (i : Nat) < 2 ^ 31 := lt_trans hb hd.grid_lt
  refine safe_convert (fun _ => ⟨_, (ConvRel.iff_cast_i32_i64).2 rfl⟩) fun r hr => ?_
  have hpid : r.item.toInt = ((i : Nat) : Int) := by
    have h := ConvRel.toInt_i32_i64 (hr ())
    rw [show r.get () = r.item from rfl] at h
    rw [h]
    show (tl.program_id ((launch d).ctx i) 0).toInt = _
    rw [tl.program_id, launch_pid]
    exact IntV.I32.toInt_ofNat hb31
  by_cases hP : d.hasPrefix = true
  · rw [ite_eq_left hP]
    have hbl : (i : Nat) < d.extend_prefix_lens.len := lt_of_lt_of_le hb (hd.prefix_len hP)
    obtain ⟨v0, hv0⟩ := Buf.read_some d.extend_prefix_lens (hpre.2.2.2.1 hP) hbl
    refine safe_load1 (v0 := v0) (by rw [ptr_zero_add _ _ hpid]; exact hv0) ?_ ?_
    · rw [ptr_zero_add _ _ hpid]
      intro x hx
      simp only [R, hP, ↓reduceIte]
      exact Or.inr ⟨(i : Nat), rfl, hx⟩
    exact safe_after hd hpre E hb r.item hpid v0 (by rw [pre, ite_eq_left hP, valOf_of_read hv0])
  · rw [ite_eq_right hP]
    exact safe_after hd hpre E hb r.item hpid (IntV.ofInt 0)
      (by rw [pre, ite_eq_right hP]; exact i32_zero)

end Claims.ComputePosition

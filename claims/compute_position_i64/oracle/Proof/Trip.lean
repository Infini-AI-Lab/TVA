import Proof.Basic

/-!
# `compute_position_i64`: one trip of the store loop

The loop `for i in range(num_loop)` of `compute_position_kernel` writes the positions
`t * 512, …, t * 512 + 511` of its request (masked by the length) into `positions`. This file
proves `safe_trip`, the program safety of one trip, with the invariant `TripI`.
-/

namespace Claims.ComputePosition

open TVA

/-- One trip of the store loop (source lines 85-90): the lanes `offset`, the `int64` values and
the masked store. `cs` is `cumsum_start`, `i` the trip's `int32` induction variable. -/
def tripBody (positions : Ptr .i64) (prefix_len seq_len : I32) (cs : I64) (i : I32) :
    Prog Unit := do
  let offset := tl.arange 0 512 + i * (IntV.ofNat 512 : I32)
  let vals ← tl.convert .i32 .i64 (prefix_len + offset)
  tl.store (positions + cs + offset) vals (some (offset.ltS seq_len))

/-- The invariant of the store loop before trip `t`: the positions below `t * 512` are written,
and both outputs are valid. -/
def TripI (d : Inst) (init : Mem) (b t : Nat) (m : Mem) : Prop :=
  QPos d init b ((t : Int) * 512) m ∧ d.positions.Valid m ∧ d.extend_start_loc.Valid m

/-- Trips `t < u` of a program write disjoint elements of `positions`. -/
theorem tripElems_disjoint {d : Inst} {init : Mem} {b t u : Nat} (htu : t < u) :
    Disjoint (tripElems d init b t) (tripElems d init b u) := by
  rw [Set.disjoint_left]
  rintro j ⟨l, hl, -, hj⟩ ⟨l', hl', -, hj'⟩
  have hu : (t : Int) + 1 ≤ (u : Int) := by exact_mod_cast htu
  have hl512 : (l : Int) < 512 := by exact_mod_cast hl
  have hl'0 : (0 : Int) ≤ (l' : Int) := by exact_mod_cast Nat.zero_le l'
  nlinarith

/-- **One trip is safe.** It writes the positions `t * 512 + l < len` of request `b`, extending the
invariant by one trip. -/
theorem safe_trip {d : Inst} {init : Mem} (hpre : Pre d init) (E : Env) {b : Nat}
    (hb : b < d.bs) (prefix_len seq_len : I32) (cs : I64)
    (hp : prefix_len.toInt = pre d init b) (hs : seq_len.toInt = len d init b)
    (hcs : cs.toInt = start d init b) (t : Nat) (i : I32) (hi : i.toInt = (t : Int))
    (ht : (t : Int) * 512 + 512 ≤ 2 ^ 31) (m : Mem) (hI : TripI d init b t m) :
    Safe E (R d b) (d.positions.elems (tripElems d init b t))
      (fun _ m' => TripI d init b (t + 1) m') m
      (tripBody (d.positions.ptr 0) prefix_len seq_len cs i) := by
  obtain ⟨hQ, hVp, hVs⟩ := hI
  obtain ⟨hl0, -, hpl, -⟩ := hpre.2.2.2.2.1 b hb
  have hpb := IntV.sfits_toInt prefix_len
  rw [hp] at hpb
  have hpb0 : -(2 ^ 31 : Int) ≤ pre d init b := by
    have := hpb.1; norm_num at this; linarith
  have hs0 : 0 ≤ start d init b := start_nonneg hpre (le_of_lt hb)
  have hsl := start_len_le hpre hb
  have ht0 : (0 : Int) ≤ (t : Int) := by exact_mod_cast Nat.zero_le t
  simp only [tripBody]
  -- the lanes of `offset`
  have hiB : (i * (IntV.ofNat 512 : I32)).toInt = (t : Int) * 512 := by
    have h512 : (IntV.ofNat 512 : I32).toInt = 512 := IntV.I32.toInt_ofNat (by norm_num)
    exact IntV.toInt_mul hi h512 (IntV.I32.sfits (by linarith) (by linarith))
  have hoff : ∀ l : Idx [512], ((tl.arange 0 512 + i * (IntV.ofNat 512 : I32)).get l).toInt =
      (t : Int) * 512 + ((l.1 : Nat) : Int) := by
    intro l
    have hl := l.1.isLt
    have ha : ((tl.arange 0 512).get l).toInt = ((l.1 : Nat) : Int) := by
      show ((tl.arange 0 512).get (l.1, ())).toInt = _
      simp only [tl.arange, Tensor.get_arange]
      have h := IntV.I32.toInt_ofInt (x := (0 : Int) + ((l.1 : Nat) : Int))
        ⟨by push_cast; omega, by push_cast; omega⟩
      simpa using h
    rw [Tensor.get_adds, IntV.toInt_add ha hiB (IntV.I32.sfits (by omega) (by omega))]
    ring
  -- the enabled lanes and their elements
  let en : Idx [512] → Bool := fun l =>
    decide ((t : Int) * 512 + ((l.1 : Nat) : Int) < len d init b)
  let g : Idx [512] → Nat := fun l =>
    (start d init b + ((t : Int) * 512 + ((l.1 : Nat) : Int))).toNat
  have hen : ∀ l, en l = true → (t : Int) * 512 + ((l.1 : Nat) : Int) < len d init b :=
    fun l h => of_decide_eq_true h
  have hgv : ∀ l : Idx [512],
      (g l : Int) = start d init b + ((t : Int) * 512 + ((l.1 : Nat) : Int)) := by
    intro l
    exact Int.toNat_of_nonneg (by have := Nat.zero_le (l.1 : Nat); omega)
  have hmask : ∀ l, ((tl.arange 0 512 + i * (IntV.ofNat 512 : I32)).ltS seq_len).get l = en l := by
    intro l
    rw [Tensor.get_ltS, IntV.lt_eq, hoff l, hs]
  have hptr : ∀ l, en l = true →
      (d.positions.ptr 0 + cs + (tl.arange 0 512 + i * (IntV.ofNat 512 : I32))).get l =
        d.positions.ptr (g l : Int) := by
    intro l _
    rw [Tensor.get_ptrAdd, ptr_zero_add d.positions cs hcs, Buf.ptr_offset, hoff l, hgv l]
  have hin : ∀ l, en l = true → g l < d.positions.len := by
    intro l he
    have h := hen l he
    have : (g l : Int) < (d.positions.len : Int) := by rw [hgv l]; linarith
    exact_mod_cast this
  have hinj : ∀ l l', en l = true → en l' = true → g l = g l' → l = l' := by
    intro l l' _ _ hg
    have h1 := hgv l
    have h2 := hgv l'
    have h3 : ((l.1 : Nat) : Int) = ((l'.1 : Nat) : Int) := by
      have : (g l : Int) = (g l' : Int) := by rw [hg]
      linarith
    have h5 : (l.1 : Nat) = (l'.1 : Nat) := by exact_mod_cast h3
    have : Subsingleton (Idx []) := ⟨fun a b => by cases a; cases b; rfl⟩
    exact Prod.ext (Fin.ext h5) (Subsingleton.elim _ _)
  have himg : Buf.img g en ⊆ tripElems d init b t := by
    rintro j ⟨l, he, hg⟩
    refine ⟨l.1.val, l.1.isLt, hen l he, ?_⟩
    rw [← hg, hgv l]
  refine safe_convert (fun _ => ⟨_, (ConvRel.iff_cast_i32_i64).2 rfl⟩) fun vals hvals => ?_
  have hval : ∀ l, en l = true → (vals.get l).toInt = pre d init b + ((t : Int) * 512 +
      ((l.1 : Nat) : Int)) := by
    intro l he
    rw [ConvRel.toInt_i32_i64 (hvals l), Tensor.get_sadd]
    have h := hen l he
    have hl := Nat.zero_le (l.1 : Nat)
    exact IntV.toInt_add hp (hoff l) (IntV.I32.sfits (by omega) (by omega))
  refine safe_last ?_
  refine safe_store_buf d.positions (Buf.addrs_of_get d.positions hmask hptr) hin hinj hVp
    (d.positions.elems_mono himg) fun m' hread hframe hpres => safe_ret ?_
  refine ⟨?_, Buf.valid_of_agree hVp (fun x hx => hpres x (hVp.2.2.2 x hx)),
    Buf.valid_of_agree hVs (fun x hx => hpres x (hVs.2.2.2 x hx))⟩
  intro j hj hjt
  by_cases hjo : (j : Int) < (t : Int) * 512
  · obtain ⟨v, hv, hvv⟩ := hQ j hj hjo
    refine ⟨v, ?_, hvv⟩
    rw [Mem.read_congr (m₂ := m) fun x hx => hframe x ?_]
    · exact hv
    · rintro ⟨j', ⟨l, he, hg⟩, hx'⟩
      have hne : start d init b + (j : Int) ≠ (j' : Int) := by
        rw [← hg, hgv l]
        have := Nat.zero_le (l.1 : Nat)
        omega
      exact Set.disjoint_left.mp (d.positions.span_disjoint hne) hx hx'
  · push Not at hjo
    have hle : t * 512 ≤ j := by
      have : ((t * 512 : Nat) : Int) ≤ (j : Int) := by push_cast; linarith
      exact_mod_cast this
    have hl512 : j - t * 512 < 512 := by
      have : (j : Int) < (t : Int) * 512 + 512 := by push_cast at hjt; linarith
      omega
    let l : Idx [512] := (⟨j - t * 512, hl512⟩, ())
    have hlj : (t : Int) * 512 + ((l.1 : Nat) : Int) = (j : Int) := by
      show (t : Int) * 512 + ((j - t * 512 : Nat) : Int) = (j : Int)
      rw [Nat.cast_sub hle]; push_cast; ring
    have hel : en l = true := decide_eq_true (by rw [hlj]; exact hj)
    have hgl : (g l : Int) = start d init b + (j : Int) := by rw [hgv l, hlj]
    refine ⟨vals.get l, ?_, ?_⟩
    · rw [← hgl]; exact hread l hel
    · rw [hval l hel, hlj]

end Claims.ComputePosition

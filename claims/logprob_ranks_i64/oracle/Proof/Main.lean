import Proof.Trip

/-!
# The claim

Program `r` reads its token id and its row of logits and writes element `r` of the output. This
file walks the body (the two conversions, the two scalar loads, the loop by `safe_forRange` with
the invariant `n = cnt d r t`, the conversion and the store), then puts the programs together with
`correct_of_safe`.
-/

namespace Claims.LogprobRanks

open TVA

/-- The body of the program with program id `pid`, written as binds. -/
def bodyP (d : Inst) (pid : I32) : Prog Unit :=
  tl.convert .i32 .i64 (Tensor.scalar pid) >>= fun r1 =>
  tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat d.stride : I32)) >>= fun r2 =>
  tl.load1 (d.tok.ptr 0 + r1.item) >>= fun token_id =>
  tl.load1 ((d.logits.ptr 0 + r1.item * r2.item) + token_id) >>= fun x =>
  tl.forRange (0 : I32) (IntV.ofNat d.V) (IntV.ofNat d.BLOCK : I32) (0 : I32)
    (tripBody d (d.logits.ptr 0 + r1.item * r2.item) x) >>= fun n =>
  tl.convert .i32 .i64 (Tensor.scalar n) >>= fun r3 =>
  tl.store1 (d.out.ptr 0 + r1.item) r3.item

theorem body_eq (d : Inst) (c : Ctx) : (model d).body c = bodyP d (tl.program_id c 0) := rfl

/-- Program `i` of the launch has program id `i`. -/
theorem pid_eq (d : Inst) (i : Nat) (hi : i < d.B) :
    tl.program_id ((launch d).ctx i) 0 = (IntV.ofNat i : I32) := by
  simp [tl.program_id, Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi]

/-- The launch is valid. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 hd.B_lt fun s hs => ?_
  simp only [List.mem_singleton] at hs
  subst hs
  refine ⟨fun k hk => ?_, ?_⟩
  · simp only [List.mem_singleton] at hk
    subst hk
    exact hd.blk_pow2
  · simpa using hd.blk_le

/-- The loop's bounds are within `tl.RangeOk`, and it runs `numTrips d` times. -/
theorem range_ok {d : Inst} (hd : d.ok) :
    tl.RangeOk (0 : I32) (IntV.ofNat d.V) (IntV.ofNat d.BLOCK : I32) ∧
    tl.tripCount (0 : I32).toInt (IntV.ofNat d.V : I32).toInt (IntV.ofNat d.BLOCK : I32).toInt
      = numTrips d := by
  have hB := blk_pos hd
  have h2 := hd.V_blk
  have h3 := hd.blk_le
  have hV : (IntV.ofNat d.V : I32).toInt = (d.V : Int) := IntV.I32.toInt_ofNat (by omega)
  have hBv : (IntV.ofNat d.BLOCK : I32).toInt = (d.BLOCK : Int) :=
    IntV.I32.toInt_ofNat (by have := hd.blk_le; omega)
  refine ⟨?_, ?_⟩
  · refine ⟨?_, ?_, ?_, ?_⟩
    · rw [hBv]; exact_mod_cast hB
    · rw [hV, hBv, intRange_32]; omega
    · rw [toInt_zero_i32, intRange_32]; norm_num
    · rw [hBv, intRange_32]; omega
  · unfold tl.tripCount
    rw [toInt_zero_i32, hV, hBv, numTrips]
    rw [show (d.V : Int) - 0 + d.BLOCK - 1 = ((d.V + d.BLOCK - 1 : Nat) : Int) by omega]
    rw [← Int.natCast_ediv, Int.toNat_natCast]

/-- **One program** (row `r`) is safe: it reads the row's token id and logits, writes element
`r` of the output, and ends with the rank there. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    {r : Nat} (hr : r < d.B) :
    Safe E (Rp d r) (Wp d r) (fun _ m => Qp d init r m) init (bodyP d (IntV.ofNat r)) := by
  unfold bodyP
  have hr31 : r < 2 ^ 31 := lt_trans hr hd.B_lt
  refine safe_convert (fun _ => ConvRel.exists_i32_i64 _) fun r1 hr1 => ?_
  have h1 : r1.item.toInt = (r : Int) :=
    (ConvRel.toInt_i32_i64 (hr1 ())).trans (IntV.I32.toInt_ofNat hr31)
  refine safe_convert (fun _ => ConvRel.exists_i32_i64 _) fun r2 hr2 => ?_
  have h2 : r2.item.toInt = (d.stride : Int) :=
    (ConvRel.toInt_i32_i64 (hr2 ())).trans (IntV.I32.toInt_ofNat hd.stride_lt)
  -- the token id
  have hpt : d.tok.ptr 0 + r1.item = d.tok.ptr (r : Int) := by
    rw [Ptr.add_int, Buf.ptr_offset, h1, Int.zero_add]
  rw [hpt]
  refine safe_load1_buf d.tok (lt_of_lt_of_le hr hd.tok_len) hpre.2.1
    (fun z hz => Or.inl (Buf.span_sub_elems d.tok (S := {r}) (j := r) rfl hz)) fun tk htk => ?_
  have htk' : tk = d.T r := by
    have := hpre.2.2.2.2.1 r (lt_of_lt_of_le hr hd.tok_len)
    rw [this] at htk
    exact (Option.some.inj htk).symm
  subst htk'
  obtain ⟨ht0, htV⟩ := hpre.2.2.2.2.2 r hr
  have htok : (d.T r).toInt = (tokOf d r : Int) := by
    rw [tokOf, Int.toNat_of_nonneg ht0]
  -- the row pointer and the token's logit
  have hmul : (r1.item * r2.item).toInt = ((r * d.stride : Nat) : Int) := by
    have := IntV.I64.mul_nat h1 h2 (by
      have : r * d.stride < 2 ^ 31 * 2 ^ 31 := Nat.mul_lt_mul'' hr31 hd.stride_lt
      omega)
    simpa using this
  have hrow : d.logits.ptr 0 + r1.item * r2.item = d.logits.ptr ((r * d.stride : Nat) : Int) := by
    rw [Ptr.add_int, Buf.ptr_offset, hmul, Int.zero_add]
  rw [hrow]
  have hlen := hd.logits_len r hr
  have htV' : tokOf d r < d.V := by
    have : (tokOf d r : Int) < d.V := htok ▸ htV
    exact_mod_cast this
  have hpx : d.logits.ptr ((r * d.stride : Nat) : Int) + d.T r =
      d.logits.ptr ((r * d.stride + tokOf d r : Nat) : Int) := by
    rw [Ptr.add_int, Buf.ptr_offset, htok]
    push_cast; ring_nf
  rw [hpx]
  refine safe_load1_buf d.logits (by omega) hpre.1
    (fun z hz => Or.inr (Buf.span_sub_elems d.logits (S := rowSet d r) (j := r * d.stride + tokOf d r)
      ⟨tokOf d r, htV', rfl⟩ hz)) fun x hx => ?_
  have hx' : x = xOf d r := by
    have := hpre.2.2.2.1 (r * d.stride + tokOf d r) (by omega)
    rw [this] at hx
    exact (Option.some.inj hx).symm
  subst hx'
  -- the loop
  obtain ⟨hrange, htrips⟩ := range_ok hd
  have hloop : Safe E (Rp d r)
      (iterW (fun _ => ∅) (tl.tripCount (0 : I32).toInt (IntV.ofNat d.V : I32).toInt
        (IntV.ofNat d.BLOCK : I32).toInt) ∪ Wp d r) (fun _ m => Qp d init r m) init
      (tl.forRange (0 : I32) (IntV.ofNat d.V) (IntV.ofNat d.BLOCK : I32) (0 : I32)
        (tripBody d (d.logits.ptr ((r * d.stride : Nat) : Int)) (xOf d r)) >>= fun n =>
        tl.convert .i32 .i64 (Tensor.scalar n) >>= fun r3 =>
        tl.store1 (d.out.ptr 0 + r1.item) r3.item) := by
    have h0 : init = init ∧ (0 : I32).toInt = (cnt d r 0 : Int) :=
      ⟨rfl, by rw [toInt_zero_i32, cnt_zero]; rfl⟩
    refine safe_forRange (I := fun t n m => m = init ∧ n.toInt = (cnt d r t : Int))
      (Wt := fun _ => ∅) (Wk := Wp d r) hrange ?_ ?_ h0 ?_ ?_
    · intro t u _ _; exact Set.empty_disjoint _
    · rw [iterW_empty]; exact Set.empty_disjoint _
    · intro t ht n m₁ hI
      rw [htrips] at ht
      obtain ⟨rfl, hn⟩ := hI
      have hBv : (IntV.ofNat d.BLOCK : I32).toInt = (d.BLOCK : Int) :=
        IntV.I32.toInt_ofNat (by have := hd.blk_le; omega)
      have hi : (IntV.ofInt ((0 : I32).toInt + (t : Int) * (IntV.ofNat d.BLOCK : I32).toInt) : I32).toInt
          = ((t * d.BLOCK : Nat) : Int) := by
        rw [toInt_zero_i32, hBv, Int.zero_add]
        have := trip_lt hd ht
        have := hd.V_blk
        rw [show (t : Int) * d.BLOCK = ((t * d.BLOCK : Nat) : Int) by push_cast; ring]
        exact IntV.I32.toInt_ofInt ⟨by omega, by omega⟩
      exact safe_trip hd hpre E hr ht hi hn
    · intro n m₁ hI
      rw [htrips] at hI
      obtain ⟨rfl, hn⟩ := hI
      rw [cnt_final hd r] at hn
      refine safe_convert (fun _ => ConvRel.exists_i32_i64 _) fun r3 hr3 => ?_
      have h3 : r3.item.toInt = (rank d r : Int) := (ConvRel.toInt_i32_i64 (hr3 ())).trans hn
      have hpo : d.out.ptr 0 + r1.item = d.out.ptr (r : Int) := by
        rw [Ptr.add_int, Buf.ptr_offset, h1, Int.zero_add]
      rw [hpo]
      refine safe_last ?_
      refine safe_store1 (Buf.writable d.out hpre.2.2.1 (lt_of_lt_of_le hr hd.out_len))
        (Buf.span_sub_elems d.out (S := {r}) (j := r) rfl) fun m' hread hframe => ?_
      exact safe_ret ⟨r3.item, h3, hread, hframe⟩
  rw [iterW_empty, Set.empty_union] at hloop
  exact hloop

/-! ## Footprints -/

/-- A program's write footprint lies in the output's first `B` elements. -/
theorem Wp_sub {d : Inst} {i : Nat} (hi : i < d.B) : Wp d i ⊆ d.out.elems {k | k < d.B} :=
  d.out.elems_mono (by intro k hk; rw [Set.mem_singleton_iff.mp hk]; exact hi)

/-- Distinct programs write distinct elements. -/
theorem disjoint_Wp_Wp {d : Inst} {i j : Nat} (hij : i ≠ j) : Disjoint (Wp d i) (Wp d j) := by
  have hcast : (i : Int) ≠ (j : Int) := by exact_mod_cast hij
  rw [Set.disjoint_left]
  rintro x ⟨k, hk, hx⟩ ⟨k', hk', hx'⟩
  rw [Set.mem_singleton_iff] at hk hk'
  rw [hk] at hx
  rw [hk'] at hx'
  exact Set.disjoint_left.mp (Buf.span_disjoint d.out hcast) hx hx'

/-- No program writes what any program reads. -/
theorem disjoint_Wp_Rp {d : Inst} (hd : d.ok) {i j : Nat} (hi : i < d.B) (hj : j < d.B) :
    Disjoint (Wp d i) (Rp d j) := by
  rw [Set.disjoint_left]
  intro x hxW hxR
  have hxB : x ∈ d.out.bytes :=
    Buf.elems_sub_bytes d.out (by
      intro k hk
      rw [Set.mem_singleton_iff.mp hk]
      exact lt_of_lt_of_le hi hd.out_len) hxW
  rcases hxR with hxt | hxl
  · exact Set.disjoint_left.mp hd.out_tok hxB
      (Buf.elems_sub_bytes d.tok (by
        intro k hk
        rw [Set.mem_singleton_iff.mp hk]
        exact lt_of_lt_of_le hj hd.tok_len) hxt)
  · exact Set.disjoint_left.mp hd.out_logits hxB
      (Buf.elems_sub_bytes d.logits (by
        rintro k ⟨q, hq, rfl⟩
        have := hd.logits_len j hj
        omega) hxl)

/-! ## The claim -/

/-- A program index of the launch is a row. -/
theorem idx_lt {d : Inst} (i : Fin (launch d).size) : (i : Nat) < d.B :=
  lt_of_lt_of_eq i.isLt (Launch.size_grid1 _ _ _)

/-- **The claim.** Program `r` is safe on its own (`safe_program`) with the footprints `Rp d r`
and `Wp d r`; distinct programs write distinct elements of the output, which no program reads. -/
theorem main : claim.Holds := by
  intro P _ d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (fun i => Rp d i.val) (fun i => Wp d i.val)
    (fun i m => Qp d init i.val m) (launch_valid hd) (fun i => ?_) ?_ ?_ ?_
  · have hi := idx_lt i
    show Safe _ _ _ _ _ (bodyP d (tl.program_id ((launch d).ctx i) 0))
    rw [pid_eq d i hi]
    exact safe_program hd hpre ((launch d).env P) hi
  · intro i j hij
    have hne : (i : Nat) ≠ (j : Nat) := fun h => hij (Fin.ext h)
    rw [Set.disjoint_union_right]
    exact ⟨disjoint_Wp_Rp hd (idx_lt i) (idx_lt j), disjoint_Wp_Wp hne⟩
  · intro i
    exact (disjoint_Wp_Rp hd (idx_lt i) (idx_lt i)).symm
  · intro fin hloc hfr
    refine ⟨?_, ?_⟩
    · intro r hr
      obtain ⟨m, ⟨v, hv, hread, -⟩, hframe⟩ :=
        hloc ⟨r, (Launch.size_grid1 d.B [[d.BLOCK]] true).symm ▸ hr⟩
      refine ⟨v, ?_, hv⟩
      rw [Mem.read_congr (m₂ := m)
        (fun x hx => hframe x (Or.inr (Buf.span_sub_elems d.out (S := {r}) (j := r) rfl hx)))]
      exact hread
    · intro z hz
      refine hfr z ?_
      intro i hzi
      exact hz (Wp_sub (idx_lt i) hzi)

end Claims.LogprobRanks

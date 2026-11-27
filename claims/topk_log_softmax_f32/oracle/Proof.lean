import Proof.Walk3

/-!
# The proof of `topk_log_softmax_f32`

Program `p` of the grid handles row `p`: a max loop and a sum loop over the row's logits in blocks
of `1024`, `lse = tl.log(se)`, and a gather loop over the row's ids in blocks of
`TOPK_BLOCK_SIZE`, which stores `(x_id − max) − lse`.

1. `Proof.Exp`, `Proof.Max`, `Proof.Lane`: constants, the exact quantities `ER`, `ZR` of a row and
   one lane's `exp` error (`lane_bound`);
2. `Proof.Sum`, `Proof.Num`, `Proof.Out`: the row sum as one tree over all trips (`SumInv`), its
   error (`sum_bound`), `lse`'s (`lse_bound`) and the outputs' (`row_num`);
3. `Proof.Lanes`, `Proof.Walk1`–`Walk3`: the lanes and addresses, and one trip of each loop
   walked with the rules of `TVA.Proof.Rules` (`safe_maxTrip`, `safe_sumTrip`, `safe_gatherTrip`);
4. `safe_program`: the body, the three loops by `safe_forRange`, with post-condition `Qp`;
5. `main`: rows' outputs are disjoint from each other and from every read, and `row_num` turns
   `Qp` into `Post`.
-/
namespace Claims.TopkLogSoftmax

open TVA

/-- The trip count of the vocabulary loops. -/
def nT (V : Nat) : Nat := (V + 1024 - 1) / 1024

/-- What program `p` leaves: the row's max `mv` (a finite logit at least every logit), the sum
`se` (`SumInv` after `nT` trips) and `lse = log(se)` on a `RowOk` row, and at every output of
the row a value `OutRel` describes. -/
def Qp (P : Platform) (d : Inst) (init : Mem) (p : Nat) (m : Mem) : Prop :=
  ∃ mv se lse : F32,
    (RowOk (xsOf d init p) d.vocab →
      (∃ i0, i0 < d.vocab ∧ mv = xsOf d init p i0 ∧ mv.isFinite = true ∧
        ∀ k, k < d.vocab → (xsOf d init p k).toRat ≤ mv.toRat) ∧
      SumInv P d.vocab (xsOf d init p) mv (nT d.vocab) se ∧ P.math .log .binary32 se lse) ∧
    ∀ j, j < d.topk → ∃ c : F32,
      m.read .f32 (d.out.ptr (outIdx d p j)).addr = some c ∧ OutRel d init p mv lse j c

/-- After the max loop, on a `RowOk` row, the max is a finite logit of the row. -/
lemma maxFacts {xs : Nat → F32} {V : Nat} {mv : F32} (hrow : RowOk xs V)
    (hI : MaxInv xs V (nT V) mv) :
    ∃ i0, i0 < V ∧ mv = xs i0 ∧ mv.isFinite = true ∧ ∀ k, k < V → (xs k).toRat ≤ mv.toRat := by
  obtain ⟨hnan, hle, hmem⟩ := hI hrow
  have hle' : ∀ k, k < V → (xs k).toRat ≤ mv.toRat := fun k hk => hle k hk (by unfold nT; omega)
  obtain ⟨k0, hk0, hk0f⟩ := hrow.2.1
  have hnot : mv ≠ FloatV.negInf := by
    intro hneg
    have h1 := hle' k0 hk0
    rw [hneg, negInf_toRat] at h1
    have h2 := FloatV.abs_toRat_le_maxVal b32_reg hk0f
    rw [abs_le] at h2
    linarith [maxVal_lt_two_pow, h2.1]
  rcases hmem with h | ⟨i0, hi0, h⟩
  · exact absurd h hnot
  · refine ⟨i0, hi0, h, ?_, hle'⟩
    rcases hrow.1 i0 hi0 with hf | hn
    · rw [h]; exact hf
    · exact absurd (h.trans hn) hnot

set_option maxHeartbeats 800000 in
/-- **One program** is safe: it reads its row of logits and ids, writes its row of outputs, and
ends with `Qp`. -/
theorem safe_program {d : Inst} (hd : d.ok) {P : Platform} (hP : Device P) {init : Mem}
    (hpre : Pre d init) (i : Fin (launch d).size) :
    Safe ((launch d).env P) (Rd d i) (d.out.elems (rowOut d i)) (fun _ m => Qp P d init i m) init
      ((model d).body ((launch d).ctx i)) := by
  have hp : (i : Nat) < d.batch := lt_of_lt_of_eq i.isLt (launch_size d)
  set p := (i : Nat) with hpdef
  have hp31 : p < 2 ^ 31 := lt_of_lt_of_le hp (le_of_lt hd.batch_lt)
  have hV19 := hd.vocab_le
  have hV1 : d.vocab + 1024 ≤ 2 ^ 31 := by omega
  obtain ⟨e, he, hTB⟩ := hd.tb
  have hTBle : d.TOPK_BLOCK_SIZE ≤ 1024 := by
    rw [hTB]; exact le_trans (Nat.pow_le_pow_right (by norm_num) he) (by norm_num)
  have hTBpos : 0 < d.TOPK_BLOCK_SIZE := by rw [hTB]; exact Nat.two_pow_pos e
  have hk := hd.topk_lt
  rw [body_eq, tl.program_id, launch_pid]
  -- `req_idx`
  refine safe_convert (fun _ => ⟨_, ConvRel.iff_cast_i32_i64.mpr rfl⟩) fun rq hrq => ?_
  have hreq : rq.item.toInt = (p : Int) := (cast64 (hrq ())).trans (IntV.I32.toInt_ofNat hp31)
  -- `logits_stride` to `int64`
  refine safe_convert (fun _ => ⟨_, ConvRel.iff_cast_i32_i64.mpr rfl⟩) fun st hst => ?_
  have hS : st.item.toInt = ((d.logits_stride : Nat) : Int) :=
    (cast64 (hst ())).trans (IntV.I32.toInt_ofNat hd.stride_lt)
  dsimp only
  rw [ptr_add64 d.logits (mul64 hreq hS hp31 hd.stride_lt)]
  have hrange := range_ok (n := d.vocab) (B := 1024) (by norm_num) hV1 (by norm_num)
  have htrips := trips_eq (n := d.vocab) (B := 1024) (by norm_num) hV1 (by norm_num)
  -- the max loop
  refine Safe.mono (W' := d.out.elems (rowOut d p))
    (safe_forRange hrange (fun t mv m => m = init ∧ MaxInv (xsOf d init p) d.vocab t mv)
    (fun _ => ∅) (Wk := d.out.elems (rowOut d p)) (by simp) (by simp) ⟨rfl, maxInv_zero _ _⟩
    ?_ ?_) (fun _ _ h => h) (by simp)
  · rintro t ht mv m₁ ⟨hm, hI⟩
    rw [hm]
    rw [htrips] at ht
    exact safe_maxTrip hd hpre.1 hp (by omega) (iv_eq (by omega) (by norm_num)) hI
      Set.subset_union_left
  rintro mv m₁ ⟨hm, hI⟩
  rw [hm]
  rw [htrips] at hI
  -- the sum loop
  refine Safe.mono (W' := d.out.elems (rowOut d p)) (safe_forRange hrange
    (fun t se m => m = init ∧
      (RowOk (xsOf d init p) d.vocab → SumInv P d.vocab (xsOf d init p) mv t se))
    (fun _ => ∅) (Wk := d.out.elems (rowOut d p)) (by simp) (by simp)
    ⟨rfl, fun _ => sumInv_zero _ _ _ _⟩ ?_ ?_) (fun _ _ h => h) (by simp)
  · rintro t ht se m₁ ⟨hm, hI⟩
    rw [hm]
    rw [htrips] at ht
    exact safe_sumTrip hd (hP.mathTotal .exp .binary32) hP.sumDepth hpre.1 hp (by omega)
      (iv_eq (by omega) (by norm_num)) hI Set.subset_union_left
  rintro se m₁ ⟨hm, hsum⟩
  rw [hm]
  rw [htrips] at hsum
  -- `lse`
  refine safe_math (fn := .log) (fun _ => hP.mathTotal .log .binary32 _) fun l hl => ?_
  -- the gather loop
  have hrange2 := range_ok (n := d.topk) (B := d.TOPK_BLOCK_SIZE) hTBpos hk (by omega)
  have htrips2 := trips_eq (n := d.topk) (B := d.TOPK_BLOCK_SIZE) hTBpos hk (by omega)
  refine safe_last ?_
  refine Safe.mono (W' := d.out.elems (rowOut d p)) (safe_forRange hrange2
    (fun t _ m => GInv d init p mv l.item t m)
    (fun t => d.out.elems (outBlk d p t)) (Wk := ∅) ?_ (by simp)
    ⟨fun _ _ => rfl, fun x hx => hx, fun j _ hj => by omega⟩ ?_ ?_) (fun _ _ h => h) ?_
  · intro t u htu _
    refine Buf.elems_disjoint d.out (Set.disjoint_left.mpr ?_)
    rintro x ⟨a, ha, -, rfl⟩ ⟨b, hb, -, hx⟩
    have : (t + 1) * d.TOPK_BLOCK_SIZE ≤ u * d.TOPK_BLOCK_SIZE := Nat.mul_le_mul_right _ htu
    rw [Nat.add_mul, Nat.one_mul] at this
    omega
  · intro t ht _ m₁ hI
    rw [htrips2] at ht
    have htB : t * d.TOPK_BLOCK_SIZE < d.topk := by
      by_contra hcon
      have : (d.topk + d.TOPK_BLOCK_SIZE - 1) / d.TOPK_BLOCK_SIZE ≤ t := by
        rw [Nat.div_le_iff_le_mul_add_pred hTBpos]
        have : t * d.TOPK_BLOCK_SIZE = d.TOPK_BLOCK_SIZE * t := Nat.mul_comm _ _
        omega
      omega
    exact safe_gatherTrip hd hpre hp htB (iv_eq (by omega) (by omega)) hreq hI
  · intro _ m₁ hG
    rw [htrips2] at hG
    refine safe_ret ⟨mv, se, l.item, fun hrow => ⟨maxFacts hrow hI, hsum hrow, hl ()⟩,
      fun j hj => hG.2.2 j hj ?_⟩
    have h1 := Nat.div_add_mod (d.topk + d.TOPK_BLOCK_SIZE - 1) d.TOPK_BLOCK_SIZE
    have h2 := Nat.mod_lt (d.topk + d.TOPK_BLOCK_SIZE - 1) hTBpos
    have : (d.topk + d.TOPK_BLOCK_SIZE - 1) / d.TOPK_BLOCK_SIZE * d.TOPK_BLOCK_SIZE =
        d.TOPK_BLOCK_SIZE * ((d.topk + d.TOPK_BLOCK_SIZE - 1) / d.TOPK_BLOCK_SIZE) :=
      Nat.mul_comm _ _
    omega
  · rintro x (⟨t, -, hx⟩ | hx)
    · exact Buf.elems_mono d.out outBlk_sub hx
    · exact absurd hx (Set.notMem_empty x)

/-- Two rows' outputs are disjoint. -/
lemma rowOut_disjoint {d : Inst} {p q : Nat} (hpq : p ≠ q) :
    Disjoint (d.out.elems (rowOut d p)) (d.out.elems (rowOut d q)) := by
  refine Buf.elems_disjoint d.out (Set.disjoint_left.mpr ?_)
  rintro x ⟨a, ha, rfl⟩ ⟨b, hb, hx⟩
  rcases Nat.lt_or_gt_of_ne hpq with h | h
  · have : (p + 1) * d.topk ≤ q * d.topk := Nat.mul_le_mul_right _ h
    rw [Nat.add_mul, Nat.one_mul] at this
    omega
  · have : (q + 1) * d.topk ≤ p * d.topk := Nat.mul_le_mul_right _ h
    rw [Nat.add_mul, Nat.one_mul] at this
    omega

/-- **The claim.** Each program is safe on its row (`safe_program`); the rows' outputs are disjoint
from each other and from every read, and `row_num` turns each program's `Qp` into `Post`. -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  have hsz : ∀ i : Fin (launch d).size, (i : Nat) < d.batch :=
    fun i => lt_of_lt_of_eq i.isLt (launch_size d)
  refine correct_of_safe (fun i => Rd d i) (fun i => d.out.elems (rowOut d i))
    (fun i m => Qp P d init i m) (launch_valid hd) (fun i => safe_program hd hP hpre i) ?_ ?_ ?_
  · intro i j hij
    have hij' : (i : Nat) ≠ (j : Nat) := fun h => hij (Fin.ext h)
    exact Disjoint.union_right (out_Rd hd (hsz i) (hsz j)) (rowOut_disjoint hij')
  · intro i
    exact (out_Rd hd (hsz i) (hsz i)).symm
  · intro fin hloc hfr
    have hV19 := hd.vocab_le
    refine ⟨fun r hr => ?_, fun z hz => hfr z fun i hzi =>
      hz (Buf.elems_mono d.out (rowOut_sub (hsz i)) hzi)⟩
    obtain ⟨m, hQ, hag⟩ := hloc ⟨r, lt_of_lt_of_eq hr (launch_size d).symm⟩
    obtain ⟨mv, se, lse, hrowf, houts⟩ := hQ
    have hread : ∀ j, j < d.topk →
        fin.read .f32 (d.out.ptr (outIdx d r j)).addr = m.read .f32 (d.out.ptr (outIdx d r j)).addr :=
      fun j hj => Mem.read_congr fun x hx =>
        hag x (Or.inr (Buf.span_sub_elems d.out (S := rowOut d r) ⟨j, hj, rfl⟩ hx))
    refine ⟨xsOf d init r, fun j => tidOf d init r j, fun k hk => read_xs hd hpre.1 hr hk,
      fun j hj => read_tid hd hpre.2.1 hr hj, ?_, ?_⟩
    · intro j hj
      obtain ⟨c, hc, -⟩ := houts j hj
      exact ⟨c, by rw [hread j hj]; exact hc⟩
    · intro hrow
      obtain ⟨⟨i0, hi0, hmv, hmvfin, hle⟩, hs, hlog⟩ := hrowf hrow
      obtain ⟨-, sh, hsh, hall⟩ := row_num hP.expF32 hP.expF32NegInf hP.logF32 hV19
        (T := nT d.vocab) (by unfold nT; omega) (by unfold nT; omega) hrow hi0 hmv hmvfin hle hs
        hlog
      refine ⟨sh, hsh, fun j hj => ?_⟩
      obtain ⟨c, hc, dd, hdd, hoc⟩ := houts j hj
      have hid : (tidOf d init r j).toInt.toNat < d.vocab := by
        have h := hpre.2.2.2 r j hr hj _ (read_tid hd hpre.2.1 hr hj)
        omega
      obtain ⟨h1, h2⟩ := hall _ hid dd c hdd hoc
      exact ⟨c, by rw [hread j hj]; exact hc, h1, h2⟩

end Claims.TopkLogSoftmax

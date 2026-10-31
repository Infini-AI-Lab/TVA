import GroupedMatmulTma.Tile

/-!
# `grouped_matmul_tma_f16`: one program

`safe_program`: program `p < NUM_SM` of the grid, run alone from the initial memory, is safe
(`TVA.Safe`): it reads only the tables and the input matrices (`R`), writes only the tiles it owns
(`W p`), each byte once, and leaves in each of them what `OutSpec` says (`Q`). The proof walks
the model: the loop of line 212 with the invariant `OI`, the `if` of line 220, the loads and
descriptors of lines 222–248, the `while` loop of line 251 with the invariant `WI` and the variant
`mu`, whose body is `safe_tile`.
-/

namespace Kernels.GroupedMatmulTma

open TVA

/-! ## The launch and the program's result -/

theorem launch_size (d : Inst) : (launch d).size = d.NUM_SM := by
  rw [launch, Launch.size_grid1]

theorem launch_pid (d : Inst) {p : Nat} (hp : p < (launch d).size) : ((launch d).ctx p).pid 0 = p := by
  rw [launch_size] at hp
  simp only [Launch.ctx, launch, Launch.grid1, Matrix.cons_val_zero]
  exact Nat.mod_eq_of_lt hp

/-- What program `p` leaves in memory `m`: every tile it owns (global index `p` modulo `NUM_SM`)
as `OutSpec` says. -/
def Q (d : Inst) (init : Mem) (p : Nat) (m : Mem) : Prop :=
  ∀ g t, g < d.group_size → t < tiles d g → (start d g + t) % d.NUM_SM = p → TileOK d init g t m

/-- The invariant of the loop of line 212 before trip `g`: `last_problem_end` is `start g`;
`tile_idx` holds some `y ≡ p` modulo `NUM_SM` within one stride from `start g`; the memory holds
`ProgInv` at `y`. -/
def OI (d : Inst) (init : Mem) (p g : Nat) (st : I32 × I32) (m : Mem) : Prop :=
  st.2.toInt = start d g ∧ ∃ y : Nat, st.1.toInt = y ∧ y % d.NUM_SM = p ∧ start d g ≤ y ∧
    y < start d g + d.NUM_SM ∧ ProgInv d init p y m

/-! ## Table elements -/

/-- `ptr + g * 3`, `+ 1`, `+ 2` (lines 214–216, 222–224) at `g = s`: elements `3s`, `3s + 1`,
`3s + 2` of the table. -/
theorem ptr_tab3 {dt : DType} (b : Buf dt) {gv : I32} {s : Nat} (hg : gv.toInt = s)
    (hs : 3 * s + 2 < 2 ^ 31) :
    b.ptr 0 + gv * (IntV.ofInt 3 : I32) = b.ptr ↑(3 * s) ∧
      b.ptr 0 + gv * (IntV.ofInt 3 : I32) + (IntV.ofInt 1 : I32) = b.ptr ↑(3 * s + 1) ∧
      b.ptr 0 + gv * (IntV.ofInt 3 : I32) + (IntV.ofInt 2 : I32) = b.ptr ↑(3 * s + 2) := by
  have h3 : (gv * (IntV.ofInt 3 : I32)).toInt = ((3 * s : Nat) : Int) := by
    rw [i32_mul hg (i32_int (by norm_num) (by norm_num))] <;> push_cast <;> omega
  have e0 : b.ptr 0 + gv * (IntV.ofInt 3 : I32) = b.ptr ↑(3 * s) := by
    show (b.ptr 0).offset (gv * (IntV.ofInt 3 : I32)).toInt = _
    rw [h3, Buf.ptr_offset, zero_add]
  refine ⟨e0, ?_, ?_⟩
  · rw [e0]
    show (b.ptr ↑(3 * s)).offset (IntV.ofInt 1 : I32).toInt = _
    rw [i32_one, Buf.ptr_offset]; push_cast; rfl
  · rw [e0]
    show (b.ptr ↑(3 * s)).offset (IntV.ofInt 2 : I32).toInt = _
    rw [i32_int (by norm_num) (by norm_num), Buf.ptr_offset]; push_cast; rfl

/-- `ptr + g` (lines 226–228) at `g = s`: element `s` of the table. -/
theorem ptr_tab1 {dt : DType} (b : Buf dt) {gv : I32} {s : Nat} (hg : gv.toInt = s) :
    b.ptr 0 + gv = b.ptr ↑s := by
  show (b.ptr 0).offset gv.toInt = _
  rw [hg, Buf.ptr_offset, zero_add]

/-- An `int32` value `n` converted to `int64` (lines 232–233 and the others) is `n`. -/
theorem conv64 {P : Platform} {x : I32} {n : Nat} (hx : x.toInt = n) (hn : n < 2 ^ 31)
    {r : Tensor [] I64} (h : ∀ i, ConvRel P .i32 .i64 ((Tensor.scalar x).get i) (r.get i)) :
    r.item.toInt = n := by
  have e : r.get () = IntV.ofInt x.toInt := h ()
  show (r.get ()).toInt = n
  rw [e, hx]
  exact IntV.toInt_ofInt (by norm_num) (by unfold IntV.SFits; constructor <;> norm_num; omega)

theorem nonneg2 {w : Nat} {a b : IntV true w} (ha : 0 ≤ a.toInt) (hb : 0 ≤ b.toInt) :
    ∀ x ∈ [a, b], 0 ≤ x.toInt := by
  intro x hx
  simp only [List.mem_cons, List.mem_nil_iff, or_false] at hx
  rcases hx with rfl | rfl
  · exact ha
  · exact hb

/-! ## One program -/

/-- **One program.** Program `p < NUM_SM`, run alone from `init`, is safe: it reads only the
tables and the input matrices, writes only the tiles it owns, and leaves there what `OutSpec`
says. -/
theorem safe_program {P : Platform} (hdot : Rules.Num.DotF16 P) (htot : Rules.Num.DotTotal P cfg16)
    {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {p : Nat} (hp : p < d.NUM_SM) :
    Safe ((launch d).env P) (R d) (W d p) (fun _ m => Q d init p m) init
      ((model d).body ((launch d).ctx p)) := by
  have hpid : ((launch d).ctx p).pid 0 = p := launch_pid d (by rw [launch_size]; exact hp)
  have hS := hd.SM_pos
  have hGt := hd.tiles_lt
  have hG3 := hd.G_le
  simp only [model, grouped_matmul_tma_kernel]
  rw [show tl.program_id ((launch d).ctx p) 0 = IntV.ofNat p by simp [tl.program_id, hpid]]
  -- line 212
  have hGv : (IntV.ofNat d.group_size : I32).toInt = d.group_size := i32_nat (by omega)
  have hrange : tl.RangeOk (IntV.ofInt 0 : I32) (IntV.ofNat d.group_size) (IntV.ofInt 1) := by
    unfold tl.RangeOk Math.Float.Format.intRange
    rw [i32_zero, i32_one, hGv]; norm_num; omega
  have htc : tl.tripCount (IntV.ofInt 0 : I32).toInt (IntV.ofNat d.group_size : I32).toInt
      (IntV.ofInt 1 : I32).toInt = d.group_size := by
    rw [i32_zero, i32_one, hGv]; simp [tl.tripCount]
  refine Safe.mono (safe_forRange (Wk := ∅) hrange (OI d init p) (Wt d p) ?_ (Set.disjoint_empty _) ?_ ?_ ?_)
    (fun _ _ h => h) (by rw [htc, Set.union_empty]; exact subset_rfl)
  · rw [htc]; intro u v huv hv; exact Wt_disj hd p huv hv
  · -- before the loop (lines 210–211)
    refine ⟨by rw [i32_zero, start_zero]; rfl, p, i32_nat (by omega), Nat.mod_eq_of_lt hp,
      by rw [start_zero]; omega, by rw [start_zero]; omega, ProgInv_start d init hp⟩
  · -- trip `s` of the loop of line 212 (lines 214–278)
    rw [htc]
    rintro s hs ⟨ti, lpe⟩ m₁ ⟨hlpe, y, hy, hyp, hlo, hhi, hinv⟩
    dsimp only at hlpe hy ⊢
    have hs3 : 3 * s + 2 < 2 ^ 31 := by omega
    have hgv : (IntV.ofInt ((IntV.ofInt 0 : I32).toInt + (s : Int) * (IntV.ofInt 1 : I32).toInt) : I32).toInt
        = s := i32_trip (by omega)
    obtain ⟨eS0, eS1, eS2⟩ := ptr_tab3 d.group_gemm_sizes hgv hs3
    obtain ⟨eL0, eL1, eL2⟩ := ptr_tab3 d.g_lds hgv hs3
    rw [eS1, eS2, eS0, eL1, eL2, eL0, ptr_tab1 d.group_a_ptrs hgv, ptr_tab1 d.group_b_ptrs hgv,
      ptr_tab1 d.group_c_ptrs hgv]
    obtain ⟨rM, rN, rK, rlda, rldb, rldc, rA, rB, rC⟩ := hpre.2.2.2.2.2.2 s hs
    obtain ⟨hvA, hvB, hvC⟩ := hpre.2.2.2.2.2.1 s hs
    have hrd : ∀ {dt : DType} (b : Buf dt) (i : Nat), Mem.span dt (b.ptr ↑i).addr ⊆ tabEl d →
        m₁.read dt (b.ptr ↑i).addr = init.read dt (b.ptr ↑i).addr :=
      fun b i h => Mem.read_congr fun x hx => ProgInv_R hd hinv x (Or.inl (h hx))
    have hRt : ∀ {X : Set Int}, X ⊆ tabEl d → X ⊆ R d := fun h => h.trans Set.subset_union_left
    have hSz : ∀ i, i < 3 * d.group_size → Mem.span .i32 (d.group_gemm_sizes.ptr ↑i).addr ⊆ tabEl d :=
      fun i hi x hx => Or.inl (Or.inr (d.group_gemm_sizes.span_sub_elems (S := {e | e < 3 * d.group_size}) hi hx))
    have hLd : ∀ i, i < 3 * d.group_size → Mem.span .i32 (d.g_lds.ptr ↑i).addr ⊆ tabEl d :=
      fun i hi x hx => Or.inr (d.g_lds.span_sub_elems (S := {e | e < 3 * d.group_size}) hi hx)
    have hPa : Mem.span .i64 (d.group_a_ptrs.ptr ↑s).addr ⊆ tabEl d := fun x hx =>
      Or.inl (Or.inl (Or.inl (Or.inl (d.group_a_ptrs.span_sub_elems (S := {e | e < d.group_size}) hs hx))))
    have hPb : Mem.span .i64 (d.group_b_ptrs.ptr ↑s).addr ⊆ tabEl d := fun x hx =>
      Or.inl (Or.inl (Or.inl (Or.inr (d.group_b_ptrs.span_sub_elems (S := {e | e < d.group_size}) hs hx))))
    have hPc : Mem.span .i64 (d.group_c_ptrs.ptr ↑s).addr ⊆ tabEl d := fun x hx =>
      Or.inl (Or.inl (Or.inr (d.group_c_ptrs.span_sub_elems (S := {e | e < d.group_size}) hs hx)))
    -- lines 214–216
    refine safe_load1 ((hrd _ _ (hSz _ (by omega))).trans rM) (hRt (hSz _ (by omega))) ?_
    refine safe_load1 ((hrd _ _ (hSz _ (by omega))).trans rN) (hRt (hSz _ (by omega))) ?_
    refine safe_load1 ((hrd _ _ (hSz _ (by omega))).trans rK) (hRt (hSz _ (by omega))) ?_
    have hMs := hd.M_le s hs
    have hNs := hd.N_le s hs
    have hKs := hd.K_le s hs
    have hend := start_add_le_total d hs
    -- lines 217–218
    have hm1 : (IntV.ofNat (d.M s) + (IntV.ofNat (128 - 1) : I32) : I32).toInt = ((d.M s + 127 : Nat) : Int) :=
      IntV.I32.add_nat (i32_nat (by omega)) (i32_nat (by norm_num)) (by omega)
    refine safe_idiv1 (by rw [hm1]; omega) (by rw [i32_nat (by norm_num)]; norm_num) fun rm hrm => ?_
    have hnm : rm.item.toInt = (nM d s : Int) := by
      rw [hrm, hm1, i32_nat (by norm_num)]; unfold nM; push_cast; omega
    have hn1 : (IntV.ofNat (d.N s) + (IntV.ofNat (128 - 1) : I32) : I32).toInt = ((d.N s + 127 : Nat) : Int) :=
      IntV.I32.add_nat (i32_nat (by omega)) (i32_nat (by norm_num)) (by omega)
    refine safe_idiv1 (by rw [hn1]; omega) (by rw [i32_nat (by norm_num)]; norm_num) fun rn hrn => ?_
    have hnn : rn.item.toInt = (nN d s : Int) := by
      rw [hrn, hn1, i32_nat (by norm_num)]; unfold nN; push_cast; omega
    -- line 219
    have hnt : (rm.item * rn.item).toInt = (tiles d s : Int) := by
      rw [tiles_eq]; exact IntV.I32.mul_nat hnm hnn (by rw [← tiles_eq]; omega)
    have hsum : (lpe + rm.item * rn.item).toInt = ((start d s + tiles d s : Nat) : Int) :=
      IntV.I32.add_nat hlpe hnt (by omega)
    -- line 220
    rw [i32_le hlpe hy, i32_lt hy hsum]
    by_cases hlt : y < start d s + tiles d s
    · rw [ite_eq_left (by simp only [Bool.and_eq_true, decide_eq_true_eq]; omega)]
      have ht0 : 0 < tiles d s := by omega
      have hnN0 : 0 < nN d s := by
        rcases Nat.eq_zero_or_pos (nN d s) with h | h
        · rw [tiles_eq, h, Nat.mul_zero] at ht0; omega
        · exact h
      have hnM0 : 0 < nM d s := by
        rcases Nat.eq_zero_or_pos (nM d s) with h | h
        · rw [tiles_eq, h, Nat.zero_mul] at ht0; omega
        · exact h
      have hM0 : 0 < d.M s := by unfold nM at hnM0; omega
      have hN0 : 0 < d.N s := by unfold nN at hnN0; omega
      obtain ⟨hlda, -⟩ := hd.lda_ok s hs
      obtain ⟨hldb, -⟩ := hd.ldb_ok s hs
      obtain ⟨hldc, -, hNldc⟩ := hd.ldc_ok s hs
      -- lines 222–224
      refine safe_load1 ((hrd _ _ (hLd _ (by omega))).trans rlda) (hRt (hLd _ (by omega))) ?_
      refine safe_load1 ((hrd _ _ (hLd _ (by omega))).trans rldb) (hRt (hLd _ (by omega))) ?_
      refine safe_load1 ((hrd _ _ (hLd _ (by omega))).trans rldc) (hRt (hLd _ (by omega))) ?_
      -- lines 226–228
      refine safe_load1 ((hrd _ _ hPa).trans rA) (hRt hPa) ?_
      refine safe_load1 ((hrd _ _ hPb).trans rB) (hRt hPb) ?_
      refine safe_load1 ((hrd _ _ hPc).trans rC) (hRt hPc) ?_
      -- the matrices' addresses
      have hbase : ∀ (b : Buf .f16), b.Valid init → 0 < b.len →
          tl.intToPtr .f16 (IntV.ofInt b.base : I64) = (⟨b.base⟩ : Ptr .f16) := by
        intro b hv hl
        refine intToPtr_ofInt hv.1 ?_
        have h1 := hv.2.1
        have h2 : DType.bytes .f16 = 2 := rfl
        rw [h2] at h1
        have : (1 : Int) ≤ b.len := by exact_mod_cast hl
        omega
      have hone : (IntV.ofInt 1 : I32).toInt = ((1 : Nat) : Int) := by rw [i32_one]; rfl
      -- lines 230–235
      refine safe_convert (fun _ => ⟨_, rfl⟩) fun c1 hc1 => ?_
      refine safe_convert (fun _ => ⟨_, rfl⟩) fun c2 hc2 => ?_
      have hc1v := conv64 (i32_nat hlda) hlda hc1
      have hc2v := conv64 hone (by norm_num) hc2
      refine safe_make_tensor_descriptor (nonneg2 (by rw [i32_nat (by omega)]; omega) (by rw [i32_nat (by omega)]; omega))
        (nonneg2 (by rw [hc1v]; omega) (by rw [hc2v]; omega)) ?_
      -- lines 237–242
      refine safe_convert (fun _ => ⟨_, rfl⟩) fun c3 hc3 => ?_
      refine safe_convert (fun _ => ⟨_, rfl⟩) fun c4 hc4 => ?_
      have hc3v := conv64 (i32_nat hldb) hldb hc3
      have hc4v := conv64 hone (by norm_num) hc4
      refine safe_make_tensor_descriptor (nonneg2 (by rw [i32_nat (by omega)]; omega) (by rw [i32_nat (by omega)]; omega))
        (nonneg2 (by rw [hc3v]; omega) (by rw [hc4v]; omega)) ?_
      -- lines 243–248
      refine safe_convert (fun _ => ⟨_, rfl⟩) fun c5 hc5 => ?_
      refine safe_convert (fun _ => ⟨_, rfl⟩) fun c6 hc6 => ?_
      have hc5v := conv64 (i32_nat hldc) hldc hc5
      have hc6v := conv64 hone (by norm_num) hc6
      refine safe_make_tensor_descriptor (nonneg2 (by rw [i32_nat (by omega)]; omega) (by rw [i32_nat (by omega)]; omega))
        (nonneg2 (by rw [hc5v]; omega) (by rw [hc6v]; omega)) ?_
      -- line 251
      have hlenA : 0 < d.K s → 0 < (d.A s).len := fun hK => by
        have := hd.A_len s hs 0 0 hM0 hK; omega
      have hlenB : 0 < d.K s → 0 < (d.B s).len := fun hK => by
        have := hd.B_len s hs 0 0 hN0 hK; omega
      have hlenC : 0 < (d.C s).len := by
        have := hd.C_len s hs 0 0 hM0 hN0; omega
      refine Safe.mono (Safe.seq (W₂ := ∅) (safe_while _ _ _ (WI d init p s) (mu d s) (Wv d p s)
        (fun u v huv => Wv_disj hd p hs huv) ⟨y, hy, hyp, hlo, by rw [start_succ]; omega, hinv⟩ ?_)
        (Set.disjoint_empty _)) (fun _ _ h => h) (by rw [Set.union_empty]; exact Wv_sub d p s _)
      rintro st m₂ ⟨y', hy', hyp', hlo', hhi', hinv'⟩
      refine safe_ret ?_
      rw [i32_le hlpe hy', i32_lt hy' hsum]
      by_cases hlt' : y' < start d s + tiles d s
      · rw [ite_eq_left (by simp only [Bool.and_eq_true, decide_eq_true_eq]; omega)]
        exact safe_tile hdot htot hd hpre hs hlpe hnn (i32_nat (by omega))
          (by simp only [List.map_cons, List.map_nil, i32_nat (show d.M s < 2 ^ 31 by omega),
            i32_nat (show d.K s < 2 ^ 31 by omega), hc1v, hc2v, Int.toNat_natCast])
          (fun hK => hbase _ hvA (hlenA hK))
          (by simp only [List.map_cons, List.map_nil, i32_nat (show d.N s < 2 ^ 31 by omega),
            i32_nat (show d.K s < 2 ^ 31 by omega), hc3v, hc4v, Int.toNat_natCast])
          (fun hK => hbase _ hvB (hlenB hK))
          (by simp only [List.map_cons, List.map_nil, i32_nat (show d.M s < 2 ^ 31 by omega),
            i32_nat (show d.N s < 2 ^ 31 by omega), hc5v, hc6v, Int.toNat_natCast, hbase _ hvC hlenC])
          hy' hlo' hlt' hyp' hinv'
      · -- the end of the group's tiles (line 278)
        rw [ite_eq_right (by simp only [Bool.and_eq_true, decide_eq_true_eq, not_and]; omega)]
        refine safe_ret ⟨by rw [hsum, start_succ], y', hy', hyp', by rw [start_succ]; omega, hhi', hinv'⟩
    · -- the `if` not taken: the group's tiles are below `tile_idx` (line 278)
      rw [ite_eq_right (by simp only [Bool.and_eq_true, decide_eq_true_eq, not_and]; omega)]
      exact safe_ret ⟨by rw [hsum, start_succ], y, hy, hyp, by rw [start_succ]; omega,
        by rw [start_succ]; omega, hinv⟩
  · -- after the loop: every tile the program owns is written
    rw [htc]
    rintro ⟨ti, lpe⟩ m₁ ⟨-, y, -, -, hlo, -, hinv⟩
    exact safe_ret fun g t hg ht hgp => ProgInv_final hinv hlo hg ht hgp

end Kernels.GroupedMatmulTma

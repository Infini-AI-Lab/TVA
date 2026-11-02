import StreamKMatmul.Tail

/-!
# `streamk_matmul_f16`: one program

`safe_iter`: one iteration of the while loop (lines 42–64), from the loop's invariant, with
`int32` values of `tile`, `k_first`, `k_last`, `pid_m`, `pid_n` and the lanes `offs_m`, `offs_n`;
the inner loop (lines 51–57) by `safe_loadDot`; the rest by `safe_storeAdd`.

`safe_program`: a whole program (lines 36–64), alone from `init`, for the theorem `TVA.shared`:
it reads only the matrices `a` and `b` (`R`), writes privately only the tiles it owns (`W`), adds
atomically only into the elements of split tiles (`cells`), and leaves what `Q` says.
-/

namespace Kernels.StreamKMatmul

open TVA

/-! ## The launch -/

theorem launch_size (d : Inst) : (launch d).size = d.num_programs := Launch.size_grid1 _ _ _

theorem launch_pid (d : Inst) {p : Nat} (hp : p < d.num_programs) : ((launch d).ctx p).pid 0 = p := by
  simp only [Launch.ctx, launch, Launch.grid1, Matrix.cons_val_zero]
  exact Nat.mod_eq_of_lt hp

theorem np_lt {d : Inst} (hd : d.ok) : d.num_programs < 2 ^ 31 := by
  have h1 := ipp_pos hd
  have h2 := hd.progs_lt
  have : d.num_programs ≤ d.num_programs * d.iters_per_program := Nat.le_mul_of_pos_right _ h1
  omega

/-! ## The inner loop's start -/

/-- The accumulator before the inner loop (line 50): `+0.0`, exact. -/
theorem acc0 (d : Inst) (init : Mem) (t lo : Nat) : AccInv d init t lo 0 (tl.zeros [128, 128] .f32) := by
  intro x _ va vb _ _
  have h := DotLoop.val?_f32.mp DotLoop.zero_val
  refine ⟨h.1, ?_⟩
  show |(FloatV.ofPat 0 : F32).toRat - partS d va vb lo (lo + 0)| ≤ _
  rw [h.2]
  simp [partS, DotLoop.bound]

/-! ## One iteration -/

/-- **One iteration of the while loop** (lines 42–64) from `it = n < end`: it reads only `a`
and `b`, writes at most the tile it owns, and keeps the loop's invariant with a smaller variant. -/
theorem safe_iter {d : Inst} (hd : d.ok) {init : Mem} {E : Env} (hdot : Rules.Num.DotF16 E.P)
    (htot : Rules.Num.DotTotal E.P { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 })
    (hat : Rules.Num.AtomicAddF32 E.P) (hatot : Rules.Num.AtomicAddTotal E.P Math.Float.Format.binary32)
    {p : Nat} (hp : p < d.num_programs) {nn T e : I32} (hnn : nn.toInt = tilesN d)
    (hT : T.toInt = steps d.K) (he : e.toInt = pend d p) {it : I32} {log : List AtomicStep} {m : Mem}
    (hI : LoopInv d init p it log m) (hc : IntV.lt it e = true) :
    SafeA E (R d) (Wv d p (mu d p it)) (cells d)
      (fun it' log' m' => LoopInv d init p it' log' m' ∧ mu d p it' < mu d p it) log m
      (iterProg d nn T e it) := by
  obtain ⟨n, hit, h1, -, h2', hR, hva, hvb, hvc, hmem, hlog⟩ := hI
  have h3 : n < pend d p := by
    rw [i32_lt hit he] at hc
    simpa using hc
  have h2 : n = pstart d p ∨ n % steps d.K = 0 := by
    rcases h2' with h | h | h
    · exact Or.inl h
    · exact Or.inr h
    · omega
  have hmu : mu d p it = pend d p - n := by unfold mu; rw [hit, Int.toNat_natCast]
  rw [hmu]
  have hT0 := steps_pos hd
  have hTle : steps d.K ≤ 128 := by have := hd.K_le; unfold steps; omega
  have hpl := pend_lt (p := p) hd
  have hn0 := tilesN_pos hd
  have htl := iter_tile_lt hd h3
  have hdm := Nat.div_add_mod n (steps d.K)
  rw [Nat.mul_comm] at hdm
  have hdml := Nat.div_mul_le_self n (steps d.K)
  unfold iterProg
  -- line 44
  refine safeA_ro (safe_last (safe_idiv1 (by rw [hit]; omega) (by rw [hT]; omega) fun r1 hr1 => safe_ret ?_))
  try dsimp only
  have htile : r1.item.toInt = ((n / steps d.K : Nat) : Int) := by rw [hr1, hit, hT, Int.natCast_div]
  -- line 45
  have hkf0 : (it - r1.item * T).toInt = ((n % steps d.K : Nat) : Int) := by
    have hm : (r1.item * T).toInt = ((n / steps d.K * steps d.K : Nat) : Int) :=
      IntV.I32.mul_nat htile hT (by omega)
    rw [IntV.I32.sub_nat hit hm hdml (by omega)]
    congr 1; omega
  have hkf : (it - r1.item * T).toInt = (kfirst d p (n / steps d.K) : Int) := by
    rw [hkf0, iter_kfirst hd h1 h2]
  -- line 46
  have hkl : (tl.iminimum (Tensor.scalar T) (Tensor.scalar (it - r1.item * T + (e - it)))).item.toInt =
      (klast d p (n / steps d.K) : Int) := by
    show (IntV.min T (it - r1.item * T + (e - it))).toInt = _
    have hd1 : (e - it).toInt = ((pend d p - n : Nat) : Int) := IntV.I32.sub_nat he hit h3.le hpl
    have hd2 : (it - r1.item * T + (e - it)).toInt = ((n % steps d.K + (pend d p - n) : Nat) : Int) :=
      IntV.I32.add_nat hkf0 hd1 (by have := Nat.mod_le n (steps d.K); omega)
    rw [i32_min hT hd2, iter_klast hd h3]
    push_cast; rfl
  -- lines 47–48
  generalize (tl.iminimum (Tensor.scalar T) (Tensor.scalar (it - r1.item * T + (e - it)))).item = k_last at hkl ⊢
  generalize it - r1.item * T = k_first at hkf hkf0 ⊢
  refine safeA_ro (safe_last (safe_idiv1 (by rw [htile]; exact Int.natCast_nonneg _) (by rw [hnn]; omega) fun r2 hr2 => safe_ret ?_))
  have hpm : r2.item.toInt = (tm d (n / steps d.K) : Int) := by
    rw [hr2, htile, hnn, ← Int.natCast_div]; rfl
  refine safeA_ro (safe_last (safe_irem1 (by rw [htile]; exact Int.natCast_nonneg _) (by rw [hnn]; omega) fun r3 hr3 => safe_ret ?_))
  have hpn : r3.item.toInt = (tn d (n / steps d.K) : Int) := by
    rw [hr3, htile, hnn, ← Int.natCast_mod]; rfl
  -- lines 49–50: the lanes
  have htm : tm d (n / steps d.K) < tilesM d := by
    unfold tm
    rw [Nat.div_lt_iff_lt_mul hn0]
    exact htl
  have htn : tn d (n / steps d.K) < tilesN d := Nat.mod_lt _ hn0
  have hMle := hd.M_le
  have hNle := hd.N_le
  have hm : ∀ i : Fin 128, (((r2.item * (IntV.ofNat 128 : I32) + tl.arange 0 128 : Tensor [128 - 0] I32) : Tensor [128] I32).get (i, ())).toInt =
      (rowOf d (n / steps d.K) i : Int) := fun i =>
    lane_add hpm (by unfold tilesM at htm; omega) (by norm_num) i
  have hn : ∀ j : Fin 128, (((r3.item * (IntV.ofNat 128 : I32) + tl.arange 0 128 : Tensor [128 - 0] I32) : Tensor [128] I32).get (j, ())).toInt =
      (colOf d (n / steps d.K) j : Int) := fun j =>
    lane_add hpn (by unfold tilesN at htn; omega) (by norm_num) j
  -- lines 51–57: the inner loop
  have htc := iter_touches hd h1 h2 h3
  have hle := kfirst_le_klast htc
  have hkl128 := part_lt (d := d) p (n / steps d.K)
  have hrange : tl.RangeOk k_first k_last (IntV.ofInt 1 : I32) := by
    unfold tl.RangeOk Math.Float.Format.intRange
    rw [hkf, hkl, i32_one]
    norm_num
    omega
  have htrip : tl.tripCount k_first.toInt k_last.toInt (IntV.ofInt 1 : I32).toInt =
      klast d p (n / steps d.K) - kfirst d p (n / steps d.K) := by
    rw [hkf, hkl, i32_one]
    simp only [tl.tripCount, add_sub_cancel_right, Int.ediv_one]
    omega
  refine safeA_ro (safe_last (Safe.mono (safe_forRange (Wk := ∅) hrange
    (fun t st m₁ => m₁ = m ∧ AccInv d init (n / steps d.K) (kfirst d p (n / steps d.K)) t st)
    (fun _ => ∅) (fun _ _ _ _ => Set.disjoint_empty _) (by rw [iterW_empty]; exact Set.empty_disjoint _)
    ⟨rfl, acc0 d init _ _⟩ ?_ ?_) (fun _ _ h => h) (by rw [iterW_empty, Set.empty_union])))
  · rw [htrip]
    rintro t ht st m₁ ⟨rfl, hacc⟩
    have hkk : (IntV.ofInt (k_first.toInt + ↑t * (IntV.ofInt 1 : I32).toInt) : I32).toInt =
        ((kfirst d p (n / steps d.K) + t : Nat) : Int) := by
      rw [hkf, i32_one, mul_one]
      exact IntV.I32.toInt_ofInt (by constructor <;> push_cast <;> omega)
    have hk : ∀ k : Fin 64, (((IntV.ofInt (k_first.toInt + ↑t * (IntV.ofInt 1 : I32).toInt) : I32) *
        (IntV.ofNat 64 : I32) + tl.arange 0 64 : Tensor [64 - 0] I32).get (k, ())).toInt =
        ((64 * (kfirst d p (n / steps d.K) + t) + k : Nat) : Int) := fun k => by
      rw [show 64 * (kfirst d p (n / steps d.K) + t) = (kfirst d p (n / steps d.K) + t) * 64 by ring]
      exact lane_add hkk (by omega) (by norm_num) k
    unfold chunkProg
    exact safe_loadDot hd hdot htot hva hvb hR (by omega) hm hn hk hacc
  · rw [htrip]
    rintro st m₁ ⟨rfl, hacc⟩
    refine safe_ret ?_
    unfold tailProg
    -- lines 58–64
    have hMv : (IntV.ofNat d.M : I32).toInt = d.M := i32_nat (by omega)
    have hNv : (IntV.ofNat d.N : I32).toInt = d.N := i32_nat (by omega)
    have hscm := i32_nat hd.cm
    have hscn := i32_nat hd.cn.2
    have hcrow : d.N * d.stride_cn < 2 ^ 31 := by have := hd.c_rows; have := hd.cm; omega
    refine safe_storeAdd hd hat hatot hp h1 h2 h3 hR hva hvb hvc hmem hlog (fun ⟨i, j, ⟨⟩⟩ => ?_)
      (fun ⟨i, j, ⟨⟩⟩ he => ?_) hit hkf hkl hT hacc
    · show (IntV.lt ((r2.item * (IntV.ofNat 128 : I32) + tl.arange 0 128 : Tensor [128 - 0] I32).get (i, ())) (IntV.ofNat d.M) && IntV.lt ((r3.item * (IntV.ofNat 128 : I32) + tl.arange 0 128 : Tensor [128 - 0] I32).get (j, ())) (IntV.ofNat d.N)) =
        cEn d (n / steps d.K) (i, j, ())
      rw [i32_lt (hm i) hMv, i32_lt (hn j) hNv]
      simp only [cEn, Nat.cast_lt]
    · simp only [cEn, Bool.and_eq_true, decide_eq_true_eq] at he
      show ((d.c.ptr 0).offset (((r2.item * (IntV.ofNat 128 : I32) + tl.arange 0 128 : Tensor [128 - 0] I32).get (i, ()) * IntV.ofNat d.stride_cm).toInt)).offset
        (((r3.item * (IntV.ofNat 128 : I32) + tl.arange 0 128 : Tensor [128 - 0] I32).get (j, ()) * IntV.ofNat d.stride_cn).toInt) = d.c.ptr ↑(cG d (n / steps d.K) (i, j, ()))
      have hb1 : rowOf d (n / steps d.K) i * d.stride_cm < 2 ^ 31 :=
        lt_of_le_of_lt (mul_le_fit he.1) hd.cm_fit
      have hb2 : colOf d (n / steps d.K) j * d.stride_cn < 2 ^ 31 :=
        lt_of_lt_of_le (Nat.mul_lt_mul_of_pos_right he.2 hd.cn.1) hcrow.le
      rw [IntV.I32.mul_nat (hm i) hscm hb1, IntV.I32.mul_nat (hn j) hscn hb2, Buf.ptr_offset, Buf.ptr_offset]
      unfold cG
      congr 1; push_cast; ring

/-! ## One program -/

/-- What a program leaves does not depend on the order of its log (`Paired.perm`). -/
theorem Q_perm (d : Inst) (init : Mem) (p : Nat) : PermInv (fun (_ : Unit) log m => Q d init p log m) := by
  rintro _ l l' m hp ⟨hmem, hlog⟩
  refine ⟨hmem, fun t x hw hen va vb hra hcb => ?_⟩
  obtain ⟨ops, hpd, h1, h2⟩ := hlog t x hw hen va vb hra hcb
  exact ⟨ops, hpd.perm hp, h1, h2⟩

/-- **One program.** Program `p` of the grid, run alone from `init`, with its atomic steps on the
split tiles' elements logged: it reads only `a` and `b`, writes privately only the tiles it owns,
and leaves what `Q` says. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {P : Platform}
    (hdot : Rules.Num.DotF16 P)
    (htot : Rules.Num.DotTotal P { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 })
    (hat : Rules.Num.AtomicAddF32 P) (hatot : Rules.Num.AtomicAddTotal P Math.Float.Format.binary32)
    {p : Nat} (hp : p < d.num_programs) :
    SafeA ((launch d).env P) (R d) (W d p) (cells d) (fun _ log m => Q d init p log m) [] init
      ((model d).body ((launch d).ctx p)) := by
  have hpid := launch_pid d hp
  have hnp := np_lt hd
  have hI := ipp_pos hd
  have hNle := hd.N_le
  have hK := hd.K_le
  have hpl := pend_lt (p := p) hd
  have hcov := hd.progs_lt
  have hp1 : p * d.iters_per_program + d.iters_per_program ≤ d.num_programs * d.iters_per_program := by
    rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hp
  have htot_le : d.total_iters ≤ d.num_programs * d.iters_per_program := hd.cover
  rw [body_eq]
  rw [show tl.program_id ((launch d).ctx p) 0 = IntV.ofNat p by simp [tl.program_id, hpid]]
  have hpv : (IntV.ofNat p : I32).toInt = p := i32_nat (by omega)
  -- line 37
  have hx1 : ((IntV.ofNat d.N : I32) + (IntV.ofNat (128 - 1) : I32)).toInt = ((d.N + 127 : Nat) : Int) :=
    IntV.I32.add_nat (i32_nat (by omega)) (i32_nat (by norm_num)) (by omega)
  refine safeA_ro (safe_last (safe_idiv1 (by rw [hx1]; exact Int.natCast_nonneg _)
    (by rw [i32_nat (by norm_num)]; norm_num) fun r1 hr1 => safe_ret ?_))
  have hnn : r1.item.toInt = tilesN d := by
    rw [hr1, hx1, i32_nat (by norm_num)]; unfold tilesN; push_cast; omega
  -- line 38
  have hx2 : ((IntV.ofNat d.K : I32) + (IntV.ofNat (64 - 1) : I32)).toInt = ((d.K + 63 : Nat) : Int) :=
    IntV.I32.add_nat (i32_nat (by omega)) (i32_nat (by norm_num)) (by omega)
  refine safeA_ro (safe_last (safe_idiv1 (by rw [hx2]; exact Int.natCast_nonneg _)
    (by rw [i32_nat (by norm_num)]; norm_num) fun r2 hr2 => safe_ret ?_))
  have hT : r2.item.toInt = steps d.K := by
    rw [hr2, hx2, i32_nat (by norm_num)]; unfold steps; push_cast; omega
  dsimp only
  -- lines 39–40
  have hit0 : (IntV.ofNat p * (IntV.ofNat d.iters_per_program : I32)).toInt = (pstart d p : Int) :=
    IntV.I32.mul_nat hpv (i32_nat (by omega)) (by omega)
  have hend : (tl.iminimum (Tensor.scalar (IntV.ofNat p * (IntV.ofNat d.iters_per_program : I32) +
      (IntV.ofNat d.iters_per_program : I32))) (Tensor.scalar (IntV.ofNat d.total_iters : I32))).item.toInt =
      (pend d p : Int) := by
    show (IntV.min (IntV.ofNat p * (IntV.ofNat d.iters_per_program : I32) + (IntV.ofNat d.iters_per_program : I32))
      (IntV.ofNat d.total_iters : I32)).toInt = _
    rw [i32_min (IntV.I32.add_nat hit0 (i32_nat (by omega)) (by unfold pstart; omega)) (i32_nat (by omega))]
    unfold pend pstart; push_cast; rfl
  generalize (tl.iminimum (Tensor.scalar (IntV.ofNat p * (IntV.ofNat d.iters_per_program : I32) +
      (IntV.ofNat d.iters_per_program : I32))) (Tensor.scalar (IntV.ofNat d.total_iters : I32))).item = e at hend ⊢
  generalize IntV.ofNat p * (IntV.ofNat d.iters_per_program : I32) = it0 at hit0 ⊢
  -- lines 41–64: the while loop, then `pure ()`
  have hmu0 : mu d p it0 = pend d p - pstart d p := by unfold mu; rw [hit0, Int.toNat_natCast]
  have h0 : LoopInv d init p it0 [] init :=
    ⟨pstart d p, hit0, le_rfl, Or.inr rfl, Or.inl rfl, fun _ _ => rfl, hpre.1, hpre.2.1, hpre.2.2.1,
      fun _ _ _ hdone _ => absurd hdone.2 (not_done_start hdone.1),
      fun _ _ _ _ _ _ _ _ => ⟨0, Paired.nil, fun h => absurd h.2 (not_done_start h.1), fun _ => rfl⟩⟩
  have hsub : iterW (Wv d p) (mu d p it0 + 1) ∪ ∅ ⊆ W d p := by
    rw [Set.union_empty]
    rintro x ⟨v, hv, hx⟩
    exact Wv_sub hd (by rw [hmu0] at hv; omega) hx
  refine (SafeA.bind (W₁ := iterW (Wv d p) (mu d p it0 + 1)) (W₂ := ∅)
    (safeA_while (fun it => IntV.lt it e) (iterProg d r1.item r2.item e) it0 (LoopInv d init p) (mu d p) (Wv d p)
      (fun u v huv => Wv_disj hd p huv) h0
      (fun st log m hI hc => safe_iter hd hdot htot hat hatot hp hnn hT hend hI hc) ?_)
    (Set.disjoint_empty _)).mono (fun _ _ _ h => h) hsub
  intro st log m hI hc
  obtain ⟨n, hn, -, -, -, -, -, -, -, hmem, hlog⟩ := hI
  have hge : pend d p ≤ n := by
    rw [i32_lt hn hend] at hc
    simpa using hc
  have hdone : doneAt d p n = touches d p :=
    funext fun t => propext ⟨fun h => h.1, fun h => ⟨h, done_end hge⟩⟩
  rw [hdone] at hmem hlog
  exact SafeA.ret ⟨hmem, hlog⟩

end Kernels.StreamKMatmul

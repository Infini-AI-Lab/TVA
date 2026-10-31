import GroupedMatmulTma.Basic

/-!
# `grouped_matmul_tma_f16`: one tile

`safe_tile`: one iteration of the `while` loop of line 251 (lines 252–275), at `tile_idx = y`,
the global index of tile `t = y − start g` of group `g`, which program `p` owns. It reads only
the tables and the input matrices (`R`), writes only that tile (`Wv`, by the loop's variant `mu`),
leaves there what `OutSpec` says, and returns `y + NUM_SM`: the `while` invariant `WI` holds
again and the variant decreases.

* `lane2`: the lanes of a block of a row-major 2-D descriptor.
* `AccInv`: the accumulator after `s` steps of the loop of line 263.
-/

namespace Kernels.GroupedMatmulTma

open TVA

/-! ## Lanes of a 2-D descriptor -/

/-- Lane `(i, j)` of a block at the offsets `O0`, `O1` (nonnegative, the values of `o0`, `o1`) of
a row-major `[s0, s1]` descriptor with row stride `l`: inside the tensor when both coordinates
are, and then element `(O0 + i)·l + (O1 + j)`. -/
theorem lane2 {d : DType} (base : Ptr d) (s0 s1 l : Nat) (bl : Shape) (pad : Padding) {m n : Nat}
    {o0 o1 : I32} {O0 O1 : Nat} (h0 : o0.toInt = O0) (h1 : o1.toInt = O1) (i : Fin m) (j : Fin n) :
    (⟨base, [s0, s1], [l, 1], bl, pad⟩ : TensorDesc d).inbAt
        (TensorDesc.coord ([o0, o1].map IntV.toInt) (bs := [m, n]) (i, j, ())) =
      decide (O0 + i < s0 ∧ O1 + j < s1) ∧
    ((⟨base, [s0, s1], [l, 1], bl, pad⟩ : TensorDesc d).elemAt
        (TensorDesc.coord ([o0, o1].map IntV.toInt) (bs := [m, n]) (i, j, ()))).toNat =
      (O0 + i) * l + (O1 + j) := by
  simp only [List.map_cons, List.map_nil]
  rw [inbAt2, elemAt2, coord2_0, coord2_1, h0, h1]
  constructor
  · refine decide_eq_decide.mpr ⟨fun h => ⟨by omega, by omega⟩, fun h => ⟨⟨by omega, by omega⟩, by omega, by omega⟩⟩
  · rw [show ((O0 : Int) + (i : Nat)) * (l : Int) + ((O1 : Int) + (j : Nat)) =
      (((O0 + i) * l + (O1 + j) : Nat) : Int) by push_cast; ring, Int.toNat_natCast]

/-! ## The accumulator and the `while` loop -/

/-- The accumulator of tile `t` of group `g` after `s` steps of the loop of line 263: each element
of the tile inside `C g` whose row of `A g` and row of `B g` hold finite values in `init` is
finite and within `DotLoop.bound rho alpha s` of the exact partial sum. -/
def AccInv (d : Inst) (init : Mem) (g t s : Nat) (acc : Tensor [128, 128] F32) : Prop :=
  ∀ i j : Fin 128, tm d g t * 128 + i < d.M g → tn d g t * 128 + j < d.N g →
    ∀ va vb : Nat → F16,
      (∀ k, k < d.K g → init.read .f16 ((d.A g).ptr ↑((tm d g t * 128 + i) * d.lda g + k)).addr =
        some (va k) ∧ (va k).isFinite = true) →
      (∀ k, k < d.K g → init.read .f16 ((d.B g).ptr ↑((tn d g t * 128 + j) * d.ldb g + k)).addr =
        some (vb k) ∧ (vb k).isFinite = true) →
      (acc.get2 i j).isFinite = true ∧
        |(acc.get2 i j).toRat - partS va vb (d.K g) s| ≤ DotLoop.bound rho alpha s (partA va vb (d.K g) s)

/-- The invariant of the `while` loop of group `g` (line 251) in program `p`: `tile_idx` holds
some `y` with `y ≡ p` modulo `NUM_SM`, from `start g` up to one stride past the group's end, and
the memory holds `ProgInv` at `y`. -/
def WI (d : Inst) (init : Mem) (p g : Nat) (st : I32) (m : Mem) : Prop :=
  ∃ y : Nat, st.toInt = y ∧ y % d.NUM_SM = p ∧ start d g ≤ y ∧ y < start d (g + 1) + d.NUM_SM ∧
    ProgInv d init p y m

/-- The variant of the `while` loop: the tiles of group `g` from `tile_idx` on. -/
def mu (d : Inst) (g : Nat) (st : I32) : Nat := (((start d g + tiles d g : Nat) : Int) - st.toInt).toNat

theorem acc0 (d : Inst) (init : Mem) (g t : Nat) : AccInv d init g t 0 (tl.zeros [128, 128] .f32) :=
  fun i j _ _ va vb _ _ => acc0_val va vb (d.K g) i j

/-- The `dot` configuration of line 266. -/
abbrev cfg16 : DotCfg := { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }

/-! ## One tile -/

/-- **One tile** (lines 252–275): the body of the `while` loop of group `g`, with the
descriptors of lines 230–248 (the bases of `a` and `b` matter only when `K g > 0`), at
`tile_idx = y` inside the group's tiles and owned by `p`. -/
theorem safe_tile {P : Platform} (hdot : Rules.Num.DotF16 P) (htot : Rules.Num.DotTotal P cfg16)
    {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {p g : Nat} (hg : g < d.group_size)
    {lpe nn gk : I32} (hlpe : lpe.toInt = start d g) (hnn : nn.toInt = nN d g)
    (hgk : gk.toInt = d.K g) {aD bD cD : TensorDesc .f16} {abase bbase : Ptr .f16}
    (haD : aD = ⟨abase, [d.M g, d.K g], [d.lda g, 1], [128, 64], .zero⟩)
    (habase : 0 < d.K g → abase = ⟨(d.A g).base⟩)
    (hbD : bD = ⟨bbase, [d.N g, d.K g], [d.ldb g, 1], [128, 64], .zero⟩)
    (hbbase : 0 < d.K g → bbase = ⟨(d.B g).base⟩)
    (hcD : cD = ⟨⟨(d.C g).base⟩, [d.M g, d.N g], [d.ldc g, 1], [128, 128], .zero⟩)
    {tile_idx : I32} {y : Nat} (hy : tile_idx.toInt = y) (hlo : start d g ≤ y)
    (hhi : y < start d g + tiles d g) (hp : y % d.NUM_SM = p) {m₁ : Mem} (hinv : ProgInv d init p y m₁) :
    Safe ((launch d).env P) (R d) (Wv d p g (mu d g tile_idx))
      (fun st' m₃ => WI d init p g st' m₃ ∧ mu d g st' < mu d g tile_idx) m₁
      (do
        let k := gk
        let tile_idx_in_gemm := tile_idx - lpe
        let tile_m_idx := (← tl.idiv (Tensor.scalar tile_idx_in_gemm) (Tensor.scalar nn)).item
        let tile_n_idx := (← tl.irem (Tensor.scalar tile_idx_in_gemm) (Tensor.scalar nn)).item
        let offs_am := tile_m_idx * (IntV.ofNat 128 : I32)
        let offs_bn := tile_n_idx * (IntV.ofNat 128 : I32)
        let accumulator := tl.zeros [128, 128] .f32
        let accumulator ← tl.forRange (IntV.ofInt 0 : I32)
            (← tl.idiv (Tensor.scalar (k + (IntV.ofNat (64 - 1) : I32)))
              (Tensor.scalar (IntV.ofNat 64 : I32))).item
            (IntV.ofInt 1 : I32) accumulator fun kk accumulator => do
          let a : Tensor [128, 64] F16 ← tl.desc_load aD [offs_am, kk * (IntV.ofNat 64 : I32)]
          let b : Tensor [128, 64] F16 ← tl.desc_load bD [offs_bn, kk * (IntV.ofNat 64 : I32)]
          let accumulator ←
            tl.dot_add { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 } a b.trans accumulator
          return accumulator
        let offs_cm := tile_m_idx * (IntV.ofNat 128 : I32)
        let offs_cn := tile_n_idx * (IntV.ofNat 128 : I32)
        let c : Tensor [128, 128] F16 ← tl.cvt accumulator
        tl.desc_store cD [offs_cm, offs_cn] c
        let tile_idx := tile_idx + (IntV.ofNat d.NUM_SM : I32)
        return tile_idx) := by
  subst haD hbD hcD
  obtain ⟨t, rfl⟩ : ∃ t, y = start d g + t := ⟨y - start d g, by omega⟩
  have ht : t < tiles d g := by omega
  -- sizes
  have hS := hd.SM_pos
  have hGt := hd.tiles_lt
  have hend := start_add_le_total d hg
  have hMle := hd.M_le g hg
  have hNle := hd.N_le g hg
  have hKle := hd.K_le g hg
  obtain ⟨hlda, hlda8⟩ := hd.lda_ok g hg
  obtain ⟨hldb, hldb8⟩ := hd.ldb_ok g hg
  obtain ⟨hldc, hldc8, hNldc⟩ := hd.ldc_ok g hg
  obtain ⟨hAal, hBal, hCal⟩ := hd.base_al g hg
  have htiles : tiles d g = nM d g * nN d g := tiles_eq d g
  have hnN0 : 0 < nN d g := by
    rcases Nat.eq_zero_or_pos (nN d g) with h | h
    · rw [htiles, h, Nat.mul_zero] at ht; omega
    · exact h
  have htm : tm d g t < nM d g := by
    unfold tm; exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm, ← htiles]; exact ht)
  have htn : tn d g t < nN d g := Nat.mod_lt _ hnN0
  have hM : tm d g t * 128 < d.M g := by unfold nM at htm; omega
  have hN : tn d g t * 128 < d.N g := by unfold nN at htn; omega
  have hmu : mu d g tile_idx = tiles d g - t := by unfold mu; rw [hy]; omega
  -- line 254
  have htig : (tile_idx - lpe).toInt = (t : Int) := by
    rw [i32_sub hy hlpe (by omega) (by omega)]; push_cast; ring
  -- lines 255–256
  refine safe_idiv1 (by rw [htig]; omega) (by rw [hnn]; exact_mod_cast hnN0) fun r1 hr1 => ?_
  have h1 : r1.item.toInt = (tm d g t : Int) := by rw [hr1, htig, hnn, ← Int.natCast_div]; rfl
  refine safe_irem1 (by rw [htig]; omega) (by rw [hnn]; exact_mod_cast hnN0) fun r2 hr2 => ?_
  have h2 : r2.item.toInt = (tn d g t : Int) := by rw [hr2, htig, hnn, ← Int.natCast_mod]; rfl
  -- lines 259–260 (and 268–269)
  have hoam : (r1.item * (IntV.ofNat 128 : I32)).toInt = ((tm d g t * 128 : Nat) : Int) :=
    IntV.I32.mul_nat h1 (i32_nat (by norm_num)) (by omega)
  have hobn : (r2.item * (IntV.ofNat 128 : I32)).toInt = ((tn d g t * 128 : Nat) : Int) :=
    IntV.I32.mul_nat h2 (i32_nat (by norm_num)) (by omega)
  -- line 263: the loop bound
  have hK63 : (gk + (IntV.ofNat (64 - 1) : I32)).toInt = ((d.K g + 63 : Nat) : Int) :=
    IntV.I32.add_nat hgk (i32_nat (by norm_num)) (by omega)
  refine safe_idiv1 (by rw [hK63]; omega) (by rw [i32_nat (by norm_num)]; norm_num) fun rK hrK => ?_
  have hT : rK.item.toInt = steps (d.K g) := by
    rw [hrK, hK63, i32_nat (by norm_num)]; unfold steps; push_cast; omega
  have hrange : tl.RangeOk (IntV.ofInt 0 : I32) rK.item (IntV.ofInt 1) := by
    unfold tl.RangeOk Math.Float.Format.intRange
    rw [i32_zero, i32_one, hT]; unfold steps; norm_num; omega
  have htc : tl.tripCount (IntV.ofInt 0 : I32).toInt rK.item.toInt (IntV.ofInt 1 : I32).toInt =
      steps (d.K g) := by
    rw [i32_zero, i32_one, hT]; simp [tl.tripCount]
  have hvA : (d.A g).Valid m₁ := ProgInv_valid hinv (hpre.2.2.2.2.2.1 g hg).1
  have hvB : (d.B g).Valid m₁ := ProgInv_valid hinv (hpre.2.2.2.2.2.1 g hg).2.1
  have hvC : (d.C g).Valid m₁ := ProgInv_valid hinv (hpre.2.2.2.2.2.1 g hg).2.2
  have hagR := ProgInv_R hd hinv
  have hb2 : DType.bytes .f16 = 2 := rfl
  refine Safe.mono (safe_forRange (Wk := Wv d p g (mu d g tile_idx)) hrange
    (fun s acc m => m = m₁ ∧ AccInv d init g t s acc)
    (fun _ => ∅) (fun _ _ _ _ => Set.disjoint_empty _) (by rw [iterW_empty]; exact Set.empty_disjoint _)
    ⟨rfl, acc0 d init g t⟩ ?_ ?_) (fun _ _ h => h) (by rw [iterW_empty, Set.empty_union])
  · -- one step of the loop of line 263 (lines 264–266)
    rw [htc]
    rintro s hs acc m₂ ⟨rfl, hacc⟩
    have hs64 : 64 * s < d.K g := by unfold steps at hs; omega
    obtain rfl := habase (by omega)
    obtain rfl := hbbase (by omega)
    have hkk : (IntV.ofInt ((IntV.ofInt 0 : I32).toInt + (s : Int) * (IntV.ofInt 1 : I32).toInt) * (IntV.ofNat 64 : I32) : I32).toInt =
        ((64 * s : Nat) : Int) := by
      rw [IntV.I32.mul_nat (i32_trip (by omega)) (i32_nat (by norm_num)) (by omega)]; push_cast; ring
    -- line 264
    refine safe_desc_load (d.A g) (buf_ptr0 _).symm
      (access2 ⟨by omega, by omega⟩ ⟨by omega, by omega⟩ hAal (by rw [hb2]; omega) rfl
        (by rw [hkk, hb2]; push_cast; omega)) ?_ hvA ?_ fun A hA1 hA2 => ?_
    · rintro ⟨i, k, ⟨⟩⟩ hin
      obtain ⟨e1, e2⟩ := lane2 _ _ _ _ _ _ hoam hkk i k
      rw [e1, decide_eq_true_iff] at hin
      rw [e2]
      exact hd.A_len g hg _ _ hin.1 (by omega)
    · refine Set.Subset.trans ((d.A g).elems_mono ?_) fun x hx => Or.inr ⟨g, hg, Or.inl hx⟩
      rintro _ ⟨⟨i, k, ⟨⟩⟩, he, rfl⟩
      dsimp only at he ⊢
      obtain ⟨e1, e2⟩ := lane2 _ _ _ _ _ _ hoam hkk i k
      rw [e1, decide_eq_true_iff] at he
      rw [e2]
      exact ⟨_, _, he.1, he.2, by ring⟩
    -- line 265
    refine safe_desc_load (d.B g) (buf_ptr0 _).symm
      (access2 ⟨by omega, by omega⟩ ⟨by omega, by omega⟩ hBal (by rw [hb2]; omega) rfl
        (by rw [hkk, hb2]; push_cast; omega)) ?_ hvB ?_ fun Bm hB1 hB2 => ?_
    · rintro ⟨j, k, ⟨⟩⟩ hin
      obtain ⟨e1, e2⟩ := lane2 _ _ _ _ _ _ hobn hkk j k
      rw [e1, decide_eq_true_iff] at hin
      rw [e2]
      exact hd.B_len g hg _ _ hin.1 (by omega)
    · refine Set.Subset.trans ((d.B g).elems_mono ?_) fun x hx => Or.inr ⟨g, hg, Or.inr hx⟩
      rintro _ ⟨⟨j, k, ⟨⟩⟩, he, rfl⟩
      dsimp only at he ⊢
      obtain ⟨e1, e2⟩ := lane2 _ _ _ _ _ _ hobn hkk j k
      rw [e1, decide_eq_true_iff] at he
      rw [e2]
      exact ⟨_, _, he.1, he.2, by ring⟩
    -- line 266
    refine safe_last (safe_dot_add ?_ ?_)
    · exact htot _ _ _
    intro r hr
    refine safe_ret ⟨rfl, ?_⟩
    intro i j hi hj va vb hva hvb
    refine acc_step hdot hr i j (by omega) hs64 (fun e he => (hva e he).2) (fun e he => (hvb e he).2)
      (fun k => ⟨fun h => ?_, fun h => ?_⟩) (fun k => ⟨fun h => ?_, fun h => ?_⟩)
      (hacc i j hi hj va vb hva hvb)
    · obtain ⟨e1, e2⟩ := lane2 _ _ _ _ _ _ hoam hkk i k
      have hr1 := hA1 (i, k, ()) (by rw [e1, decide_eq_true_iff]; exact ⟨hi, by omega⟩)
      have hag : m₂.read .f16 ((d.A g).ptr ↑((tm d g t * 128 + ↑i) * d.lda g + (64 * s + ↑k))).addr =
          init.read .f16 ((d.A g).ptr ↑((tm d g t * 128 + ↑i) * d.lda g + (64 * s + ↑k))).addr :=
        Mem.read_congr fun x hx => hagR x (Or.inr ⟨g, hg, Or.inl ((d.A g).span_sub_elems (S := aIdx d g) ⟨_, _, hi, h, rfl⟩ hx)⟩)
      rw [e2, hag] at hr1
      have hr2 := (hva _ h).1
      exact Option.some.inj (hr1.symm.trans hr2)
    · obtain ⟨e1, -⟩ := lane2 _ _ _ _ _ _ hoam hkk i k
      have h0 := hA2 (i, k, ()) (by rw [e1, decide_eq_false_iff_not]; omega)
      show (A.get (i, k, ())).isFinite = true ∧ (A.get (i, k, ())).toRat = 0
      rw [show A.get (i, k, ()) = FloatV.ofPat 0 from h0]
      exact f16_zero
    · obtain ⟨e1, e2⟩ := lane2 _ _ _ _ _ _ hobn hkk j k
      have hr1 := hB1 (j, k, ()) (by rw [e1, decide_eq_true_iff]; exact ⟨hj, by omega⟩)
      have hag : m₂.read .f16 ((d.B g).ptr ↑((tn d g t * 128 + ↑j) * d.ldb g + (64 * s + ↑k))).addr =
          init.read .f16 ((d.B g).ptr ↑((tn d g t * 128 + ↑j) * d.ldb g + (64 * s + ↑k))).addr :=
        Mem.read_congr fun x hx => hagR x (Or.inr ⟨g, hg, Or.inr ((d.B g).span_sub_elems (S := bIdx d g) ⟨_, _, hj, h, rfl⟩ hx)⟩)
      rw [e2, hag] at hr1
      have hr2 := (hvb _ h).1
      exact Option.some.inj (hr1.symm.trans hr2)
    · obtain ⟨e1, -⟩ := lane2 _ _ _ _ _ _ hobn hkk j k
      have h0 := hB2 (j, k, ()) (by rw [e1, decide_eq_false_iff_not]; omega)
      show (Bm.get (j, k, ())).isFinite = true ∧ (Bm.get (j, k, ())).toRat = 0
      rw [show Bm.get (j, k, ()) = FloatV.ofPat 0 from h0]
      exact f16_zero
  · -- after the loop (lines 268–275)
    rw [htc]
    rintro acc m₂ ⟨rfl, hacc⟩
    -- line 271
    refine safe_cvt ⟨by unfold Math.Float.Format.Std; simp, by unfold Math.Float.Format.Std; simp, Or.inl rfl⟩
      fun c hc => ?_
    -- line 272
    have himg : Buf.img (fun i : Idx [128, 128] => ((⟨⟨(d.C g).base⟩, [d.M g, d.N g], [d.ldc g, 1], [128, 128], .zero⟩ :
          TensorDesc .f16).elemAt (TensorDesc.coord ([r1.item * (IntV.ofNat 128 : I32), r2.item * (IntV.ofNat 128 : I32)].map IntV.toInt) i)).toNat)
        (fun i => (⟨⟨(d.C g).base⟩, [d.M g, d.N g], [d.ldc g, 1], [128, 128], .zero⟩ : TensorDesc .f16).inbAt
          (TensorDesc.coord ([r1.item * (IntV.ofNat 128 : I32), r2.item * (IntV.ofNat 128 : I32)].map IntV.toInt) i)) ⊆
        tileIdx d g t := by
      rintro _ ⟨⟨i, j, ⟨⟩⟩, he, rfl⟩
      dsimp only at he ⊢
      obtain ⟨e1, e2⟩ := lane2 _ _ _ _ _ _ hoam hobn i j
      rw [e1, decide_eq_true_iff] at he
      rw [e2]
      exact ⟨i, j, i.2, j.2, he.1, he.2, rfl⟩
    have hWv : Wv d p g (mu d g tile_idx) = tileBytes d g t := by
      rw [hmu]; unfold Wv
      rw [ite_eq_left ⟨by omega, by omega, by rw [show start d g + (tiles d g - (tiles d g - t)) = start d g + t by omega]; exact hp⟩,
        show tiles d g - (tiles d g - t) = t by omega]
    refine safe_desc_store (d.C g) (buf_ptr0 _).symm
      (access2 ⟨by omega, by omega⟩ ⟨by omega, by omega⟩ hCal (by rw [hb2]; omega) rfl
        (by rw [hobn, hb2]; push_cast; omega)) ?_ ?_ hvC (by rw [hWv]; exact (d.C g).elems_mono himg)
      fun m' hm'1 hm'2 hm'3 => safe_ret ⟨?_, ?_⟩
    · rintro ⟨i, j, ⟨⟩⟩ hin
      obtain ⟨e1, e2⟩ := lane2 _ _ _ _ _ _ hoam hobn i j
      rw [e1, decide_eq_true_iff] at hin
      rw [e2]
      exact hd.C_len g hg _ _ hin.1 hin.2
    · rintro ⟨i, j, ⟨⟩⟩ ⟨i', j', ⟨⟩⟩ hin hin' he
      obtain ⟨e1, e2⟩ := lane2 _ _ _ _ _ _ hoam hobn i j
      obtain ⟨e1', e2'⟩ := lane2 _ _ _ _ _ _ hoam hobn i' j'
      rw [e1, decide_eq_true_iff] at hin
      rw [e1', decide_eq_true_iff] at hin'
      rw [e2, e2'] at he
      obtain ⟨hr, hc'⟩ := c_inj hNldc hin.2 hin'.2 he
      have hi : i = i' := Fin.ext (by omega)
      have hj : j = j' := Fin.ext (by omega)
      subst hi hj; rfl
    · -- line 275: the invariant at `y + NUM_SM`
      have hSlt : d.NUM_SM < 2 ^ 31 := by omega
      refine ⟨start d g + t + d.NUM_SM, ?_, ?_, by omega, by rw [start_succ]; omega, ?_⟩
      · rw [IntV.I32.add_nat hy (i32_nat hSlt) (by omega)]
      · rw [Nat.add_mod_right]; exact hp
      · refine ProgInv_step hd hinv hg ht rfl hp
          (fun x hx => hm'2 x fun hx' => hx ((d.C g).elems_mono himg hx')) hm'3 ?_
        intro i j hi hj hiM hjN
        obtain ⟨e1, e2⟩ := lane2 _ _ _ _ _ _ hoam hobn (⟨i, hi⟩ : Fin 128) (⟨j, hj⟩ : Fin 128)
        have hr := hm'1 (⟨i, hi⟩, ⟨j, hj⟩, ()) (by rw [e1, decide_eq_true_iff]; exact ⟨hiM, hjN⟩)
        rw [e2] at hr
        refine ⟨c.get (⟨i, hi⟩, ⟨j, hj⟩, ()), hr, fun va vb hva hvb => ?_⟩
        obtain ⟨hfin, herr⟩ := hacc ⟨i, hi⟩ ⟨j, hj⟩ hiM hjN va vb hva hvb
        obtain ⟨hS', hA'⟩ := partS_end va vb (T := steps (d.K g)) (K := d.K g) (by unfold steps; omega)
        rw [hS', hA'] at herr
        have hcv : FloatV.NaNEq (FloatV.cvt (acc.get2 ⟨i, hi⟩ ⟨j, hj⟩)) (c.get (⟨i, hi⟩, ⟨j, hj⟩, ())) :=
          hc (⟨i, hi⟩, ⟨j, hj⟩, ())
        exact ⟨acc.get2 ⟨i, hi⟩ ⟨j, hj⟩, hfin, herr, hcv, fun hroom => out_f16 hfin herr hcv hroom⟩
    · -- the variant decreases
      rw [hmu]
      unfold mu
      rw [IntV.I32.add_nat hy (i32_nat (show d.NUM_SM < 2 ^ 31 by omega)) (by omega)]
      omega

end Kernels.GroupedMatmulTma

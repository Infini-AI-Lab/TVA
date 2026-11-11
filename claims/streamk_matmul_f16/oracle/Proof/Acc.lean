import Claim
import TVA.Proof
import Proof.Count

/-!
# `Proof.Acc`

The accumulator invariant `AccI` of the inner loop, the gated invariant `Qg`, and the per-tile write footprints.
-/

/-!
# The inner chunk loop

One iteration of the inner loop of the ``stream-K`` kernel multiplies a `128 × 64` block of `a` by
a `64 × 128` block of `b` and adds the result into the `128 × 128` accumulator.

`AccI` is the invariant of the accumulator after `t` chunk iterations of the tile `u` of program
`i`: for the lanes of the tile's output elements the value is the finite `float32` partial sum
`pre (c0 + t) - pre c0` of the first `c0 + t` chunks, up to the `fineOne` quantity of
`Proof/Tol.lean`.  Nothing is required of the other lanes (both the loads and the store are
masked).  `chunkSafe` is one step of the loop, at the `Safe` level of the library.
-/

namespace Claims.StreamKMatmul

open TVA

variable {d : Inst} {P : Platform}

/-- **The accumulator invariant.**  After `t` chunk iterations of tile `u`, every in-bounds lane
holds the tile's partial dot product over the first `c0 + t` chunks up to `fineOne`. -/
def AccI (d : Inst) (init : Mem) (i u t : Nat) (acc : Tensor [128, 128] F32) : Prop :=
  ∀ (ii jj : Fin 128), pidMOf d u * 128 + ii.val < d.M → pidNOf d u * 128 + jj.val < d.N →
    ∀ va vb : Nat → F16,
      ReadsA d init (pidMOf d u * 128 + ii.val) (pidNOf d u * 128 + jj.val) va →
      ReadsB d init (pidMOf d u * 128 + ii.val) (pidNOf d u * 128 + jj.val) vb →
      ∃ v : F32, acc.get2 ii jj = v ∧ good v ∧
        |v.toRat - (pre va vb d.K (c0 d i u + t) - pre va vb d.K (c0 d i u))| ≤
          fineOne va vb d.K (c0 d i u) t

/-- The accumulator starts zeroed. -/
theorem accI_zero (d : Inst) (init : Mem) (i u : Nat) :
    AccI d init i u 0 (tl.zeros [128, 128] DType.f32) := by
  intro ii jj _ _ va vb _ _
  have hget : (tl.zeros [128, 128] DType.f32).get2 ii jj = (FloatV.ofPat 0 : F32) := rfl
  refine ⟨FloatV.ofPat 0, hget, FloatV.isFinite_zero_f32, ?_⟩
  rw [FloatV.toRat_ofPat_zero, zero_sub]
  rw [show c0 d i u + 0 = c0 d i u from by omega, sub_self, neg_zero, abs_zero]
  rw [show fineOne va vb d.K (c0 d i u) 0 = 0 from by unfold fineOne; simp]

/-- One chunk iteration of the inner loop, on the accumulator `acc`. -/
def ChunkBody (d : Inst) (offs_m offs_n : Tensor [128] I32) (kk : I32)
    (acc : Tensor [128, 128] F32) : Prog (Tensor [128, 128] F32) :=
  have offs_k := kk * (IntV.ofNat 64 : I32) + tl.arange 0 64;
  do
    let __do_lift ← tl.convert DType.f32 DType.f16 (tl.full [128, 64] (FloatV.ofPat 0).litF32)
    let a ←
      tl.load (d.a.ptr 0 + offs_m.col 64 * (IntV.ofNat d.stride_am : I32) + offs_k.row 128 * (IntV.ofNat d.stride_ak : I32))
          (some (((offs_m.col 64).ltS ((IntV.ofNat d.M : I32))).and ((offs_k.row 128).ltS ((IntV.ofNat d.K : I32)))))
          (some __do_lift)
    let __do_lift ← tl.convert DType.f32 DType.f16 (tl.full [64, 128] (FloatV.ofPat 0).litF32)
    let b ←
      tl.load (d.b.ptr 0 + offs_k.col 128 * (IntV.ofNat d.stride_bk : I32) + offs_n.row 64 * (IntV.ofNat d.stride_bn : I32))
          (some (((offs_k.col 128).ltS ((IntV.ofNat d.K : I32))).and ((offs_n.row 64).ltS ((IntV.ofNat d.N : I32)))))
          (some __do_lift)
    tl.dot { a := DType.f16, b := DType.f16, acc := DType.f32 } a b acc

/-- One lane of the accumulator: the partial dot product of the tile's output element. -/
theorem accI_cell {init : Mem} {i u t : Nat} {acc : Tensor [128, 128] F32}
    (hacc : AccI d init i u t acc) {ii jj : Fin 128}
    (hi : pidMOf d u * 128 + ii.val < d.M) (hj : pidNOf d u * 128 + jj.val < d.N)
    {va vb : Nat → F16}
    (hva : ReadsA d init (pidMOf d u * 128 + ii.val) (pidNOf d u * 128 + jj.val) va)
    (hvb : ReadsB d init (pidMOf d u * 128 + ii.val) (pidNOf d u * 128 + jj.val) vb) :
    ∃ v : F32, acc.get2 ii jj = v ∧ good v ∧
      |v.toRat - (pre va vb d.K (c0 d i u + t) - pre va vb d.K (c0 d i u))| ≤
        fineOne va vb d.K (c0 d i u) t :=
  hacc ii jj hi hj va vb hva hvb

/-- **The tight bound on a lane**: the partial sum's deviation plus the accumulated fine error. -/
theorem accI_bnd {init : Mem} {i u t : Nat} {acc : Tensor [128, 128] F32}
    (hacc : AccI d init i u t acc) {ii jj : Fin 128}
    (hi : pidMOf d u * 128 + ii.val < d.M) (hj : pidNOf d u * 128 + jj.val < d.N)
    {va vb : Nat → F16}
    (hva : ReadsA d init (pidMOf d u * 128 + ii.val) (pidNOf d u * 128 + jj.val) va)
    (hvb : ReadsB d init (pidMOf d u * 128 + ii.val) (pidNOf d u * 128 + jj.val) vb) :
    ∃ v : F32, acc.get2 ii jj = v ∧ good v ∧
      |v.toRat| ≤ dev va vb d.K (c0 d i u + t) + fineOne va vb d.K (c0 d i u) t := by
  obtain ⟨v, hv, hg, hb⟩ := hacc ii jj hi hj va vb hva hvb
  refine ⟨v, hv, hg, ?_⟩
  have hdev : |pre va vb d.K (c0 d i u + t) - pre va vb d.K (c0 d i u)| ≤
      dev va vb d.K (c0 d i u + t) :=
    le_dev (va := va) (vb := vb) (K := d.K) (s := c0 d i u) (t := c0 d i u + t)
      (Nat.le_add_right _ _)
  have hsplit : |v.toRat| ≤
      |v.toRat - (pre va vb d.K (c0 d i u + t) - pre va vb d.K (c0 d i u))| +
        |pre va vb d.K (c0 d i u + t) - pre va vb d.K (c0 d i u)| := by
    have := abs_sub_abs_le_abs_sub v.toRat
      (pre va vb d.K (c0 d i u + t) - pre va vb d.K (c0 d i u))
    linarith [neg_le_abs v.toRat]
  linarith [abs_nonneg v.toRat]

/-- An in-bounds lane of the accumulator is a finite `float32` value of magnitude at most
`2^46` (`2^45` of partial-sum deviation plus `2^38` of fine error), which gives the room of
`Mag.dot_room`. -/
theorem accI_mag (hd : d.ok) {init : Mem} {i u t : Nat} {acc : Tensor [128, 128] F32}
    (hacc : AccI d init i u t acc) (ht : t ≤ 128) {ii jj : Fin 128}
    (hi : pidMOf d u * 128 + ii.val < d.M) (hj : pidNOf d u * 128 + jj.val < d.N)
    {va vb : Nat → F16}
    (hva : ReadsA d init (pidMOf d u * 128 + ii.val) (pidNOf d u * 128 + jj.val) va)
    (hvb : ReadsB d init (pidMOf d u * 128 + ii.val) (pidNOf d u * 128 + jj.val) vb) :
    ∃ v : F32, acc.get2 ii jj = v ∧ good v ∧ |v.toRat| ≤ 2 ^ 46 := by
  obtain ⟨v, hv, hg, hb⟩ := hacc ii jj hi hj va vb hva hvb
  have hbva : ∀ k, k < d.K → |(va k).toRat| ≤ 65504 := fun k hk =>
    f16_toRat_abs_le (hva k hk).2
  have hbvb : ∀ k, k < d.K → |(vb k).toRat| ≤ 65504 := fun k hk =>
    f16_toRat_abs_le (hvb k hk).2
  have hdev : |pre va vb d.K (c0 d i u + t) - pre va vb d.K (c0 d i u)| ≤ 2 ^ 45 :=
    (le_dev (va := va) (vb := vb) (K := d.K) (s := c0 d i u) (t := c0 d i u + t)
      (Nat.le_add_right _ _)).trans
      (abs_le.mp (dev_abs_le (va := va) (vb := vb) (t := c0 d i u + t) hd.K_le hbva hbvb)).2
  have hfine : fineOne va vb d.K (c0 d i u) t ≤ 2 ^ 38 :=
    fineOne_le hd.K_le ht hbva hbvb
  refine ⟨v, hv, hg, ?_⟩
  have hsplit : |v.toRat| ≤
      |v.toRat - (pre va vb d.K (c0 d i u + t) - pre va vb d.K (c0 d i u))| +
        |pre va vb d.K (c0 d i u + t) - pre va vb d.K (c0 d i u)| := by
    have := abs_sub_abs_le_abs_sub v.toRat
      (pre va vb d.K (c0 d i u + t) - pre va vb d.K (c0 d i u))
    linarith [neg_le_abs v.toRat]
  linarith

/-- **The end of an owned tile's chunk run**: the accumulator holds the full dot product up to
`fineOne`. -/
theorem accI_val {init : Mem} {i u : Nat} {acc : Tensor [128, 128] F32} (hd : d.ok)
    (ho : owns d i u) (hacc : AccI d init i u (steps d.K) acc) {ii jj : Fin 128}
    (hi : pidMOf d u * 128 + ii.val < d.M) (hj : pidNOf d u * 128 + jj.val < d.N)
    {va vb : Nat → F16}
    (hva : ReadsA d init (pidMOf d u * 128 + ii.val) (pidNOf d u * 128 + jj.val) va)
    (hvb : ReadsB d init (pidMOf d u * 128 + ii.val) (pidNOf d u * 128 + jj.val) vb) :
    ∃ v : F32, acc.get2 ii jj = v ∧ good v ∧
      |v.toRat - dotSum va vb d.K| ≤ fineOne va vb d.K 0 (steps d.K) := by
  obtain ⟨v, hv, hg, hb⟩ := hacc ii jj hi hj va vb hva hvb
  refine ⟨v, hv, hg, ?_⟩
  have h0 := ho.1
  rw [h0] at hb
  rw [Nat.zero_add] at hb
  rw [show pre va vb d.K 0 = 0 from by unfold pre; simp] at hb
  rw [sub_zero] at hb
  rw [pre_steps (va := va) (vb := vb) hd.K_pos] at hb
  exact hb

/-- **The end of a split run** is the `RunVal` of the tile's output element. -/
theorem accI_runVal {init : Mem} {i u : Nat} {acc : Tensor [128, 128] F32}
    (hacc : AccI d init i u (c1 d i u - c0 d i u) acc) (hle : c0 d i u ≤ c1 d i u)
    {ii jj : Fin 128}
    (hi : pidMOf d u * 128 + ii.val < d.M) (hj : pidNOf d u * 128 + jj.val < d.N) :
    RunVal d init i u (pidMOf d u * 128 + ii.val, pidNOf d u * 128 + jj.val)
      (acc.get2 ii jj).toRat := by
  refine ⟨acc.get2 ii jj, rfl, fun va vb hva hvb => ?_⟩
  obtain ⟨v, hv, hg, hb⟩ := hacc ii jj hi hj va vb hva hvb
  have hpre : pre va vb d.K (c0 d i u + (c1 d i u - c0 d i u)) = pre va vb d.K (c1 d i u) := by
    rw [Nat.add_sub_cancel' hle]
  rw [hpre] at hb
  rw [hv]
  exact ⟨hg, hb⟩

end Claims.StreamKMatmul


/-!
# The loop invariant of one program

While the outer `while` loop runs, only the tiles the program has already handled are finished:
a tile it owns holds its full dot product, and a split tile has received the program's atomic
addition.  So the invariant `Q` of `Proof/Q.lean` is asserted for a tile only once the tile is
*available*, which the predicate `gate` records (a tile is available when its chunk range ends at
or before the current iteration value).

The measures for the loop rule (`Proof/SharedA.lean`, `SafeA.whileLoop`) are also here: `mu`
counts the tiles still to come, and `Wv` is the write footprint of each value of `mu`.
-/

namespace Claims.StreamKMatmul

open TVA

variable {d : Inst}

/-! ## The availability of a tile -/

/-- Tiles of the launch. -/
def T (d : Inst) : Nat := tilesM d * tilesN d

/-- `steps K` is positive, as a natural number. -/
theorem T_nat_pos {d : Inst} (hd : d.ok) : 0 < T d := by
  unfold T
  exact Nat.mul_pos (tilesM_pos hd) (tilesN_pos hd)

/-- The `Nat` value of the loop's `it`. -/
def itN (st : I32) : Nat := st.toInt.toNat

/-- Tile `u` has been handled once the iteration value reaches the end of its chunk range. -/
abbrev gate (d : Inst) (i it u : Nat) : Prop := min (u * steps d.K + steps d.K) (hi d i) ≤ it

/-- The gate stays open when `it` grows. -/
theorem gate_mono {d : Inst} {i u it it' : Nat} (h : it ≤ it') (hg : gate d i it u) :
    gate d i it' u := le_trans hg h

/-- **The gated invariant.**  `Q`, with the finished-tile clauses only for tiles `u` that
`gate d i it u` admits. -/
def Qg (d : Inst) (init : Mem) (i it : Nat) (log : List AtomicStep) (m : Mem) : Prop :=
  ∃ L : List (AtomicStep × (Nat × Nat → Option ℚ)),
    L.map Prod.fst = log ∧
    (∀ p ∈ L, ∀ q ∈ splitOuts d, ∀ va vb : Nat → F16,
      ReadsA d init q.1 q.2 va → ReadsB d init q.1 q.2 vb →
      AccStep .f32 (cAddr d q.1 q.2) FloatV.toRat good aRho aAlpha
        Format.binary32.maxVal p.1 (p.2 q)) ∧
    (∀ x, x ∉ touchedBytes d i → m x = init x) ∧
    (∀ u ∈ tiles d, owns d i u → gate d i it u → ∀ q ∈ tileOut d u,
      ∃ v : F32, m.read .f32 (cAddr d q.1 q.2) = some v ∧
        ∀ va vb : Nat → F16, ReadsA d init q.1 q.2 va → ReadsB d init q.1 q.2 vb →
          good v ∧ |v.toRat - dotSum va vb d.K| ≤ fineOne va vb d.K 0 (steps d.K)) ∧
    (∀ p ∈ L, ∀ q ∈ splitOuts d, ∀ w : ℚ, p.2 q = some w →
      ∃ u ∈ tiles d, splitTile d u ∧ q ∈ tileOut d u ∧ ¬ owns d i u ∧ RunVal d init i u q w) ∧
    (∀ q ∈ splitOuts d, ∀ u ∈ tiles d, q ∈ tileOut d u →
      (L.filter (fun p => (p.2 q).isSome)).length =
        if (c0 d i u < c1 d i u ∧ gate d i it u) then 1 else 0)

/-- The gated invariant is invariant under a permutation of the log. -/
theorem Qg_perm {init : Mem} {i it : Nat} {log log' : List AtomicStep} {m : Mem}
    (h : Qg d init i it log m) (hp : log.Perm log') : Qg d init i it log' m := by
  obtain ⟨L, hm, hacc, hframe, hval, hrun, hcnt⟩ := h
  obtain ⟨L', hL', hm'⟩ := exists_perm_of_map_perm Prod.fst (hm ▸ hp : (L.map Prod.fst).Perm log')
  exact ⟨L', hm', fun p hp' q hq va vb hva hvb => hacc p (hL'.subset hp') q hq va vb hva hvb,
    hframe,
    fun u hu ho hg q hq => hval u hu ho hg q hq,
    fun p hp' q hq w hw => hrun p (hL'.subset hp') q hq w hw,
    fun q hq u hu hqu => by
      have := hcnt q hq u hu hqu
      have heq : (L.filter (fun p => (p.2 q).isSome)).length
          = (L'.filter (fun p => (p.2 q).isSome)).length :=
        (hL'.filter (fun p => (p.2 q).isSome)).length_eq.symm
      rw [← heq]
      exact this⟩

/-- `Qg` is a `PermInv`, as `TVA.shared` demands. -/
theorem Qg_permInv (init : Mem) (i it : Nat) :
    PermInv (fun (_ : Unit) log m => Qg d init i it log m) := by
  intro a l l' m hp hQ
  exact Qg_perm hQ hp

/-! ## The invariant at the ends of the loop -/

/-- At the end of the loop every tile is available, so the gated invariant is the invariant. -/
theorem Qg_end {init : Mem} {i it : Nat} {log : List AtomicStep} {m : Mem}
    (h : Qg d init i it log m) (hge : hi d i ≤ it) : Q d init i log m := by
  obtain ⟨L, hm, hacc, hframe, hval, hrun, hcnt⟩ := h
  refine ⟨L, hm, hacc, hframe,
    fun u hu ho hq => hval u hu ho (le_trans (min_le_right _ _) hge) hq,
    hrun, fun q hq u hu hqu => ?_⟩
  have hgate : gate d i it u := le_trans (min_le_right _ _) hge
  have h := hcnt q hq u hu hqu
  by_cases hlt : c0 d i u < c1 d i u
  · simpa only [ite_eq_left (show c0 d i u < c1 d i u ∧ gate d i it u from ⟨hlt, hgate⟩), ite_eq_left hlt]
      using h
  · simpa only [ite_eq_right hlt, ite_eq_right (show ¬ (c0 d i u < c1 d i u ∧ gate d i it u) from
      fun hh => hlt hh.1)] using h

/-- The initial memory satisfies the gated invariant before the loop: the tiles available at `lo`
are the ones before it, whose chunk range is empty. -/
theorem Qg_init (hd : d.ok) {init : Mem} {i : Nat} :
    Qg d init i (lo d i) [] init := by
  have hT : 0 < steps d.K := T_pos hd
  refine ⟨[], rfl, ?_, ?_, ?_, ?_, ?_⟩
  · intro p hp
    simp at hp
  · intro x _
    rfl
  · intro u hu ho hg q hq
    obtain ⟨hc0, hc1⟩ := ho
    unfold gate at hg
    by_cases hA : u * steps d.K + steps d.K ≤ lo d i
    · have hc : c0 d i u = steps d.K := by
        unfold c0
        rw [min_eq_left]
        exact le_max_of_le_right (by omega)
      have hcl : c1 d i u ≤ steps d.K := c1_le
      omega
    · have hB : hi d i ≤ lo d i := by omega
      have hle : c1 d i u ≤ c0 d i u := by
        unfold c1 c0
        exact min_le_min le_rfl (max_le_max le_rfl (by omega))
      omega
  · intro p hp
    simp at hp
  · intro q hq u hu hqu
    simp only [List.filter_nil, List.length_nil]
    have hnot : ¬ (c0 d i u < c1 d i u ∧ min (u * steps d.K + steps d.K) (hi d i) ≤ lo d i) := by
      rintro ⟨hlt, hg⟩
      rcases min_le_iff.mp hg with h | h
      · have hc : c0 d i u = steps d.K := by
          unfold c0
          rw [min_eq_left]
          exact le_max_of_le_right (by omega)
        have hcl : c1 d i u ≤ steps d.K := c1_le
        omega
      · have hle : c1 d i u ≤ c0 d i u := by
          unfold c1 c0
          exact min_le_min le_rfl (max_le_max le_rfl (by omega))
        omega
    rw [ite_eq_right hnot]

/-! ## The measure and the per-tile write footprint -/

/-- The number of tiles still to come: once the loop has passed the last tile, `0`. -/
def mu (d : Inst) (i : Nat) (st : I32) : Nat :=
  T d - (if itN st < hi d i then tileOf d (itN st) else T d)

/-- The tile index `T - r` is a tile for `1 ≤ r ≤ T`. -/
theorem sub_mem_tiles {r : Nat} (hr : 1 ≤ r ∧ r ≤ T d) : T d - r ∈ tiles d := by
  show T d - r < T d
  omega

/-- The write footprint indexed by `mu`: program `i` stores the output of the tile `T - r` when it
owns it, and writes nothing there when it does not. -/
def Wv (d : Inst) (i r : Nat) : Set Int :=
  if 1 ≤ r ∧ r ≤ T d then
    (if owns d i (T d - r) then outBytes d (tileOut d (T d - r)) else ∅)
  else ∅

/-- The footprint `Wv d i r` lies in `Wset d i`. -/
theorem Wv_sub_Wset {i r : Nat} : Wv d i r ⊆ Wset d i := by
  intro x hx
  unfold Wv at hx
  by_cases hr : 1 ≤ r ∧ r ≤ T d
  · rw [ite_eq_left hr] at hx
    by_cases ho : owns d i (T d - r)
    · rw [ite_eq_left ho] at hx
      have hsub : tileOut d (T d - r) ⊆ storedOuts d i :=
        fun q hq => ⟨T d - r, sub_mem_tiles hr, ho, hq⟩
      exact outBytes_mono hsub hx
    · rw [ite_eq_right ho] at hx
      simp at hx
  · rw [ite_eq_right hr] at hx
    simp at hx

/-- Footprints `Wv d i u` and `Wv d i v` of different indices are disjoint. -/
theorem Wv_disjoint (hd : d.ok) {i u v : Nat} (h : u < v) :
    Disjoint (Wv d i u) (Wv d i v) := by
  rw [Set.disjoint_left]
  rintro x hx hx'
  unfold Wv at hx hx'
  by_cases hu : 1 ≤ u ∧ u ≤ T d
  · rw [ite_eq_left hu] at hx
    by_cases hv : 1 ≤ v ∧ v ≤ T d
    · rw [ite_eq_left hv] at hx'
      have hne : T d - u ≠ T d - v := by omega
      by_cases ho : owns d i (T d - u)
      · rw [ite_eq_left ho] at hx
        by_cases ho' : owns d i (T d - v)
        · rw [ite_eq_left ho'] at hx'
          have h1 : T d - u ∈ tiles d := sub_mem_tiles hu
          have h2 : T d - v ∈ tiles d := sub_mem_tiles hv
          refine Set.disjoint_left.mp (disjoint_outBytes ?_) hx hx'
          intro q hq q' hq'
          have hqne : q ≠ q' := by
            intro he
            exact Set.disjoint_left.mp (tileOut_disjoint h1 h2 hne) hq (he ▸ hq')
          exact span_disjoint_of_ne hd (tileOut_sub_allOuts _ hq) (tileOut_sub_allOuts _ hq')
            hqne
        · rw [ite_eq_right ho'] at hx'
          simp at hx'
      · rw [ite_eq_right ho] at hx
        simp at hx
    · rw [ite_eq_right hv] at hx'
      simp at hx'
  · rw [ite_eq_right hu] at hx
    simp at hx

/-- The loop's write footprint is inside the program's stores. -/
theorem iterW_Wv_sub_Wset {i n : Nat} : iterW (Wv d i) n ⊆ Wset d i := by
  rintro x ⟨r, hr, hx⟩
  exact Wv_sub_Wset hx

end Claims.StreamKMatmul


/-!
# Buffer accesses inside one chunk iteration

The loads of `a` and `b` and the store to `c` touch single elements of the three matrices.  The
memory they run on is not the initial one: earlier programs' atomic additions have changed `c`.
The library's `safe_load_buf` / `safe_store_buf` want the whole buffer present (`Buf.Valid`), which
is not known after the atomics; what *is* known is that every byte of the three matrices is
present (`Pres`), because atomics write cells with values and stores never remove a byte.  From
`Pres` and the initial `Buf.Valid` we get `Buf.Valid` again (`valid_of_buf`), and so line-level
loads can read and store lanes can write.

This file also identifies the footprints: the `c` lanes of a tile cover exactly the bytes of the
tile's output elements (`cAccBytes`), which is the per-tile write footprint `Wv` of `Proof/Inv.lean`
(`Wv_of_owns`, `Wv_of_not_owns`).
-/

namespace Claims.StreamKMatmul

open TVA

variable {d : Inst}

/-! ## Presence of the matrix bytes -/

/-- Every byte of the three matrices is present in memory. -/
def Pres (d : Inst) (m : Mem) : Prop :=
  ∀ x ∈ d.c.bytes ∪ d.a.bytes ∪ d.b.bytes, m x ≠ none

/-- The bytes of an `a` tile lie in the input footprint. -/
theorem aBytes_sub : d.a.bytes ⊆ d.c.bytes ∪ d.a.bytes ∪ d.b.bytes :=
  fun _ hx => Or.inl (Or.inr hx)

/-- The bytes of a `b` tile lie in the input footprint. -/
theorem bBytes_sub : d.b.bytes ⊆ d.c.bytes ∪ d.a.bytes ∪ d.b.bytes :=
  fun _ hx => Or.inr hx

/-- The bytes of a `c` tile lie in the output matrix. -/
theorem cBytes_sub : d.c.bytes ⊆ d.c.bytes ∪ d.a.bytes ∪ d.b.bytes :=
  fun _ hx => Or.inl (Or.inl hx)

/-- The initial memory satisfies `Pres`: the buffers are present. -/
theorem pres_init {init : Mem} (hpre : Pre d init) : Pres d init := by
  have ha := hpre.1
  have hb := hpre.2.1
  have hc := hpre.2.2.1
  rintro x ((hx | hx) | hx)
  · exact hc.2.2.2 x hx
  · exact ha.2.2.2 x hx
  · exact hb.2.2.2 x hx

/-- Presence is kept by any memory whose bytes include the ones already present. -/
theorem pres_of_ne {m m' : Mem} (h : Pres d m) (hfr : ∀ x, m x ≠ none → m' x ≠ none) :
    Pres d m' :=
  fun x hx => hfr x (h x hx)

/-- A write outside the matrices keeps presence. -/
theorem pres_write {m : Mem} {dd : DType} {a : Int} (v : dd.Val) (h : Pres d m)
    (hd : Disjoint (Mem.span dd a) (d.c.bytes ∪ d.a.bytes ∪ d.b.bytes)) :
    Pres d (m.write dd a v) := by
  intro x hx
  rw [Mem.write_frame _ _ _ _ (fun hs => Set.disjoint_left.mp hd hs hx)]
  exact h x hx

/-- **Presence gives validity back**: the bounds of a buffer are fixed, so only the presence of
its bytes is needed. -/
theorem valid_of_buf {dd : DType} {b : Buf dd} {init m : Mem} (hv : b.Valid init) (hp : Pres d m)
    (hsub : b.bytes ⊆ d.c.bytes ∪ d.a.bytes ∪ d.b.bytes) : b.Valid m :=
  Buf.valid_of_agree hv fun x hx => hp x (hsub hx)

/-- The enabled lanes of an access to `b` can be read in any memory in which the matrix bytes are
present.  This is the `hval` hypothesis of `safe_load`. -/
theorem read_of_addrs {dd : DType} {b : Buf dd} {init m : Mem} (hv : b.Valid init) (hp : Pres d m)
    (hsub : b.bytes ⊆ d.c.bytes ∪ d.a.bytes ∪ d.b.bytes) {s : Shape}
    {ptrs : Tensor s (Ptr dd)} {mask : Option (Tensor s Bool)} {g : Idx s → Nat} {en : Idx s → Bool}
    (hA : b.Addrs (tl.addrs ptrs mask) g en) (hin : ∀ i, en i = true → g i < b.len) :
    ∀ i a, tl.addrs ptrs mask i = some a → ∃ w, m.read dd a = some w := by
  intro i a ha
  rw [hA i] at ha
  by_cases he : en i = true
  · rw [ite_eq_left he] at ha
    cases ha
    exact b.read_some (valid_of_buf hv hp hsub) (hin i he)
  · rw [ite_eq_right he] at ha
    exact absurd ha (by simp)

/-- The enabled lanes of an access to `b` are writable in any memory in which the matrix bytes are
present.  This is the `StoreOk` hypothesis of `safe_store`. -/
theorem storeOk_of_addrs {dd : DType} {b : Buf dd} {init m : Mem} (hv : b.Valid init) (hp : Pres d m)
    (hsub : b.bytes ⊆ d.c.bytes ∪ d.a.bytes ∪ d.b.bytes) {s : Shape}
    {ptrs : Tensor s (Ptr dd)} {mask : Option (Tensor s Bool)} {g : Idx s → Nat} {en : Idx s → Bool}
    (hA : b.Addrs (tl.addrs ptrs mask) g en) (hin : ∀ i, en i = true → g i < b.len) :
    StoreOk m dd (tl.addrs ptrs mask) := by
  intro i a ha
  rw [hA i] at ha
  by_cases he : en i = true
  · rw [ite_eq_left he] at ha
    cases ha
    exact b.writable (valid_of_buf hv hp hsub) (hin i he)
  · rw [ite_eq_right he] at ha
    exact absurd ha (by simp)

/-! ## The read footprints -/

/-- The bytes a masked load of an `a` tile reads lie in the input footprint. -/
theorem aAccBytes_sub (hd : d.ok) {pm kk : I32} {u v : Nat}
    (hpm : pm.toInt = (pidMOf d u : Int)) (hkk : kk.toInt = (v : Int))
    (hu : u < tilesM d * tilesN d) (hv : v < steps d.K) :
    accBytes .f16
        (tl.addrs (ptrsA (d.a.ptr 0) (offs1 pm 128) (offs1 kk 64) (IntV.ofNat d.stride_am)
            (IntV.ofNat d.stride_ak))
          (some (maskA (offs1 pm 128) (offs1 kk 64) (IntV.ofNat d.M) (IntV.ofNat d.K))))
      ⊆ inElems d := by
  rw [(aAddrs hd hpm hkk hu hv).accBytes]
  exact fun x hx => Or.inl (Buf.elems_mono _ (aImg_sub hd hu hv) hx)

/-- The bytes a masked load of a `b` tile reads lie in the input footprint. -/
theorem bAccBytes_sub (hd : d.ok) {kk pn : I32} {v w : Nat}
    (hkk : kk.toInt = (v : Int)) (hpn : pn.toInt = (w : Int))
    (hv : v < steps d.K) (hw : w < tilesN d) :
    accBytes .f16
        (tl.addrs (ptrsB (d.b.ptr 0) (offs1 kk 64) (offs1 pn 128) (IntV.ofNat d.stride_bk)
            (IntV.ofNat d.stride_bn))
          (some (maskB (offs1 kk 64) (offs1 pn 128) (IntV.ofNat d.K) (IntV.ofNat d.N))))
      ⊆ inElems d := by
  rw [(bAddrs hd hkk hpn hv hw).accBytes]
  exact fun x hx => Or.inr (Buf.elems_mono _ (bImg_sub hd hv hw) hx)

/-! ## The write footprint of a tile -/

/-- The `c` lanes of a tile touch exactly the bytes of the tile's output elements. -/
theorem cAccBytes (hd : d.ok) {pm pn : I32} {u : Nat}
    (hpm : pm.toInt = (pidMOf d u : Int)) (hpn : pn.toInt = (pidNOf d u : Int))
    (hu : u < tilesM d * tilesN d) :
    accBytes .f32
        (tl.addrs (ptrsC (d.c.ptr 0) (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.stride_cm)
            (IntV.ofNat d.stride_cn))
          (some (maskC (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.M) (IntV.ofNat d.N))))
      = outBytes d (tileOut d u) := by
  rw [(cAddrs hd hpm hpn hu).accBytes]
  ext x
  refine Iff.intro ?_ ?_
  · rintro ⟨j, ⟨i, hen, rfl⟩, hx⟩
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hen
    obtain ⟨h1, h2⟩ := hen
    have e1 : (pidMOf d u * 128 + i.1.val) / 128 = pidMOf d u := by
      rw [Nat.mul_comm (pidMOf d u) 128,
        Nat.mul_add_div (by norm_num : 0 < 128), Nat.div_eq_of_lt i.1.isLt, Nat.add_zero]
    have e2 : (pidNOf d u * 128 + i.2.1.val) / 128 = pidNOf d u := by
      rw [Nat.mul_comm (pidNOf d u) 128,
        Nat.mul_add_div (by norm_num : 0 < 128), Nat.div_eq_of_lt i.2.1.isLt, Nat.add_zero]
    exact ⟨(pidMOf d u * 128 + i.1.val, pidNOf d u * 128 + i.2.1.val),
      ⟨⟨h1, h2⟩, e1, e2⟩, by simpa only [cAddr] using hx⟩
  · rintro ⟨q, hq, hx⟩
    obtain ⟨⟨hq1, hq2⟩, hqm, hqn⟩ := hq
    have hm : pidMOf d u * 128 + q.1 % 128 = q.1 := by
      have h := Nat.div_add_mod q.1 128
      rw [hqm, Nat.mul_comm 128 (pidMOf d u)] at h
      exact h
    have hn : pidNOf d u * 128 + q.2 % 128 = q.2 := by
      have h := Nat.div_add_mod q.2 128
      rw [hqn, Nat.mul_comm 128 (pidNOf d u)] at h
      exact h
    have hen : (fun i : Idx [128, 128] =>
        decide (pidMOf d u * 128 + i.1.val < d.M) &&
          decide (pidNOf d u * 128 + i.2.1.val < d.N))
        (⟨q.1 % 128, Nat.mod_lt q.1 (by norm_num)⟩,
          ⟨q.2 % 128, Nat.mod_lt q.2 (by norm_num)⟩, ()) = true := by
      simp only [Bool.and_eq_true, decide_eq_true_eq]
      exact ⟨by rw [hm]; exact hq1, by rw [hn]; exact hq2⟩
    exact ⟨(pidMOf d u * 128 + q.1 % 128) * d.stride_cm +
        (pidNOf d u * 128 + q.2 % 128) * d.stride_cn,
      ⟨(⟨q.1 % 128, Nat.mod_lt q.1 (by norm_num)⟩,
        ⟨q.2 % 128, Nat.mod_lt q.2 (by norm_num)⟩, ()), hen, rfl⟩,
      by
        have hx' : x ∈ Mem.span .f32
            (d.c.ptr ((q.1 * d.stride_cm + q.2 * d.stride_cn : Nat) : Int)).addr := by
          simpa only [cAddr] using hx
        rw [← hm, ← hn] at hx'
        exact hx'⟩

/-- The tiles of a chunk range fit in the write footprint of the program. -/
theorem outBytes_tileOut_sub_touched (_hd : d.ok) {i u : Nat} (hu : u < T d)
    (hne : c0 d i u < c1 d i u) :
    outBytes d (tileOut d u) ⊆ touchedBytes d i := by
  rintro x ⟨q, hq, hx⟩
  exact ⟨q, ⟨u, hu, hne, hq⟩, hx⟩

/-- The operands and the bytes program `i` writes are disjoint. -/
theorem inElems_disjoint_touched (hd : d.ok) (p : Nat) :
    Disjoint (inElems d) (touchedBytes d p) := by
  have hE : touchedBytes d p ⊆ outElems d := by
    rw [touchedBytes, outElems_eq]
    exact outBytes_mono (fun q hq => by
      obtain ⟨u, _hu, _hne, hqu⟩ := hq
      exact tileOut_sub_allOuts u hqu)
  have hC : touchedBytes d p ⊆ d.c.elems (cIdx d) := hE
  have ha : Disjoint (d.a.elems (aIdx d)) (touchedBytes d p) := hd.c_a.symm.mono_right hC
  have hb : Disjoint (d.b.elems (bIdx d)) (touchedBytes d p) := hd.c_b.symm.mono_right hC
  show Disjoint (d.a.elems (aIdx d) ∪ d.b.elems (bIdx d)) (touchedBytes d p)
  exact ha.union_left hb

/-- The lanes of the current tile write exactly `Wv` at the measure's current value, when the
program owns the tile. -/
theorem Wv_of_owns {i u : Nat} (hu : u < T d) (ho : owns d i u) :
    Wv d i (T d - u) = outBytes d (tileOut d u) := by
  have h : T d - (T d - u) = u := by omega
  unfold Wv
  rw [ite_eq_left ⟨by omega, by omega⟩, h, ite_eq_left ho]

/-- … and nothing at all when it does not. -/
theorem Wv_of_not_owns {i u : Nat} (hu : u < T d) (ho : ¬ owns d i u) :
    Wv d i (T d - u) = ∅ := by
  have h : T d - (T d - u) = u := by omega
  unfold Wv
  rw [ite_eq_left ⟨by omega, by omega⟩, h, ite_eq_right ho]

end Claims.StreamKMatmul


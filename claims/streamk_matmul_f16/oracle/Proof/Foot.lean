import Claim
import TVA.Proof
import Proof.Tol

/-!
# `Proof.Foot`

The per-program footprints: the stored output elements `Wset`, the shared cells `Ccells`, the inputs, and the invariant `Q` on the log of atomic steps.
-/

/-!
# The per-program invariant `Q` of `TVA.shared`

`TVA.shared` needs, for every program `i`, a predicate `Q i log m` on the program's atomic log and
on the memory it ends with.  The program

* writes only the output cells of the tiles it handles (`touchedBytes`), so outside them its
  memory is still the initial memory;
* leaves each output element of a tile it **owns** (its chunk run is the whole tile) holding the
  `fineOne`-accurate full dot product (it takes the `tl.store` branch);
* logs, for every atomic accumulation into a split tile's element `q`, the operand `acc.toRat` it
  added (the `AccStep` pairing, which `replay_err_shared` consumes cell by cell).

`Q` is permutation invariant, as `TVA.shared` requires.
-/

namespace Claims.StreamKMatmul

open TVA

variable {d : Inst}

/-! ## The float-accumulation parameters -/

/-- A float32 value is *good* if it is finite. -/
def good (x : F32) : Prop := x.isFinite = true

/-- The relative constant of one `float32` atomic add (`Device.AtomicAddF32`). -/
def aRho : ℚ := 2 ^ (-23 : ℤ)

/-- The absolute constant of one `float32` atomic add. -/
def aAlpha : ℚ := 2 ^ (-124 : ℤ)

/-- `aRho` is nonnegative. -/
theorem aRho_nonneg : 0 ≤ aRho := by unfold aRho; positivity
/-- `aAlpha` is nonnegative. -/
theorem aAlpha_nonneg : 0 ≤ aAlpha := by unfold aAlpha; positivity

/-! ## The initial reads the output specification quantifies over -/

/-- The `a` reads of row `i` are `va`. -/
def ReadsA (d : Inst) (init : Mem) (i _j : Nat) (va : Nat → F16) : Prop :=
  ∀ kk, kk < d.K →
    init.read .f16 (d.a.ptr ((i * d.stride_am + kk * d.stride_ak : Nat) : Int)).addr = some (va kk) ∧
      (va kk).isFinite = true

/-- The `b` reads of column `j` are `vb`. -/
def ReadsB (d : Inst) (init : Mem) (_i j : Nat) (vb : Nat → F16) : Prop :=
  ∀ kk, kk < d.K →
    init.read .f16 (d.b.ptr ((kk * d.stride_bk + j * d.stride_bn : Nat) : Int)).addr = some (vb kk) ∧
      (vb kk).isFinite = true

/-! ## The bytes a program writes -/

/-- The output elements of the tiles program `i` handles at all (a nonempty chunk range). -/
def touchedOuts (d : Inst) (i : Nat) : Set (Nat × Nat) :=
  {q | ∃ u ∈ tiles d, c0 d i u < c1 d i u ∧ q ∈ tileOut d u}

/-- The bytes program `i` writes. -/
def touchedBytes (d : Inst) (i : Nat) : Set Int := outBytes d (touchedOuts d i)

/-- A program writes only inside the bytes it touches. -/
theorem Wset_sub_touched (hd : d.ok) (i : Nat) : Wset d i ⊆ touchedBytes d i := by
  rintro x ⟨q, ⟨u, hu, how, hq⟩, hx⟩
  exact ⟨q, ⟨u, hu, by rw [how.1, how.2]; exact T_pos hd, hq⟩, hx⟩

/-! ## One atomic run's value -/

/-- The value one atomic run of program `i` on tile `u` adds to the output element `q`: it is the
`fineOne`-accurate partial sum of the run's chunks, `pre (c1) - pre (c0)`. -/
def RunVal (d : Inst) (init : Mem) (i u : Nat) (q : Nat × Nat) (w : ℚ) : Prop :=
  ∃ v : F32, w = v.toRat ∧
    ∀ va vb : Nat → F16, ReadsA d init q.1 q.2 va → ReadsB d init q.1 q.2 vb →
      good v ∧
        |v.toRat - (pre va vb d.K (c1 d i u) - pre va vb d.K (c0 d i u))| ≤
          fineOne va vb d.K (c0 d i u) (c1 d i u - c0 d i u)

/-! ## The invariant -/

/-- **The per-program invariant.**  Program `i` starts from `init` and, having performed the
atomic steps of `log`, ends with the memory `m` such that

* `log` is paired, cell by cell over the split output elements, with the operands it added;
* `m` is `init` outside the bytes the program writes;
* every output element of a tile the program owns holds the full dot product of its row times its
  column, up to the `fineOne` error of `steps K` dot steps.
-/
def Q (d : Inst) (init : Mem) (i : Nat) (log : List AtomicStep) (m : Mem) : Prop :=
  ∃ L : List (AtomicStep × (Nat × Nat → Option ℚ)),
    L.map Prod.fst = log ∧
    (∀ p ∈ L, ∀ q ∈ splitOuts d, ∀ va vb : Nat → F16,
      ReadsA d init q.1 q.2 va → ReadsB d init q.1 q.2 vb →
      AccStep .f32 (cAddr d q.1 q.2) FloatV.toRat good aRho aAlpha
        Format.binary32.maxVal p.1 (p.2 q)) ∧
    (∀ x, x ∉ touchedBytes d i → m x = init x) ∧
    (∀ u ∈ tiles d, owns d i u → ∀ q ∈ tileOut d u,
      ∃ v : F32, m.read .f32 (cAddr d q.1 q.2) = some v ∧
        ∀ va vb : Nat → F16, ReadsA d init q.1 q.2 va → ReadsB d init q.1 q.2 vb →
          good v ∧ |v.toRat - dotSum va vb d.K| ≤ fineOne va vb d.K 0 (steps d.K)) ∧
    (∀ p ∈ L, ∀ q ∈ splitOuts d, ∀ w : ℚ, p.2 q = some w →
      ∃ u ∈ tiles d, splitTile d u ∧ q ∈ tileOut d u ∧ ¬ owns d i u ∧ RunVal d init i u q w) ∧
    (∀ q ∈ splitOuts d, ∀ u ∈ tiles d, q ∈ tileOut d u →
      (L.filter (fun p => (p.2 q).isSome)).length =
        if c0 d i u < c1 d i u then 1 else 0)

/-- A list whose images under `f` are a permutation of `l'` can itself be permuted so that the images
are exactly `l'`. -/
theorem exists_perm_of_map_perm {α β : Type} (f : α → β) {L : List α} {l' : List β}
    (h : (L.map f).Perm l') : ∃ L' : List α, L'.Perm L ∧ L'.map f = l' := by
  generalize hl : L.map f = l at h
  induction h generalizing L with
  | nil => exact ⟨[], by simp_all⟩
  | cons x _ ih =>
    obtain ⟨a, L₁, rfl, rfl, rfl⟩ := List.map_eq_cons_iff.mp hl
    obtain ⟨L₁', hp, hm⟩ := ih rfl
    exact ⟨a :: L₁', hp.cons a, by simp [hm]⟩
  | swap x y l =>
    obtain ⟨a, L₁, rfl, rfl, h₁⟩ := List.map_eq_cons_iff.mp hl
    obtain ⟨b, L₂, rfl, rfl, rfl⟩ := List.map_eq_cons_iff.mp h₁
    exact ⟨b :: a :: L₂, List.Perm.swap _ _ _, rfl⟩
  | trans _ _ ih₁ ih₂ =>
    obtain ⟨L₁, hp₁, hm₁⟩ := ih₁ hl
    obtain ⟨L₂, hp₂, hm₂⟩ := ih₂ hm₁
    exact ⟨L₂, hp₂.trans hp₁, hm₂⟩

/-- The invariant is invariant under a permutation of the log, so it can be handed to
`TVA.shared`. -/
theorem Q_perm {init : Mem} {i : Nat} {log log' : List AtomicStep} {m : Mem}
    (h : Q d init i log m) (hp : log.Perm log') : Q d init i log' m := by
  obtain ⟨L, hm, hacc, hframe, hval, hrun, hcnt⟩ := h
  obtain ⟨L', hL', hm'⟩ := exists_perm_of_map_perm Prod.fst (hm ▸ hp : (L.map Prod.fst).Perm log')
  exact ⟨L', hm', fun p hp' q hq va vb hva hvb => hacc p (hL'.subset hp') q hq va vb hva hvb,
    hframe, hval, fun p hp' q hq w hw => hrun p (hL'.subset hp') q hq w hw,
    fun q hq u hu hqu => by
      have := hcnt q hq u hu hqu
      have heq : (L.filter (fun p => (p.2 q).isSome)).length
          = (L'.filter (fun p => (p.2 q).isSome)).length :=
        (hL'.filter (fun p => (p.2 q).isSome)).length_eq.symm
      rw [← heq]
      exact this⟩

/-- `Q` is a `PermInv`, as `TVA.shared` demands. -/
theorem Q_permInv (init : Mem) (i : Nat) :
    PermInv (fun (_ : Unit) log m => Q d init i log m) := by
  intro a l l' m hp hQ
  exact Q_perm hQ hp

end Claims.StreamKMatmul


/-!
# The footprints of the Stream-K launch

The tiles of the launch cover the output elements, they are pairwise disjoint, and the byte sets
`Wset d p` (a program's private writes) and `Cbytes d` (the atomic cells) are what
`TVA.shared` needs: the private sets are pairwise disjoint, the cells are disjoint from every
private set and from the operands, the cells are equal-or-disjoint and present initially.
-/

namespace Claims.StreamKMatmul

open TVA

variable {d : Inst}

set_option maxHeartbeats 1000000

/-! ## Membership and the address formula -/

/-- Membership in `allOuts`. -/
theorem mem_allOuts {q : Nat × Nat} : q ∈ allOuts d ↔ q.1 < d.M ∧ q.2 < d.N := Iff.rfl

/-- Membership in `tiles`. -/
theorem mem_tiles {u : Nat} : u ∈ tiles d ↔ u < tilesM d * tilesN d := Iff.rfl

/-- The address of output element `(i, j)`, unfolded. -/
theorem cAddr_eq (i j : Nat) :
    cAddr d i j = d.c.base + ((i * d.stride_cm + j * d.stride_cn : Nat) : Int) * 4 := rfl

/-- A row of `c` fits inside `c`'s row stride. -/
theorem row_lt (hd : d.ok) {i j : Nat} (hj : j < d.N) :
    i * d.stride_cm + j * d.stride_cn < (i + 1) * d.stride_cm := by
  have hcn := hd.cn.1
  have hrows := hd.c_rows
  have hj' : j ≤ d.N - 1 := by omega
  have h1 : j * d.stride_cn ≤ (d.N - 1) * d.stride_cn := Nat.mul_le_mul_right _ hj'
  have h2 : (d.N - 1) * d.stride_cn + d.stride_cn = d.N * d.stride_cn := by
    rw [← Nat.succ_mul]
    congr 1
    omega
  have h3 : (d.N - 1) * d.stride_cn < d.stride_cm := by omega
  rw [Nat.succ_mul]
  omega

/-- Distinct in-bounds output elements have distinct element indices. -/
theorem idx_inj (hd : d.ok) {q q' : Nat × Nat} (hq : q ∈ allOuts d) (hq' : q' ∈ allOuts d)
    (h : q.1 * d.stride_cm + q.2 * d.stride_cn = q'.1 * d.stride_cm + q'.2 * d.stride_cn) :
    q = q' := by
  obtain ⟨hq1, hq2⟩ := hq
  obtain ⟨hq1', hq2'⟩ := hq'
  have hrow : q.1 * d.stride_cm + q.2 * d.stride_cn < q.1 * d.stride_cm + d.stride_cm := by
    simpa only [Nat.succ_mul] using row_lt hd hq2
  have hrow' : q'.1 * d.stride_cm + q'.2 * d.stride_cn < q'.1 * d.stride_cm + d.stride_cm := by
    simpa only [Nat.succ_mul] using row_lt hd hq2'
  have hii : q.1 = q'.1 := by
    rcases Nat.lt_trichotomy q.1 q'.1 with hlt | heq | hgt
    · exfalso
      have h1 : q.1 * d.stride_cm + d.stride_cm ≤ q'.1 * d.stride_cm := by
        rw [← Nat.succ_mul]
        exact Nat.mul_le_mul_right _ hlt
      omega
    · exact heq
    · exfalso
      have h1 : q'.1 * d.stride_cm + d.stride_cm ≤ q.1 * d.stride_cm := by
        rw [← Nat.succ_mul]
        exact Nat.mul_le_mul_right _ hgt
      omega
  have hjj : q.2 = q'.2 := by
    have h2 : q.2 * d.stride_cn = q'.2 * d.stride_cn := by
      have h' := h
      rw [hii] at h'
      omega
    exact Nat.mul_right_cancel hd.cn.1 h2
  exact Prod.ext hii hjj

/-- Distinct in-bounds output elements have distinct addresses. -/
theorem cAddr_inj (hd : d.ok) {q q' : Nat × Nat} (hq : q ∈ allOuts d) (hq' : q' ∈ allOuts d)
    (h : cAddr d q.1 q.2 = cAddr d q'.1 q'.2) : q = q' := by
  apply idx_inj hd hq hq'
  have h2 : ((q.1 * d.stride_cm + q.2 * d.stride_cn : Nat) : Int) * 4
      = ((q'.1 * d.stride_cm + q'.2 * d.stride_cn : Nat) : Int) * 4 := by
    rw [cAddr_eq, cAddr_eq] at h
    exact add_left_cancel h
  have h3 : ((q.1 * d.stride_cm + q.2 * d.stride_cn : Nat) : Int)
      = ((q'.1 * d.stride_cm + q'.2 * d.stride_cn : Nat) : Int) :=
    mul_right_cancel₀ (by norm_num : (4 : Int) ≠ 0) h2
  exact_mod_cast h3

/-- Distinct in-bounds output elements lie in disjoint blocks. -/
theorem span_disjoint_of_ne (hd : d.ok) {q q' : Nat × Nat} (hq : q ∈ allOuts d) (hq' : q' ∈ allOuts d)
    (hne : q ≠ q') :
    Disjoint (Mem.span .f32 (cAddr d q.1 q.2)) (Mem.span .f32 (cAddr d q'.1 q'.2)) := by
  have hidx : ((q.1 * d.stride_cm + q.2 * d.stride_cn : Nat) : Int)
      ≠ ((q'.1 * d.stride_cm + q'.2 * d.stride_cn : Nat) : Int) := by
    intro he
    exact hne (idx_inj hd hq hq' (by exact_mod_cast he))
  simpa only [cAddr] using Buf.span_disjoint d.c hidx

/-! ## Tiles cover the output elements -/

/-- `M ≤ 128 * tilesM`. -/
theorem M_le_tilesM : d.M ≤ tilesM d * 128 := by
  have h := Nat.div_add_mod (d.M + 127) 128
  rw [Nat.mul_comm 128 ((d.M + 127) / 128)] at h
  have h2 := Nat.mod_lt (d.M + 127) (by norm_num : 0 < 128)
  unfold tilesM
  omega

/-- `N ≤ 128 * tilesN`. -/
theorem N_le_tilesN : d.N ≤ tilesN d * 128 := by
  have h := Nat.div_add_mod (d.N + 127) 128
  rw [Nat.mul_comm 128 ((d.N + 127) / 128)] at h
  have h2 := Nat.mod_lt (d.N + 127) (by norm_num : 0 < 128)
  unfold tilesN
  omega

/-- Every in-bounds output element lies in a tile. -/
theorem mem_tiles_cover {q : Nat × Nat} (hq : q ∈ allOuts d) :
    ∃ u ∈ tiles d, q ∈ tileOut d u := by
  have hq1 : q.1 < d.M := hq.1
  have hq2 : q.2 < d.N := hq.2
  set pm := q.1 / 128 with hpm
  set pn := q.2 / 128 with hpn
  have hpm_lt : pm < tilesM d := by
    rw [Nat.div_lt_iff_lt_mul (by norm_num : 0 < 128)]
    have := M_le_tilesM (d := d)
    omega
  have hpn_lt : pn < tilesN d := by
    rw [Nat.div_lt_iff_lt_mul (by norm_num : 0 < 128)]
    have := N_le_tilesN (d := d)
    omega
  have hNpos : 0 < tilesN d := lt_of_le_of_lt (Nat.zero_le _) hpn_lt
  refine ⟨pm * tilesN d + pn, ?_, hq, ?_, ?_⟩
  · show pm * tilesN d + pn < tilesM d * tilesN d
    have h1 : (pm + 1) * tilesN d ≤ tilesM d * tilesN d := Nat.mul_le_mul_right _ hpm_lt
    rw [Nat.succ_mul] at h1
    omega
  · unfold pidMOf
    have h1 : (pm * tilesN d + pn) / tilesN d = pm := by
      rw [Nat.mul_comm pm (tilesN d), Nat.mul_add_div hNpos, Nat.div_eq_of_lt hpn_lt,
        Nat.add_zero]
    rw [h1, hpm]
  · unfold pidNOf
    have h2 : (pm * tilesN d + pn) % tilesN d = pn := by
      rw [Nat.mul_comm pm (tilesN d), Nat.mul_add_mod_self_left, Nat.mod_eq_of_lt hpn_lt]
    rw [h2, hpn]

/-- A tile's elements are elements of the matrix. -/
theorem tileOut_sub_allOuts (u : Nat) : tileOut d u ⊆ allOuts d := fun _ hq => hq.1

/-- Distinct tiles have disjoint output elements. -/
theorem tileOut_disjoint {u v : Nat} (_hu : u ∈ tiles d) (_hv : v ∈ tiles d) (hne : u ≠ v) :
    Disjoint (tileOut d u) (tileOut d v) := by
  rw [Set.disjoint_left]
  rintro q ⟨_, hmu, hnu⟩ ⟨_, hmv, hnv⟩
  apply hne
  have h1 : pidMOf d u = pidMOf d v := by rw [← hmu, ← hmv]
  have h2 : pidNOf d u = pidNOf d v := by rw [← hnu, ← hnv]
  have hu' : u = pidMOf d u * tilesN d + pidNOf d u := by
    have h := (Nat.div_add_mod u (tilesN d)).symm
    rw [Nat.mul_comm (tilesN d) (u / tilesN d)] at h
    simpa only [pidMOf, pidNOf] using h
  have hv' : v = pidMOf d v * tilesN d + pidNOf d v := by
    have h := (Nat.div_add_mod v (tilesN d)).symm
    rw [Nat.mul_comm (tilesN d) (v / tilesN d)] at h
    simpa only [pidMOf, pidNOf] using h
  calc u = pidMOf d u * tilesN d + pidNOf d u := hu'
    _ = pidMOf d v * tilesN d + pidNOf d v := by rw [h1, h2]
    _ = v := hv'.symm

/-- The output bytes are covered by the atomic cells and the private writes. -/
theorem outBytes_cover' :
    outBytes d (allOuts d) ⊆ Cbytes d ∪ {x | ∃ p, x ∈ Wset d p} := by
  rintro x ⟨q, hq, hx⟩
  obtain ⟨u, hu, hqu⟩ := mem_tiles_cover hq
  by_cases h : ∃ p, p < d.num_programs ∧ owns d p u
  · obtain ⟨p, hp, how⟩ := h
    exact Or.inr ⟨p, ⟨q, ⟨u, hu, how, hqu⟩, hx⟩⟩
  · refine Or.inl ⟨q, ⟨u, hu, ?_, hqu⟩, hx⟩
    intro p hp how
    exact h ⟨p, hp, how⟩

/-- The output elements are exactly the bytes of the output buffer. -/
theorem outElems_eq : outElems d = outBytes d (allOuts d) := by
  ext x
  simp only [outElems, Buf.elems, cIdx, outBytes, allOuts, cAddr, Set.mem_ofPred_eq]
  constructor
  · rintro ⟨e, ⟨i, j, hi, hj, rfl⟩, hx⟩
    exact ⟨(i, j), ⟨hi, hj⟩, hx⟩
  · rintro ⟨⟨i, j⟩, ⟨hi, hj⟩, hx⟩
    exact ⟨i * d.stride_cm + j * d.stride_cn, ⟨i, j, hi, hj, rfl⟩, hx⟩

/-! ## Ownership -/

/-- A program owning a tile is a program of the launch. -/
theorem owns_lt (hd : d.ok) {p u : Nat} (hu : u ∈ tiles d) (h : owns d p u) :
    p < d.num_programs := by
  obtain ⟨h0, h1⟩ := h
  have hT : 0 < steps d.K := T_pos hd
  have hX : max 0 (lo d p - u * steps d.K) = 0 := by
    unfold c0 at h0
    rcases Nat.lt_or_ge (steps d.K) (max 0 (lo d p - u * steps d.K)) with hlt | hge
    · rw [min_eq_left (le_of_lt hlt)] at h0
      omega
    · rwa [min_eq_right hge] at h0
  have hlo : lo d p ≤ u * steps d.K := by
    rw [Nat.zero_max] at hX
    omega
  have hu' : u < tilesM d * tilesN d := hu
  have htu : u * steps d.K < d.total_iters := by
    rw [total_eq hd]
    have h1 : u + 1 ≤ tilesM d * tilesN d := hu'
    have h2 : u * steps d.K < (u + 1) * steps d.K := by
      rw [Nat.succ_mul]; omega
    exact lt_of_lt_of_le h2 (Nat.mul_le_mul_right _ h1)
  have h2 : lo d p < d.num_programs * d.iters_per_program :=
    lt_of_le_of_lt hlo (lt_of_lt_of_le htu hd.cover)
  unfold lo at h2
  exact Nat.lt_of_mul_lt_mul_right h2

/-- A tile is owned by at most one program. -/
theorem owns_unique (hd : d.ok) {p q u : Nat} (hu : u ∈ tiles d) (hp : owns d p u)
    (hq : owns d q u) : p = q := by
  have hpp : p < d.num_programs := owns_lt hd hu hp
  have hqq : q < d.num_programs := owns_lt hd hu hq
  obtain ⟨hp0, hp1⟩ := hp
  obtain ⟨hq0, hq1⟩ := hq
  have hT : 0 < steps d.K := T_pos hd
  obtain ⟨r, ⟨hr, hrc0, hrc1⟩, huniq⟩ := chunk_partition hd hu (v := 0) (T_pos hd)
  have hpr : p = r := huniq p ⟨hpp, by omega, by omega⟩
  have hqr : q = r := huniq q ⟨hqq, by omega, by omega⟩
  rw [hpr, hqr]

/-! ## Private writes -/

/-- Stored elements are elements of the matrix. -/
theorem storedOuts_sub_allOuts (p : Nat) : storedOuts d p ⊆ allOuts d := by
  rintro q ⟨u, hu, how, hqu⟩
  exact hqu.1

/-- Split elements are elements of the matrix. -/
theorem splitOuts_sub_allOuts : splitOuts d ⊆ allOuts d := by
  rintro q ⟨u, hu, hsp, hqu⟩
  exact hqu.1

/-- The output bytes are the atomic cells together with the private writes. -/
theorem outBytes_cover : outBytes d (allOuts d) = Cbytes d ∪ ⋃ p, Wset d p := by
  refine Set.Subset.antisymm ?_ ?_
  · intro x hx
    have h := outBytes_cover' hx
    simpa only [Set.mem_union, Set.mem_ofPred_eq, Set.mem_iUnion] using h
  · intro x hx
    rcases hx with hx | hx
    · obtain ⟨q, ⟨u, hu, hsp, hqu⟩, hxs⟩ := hx
      exact ⟨q, hqu.1, hxs⟩
    · rw [Set.mem_iUnion] at hx
      obtain ⟨p, hx⟩ := hx
      obtain ⟨q, ⟨u, hu, how, hqu⟩, hxs⟩ := hx
      exact ⟨q, hqu.1, hxs⟩

/-- Two program's stored tiles are disjoint. -/
theorem storedOuts_disjoint (hd : d.ok) {p q : Nat} (hne : p ≠ q) :
    Disjoint (storedOuts d p) (storedOuts d q) := by
  rw [Set.disjoint_left]
  rintro x ⟨u, hu, hpu, hxu⟩ ⟨v, hv, hqv, hxv⟩
  by_cases huv : u = v
  · subst huv
    exact hne (owns_unique hd hv hpu hqv)
  · exact Set.disjoint_left.mp (tileOut_disjoint hu hv huv) hxu hxv

/-- The split tiles and a program's stored tiles are disjoint. -/
theorem splitOuts_disjoint_storedOuts (hd : d.ok) (p : Nat) :
    Disjoint (splitOuts d) (storedOuts d p) := by
  rw [Set.disjoint_left]
  rintro x ⟨u, hu, hsp, hxu⟩ ⟨v, hv, hov, hxv⟩
  by_cases huv : u = v
  · subst huv
    exact hsp p (owns_lt hd hv hov) hov
  · exact Set.disjoint_left.mp (tileOut_disjoint hu hv huv) hxu hxv

/-- `outBytes` is monotone. -/
theorem outBytes_mono {S T : Set (Nat × Nat)} (h : S ⊆ T) : outBytes d S ⊆ outBytes d T := by
  rintro x ⟨q, hq, hx⟩
  exact ⟨q, h hq, hx⟩

/-- Two sets whose elements lie in disjoint blocks have disjoint bytes. -/
theorem disjoint_outBytes {S T : Set (Nat × Nat)}
    (h : ∀ q ∈ S, ∀ q' ∈ T,
      Disjoint (Mem.span .f32 (cAddr d q.1 q.2)) (Mem.span .f32 (cAddr d q'.1 q'.2))) :
    Disjoint (outBytes d S) (outBytes d T) := by
  rw [Set.disjoint_left]
  rintro x ⟨q, hqS, hx⟩ ⟨q', hqT, hx'⟩
  exact Set.disjoint_left.mp (h q hqS q' hqT) hx hx'

/-- The private writes of distinct programs are disjoint. -/
theorem Wset_disjoint (hd : d.ok) {p q : Nat} (hne : p ≠ q) :
    Disjoint (Wset d p) (Wset d q) := by
  rw [Wset, Wset]
  apply disjoint_outBytes
  intro q₀ hq₀ q₁ hq₁
  have hne' : q₀ ≠ q₁ := by
    intro he
    subst he
    exact Set.disjoint_left.mp (storedOuts_disjoint hd hne) hq₀ hq₁
  exact span_disjoint_of_ne hd (storedOuts_sub_allOuts p hq₀) (storedOuts_sub_allOuts q hq₁) hne'

/-- The private writes and the atomic cells are disjoint. -/
theorem Cbytes_disjoint_Wset (hd : d.ok) (p : Nat) : Disjoint (Cbytes d) (Wset d p) := by
  rw [Cbytes, Wset]
  apply disjoint_outBytes
  intro q₀ hq₀ q₁ hq₁
  have hne' : q₀ ≠ q₁ := by
    intro he
    subst he
    exact Set.disjoint_left.mp (splitOuts_disjoint_storedOuts hd p) hq₀ hq₁
  exact span_disjoint_of_ne hd (splitOuts_sub_allOuts hq₀) (storedOuts_sub_allOuts p hq₁) hne'

/-! ## The operands -/

/-- A write footprint lies in the output matrix. -/
theorem Wset_sub_outElems (p : Nat) : Wset d p ⊆ outElems d := by
  rw [Wset, outElems_eq]
  exact outBytes_mono (storedOuts_sub_allOuts p)

/-- A write footprint lies in the matrix `c`. -/
theorem Wset_sub_cElems (p : Nat) : Wset d p ⊆ d.c.elems (cIdx d) := Wset_sub_outElems p

/-- The shared cells lie in the output matrix. -/
theorem Cbytes_sub_outElems : Cbytes d ⊆ outElems d := by
  rw [Cbytes, outElems_eq]
  exact outBytes_mono splitOuts_sub_allOuts

/-- The shared cells lie in the matrix `c`. -/
theorem Cbytes_sub_cElems : Cbytes d ⊆ d.c.elems (cIdx d) := Cbytes_sub_outElems

/-- The operands and the private writes are disjoint. -/
theorem inElems_disjoint_Wset (hd : d.ok) (p : Nat) : Disjoint (inElems d) (Wset d p) := by
  have hW : Wset d p ⊆ d.c.elems (cIdx d) := Wset_sub_cElems p
  have ha : Disjoint (d.a.elems (aIdx d)) (Wset d p) := hd.c_a.symm.mono_right hW
  have hb : Disjoint (d.b.elems (bIdx d)) (Wset d p) := hd.c_b.symm.mono_right hW
  show Disjoint (d.a.elems (aIdx d) ∪ d.b.elems (bIdx d)) (Wset d p)
  exact ha.union_left hb

/-- The operands and the atomic cells are disjoint. -/
theorem inElems_disjoint_Cbytes (hd : d.ok) : Disjoint (inElems d) (Cbytes d) := by
  have hC : Cbytes d ⊆ d.c.elems (cIdx d) := Cbytes_sub_cElems
  have ha : Disjoint (d.a.elems (aIdx d)) (Cbytes d) := hd.c_a.symm.mono_right hC
  have hb : Disjoint (d.b.elems (bIdx d)) (Cbytes d) := hd.c_b.symm.mono_right hC
  show Disjoint (d.a.elems (aIdx d) ∪ d.b.elems (bIdx d)) (Cbytes d)
  exact ha.union_left hb

/-! ## The atomic cells -/

/-- The bytes of the cells are the atomic bytes. -/
theorem cellBytes_Ccells : cellBytes (Ccells d) = Cbytes d := by
  ext x
  constructor
  · rintro ⟨c, hc, hx⟩
    obtain ⟨q, hq, rfl⟩ := hc
    exact ⟨q, hq, hx⟩
  · rintro ⟨q, hq, hx⟩
    exact ⟨(.f32, cAddr d q.1 q.2), ⟨q, hq, rfl⟩, hx⟩

/-- Distinct cells are disjoint or equal. -/
theorem Ccells_distinct (hd : d.ok) :
    ∀ c ∈ Ccells d, ∀ c' ∈ Ccells d,
      (Mem.span c.1 c.2 ∩ Mem.span c'.1 c'.2).Nonempty → c = c' := by
  rintro c ⟨q, hq, rfl⟩ c' ⟨q', hq', rfl⟩ ⟨x, hx1, hx2⟩
  have hqA : q ∈ allOuts d := splitOuts_sub_allOuts hq
  have hqA' : q' ∈ allOuts d := splitOuts_sub_allOuts hq'
  have hqq : q = q' := by
    by_contra hne
    exact Set.disjoint_left.mp (span_disjoint_of_ne hd hqA hqA' hne) hx1 hx2
  rw [hqq]

/-- The atomic cells lie in `c`'s storage. -/
theorem Cbytes_sub_cBytes (hd : d.ok) : Cbytes d ⊆ d.c.bytes := by
  rintro x ⟨q, hq, hx⟩
  obtain ⟨h1, h2⟩ := splitOuts_sub_allOuts hq
  have hlen : q.1 * d.stride_cm + q.2 * d.stride_cn < d.c.len := hd.c_len q.1 q.2 h1 h2
  have hsub := Buf.span_sub d.c hlen
  exact hsub hx

/-- The atomic cells are present in a valid initial memory. -/
theorem cellBytes_present (hd : d.ok) {init : Mem} (hv : d.c.Valid init) :
    ∀ x ∈ cellBytes (Ccells d), init x ≠ none := by
  intro x hx
  rw [cellBytes_Ccells] at hx
  exact hv.2.2.2 x (Cbytes_sub_cBytes hd hx)

end Claims.StreamKMatmul


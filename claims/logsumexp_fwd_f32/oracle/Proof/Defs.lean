import Proof.Final

/-!
# The grid, the footprints, and what each program leaves

Program `p` of the grid `(N, ND)` handles row `pn p = p % N` and block `pi p = p / N % ND`. It reads
the elements `n·D + j`, `j ∈ blk i`, of `x` (`Rf`), writes element `n·ND + i` of `z` (`Wf`), and
leaves there the value `Post` asks for (`Qf`, with the row's inputs `xsOf` read from the initial
memory).
-/

namespace Claims.LogsumexpFwd

open TVA

/-- `BD` is positive. -/
theorem bd_pos {d : Inst} (hd : d.ok) : 0 < d.BD := by
  obtain ⟨e, he⟩ := hd.bd_pow2
  rw [he]; exact Nat.two_pow_pos e

/-- The launch has `N · ND` programs. -/
theorem launch_size (d : Inst) : (launch d).size = d.N * ND d := by
  simp [Launch.size, launch]

/-- The row of program `p`. -/
def pn (d : Inst) (p : Nat) : Nat := p % d.N

/-- The block of program `p`. -/
def pi (d : Inst) (p : Nat) : Nat := p / d.N % ND d

/-- Program `p`'s coordinates. -/
theorem launch_pid0 (d : Inst) (p : Nat) : ((launch d).ctx p).pid 0 = pn d p := by
  simp [Launch.ctx, launch, pn]

theorem launch_pid1 (d : Inst) (p : Nat) : ((launch d).ctx p).pid 1 = pi d p := by
  simp [Launch.ctx, launch, pi]

/-- A program's row and block are in range. -/
theorem pn_lt {d : Inst} {p : Nat} (hp : p < d.N * ND d) : pn d p < d.N := by
  have hN : 0 < d.N := by
    rcases Nat.eq_zero_or_pos d.N with h | h
    · rw [h, Nat.zero_mul] at hp; omega
    · exact h
  exact Nat.mod_lt _ hN

theorem pi_lt {d : Inst} {p : Nat} (hp : p < d.N * ND d) : pi d p < ND d := by
  have hN : 0 < ND d := by
    rcases Nat.eq_zero_or_pos (ND d) with h | h
    · rw [h, Nat.mul_zero] at hp; omega
    · exact h
  exact Nat.mod_lt _ hN

/-- A program is determined by its row and block. -/
theorem pid_inj {d : Inst} {p q : Nat} (hp : p < d.N * ND d) (hq : q < d.N * ND d)
    (h1 : pn d p = pn d q) (h2 : pi d p = pi d q) : p = q := by
  have hN : 0 < d.N := by
    rcases Nat.eq_zero_or_pos d.N with h | h
    · rw [h, Nat.zero_mul] at hp; omega
    · exact h
  have hpd : p / d.N < ND d := by rw [Nat.div_lt_iff_lt_mul hN, Nat.mul_comm]; exact hp
  have hqd : q / d.N < ND d := by rw [Nat.div_lt_iff_lt_mul hN, Nat.mul_comm]; exact hq
  simp only [pi, Nat.mod_eq_of_lt hpd, Nat.mod_eq_of_lt hqd] at h2
  simp only [pn] at h1
  calc p = d.N * (p / d.N) + p % d.N := (Nat.div_add_mod _ _).symm
    _ = d.N * (q / d.N) + q % d.N := by rw [h2, h1]
    _ = q := Nat.div_add_mod _ _

/-- The program of row `n`, block `i` is `n + N·i`. -/
theorem pid_of {d : Inst} {n i : Nat} (hn : n < d.N) (hi : i < ND d) :
    n + d.N * i < d.N * ND d ∧ pn d (n + d.N * i) = n ∧ pi d (n + d.N * i) = i := by
  have hN : 0 < d.N := by omega
  refine ⟨?_, ?_, ?_⟩
  · have : d.N * i + d.N ≤ d.N * ND d := by
      rw [← Nat.mul_succ]; exact Nat.mul_le_mul_left _ hi
    omega
  · simp only [pn]; rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hn]
  · simp only [pi]
    rw [show (n + d.N * i) / d.N = i by
      rw [Nat.add_mul_div_left _ _ hN, Nat.div_eq_of_lt hn, Nat.zero_add]]
    exact Nat.mod_eq_of_lt hi

/-- The launch is valid. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  have hN : d.N < 2 ^ 31 := by
    have h := hd.grid_lt
    have hND : 0 < ND d ∨ ND d = 0 := by omega
    rcases hND with h1 | h1
    · have : d.N ≤ d.N * ND d := Nat.le_mul_of_pos_right _ h1
      omega
    · -- `ND ≥ 1` since `D ≥ 1`
      exfalso
      have hbd := bd_pos hd
      have : 1 ≤ ND d := by
        unfold ND
        rw [Nat.le_div_iff_mul_le hbd]
        have := hd.d_pos
        omega
      omega
  refine ⟨by simpa [launch] using hN, by simpa [launch] using hd.nd_lt, by simp [launch],
    by rw [launch_size]; exact hd.grid_lt, rfl, rfl, rfl, ?_⟩
  intro s hs
  simp only [launch, List.mem_singleton] at hs
  subst hs
  obtain ⟨e, he⟩ := hd.bd_pow2
  refine ⟨fun k hk => ⟨e, ?_⟩, ?_⟩
  · simp only [List.mem_singleton] at hk
    rw [hk, he]
  · have := hd.bd_le
    simp only [List.prod_cons, List.prod_nil, Nat.mul_one]
    omega

/-! ## Footprints -/

/-- The elements of `x` that row `n`, block `i` reads. -/
def xBlk (d : Inst) (n i : Nat) : Set Nat := {k | ∃ j ∈ blk d i, k = xIdx d n j}

/-- The read footprint of row `n`, block `i`. -/
def Rf (d : Inst) (n i : Nat) : Set Int := d.x.elems (xBlk d n i)

/-- The write footprint of row `n`, block `i`: its element of `z`. -/
def Wf (d : Inst) (n i : Nat) : Set Int := d.z.elems {zIdx d n i}

/-- The row's inputs as read in the initial memory (any value where nothing is read). -/
def xsOf (d : Inst) (init : Mem) (n j : Nat) : F32 :=
  (init.read .f32 (d.x.ptr (xIdx d n j)).addr).getD default

/-- What row `n`, block `i` leaves in `z`: the value `Post` asks for. -/
def Qf (d : Inst) (init : Mem) (n i : Nat) (m : Mem) : Prop :=
  ∃ c : F32, m.read .f32 (d.z.ptr (zIdx d n i)).addr = some c ∧
    (BlockOk d (xsOf d init n) i → c.isFinite = true ∧
      ∀ M : ℝ, (∀ j ∈ blk d i, |(xsOf d init n j).toReal * sig d| ≤ M) →
        |c.toReal - lseRef d (xsOf d init n) i| ≤ (zTolM : ℝ) * M + (zTol0 : ℝ))

/-- The block's columns are below `D`. -/
theorem blk_lt {d : Inst} {i j : Nat} (hj : j ∈ blk d i) : j < d.D := by
  simp only [blk, Finset.mem_Ico] at hj
  omega

/-- `x`'s elements read are among the first `N·D`. -/
theorem xIdx_lt {d : Inst} {n j : Nat} (hn : n < d.N) (hj : j < d.D) : xIdx d n j < d.N * d.D := by
  unfold xIdx
  have : (n + 1) * d.D ≤ d.N * d.D := Nat.mul_le_mul_right _ hn
  rw [Nat.succ_mul] at this
  omega

/-- `z`'s elements written are among the first `N·ND`. -/
theorem zIdx_lt {d : Inst} {n i : Nat} (hn : n < d.N) (hi : i < ND d) : zIdx d n i < d.N * ND d := by
  unfold zIdx
  have : (n + 1) * ND d ≤ d.N * ND d := Nat.mul_le_mul_right _ hn
  rw [Nat.succ_mul] at this
  omega

/-- The read elements are among `xSet`. -/
theorem xBlk_sub {d : Inst} {n i : Nat} (hn : n < d.N) : xBlk d n i ⊆ xSet d := by
  rintro k ⟨j, hj, rfl⟩
  exact xIdx_lt hn (blk_lt hj)

/-- The written element is in `zSet`. -/
theorem zBlk_sub {d : Inst} {n i : Nat} (hn : n < d.N) (hi : i < ND d) :
    ({zIdx d n i} : Set Nat) ⊆ zSet d := by
  rintro k rfl
  exact zIdx_lt hn hi

/-- `zIdx` is injective on the grid. -/
theorem zIdx_inj {d : Inst} {n i n' i' : Nat} (hi : i < ND d) (hi' : i' < ND d)
    (h : zIdx d n i = zIdx d n' i') : n = n' ∧ i = i' := by
  unfold zIdx at h
  have h1 : (n * ND d + i) / ND d = n := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hi, Nat.zero_add]
  have h2 : (n' * ND d + i') / ND d = n' := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hi', Nat.zero_add]
  have hn : n = n' := by rw [← h1, ← h2, h]
  subst hn
  exact ⟨rfl, by omega⟩

end Claims.LogsumexpFwd

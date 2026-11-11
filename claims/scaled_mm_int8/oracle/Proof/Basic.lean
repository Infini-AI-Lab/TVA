import Claim
import TVA.Proof

/-!
# `scaled_mm_int8`: the launch and the footprints

The grid of the launch, its validity, and the read and write footprints of its programs.
-/

namespace Claims.ScaledMmInt8

open TVA

/-- The block shapes of a `TritonOk` proof: every entry of `s` is a power of two. -/
def pow2s : List Nat := [1, 2, 4, 8, 16, 32, 64, 128, 256]

/-- Every entry of `pow2s` is a power of two. -/
theorem pow2_of_mem : ∀ k ∈ pow2s, ∃ e, k = 2 ^ e := by
  intro k hk
  simp only [pow2s, List.mem_cons, List.not_mem_nil, or_false] at hk
  rcases hk with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  exacts [⟨0, rfl⟩, ⟨1, rfl⟩, ⟨2, rfl⟩, ⟨3, rfl⟩, ⟨4, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨7, rfl⟩,
    ⟨8, rfl⟩]

/-- A boolean check that every shape of `L` has power-of-two entries and at most `2^20` elements gives `TritonOk` for each. -/
theorem tOk_all {L : List Shape}
    (h : L.all (fun s => s.all (fun k => decide (k ∈ pow2s)) && decide (s.prod ≤ 2 ^ 20)) = true) :
    ∀ s ∈ L, s.TritonOk := by
  intro s hs
  have h1 := (List.all_eq_true.mp h) s hs
  rw [Bool.and_eq_true] at h1
  obtain ⟨h1l, h1r⟩ := h1
  refine ⟨fun k hk => pow2_of_mem k ?_, ?_⟩
  · have := (List.all_eq_true.mp h1l) k hk
    simpa using this
  · simpa using h1r

/-- The number of column tiles, `cdiv(N, BLOCK_SIZE_N)`. -/
abbrev gridN (d : Inst) : Nat := (d.N + d.BN - 1) / d.BN

/-- The number of programs, `cdiv(M, BLOCK_SIZE_M) · cdiv(N, BLOCK_SIZE_N)`. -/
abbrev grid (d : Inst) : Nat := (d.M + d.BM - 1) / d.BM * gridN d

/-- Program `p`'s row tile. -/
abbrev pm (d : Inst) (p : Nat) : Nat := p / gridN d

/-- Program `p`'s column tile. -/
abbrev pn (d : Inst) (p : Nat) : Nat := p % gridN d

/-- The tile of every covered instance is one of the heuristic's four, or the default. -/
theorem tile_cases {d : Inst} (hd : d.ok) :
    (d.BM, d.BN, d.BK) = (64, 64, 256) ∨ (d.BM, d.BN, d.BK) = (64, 128, 256) ∨
      (d.BM, d.BN, d.BK) = (64, 128, 128) ∨ (d.BM, d.BN, d.BK) = (128, 128, 128) ∨
      (d.BM, d.BN, d.BK) = (32, 32, 32) := by
  rcases hd.tile_ok with h | h
  · have h1 : d.BM = (heurTile d.M d.N).1 := congrArg Prod.fst h
    have h2 : d.BN = (heurTile d.M d.N).2.1 :=
      congrArg (fun p : Nat × Nat × Nat => p.2.1) h
    have h3 : d.BK = (heurTile d.M d.N).2.2 :=
      congrArg (fun p : Nat × Nat × Nat => p.2.2) h
    clear h
    rw [h1, h2, h3]
    unfold heurTile
    split_ifs <;> simp
  · exact Or.inr (Or.inr (Or.inr (Or.inr h)))

/-- The tile has at least one column. -/
theorem BN_pos {d : Inst} (hd : d.ok) : 0 < d.BN := by
  rcases tile_cases hd with h | h | h | h | h <;>
    (first
      | (have h2 : d.BN = 64 := congrArg (fun p : Nat × Nat × Nat => p.2.1) h; omega)
      | (have h2 : d.BN = 128 := congrArg (fun p : Nat × Nat × Nat => p.2.1) h; omega)
      | (have h2 : d.BN = 32 := congrArg (fun p : Nat × Nat × Nat => p.2.1) h; omega))

/-- The tile has at least one row. -/
theorem BM_pos {d : Inst} (hd : d.ok) : 0 < d.BM := by
  rcases tile_cases hd with h | h | h | h | h <;>
    (first
      | (have h1 : d.BM = 64 := congrArg Prod.fst h; omega)
      | (have h1 : d.BM = 128 := congrArg Prod.fst h; omega)
      | (have h1 : d.BM = 32 := congrArg Prod.fst h; omega))

/-- The reduction tile is not empty. -/
theorem BK_pos {d : Inst} (hd : d.ok) : 0 < d.BK := by
  rcases tile_cases hd with h | h | h | h | h <;>
    (first
      | (have h3 : d.BK = 256 := congrArg (fun p : Nat × Nat × Nat => p.2.2) h; omega)
      | (have h3 : d.BK = 128 := congrArg (fun p : Nat × Nat × Nat => p.2.2) h; omega)
      | (have h3 : d.BK = 32 := congrArg (fun p : Nat × Nat × Nat => p.2.2) h; omega))

/-- There is at least one column tile. -/
theorem gridN_pos {d : Inst} (hd : d.ok) : 0 < gridN d := by
  have hB := BN_pos hd
  have hN := hd.N_pos
  show 0 < (d.N + d.BN - 1) / d.BN
  exact Nat.div_pos (by omega) hB

/-- The launch has `grid d` programs. -/
theorem launch_size (d : Inst) : (launch d).size = grid d := Launch.size_grid1 _ _ _

/-- Program `i` of the launch has `pid 0 = i`. -/
theorem launch_pid (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = i := by
  have hi : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  simp [Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi]

/-- The launch does not set `debug`. -/
theorem launch_debug (d : Inst) : (launch d).debug = false := rfl

/-- The launch is valid: fewer than `2^31` programs and Triton-acceptable block shapes (every tile of the four heuristic sizes, with either scale layout). -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 (by simpa [launch, grid, Launch.grid1] using hd.grid_lt) ?_
  intro s hs
  rcases tile_cases hd with h | h | h | h | h
  all_goals
    simp only [Prod.mk.injEq] at h
    obtain ⟨h1, h2, h3⟩ := h
    simp only [h1, h2, h3] at hs ⊢
    cases hsa : d.saVec <;> cases hsb : d.sbVec <;>
      simp only [hsa, hsb, BSA, BSB, ite_true, h1, h2] at hs <;>
      exact tOk_all (by decide) s hs

/-- The reduction tile is at most `256`. -/
theorem BK_le {d : Inst} (hd : d.ok) : d.BK ≤ 256 := by
  rcases tile_cases hd with h | h | h | h | h <;>
    (first
      | (have h3 : d.BK = 256 := congrArg (fun p : Nat × Nat × Nat => p.2.2) h; omega)
      | (have h3 : d.BK = 128 := congrArg (fun p : Nat × Nat × Nat => p.2.2) h; omega)
      | (have h3 : d.BK = 32 := congrArg (fun p : Nat × Nat × Nat => p.2.2) h; omega))

/-- `N + (BLOCK_SIZE_N - 1)` in `int32` does not wrap. -/
theorem toInt_Nadd {d : Inst} (hd : d.ok) :
    (IntV.ofNat d.N + (IntV.ofNat (d.BN - 1) : I32)).toInt = ((d.N + (d.BN - 1) : Nat) : Int) := by
  exact IntV.I32.add_nat (x := d.N) (y := d.BN - 1)
    (IntV.I32.toInt_ofNat (by have := hd.N_le; have := BN_pos hd; omega))
    (IntV.I32.toInt_ofNat (by have := hd.N_le; have := BN_pos hd; omega))
    (by have := hd.N_le; have := BN_pos hd; omega)

/-- `BLOCK_SIZE_N` as an `int32` constant has its value. -/
theorem toInt_BN {d : Inst} (hd : d.ok) : (IntV.ofNat d.BN : I32).toInt = (d.BN : Int) :=
  IntV.I32.toInt_ofNat (by have := hd.N_le; have := hd.N_pos; omega)

/-- `K + (BLOCK_SIZE_K - 1)` in `int32` does not wrap. -/
theorem toInt_Kadd {d : Inst} (hd : d.ok) :
    (IntV.ofNat d.K + (IntV.ofNat (d.BK - 1) : I32)).toInt = ((d.K + (d.BK - 1) : Nat) : Int) := by
  exact IntV.I32.add_nat (x := d.K) (y := d.BK - 1)
    (IntV.I32.toInt_ofNat (by have := hd.K_lt; omega))
    (IntV.I32.toInt_ofNat (by have := hd.K_lt; have := BK_le hd; omega))
    (by have := hd.K_lt; have := BK_le hd; omega)

/-- `BLOCK_SIZE_K` as an `int32` constant has its value. -/
theorem toInt_BK {d : Inst} (hd : d.ok) : (IntV.ofNat d.BK : I32).toInt = (d.BK : Int) :=
  IntV.I32.toInt_ofNat (by have := hd.K_lt; have := BK_le hd; omega)

/-- `tl.cdiv(N, BLOCK_SIZE_N)` is positive. -/
theorem divN_pos {d : Inst} (hd : d.ok) :
    0 < ((d.N + (d.BN - 1) : Nat) : Int) / (d.BN : Int) := by
  rw [← Int.natCast_ediv]
  exact_mod_cast Nat.div_pos (by have := hd.N_pos; omega) (BN_pos hd)

/-- `tl.cdiv(K, BLOCK_SIZE_K)` is positive. -/
theorem divK_pos {d : Inst} (hd : d.ok) :
    0 < ((d.K + (d.BK - 1) : Nat) : Int) / (d.BK : Int) := by
  rw [← Int.natCast_ediv]
  exact_mod_cast Nat.div_pos (by have := hd.K_pos; omega) (BK_pos hd)

/-- `tl.cdiv(K, BLOCK_SIZE_K) ≤ K`. -/
theorem divK_le {d : Inst} (hd : d.ok) :
    ((d.K + (d.BK - 1) : Nat) : Int) / (d.BK : Int) ≤ (d.K : Int) := by
  rw [← Int.natCast_ediv]
  have hle : d.K + (d.BK - 1) ≤ d.BK * d.K := by
    have hbk1 : 1 ≤ d.BK := BK_pos hd
    have hcast : ((d.K + (d.BK - 1) : Nat) : ℤ) ≤ ((d.BK * d.K : Nat) : ℤ) := by
      have hs : ((d.BK - 1 : Nat) : ℤ) = (d.BK : ℤ) - 1 := by
        rw [Nat.cast_sub hbk1]; norm_num
      have hK1 : 1 ≤ d.K := hd.K_pos
      have h1 : (0:ℤ) ≤ ((d.BK - 1 : Nat) : ℤ) := by exact_mod_cast Nat.zero_le _
      have h2 : (0:ℤ) ≤ ((d.K - 1 : Nat) : ℤ) := by exact_mod_cast Nat.zero_le _
      have hs2 : ((d.K - 1 : Nat) : ℤ) = (d.K : ℤ) - 1 := by rw [Nat.cast_sub hK1]; norm_num
      push_cast
      rw [hs]
      nlinarith [mul_nonneg h1 h2, hs, hs2]
    exact_mod_cast hcast
  have h3 : (d.K + (d.BK - 1)) / d.BK ≤ d.K := Nat.div_le_of_le_mul hle
  exact_mod_cast h3

/-- Since `BLOCK_SIZE_K` divides `K`, `cdiv(K, BLOCK_SIZE_K) = K / BLOCK_SIZE_K`. -/
theorem divK_eq {d : Inst} (hd : d.ok) : (d.K + (d.BK - 1)) / d.BK = d.K / d.BK := by
  have hB := BK_pos hd
  obtain ⟨c, hc⟩ : d.BK ∣ d.K := Nat.dvd_of_mod_eq_zero hd.K_div
  rw [hc, Nat.mul_add_div hB, Nat.div_eq_of_lt (by omega), Nat.add_zero]
  exact (Nat.mul_div_right c hB).symm


/-- The row index of a launched program is below the number of row tiles. -/
theorem pm_lt_gridM {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) :
    pm d p < (d.M + d.BM - 1) / d.BM := by
  simp only [pm, grid] at hp ⊢
  rwa [Nat.div_lt_iff_lt_mul (gridN_pos hd)]

/-- In a launched program the row tile starts below `2^31`. -/
theorem pmBM_lt31 {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) :
    pm d p * d.BM < 2 ^ 31 := by
  have h1 := pm_lt_gridM hd hp
  have h2 : ((d.M + d.BM - 1) / d.BM) * d.BM ≤ d.M + d.BM - 1 := Nat.div_mul_le_self _ _
  have h3 : pm d p * d.BM < ((d.M + d.BM - 1) / d.BM) * d.BM :=
    Nat.mul_lt_mul_of_pos_right h1 (BM_pos hd)
  have h4 : d.M + d.BM - 1 < 2 ^ 31 := by have := hd.M_le; omega
  omega

/-- The column tile starts below `2^31` for every program index. -/
theorem pnBN_lt31 {d : Inst} (hd : d.ok) (p : Nat) : pn d p * d.BN < 2 ^ 31 := by
  have h1 : pn d p < gridN d := by
    simp only [pn]; exact Nat.mod_lt _ (gridN_pos hd)
  have h2 : gridN d * d.BN ≤ d.N + d.BN - 1 := by
    simp only [gridN]; exact Nat.div_mul_le_self _ _
  have h3 : pn d p * d.BN < gridN d * d.BN := Nat.mul_lt_mul_of_pos_right h1 (BN_pos hd)
  have h4 : d.N + d.BN - 1 < 2 ^ 31 := by have := hd.N_le; omega
  omega

/-- The tile has at most `256` rows. -/
theorem BM_le256 {d : Inst} (hd : d.ok) : d.BM ≤ 256 := by
  rcases tile_cases hd with h | h | h | h | h <;>
    (simp only [Prod.mk.injEq] at h; obtain ⟨h1, -, -⟩ := h; omega)

/-- The tile has at most `256` columns. -/
theorem BN_le256 {d : Inst} (hd : d.ok) : d.BN ≤ 256 := by
  rcases tile_cases hd with h | h | h | h | h <;>
    (simp only [Prod.mk.injEq] at h; obtain ⟨-, h1, -⟩ := h; omega)

/-- `s·a + t·b` of naturals below `2^63` fits `int64`. -/
theorem sfits_sum_mul {s a t b : Nat} (h : s * a + t * b < 2 ^ 63) :
    IntV.SFits 64 ((s : ℤ) * (a : ℤ) + (t : ℤ) * (b : ℤ)) := by
  have e1 : ((s : ℤ) * (a : ℤ)) = ((s * a : Nat) : ℤ) := by push_cast; ring
  have e2 : ((t : ℤ) * (b : ℤ)) = ((t * b : Nat) : ℤ) := by push_cast; ring
  rw [e1, e2]
  simp only [IntV.SFits]
  push_cast
  constructor <;> omega

/-- `s·a` of naturals below `2^63` fits `int64`. -/
theorem sfits_mul {s a : Nat} (h : s * a < 2 ^ 63) :
    IntV.SFits 64 ((s : ℤ) * (a : ℤ)) := by
  have e1 : ((s : ℤ) * (a : ℤ)) = ((s * a : Nat) : ℤ) := by push_cast; ring
  rw [e1]
  simp only [IntV.SFits]
  push_cast
  constructor <;> omega

/-- `a + b` of naturals below `2^63` fits `int64`. -/
theorem sfits_add_nat {a b : Nat} (h : a + b < 2 ^ 63) : IntV.SFits 64 ((a : ℤ) + (b : ℤ)) := by
  simp only [IntV.SFits]
  push_cast
  constructor <;> omega

end Claims.ScaledMmInt8

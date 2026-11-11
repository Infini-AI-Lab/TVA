import TVA.Proof
import Claim
import Proof.Error

/-!
# The launch and its tiles

The launch and its tiles: the grouped program-id arithmetic (`pidM`, `pidN`), the launch facts, the unfolded kernel body, and the footprints `R`, `W` and the postcondition `Q`.
-/


/-!
# The grouped tile map `pid ↦ (pid_m, pid_n)`

Program `p` of the grid is the tile `(pidM d p, pidN d p)`. The map is inside the tile grid and
one-to-one there.
-/

namespace Claims.Matmul

open TVA

/-- `cdiv(M, 128)`, the number of row tiles. -/
def npm (d : Inst) : Nat := (d.M + 127) / 128

/-- `cdiv(N, 256)`, the number of column tiles. -/
def npn (d : Inst) : Nat := (d.N + 255) / 256

/-- The number of programs, `cdiv(M, 128) · cdiv(N, 256)`. -/
def grid (d : Inst) : Nat := npm d * npn d

/-- The group of program `p`, `p // (GROUP_SIZE_M · num_pid_n)`. -/
def grp (d : Inst) (p : Nat) : Nat := p / (8 * npn d)

/-- `first_pid_m` of program `p`. -/
def fpm (d : Inst) (p : Nat) : Nat := grp d p * 8

/-- `group_size_m` of program `p`. -/
def gsize (d : Inst) (p : Nat) : Nat := min (npm d - fpm d p) 8

/-- The row tile of program `p`. -/
def pidM (d : Inst) (p : Nat) : Nat := fpm d p + (p % (8 * npn d)) % gsize d p

/-- The column tile of program `p`. -/
def pidN (d : Inst) (p : Nat) : Nat := (p % (8 * npn d)) / gsize d p

/-- `fpm` is the group number times `8`. -/
theorem pid_fpm_eq (d : Inst) (p : Nat) : fpm d p = 8 * grp d p := by
  simp only [fpm]
  exact Nat.mul_comm _ _

/-- A positive `M` gives a positive number of row tiles. -/
theorem npm_pos {d : Inst} (h : 0 < d.M) : 0 < npm d := by
  simp only [npm]
  exact Nat.div_pos (by omega) (by norm_num)

/-- A positive `N` gives a positive number of column tiles. -/
theorem npn_pos {d : Inst} (h : 0 < d.N) : 0 < npn d := by
  simp only [npn]
  exact Nat.div_pos (by omega) (by norm_num)

/-- A group has at most `8` row tiles. -/
theorem gsize_le (d : Inst) (p : Nat) : gsize d p ≤ 8 := by
  simp only [gsize]
  exact Nat.min_le_right _ _

/-- On a program of the grid the column-tile count is positive. -/
theorem pid_npn_pos_of_grid {d : Inst} {p : Nat} (hp : p < grid d) : 0 < npn d := by
  rcases Nat.eq_zero_or_pos (npn d) with h | h
  · exfalso
    rw [grid, h, Nat.mul_zero] at hp
    omega
  · exact h

/-- On a program of the grid the row-tile count is positive. -/
theorem pid_npm_pos_of_grid {d : Inst} {p : Nat} (hp : p < grid d) : 0 < npm d := by
  rcases Nat.eq_zero_or_pos (npm d) with h | h
  · exfalso
    rw [grid, h, Nat.zero_mul] at hp
    omega
  · exact h

/-- The first row tile of `p`'s group is below the number of row tiles. -/
theorem pid_fpm_lt_npm {d : Inst} {p : Nat} (hp : p < grid d) : fpm d p < npm d := by
  have hn : 0 < npn d := pid_npn_pos_of_grid hp
  have hG : 0 < 8 * npn d := Nat.mul_pos (by norm_num) hn
  have hdiv : 8 * npn d * (p / (8 * npn d)) + p % (8 * npn d) = p :=
    Nat.div_add_mod p (8 * npn d)
  have hle : 8 * npn d * (p / (8 * npn d)) ≤ p := by omega
  have hkey : 8 * (p / (8 * npn d)) * npn d ≤ p := by
    calc 8 * (p / (8 * npn d)) * npn d = 8 * npn d * (p / (8 * npn d)) := by ring
      _ ≤ p := hle
  have hfpm : fpm d p = 8 * (p / (8 * npn d)) := by
    simp only [fpm, grp]
    exact Nat.mul_comm _ _
  by_contra hcon
  have hcon' : npm d ≤ fpm d p := not_lt.mp hcon
  rw [hfpm] at hcon'
  have h2 : npm d * npn d ≤ 8 * (p / (8 * npn d)) * npn d := Nat.mul_le_mul_right _ hcon'
  rw [grid] at hp
  omega

/-- A group of a program of the grid is nonempty. -/
theorem gsize_pos {d : Inst} {p : Nat} (hp : p < grid d) : 0 < gsize d p := by
  have hlt : fpm d p < npm d := pid_fpm_lt_npm hp
  have hsub : 0 < npm d - fpm d p := Nat.sub_pos_of_lt hlt
  simp only [gsize]
  exact Nat.lt_min.mpr ⟨hsub, by norm_num⟩

/-- The group's rows, `first_pid_m + group_size_m`, fit in the row tiles. -/
theorem pid_fpm_add_gsize_le {d : Inst} {p : Nat} (hp : p < grid d) :
    fpm d p + gsize d p ≤ npm d := by
  have hf : fpm d p ≤ npm d := le_of_lt (pid_fpm_lt_npm hp)
  have hgs : gsize d p ≤ npm d - fpm d p := by
    simp only [gsize]
    exact Nat.min_le_left _ _
  have h := Nat.add_sub_of_le hf
  omega

/-- The row tile stays below `first_pid_m + 8`. -/
theorem pid_pidM_lt_fpm_add_8 {d : Inst} {p : Nat} (hp : p < grid d) :
    pidM d p < fpm d p + 8 := by
  have hgs : 0 < gsize d p := gsize_pos hp
  have h8 : gsize d p ≤ 8 := gsize_le d p
  have hmod : p % (8 * npn d) % gsize d p < gsize d p := Nat.mod_lt _ hgs
  simp only [pidM]
  omega

/-- The offset inside the group is below `group_size_m · num_pid_n`. -/
theorem pid_mod_lt_gsize_mul {d : Inst} {p : Nat} (hp : p < grid d) :
    p % (8 * npn d) < gsize d p * npn d := by
  have hn : 0 < npn d := pid_npn_pos_of_grid hp
  have hG : 0 < 8 * npn d := Nat.mul_pos (by norm_num) hn
  have hdiv : 8 * npn d * (p / (8 * npn d)) + p % (8 * npn d) = p :=
    Nat.div_add_mod p (8 * npn d)
  have hmodlt : p % (8 * npn d) < 8 * npn d := Nat.mod_lt p hG
  rcases Nat.lt_or_ge (npm d - fpm d p) 8 with hpart | hfull
  · have hgs : gsize d p = npm d - fpm d p := by
      simp only [gsize]
      exact Nat.min_eq_left (le_of_lt hpart)
    have hf : fpm d p ≤ npm d := le_of_lt (pid_fpm_lt_npm hp)
    have hsum : fpm d p + gsize d p = npm d := by
      rw [hgs]
      exact Nat.add_sub_of_le hf
    have hp' : p < npm d * npn d := by rw [← grid]; exact hp
    have hexp : npm d * npn d = 8 * npn d * (p / (8 * npn d)) + gsize d p * npn d := by
      rw [← hsum, Nat.add_mul]
      simp only [fpm, grp]
      ring
    omega
  · have hgs : gsize d p = 8 := by
      simp only [gsize]
      exact Nat.min_eq_right hfull
    rw [hgs]
    exact hmodlt

/-- The row tile of a program of the grid is inside the grid. -/
theorem pidM_lt {d : Inst} {p : Nat} (hp : p < grid d) : pidM d p < npm d := by
  have hgs : 0 < gsize d p := gsize_pos hp
  have hmod : p % (8 * npn d) % gsize d p < gsize d p := Nat.mod_lt _ hgs
  have hle : fpm d p + gsize d p ≤ npm d := pid_fpm_add_gsize_le hp
  simp only [pidM]
  omega

/-- The column tile of a program of the grid is inside the grid. -/
theorem pidN_lt {d : Inst} {p : Nat} (hp : p < grid d) : pidN d p < npn d := by
  have h := pid_mod_lt_gsize_mul hp
  simp only [pidN]
  exact Nat.div_lt_of_lt_mul h

/-- The first row tiles of two groups differ by at least `8` when they differ. -/
theorem pid_fpm_lt_add_8 {d : Inst} {a b : Nat} (h : fpm d a < fpm d b) :
    fpm d a + 8 ≤ fpm d b := by
  have h1 : 8 * grp d a < 8 * grp d b := by
    simpa only [pid_fpm_eq] using h
  have hg : grp d a < grp d b := Nat.lt_of_mul_lt_mul_left h1
  have h2 : 8 * (grp d a + 1) ≤ 8 * grp d b :=
    Nat.mul_le_mul_left 8 (Nat.succ_le_of_lt hg)
  rw [Nat.mul_add, Nat.mul_one] at h2
  rw [pid_fpm_eq, pid_fpm_eq]
  exact h2

/-- **The tile map is one-to-one on the grid.** -/
theorem pid_inj {d : Inst} {p q : Nat} (hp : p < grid d) (hq : q < grid d)
    (hm : pidM d p = pidM d q) (hn : pidN d p = pidN d q) : p = q := by
  have hpM : pidM d p < fpm d p + 8 := pid_pidM_lt_fpm_add_8 hp
  have hqM : pidM d q < fpm d q + 8 := pid_pidM_lt_fpm_add_8 hq
  have hpMge : fpm d p ≤ pidM d p := by simp only [pidM]; omega
  have hqMge : fpm d q ≤ pidM d q := by simp only [pidM]; omega
  have hfp : fpm d p = fpm d q := by
    rcases Nat.lt_trichotomy (fpm d p) (fpm d q) with hlt | heq | hgt
    · have h1 := pid_fpm_lt_add_8 hlt
      omega
    · exact heq
    · have h1 := pid_fpm_lt_add_8 hgt
      omega
  have hgp : grp d p = grp d q := by
    have h1 : 8 * grp d p = 8 * grp d q := by
      rw [← pid_fpm_eq, ← pid_fpm_eq]
      exact hfp
    exact Nat.mul_left_cancel (by norm_num) h1
  have hgs_eq : gsize d p = gsize d q := by
    simp only [gsize]
    rw [hfp]
  have hmod_eq : p % (8 * npn d) % gsize d p = q % (8 * npn d) % gsize d q := by
    simp only [pidM] at hm
    omega
  have hdiv_eq : p % (8 * npn d) / gsize d p = q % (8 * npn d) / gsize d q := by
    simp only [pidN] at hn
    exact hn
  have hgpq : p / (8 * npn d) = q / (8 * npn d) := by
    have h := hgp
    simp only [grp] at h
    exact h
  have hr : p % (8 * npn d) = q % (8 * npn d) := by
    have h1 : p % (8 * npn d) =
        gsize d p * (p % (8 * npn d) / gsize d p) + p % (8 * npn d) % gsize d p :=
      (Nat.div_add_mod _ _).symm
    have h2 : q % (8 * npn d) =
        gsize d q * (q % (8 * npn d) / gsize d q) + q % (8 * npn d) % gsize d q :=
      (Nat.div_add_mod _ _).symm
    rw [h1, h2, hdiv_eq, hmod_eq, hgs_eq]
  have hp_eq : p = 8 * npn d * (p / (8 * npn d)) + p % (8 * npn d) :=
    (Nat.div_add_mod p (8 * npn d)).symm
  have hq_eq : q = 8 * npn d * (q / (8 * npn d)) + q % (8 * npn d) :=
    (Nat.div_add_mod q (8 * npn d)).symm
  rw [hp_eq, hq_eq, hr, hgpq]

end Claims.Matmul

/- ===== Basic ===== -/

/-!
# The launch and the model's body

The body of the model on an instance, spelled out at the instance's parameters (the kernel's
parameters are all fixed by `model`), the launch's size, program number, validity, and the
`debug` option.
-/

namespace Claims.Matmul

open TVA

/-- The block shapes the launch lists. -/
def blocks : List Shape :=
  [[128], [256], [64], [128, 1], [1, 64], [128, 64], [64, 1], [1, 256], [64, 256], [128, 256]]

/-- The block extents `1, 64, 128, 256` are powers of two. -/
theorem pow2_of_small {k : Nat} (h : k = 1 ∨ k = 64 ∨ k = 128 ∨ k = 256) : ∃ e, k = 2 ^ e := by
  rcases h with rfl | rfl | rfl | rfl
  · exact ⟨0, by norm_num⟩
  · exact ⟨6, by norm_num⟩
  · exact ⟨7, by norm_num⟩
  · exact ⟨8, by norm_num⟩

/-- Every extent of the launch's block shapes is `1`, `64`, `128` or `256`. -/
theorem flatten_dims : ∀ k ∈ blocks.flatten, k = 1 ∨ k = 64 ∨ k = 128 ∨ k = 256 := by
  decide

/-- Each block shape has at most `2^20` elements. -/
theorem blocks_prod : ∀ s ∈ blocks, s.prod ≤ 2 ^ 20 := by
  decide

/-- Each block shape of the launch is accepted by Triton. -/
theorem blocks_ok (s : Shape) (hs : s ∈ blocks) : s.TritonOk := by
  refine ⟨fun k hk => pow2_of_small (flatten_dims k ?_), blocks_prod s hs⟩
  exact List.mem_flatten.mpr ⟨s, hs, hk⟩

/-- The launch has `grid d` programs. -/
theorem launch_size (d : Inst) : (launch d).size = grid d := by
  have h : (launch d).size = (d.M + 128 - 1) / 128 * ((d.N + 256 - 1) / 256) :=
    Launch.size_grid1 _ _ _
  rw [h]
  rfl

/-- Program `i` has `program_id(0) = i`. -/
theorem launch_pid (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = i := by
  have hi : (i : Nat) < (d.M + 128 - 1) / 128 * ((d.N + 256 - 1) / 256) :=
    lt_of_lt_of_eq i.isLt (launch_size d)
  show (i : Nat) % ((d.M + 128 - 1) / 128 * ((d.N + 256 - 1) / 256)) = (i : Nat)
  exact Nat.mod_eq_of_lt hi

/-- The launch is valid on a covered instance. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 ?_ ?_
  · exact hd.grid_lt
  · exact blocks_ok

/-- The launch has `debug = false`. -/
theorem launch_debug (d : Inst) : (launch d).debug = false := rfl

/-- The model's body on instance `d`, in the instance's parameters. -/
theorem body_eq (d : Inst) (c : Ctx) :
    (model d).body c =
      (do
        let pid := tl.program_id c 0
        let num_pid_m := (← tl.idiv (Tensor.scalar (IntV.ofNat d.M + (IntV.ofNat (128 - 1) : I32)))
          (Tensor.scalar (IntV.ofNat 128 : I32))).item
        let num_pid_n := (← tl.idiv (Tensor.scalar (IntV.ofNat d.N + (IntV.ofNat (256 - 1) : I32)))
          (Tensor.scalar (IntV.ofNat 256 : I32))).item
        let num_pid_in_group := (IntV.ofNat 8 : I32) * num_pid_n
        let group_id := (← tl.idiv (Tensor.scalar pid) (Tensor.scalar num_pid_in_group)).item
        let first_pid_m := group_id * (IntV.ofNat 8 : I32)
        let group_size_m := (tl.iminimum (Tensor.scalar (num_pid_m - first_pid_m))
          (Tensor.scalar (IntV.ofNat 8 : I32))).item
        let pid_m := first_pid_m + (← tl.irem (Tensor.scalar (← tl.irem (Tensor.scalar pid)
          (Tensor.scalar num_pid_in_group)).item) (Tensor.scalar group_size_m)).item
        let pid_n := (← tl.idiv (Tensor.scalar (← tl.irem (Tensor.scalar pid)
          (Tensor.scalar num_pid_in_group)).item) (Tensor.scalar group_size_m)).item
        tl.assume (IntV.le (IntV.ofInt 0 : I32) pid_m)
        tl.assume (IntV.le (IntV.ofInt 0 : I32) pid_n)
        tl.assume (IntV.lt (IntV.ofInt 0 : I32) (IntV.ofNat d.stride_am : I32))
        tl.assume (IntV.lt (IntV.ofInt 0 : I32) (IntV.ofNat d.stride_ak : I32))
        tl.assume (IntV.lt (IntV.ofInt 0 : I32) (IntV.ofNat d.stride_bn : I32))
        tl.assume (IntV.lt (IntV.ofInt 0 : I32) (IntV.ofNat d.stride_bk : I32))
        tl.assume (IntV.lt (IntV.ofInt 0 : I32) (IntV.ofNat d.stride_cm : I32))
        tl.assume (IntV.lt (IntV.ofInt 0 : I32) (IntV.ofNat d.stride_cn : I32))
        let offs_am : Tensor [128] I32 ←
          tl.irem (pid_m * (IntV.ofNat 128 : I32) + tl.arange 0 128) (bc (IntV.ofNat d.M))
        let offs_bn : Tensor [256] I32 ←
          tl.irem (pid_n * (IntV.ofNat 256 : I32) + tl.arange 0 256) (bc (IntV.ofNat d.N))
        let offs_k : Tensor [64] I32 := tl.arange 0 64
        let a_ptrs := d.a.ptr 0 +
          (offs_am.col 64 * (IntV.ofNat d.stride_am : I32) + offs_k.row 128 * (IntV.ofNat d.stride_ak : I32))
        let b_ptrs := d.b.ptr 0 +
          (offs_k.col 256 * (IntV.ofNat d.stride_bk : I32) + offs_bn.row 64 * (IntV.ofNat d.stride_bn : I32))
        let accumulator := tl.zeros [128, 256] .f32
        let (_a_ptrs, _b_ptrs, accumulator) ← tl.forRange (IntV.ofInt 0 : I32)
            (← tl.idiv (Tensor.scalar (IntV.ofNat d.K + (IntV.ofNat (64 - 1) : I32)))
              (Tensor.scalar (IntV.ofNat 64 : I32))).item
            (IntV.ofInt 1 : I32) (a_ptrs, b_ptrs, accumulator)
            fun k (a_ptrs, b_ptrs, accumulator) => do
          let a ← tl.load a_ptrs
            (some ((offs_k.row 128).ltS (IntV.ofNat d.K - k * (IntV.ofNat 64 : I32))))
            (some (← tl.convert .f32 .f16 (tl.full [128, 64] (FloatV.litF32 (FloatV.ofPat 0)))))
          let b ← tl.load b_ptrs
            (some ((offs_k.col 256).ltS (IntV.ofNat d.K - k * (IntV.ofNat 64 : I32))))
            (some (← tl.convert .f32 .f16 (tl.full [64, 256] (FloatV.litF32 (FloatV.ofPat 0)))))
          let accumulator ← tl.dot
            { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }
            a b accumulator
          let a_ptrs := a_ptrs + (bc ((IntV.ofNat 64 : I32) * (IntV.ofNat d.stride_ak : I32)) :
            Tensor [128, 64] I32)
          let b_ptrs := b_ptrs + (bc ((IntV.ofNat 64 : I32) * (IntV.ofNat d.stride_bk : I32)) :
            Tensor [64, 256] I32)
          return (a_ptrs, b_ptrs, accumulator)
        let c : Tensor [128, 256] F16 ← tl.cvt accumulator
        let offs_cm : Tensor [128] I32 := pid_m * (IntV.ofNat 128 : I32) + tl.arange 0 128
        let offs_cn : Tensor [256] I32 := pid_n * (IntV.ofNat 256 : I32) + tl.arange 0 256
        let c_ptrs : Tensor [128, 256] (Ptr .f16) := d.c.ptr 0 +
          (IntV.ofNat d.stride_cm : I32) * offs_cm.col 256 + (IntV.ofNat d.stride_cn : I32) * offs_cn.row 128
        let c_mask := ((offs_cm.col 256).ltS (IntV.ofNat d.M)).and ((offs_cn.row 128).ltS (IntV.ofNat d.N))
        tl.store c_ptrs c (some c_mask)) := rfl

end Claims.Matmul

/- ===== Spec ===== -/

/-!
# Footprints and the per-program postcondition

The elements of `a` and `b` a program reads, the elements of `c` it writes, and what it leaves in
its tile of `c`.
-/

namespace Claims.Matmul

open TVA

/-- The elements of `a`'s storage that hold the matrix `a`. -/
def aSet (d : Inst) : Set Nat :=
  {e | ∃ i k, i < d.M ∧ k < d.K ∧ e = i * d.stride_am + k * d.stride_ak}

/-- The elements of `b`'s storage that hold the matrix `b`. -/
def bSet (d : Inst) : Set Nat :=
  {e | ∃ k j, k < d.K ∧ j < d.N ∧ e = k * d.stride_bk + j * d.stride_bn}

/-- The bytes of `a` and of `b` that make up the two inputs: every program's read footprint. -/
def R (d : Inst) (_ : Mem) (_ : Fin (launch d).size) : Set Int :=
  d.a.elems (aSet d) ∪ d.b.elems (bSet d)

/-- The matrix row of lane `r` of program `p`'s tile. -/
def cRow (d : Inst) (p : Nat) (r : Nat) : Nat := pidM d p * 128 + r

/-- The matrix column of lane `c` of program `p`'s tile. -/
def cCol (d : Inst) (p : Nat) (c : Nat) : Nat := pidN d p * 256 + c

/-- The elements of `c`'s storage that program `p` writes: its in-range tile. -/
def tileSet (d : Inst) (p : Nat) : Set Nat :=
  {e | ∃ r c, r < 128 ∧ c < 256 ∧ cRow d p r < d.M ∧ cCol d p c < d.N ∧
    e = cRow d p r * d.stride_cm + cCol d p c * d.stride_cn}

/-- Program `i`'s write footprint. -/
def W (d : Inst) (_ : Mem) (i : Fin (launch d).size) : Set Int :=
  d.c.elems (tileSet d (i : Nat))

/-- Program `i` leaves `OutSpec` at each element of its tile. -/
def Q (d : Inst) (init : Mem) (i : Fin (launch d).size) (m : Mem) : Prop :=
  ∀ r c, r < 128 → c < 256 → cRow d i r < d.M → cCol d i c < d.N →
    OutSpec d init m (cRow d i r) (cCol d i c)

end Claims.Matmul

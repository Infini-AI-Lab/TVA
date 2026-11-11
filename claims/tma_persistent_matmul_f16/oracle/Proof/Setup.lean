import Claim
import TVA.Proof

/-!
# `tma_persistent_matmul_f16`: The launch, the tiles, the geometry of the grid and the integer arithmetic of the program ids.
-/

namespace Claims.TmaPersistentMatmul

open TVA

/-!
## `tma_persistent_matmul_f16`: the launch, the tiles and the loops

The launch is valid, each program of the grid is program `i`, and the sizes of the two loops of
the kernel are the tile count and `steps K`.
-/


/-! ## The launch -/

/-- The launch has `min NUM_SMS tiles` programs. -/
theorem launch_size (d : Inst) : (launch d).size = min d.NUM_SMS (tiles d) :=
  Launch.size_grid1 _ _ _

/-- Program `i` of the grid has `program_id(0) = i`. -/
theorem launch_pid (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = i := by
  have hi : (i : Nat) < min d.NUM_SMS (tiles d) := lt_of_lt_of_eq i.isLt (launch_size d)
  simp [Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi]

/-- Every dimension of the tutorial's blocks is a power of two. -/
theorem pow_two_of_mem : ∀ k, k ∈ ([128, 256, 64, 2] : Shape) → ∃ e, k = 2 ^ e := by
  intro k hk
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
  rcases hk with rfl | rfl | rfl | rfl
  · exact ⟨7, by norm_num⟩
  · exact ⟨8, by norm_num⟩
  · exact ⟨6, by norm_num⟩
  · exact ⟨1, by norm_num⟩

/-- Every block shape of the kernel is accepted by Triton. -/
theorem block_ok {s : Shape}
    (hs : s ∈ [[128, 256], [128, 64], [256, 64], [64, 256], [128, 2, 128], [128, 128, 2], [128, 128]]) :
    Shape.TritonOk s := by
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals
    exact ⟨fun k hk => pow_two_of_mem k (by
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hk ⊢; tauto), by norm_num⟩

/-- Every covered launch is valid. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 ?_ fun s hs => block_ok hs
  exact lt_of_le_of_lt (Nat.min_le_left _ _) hd.sms.2

/-! ## The tile count and the inner loop's length -/

/-- The tile count in closed form. -/
theorem tiles_eq (d : Inst) : tiles d = (d.M + 127) / 128 * ((d.N + 255) / 256) := by
  simp only [tiles]
  norm_num

/-- `steps K` is `cdiv(K, 64)`. -/
theorem steps_eq (K : Nat) : steps K = (K + 63) / 64 := rfl

/-- A positive `K` has at least one step. -/
theorem steps_pos {K : Nat} (hK : 0 < K) : 0 < steps K := by
  simp only [steps]; omega

/-- With `128 ∣ K` every inner step covers a full block of `64` columns. -/
theorem steps_mul {K : Nat} (hK : 128 ∣ K) : 64 * steps K = K := by
  obtain ⟨a, rfl⟩ := hK
  simp only [steps]
  rw [show 128 * a + 63 = 64 * (2 * a) + 63 by ring]
  rw [Nat.mul_add_div (by norm_num : 0 < 64)]
  ring

/-- `K ≤ 2^13` has at most `128` steps. -/
theorem steps_le {K : Nat} (hK : K ≤ 2 ^ 13) : steps K ≤ 128 := by
  simp only [steps]; omega

/-!
## `tma_persistent_matmul_f16`: the objects of the proof

The transcription of the kernel body, the grouped tile order the kernel computes, the values the
`a` and `b` matrices hold in the initial memory, the read and write footprints of one program, and
the invariant of the inner reduction loop.
-/


/-! ## The kernel body -/

/-- The kernel's body, unfolded line by line. -/
theorem body_eq (d : Inst) (c : Ctx) :
    (model d).body c =
      (do
        let start_pid := tl.program_id c 0
        let num_pid_m := (← tl.idiv (Tensor.scalar (IntV.ofNat d.M + (IntV.ofNat 127 : I32)))
          (Tensor.scalar (IntV.ofNat 128 : I32))).item
        let num_pid_n := (← tl.idiv (Tensor.scalar (IntV.ofNat d.N + (IntV.ofNat 255 : I32)))
          (Tensor.scalar (IntV.ofNat 256 : I32))).item
        let k_tiles := (← tl.idiv (Tensor.scalar (IntV.ofNat d.K + (IntV.ofNat 63 : I32)))
          (Tensor.scalar (IntV.ofNat 64 : I32))).item
        let num_tiles := num_pid_m * num_pid_n
        let tile_id_c := start_pid - (IntV.ofNat d.NUM_SMS : I32)
        let num_pid_in_group := (IntV.ofNat 8 : I32) * num_pid_n
        let _tile_id_c ← tl.forRange start_pid num_tiles (IntV.ofNat d.NUM_SMS : I32) tile_id_c
            (fun tile_id tile_id_c => do
          let (pid_m, pid_n) ← compute_pid tile_id num_pid_in_group num_pid_m 8
          let offs_am := pid_m * (IntV.ofNat 128 : I32)
          let offs_bn := pid_n * (IntV.ofNat 256 : I32)
          let accumulator := tl.zeros [128, 256] .f32
          let accumulator ← tl.forRange (IntV.ofInt 0 : I32) k_tiles (IntV.ofInt 1 : I32) accumulator
              (fun ki accumulator => do
            let offs_k := ki * (IntV.ofNat 64 : I32)
            let a ← tl.desc_load (bs := [128, 64]) (aDesc d) [offs_am, offs_k]
            let b ← tl.desc_load (bs := [256, 64]) (bDesc d) [offs_bn, offs_k]
            tl.dot { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }
              a b.trans accumulator)
          let tile_id_c := tile_id_c + (IntV.ofNat d.NUM_SMS : I32)
          let (pid_m, pid_n) ← compute_pid tile_id_c num_pid_in_group num_pid_m 8
          let offs_am_c := pid_m * (IntV.ofNat 128 : I32)
          let offs_bn_c := pid_n * (IntV.ofNat 256 : I32)
          let acc := accumulator.reshape [128, 2, 128] (by norm_num)
          let acc : Tensor [128, 128, 2] F32 := acc.permute [0, 2, 1]
          let (acc0, acc1) := Tensor.split (s := [128, 128]) acc
          let c0 : Tensor [128, 128] F16 ← tl.cvt acc0
          tl.desc_store (cDesc d) [offs_am_c, offs_bn_c] c0
          let c1 : Tensor [128, 128] F16 ← tl.cvt acc1
          tl.desc_store (cDesc d) [offs_am_c, offs_bn_c + (IntV.ofNat 128 : I32)] c1
          return tile_id_c)
        pure ()) := rfl

/-! ## The grouped tile order -/

/-- `num_pid_m = cdiv(M, 128)`. -/
def numPidM (d : Inst) : Nat := (d.M + 127) / 128

/-- `num_pid_n = cdiv(N, 256)`. -/
def numPidN (d : Inst) : Nat := (d.N + 255) / 256

/-- The group of tile id `t`: `t // num_pid_in_group`. -/
def grp (d : Inst) (t : Nat) : Nat := t / (8 * numPidN d)

/-- The number of rows of tile id `t`'s group: `min (num_pid_m - first_pid_m) 8`. -/
def gsm (d : Inst) (t : Nat) : Nat := min (numPidM d - 8 * grp d t) 8

/-- The row tile of tile id `t`. -/
def pidM (d : Inst) (t : Nat) : Nat := 8 * grp d t + t % gsm d t

/-- The column tile of tile id `t`. -/
def pidN (d : Inst) (t : Nat) : Nat := (t % (8 * numPidN d)) / gsm d t

/-- The tile count is the number of row blocks times the number of column blocks. -/
theorem tiles_eq_numPid (d : Inst) : tiles d = numPidM d * numPidN d := by
  simp only [tiles, numPidM, numPidN]
  have h : d.M + 128 - 1 = d.M + 127 := by omega
  have h2 : d.N + 256 - 1 = d.N + 255 := by omega
  rw [h, h2]

/-! ## The values of the input matrices in the initial memory -/

/-- The `float16` value `a`'s element `(i, k)` holds in `init` (junk `0` if memory has none; under
`Pre` every element of the matrix is present). -/
noncomputable def aGet (d : Inst) (init : Mem) (i : Nat) : Nat → F16 :=
  fun k => (init.read .f16 (d.a.ptr (↑(i * d.lda + k))).addr).getD default

/-- The `float16` value `b`'s element `(j, k)` holds in `init`. -/
noncomputable def bGet (d : Inst) (init : Mem) (j : Nat) : Nat → F16 :=
  fun k => (init.read .f16 (d.b.ptr (↑(j * d.ldb + k))).addr).getD default

/-- Row `i` of `a` holds finite values up to column `K`. -/
def RowFin (d : Inst) (init : Mem) (i : Nat) : Prop :=
  ∀ k, k < d.K → (aGet d init i k).isFinite = true

/-- Row `j` of `b` holds finite values up to column `K`. -/
def ColFin (d : Inst) (init : Mem) (j : Nat) : Prop :=
  ∀ k, k < d.K → (bGet d init j k).isFinite = true

/-- The exact partial sum of the products of row `i` of `a` and row `j` of `b` before step `t`. -/
noncomputable def pSum (d : Inst) (init : Mem) (i j t : Nat) : ℚ :=
  partSum (prodV (aGet d init i) (bGet d init j) d.K) t

/-- The lifted magnitude of step `t` of row `i` and row `j`. -/
noncomputable def lSum (d : Inst) (init : Mem) (i j t : Nat) : ℚ :=
  stepMag (liftV (aGet d init i) (bGet d init j) d.K) t

/-- The exact sum of the products of row `i` of `a` and row `j` of `b`. -/
noncomputable def dSum (d : Inst) (init : Mem) (i j : Nat) : ℚ :=
  dotSum (aGet d init i) (bGet d init j) d.K

/-- The tolerance of `Post` for row `i` of `a` and row `j` of `b`. -/
noncomputable def dTol (d : Inst) (init : Mem) (i j : Nat) : ℚ :=
  tol (aGet d init i) (bGet d init j) d.K

/-- The path magnitude of `Post` for row `i` of `a` and row `j` of `b`. -/
noncomputable def dPath (d : Inst) (init : Mem) (i j : Nat) : ℚ :=
  pathMag (aGet d init i) (bGet d init j) d.K

/-! ## The error recursion of the reduction -/

/-- The per-step relative error of `tl.dot` of `16`-bit operands with `64` products. -/
def rho : ℚ := DotLoop.stepRel Device.relDot16 64

/-- The per-step absolute error of `tl.dot` of `16`-bit operands with `64` products. -/
def alpha : ℚ := DotLoop.stepAbs Device.absDot16 64

/-- The error bound of `t` inner steps, on the products of row `i` and row `j`:
`E 0 = 0`, `E (t+1) = (1 + ρ)·E t + ρ·(|S t| + L t) + α`. -/
noncomputable def E (d : Inst) (init : Mem) (i j : Nat) : Nat → ℚ
  | 0 => 0
  | t + 1 => (1 + rho) * E d init i j t + rho * (|pSum d init i j t| + lSum d init i j t) + alpha

/-! ## The footprints of one program -/

/-- The elements of `c` that tile `t` writes: the in-bounds lanes of its two `128`-column halves. -/
def tileIdx (d : Inst) (t : Nat) : Set Nat :=
  {e | ∃ r n, r < 128 ∧ n < 256 ∧
    128 * pidM d t + r < d.M ∧ 256 * pidN d t + n < d.N ∧
    e = (128 * pidM d t + r) * d.ldc + (256 * pidN d t + n)}

/-- The tile ids program `i` runs: `[0, tiles d)` with `t % NUM_SMS = i`. -/
def progTiles (d : Inst) (i : Nat) : Set Nat := {t | t < tiles d ∧ t % d.NUM_SMS = i}

/-- The write footprint of program `i`: the elements of the tiles it runs. -/
def Wset (d : Inst) (i : Nat) : Set Nat := {e | ∃ t ∈ progTiles d i, e ∈ tileIdx d t}

/-- The write footprint of program `i`. -/
def W (d : Inst) (i : Nat) : Set Int := d.c.elems (Wset d i)

/-- The read footprint: the elements of `a`'s and `b`'s matrices. -/
def R (d : Inst) : Set Int := d.a.elems (aIdx d) ∪ d.b.elems (bIdx d)

/-! ## What one program leaves -/

/-- The specification of one output element: a finite `float32` within `dTol` of the exact sum,
whose rounding to `float16` is the stored value. -/
noncomputable def LaneSpec (d : Inst) (init : Mem) (i j : Nat) (m : Mem) : Prop :=
  ∃ v : F32, v.isFinite = true ∧ |v.toRat - dSum d init i j| ≤ dTol d init i j ∧
    ∃ c : F16, m.read .f16 (d.c.ptr (↑(i * d.ldc + j))).addr = some c ∧
      FloatV.NaNEq (FloatV.cvt v) c ∧
      (|dSum d init i j| + dTol d init i j ≤ Format.binary16.maxVal →
        c.isFinite = true)

/-- What program `i` leaves in the tiles it runs: every in-bounds lane holds a value, and a finite
row and column give the specified element. -/
noncomputable def Q (d : Inst) (init : Mem) (i : Fin (launch d).size) (m : Mem) : Prop :=
  ∀ t ∈ progTiles d (i : Nat), ∀ r n, r < 128 → n < 256 →
    128 * pidM d t + r < d.M → 256 * pidN d t + n < d.N →
    (∃ c : F16, m.read .f16
      (d.c.ptr (↑((128 * pidM d t + r) * d.ldc + (256 * pidN d t + n)))).addr = some c) ∧
    (RowFin d init (128 * pidM d t + r) → ColFin d init (256 * pidN d t + n) →
      LaneSpec d init (128 * pidM d t + r) (256 * pidN d t + n) m)

/-!
## The grouped tile order is a bijection, and the tiles are disjoint

The kernel enumerates its tiles in the grouped order (`GROUP_SIZE_M = 8`): tile id `t` of
`[0, tiles d)` gives the row tile `pidM d t` and the column tile `pidN d t` of `Defs.lean`. This
file proves that this map is a bijection onto `[0, numPidM d) × [0, numPidN d)`, hence that the
tiles of different trips of one program, and of different programs, are disjoint.
-/


/-- There is at least one row block. -/
theorem numPidM_pos {d : Inst} (hd : d.ok) : 0 < numPidM d := by
  simp only [numPidM]
  have := hd.pos.1
  omega

/-- There is at least one column block. -/
theorem numPidN_pos {d : Inst} (hd : d.ok) : 0 < numPidN d := by
  simp only [numPidN]
  have := hd.pos.2.1
  omega

/-- There is at least one tile. -/
theorem tiles_pos {d : Inst} (hd : d.ok) : 0 < tiles d := by
  rw [tiles_eq_numPid]
  exact Nat.mul_pos (numPidM_pos hd) (numPidN_pos hd)

/-- The row blocks cover at most `M + 127` rows. -/
theorem mul128_numPidM_le (d : Inst) : 128 * numPidM d ≤ d.M + 127 := by
  simp only [numPidM]; omega

/-- The column blocks cover at most `N + 255` columns. -/
theorem mul256_numPidN_le (d : Inst) : 256 * numPidN d ≤ d.N + 255 := by
  simp only [numPidN]; omega

/-- The number of rows of the group of tile `t` is positive. -/
theorem grp_mul_lt {d : Inst} (hd : d.ok) {t : Nat} (ht : t < tiles d) :
    8 * grp d t < numPidM d := by
  have hn : 0 < numPidN d := numPidN_pos hd
  have ht' : t < numPidM d * numPidN d := by rw [← tiles_eq_numPid]; exact ht
  have h1 : 8 * numPidN d * grp d t ≤ t := Nat.mul_div_le t (8 * numPidN d)
  have h2 : (8 * grp d t) * numPidN d < numPidM d * numPidN d := by
    calc (8 * grp d t) * numPidN d = 8 * numPidN d * grp d t := by ring
      _ ≤ t := h1
      _ < numPidM d * numPidN d := ht'
  exact Nat.lt_of_mul_lt_mul_right h2

/-- The group size of a tile is positive. -/
theorem gsm_pos {d : Inst} (hd : d.ok) {t : Nat} (ht : t < tiles d) : 0 < gsm d t := by
  have hg : 8 * grp d t < numPidM d := grp_mul_lt hd ht
  have hsub : 0 < numPidM d - 8 * grp d t := by omega
  have : 0 < min (numPidM d - 8 * grp d t) 8 := lt_min hsub (by omega)
  simpa [gsm] using this

/-- The second half of the key bound: the offset inside a group of `8` row tiles is short. -/
theorem mod_lt_gsm_mul {d : Inst} (hd : d.ok) {t : Nat} (ht : t < tiles d) :
    t % (8 * numPidN d) < gsm d t * numPidN d := by
  have hn : 0 < numPidN d := numPidN_pos hd
  have hG : 0 < 8 * numPidN d := by omega
  have ht' : t < numPidM d * numPidN d := by rw [← tiles_eq_numPid]; exact ht
  have hdec : t = 8 * numPidN d * grp d t + t % (8 * numPidN d) :=
    (Nat.div_add_mod t (8 * numPidN d)).symm
  by_cases hcase : 8 ≤ numPidM d - 8 * grp d t
  · have hgsm : gsm d t = 8 := by
      simp only [gsm]
      exact min_eq_right hcase
    rw [hgsm]
    exact Nat.mod_lt _ hG
  · have hgsm : gsm d t = numPidM d - 8 * grp d t := by
      simp only [gsm]
      exact min_eq_left (le_of_lt (not_le.mp hcase))
    have hgrp : 8 * grp d t < numPidM d := grp_mul_lt hd ht
    have hle : 8 * grp d t ≤ numPidM d := le_of_lt hgrp
    have hsum : 8 * grp d t + (numPidM d - 8 * grp d t) = numPidM d :=
      Nat.add_sub_cancel' hle
    have hB : numPidM d * numPidN d =
        8 * numPidN d * grp d t + gsm d t * numPidN d := by
      rw [hgsm]
      calc numPidM d * numPidN d
          = (8 * grp d t + (numPidM d - 8 * grp d t)) * numPidN d := by rw [hsum]
        _ = 8 * numPidN d * grp d t + (numPidM d - 8 * grp d t) * numPidN d := by ring
    have htlt : 8 * numPidN d * grp d t + t % (8 * numPidN d) <
        8 * numPidN d * grp d t + gsm d t * numPidN d := by
      rw [← hdec, ← hB]
      exact ht'
    omega

/-- The row block of a tile is a row block. -/
theorem pidM_lt {d : Inst} (hd : d.ok) {t : Nat} (ht : t < tiles d) : pidM d t < numPidM d := by
  have hg : 8 * grp d t < numPidM d := grp_mul_lt hd ht
  have hpos : 0 < gsm d t := gsm_pos hd ht
  have hmod : t % gsm d t < gsm d t := Nat.mod_lt _ hpos
  have hle : gsm d t ≤ numPidM d - 8 * grp d t := by
    simp only [gsm]; exact min_le_left _ _
  have hmod2 : t % gsm d t < numPidM d - 8 * grp d t := lt_of_lt_of_le hmod hle
  simp only [pidM]
  omega

/-- The column block of a tile is a column block. -/
theorem pidN_lt {d : Inst} (hd : d.ok) {t : Nat} (ht : t < tiles d) : pidN d t < numPidN d := by
  have hpos : 0 < gsm d t := gsm_pos hd ht
  have hkey : t % (8 * numPidN d) < gsm d t * numPidN d := mod_lt_gsm_mul hd ht
  show t % (8 * numPidN d) / gsm d t < numPidN d
  rw [Nat.div_lt_iff_lt_mul hpos, Nat.mul_comm (numPidN d) (gsm d t)]
  exact hkey

/-- Distinct tiles have distinct (row block, column block). -/
theorem pid_inj {d : Inst} (hd : d.ok) {t t' : Nat} (ht : t < tiles d) (ht' : t' < tiles d)
    (hm : pidM d t = pidM d t') (hn : pidN d t = pidN d t') : t = t' := by
  have hdec : t = 8 * numPidN d * grp d t + t % (8 * numPidN d) :=
    (Nat.div_add_mod t (8 * numPidN d)).symm
  simp only [pidM] at hm
  simp only [pidN] at hn
  have hmodlt : t % gsm d t < 8 := by
    have h1 : t % gsm d t < gsm d t := Nat.mod_lt _ (gsm_pos hd ht)
    have h2 : gsm d t ≤ 8 := by simp only [gsm]; exact min_le_right _ _
    omega
  have hmodlt' : t' % gsm d t' < 8 := by
    have h1 : t' % gsm d t' < gsm d t' := Nat.mod_lt _ (gsm_pos hd ht')
    have h2 : gsm d t' ≤ 8 := by simp only [gsm]; exact min_le_right _ _
    omega
  have hg : grp d t = grp d t' := by omega
  have hdec' : t' = 8 * numPidN d * grp d t + t' % (8 * numPidN d) := by
    rw [hg]
    exact (Nat.div_add_mod t' (8 * numPidN d)).symm
  have hgsm : gsm d t = gsm d t' := by
    simp only [gsm]
    rw [hg]
  have hm2 : t % gsm d t' = t' % gsm d t' := by
    have h1 : t % gsm d t = t' % gsm d t' := by omega
    rwa [hgsm] at h1
  have hn2 : t % (8 * numPidN d) / gsm d t' = t' % (8 * numPidN d) / gsm d t' := by
    have h1 : t % (8 * numPidN d) / gsm d t = t' % (8 * numPidN d) / gsm d t' := hn
    rwa [hgsm] at h1
  have hpos' : 0 < gsm d t' := gsm_pos hd ht'
  have hgen : ∀ (b a : Nat), b = 8 * numPidN d * grp d t + a →
      b % gsm d t' =
        (8 * numPidN d * grp d t + a % gsm d t') % gsm d t' := by
    intro b a h
    rw [h]
    conv_lhs =>
      rw [← Nat.div_add_mod a (gsm d t')]
      rw [show 8 * numPidN d * grp d t +
            (gsm d t' * (a / gsm d t') + a % gsm d t')
          = (8 * numPidN d * grp d t + a % gsm d t') + (a / gsm d t') * gsm d t'
          from by ring]
      rw [Nat.add_mul_mod_self_right]
  have hf1 : t % gsm d t' =
      (8 * numPidN d * grp d t + t % (8 * numPidN d) % gsm d t') % gsm d t' :=
    hgen t (t % (8 * numPidN d)) hdec
  have hf2 : t' % gsm d t' =
      (8 * numPidN d * grp d t + t' % (8 * numPidN d) % gsm d t') % gsm d t' :=
    hgen t' (t' % (8 * numPidN d)) hdec'
  have hrr : t % (8 * numPidN d) % gsm d t' = t' % (8 * numPidN d) % gsm d t' := by
    have h1 : (8 * numPidN d * grp d t + t % (8 * numPidN d) % gsm d t') % gsm d t' =
        (8 * numPidN d * grp d t + t' % (8 * numPidN d) % gsm d t') % gsm d t' := by
      rw [← hf1, ← hf2]
      exact hm2
    have h2 := Nat.ModEq.add_left_cancel' (8 * numPidN d * grp d t) h1
    exact Nat.ModEq.eq_of_lt_of_lt h2 (Nat.mod_lt _ hpos') (Nat.mod_lt _ hpos')
  have hs : t % (8 * numPidN d) = t' % (8 * numPidN d) := by
    rw [← Nat.div_add_mod (t % (8 * numPidN d)) (gsm d t'),
      ← Nat.div_add_mod (t' % (8 * numPidN d)) (gsm d t'), hrr, hn2]
  rw [hdec, hdec', hs]

/-- Every (row block, column block) is the pair of some tile. -/
theorem pid_surj {d : Inst} (hd : d.ok) {pm pn : Nat} (hm : pm < numPidM d) (hn : pn < numPidN d) :
    ∃ t, t < tiles d ∧ pidM d t = pm ∧ pidN d t = pn := by
  let f : Fin (tiles d) → Fin (numPidM d) × Fin (numPidN d) := fun t =>
    (⟨pidM d t.1, pidM_lt hd t.2⟩, ⟨pidN d t.1, pidN_lt hd t.2⟩)
  have hf : Function.Injective f := by
    intro a b hab
    have h1 : pidM d a.1 = pidM d b.1 := by
      have := congrArg (fun x : Fin (numPidM d) × Fin (numPidN d) => (x.1 : Nat)) hab
      simpa [f] using this
    have h2 : pidN d a.1 = pidN d b.1 := by
      have := congrArg (fun x : Fin (numPidM d) × Fin (numPidN d) => (x.2 : Nat)) hab
      simpa [f] using this
    exact Fin.ext (pid_inj hd a.2 b.2 h1 h2)
  have hcard : Fintype.card (Fin (tiles d)) =
      Fintype.card (Fin (numPidM d) × Fin (numPidN d)) := by
    simp [Fintype.card_prod, tiles_eq_numPid]
  have hbij := (Fintype.bijective_iff_injective_and_card f).mpr ⟨hf, hcard⟩
  obtain ⟨t, ht⟩ := hbij.2 (⟨pm, hm⟩, ⟨pn, hn⟩)
  refine ⟨t.1, t.2, ?_, ?_⟩
  · have := congrArg (fun x : Fin (numPidM d) × Fin (numPidN d) => (x.1 : Nat)) ht
    simpa [f] using this
  · have := congrArg (fun x : Fin (numPidM d) × Fin (numPidN d) => (x.2 : Nat)) ht
    simpa [f] using this

/-- The tile id, and the lane inside the tile, of an in-bounds element of `c`. -/
theorem pid_surj' {d : Inst} (hd : d.ok) {I J : Nat} (hI : I < d.M) (hJ : J < d.N) :
    ∃ t, t < tiles d ∧ 128 * pidM d t + I % 128 = I ∧ 256 * pidN d t + J % 256 = J := by
  have hMle : d.M ≤ numPidM d * 128 := by
    have h : (d.M + 127) / 128 ≤ numPidM d := by
      simp only [numPidM]
      exact le_refl _
    have h2 := (Nat.div_le_iff_le_mul (k := 128) (x := d.M + 127)
      (y := numPidM d) (by norm_num)).mp h
    omega
  have hNle : d.N ≤ numPidN d * 256 := by
    have h : (d.N + 255) / 256 ≤ numPidN d := by
      simp only [numPidN]
      exact le_refl _
    have h2 := (Nat.div_le_iff_le_mul (k := 256) (x := d.N + 255)
      (y := numPidN d) (by norm_num)).mp h
    omega
  have hpm : I / 128 < numPidM d := by
    rw [Nat.div_lt_iff_lt_mul (by norm_num : 0 < 128)]
    calc I < d.M := hI
      _ ≤ numPidM d * 128 := hMle
  have hpn : J / 256 < numPidN d := by
    rw [Nat.div_lt_iff_lt_mul (by norm_num : 0 < 256)]
    calc J < d.N := hJ
      _ ≤ numPidN d * 256 := hNle
  obtain ⟨t, ht, hm, hn⟩ := pid_surj hd hpm hpn
  refine ⟨t, ht, ?_, ?_⟩
  · rw [hm]
    exact Nat.div_add_mod I 128
  · rw [hn]
    exact Nat.div_add_mod J 256

/-- The storage index of a lane of a block with two offsets, in a descriptor of shape `[A, B]`
and strides `[s, 1]`. -/
theorem elemAt_two {δ : DType} (D : TensorDesc δ) {A B s : Nat}
    (hshape : D.shape = [A, B]) (hstr : D.strides = [s, 1])
    (o0 o1 : Int) {bs : Shape} (i : Idx bs) :
    D.elemAt (TensorDesc.coord [o0, o1] i)
      = (o0 + (i.coord 0 : Int)) * (s : Int) + (o1 + (i.coord 1 : Int)) := by
  simp only [TensorDesc.elemAt, TensorDesc.rank, TensorDesc.coord, hshape, hstr,
    List.length_cons, List.length_nil]
  rw [show List.range 2 = [0, 1] from rfl]
  simp

/-- The storage index of a lane of a two-dimensional descriptor block, from its offsets. -/
theorem elemAt_toNat_two {δ : DType} (D : TensorDesc δ) {A B s : Nat}
    (hshape : D.shape = [A, B]) (hstr : D.strides = [s, 1])
    (o0 o1 : Int) {bs : Shape} (i : Idx bs)
    (h0 : 0 ≤ o0 + (i.coord 0 : Int)) (h2 : 0 ≤ o1 + (i.coord 1 : Int)) :
    (D.elemAt (TensorDesc.coord [o0, o1] i)).toNat
      = (o0 + (i.coord 0 : Int)).toNat * s + (o1 + (i.coord 1 : Int)).toNat := by
  rw [elemAt_two D hshape hstr o0 o1 i]
  rw [Int.toNat_add (mul_nonneg h0 (Int.natCast_nonneg s)) h2]
  rw [Int.toNat_mul h0 (Int.natCast_nonneg s), Int.toNat_natCast]

/-- Different elements of the matrix `c` have different storage indices. -/
theorem elem_inj {d : Inst} (hd : d.ok) {i j i' j' : Nat} (_hi : i < d.M) (hj : j < d.N)
    (_hi' : i' < d.M) (hj' : j' < d.N) (h : i * d.ldc + j = i' * d.ldc + j') :
    i = i' ∧ j = j' := by
  have hldc : 0 < d.ldc := by
    have h1 := hd.pos.2.1
    have h2 := hd.N_ldc
    omega
  have hjlt : j < d.ldc := by
    have h2 := hd.N_ldc
    omega
  have hj'lt : j' < d.ldc := by
    have h2 := hd.N_ldc
    omega
  have hjeq : j = j' := by
    have h1 := congrArg (fun x => x % d.ldc) h
    simp only [Nat.mul_add_mod_self_right, Nat.mod_eq_of_lt hjlt,
      Nat.mod_eq_of_lt hj'lt] at h1
    exact h1
  have hieq : i = i' := by
    have h2 : i * d.ldc = i' * d.ldc := by omega
    exact Nat.mul_right_cancel hldc h2
  exact ⟨hieq, hjeq⟩

/-- The tiles `t` and `t'` write disjoint elements of `c`. -/
theorem tileIdx_disjoint {d : Inst} (hd : d.ok) {t t' : Nat} (ht : t < tiles d) (ht' : t' < tiles d)
    (h : t ≠ t') : Disjoint (tileIdx d t) (tileIdx d t') := by
  rw [Set.disjoint_left]
  rintro e ⟨r, n, hr, hn, hrM, hnN, he⟩ ⟨r', n', hr', hn', hrM', hnN', he'⟩
  have hmain : (128 * pidM d t + r) * d.ldc + (256 * pidN d t + n)
      = (128 * pidM d t' + r') * d.ldc + (256 * pidN d t' + n') := he.symm.trans he'
  obtain ⟨hI, hJ⟩ := elem_inj hd hrM hnN hrM' hnN' hmain
  have hpidM : pidM d t = pidM d t' := by
    have h1 : (128 * pidM d t + r) / 128 = (128 * pidM d t' + r') / 128 := by rw [hI]
    have h2 : (128 * pidM d t + r) / 128 = pidM d t := by
      rw [Nat.mul_add_div (by norm_num : 0 < 128), Nat.div_eq_of_lt hr, Nat.add_zero]
    have h3 : (128 * pidM d t' + r') / 128 = pidM d t' := by
      rw [Nat.mul_add_div (by norm_num : 0 < 128), Nat.div_eq_of_lt hr', Nat.add_zero]
    rw [h2, h3] at h1
    exact h1
  have hpidN : pidN d t = pidN d t' := by
    have h1 : (256 * pidN d t + n) / 256 = (256 * pidN d t' + n') / 256 := by rw [hJ]
    have h2 : (256 * pidN d t + n) / 256 = pidN d t := by
      rw [Nat.mul_add_div (by norm_num : 0 < 256), Nat.div_eq_of_lt hn, Nat.add_zero]
    have h3 : (256 * pidN d t' + n') / 256 = pidN d t' := by
      rw [Nat.mul_add_div (by norm_num : 0 < 256), Nat.div_eq_of_lt hn', Nat.add_zero]
    rw [h2, h3] at h1
    exact h1
  exact h (pid_inj hd ht ht' hpidM hpidN)

/-- A tile's elements are elements of the matrix `c`. -/
theorem tileIdx_subset_cIdx (d : Inst) (t : Nat) : tileIdx d t ⊆ cIdx d := by
  rintro e ⟨r, n, hr, hn, hrM, hnN, he⟩
  exact ⟨128 * pidM d t + r, 256 * pidN d t + n, hrM, hnN, he⟩

/-- A program's elements are elements of the matrix `c`. -/
theorem Wset_subset_cIdx (d : Inst) (i : Nat) : Wset d i ⊆ cIdx d := by
  rintro e ⟨t, ht, he⟩
  exact tileIdx_subset_cIdx d t he

/-- Different programs write disjoint elements of `c`. -/
theorem Wset_disjoint {d : Inst} (hd : d.ok) {i j : Nat} (h : i ≠ j) :
    Disjoint (Wset d i) (Wset d j) := by
  rw [Set.disjoint_left]
  rintro e ⟨t, ht, het⟩ ⟨t', ht', het'⟩
  obtain ⟨ht1, ht2⟩ := ht
  obtain ⟨ht1', ht2'⟩ := ht'
  have htt' : t = t' := by
    by_contra hne
    exact Set.disjoint_left.mp (tileIdx_disjoint hd ht1 ht1' hne) het het'
  exact h (by rw [← ht2, htt', ht2'])

/-- A program writes only elements of the matrix `c`. -/
theorem W_subset_c {d : Inst} (i : Nat) : W d i ⊆ d.c.elems (cIdx d) := by
  exact Buf.elems_mono d.c (Wset_subset_cIdx d i)

/-- Different programs write disjoint memory. -/
theorem W_disjoint_W {d : Inst} (hd : d.ok) {i j : Nat} (h : i ≠ j) : Disjoint (W d i) (W d j) := by
  exact Buf.elems_disjoint d.c (Wset_disjoint hd h)

/-- A program's writes stay off the input matrices. -/
theorem W_disjoint_reads {d : Inst} (hd : d.ok) (i : Nat) : Disjoint (W d i) (R d) := by
  have hR : Disjoint (d.c.elems (cIdx d)) (R d) := by
    show Disjoint (d.c.elems (cIdx d)) (d.a.elems (aIdx d) ∪ d.b.elems (bIdx d))
    exact Disjoint.union_right hd.c_a hd.c_b
  exact Disjoint.mono_left (W_subset_c i) hR

/-- A program's writes are disjoint from the reads of every program. -/
theorem reads_disjoint_W {d : Inst} (hd : d.ok) (i : Nat) : Disjoint (R d) (W d i) :=
  (W_disjoint_reads hd i).symm

/-!
## The trips of one program

The persistent loop of program `i` runs the tile ids `i, i + NUM_SMS, i + 2·NUM_SMS, …` below
`tiles d`. This file names their number (`trips d i`), identifies it with the `tripCount` of the
loop, and proves that the union of the tiles' elements is the program's write footprint.
-/


/-- The number of tiles program `i` runs. -/
def trips (d : Inst) (i : Nat) : Nat := (tiles d - i + d.NUM_SMS - 1) / d.NUM_SMS

/-- The kernel's loop trip count is `trips d i`. -/
theorem tripCount_eq {d : Inst} {i : Nat} (hS : 0 < d.NUM_SMS) (hi : i ≤ tiles d) :
    tl.tripCount (i : Int) (tiles d : Int) (d.NUM_SMS : Int) = trips d i := by
  unfold tl.tripCount trips
  have hcast : (tiles d : Int) - (i : Int) + (d.NUM_SMS : Int) - 1
      = ((tiles d - i + d.NUM_SMS - 1 : Nat) : Int) := by
    rw [← Int.ofNat_sub hi]
    have hlt : 1 ≤ tiles d - i + d.NUM_SMS := by omega
    rw [Int.ofNat_sub hlt]
    push_cast
    ring
  rw [hcast, ← Int.natCast_ediv, Int.toNat_natCast]

/-- Trip `t` of program `i` is a tile. -/
theorem trip_lt {d : Inst} {i t : Nat} (hS : 0 < d.NUM_SMS) (hi : i ≤ tiles d)
    (ht : t < trips d i) : i + t * d.NUM_SMS < tiles d := by
  rw [trips, Nat.lt_div_iff_mul_lt hS] at ht
  have hsub : (tiles d - i + d.NUM_SMS - 1) - (d.NUM_SMS - 1) = tiles d - i := by omega
  rw [hsub] at ht
  omega

/-- Trips below the last one are below the trip count. -/
theorem trips_mono {d : Inst} {i t u : Nat} (_hS : 0 < d.NUM_SMS) (ht : t < u) (hu : u < trips d i) :
    t < trips d i := lt_trans ht hu

/-- Distinct trips of one program are distinct tiles. -/
theorem tile_id_ne {d : Inst} {i t u : Nat} (hS : 0 < d.NUM_SMS) (htu : t < u) :
    i + t * d.NUM_SMS ≠ i + u * d.NUM_SMS := by
  intro h
  have hlt : t * d.NUM_SMS < u * d.NUM_SMS := Nat.mul_lt_mul_of_pos_right htu hS
  omega

/-- The tile ids program `i` runs are `i + t·NUM_SMS` for `t < trips d i`. -/
theorem progTiles_eq {d : Inst} (hS : 0 < d.NUM_SMS) {i : Nat} (hi : i < d.NUM_SMS) :
    progTiles d i = {T | ∃ t, t < trips d i ∧ T = i + t * d.NUM_SMS} := by
  ext T
  simp only [progTiles, Set.mem_ofPred_eq]
  constructor
  · rintro ⟨hT, hmod⟩
    refine ⟨T / d.NUM_SMS, ?_, ?_⟩
    · rw [trips, Nat.lt_div_iff_mul_lt hS]
      have hdec : T = d.NUM_SMS * (T / d.NUM_SMS) + i := by
        have h := (Nat.div_add_mod T d.NUM_SMS).symm
        omega
      have hlt : T / d.NUM_SMS * d.NUM_SMS < tiles d - i := by
        have h1 : T / d.NUM_SMS * d.NUM_SMS + i < tiles d := by
          rw [Nat.mul_comm]
          rw [← hdec]
          exact hT
        omega
      have hsub : (tiles d - i + d.NUM_SMS - 1) - (d.NUM_SMS - 1) = tiles d - i := by omega
      rwa [hsub]
    · show T = i + T / d.NUM_SMS * d.NUM_SMS
      rw [Nat.mul_comm (T / d.NUM_SMS) d.NUM_SMS]
      have hdec : T = d.NUM_SMS * (T / d.NUM_SMS) + T % d.NUM_SMS :=
        (Nat.div_add_mod T d.NUM_SMS).symm
      omega
  · rintro ⟨t, ht, rfl⟩
    have hlt : t * d.NUM_SMS < tiles d - i := by
      rw [trips, Nat.lt_div_iff_mul_lt hS] at ht
      have hsub : (tiles d - i + d.NUM_SMS - 1) - (d.NUM_SMS - 1) = tiles d - i := by omega
      rwa [hsub] at ht
    refine ⟨by omega, ?_⟩
    rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hi]

/-- The write footprint of program `i` is the union of its tiles' elements. -/
theorem Wset_eq {d : Inst} (hS : 0 < d.NUM_SMS) {i : Nat} (hi : i < d.NUM_SMS) :
    Wset d i = {e | ∃ t, t < trips d i ∧ e ∈ tileIdx d (i + t * d.NUM_SMS)} := by
  rw [Wset, progTiles_eq hS hi]
  ext e
  simp only [Set.mem_ofPred_eq]
  constructor
  · rintro ⟨T, ⟨t, ht, rfl⟩, he⟩
    exact ⟨t, ht, he⟩
  · rintro ⟨t, ht, he⟩
    exact ⟨i + t * d.NUM_SMS, ⟨t, ht, rfl⟩, he⟩

/-- Iterated write footprints of one buffer are the elements of the union of the index sets. -/
theorem iterW_elems {dty : DType} (c : Buf dty) (F : Nat → Set Nat) (n : Nat) :
    iterW (fun t => c.elems (F t)) n = c.elems {j | ∃ t, t < n ∧ j ∈ F t} := by
  induction n with
  | zero =>
    rw [iterW_zero]
    ext x
    simp [Buf.elems]
  | succ n ih =>
    have hsucc : iterW (fun t => c.elems (F t)) (n + 1)
        = iterW (fun t => c.elems (F t)) n ∪ c.elems (F n) := by
      ext x
      simp only [iterW, Set.mem_union, Set.mem_ofPred_eq]
      constructor
      · rintro ⟨t, ht, hx⟩
        rcases Nat.lt_or_ge t n with h | h
        · exact Or.inl ⟨t, h, hx⟩
        · exact Or.inr (by rw [show n = t by omega]; exact hx)
      · rintro (⟨t, ht, hx⟩ | hx)
        · exact ⟨t, by omega, hx⟩
        · exact ⟨n, by omega, hx⟩
    rw [hsucc, ih, ← Buf.elems_union]
    congr 1
    ext j
    simp only [Set.mem_ofPred_eq]
    constructor
    · rintro (⟨t, ht, hj⟩ | hj)
      · exact ⟨t, by omega, hj⟩
      · exact ⟨n, by omega, hj⟩
    · rintro ⟨t, ht, hj⟩
      rcases Nat.lt_or_ge t n with h | h
      · exact Or.inl ⟨t, h, hj⟩
      · exact Or.inr (by rw [show n = t by omega]; exact hj)

/-- Different trips of one program write disjoint memory. -/
theorem Wt_disjoint {d : Inst} (hd : d.ok) {i : Nat} (hi : i < min d.NUM_SMS (tiles d)) :
    ∀ t u, t < u → u < tl.tripCount (i : Int) (tiles d : Int) (d.NUM_SMS : Int) →
      Disjoint (d.c.elems (tileIdx d (i + t * d.NUM_SMS)))
        (d.c.elems (tileIdx d (i + u * d.NUM_SMS))) := by
  intro t u htu hu
  have hS : 0 < d.NUM_SMS := hd.sms.1
  have hile : i ≤ tiles d := le_of_lt (lt_of_lt_of_le hi (min_le_right _ _))
  rw [tripCount_eq hS hile] at hu
  have ht' : t < trips d i := lt_trans htu hu
  have htiles : i + t * d.NUM_SMS < tiles d := trip_lt hS hile ht'
  have hutiles : i + u * d.NUM_SMS < tiles d := trip_lt hS hile hu
  exact Buf.elems_disjoint d.c (tileIdx_disjoint hd htiles hutiles (tile_id_ne hS htu))

/-- The loop's iterated footprint is the program's write footprint. -/
theorem iterW_tileIdx_eq {d : Inst} (hd : d.ok) {i : Nat}
    (hi : i < min d.NUM_SMS (tiles d)) :
    iterW (fun t => d.c.elems (tileIdx d (i + t * d.NUM_SMS)))
      (tl.tripCount (i : Int) (tiles d : Int) (d.NUM_SMS : Int)) = W d i := by
  have hS : 0 < d.NUM_SMS := hd.sms.1
  have hile : i ≤ tiles d := le_of_lt (lt_of_lt_of_le hi (min_le_right _ _))
  have hiS : i < d.NUM_SMS := lt_of_lt_of_le hi (min_le_left _ _)
  have hI : iterW (fun t => d.c.elems (tileIdx d (i + t * d.NUM_SMS))) (trips d i)
      = d.c.elems {j | ∃ t, t < trips d i ∧ j ∈ tileIdx d (i + t * d.NUM_SMS)} :=
    iterW_elems d.c (fun t => tileIdx d (i + t * d.NUM_SMS)) (trips d i)
  have hW : W d i =
      d.c.elems {j | ∃ t, t < trips d i ∧ j ∈ tileIdx d (i + t * d.NUM_SMS)} := by
    show d.c.elems (Wset d i) = _
    rw [Wset_eq hS hiS]
  rw [tripCount_eq hS hile, hI, hW]

/-!
## The tile coordinates

`compute_pid` computes the grouped-order row and column tile of a tile id. It reads no memory and
writes none, so it is `Safe` at any footprint; this file identifies its result with `pidM` and
`pidN`.
-/


set_option maxHeartbeats 800000 in
/-- `_compute_pid` (lines 291–298) computes the tile's row and column block. -/
theorem safe_compute_pid {d : Inst} (hd : d.ok) {E : Env} {R W : Set Int}
    {β : Type} {Qq : β → Mem → Prop} {m : Mem}
    {tile_id num_pid_in_group num_pid_m : I32} {T : Nat}
    (htile : tile_id.toInt = (T : Int))
    (hG : num_pid_in_group.toInt = ((8 * numPidN d : Nat) : Int))
    (hM : num_pid_m.toInt = ((numPidM d : Nat) : Int))
    (hT : T < tiles d)
    {k : I32 × I32 → Prog β}
    (hk : ∀ p : I32 × I32, p.1.toInt = ((pidM d T : Nat) : Int) →
        p.2.toInt = ((pidN d T : Nat) : Int) → Safe E R W Qq m (k p)) :
    Safe E R W Qq m (compute_pid tile_id num_pid_in_group num_pid_m 8 >>= k) := by
  have hG8 : 0 < 8 * numPidN d := by
    have := numPidN_pos hd; omega
  have hglt : 8 * grp d T < numPidM d := grp_mul_lt hd hT
  have hMlt : numPidM d < 2 ^ 31 := by
    have h1 : numPidM d ≤ d.M + 127 := by
      simp only [numPidM]; exact Nat.div_le_self _ _
    have h2 := hd.M_ctx
    omega
  have h8g_lt : grp d T * 8 < 2 ^ 31 := by omega
  have hgsmpos : 0 < gsm d T := gsm_pos hd hT
  have hpMlt : pidM d T < 2 ^ 31 := by
    have := pidM_lt hd hT; omega
  have hsum_lt : grp d T * 8 + T % gsm d T < 2 ^ 31 := by
    have h := hpMlt
    simp only [pidM] at h
    omega
  simp only [compute_pid, Prog.bind_eq, Prog.bind_assoc, Prog.pure_eq, Prog.ret_bind]
  refine safe_idiv1 (x := tile_id) (y := num_pid_in_group) ?_ ?_ ?_
  · rw [htile]; exact Int.natCast_nonneg T
  · rw [hG]; exact_mod_cast hG8
  · intro r1 hr1
    have h1 : r1.item.toInt = ((grp d T : Nat) : Int) := by
      rw [hr1, htile, hG, ← Int.natCast_ediv]
      rfl
    set g : I32 := (tl.iminimum (Tensor.scalar (num_pid_m - r1.item * IntV.ofNat 8))
      (Tensor.scalar (IntV.ofNat 8))).item with hg_def
    have hA : (num_pid_m - r1.item * IntV.ofNat 8).toInt
        = ((numPidM d - 8 * grp d T : Nat) : Int) := by
      have hmul : (r1.item * IntV.ofNat 8).toInt = ((grp d T * 8 : Nat) : Int) :=
        IntV.I32.mul_nat h1 (IntV.I32.toInt_ofNat (by norm_num)) h8g_lt
      have hsub := IntV.I32.sub_nat hM hmul (by omega : grp d T * 8 ≤ numPidM d) hMlt
      rw [hsub, Nat.mul_comm (grp d T) 8]
    have hgsm : g.toInt = ((gsm d T : Nat) : Int) := by
      have hgeq : g = IntV.min (num_pid_m - r1.item * IntV.ofNat 8) (IntV.ofNat 8) := by
        rw [hg_def]
        simp only [tl.iminimum, Tensor.item, Tensor.scalar, Tensor.map2]
      rw [hgeq, IntV.toInt_min, hA, IntV.I32.toInt_ofNat (by norm_num), ← Nat.cast_min]
      rfl
    refine safe_irem1 (x := tile_id) (y := g) ?_ ?_ ?_
    · rw [htile]; exact Int.natCast_nonneg T
    · rw [hgsm]; exact_mod_cast hgsmpos
    · intro r2 hr2
      have h2 : r2.item.toInt = ((T % gsm d T : Nat) : Int) := by
        rw [hr2, htile, hgsm, ← Int.natCast_emod]
      refine safe_irem1 (x := tile_id) (y := num_pid_in_group) ?_ ?_ ?_
      · rw [htile]; exact Int.natCast_nonneg T
      · rw [hG]; exact_mod_cast hG8
      · intro r3 hr3
        have h3 : r3.item.toInt = ((T % (8 * numPidN d) : Nat) : Int) := by
          rw [hr3, htile, hG, ← Int.natCast_emod]
        refine safe_idiv1 (x := r3.item) (y := g) ?_ ?_ ?_
        · rw [h3]; exact Int.natCast_nonneg _
        · rw [hgsm]; exact_mod_cast hgsmpos
        · intro r4 hr4
          refine hk (r1.item * IntV.ofNat 8 + r2.item, r4.item) ?_ ?_
          · have hfirst : (r1.item * IntV.ofNat 8).toInt = ((grp d T * 8 : Nat) : Int) :=
              IntV.I32.mul_nat h1 (IntV.I32.toInt_ofNat (by norm_num)) h8g_lt
            have hadd := IntV.I32.add_nat hfirst h2 hsum_lt
            rw [hadd]
            have hpid : grp d T * 8 + T % gsm d T = pidM d T := by
              simp only [pidM]; omega
            rw [hpid]
          · rw [hr4, h3, hgsm, ← Int.natCast_ediv]
            rfl

end Claims.TmaPersistentMatmul

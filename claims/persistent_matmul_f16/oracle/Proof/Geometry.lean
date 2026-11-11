import Claim
import TVA.Proof

/-!
# The shape of the launch: the number of programs, the tile of each trip, and the footprints
(`Rd`, `Wr`) of the programs, with their disjointness and the cover of `c` by tiles.
-/

set_option maxRecDepth 8000

namespace Claims.PersistentMatmul

open TVA

/-- The model's body, with `d`'s buffers, sizes and strides substituted. -/
theorem body_eq (d : Inst) (ctx : Ctx) :
    (model d).body ctx =
      (do
        let start_pid := tl.program_id ctx 0
        let num_pid_m := (← tl.idiv (Tensor.scalar ((IntV.ofNat d.M : I32) + (IntV.ofNat 127 : I32)))
          (Tensor.scalar (IntV.ofNat 128 : I32))).item
        let num_pid_n := (← tl.idiv (Tensor.scalar ((IntV.ofNat d.N : I32) + (IntV.ofNat 255 : I32)))
          (Tensor.scalar (IntV.ofNat 256 : I32))).item
        let k_tiles := (← tl.idiv (Tensor.scalar ((IntV.ofNat d.K : I32) + (IntV.ofNat 63 : I32)))
          (Tensor.scalar (IntV.ofNat 64 : I32))).item
        let num_tiles := num_pid_m * num_pid_n
        let tile_id_c := start_pid - (IntV.ofNat d.NUM_SMS : I32)
        let offs_k_for_mask : Tensor [64] I32 := tl.arange 0 64
        let num_pid_in_group := (IntV.ofNat 8 : I32) * num_pid_n
        let _tile_id_c ← tl.forRange start_pid num_tiles (IntV.ofNat d.NUM_SMS : I32) tile_id_c
            fun tile_id tile_id_c => do
          let (pid_m, pid_n) ← compute_pid tile_id num_pid_in_group num_pid_m 8
          let start_m := pid_m * (IntV.ofNat 128 : I32)
          let start_n := pid_n * (IntV.ofNat 256 : I32)
          let offs_am : Tensor [128] I32 := start_m + tl.arange 0 128
          let offs_bn : Tensor [256] I32 := start_n + tl.arange 0 256
          let offs_am := tl.where (offs_am.ltS (IntV.ofNat d.M : I32)) offs_am
            (tl.full [128] (IntV.ofInt 0 : I32))
          let offs_bn := tl.where (offs_bn.ltS (IntV.ofNat d.N : I32)) offs_bn
            (tl.full [256] (IntV.ofInt 0 : I32))
          let offs_am ← tl.max_contiguous_multiple_of offs_am 128 128
          let offs_bn ← tl.max_contiguous_multiple_of offs_bn 256 256
          let accumulator := tl.zeros [128, 256] .f32
          let accumulator ← tl.forRange (IntV.ofInt 0 : I32) k_tiles (IntV.ofInt 1 : I32) accumulator
              fun ki accumulator => do
            let offs_k : Tensor [64] I32 := ki * (IntV.ofNat 64 : I32) + tl.arange 0 64
            let a_ptrs := d.a.ptr 0 +
              (offs_am.col 64 * (IntV.ofNat d.stride_am : I32) + offs_k.row 128 * (IntV.ofNat d.stride_ak : I32))
            let b_ptrs := d.b.ptr 0 +
              (offs_k.col 256 * (IntV.ofNat d.stride_bk : I32) + offs_bn.row 64 * (IntV.ofNat d.stride_bn : I32))
            let a ← tl.load a_ptrs
              (some ((offs_k_for_mask.row 128).ltS ((IntV.ofNat d.K : I32) - ki * (IntV.ofNat 64 : I32))))
              (some (← tl.convert .f32 .f16 (tl.full [128, 64] (FloatV.litF32 (FloatV.ofPat 0)))))
            let b ← tl.load b_ptrs
              (some ((offs_k_for_mask.col 256).ltS ((IntV.ofNat d.K : I32) - ki * (IntV.ofNat 64 : I32))))
              (some (← tl.convert .f32 .f16 (tl.full [64, 256] (FloatV.litF32 (FloatV.ofPat 0)))))
            tl.dot { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }
              a b accumulator
          let tile_id_c := tile_id_c + (IntV.ofNat d.NUM_SMS : I32)
          let (pid_m, pid_n) ← compute_pid tile_id_c num_pid_in_group num_pid_m 8
          let offs_cm : Tensor [128] I32 := pid_m * (IntV.ofNat 128 : I32) + tl.arange 0 128
          let offs_cn : Tensor [256] I32 := pid_n * (IntV.ofNat 256 : I32) + tl.arange 0 256
          let c_ptrs := d.c.ptr 0 + (IntV.ofNat d.stride_cm : I32) * offs_cm.col 256 +
            (IntV.ofNat d.stride_cn : I32) * offs_cn.row 128
          let c_mask := ((offs_cm.col 256).ltS (IntV.ofNat d.M : I32)).and
            ((offs_cn.row 128).ltS (IntV.ofNat d.N : I32))
          let c : Tensor [128, 256] F16 ← tl.cvt accumulator
          tl.store c_ptrs c (some c_mask)
          return tile_id_c) := rfl

/-- The launch has `min NUM_SMS tiles` programs. -/
theorem launch_size (d : Inst) : (launch d).size = min d.NUM_SMS (tiles d) :=
  Launch.size_grid1 _ _ _

/-- Program `i` of the launch has index `i`. -/
theorem launch_pid (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = i := by
  have hi : (i : Nat) < min d.NUM_SMS (tiles d) := lt_of_lt_of_eq i.isLt (launch_size d)
  simp [Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi]

/-- Each block size of the kernel (1, 64, 128, 256) is a power of two. -/
theorem one_mem_pow2 {k : Nat} (h : k ∈ ({1, 64, 128, 256} : Finset Nat)) : ∃ e, k = 2 ^ e := by
  simp only [Finset.mem_insert, Finset.mem_singleton] at h
  rcases h with rfl | rfl | rfl | rfl
  · exact ⟨0, rfl⟩
  · exact ⟨6, rfl⟩
  · exact ⟨7, rfl⟩
  · exact ⟨8, rfl⟩

/-- The ten block shapes of the launch are Triton block shapes. -/
theorem blocks_ok :
    ∀ s ∈ ([[128], [256], [64], [128, 1], [1, 64], [128, 64], [64, 1], [1, 256], [64, 256],
      [128, 256]] : List Shape), s.TritonOk := by
  intro s hs
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals
    refine ⟨?_, by decide⟩
    intro k hk
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
    exact one_mem_pow2 (by simp only [Finset.mem_insert, Finset.mem_singleton]; omega)

/-- The grid has fewer than `2^31` programs. -/
theorem grid_lt {d : Inst} (hd : d.ok) : min d.NUM_SMS (tiles d) < 2 ^ 31 := by
  have h1 := hd.sms.1
  have h2 := hd.sms.2
  have h3 := hd.tiles_le
  have h4 : tiles d < 2 ^ 31 := by omega
  omega

/-- The launch is valid. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  have hlt : min d.NUM_SMS (tiles d) < 2 ^ 31 := grid_lt hd
  have h := Launch.valid_grid1 (n := min d.NUM_SMS (tiles d)) (fusion := true) hlt blocks_ok
  simpa [launch] using h

/-- The number of tile rows, `M / 128`. -/
def nPM (d : Inst) : Nat := d.M / 128

/-- The number of tile columns, `N / 256`. -/
def nPN (d : Inst) : Nat := d.N / 256

/-- The number of tiles. -/
def nT (d : Inst) : Nat := nPM d * nPN d

/-- The group of tile `t` (groups of `8` tile rows, the model's `_compute_pid`). -/
def gid (d : Inst) (t : Nat) : Nat := t / (8 * nPN d)

/-- The number of tile rows of the group of tile `t`. -/
def gsz (d : Inst) (t : Nat) : Nat := min (nPM d - 8 * gid d t) 8

/-- The tile row of tile `t`. -/
def pm (d : Inst) (t : Nat) : Nat := 8 * gid d t + t % gsz d t

/-- The tile column of tile `t`. -/
def pn (d : Inst) (t : Nat) : Nat := (t % (8 * nPN d)) / gsz d t

/-! ### Helpers -/

/-- `cdiv(m, k) = m / k` when `k` divides `m`. -/
theorem cdiv_eq {m k : Nat} (hk : 0 < k) (hm : m % k = 0) : (m + k - 1) / k = m / k := by
  have hmul : k * (m / k) = m := by
    have h := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hm)
    rw [Nat.mul_comm]
    exact h
  have hmul' : (m / k) * k = m := by rw [Nat.mul_comm]; exact hmul
  have hsub : m + k - 1 = (k - 1) + (m / k) * k := by omega
  rw [hsub, Nat.add_mul_div_right _ _ hk, Nat.div_eq_of_lt (by omega), Nat.zero_add]

/-- The launch's tile count is `nT d`. -/
theorem tiles_eq_ok {d : Inst} (hd : d.ok) : tiles d = nT d := by
  simp only [tiles, nT, nPM, nPN]
  rw [cdiv_eq (by norm_num : 0 < 128) hd.M_tile,
    cdiv_eq (by norm_num : 0 < 256) hd.N_tile]

/-- `tiles d` is the tile count, on a covered instance. -/
theorem tiles_eq {d : Inst} (hd : d.ok) : tiles d = nT d := tiles_eq_ok hd

/-- There is at least one tile column. -/
theorem nPN_pos {d : Inst} (hd : d.ok) : 0 < nPN d := by
  have hN : 0 < d.N := hd.N_dim.1
  have h : d.N = 256 * (d.N / 256) := by
    have hh := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hd.N_tile)
    omega
  simp only [nPN]
  omega

/-- A tile's group starts inside the tile rows. -/
theorem gid_lt {d : Inst} (hd : d.ok) {t : Nat} (ht : t < nT d) : 8 * gid d t < nPM d := by
  have hS := nPN_pos hd
  have h1 : gid d t * (8 * nPN d) ≤ t := Nat.div_mul_le_self t (8 * nPN d)
  have h1' : (8 * gid d t) * nPN d ≤ t := by
    calc (8 * gid d t) * nPN d = gid d t * (8 * nPN d) := by ring
      _ ≤ t := h1
  have h2 : t < nPM d * nPN d := by simpa only [nT] using ht
  exact Nat.lt_of_mul_lt_mul_right (lt_of_le_of_lt h1' h2)

/-- A group has at least one tile row. -/
theorem gsz_pos {d : Inst} (hd : d.ok) {t : Nat} (ht : t < nT d) : 0 < gsz d t := by
  have h := gid_lt hd ht
  simp only [gsz]
  rw [Nat.lt_min]
  omega

/-- The tile row of tile `t` is `t % gsz` modulo `8`. -/
theorem pm_mod_eight {d : Inst} (hd : d.ok) {t : Nat} (ht : t < nT d) :
    pm d t % 8 = t % gsz d t := by
  have hz8 : gsz d t ≤ 8 := by simp only [gsz]; exact Nat.min_le_right _ _
  have hmod : t % gsz d t < gsz d t := Nat.mod_lt _ (gsz_pos hd ht)
  have hr : t % gsz d t < 8 := lt_of_lt_of_le hmod hz8
  have hpm : pm d t = t % gsz d t + 8 * gid d t := by
    simp only [pm]; ring
  rw [hpm, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hr]

/-- The tile row of tile `t` divided by `8` is its group. -/
theorem pm_div_eight {d : Inst} (hd : d.ok) {t : Nat} (ht : t < nT d) :
    pm d t / 8 = gid d t := by
  have hz8 : gsz d t ≤ 8 := by simp only [gsz]; exact Nat.min_le_right _ _
  have hmod : t % gsz d t < gsz d t := Nat.mod_lt _ (gsz_pos hd ht)
  have hr : t % gsz d t < 8 := lt_of_lt_of_le hmod hz8
  have hpm : pm d t = t % gsz d t + 8 * gid d t := by
    simp only [pm]; ring
  rw [hpm, Nat.add_mul_div_left _ _ (by norm_num : 0 < 8), Nat.div_eq_of_lt hr, Nat.zero_add]

/-- `(x + a) % z = (x + a % z) % z`. -/
theorem add_mod_mod (x a z : Nat) : (x + a) % z = (x + a % z) % z := by
  conv_lhs => rw [show a = a % z + z * (a / z) from (Nat.mod_add_div a z).symm]
  rw [← Nat.add_assoc, Nat.add_mul_mod_self_left]

/-- Two residues below `z` that agree after adding `c` are equal. -/
theorem add_mod_cancel {z c x y : Nat} (hx : x < z) (hy : y < z)
    (h : (c + x) % z = (c + y) % z) : x = y := by
  have h1 : c + x ≡ c + y [MOD z] := h
  have h2 : x ≡ y [MOD z] := Nat.ModEq.add_left_cancel (Nat.ModEq.refl c) h1
  exact Nat.ModEq.eq_of_lt_of_lt h2 hx hy

/-- The trip count `tl.tripCount p n s` is `(n - p + s - 1) / s` for `p ≤ n`. -/
theorem tripCount_eq {p n s : Nat} (hpn : p ≤ n) (hs : 0 < s) :
    tl.tripCount (p : Int) (n : Int) (s : Int) = (n - p + s - 1) / s := by
  simp only [tl.tripCount]
  have h1 : ((n : Int) - (p : Int) + (s : Int) - 1) = ((n - p + s - 1 : Nat) : Int) := by
    omega
  rw [h1, ← Int.natCast_ediv, Int.toNat_natCast]

/-! ### The statements -/

/-- The tile row of a tile is a tile row. -/
theorem pm_lt {d : Inst} (hd : d.ok) {t : Nat} (ht : t < nT d) : pm d t < nPM d := by
  have hp := gsz_pos hd ht
  have hgl := gid_lt hd ht
  have hmod : t % gsz d t < gsz d t := Nat.mod_lt _ hp
  have hle : gsz d t ≤ nPM d - 8 * gid d t := by
    simp only [gsz]; exact Nat.min_le_left _ _
  simp only [pm]
  omega

/-- The tile column of a tile is a tile column. -/
theorem pn_lt {d : Inst} (hd : d.ok) {t : Nat} (ht : t < nT d) : pn d t < nPN d := by
  have hS := nPN_pos hd
  have hp := gsz_pos hd ht
  have hgl := gid_lt hd ht
  have hzle : gsz d t ≤ nPM d - 8 * gid d t := by
    simp only [gsz]; exact Nat.min_le_left _ _
  have hz8 : gsz d t ≤ 8 := by simp only [gsz]; exact Nat.min_le_right _ _
  have h2 : t < nPM d * nPN d := by simpa only [nT] using ht
  have htS : t = 8 * gid d t * nPN d + t % (8 * nPN d) := by
    have h := Nat.div_add_mod t (8 * nPN d)
    have h2 : (8 * nPN d) * (t / (8 * nPN d)) = 8 * gid d t * nPN d := by
      simp only [gid]; ring
    rw [← h2]
    exact h.symm
  have hkey : t % (8 * nPN d) < gsz d t * nPN d := by
    rcases le_total (nPM d - 8 * gid d t) 8 with hle | hle
    · have hz : gsz d t = nPM d - 8 * gid d t := by
        simp only [gsz]; exact Nat.min_eq_left hle
      rw [hz]
      have hdist : (nPM d - 8 * gid d t) * nPN d =
          nPM d * nPN d - 8 * gid d t * nPN d := by
        rw [Nat.sub_mul]
      rw [hdist]
      omega
    · have hz : gsz d t = 8 := by
        simp only [gsz]; exact Nat.min_eq_right hle
      rw [hz]
      exact Nat.mod_lt _ (by omega)
  have hdiv := Nat.div_lt_iff_lt_mul (k := gsz d t) (x := t % (8 * nPN d)) (y := nPN d) hp
  simp only [pn]
  rw [hdiv]
  simpa only [Nat.mul_comm (gsz d t) (nPN d)] using hkey

/-- Distinct tiles have distinct `(pm, pn)`. -/
theorem pid_inj {d : Inst} (hd : d.ok) {t u : Nat} (ht : t < nT d) (hu : u < nT d)
    (h : pm d t = pm d u ∧ pn d t = pn d u) : t = u := by
  obtain ⟨h1, h2⟩ := h
  have hS := nPN_pos hd
  have htS : t = gid d t * (8 * nPN d) + t % (8 * nPN d) := by
    have hh := Nat.div_add_mod t (8 * nPN d)
    rw [gid]
    rw [Nat.mul_comm]
    exact hh.symm
  have huS : u = gid d u * (8 * nPN d) + u % (8 * nPN d) := by
    have hh := Nat.div_add_mod u (8 * nPN d)
    rw [gid]
    rw [Nat.mul_comm]
    exact hh.symm
  have hg : gid d t = gid d u := by
    rw [← pm_div_eight hd ht, ← pm_div_eight hd hu, h1]
  have hz : gsz d t = gsz d u := by simp only [gsz, hg]
  have hr : t % gsz d t = u % gsz d u := by
    rw [← pm_mod_eight hd ht, ← pm_mod_eight hd hu, h1]
  have hq : (t % (8 * nPN d)) / gsz d t = (u % (8 * nPN d)) / gsz d u := by
    simp only [pn] at h2
    exact h2
  have huS' : u = gid d t * (8 * nPN d) + u % (8 * nPN d) := by
    rw [hg]; exact huS
  have hmain : t % (8 * nPN d) = u % (8 * nPN d) := by
    have hzpos : 0 < gsz d t := gsz_pos hd ht
    have hrz : t % gsz d t = u % gsz d t := by rw [hr, hz]
    have hc : (gid d t * (8 * nPN d) + t % (8 * nPN d)) % gsz d t =
        (gid d t * (8 * nPN d) + u % (8 * nPN d)) % gsz d t := by
      rw [← htS, ← huS']
      exact hrz
    have hc' : (gid d t * (8 * nPN d) + (t % (8 * nPN d)) % gsz d t) % gsz d t =
        (gid d t * (8 * nPN d) + (u % (8 * nPN d)) % gsz d t) % gsz d t := by
      rw [← add_mod_mod, ← add_mod_mod]
      exact hc
    have hres : (t % (8 * nPN d)) % gsz d t = (u % (8 * nPN d)) % gsz d t :=
      add_mod_cancel (Nat.mod_lt _ hzpos) (Nat.mod_lt _ hzpos) hc'
    have e1 : t % (8 * nPN d) =
        (t % (8 * nPN d)) / gsz d t * gsz d t + (t % (8 * nPN d)) % gsz d t := by
      have h := Nat.div_add_mod (t % (8 * nPN d)) (gsz d t)
      rw [Nat.mul_comm] at h
      exact h.symm
    have e2 : (u % (8 * nPN d)) / gsz d u * gsz d u + (u % (8 * nPN d)) % gsz d u =
        u % (8 * nPN d) := by
      have h := Nat.div_add_mod (u % (8 * nPN d)) (gsz d u)
      rw [Nat.mul_comm] at h
      exact h
    calc t % (8 * nPN d) =
        (t % (8 * nPN d)) / gsz d t * gsz d t + (t % (8 * nPN d)) % gsz d t := e1
      _ = (u % (8 * nPN d)) / gsz d u * gsz d t + (u % (8 * nPN d)) % gsz d t := by
          rw [hq, hres]
      _ = (u % (8 * nPN d)) / gsz d u * gsz d u + (u % (8 * nPN d)) % gsz d u := by
          rw [hz]
      _ = u % (8 * nPN d) := e2
  rw [htS, huS, hg, hmain]

/-- Every `(row, column)` of tiles is some tile's `(pm, pn)`. -/
theorem pid_surj {d : Inst} (hd : d.ok) {i j : Nat} (hi : i < nPM d) (hj : j < nPN d) :
    ∃ t, t < nT d ∧ pm d t = i ∧ pn d t = j := by
  classical
  let f : Fin (nT d) → Fin (nPM d) × Fin (nPN d) := fun t =>
    (⟨pm d t.val, pm_lt hd t.isLt⟩, ⟨pn d t.val, pn_lt hd t.isLt⟩)
  have hf : Function.Injective f := by
    intro a b hab
    have h1 : pm d a.val = pm d b.val := by
      have hh := congrArg (fun p : Fin (nPM d) × Fin (nPN d) => (p.1 : Nat)) hab
      simpa [f] using hh
    have h2 : pn d a.val = pn d b.val := by
      have hh := congrArg (fun p : Fin (nPM d) × Fin (nPN d) => (p.2 : Nat)) hab
      simpa [f] using hh
    exact Fin.ext (pid_inj hd a.isLt b.isLt ⟨h1, h2⟩)
  have hcard : Fintype.card (Fin (nT d)) =
      Fintype.card (Fin (nPM d) × Fin (nPN d)) := by
    rw [Fintype.card_fin, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin, nT]
  have hbij : Function.Bijective f :=
    (Fintype.bijective_iff_injective_and_card f).2 ⟨hf, hcard⟩
  obtain ⟨t, ht⟩ := hbij.2 (⟨i, hi⟩, ⟨j, hj⟩)
  refine ⟨t.val, t.isLt, ?_, ?_⟩
  · have hh := congrArg (fun p : Fin (nPM d) × Fin (nPN d) => (p.1 : Nat)) ht
    simpa [f] using hh
  · have hh := congrArg (fun p : Fin (nPM d) × Fin (nPN d) => (p.2 : Nat)) ht
    simpa [f] using hh

/-- Programs `p`, `q` reach the same tile on trips `t`, `u` only when `p = q` and `t = u`. -/
theorem tile_id_inj {d : Inst} (hd : d.ok) {p q t u : Nat} (hp : p < d.NUM_SMS) (hq : q < d.NUM_SMS)
    (h : p + t * d.NUM_SMS = q + u * d.NUM_SMS) : p = q ∧ t = u := by
  have hS := hd.sms.1
  have e1 : (p + t * d.NUM_SMS) % d.NUM_SMS = p := by
    rw [Nat.mul_comm t, Nat.add_mul_mod_self_left]
    exact Nat.mod_eq_of_lt hp
  have e2 : (q + u * d.NUM_SMS) % d.NUM_SMS = q := by
    rw [Nat.mul_comm u, Nat.add_mul_mod_self_left]
    exact Nat.mod_eq_of_lt hq
  have hpq : p = q := by rw [← e1, ← e2, h]
  subst hpq
  have hmul : t * d.NUM_SMS = u * d.NUM_SMS := by omega
  exact ⟨rfl, Nat.eq_of_mul_eq_mul_right hS hmul⟩

/-- Trip `t` of program `p` is a tile that exists. -/
theorem trip_lt {d : Inst} (hd : d.ok) {p : Nat} (hp : p ≤ nT d) {t : Nat}
    (ht : t < tl.tripCount (p : Int) (nT d : Int) (d.NUM_SMS : Int)) :
    p + t * d.NUM_SMS < nT d := by
  have hS := hd.sms.1
  rw [tripCount_eq hp hS] at ht
  have hle : t + 1 ≤ (nT d - p + d.NUM_SMS - 1) / d.NUM_SMS := ht
  have hmul : t * d.NUM_SMS + d.NUM_SMS ≤ nT d - p + d.NUM_SMS - 1 := by
    have h := (Nat.le_div_iff_mul_le hS).1 hle
    rwa [Nat.add_mul, Nat.one_mul] at h
  omega

/-- Every tile is a tile of some trip of some program. -/
theorem exists_trip {d : Inst} (hd : d.ok) {q : Nat} (hq : q < nT d) :
    ∃ p t, p < min d.NUM_SMS (tiles d) ∧
      t < tl.tripCount (p : Int) (nT d : Int) (d.NUM_SMS : Int) ∧
      p + t * d.NUM_SMS = q := by
  have hS := hd.sms.1
  refine ⟨q % d.NUM_SMS, q / d.NUM_SMS, ?_, ?_, ?_⟩
  · rw [tiles_eq_ok hd, Nat.lt_min]
    constructor
    · exact Nat.mod_lt _ hS
    · have h1 : q % d.NUM_SMS ≤ q := Nat.mod_le _ _
      omega
  · rw [tripCount_eq (by
      have h1 : q % d.NUM_SMS ≤ q := Nat.mod_le _ _
      omega) hS]
    rw [Nat.lt_div_iff_mul_lt hS]
    have hqS : q % d.NUM_SMS + d.NUM_SMS * (q / d.NUM_SMS) = q := Nat.mod_add_div q d.NUM_SMS
    rw [Nat.mul_comm (q / d.NUM_SMS) d.NUM_SMS]
    omega
  · rw [Nat.mul_comm (q / d.NUM_SMS) d.NUM_SMS]
    exact Nat.mod_add_div q d.NUM_SMS

/-- `range(0, n, 1)` has `n` trips. -/
theorem tripCount_zero_one (n : Nat) :
    tl.tripCount (0 : Int) (n : Int) (1 : Int) = n := by
  have h : tl.tripCount (0 : Int) (n : Int) (1 : Int) = (n - 0 + 1 - 1) / 1 :=
    tripCount_eq (p := 0) (n := n) (s := 1) (Nat.zero_le n) (by norm_num)
  rw [h]
  omega

/-- The elements of `c`'s storage of the tile of tile index `t`: rows `128·pm t + i'`,
columns `256·pn t + j'`. -/
def tileOf (d : Inst) (t : Nat) : Set Nat :=
  {e | ∃ i' j', i' < 128 ∧ j' < 256 ∧
    e = (pm d t * 128 + i') * d.stride_cm + (pn d t * 256 + j') * d.stride_cn}

/-- The bytes of the tile of tile index `t`. -/
def tileSet (d : Inst) (t : Nat) : Set Int := d.c.elems (tileOf d t)

/-- The reads of program `i`: the matrices `a` and `b`. -/
def Rd (d : Inst) (_init : Mem) (_i : Fin (launch d).size) : Set Int :=
  d.a.elems (aIdx d) ∪ d.b.elems (bIdx d)

/-- The writes of program `i`: the `c` tiles of its outer trips. -/
def Wr (d : Inst) (_init : Mem) (i : Fin (launch d).size) : Set Int :=
  {x | ∃ t, t < tl.tripCount ((i : Nat) : Int) (nT d : Int) (d.NUM_SMS : Int) ∧
    x ∈ tileSet d ((i : Nat) + t * d.NUM_SMS)}

/-- What program `i` leaves: every `c` element of its tiles satisfies `OutSpec`. -/
def Qd (d : Inst) (init : Mem) (i : Fin (launch d).size) (m : Mem) : Prop :=
  ∀ t, t < tl.tripCount ((i : Nat) : Int) (nT d : Int) (d.NUM_SMS : Int) →
    ∀ a b : Nat, a < 128 → b < 256 → OutSpec d init m
      (pm d ((i : Nat) + t * d.NUM_SMS) * 128 + a) (pn d ((i : Nat) + t * d.NUM_SMS) * 256 + b)

/-! ### A tile is a piece of the matrix `c` -/

/-- `M` is made of whole tiles. -/
theorem M_eq {d : Inst} (hd : d.ok) : d.M = 128 * nPM d := by
  have h := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hd.M_tile)
  simp only [nPM]
  omega

/-- `N` is made of whole tiles. -/
theorem N_eq {d : Inst} (hd : d.ok) : d.N = 256 * nPN d := by
  have h := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hd.N_tile)
  simp only [nPN]
  omega

/-- A tile lies inside the matrix `c`. -/
theorem tileOf_sub_cIdx {d : Inst} (hd : d.ok) {t : Nat} (ht : t < nT d) :
    tileOf d t ⊆ cIdx d := by
  rintro e ⟨i', j', hi', hj', rfl⟩
  have hp := pm_lt hd ht
  have hq := pn_lt hd ht
  refine ⟨pm d t * 128 + i', pn d t * 256 + j', ?_, ?_, rfl⟩
  · have := M_eq hd; omega
  · have := N_eq hd; omega

/-- Distinct tile indices have disjoint tiles. -/
theorem tileOf_disjoint {d : Inst} (hd : d.ok) {t u : Nat} (ht : t < nT d) (hu : u < nT d)
    (hne : t ≠ u) : Disjoint (tileOf d t) (tileOf d u) := by
  rw [Set.disjoint_left]
  rintro e ⟨i', j', hi', hj', he⟩ ⟨i'', j'', hi'', hj'', he'⟩
  have hp := pm_lt hd ht
  have hq := pn_lt hd ht
  have hp' := pm_lt hd hu
  have hq' := pn_lt hd hu
  have hM := M_eq hd
  have hN := N_eq hd
  have hrow : pm d t * 128 + i' < d.M := by omega
  have hcol : pn d t * 256 + j' < d.N := by omega
  have hrow' : pm d u * 128 + i'' < d.M := by omega
  have hcol' : pn d u * 256 + j'' < d.N := by omega
  obtain ⟨h1, h2⟩ := hd.c_inj _ _ _ _ hrow hcol hrow' hcol' (he.symm.trans he')
  have hpm : pm d t = pm d u := by omega
  have hpn : pn d t = pn d u := by omega
  exact hne (pid_inj hd ht hu ⟨hpm, hpn⟩)

/-- Distinct tile indices have disjoint byte sets. -/
theorem tileSet_disjoint {d : Inst} (hd : d.ok) {t u : Nat} (ht : t < nT d) (hu : u < nT d)
    (hne : t ≠ u) : Disjoint (tileSet d t) (tileSet d u) :=
  Buf.elems_disjoint _ (tileOf_disjoint hd ht hu hne)

/-! ### The footprints against each other -/

/-- A program's writes are inside the matrix `c`. -/
theorem Wr_sub_cIdx {d : Inst} (hd : d.ok) {init : Mem} {i : Fin (launch d).size} :
    Wr d init i ⊆ d.c.elems (cIdx d) := by
  rintro x ⟨t, ht, hx⟩
  have hi : (i : Nat) < min d.NUM_SMS (tiles d) := lt_of_lt_of_eq i.isLt (launch_size d)
  have hp : (i : Nat) ≤ nT d := by
    have h1 : (i : Nat) < tiles d := by omega
    have h2 : tiles d = nT d := tiles_eq hd
    omega
  have hlt : (i : Nat) + t * d.NUM_SMS < nT d := trip_lt hd hp ht
  exact Buf.elems_mono d.c (tileOf_sub_cIdx hd hlt) hx

/-- A program's writes share no byte with its reads. -/
theorem Wr_disjoint_Rd {d : Inst} (hd : d.ok) {init : Mem} {i : Fin (launch d).size} :
    Disjoint (Wr d init i) (Rd d init i) := by
  rw [Set.disjoint_left]
  rintro x hx (hxa | hxb)
  · exact Set.disjoint_left.1 hd.c_a (Wr_sub_cIdx hd hx) hxa
  · exact Set.disjoint_left.1 hd.c_b (Wr_sub_cIdx hd hx) hxb

/-- Two programs' writes share no byte. -/
theorem Wr_disjoint {d : Inst} (hd : d.ok) {init : Mem} {i j : Fin (launch d).size}
    (hij : i ≠ j) : Disjoint (Wr d init i) (Wr d init j) := by
  rw [Set.disjoint_left]
  rintro x ⟨t, ht, hx⟩ ⟨u, hu, hx'⟩
  have hi : (i : Nat) < min d.NUM_SMS (tiles d) := lt_of_lt_of_eq i.isLt (launch_size d)
  have hj : (j : Nat) < min d.NUM_SMS (tiles d) := lt_of_lt_of_eq j.isLt (launch_size d)
  have hiS : (i : Nat) < d.NUM_SMS := by omega
  have hjS : (j : Nat) < d.NUM_SMS := by omega
  have hle : (i : Nat) ≤ nT d := by
    have h1 : (i : Nat) < tiles d := by omega
    have h2 : tiles d = nT d := tiles_eq hd
    omega
  have hle' : (j : Nat) ≤ nT d := by
    have h1 : (j : Nat) < tiles d := by omega
    have h2 : tiles d = nT d := tiles_eq hd
    omega
  have hlt : (i : Nat) + t * d.NUM_SMS < nT d := trip_lt hd hle ht
  have hlt' : (j : Nat) + u * d.NUM_SMS < nT d := trip_lt hd hle' hu
  by_cases hEq : (i : Nat) + t * d.NUM_SMS = (j : Nat) + u * d.NUM_SMS
  · exact hij (Fin.ext (tile_id_inj hd hiS hjS hEq).1)
  · exact (Set.disjoint_left.1 (tileSet_disjoint hd hlt hlt' hEq)) hx hx'

/-- Two programs' writes share no byte with each other's reads. -/
theorem Wr_disjoint_Rd' {d : Inst} (hd : d.ok) {init : Mem} {i j : Fin (launch d).size} :
    Disjoint (Wr d init i) (Rd d init j) := by
  rw [Set.disjoint_left]
  rintro x hx (hxa | hxb)
  · exact Set.disjoint_left.1 hd.c_a (Wr_sub_cIdx hd hx) hxa
  · exact Set.disjoint_left.1 hd.c_b (Wr_sub_cIdx hd hx) hxb

/-! ### Every element of the matrix is in some tile -/

/-- Every element of `c` is in some tile. -/
theorem exists_tile {d : Inst} (hd : d.ok) {i j : Nat} (hi : i < d.M) (hj : j < d.N) :
    ∃ t, t < nT d ∧ i * d.stride_cm + j * d.stride_cn ∈ tileOf d t := by
  have hM := M_eq hd
  have hN := N_eq hd
  have hi' : i / 128 < nPM d := by
    rw [Nat.div_lt_iff_lt_mul (by norm_num : 0 < 128)]
    omega
  have hj' : j / 256 < nPN d := by
    rw [Nat.div_lt_iff_lt_mul (by norm_num : 0 < 256)]
    omega
  obtain ⟨t, ht, hpm, hpn⟩ := pid_surj hd hi' hj'
  refine ⟨t, ht, i % 128, j % 256, Nat.mod_lt _ (by norm_num), Nat.mod_lt _ (by norm_num), ?_⟩
  have hi2 : i = pm d t * 128 + i % 128 := by
    rw [hpm]; omega
  have hj2 : j = pn d t * 256 + j % 256 := by
    rw [hpn]; omega
  rw [show i * d.stride_cm + j * d.stride_cn =
      (pm d t * 128 + i % 128) * d.stride_cm + (pn d t * 256 + j % 256) * d.stride_cn from by
    rw [← hi2, ← hj2]]

/-- One step of the reduction (source lines 339–345): load a `128 × 64` block of `a` and a
`64 × 256` block of `b`, and accumulate their product into `accumulator`. -/
def innerLoop (d : Inst) (offs_am : Tensor [128] I32) (offs_bn : Tensor [256] I32)
    (offs_k_for_mask : Tensor [64] I32) (ki : I32) (accumulator : Tensor [128, 256] F32) :
    Prog (Tensor [128, 256] F32) := do
  let offs_k : Tensor [64] I32 := ki * (IntV.ofNat 64 : I32) + tl.arange 0 64
  let a_ptrs := d.a.ptr 0 +
    (offs_am.col 64 * (IntV.ofNat d.stride_am : I32) + offs_k.row 128 * (IntV.ofNat d.stride_ak : I32))
  let b_ptrs := d.b.ptr 0 +
    (offs_k.col 256 * (IntV.ofNat d.stride_bk : I32) + offs_bn.row 64 * (IntV.ofNat d.stride_bn : I32))
  let a ← tl.load a_ptrs
    (some ((offs_k_for_mask.row 128).ltS ((IntV.ofNat d.K : I32) - ki * (IntV.ofNat 64 : I32))))
    (some (← tl.convert .f32 .f16 (tl.full [128, 64] (FloatV.litF32 (FloatV.ofPat 0)))))
  let b ← tl.load b_ptrs
    (some ((offs_k_for_mask.col 256).ltS ((IntV.ofNat d.K : I32) - ki * (IntV.ofNat 64 : I32))))
    (some (← tl.convert .f32 .f16 (tl.full [64, 256] (FloatV.litF32 (FloatV.ofPat 0)))))
  tl.dot { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }
    a b accumulator

/-- One trip of the outer loop (source lines 327–357): the `c` tile of `tile_id`, computed and
stored; the trip's result is `tile_id_c + NUM_SMS`, the state the next trip continues from. -/
def trip (d : Inst) (offs_k_for_mask : Tensor [64] I32) (num_pid_in_group num_pid_m kt : I32)
    (tile_id tile_id_c : I32) : Prog I32 := do
  let (pid_m, pid_n) ← compute_pid tile_id num_pid_in_group num_pid_m 8
  let start_m := pid_m * (IntV.ofNat 128 : I32)
  let start_n := pid_n * (IntV.ofNat 256 : I32)
  let offs_am : Tensor [128] I32 := start_m + tl.arange 0 128
  let offs_bn : Tensor [256] I32 := start_n + tl.arange 0 256
  let offs_am := tl.where (offs_am.ltS (IntV.ofNat d.M : I32)) offs_am
    (tl.full [128] (IntV.ofInt 0 : I32))
  let offs_bn := tl.where (offs_bn.ltS (IntV.ofNat d.N : I32)) offs_bn
    (tl.full [256] (IntV.ofInt 0 : I32))
  let offs_am ← tl.max_contiguous_multiple_of offs_am 128 128
  let offs_bn ← tl.max_contiguous_multiple_of offs_bn 256 256
  let accumulator := tl.zeros [128, 256] .f32
  let accumulator ← tl.forRange (IntV.ofInt 0 : I32) kt (IntV.ofInt 1 : I32) accumulator
      fun ki accumulator => innerLoop d offs_am offs_bn offs_k_for_mask ki accumulator
  let tile_id_c := tile_id_c + (IntV.ofNat d.NUM_SMS : I32)
  let (pid_m, pid_n) ← compute_pid tile_id_c num_pid_in_group num_pid_m 8
  let offs_cm : Tensor [128] I32 := pid_m * (IntV.ofNat 128 : I32) + tl.arange 0 128
  let offs_cn : Tensor [256] I32 := pid_n * (IntV.ofNat 256 : I32) + tl.arange 0 256
  let c_ptrs := d.c.ptr 0 + (IntV.ofNat d.stride_cm : I32) * offs_cm.col 256 +
    (IntV.ofNat d.stride_cn : I32) * offs_cn.row 128
  let c_mask := ((offs_cm.col 256).ltS (IntV.ofNat d.M : I32)).and
    ((offs_cn.row 128).ltS (IntV.ofNat d.N : I32))
  let c : Tensor [128, 256] F16 ← tl.cvt accumulator
  tl.store c_ptrs c (some c_mask)
  return tile_id_c

/-- **The model's body** is the setup, then `trip` at each trip of the outer loop. -/
theorem body_trip (d : Inst) (ctx : Ctx) :
    (model d).body ctx =
      (do
        let start_pid := tl.program_id ctx 0
        let num_pid_m := (← tl.idiv (Tensor.scalar ((IntV.ofNat d.M : I32) + (IntV.ofNat 127 : I32)))
          (Tensor.scalar (IntV.ofNat 128 : I32))).item
        let num_pid_n := (← tl.idiv (Tensor.scalar ((IntV.ofNat d.N : I32) + (IntV.ofNat 255 : I32)))
          (Tensor.scalar (IntV.ofNat 256 : I32))).item
        let k_tiles := (← tl.idiv (Tensor.scalar ((IntV.ofNat d.K : I32) + (IntV.ofNat 63 : I32)))
          (Tensor.scalar (IntV.ofNat 64 : I32))).item
        let num_tiles := num_pid_m * num_pid_n
        let tile_id_c := start_pid - (IntV.ofNat d.NUM_SMS : I32)
        let offs_k_for_mask : Tensor [64] I32 := tl.arange 0 64
        let num_pid_in_group := (IntV.ofNat 8 : I32) * num_pid_n
        let _tile_id_c ← tl.forRange start_pid num_tiles (IntV.ofNat d.NUM_SMS : I32) tile_id_c
            fun tile_id tile_id_c =>
              trip d offs_k_for_mask num_pid_in_group num_pid_m k_tiles tile_id tile_id_c
        pure ()) := rfl

end Claims.PersistentMatmul

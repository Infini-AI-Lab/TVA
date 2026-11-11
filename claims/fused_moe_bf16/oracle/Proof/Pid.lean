import TVA.Proof
import Claim

/-!
# The launch and the grouped tile map

The grid of `cdiv(EM, BM) * cdiv(N, BN)` programs, the grouped order of lines 198-206 as plain
arithmetic (`Grp`: bijective onto the tile grid), the launch facts, and `safe_pidPrefix`: the
kernel's first lines (`pid`, `num_pid_*`, `group_*`) leave `pid_m`, `pid_n` as `int32` scalars of the
values `pidM`, `pidN`.
-/

namespace Grp

/-- first row tile of the group of `p` -/
def fp (G n p : Nat) : Nat := p / (G * n) * G
/-- `group_size_m`: the rows left in the last group, at most `G`. -/
def gs (G m n p : Nat) : Nat := min (m - fp G n p) G
/-- The row tile of program `p`. -/
def pM (G m n p : Nat) : Nat := fp G n p + (p % (G * n)) % gs G m n p
/-- The column tile of program `p`. -/
def pN (G m n p : Nat) : Nat := (p % (G * n)) / gs G m n p

/-- The first row tile of the group of a program of the grid is below the row tile count. -/
theorem fp_lt {G m n p : Nat} (_hG : 0 < G) (hp : p < m * n) : fp G n p < m := by
  have hn : 0 < n := by
    rcases Nat.eq_zero_or_pos n with h | h
    · subst h; simp at hp
    · exact h
  have h1 : p / (G * n) * (G * n) ≤ p := Nat.div_mul_le_self p (G * n)
  by_contra hc
  have h2 : m ≤ fp G n p := not_lt.mp hc
  have h3 : m * n ≤ fp G n p * n := Nat.mul_le_mul_right n h2
  have h4 : fp G n p * n = p / (G * n) * (G * n) := by unfold fp; ring
  omega

/-- The group size is positive on the grid. -/
theorem gs_pos {G m n p : Nat} (hG : 0 < G) (hp : p < m * n) : 0 < gs G m n p := by
  have := fp_lt hG hp
  unfold gs
  exact Nat.lt_min.mpr ⟨Nat.sub_pos_of_lt this, hG⟩

/-- The group's rows fit in the row tiles. -/
theorem fp_add_gs {G m n p : Nat} (hG : 0 < G) (hp : p < m * n) : fp G n p + gs G m n p ≤ m := by
  have := fp_lt hG hp
  have h2 : gs G m n p ≤ m - fp G n p := Nat.min_le_left _ _
  omega

/-- A group has at most `G` rows. -/
theorem gs_le {G m n p : Nat} : gs G m n p ≤ G := Nat.min_le_right _ _

/-- The row tile is below the row tile count. -/
theorem pM_lt {G m n p : Nat} (hG : 0 < G) (hp : p < m * n) : pM G m n p < m := by
  have h1 := gs_pos hG hp
  have h2 := fp_add_gs hG hp
  have h3 : p % (G * n) % gs G m n p < gs G m n p := Nat.mod_lt _ h1
  unfold pM
  omega

/-- The offset inside the group is below `group_size_m * n`. -/
theorem mod_lt_gs {G m n p : Nat} (hG : 0 < G) (hp : p < m * n) : p % (G * n) < gs G m n p * n := by
  have hn : 0 < n := by
    rcases Nat.eq_zero_or_pos n with h | h
    · subst h; simp at hp
    · exact h
  have hGn : 0 < G * n := Nat.mul_pos hG hn
  have hdm := Nat.div_add_mod p (G * n)
  have hml : p % (G * n) < G * n := Nat.mod_lt p hGn
  rcases Nat.lt_or_ge (m - fp G n p) G with hpart | hfull
  · have hgs : gs G m n p = m - fp G n p := Nat.min_eq_left (le_of_lt hpart)
    have hf := fp_lt hG hp
    have hsum : fp G n p + gs G m n p = m := by rw [hgs]; omega
    have hexp : m * n = G * n * (p / (G * n)) + gs G m n p * n := by
      have e : m * n = (fp G n p + gs G m n p) * n := by rw [hsum]
      rw [e, Nat.add_mul]; unfold fp; ring
    omega
  · have hgs : gs G m n p = G := Nat.min_eq_right hfull
    rw [hgs]; exact hml

/-- The column tile is below the column tile count. -/
theorem pN_lt {G m n p : Nat} (hG : 0 < G) (hp : p < m * n) : pN G m n p < n :=
  Nat.div_lt_of_lt_mul (mod_lt_gs hG hp)

/-- `first_pid_m` is `G` times the group number. -/
theorem fp_mul (G n p : Nat) : fp G n p = G * (p / (G * n)) := by unfold fp; exact Nat.mul_comm _ _

/-- The row tile stays below `first_pid_m + G`. -/
theorem pM_lt_fp {G m n p : Nat} (hG : 0 < G) (hp : p < m * n) : pM G m n p < fp G n p + G := by
  have h1 := gs_pos hG hp
  have h3 : p % (G * n) % gs G m n p < gs G m n p := Nat.mod_lt _ h1
  have h4 : gs G m n p ≤ G := gs_le
  unfold pM
  omega

/-- The row tile lies in the group of `p`. -/
theorem pM_div {G m n p : Nat} (hG : 0 < G) (hp : p < m * n) : pM G m n p / G = p / (G * n) := by
  have h1 := pM_lt_fp hG hp
  have h2 : fp G n p ≤ pM G m n p := by unfold pM; omega
  have e : pM G m n p = (pM G m n p - fp G n p) + fp G n p := by omega
  have f : fp G n p = G * (p / (G * n)) := fp_mul G n p
  rw [e, f, Nat.add_mul_div_left _ _ hG, Nat.div_eq_of_lt (by omega)]; simp

/-- The grouped tile map is one-to-one on the grid. -/
theorem inj {G m n p q : Nat} (hG : 0 < G) (hp : p < m * n) (hq : q < m * n)
    (h1 : pM G m n p = pM G m n q) (h2 : pN G m n p = pN G m n q) : p = q := by
  have hg : p / (G * n) = q / (G * n) := by rw [← pM_div hG hp, ← pM_div hG hq, h1]
  have hf : fp G n p = fp G n q := by unfold fp; rw [hg]
  have hgs : gs G m n p = gs G m n q := by unfold gs; rw [hf]
  have hm : p % (G * n) % gs G m n p = q % (G * n) % gs G m n q := by
    unfold pM at h1; omega
  have hr : p % (G * n) = q % (G * n) := by
    have e1 := Nat.div_add_mod (p % (G * n)) (gs G m n p)
    have e2 := Nat.div_add_mod (q % (G * n)) (gs G m n q)
    unfold pN at h2
    rw [← e1, ← e2, h2, hm, hgs]
  have e1 := Nat.div_add_mod p (G * n)
  have e2 := Nat.div_add_mod q (G * n)
  rw [← e1, ← e2, hr, hg]

/-- The grouped tile map is onto the tile grid. -/
theorem surj {G m n a b : Nat} (hG : 0 < G) (ha : a < m) (hb : b < n) :
    ∃ p, p < m * n ∧ pM G m n p = a ∧ pN G m n p = b := by
  have hn : 0 < n := by omega
  have hGn : 0 < G * n := Nat.mul_pos hG hn
  obtain ⟨g, hgdef⟩ : ∃ g, g = a / G := ⟨_, rfl⟩
  obtain ⟨f, hf⟩ : ∃ f, f = G * g := ⟨_, rfl⟩
  have hdm := Nat.div_add_mod a G
  have hmod := Nat.mod_lt a hG
  rw [← hgdef] at hdm
  have hfa : f ≤ a := by omega
  have hrG : a - f < G := by omega
  set gsz := min (m - f) G with hgsz
  have hgsz_le : gsz ≤ G := Nat.min_le_right _ _
  have hgsz_le2 : gsz ≤ m - f := Nat.min_le_left _ _
  have hgsz_pos : 0 < gsz := Nat.lt_min.mpr ⟨by omega, hG⟩
  have hrgs : a - f < gsz := Nat.lt_min.mpr ⟨by omega, hrG⟩
  set mm := (a - f) + gsz * b with hmm
  have hmm_lt : mm < gsz * n := by
    have h2 : gsz * (b + 1) ≤ gsz * n := Nat.mul_le_mul_left gsz (by omega)
    rw [Nat.mul_add, Nat.mul_one] at h2
    omega
  have hmm_lt' : mm < G * n := lt_of_lt_of_le hmm_lt (Nat.mul_le_mul_right n hgsz_le)
  set q := G * n * g + mm with hq
  have hqdiv : q / (G * n) = g := by
    have : q = mm + G * n * g := by omega
    rw [this, Nat.add_mul_div_left mm g hGn, Nat.div_eq_of_lt hmm_lt', Nat.zero_add]
  have hqmod : q % (G * n) = mm := by
    have h := Nat.div_add_mod q (G * n)
    rw [hqdiv] at h; omega
  have hfp : fp G n q = f := by unfold fp; rw [hqdiv]; rw [hf]; exact Nat.mul_comm g G
  have hgs : gs G m n q = gsz := by unfold gs; rw [hfp]
  have hmodm : mm % gsz = a - f := by
    rw [hmm, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hrgs]
  have hdivm : mm / gsz = b := by
    rw [hmm, Nat.add_mul_div_left _ b hgsz_pos, Nat.div_eq_of_lt hrgs, Nat.zero_add]
  refine ⟨q, ?_, ?_, ?_⟩
  · have h3 : (f + gsz) * n ≤ m * n := Nat.mul_le_mul_right n (by omega)
    have h4 : (f + gsz) * n = G * n * g + gsz * n := by rw [hf]; ring
    omega
  · unfold pM; rw [hfp, hqmod, hgs, hmodm]; omega
  · unfold pN; rw [hqmod, hgs, hdivm]

end Grp

namespace Claims.FusedMoe

open TVA

/-- `cdiv(EM, BM)`, the number of row tiles (blocks of the assignment). -/
def npm (d : Inst) : Nat := (d.EM + d.BM - 1) / d.BM

/-- `cdiv(N, BN)`, the number of column tiles. -/
def npn (d : Inst) : Nat := (d.N + d.BN - 1) / d.BN

/-- The number of programs. -/
def grid (d : Inst) : Nat := npm d * npn d

/-- The row tile (block of the assignment) of program `p`. -/
def pidM (d : Inst) (p : Nat) : Nat := Grp.pM d.GM (npm d) (npn d) p

/-- The column tile of program `p`. -/
def pidN (d : Inst) (p : Nat) : Nat := Grp.pN d.GM (npm d) (npn d) p

/-- The group of program `p`, `pid // (GROUP_SIZE_M * num_pid_n)`. -/
def grp (d : Inst) (p : Nat) : Nat := p / (d.GM * npn d)

/-- `group_size_m` of program `p`. -/
def gsize (d : Inst) (p : Nat) : Nat := Grp.gs d.GM (npm d) (npn d) p

/-- The tile sizes and group size the hosts pick, as small numerals. -/
theorem cfg_facts {d : Inst} (hd : d.ok) :
    (d.BM = 16 ∨ d.BM = 32 ∨ d.BM = 64 ∨ d.BM = 128) ∧
    (d.BN = 32 ∨ d.BN = 64 ∨ d.BN = 128 ∨ d.BN = 256) ∧
    (d.BK = 64 ∨ d.BK = 128 ∨ d.BK = 256) ∧ 0 < d.GM := by
  have h : ∀ x ∈ cfgs, (x.1 = 16 ∨ x.1 = 32 ∨ x.1 = 64 ∨ x.1 = 128) ∧
      (x.2.1 = 32 ∨ x.2.1 = 64 ∨ x.2.1 = 128 ∨ x.2.1 = 256) ∧
      (x.2.2.1 = 64 ∨ x.2.2.1 = 128 ∨ x.2.2.1 = 256) ∧ 0 < x.2.2.2 := by decide
  exact h _ hd.cfg

/-- `GROUP_SIZE_M` is at most `64`. -/
theorem gm_le {d : Inst} (hd : d.ok) : d.GM ≤ 64 := by
  have h : ∀ x ∈ cfgs, x.2.2.2 ≤ 64 := by decide
  exact h _ hd.cfg

/-- The block sizes the hosts pick are powers of two. -/
theorem pow2_of_cfg {k : Nat} (h : k = 1 ∨ k = 16 ∨ k = 32 ∨ k = 64 ∨ k = 128 ∨ k = 256) :
    ∃ e, k = 2 ^ e := by
  rcases h with rfl | rfl | rfl | rfl | rfl | rfl
  · exact ⟨0, by norm_num⟩
  · exact ⟨4, by norm_num⟩
  · exact ⟨5, by norm_num⟩
  · exact ⟨6, by norm_num⟩
  · exact ⟨7, by norm_num⟩
  · exact ⟨8, by norm_num⟩

/-- The launch has `grid d` programs. -/
theorem launch_size (d : Inst) : (launch d).size = grid d := by
  have h : (launch d).size = (d.EM + d.BM - 1) / d.BM * ((d.N + d.BN - 1) / d.BN) :=
    Launch.size_grid1 _ _ _
  rw [h]
  rfl

/-- Program `i` has `tl.program_id(0) = i`. -/
theorem launch_pid (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = i := by
  have hi : (i : Nat) < (d.EM + d.BM - 1) / d.BM * ((d.N + d.BN - 1) / d.BN) :=
    lt_of_lt_of_eq i.isLt (launch_size d)
  show (i : Nat) % ((d.EM + d.BM - 1) / d.BM * ((d.N + d.BN - 1) / d.BN)) = (i : Nat)
  exact Nat.mod_eq_of_lt hi

/-- The launch is valid: fewer than `2^31` programs, every block shape a power of two. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  obtain ⟨hm, hn, hk, _⟩ := cfg_facts hd
  refine Launch.valid_grid1 hd.grid_lt ?_
  intro s hs
  have hdims : ∀ x ∈ s, x = d.BM ∨ x = d.BN ∨ x = d.BK := by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hs
    rcases hs with rfl | rfl | rfl | rfl | rfl | rfl <;> simp
  have hp2 : ∀ x ∈ s, ∃ e, x = 2 ^ e := by
    intro x hx
    apply pow2_of_cfg
    rcases hdims x hx with rfl | rfl | rfl <;> omega
  refine ⟨hp2, ?_⟩
  have hle : d.BM ≤ 256 ∧ d.BN ≤ 256 ∧ d.BK ≤ 256 := by omega
  obtain ⟨h1, h2, h3⟩ := hle
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp only [List.prod_cons, List.prod_nil, mul_one] <;> nlinarith

/-! ## The pid opening -/

theorem s32 {x : Int} (h1 : -(2 ^ 31 : Int) ≤ x) (h2 : x < 2 ^ 31) : IntV.SFits 32 x := by
  have e : ((2 : Int) ^ (32 - 1)) = 2 ^ 31 := by norm_num
  unfold IntV.SFits
  rw [e]
  exact ⟨h1, h2⟩

/-- A natural number below `2^31` fits `int32`. -/
theorem sfits_nat {n : Nat} (h : n < 2 ^ 31) : IntV.SFits 32 (n : Int) := s32 (by omega) (by omega)

/-- A product of naturals below `2^31` fits `int32`. -/
theorem sfits_mul_nat {a b : Nat} (h : a * b < 2 ^ 31) : IntV.SFits 32 ((a : Int) * (b : Int)) := by
  have e : (a : Int) * (b : Int) = ((a * b : Nat) : Int) := by push_cast; ring
  rw [e]; exact sfits_nat h

/-- A numeral below `2^31` as an `int32`. -/
theorem i32_ofNat {n : Nat} (h : n < 2 ^ 31) : (IntV.ofNat n : I32).toInt = (n : Int) :=
  IntV.I32.toInt_ofNat h

/-- The `int32` sum of two small numerals. -/
theorem toInt_add_ofNat {a b : Nat} (hab : a + b < 2 ^ 31) :
    ((IntV.ofNat a : I32) + (IntV.ofNat b : I32)).toInt = (a : Int) + (b : Int) := by
  have hsa := i32_ofNat (n := a) (by omega)
  have hsb := i32_ofNat (n := b) (by omega)
  rw [IntV.toInt_add rfl rfl (by rw [hsa, hsb]; exact s32 (by omega) (by omega)), hsa, hsb]

/-- `x.item` of a quotient `(a + (b-1)) // b` is `cdiv(a, b)`. -/
theorem cdiv_val {a b : Nat} (hb : 0 < b) (hb31 : b < 2 ^ 31) (hab : a + b ≤ 2 ^ 31) {r : Tensor [] I32}
    (h : r.item.toInt = ((IntV.ofNat a : I32) + (IntV.ofNat (b - 1) : I32)).toInt /
      (IntV.ofNat b : I32).toInt) : r.item.toInt = (((a + b - 1) / b : Nat) : Int) := by
  rw [h, toInt_add_ofNat (a := a) (b := b - 1) (by omega), i32_ofNat (n := b) (by omega)]
  have : ((a + b - 1 : Nat) : Int) = (a : Int) + ((b - 1 : Nat) : Int) := by omega
  rw [← this, Int.natCast_div]

/-- `num_pid_m` is `cdiv(EM, BM)`. -/
theorem npm_val {d : Inst} {r : Tensor [] I32}
    (hd : d.ok) (hb : 0 < d.BM)
    (h : r.item.toInt = ((IntV.ofNat d.EM : I32) + (IntV.ofNat (d.BM - 1) : I32)).toInt /
      (IntV.ofNat d.BM : I32).toInt) : r.item.toInt = (npm d : Int) :=
  cdiv_val hb (by have := (cfg_facts hd).1; omega) (by have := hd.EM_le; omega) h

/-- `num_pid_n` is `cdiv(N, BN)`. -/
theorem npn_val {d : Inst} {r : Tensor [] I32}
    (hd : d.ok) (hb : 0 < d.BN)
    (h : r.item.toInt = ((IntV.ofNat d.N : I32) + (IntV.ofNat (d.BN - 1) : I32)).toInt /
      (IntV.ofNat d.BN : I32).toInt) : r.item.toInt = (npn d : Int) :=
  cdiv_val hb (by have := (cfg_facts hd).2.1; omega) (by have := hd.N_le; omega) h

/-- The grid has fewer than `2^31` programs. -/
theorem grid_lt' {d : Inst} (hd : d.ok) : grid d < 2 ^ 31 := hd.grid_lt

/-- A program of the grid has a positive column tile count. -/
theorem npn_pos {d : Inst} {p : Nat} (hp : p < grid d) : 0 < npn d := by
  rcases Nat.eq_zero_or_pos (npn d) with h | h
  · rw [grid, h, Nat.mul_zero] at hp; omega
  · exact h

/-- `num_pid_in_group` does not wrap. -/
theorem gm_npn_lt {d : Inst} (hd : d.ok) : d.GM * npn d < 2 ^ 31 := by
  have := hd.N_group
  rw [Nat.mul_comm] at this
  exact this

/-- `pidM p`, `pidN p` of a program of the grid are below the tile counts. -/
theorem pidM_lt {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) : pidM d p < npm d :=
  Grp.pM_lt (cfg_facts hd).2.2.2 hp

/-- The column tile of a program of the grid is below the column tile count. -/
theorem pidN_lt {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) : pidN d p < npn d :=
  Grp.pN_lt (cfg_facts hd).2.2.2 hp

/-- **The tile map is one-to-one on the grid.** -/
theorem pid_inj {d : Inst} (hd : d.ok) {p q : Nat} (hp : p < grid d) (hq : q < grid d)
    (hm : pidM d p = pidM d q) (hn : pidN d p = pidN d q) : p = q :=
  Grp.inj (cfg_facts hd).2.2.2 hp hq hm hn

/-- **The tile map is onto the tile grid.** -/
theorem pid_surj {d : Inst} (hd : d.ok) {a b : Nat} (ha : a < npm d) (hb : b < npn d) :
    ∃ p, p < grid d ∧ pidM d p = a ∧ pidN d p = b :=
  Grp.surj (cfg_facts hd).2.2.2 ha hb

/-- The kernel's first lines (198-206): `pid`, `num_pid_m`, `num_pid_n`, `num_pid_in_group`,
`group_id`, `first_pid_m`, `group_size_m`, `pid_m`, `pid_n`, then the rest `k`. -/
def pidPrefix (d : Inst) (ctx : Ctx) {α : Type} (k : I32 → I32 → Prog α) : Prog α := do
  let pid := tl.program_id ctx 0
  let num_pid_m := (← tl.idiv (Tensor.scalar ((IntV.ofNat d.EM : I32) + (IntV.ofNat (d.BM - 1) : I32)))
    (Tensor.scalar (IntV.ofNat d.BM : I32))).item
  let num_pid_n := (← tl.idiv (Tensor.scalar ((IntV.ofNat d.N : I32) + (IntV.ofNat (d.BN - 1) : I32)))
    (Tensor.scalar (IntV.ofNat d.BN : I32))).item
  let num_pid_in_group := (IntV.ofNat d.GM : I32) * num_pid_n
  let group_id := (← tl.idiv (Tensor.scalar pid) (Tensor.scalar num_pid_in_group)).item
  let first_pid_m := group_id * (IntV.ofNat d.GM : I32)
  let group_size_m := (tl.iminimum (Tensor.scalar (num_pid_m - first_pid_m))
    (Tensor.scalar (IntV.ofNat d.GM : I32))).item
  let pid_m := first_pid_m + (← tl.irem (Tensor.scalar (← tl.irem (Tensor.scalar pid)
    (Tensor.scalar num_pid_in_group)).item) (Tensor.scalar group_size_m)).item
  let pid_n := (← tl.idiv (Tensor.scalar (← tl.irem (Tensor.scalar pid)
    (Tensor.scalar num_pid_in_group)).item) (Tensor.scalar group_size_m)).item
  k pid_m pid_n

/-- The row tile count is below `2^31`. -/
theorem npm_lt {d : Inst} (hd : d.ok) : npm d < 2 ^ 31 := by
  have h1 := hd.EM_le
  have h2 := (cfg_facts hd).1
  unfold npm
  have := Nat.div_le_self (d.EM + d.BM - 1) d.BM
  omega

/-- The program number as an `int32`. -/
theorem i32_pid (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    ((IntV.ofNat (i : Nat) : I32)).toInt = (i : Int) := by
  have hi : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  exact IntV.I32.toInt_ofNat (by have := grid_lt' hd; omega)

/-- `num_pid_in_group` has the value `GM * npn`. -/
theorem gm_mul_val {d : Inst} (hd : d.ok) {rn : Tensor [] I32}
    (hn : rn.item.toInt = (npn d : Int)) :
    ((IntV.ofNat d.GM : I32) * rn.item).toInt = ((d.GM * npn d : Nat) : Int) := by
  have hgm := i32_ofNat (n := d.GM) (by have := gm_le hd; omega)
  rw [IntV.toInt_mul rfl rfl (by rw [hgm, hn]; exact sfits_mul_nat (gm_npn_lt hd)), hgm, hn]
  push_cast; ring

/-- `num_pid_in_group` is positive. -/
theorem gm_mul_pos {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) {rn : Tensor [] I32}
    (hn : rn.item.toInt = (npn d : Int)) : 0 < ((IntV.ofNat d.GM : I32) * rn.item).toInt := by
  rw [gm_mul_val hd hn]
  have := npn_pos hp
  have := (cfg_facts hd).2.2.2
  have : 0 < d.GM * npn d := Nat.mul_pos (by omega) (by omega)
  exact_mod_cast this

/-- `group_id` is the group of the program. -/
theorem group_val {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {rn : Tensor [] I32}
    (hn : rn.item.toInt = (npn d : Int)) {r : Tensor [] I32}
    (h : r.item.toInt = (IntV.ofNat (i : Nat) : I32).toInt /
      ((IntV.ofNat d.GM : I32) * rn.item).toInt) : r.item.toInt = (grp d i : Int) := by
  rw [h, i32_pid d hd i, gm_mul_val hd hn]
  unfold grp
  exact (Int.natCast_div _ _).symm

/-- `pid % num_pid_in_group` has its value. -/
theorem rem_val {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {rn : Tensor [] I32}
    (hn : rn.item.toInt = (npn d : Int)) {r : Tensor [] I32}
    (h : r.item.toInt = (IntV.ofNat (i : Nat) : I32).toInt %
      ((IntV.ofNat d.GM : I32) * rn.item).toInt) :
    r.item.toInt = (((i : Nat) % (d.GM * npn d) : Nat) : Int) := by
  rw [h, i32_pid d hd i, gm_mul_val hd hn, Int.natCast_mod]

/-- `pid % num_pid_in_group` is nonnegative. -/
theorem rem_nonneg {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {rn : Tensor [] I32}
    (hn : rn.item.toInt = (npn d : Int)) {r : Tensor [] I32}
    (h : r.item.toInt = (IntV.ofNat (i : Nat) : I32).toInt %
      ((IntV.ofNat d.GM : I32) * rn.item).toInt) : 0 ≤ r.item.toInt := by
  rw [rem_val hd i hn h]; omega

/-- `group_size_m` has the value `gsize`. -/
theorem gsize_val {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {rm rg : Tensor [] I32}
    (hm : rm.item.toInt = (npm d : Int)) (hg : rg.item.toInt = (grp d i : Int)) :
    (tl.iminimum (Tensor.scalar (rm.item - rg.item * (IntV.ofNat d.GM : I32)))
      (Tensor.scalar (IntV.ofNat d.GM : I32))).item.toInt = (gsize d i : Int) := by
  have hi : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  have hG := (cfg_facts hd).2.2.2
  have hfp : Grp.fp d.GM (npn d) i < npm d := Grp.fp_lt hG hi
  have hfpe : Grp.fp d.GM (npn d) i = grp d i * d.GM := rfl
  have hnpm := npm_lt hd
  have hgm : (IntV.ofNat d.GM : I32).toInt = (d.GM : Int) :=
    i32_ofNat (by have := gm_le hd; omega)
  rw [hfpe] at hfp
  have hmul : (rg.item * (IntV.ofNat d.GM : I32)).toInt = ((grp d i * d.GM : Nat) : Int) := by
    rw [IntV.toInt_mul rfl rfl (by rw [hg, hgm]; exact sfits_mul_nat (by omega)), hg, hgm]
    push_cast; ring
  have hsub : (rm.item - rg.item * (IntV.ofNat d.GM : I32)).toInt
      = ((npm d - grp d i * d.GM : Nat) : Int) := by
    generalize grp d i * d.GM = F at hfp hmul
    rw [IntV.toInt_sub rfl rfl (by rw [hm, hmul]; exact s32 (by omega) (by omega)), hm, hmul]
    omega
  have hmin : (IntV.min (rm.item - rg.item * (IntV.ofNat d.GM : I32))
      (IntV.ofNat d.GM : I32)).toInt = (gsize d i : Int) := by
    rw [IntV.toInt_min, hsub, hgm]
    unfold gsize Grp.gs
    rw [hfpe]
    omega
  simpa [tl.iminimum, Tensor.item, Tensor.map2, Tensor.scalar] using hmin

/-- `group_size_m` is positive. -/
theorem gsize_pos {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {rm rg : Tensor [] I32}
    (hm : rm.item.toInt = (npm d : Int)) (hg : rg.item.toInt = (grp d i : Int)) :
    0 < (tl.iminimum (Tensor.scalar (rm.item - rg.item * (IntV.ofNat d.GM : I32)))
      (Tensor.scalar (IntV.ofNat d.GM : I32))).item.toInt := by
  rw [gsize_val hd i hm hg]
  have := Grp.gs_pos (cfg_facts hd).2.2.2 (lt_of_lt_of_eq i.isLt (launch_size d))
  exact_mod_cast this

/-- The row offset inside the group has its value. -/
theorem off_val {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {rn rm rg : Tensor [] I32}
    (hn : rn.item.toInt = (npn d : Int)) (hm : rm.item.toInt = (npm d : Int))
    (hg : rg.item.toInt = (grp d i : Int)) {r4 r5 : Tensor [] I32}
    (h4 : r4.item.toInt = (IntV.ofNat (i : Nat) : I32).toInt %
      ((IntV.ofNat d.GM : I32) * rn.item).toInt)
    (h5 : r5.item.toInt = r4.item.toInt %
      (tl.iminimum (Tensor.scalar (rm.item - rg.item * (IntV.ofNat d.GM : I32)))
          (Tensor.scalar (IntV.ofNat d.GM : I32))).item.toInt) :
    r5.item.toInt = (((i : Nat) % (d.GM * npn d) % gsize d i : Nat) : Int) := by
  rw [h5, rem_val hd i hn h4, gsize_val hd i hm hg]
  norm_cast

/-- `pid_n` has the value `pidN`. -/
theorem pidn_val {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {rn rm rg : Tensor [] I32}
    (hn : rn.item.toInt = (npn d : Int)) (hm : rm.item.toInt = (npm d : Int))
    (hg : rg.item.toInt = (grp d i : Int)) {r6 r7 : Tensor [] I32}
    (h6 : r6.item.toInt = (IntV.ofNat (i : Nat) : I32).toInt %
      ((IntV.ofNat d.GM : I32) * rn.item).toInt)
    (h7 : r7.item.toInt = r6.item.toInt /
      (tl.iminimum (Tensor.scalar (rm.item - rg.item * (IntV.ofNat d.GM : I32)))
          (Tensor.scalar (IntV.ofNat d.GM : I32))).item.toInt) :
    r7.item.toInt = (pidN d i : Int) := by
  rw [h7, rem_val hd i hn h6, gsize_val hd i hm hg]
  unfold pidN Grp.pN
  rw [Int.natCast_div]
  rfl

/-- The pid opening leaves `pid_m`, `pid_n` as `int32` scalars of the values `pidM`, `pidN`. -/
theorem safe_pidPrefix {P : Platform} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {k : I32 → I32 → Prog α}
    (hk : ∀ pm pn : I32, pm.toInt = (pidM d i : Int) → pn.toInt = (pidN d i : Int) →
      Safe ((launch d).env P) R W Q m (k pm pn)) :
    Safe ((launch d).env P) R W Q m (pidPrefix d ((launch d).ctx i) k) := by
  have hi : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  obtain ⟨hBM, hBN, hBK, hGM⟩ := cfg_facts hd
  have hBM0 : 0 < d.BM := by omega
  have hBN0 : 0 < d.BN := by omega
  have hBM31 : d.BM < 2 ^ 31 := by omega
  have hBN31 : d.BN < 2 ^ 31 := by omega
  have hpid : (IntV.ofNat (i : Nat) : I32).toInt = (i : Int) := i32_pid d hd i
  unfold pidPrefix
  simp only [tl.program_id, launch_pid]
  refine safe_idiv1 (w := 32) ?_ ?_ ?_
  · rw [toInt_add_ofNat (a := d.EM) (b := d.BM - 1) (by have := hd.EM_le; omega)]; positivity
  · rw [i32_ofNat hBM31]; exact_mod_cast hBM0
  · intro r1 h1
    have hm : r1.item.toInt = (npm d : Int) := npm_val hd hBM0 h1
    refine safe_idiv1 (w := 32) ?_ ?_ ?_
    · rw [toInt_add_ofNat (a := d.N) (b := d.BN - 1) (by have := hd.N_le; omega)]; positivity
    · rw [i32_ofNat hBN31]; exact_mod_cast hBN0
    · intro r2 h2
      have hn : r2.item.toInt = (npn d : Int) := npn_val hd hBN0 h2
      have hy3 := gm_mul_pos hd hi hn
      refine safe_idiv1 (w := 32) ?_ ?_ ?_
      · rw [hpid]; exact_mod_cast Nat.zero_le (i : Nat)
      · exact hy3
      · intro r3 h3
        have hg : r3.item.toInt = (grp d i : Int) := group_val hd i hn h3
        refine safe_irem1 (w := 32) ?_ ?_ ?_
        · rw [hpid]; exact_mod_cast Nat.zero_le (i : Nat)
        · exact hy3
        · intro r4 h4
          have hx5 := rem_nonneg hd i hn h4
          have hy5 := gsize_pos hd i hm hg
          refine safe_irem1 (w := 32) ?_ ?_ ?_
          · exact hx5
          · exact hy5
          · intro r5 h5
            refine safe_irem1 (w := 32) ?_ ?_ ?_
            · rw [hpid]; exact_mod_cast Nat.zero_le (i : Nat)
            · exact hy3
            · intro r6 h6
              have hx7 := rem_nonneg hd i hn h6
              refine safe_idiv1 (w := 32) ?_ ?_ ?_
              · exact hx7
              · exact hy5
              · intro r7 h7
                have ho := off_val hd i hn hm hg h4 h5
                have hpn := pidn_val hd i hn hm hg h6 h7
                refine hk _ _ ?_ hpn
                have hG := (cfg_facts hd).2.2.2
                have hfp : Grp.fp d.GM (npn d) i < npm d := Grp.fp_lt hG hi
                have hnpm := npm_lt hd
                have hgm : (IntV.ofNat d.GM : I32).toInt = (d.GM : Int) :=
                  i32_ofNat (by have := gm_le hd; omega)
                have hfpe : Grp.fp d.GM (npn d) i = grp d i * d.GM := rfl
                rw [hfpe] at hfp
                have hmul : (r3.item * (IntV.ofNat d.GM : I32)).toInt =
                    ((grp d i * d.GM : Nat) : Int) := by
                  rw [IntV.toInt_mul rfl rfl (by rw [hg, hgm]; exact sfits_mul_nat (by omega)), hg, hgm]
                  push_cast; ring
                have hlt := Grp.pM_lt hG hi
                have hpmE : pidM d i = grp d i * d.GM + (i : Nat) % (d.GM * npn d) % gsize d i := rfl
                have hlt' : pidM d i < npm d := hlt
                rw [hpmE] at hlt'
                generalize grp d i * d.GM = F at hfp hmul hpmE hlt'
                rw [IntV.toInt_add rfl rfl (by rw [hmul, ho]; exact s32 (by omega) (by omega)),
                  hmul, ho, hpmE]
                push_cast; ring

/-- The tile counts bound the tile origins: `(pidM + 1) * BM ≤ EM + BM - 1`. -/
theorem pidM_mul_le {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) :
    (pidM d p + 1) * d.BM ≤ d.EM + d.BM - 1 := by
  have h := pidM_lt hd hp
  have hBM := (cfg_facts hd).1
  have h1 : (pidM d p + 1) * d.BM ≤ npm d * d.BM := Nat.mul_le_mul_right _ h
  have h2 : npm d * d.BM ≤ d.EM + d.BM - 1 := Nat.div_mul_le_self _ _
  omega

/-- `(pidN + 1) * BN ≤ N + BN - 1`. -/
theorem pidN_mul_le {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) :
    (pidN d p + 1) * d.BN ≤ d.N + d.BN - 1 := by
  have h := pidN_lt hd hp
  have hBN := (cfg_facts hd).2.1
  have h1 : (pidN d p + 1) * d.BN ≤ npn d * d.BN := Nat.mul_le_mul_right _ h
  have h2 : npn d * d.BN ≤ d.N + d.BN - 1 := Nat.div_mul_le_self _ _
  omega

end Claims.FusedMoe

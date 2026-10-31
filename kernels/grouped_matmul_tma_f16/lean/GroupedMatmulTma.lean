import GroupedMatmulTma.Program

/-!
# `grouped_matmul_tma_f16`: tutorial 08, `grouped_matmul_tma_kernel` on `float16`

Source: `source/08-grouped-gemm.py`. The kernel is lines 186–278 and it reads the module-level
names `triton` and `tl` (lines 32–33). Its host functions `group_gemm_tma_fn` (lines 281–326) and
`triton_tma_perf_fn` (lines 374–378) launch it with `FP8=False` on groups of `float16` matrices:
a persistent grid of `NUM_SM` programs, program `p` computing the `128 × 128` tiles of all groups
whose global index is `p` modulo `NUM_SM`.

The claim: on every covered instance, every launch that runs ends normally; each element `(i, j)`
of each `C g` is the `float16` rounding of a `float32` accumulation within an explicit tolerance
of the exact dot product of row `i` of `A g` and row `j` of `B g` (when these are finite); every
byte outside the matrices `C g` is unchanged.

* `GroupedMatmulTma.Defs`: the model, the conditions, the specification, the assumptions.
* `GroupedMatmulTma.Sched`: the schedule (which program owns which tile), the footprints, the
  invariant `ProgInv`.
* `GroupedMatmulTma.Basic`: `int32` arithmetic, descriptor lanes, one accumulation step, the last
  rounding.
* `GroupedMatmulTma.Tile`: one iteration of the `while` loop (`safe_tile`).
* `GroupedMatmulTma.Program`: one program (`safe_program`).
* Here: `main` by `TVA.meets_independent`, `consistent` and `satisfiable`.
-/

namespace Kernels.GroupedMatmulTma

open TVA

/-! ## The programs together -/

theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 (by have := hd.tiles_lt; omega) ?_
  have hpow : ∀ k : Nat, k = 64 ∨ k = 128 → ∃ e, k = 2 ^ e := by
    rintro k (rfl | rfl)
    exacts [⟨6, rfl⟩, ⟨7, rfl⟩]
  intro s hs
  simp only [List.mem_cons, List.mem_nil_iff, or_false] at hs
  rcases hs with rfl | rfl | rfl <;>
    exact ⟨fun k hk => hpow k (by simp only [List.mem_cons, List.mem_nil_iff, or_false] at hk; omega),
      by simp⟩

/-- Element `(i, j)` of `C g` lies in tile `(i / 128) · cdiv(N g, 128) + j / 128` of the group, at
lane `(i % 128, j % 128)`. -/
theorem tile_of {d : Inst} {g i j : Nat} (hi : i < d.M g) (hj : j < d.N g) :
    i / 128 * nN d g + j / 128 < tiles d g ∧ tm d g (i / 128 * nN d g + j / 128) = i / 128 ∧
      tn d g (i / 128 * nN d g + j / 128) = j / 128 := by
  have hi' : i / 128 < nM d g := by unfold nM; omega
  have hj' : j / 128 < nN d g := by unfold nN; omega
  refine ⟨?_, ?_, ?_⟩
  · rw [tiles_eq]
    calc i / 128 * nN d g + j / 128 < i / 128 * nN d g + nN d g := by omega
      _ = (i / 128 + 1) * nN d g := by ring
      _ ≤ nM d g * nN d g := Nat.mul_le_mul_right _ hi'
  · unfold tm
    rw [Nat.add_comm, Nat.mul_comm, Nat.add_mul_div_left _ _ (by omega), Nat.div_eq_of_lt hj', zero_add]
  · unfold tn
    rw [Nat.add_comm, Nat.mul_comm, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hj']

/-! ## The claim -/

theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K := by
  intro K P hA
  refine meets_independent hA.faithful (fun _ _ => rfl) (fun _ hd => launch_valid hd)
    (fun d _ _ => R d) (fun d _ i => W d i) (fun d init i m => Q d init i m) ?_ ?_ ?_ ?_
  · -- each program is safe
    intro d hd init hpre i
    exact safe_program hA.dot hA.dot_total hd hpre (lt_of_lt_of_eq i.isLt (launch_size d))
  · -- no program writes another's footprint
    intro d hd init _ i j hij
    have hij' : (i : Nat) ≠ j := fun h => hij (Fin.ext h)
    exact Disjoint.union_right (R_W hd i).symm (W_disj hd hij')
  · -- no program reads what it writes
    intro d hd init _ i
    exact R_W hd i
  · -- the programs' results give `Post`
    intro d hd init _ fin hloc hfr
    refine ⟨fun g i j hg hi hj => ?_, fun z hz => hfr z fun p hzp => ?_⟩
    · obtain ⟨ht, htm, htn⟩ := tile_of hi hj
      have hpL : (start d g + (i / 128 * nN d g + j / 128)) % d.NUM_SM < (launch d).size := by
        rw [launch_size]; exact Nat.mod_lt _ hd.SM_pos
      obtain ⟨m, hQ, hagree⟩ := hloc ⟨_, hpL⟩
      have hO := hQ g _ hg ht rfl (i % 128) (j % 128) (Nat.mod_lt _ (by norm_num))
        (Nat.mod_lt _ (by norm_num)) (by rw [htm]; omega) (by rw [htn]; omega)
      rw [htm, htn, show i / 128 * 128 + i % 128 = i by omega, show j / 128 * 128 + j % 128 = j by omega] at hO
      obtain ⟨c, hc, hrest⟩ := hO
      refine ⟨c, ?_, hrest⟩
      rw [← hc]
      refine Mem.read_congr fun x hx => hagree x (Or.inr ⟨g, hg, _, ht, rfl, ?_⟩)
      refine (d.C g).span_sub_elems (S := tileIdx d g _) ⟨i % 128, j % 128, Nat.mod_lt _ (by norm_num),
        Nat.mod_lt _ (by norm_num), by rw [htm]; omega, by rw [htn]; omega, ?_⟩ hx
      rw [htm, htn, show i / 128 * 128 + i % 128 = i by omega, show j / 128 * 128 + j % 128 = j by omega]
    · obtain ⟨g, hg, hc⟩ := W_sub d p hzp
      exact hz g hg hc

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, htot, ⟨fun d _ => hf d, Platform.sm90ref.dotF16, Platform.ref.dotTotal _⟩⟩

/-! ## A covered instance -/

/-- Memory `m` with buffer `b` holding `v e` in each element `e`. -/
def fillBuf {dt : DType} (m : Mem) (b : Buf dt) (v : Nat → dt.Val) : Mem := fun x =>
  if b.base ≤ x ∧ x < b.base + b.len * dt.bytes then
    (Math.Bytes.leBytes dt.bytes (dt.toBits (v ((x - b.base) / dt.bytes).toNat)))[((x - b.base) % dt.bytes).toNat]?
  else m x

theorem fillBuf_frame {dt : DType} (m : Mem) (b : Buf dt) (v : Nat → dt.Val) {x : Int}
    (hx : x ∉ b.bytes) : fillBuf m b v x = m x := by
  unfold fillBuf
  rw [ite_eq_right (show ¬ (b.base ≤ x ∧ x < b.base + b.len * dt.bytes) from hx)]

theorem fillBuf_present {dt : DType} {m : Mem} (hm : ∀ x, m x ≠ none) (b : Buf dt) (v : Nat → dt.Val) :
    ∀ x, fillBuf m b v x ≠ none := by
  intro x
  unfold fillBuf
  split_ifs with h
  · have hB := dt.bytes_pos
    have hlt : ((x - b.base) % dt.bytes).toNat <
        (Math.Bytes.leBytes dt.bytes (dt.toBits (v ((x - b.base) / dt.bytes).toNat))).length := by
      rw [Mem.leBytes_len]
      have h1 := Int.emod_nonneg (x - b.base) (show (dt.bytes : Int) ≠ 0 by omega)
      have h2 := Int.emod_lt_of_pos (x - b.base) (show (0 : Int) < dt.bytes by omega)
      omega
    rw [List.getElem?_eq_getElem hlt]; simp
  · exact hm x

theorem fillBuf_read {dt : DType} {m : Mem} (hm : ∀ x, m x ≠ none) {b : Buf dt} (v : Nat → dt.Val)
    (h0 : 0 ≤ b.base) (h64 : b.base + b.len * dt.bytes ≤ 2 ^ 64) (hal : b.base % dt.bytes = 0)
    {i : Nat} (hi : i < b.len) : (fillBuf m b v).read dt (b.ptr ↑i).addr = some (v i) := by
  have hw : m.Writable dt (b.ptr ↑i).addr := b.writable ⟨h0, h64, hal, fun x _ => hm x⟩ hi
  rw [← Mem.read_write_self hw (v i)]
  apply Mem.read_congr
  intro x hx
  have hx' : b.base + (i : Int) * dt.bytes ≤ x ∧ x < b.base + (i : Int) * dt.bytes + dt.bytes := hx
  have hB := dt.bytes_pos
  obtain ⟨r, hr0, hrB, rfl⟩ : ∃ r : Int, 0 ≤ r ∧ r < dt.bytes ∧ x = b.base + (i : Int) * dt.bytes + r :=
    ⟨x - (b.base + (i : Int) * dt.bytes), by omega, by omega, by ring⟩
  have hiL : ((i : Int) + 1) * dt.bytes ≤ (b.len : Int) * dt.bytes :=
    mul_le_mul_of_nonneg_right (by exact_mod_cast hi) (by positivity)
  have hc1 : b.base ≤ b.base + (i : Int) * dt.bytes + r ∧
      b.base + (i : Int) * dt.bytes + r < b.base + b.len * dt.bytes := ⟨by nlinarith, by nlinarith⟩
  have hc2 : (b.ptr ↑i).addr ≤ b.base + (i : Int) * dt.bytes + r ∧
      b.base + (i : Int) * dt.bytes + r < (b.ptr ↑i).addr + dt.bytes := by
    simp only [Buf.ptr_addr]; constructor <;> omega
  unfold fillBuf Mem.write Mem.writeBytes
  rw [ite_eq_left hc1, Mem.leBytes_len, ite_eq_left hc2]
  have hBne : (dt.bytes : Int) ≠ 0 := by omega
  have e1 : (b.base + (i : Int) * dt.bytes + r - b.base) / dt.bytes = i := by
    rw [show b.base + (i : Int) * dt.bytes + r - b.base = r + (i : Int) * dt.bytes by ring,
      Int.add_mul_ediv_right _ _ hBne, Int.ediv_eq_zero_of_lt hr0 hrB, zero_add]
  have e2 : (b.base + (i : Int) * dt.bytes + r - b.base) % dt.bytes = r := by
    rw [show b.base + (i : Int) * dt.bytes + r - b.base = r + (i : Int) * dt.bytes by ring,
      Int.add_mul_emod_self_right, Int.emod_eq_of_lt hr0 hrB]
  rw [e1, e2, Int.toNat_natCast, show b.base + (i : Int) * dt.bytes + r - (b.ptr ↑i).addr = r by
    simp only [Buf.ptr_addr]; ring]

theorem fillBuf_read_other {dt dt' : DType} (m : Mem) (b : Buf dt) (v : Nat → dt.Val) {a : Int}
    (h : a + dt'.bytes ≤ b.base ∨ b.base + b.len * dt.bytes ≤ a) :
    (fillBuf m b v).read dt' a = m.read dt' a :=
  Mem.read_congr fun x hx => fillBuf_frame m b v fun hb => by
    have hx' : a ≤ x ∧ x < a + dt'.bytes := hx
    have hb' : b.base ≤ x ∧ x < b.base + b.len * dt.bytes := hb
    omega

/-- Byte ranges that do not overlap. -/
theorem disj_range {S T : Set Int} {a1 b1 a2 b2 : Int} (hS : S ⊆ {x | a1 ≤ x ∧ x < b1})
    (hT : T ⊆ {x | a2 ≤ x ∧ x < b2}) (h : b1 ≤ a2 ∨ b2 ≤ a1) : Disjoint S T :=
  Set.disjoint_left.mpr fun x hx hy => by
    have h1 := hS hx
    have h2 := hT hy
    simp only [Set.mem_ofPred_eq] at h1 h2
    omega

/-- The test's sizes (line 337): `M = N = K = 1024, 512, 256, 128`. -/
def testSz : Nat → Nat
  | 0 => 1024
  | 1 => 512
  | 2 => 256
  | _ => 128

theorem testSz_cases (g : Nat) : testSz g = 1024 ∨ testSz g = 512 ∨ testSz g = 256 ∨ testSz g = 128 := by
  unfold testSz; split <;> simp

/-- The matrices: `A g`, `B g`, `C g` of `2^20` elements each, at 16-byte-aligned addresses
`2^24` apart. -/
def testA (g : Nat) : Buf .f16 := ⟨2 ^ 24 * ((g : Int) + 1), 2 ^ 20⟩
def testB (g : Nat) : Buf .f16 := ⟨2 ^ 24 * ((g : Int) + 5), 2 ^ 20⟩
def testC (g : Nat) : Buf .f16 := ⟨2 ^ 24 * ((g : Int) + 9), 2 ^ 20⟩

/-- The source's test instance (lines 329–357): four groups of `M = N = K = 1024, 512, 256, 128`,
row-major `A g` (`lda = K`), `B g` (the transpose, `ldb = K`) and `C g` (`ldc = N`), and
`NUM_SM = 132` (an H200's multiprocessor count). -/
def testInst : Inst where
  group_size := 4
  NUM_SM := 132
  M := testSz
  N := testSz
  K := testSz
  lda := testSz
  ldb := testSz
  ldc := testSz
  A := testA
  B := testB
  C := testC
  group_a_ptrs := ⟨12288, 4⟩
  group_b_ptrs := ⟨16384, 4⟩
  group_c_ptrs := ⟨20480, 4⟩
  group_gemm_sizes := ⟨4096, 12⟩
  g_lds := ⟨8192, 12⟩

/-- The tables' values: `M g, N g, K g` (and `lda g, ldb g, ldc g`) at elements `3g, 3g + 1,
3g + 2`; the matrices' addresses. -/
def sizeV (e : Nat) : I32 := IntV.ofNat (testSz (e / 3))
def aV (e : Nat) : I64 := IntV.ofInt (testA e).base
def bV (e : Nat) : I64 := IntV.ofInt (testB e).base
def cV (e : Nat) : I64 := IntV.ofInt (testC e).base

/-- A memory in which every byte is present, holding the tables' values. -/
def testMem : Mem :=
  fillBuf (fillBuf (fillBuf (fillBuf (fillBuf (fun _ => some 0) testInst.group_gemm_sizes sizeV)
    testInst.g_lds sizeV) testInst.group_a_ptrs aV) testInst.group_b_ptrs bV) testInst.group_c_ptrs cV

theorem testMem_present : ∀ x, testMem x ≠ none :=
  fillBuf_present (fillBuf_present (fillBuf_present (fillBuf_present (fillBuf_present
    (fun _ => Option.some_ne_none 0) _ _) _ _) _ _) _ _) _ _

theorem test_ok : testInst.ok := by
  have hle : ∀ g, testSz g ≤ 1024 := fun g => by rcases testSz_cases g with h | h | h | h <;> omega
  have hpos : ∀ g, 128 ≤ testSz g := fun g => by rcases testSz_cases g with h | h | h | h <;> omega
  have h8 : ∀ g, testSz g % 8 = 0 := fun g => by rcases testSz_cases g with h | h | h | h <;> rw [h]
  have hlen : ∀ g i k, i < testSz g → k < testSz g → i * testSz g + k < 2 ^ 20 := by
    intro g i k hi hk
    have h1 : (i + 1) * testSz g ≤ 1024 * 1024 := Nat.mul_le_mul (by have := hle g; omega) (hle g)
    nlinarith
  have hA : ∀ g, ∀ S : Set Nat, (∀ j ∈ S, j < 2 ^ 20) →
      (testA g).elems S ⊆ {x | 2 ^ 24 * ((g : Int) + 1) ≤ x ∧ x < 2 ^ 24 * ((g : Int) + 1) + 2 ^ 21} :=
    fun g S hS => ((testA g).elems_sub_bytes hS).trans fun x hx => by
      have : (testA g).base ≤ x ∧ x < (testA g).base + (testA g).len * DType.bytes .f16 := hx
      simp only [testA, DType.bytes] at this; constructor <;> omega
  have hB : ∀ g, ∀ S : Set Nat, (∀ j ∈ S, j < 2 ^ 20) →
      (testB g).elems S ⊆ {x | 2 ^ 24 * ((g : Int) + 5) ≤ x ∧ x < 2 ^ 24 * ((g : Int) + 5) + 2 ^ 21} :=
    fun g S hS => ((testB g).elems_sub_bytes hS).trans fun x hx => by
      have : (testB g).base ≤ x ∧ x < (testB g).base + (testB g).len * DType.bytes .f16 := hx
      simp only [testB, DType.bytes] at this; constructor <;> omega
  have hC : ∀ g, ∀ S : Set Nat, (∀ j ∈ S, j < 2 ^ 20) →
      (testC g).elems S ⊆ {x | 2 ^ 24 * ((g : Int) + 9) ≤ x ∧ x < 2 ^ 24 * ((g : Int) + 9) + 2 ^ 21} :=
    fun g S hS => ((testC g).elems_sub_bytes hS).trans fun x hx => by
      have : (testC g).base ≤ x ∧ x < (testC g).base + (testC g).len * DType.bytes .f16 := hx
      simp only [testC, DType.bytes] at this; constructor <;> omega
  have hT : tabEl testInst ⊆ {x | (4096 : Int) ≤ x ∧ x < 20512} := by
    have h1 : ∀ {dt : DType} (b : Buf dt) (S : Set Nat), (∀ j ∈ S, j < b.len) → 4096 ≤ b.base →
        b.base + b.len * dt.bytes ≤ 20512 → b.elems S ⊆ {x | (4096 : Int) ≤ x ∧ x < 20512} :=
      fun b S hS h1 h2 => (b.elems_sub_bytes hS).trans fun x hx => by
        have : b.base ≤ x ∧ x < b.base + b.len * _ := hx
        constructor <;> omega
    rintro x ((((hx | hx) | hx) | hx) | hx)
    · exact h1 _ _ (fun j hj => hj) (by norm_num [testInst]) (by norm_num [testInst, DType.bytes]) hx
    · exact h1 _ _ (fun j hj => hj) (by norm_num [testInst]) (by norm_num [testInst, DType.bytes]) hx
    · exact h1 _ _ (fun j hj => hj) (by norm_num [testInst]) (by norm_num [testInst, DType.bytes]) hx
    · exact h1 _ _ (fun j hj => hj) (by norm_num [testInst]) (by norm_num [testInst, DType.bytes]) hx
    · exact h1 _ _ (fun j hj => hj) (by norm_num [testInst]) (by norm_num [testInst, DType.bytes]) hx
  have haI : ∀ g, ∀ j ∈ aIdx testInst g, j < 2 ^ 20 := by
    rintro g _ ⟨i, k, hi, hk, rfl⟩; exact hlen g i k hi hk
  have hbI : ∀ g, ∀ j ∈ bIdx testInst g, j < 2 ^ 20 := by
    rintro g _ ⟨i, k, hi, hk, rfl⟩; exact hlen g i k hi hk
  have hcI : ∀ g, ∀ j ∈ cIdx testInst g, j < 2 ^ 20 := by
    rintro g _ ⟨i, k, hi, hk, rfl⟩; exact hlen g i k hi hk
  refine ⟨by norm_num [testInst], by norm_num [testInst], ?_, fun g _ => by have := hle g; simp only [testInst]; omega,
    fun g _ => by have := hle g; simp only [testInst]; omega, fun g _ => by have := hle g; simp only [testInst]; omega,
    fun g _ => ⟨by have := hle g; simp only [testInst]; omega, h8 g⟩,
    fun g _ => ⟨by have := hle g; simp only [testInst]; omega, h8 g⟩,
    fun g _ => ⟨by have := hle g; simp only [testInst]; omega, h8 g, le_rfl⟩,
    fun g _ => ⟨by simp only [testInst, testA]; omega, by simp only [testInst, testB]; omega,
      by simp only [testInst, testC]; omega⟩,
    fun g _ i k hi hk => hlen g i k hi hk, fun g _ i k hi hk => hlen g i k hi hk,
    fun g _ i k hi hk => hlen g i k hi hk, by norm_num [testInst], ?_, ?_, ?_, ?_⟩
  · simp only [start, tiles, testInst, Finset.sum_range_succ, Finset.sum_range_zero, testSz]
    norm_num
  · intro g h hg hh hgh
    have hg' : g < 4 := hg
    have hh' : h < 4 := hh
    exact disj_range (hC g _ (hcI g)) (hC h _ (hcI h)) (by omega)
  · intro g h hg hh
    have hg' : g < 4 := hg
    have hh' : h < 4 := hh
    exact ⟨disj_range (hC g _ (hcI g)) (hA h _ (haI h)) (by omega),
      disj_range (hC g _ (hcI g)) (hB h _ (hbI h)) (by omega)⟩
  · intro g hg
    exact disj_range (hC g _ (hcI g)) hT (by omega)
  · intro h hh
    exact ⟨disj_range (hA h _ (haI h)) hT (by omega), disj_range (hB h _ (hbI h)) hT (by omega)⟩

theorem test_pre : Pre testInst testMem := by
  have hv : ∀ {dt : DType} (b : Buf dt), 0 ≤ b.base → b.base + b.len * dt.bytes ≤ 2 ^ 64 →
      b.base % dt.bytes = 0 → b.Valid testMem := fun b h0 h1 h2 => ⟨h0, h1, h2, fun x _ => testMem_present x⟩
  have hm0 : ∀ x, (fun _ => some 0 : Mem) x ≠ none := fun _ => Option.some_ne_none 0
  refine ⟨hv _ (by norm_num [testInst]) (by norm_num [testInst, DType.bytes]) (by norm_num [testInst, DType.bytes]),
    hv _ (by norm_num [testInst]) (by norm_num [testInst, DType.bytes]) (by norm_num [testInst, DType.bytes]),
    hv _ (by norm_num [testInst]) (by norm_num [testInst, DType.bytes]) (by norm_num [testInst, DType.bytes]),
    hv _ (by norm_num [testInst]) (by norm_num [testInst, DType.bytes]) (by norm_num [testInst, DType.bytes]),
    hv _ (by norm_num [testInst]) (by norm_num [testInst, DType.bytes]) (by norm_num [testInst, DType.bytes]),
    fun g hg => ?_, fun g hg => ?_⟩
  · have hg' : g < 4 := hg
    refine ⟨hv _ (by simp only [testInst, testA]; omega) (by simp only [testInst, testA, DType.bytes]; omega)
        (by simp only [testInst, testA, DType.bytes]; omega),
      hv _ (by simp only [testInst, testB]; omega) (by simp only [testInst, testB, DType.bytes]; omega)
        (by simp only [testInst, testB, DType.bytes]; omega),
      hv _ (by simp only [testInst, testC]; omega) (by simp only [testInst, testC, DType.bytes]; omega)
        (by simp only [testInst, testC, DType.bytes]; omega)⟩
  · have hg' : g < 4 := hg
    have p1 := fillBuf_present hm0 testInst.group_gemm_sizes sizeV
    have p2 := fillBuf_present p1 testInst.g_lds sizeV
    have p3 := fillBuf_present p2 testInst.group_a_ptrs aV
    have p4 := fillBuf_present p3 testInst.group_b_ptrs bV
    have hsz : ∀ c, c < 3 → testMem.read .i32 (testInst.group_gemm_sizes.ptr ↑(3 * g + c)).addr =
        some (IntV.ofNat (testSz g)) := by
      intro c hc
      unfold testMem
      rw [fillBuf_read_other _ _ _ (by simp only [testInst, Buf.ptr_addr, DType.bytes]; omega),
        fillBuf_read_other _ _ _ (by simp only [testInst, Buf.ptr_addr, DType.bytes]; omega),
        fillBuf_read_other _ _ _ (by simp only [testInst, Buf.ptr_addr, DType.bytes]; omega),
        fillBuf_read_other _ _ _ (by simp only [testInst, Buf.ptr_addr, DType.bytes]; omega),
        fillBuf_read hm0 _ (by norm_num [testInst]) (by norm_num [testInst, DType.bytes])
          (by norm_num [testInst, DType.bytes]) (by simp only [testInst]; omega)]
      simp only [sizeV, show (3 * g + c) / 3 = g by omega]
    have hld : ∀ c, c < 3 → testMem.read .i32 (testInst.g_lds.ptr ↑(3 * g + c)).addr =
        some (IntV.ofNat (testSz g)) := by
      intro c hc
      unfold testMem
      rw [fillBuf_read_other _ _ _ (by simp only [testInst, Buf.ptr_addr, DType.bytes]; omega),
        fillBuf_read_other _ _ _ (by simp only [testInst, Buf.ptr_addr, DType.bytes]; omega),
        fillBuf_read_other _ _ _ (by simp only [testInst, Buf.ptr_addr, DType.bytes]; omega),
        fillBuf_read p1 _ (by norm_num [testInst]) (by norm_num [testInst, DType.bytes])
          (by norm_num [testInst, DType.bytes]) (by simp only [testInst]; omega)]
      simp only [sizeV, show (3 * g + c) / 3 = g by omega]
    have h3 := hsz 0 (by norm_num)
    have h4 := hld 0 (by norm_num)
    rw [Nat.add_zero] at h3 h4
    refine ⟨h3, hsz 1 (by norm_num), hsz 2 (by norm_num), h4, hld 1 (by norm_num), hld 2 (by norm_num), ?_, ?_, ?_⟩
    · unfold testMem
      rw [fillBuf_read_other _ _ _ (by simp only [testInst, Buf.ptr_addr, DType.bytes]; omega),
        fillBuf_read_other _ _ _ (by simp only [testInst, Buf.ptr_addr, DType.bytes]; omega),
        fillBuf_read p2 _ (by norm_num [testInst]) (by norm_num [testInst, DType.bytes])
          (by norm_num [testInst, DType.bytes]) (by simp only [testInst]; omega)]
      rfl
    · unfold testMem
      rw [fillBuf_read_other _ _ _ (by simp only [testInst, Buf.ptr_addr, DType.bytes]; omega),
        fillBuf_read p3 _ (by norm_num [testInst]) (by norm_num [testInst, DType.bytes])
          (by norm_num [testInst, DType.bytes]) (by simp only [testInst]; omega)]
      rfl
    · unfold testMem
      rw [fillBuf_read p4 _ (by norm_num [testInst]) (by norm_num [testInst, DType.bytes])
          (by norm_num [testInst, DType.bytes]) (by simp only [testInst]; omega)]
      rfl

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := ⟨testInst, testMem, test_ok, test_pre⟩

end Kernels.GroupedMatmulTma

import Claim
import TVA.Proof

/-!
# The proof of `vec_add_f32`

Program `p` of the grid owns the elements `blk d p` (its block of `BLOCK_SIZE` elements, cut at
`n`). It reads them in `x` and `y` and writes them in `output`, so the footprints of distinct
programs are disjoint, and `correct_of_safe` composes the programs:

1. `start_bound`: the block's `int32` offsets do not wrap (`BLOCK_SIZE` divides `2^31`);
2. `safe_program`: one program, walked line by line with the rules of `TVA.Proof.Rules`;
3. `main`: the launch is valid, the footprints are disjoint, and the blocks cover `[0, n)`.
-/

namespace Claims.VecAdd

open TVA

/-- The number of programs, `cdiv(n, BLOCK_SIZE)`. -/
def G (d : Inst) : Nat := (d.n + d.BLOCK_SIZE - 1) / d.BLOCK_SIZE

/-- The elements program `p` owns: its block, cut at `n`. -/
def blk (d : Inst) (p : Nat) : Set Nat :=
  {j | p * d.BLOCK_SIZE ≤ j ∧ j < p * d.BLOCK_SIZE + d.BLOCK_SIZE ∧ j < d.n}

/-- Program `p` reads its block of `x` and `y`. -/
def R (d : Inst) (p : Nat) : Set Int := d.x.elems (blk d p) ∪ d.y.elems (blk d p)

/-- Program `p` writes its block of `output`. -/
def W (d : Inst) (p : Nat) : Set Int := d.output.elems (blk d p)

/-- After program `p`, each element of its block of `output` is the sum of the initial `x` and `y`. -/
def Q (d : Inst) (init : Mem) (p : Nat) (m : Mem) : Prop :=
  ∀ j ∈ blk d p, ∃ a b c : F32,
    init.read .f32 (d.x.ptr j).addr = some a ∧ init.read .f32 (d.y.ptr j).addr = some b ∧
    m.read .f32 (d.output.ptr j).addr = some c ∧ FloatV.NaNEq (FloatV.add a b) c

/-- A program's block start and end fit in `int32`: `p · B < n` and, since `B` divides `2^31`,
`p · B + B ≤ 2^31`. -/
theorem start_bound {d : Inst} (hd : d.ok) {p : Nat} (hp : p < G d) :
    p * d.BLOCK_SIZE < d.n ∧ p * d.BLOCK_SIZE + d.BLOCK_SIZE ≤ 2 ^ 31 := by
  obtain ⟨e, he⟩ := hd.block_pow2
  have hB : 0 < d.BLOCK_SIZE := by rw [he]; positivity
  have hlt : p * d.BLOCK_SIZE < d.n := by
    have h1 := (Nat.le_div_iff_mul_le hB).mp (Nat.succ_le_of_lt hp)
    rw [Nat.succ_mul] at h1
    omega
  refine ⟨hlt, ?_⟩
  have he20 : e ≤ 20 := by
    by_contra h
    have : 2 ^ 21 ≤ 2 ^ e := Nat.pow_le_pow_right (by norm_num) (by omega)
    have := hd.block_le
    omega
  have h31 : 2 ^ 31 = 2 ^ (31 - e) * 2 ^ e := by rw [← pow_add]; congr 1; omega
  have hp' : p < 2 ^ (31 - e) := by
    by_contra h
    have h2 := Nat.mul_le_mul_right (2 ^ e) (Nat.le_of_not_lt h)
    rw [← h31, ← he] at h2
    have := hd.n_lt
    omega
  rw [he, h31, ← Nat.succ_mul]
  exact Nat.mul_le_mul_right _ hp'

/-- **One program** of the launch, the one with `program_id(0) = p`, is safe from `init`: it
reads its block of `x` and `y`, writes its block of `output`, and ends with `Q`. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env) {p : Nat}
    (hp : p < G d) (c : Ctx) (hc : c.pid 0 = p) :
    Safe E (R d p) (W d p) (fun _ m => Q d init p m) init ((model d).body c) := by
  obtain ⟨hlt, hend⟩ := start_bound hd hp
  obtain ⟨hvx, hvy, hvo⟩ := hpre
  have hB : d.BLOCK_SIZE ≤ 2 ^ 20 := hd.block_le
  have hn31 : d.n < 2 ^ 31 := hd.n_lt
  have hp31 : p < 2 ^ 31 := by
    obtain ⟨e, he⟩ := hd.block_pow2
    have : 1 ≤ d.BLOCK_SIZE := by rw [he]; exact Nat.one_le_two_pow
    have : p ≤ p * d.BLOCK_SIZE := Nat.le_mul_of_pos_right p this
    omega
  -- the lanes: lane `i` is element `p · B + i`, enabled when it is below `n`
  let g : Idx [d.BLOCK_SIZE - 0] → Nat := fun i => p * d.BLOCK_SIZE + i.1.val
  let en : Idx [d.BLOCK_SIZE - 0] → Bool := fun i => decide (p * d.BLOCK_SIZE + i.1.val < d.n)
  have hpid : (tl.program_id c 0).toInt = (p : Int) := by
    rw [tl.program_id, hc]
    exact IntV.I32.toInt_ofNat (by omega)
  have hstart : (tl.program_id c 0 * (IntV.ofNat d.BLOCK_SIZE : I32)).toInt =
      ((p * d.BLOCK_SIZE : Nat) : Int) :=
    IntV.I32.mul_nat hpid (IntV.I32.toInt_ofNat (by omega)) (by omega)
  have hoff : ∀ i : Idx [d.BLOCK_SIZE - 0],
      ((tl.program_id c 0 * (IntV.ofNat d.BLOCK_SIZE : I32) + tl.arange 0 d.BLOCK_SIZE).get i).toInt
        = ((g i : Nat) : Int) := by
    rintro ⟨i, ⟨⟩⟩
    exact IntV.I32.toInt_sadd_arange hstart (by omega) i
  have hn : (IntV.ofNat d.n : I32).toInt = (d.n : Int) := IntV.I32.toInt_ofNat hd.n_lt
  have hmask : ∀ i, ((tl.program_id c 0 * (IntV.ofNat d.BLOCK_SIZE : I32) +
      tl.arange 0 d.BLOCK_SIZE).ltS (IntV.ofNat d.n : I32)).get i = en i := by
    intro i
    simp only [Tensor.get_ltS, IntV.lt_eq, hoff i, hn, en, g]
    norm_cast
  have hin : ∀ (b : Buf .f32), d.n ≤ b.len → ∀ i, en i = true → g i < b.len := by
    intro b hb i hi
    simp only [en, decide_eq_true_eq] at hi
    show p * d.BLOCK_SIZE + i.1.val < b.len
    omega
  have hblk : ∀ i, en i = true → g i ∈ blk d p := by
    intro i hi
    simp only [en, decide_eq_true_eq] at hi
    have := i.1.isLt
    exact ⟨Nat.le_add_right _ _, by show p * d.BLOCK_SIZE + i.1.val < _; omega, hi⟩
  have himg : Buf.img g en ⊆ blk d p := by
    rintro _ ⟨i, hi, rfl⟩; exact hblk i hi
  have hptr : ∀ (b : Buf .f32) i, (b.ptr 0 + (tl.program_id c 0 * (IntV.ofNat d.BLOCK_SIZE : I32) +
      tl.arange 0 d.BLOCK_SIZE)).get i = b.ptr (g i : Int) := by
    intro b i
    show (b.ptr 0).offset _ = _
    rw [hoff i, Buf.ptr_offset, zero_add]
  simp only [model, add_kernel]
  refine safe_load_buf d.x (Buf.addrs_of_get d.x hmask (fun i _ => hptr d.x i)) (hin d.x hd.x_len)
    hvx (fun z hz => Or.inl (d.x.elems_mono himg hz)) fun vx hx _ => ?_
  refine safe_load_buf d.y (Buf.addrs_of_get d.y hmask (fun i _ => hptr d.y i)) (hin d.y hd.y_len)
    hvy (fun z hz => Or.inr (d.y.elems_mono himg hz)) fun vy hy _ => ?_
  refine safe_add fun vo hvadd => ?_
  refine safe_last ?_
  refine safe_store_buf d.output (Buf.addrs_of_get d.output hmask (fun i _ => hptr d.output i))
    (hin d.output hd.output_len) ?_ hvo (d.output.elems_mono himg) fun m' hm' _ _ => safe_ret ?_
  · intro i j _ _ h
    rcases i with ⟨i, ⟨⟩⟩; rcases j with ⟨j, ⟨⟩⟩
    dsimp only [g] at h
    exact Prod.ext (Fin.ext (by dsimp only; omega)) rfl
  · intro j ⟨hj1, hj2, hj3⟩
    have hjB : j - p * d.BLOCK_SIZE < d.BLOCK_SIZE - 0 := by omega
    let i : Idx [d.BLOCK_SIZE - 0] := (⟨j - p * d.BLOCK_SIZE, hjB⟩, ())
    have hgi : g i = j := by simp only [g, i]; omega
    have hei : en i = true := by simp only [en, decide_eq_true_eq]; rw [← hgi] at hj3; exact hj3
    refine ⟨vx.get i, vy.get i, vo.get i, ?_, ?_, ?_, ?_⟩
    · have := hx i hei; rwa [hgi] at this
    · have := hy i hei; rwa [hgi] at this
    · have := hm' i hei; rwa [hgi] at this
    · simpa using hvadd i

/-- The blocks of distinct programs are disjoint. -/
theorem blk_disjoint {d : Inst} {p q : Nat} (hpq : p ≠ q) : Disjoint (blk d p) (blk d q) := by
  rw [Set.disjoint_left]
  rintro j ⟨hp1, hp2, -⟩ ⟨hq1, hq2, -⟩
  have hB : 0 < d.BLOCK_SIZE := by omega
  have e1 : j / d.BLOCK_SIZE = p := Nat.div_eq_of_lt_le hp1 (by rw [Nat.succ_mul]; exact hp2)
  have e2 : j / d.BLOCK_SIZE = q := Nat.div_eq_of_lt_le hq1 (by rw [Nat.succ_mul]; exact hq2)
  exact hpq (e1.symm.trans e2)

/-- A block holds elements below `n` only. -/
theorem blk_sub {d : Inst} (p : Nat) : blk d p ⊆ {j | j < d.n} := fun _ h => h.2.2

/-- Element `j < n` is in the block of program `j / B`, and that program is in the grid. -/
theorem mem_blk {d : Inst} (hd : d.ok) {j : Nat} (hj : j < d.n) :
    j ∈ blk d (j / d.BLOCK_SIZE) ∧ j / d.BLOCK_SIZE < G d := by
  obtain ⟨e, he⟩ := hd.block_pow2
  have hB : 0 < d.BLOCK_SIZE := by rw [he]; positivity
  have h1 := Nat.div_mul_le_self j d.BLOCK_SIZE
  have h2 := Nat.lt_div_mul_add (a := j) hB
  refine ⟨⟨h1, h2, hj⟩, ?_⟩
  apply Nat.lt_of_succ_le
  rw [G, Nat.le_div_iff_mul_le hB, Nat.succ_mul]
  omega

/-- **The claim.** Each program is safe on its own block (`safe_program`); the blocks' footprints
are disjoint, and together they give `Post`. -/
theorem main : claim.Holds := by
  intro P _ d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  have hB : 0 < d.BLOCK_SIZE := by obtain ⟨e, he⟩ := hd.block_pow2; rw [he]; positivity
  have hsize : (launch d).size = G d := by simp [launch, G]
  have hW : ∀ p, W d p ⊆ d.output.elems {j | j < d.n} := fun p => d.output.elems_mono (blk_sub p)
  have hRx : ∀ p, d.x.elems (blk d p) ⊆ d.x.elems {j | j < d.n} := fun p => d.x.elems_mono (blk_sub p)
  have hRy : ∀ p, d.y.elems (blk d p) ⊆ d.y.elems {j | j < d.n} := fun p => d.y.elems_mono (blk_sub p)
  have hWR : ∀ p q, Disjoint (W d p) (R d q) := fun p q =>
    Disjoint.union_right (hd.output_x.mono (hW p) (hRx q)) (hd.output_y.mono (hW p) (hRy q))
  refine correct_of_safe (fun i => R d i) (fun i => W d i) (fun i m => Q d init i m) ?_
    (fun i => ?_) ?_ ?_ ?_
  · -- the launch is valid
    refine Launch.valid_grid1 ?_ ?_
    · show (d.n + d.BLOCK_SIZE - 1) / d.BLOCK_SIZE < 2 ^ 31
      rw [Nat.div_lt_iff_lt_mul hB]
      have := hd.n_lt
      omega
    · intro s hs
      simp only [List.mem_singleton] at hs
      subst hs
      exact ⟨by simpa using hd.block_pow2, by simpa using hd.block_le⟩
  · -- program `i`
    have hi : (i : Nat) < G d := hsize ▸ i.isLt
    refine safe_program hd hpre _ hi _ ?_
    simp only [Launch.ctx, launch, Launch.grid1]
    exact Nat.mod_eq_of_lt (by simpa [G] using hi)
  · -- no program writes what another reads or writes
    intro i j hij
    refine Disjoint.union_right (hWR i j) (d.output.elems_disjoint (blk_disjoint ?_))
    exact fun h => hij (Fin.ext h)
  · -- no program reads what it writes
    intro i
    exact (hWR i i).symm
  · -- the programs' results give `Post`
    intro fin hloc hfr
    refine ⟨fun j hj => ?_, fun z hz => hfr z fun i hzi => hz (hW i hzi)⟩
    obtain ⟨hjb, hjG⟩ := mem_blk hd hj
    obtain ⟨m, hQ, hagree⟩ := hloc ⟨j / d.BLOCK_SIZE, hsize ▸ hjG⟩
    obtain ⟨a, b, c, ha, hb, hc, hab⟩ := hQ j hjb
    refine ⟨a, b, c, ha, hb, ?_, hab⟩
    rw [Mem.read_congr fun x hx => hagree x (Or.inr (d.output.span_sub_elems hjb hx))]
    exact hc

end Claims.VecAdd

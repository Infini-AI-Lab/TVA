import Proof.Lanes
import Proof.Ubounds

/-!
# `seeded_dropout_f32`

Program `p` owns the elements of its tile (`Buf.tile`, `BLOCK_SIZE` elements cut at `n`). It reads
them in `x` and writes them in `output`, so the footprints of distinct programs are disjoint and
`correct_of_safe` composes the programs. The kernel drops element `j` (writes `+0`) when its random
number is not above `p`, and otherwise writes the platform's quotient `x[j] / (1 - p)`.

1. `Proof.Consts`, `Proof.Bounds`, `Proof.Ubounds`: the constants `1` and `scale32`; each random
   number `uRef` is finite and below `1`; the divisor `dRef = 1 - p` is finite, and nonzero when
   `p < 1`, so a kept element has `p < 1`;
2. `Proof.Keys`, `Proof.Lanes`: for every seed type, lane `l` of `tl.rand` is the claim's `uRef`
   of its element, and the lane's comparison with `p` is the claim's `keep`;
3. `safe_program`: one program walked line by line (`safe_load_block`, `safe_rand`, `safe_sub`,
   `safe_div`, `safe_store_block`); a dropped lane stores `+0`, a kept lane the quotient, which the
   device's division rule (`hP.divFull`) bounds;
4. `main`: the launch is valid, the three seed types each give a safe program, the tiles are
   disjoint and cover `[0, n)`.
-/

namespace Claims.SeededDropout

open TVA
open TVA.Format
open Block

/-- The number of programs, `cdiv(n, BLOCK_SIZE)`. -/
abbrev grid (d : Inst) : Nat := (d.n + d.BLOCK_SIZE - 1) / d.BLOCK_SIZE

/-- Program `p`'s body, with the block pattern of `TVA.Proof.Pattern.Block`. -/
theorem body_eq {sg : Bool} {w : Nat} (d : Inst) (seed : IntV sg w) (c : Ctx) :
    (modelAt d seed).body c =
      (do
        let x ← tl.load (d.x.ptr 0 + offs (c.pid 0) d.BLOCK_SIZE)
          (some (mask (c.pid 0) d.BLOCK_SIZE d.n))
        let random ← tl.rand seed (offs (c.pid 0) d.BLOCK_SIZE)
        let one_minus_p ← tl.sub (Tensor.scalar one32) (Tensor.scalar (p32 d))
        let q ← tl.div x (bc one_minus_p.item)
        tl.store (d.output.ptr 0 + offs (c.pid 0) d.BLOCK_SIZE)
          (tl.where (Tensor.flt (bc (p32 d)) random) q (bc (FloatV.litF32 (FloatV.ofPat 0))))
          (some (mask (c.pid 0) d.BLOCK_SIZE d.n))) := rfl

/-- The grid has `cdiv(n, BLOCK_SIZE)` programs. -/
theorem launch_size (d : Inst) : (launch d).size = grid d := Launch.size_grid1 _ _ _

/-- Program `i` of the grid has `program_id(0) = i`. -/
theorem launch_pid (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = i := by
  have hi : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  simp [Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi]

/-- `BLOCK_SIZE` is positive (a power of two). -/
theorem block_pos {d : Inst} (hd : d.ok) : 0 < d.BLOCK_SIZE := by
  obtain ⟨e, he⟩ := hd.block_pow2
  rw [he]; exact Nat.two_pow_pos e

/-- The grid has fewer than `2^31` programs: `cdiv(n, BLOCK_SIZE) ≤ n` for `BLOCK_SIZE ≥ 1`. -/
theorem grid_lt {d : Inst} (hd : d.ok) : grid d < 2 ^ 31 := by
  have hB := block_pos hd
  have hn := hd.n_lt
  show (d.n + d.BLOCK_SIZE - 1) / d.BLOCK_SIZE < 2 ^ 31
  rw [Nat.div_lt_iff_lt_mul hB]
  have : 2 ^ 31 * 1 ≤ 2 ^ 31 * d.BLOCK_SIZE := Nat.mul_le_mul_left _ hB
  omega

/-- The launch is valid: fewer than `2^31` programs, and a block shape that is a power of two at most `2^20`. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 (grid_lt hd) fun s hs => ?_
  simp only [List.mem_singleton] at hs
  subst hs
  refine ⟨fun k hk => ?_, ?_⟩
  · simp only [List.mem_singleton] at hk
    subst hk; exact hd.block_pow2
  · simpa using hd.block_le

/-- Every program of the grid has `int32` offsets that do not wrap. -/
theorem fits {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) : Fits p d.BLOCK_SIZE := by
  obtain ⟨e, he⟩ := hd.block_pow2
  have he20 : e ≤ 20 := by
    have h := hd.block_le
    rw [he] at h
    exact (Nat.pow_le_pow_iff_right (by norm_num)).mp h
  exact fits_of_grid_pow2 he (by omega) hd.n_lt hp

/-- Every element below `n` is in the tile of a program of the grid. -/
theorem div_lt_grid {d : Inst} (hd : d.ok) {j : Nat} (hj : j < d.n) : j / d.BLOCK_SIZE < grid d := by
  have hB := block_pos hd
  show j / d.BLOCK_SIZE < (d.n + d.BLOCK_SIZE - 1) / d.BLOCK_SIZE
  rw [Nat.div_lt_iff_lt_mul hB]
  have h1 := Nat.div_add_mod (d.n + d.BLOCK_SIZE - 1) d.BLOCK_SIZE
  have h2 := Nat.mod_lt (d.n + d.BLOCK_SIZE - 1) hB
  rw [Nat.mul_comm] at h1
  omega

/-- `tl.where` on a lane of a rank-1 block. -/
theorem get_where_block {n : Nat} (cnd : Tensor [n] Bool) (x y : Tensor [n] F32) (l : Fin n) :
    (tl.where cnd x y).get1 l =
      if cnd.get1 l then x.get1 l else y.get1 l := rfl

/-- A broadcast scalar's value at a lane of a rank-1 block. -/
theorem get_bc {α : Type} {n : Nat} (a : α) (l : Fin n) : (bc a : Tensor [n] α).get1 l = a :=
  rfl

/-- An element of a tile is a lane of its block. -/
theorem tile_lane {p B n j : Nat} (h : Tile p B n j) :
    ∃ l : Fin (B - 0), j = p * B + l := by
  obtain ⟨h1, h2, -⟩ := h
  exact ⟨⟨j - p * B, by omega⟩, by show j = p * B + (j - p * B); omega⟩

/-- The read footprint of program `p`: its tile of `x`. -/
def R (d : Inst) (p : Nat) : Set Int := d.x.tile p d.BLOCK_SIZE d.n

/-- The write footprint of program `p`: its tile of `output`. -/
def W (d : Inst) (p : Nat) : Set Int := d.output.tile p d.BLOCK_SIZE d.n

/-- What program `p` leaves in its own memory `m`: every element of its tile of `output` is either
`0` (the element was dropped) or the quotient `x / (1 - p)` within the division rule's error. -/
def Q (d : Inst) (init : Mem) (p : Nat) (m : Mem) : Prop :=
  ∀ j, Tile p d.BLOCK_SIZE d.n j → ∃ a c : F32,
    init.read .f32 (d.x.ptr j).addr = some a ∧
    m.read .f32 (d.output.ptr j).addr = some c ∧
    (keep d j = false → c = FloatV.ofPat 0) ∧
    (keep d j = true → a.isFinite = true → |a.toRat / (dRef d).toRat| ≤ 2 ^ 126 →
      c.isFinite = true ∧
      |c.toRat - a.toRat / (dRef d).toRat| ≤
        2 ^ (-22 : ℤ) * |a.toRat / (dRef d).toRat| + 2 ^ (-148 : ℤ))

/-- **One program.** Program `p` of the grid, run alone from `init`, is safe, reads its tile of
`x`, writes its tile of `output`, and leaves there either `0` or the quotient. -/
theorem safe_program {sg : Bool} {w : Nat} {d : Inst} (hd : d.ok) (seed : IntV sg w)
    (hk0 : (seed.cast : U64).bits.setWidth 32 = key0 d.seed)
    (hk1 : ((seed.cast : U64).bits >>> 32).setWidth 32 = key1 d.seed)
    {init : Mem} (hpre : Pre d init) (E : Env) {p : Nat} (hp : p < grid d)
    (hdivt : ∀ x y : F32, ∃ r, E.P.div .binary32 x y r)
    (hdivfull : ∀ (x y r : F32), E.P.div .binary32 x y r → x.isFinite = true →
      y.isFinite = true → y.toRat ≠ 0 → |x.toRat / y.toRat| ≤ 2 ^ 126 →
      r.isFinite = true ∧
      |r.toRat - x.toRat / y.toRat| ≤ 2 ^ (-22 : ℤ) * |x.toRat / y.toRat| + 2 ^ (-148 : ℤ)) :
    Safe E (R d p) (W d p) (fun _ m => Q d init p m) init
      (do
        let x ← tl.load (d.x.ptr 0 + offs p d.BLOCK_SIZE) (some (mask p d.BLOCK_SIZE d.n))
        let random ← tl.rand seed (offs p d.BLOCK_SIZE)
        let one_minus_p ← tl.sub (Tensor.scalar one32) (Tensor.scalar (p32 d))
        let q ← tl.div x (bc one_minus_p.item)
        tl.store (d.output.ptr 0 + offs p d.BLOCK_SIZE)
          (tl.where (Tensor.flt (bc (p32 d)) random) q (bc (FloatV.litF32 (FloatV.ofPat 0))))
          (some (mask p d.BLOCK_SIZE d.n))) := by
  obtain ⟨hvx, hvo⟩ := hpre
  have hfit : Fits p d.BLOCK_SIZE := fits hd hp
  have hpNaN : (p32 d).isNaN = false := (FloatV.isFinite_iff.mp hd.p_fin).1
  refine safe_load_block d.x hfit hd.n_lt hd.x_len hvx subset_rfl fun vx hx _ => ?_
  refine safe_rand (seed := seed) fun r hr => ?_
  refine safe_sub fun dsub hsub => ?_
  refine safe_div (fun i => hdivt (vx.get i) dsub.item) fun q hq => ?_
  refine safe_last (safe_store_block d.output hfit hd.n_lt hd.output_len hvo subset_rfl
    fun m' hm' _ _ => safe_ret ?_)
  intro j hj
  obtain ⟨l, rfl⟩ := tile_lane hj
  have hl : p * d.BLOCK_SIZE + l < d.n := hj.2.2
  have hrand : r.get1 l = uRef d.seed (p * d.BLOCK_SIZE + l) := by
    obtain ⟨v, hv, hvv⟩ := hr (l, ())
    exact lane_rand (d := d) (p := p) seed hk0 hk1 hfit l hv hvv
  have hkeep : (Tensor.flt (bc (p32 d)) r).get1 l = keep d (p * d.BLOCK_SIZE + l) := by
    show FloatV.lt (p32 d) (r.get1 l) =
      FloatV.lt (p32 d) (uRef d.seed (p * d.BLOCK_SIZE + l))
    rw [hrand]
  refine ⟨vx.get1 l, _, hx l hl, hm' l hl, ?_, ?_⟩
  · intro hk
    have hval : (tl.where (Tensor.flt (bc (p32 d)) r) q
        (bc (FloatV.litF32 (FloatV.ofPat 0)))).get1 l = FloatV.ofPat 0 := by
      rw [get_where_block, hkeep, hk]
      simp [get_bc]
    exact hval
  · intro hk hafin hqmax
    have hval : (tl.where (Tensor.flt (bc (p32 d)) r) q
        (bc (FloatV.litF32 (FloatV.ofPat 0)))).get1 l = q.get1 l := by
      rw [get_where_block, hkeep, hk]
      simp
    rw [hval]
    have hsub' : FloatV.NaNEq (FloatV.sub one32 (p32 d)) dsub.item := hsub ()
    have hnan : (FloatV.sub one32 (p32 d)).isNaN = false :=
      (FloatV.isFinite_iff.mp (dRef_isFinite (d := d) hd)).1
    have hy : dsub.item = dRef d := by
      have h := FloatV.NaNEq.eq hsub' hnan
      rw [h, dRef_eq]
    have hlt : FloatV.lt (p32 d) (uRef d.seed (p * d.BLOCK_SIZE + l)) = true := hk
    have hp1 : (p32 d).toRat < 1 :=
      lt_trans ((FloatV.lt_iff hpNaN (uRef_isFinite d.seed _).1).mp hlt) (uRef_toRat_lt_one d.seed _)
    have hyfin : dsub.item.isFinite = true := by rw [hy]; exact dRef_isFinite (d := d) hd
    have hyne : dsub.item.toRat ≠ 0 := by rw [hy]; exact dRef_toRat_ne_zero (d := d) hd hp1
    have hqmax' : |(vx.get1 l).toRat / dsub.item.toRat| ≤ 2 ^ 126 := by
      rw [hy]; exact hqmax
    have hres := hdivfull (vx.get1 l) dsub.item (q.get1 l) (hq (l, ()))
      hafin hyfin hyne hqmax'
    rw [hy] at hres
    exact hres

/-- A tile of `b` and a tile of `c`, of any two programs, share no byte when the first `n`
elements of `b` and of `c` share none. -/
theorem tile_disjoint_in {b c : Buf .f32} {n B : Nat}
    (h : Disjoint (b.elems {j | j < n}) (c.elems {j | j < n})) (p q : Nat) :
    Disjoint (b.tile p B n) (c.tile q B n) :=
  h.mono (b.elems_mono fun _ hj => hj.2.2) (c.elems_mono fun _ hj => hj.2.2)

/-- **The claim.** Each program is safe on its own tile (`safe_program`, for the seed's type); the
tiles' footprints are disjoint, and together they give `Post`. -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  have hdivt : ∀ x y : F32, ∃ r, P.div .binary32 x y r := hP.divTotal .binary32
  have hdivfull := hP.divFull
  have hprog : ∀ i : Fin (launch d).size, Safe ((launch d).env P) (R d i) (W d i)
      (fun _ m => Q d init i m) init ((model d).body ((launch d).ctx i)) := by
    intro i
    have hp : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
    by_cases h1 : -2 ^ 31 ≤ d.seed ∧ d.seed < 2 ^ 31
    · have hb : ((IntV.ofInt d.seed : I32).cast : U64).bits = BitVec.ofInt 64 d.seed :=
        cast_u64_of_i32 d.seed h1.1 h1.2
      have hm : model d = modelAt d (IntV.ofInt d.seed : I32) := by
        unfold model
        rw [ite_eq_left h1]
      rw [hm, body_eq d (IntV.ofInt d.seed : I32) ((launch d).ctx i), launch_pid d i]
      exact safe_program hd (IntV.ofInt d.seed : I32) (by rw [hb]; exact ofInt64_setWidth d.seed)
        (by rw [hb]; exact ofInt64_hi d.seed) hpre _ hp hdivt hdivfull
    · by_cases h2 : d.seed < 2 ^ 63
      · have hb : ((IntV.ofInt d.seed : I64).cast : U64).bits = BitVec.ofInt 64 d.seed :=
          cast_u64_of_i64 d.seed
        have hm : model d = modelAt d (IntV.ofInt d.seed : I64) := by
          unfold model
          rw [ite_eq_right h1, ite_eq_left h2]
        rw [hm, body_eq d (IntV.ofInt d.seed : I64) ((launch d).ctx i), launch_pid d i]
        exact safe_program hd (IntV.ofInt d.seed : I64) (by rw [hb]; exact ofInt64_setWidth d.seed)
          (by rw [hb]; exact ofInt64_hi d.seed) hpre _ hp hdivt hdivfull
      · have hb : ((IntV.ofInt d.seed : U64).cast : U64).bits = BitVec.ofInt 64 d.seed :=
          cast_u64_of_u64 d.seed
        have hm : model d = modelAt d (IntV.ofInt d.seed : U64) := by
          unfold model
          rw [ite_eq_right h1, ite_eq_right h2]
        rw [hm, body_eq d (IntV.ofInt d.seed : U64) ((launch d).ctx i), launch_pid d i]
        exact safe_program hd (IntV.ofInt d.seed : U64) (by rw [hb]; exact ofInt64_setWidth d.seed)
          (by rw [hb]; exact ofInt64_hi d.seed) hpre _ hp hdivt hdivfull
  refine correct_of_safe (fun i => R d i) (fun i => W d i) (fun i m => Q d init i m)
    (launch_valid hd) hprog ?_ ?_ ?_
  · intro i j hij
    have hij' : (i : Nat) ≠ j := fun h => hij (Fin.ext h)
    simp only [R, W]
    exact Disjoint.union_right (tile_disjoint_in hd.output_x _ _) (d.output.tile_disjoint hij')
  · intro i
    simp only [R, W]
    exact (tile_disjoint_in hd.output_x _ _).symm
  · intro fin hloc hfr
    refine ⟨fun j hj => ?_, fun z hz => hfr z fun i hzi => hz ?_⟩
    · have hB := block_pos hd
      have hp : j / d.BLOCK_SIZE < (launch d).size := lt_of_lt_of_eq (div_lt_grid hd hj) (launch_size d).symm
      have ht : Tile (j / d.BLOCK_SIZE) d.BLOCK_SIZE d.n j := tile_div hB hj
      obtain ⟨m, hQ, hagree⟩ := hloc ⟨j / d.BLOCK_SIZE, hp⟩
      obtain ⟨a, c, ha, hc, h0, h1⟩ := hQ j ht
      refine ⟨a, c, ha, ?_, h0, h1⟩
      rw [← hc]
      exact Mem.read_congr fun x hx => hagree x (Or.inr (d.output.span_sub_tile ht hx))
    · exact d.output.elems_mono (fun j hj => hj.2.2) hzi

end Claims.SeededDropout

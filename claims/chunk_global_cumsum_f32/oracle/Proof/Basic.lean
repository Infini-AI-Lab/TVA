import Claim
import TVA.Proof

/-!
# The model's body, unfolded, the launch, and the program's geometry

`body_eq` writes the model's body as its prologue followed by `tail`, the loop over the blocks
whose trip is `tripBody`. Program `i` of the grid handles sequence `seqOf d i = i / H` and head
`headOf d i = i % H`; trip `t` of its loop handles block `bt d i t` (`t`, or `nBlk − 1 − t`
reversed), whose lane `l` is position `lo + bt·BT + l`.
-/

namespace Claims.ChunkGlobalCumsum

open TVA

/-! ## The body -/

/-- One trip of the loop (source lines 85–101), with the loop's scalars as arguments. -/
def tripBody (d : Inst) (NT T bos i_h : I64) (i_c : I64) (b_z : F32) : Prog F32 := do
  let i_t := if d.reverse then NT - IntV.ofInt 1 - i_c else i_c
  let o_t := bc (i_t * (IntV.ofNat d.BT : I64)) + (← tl.convert .i32 .i64 (tl.arange 0 d.BT))
  let m_t := o_t.ltS T
  let p_s := d.s.ptr 0 + bos * (IntV.ofNat d.H : I64) + i_h + o_t * (IntV.ofNat d.H : I64)
  let p_o := d.o.ptr 0 + bos * (IntV.ofNat d.H : I64) + i_h + o_t * (IntV.ofNat d.H : I64)
  let b_s ← tl.load p_s (some m_t) (some (bc (FloatV.litF32 (FloatV.ofPat 0))))
  let b_o ← tl.cumsum b_s 0 d.reverse
  let b_ss ← tl.sum b_s
  let b_o ← tl.add b_o (bc b_z)
  let b_z ← if IntV.le (IntV.ofInt 0 : I64) i_c then do
      pure (← tl.add (Tensor.scalar b_z) (Tensor.scalar b_ss)).item
    else pure b_z
  let b_o ← if d.hasScale then tl.mul b_o (bc (sigma32 d)) else pure b_o
  tl.store p_o b_o (some m_t)
  pure b_z

/-- The body after the sequence's bounds (source lines 80–101). -/
def tail (d : Inst) (bos eos i_h : I64) : Prog Unit := do
  let T := eos - bos
  let b_z : F32 := (tl.zeros [] .f32).item
  let NT := (← tl.idiv (Tensor.scalar (T + (IntV.ofNat (d.BT - 1) : I64)))
    (Tensor.scalar (IntV.ofNat d.BT : I64))).item
  let _b_z ← tl.forRange (IntV.ofInt 0 : I64) NT (IntV.ofInt 1 : I64) b_z (tripBody d NT T bos i_h)
  pure ()

/-- The model's body, unfolded. -/
theorem body_eq (d : Inst) (c : Ctx) :
    (model d).body c =
      (do
        let i_nh := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id c 0))).item
        let i_n := (← tl.idiv (Tensor.scalar i_nh) (Tensor.scalar (IntV.ofNat d.H : I64))).item
        let i_h := (← tl.irem (Tensor.scalar i_nh) (Tensor.scalar (IntV.ofNat d.H : I64))).item
        let (bos, eos) ← if d.varlen then do
            let bos := (← tl.convert d.idxD .i64 (Tensor.scalar (← tl.load1 (d.cu.ptr 0 + i_n)))).item
            let eos := (← tl.convert d.idxD .i64
              (Tensor.scalar (← tl.load1 (d.cu.ptr 0 + i_n + (IntV.ofNat 1 : I32))))).item
            pure (bos, eos)
          else do
            let T64 := (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat d.T : I32))).item
            pure (i_n * T64, i_n * T64 + T64)
        tail d bos eos i_h) := rfl

/-! ## The launch -/

/-- The grid has `gridSize d` programs. -/
theorem launch_size (d : Inst) : (launch d).size = gridSize d := Launch.size_grid1 _ _ _

/-- Program number `i`'s first grid coordinate. -/
theorem launch_pid (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = i := by
  have hi : (i : Nat) < gridSize d := lt_of_lt_of_eq i.isLt (launch_size d)
  simp [Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi]

/-- `BT` is positive. -/
theorem bt_pos {d : Inst} (hd : d.ok) : 0 < d.BT := by
  obtain ⟨e, he⟩ := hd.bt_pow2
  rw [he]
  exact Nat.two_pow_pos e

/-- The number of sequences is at most `8192`. -/
theorem nSeq_le {d : Inst} (hd : d.ok) : nSeq d ≤ 8192 := by
  unfold nSeq
  split
  · rename_i h; exact hd.N_le h
  · exact hd.B_le

/-- The grid has fewer than `2^31` programs. -/
theorem grid_lt {d : Inst} (hd : d.ok) : gridSize d < 2 ^ 31 := by
  unfold gridSize
  have h1 := nSeq_le hd
  have h2 := hd.H_le
  have : nSeq d * d.H ≤ 8192 * 2 ^ 17 := Nat.mul_le_mul h1 h2
  omega

/-- The launch is valid. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 (grid_lt hd) ?_
  intro s hs
  simp only [List.mem_singleton] at hs
  subst hs
  refine ⟨?_, ?_⟩
  · intro k hk
    simp only [List.mem_singleton] at hk
    subst hk
    exact hd.bt_pow2
  · simpa using hd.bt_le

/-! ## The geometry -/

/-- Program `i`'s sequence and head. -/
def seqOf (d : Inst) (i : Nat) : Nat := i / d.H
def headOf (d : Inst) (i : Nat) : Nat := i % d.H

/-- Program `i`'s sequence bounds and length. -/
def loOf (d : Inst) (i : Nat) : Nat := lo d (seqOf d i)
def hiOf (d : Inst) (i : Nat) : Nat := hi d (seqOf d i)
def lenOf (d : Inst) (i : Nat) : Nat := hiOf d i - loOf d i

/-- The number of blocks of program `i`'s sequence. -/
def nbOf (d : Inst) (i : Nat) : Nat := nBlk d (seqOf d i)

/-- The block trip `t` handles. -/
def bt (d : Inst) (i t : Nat) : Nat := if d.reverse then nbOf d i - 1 - t else t

/-- The leaf values of program `i`: position `q`'s gate inside the sequence, `+0` outside. -/
def X (d : Inst) (i : Nat) (q : Nat) : F32 :=
  if loOf d i ≤ q ∧ q < hiOf d i then sAt d q (headOf d i) else FloatV.ofPat 0

/-- The positions whose blocks trips `0, …, t − 1` have done (all their lanes, also past the
sequence's end). -/
def Pset (d : Inst) (i t : Nat) : Finset Nat :=
  if d.reverse then Finset.Ico (loOf d i + (nbOf d i - t) * d.BT) (loOf d i + nbOf d i * d.BT)
  else Finset.Ico (loOf d i) (loOf d i + t * d.BT)

/-- The depth bound of the running sum after `t` trips. -/
def zD (d : Inst) (t : Nat) : Nat := if t = 0 then 0 else Nat.clog 2 d.BT + t

/-- The output elements of block `b` of program `i`. -/
def blkElems (d : Inst) (i b : Nat) : Set Nat :=
  {e | ∃ l, l < d.BT ∧ b * d.BT + l < lenOf d i ∧ e = (loOf d i + b * d.BT + l) * d.H + headOf d i}

/-- Program `i`'s output elements: the positions of its sequence, at its head. -/
def seqElems (d : Inst) (i : Nat) : Set Nat :=
  {e | ∃ p, loOf d i ≤ p ∧ p < hiOf d i ∧ e = p * d.H + headOf d i}

/-- The read footprint: all of `s`, and on a variable-length launch all of `cu_seqlens`. -/
def R (d : Inst) : Set Int := d.s.bytes ∪ (if d.varlen then d.cu.bytes else ∅)

/-- Program `i`'s write footprint. -/
def W (d : Inst) (i : Nat) : Set Int := d.o.elems (seqElems d i)

/-- Trip `t`'s write footprint. -/
def Wt (d : Inst) (i t : Nat) : Set Int := d.o.elems (blkElems d i (bt d i t))

/-- An output value is good: in range, finite and within the tolerance. -/
def Good (d : Inst) (n p h : Nat) (v : F32) : Prop :=
  InRange d n p h → v.isFinite = true ∧ |v.toRat - sig d * refSum d n p h| ≤ tol d n p h

/-- What program `i` leaves: every position of its sequence holds a good value at its head. -/
def Q (d : Inst) (i : Nat) (m : Mem) : Prop :=
  ∀ p, loOf d i ≤ p → p < hiOf d i → ∃ v : F32,
    m.read .f32 (d.o.ptr (p * d.H + headOf d i : Nat)).addr = some v ∧
      Good d (seqOf d i) p (headOf d i) v

end Claims.ChunkGlobalCumsum

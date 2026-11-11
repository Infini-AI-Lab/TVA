import Claim
import Proof.Defs
import Proof.Loop
import Proof.Num
import Proof.Ptr
import TVA.Proof

/-!
# BClause

The load, scale and rounding of `B` in one loop trip, and the `tl.dot` step it feeds.
-/

namespace Claims.SsdChunkState

open TVA

set_option maxHeartbeats 4000000

/-!
# The `b` value clause after one loop iteration

`Inv` carries, for a partial token window, a table `bh` of the exact `ℚ` values the `b` operand
holds (up to the claimed error `errB`) and zeros where the token is not in the chunk.  One
iteration extends the window by `BK` tokens: this file shows that the extended table `bh'` (built
from the converted tile `bt`) satisfies the same clause for the window `(t+1) * BK`, and that every
lane of `bt` is a finite `bfloat16`.
-/

open TVA

set_option maxHeartbeats 800000

/-- Extending the `b` value table by one step of the loop. -/
theorem bClause {d : Inst} (hd : d.ok) {E : Env} {m₁ : Mem}
    {i : Fin (launch d).size} {t : Nat} {st : LoopSt d} {Nch : Nat}
    {bt b0f bo : Tensor [BK, BN] DType.bf16.Val}
    {b32f b2f : Tensor [BK, BN] DType.f32.Val}
    {dAk dtk dsub dexp w zo : Tensor [BK] DType.f32.Val}
    {sq zoi : Tensor [BK] DType.i32.Val}
    {v0 : DType.f32.Val} {v1 : DType.i32.Val}
    {mb : Tensor [BK, BN] Bool} {mo : Tensor [BK] Bool}
    {bh bh' : Nat → Fin BN → ℚ}
    (hNch : Nch = endTok d (chunkOf d i) + 1 - chunkOf d i * d.chunk)
    (hseq : endTok d (chunkOf d i) < d.seqlen)
    (hmeta : MetaOk d m₁)
    (hV : Vals m₁ d)
    (hchunk : 0 < d.chunk)
    (hhh : headOf d i < d.nheads)
    (hgl : grp d (headOf d i) < d.ngroups)
    (hv0 : m₁.read .f32
      (d.dA.ptr (dIdx d (headOf d i) (chunkOf d i) (d.chunk - 1))).addr = some v0)
    (hv1 : m₁.read .i32 (d.seqIdx.ptr (endTok d (chunkOf d i))).addr = some v1)
    (hP2' : ∀ (j : Fin BK) (n : Fin BN), (st.2.1.get2 j n).addr =
      (d.b.ptr (bIdx d (chunkOf d i * d.chunk + (t * BK + (j : Nat)))
        (grp d (headOf d i)) (pnOf d i * BN + (n : Nat)))).addr)
    (hP3 : ∀ j : Fin BK, (st.2.2.1.get1 j).addr =
      (d.dt.ptr (dIdx d (headOf d i) (chunkOf d i) (t * BK + (j : Nat)))).addr)
    (hP4 : ∀ j : Fin BK, (st.2.2.2.1.get1 j).addr =
      (d.dA.ptr (dIdx d (headOf d i) (chunkOf d i) (t * BK + (j : Nat)))).addr)
    (hP5 : ∀ j : Fin BK, (st.2.2.2.2.1.get1 j).addr =
      (d.seqIdx.ptr (chunkOf d i * d.chunk + (t * BK + (j : Nat)))).addr)
    (hchunkK : ∀ j : Fin BK, (j : Nat) < Nch - t * BK → t * BK + (j : Nat) < d.chunk)
    (hmaskB : ∀ (j : Fin BK) (n : Fin BN), mb.get2 j n =
      (decide ((j : Nat) < Nch - t * BK) && decide (pnOf d i * BN + (n : Nat) < d.dstate)))
    (hbo0 : ∀ (j : Fin BK) (n : Fin BN), bo.get2 j n = (FloatV.ofPat (0 : Nat) : BF16))
    (hb0 : LoadRes m₁ st.2.1 (some mb) (some bo) b0f)
    (_hmaskA : ∀ j : Fin BK, (j : Nat) < Nch - t * BK → mo.get (j, ()) = true)
    (_hzo0 : ∀ ix : Idx [BK], zo.get ix = (FloatV.ofPat (0 : Nat) : F32))
    (hzoi0 : ∀ ix : Idx [BK], zoi.get ix = (IntV.ofInt (-1) : I32))
    (hdAk : LoadRes m₁ st.2.2.2.1 (some mo) (some zo) dAk)
    (hsq : LoadRes m₁ st.2.2.2.2.1 (some mo) (some zoi) sq)
    (hdtk : LoadRes m₁ st.2.2.1 (some mo) (some zo) dtk)
    (hb32 : ∀ ix : Idx [BK, BN], (FloatV.cvtWith .rn (b0f.get ix)).NaNEq (b32f.get ix))
    (hsub : ∀ ix : Idx [BK], (((bc v0).get ix).sub (dAk.get ix)).NaNEq (dsub.get ix))
    (hdexp : ∀ ix : Idx [BK], E.P.math .exp .binary32 (dsub.get ix) (dexp.get ix))
    (hw : ∀ ix : Idx [BK], ((dexp.get ix).mul (dtk.get ix)).NaNEq (w.get ix))
    (hb2 : ∀ ix : Idx [BK, BN],
      ((b32f.get ix).mul (((tl.where (sq.eq (bc v1)) w
        (bc (FloatV.ofPat (0 : Nat)).litF32)).col BN).get ix)).NaNEq (b2f.get ix))
    (hbt : ∀ ix : Idx [BK, BN], (FloatV.cvtWith .rn (b2f.get ix)).NaNEq (bt.get ix))
    (hbagree : ∀ (s : Nat) (n : Fin BN), s < t * BK → bh' s n = bh s n)
    (hbnew : ∀ (j : Fin BK) (n : Fin BN), bh' (t * BK + (j : Nat)) n = (bt.get2 j n).toRat)
    (hbcl : ∀ (s : Nat) (n : Fin BN), s < t * BK →
      (((chunkOf d i * d.chunk + s) ∈ toks d (chunkOf d i) ∧
          pnOf d i * BN + (n : Nat) < d.dstate →
        |(bh s n : ℝ) - bV m₁ d (chunkOf d i * d.chunk + s) (headOf d i)
            (pnOf d i * BN + (n : Nat)) *
            wRef m₁ d (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + s)|
          ≤ errB m₁ d (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + s)
            (pnOf d i * BN + (n : Nat))) ∧
       (¬((chunkOf d i * d.chunk + s) ∈ toks d (chunkOf d i) ∧
          pnOf d i * BN + (n : Nat) < d.dstate) → bh s n = 0)))
    (hexpacc : Device.ExpF32 E.P)
    (hmaskAo : ∀ j : Fin BK, mo.get (j, ()) = decide ((j : Nat) < Nch - t * BK)) :
    (∀ (s : Nat) (n : Fin BN), s < (t + 1) * BK →
      (((chunkOf d i * d.chunk + s) ∈ toks d (chunkOf d i) ∧
          pnOf d i * BN + (n : Nat) < d.dstate →
        |(bh' s n : ℝ) - bV m₁ d (chunkOf d i * d.chunk + s) (headOf d i)
            (pnOf d i * BN + (n : Nat)) *
            wRef m₁ d (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + s)|
          ≤ errB m₁ d (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + s)
            (pnOf d i * BN + (n : Nat))) ∧
       (¬((chunkOf d i * d.chunk + s) ∈ toks d (chunkOf d i) ∧
          pnOf d i * BN + (n : Nat) < d.dstate) → bh' s n = 0))) ∧
    (∀ (j : Fin BK) (n : Fin BN), (bt.get2 j n).isFinite = true) := by
  have hc : chunkOf d i < nc d := chunkOf_lt hd (i : Nat)
  have hch1 : d.chunk - 1 < d.chunk := by omega
  -- the `dA_cumsum` entry at the chunk's end is non-positive
  obtain ⟨v0', hv0r, hv0f', hv0n'⟩ := (hV.2.2.1 (headOf d i) (chunkOf d i) (d.chunk - 1) hhh hc hch1).2
  have hv0eq : v0 = v0' := by
    rw [hv0r] at hv0
    exact (Option.some.inj hv0).symm
  have hv0f : v0.isFinite = true := by rw [hv0eq]; exact hv0f'
  have hv0n : v0.toRat ≤ 0 := by rw [hv0eq]; exact hv0n'
  -- the request at the chunk's end
  have hv1v : v1 = (IntV.ofNat (eReq d (chunkOf d i)) : I32) := by
    have h1 := hmeta (endTok d (chunkOf d i)) hseq
    rw [hv1] at h1
    exact Option.some.inj h1
  -- the `tl.where` lane value
  have hkey : ∀ (j : Fin BK) (n : Fin BN),
      ((tl.«where» (sq.eq (bc v1)) w (bc (FloatV.ofPat (0 : Nat)).litF32)).col BN).get (j, n, ())
        = if (sq.eq (bc v1)).get (j, ()) = true then w.get (j, ())
          else (FloatV.ofPat (0 : Nat) : F32) := by
    intro j n
    rfl
  have hcond : ∀ j : Fin BK,
      (sq.eq (bc (s := [BK]) v1)).get (j, ()) = decide (sq.get1 j = v1) := by
    intro j
    show IntV.eq (sq.get1 j) v1 = decide (sq.get1 j = v1)
    exact IntV.eq_eq_decide _ _
  have hselW : ∀ (j : Fin BK) (n : Fin BN), (sq.eq (bc v1)).get (j, ()) = true →
      ((tl.«where» (sq.eq (bc v1)) w (bc (FloatV.ofPat (0 : Nat)).litF32)).col BN).get (j, n, ())
        = w.get1 j := by
    intro j n h
    rw [hkey j n, h]
    rfl
  have hsel0 : ∀ (j : Fin BK) (n : Fin BN), (sq.eq (bc v1)).get (j, ()) = false →
      ((tl.«where» (sq.eq (bc v1)) w (bc (FloatV.ofPat (0 : Nat)).litF32)).col BN).get (j, n, ())
        = (FloatV.ofPat (0 : Nat) : F32) := by
    intro j n h
    rw [hkey j n, h]
    simp
  -- a token lane inside the window
  have htokle : ∀ (j : Fin BK), (j : Nat) < Nch - t * BK →
      chunkOf d i * d.chunk + (t * BK + (j : Nat)) ≤ endTok d (chunkOf d i) := by
    intro j hj
    have h1 : t * BK + (j : Nat) < Nch := by omega
    have h2 : Nch = endTok d (chunkOf d i) + 1 - chunkOf d i * d.chunk := hNch
    omega
  have hseqlen : ∀ (j : Fin BK), (j : Nat) < Nch - t * BK →
      chunkOf d i * d.chunk + (t * BK + (j : Nat)) < d.seqlen := by
    intro j hj
    have := htokle j hj
    omega
  -- `b2f → bt` carries a finite zero
  have hfin_zero : ∀ (x : F32) (j : Fin BK) (n : Fin BN),
      FloatV.NaNEq x (b2f.get2 j n) → x.toRat = 0 → x.isFinite = true →
      (bt.get2 j n).toRat = 0 ∧ (bt.get2 j n).isFinite = true := by
    intro x j n hx h0 hf
    have hnan : x.isNaN = false := (FloatV.isFinite_iff.mp hf).1
    have hb2z : (b2f.get2 j n).toRat = 0 := by rw [FloatV.NaNEq.eq hx hnan, h0]
    have hb2fin : (b2f.get2 j n).isFinite = true := by
      rw [FloatV.NaNEq.eq hx hnan]
      exact hf
    have hcv0 : (FloatV.cvt (b2f.get2 j n) : BF16).toRat = 0 :=
      cvt_f32_bf16_toRat_zero hb2fin hb2z
    have hcvf : (FloatV.cvt (b2f.get2 j n) : BF16).isFinite = true :=
      cvt_f32_bf16_zero_finite hb2fin hb2z
    have hnan2 : (FloatV.cvt (b2f.get2 j n) : BF16).isNaN = false :=
      (FloatV.isFinite_iff.mp hcvf).1
    have he : bt.get2 j n = (FloatV.cvt (b2f.get2 j n) : BF16) :=
      FloatV.NaNEq.eq (hbt (j, n, ())) hnan2
    exact ⟨by rw [he, hcv0], by rw [he]; exact hcvf⟩
  -- the pointer facts of the hypotheses, with `get1`/`get2` unfolded
  have hP2g : ∀ (j : Fin BK) (n : Fin BN), (st.2.1.get (j, n, ())).addr =
      (d.b.ptr (bIdx d (chunkOf d i * d.chunk + (t * BK + (j : Nat)))
        (grp d (headOf d i)) (pnOf d i * BN + (n : Nat)))).addr := fun j n => hP2' j n
  have hP3g : ∀ j : Fin BK, (st.2.2.1.get (j, ())).addr =
      (d.dt.ptr (dIdx d (headOf d i) (chunkOf d i) (t * BK + (j : Nat)))).addr :=
    fun j => hP3 j
  have hP4g : ∀ j : Fin BK, (st.2.2.2.1.get (j, ())).addr =
      (d.dA.ptr (dIdx d (headOf d i) (chunkOf d i) (t * BK + (j : Nat)))).addr :=
    fun j => hP4 j
  have hP5g : ∀ j : Fin BK, (st.2.2.2.2.1.get (j, ())).addr =
      (d.seqIdx.ptr (chunkOf d i * d.chunk + (t * BK + (j : Nat)))).addr :=
    fun j => hP5 j
  have hmain : ∀ (j : Fin BK) (n : Fin BN),
      (((chunkOf d i * d.chunk + (t * BK + (j : Nat))) ∈ toks d (chunkOf d i) ∧
          pnOf d i * BN + (n : Nat) < d.dstate →
        |((bt.get2 j n).toRat : ℝ) -
            bV m₁ d (chunkOf d i * d.chunk + (t * BK + (j : Nat))) (headOf d i)
              (pnOf d i * BN + (n : Nat)) *
            wRef m₁ d (chunkOf d i) (headOf d i)
              (chunkOf d i * d.chunk + (t * BK + (j : Nat)))|
          ≤ errB m₁ d (chunkOf d i) (headOf d i)
              (chunkOf d i * d.chunk + (t * BK + (j : Nat))) (pnOf d i * BN + (n : Nat))) ∧
       (¬((chunkOf d i * d.chunk + (t * BK + (j : Nat))) ∈ toks d (chunkOf d i) ∧
          pnOf d i * BN + (n : Nat) < d.dstate) → (bt.get2 j n).toRat = 0)) ∧
      (bt.get2 j n).isFinite = true := by
    intro j n
    by_cases hmb : mb.get2 j n = true
    · -- ===================== the lane is inside the tile
      have hm := hmaskB j n
      rw [hmb] at hm
      have hm2 : (decide ((j : Nat) < Nch - t * BK) &&
          decide (pnOf d i * BN + (n : Nat) < d.dstate)) = true := hm.symm
      rw [Bool.and_eq_true] at hm2
      obtain ⟨hjd, hpd⟩ := hm2
      have hjN : (j : Nat) < Nch - t * BK := of_decide_eq_true hjd
      have hp : pnOf d i * BN + (n : Nat) < d.dstate := of_decide_eq_true hpd
      have hmo : mo.get (j, ()) = true := by
        rw [hmaskAo j]
        exact decide_eq_true hjN
      have hle := htokle j hjN
      have hlt := hseqlen j hjN
      have hsqv : sq.get1 j =
          (IntV.ofNat (reqOf d (chunkOf d i * d.chunk + (t * BK + (j : Nat)))) : I32) := by
        have h := load_read (m := m₁) hsq (i := (j, ())) hmo
        rw [hP5g j] at h
        have h2 := hmeta _ hlt
        push_cast at h2
        rw [h] at h2
        exact Option.some.inj h2
      have hwcond : (sq.eq (bc v1)).get (j, ()) = decide (sq.get1 j = v1) := hcond j
      have hb0read : m₁.read .bf16 (d.b.ptr ↑(bIdx d
            (chunkOf d i * d.chunk + (t * BK + (j : Nat)))
            (grp d (headOf d i)) (pnOf d i * BN + (n : Nat)))).addr = some (b0f.get2 j n) := by
        have h := load_read (m := m₁) hb0 (i := (j, n, ())) hmb
        rw [hP2g j n] at h
        exact h
      have hb0f : (b0f.get2 j n).isFinite = true := by
        obtain ⟨v, hv, hf⟩ := hV.2.1 _ (grp d (headOf d i)) (pnOf d i * BN + (n : Nat)) hlt hgl hp
        exact (Option.some.inj (hv.symm.trans hb0read)) ▸ hf
      -- the reads of this token's `dA_cumsum` and `dt`
      have hdkr : m₁.read .f32 (d.dA.ptr ↑(dIdx d (headOf d i) (chunkOf d i)
          (t * BK + (j : Nat)))).addr = some (dAk.get1 j) := by
        have h := load_read (m := m₁) hdAk (i := (j, ())) hmo
        rw [hP4g j] at h
        exact h
      have hdtr : m₁.read .f32 (d.dt.ptr ↑(dIdx d (headOf d i) (chunkOf d i)
          (t * BK + (j : Nat)))).addr = some (dtk.get1 j) := by
        have h := load_read (m := m₁) hdtk (i := (j, ())) hmo
        rw [hP3g j] at h
        exact h
      have hdkK : t * BK + (j : Nat) < d.chunk := hchunkK j hjN
      obtain ⟨dk', hdkr', hdkf', hdkn'⟩ := (hV.2.2.1 (headOf d i) (chunkOf d i) (t * BK + (j : Nat)) hhh hc hdkK).2
      have hdkeq : dk' = dAk.get1 j := Option.some.inj (hdkr'.symm.trans hdkr)
      have hdkf : (dAk.get1 j).isFinite = true := by rw [← hdkeq]; exact hdkf'
      have hdkn : (dAk.get1 j).toRat ≤ 0 := by rw [← hdkeq]; exact hdkn'
      obtain ⟨dt', hdtr', hdtf'⟩ := (hV.2.2.1 (headOf d i) (chunkOf d i) (t * BK + (j : Nat)) hhh hc hdkK).1
      have hdteq : dt' = dtk.get1 j := Option.some.inj (hdtr'.symm.trans hdtr)
      have hdtf : (dtk.get1 j).isFinite = true := by rw [← hdteq]; exact hdtf'
      -- the `hb_chain` arguments that carry the `t - c · chunk` index
      have hdkind : chunkOf d i * d.chunk + (t * BK + (j : Nat)) - chunkOf d i * d.chunk
          = t * BK + (j : Nat) := by omega
      have hdksub : m₁.read .f32 (d.dA.ptr ↑(dIdx d (headOf d i) (chunkOf d i)
          (chunkOf d i * d.chunk + (t * BK + (j : Nat)) - chunkOf d i * d.chunk))).addr
          = some (dAk.get1 j) := by rw [hdkind]; exact hdkr
      have hdtsub : m₁.read .f32 (d.dt.ptr ↑(dIdx d (headOf d i) (chunkOf d i)
          (chunkOf d i * d.chunk + (t * BK + (j : Nat)) - chunkOf d i * d.chunk))).addr
          = some (dtk.get1 j) := by rw [hdkind]; exact hdtr
      by_cases hsel : (sq.eq (bc v1)).get (j, ()) = true
      · -- the lane contributes: the `b` chain of `hb_chain`
        have hsqeq : sq.get1 j = v1 := by
          have h' : decide (sq.get1 j = v1) = true := by rw [← hwcond]; exact hsel
          exact of_decide_eq_true h'
        have hreq : reqOf d (chunkOf d i * d.chunk + (t * BK + (j : Nat)))
            = eReq d (chunkOf d i) := by
          have hof : (IntV.ofNat (reqOf d (chunkOf d i * d.chunk + (t * BK + (j : Nat))))
              : I32) = IntV.ofNat (eReq d (chunkOf d i)) := by
            rw [← hsqv, ← hv1v]
            exact hsqeq
          have h2 := congrArg (fun v : I32 => v.bits.toNat) hof
          simp only [IntV.ofNat, BitVec.toNat_ofNat] at h2
          have her : eReq d (chunkOf d i) = reqOf d (endTok d (chunkOf d i)) := rfl
          rw [her, Nat.mod_eq_of_lt (reqOf_lt hd _),
            Nat.mod_eq_of_lt (reqOf_lt hd (endTok d (chunkOf d i)))] at h2
          exact h2
        have htok : (chunkOf d i * d.chunk + (t * BK + (j : Nat))) ∈ toks d (chunkOf d i) :=
          Finset.mem_filter.mpr ⟨Finset.mem_Icc.mpr ⟨by omega, hle⟩, hreq⟩
        obtain ⟨hD64, hr1', hr2'⟩ := hV.2.2.2.1 (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + (t * BK + (j : Nat))) hc hhh htok
        have hr1 : eHat m₁ d (chunkOf d i) (headOf d i)
              (chunkOf d i * d.chunk + (t * BK + (j : Nat))) *
            |dtV m₁ d (chunkOf d i) (headOf d i)
              (chunkOf d i * d.chunk + (t * BK + (j : Nat)))| ≤ (2 : ℝ) ^ (126 : ℤ) := by
          simpa [zpow_natCast] using hr1'
        have hr2 : |bV m₁ d (chunkOf d i * d.chunk + (t * BK + (j : Nat))) (headOf d i)
              (pnOf d i * BN + (n : Nat))| *
            sHat m₁ d (chunkOf d i) (headOf d i)
              (chunkOf d i * d.chunk + (t * BK + (j : Nat))) ≤ (2 : ℝ) ^ (126 : ℤ) := by
          simpa [zpow_natCast] using hr2' _ hp
        have hb2' : FloatV.NaNEq (FloatV.mul (b32f.get2 j n) (w.get1 j)) (b2f.get2 j n) := by
          have h := hb2 (j, n, ())
          rw [hselW j n hsel] at h
          exact h
        obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, hrfin, hrerr, -⟩ :=
          hb_chain (E := E) (d := d) (m₀ := m₁)
            (c := chunkOf d i) (h := headOf d i)
            (t := chunkOf d i * d.chunk + (t * BK + (j : Nat)))
            (n := pnOf d i * BN + (n : Nat))
            (by omega) hv0 hdksub hdtsub hb0read hv0f hdkf hdtf hb0f hv0n hdkn
            hD64 hr1 hr2 hexpacc
            (hsub (j, ())) (hdexp (j, ())) (hw (j, ())) (hb32 (j, n, ())) hb2' (hbt (j, n, ()))
        exact ⟨⟨fun _ => by simpa only [FloatV.toReal, Tensor.get2] using hrerr,
          fun hc' => absurd ⟨htok, hp⟩ hc'⟩, hrfin⟩
      · -- the sequence index does not match: the multiplier is zero
        have hself : (sq.eq (bc v1)).get (j, ()) = false := by
          cases h : (sq.eq (bc v1)).get (j, ())
          · rfl
          · exact absurd h hsel
        have hne : sq.get1 j ≠ v1 := by
          intro he
          have h1 : decide (sq.get1 j = v1) = true := decide_eq_true he
          rw [← hwcond] at h1
          rw [hself] at h1
          exact Bool.noConfusion h1
        have hcf : ¬((chunkOf d i * d.chunk + (t * BK + (j : Nat))) ∈ toks d (chunkOf d i) ∧
            pnOf d i * BN + (n : Nat) < d.dstate) := by
          rintro ⟨htok, -⟩
          apply hne
          rw [hsqv, hv1v, (Finset.mem_filter.mp htok).2]
        have hb32fin : (b32f.get2 j n).isFinite = true := by
          show (b32f.get (j, n, ())).isFinite = true
          have hN := hb32 (j, n, ())
          have hcvt : (FloatV.cvt (b0f.get2 j n) : F32).isFinite = true :=
            cvt_bf16_f32_finite hb0f
          have hnan : (FloatV.cvt (b0f.get2 j n) : F32).isNaN = false :=
            (FloatV.isFinite_iff.mp hcvt).1
          rw [FloatV.NaNEq.eq hN hnan]
          exact hcvt
        have hmulres := mul_zero_right hb32fin
        have hb2'' : FloatV.NaNEq (FloatV.mul (b32f.get2 j n) (FloatV.ofPat (0 : Nat)))
            (b2f.get2 j n) := by
          have h := hb2 (j, n, ())
          rw [hsel0 j n hself] at h
          exact h
        obtain ⟨h0, hf⟩ := hfin_zero _ j n hb2'' hmulres.1 hmulres.2
        exact ⟨⟨fun hc' => absurd hc' hcf, fun _ => h0⟩, hf⟩
    · -- ===================== the lane is outside the tile
      have hmbf : mb.get2 j n = false := by
        cases h : mb.get2 j n <;> simp_all
      have hcf : ¬((chunkOf d i * d.chunk + (t * BK + (j : Nat))) ∈ toks d (chunkOf d i) ∧
          pnOf d i * BN + (n : Nat) < d.dstate) := by
        rintro ⟨htok, hp⟩
        have hle := (Finset.mem_Icc.mp (Finset.mem_filter.mp htok).1).2
        have hjN : (j : Nat) < Nch - t * BK := by
          have h2 : Nch = endTok d (chunkOf d i) + 1 - chunkOf d i * d.chunk := hNch
          omega
        rw [hmaskB j n, decide_eq_true hjN, decide_eq_true hp] at hmbf
        exact absurd hmbf (by simp)
      have hb0z : b0f.get2 j n = (FloatV.ofPat (0 : Nat) : BF16) := by
        have h := load_mask_false (m := m₁) hb0 (i := (j, n, ())) hmbf
        have hbo : bo.get (j, n, ()) = (FloatV.ofPat (0 : Nat) : BF16) := hbo0 j n
        rw [hbo] at h
        exact h
      have hb32z : b32f.get2 j n = (FloatV.ofPat (0 : Nat) : F32) := by
        have hN := hb32 (j, n, ())
        rw [show b0f.get (j, n, ()) = (FloatV.ofPat (0 : Nat) : BF16) from hb0z] at hN
        rw [FloatV.cvtWith_rn] at hN
        rw [cvt_bf16_f32_zero] at hN
        exact FloatV.NaNEq.eq hN F32_zero_notNaN
      have hb32fin : (b32f.get2 j n).isFinite = true := by
        rw [hb32z]
        exact F32_zero_finite
      by_cases hsel : (sq.eq (bc v1)).get (j, ()) = true
      · -- the sequence index matches: the weight is bounded by `wFinite`
        have hsqeq : sq.get1 j = v1 := by
          have h' : decide (sq.get1 j = v1) = true := by
            rw [← hcond j]
            exact hsel
          exact of_decide_eq_true h'
        have hjN : (j : Nat) < Nch - t * BK := by
          by_contra hcon
          have hmof : mo.get (j, ()) = false := by
            rw [hmaskAo j]
            exact decide_eq_false hcon
          have hsqm : sq.get1 j = (IntV.ofInt (-1) : I32) := by
            have h := load_mask_false (m := m₁) hsq (i := (j, ())) hmof
            rw [hzoi0 (j, ())] at h
            exact h
          rw [hsqm, hv1v] at hsqeq
          exact i32_ofNat_ne_negOne (reqOf_lt_2_31 hd (endTok d (chunkOf d i))) hsqeq.symm
        have hmo : mo.get (j, ()) = true := by
          rw [hmaskAo j]
          exact decide_eq_true hjN
        have hle := htokle j hjN
        have hlt := hseqlen j hjN
        have hsqv : sq.get1 j =
            (IntV.ofNat (reqOf d (chunkOf d i * d.chunk + (t * BK + (j : Nat)))) : I32) := by
          have h := load_read (m := m₁) hsq (i := (j, ())) hmo
          rw [hP5g j] at h
          have h2 := hmeta _ hlt
          push_cast at h2
          rw [h] at h2
          exact Option.some.inj h2
        have hwcond : (sq.eq (bc v1)).get (j, ()) = decide (sq.get1 j = v1) := hcond j
        have hreq : reqOf d (chunkOf d i * d.chunk + (t * BK + (j : Nat)))
            = eReq d (chunkOf d i) := by
          have hof : (IntV.ofNat (reqOf d (chunkOf d i * d.chunk + (t * BK + (j : Nat))))
              : I32) = IntV.ofNat (reqOf d (endTok d (chunkOf d i))) := by
            rw [← hsqv, hsqeq, hv1v]
            rfl
          have h2 := congrArg (fun v : I32 => v.bits.toNat) hof
          simp only [IntV.ofNat, BitVec.toNat_ofNat] at h2
          rw [Nat.mod_eq_of_lt (reqOf_lt hd _),
            Nat.mod_eq_of_lt (reqOf_lt hd (endTok d (chunkOf d i)))] at h2
          exact h2
        have htok : (chunkOf d i * d.chunk + (t * BK + (j : Nat))) ∈ toks d (chunkOf d i) :=
          Finset.mem_filter.mpr ⟨Finset.mem_Icc.mpr ⟨by omega, hle⟩, hreq⟩
        obtain ⟨hD64, hr1', -⟩ := hV.2.2.2.1 (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + (t * BK + (j : Nat))) hc hhh htok
        have hr1 : eHat m₁ d (chunkOf d i) (headOf d i)
              (chunkOf d i * d.chunk + (t * BK + (j : Nat))) *
            |dtV m₁ d (chunkOf d i) (headOf d i)
              (chunkOf d i * d.chunk + (t * BK + (j : Nat)))| ≤ (2 : ℝ) ^ (126 : ℤ) := by
          simpa [zpow_natCast] using hr1'
        have hdkr : m₁.read .f32 (d.dA.ptr ↑(dIdx d (headOf d i) (chunkOf d i)
            (t * BK + (j : Nat)))).addr = some (dAk.get1 j) := by
          have h := load_read (m := m₁) hdAk (i := (j, ())) hmo
          rw [hP4g j] at h
          exact h
        have hdtr : m₁.read .f32 (d.dt.ptr ↑(dIdx d (headOf d i) (chunkOf d i)
            (t * BK + (j : Nat)))).addr = some (dtk.get1 j) := by
          have h := load_read (m := m₁) hdtk (i := (j, ())) hmo
          rw [hP3g j] at h
          exact h
        have hdkK : t * BK + (j : Nat) < d.chunk := hchunkK j hjN
        obtain ⟨dk', hdkr', hdkf', hdkn'⟩ := (hV.2.2.1 (headOf d i) (chunkOf d i) (t * BK + (j : Nat)) hhh hc hdkK).2
        have hdkeq : dk' = dAk.get1 j := Option.some.inj (hdkr'.symm.trans hdkr)
        have hdkf : (dAk.get1 j).isFinite = true := by rw [← hdkeq]; exact hdkf'
        have hdkn : (dAk.get1 j).toRat ≤ 0 := by rw [← hdkeq]; exact hdkn'
        obtain ⟨dt', hdtr', hdtf'⟩ := (hV.2.2.1 (headOf d i) (chunkOf d i) (t * BK + (j : Nat)) hhh hc hdkK).1
        have hdteq : dt' = dtk.get1 j := Option.some.inj (hdtr'.symm.trans hdtr)
        have hdtf : (dtk.get1 j).isFinite = true := by rw [← hdteq]; exact hdtf'
        have hdkind : chunkOf d i * d.chunk + (t * BK + (j : Nat)) - chunkOf d i * d.chunk
            = t * BK + (j : Nat) := by omega
        have hdksub : m₁.read .f32 (d.dA.ptr ↑(dIdx d (headOf d i) (chunkOf d i)
            (chunkOf d i * d.chunk + (t * BK + (j : Nat)) - chunkOf d i * d.chunk))).addr
            = some (dAk.get1 j) := by rw [hdkind]; exact hdkr
        have hdtsub : m₁.read .f32 (d.dt.ptr ↑(dIdx d (headOf d i) (chunkOf d i)
            (chunkOf d i * d.chunk + (t * BK + (j : Nat)) - chunkOf d i * d.chunk))).addr
            = some (dtk.get1 j) := by rw [hdkind]; exact hdtr
        obtain ⟨-, -, hwfin, -⟩ :=
          wFinite (E := E) (d := d) (m₀ := m₁) (c := chunkOf d i) (h := headOf d i)
            (t := chunkOf d i * d.chunk + (t * BK + (j : Nat)))
            (by omega) hv0 hdksub hdtsub hv0f hdkf hdtf hv0n hdkn hD64 hr1 hexpacc
            (hsub (j, ())) (hdexp (j, ())) (hw (j, ()))
        have hmulres := mul_zero_left hwfin
        have hb2'' : FloatV.NaNEq (FloatV.mul (FloatV.ofPat (0 : Nat)) (w.get1 j))
            (b2f.get2 j n) := by
          have h := hb2 (j, n, ())
          rw [hselW j n hsel] at h
          rw [show b32f.get (j, n, ()) = (FloatV.ofPat (0 : Nat) : F32) from hb32z] at h
          exact h
        obtain ⟨h0, hf⟩ := hfin_zero _ j n hb2'' hmulres.1 hmulres.2
        exact ⟨⟨fun hc' => absurd hc' hcf, fun _ => h0⟩, hf⟩
      · -- the sequence index does not match: the multiplier is zero
        have hself : (sq.eq (bc v1)).get (j, ()) = false := by
          cases h : (sq.eq (bc v1)).get (j, ())
          · rfl
          · exact absurd h hsel
        have hmulres := mul_zero_right hb32fin
        have hb2'' : FloatV.NaNEq (FloatV.mul (b32f.get2 j n) (FloatV.ofPat (0 : Nat)))
            (b2f.get2 j n) := by
          have h := hb2 (j, n, ())
          rw [hsel0 j n hself] at h
          exact h
        obtain ⟨h0, hf⟩ := hfin_zero _ j n hb2'' hmulres.1 hmulres.2
        exact ⟨⟨fun hc' => absurd hc' hcf, fun _ => h0⟩, hf⟩
  constructor
  · intro s n hs
    by_cases hsold : s < t * BK
    · obtain ⟨h1, h2⟩ := hbcl s n hsold
      exact ⟨fun hc' => by rw [hbagree s n hsold]; exact h1 hc',
        fun hc' => by rw [hbagree s n hsold]; exact h2 hc'⟩
    · have hjlt : s - t * BK < BK := by
        have hmul : (t + 1) * BK = t * BK + BK := by ring
        omega
      have hsj' : s = t * BK + (s - t * BK) := by omega
      obtain ⟨j, hsj⟩ : ∃ j : Fin BK, s = t * BK + (j : Nat) :=
        ⟨⟨s - t * BK, hjlt⟩, by simpa using hsj'⟩
      rw [hsj, hbnew j n]
      exact (hmain j n).1
  · intro j n
    exact (hmain j n).2


/-! ## AccClause -/

/-!
# The accumulator clause after one loop iteration

`Inv` says the accumulator is finite and within `DotLoop.bound` of the exact partial sums over the
first `t` steps.  This file shows that after the `tl.dot_add` of step `t` (whose operands are the
`ℚ` tables `xh'`/`bh'`), the new accumulator satisfies the same clause for `t + 1` steps.  The
room hypothesis of `accStep` comes from the claim's `dotMag ≤ 2^120`.
-/

open TVA

set_option maxHeartbeats 800000

/-- An index of an earlier chunk is below the current chunk's start. -/
private lemma idx_lt_BK' {t t' : Nat} (k : Fin BK) (ht' : t' < t) :
    t' * BK + (k : Nat) < t * BK := by
  have hk : (k : Nat) < BK := k.isLt
  have h1 : (t' + 1) * BK ≤ t * BK := Nat.mul_le_mul_right BK (by omega)
  have h2 : t' * BK + (k : Nat) < (t' + 1) * BK := by
    rw [Nat.succ_mul]
    omega
  omega

/-- The index of a step of `T` iterations is below `T * BK`. -/
private lemma step_idx_lt {T t' : Nat} (k : Fin BK) (ht' : t' < T) :
    t' * BK + (k : Nat) < T * BK := by
  have hk : (k : Nat) < BK := k.isLt
  have h1 : (t' + 1) * BK ≤ T * BK := Nat.mul_le_mul_right BK (by omega)
  have h2 : t' * BK + (k : Nat) < (t' + 1) * BK := by
    rw [Nat.succ_mul]
    omega
  omega

/-- If the `x` operand is zero on the first `T` steps, the lifted sums vanish. -/
private lemma sum_Af_zero_left {xh : Nat → Fin BM → ℚ} {bh : Nat → Fin BN → ℚ}
    {a : Fin BM} {n : Fin BN} {T : Nat}
    (h : ∀ s, s < T * BK → xh s a = 0) :
    ∑ t' ∈ Finset.range T, Af xh bh a n t' = 0 := by
  refine Finset.sum_eq_zero ?_
  intro t' ht'
  rw [Finset.mem_range] at ht'
  unfold Af
  refine Finset.sum_eq_zero ?_
  intro k _
  rw [h (t' * BK + (k : Nat)) (step_idx_lt k ht'), Device.liftMag_zero .bf16, zero_mul]

/-- If the `b` operand is zero on the first `T` steps, the lifted sums vanish. -/
private lemma sum_Af_zero_right {xh : Nat → Fin BM → ℚ} {bh : Nat → Fin BN → ℚ}
    {a : Fin BM} {n : Fin BN} {T : Nat}
    (h : ∀ s, s < T * BK → bh s n = 0) :
    ∑ t' ∈ Finset.range T, Af xh bh a n t' = 0 := by
  refine Finset.sum_eq_zero ?_
  intro t' ht'
  rw [Finset.mem_range] at ht'
  unfold Af
  refine Finset.sum_eq_zero ?_
  intro k _
  rw [h (t' * BK + (k : Nat)) (step_idx_lt k ht'), Device.liftMag_zero .bf16, mul_zero]

/-- The lifted sums dominate the absolute value of the exact sums. -/
private lemma abs_Sf_sum_le {xh : Nat → Fin BM → ℚ} {bh : Nat → Fin BN → ℚ}
    (a : Fin BM) (n : Fin BN) (t : Nat) :
    |∑ t' ∈ Finset.range t, Sf xh bh a n t'| ≤ ∑ t' ∈ Finset.range t, Af xh bh a n t' := by
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum ?_)
  intro t' _
  unfold Sf Af
  exact (Finset.abs_sum_le_sum_abs _ _).trans
    (Finset.sum_le_sum fun k _ => Device.abs_mul_le_liftMag _ _ _ _)

/-- Extending the accumulator clause by one step of the loop. -/
theorem accClause {d : Inst} (hd : d.ok) {E : Env} {m₁ : Mem}
    {i : Fin (launch d).size} {t : Nat}
    {xh xh' : Nat → Fin BM → ℚ} {bh bh' : Nat → Fin BN → ℚ}
    {xt : Tensor [BM, BK] DType.bf16.Val} {bt : Tensor [BK, BN] DType.bf16.Val}
    {acc r : Tensor [BM, BN] DType.f32.Val}
    (hV : Vals m₁ d)
    (hdotacc : Device.DotBF16 E.P)
    (hxagree : ∀ (s : Nat) (a : Fin BM), s < t * BK → xh' s a = xh s a)
    (hbagree : ∀ (s : Nat) (n : Fin BN), s < t * BK → bh' s n = bh s n)
    (hxnew : ∀ (a : Fin BM) (k : Fin BK), xh' (t * BK + (k : Nat)) a = (xt.get2 a k).toRat)
    (hbnew : ∀ (k : Fin BK) (n : Fin BN), bh' (t * BK + (k : Nat)) n = (bt.get2 k n).toRat)
    (hxtfin : ∀ (a : Fin BM) (k : Fin BK), (xt.get2 a k).isFinite = true)
    (hbtfin : ∀ (k : Fin BK) (n : Fin BN), (bt.get2 k n).isFinite = true)
    (hr : E.P.dot dotCfg xt bt acc r ∨ ∃ d, E.P.dot dotCfg xt bt
      (Tensor.full (FloatV.ofPat 0)) d ∧
      ∀ i, AccAdd E.fusion DType.f32 (d.get i) (acc.get i) (r.get i))
    (hacc : (∀ (a : Fin BM) (n : Fin BN), (acc.get2 a n).isFinite = true) ∧
      (∀ (a : Fin BM) (n : Fin BN),
        |(((acc.get2 a n).toRat : ℚ) : ℝ) -
            ((∑ t' ∈ Finset.range t, Sf xh bh a n t' : ℚ) : ℝ)| ≤
          ((DotLoop.bound rho alp t (∑ t' ∈ Finset.range t, Af xh bh a n t') : ℚ) : ℝ)))
    (hxcl' : ∀ (s : Nat) (a : Fin BM), s < (t + 1) * BK →
      ((chunkOf d i * d.chunk + s ≤ endTok d (chunkOf d i) ∧
          pmOf d i * BM + (a : Nat) < d.headdim →
        (xh' s a : ℝ) = xV m₁ d (chunkOf d i * d.chunk + s) (headOf d i)
          (pmOf d i * BM + (a : Nat))) ∧
       (¬(chunkOf d i * d.chunk + s ≤ endTok d (chunkOf d i) ∧
          pmOf d i * BM + (a : Nat) < d.headdim) → xh' s a = 0)))
    (hbcl' : ∀ (s : Nat) (n : Fin BN), s < (t + 1) * BK →
      (((chunkOf d i * d.chunk + s) ∈ toks d (chunkOf d i) ∧
          pnOf d i * BN + (n : Nat) < d.dstate →
        |(bh' s n : ℝ) - bV m₁ d (chunkOf d i * d.chunk + s) (headOf d i)
            (pnOf d i * BN + (n : Nat)) *
            wRef m₁ d (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + s)|
          ≤ errB m₁ d (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + s)
            (pnOf d i * BN + (n : Nat))) ∧
       (¬((chunkOf d i * d.chunk + s) ∈ toks d (chunkOf d i) ∧
          pnOf d i * BN + (n : Nat) < d.dstate) → bh' s n = 0)))
    (ht1 : t + 1 ≤ steps d (chunkOf d i)) :
    (∀ (a : Fin BM) (n : Fin BN), (r.get2 a n).isFinite = true) ∧
    (∀ (a : Fin BM) (n : Fin BN),
      |(((r.get2 a n).toRat : ℚ) : ℝ) -
          ((∑ t' ∈ Finset.range (t + 1), Sf xh' bh' a n t' : ℚ) : ℝ)| ≤
        ((DotLoop.bound rho alp (t + 1)
          (∑ t' ∈ Finset.range (t + 1), Af xh' bh' a n t') : ℚ) : ℝ)) := by
  -- the accumulator clause, in `ℚ`
  have haccQ : ∀ (a : Fin BM) (n : Fin BN),
      |(acc.get2 a n).toRat - ∑ t' ∈ Finset.range t, Sf xh bh a n t'| ≤
        DotLoop.bound rho alp t (∑ t' ∈ Finset.range t, Af xh bh a n t') := by
    intro a n
    exact_mod_cast hacc.2 a n
  -- the `ℚ` values of this step's operands
  have hva : ∀ (a : Fin BM) (k : Fin BK),
      DType.bf16.val? (xt.get2 a k) = some (xh' (t * BK + (k : Nat)) a) := by
    intro a k
    rw [DType.val?_bf16]
    exact ⟨hxtfin a k, (hxnew a k).symm⟩
  have hvb : ∀ (k : Fin BK) (n : Fin BN),
      DType.bf16.val? (bt.get2 k n) = some (bh' (t * BK + (k : Nat)) n) := by
    intro k n
    rw [DType.val?_bf16]
    exact ⟨hbtfin k n, (hbnew k n).symm⟩
  -- the loop runs at most 1024 times
  have ht1024 : t ≤ 1024 := by
    have h := steps_le hd (chunkOf_lt hd (i : Nat))
    omega
  have htq : (t : ℚ) * rho ≤ 1 / 2 := t_mul_rho_le_half ht1024
  -- room for the accumulation step
  have hbig : ∀ (a : Fin BM) (n : Fin BN),
      |(acc.get2 a n).toRat| + ∑ k : Fin BK,
          Device.liftMag .bf16 (xh' (t * BK + (k : Nat)) a) *
            Device.liftMag .bf16 (bh' (t * BK + (k : Nat)) n) ≤ 2 ^ (121 : ℤ) := by
    intro a n
    -- the lifted sums over `t + 1` steps are the earlier ones plus this step's
    have hAsum : (∑ t' ∈ Finset.range (t + 1), Af xh' bh' a n t') =
        (∑ t' ∈ Finset.range t, Af xh bh a n t') +
          ∑ k : Fin BK, Device.liftMag .bf16 (xh' (t * BK + (k : Nat)) a) *
            Device.liftMag .bf16 (bh' (t * BK + (k : Nat)) n) := by
      rw [Finset.sum_range_succ]
      have hpre : (∑ t' ∈ Finset.range t, Af xh' bh' a n t')
          = ∑ t' ∈ Finset.range t, Af xh bh a n t' := by
        refine Finset.sum_congr rfl ?_
        intro t' ht'
        rw [Finset.mem_range] at ht'
        unfold Af
        refine Finset.sum_congr rfl ?_
        intro k _
        rw [hxagree (t' * BK + (k : Nat)) a (idx_lt_BK' k ht'),
          hbagree (t' * BK + (k : Nat)) n (idx_lt_BK' k ht')]
      rw [hpre]
      rfl
    -- the lifted sum over `t + 1` steps is at most `2^120`
    have hstep : (∑ t' ∈ Finset.range (t + 1), Af xh' bh' a n t') ≤ 2 ^ (120 : ℤ) := by
      by_cases hp : pmOf d i * BM + (a : Nat) < d.headdim
      · by_cases hn : pnOf d i * BN + (n : Nat) < d.dstate
        · -- both indices are inside the tile: the claim's `dotMag ≤ 2^120`
          have hb := aSum_le_dotMag (d := d) (init := m₁) (i := i) hd (t := t + 1) ht1
            xh' bh' hxcl' hbcl' hV a n
          have hdm := hV.2.2.2.2 (chunkOf d i) (headOf d i) (pmOf d i * BM + (a : Nat))
            (pnOf d i * BN + (n : Nat)) (chunkOf_lt hd (i : Nat)) (headOf_lt hd i.isLt) hp hn
          have hR : ((∑ t' ∈ Finset.range (t + 1), Af xh' bh' a n t' : ℚ) : ℝ) ≤
              ((2 : ℚ) ^ (120 : ℤ) : ℝ) := by
            have h2 : ((2 : ℚ) ^ (120 : ℤ) : ℝ) = (2 : ℝ) ^ (120 : ℤ) := by
              push_cast
              norm_num
            rw [h2]
            exact hb.trans hdm
          exact_mod_cast hR
        · -- the `n` index is outside the tile: the `b` operand is zero
          have hz : ∀ s, s < (t + 1) * BK → bh' s n = 0 := by
            intro s hs
            exact (hbcl' s n hs).2 (fun hh => hn hh.2)
          rw [sum_Af_zero_right hz]
          norm_num
      · -- the `a` index is outside the tile: the `x` operand is zero
        have hz : ∀ s, s < (t + 1) * BK → xh' s a = 0 := by
          intro s hs
          exact (hxcl' s a hs).2 (fun hh => hp hh.2)
        rw [sum_Af_zero_left hz]
        norm_num
    -- `|St| ≤ Bt`
    have hSt := abs_Sf_sum_le (xh := xh) (bh := bh) a n t
    -- the magnitudes of this step are nonnegative
    have hA0 : (0 : ℚ) ≤ ∑ k : Fin BK,
        Device.liftMag .bf16 (xh' (t * BK + (k : Nat)) a) *
          Device.liftMag .bf16 (bh' (t * BK + (k : Nat)) n) :=
      Finset.sum_nonneg fun k _ =>
        mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
    have hBA : (∑ t' ∈ Finset.range t, Af xh bh a n t') +
        ∑ k : Fin BK, Device.liftMag .bf16 (xh' (t * BK + (k : Nat)) a) *
          Device.liftMag .bf16 (bh' (t * BK + (k : Nat)) n) ≤ 2 ^ (120 : ℤ) := by
      rw [← hAsum]
      exact hstep
    have hBt0 : (0 : ℚ) ≤ ∑ t' ∈ Finset.range t, Af xh bh a n t' := by
      refine Finset.sum_nonneg ?_
      intro t' _
      unfold Af
      exact Finset.sum_nonneg fun k _ =>
        mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
    have hbnd : DotLoop.bound rho alp t (∑ t' ∈ Finset.range t, Af xh bh a n t')
        ≤ 2 ^ (120 : ℤ) :=
      bound_le_pow120 ht1024 hBt0 (by linarith)
    exact abs_add_le_pow121 (haccQ a n) hSt hBA hbnd hA0
  -- one accumulation step
  have hres := accStep (E := E) (t := t) hdotacc (xh := xh) (bh := bh) (xh' := xh')
    (bh' := bh') hxagree hbagree (acc := acc) (r := r) ⟨hacc.1, haccQ⟩ hr hva hvb htq hbig
  exact ⟨hres.1, fun a n => by exact_mod_cast hres.2 a n⟩


end Claims.SsdChunkState

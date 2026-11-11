import Claim
import Proof.Basic
import Proof.Index
import Proof.Lanes
import Proof.Value

/-!
# `attn_bwd_pre_f16`: the launch and the safety of one program

The launch is valid, each program reads rows of `O` and `DO` and writes its own row of `Delta`,
and running it is safe with the read set and the write set of that program.
-/

namespace Claims.AttnBwdPre

open TVA


/-! ### The launch is valid -/

theorem tritonOk_nil : Shape.TritonOk [] :=
  ⟨by intro k hk; exact absurd hk List.not_mem_nil, by decide⟩

/-- The shape `[1]` is accepted by Triton. -/
theorem tritonOk_one : Shape.TritonOk [1] :=
  ⟨by
    intro k hk
    rw [List.mem_singleton] at hk
    subst hk
    exact ⟨0, rfl⟩, by decide⟩

/-- A power-of-two dimension in front of an accepted shape, within `2^20` elements, is accepted. -/
theorem tritonOk_cons {a : Nat} {s : Shape} (ha : ∃ e, a = 2 ^ e) (hs : Shape.TritonOk s)
    (h : (a :: s).prod ≤ 2 ^ 20) : Shape.TritonOk (a :: s) :=
  ⟨fun k hk => by
    rcases List.mem_cons.mp hk with rfl | hk
    · exact ha
    · exact hs.1 k hk,
   h⟩

/-- **The launch is valid** on every covered instance. -/
theorem launch_valid (d : Inst) (hd : d.ok) : (launch d).Valid := by
  have hHD : d.HEAD_DIM ≤ 2 ^ 20 := hd.head_le.trans (by norm_num)
  have hBM : d.BLOCK_M ≤ 2 ^ 20 := by
    calc d.BLOCK_M = d.BLOCK_M * 1 := (Nat.mul_one _).symm
      _ ≤ d.BLOCK_M * d.HEAD_DIM := Nat.mul_le_mul_left _ (head_pos hd)
      _ ≤ 2 ^ 20 := hd.block_le
  refine ⟨?_, ?_, ?_, ?_, rfl, rfl, rfl, ?_⟩
  · show d.N_CTX / d.BLOCK_M < 2 ^ 31
    exact lt_of_le_of_lt (Nat.div_le_self _ _) hd.ctx_lt
  · show d.Z * d.H < 2 ^ 16
    exact hd.bh_lt
  · show (1 : Nat) < 2 ^ 16
    norm_num
  · rw [launch_size]
    show gx d * gy d < 2 ^ 31
    simpa [gx, gy] using hd.grid_lt
  · intro s hs
    simp only [launch, List.mem_cons, List.not_mem_nil, or_false] at hs
    rcases hs with rfl | rfl | rfl | rfl | rfl
    · exact tritonOk_cons (s := []) hd.block_pow2 tritonOk_nil (by simpa using hBM)
    · exact tritonOk_cons (s := []) hd.head_pow2 tritonOk_nil (by simpa using hHD)
    · exact tritonOk_cons (s := [1]) hd.block_pow2 tritonOk_one (by simpa using hBM)
    · exact tritonOk_cons (s := [d.HEAD_DIM]) ⟨0, rfl⟩
        (tritonOk_cons (s := []) hd.head_pow2 tritonOk_nil (by simpa using hHD))
        (by simpa using hHD)
    · exact tritonOk_cons (s := [d.HEAD_DIM]) hd.block_pow2
        (tritonOk_cons (s := []) hd.head_pow2 tritonOk_nil (by simpa using hHD))
        (by simpa using hd.block_le)

/-! ### The lanes as pointers -/

theorem ptrO_get' (d : Inst) (hd : d.ok) (i : Fin (launch d).size)
    (p : Idx [d.BLOCK_M, d.HEAD_DIM]) :
    (ptrO d i).get p = d.O.ptr ((readIdx d i p : Nat) : Int) := by
  obtain ⟨l, k, u⟩ := p
  exact ptrO_get d hd i l k

/-- Lane `p` of the `DO` pointers addresses element `readIdx i p` of `DO`. -/
theorem ptrDO_get' (d : Inst) (hd : d.ok) (i : Fin (launch d).size)
    (p : Idx [d.BLOCK_M, d.HEAD_DIM]) :
    (ptrDO d i).get p = d.DO.ptr ((readIdx d i p : Nat) : Int) := by
  obtain ⟨l, k, u⟩ := p
  exact ptrDO_get d hd i l k

/-- Lane `p` of the `Delta` pointers addresses element `writeIdx i p` of `Delta`. -/
theorem ptrD_get' (d : Inst) (hd : d.ok) (i : Fin (launch d).size)
    (p : Idx [d.BLOCK_M]) :
    (ptrD d i).get p = d.Delta.ptr ((writeIdx d i p : Nat) : Int) := by
  obtain ⟨l, u⟩ := p
  exact ptrD_get d hd i l

/-- The `O`/`DO` element of lane `(l, k)` is column `k` of the lane's row. -/
theorem readIdx_writeRow (d : Inst) (i : Nat) (l : Fin d.BLOCK_M) (k : Fin d.HEAD_DIM) :
    readIdx d i (l, (k, ())) = writeRow d i l * d.HEAD_DIM + k.val := by
  rw [readIdx_val, writeRow_val]

/-- The `Delta` row of lane `l` of program `i`, taken modulo `BLOCK_M`, is `l`. -/
theorem writeIdx_mod (d : Inst) (hd : d.ok) (i : Nat) (l : Fin d.BLOCK_M) :
    writeIdx d i (l, ()) % d.BLOCK_M = l.val := by
  have hN : d.N_CTX = d.BLOCK_M * gx d := (block_mul_gx d hd).symm
  have hw : writeIdx d i (l, ()) = bh d i * d.N_CTX + pm d i * d.BLOCK_M + l.val := rfl
  rw [hw, hN,
    show bh d i * (d.BLOCK_M * gx d) + pm d i * d.BLOCK_M + l.val =
      l.val + d.BLOCK_M * (bh d i * gx d + pm d i) by ring,
    Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt l.isLt]

/-- Two lanes writing the same `Delta` row are the same lane. -/
theorem writeIdx_inj (d : Inst) (hd : d.ok) (i : Nat) :
    Function.Injective (writeIdx d i) := by
  intro p q h
  obtain ⟨l, u⟩ := p
  obtain ⟨l', u'⟩ := q
  cases u
  cases u'
  have h1 : l.val = l'.val := by
    rw [← writeIdx_mod d hd i l, ← writeIdx_mod d hd i l', h]
  have hl : l = l' := Fin.ext h1
  subst hl
  rfl

/-- A `float16` widened by a conversion is related to it by `ConvRel`. -/
theorem convRel_cvt {P : Platform} {x : F16} {y : F32}
    (h : FloatV.NaNEq (FloatV.cvt x) y) : ConvRel P .f16 .f32 x y := by
  have hne : (Format.binary16 : Format) ≠ Format.binary32 := by
    decide
  simp only [ConvRel, ofFloatRel, FloatRel, hne]
  exact Or.inl ⟨Or.inl rfl, Or.inr (Or.inr (Or.inl rfl)), h⟩

/-! ### The read set, the write set and what one program establishes -/

/-- The bytes program `i` reads. -/
def R (d : Inst) (_ : Mem) (i : Nat) : Set Int :=
  d.O.elems (readSet d i) ∪ d.DO.elems (readSet d i)

/-- The bytes program `i` writes. -/
def W (d : Inst) (_ : Mem) (i : Nat) : Set Int := d.Delta.elems (writeSet d i)

/-- Every row program `i` writes holds the value the claim asks for. -/
def Q (d : Inst) (init : Mem) (i : Nat) (m : Mem) : Prop :=
  ∀ l : Fin d.BLOCK_M, OutSpec d init m (writeRow d i l)

/-! ### The program is safe -/

theorem prog_safe {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    (hSD : Device.SumDepth E.P) (i : Fin (launch d).size) :
    Safe E (R d init i) (W d init i) (fun _ m => Q d init i m) init
      ((model d).body ((launch d).ctx i)) := by
  obtain ⟨hOv, hDOv, hDv⟩ := hpre
  rw [body_eq d (i : Nat)]
  refine safe_load_buf (b := d.O) (g := readIdx d i) (en := fun _ => true)
    (Buf.addrs_of_get_all d.O (fun p => ptrO_get' d hd i p)) ?_ hOv ?_ ?_
  · intro p _
    exact readIdx_lt_O d hd i p.1 p.2.1
  · rw [img_readSet]
    intro x hx
    exact Or.inl hx
  · intro vo hvo _
    refine safe_load_buf (b := d.DO) (g := readIdx d i) (en := fun _ => true)
      (Buf.addrs_of_get_all d.DO (fun p => ptrDO_get' d hd i p)) ?_ hDOv ?_ ?_
    · intro p _
      exact readIdx_lt_DO d hd i p.1 p.2.1
    · rw [img_readSet]
      intro x hx
      exact Or.inr hx
    · intro vdo hvdo _
      refine safe_cvt (g := .binary32)
        ⟨Or.inl rfl, Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩ ?_
      intro doc hdoc
      refine safe_convert (d := .f16) (d' := .f32) (x := vo) ?_ ?_
      · intro p
        refine ⟨FloatV.cvt (vo.get p), ?_⟩
        have hne : (Format.binary16 : Format) ≠
            Format.binary32 := by decide
        simp only [ConvRel, ofFloatRel, FloatRel, hne]
        exact Or.inl ⟨Or.inl rfl, Or.inr (Or.inr (Or.inl rfl)), FloatV.NaNEq.refl _⟩
      · intro oc hoc
        refine safe_mul ?_
        intro prod hprod
        refine safe_sum1c (head_pos hd) ?_
        intro delta hdelta
        have hb : tl.store (ptrD d i) delta >>= (fun () => .ret ()) = tl.store (ptrD d i) delta :=
          rfl
        rw [← hb]
        refine safe_store_buf (b := d.Delta) (g := writeIdx d i) (en := fun _ => true)
          (k := fun () => .ret ())
          (Buf.addrs_of_get_all d.Delta (fun p => ptrD_get' d hd i p)) ?_ ?_ hDv ?_ ?_
        · intro p _
          exact writeRow_lt_Delta d hd i p.1
        · intro p q _ _ hpq
          exact writeIdx_inj d hd i hpq
        · rw [img_writeSet]
          intro x hx
          exact hx
        · intro m' hread _ _
          refine Safe.ret ?_
          intro l
          obtain ⟨t, hcov, hdepB, ht⟩ := hdelta l
          have hdep : t.depth ≤ Nat.clog 2 d.HEAD_DIM :=
            hdepB.trans (max_le (hSD _) le_rfl)
          refine ⟨delta.get (l, ()), hread (l, ()) rfl, ?_⟩
          intro wo wdo hrowwo hrowwdo
          let a : Nat → F32 := fun k =>
            if h : k < d.HEAD_DIM then oc.get (l, (⟨k, h⟩, ())) else default
          let b : Nat → F32 := fun k =>
            if h : k < d.HEAD_DIM then doc.get (l, (⟨k, h⟩, ())) else default
          let pp : Nat → F32 := fun k =>
            if h : k < d.HEAD_DIM then prod.get (l, (⟨k, h⟩, ())) else default
          have ha : ∀ k, k < d.HEAD_DIM → ConvRel E.P .f16 .f32 (wo k) (a k) := by
            intro k hk
            have h1 := hoc (l, (⟨k, hk⟩, ()))
            have hlv := hvo (l, (⟨k, hk⟩, ())) rfl
            rw [readIdx_writeRow] at hlv
            have hv : vo.get (l, (⟨k, hk⟩, ())) = wo k :=
              Option.some.inj (hlv.symm.trans (hrowwo k hk).1)
            rw [hv] at h1
            have hif : a k = oc.get (l, (⟨k, hk⟩, ())) := by
              show (if h : k < d.HEAD_DIM then oc.get (l, (⟨k, h⟩, ())) else default) = _
              simp [hk]
            rw [hif]
            exact h1
          have hbb : ∀ k, k < d.HEAD_DIM → ConvRel E.P .f16 .f32 (wdo k) (b k) := by
            intro k hk
            have h1 := hdoc (l, (⟨k, hk⟩, ()))
            rw [FloatV.cvtWith_rn] at h1
            have hcr := convRel_cvt (P := E.P) h1
            have hlv := hvdo (l, (⟨k, hk⟩, ())) rfl
            rw [readIdx_writeRow] at hlv
            have hv : vdo.get (l, (⟨k, hk⟩, ())) = wdo k :=
              Option.some.inj (hlv.symm.trans (hrowwdo k hk).1)
            rw [hv] at hcr
            have hif : b k = doc.get (l, (⟨k, hk⟩, ())) := by
              show (if h : k < d.HEAD_DIM then doc.get (l, (⟨k, h⟩, ())) else default) = _
              simp [hk]
            rw [hif]
            exact hcr
          have hp : ∀ k, k < d.HEAD_DIM → tl.MulRel .f32 (a k) (b k) (pp k) := by
            intro k hk
            have h1 := hprod (l, (⟨k, hk⟩, ()))
            have ha' : a k = oc.get (l, (⟨k, hk⟩, ())) := by
              show (if h : k < d.HEAD_DIM then oc.get (l, (⟨k, h⟩, ())) else default) = _
              simp [hk]
            have hb' : b k = doc.get (l, (⟨k, hk⟩, ())) := by
              show (if h : k < d.HEAD_DIM then doc.get (l, (⟨k, h⟩, ())) else default) = _
              simp [hk]
            have hpp : pp k = prod.get (l, (⟨k, hk⟩, ())) := by
              show (if h : k < d.HEAD_DIM then prod.get (l, (⟨k, h⟩, ())) else default) = _
              simp [hk]
            show FloatV.NaNEq (FloatV.mul (a k) (b k)) (pp k)
            rw [ha', hb', hpp]
            exact h1
          have ht' : t.Eval (AccAdd E.fusion .f32) pp (delta.get (l, ())) := by
            have hacc : AccAdd E.fusion DType.f32 = (AddRel E.fusion : F32 → F32 → F32 → Prop) :=
              rfl
            rw [hacc]
            exact ht
          exact row_value (P := E.P) (fusion := E.fusion) (K := d.HEAD_DIM) hd.head_le
            (vo := wo) (vdo := wdo) (a := a) (b := b) (p := pp) (t := t)
            (r := delta.get (l, ()))
            (fun k hk => (hrowwo k hk).2) (fun k hk => (hrowwdo k hk).2)
            ha hbb hp hcov hdep ht'

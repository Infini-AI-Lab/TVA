import Proof.Basic

/-!
# `write_req_to_token_pool_i32`: one trip of a copy loop

Both write loops of the kernel (source lines 77-85 and 93-104) have the same body: the offsets
`tl.arange(0, 512) + i * 512` in `int64`, masked by `offset < lim`, a load from a source pointer
plus the offsets, the `int32` cast, and a store to a table pointer plus the offsets. `tripProg`
is that body with the two pointer tensors as functions of the offsets; `safe_trip` proves one trip
for a source buffer `S` read from element `s0` and the table written from element `t0`.
-/

namespace Claims.WriteReqToTokenPool

open TVA

/-- The body of a copy loop (source lines 78-85, and 94-104). -/
def tripProg (lim i : I64) (srcOf : Tensor [512] I64 → Tensor [512] (Ptr .i64))
    (dstOf : Tensor [512] I64 → Tensor [512] (Ptr .i32)) : Prog Unit := do
  let a64 ← tl.convert .i32 .i64 (tl.arange 0 512)
  let offset : Tensor [512] I64 := a64 + i * (IntV.ofNat 512 : I64)
  let mask := offset.ltS lim
  let value : Tensor [512] I64 ← tl.load (srcOf offset) (some mask)
  let v32 ← tl.convert .i64 .i32 value
  tl.store (dstOf offset) v32 (some mask)

/-- Lane `l` of a `[512]` block. -/
abbrev lane (l : Nat) (hl : l < 512) : Idx [512] := (⟨l, hl⟩, ())

/-- **One trip is safe.** Trip `t` (`i = t`) of a copy loop over `L` elements reads the source
elements `s0 + t·512 + l` and writes the table elements `t0 + t·512 + l` (`l < 512`,
`t·512 + l < L`), each the cast of the source's initial value; nothing else changes. -/
theorem safe_trip {d : Inst} {E : Env} {R : Set Int} {init m : Mem} (S : Buf .i64)
    {s0 t0 L : Int} {lim i : I64} (hlim : lim.toInt = L) {t : Nat} (hi : i.toInt = t)
    (hL : L ≤ 2 ^ 22) (ht : (t : Int) * 512 < L)
    {srcOf : Tensor [512] I64 → Tensor [512] (Ptr .i64)}
    {dstOf : Tensor [512] I64 → Tensor [512] (Ptr .i32)}
    (hsrc : ∀ off l, (srcOf off).get l = S.ptr (s0 + (off.get l).toInt))
    (hdst : ∀ off l, (dstOf off).get l = d.req_to_token.ptr (t0 + (off.get l).toInt))
    (hS : ∀ o : Nat, (o : Int) < L → 0 ≤ s0 + o ∧ s0 + o < S.len)
    (hT : ∀ o : Nat, (o : Int) < L → 0 ≤ t0 + o ∧ t0 + o < d.req_to_token.len)
    (hTS : Disjoint d.req_to_token.bytes S.bytes) (hSv : S.Valid init)
    (hR : S.elems (win s0 L) ⊆ R)
    (hfr : ∀ z, z ∉ d.req_to_token.bytes → m z = init z) (hTv : d.req_to_token.Valid m) :
    Safe E R (d.req_to_token.elems (tripWin t0 L t))
      (fun _ m' => (∀ l : Nat, l < 512 → (t : Int) * 512 + l < L →
          Copied d init m' S s0 t0 (t * 512 + l)) ∧
        (∀ z, z ∉ d.req_to_token.elems (tripWin t0 L t) → m' z = m z) ∧
        d.req_to_token.Valid m')
      m (tripProg lim i srcOf dstOf) := by
  have ht0 : (0 : Int) ≤ t := by positivity
  simp only [tripProg]
  refine safe_convert (d := .i32) (d' := .i64) (x := tl.arange 0 512) (fun _ => ⟨_, rfl⟩) ?_
  intro a64 ha64
  -- the lanes of the offsets: `t·512 + l`
  have ha : ∀ l : Idx [512], (a64.get l).toInt = ((l.1 : Nat) : Int) := by
    intro l
    have h := ConvRel.toInt_i32_i64 (ha64 l)
    rw [h]
    show (IntV.ofInt ((0 : Nat) + ((l.1 : Nat) : Int)) : I32).toInt = _
    have hl := l.1.isLt
    rw [IntV.I32.toInt_ofInt ⟨by omega, by omega⟩]
    simp
  have hiB : (i * (IntV.ofNat 512 : I64)).toInt = (t : Int) * 512 := by
    have h512 : (IntV.ofNat 512 : I64).toInt = 512 := IntV.I64.toInt_ofNat (by norm_num)
    rw [IntV.toInt_mul hi h512 (IntV.I64.sfits (by nlinarith) (by nlinarith))]
  have hoff : ∀ l : Idx [512], ((a64 + i * (IntV.ofNat 512 : I64)).get l).toInt =
      (t : Int) * 512 + ((l.1 : Nat) : Int) := by
    intro l
    have hl : ((l.1 : Nat) : Int) < 512 := by exact_mod_cast l.1.isLt
    have hl0 : (0 : Int) ≤ ((l.1 : Nat) : Int) := by positivity
    rw [Tensor.get_adds, IntV.toInt_add (ha l) hiB (IntV.I64.sfits (by nlinarith) (by nlinarith))]
    ring
  set offset := a64 + i * (IntV.ofNat 512 : I64) with hoffset
  -- the enabled lanes
  let en : Idx [512] → Bool := fun l => decide ((t : Int) * 512 + ((l.1 : Nat) : Int) < L)
  have hen : ∀ l : Idx [512], en l = true → (t : Int) * 512 + ((l.1 : Nat) : Int) < L :=
    fun l h => of_decide_eq_true h
  have hmask : ∀ l : Idx [512], (offset.ltS lim).get l = en l := by
    intro l
    rw [Tensor.get_ltS, IntV.lt_eq, hoff l, hlim]
  have hoN : ∀ l : Idx [512], en l = true →
      (((t * 512 + (l.1 : Nat) : Nat) : Int)) < L := by
    intro l h
    push_cast
    exact hen l h
  -- the load: source element `s0 + t·512 + l`
  let gL : Idx [512] → Nat := fun l => (s0 + ((t : Int) * 512 + ((l.1 : Nat) : Int))).toNat
  have hgL : ∀ l : Idx [512], en l = true →
      (gL l : Int) = s0 + ((t : Int) * 512 + ((l.1 : Nat) : Int)) := by
    intro l h
    have := (hS _ (hoN l h)).1
    push_cast at this
    exact Int.toNat_of_nonneg this
  have hinL : ∀ l : Idx [512], en l = true → gL l < S.len := by
    intro l h
    have h2 := (hS _ (hoN l h)).2
    push_cast at h2
    have := hgL l h
    omega
  have hptrL : ∀ l : Idx [512], en l = true → (srcOf offset).get l = S.ptr (gL l : Int) := by
    intro l h
    rw [hsrc, hoff l, hgL l h]
  have himgL : Buf.img gL en ⊆ win s0 L := by
    rintro e ⟨l, h, rfl⟩
    exact ⟨t * 512 + (l.1 : Nat), hoN l h, by rw [hgL l h]; push_cast; ring⟩
  have hSv' : S.Valid m := valid_init (d := d) S hTS hfr hSv
  refine safe_load_buf S (Buf.addrs_of_get S hmask hptrL) hinL hSv'
    ((S.elems_mono himgL).trans hR) ?_
  intro value hval _
  refine safe_convert (d := .i64) (d' := .i32) (x := value)
    (fun _ => ⟨_, ConvRel.iff_cast_i64_i32.mpr rfl⟩) ?_
  intro v32 hv32
  have hv32' : ∀ l, v32.get l = IntV.cast (value.get l) :=
    fun l => ConvRel.iff_cast_i64_i32.mp (hv32 l)
  -- the store: table element `t0 + t·512 + l`
  let gS : Idx [512] → Nat := fun l => (t0 + ((t : Int) * 512 + ((l.1 : Nat) : Int))).toNat
  have hgS : ∀ l : Idx [512], en l = true →
      (gS l : Int) = t0 + ((t : Int) * 512 + ((l.1 : Nat) : Int)) := by
    intro l h
    have := (hT _ (hoN l h)).1
    push_cast at this
    exact Int.toNat_of_nonneg this
  have hinS : ∀ l : Idx [512], en l = true → gS l < d.req_to_token.len := by
    intro l h
    have h2 := (hT _ (hoN l h)).2
    push_cast at h2
    have := hgS l h
    omega
  have hptrS : ∀ l : Idx [512], en l = true →
      (dstOf offset).get l = d.req_to_token.ptr (gS l : Int) := by
    intro l h
    rw [hdst, hoff l, hgS l h]
  have hinj : ∀ l l' : Idx [512], en l = true → en l' = true → gS l = gS l' → l = l' := by
    intro l l' h h' he
    have h1 := hgS l h
    have h2 := hgS l' h'
    have h3 : ((l.1 : Nat) : Int) = ((l'.1 : Nat) : Int) := by
      have : (gS l : Int) = (gS l' : Int) := by rw [he]
      linarith
    have h5 : (l.1 : Nat) = (l'.1 : Nat) := by exact_mod_cast h3
    have : Subsingleton (Idx []) := ⟨fun a b => by cases a; cases b; rfl⟩
    exact Prod.ext (Fin.ext h5) (Subsingleton.elim _ _)
  have himgS : Buf.img gS en = tripWin t0 L t := by
    ext e
    constructor
    · rintro ⟨l, h, rfl⟩
      exact ⟨(l.1 : Nat), l.1.isLt, hen l h, hgS l h⟩
    · rintro ⟨l, hl, hlt, he⟩
      have h : en (lane l hl) = true := decide_eq_true hlt
      exact ⟨lane l hl, h, by
        have := hgS (lane l hl) h
        have : (gS (lane l hl) : Int) = (e : Int) := by rw [this, he]
        exact_mod_cast this⟩
  refine safe_last ?_
  refine safe_store_buf d.req_to_token (Buf.addrs_of_get d.req_to_token hmask hptrS) hinS hinj hTv
    (by rw [himgS]) ?_
  intro m' hread hframe hpres
  rw [himgS] at hframe
  refine safe_ret ⟨?_, hframe, Buf.valid_of_agree hTv fun x hx => hpres x (hTv.2.2.2 x hx)⟩
  intro l hl hlt
  have h : en (lane l hl) = true := decide_eq_true hlt
  have e : (((t * 512 + l : Nat)) : Int) = (t : Int) * 512 + (l : Int) := by push_cast; ring
  refine ⟨value.get (lane l hl), ?_, ?_⟩
  · have h1 := hval (lane l hl) h
    rw [hgL _ h] at h1
    have h2 := (hS _ (hoN _ h)).2
    have h0 := (hS _ (hoN _ h)).1
    rw [e] at h0 h2
    rw [e, ← read_init (d := d) S hTS hfr h0 h2]
    exact h1
  · have h1 := hread (lane l hl) h
    rw [hgS _ h, hv32'] at h1
    rw [e]
    exact h1

end Claims.WriteReqToTokenPool

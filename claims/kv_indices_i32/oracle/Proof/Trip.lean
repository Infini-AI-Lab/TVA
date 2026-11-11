import Proof.Basic

/-!
# `kv_indices_i32`: one trip of the inner loop

The loop `for i in range(0, num_loop, 1)` of `create_flashinfer_kv_indices_triton` copies
`BLOCK = 512` entries of the table row of a request into `kv_indices`, masked by the request's
length. This file states and proves `safe_trip`, the program safety of one such trip.

The set of table entries trip `t` reads is `tabTripElems`; the entries of `kv_indices` it writes
are `tripElems` (`Proof/Basic.lean`).
-/

namespace Claims.KvIndices

open TVA

/-! ## Small helpers -/

/-- `b.ptr 0 + k` is `b.ptr k`: the pointer of element `k`. -/
theorem ptr_zero_add {d : DType} (b : Buf d) (k : Nat) (hk : k < 2 ^ 31) :
    (b.ptr 0 + (IntV.ofNat k : I32)).addr = (b.ptr (k : Int)).addr := by
  simp [Buf.ptr, Ptr.offset, IntV.I32.toInt_ofNat hk]

/-- An `int64` value converted from an `int64` or `int32` element. -/
theorem exists_convRel_i64 {ds : DType} (h : ds = .i64 ∨ ds = .i32) (P : Platform) (v : ds.Val) :
    ∃ r : I64, ConvRel P ds .i64 v r := by
  rcases h with rfl | rfl <;> exact ⟨_, rfl⟩

/-- An `int32` value converted from an `int64` or `int32` element. -/
theorem exists_convRel_i32 {ds : DType} (h : ds = .i64 ∨ ds = .i32) (P : Platform) (v : ds.Val) :
    ∃ r : I32, ConvRel P ds .i32 v r := by
  rcases h with rfl | rfl <;> exact ⟨_, rfl⟩

/-- The value of a converted `int64`, as `ivOf`. -/
theorem convRel_i64_ivOf {P : Platform} {ds : DType} (hds : ds = .i64 ∨ ds = .i32)
    {v : ds.Val} {r : I64} (h : ConvRel P ds .i64 v r) : r.toInt = ivOf v := by
  rcases hds with rfl | rfl
  · exact ConvRel.toInt_i64_i64 h
  · exact ConvRel.toInt_i32_i64 h

/-- The value of a converted `int32`, as `ivOf`, when it does not wrap. -/
theorem convRel_i32_ivOf {P : Platform} {ds : DType} (hds : ds = .i64 ∨ ds = .i32)
    {v : ds.Val} {r : I32} (h : ConvRel P ds .i32 v r) (hfit : IntV.SFits 32 (ivOf v)) :
    r.toInt = ivOf v := by
  rcases hds with rfl | rfl
  · exact ConvRel.toInt_i64_i32 h hfit
  · exact ConvRel.toInt_i32_i32 h

/-- `(x + s) - x = s` in the wrapped arithmetic, when `s` is a non-negative small value. -/
theorem sub_add_self_toInt {w : Nat} {x s : IntV true w} (hw : 0 < w)
    (h0 : 0 ≤ s.toInt) (h1 : s.toInt < 2 ^ (w - 1)) :
    ((x + s) - x).toInt = s.toInt := by
  have hto : ((x + s) - x).toInt = (((x.toInt + s.toInt).bmod (2 ^ w)) - x.toInt).bmod (2 ^ w) := by
    have e1 : x.toInt = x.bits.toInt := rfl
    have e2 : s.toInt = s.bits.toInt := rfl
    rw [e1, e2]
    show ((x.bits + s.bits) - x.bits).toInt
      = (((x.bits.toInt + s.bits.toInt).bmod (2 ^ w)) - x.bits.toInt).bmod (2 ^ w)
    rw [BitVec.toInt_sub, BitVec.toInt_add]
  rw [hto]
  have hB : (x.toInt + s.toInt).bmod (2 ^ w)
      = (x.toInt + s.toInt) - ((2 ^ w : Nat) : Int) * Int.bdiv (x.toInt + s.toInt) (2 ^ w) :=
    Int.bmod_eq_self_sub_mul_bdiv _ _
  have harg : (x.toInt + s.toInt).bmod (2 ^ w) - x.toInt
      = s.toInt - Int.bdiv (x.toInt + s.toInt) (2 ^ w) * ((2 ^ w : Nat) : Int) := by
    rw [hB]; ring
  rw [harg, Int.sub_mul_bmod_self_right]
  refine Int.bmod_eq_of_le_mul_two ?_ ?_
  · have : (0 : Int) ≤ 2 ^ w := by positivity
    linarith
  · have e : (((2 ^ w : Nat)) : Int) = 2 ^ (w - 1) * 2 := by
      push_cast
      rw [← pow_succ]; congr 1; omega
    linarith

/-- The row offset `req_pool_indices[b] * max_context_len` fits `int64`. -/
theorem tab_fits {d : Inst} (_hd : d.ok) {init : Mem} (hpre : Pre d init) {b : Nat}
    (hb : b < d.bs) (hslen : 0 < slen d init b) :
    IntV.SFits 64 (rpi d init b * (d.max_context_len : Int)) := by
  obtain ⟨hs0, hs1, hi0, hi1, hw⟩ := hpre.2.2.2.2.2.2.1 b hb
  obtain ⟨hcl, hci⟩ := hpre.2.2.2.2.2.2.2.1 b hb
  obtain ⟨hb0, hb1, hb2, hb3⟩ := hpre.1
  have hb4 : (DType.bytes .i32 : Int) = 4 := by simp [DType.bytes]
  rw [hb4] at hb1
  have hlen : (d.req_to_token.len : Int) ≤ 2 ^ 62 := by linarith
  have h0s : ((0 : Nat) : Int) < slen d init b := by exact_mod_cast hslen
  have hupper : rpi d init b * (d.max_context_len : Int) < 2 ^ 62 := by
    linarith [(hw 0 h0s).2]
  have hlower : -(2 ^ 63 : Int) ≤ rpi d init b * (d.max_context_len : Int) := by
    linarith [(hw 0 h0s).1]
  have e : ((2 : Int) ^ (64 - 1 : Nat)) = 2 ^ 63 := by norm_num
  simp only [IntV.SFits, e]
  constructor <;> linarith

/-! ## The table entries one trip reads -/

/-- The elements of `req_to_token` that trip `t` of program `b` reads. -/
def tabTripElems (d : Inst) (init : Mem) (b t : Nat) : Set Nat :=
  {j | ∃ l : Nat, l < 512 ∧ (t : Int) * 512 + (l : Int) < slen d init b ∧
    (j : Int) = rpi d init b * (d.max_context_len : Int) + sidx d init b +
      ((t : Int) * 512 + (l : Int))}

/-- The table elements of one trip lie in the request's window. -/
theorem tabTripElems_sub (d : Inst) (init : Mem) (b t : Nat) :
    tabTripElems d init b t ⊆ tabElems d init b := by
  rintro j ⟨l, hl, hlt, hj⟩
  exact ⟨t * 512 + l, hlt, hj⟩

/-- The output elements of one trip lie in the request's window. -/
theorem tripElems_sub (d : Inst) (init : Mem) (b t : Nat) :
    tripElems d init b t ⊆ outElems d init b := by
  rintro j ⟨l, hl, hlt, hj⟩
  refine ⟨t * 512 + l, hlt, ?_⟩
  push_cast
  linarith

/-! ## One trip -/

/-- The body of the inner loop: copy the 512 entries of block `i` of the request's row to
`kv_indices`. `p` is `req_pool_indices[b]`, `x` is `kv_start` (the request's first position),
`s` is `seq_lens[b]` and `off` is `kv_indptr[b]`. -/
def tripProg {sg : Bool} {w : Nat} (d : Inst) (p : I64) (x s i : I32) (off : IntV sg w) :
    Prog Unit := do
  let iB := i * (IntV.ofNat 512 : I32)
  let i64 := (← tl.convert .i32 .i64 (Tensor.scalar iB)).item
  let a64 ← tl.convert .i32 .i64 (tl.arange 0 512)
  let offset : Tensor [512] I64 := a64 + i64
  let lim : I64 := (← tl.convert .i32 .i64 (Tensor.scalar (x + s - x))).item
  let mask := offset.ltS lim
  let rsp : I64 := (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat d.max_context_len))).item
  let data : Tensor [512] I32 ← tl.load
    (d.req_to_token.ptr 0 + p * rsp + x + offset) (some mask)
  tl.store (d.kv_indices.ptr 0 + off + offset) data (some mask)

/-- **One trip is safe.** It reads the request's table entry of each of the 512 positions
`t * 512, …, t * 512 + 511` (masked by `slen`) and writes them to `kv_indices`, extending
`Qpart` by one trip. -/
theorem safe_trip {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    (b : Nat) (hb : b < d.bs) {sg : Bool} {w : Nat} (p : I64) (x s i : I32) (off : IntV sg w)
    (hp : p.toInt = rpi d init b) (hx : x.toInt = sidx d init b)
    (hs : s.toInt = slen d init b) (hoff : off.toInt = ind d init b)
    (t : Nat) (hi : i.toInt = (t : Int)) (ht : (t : Int) * 512 < 2 ^ 31) (m : Mem)
    (hQ : Qpart d init b t m)
    (hT : ∀ z ∈ d.req_to_token.bytes, m z = init z)
    (hVi : d.kv_indices.Valid m) (hVt : d.req_to_token.Valid m) :
    Safe E (R d init b) (d.kv_indices.elems (tripElems d init b t))
      (fun _ m' => Qpart d init b (t + 1) m' ∧
        (∀ z ∈ d.req_to_token.bytes, m' z = init z) ∧
        d.kv_indices.Valid m' ∧ d.req_to_token.Valid m')
      m (tripProg d p x s i off) := by
  simp only [tripProg]
  refine safe_convert (d := .i32) (d' := .i64)
    (x := Tensor.scalar (i * (IntV.ofNat 512 : I32))) ?_ ?_
  · intro u
    exact exists_convRel_i64 (Or.inr rfl) E.P _
  · intro i64 hi64
    have hiB : i64.item.toInt = (t : Int) * 512 := by
      have h := hi64 ()
      have hv : i64.item.toInt = (i * (IntV.ofNat 512 : I32)).toInt := by
        simpa only [Tensor.scalar, Tensor.item] using ConvRel.toInt_i32_i64 h
      rw [hv]
      have h512 : (IntV.ofNat 512 : I32).toInt = 512 := IntV.I32.toInt_ofNat (by norm_num)
      have hsfit : IntV.SFits 32 (i.toInt * (IntV.ofNat 512 : I32).toInt) := by
        rw [h512, hi]
        exact IntV.I32.sfits (by
          have h0 : (0 : Int) ≤ (t : Int) * 512 := by exact_mod_cast Nat.zero_le (t * 512)
          linarith) ht
      rw [IntV.toInt_mul rfl rfl hsfit, h512, hi]
    refine safe_convert (d := .i32) (d' := .i64) (x := tl.arange 0 512) ?_ ?_
    · intro u
      exact exists_convRel_i64 (Or.inr rfl) E.P _
    · intro a64 ha64
      have ha : ∀ i : Idx [512], (a64.get i).toInt = ((i.1 : Nat) : Int) := by
        intro i
        show (a64.get (i.1, ())).toInt = ((i.1 : Nat) : Int)
        have h := ha64 (i.1, ())
        have hv : (a64.get (i.1, ())).toInt = ((tl.arange 0 512).get (i.1, ())).toInt :=
          ConvRel.toInt_i32_i64 h
        rw [hv]
        simp only [tl.arange, Tensor.get_arange]
        have hlt : (0 : Int) + ((i.1 : Nat) : Int) < 2 ^ 31 := by
          have := i.1.isLt
          push_cast
          omega
        have h0 : -(2 ^ 31 : Int) ≤ (0 : Int) + ((i.1 : Nat) : Int) := by
          have h2 : (0 : Int) ≤ ((i.1 : Nat) : Int) := by exact_mod_cast Nat.zero_le (i.1 : Nat)
          linarith
        have h := IntV.I32.toInt_ofInt (x := (0 : Int) + ((i.1 : Nat) : Int)) ⟨h0, hlt⟩
        simpa using h
      refine safe_convert (d := .i32) (d' := .i64) (x := Tensor.scalar (x + s - x)) ?_ ?_
      · intro u
        exact exists_convRel_i64 (Or.inr rfl) E.P _
      · intro lim64 hlim64
        have hlim : lim64.item.toInt = slen d init b := by
          have h := hlim64 ()
          have hv : lim64.item.toInt = (x + s - x).toInt := by
            simpa only [Tensor.scalar, Tensor.item] using ConvRel.toInt_i32_i64 h
          rw [hv]
          obtain ⟨hs0, hs1, -⟩ := hpre.2.2.2.2.2.2.1 b hb
          have h0 : 0 ≤ s.toInt := by rw [hs]; exact hs0
          have h1 : s.toInt < (2 : Int) ^ (32 - 1) := by
            have e : ((2 : Int) ^ (32 - 1 : Nat)) = 2 ^ 31 := by norm_num
            rw [e, hs]; linarith
          rw [sub_add_self_toInt (w := 32) (by norm_num) h0 h1, hs]
        refine safe_convert (d := .i32) (d' := .i64)
          (x := Tensor.scalar (IntV.ofNat d.max_context_len)) ?_ ?_
        · intro u
          exact exists_convRel_i64 (Or.inr rfl) E.P _
        · intro rsp64 hrsp64
          have hrsp : rsp64.item.toInt = (d.max_context_len : Int) := by
            have h := hrsp64 ()
            have hv : rsp64.item.toInt = ((IntV.ofNat d.max_context_len : I32)).toInt := by
              simpa only [Tensor.scalar, Tensor.item] using ConvRel.toInt_i32_i64 h
            rw [hv, IntV.I32.toInt_ofNat hd.mcl]
          -- the lanes of the load: the table element of each of the 512 positions of trip `t`
          let gL : Idx [512] → Nat := fun i =>
            (rpi d init b * (d.max_context_len : Int) + sidx d init b +
              ((t : Int) * 512 + ((i.1 : Nat) : Int))).toNat
          let enL : Idx [512] → Bool := fun i =>
            decide ((t : Int) * 512 + ((i.1 : Nat) : Int) < slen d init b)
          have henL : ∀ i : Idx [512], enL i =
              decide ((t : Int) * 512 + ((i.1 : Nat) : Int) < slen d init b) := fun _ => rfl
          have hgL : ∀ i : Idx [512], gL i =
              (rpi d init b * (d.max_context_len : Int) + sidx d init b +
                ((t : Int) * 512 + ((i.1 : Nat) : Int))).toNat := fun _ => rfl
          have henLt : ∀ i : Idx [512], enL i = true →
              (t : Int) * 512 + ((i.1 : Nat) : Int) < slen d init b := by
            intro i he
            rw [henL i] at he
            exact of_decide_eq_true he
          have hreq : ∀ i : Idx [512], enL i = true →
              0 ≤ rpi d init b * (d.max_context_len : Int) + sidx d init b +
                  ((t : Int) * 512 + ((i.1 : Nat) : Int)) ∧
              rpi d init b * (d.max_context_len : Int) + sidx d init b +
                  ((t : Int) * 512 + ((i.1 : Nat) : Int)) < (d.req_to_token.len : Int) := by
            intro i he
            have hltc := henLt i he
            have hcast : ((t * 512 + i.1.val : Nat) : Int) =
                (t : Int) * 512 + ((i.1 : Nat) : Int) := by
              push_cast
              ring
            obtain ⟨-, -, -, -, hw⟩ := hpre.2.2.2.2.2.2.1 b hb
            obtain ⟨h1, h2⟩ := hw (t * 512 + i.1.val) (by rw [hcast]; exact hltc)
            rw [hcast] at h1 h2
            exact ⟨h1, h2⟩
          have hvalL : ∀ i : Idx [512], enL i = true →
              (gL i : Int) = rpi d init b * (d.max_context_len : Int) + sidx d init b +
                ((t : Int) * 512 + ((i.1 : Nat) : Int)) := by
            intro i he
            rw [hgL i]
            exact Int.toNat_of_nonneg (hreq i he).1
          have hoffL : ∀ i : Idx [512], ((a64 + i64.item).get i).toInt =
              (t : Int) * 512 + ((i.1 : Nat) : Int) := by
            intro i
            have hfit : IntV.SFits 64 ((a64.get i).toInt + i64.item.toInt) := by
              rw [ha i, hiB]
              refine IntV.I64.sfits ?_ ?_
              · have h0 : (0 : Int) ≤ ((i.1 : Nat) : Int) := by
                  exact_mod_cast Nat.zero_le (i.1 : Nat)
                have h4 : (-(2 ^ 63) : Int) ≤ 0 := by norm_num
                linarith
              · have h1 : ((i.1 : Nat) : Int) < 512 := by exact_mod_cast i.1.isLt
                have h3 : (512 : Int) + 2 ^ 31 < 2 ^ 63 := by norm_num
                linarith
            rw [Tensor.get_adds, IntV.toInt_add rfl rfl hfit, ha i, hiB]
            ring
          have hmask : ∀ i : Idx [512], ((a64 + i64.item).ltS lim64.item).get i = enL i := by
            intro i
            rw [Tensor.get_ltS, IntV.lt_eq, henL i]
            congr 1
            rw [hoffL i, hlim]
          have hptrL : ∀ i : Idx [512], enL i = true →
              (d.req_to_token.ptr 0 + p * rsp64.item + x + (a64 + i64.item)).get i
                = d.req_to_token.ptr (gL i : Int) := by
            intro i he
            simp only [Tensor.get_ptrAdd, Ptr.add_int, Buf.ptr_offset,
              zero_add]
            have hmul : (p * rsp64.item).toInt =
                rpi d init b * (d.max_context_len : Int) := by
              have hsl : 0 < slen d init b := by
                have hltc := henLt i he
                have hl0 : (0 : Int) ≤ ((i.1 : Nat) : Int) := by
                  exact_mod_cast Nat.zero_le (i.1 : Nat)
                linarith
              have hfit : IntV.SFits 64 (p.toInt * rsp64.item.toInt) := by
                rw [hp, hrsp]
                exact tab_fits hd hpre hb hsl
              rw [IntV.toInt_mul rfl rfl hfit, hp, hrsp]
            rw [hmul, hoffL i, hx, hvalL i he]
          have hinL : ∀ i : Idx [512], enL i = true → gL i < d.req_to_token.len := by
            intro i he
            have h := (hreq i he).2
            rw [← hvalL i he] at h
            exact_mod_cast h
          have himgL : Buf.img gL enL = tabTripElems d init b t := by
            ext j
            constructor
            · rintro ⟨i, he, hg⟩
              refine ⟨i.1.val, i.1.isLt, henLt i he, ?_⟩
              rw [← hg, hvalL i he]
            · rintro ⟨l, hl, hltc, hj⟩
              have hen : enL ((⟨l, hl⟩, ()) : Idx [512]) = true := decide_eq_true hltc
              have hgl : (gL (⟨l, hl⟩, ()) : Int) = (j : Int) := by
                rw [hvalL (⟨l, hl⟩, ()) hen]
                show rpi d init b * (d.max_context_len : Int) + sidx d init b +
                    ((t : Int) * 512 + (l : Int)) = (j : Int)
                exact hj.symm
              exact ⟨(⟨l, hl⟩, ()), hen, by exact_mod_cast hgl⟩
          have hRL : d.req_to_token.elems (Buf.img gL enL) ⊆ R d init b := by
            rw [himgL]
            intro z hz
            simp only [R]
            exact Or.inr (Buf.elems_mono d.req_to_token (tabTripElems_sub d init b t) hz)
          refine safe_load_buf (b := d.req_to_token) (g := gL) (en := enL)
            (Buf.addrs_of_get d.req_to_token hmask hptrL) hinL hVt hRL ?_
          intro data hread hother
          -- the lanes of the store: the 512 elements of `kv_indices` that trip `t` writes
          let gS : Idx [512] → Nat := fun i =>
            (ind d init b + (t : Int) * 512 + ((i.1 : Nat) : Int)).toNat
          have hgS : ∀ i : Idx [512], gS i =
              (ind d init b + (t : Int) * 512 + ((i.1 : Nat) : Int)).toNat := fun _ => rfl
          have hnonnegS : ∀ i : Idx [512], enL i = true →
              0 ≤ ind d init b + (t : Int) * 512 + ((i.1 : Nat) : Int) := by
            intro i _
            have h0 : (0 : Int) ≤ ind d init b :=
              (hpre.2.2.2.2.2.2.2.2.2 b (Nat.le_of_lt hb)).1
            have h1 : (0 : Int) ≤ ((i.1 : Nat) : Int) := by
              exact_mod_cast Nat.zero_le (i.1 : Nat)
            linarith
          have hvalS : ∀ i : Idx [512], enL i = true →
              (gS i : Int) = ind d init b + (t : Int) * 512 + ((i.1 : Nat) : Int) := by
            intro i he
            rw [hgS i]
            exact Int.toNat_of_nonneg (hnonnegS i he)
          have hinS : ∀ i : Idx [512], enL i = true → gS i < d.kv_indices.len := by
            intro i he
            have hstep : ind d init b + slen d init b = ind d init (b + 1) := by
              have h := hpre.2.2.2.2.2.2.2.2.1 b hb
              linarith
            have hrange : ind d init (b + 1) ≤ (d.kv_indices.len : Int) :=
              (hpre.2.2.2.2.2.2.2.2.2 (b + 1) (Nat.succ_le_of_lt hb)).2
            have hltc := henLt i he
            have h : (gS i : Int) < (d.kv_indices.len : Int) := by
              rw [hvalS i he]
              linarith
            exact_mod_cast h
          have hptrS : ∀ i : Idx [512], enL i = true →
              (d.kv_indices.ptr 0 + off + (a64 + i64.item)).get i
                = d.kv_indices.ptr (gS i : Int) := by
            intro i he
            simp only [Tensor.get_ptrAdd, Ptr.add_int, Buf.ptr_offset,
              zero_add]
            rw [hoffL i, hoff, hvalS i he]
            ring_nf
          have hinj : ∀ i j : Idx [512], enL i = true → enL j = true → gS i = gS j → i = j := by
            intro i j hei hej hg
            have h1 : (gS i : Int) = ind d init b + (t : Int) * 512 + ((i.1 : Nat) : Int) :=
              hvalS i hei
            have h2 : (gS j : Int) = ind d init b + (t : Int) * 512 + ((j.1 : Nat) : Int) :=
              hvalS j hej
            have h3 : ((i.1 : Nat) : Int) = ((j.1 : Nat) : Int) := by
              have h4 : (gS i : Int) = (gS j : Int) := by rw [hg]
              linarith
            have h5 : (i.1 : Nat) = (j.1 : Nat) := by exact_mod_cast h3
            have : Subsingleton (Idx []) := ⟨fun a b => by cases a; cases b; rfl⟩
            exact Prod.ext (Fin.ext h5) (Subsingleton.elim _ _)
          have himgS : Buf.img gS enL = tripElems d init b t := by
            ext j
            constructor
            · rintro ⟨i, he, hg⟩
              refine ⟨i.1.val, i.1.isLt, henLt i he, ?_⟩
              rw [← hg, hvalS i he]
            · rintro ⟨l, hl, hltc, hj⟩
              have hen : enL ((⟨l, hl⟩, ()) : Idx [512]) = true := decide_eq_true hltc
              have hgl : (gS (⟨l, hl⟩, ()) : Int) = (j : Int) := by
                rw [hvalS (⟨l, hl⟩, ()) hen]
                show ind d init b + (t : Int) * 512 + (l : Int) = (j : Int)
                exact hj.symm
              exact ⟨(⟨l, hl⟩, ()), hen, by exact_mod_cast hgl⟩
          have houtLen : ∀ j ∈ tripElems d init b t, j < d.kv_indices.len := by
            rintro j ⟨l, hl, hltc, hj⟩
            have hstep : ind d init b + slen d init b = ind d init (b + 1) := by
              have h := hpre.2.2.2.2.2.2.2.2.1 b hb
              linarith
            have hrange : ind d init (b + 1) ≤ (d.kv_indices.len : Int) :=
              (hpre.2.2.2.2.2.2.2.2.2 (b + 1) (Nat.succ_le_of_lt hb)).2
            have h : (j : Int) < (d.kv_indices.len : Int) := by linarith
            exact_mod_cast h
          have hgoal : tl.store (d.kv_indices.ptr 0 + off + (a64 + i64.item)) data
                (some ((a64 + i64.item).ltS lim64.item)) =
              (tl.store (d.kv_indices.ptr 0 + off + (a64 + i64.item)) data
                (some ((a64 + i64.item).ltS lim64.item)) >>= fun _ => .ret ()) := by
            simp only [tl.store, Prog.bind_eq, Prog.bind]
          rw [hgoal]
          refine safe_store_buf (b := d.kv_indices) (g := gS) (en := enL)
            (k := fun _ => .ret ())
            (Buf.addrs_of_get d.kv_indices hmask hptrS) hinS hinj hVi ?_ ?_
          · rw [himgS]
          · intro m' hreadS hframeS hpres
            rw [himgS] at hframeS
            refine Safe.ret ?_
            refine ⟨?_, ?_, ?_, ?_⟩
            · intro j hj hsl
              push_cast at hsl
              by_cases hjt : (j : Int) < (t : Int) * 512
              · have hnotin : ∀ x ∈ Mem.span .i32
                    (d.kv_indices.ptr (ind d init b + (j : Int))).addr,
                    x ∉ d.kv_indices.elems (tripElems d init b t) := by
                  intro x hx hx'
                  obtain ⟨j', ⟨l, hl, hltrip, hj'⟩, hx''⟩ := hx'
                  have hne : ind d init b + (j : Int) ≠ (j' : Int) := by
                    rw [hj']
                    intro h
                    have hl0 : (0 : Int) ≤ (l : Int) := by exact_mod_cast Nat.zero_le l
                    linarith
                  exact Set.disjoint_left.mp
                    (d.kv_indices.span_disjoint (i := ind d init b + (j : Int))
                      (j := (j' : Int)) hne) hx hx''
                have heq : m'.read .i32
                      (d.kv_indices.ptr (ind d init b + (j : Int))).addr
                    = m.read .i32
                      (d.kv_indices.ptr (ind d init b + (j : Int))).addr :=
                  Mem.read_congr fun x hx => hframeS x (hnotin x hx)
                rw [heq]
                exact hQ j hj hjt
              · push Not at hjt
                have hle : t * 512 ≤ j := by
                  have h2 : ((t * 512 : Nat) : Int) ≤ (j : Int) := by
                    push_cast
                    linarith
                  exact_mod_cast h2
                let l : Nat := j - t * 512
                have hlj : ((l : Nat) : Int) = (j : Int) - (t : Int) * 512 := by
                  simp only [l]
                  rw [Nat.cast_sub hle]
                  push_cast
                  ring
                have hl512 : l < 512 := by
                  have h2 : ((l : Nat) : Int) < 512 := by
                    rw [hlj]
                    linarith
                  exact_mod_cast h2
                have hjeq : (t : Int) * 512 + ((l : Nat) : Int) = (j : Int) := by
                  rw [hlj]
                  ring
                have hlane' : ((((⟨l, hl512⟩, ()) : Idx [512]).1 : Nat) : Int) = (l : Int) := rfl
                have hen : enL ((⟨l, hl512⟩, ()) : Idx [512]) = true :=
                  decide_eq_true (show (t : Int) * 512 + (l : Int) < slen d init b from by
                    rw [hjeq]; exact hj)
                have hgS_i : (gS (⟨l, hl512⟩, ()) : Int) = ind d init b + (j : Int) := by
                  rw [hvalS (⟨l, hl512⟩, ()) hen]
                  linarith [hjeq, hlane']
                have hgL_i : (gL (⟨l, hl512⟩, ()) : Int) =
                    rpi d init b * (d.max_context_len : Int) + sidx d init b + (j : Int) := by
                  rw [hvalL (⟨l, hl512⟩, ()) hen]
                  linarith [hjeq, hlane']
                have hmread : m'.read .i32
                      (d.kv_indices.ptr (ind d init b + (j : Int))).addr
                    = some (data.get (⟨l, hl512⟩, ())) := by
                  rw [← hgS_i]
                  exact hreadS (⟨l, hl512⟩, ()) hen
                have hminit : m.read .i32
                      (d.req_to_token.ptr (gL (⟨l, hl512⟩, ()) : Int)).addr
                    = init.read .i32
                      (d.req_to_token.ptr (gL (⟨l, hl512⟩, ()) : Int)).addr :=
                  Mem.read_congr fun x hx =>
                    hT x (Buf.span_sub d.req_to_token (hinL (⟨l, hl512⟩, ()) hen) hx)
                have hinitread : init.read .i32
                      (d.req_to_token.ptr
                        (rpi d init b * (d.max_context_len : Int) + sidx d init b +
                          (j : Int))).addr
                    = some (data.get (⟨l, hl512⟩, ())) := by
                  rw [← hgL_i, ← hminit]
                  exact hread (⟨l, hl512⟩, ()) hen
                rw [hmread, hinitread]
            · intro z hz
              have hnot : z ∉ d.kv_indices.elems (tripElems d init b t) := by
                intro hz'
                have hzbytes : z ∈ d.kv_indices.bytes :=
                  Buf.elems_sub_bytes d.kv_indices houtLen hz'
                exact Set.disjoint_left.mp hd.disj.1 hzbytes hz
              rw [hframeS z hnot]
              exact hT z hz
            · exact Buf.valid_of_agree hVi (fun x hx => hpres x (hVi.2.2.2 x hx))
            · exact Buf.valid_of_agree hVt (fun x hx => hpres x (hVt.2.2.2 x hx))

/-! ## The tail of the program, after the row pointer is known -/

/-- The rest of the program once `p` (`req_pool_indices[b]`), `x` (`kv_start`) and `off`
(`kv_indptr[b]`) are known: load the length, compute the trip count, and run the loop. -/
def tailProg (d : Inst) (b : Nat) (p : I64) (x : I32) (off : I32) : Prog Unit := do
  let s ← tl.load1 (d.seq_lens.ptr 0 + (IntV.ofNat b : I32))
  let s32 ← tl.convert d.dL .i32 (Tensor.scalar s)
  let n ← tl.idiv (Tensor.scalar (x + s32.item - x + (IntV.ofNat 511 : I32)))
    (Tensor.scalar (IntV.ofNat 512 : I32))
  let _ ← tl.forRange (IntV.ofInt 0) n.item (IntV.ofInt 1) () fun i _ =>
    tripProg d p x s32.item i off
  pure ()

end Claims.KvIndices

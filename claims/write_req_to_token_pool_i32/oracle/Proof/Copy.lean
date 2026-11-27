import Proof.Trip

/-!
# `write_req_to_token_pool_i32`: a whole copy loop

A copy loop is `num_loop = tl.cdiv(x, 512)` followed by `for i in range(num_loop)` over
`tripProg` (source lines 76-85 and 92-104). `safe_copy` proves the loop and hands the rest of the
program a memory in which each of the `L = x` table elements from `t0` holds the cast of its source
element, and nothing else of the table's window changed.
-/

namespace Claims.WriteReqToTokenPool

open TVA

/-- A copy loop followed by the rest of the program `k`. -/
def copyProg {α : Type} (x lim : I64) (srcOf : Tensor [512] I64 → Tensor [512] (Ptr .i64))
    (dstOf : Tensor [512] I64 → Tensor [512] (Ptr .i32)) (k : Unit → Prog α) : Prog α := do
  let num_loop := (← tl.idiv (Tensor.scalar (x + (IntV.ofNat 511 : I64)))
    (Tensor.scalar (IntV.ofNat 512 : I64))).item
  let _ ← tl.forRange (IntV.ofInt 0 : I64) num_loop (IntV.ofInt 1 : I64) () fun i _ =>
    tripProg lim i srcOf dstOf
  k ()

/-- A read of a buffer element outside a written set of elements is unchanged. -/
theorem read_frame {dt : DType} {b : Buf dt} {X : Set Nat} {m₁ m₂ : Mem}
    (h : ∀ z, z ∉ b.elems X → m₂ z = m₁ z) {e : Int}
    (he : ∀ j ∈ X, (j : Int) ≠ e) :
    m₂.read dt (b.ptr e).addr = m₁.read dt (b.ptr e).addr := by
  refine Mem.read_congr fun x hx => h x ?_
  rintro ⟨j, hj, hx'⟩
  exact Set.disjoint_left.mp (b.span_disjoint (he j hj)) hx' hx

/-- **A copy loop is safe.** With `x = L` source elements `S[s0 …]` (`0 ≤ L ≤ 2^22`) and the
table window `t0 …`, the loop writes exactly that window, each element the cast of its source
element's initial value; then the rest `k` runs. -/
theorem safe_copy {d : Inst} {E : Env} {R : Set Int} {α : Type} {Q : α → Mem → Prop}
    {Wk : Set Int} {init m : Mem} (S : Buf .i64) {s0 t0 L : Int} {x : I64} (hx : x.toInt = L)
    (hL0 : 0 ≤ L) (hL : L ≤ 2 ^ 22)
    {srcOf : Tensor [512] I64 → Tensor [512] (Ptr .i64)}
    {dstOf : Tensor [512] I64 → Tensor [512] (Ptr .i32)}
    (hsrc : 0 < L → ∀ off l, (srcOf off).get l = S.ptr (s0 + (off.get l).toInt))
    (hdst : ∀ off l, (dstOf off).get l = d.req_to_token.ptr (t0 + (off.get l).toInt))
    (hS : ∀ o : Nat, (o : Int) < L → 0 ≤ s0 + o ∧ s0 + o < S.len)
    (hT : ∀ o : Nat, (o : Int) < L → 0 ≤ t0 + o ∧ t0 + o < d.req_to_token.len)
    (hTS : Disjoint d.req_to_token.bytes S.bytes) (hSv : 0 < L → S.Valid init)
    (hR : S.elems (win s0 L) ⊆ R)
    (hfr : ∀ z, z ∉ d.req_to_token.bytes → m z = init z) (hTv : d.req_to_token.Valid m)
    (hWk : Disjoint (d.req_to_token.elems (win t0 L)) Wk) {k : Unit → Prog α}
    (hk : ∀ m', (∀ o : Nat, (o : Int) < L → Copied d init m' S s0 t0 o) →
      (∀ z, z ∉ d.req_to_token.elems (win t0 L) → m' z = m z) → d.req_to_token.Valid m' →
      Safe E R Wk Q m' (k ())) :
    Safe E R (d.req_to_token.elems (win t0 L) ∪ Wk) Q m (copyProg x x srcOf dstOf k) := by
  have hwin : d.req_to_token.elems (win t0 L) ⊆ d.req_to_token.bytes :=
    d.req_to_token.elems_sub_bytes (win_lt hT)
  have h511 : (IntV.ofNat 511 : I64).toInt = 511 := IntV.I64.toInt_ofNat (by norm_num)
  have h512 : (IntV.ofNat 512 : I64).toInt = 512 := IntV.I64.toInt_ofNat (by norm_num)
  have hnum : (x + (IntV.ofNat 511 : I64)).toInt = L + 511 :=
    IntV.toInt_add hx h511 (IntV.I64.sfits (by linarith) (by linarith))
  simp only [copyProg]
  refine safe_idiv1 (by rw [hnum]; linarith) (by rw [h512]; norm_num) fun n hn => ?_
  rw [hnum, h512] at hn
  have hlo : (IntV.ofInt 0 : I64).toInt = 0 := IntV.I64.toInt_ofInt ⟨by norm_num, by norm_num⟩
  have hst : (IntV.ofInt 1 : I64).toInt = 1 := IntV.I64.toInt_ofInt ⟨by norm_num, by norm_num⟩
  have hq0 : 0 ≤ (L + 511) / 512 := Int.ediv_nonneg (by linarith) (by norm_num)
  have hqid : (L + 511) / 512 * 512 + (L + 511) % 512 = L + 511 := Int.ediv_mul_add_emod _ _
  have hqm0 : 0 ≤ (L + 511) % 512 := Int.emod_nonneg _ (by norm_num)
  have hqm1 : (L + 511) % 512 < 512 := Int.emod_lt_of_pos _ (by norm_num)
  have hN : tl.tripCount (IntV.ofInt 0 : I64).toInt n.item.toInt (IntV.ofInt 1 : I64).toInt =
      ((L + 511) / 512).toNat := by
    simp only [tl.tripCount, hlo, hst, hn]
    norm_num
  have hrange : tl.RangeOk (IntV.ofInt 0 : I64) n.item (IntV.ofInt 1 : I64) := by
    have hir : IntV.maxInt true 64 = 2 ^ 63 - 1 := by norm_num [IntV.maxInt]
    refine ⟨by rw [hst]; norm_num, ?_, ?_, ?_⟩
    · rw [hn, hst, hir]; omega
    · rw [hlo, hir]; norm_num
    · rw [hst, hir]; norm_num
  let I : Nat → Unit → Mem → Prop := fun t _ m₁ =>
    (∀ o : Nat, (o : Int) < L → (o : Int) < (t : Int) * 512 → Copied d init m₁ S s0 t0 o) ∧
      (∀ z, z ∉ d.req_to_token.elems (win t0 L) → m₁ z = m z) ∧ d.req_to_token.Valid m₁
  let Wt : Nat → Set Int := fun t => d.req_to_token.elems (tripWin t0 L t)
  have hiter : iterW Wt (tl.tripCount (IntV.ofInt 0 : I64).toInt n.item.toInt
      (IntV.ofInt 1 : I64).toInt) ⊆ d.req_to_token.elems (win t0 L) := by
    rintro z ⟨t, -, hz⟩
    exact d.req_to_token.elems_mono (tripWin_sub t0 L t) hz
  refine Safe.mono (Q := Q) (safe_forRange (k := fun _ => k ()) hrange I Wt
    (fun t u htu _ => d.req_to_token.elems_disjoint (tripWin_disjoint (Nat.ne_of_lt htu)))
    (hWk.mono_left hiter) ⟨fun o _ ho => by
      have : (0 : Int) ≤ o := by positivity
      simp at ho; omega, fun _ _ => rfl, hTv⟩ ?_ ?_) (fun _ _ h => h)
    (Set.union_subset_union_left _ hiter)
  · intro t ht _ m₁ hI
    rw [hN] at ht
    obtain ⟨hP, hF, hV⟩ := hI
    have htq : (t : Int) < (L + 511) / 512 := by
      have : ((t : Nat) : Int) < (((L + 511) / 512).toNat : Int) := by exact_mod_cast ht
      rwa [Int.toNat_of_nonneg hq0] at this
    have ht512 : (t : Int) * 512 < L := by nlinarith
    have hLpos : 0 < L := by
      have : (0 : Int) ≤ t := by positivity
      nlinarith
    have hi : (IntV.ofInt ((IntV.ofInt 0 : I64).toInt + (t : Int) * (IntV.ofInt 1 : I64).toInt) :
        I64).toInt = t := by
      rw [hlo, hst, IntV.I64.toInt_ofInt ⟨by
        have : (0 : Int) ≤ t := by positivity
        linarith, by nlinarith⟩]
      ring
    have hfr1 : ∀ z, z ∉ d.req_to_token.bytes → m₁ z = init z := fun z hz =>
      (hF z fun h => hz (hwin h)).trans (hfr z hz)
    refine (safe_trip S hx hi hL ht512 (hsrc hLpos) hdst hS hT hTS (hSv hLpos) hR hfr1 hV).mono
      ?_ le_rfl
    rintro _ m₂ ⟨hC, hF2, hV2⟩
    refine ⟨?_, fun z hz => (hF2 z fun h => hz
      (d.req_to_token.elems_mono (tripWin_sub t0 L t) h)).trans (hF z hz), hV2⟩
    intro o ho ho'
    by_cases hot : (o : Int) < (t : Int) * 512
    · obtain ⟨v, hv1, hv2⟩ := hP o ho hot
      refine ⟨v, hv1, ?_⟩
      rw [read_frame hF2 ?_]
      · exact hv2
      · rintro j ⟨l, -, -, hj⟩ hje
        have : (0 : Int) ≤ l := by positivity
        rw [hj] at hje
        linarith
    · push Not at hot
      have hle : t * 512 ≤ o := by exact_mod_cast hot
      have hl : o - t * 512 < 512 := by
        have : (o : Int) < (t : Int) * 512 + 512 := by push_cast at ho'; linarith
        omega
      have := hC (o - t * 512) hl (by
        have e : ((o - t * 512 : Nat) : Int) = (o : Int) - (t : Int) * 512 := by
          rw [Nat.cast_sub hle]; push_cast; ring
        rw [e]; linarith)
      rwa [show t * 512 + (o - t * 512) = o by omega] at this
  · intro _ m₁ hI
    rw [hN] at hI
    obtain ⟨hP, hF, hV⟩ := hI
    refine hk m₁ (fun o ho => hP o ho ?_) hF hV
    rw [Int.toNat_of_nonneg hq0]
    linarith

end Claims.WriteReqToTokenPool

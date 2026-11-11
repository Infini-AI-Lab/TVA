import Claim
import TVA.Proof
import Proof.Trip
import Proof.Peel

/-!
# The loop over `K`

`loop_safe`: `for i_k in range(tl.cdiv(K, BK))` (line 135) runs `nK d` trips from zero accumulators
and ends with `OInv`, `AInv` at `nK d`.
-/

open TVA
namespace Claims.ChunkFwdO

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
variable {d : Inst} {g : Geo} {r : Nat} {init : Mem}

/-- A loop whose trips write nothing. -/
theorem safe_forRange_ne {σ : Type} {sg : Bool} {w : Nat} {lo hi step : IntV sg w} {init : σ}
    {body : IntV sg w → σ → Prog σ} {k : σ → Prog α}
    (hrange : tl.RangeOk lo hi step) (I : Nat → σ → Mem → Prop) {Wk : Set Int}
    (h0 : I 0 init m)
    (hstep : ∀ t, t < tl.tripCount lo.toInt hi.toInt step.toInt → ∀ st m₁, I t st m₁ →
      Safe E R ∅ (fun st' m₂ => I (t + 1) st' m₂) m₁
        (body (IntV.ofInt (lo.toInt + t * step.toInt)) st))
    (hk : ∀ st m₁, I (tl.tripCount lo.toInt hi.toInt step.toInt) st m₁ →
      Safe E R Wk Q m₁ (k st)) :
    Safe E R Wk Q m (tl.forRange lo hi step init body >>= k) := by
  have h := safe_forRange (E := E) (R := R) (Q := Q) (m := m) hrange I (fun _ => ∅) (Wk := Wk)
    (fun _ _ _ _ => Set.disjoint_empty _) (by rw [iterW_empty]; exact Set.empty_disjoint Wk) h0
    hstep hk
  rw [iterW_empty, Set.empty_union] at h
  exact h

/-- **The loop over `K`** (lines 135–158) from zero accumulators. -/
theorem loop_safe (hP : Device E.P) (hd : d.ok) (hpre : Pre d init) (hg : Geo.Ok d g r)
    {q k v h o : Ptr .bf16} {o_t : Tensor [BT] I64} {T : I64} {o_v : Tensor [d.bk] I64}
    (hl : TailLanes d g q k v h o o_t T o_v) {K' : Tensor [BT, d.bk] F32 × Tensor [BT, BT] F32 → Prog α}
    (hK : ∀ st : Tensor [BT, d.bk] F32 × Tensor [BT, BT] F32,
      OInv d g r (nK d) st.1 → AInv d g (nK d) st.2 → Safe E (Rs d) W Q init (K' st)) :
    Safe E (Rs d) W Q init (tl.forRange (IntV.ofInt 0 : I32)
      (IntV.ofNat ((d.K + (d.bk - 1)) / d.bk) : I32) (IntV.ofInt 1 : I32)
      (tl.zeros [BT, d.bk] .f32, tl.zeros [BT, BT] .f32)
      (loopBody d q k h o_t (o_t.ltS T) o_v) >>= K') := by
  have hn := nK_le hd
  have hlo : (IntV.ofInt 0 : I32).toInt = 0 := by decide
  have hst : (IntV.ofInt 1 : I32).toInt = 1 := by decide
  have hhi : (IntV.ofNat ((d.K + (d.bk - 1)) / d.bk) : I32).toInt = (nK d : Int) :=
    IntV.I32.toInt_ofNat (by unfold nK at hn; omega)
  have hrange : tl.RangeOk (IntV.ofInt 0 : I32) (IntV.ofNat ((d.K + (d.bk - 1)) / d.bk) : I32)
      (IntV.ofInt 1 : I32) := by
    unfold tl.RangeOk IntV.maxInt
    rw [hlo, hst, hhi]
    refine ⟨by norm_num, ?_, by norm_num, by norm_num⟩
    norm_num; omega
  have htrip : tl.tripCount (IntV.ofInt 0 : I32).toInt
      (IntV.ofNat ((d.K + (d.bk - 1)) / d.bk) : I32).toInt (IntV.ofInt 1 : I32).toInt = nK d := by
    rw [hlo, hst, hhi]; simp [tl.tripCount]
  refine safe_forRange_ne hrange
    (fun t st m' => m' = init ∧ OInv d g r t st.1 ∧ AInv d g t st.2) ?_ ?_ ?_
  · obtain ⟨h1, h2⟩ := inv_zero d g r
    exact ⟨rfl, h1, h2⟩
  · intro t ht st m₁ ⟨hm, hO, hA⟩
    rw [htrip] at ht
    subst hm
    refine trip_safe hP hd hpre hg hl ht ?_ hO hA
    rw [hlo, hst]
    simp only [zero_add, mul_one]
    exact IntV.toInt_ofInt (w := 32) (by norm_num) (sfits_nat (by norm_num) (by norm_num; omega))
  · intro st m₁ ⟨hm, hO, hA⟩
    rw [htrip] at hO hA
    subst hm
    exact hK st hO hA

end Claims.ChunkFwdO

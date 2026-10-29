import TVA

/-!
lr-14b, round 14: the batch's lemmas and facts applied as a kernel proof would.
Checked with `lake env lean` from the `lean` folder; it uses only the library.

* U1 (L34): one program's masked tensor `tl.atomic_add` on `float32` cells, proved with
  `SafeA.atomic`, its log paired per position with its lanes' operands (`Paired.lanes`,
  `accStep_atomicAdd` from `AtomicAddF32`, `accStep_other`).
* U2 (L34): the postcondition is `PermInv` (`Paired.perm`), as `shared` needs.
* U3 (L34): `shared`'s outcome with that postcondition gives `replay_err_shared`'s bound.
* U4 (L34): lr-9b's case, two programs adding `1` and `1/2` to a cell holding `2^24`: the bound
  is against `2^24 + 3/2`, and it is about `4`.
* U5 (L34): integer `atomic_max` at `int32` and `atomic_min` at `uint8`, one step each,
  through `replay_max`/`replay_min`.
* U6 (L64): from `CvtSatF32E4m3`, a finite accumulator's conversion is within one rounding of the
  clamped value; `+∞` and `1000` give `448`; a NaN gives a NaN.
* U7 (L66): `negInfF32_eq_negInf` rewrites a fact at `negInfF32` into one at `FloatV.negInf`.
* U8 (L64): `Platform.sm90ref` satisfies the six `CvtSat` facts with the other facts, and the
  `CvtRn` facts follow from the `CvtSat` ones for any platform.
* U9 (L34): `shared` composes with `meets_of_faithful` into a claim.
* U10 (L64): the model SEMANTICS gives `tl.full([B], float('inf'), tl.float8e5)`, with
  `CvtSatF32E5m2`, has the value `57344` (the device stores `+inf`, work/cvt8_inf.out).
-/

open TVA TVA.Math.Float

namespace LR14b

/-! ## U1: one program -/

/-- The per-step relation `AtomicAddF32` gives, for the cell `a`. -/
abbrev R32 (a : Int) : AtomicStep → Option ℚ → Prop :=
  AccStep .f32 a FloatV.toRat (fun x : F32 => x.isFinite = true) (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ))
    Format.binary32.maxVal

/-- Lane `i`'s operand for the cell `a`. -/
def opOf {s : Shape} (ptrs : Tensor s (Ptr .f32)) (vals : Tensor s F32) (a : Int) (i : Idx s) :
    Option ℚ :=
  if (ptrs.get i).addr = a then some (vals.get i).toRat else none

/-- The program's operands for the cell `a`: its enabled lanes' values whose pointer is `a`. -/
def opsOf {s : Shape} (ptrs : Tensor s (Ptr .f32)) (vals : Tensor s F32)
    (mask : Option (Tensor s Bool)) (a : Int) : Multiset ℚ :=
  (((Idx.all s).filter fun i => (tl.addrs ptrs mask i).isSome).filterMap (opOf ptrs vals a) : List ℚ)

theorem u1 {E : Env} {R W : Set Int} {C : Set (DType × Int)} {m : Mem} {s : Shape}
    (ptrs : Tensor s (Ptr .f32)) (vals : Tensor s F32) (mask : Option (Tensor s Bool)) (a : Int)
    (hacc : Rules.Num.AtomicAddF32 E.P) (htot : Rules.Num.AtomicAddTotal E.P .binary32)
    (hcell : ∀ i, (tl.addrs ptrs mask i).isSome = true →
      Mem.Aligned .f32 (ptrs.get i).addr ∧ (DType.f32, (ptrs.get i).addr) ∈ C)
    (hfin : ∀ i, (vals.get i).isFinite = true)
    (hsep : ∀ i, (ptrs.get i).addr ≠ a →
      Disjoint (Mem.span .f32 (ptrs.get i).addr) (Mem.span .f32 a)) :
    SafeA E R W C (fun (_ : Unit) log _ => Paired (R32 a) log (opsOf ptrs vals mask a)) [] m
      (tl.atomic_add ptrs vals mask >>= fun _ => .ret ()) := by
  simp only [tl.atomic_add, Prog.bind_eq, Prog.bind]
  refine .env ?_
  refine SafeA.atomic hcell rfl (by decide) (fun i old => htot old (vals.get i)) ?_
  intro order horder junk olds news hupd
  refine .ret ?_
  have hp := Paired.lanes (R := R32 a) order
    (fun i => (⟨.f32, (ptrs.get i).addr, olds i, news i⟩ : AtomicStep)) (opOf ptrs vals a)
    (fun i hi => by
      unfold opOf
      by_cases ha : (ptrs.get i).addr = a
      · simp only [ha, ite_true]
        have h := accStep_atomicAdd hacc (a := (ptrs.get i).addr) (hupd i hi) (hfin i)
        rw [ha] at h
        exact h
      · simp only [ha, ite_false]
        exact accStep_other (hsep i ha))
  have hops : ((order.filterMap (opOf ptrs vals a) : List ℚ) : Multiset ℚ) = opsOf ptrs vals mask a :=
    Multiset.coe_eq_coe.mpr (horder.filterMap _)
  simpa [hops] using hp

/-! ## U2: the postcondition is `PermInv` -/

theorem u2 {β : Type} (Rel : AtomicStep → Option β → Prop) (ops : Multiset β) :
    PermInv (fun (_ : Unit) log (_ : Mem) => Paired Rel log ops) :=
  fun _ _ _ _ hp h => h.perm hp

/-! ## U3: `shared`'s outcome gives the bound against every program's operands -/

theorem u3 {N : Nat} {G : List (Fin N × AtomicStep)} {init mA : Mem} {a : Int} {x : F32}
    {ops : Fin N → Multiset ℚ}
    (hQ : ∀ i, ∃ _m : Mem, Paired (R32 a) (proj G i) (ops i) ∧ True)
    (hrep : Replay init (G.map Prod.snd) mA) (h0 : init.read .f32 a = some x) (hx : x.isFinite = true)
    (hroom : |x.toRat| + (∑ i, ((ops i).map fun w => |w|).sum) +
      DotLoop.bound (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) (∑ i, Multiset.card (ops i))
        (|x.toRat| + ∑ i, ((ops i).map fun w => |w|).sum) ≤ Format.binary32.maxVal) :
    ∃ y : F32, mA.read .f32 a = some y ∧ y.isFinite = true ∧
      |y.toRat - (x.toRat + ∑ i, (ops i).sum)| ≤
        DotLoop.bound (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) (∑ i, Multiset.card (ops i))
          (|x.toRat| + ∑ i, ((ops i).map fun w => |w|).sum) := by
  have hp : ∀ i, Paired (R32 a) (proj G i) (ops i) := fun i => by
    obtain ⟨_, h, -⟩ := hQ i; exact h
  obtain ⟨y, hy, hfy, he⟩ := replay_err_shared (a := a) FloatV.toRat (fun x : F32 => x.isFinite = true)
    (ρ := 2 ^ (-23 : ℤ)) (α := 2 ^ (-124 : ℤ)) (B := Format.binary32.maxVal)
    (by positivity) (by positivity) hrep hp h0 hx (by exact hroom)
  exact ⟨y, hy, hfy, he⟩

/-! ## U4: lr-9b's case -/

/-- Two programs, operands `{1}` and `{1/2}`, cell `2^24`: the cell is within the bound of
`2^24 + 3/2`, and that bound is below `5` (about `4 + 2^-21`). -/
theorem u4 {G : List (Fin 2 × AtomicStep)} {init mA : Mem} {a : Int} {x : F32}
    (hp : ∀ i, Paired (R32 a) (proj G i) (if i = 0 then {1} else {1 / 2}))
    (hrep : Replay init (G.map Prod.snd) mA) (h0 : init.read .f32 a = some x) (hx : x.isFinite = true)
    (hx24 : x.toRat = 2 ^ 24) :
    ∃ y : F32, mA.read .f32 a = some y ∧ y.isFinite = true ∧
      |y.toRat - (2 ^ 24 + 3 / 2)| ≤ 5 := by
  have hmax : (2 : ℚ) ^ 127 ≤ Format.binary32.maxVal := by
    rw [Format.binary32_maxVal]; norm_num
  obtain ⟨y, hy, hfy, he⟩ := replay_err_shared (a := a) FloatV.toRat (fun x : F32 => x.isFinite = true)
    (ρ := 2 ^ (-23 : ℤ)) (α := 2 ^ (-124 : ℤ)) (B := Format.binary32.maxVal)
    (by positivity) (by positivity) hrep hp h0 hx (by
      simp only [Fin.sum_univ_two, Fin.isValue, hx24]
      simp [DotLoop.bound]
      refine le_trans ?_ hmax
      norm_num)
  refine ⟨y, hy, hfy, ?_⟩
  simp only [Fin.sum_univ_two, Fin.isValue, hx24] at he
  simp [DotLoop.bound] at he
  have : (2 : ℚ) ^ 24 + 3 / 2 = 2 ^ 24 + (1 + 2⁻¹) := by norm_num
  rw [this]
  refine he.trans ?_
  norm_num

/-! ## U5: integer `max` and `min` -/

theorem u5max {a : Int} {old v new : I32} (h : TVA.tl.maxUpd true .i32 v old new) :
    MapsCell .i32 a (fun y => (some v).elim y (IntV.max y)) ⟨.i32, a, old, new⟩ :=
  mapsCell_self (by simpa [TVA.tl.maxUpd] using h.symm)

theorem u5 {N : Nat} {G : List (Fin N × AtomicStep)} {init mA : Mem} {a : Int} {x : I32}
    {ops : Fin N → Multiset I32}
    (hp : ∀ i, Paired (fun s o => MapsCell .i32 a (fun y => o.elim y (IntV.max y)) s) (proj G i) (ops i))
    (hrep : Replay init (G.map Prod.snd) mA) (h0 : init.read .i32 a = some x) :
    ∃ y : I32, mA.read .i32 a = some y ∧ x.toInt ≤ y.toInt ∧ (∀ i, ∀ v ∈ ops i, v.toInt ≤ y.toInt) :=
  let ⟨y, hy, h1, h2, _⟩ := replay_max (d := .i32) IntV.max IntV.toInt IntV.toInt_max hrep hp h0
  ⟨y, hy, h1, h2⟩

theorem u5min {N : Nat} {G : List (Fin N × AtomicStep)} {init mA : Mem} {a : Int} {x : U8}
    {ops : Fin N → Multiset U8}
    (hp : ∀ i, Paired (fun s o => MapsCell .u8 a (fun y => o.elim y (IntV.min y)) s) (proj G i) (ops i))
    (hrep : Replay init (G.map Prod.snd) mA) (h0 : init.read .u8 a = some x) :
    ∃ y : U8, mA.read .u8 a = some y ∧ y.toInt ≤ x.toInt ∧ (∀ i, ∀ v ∈ ops i, y.toInt ≤ v.toInt) :=
  let ⟨y, hy, h1, h2, _⟩ := replay_min (d := .u8) IntV.min IntV.toInt IntV.toInt_min hrep hp h0
  ⟨y, hy, h1, h2⟩

/-! ## U6: saturation into `float8e4nv` -/

theorem u6 {P : Platform} (h : Rules.Num.CvtSatF32E4m3 P) {x : F32} {r : F8E4}
    (hr : P.cvt .binary32 .e4m3 .rn x r) (hx : x.isFinite = true) :
    r.isFinite = true ∧
      |r.toRat - max (-448) (min x.toRat 448)| ≤ |max (-448) (min x.toRat 448)| * 2 ^ (-4 : ℤ) + 2 ^ (-10 : ℤ) := by
  have hn := h x r hr
  obtain ⟨hfin, herr⟩ := FloatV.cvtSat_rn_err (g := Format.e4m3) Format.e4m3_regular hx
  have he := hn.eq (FloatV.isFinite_iff.mp hfin).1
  rw [← he] at hfin herr
  have hmax : Format.e4m3.maxVal = 448 := Format.e4m3_maxVal
  rw [hmax] at herr
  refine ⟨hfin, herr.trans (le_of_eq ?_)⟩
  have hu : Format.e4m3.u = 2 ^ (-4 : ℤ) := by rw [Format.u_eq]; rfl
  have he : Format.e4m3.eta = 2 ^ (-10 : ℤ) := by rw [Format.eta_eq]; rfl
  rw [hu, he]

theorem u6inf {P : Platform} (h : Rules.Num.CvtSatF32E4m3 P) {r : F8E4}
    (hr : P.cvt .binary32 .e4m3 .rn (FloatV.ofPat 0x7f800000) r) :
    r.isFinite = true ∧ r.toRat = 448 := by
  have hinf : (FloatV.ofPat 0x7f800000 : F32).isInf = true := by decide
  obtain ⟨hfin, hval⟩ := FloatV.cvtSat_of_isInf (g := Format.e4m3) Format.e4m3_regular .rn hinf
  have hn := h _ r hr
  have he := hn.eq (FloatV.isFinite_iff.mp hfin).1
  rw [← he] at hfin hval
  refine ⟨hfin, ?_⟩
  rw [hval, Format.e4m3_maxVal]
  decide

theorem u6big {P : Platform} (h : Rules.Num.CvtSatF32E4m3 P) {r : F8E4}
    (hr : P.cvt .binary32 .e4m3 .rn (FloatV.ofRat 1000) r) : r.toRat = 448 := by
  have hx : (FloatV.ofRat 1000 : F32).isFinite = true := by decide +kernel
  have hv : (FloatV.ofRat 1000 : F32).toRat = 1000 := by decide +kernel
  obtain ⟨hfin, hval⟩ := FloatV.cvtSat_of_gt (g := Format.e4m3) Format.e4m3_regular .rn hx
    (by rw [hv, Format.e4m3_maxVal]; norm_num)
  have hn := h _ r hr
  have he := hn.eq (FloatV.isFinite_iff.mp hfin).1
  rw [← he] at hval
  rw [hval, hv, Format.e4m3_maxVal]
  norm_num

theorem u6nan {P : Platform} (h : Rules.Num.CvtSatF32E4m3 P) {r : F8E4}
    (hr : P.cvt .binary32 .e4m3 .rn (FloatV.ofPat 0x7fc00000) r) : r.isNaN = true := by
  have hn := h _ r hr
  rcases hn with h1 | ⟨_, h2⟩
  · rw [← h1]; decide
  · exact h2

/-! ## U7: one name for `float32` `−∞` -/

example {P : Platform} (h : Rules.Num.ExpF32NegInf P) :
    Rules.Num.MathAt P .exp .binary32 FloatV.negInf Rules.Num.zeroF32 := by
  rw [← Rules.Num.negInfF32_eq_negInf]; exact h

/-! ## U8: satisfiable together -/

structure Bundle (P : Platform) : Prop where
  satF32E4m3 : Rules.Num.CvtSatF32E4m3 P
  satF32E5m2 : Rules.Num.CvtSatF32E5m2 P
  satF16E4m3 : Rules.Num.CvtSatF16E4m3 P
  satF16E5m2 : Rules.Num.CvtSatF16E5m2 P
  satBF16E4m3 : Rules.Num.CvtSatBF16E4m3 P
  satBF16E5m2 : Rules.Num.CvtSatBF16E5m2 P
  rnF32E4m3 : Rules.Num.CvtF32E4m3 P
  rnBF16E5m2 : Rules.Num.CvtBF16E5m2 P
  cvtTot : Rules.Num.CvtTotal P .binary32 .e4m3
  cvtTotRz : Rules.Num.CvtTotal P .binary32 .e5m2 .rz
  atomic : Rules.Num.AtomicAddF32 P
  div64 : Rules.Num.DivRn P .binary64
  divFull : Rules.Num.DivFull P
  exp : Rules.Num.ExpF32 P
  expNegInf : Rules.Num.ExpF32NegInf P
  dot : Rules.Num.DotF32Ieee P

theorem u8 : Bundle Platform.sm90ref :=
  ⟨Platform.sm90ref.cvtSatF32E4m3, Platform.sm90ref.cvtSatF32E5m2, Platform.sm90ref.cvtSatF16E4m3,
    Platform.sm90ref.cvtSatF16E5m2, Platform.sm90ref.cvtSatBF16E4m3, Platform.sm90ref.cvtSatBF16E5m2,
    Platform.sm90ref.cvtF32E4m3, Platform.sm90ref.cvtBF16E5m2, Platform.ref.cvtTotal _ _ _,
    Platform.ref.cvtTotal _ _ _, Platform.sm90ref.atomicAddF32, Platform.sm90ref.divRn _,
    Platform.sm90ref.divFull, Platform.sm90ref.expF32, Platform.sm90ref.expF32NegInf,
    Platform.sm90ref.dotF32Ieee⟩

/-- Any platform: each `CvtSat` fact gives its pair's `CvtRn` fact. -/
example {P : Platform} (h : Rules.Num.CvtSatBF16E5m2 P) : Rules.Num.CvtBF16E5m2 P :=
  Rules.Num.CvtSat.cvtRn Format.e5m2_regular h

/-! ## U9: `shared` gives the claim (`meets_of_faithful`), as `meets_independent` does for `independent` -/

theorem u9 {Inst : Type} {ok : Inst → Prop} {Pre : Inst → Mem → Prop}
    {Post : Inst → Mem → Mem → Prop} {model : Inst → Kernel} {launch : Inst → Launch}
    {K : Claim.Behaviour Inst} {P : Platform}
    (hf : ∀ d, ok d → Rules.Compiler.Faithful P (model d) (launch d) (K d))
    (hdebug : ∀ d, ok d → (launch d).debug = false) (hvalid : ∀ d, ok d → (launch d).Valid)
    (C : Inst → Set (DType × Int)) (R W : (d : Inst) → Fin (launch d).size → Set Int)
    (Q : (d : Inst) → Fin (launch d).size → List AtomicStep → Mem → Prop)
    (hsafe : ∀ d, ok d → ∀ init, Pre d init → ∀ i,
      SafeA ((launch d).env P) (R d i) (W d i) (C d) (fun _ log m => Q d i log m) [] init
        ((model d).body ((launch d).ctx i)))
    (hQ : ∀ d i, PermInv (fun (_ : Unit) log m => Q d i log m))
    (hdisj : ∀ d i j, i ≠ j → Disjoint (W d i) (R d j ∪ W d j)) (hRW : ∀ d i, Disjoint (R d i) (W d i))
    (hA : ∀ d i, Disjoint (cellBytes (C d)) (R d i ∪ W d i))
    (hcells : ∀ d, ∀ c ∈ C d, ∀ c' ∈ C d, (Mem.span c.1 c.2 ∩ Mem.span c'.1 c'.2).Nonempty → c = c')
    (hpres : ∀ d init, Pre d init → ∀ x ∈ cellBytes (C d), init x ≠ none)
    (hpost : ∀ d, ok d → ∀ init, Pre d init → ∀ fin (G : List (Fin (launch d).size × AtomicStep)),
      (∀ i, ∃ m, Q d i (proj G i) m ∧ ∀ x ∈ R d i ∪ W d i, fin x = m x) →
      (∃ mA, Replay init (G.map Prod.snd) mA ∧ ∀ x ∈ cellBytes (C d), fin x = mA x) →
      (∀ x, x ∉ cellBytes (C d) → (∀ i, x ∉ W d i) → fin x = init x) → Post d init fin) :
    Claim.Meets ok Pre Post K :=
  meets_of_faithful hf hdebug hvalid fun d hd init hpre => by
    obtain ⟨hwb, hout⟩ := shared (hsafe d hd init hpre) (hQ d) (hdisj d) (hRW d) (hA d) (hcells d)
      (hpres d init hpre)
    refine ⟨hwb, fun o ho => ?_⟩
    obtain ⟨fin, rfl, G, h1, h2, h3⟩ := hout o ho
    exact ⟨fin, rfl, hpost d hd init hpre fin G h1 h2 h3⟩

/-! ## U10: the library's meaning of `tl.full([B], float('inf'), tl.float8e5)`

SEMANTICS ("Constants"): a Python number built in an fp8 dtype is `FloatV.litF32`, then
`tl.cvt8`. With `CvtSatF32E5m2`, every element is then finite with value `57344`. The device stores
`0x7c` (`+inf`): work/cvt8_inf.out. -/

theorem u10 {P : Platform} (h : Rules.Num.CvtSatF32E5m2 P) {r : F8E5}
    (hr : P.cvt .binary32 .e5m2 .rn (FloatV.litF32 (FloatV.ofPat 0x7ff0000000000000 : F64)) r) :
    r.isFinite = true ∧ r.toRat = 57344 := by
  have hinf : (FloatV.litF32 (FloatV.ofPat 0x7ff0000000000000 : F64)).isInf = true := by decide +kernel
  have hs : (FloatV.litF32 (FloatV.ofPat 0x7ff0000000000000 : F64)).sign = false := by decide +kernel
  obtain ⟨hfin, hval⟩ := FloatV.cvtSat_of_isInf (g := Format.e5m2) Format.e5m2_regular .rn hinf
  have he := (h _ r hr).eq (FloatV.isFinite_iff.mp hfin).1
  rw [← he] at hfin hval
  refine ⟨hfin, ?_⟩
  rw [hval, hs, Format.e5m2_maxVal]
  rfl

/-- The pattern `0x7c` is `+inf` in `float8e5`, so it is not the value the model gives. -/
example : (FloatV.ofPat 0x7c : F8E5).isInf = true := by decide

end LR14b

#print axioms LR14b.u1
#print axioms LR14b.u3
#print axioms LR14b.u4
#print axioms LR14b.u5
#print axioms LR14b.u5min
#print axioms LR14b.u6
#print axioms LR14b.u6inf
#print axioms LR14b.u6big
#print axioms LR14b.u8
#print axioms LR14b.u9
#print axioms LR14b.u10
#print axioms TVA.replay_err_shared
#print axioms TVA.replay_max
#print axioms TVA.replay_min
#print axioms TVA.Paired.merge
#print axioms TVA.FloatV.cvtSat_rn_err
#print axioms TVA.FloatV.cvtSat_of_gt
#print axioms TVA.FloatV.cvtSat_of_isInf
#print axioms TVA.Math.Float.Format.roundMag_sat_of_gt
#print axioms TVA.Rules.Num.negInfF32_eq_negInf
#print axioms TVA.Platform.sm90ref.cvtSat
#print axioms TVA.Platform.ref.cvtAcc

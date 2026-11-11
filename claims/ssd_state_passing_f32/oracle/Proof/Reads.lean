import Claim
import TVA.Proof
import Proof.Inv

/-!
# Reading the inputs

The inputs' bytes are untouched while a program runs (`Rin`), so a load returns what the initial
memory holds; and under `Vals` what it holds is a finite value of the claim's reference.
-/
namespace Claims.SsdStatePassing

open TVA

/-- Elements below `len` of the input buffers lie in `Rin`. -/
theorem sub_Rin_states {d : Inst} {S : Set Nat} (h : ∀ j ∈ S, j < d.states.len) :
    d.states.elems S ⊆ Rin d := fun _ hx =>
  Or.inl (Or.inl (Or.inl (Or.inl (d.states.elems_sub_bytes h hx))))

/-- Elements of `dA_cumsum` below its length lie in `Rin`. -/
theorem sub_Rin_dA {d : Inst} {S : Set Nat} (h : ∀ j ∈ S, j < d.dA.len) :
    d.dA.elems S ⊆ Rin d := fun _ hx =>
  Or.inl (Or.inl (Or.inl (Or.inr (d.dA.elems_sub_bytes h hx))))

/-- Elements of `initial_states` below its length lie in `Rin`. -/
theorem sub_Rin_initst {d : Inst} {S : Set Nat} (h : ∀ j ∈ S, j < d.initst.len) :
    d.initst.elems S ⊆ Rin d := fun _ hx =>
  Or.inl (Or.inl (Or.inr (d.initst.elems_sub_bytes h hx)))

/-- Elements of `seq_idx` below its length lie in `Rin`. -/
theorem sub_Rin_seqIdx {d : Inst} {S : Set Nat} (h : ∀ j ∈ S, j < d.seqIdx.len) :
    d.seqIdx.elems S ⊆ Rin d := fun _ hx =>
  Or.inl (Or.inr (d.seqIdx.elems_sub_bytes h hx))

/-- Elements of `chunk_offsets` below its length lie in `Rin`. -/
theorem sub_Rin_chunkOffs {d : Inst} {S : Set Nat} (h : ∀ j ∈ S, j < d.chunkOffs.len) :
    d.chunkOffs.elems S ⊆ Rin d := fun _ hx =>
  Or.inr (d.chunkOffs.elems_sub_bytes h hx)

/-- The bytes of an element of `states` lie in `Rin`. -/
theorem span_Rin_states {d : Inst} {j : Nat} (hj : j < d.states.len) :
    Mem.span .f32 (d.states.ptr (j : Int)).addr ⊆ Rin d := fun _ hx =>
  sub_Rin_states (S := {j}) (fun _ hj' => hj' ▸ hj) (d.states.span_sub_elems (S := {j}) (Set.mem_singleton j) hx)

/-- The bytes of an element of `dA_cumsum` lie in `Rin`. -/
theorem span_Rin_dA {d : Inst} {j : Nat} (hj : j < d.dA.len) :
    Mem.span .f32 (d.dA.ptr (j : Int)).addr ⊆ Rin d := fun _ hx =>
  sub_Rin_dA (S := {j}) (fun _ hj' => hj' ▸ hj) (d.dA.span_sub_elems (S := {j}) (Set.mem_singleton j) hx)

/-- The bytes of an element of `seq_idx` lie in `Rin`. -/
theorem span_Rin_seqIdx {d : Inst} {j : Nat} (hj : j < d.seqIdx.len) :
    Mem.span .i32 (d.seqIdx.ptr (j : Int)).addr ⊆ Rin d := fun _ hx =>
  sub_Rin_seqIdx (S := {j}) (fun _ hj' => hj' ▸ hj) (d.seqIdx.span_sub_elems (S := {j}) (Set.mem_singleton j) hx)

/-- The bytes of an element of `initial_states` lie in `Rin`. -/
theorem span_Rin_initst {d : Inst} {j : Nat} (hj : j < d.initst.len) :
    Mem.span .f32 (d.initst.ptr (j : Int)).addr ⊆ Rin d := fun _ hx =>
  sub_Rin_initst (S := {j}) (fun _ hj' => hj' ▸ hj) (d.initst.span_sub_elems (S := {j}) (Set.mem_singleton j) hx)

/-- The bytes of an element of `chunk_offsets` lie in `Rin`. -/
theorem span_Rin_chunkOffs {d : Inst} {j : Nat} (hj : j < d.chunkOffs.len) :
    Mem.span .i32 (d.chunkOffs.ptr (j : Int)).addr ⊆ Rin d := fun _ hx =>
  sub_Rin_chunkOffs (S := {j}) (fun _ hj' => hj' ▸ hj)
    (d.chunkOffs.span_sub_elems (S := {j}) (Set.mem_singleton j) hx)

/-- A read of an input element in the running memory is the read in the initial memory. -/
theorem read_agree {d : Inst} {m₁ init : Mem} (hagree : ∀ x ∈ Rin d, m₁ x = init x) {dt : DType}
    {a : Int} (hsub : Mem.span dt a ⊆ Rin d) : m₁.read dt a = init.read dt a :=
  Mem.read_congr fun x hx => hagree x (hsub hx)

/-- An element of `states` is a finite `float32` with the reference's value `newV`. -/
theorem states_read {m₀ : Mem} {d : Inst} (hV : Vals m₀ d) {c h m : Nat} (hc : c < d.nchunks)
    (hh : h < d.nheads) (hm : m < d.dim) {v : F32}
    (hr : m₀.read .f32 (d.states.ptr (stIdx d c h m : Int)).addr = some v) :
    v.isFinite = true ∧ v.toReal = newV m₀ d c h m := by
  obtain ⟨v', hv', hf⟩ := hV.1 c h m hc hh hm
  rw [hr] at hv'
  cases hv'
  have e : valRat m₀ d.states (stIdx d c h m) = v.toRat := by simp only [valRat, hr]
  exact ⟨hf, by unfold newV; rw [e]; rfl⟩

/-- An element of `initial_states` is a finite `float32` with the reference's value `startV`. -/
theorem init_read {m₀ : Mem} {d : Inst} (hV : Vals m₀ d) (hinit : d.hasInit = true) {r h m : Nat}
    (hr : r < d.nreq) (hh : h < d.nheads) (hm : m < d.dim) {v : F32}
    (hrd : m₀.read .f32 (d.initst.ptr (inIdx d r h m : Int)).addr = some v) :
    v.isFinite = true ∧ v.toReal = startV m₀ d r h m := by
  obtain ⟨v', hv', hf⟩ := hV.2.1 hinit r h m hr hh hm
  rw [hrd] at hv'
  cases hv'
  have e : valRat m₀ d.initst (inIdx d r h m) = v.toRat := by simp only [valRat, hrd]
  exact ⟨hf, by unfold startV initV; rw [ite_eq_left hinit, e]; rfl⟩

/-- An element of `dA_cumsum` lies inside it. -/
theorem dAIdx_lt {d : Inst} {h c p : Nat} (hh : h < d.nheads) (hc : c < d.nchunks) (hp : p < d.chunk) :
    dAIdx d h c p < d.nheads * (d.nchunks * d.chunk) := by
  unfold dAIdx
  have h1 : (h + 1) * (d.nchunks * d.chunk) ≤ d.nheads * (d.nchunks * d.chunk) :=
    Nat.mul_le_mul_right _ hh
  have h2 : (c + 1) * d.chunk ≤ d.nchunks * d.chunk := Nat.mul_le_mul_right _ hc
  rw [Nat.succ_mul] at h1 h2
  omega

/-- An element of `initial_states` lies inside it. -/
theorem inIdx_lt {d : Inst} {r h m : Nat} (hr : r < d.nreq) (hh : h < d.nheads) (hm : m < d.dim) :
    inIdx d r h m < d.nreq * (d.nheads * d.dim) := by
  unfold inIdx
  have h1 : (h + 1) * d.dim ≤ d.nheads * d.dim := Nat.mul_le_mul_right _ hh
  have h2 : (r + 1) * (d.nheads * d.dim) ≤ d.nreq * (d.nheads * d.dim) := Nat.mul_le_mul_right _ hr
  rw [Nat.succ_mul] at h1 h2
  omega

end Claims.SsdStatePassing

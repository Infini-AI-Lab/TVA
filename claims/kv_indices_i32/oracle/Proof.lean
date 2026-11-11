import Proof.Basic
import Proof.Trip
import Proof.Tail
import Proof.Disj1
import Proof.Disj2
import Proof.Post

/-!
# `kv_indices_i32`: `create_flashinfer_kv_indices_triton`

Program `b` serves request `b`: it reads its index entries and its window of `req_to_token`,
and writes its window of `kv_indices`. The proof has four steps:

1. `Proof.Trip`: one trip of the 512-wide copy loop is safe (`safe_trip`);
2. `Proof.Tail`: the loop and the trip count (`safe_tail`);
3. `safe_program`: the scalar loads and conversions, then the tail;
4. `main`: by `correct_of_safe`, using `Proof.Disj1` and `Proof.Disj2` (footprints of distinct
   programs do not meet) and `Proof.Post` (the windows give `Post`).
-/

namespace Claims.KvIndices

open TVA

/-! ## One program -/

/-- **One program** of the grid is safe from `init`: it reads its request's index entries and
table window, writes its window of `kv_indices`, and ends with `Q`. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    (i : Fin (launch d).size) :
    Safe E (R d init (i : Nat)) (W d init (i : Nat)) (fun _ m => Q d init (i : Nat) m) init
      ((model d).body ((launch d).ctx i)) := by
  simp only [model, create_flashinfer_kv_indices_triton, tl.program_id, launch_pid]
  have hb : (i : Nat) < d.bs := lt_of_lt_of_eq i.isLt (launch_size d)
  have hi31 : (i : Nat) < 2 ^ 31 := by have := hd.grid_lt; omega
  have hlen : (i : Nat) < d.req_pool_indices.len := by have := hd.rplen; omega
  obtain ⟨v0, hv0⟩ := Buf.read_some d.req_pool_indices hpre.2.1 hlen
  refine safe_load1 (v0 := v0) ?_ ?_ ?_
  · rw [ptr_zero_add d.req_pool_indices (i : Nat) hi31]
    exact hv0
  · intro x hx
    rw [ptr_zero_add d.req_pool_indices (i : Nat) hi31] at hx
    simp only [R]
    exact Or.inl (Or.inl (Or.inl (Or.inl ⟨(i : Nat), rfl, hx⟩)))
  refine safe_convert (d := d.dP) (d' := .i64) (x := Tensor.scalar v0) ?_ ?_
  · intro u
    simpa only [Tensor.scalar] using exists_convRel_i64 hd.dP_ok E.P v0
  · intro r hr
    have hp : r.item.toInt = rpi d init (i : Nat) := by
      have h := hr ()
      have hv : r.item.toInt = ivOf v0 := by
        simpa only [Tensor.scalar, Tensor.item] using convRel_i64_ivOf hd.dP_ok h
      rw [hv]
      simp only [rpi, valOf, hv0]
    have hkl : (i : Nat) < d.kv_indptr.len := by have := hd.plen; omega
    obtain ⟨v1, hv1⟩ := Buf.read_some d.kv_indptr hpre.2.2.2.1 hkl
    refine safe_load1 (v0 := v1) ?_ ?_ ?_
    · rw [ptr_zero_add d.kv_indptr (i : Nat) hi31]
      exact hv1
    · intro x hx
      rw [ptr_zero_add d.kv_indptr (i : Nat) hi31] at hx
      simp only [R]
      exact Or.inl (Or.inl (Or.inl (Or.inr ⟨(i : Nat), rfl, hx⟩)))
    have hoff : v1.toInt = ind d init (i : Nat) := by
      simp only [ind, valOf, hv1, ivOf]
    by_cases hS : d.hasStart = true
    · rw [ite_eq_left hS]
      dsimp only
      have hstart : (d.kv_start_idx.ptr 0).addr ≠ 0 := by
        simpa only [Buf.ptr, Int.zero_mul, Int.add_zero] using hd.start_nz hS
      rw [ite_eq_left hstart]
      have hsl : (i : Nat) < d.kv_start_idx.len := by have := hd.stlen hS; omega
      have hvs : d.kv_start_idx.Valid init := hpre.2.2.2.2.2.1 hS
      obtain ⟨v2, hv2⟩ := Buf.read_some d.kv_start_idx hvs hsl
      simp only [Prog.bind_eq]
      rw [Prog.bind_assoc]
      refine safe_load1 (v0 := v2) ?_ ?_ ?_
      · rw [ptr_zero_add d.kv_start_idx (i : Nat) hi31]
        exact hv2
      · intro z hz
        rw [ptr_zero_add d.kv_start_idx (i : Nat) hi31] at hz
        simp only [R, hS]
        exact Or.inl (Or.inr ⟨(i : Nat), rfl, hz⟩)
      rw [Prog.bind_assoc]
      refine safe_convert (d := d.dS) (d' := .i32) (x := Tensor.scalar v2) ?_ ?_
      · intro u
        simpa only [Tensor.scalar] using exists_convRel_i32 hd.dS_ok E.P v2
      · intro c hc
        obtain ⟨-, -, hi0, hi1, -⟩ := hpre.2.2.2.2.2.2.1 (i : Nat) hb
        have hsidx : sidx d init (i : Nat) = ivOf v2 := by
          simp only [sidx, hS, valOf, hv2, ↓reduceIte]
        have hfit : IntV.SFits 32 (ivOf v2) := by
          rw [← hsidx]
          exact IntV.I32.sfits (by linarith) hi1
        have hx : c.item.toInt = sidx d init (i : Nat) := by
          have h := hc ()
          have hv : c.item.toInt = ivOf v2 := by
            simpa only [Tensor.scalar, Tensor.item] using convRel_i32_ivOf hd.dS_ok h hfit
          rw [hv, hsidx]
        show Safe E (R d init (i : Nat)) (W d init (i : Nat)) (fun _ m => Q d init (i : Nat) m)
          init (tailProg d (i : Nat) r.item c.item v1)
        exact safe_tail hd hpre E (i : Nat) hb r.item c.item v1 hp hx hoff
    · rw [ite_eq_right hS]
      dsimp only
      have hs0 : sidx d init (i : Nat) = 0 := by
        simp only [sidx]
        rw [ite_eq_right hS]
      have hx : (IntV.ofInt 0 : I32).toInt = sidx d init (i : Nat) := by
        rw [IntV.toInt_ofInt (w := 32) (by norm_num) (IntV.I32.sfits (by norm_num) (by norm_num)), hs0]
      show Safe E (R d init (i : Nat)) (W d init (i : Nat)) (fun _ m => Q d init (i : Nat) m)
        init (tailProg d (i : Nat) r.item (IntV.ofInt 0) v1)
      exact safe_tail hd hpre E (i : Nat) hb r.item (IntV.ofInt 0) v1 hp hx hoff

/-! ## The claim -/

/-- **The claim.** Each program is safe on its own window (`safe_program`), the footprints are
disjoint, and the programs' results give `Post`. -/
theorem main : claim.Holds := by
  intro P _ d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (fun i => R d init (i : Nat)) (fun i => W d init (i : Nat))
    (fun i m => Q d init (i : Nat) m) (launch_valid hd) (fun i => safe_program hd hpre _ i)
    (fun i j hij => W_disj_aux hd hpre hij) (fun i => R_disj_W_aux hd hpre i)
    (fun fin hloc hfr => post_of_aux hd hpre hloc hfr)

end Claims.KvIndices

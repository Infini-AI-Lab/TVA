import Proof.PhaseC

/-!
# The core is safe

The three phases in a row: `core` leaves its tile of `out` as `Qt` says, given the pointers, index
tensors, limits and the scale the variant's set-up computes.
-/

namespace Claims.SsdChunkScan

open TVA

/-- **The core is safe.** Given the pointers, index tensors, limits and scale of a program `(k, pm, pn, h)`, `core` leaves its tile of `out` as `Qt` says. -/
theorem core_safe {w : Nat} {E : Env} {d : Inst} {init : Mem} {k pm pn h : Nat}
    (S : Std d init E k pm pn h)
    {C_ptr : Ptr .bf16} {prev_ptr : Ptr .f32} {hdimS dstS : I32} {cb_ptr : Ptr .f32}
    {x_ptr : Ptr .bf16} {dt_ptr dA_ptr : Ptr .f32} {out_ptr D_ptr : Ptr .bf16}
    {pid_h pid_m pid_n c_off : I32} {offs_m : Tensor [BM] I32} {offs_n : Tensor [BN] I32}
    {widen : {n : Nat} → Tensor [n] I32 → Tensor [n] (IntV true w)}
    {csw limit kmaxarg : IntV true w} {dA_cs_m scale_m : Tensor [BM] F32} {u : Bool} {r : Nat}
    (hw : 32 ≤ w)
    (hui : u = true → d.hasInit = true ∧ r < d.nreq)
    (hC : C_ptr = d.C.ptr ((lc d k * d.cs * d.cT + grp d h * d.N : Nat) : Int))
    (hS : prev_ptr = (stBuf d u).ptr (stBase d u (lc d k) r h : Nat))
    (hu : ∀ ii, RowOk d k pm ii → (d.hasInit && fromInit d (tokOf d k pm ii)) = u)
    (hr : u = true → ∀ ii, RowOk d k pm ii → reqOf d (tokOf d k pm ii) = r)
    (hhd : hdimS = IntV.ofNat d.N) (hds : dstS = IntV.ofNat 1)
    (hcb : cb_ptr = d.cb.ptr (cbIdx d (lc d k) (grp d h) 0 0 : Int))
    (hx : x_ptr = d.x.ptr (xIdx d (lc d k * d.cs) h 0 : Int))
    (hdt : dt_ptr = d.dt.ptr (dtIdx d h (lc d k) 0 : Int))
    (hdA : dA_ptr = d.dA.ptr (dtIdx d h (lc d k) 0 : Int))
    (hout : out_ptr = d.out.ptr ((lc d k * d.cs * (d.H * d.P) + h * d.P : Nat) : Int))
    (hD : D_ptr = d.D.ptr 0) (hph : pid_h.toInt = (h : Int)) (hpm : pid_m.toInt = (pm : Int))
    (hpn : pid_n.toInt = (pn : Int)) (hco : c_off.toInt = (lo d k : Int))
    (hom : ∀ ii : Fin BM, (offs_m.get1 ii).toInt = (rowOf d k pm ii : Int))
    (hon : ∀ jj : Fin BN, (offs_n.get1 jj).toInt = (colOf pn jj : Int))
    (hwiden : ∀ {n : Nat} (x : Tensor [n] I32) (i : Fin n),
      ((widen x).get1 i).toInt = (x.get1 i).toInt)
    (hcsw : csw.toInt = (d.cs : Int)) (hlim : limit.toInt = (lend d k : Int))
    (hkm : kmaxarg.toInt = ((BM * (pm + 1) : Nat) : Int))
    (hdAm : ∀ ii : Fin BM, RowOk d k pm ii → ∃ v : F32,
      init.read .f32 (d.dA.ptr (dtIdx d h (lc d k) (rowOf d k pm ii) : Int)).addr = some v ∧
        dA_cs_m.get1 ii = v)
    (hsc : Vals init d → ∀ ii : Fin BM, RowOk d k pm ii →
      ScaleOK init d (tokOf d k pm ii) h (scale_m.get1 ii)) :
    Safe E (R d) (Wt d k pm pn h) (fun _ m => Qt d init k pm pn h m) init
      (core d C_ptr prev_ptr hdimS dstS cb_ptr x_ptr dt_ptr dA_ptr out_ptr D_ptr pid_h pid_m pid_n
        c_off offs_m offs_n widen csw limit kmaxarg dA_cs_m scale_m) := by
  unfold core
  refine phaseA_safe S hui hC hS hu hr hhd hds hom hon
    (fun ii => (hwiden offs_m ii).trans (hom ii)) hlim hsc (fun acc₀ hA => ?_)
  refine phaseB_safe S hw hcb hx hdt hdA hom hon hco hwiden hcsw hlim hkm hdAm hA (fun acc hB => ?_)
  exact phaseC_safe S hw hx hout hD hph hpm hpn hco hom hon hwiden hlim hB

end Claims.SsdChunkScan

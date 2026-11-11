import Proof.Program

/-!
# The proof of `matmul_f16`

Program `p` of the grid computes the `128 × 256` tile `(pidM p, pidN p)` of `c`. It reads the
matrices `a` and `b` (`R`), writes its tile of `c` (`W`), and leaves `OutSpec` in it (`Q`).

1. `safe_program`: one program, walked line by line with the rules of `TVA.Proof.Rules`. The
   grouped program-id arithmetic gives the tile; the loop over `K` carries the invariant `InInv`
   (each accumulator lane is within `AccB` of the exact partial sum), one step by `lane_step`
   from the device's `float16` dot fact; the final `float32 → float16` rounding gives `OutSpec`.
2. `tiles_disjoint`, `read_write_disjoint`: distinct programs own distinct tiles of `c`, and `c`
   shares no byte with `a` or `b`.
3. `post_of_tiles`: the tiles cover the matrix `c`, so the programs' results give `Post`.
4. `main`: `correct_of_safe` composes the programs.
-/

namespace Claims.Matmul

open TVA

/-- **The claim.** -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (R d init) (W d init) (Q d init) (launch_valid hd) (fun i => ?_)
    (fun i j hij => tiles_disjoint hd init hpre i j hij)
    (fun i => read_write_disjoint hd init hpre i) (fun fin hQ hfr => post_of_tiles hd init hpre fin hQ hfr)
  exact safe_program hd P init hpre i hP.dotF16 (hP.dotTotal _)

end Claims.Matmul

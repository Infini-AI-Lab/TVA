import Proof.Program

/-!
# The proof of `tma_persistent_matmul_f16`

Program `i` of the grid runs the tiles `i, i + NUM_SMS, ...` of `c = a · bᵀ`; it reads `a` and `b`
and writes only the elements of `c` in its tiles, so the programs are independent.

1. `Proof.Setup`: the launch, the tiles of a program and their disjointness, the program-id arithmetic;
2. `Proof.Reduction`: the error recursion of the inner loop (`acc += tl.dot`) and its magnitudes;
3. `Proof.Epilogue`: the two stored halves of a tile, and `Post` from the programs' tile facts;
4. `Proof.InnerLoop`: the inner loop keeps its invariant and is safe;
5. `Proof.Program`: one program is safe (`safe_body`), and `launch_correct` composes the programs
   with `correct_of_safe`; `main` applies it to every covered instance.
-/

namespace Claims.TmaPersistentMatmul

open TVA

/-- **The claim**: `launch_correct` on every covered instance and memory. -/
theorem main : claim.Holds := fun _ hP _ hd _ hPre => launch_correct hP hd hPre

end Claims.TmaPersistentMatmul

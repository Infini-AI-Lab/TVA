# What `tvr audit` catches

Library evidence (not a kernel's), for library issue L7. It tests TVA's tool, not the device:
`tools/tvr.py` (Python 3.13, standard library only), Lean 4.35.0 header syntax, 2026-10-27.
Script: [probes.py](probes.py); output: [out.txt](out.txt).

**Question.** Does `tvr audit` flag each kind of reference to another project that the protocol
says it checks ([REVIEW_PROTOCOL.md](../../docs/REVIEW_PROTOCOL.md) §2, "Enforcement"), and does it
pass TVA's own text?

**Method.** Each probe copies the TVA tree into a scratch directory, changes one thing, and runs
the copy's own `tools/tvr.py audit`. The probe passes when the audit flags it exactly when the rule
asks. The other project's name is given on the command line (it must be one of the tool's hashed
names), so neither the script nor its output contains it, and the script joins its paths at run
time. Controls (I15, I16, N8, P9) are changes the audit must *not* flag: a kernel importing its own
module, the word `import` inside a docstring, the name's letters split across a case boundary
inside a longer word, and paths inside TVA, the toolchain and the system, and a URL.

**What the audit reads** (`tools/tvr.py`):
- `lean_header`: the header of each Lean file as Lean 4.35 parses it
  (`Lean.Parser.Module.header`: `[module] [prelude] ([public] [meta] import [all] ident)*`,
  comments as whitespace, `«»` names). An unreadable header, or an `import` line after the header
  (outside comments), is a problem. The files are every `.lean` file under `lean/` (outside
  `.lake`) for the library, and each kernel's `lean/` folder.
- `lake_violations`: `lean/lakefile.toml` parsed with `tomllib`; `lakefile.lean`;
  `lake-manifest.json`'s direct packages.
- `scan_files` over `text_files`: every file outside `.lake`, `.venv`, `.git` and `__pycache__`,
  whatever its extension; non-UTF-8 files and symbolic links out of TVA are problems.
  `foreign_refs` matches the hashed names on words and word pieces, and finds home paths (by whole
  component), other absolute paths, and relative paths that climb out of TVA.

**Result.** The baseline tree is independent, and each of the 53 probes is flagged or passed as the
rule asks (`out.txt`), including the 21 kinds of the round-1 library review (I1–I3, K1–K4, N1–N5,
P1–P5, F1–F4) and 32 more.

**Limits.** The name check is lexical: a name spelled with other letters (a transliteration) or
split by punctuation is not found. The import check reads the files lake builds; a review's work
file may import any module TVA's lake environment provides (TVA, the kernels, Mathlib and Mathlib's
own dependencies), since the lake check leaves no other package to import from.

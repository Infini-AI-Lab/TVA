# What `tvr audit` catches

Library evidence (not a kernel's), for library issue L7. It tests TVA's tool, not the device:
`tools/tvr.py` (Python 3.13, standard library only), Lean 4.35.0 header syntax, 2026-10-28.
Script: [probes.py](probes.py); output: [out.txt](out.txt).

**Question.** Does `tvr audit` flag each kind of reference to another project that the protocol
says it checks ([REVIEW_PROTOCOL.md](../../docs/REVIEW_PROTOCOL.md) §2, "Enforcement"), and does it
pass TVA's own text?

**Method.** Each probe copies the TVA tree into a scratch directory, changes one thing, and runs
the copy's own `tools/tvr.py audit`. The probe passes when the audit flags it exactly when the rule
asks. The other project's name is given on the command line (it must be one of the tool's hashed
names), so neither the script nor its output contains it, and the script joins its paths at run
time. Controls (I15, I16, N8, P9, P19) are changes the audit must *not* flag: a kernel importing
its own module, the word `import` inside a docstring, the name's letters split across a case
boundary inside a longer word, paths inside TVA, the toolchain and the system, and a URL; and the
round-3 spellings below pointing inside TVA or at its parent folder, placeholder paths
(`round-<N>/author/notes.md`, `{kernel}/review/x`), and a web URL whose path looks like a home path.

Round 5 (library issue L7) added P35–P51: the spellings the round-4 scanner missed, from library
review round 4 (lr-4b), and the files pinned by sha256. P35 `fpv` in double quotes, P36 the sibling
in single quotes, P37 empty quotes after `fpv`, P38 the named entity `&sol;`, P39 Python octal
`\057`, P40 `\N{SOLIDUS}`, P41 brace expansion after `fpv/`, P42 `${HOME:-${X}}`, P43
`%USERPROFILE%` then backslashes, P44 PowerShell `$HOME` then backslashes, P45 a backquote command
substitution of the home directory, P46 `$(…)` nested three deep, P47 a line continuation inside a
path, P48 a kernel's source with a home path into a sibling and its sha256 re-pinned in
`kernel.json`, P49 a new file in `primitives/` pinned in the semantics manifest; P50 is their
control (the same spellings inside TVA, `fpv/"<dir>"`, `fpv/{tva,<dir>}`, and Markdown's
`` `max`/`min` ``), and P51 the lexical limit protocol §2 states (a glob, `cd ~/fpv` then a
relative path, a shell variable), which passes. The baseline also shows that the kernel's source,
now read by every test, passes with its tutorial placeholder `save_path='/path/to/results/'`.

Round 4 (library issue L7) added P20–P34 and N9: the spellings round 3's scanner missed, from both
library reviewers of round 3, and the rule that does not depend on what precedes a path. P20
`${HOME:-}/…`, P21 `${HOME%/}/…`, P22 `$(echo ~)/…`, P23 `\u002F` escapes in JSON, P24 and P25
shell redirects with no space (`>/home/…`, `2>~/…`), P26–P28 home paths not under `fpv` right after
`)`, `]` and `-`, P29 `$(dirname $0)` followed by `..` out of TVA, P30 a Python f-string
`{os.environ['HOME']}/fpv/<sibling>`, P31 `%2F` escapes, P32 `fpv/<sibling>` with no home prefix,
P33 a sibling whose name starts with `tva` (`…/tva-old`); P34 is their control (TVA's own `fpv/tva`, placeholders `fpv/<dir>`, `fpv/…`,
redirects into TVA, `$(date)`, `100%`); N9 the name with its first letter as a `\u` escape.

Round 3 (library issue L7) added P10–P19, home paths in the spellings a shell or a browser
resolves and paths glued to punctuation, each pointing at a sibling folder of TVA:
`$HOME/…` (P10), `${HOME}/…` (P11), a `file:` URL with three slashes (P12) and in angle brackets
(P13), `~user/…` (P14), an absolute home path right after a semicolon (P15), `%USERPROFILE%/…`
(P16), a relative path after `=` that climbs out of TVA (P17), and a home path with JSON-escaped
slashes in an evidence `.json` file (P18); P19 is their control.

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

**Result** (rerun 2026-10-28, round 5). The baseline tree is independent, and each of the 96
probes is flagged or passed as the rule asks (`out.txt`: "96 probes, 0 wrong"), including the 21
kinds of the round-1 library review (I1–I3, K1–K4, N1–N5, P1–P5, F1–F4) and 75 more. P10–P18,
P20–P33 and P35–P49 are each flagged (P17 as a relative path, the others as a path out of TVA),
and N9 as the name; the controls P19, P34 and P50, and the stated limit P51, pass. The round-3 reviewers' own probes
(`docs/library-review/round-3/lr-3a/work/l7_more_spellings.py`, `lr-3b/work/l7_probes.py`) give
every case as their rule asks against the round-4 tool.

**Limits.** The text scan is lexical: it reads paths as written, so a path a program or a shell
computes at run time (`os.path.join`, an f-string with a variable directory, `Path.home()`, a glob,
`cd` then a relative path, a shell variable) is not found, unless it shows `fpv/<sibling>` (P51). The guarantee for results and lemmas is the import and lake checks. The
name check is lexical too: a name spelled with other letters (a transliteration) or
split by punctuation is not found. The import check reads the files lake builds; a review's work
file may import any module TVA's lake environment provides (TVA, the kernels, Mathlib and Mathlib's
own dependencies), since the lake check leaves no other package to import from.

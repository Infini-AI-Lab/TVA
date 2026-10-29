"""lr-3a, round 3 (L7): more spellings of a home path, tested with the tool's own `foreign_refs`.

Run from the TVA root: python3 -I docs/library-review/round-3/lr-3a/work/l7_more_spellings.py

`otherproj` stands for a folder of another project that is not named in the text (the name check
is a separate test). Each probe should be flagged: it is a path a shell, Python or a JSON reader
resolves to a home path outside TVA. The paths are assembled at run time, so this file holds none.
"""
import importlib.util
import os

spec = importlib.util.spec_from_file_location("tvr", os.path.join(os.getcwd(), "tools", "tvr.py"))
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)

S, H, T = "/", "~", "fpv" + "/" + "otherproj"
PROBES = [
    ("control: $HOME/...", "cd $HOME" + S + T),
    ("control: ~/fpv/tva/../...", "cd " + H + S + "fpv/tva/.." + S + "otherproj"),
    ("${HOME:-}/... (parameter expansion with a default)", "cd ${HOME:-}" + S + T),
    ("${HOME%/}/... (parameter expansion, suffix removed)", "cd ${HOME%/}" + S + T),
    ("$(echo ~)/... (command substitution)", "cd $(echo " + H + ")" + S + T),
    ("`echo ~`/... (backquote substitution)", "cd `echo " + H + "`" + S + T),
    ("\"$(printenv HOME)\"/...", "cd \"$(printenv HOME)\"" + S + T),
    ("JSON \\u002F escapes", "{\"p\": \"" + H + "\\u002F" + T.replace("/", "\\u002F") + "\"}"),
    ("control: JSON \\/ escapes", "{\"p\": \"" + H + "\\/" + T.replace("/", "\\/") + "\"}"),
]
for name, text in PROBES:
    refs = t.foreign_refs(text)
    print(f"{'flagged' if refs else 'PASSES '}: {name}")

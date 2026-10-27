#!/usr/bin/env python3
"""tvr: TVA's prover/reviewer protocol (docs/REVIEW_PROTOCOL.md).

Each kernel is one folder, kernels/<kernel>/ (the layout is section 1 of the protocol).

  tvr new     KERNEL --module M --namespace NS --source FILE
                                scaffold kernels/KERNEL (folder, manifest, lake library)
  tvr check   KERNEL            run the gate: build, read the Lean claim (tools/claim_reader.lean.in); write gate.json,
                                claim.md and a snapshot of the item cones
  tvr assign  KERNEL REVIEWER   assign a reviewer
  tvr file    KERNEL REVIEWER   record the reviewer's review/round-N/REVIEWER/filing.json
  tvr respond KERNEL            record the prover's review/round-N/prover/response.json
  tvr decide  KERNEL ID TEXT    record the user's decision on an escalated finding
  tvr note    KERNEL TEXT       record a coordinator's note
  tvr status  KERNEL [--as R]   print the status (REVIEW.md has the whole record)
  tvr prompt  KERNEL prover|reviewer [ID]
  tvr prompt  library author|reviewer ROUND [ID] [--area AREA]
                                print a role's prompt (docs/prompts/) with its fields filled in
  tvr index                     regenerate kernels/README.md and docs/library-review/REPORTS.md
  tvr audit                     check that the library and every kernel are independent (protocol §2)
  tvr start   KERNEL reviewer ID | KERNEL prover | library ROUND author|ID
                                write the skeleton of a role's output (the formats: protocol §11)
  tvr lint    KERNEL reviewer ID | KERNEL prover | library ROUND author|ID
                                check that a role's output is regular before it is recorded

Run it as `python3 -I tools/tvr.py ...` from the TVA root. Standard library only.
"""
import datetime
import hashlib
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tomllib
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KROOT = os.path.join(ROOT, "kernels")
LAKE = os.environ.get("TVR_LAKE", os.path.expanduser("~/.elan/bin/lake"))
STD_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
KINDS = {
    # kind: (blocking by default, allowed target prefixes, assumption kind it may target)
    "MODEL_MISMATCH": (True, ("item:model", "map:", "src:", "decl:"), None),
    "MAP_INCOMPLETE": (True, ("item:map", "map:", "src:"), None),
    "UNSUPPORTED": (True, ("src:", "map:"), None),
    "CONDITION_TOO_NARROW": (True, ("item:conditions", "decl:"), None),
    "CONDITION_WRONG": (True, ("item:conditions", "decl:"), None),
    "SPEC_TOO_WEAK": (True, ("item:spec", "decl:"), None),
    "SPEC_WRONG": (True, ("item:spec", "decl:"), None),
    "RULE_MISCITED": (True, ("item:assumption:",), "rule"),
    "ASSUMPTION_FALSE": (True, ("item:assumption:",), "kernel"),
    "ASSUMPTION_TOO_STRONG": (True, ("item:assumption:",), "kernel"),
    "ASSUMPTION_UNCLEAR": (True, ("item:assumption:",), "kernel"),
    "EVIDENCE_REQUEST": (True, ("item:assumption:",), "kernel"),
    "LIBRARY_REPORT": (False, ("decl:", "lib:", "src:"), None),
    "DOC": (False, ("item:", "decl:", "src:", "map:", "lib:", "doc:"), None),
    "QUESTION": (False, ("item:", "decl:", "src:", "map:", "lib:", "doc:"), None),
}
RESPONSES = {"REVISED", "DISPUTED", "CLARIFIED", "DEFERRED", "REFERRED"}
# The drop reasons (protocol §7). Each is a rule of docs/TRANSCRIPTION.md, which says what it drops.
DROPS = {"no-op:debug_barrier", "no-op:print", "no-op:cache", "no-op:range-kw", "no-op:gdc",
         "decl", "config", "compile-check", "dead", "comment"}
NOTE_REQUIRED = {"config", "compile-check", "dead"}
ESCALATE_AFTER = 2  # re-raises after which a finding goes to the user
DOC_SECTIONS = ("Specification", "Conditions", "Assumptions", "Proof")
RULE_ARGS = ("Inst", "Inst.ok", "model", "launch")  # the kernel constants a [rule] may mention
# Independence (docs/REVIEW_PROTOCOL.md §2): TVA stands on itself and Mathlib only.
LEAN_ROOTS = ("TVA", "Mathlib")  # the import roots allowed besides a kernel's own modules
# Other projects, by the sha256 of their lower-case name (with the name's length), so that TVA does
# not name them. A name is found anywhere inside a run of letters and digits, whatever the case.
FOREIGN = {"6419784d55bd27db790c84bc28d5a48e7b7d4cdac419329aea87ffb252f2c2ed": 3}
# Paths under the home directory that are not another project: TVA itself, the Lean toolchain, and
# Triton's own compile cache. Each is a sequence of whole path components.
HOME_OK = (("fpv", "tva"), (".elan",), (".triton",))
# Absolute paths outside /home that are the system's, not a project's.
SYSTEM_DIRS = {"usr", "bin", "sbin", "lib", "lib64", "etc", "dev", "proc", "sys"}
SKIP_DIRS = {".lake", ".venv", ".git", "__pycache__"}  # build outputs, the Python environment, git
LREVIEW = os.path.join(ROOT, "docs", "library-review")


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def file_sha(path):
    with open(path, "rb") as f:
        return sha(f.read())


def die(msg):
    print(f"tvr: {msg}", file=sys.stderr)
    sys.exit(1)


def kpath(k, *parts):
    return os.path.join(KROOT, k, *parts)


def rel(path, base=ROOT):
    return os.path.relpath(path, base)


def link(target, page_dir):
    """A Markdown link to `target` (a path from the TVA root) from a page in `page_dir`."""
    return os.path.relpath(os.path.join(ROOT, target), page_dir)


def load_json(path, default=None):
    if not os.path.exists(path):
        if default is not None:
            return default
        die(f"missing {rel(path)}")
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            die(f"{rel(path)} is not valid JSON: {e}")


def dump_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def manifest(k):
    m = load_json(kpath(k, "kernel.json"))
    m["name"] = k
    return m


def lean_file(k, man, mod):
    """The file of a module: the kernel's own modules live in kernels/K/lean, the library's in lean/."""
    parts = mod.split(".")
    if parts[0] == man["lean"]["module"].split(".")[0]:
        return kpath(k, "lean", *parts) + ".lean"
    return os.path.join(ROOT, "lean", *parts) + ".lean"


def kernel_lean_files(k):
    out = []
    for dirpath, dirnames, names in os.walk(kpath(k, "lean")):
        dirnames[:] = [n for n in dirnames if n not in SKIP_DIRS]
        out += [os.path.join(dirpath, n) for n in sorted(names) if n.endswith(".lean")]
    return sorted(out)


def parse_lines(spec):
    out = []
    for x in spec:
        if isinstance(x, int):
            out.append(x)
        else:
            a, b = str(x).split("-")
            out += list(range(int(a), int(b) + 1))
    return out


_PIECES = {}  # identifier piece -> whether it is, starts with or ends with a hashed name


def _piece_hit(piece):
    low = piece.lower()
    if low not in _PIECES:
        _PIECES[low] = any(sha(low) == h or (len(low) > n and (sha(low[:n]) == h or sha(low[-n:]) == h))
                           for h, n in FOREIGN.items())
    return _PIECES[low]


def _pieces(text):
    """The pieces of every word: split on `_`, and between lower and upper case, letters and digits."""
    out = set()
    for word in set(re.findall(r"\w+", text)):
        out.add(word)
        for part in word.split("_"):
            out.update(re.findall(r"[^\W\d_](?:[^\W\d_A-Z]*)|[A-Z]+(?![a-z])|[A-Z][a-z]+|\d+", part))
            out.update(re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[^\W\d_A-Z]+|\d+", part))
    return out


def upstream_files():
    """Triton 3.8.0's own files, as pinned by sha256: the primitives, the tutorials and each kernel's
    source. Their text is Triton's, so only the name check applies to them (a placeholder path such
    as a tutorial's `save_path` is not a path into a project)."""
    out = set()
    for d, man, key in (("primitives", "semantics-manifest.json", "sources"), ("primitives", "manifest.json", "primitives"),
                        ("tutorials", "inventory.json", "tutorials")):
        for e in load_json(os.path.join(ROOT, d, man), {}).get(key, []):
            f = os.path.join(ROOT, d, e.get("source", ""))
            if os.path.isfile(f) and file_sha(f) == e.get("sha256"):
                out.add(os.path.realpath(f))
    for k in kernels():
        src = manifest(k)["source"]
        f = kpath(k, src["file"])
        if os.path.isfile(f) and file_sha(f) == src["sha256"]:
            out.add(os.path.realpath(f))
    return out


# Paths. Before the path test, a text is normalised (NFKC, `\/` read as `/`), a `file:` URL is read as
# the path it names, every other URL is removed, and the home directory's spellings (`$HOME`,
# `${HOME}`, `%HOME%`, `%USERPROFILE%`, `$env:HOME`) are read as `~`. Then an absolute path is a `/`, `~/`
# or `~<user>/` that follows any character that cannot be part of a path name (a space, a quote, a
# bracket, `=`, `:`, `;`, `,`, …; not a closing bracket, `>` or `}`, which end a placeholder such as
# `<N>` or `{kernel}` inside a path), and a relative path is any run of path characters.
_PATH_CHARS = r"[^\s`'\"()\[\]{}<>,;|*]"
_FILE_URL = re.compile(r"(?i)\bfile:(?://(?:localhost)?)?(?=/)")
_URL = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s`'\"<>()\[\]{}|;,]*")
_HOME_VARS = re.compile(r"\$\{HOME\}|\$HOME\b|%HOME%|%USERPROFILE%|\$env:(?:HOME|USERPROFILE)\b")
_ABS = re.compile(r"(?<![\w.~/\\>})\]-])(?:~[\w.-]*)?/(?=[\w.~])" + _PATH_CHARS + r"*")
_REL = re.compile(r"[^\s`'\"()\[\]{}<>,;|*=:]+")


def _home_ok(parts):
    """A path under the home directory (as components) is TVA's, the toolchain's, Triton's cache, or a
    folder that holds one of them (such as the home directory itself)."""
    return any(tuple(parts[:len(ok)]) == ok or tuple(ok[:len(parts)]) == tuple(parts) for ok in HOME_OK)


def _path_text(text):
    """A text as the path test reads it (see the comment above `_PATH_CHARS`)."""
    t = unicodedata.normalize("NFKC", text).replace("\\/", "/")
    t = _FILE_URL.sub(" ", t)
    t = _URL.sub(" ", t)
    return _HOME_VARS.sub(" ~", t)


def foreign_refs(text, where=None, paths=True):
    """References to other projects in a text: a hashed name, or a path that leads outside TVA.

    A name counts as a word, as a piece of a word (`Name_v2`, `NameLemmas`, `nameLemmas`), or at the
    start or end of a one-case piece (`namelemmas`), in any case. A path counts when it is a home path
    (`~/…`, `~<user>/…`, `/home/<user>/…`, and the spellings `_path_text` reads as these, a `file:`
    URL included) not under TVA, the Lean toolchain or Triton's cache; when it is another absolute
    path of two components or more outside the system directories; or when it is relative, climbs
    with `..`, and leaves TVA from `where` (the directory of the file that holds it). `paths=False`
    checks the name only."""
    out = set()
    if any(_piece_hit(x) for x in _pieces(unicodedata.normalize("NFKC", text))):
        out.add("the name of another project")
    if not paths:
        return sorted(out)
    t = _path_text(text)
    for m in _ABS.finditer(t):
        tok = m.group(0).rstrip(".:")
        if tok.startswith("~"):
            rest = tok.split("/", 1)[1] if "/" in tok else ""
            ok = _home_ok([x for x in posixpath.normpath("/" + rest).split("/") if x])
        else:
            parts = [x for x in posixpath.normpath(tok).split("/") if x]
            if parts and parts[0] == "home":
                ok = len(parts) <= 2 or _home_ok(parts[2:])
            else:
                ok = len(parts) < 2 or parts[0] in SYSTEM_DIRS
        if not ok:
            out.add(f"a path outside TVA ({tok[:60]})")
    if where is not None:
        root = os.path.realpath(ROOT)
        for m in _REL.finditer(t):
            tok = m.group(0)
            tok = tok if tok.endswith("..") else tok.rstrip(".")
            if tok.startswith(("/", "~")) or ".." not in tok.split("/"):
                continue
            dest = os.path.normpath(os.path.join(where, tok))
            if dest != root and not dest.startswith(root + os.sep):
                out.add(f"a relative path that leaves TVA ({tok[:60]})")
    return sorted(out)


def scan_files(paths):
    """Every file must be UTF-8 text that names no other project and has no path into one; a symbolic
    link must stay inside TVA."""
    bad = []
    root = os.path.realpath(ROOT)
    upstream = upstream_files()
    for p in paths:
        if os.path.islink(p):
            dest = os.path.realpath(p)
            if dest != root and not dest.startswith(root + os.sep):
                bad.append(f"{rel(p)}: a symbolic link out of TVA")
                continue
        try:
            text = open(p, encoding="utf-8").read()
        except UnicodeDecodeError:
            bad.append(f"{rel(p)}: not UTF-8 text, so it cannot be checked")
            continue
        except OSError as e:
            bad.append(f"{rel(p)}: unreadable ({e.strerror})")
            continue
        for r in foreign_refs(text, os.path.dirname(os.path.realpath(p)), os.path.realpath(p) not in upstream):
            bad.append(f"{rel(p)}: {r}")
    return bad


def text_files(*dirs, skip=()):
    """Every file under the given files and directories, whatever its extension, except build
    outputs, the Python environment and git (SKIP_DIRS), and the directories in `skip`. Symbolic
    links to directories are listed, not followed (scan_files checks where they lead)."""
    out = []
    skip = {os.path.realpath(x) for x in skip}
    for d in dirs:
        if os.path.isfile(d) or os.path.islink(d):
            out.append(d)
            continue
        for dirpath, dirnames, names in os.walk(d):
            keep = []
            for n in dirnames:
                full = os.path.join(dirpath, n)
                if n in SKIP_DIRS or os.path.realpath(full) in skip:
                    continue
                if os.path.islink(full):
                    out.append(full)
                    continue
                keep.append(n)
            dirnames[:] = keep
            out += [os.path.join(dirpath, n) for n in names]
    return sorted(set(out))


_IDENT = r"(?:«[^»]*»|[^\W\d][\w'!?\u2080-\u209c\u1d62-\u1d6a\u2c7c]*)(?:\.(?:«[^»]*»|[^\W\d][\w'!?\u2080-\u209c\u1d62-\u1d6a\u2c7c]*))*\.?"


def lean_header(text):
    """The modules a Lean file imports, read from its header as Lean 4.35 parses it
    (`Lean.Parser.Module.header`: `[module] [prelude] ([public] [meta] import [all] ident)*`, with
    comments as whitespace). Returns `(imports, problem)`; `problem` is set when the header cannot
    be read, or when an import appears after it."""
    pos, n, out = 0, len(text), []

    def skip(i):
        while i < n:
            if text[i].isspace():
                i += 1
            elif text.startswith("--", i):
                j = text.find("\n", i)
                i = n if j < 0 else j + 1
            elif text.startswith("/-", i) and not text.startswith(("/--", "/-!"), i):
                depth, i = 1, i + 2
                while i < n and depth:
                    if text.startswith("/-", i):
                        depth, i = depth + 1, i + 2
                    elif text.startswith("-/", i):
                        depth, i = depth - 1, i + 2
                    else:
                        i += 1
                if depth:
                    return None
            else:
                break
        return i

    def word(i):
        m = re.compile(_IDENT).match(text, i)
        return (m.group(0), m.end()) if m else (None, i)

    pos = skip(pos)
    if pos is None:
        return out, "an unterminated comment in the header"
    w, j = word(pos)
    if w == "module":
        pos = skip(j)
        w, j = word(pos)
    if w == "prelude":
        pos = skip(j)
    while pos is not None:
        i = pos
        w, j = word(i)
        mods = []
        while w in ("public", "meta") and w not in mods:
            mods.append(w)
            i = skip(j)
            if i is None:
                return out, "an unterminated comment in the header"
            w, j = word(i)
        if w != "import":
            break
        i = skip(j)
        if i is None:
            return out, "an unterminated comment in the header"
        w, j = word(i)
        if w == "all":
            i = skip(j)
            if i is None:
                return out, "an unterminated comment in the header"
            w, j = word(i)
        if w is None:
            return out, f"an import without a module name at offset {i}"
        out.append(w.rstrip(".").replace("«", "").replace("»", ""))
        pos = skip(j)
    if pos is None:
        return out, "an unterminated comment in the header"
    later = re.search(r"^[ \t]*(?:public[ \t]+)?(?:meta[ \t]+)?import[ \t]", _uncommented(text[pos:]), re.M)
    if later:
        return out, "an import after the header"
    return out, None


def _uncommented(text):
    """Lean text with its comments and docstrings (nested `/- -/`, `--`) replaced by spaces."""
    out, i, n = [], 0, len(text)
    while i < n:
        if text.startswith("--", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/-", i):
            depth, i = 1, i + 2
            while i < n and depth:
                if text.startswith("/-", i):
                    depth, i = depth + 1, i + 2
                elif text.startswith("-/", i):
                    depth, i = depth - 1, i + 2
                else:
                    out.append("\n" if text[i] == "\n" else " ")
                    i += 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def import_violations(files, roots):
    """Lean files whose header imports a module outside `roots`, or cannot be read."""
    bad = []
    for p in files:
        try:
            imports, problem = lean_header(open(p, encoding="utf-8").read())
        except (UnicodeDecodeError, OSError):
            bad.append(f"{rel(p)}: not readable UTF-8 Lean")
            continue
        if problem:
            bad.append(f"{rel(p)}: {problem}")
        bad += [f"{rel(p)} imports {m}" for m in imports if m.split(".")[0] not in roots]
    return bad


def lake_violations():
    """The lake configuration: Mathlib is the only requirement, and the libraries are TVA's own and one
    per kernel folder (`../kernels/<kernel>/lean`, with the kernel's module as its root)."""
    lean = os.path.join(ROOT, "lean")
    bad = [f"lean/{f}: lake reads it, and TVA's configuration is lakefile.toml only"
           for f in ("lakefile.lean", "lakefile") if os.path.exists(os.path.join(lean, f))]
    try:
        with open(os.path.join(lean, "lakefile.toml"), "rb") as f:
            cfg = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        return bad + [f"lean/lakefile.toml: unreadable ({e})"]
    bad += [f"lean/lakefile.toml: key {k!r} is not allowed" for k in sorted(set(cfg) - {
        "name", "version", "defaultTargets", "leanOptions", "require", "lean_lib"})]
    reqs = cfg.get("require", [])
    if not (isinstance(reqs, list) and len(reqs) == 1 and isinstance(reqs[0], dict)
            and reqs[0].get("name") == "mathlib" and reqs[0].get("scope") == "leanprover-community"
            and set(reqs[0]) <= {"name", "scope", "rev", "version"}):
        bad.append("lean/lakefile.toml: the only requirement must be mathlib (name, scope leanprover-community, "
                   "rev or version; no path or git)")
    libs = cfg.get("lean_lib", [])
    libs = libs if isinstance(libs, list) else [libs]
    names = [lib.get("name") for lib in libs if isinstance(lib, dict)]
    if names.count("TVA") != 1:
        bad.append("lean/lakefile.toml: exactly one lean_lib must be TVA")
    root = os.path.realpath(ROOT)
    for lib in libs:
        if not isinstance(lib, dict):
            bad.append("lean/lakefile.toml: a lean_lib that is not a table")
            continue
        name = lib.get("name")
        if name == "TVA":
            if set(lib) != {"name"}:
                bad.append("lean/lakefile.toml: the TVA lean_lib takes only its name (its sources are lean/TVA)")
            continue
        if not (isinstance(name, str) and os.path.exists(kpath(name, "kernel.json"))):
            bad.append(f"lean/lakefile.toml: lean_lib {name!r} is neither TVA nor a kernel folder's library")
            continue
        want_root = manifest(name)["lean"]["module"].split(".")[0]
        want = {"name": name, "srcDir": f"../kernels/{name}/lean", "roots": [want_root]}
        if lib != want or want_root in LEAN_ROOTS:
            bad.append(f"lean/lakefile.toml: lean_lib {name!r} must be exactly name, srcDir = \"../kernels/{name}/lean\" "
                       f"and roots = [\"{want_root}\"] (the kernel's module), with a root other than {', '.join(LEAN_ROOTS)}")
        src = os.path.realpath(os.path.join(lean, str(lib.get("srcDir", "."))))
        if src != os.path.realpath(kpath(name, "lean")) or not src.startswith(root + os.sep):
            bad.append(f"lean/lakefile.toml: lean_lib {name!r} has sources outside its kernel folder")
    for t in cfg.get("defaultTargets", []):
        if t not in names:
            bad.append(f"lean/lakefile.toml: default target {t!r} is not one of the libraries")
    man = load_json(os.path.join(lean, "lake-manifest.json"), {"packages": []})
    for p in man.get("packages", []):
        if not p.get("inherited") and (p.get("name") != "mathlib" or p.get("type") != "git" or
                                       p.get("url") != "https://github.com/leanprover-community/mathlib4"):
            bad.append(f"lean/lake-manifest.json: direct package {p.get('name')} (only Mathlib's repository is allowed)")
        elif p.get("type") != "git":
            bad.append(f"lean/lake-manifest.json: package {p.get('name')} is not a git package of Mathlib's")
    return bad


def library_lean_files():
    """The Lean files of the library: every `.lean` file under lean/ outside the build directory."""
    return sorted(f for f in text_files(os.path.join(ROOT, "lean")) if f.endswith(".lean"))


def library_independence():
    """The library's own independence: its imports, the lake configuration, and the text of every file
    of the project outside the kernel folders."""
    bad = import_violations(library_lean_files(), LEAN_ROOTS)
    bad += lake_violations()
    bad += scan_files(text_files(ROOT, skip=[KROOT]))
    return bad


def kernel_independence(k):
    """One kernel folder: its Lean imports (TVA, Mathlib and its own module) and the text of every file."""
    bad = import_violations(kernel_lean_files(k), LEAN_ROOTS + (manifest(k)["lean"]["module"].split(".")[0],))
    bad += scan_files(text_files(kpath(k)))
    return bad


def kernels():
    if not os.path.isdir(KROOT):
        return []
    return sorted(d for d in os.listdir(KROOT) if os.path.exists(kpath(d, "kernel.json")))


# ---------------------------------------------------------------- the transcription rules

TRANSCRIPTION = os.path.join(ROOT, "docs", "TRANSCRIPTION.md")
EXCLUSIONS = os.path.join(ROOT, "docs", "exclusions.md")
_RULE_DEF = re.compile(r"^- \*\*\[([a-z][a-z0-9_:-]*)\]\*\*(?=\s|$)")
_RULE_REF = re.compile(r"(?<![\w`\[])\[([a-z][a-z0-9_:-]*)\](?![(\[])")


def _norm(text):
    return " ".join(text.split())


def transcription_rules():
    """The transcription rules of docs/TRANSCRIPTION.md (protocol §7).

    A rule is a list item that starts with its identity in bold brackets, `- **[id]** …`, and runs to
    the next rule or the next heading. Its text may name other rules as `[id]`. The page's text outside
    every rule, and docs/exclusions.md, apply to every model (the *general* text).

    Returns `(general, rules, refs, problems)`: the general text, `{id: text}`, `{id: ids it names}`
    (the key `""` for the general text), and the page's inconsistencies (an identity defined twice, a
    name of no rule, a drop reason of the protocol that the page does not define)."""
    general, rules, problems, cur = [], {}, [], None
    try:
        text = open(TRANSCRIPTION, encoding="utf-8").read()
    except OSError:
        return "", {}, {}, ["docs/TRANSCRIPTION.md: missing"]
    for line in text.splitlines():
        m = _RULE_DEF.match(line)
        if m:
            cur = m.group(1)
            if cur in rules:
                problems.append(f"docs/TRANSCRIPTION.md: rule [{cur}] is defined twice")
            rules[cur] = [line]
        elif line.startswith("#"):
            cur = None
            general.append(line)
        elif cur is not None:
            rules[cur].append(line)
        else:
            general.append(line)
    excl = open(EXCLUSIONS, encoding="utf-8").read() if os.path.exists(EXCLUSIONS) else ""
    gtext = "\n".join(general) + "\n\n" + excl
    rules = {i: "\n".join(ls) for i, ls in rules.items()}
    refs = {i: {r for r in _RULE_REF.findall(t) if r != i} for i, t in rules.items()}
    refs[""] = set(_RULE_REF.findall(gtext))
    for i, rs in sorted(refs.items()):
        for r in sorted(rs - set(rules)):
            problems.append(f"docs/TRANSCRIPTION.md: {'rule [' + i + ']' if i else 'the general text'} names [{r}], "
                            "which no rule defines")
    problems += [f"docs/TRANSCRIPTION.md: the drop reason {d} has no rule [{d}]" for d in sorted(DROPS - set(rules))]
    return gtext, rules, refs, problems


def rule_closure(ids, refs):
    """The rules `ids` name, with every rule their texts name in turn, and those the general text names."""
    out, todo = set(), list(ids) + sorted(refs.get("", ()))
    while todo:
        i = todo.pop()
        if i not in out:
            out.add(i)
            todo += sorted(refs.get(i, ()))
    return out


def rule_parts(ids):
    """What a `map` or `model` digest folds in: the general text and the text of each rule in the closure
    of `ids`, whitespace normalised, as `{"(general)": text, id: text}`."""
    gtext, rules, refs, _ = transcription_rules()
    parts = {"(general)": _norm(gtext)}
    for i in sorted(rule_closure(ids, refs)):
        if i in rules:
            parts[i] = _norm(rules[i])
    return parts


# ---------------------------------------------------------------- kernel Lean hygiene

# What a kernel's Lean files may not contain, matched on the whole text (comments and strings
# included, so that nothing hides behind a comment marker inside a string). The gate runs this before
# it builds the kernel, since meta code would run during the build. Proof escapes, kernel bypasses, and
# every common way to run code while Lean elaborates: commands, attributes that install handlers, and
# the meta monads such code is written in. The replay check (leanchecker) re-checks every declaration
# whatever this list misses.
HYGIENE = [
    ("a proof escape", r"\bsorry\b|\badmit\b|\bnative_decide\b|\+\s*native\b|\bofReduce(?:Bool|Nat)\b|^\s*axiom\b"),
    ("a kernel or compiler bypass", r"\bimplemented_by\b|\bextern\b|\bunsafe\b|set_option\s+debug\.|\bskipKernelTC\b|"
     r"\bdoCheck\b|\baddDecl\w*|\baddAndCompile\b|\bmodifyEnv\b|\bsetEnv\b|\bEnvironment\b"),
    ("meta code", r"#eval\b|#exit\b|\brun_(?:cmd|elab|meta|tac)\b|\bby_elab\b|\beval%|\belab(?:_rules)?\b|"
     r"\bmacro(?:_rules)?\b|\bsyntax\b|\bdeclare_syntax_cat\b|\b(?:builtin_)?initialize\b|\bregister_\w+|"
     r"\binclude_str\b|\b[d]?simproc\b|\bdelab\b|\bapp_unexpander\b|\b(?:command|term)_elab\b|\bcode_action\b"),
    ("a meta type", r"\b(?:IO|EIO|BaseIO|CoreM|MetaM|TermElabM|CommandElabM|TacticM|MacroM)\b|\bLean\.(?:Elab|Meta|Syntax)\b"),
    ("an attribute that installs a handler",
     r"(?:@\[|\battribute\s*\[)[^\]]*\b(?:tactic|init|norm_num|positivity|aesop|export|env_linter)\b"),
]


def hygiene_problems(k):
    bad = []
    for p in kernel_lean_files(k):
        text = open(p, encoding="utf-8", errors="replace").read()
        for what, pat in HYGIENE:
            for m in re.finditer(pat, text, re.M):
                bad.append(f"{rel(p, kpath(k))}:{text.count(chr(10), 0, m.start()) + 1}: {what} ({m.group(0).strip()[:24]})")
    return bad


def replay(k):
    """Re-check every declaration of the kernel's modules with Lean's kernel, from the compiled files:
    `leanchecker <root>` replays each module's constants into the environment of its imports. A
    declaration that meta code added without checking fails here."""
    lean_dir = os.path.join(ROOT, "lean")
    root_mod = manifest(k)["lean"]["module"].split(".")[0]
    pre = subprocess.run([LAKE, "env", "lean", "--print-prefix"], cwd=lean_dir, capture_output=True, text=True)
    prefix = pre.stdout.strip()
    exe = os.path.join(prefix, "bin", "leanchecker")
    if pre.returncode != 0 or not os.path.exists(exe):
        return False, "leanchecker not found in the Lean toolchain"
    env = dict(os.environ, LEAN_SYSROOT=prefix, PATH=os.path.join(prefix, "bin") + os.pathsep + os.environ.get("PATH", ""))
    r = subprocess.run([LAKE, "env", exe, root_mod], cwd=lean_dir, capture_output=True, text=True, env=env)
    if r.returncode == 0:
        return True, f"leanchecker {root_mod}: every declaration of the kernel's modules re-checked by Lean's kernel"
    msg = " | ".join(l.strip() for l in (r.stderr + r.stdout).splitlines() if l.strip())[:300]
    return False, f"leanchecker {root_mod} rejects the compiled kernel: {msg}"


# ---------------------------------------------------------------- ledger

def ledger(k):
    path = kpath(k, "review", "ledger.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def append(k, events):
    path = kpath(k, "review", "ledger.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    seq = len(ledger(k))
    with open(path, "a") as f:
        for e in events:
            seq += 1
            f.write(json.dumps({"seq": seq, "time": now(), **e}, sort_keys=True, ensure_ascii=False) + "\n")


def state(k):
    """Fold the ledger. Events written before rounds were recorded belong to round 1."""
    S = {"reviewers": {}, "findings": {}, "accepts": {}, "filed": {}, "responses": 0,
         "archive": {}, "events": ledger(k)}
    for e in S["events"]:
        t, rnd = e["type"], e.get("round", 1)
        if t == "assign":
            S["reviewers"][e["actor"]] = e.get("focus", "all")
        elif t == "file":
            S["filed"].setdefault(e["actor"], {}).setdefault(rnd, 0)
            S["filed"][e["actor"]][rnd] += 1
        elif t == "response":
            S["responses"] += 1
        elif t == "raise":
            S["findings"][e["id"]] = {**e, "round": rnd, "state": "OPEN", "raisers": {e["actor"]},
                                      "released": set(), "reraises": 0, "history": [e]}
        elif t in ("respond", "release", "reraise", "escalate", "decide"):
            f = S["findings"][e["id"]]
            f["history"].append(e)
            if t == "respond":
                f["state"] = "REFERRED" if e["response"] == "REFERRED" else "ANSWERED"
                f["last_response"] = e["response"]
            elif t == "release":
                f["released"].add(e["actor"])
                if f["released"] >= f["raisers"]:
                    f["state"] = "CLOSED"
            elif t == "reraise":
                f["reraises"] += 1
                f["released"].discard(e["actor"])
                f["state"] = "OPEN"
                f["raise_items"] = e.get("items", f["items"])
            elif t == "escalate":
                f["state"] = "ESCALATED"
            elif t == "decide":
                f["state"] = "CLOSED"
                f["decision"] = e["decision"]
        elif t == "accept":
            S["accepts"].setdefault(e["actor"], {}).update(e["items"])
        for p, h in e.get("files", {}).items():
            S["archive"][p] = h
    S["round"] = S["responses"] + 1
    return S


def snapshot(k, gate_digest):
    return load_json(kpath(k, "review", "snapshots", gate_digest[:16] + ".json"), {"items": {}})


def affected(k, f, items):
    """The items a finding is about: an acceptance of one of them waits for the finding."""
    t = f["target"]
    if t.startswith("item:"):
        return {t[5:]}
    if t.startswith(("map:", "src:")):
        return {"model", "map"}
    if f.get("affects"):
        return {a[5:] if a.startswith("item:") else a for a in f["affects"]}
    if t.startswith("decl:"):
        hit = {it for it, v in snapshot(k, f["gate"])["items"].items() if t[5:] in v.get("cone", {})}
        return hit or set(items)
    return set(items)


# ---------------------------------------------------------------- the gate

def check(k):
    man = manifest(k)
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    src = kpath(k, man["source"]["file"])
    src_text = open(src).read() if os.path.exists(src) else ""
    src_sha = sha(src_text)
    add("source", src_sha == man["source"]["sha256"],
        f"{man['source']['file']} sha256 {src_sha[:16]}" +
        ("" if src_sha == man["source"]["sha256"] else f" (the manifest says {man['source']['sha256'][:16]})"))
    lean_dir = os.path.join(ROOT, "lean")
    mod, ns = man["lean"]["module"], man["lean"]["namespace"]
    # hygiene and independence read the files only; they run first, and the build (which runs the
    # kernel's Lean) only when both pass
    bad = hygiene_problems(k)
    add("hygiene", not bad, "no proof escape, kernel bypass, meta code, meta type or handler attribute" if not bad
        else "; ".join(bad[:5]) + (f" (+{len(bad) - 5} more)" if len(bad) > 5 else ""))
    ind = kernel_independence(k) + library_independence()
    add("independence", not ind, "the kernel and the library import only TVA, the kernel's own modules and Mathlib, "
        "and refer to no other project" if not ind else "; ".join(ind[:5]) + (f" (+{len(ind) - 5} more)" if len(ind) > 5 else ""))
    built = False
    if bad or ind:
        add("build", False, "not run (hygiene or independence failed)")
    else:
        r = subprocess.run([LAKE, "build", mod], cwd=lean_dir, capture_output=True, text=True)
        errs = [l for l in r.stdout.splitlines() if l.startswith("error")]
        built = r.returncode == 0
        add("build", built, f"lake build {mod}" + ("" if built else ": " + " | ".join(errs[:3])))
    if built:
        add("replay", *replay(k))
    else:
        add("replay", False, "not run (the build did not run or failed)")
    tmpl = open(os.path.join(ROOT, "tools", "claim_reader.lean.in")).read()
    gen = tmpl.replace("@MODULE@", mod).replace("@NS@", ns).replace("@PREFIX@", mod.split(".")[0])
    gpath = os.path.join(lean_dir, ".lake", "tvr", k, "claim_reader.lean")
    write(gpath, gen)
    surf = None
    if built:
        r2 = subprocess.run([LAKE, "env", "lean", gpath], cwd=lean_dir, capture_output=True, text=True)
        js = [l for l in r2.stdout.splitlines() if l.startswith("TVR_JSON ")]
        if r2.returncode == 0 and js:
            surf = json.loads(js[0][len("TVR_JSON "):])
        add("shape", surf is not None,
            "main : ∀ K P, Assumptions K P → Spec K; consistent : Claim.Consistent Assumptions; Spec K = Meets Inst.ok Pre Post K"
            if surf else "the claim does not have the protocol's shape: " + " | ".join(r2.stdout.splitlines()[:4]))
    else:
        add("shape", False, "not checked (the build did not run or failed)")
    items, consts = {}, {}
    if surf:
        consts = {c["name"]: c for c in surf["consts"] if c}
        add("axioms", all(set(v) <= STD_AXIOMS for v in surf["axioms"].values()),
            "; ".join(f"{a}: {', '.join(v)}" for a, v in surf["axioms"].items()))
        unused = [f for f, u in surf["usage"].items() if not u]
        add("usage", not unused and not surf["destructured"],
            "every assumption is used by main" if not unused and not surf["destructured"] else
            ("unused: " + ", ".join(unused) if unused else "") +
            (" (Assumptions destructured: use projections)" if surf["destructured"] else ""))
        tva_insts = [i for i in surf["instances"] if i["class"].startswith("TVA.")]
        add("namespace", not surf["shadow"] and not tva_insts,
            "the kernel declares nothing in TVA and no instance of a library class" if not surf["shadow"] and not tva_insts
            else "declares " + ", ".join(surf["shadow"][:5] + [i["name"] for i in tva_insts][:5]))
        bad_kinds = []
        for f in surf["fields"]:
            m = re.match(r"\s*\[(rule|kernel)\]", surf["field_docs"].get(f, ""))
            u = surf["field_uses"].get(f, {"consts": [], "K": False, "P": False})
            own = sorted(c for c in u["consts"] if c.startswith(ns + ".") and c[len(ns) + 1:] not in RULE_ARGS)
            rules = [c for c in u["consts"] if c.startswith("TVA.Rules.")]
            if not m:
                bad_kinds.append(f"{f}: not tagged [rule] or [kernel]")
            elif m.group(1) == "rule" and (own or not rules):
                bad_kinds.append(f"{f}: a [rule] must cite a TVA.Rules fact" + (" with no constant of the kernel's own (" +
                                 ", ".join(c[len(ns) + 1:] for c in own) + ")" if own else ""))
            elif m.group(1) == "kernel" and (u["K"] or u["P"] or rules):
                bad_kinds.append(f"{f}: a [kernel] assumption may not mention K, P or a TVA.Rules fact")
        add("kinds", not bad_kinds, "every assumption is a [rule] citing the library or a [kernel] fact about the kernel's own values"
            if not bad_kinds else "; ".join(bad_kinds))
    # The model map, and the transcription rules it names (protocol §4, §7): their text goes into the
    # `map` and `model` digests, so a change of a rule the kernel uses re-opens both items.
    mm_path = kpath(k, "model-map.json")
    mm = load_json(mm_path, {"entries": []})
    _, trules, _, tproblems = transcription_rules()
    claim_text = re.sub(r"\s+", " ", " ".join(open(p).read() for p in kernel_lean_files(k)))
    covered, problems, used = set(), [], set()
    problems += [f"the library's transcription page is inconsistent: {x}" for x in tproblems]
    for i, e in enumerate(mm["entries"]):
        covered |= set(parse_lines(e["src"]))
        names = e.get("rules", [])
        if not (isinstance(names, list) and all(isinstance(n, str) for n in names)):
            problems.append(f"entry {i} (src {e['src']}): `rules` is a list of rule identities")
            names = []
        unknown = [n for n in names if n not in trules]
        if unknown:
            problems.append(f"entry {i} (src {e['src']}): no transcription rule {', '.join('[' + n + ']' for n in unknown)}")
        used |= set(names)
        if "model" in e:
            if not names:
                problems.append(f"entry {i} (src {e['src']}): a model entry names the transcription rules it applies (`rules`)")
            for snip in ([e["model"]] if isinstance(e["model"], str) else e["model"]):
                if re.sub(r"\s+", " ", snip).strip() not in claim_text:
                    problems.append(f"entry {i} (src {e['src']}): model text not found: {snip[:50]!r}")
        elif e.get("drop") not in DROPS:
            problems.append(f"entry {i}: drop reason {e.get('drop')!r} is not allowed")
        else:
            used.add(e["drop"])
            if e["drop"] in NOTE_REQUIRED and not e.get("note"):
                problems.append(f"entry {i}: a {e['drop']} drop needs a note (the condition that makes it hold)")
    tparts = rule_parts(used)
    trans = [f"[{i}] {t}" for i, t in sorted(tparts.items())]
    trule_hashes = {f"rule:{i}": sha(t)[:16] for i, t in tparts.items()}
    if surf:
        for it, v in surf["items"].items():
            lines = sorted(f"{n} {consts[n]['hash']}" for n in v["cone"] if n in consts)
            docs = [consts[n].get("doc") or "" for n in v["roots"] if n in consts]
            if it.startswith("assumption:"):
                docs.append(surf["field_docs"].get(it.split(":", 1)[1], ""))
            extra = [src_sha] + trans if it == "model" else []
            items[it] = {"digest": sha("\n".join(lines + docs + extra + [surf["lean"]])), "roots": v["roots"],
                         "cone": sorted(v["cone"]), "rules": trule_hashes if it == "model" else {}}
    src_lines = src_text.splitlines()
    missing, in_doc = [], False
    for a, b in man["source"]["ranges"]:
        for i in range(a, b + 1):
            t = src_lines[i - 1].strip() if i <= len(src_lines) else ""
            q = t.count('"""')
            if in_doc or q:
                if q % 2 == 1:
                    in_doc = not in_doc
                continue
            if t and not t.startswith("#") and i not in covered:
                missing.append(i)
    add("map", not missing and not problems,
        f"{len(mm['entries'])} entries cover every code line of {man['source']['ranges']}" if not missing and not problems
        else "; ".join(([f"uncovered source lines {missing[:12]}"] if missing else []) + problems[:4]))
    items["map"] = {"digest": sha("\n".join([open(mm_path).read(), src_sha] + trans)) if os.path.exists(mm_path) else "",
                    "roots": [], "cone": [], "rules": trule_hashes}
    doc = kpath(k, man.get("doc", "README.md"))
    heads = re.findall(r"^## (\w+)", open(doc).read(), re.M) if os.path.exists(doc) else []
    lack = [s for s in DOC_SECTIONS if s not in heads]
    add("doc", os.path.exists(doc) and not lack, f"{man.get('doc', 'README.md')} has the sections {', '.join(DOC_SECTIONS)}"
        if not lack else f"{man.get('doc', 'README.md')} lacks: {', '.join(lack)}")
    ok = all(c["ok"] for c in checks)
    gate_digest = sha("\n".join(f"{i} {v['digest']}" for i, v in sorted(items.items())))
    kinds = {}
    for f in (surf["fields"] if surf else []):
        m = re.match(r"\s*\[(rule|kernel)\]", surf["field_docs"].get(f, ""))
        kinds[f] = m.group(1) if m else ""
    gate = {"protocol": "tvr-2", "kernel": k, "time": now(), "ok": ok, "digest": gate_digest,
            "checks": checks, "items": {i: v["digest"] for i, v in items.items()}, "kinds": kinds}
    dump_json(kpath(k, "gate.json"), gate)
    if ok and surf:
        dump_json(kpath(k, "review", "snapshots", gate_digest[:16] + ".json"),
                  {"items": {i: {"digest": v["digest"], "cone": {**{n: consts[n]["hash"] for n in v["cone"] if n in consts},
                                                                 **v.get("rules", {})}}
                             for i, v in items.items()}})
    if surf:
        render_claim(k, man, gate, surf, consts, items, mm, src_lines)
    refresh(k)
    for c in checks:
        print(f"{'PASS' if c['ok'] else 'FAIL'}  {c['name']:<10} {c['detail']}")
    print(f"gate {'PASS' if ok else 'FAIL'}  digest {gate_digest[:16]}")
    return ok


# ---------------------------------------------------------------- claim.md

def visible(name):
    return not any(part.startswith("_") or part.startswith("match_") or part in (
        "mk", "rec", "casesOn", "recOn", "noConfusion", "noConfusionType", "brecOn", "below", "eq_def",
        "eq_1", "inj", "injEq", "sizeOf_spec", "ext", "ext_iff") for part in name.split("."))


def render_claim(k, man, gate, surf, consts, items, mm, src_lines):
    ns = man["lean"]["namespace"]
    out = []
    w = out.append
    w(f"# Claim: `{k}`\n")
    w(f"> Generated by `tvr check` at {gate['time']}; do not edit. Gate: **{'PASS' if gate['ok'] else 'FAIL'}**, "
      f"digest `{gate['digest'][:16]}`. How to review: [docs/REVIEW_PROTOCOL.md]({link('docs/REVIEW_PROTOCOL.md', kpath(k))}) and "
      f"[docs/REVIEWER_GUIDE.md]({link('docs/REVIEWER_GUIDE.md', kpath(k))}). The review so far: [REVIEW.md](REVIEW.md).\n")
    w(f"{man.get('title', k)}. The prover's write-up: [README.md](README.md).\n")
    w("## 1. The claim\n")
    w(f"```lean\ntheorem {ns}.main : {surf['main_type']}\ntheorem {ns}.consistent : TVA.Claim.Consistent {ns}.Assumptions\n```\n")
    w("`Spec K` is checked to be `Claim.Meets Inst.ok Pre Post K`: on every instance `Inst.ok` covers and every input "
      "`Pre` allows, every outcome of the device `K` is a normal end whose memory satisfies `Post`. Lean checked "
      "`main` and `consistent`; you judge the items below.\n")
    w("## 2. The source\n")
    s = man["source"]
    w(f"[`{s['file']}`]({s['file']}) (sha256 `{s['sha256'][:16]}…`), kernel lines {s['ranges']}" +
      (f", host lines {s['host']}" if s.get("host") else "") + ".\n")

    def section(title, item, intro):
        w(f"## {title}\n")
        w(f"Item `{item}`, digest `{items[item]['digest'][:16]}`. {intro}\n")
        decls = []
        for n in items[item]["cone"]:
            c = consts.get(n)
            if not c or not n.startswith(ns + ".") or not visible(n) or c["kind"] == "ctor":
                continue
            if not c.get("range") or not c.get("module"):
                continue
            p = lean_file(k, man, c["module"])
            if os.path.exists(p):
                decls.append((rel(p, kpath(k)), c["range"][0], c["range"][1], p))
        last = {}
        for f, a, b, p in sorted(set(decls), key=lambda x: (x[0], x[1], -x[2])):
            if f in last and a <= last[f]:
                continue  # inside a declaration already shown (a structure's field)
            last[f] = b
            text = "\n".join(open(p).read().splitlines()[a - 1:b])
            w(f"[`{f}:{a}`]({f}#L{a})\n```lean\n{text}\n```\n")
        lib = sorted(n for n in items[item]["cone"] if n.startswith("TVA.") and visible(n) and n in consts
                     and consts[n]["kind"] in ("def", "inductive", "opaque", "axiom"))
        if lib:
            w(f"<details><summary>Library definitions this item reaches ({len(lib)})</summary>\n")
            for n in lib:
                d = (consts[n].get("doc") or "").strip().split("\n")[0]
                w(f"- `{n}` ({consts[n]['module']}){': ' + d if d else ''}")
            w("\n</details>\n")

    section("3. Conditions", "conditions", "The instances (`Inst.ok`) and inputs (`Pre`) covered.")
    section("4. Specification", "spec", "What the outputs are (`Post`).")
    w("## 5. Assumptions\n")
    for f in surf["fields"]:
        it = f"assumption:{f}"
        w(f"### `{f}` — item `{it}`, digest `{items[it]['digest'][:16]}`, used by `main`: "
          f"{'yes' if surf['usage'].get(f) else '**no**'}\n")
        w(surf["field_docs"].get(f, "").strip() + "\n")
        w(f"```lean\n{surf['field_types'].get(f, '')}\n```\n")
        cited = sorted(c for c in surf["field_uses"].get(f, {}).get("consts", []) if c.startswith("TVA.Rules."))
        if cited:
            w("Library facts cited: " + ", ".join(f"`{c}`" for c in cited) + ". A kernel reviewer checks that they are "
              "the library's facts for the operation the model uses, not whether they are true.\n")
    section("6. Model", "model", "The kernel in the `tl.*` vocabulary and the launch.")
    w("### The model map (item `map`)\n")
    w(f"Digest `{items['map']['digest'][:16]}`. Every code line of the source is mapped to model text, with the "
      "transcription rules it applies, or dropped with a reason (a rule too). The rules are below the table.\n")
    w("| Source | Model / reason |\n|---|---|")
    for e in mm["entries"]:
        ls = parse_lines(e["src"])
        st = " ⏎ ".join(src_lines[i - 1].strip() for i in ls if i <= len(src_lines)).replace("|", "\\|")
        named = " (" + ", ".join(f"[{n}]" for n in e.get("rules", [])) + ")" if e.get("rules") else ""
        if "model" in e:
            m = e["model"] if isinstance(e["model"], str) else " ⏎ ".join(e["model"])
            right = f"`{m}`".replace("|", "\\|") + named + (f" — {e['note']}" if e.get("note") else "")
        else:
            right = f"**[{e['drop']}]**" + named + (f": {e['note']}" if e.get("note") else "")
        w(f"| [{e['src']}]({s['file']}#L{ls[0]}): `{st}` | {right} |")
    w("")
    _, trules, _, _ = transcription_rules()
    shown = sorted(r[5:] for r in items["map"].get("rules", {}) if r != "rule:(general)")
    w(f"<details><summary>The transcription rules this map applies ({len(shown)}, with the rules they name), from "
      f"[docs/TRANSCRIPTION.md]({link('docs/TRANSCRIPTION.md', kpath(k))})</summary>\n")
    for r in shown:
        # the rules' links are relative to docs/; make them relative to the kernel's folder
        text = re.sub(r"\]\((?![a-z]+:|#)([^)]+)\)", lambda m: "](" + link(posixpath.join("docs", m.group(1)), kpath(k)) + ")",
                      trules.get(r, f"- **[{r}]** (missing)"))
        w(text + "\n")
    w("</details>\n")
    w("## 7. Gate\n")
    w("| Check | | Detail |\n|---|---|---|")
    for c in gate["checks"]:
        w(f"| {c['name']} | {'PASS' if c['ok'] else 'FAIL'} | {c['detail'].replace('|', '/')} |")
    w("\n| Item | Digest |\n|---|---|")
    for i, v in sorted(items.items()):
        w(f"| `{i}` | `{v['digest'][:16]}` |")
    write(kpath(k, "claim.md"), "\n".join(out) + "\n")


# ---------------------------------------------------------------- review commands

def current_gate(k):
    g = load_json(kpath(k, "gate.json"))
    if not g["ok"]:
        die("the gate fails; run `tvr check` after fixing it")
    newest = max(os.path.getmtime(p) for p in kernel_lean_files(k) + [kpath(k, "model-map.json"), kpath(k, "kernel.json")])
    gt = datetime.datetime.strptime(g["time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()
    if newest > gt + 1:
        die("the gate is stale (a kernel file changed after it ran); run `tvr check`")
    return g


def valid_target(k, t, gate, man):
    if t.startswith("item:"):
        return t[5:] in gate["items"]
    if t.startswith("decl:"):
        return any(t[5:] in v["cone"] for v in snapshot(k, gate["digest"])["items"].values())
    if t.startswith(("src:", "map:")):
        try:
            n = int(t.split(":", 1)[1])
        except ValueError:
            return False
        return any(a <= n <= b for a, b in man["source"]["ranges"] + man["source"].get("host", []))
    if t.startswith(("lib:", "doc:")):
        return os.path.exists(os.path.join(ROOT, t.split(":", 1)[1].split("#")[0]))
    return False


def round_file(k, rnd, actor, base, count):
    """review/round-N/ACTOR/BASE.json for an actor's first filing of the round, BASE-2.json for the second, …"""
    stem, ext = os.path.splitext(base)
    return kpath(k, "review", f"round-{rnd}", actor, base if count == 0 else f"{stem}-{count + 1}{ext}")


def archive_entry(*paths):
    return {rel(p): file_sha(p) for p in paths if os.path.exists(p)}


def cmd_assign(k, reviewer, focus):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", reviewer) or reviewer in ("prover", "user", "tvr", "coordinator"):
        die(f"bad reviewer id {reviewer!r} (lower case, digits, '-')")
    S = state(k)
    append(k, [{"type": "assign", "actor": reviewer, "focus": focus, "round": S["round"]}])
    refresh(k)
    print(f"assigned {reviewer} ({focus}) to {k}")


def cmd_file(k, who):
    S = state(k)
    N = S["round"]
    if who not in S["reviewers"]:
        die(f"{who} is not assigned to {k}")
    count = S["filed"].get(who, {}).get(N, 0)
    path = round_file(k, N, who, "filing.json", count)
    report = round_file(k, N, who, "report.md", count)
    filing = load_json(path)
    if filing.get("reviewer", who) != who:
        die(f"{rel(path)} names reviewer {filing['reviewer']}")
    bad = scan_files([path, report] + text_files(os.path.join(os.path.dirname(path), "work")))
    if bad:
        die("a filing may cite only TVA and Mathlib (the protocol, §2): " + "; ".join(bad[:5]))
    gate = current_gate(k)
    P = lint_filing(k, who, N, filing, gate["items"], rel(path))
    P += lint_review_report(k, who, N, filing if isinstance(filing, dict) else {}, gate["items"], report)
    report_problems(P, "the filing")
    man = manifest(k)
    findings = S["findings"]
    events = [{"type": "file", "actor": who, "round": N, "gate": gate["digest"], "path": rel(path),
               "report": rel(report) if os.path.exists(report) else "", "notes": filing.get("notes", ""),
               "files": archive_entry(path, report)}]
    nxt = 1 + max([int(i[1:]) for i in findings] or [0])
    for f in filing.get("findings", []):
        if f.get("kind") not in KINDS:
            die(f"unknown kind {f.get('kind')!r}; the kinds are {', '.join(KINDS)}")
        if not any(f["target"].startswith(p) for p in KINDS[f["kind"]][1]) or not valid_target(k, f["target"], gate, man):
            die(f"invalid target {f['target']!r} for {f['kind']} (allowed: {', '.join(KINDS[f['kind']][1])})")
        need = KINDS[f["kind"]][2]
        if need:
            have = gate.get("kinds", {}).get(f["target"].split(":", 2)[2], "")
            if have != need:
                die(f"{f['kind']} applies to [{need}] assumptions; {f['target']} is [{have}]" +
                    (" (a library fact: file LIBRARY_REPORT, or RULE_MISCITED if it is the wrong fact)" if have == "rule" else ""))
        for a in f.get("affects", []):
            if (a[5:] if a.startswith("item:") else a) not in gate["items"]:
                die(f"`affects` names no item: {a}")
        events.append({"type": "raise", "actor": who, "round": N, "id": f"F{nxt}", "kind": f["kind"],
                       "target": f["target"], "affects": f.get("affects", []),
                       "blocking": f.get("blocking", KINDS[f["kind"]][0]), "summary": f["summary"],
                       "detail": f.get("detail", ""), "evidence": f.get("evidence", ""),
                       "gate": gate["digest"], "items": gate["items"]})
        nxt += 1
    released = set()
    for r in filing.get("release", []):
        if r["id"] not in findings or who not in findings[r["id"]]["raisers"]:
            die(f"{who} cannot release {r['id']} (not raised by {who})")
        events.append({"type": "release", "actor": who, "round": N, "id": r["id"], "note": r.get("note", "")})
        released.add(r["id"])
    reraised = set()
    for r in filing.get("reraise", []):
        f = findings.get(r["id"])
        if not f or who not in f["raisers"]:
            die(f"{who} cannot re-raise {r['id']} (not raised by {who})")
        if f["state"] not in ("ANSWERED", "REFERRED"):
            die(f"{r['id']} is {f['state']}; only an answered or referred finding can be re-raised")
        events.append({"type": "reraise", "actor": who, "round": N, "id": r["id"], "note": r.get("note", ""),
                       "items": gate["items"]})
        reraised.add(r["id"])
        if f["reraises"] + 1 >= ESCALATE_AFTER:
            events.append({"type": "escalate", "actor": "tvr", "round": N, "id": r["id"],
                           "memo": f"re-raised {f['reraises'] + 1} times; the user decides"})
    acc = filing.get("accept", [])
    if acc:
        names = list(gate["items"]) if acc == "all" else [a[5:] if a.startswith("item:") else a for a in acc]
        own = [f for i, f in findings.items() if who in f["raisers"] and f["blocking"]
               and ((f["state"] not in ("CLOSED", "REFERRED") and i not in released) or i in reraised)]
        own += [e for e in events if e["type"] == "raise" and e["blocking"]]
        for n in names:
            if n not in gate["items"]:
                die(f"no item {n}")
            block = sorted({f["id"] for f in own if n in affected(k, f, gate["items"])})
            if block:
                die(f"{who} cannot accept {n} while their blocking finding(s) on it are open: {', '.join(block)}")
        events.append({"type": "accept", "actor": who, "round": N, "items": {n: gate["items"][n] for n in names},
                       "gate": gate["digest"]})
    append(k, events)
    refresh(k)
    print(f"recorded {rel(path)}: {len(events) - 1} event(s)")
    print(status_text(k, who))


def cmd_respond(k):
    S = state(k)
    N = S["round"]
    waiting = [r for r in S["reviewers"] if not S["filed"].get(r, {}).get(N)]
    if waiting:
        die(f"round {N} is waiting for {', '.join(waiting)}")
    if not any(f["state"] == "OPEN" for f in S["findings"].values()):
        die(f"round {N} has no open finding to answer (a first submission needs only a passing gate)")
    path = kpath(k, "review", f"round-{N}", "prover", "response.json")
    notes = kpath(k, "review", f"round-{N}", "prover", "notes.md")
    resp = load_json(path)
    bad = scan_files([path, notes])
    if bad:
        die("a response may cite only TVA and Mathlib (the protocol, §2): " + "; ".join(bad[:5]))
    gate = current_gate(k)
    findings = S["findings"]
    open_ids = sorted((i for i, f in findings.items() if f["state"] == "OPEN"), key=lambda x: int(x[1:]))
    P = lint_response(k, N, resp, open_ids, gate["items"], rel(path))
    P += lint_prover_notes(k, N, resp if isinstance(resp, dict) else {}, gate, notes)
    report_problems(P, "the response")
    events = [{"type": "response", "actor": "prover", "round": N, "gate": gate["digest"], "path": rel(path),
               "notes": rel(notes) if os.path.exists(notes) else "", "files": archive_entry(path, notes)}]
    answered = set()
    for r in resp["responses"]:
        f = findings.get(r["id"])
        if not f:
            die(f"no finding {r['id']}")
        if f["state"] != "OPEN":
            die(f"{r['id']} is {f['state']}; only open findings are answered")
        if r["response"] not in RESPONSES:
            die(f"unknown response {r['response']!r}; the responses are {', '.join(sorted(RESPONSES))}")
        if r["response"] == "REVISED":
            base = f.get("raise_items", f["items"])
            ch = [c[5:] if c.startswith("item:") else c for c in r.get("changed", [])]
            if not ch:
                die(f"{r['id']}: REVISED must name the changed items")
            unknown = [c for c in ch if c not in gate["items"]]
            same = [c for c in ch if c in gate["items"] and gate["items"][c] == base.get(c)]
            if unknown or same:
                die(f"{r['id']}: REVISED, but " + "; ".join(
                    ([f"no such item: {', '.join(unknown)}"] if unknown else []) +
                    ([f"these items did not change since the finding: {', '.join(same)}"] if same else [])))
        events.append({"type": "respond", "actor": "prover", "round": N, "id": r["id"], "response": r["response"],
                       "text": r["text"], "changed": r.get("changed", []), "issue": r.get("issue", ""),
                       "gate": gate["digest"]})
        answered.add(r["id"])
    left = sorted((i for i, f in findings.items() if f["state"] == "OPEN" and i not in answered), key=lambda x: int(x[1:]))
    if left:
        die(f"one batch answers every open finding (the batch policy); not answered: {', '.join(left)}")
    append(k, events)
    refresh(k)
    print(f"recorded {rel(path)}: {len(events) - 1} response(s); round {N + 1} is open for the reviewers")


def cmd_decide(k, fid, text):
    S = state(k)
    if S["findings"].get(fid, {}).get("state") != "ESCALATED":
        die(f"{fid} is not escalated")
    append(k, [{"type": "decide", "actor": "user", "round": S["round"], "id": fid, "decision": text}])
    refresh(k)


def cmd_note(k, text, files=()):
    S = state(k)
    append(k, [{"type": "note", "actor": "coordinator", "round": S["round"], "text": text,
                "files": archive_entry(*[os.path.join(ROOT, f) for f in files])}])
    refresh(k)


# ---------------------------------------------------------------- views

def cert_state(k, S, gate):
    blocking = sorted((i for i, f in S["findings"].items() if f["blocking"] and f["state"] not in ("CLOSED", "REFERRED")),
                      key=lambda x: int(x[1:]))
    full = bool(S["reviewers"]) and gate["ok"] and all(
        S["accepts"].get(r, {}).get(i) == d for r in S["reviewers"] for i, d in gate["items"].items())
    return full and not blocking, blocking, full


def blind_round1(S):
    return any(not S["filed"].get(r, {}).get(1) for r in S["reviewers"])


def status_text(k, who=None):
    gate = load_json(kpath(k, "gate.json"), {"items": {}, "ok": False, "digest": ""})
    S = state(k)
    L = []
    if not S["reviewers"]:
        return "No reviewer assigned yet.\n"
    revs = list(S["reviewers"])
    L.append("| Item | Digest | " + " | ".join(revs) + " |")
    L.append("|---|---|" + "---|" * len(revs))
    cur = snapshot(k, gate["digest"])["items"]
    for it, dg in sorted(gate["items"].items()):
        row = []
        for r in revs:
            a = S["accepts"].get(r, {}).get(it)
            if a is None:
                row.append("—")
            elif a == dg:
                row.append("accepted")
            else:
                old = {}
                for e in S["events"]:
                    if e["type"] == "accept" and e["actor"] == r and it in e["items"]:
                        old = snapshot(k, e["gate"])["items"].get(it, {}).get("cone", {})
                new = cur.get(it, {}).get("cone", {})
                diff = sorted(n for n in set(old) | set(new) if old.get(n) != new.get(n))
                row.append("STALE" + (" (changed: " + ", ".join(f"`{n}`" for n in diff[:5]) +
                                      (" …" if len(diff) > 5 else "") + ")" if diff else ""))
        L.append(f"| `{it}` | `{dg[:12]}` | " + " | ".join(row) + " |")
    L.append("")
    hide = blind_round1(S)
    shown = {i: f for i, f in S["findings"].items()
             if not (hide and f["round"] == 1 and (who is None or who not in f["raisers"]))}
    if hide:
        L.append("Round 1 is blind: findings are hidden until every assigned reviewer has filed.\n")
    if shown:
        L.append("| Id | State | Kind | Target | By | Round | Summary |\n|---|---|---|---|---|---|---|")
        for i in sorted(shown, key=lambda x: int(x[1:])):
            f = shown[i]
            st = f["state"] + ("" if f["blocking"] else " (non-blocking)")
            if f["state"] == "ANSWERED":
                st += f" — {f['last_response']}"
            L.append(f"| [{i}](#{i.lower()}) | {st} | {f['kind']} | `{f['target']}` | {', '.join(sorted(f['raisers']))} "
                     f"| {f['round']} | {f['summary'][:120].replace('|', '/')} |")
        L.append("")
    cert, blocking, full = cert_state(k, S, gate)
    L.append(f"**Certificate: {'GRANTED' if cert else 'not yet'}**" + ("" if cert else (
        f" — {len(blocking)} blocking finding(s) unresolved" + ("" if full else
        "; not every item is accepted by every reviewer at its current digest"))) + "\n")
    return "\n".join(L)


def phase(k, S):
    N = S["round"]
    waiting = [r for r in S["reviewers"] if not S["filed"].get(r, {}).get(N)]
    open_ = [i for i, f in S["findings"].items() if f["state"] == "OPEN"]
    if not S["reviewers"]:
        return "no reviewer assigned"
    if waiting:
        return f"round {N}: waiting for {', '.join(waiting)}"
    if open_:
        return f"round {N}: waiting for the prover ({len(open_)} open finding(s))"
    return f"round {N}: all reviewers have filed"


def render_review(k):
    man = manifest(k)
    gate = load_json(kpath(k, "gate.json"), {"items": {}, "ok": False, "digest": "", "time": ""})
    S = state(k)
    out = []
    w = out.append
    w(f"# Review: `{k}`\n")
    w(f"> Generated by `tvr` from [review/ledger.jsonl](review/ledger.jsonl) at {now()}; do not edit. "
      f"Protocol: [docs/REVIEW_PROTOCOL.md]({link('docs/REVIEW_PROTOCOL.md', kpath(k))}).\n")
    w(f"{man.get('title', k)}. The claim sheet: [claim.md](claim.md); the prover's write-up: [README.md](README.md).\n")
    cert, blocking, _ = cert_state(k, S, gate)
    w("## Status\n")
    w(f"- Gate: **{'PASS' if gate['ok'] else 'FAIL'}**, digest `{gate['digest'][:16]}` ({gate.get('time', '')}).")
    w(f"- Phase: {phase(k, S)}.")
    w(f"- Reviewers: {', '.join(f'{r} ({fo})' for r, fo in S['reviewers'].items()) or 'none'}.")
    w(f"- Certificate: **{'GRANTED' if cert else 'not yet'}**.")
    bad = [p for p, h in S["archive"].items() if not os.path.exists(os.path.join(ROOT, p)) or file_sha(os.path.join(ROOT, p)) != h]
    w(f"- Recorded files: {'all unchanged since they were recorded' if not bad else '**changed after recording: ' + ', '.join(bad) + '**'}.\n")
    w(status_text(k))
    rows = issues_for(k)
    if rows:
        w("### Library issues raised from this kernel\n")
        w("Kernel review trusts the library; these are tracked in "
          f"[docs/library-review/ISSUES.md]({link('docs/library-review/ISSUES.md', kpath(k))}).\n")
        w("| Issue | State | Source | Summary |\n|---|---|---|---|")
        for c in rows:
            w(f"| {c[0].replace('(#', '(' + link('docs/library-review/ISSUES.md', kpath(k)) + '#')} | {c[1]} | {c[2]} | {c[3]} |")
        w("")
    hide = blind_round1(S)
    rounds = sorted({e.get("round", 1) for e in S["events"]})
    for rnd in rounds:
        w(f"## Round {rnd}\n")
        if rnd == 1 and hide:
            done = [r for r in S["reviewers"] if S["filed"].get(r, {}).get(1)]
            w(f"Blind: {len(done)} of {len(S['reviewers'])} reviewers have filed; their filings are shown once all have.\n")
            continue
        evs = [e for e in S["events"] if e.get("round", 1) == rnd]
        for e in evs:
            t = e["type"]
            if t == "assign":
                w(f"- `{e['time']}` {e['actor']} assigned (focus: {e.get('focus', 'all')}).")
            elif t == "note":
                w(f"- `{e['time']}` coordinator's note: {e['text']}" +
                  (" Files: " + ", ".join(f"[{os.path.basename(p)}]({link(p, kpath(k))})" for p in e.get("files", {})) if e.get("files") else ""))
            elif t == "file":
                # filings recorded before the kernels/ layout carry no path: use the round folder's
                d = os.path.dirname(e.get("path") or f"kernels/{k}/review/round-{rnd}/{e['actor']}/filing.json")
                fp = e.get("path") or os.path.join(d, "filing.json")
                rep = e.get("report") or (os.path.join(d, "report.md") if not e.get("path") else "")
                links = []
                for label, p in (("filing", fp), ("report", rep)):
                    if p and os.path.exists(os.path.join(ROOT, p)):
                        links.append(f"[{label}]({rel(os.path.join(ROOT, p), kpath(k))})")
                work = os.path.join(ROOT, d, "work")
                if os.path.isdir(work):
                    links.append(f"[work]({rel(work, kpath(k))}/)")
                w(f"\n### {e['actor']} files ({', '.join(l for l in links if l)})\n")
                w(f"`{e['time']}`, at gate `{e['gate'][:16]}`." + (f" Notes: {e['notes']}" if e.get("notes") else "") + "\n")
            elif t == "raise":
                w(f"#### {e['id']}\n")
                w(f"**{e['kind']}** on `{e['target']}`" + (" (blocking)" if e["blocking"] else " (non-blocking)") +
                  (f", affects {', '.join(e['affects'])}" if e.get("affects") else "") + f": {e['summary']}\n")
                if e.get("detail"):
                    w(e["detail"] + "\n")
                if e.get("evidence"):
                    w(f"*Evidence:* {e['evidence']}\n")
            elif t == "release":
                w(f"- **released {e['id']}**: {e.get('note', '')}")
            elif t == "reraise":
                w(f"- **re-raised {e['id']}**: {e.get('note', '')}")
            elif t == "escalate":
                w(f"- **{e['id']} escalated to the user**: {e.get('memo', '')}")
            elif t == "accept":
                w(f"- **accepted** at gate `{e['gate'][:16]}`: " + ", ".join(f"`{i}`" for i in sorted(e["items"])))
            elif t == "response":
                links = [f"[response]({rel(os.path.join(ROOT, e['path']), kpath(k))})"]
                if e.get("notes"):
                    links.append(f"[notes]({rel(os.path.join(ROOT, e['notes']), kpath(k))})")
                w(f"\n### The prover responds ({', '.join(links)})\n")
                w(f"`{e['time']}`, at gate `{e['gate'][:16]}`.\n")
            elif t == "respond":
                w(f"- **{e['id']} {e['response']}**" + (f" (changed: {', '.join(e['changed'])})" if e.get("changed") else "") +
                  f": {e['text']}")
            elif t == "decide":
                w(f"- **the user decides {e['id']}**: {e['decision']}")
        w("")
    write(kpath(k, "REVIEW.md"), "\n".join(out) + "\n")
    if cert:
        dump_json(kpath(k, "review", "certificate.json"),
                  {"kernel": k, "gate": gate["digest"], "items": gate["items"], "reviewers": sorted(S["reviewers"]),
                   "time": now()})
    elif os.path.exists(kpath(k, "review", "certificate.json")):
        os.remove(kpath(k, "review", "certificate.json"))


def render_index():
    out = ["# Kernels\n",
           "> Generated by `tvr`; do not edit. One folder per kernel; the layout and the process are in "
           "[docs/REVIEW_PROTOCOL.md](../docs/REVIEW_PROTOCOL.md).\n",
           "| Kernel | What | Gate | Phase | Open blocking | Certificate |", "|---|---|---|---|---|---|"]
    for k in kernels():
        man = manifest(k)
        gate = load_json(kpath(k, "gate.json"), {"items": {}, "ok": False, "digest": ""})
        S = state(k)
        cert, blocking, _ = cert_state(k, S, gate)
        out.append(f"| [{k}]({k}/README.md) | {man.get('title', '')} | {'PASS' if gate['ok'] else 'FAIL'} | "
                   f"[{phase(k, S)}]({k}/REVIEW.md) | {len(blocking)} | {'GRANTED' if cert else '—'} |")
    write(os.path.join(KROOT, "README.md"), "\n".join(out) + "\n")


def issue_ids():
    p = os.path.join(LREVIEW, "ISSUES.md")
    return set(re.findall(r"^\| \[(L\d+)\]", open(p).read(), re.M)) if os.path.exists(p) else set()


def issues_for(k):
    """The rows of ISSUES.md that name one of this kernel's findings."""
    p = os.path.join(LREVIEW, "ISSUES.md")
    if not os.path.exists(p):
        return []
    rows, head = [], []
    for l in open(p).read().splitlines():
        if l.startswith("| Issue |"):
            head = [c.strip() for c in l.strip("|").split("|")]
        if not l.startswith("| [L"):
            continue
        cells = [c.strip() for c in l.strip("|").split("|")]
        col = {h: cells[i] for i, h in enumerate(head) if i < len(cells)}
        cells = [col.get("Issue", ""), col.get("State", ""), col.get("Source", ""), col.get("Summary", "")]
        src = cells[2]
        kern = None
        for part in re.split(r",\s*", src):
            m = re.match(r"([a-z0-9_]+)#F\d+", part)
            if m:
                kern = m.group(1)
            if kern == k and re.search(r"#F\d+", part):
                rows.append(cells)
                break
    return rows


def render_reports():
    out = ["# Library reports from kernel review\n",
           "> Generated by `tvr`; do not edit. Every `LIBRARY_REPORT` a kernel reviewer filed, and every finding the "
           "prover referred to the library. The coordinator triages them into [ISSUES.md](ISSUES.md).\n",
           "| Finding | Kind | Target | State | Summary | Referred as |", "|---|---|---|---|---|---|"]
    for k in kernels():
        S = state(k)
        for i in sorted(S["findings"], key=lambda x: int(x[1:])):
            f = S["findings"][i]
            if f["kind"] != "LIBRARY_REPORT" and f["state"] != "REFERRED" and f.get("last_response") != "REFERRED":
                continue
            ref = ""
            for e in f["history"]:
                if e["type"] == "respond" and e["response"] == "REFERRED":
                    ref = e.get("issue") or ", ".join(sorted(set(re.findall(r"\bL\d+\b", e["text"]))))
            out.append(f"| [{k}#{i}]({link(f'kernels/{k}/REVIEW.md', LREVIEW)}#{i.lower()}) | {f['kind']} | `{f['target']}` | {f['state']} | "
                       f"{f['summary'][:120].replace('|', '/')} | {ref} |")
    write(os.path.join(LREVIEW, "REPORTS.md"), "\n".join(out) + "\n")


def survey_areas():
    """The survey areas of library review: the table of docs/LIBRARY_REVIEW.md § Survey areas."""
    text = open(os.path.join(ROOT, "docs", "LIBRARY_REVIEW.md")).read()
    body = text.split("## Survey areas", 1)[1].split("\n## ", 1)[0] if "## Survey areas" in text else ""
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in body.splitlines() if l.startswith("| `")]
    return {r[0].strip("`"): (r[1], r[2]) for r in rows if len(r) >= 3}


def cmd_prompt(argv):
    if argv[2] == "library":
        argv = list(argv)
        area = None
        if "--area" in argv:
            i = argv.index("--area")
            area = argv[i + 1] if i + 1 < len(argv) else ""
            del argv[i:i + 2]
        role, rnd = argv[3], argv[4]
        rid = argv[5] if len(argv) > 5 else "lr-1"
        name = {"author": "library-author.md", "reviewer": "library-reviewer.md"}.get(role)
        fields = {"round": rnd, "reviewer": rid}
        if role == "reviewer":
            A = survey_areas()
            if area is not None and area not in A:
                die(f"no survey area {area!r} (docs/LIBRARY_REVIEW.md § Survey areas: {', '.join(A)})")
            fields["survey"] = "" if area is None else (
                f"## Your survey area: `{area}`\n\n"
                f"- **What it covers:** {A[area][0]}\n- **Where to read:** {A[area][1]}\n\n"
                "Beyond the batch, survey this area as it stands now. Look for statements that are false of\n"
                "Triton 3.8.0, unusable by a kernel reviewer, inconsistent with the rest, stronger than their\n"
                "evidence, or missing where kernels need them. Test what you doubt, as for the batch. File\n"
                "each defect under New issues, with its class (`soundness`, `gap` or `wording`, defined in\n"
                "`docs/LIBRARY_REVIEW.md`), where it is, and what would fix it.\n\n")
    else:
        k, role = argv[2], argv[3]
        if not os.path.exists(kpath(k, "kernel.json")):
            die(f"no kernel {k}")
        name = {"prover": "kernel-prover.md", "reviewer": "kernel-reviewer.md"}.get(role)
        fields = {"kernel": k, "round": str(state(k)["round"]), "reviewer": argv[4] if len(argv) > 4 else ""}
        if role == "reviewer" and not fields["reviewer"]:
            die("tvr prompt KERNEL reviewer ID")
    if not name:
        die("roles: prover, reviewer (kernel); author, reviewer (library)")
    text = open(os.path.join(ROOT, "docs", "prompts", name)).read()
    for a, b in fields.items():
        text = text.replace("{{" + a + "}}", b)
    # links in the prompts are relative to docs/prompts; an agent reads them from the TVA root
    text = re.sub(r"\]\(\.\./([^)]+)\)", lambda m: f"](docs/{m.group(1)})", text)
    left = re.findall(r"\{\{\w+\}\}", text)
    if left:
        die(f"unfilled fields: {', '.join(sorted(set(left)))}")
    print(text)



# ---------------------------------------------------------------- formats: every output is regular

class Problems(list):
    def need(self, cond, msg):
        if not cond:
            self.append(msg)


def md_sections(text):
    """The `# ` title and the `## ` sections (heading, body lines) of a Markdown file."""
    lines = text.splitlines()
    title = next((l[2:].strip() for l in lines if l.startswith("# ")), None)
    secs = []
    for l in lines:
        if l.startswith("## "):
            secs.append((l[3:].strip(), []))
        elif secs:
            secs[-1][1].append(l)
    return title, secs


def bullets(body):
    out = []
    for l in body:
        if l.startswith("- "):
            out.append(l[2:].strip())
        elif l.startswith("  ") and out:
            out[-1] += " " + l.strip()
    return out


def body_text(body):
    return " ".join(l.strip() for l in body).strip()


def sections(P, path, text, title, names):
    """Check the title and that the sections are exactly `names`, in order."""
    got_title, secs = md_sections(text)
    P.need(got_title == title, f"{path}: the title must be `# {title}` (found: {got_title!r})")
    got = [h for h, _ in secs]
    P.need(got == list(names), f"{path}: the sections must be exactly {', '.join(names)}, in this order "
           f"(found: {', '.join(got) or 'none'})")
    P.need("TODO" not in text, f"{path}: placeholders (TODO) are left")
    return {h: b for h, b in secs}


def listed(P, path, S, sec, empty):
    """A section of bullets, or the single word `empty` ("None.", "Nothing.")."""
    body = S.get(sec, [])
    bs = bullets(body)
    P.need(bs or body_text(body) == empty, f"{path} § {sec}: bullets, or `{empty}`")
    return bs


def keys(P, where, obj, required, optional=()):
    if not isinstance(obj, dict):
        P.append(f"{where}: must be an object")
        return False
    missing = [k for k in required if k not in obj]
    extra = sorted(set(obj) - set(required) - set(optional))
    P.need(not missing, f"{where}: missing {', '.join(missing)}")
    P.need(not extra, f"{where}: unknown keys {', '.join(extra)}")
    return not missing


def text_field(P, where, v, maxlen=None, one_line=False):
    ok = isinstance(v, str) and v.strip() != ""
    P.need(ok, f"{where}: a non-empty string")
    if ok and maxlen:
        P.need(len(v) <= maxlen, f"{where}: at most {maxlen} characters ({len(v)})")
    if ok and one_line:
        P.need("\n" not in v, f"{where}: one line")


def lint_filing(k, who, rnd, filing, items, path):
    P = Problems()
    if not keys(P, path, filing, ("reviewer", "kernel", "round"), ("findings", "release", "reraise", "accept", "notes")):
        return P
    P.need(filing["reviewer"] == who, f"{path}: reviewer must be {who!r}")
    P.need(filing["kernel"] == k, f"{path}: kernel must be {k!r}")
    P.need(filing["round"] == rnd, f"{path}: round must be {rnd} (the current round)")
    for i, f in enumerate(filing.get("findings", [])):
        w = f"{path}: findings[{i}]"
        if not keys(P, w, f, ("kind", "target", "summary"), ("detail", "evidence", "affects", "blocking")):
            continue
        P.need(f["kind"] in KINDS, f"{w}.kind: one of {', '.join(KINDS)}")
        text_field(P, f"{w}.target", f["target"], 200, True)
        text_field(P, f"{w}.summary", f["summary"], 200, True)
        if f["kind"] in KINDS:
            blocking = KINDS[f["kind"]][0]
            P.need(f.get("blocking", blocking) == blocking, f"{w}.blocking: {f['kind']} is "
                   f"{'blocking' if blocking else 'non-blocking'} (use DOC or QUESTION for a non-blocking remark)")
            if blocking:
                text_field(P, f"{w}.detail (a blocking finding explains itself)", f.get("detail"))
        for key in ("detail", "evidence"):
            if key in f:
                P.need(isinstance(f[key], str), f"{w}.{key}: a string")
        if "affects" in f:
            P.need(isinstance(f["affects"], list) and all(isinstance(a, str) and a.startswith("item:") and a[5:] in items
                                                           for a in f["affects"]), f"{w}.affects: a list of item:<name>")
    for key in ("release", "reraise"):
        for i, r in enumerate(filing.get(key, [])):
            w = f"{path}: {key}[{i}]"
            if keys(P, w, r, ("id", "note")):
                P.need(isinstance(r["id"], str) and re.fullmatch(r"F\d+", r["id"]) is not None, f"{w}.id: F<n>")
                text_field(P, f"{w}.note", r["note"])
    acc = filing.get("accept", [])
    P.need(acc == "all" or (isinstance(acc, list) and all(isinstance(a, str) and a.startswith("item:") and a[5:] in items
                                                          for a in acc)), f"{path}: accept is \"all\" or a list of item:<name>")
    if "notes" in filing:
        P.need(isinstance(filing["notes"], str), f"{path}: notes is a string")
    return P


def lint_review_report(k, who, rnd, filing, items, path):
    P = Problems()
    name = rel(path) if os.path.isabs(path) else path
    if not os.path.exists(path):
        P.append(f"{name}: missing")
        return P
    text = open(path).read()
    S = sections(P, name, text, f"Review of {k}, round {rnd}, by {who}",
                 ("Verdict", "Findings", "Accepted", "Checked", "Not checked"))
    verdict = next((l.strip() for l in S.get("Verdict", []) if l.strip()), "")
    P.need(verdict in ("accept", "accept-partly", "reject"),
           f"{name} § Verdict: the first line is accept, accept-partly or reject")
    acc = filing.get("accept", [])
    accepted = set(items) if acc == "all" else {a[5:] for a in acc}
    new_blocking = [f for f in filing.get("findings", []) if KINDS.get(f.get("kind"), (False,))[0]]
    if verdict == "accept":
        P.need(accepted == set(items) and not new_blocking and not filing.get("reraise"),
               f"{name} § Verdict: accept means every item accepted, with no new blocking finding and no re-raise")
    elif verdict == "reject":
        P.need(not accepted, f"{name} § Verdict: reject means no item accepted")
    elif verdict == "accept-partly":
        P.need(accepted and accepted != set(items), f"{name} § Verdict: accept-partly means some items, not all")
    want = ([f"**{f.get('kind')}** `{f.get('target')}`" for f in filing.get("findings", [])] +
            [f"**release {r.get('id')}**" for r in filing.get("release", [])] +
            [f"**re-raise {r.get('id')}**" for r in filing.get("reraise", [])])
    got = listed(P, name, S, "Findings", "None.")
    P.need(len(got) == len(want) and all(g.startswith(x) for g, x in zip(got, want)),
           f"{name} § Findings: one bullet per finding, release and re-raise of the filing, in its order, "
           f"starting {' / '.join(want[:3]) or '(none: write None.)'}{' …' if len(want) > 3 else ''}")
    body = body_text(S.get("Accepted", []))
    if acc == "all":
        P.need(body == "All items.", f"{name} § Accepted: `All items.` (the filing accepts all)")
    else:
        got = {re.match(r"`item:([^`]+)`", b).group(1) for b in bullets(S.get("Accepted", [])) if re.match(r"`item:([^`]+)`", b)}
        P.need(got == accepted and (accepted or body == "None."),
               f"{name} § Accepted: one bullet `item:<name>` per accepted item, or `None.`")
    P.need(len(listed(P, name, S, "Checked", "")) >= 1, f"{name} § Checked: at least one bullet")
    listed(P, name, S, "Not checked", "Nothing.")
    return P


def lint_response(k, rnd, resp, open_ids, items, path):
    P = Problems()
    if not keys(P, path, resp, ("kernel", "round", "responses")):
        return P
    P.need(resp["kernel"] == k, f"{path}: kernel must be {k!r}")
    P.need(resp["round"] == rnd, f"{path}: round must be {rnd}")
    seen = []
    for i, r in enumerate(resp["responses"] if isinstance(resp["responses"], list) else []):
        w = f"{path}: responses[{i}]"
        if not keys(P, w, r, ("id", "response", "text"), ("changed", "issue")):
            continue
        seen.append(r["id"])
        P.need(r["response"] in RESPONSES, f"{w}.response: one of {', '.join(sorted(RESPONSES))}")
        text_field(P, f"{w}.text", r["text"])
        if r["response"] == "REVISED":
            P.need(isinstance(r.get("changed"), list) and r["changed"] and
                   all(isinstance(c, str) and c.startswith("item:") and c[5:] in items for c in r["changed"]),
                   f"{w}.changed: REVISED names the changed items (item:<name>)")
        else:
            P.need("changed" not in r, f"{w}.changed: only a REVISED response names changed items")
        if r["response"] == "REFERRED":
            P.need(isinstance(r.get("issue"), str) and r["issue"] in issue_ids(),
                   f"{w}.issue: REFERRED names a library issue of docs/library-review/ISSUES.md (L<n>)")
        else:
            P.need("issue" not in r, f"{w}.issue: only a REFERRED response names an issue")
    P.need(sorted(seen) == sorted(open_ids) and len(seen) == len(set(seen)),
           f"{path}: exactly one response per open finding ({', '.join(open_ids) or 'none'})")
    return P


def lint_prover_notes(k, rnd, resp, gate, path):
    P = Problems()
    name = rel(path) if os.path.isabs(path) else path
    if not os.path.exists(path):
        P.append(f"{name}: missing")
        return P
    S = sections(P, name, open(path).read(), f"Prover's notes on {k}, round {rnd}",
                 ("Changes", "Responses", "Library requests", "Gate"))
    listed(P, name, S, "Changes", "None.")
    want = [f"**{r.get('id')}** {r.get('response')}" for r in resp.get("responses", [])]
    got = bullets(S.get("Responses", []))
    P.need(len(got) == len(want) and all(g.startswith(x) for g, x in zip(got, want)),
           f"{name} § Responses: one bullet per response, in order, starting {' / '.join(want[:3])}")
    listed(P, name, S, "Library requests", "None.")
    P.need(body_text(S.get("Gate", [])) == f"PASS, digest `{gate['digest'][:16]}`",
           f"{name} § Gate: `PASS, digest `{gate['digest'][:16]}`` (the current gate)")
    return P


def issues_of_round(rnd):
    p = os.path.join(LREVIEW, "ISSUES.md")
    out, head = [], []
    for l in (open(p).read().splitlines() if os.path.exists(p) else []):
        if l.startswith("| Issue |"):
            head = [c.strip() for c in l.strip("|").split("|")]
        elif l.startswith("| [L"):
            cells = dict(zip(head, [c.strip() for c in l.strip("|").split("|")]))
            if cells.get("Round") == str(rnd):
                out.append(re.match(r"\[(L\d+)\]", cells["Issue"]).group(1))
    return out


def lint_author_notes(rnd, path):
    P = Problems()
    name = rel(path) if os.path.isabs(path) else path
    if not os.path.exists(path):
        P.append(f"{name}: missing")
        return P
    want = issues_of_round(rnd)
    text = open(path).read()
    title, secs = md_sections(text)
    P.need(title == f"Library batch, round {rnd}: author's notes", f"{name}: the title must be "
           f"`# Library batch, round {rnd}: author's notes`")
    P.need("TODO" not in text, f"{name}: placeholders (TODO) are left")
    got = [re.match(r"(L\d+): ", h).group(1) if re.match(r"(L\d+): ", h) else h for h, _ in secs]
    P.need(got == want, f"{name}: one section `## L<n>: <title>` per issue of round {rnd}, in order "
           f"({', '.join(want) or 'none'}; found {', '.join(got) or 'none'})")
    for h, body in secs:
        bs = bullets(body)
        for lead in ("**What changed", "**Why", "**Constants"):
            P.need(any(b.startswith(lead) for b in bs), f"{name} § {h}: a bullet starting {lead}…**")
    return P


def lint_library_report(rnd, who, path):
    P = Problems()
    name = rel(path) if os.path.isabs(path) else path
    if not os.path.exists(path):
        P.append(f"{name}: missing")
        return P
    S = sections(P, name, open(path).read(), f"Library review, round {rnd}, by {who}",
                 ("Verdicts", "New issues", "Re-ran", "Not checked"))
    want = issues_of_round(rnd)
    got = []
    for b in bullets(S.get("Verdicts", [])):
        m = re.match(r"\*\*(L\d+)\*\* (ACCEPT|REOPEN) — \S", b)
        P.need(m is not None, f"{name} § Verdicts: `- **L<n>** ACCEPT — <reason>` or `REOPEN — <what is wrong>`: {b[:60]!r}")
        if m:
            got.append(m.group(1))
    P.need(sorted(got) == sorted(want) and len(got) == len(set(got)),
           f"{name} § Verdicts: one verdict per issue of round {rnd} ({', '.join(want)})")
    for b in listed(P, name, S, "New issues", "None."):
        P.need(re.match(r"\*\*NEW\*\* (soundness|gap|wording) — \S", b) is not None,
               f"{name} § New issues: `- **NEW** soundness|gap|wording — <summary>`: {b[:60]!r}")
    listed(P, name, S, "Re-ran", "Nothing.")
    listed(P, name, S, "Not checked", "Nothing.")
    return P


def report_problems(P, what):
    if P:
        die(f"{what} is not regular ({len(P)} problem(s); `tvr lint` shows the same):\n  - " + "\n  - ".join(P))


def kernel_round_files(k, role, who=None):
    S = state(k)
    N = S["round"]
    if role == "reviewer":
        count = S["filed"].get(who, {}).get(N, 0)
        return N, round_file(k, N, who, "filing.json", count), round_file(k, N, who, "report.md", count)
    return N, kpath(k, "review", f"round-{N}", "prover", "response.json"), kpath(k, "review", f"round-{N}", "prover", "notes.md")


def cmd_start(argv):
    """Write the skeleton of a role's output (never over an existing file)."""
    made = []

    def put(path, text):
        if not os.path.exists(path):
            write(path, text)
            made.append(rel(path))

    if argv[2] == "library":
        rnd, who = int(argv[3]), argv[4]
        issues = issues_of_round(rnd)
        if who == "author":
            put(os.path.join(LREVIEW, f"round-{rnd}", "author", "notes.md"),
                f"# Library batch, round {rnd}: author's notes\n\n" + "".join(
                    f"## {i}: TODO title\n\n- **What changed.** TODO\n- **Why.** TODO\n- **Evidence.** TODO\n"
                    f"- **Constants whose meaning changed.** TODO\n\n" for i in issues))
        else:
            put(os.path.join(LREVIEW, f"round-{rnd}", who, "report.md"),
                f"# Library review, round {rnd}, by {who}\n\n## Verdicts\n\n" +
                "".join(f"- **{i}** TODO — TODO\n" for i in issues) +
                "\n## New issues\n\n- **NEW** TODO — TODO (soundness, gap or wording; or replace the list with None.)\n"
                "\n## Re-ran\n\n- TODO\n\n## Not checked\n\nNothing.\n")
            os.makedirs(os.path.join(LREVIEW, f"round-{rnd}", who, "work"), exist_ok=True)
    else:
        k, role = argv[2], argv[3]
        if role == "reviewer":
            who = argv[4]
            N, fp, rp = kernel_round_files(k, "reviewer", who)
            put(fp, json.dumps({"reviewer": who, "kernel": k, "round": N, "findings": [], "release": [],
                                "reraise": [], "accept": [], "notes": ""}, indent=2) + "\n")
            put(rp, f"# Review of {k}, round {N}, by {who}\n\n## Verdict\n\nTODO (accept, accept-partly or reject)\n\n"
                    "## Findings\n\nNone.\n\n## Accepted\n\nNone.\n\n## Checked\n\n- TODO\n\n## Not checked\n\nNothing.\n")
            os.makedirs(os.path.join(os.path.dirname(fp), "work"), exist_ok=True)
        else:
            N, fp, np_ = kernel_round_files(k, "prover")
            open_ids = sorted((i for i, f in state(k)["findings"].items() if f["state"] == "OPEN"), key=lambda x: int(x[1:]))
            put(fp, json.dumps({"kernel": k, "round": N, "responses": [
                {"id": i, "response": "TODO", "text": "TODO"} for i in open_ids]}, indent=2) + "\n")
            put(np_, f"# Prover's notes on {k}, round {N}\n\n## Changes\n\n- TODO\n\n## Responses\n\n" +
                "".join(f"- **{i}** TODO — TODO\n" for i in open_ids) +
                "\n## Library requests\n\nNone.\n\n## Gate\n\nTODO (PASS, digest `<16 hex digits>`, from tvr check)\n")
    print("\n".join(f"wrote {m}" for m in made) if made else "nothing to write: the files exist")


def cmd_lint(argv):
    if argv[2] == "library":
        rnd, who = int(argv[3]), argv[4]
        if who == "author":
            path = os.path.join(LREVIEW, f"round-{rnd}", "author", "notes.md")
            P = lint_author_notes(rnd, path)
            P += scan_files([path])
        else:
            path = os.path.join(LREVIEW, f"round-{rnd}", who, "report.md")
            P = lint_library_report(rnd, who, path)
            P += scan_files([path] + text_files(os.path.join(os.path.dirname(path), "work")))
    else:
        k, role = argv[2], argv[3]
        gate = load_json(kpath(k, "gate.json"), {"items": {}, "digest": ""})
        if role == "reviewer":
            who = argv[4]
            N, fp, rp = kernel_round_files(k, "reviewer", who)
            count = state(k)["filed"].get(who, {}).get(N, 0)
            if count and not os.path.exists(fp):
                print(f"{who} has filed round {N}; a further filing this round goes in {rel(fp)} "
                      f"(`tvr start {k} reviewer {who}` writes it)")
                return 0
            filing = load_json(fp)
            P = lint_filing(k, who, N, filing, gate["items"], rel(fp))
            P += lint_review_report(k, who, N, filing if isinstance(filing, dict) else {}, gate["items"], rp)
            P += scan_files([fp, rp] + text_files(os.path.join(os.path.dirname(fp), "work")))
        else:
            N, fp, np_ = kernel_round_files(k, "prover")
            resp = load_json(fp)
            open_ids = sorted((i for i, f in state(k)["findings"].items() if f["state"] == "OPEN"), key=lambda x: int(x[1:]))
            P = lint_response(k, N, resp, open_ids, gate["items"], rel(fp))
            P += lint_prover_notes(k, N, resp if isinstance(resp, dict) else {}, gate, np_)
            P += scan_files([fp, np_])
    if P:
        print("not regular:\n  - " + "\n  - ".join(P))
        return 1
    print("regular")
    return 0


def refresh(k):
    render_review(k)
    render_index()
    render_reports()


# ---------------------------------------------------------------- scaffold

README_TEMPLATE = """# {title}

The prover's write-up. The reviewers judge [claim.md](claim.md) (generated); the review is
[REVIEW.md](REVIEW.md) (generated).

Source: [`{src}`]({src}). Lean: [`lean/`](lean/).

## Specification

## Conditions

## Assumptions

## Proof

## Narrowings and modelling decisions
"""


def cmd_new(k, module, ns, source):
    if not re.fullmatch(r"[a-z0-9_]+", k):
        die("a kernel name is lower case, digits and '_'")
    if os.path.exists(kpath(k)):
        die(f"kernels/{k} exists")
    base = os.path.basename(source)
    for d in ("source", "lean", "review"):
        os.makedirs(kpath(k, d))
    shutil.copyfile(os.path.join(ROOT, source), kpath(k, "source", base))
    dump_json(kpath(k, "kernel.json"), {
        "protocol": "tvr-2", "title": k,
        "lean": {"module": module, "namespace": ns},
        "source": {"file": f"source/{base}", "sha256": file_sha(kpath(k, "source", base)), "ranges": [], "host": []},
        "doc": "README.md"})
    dump_json(kpath(k, "model-map.json"), {"entries": []})
    write(kpath(k, "README.md"), README_TEMPLATE.format(title=k, src=f"source/{base}"))
    add_lake_lib(k, module)
    render_index()
    print(f"created kernels/{k}; write lean/{module.replace('.', '/')}.lean, fill kernel.json (ranges) and model-map.json")


def add_lake_lib(k, module):
    p = os.path.join(ROOT, "lean", "lakefile.toml")
    s = open(p).read()
    if re.search(rf'^name = "{re.escape(k)}"$', s, re.M):
        return
    s = re.sub(r'^defaultTargets = \[(.*?)\]', lambda m: f'defaultTargets = [{m.group(1)}, "{k}"]', s, count=1, flags=re.M)
    s = s.rstrip("\n") + f'\n\n[[lean_lib]]\nname = "{k}"\nsrcDir = "../kernels/{k}/lean"\nroots = ["{module.split(".")[0]}"]\n'
    write(p, s)


def main(argv):
    if len(argv) >= 2 and argv[1] == "index":
        for k in kernels():
            render_review(k)
        render_index()
        render_reports()
        return 0
    if len(argv) >= 2 and argv[1] == "audit":
        bad = library_independence()
        for k in kernels():
            bad += kernel_independence(k)
        bad += scan_files([p for p in text_files(KROOT) if os.path.dirname(p) == KROOT or os.path.islink(p)])
        bad += transcription_rules()[3]
        print("\n".join(bad) if bad else "independent: TVA imports only TVA and Mathlib, and refers to no other project")
        return 1 if bad else 0
    if len(argv) >= 4 and argv[1] == "prompt":
        cmd_prompt(argv)
        return 0
    if len(argv) >= 4 and argv[1] == "start":
        cmd_start(argv)
        return 0
    if len(argv) >= 4 and argv[1] == "lint":
        return cmd_lint(argv)
    if len(argv) < 3:
        print(__doc__)
        return 1
    cmd, k = argv[1], argv[2]

    def opt(name):
        return argv[argv.index(name) + 1] if name in argv else None

    if cmd == "new":
        cmd_new(k, opt("--module"), opt("--namespace"), opt("--source"))
        return 0
    if not os.path.exists(kpath(k, "kernel.json")):
        die(f"no kernel {k} (kernels/{k}/kernel.json)")
    if cmd == "check":
        return 0 if check(k) else 1
    if cmd == "assign":
        cmd_assign(k, argv[3], opt("--focus") or "all")
    elif cmd == "file":
        cmd_file(k, argv[3])
    elif cmd == "respond":
        cmd_respond(k)
    elif cmd == "decide":
        cmd_decide(k, argv[3], " ".join(argv[4:]))
    elif cmd == "note":
        files = argv[argv.index("--files") + 1:] if "--files" in argv else []
        text = " ".join(argv[3:argv.index("--files")] if "--files" in argv else argv[3:])
        cmd_note(k, text, files)
    elif cmd == "status":
        S = state(k)
        print(phase(k, S))
        print(status_text(k, opt("--as")))
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

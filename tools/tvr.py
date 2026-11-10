#!/usr/bin/env python3
"""tvr: TVA's prover/reviewer protocol (docs/REVIEW_PROTOCOL.md).

Each kernel is one folder, kernels/<kernel>/ (the layout is section 1 of the protocol).

  tvr new     KERNEL --module M --namespace NS --source FILE
                                scaffold kernels/KERNEL (folder, manifest; its library in lean/.lake/tvr/lakefile.toml)
  tvr check   KERNEL            run the gate: build, read the Lean claim (tools/claim_reader.lean.in); write gate.json,
                                claim.md and a snapshot of the item cones
  tvr assign  KERNEL REVIEWER   assign a reviewer
  tvr file    KERNEL REVIEWER   record the reviewer's review/round-N/REVIEWER/filing.json
  tvr respond KERNEL            record the prover's review/round-N/prover/response.json
  tvr decide  KERNEL ID TEXT    record the user's decision on an escalated finding
  tvr note    KERNEL TEXT [--files F …]
                                record a coordinator's note (never a file of a role folder)
  tvr freeze  KERNEL [--dry-run]
                                record, once, the work files of each role folder whose filing or response was
                                recorded before the tool named folders, and left them out (the coordinator's)
  tvr status  KERNEL [--as R]   print the status (REVIEW.md has the whole record)
  tvr prompt  KERNEL prover|reviewer [ID]
  tvr prompt  library author|reviewer|coverage ROUND [ID] [--area AREA]
                                print a role's prompt (docs/prompts/) with its fields filled in
  tvr index                     regenerate kernels/README.md and docs/library-review/REPORTS.md
  tvr audit                     check that the library and every kernel are independent (protocol §2)
  tvr release [--dir DIR] [--keep] [--no-mask]
                                the release check (protocol §2): copy the files TVA ships into an empty
                                allowlist sandbox, make Mathlib's packages there from their upstream
                                repositories (network needed), and build, re-check the library, check the
                                ledgers and the events they hold, lint the records not yet recorded and audit
                                there (no kernel's gate is recomputed, library issue L160), running only the
                                system's git, bwrap and find (about 25 minutes and 16 GB: Mathlib is built
                                from source; the sandbox is removed afterwards
                                unless --keep, release.json stays; the output masks the folder as <dir>)
  tvr library-check             the library's hygiene, that lean/TVA.lean reaches every library module, and
                                `leanchecker TVA` (after `lake build TVA`)
  tvr record  library ROUND WHO [--no-lint]
                                record a library role's folder (round-ROUND/WHO) in the library's ledger,
                                docs/library-review/ledger.jsonl, after checking it as `tvr lint` does
                                (--no-lint only for rounds 1 to 7, before the ledger)
  tvr start   KERNEL reviewer ID | KERNEL prover | library ROUND author|coverage|ID
                                write the skeleton of a role's output (the formats: protocol §11)
  tvr lint    KERNEL reviewer ID | KERNEL prover | library ROUND author|coverage|ID
                                check that a role's output is regular before it is recorded

Run it as `python3 -I tools/tvr.py ...` from the TVA root. Standard library only.
"""
import ast
import datetime
import hashlib
import html.entities
import io
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tokenize
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
    # the write-up states a claim other than the Lean's (library issue L136): blocks, unlike wording
    "DOC_WRONG": (True, ("item:", "decl:", "doc:"), None),
    "DOC": (False, ("item:", "decl:", "src:", "map:", "lib:", "doc:"), None),
    "QUESTION": (False, ("item:", "decl:", "src:", "map:", "lib:", "doc:"), None),
}
RESPONSES = {"REVISED", "DISPUTED", "CLARIFIED", "DEFERRED", "REFERRED"}
# A model-map entry's drop reason is the prover's, judged by the kernel reviewer (protocol §7, library
# issue L36); docs/TRANSCRIPTION.md lists the common ones as guidance, and the gate checks none of them.
ESCALATE_AFTER = 2  # re-raises after which a finding goes to the user
# `filing_events` derives the events of every recorded filing again with today's values of a kind's default
# `blocking` (KINDS), ESCALATE_AFTER and the escalation memo, not with those of the day it was filed (protocol
# §3, library issue L82): a change to any of them makes every ledger with an event they decided fail the
# release check's records step and the certificate, until the derivation keeps the old values for old events.
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
# Absolute paths outside /home that are the system's, not a project's, and the conventional
# placeholder `/path/to/…` (a tutorial's `save_path`).
SYSTEM_DIRS = {"usr", "bin", "sbin", "lib", "lib64", "etc", "dev", "proc", "sys"}
PLACEHOLDER_ROOTS = {"path"}
SKIP_DIRS = {".lake", ".venv", ".git", "__pycache__"}  # build outputs, the Python environment, git
# The sandbox (library issue L135): the containers in which agents prove and review a kernel. Its files run
# there and name the containers' own paths, so the text scan skips its generic path tests for them.
SANDBOX = os.path.join(ROOT, "sandbox")
LREVIEW = os.path.join(ROOT, "docs", "library-review")
# A Lean name as the claim reader takes it from a kernel's manifest (library issue L38): plain ASCII
# components, so that the text spliced into the reader can be nothing but a name.
LEAN_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*")
# The package-wide Lean options lean/lakefile.toml may set (library issue L40): the same as a kernel's
# `set_option` allowlist (HYGIENE), nothing that changes what Lean checks.
LEAN_OPTIONS = re.compile(r"maxHeartbeats|maxRecDepth|synthInstance\.maxHeartbeats|synthInstance\.maxSize|"
                          r"autoImplicit|relaxedAutoImplicit|linter\.[\w.]+|pp\.[\w.]+")
# lean/lean-toolchain (library issue L54): a release of Lean from its own organisation, as elan installs it.
TOOLCHAIN = re.compile(r"leanprover/lean4:v\d+\.\d+\.\d+(?:-rc\d+)?")
# Line breaks other than a line feed (library issue L51): `str.splitlines` breaks at them, while Python's
# tokenizer, Triton and Lean count line feeds only; a source or kernel Lean file holding one is refused, so
# every reader numbers its lines alike. A carriage return is allowed only before a line feed.
LINE_SEPARATORS = re.compile("\r(?!\n)|[\x0b\x0c\x1c\x1d\x1e\x85  ]")


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


def rel(path, base=None):
    return os.path.relpath(path, base or ROOT)


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


def manifest_problems(k, man):
    """What the gate takes from a kernel's manifest, checked before it is used (library issue L38): the
    Lean module and namespace are plain names (`LEAN_NAME`), since the claim reader is Lean text built
    around them; the source and the write-up are files of the kernel folder; the line ranges are pairs
    of line numbers."""
    bad = []
    lean = man.get("lean") if isinstance(man.get("lean"), dict) else {}
    for key in ("module", "namespace"):
        v = lean.get(key)
        if not (isinstance(v, str) and LEAN_NAME.fullmatch(v)):
            bad.append(f"lean.{key} must be a dotted name of ASCII letters, digits, `_` and `'` (found {str(v)[:60]!r})")
    src = man.get("source") if isinstance(man.get("source"), dict) else {}
    for key, v in (("source.file", src.get("file")), ("doc", man.get("doc", "README.md"))):
        inside = isinstance(v, str) and v and not os.path.isabs(v) and \
            not os.path.normpath(v).split(os.sep)[0] == os.pardir
        if not inside:
            bad.append(f"{key} must be a path inside the kernel folder (found {str(v)[:60]!r})")
    if not isinstance(src.get("sha256"), str):
        bad.append("source.sha256 must be a string")
    for key in ("ranges", "host"):
        v = src.get(key, [] if key == "host" else None)
        if not (isinstance(v, list) and all(isinstance(r, list) and len(r) == 2 and
                                            all(isinstance(x, int) and not isinstance(x, bool) for x in r) for r in v)):
            bad.append(f"source.{key} must be a list of [first, last] line numbers")
    # the functions of the source that nothing runs (library issue L50): every other function lies in the
    # kernel lines or the host lines, and the kernel reviewers judge the lists
    v = src.get("unused", [])
    if not (isinstance(v, list) and all(isinstance(x, str) and x.isidentifier() for x in v)):
        bad.append("source.unused must be a list of function names")
    return bad


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
    """Triton 3.8.0's own files, as pinned by sha256 in `primitives/` and `tutorials/`. Their text is
    Triton's, so they are exempt from the generic absolute-path and relative-path tests (a tutorial's
    `save_path` is a placeholder, not a path into a project); the name, `fpv/<name>` and home-path
    tests still apply to them (library issue L7). A kernel's source is not among them: the prover
    writes and pins it, so every test applies to it."""
    out = set()
    for d, man, key in (("primitives", "semantics-manifest.json", "sources"), ("primitives", "manifest.json", "primitives"),
                        ("tutorials", "inventory.json", "tutorials")):
        for e in load_json(os.path.join(ROOT, d, man), {}).get(key, []):
            f = os.path.join(ROOT, d, e.get("source", ""))
            if os.path.isfile(f) and file_sha(f) == e.get("sha256"):
                out.add(os.path.realpath(f))
    return out


# Paths. The scan is lexical: it reads paths as they are written, and a path a program computes at run
# time is not checked (protocol §2). Before the path test, a text is normalised (`_path_views`):
# - NFKC, before and after decoding the escapes of a path character, a letter or a digit that a reader
#   of the text decodes: `\/`, `\uXXXX`, `\UXXXXXXXX`, `\xXX`, a Python octal `\NNN` and `\N{NAME}`,
#   `%XX`, `&#N;`, `&#xN;` and named HTML entities (`&sol;`, `&bsol;`, …);
# - a line continuation (a backslash before a line break) joins its lines;
# - a `file:` URL is read as the path it names, and every other URL is removed;
# - the home directory's spellings are read as `~`: `$HOME`, `${HOME…}` with any expansion operator
#   and nested braces (`${HOME:-${X}}`), `%HOME%`, `%USERPROFILE%`, `$env:HOME`, `$env:USERPROFILE`,
#   every command substitution `$(…)` (nested to any depth), and a command substitution in backquotes
#   that mentions the home directory (`` `echo ~` ``, `` `echo $HOME` ``) right before a separator.
# The tests then read several views of that text: the text itself; the text with its quotes (`"`, `'`)
# removed, as a shell joins a quoted path component with its neighbours into one word; and, for each,
# the views in which every brace
# group without spaces (`{a,b}`) is replaced by one of its alternatives, as a shell expands it. In each
# view:
# - wherever it occurs, a path component `fpv` followed by a component other than `tva` names a sibling
#   of TVA (`_FPV`), whatever precedes it;
# - a *home* path (`~/`, `~<user>/`, `/home/`, also with `\` after the home prefix) starts after any
#   character other than a letter, a digit, `_`, `.`, `/`, `\` or `~`;
# - another absolute path is a `/` after any character that cannot be part of a path name (a space, a
#   quote, an opening bracket, `=`, `:`, `;`, `,`, …), except a closing bracket, `>`, `}` or `-`, which end
#   a placeholder such as `<N>` or `{kernel}` inside a relative path, and except a closing quote (one
#   right after a path character), whose word the view without quotes reads whole;
# - a relative path is any run of path characters.
# Triton's own pinned files (`upstream_files`) are read by the name, `fpv` and home tests only; every
# other file, a kernel's source included, by every test.
_PATH_CHARS = r"[^\s`'\"()\[\]{}<>,;|*]"
_FILE_URL = re.compile(r"(?i)\bfile:(?://(?:localhost)?)?(?=/)")
_URL = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s`'\"<>()\[\]{}|;,]*")
_HOME_SIMPLE = re.compile(r"\$HOME(?!\w)|%HOME%|%USERPROFILE%|\$env:(?:HOME|USERPROFILE)(?!\w)")
_HOME_TICKS = re.compile(r"\x60[^\x60\n]*(?:~|(?<!\w)HOME(?!\w))[^\x60\n]*\x60(?=[/\\])")  # \x60: a backquote
_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{8})|\\x([0-9a-fA-F]{2})|%([0-9a-fA-F]{2})|&#(\d{1,7});|"
                     r"&#[xX]([0-9a-fA-F]{1,6});|\\([0-7]{1,3})|\\N\{([A-Za-z0-9 \-]{1,80})\}|&([A-Za-z][A-Za-z0-9]{1,31});")
_ESCAPED = set("/\\~.$-_:") | set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
_ENTITIES = {k[:-1]: unicodedata.normalize("NFKC", v) for k, v in html.entities.html5.items()
             if k.endswith(";") and len(unicodedata.normalize("NFKC", v)) == 1 and unicodedata.normalize("NFKC", v) in _ESCAPED}
_CONTINUATION = re.compile(r"\\\r?\n[ \t]*")
_BRACES = re.compile(r"\{([^{}\s]*,[^{}\s]*)\}")
_HOME_START = re.compile(r"(?<![\w./\\~])(?:~[\w.-]*[/\\]|/home/)" + _PATH_CHARS + r"*")
_ABS = re.compile(r"(?<![\w.~/\\>})\]-])(?<![\w.~/\\>})\]-][\"'])(?:~[\w.-]*)?/(?=[\w.~])" + _PATH_CHARS + r"*")
_FPV = re.compile(r"(?i)(?<![\w.-])fpv[/\\]+(" + _PATH_CHARS + r"*)")
_REL = re.compile(r"[^\s`'\"()\[\]{}<>,;|*=:]+")


def _unescape(text):
    """Decode the escapes of path characters (and letters and digits) that a reader of the text would
    decode: JSON and Python `\\u`, `\\U`, `\\x`, Python octal `\\NNN` and `\\N{NAME}`, URL `%XX`, HTML
    `&#N;`, `&#xN;` and named entities, and `\\/`."""
    def dec(m):
        g = m.groups()
        if g[7] is not None:
            try:
                c = unicodedata.normalize("NFKC", unicodedata.lookup(g[7]))
            except KeyError:
                return m.group(0)
        elif g[8] is not None:
            c = _ENTITIES.get(g[8], "")
        else:
            i, v = next((i, v) for i, v in enumerate(g) if v is not None)
            n = int(v, 10 if i == 4 else 8 if i == 6 else 16)
            c = chr(n) if n < 0x110000 else ""
        return c if c in _ESCAPED else m.group(0)
    return _ESCAPE.sub(dec, text.replace("\\/", "/"))


def _home_vars(t):
    """Read the home directory's spellings and every command substitution as ` ~` (balanced brackets,
    so `${HOME:-${X}}` and `$(a $(b $(echo ~)))` are read whole)."""
    out, i, n = [], 0, len(t)
    while i < n:
        j = -1
        if t.startswith("${HOME", i) and not (i + 6 < n and (t[i + 6].isalnum() or t[i + 6] == "_")):
            j, opn, cls = i + 6, "{", "}"
        elif t.startswith("$(", i):
            j, opn, cls = i + 2, "(", ")"
        if j < 0:
            out.append(t[i])
            i += 1
            continue
        depth = 1
        while j < n and depth:
            depth += (t[j] == opn) - (t[j] == cls)
            j += 1
        out.append(" ~")
        i = j
    t = "".join(out)
    t = _HOME_TICKS.sub(" ~", t)
    return _HOME_SIMPLE.sub(" ~", t)


def _path_views(text):
    """The views of a text the path tests read (see the comment above `_PATH_CHARS`)."""
    t = unicodedata.normalize("NFKC", _unescape(unicodedata.normalize("NFKC", text)))
    t = _CONTINUATION.sub("", t)
    views = [t, t.replace('"', "").replace("'", "")]
    for v in list(views):
        groups = [m.group(1).split(",") for m in _BRACES.finditer(v)]
        for k in range(min(16, max((len(g) for g in groups), default=0))):
            def pick(m, k=k):
                alts = m.group(1).split(",")
                return alts[min(k, len(alts) - 1)]
            views.append(_BRACES.sub(pick, v))
    out = []
    for v in views:
        v = _FILE_URL.sub(" ", v)
        v = _URL.sub(" ", v)
        out.append(_home_vars(v))
    return out


def _home_ok(parts):
    """A path under the home directory (as components) is TVA's, the toolchain's, Triton's cache, or a
    folder that holds one of them (such as the home directory itself)."""
    return any(tuple(parts[:len(ok)]) == ok or tuple(ok[:len(parts)]) == tuple(parts) for ok in HOME_OK)


def md_cell(text):
    """Text for a cell of a Markdown table: a `|` escaped as `\\|` (library issue L138). Writing it as
    `/` turned a reviewer's `|x|=2^30` into `/x/=2^30`, which `foreign_refs` reads as an absolute path,
    so the generated REVIEW.md failed the kernel's own independence check."""
    return text.replace("\n", " ").replace("|", "\\|")


def foreign_refs(text, where=None, absolute=True, relative=True, home=True):
    """References to other projects in a text: a hashed name, or a path that leads outside TVA.

    A name counts as a word, as a piece of a word (`Name_v2`, `NameLemmas`, `nameLemmas`), or at the
    start or end of a one-case piece (`namelemmas`), in any case. A path counts, in any view of the text
    (`_path_views`), when it holds a component `fpv` followed by one other than `tva`, wherever it
    starts; when it is a home path (`~/…`, `~<user>/…`, `/home/<user>/…`, and the spellings read as
    these, a `file:` URL included) not under TVA, the Lean toolchain or Triton's cache; when it is
    another absolute path of two components or more outside the system directories; or when it is
    relative, climbs with `..`, and leaves TVA from `where` (the directory of the file that holds it).
    `absolute=False` skips the generic absolute-path test, `relative=False` the relative one (Triton's
    own pinned files skip both), `home=False` the home-path test (the sandbox's files skip all three).
    The test is lexical: a path computed at run time is not seen."""
    out = set()
    norm = unicodedata.normalize("NFKC", text)
    if any(_piece_hit(x) for x in _pieces(norm) | _pieces(_unescape(norm))):
        out.add("the name of another project")
    root = os.path.realpath(ROOT)
    for t in _path_views(text):
        for m in _FPV.finditer(t):
            rest = m.group(1).replace("\\", "/").rstrip(".:")
            comps = [x for x in posixpath.normpath(rest).split("/") if x and x != "."] if rest else []
            if comps and comps[0] != "tva":
                out.add(f"a path outside TVA (a sibling folder, fpv/{rest[:50]})")
        toks = {m.group(0) for m in _HOME_START.finditer(t)} if home else set()
        if absolute:
            toks |= {m.group(0) for m in _ABS.finditer(t)}
        for raw in sorted(toks):
            tok = raw.rstrip(".:")
            if tok.startswith("~"):
                tok = tok.replace("\\", "/")
                rest = tok.split("/", 1)[1] if "/" in tok else ""
                ok = _home_ok([x for x in posixpath.normpath("/" + rest).split("/") if x])
            else:
                parts = [x for x in posixpath.normpath(tok).split("/") if x]
                if parts and parts[0] == "home":
                    ok = len(parts) <= 2 or _home_ok(parts[2:])
                else:
                    ok = len(parts) < 2 or parts[0] in SYSTEM_DIRS or parts[0] in PLACEHOLDER_ROOTS
            if not ok:
                out.add(f"a path outside TVA ({tok[:60]})")
        if where is not None and relative:
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
    link must stay inside TVA. Triton's own pinned files skip the generic absolute-path and relative-path
    tests (`upstream_files`), and the sandbox's files skip those and the home test too, since they name
    the paths of the containers they run in (`SANDBOX`, library issue L135); the name and `fpv` tests
    apply to every file, and a kernel's source gets every test (library issue L7)."""
    bad = []
    root = os.path.realpath(ROOT)
    upstream = upstream_files()
    sandbox = os.path.realpath(SANDBOX) + os.sep
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
        real = os.path.realpath(p)
        own = real not in upstream and not real.startswith(sandbox)
        for r in foreign_refs(text, os.path.dirname(real), absolute=own, relative=own,
                              home=not real.startswith(sandbox)):
            bad.append(f"{rel(p)}: {r}")
    return bad


# Cited paths (library issue L70): a library page, a Lean file or a kernel's write-up that cites a file of TVA
# (a path from TVA's root, `docs/…` or `evidence/…`, or a Markdown link) must cite one that exists. Records
# (`docs/library-review/`, a kernel's `review/`) are history: they cite files as they were, and are frozen.
# Evidence folders hold test inputs (probe paths in a scratch copy), not citations. Triton's pinned files
# cite Triton's own tree. A path that a placeholder ends (`round-<N>/`, `{kernel}`, `*.md`, `…`, `...`, or a
# lower-case word with an upper-case one after its last hyphen, such as `round-N`) is checked up to the folder
# before it. A component in capitals only (`docs/DESIGN`) is a name, and checked (library issue L82).
CITE_SKIP = ("docs/library-review/", "evidence/", "primitives/", "tutorials/")
_CITE = re.compile(r"(?<![\w./\\-])((?:docs|evidence)/[^\s`'\"()\[\]{}<>,;|*…]*)")
_MD_LINK = re.compile(r"\]\(([^)\s]+)\)")
_CODE_SPAN = re.compile(r"`[^`\n]*`")
_PLACEHOLDER = re.compile(r"[a-z0-9_.]+(?:-[a-z0-9_.]+)*-[A-Z]+|.*\.\.\.|")


def git_checkout():
    """Whether TVA's root is a git checkout. A release copy is not: it has no git metadata and holds only
    what ships (the release check's `shipped` step made it so)."""
    return os.path.exists(os.path.join(ROOT, ".git"))


def shipped_set():
    """What ships, by path from TVA's root with `/`, folders included (every folder above a shipped
    file), in a git checkout (`shipped_files`); `None` outside one (library issues L186, L187)."""
    if not git_checkout():
        return None
    out = set()
    for f in shipped_files():
        parts = f.rstrip("/").split("/")
        out.update("/".join(parts[:n]) for n in range(1, len(parts) + 1))
    return out


def cited_files(shipped=None):
    """The files whose citations `citation_problems` checks: every Markdown and Lean file of the project but
    the records, the evidence and Triton's pinned files (`CITE_SKIP`); in a git checkout only files TVA
    ships (`shipped_set`), so a folder git ignores, such as a local `kernels/`, is not read; outside one
    (a release copy, which holds only what ships; library issue L186) every such file."""
    out = []
    shipped = shipped_set() if shipped is None else shipped
    for p in text_files(ROOT):
        r = rel(p).replace(os.sep, "/")
        if shipped is not None and r not in shipped:
            continue
        if p.endswith((".md", ".lean")) and not r.startswith(CITE_SKIP) and "/review/" not in r and not os.path.islink(p):
            out.append(p)
    return out


def citation_problems(paths=None):
    """The citations of files of TVA that do not exist (library issue L70): in each file of `paths`
    (`cited_files` by default), every path from TVA's root that starts `docs/` or `evidence/`, and in a
    Markdown file every link to a relative path (code spans aside), up to a placeholder; a line number
    (`:12`), an anchor (`#…`) and a closing punctuation mark are not part of the path. In a git checkout
    a cited file or folder of TVA must also ship (`shipped_set`; library issue L187): a git-ignored
    `kernels/` exists locally but not in what ships."""
    bad = []
    root = os.path.realpath(ROOT)
    shipped = shipped_set()
    for p in paths if paths is not None else cited_files(shipped):
        try:
            text = open(p, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        cites = []
        for m in _CITE.finditer(text):
            tok, nxt = m.group(1), text[m.end():m.end() + 1]
            parts = tok.split("/")
            if nxt in "<{*…" and nxt:
                parts = parts[:-1]  # the last part goes on in a placeholder
            keep = []
            for x in parts:
                if _PLACEHOLDER.fullmatch(re.sub(r"[.:,]+$", "", x)) and keep:
                    break
                keep.append(x)
            tok = re.sub(r"#.*$", "", "/".join(keep))
            tok = re.sub(r"(?::\d+)+(?:-\d+)?$", "", tok).rstrip(".:,")
            cites.append((tok, os.path.join(root, tok)))
        if p.endswith(".md"):
            for m in _MD_LINK.finditer(_CODE_SPAN.sub(lambda c: " " * len(c.group(0)), text)):
                tgt = m.group(1)
                if re.match(r"[A-Za-z][A-Za-z0-9+.-]*:", tgt) or tgt.startswith(("#", "<", "{", "…")):
                    continue
                tgt = re.sub(r"[#?].*$", "", tgt)
                if tgt:
                    cites.append((tgt, os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(p)), tgt))))
        for tok, dest in cites:
            if not tok or not (dest == root or dest.startswith(root + os.sep)):
                continue
            if not os.path.exists(dest):
                bad.append(f"{rel(p)}: cites {tok}, which does not exist")
            elif shipped is not None and dest != root and \
                    os.path.relpath(dest, root).replace(os.sep, "/") not in shipped:
                bad.append(f"{rel(p)}: cites {tok}, which TVA does not ship (git ignores it)")
    return sorted(set(bad))


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


_CHAR_LIT = re.compile(r"'(?:\\(?:x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|.)|[^\\'\n])'")
_RAW_OPEN = re.compile(r"r(#*)\"")


def _uncommented(text):
    """Lean text with its comments and docstrings (nested `/- -/`, `--`) replaced by spaces, line breaks
    kept. String literals (raw ones included), character literals and `«…»` names are read as Lean's
    lexer reads them and kept, so a `--` or `/-` inside one starts no comment (library issue L40).

    A string literal is read as a plain one. Lean reads `"…"` as an interpolated string (`s!"…"`,
    `m!"…"`, `throwError "…"`, …), whose `{…}` is code with literals of its own, or as a plain one,
    depending on the syntax around it, so the two readings differ only for a string that holds `{`. The
    hygiene refuses such a string (`brace_string_problems`), so wherever it passes, this reading is
    Lean's."""
    return _lean_scan(text)[0]


def brace_string_problems(text):
    """The lines of the string literals of a Lean text (read as `_uncommented` reads them) that hold `{`
    (library issue L40). Lean reads such a literal as a plain string or as an interpolated one, whose
    `{…}` is code with string literals and comments of its own, depending on the syntax before it: in
    `s!"{"--"}" #eval 7` the `#eval` is a command, while a plain reading takes it for a comment. A
    literal without `{` reads the same both ways, so refusing these makes the comment reader exact."""
    return sorted({text.count("\n", 0, a) + 1 for a, b in _lean_scan(text)[1] if "{" in text[a:b]})


def _lean_scan(text):
    """`(code, strings)`: the text with comments blanked (`_uncommented`), and the spans `(start, end)` of
    its string literals, raw ones included."""
    out, i, n = [], 0, len(text)
    strings = []

    def blank(a, b):
        out.append("".join("\n" if c == "\n" else " " for c in text[a:b]))

    while i < n:
        c = text[i]
        prev = text[i - 1] if i else " "
        ident_before = prev.isalnum() or prev in "_'!?."
        if text.startswith("--", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif text.startswith("/-", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith("/-", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("-/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            blank(i, j)
            i = j
        elif c == "r" and not ident_before and _RAW_OPEN.match(text, i):
            close = '"' + _RAW_OPEN.match(text, i).group(1)
            j = text.find(close, _RAW_OPEN.match(text, i).end())
            j = n if j < 0 else j + len(close)
            out.append(text[i:j])
            strings.append((i, j))
            i = j
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            strings.append((i, min(j + 1, n)))
            i = j + 1
        elif c == "«":
            j = text.find("»", i)
            j = n if j < 0 else j + 1
            out.append(text[i:j])
            i = j
        elif c == "'" and not ident_before and _CHAR_LIT.match(text, i):
            m = _CHAR_LIT.match(text, i)
            out.append(m.group(0))
            i = m.end()
        else:
            out.append(c)
            i += 1
    return "".join(out), strings


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
    """The lake configuration: Mathlib is the only requirement, and the only library is TVA's own.
    `kernels/` is local and does not ship (library issue L187), so the shipped `lean/lakefile.toml`
    declares no kernel's library: `tvr` builds a kernel with a configuration of its own,
    `lean/.lake/tvr/lakefile.toml` (`kernel_lakefile`), written from this one."""
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
    # package-wide options apply to every module of the library and the kernels (library issue L40)
    opts = cfg.get("leanOptions", {})

    def flat(t, pre=""):  # `linter.x = false` under [leanOptions] is a nested table in TOML
        out = {}
        for key, val in t.items():
            out.update(flat(val, pre + key + ".") if isinstance(val, dict) else {pre + key: val})
        return out

    if not isinstance(opts, dict):
        bad.append("lean/lakefile.toml: leanOptions must be a table")
    else:
        for key, val in sorted(flat(opts).items()):
            if not (LEAN_OPTIONS.fullmatch(key) and isinstance(val, (bool, int))):
                bad.append(f"lean/lakefile.toml: leanOptions {key!r} is not allowed (only resource limits, "
                           "autoImplicit, relaxedAutoImplicit, linter.* and pp.*, with a boolean or number value)")
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
    for lib in libs:
        if not isinstance(lib, dict):
            bad.append("lean/lakefile.toml: a lean_lib that is not a table")
            continue
        name = lib.get("name")
        if name == "TVA":
            if set(lib) != {"name"}:
                bad.append("lean/lakefile.toml: the TVA lean_lib takes only its name (its sources are lean/TVA)")
            continue
        bad.append(f"lean/lakefile.toml: lean_lib {str(name)[:60]!r} is not TVA: the shipped configuration declares "
                   "TVA alone, since kernels/ does not ship (library issue L187); tvr builds a kernel with "
                   "lean/.lake/tvr/lakefile.toml")
    for t in cfg.get("defaultTargets", []):
        if t not in names:
            bad.append(f"lean/lakefile.toml: default target {t!r} is not one of the libraries")
    man = load_json(os.path.join(lean, "lake-manifest.json"), {"packages": []})
    # where Lake looks for the packages (library issue L7): its own folder, which the release check makes
    for key, want in (("packagesDir", ".lake/packages"), ("lakeDir", ".lake")):
        if man.get(key, want) != want:
            bad.append(f"lean/lake-manifest.json: {key} must be {want!r} (found {str(man.get(key))[:60]!r})")
    req_rev = reqs[0].get("rev", reqs[0].get("version")) if isinstance(reqs, list) and reqs and isinstance(reqs[0], dict) else None
    for p in man.get("packages", []):
        if not p.get("inherited") and (p.get("name") != "mathlib" or p.get("type") != "git" or
                                       p.get("url") != "https://github.com/leanprover-community/mathlib4"):
            bad.append(f"lean/lake-manifest.json: direct package {p.get('name')} (only Mathlib's repository is allowed)")
        elif not p.get("inherited") and p.get("inputRev") != req_rev:
            bad.append(f"lean/lake-manifest.json: Mathlib's inputRev {p.get('inputRev')!r} is not the lakefile's rev {req_rev!r}")
        elif p.get("type") != "git" or p.get("subDir") is not None:
            bad.append(f"lean/lake-manifest.json: package {p.get('name')} is not a git package at its repository's root")
    # the toolchain (library issue L54): an official release of Lean, which the release check shows read-only
    try:
        tc = open(os.path.join(lean, "lean-toolchain"), encoding="utf-8").read()
    except (OSError, UnicodeDecodeError):
        tc = None
    if tc is None or not TOOLCHAIN.fullmatch(tc.rstrip("\n")) or tc.count("\n") > 1:
        bad.append("lean/lean-toolchain must be one line `leanprover/lean4:v<major>.<minor>.<patch>` (a release of "
                   f"Lean, optionally `-rc<n>`; found {str(tc)[:60]!r})")
    # Mathlib's revision (library issue L7): the release tag of the toolchain's Lean, or master
    elif req_rev not in mathlib_ref(tc):
        bad.append(f"lean/lakefile.toml: Mathlib's rev {str(req_rev)[:60]!r} must be the release tag of the toolchain's Lean "
                   f"or master ({', '.join(sorted(mathlib_ref(tc)))}; another branch is not reviewed Mathlib)")
    return bad


def library_lean_files():
    """The Lean files of the library: every `.lean` file under lean/ outside the build directory."""
    return sorted(f for f in text_files(os.path.join(ROOT, "lean")) if f.endswith(".lean"))


def library_module_problems():
    """Every Lean file of the library is a module that `lean/TVA.lean` reaches through its imports
    (library issue L40): `lake build TVA` builds, and `leanchecker TVA` replays, only those, so a module
    outside that closure would be built only as some kernel's import and replayed never."""
    lean_dir = os.path.join(ROOT, "lean")
    files = {}
    for f in library_lean_files():
        mod = os.path.relpath(f, lean_dir)[:-5].replace(os.sep, ".")
        files[mod] = f
    seen, todo = set(), ["TVA"] if "TVA" in files else []
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        try:
            imports, _ = lean_header(open(files[m], encoding="utf-8").read())
        except (UnicodeDecodeError, OSError):
            continue
        todo += [x for x in imports if x in files and x not in seen]
    return [f"{rel(f)}: a library file that lean/TVA.lean does not reach through its imports, so `lake build TVA` "
            "does not build it and `leanchecker TVA` does not replay it" for m, f in sorted(files.items()) if m not in seen]


def library_independence():
    """The library's own independence: its imports, the lake configuration, and the text of every file
    of the project outside the kernel folders."""
    bad = import_violations(library_lean_files(), LEAN_ROOTS)
    bad += lake_violations()
    bad += scan_files(text_files(ROOT, skip=[KROOT]))
    return bad


def kernel_independence(k):
    """One kernel folder: its Lean imports (TVA, Mathlib and its own module) and the text of every file."""
    own = str((manifest(k).get("lean") or {}).get("module", "")).split(".")[0]
    bad = import_violations(kernel_lean_files(k), LEAN_ROOTS + ((own,) if own else ()))
    bad += scan_files(text_files(kpath(k)))
    return bad


def kernels():
    if not os.path.isdir(KROOT):
        return []
    return sorted(d for d in os.listdir(KROOT) if os.path.exists(kpath(d, "kernel.json")))


# ---------------------------------------------------------------- what an item's digest covers

# The library files a kernel item's meaning rests on (protocol §4, library issue L36). Digests are
# bookkeeping of what changed, and they are complete by construction: an item's digest holds the whole
# text of every library file its cone reaches, not a curated list of docstrings.
# - the Lean file of every library module that holds a constant of the cone (its definitions,
#   statements, docstrings, module docstrings and comments, all of it);
# - the general fact pages, which say what every library construct means and what is out of scope,
#   for an item whose cone reaches the library at all;
# - the rule page of each `TVA.Rules` module the cone reaches (`tvr audit` checks that every module of
#   lean/TVA/Rules has one and that every page of docs/rules belongs to one);
# - the versions of the packages Lean resolves (Mathlib's, from lean/lake-manifest.json).
# docs/TRANSCRIPTION.md is guidance (worked patterns and pitfalls); it states no fact of its own, so it
# is in no digest, and a change to it re-opens nothing.
GENERAL_PAGES = ("docs/SEMANTICS.md", "docs/exclusions.md")
RULE_PAGES = {"TVA.Rules.Num": "docs/rules/num.md", "TVA.Rules.Compiler": "docs/rules/compiler-faithful.md"}


def module_file(mod):
    """The source file of a library module, relative to the TVA root."""
    return "lean/" + mod.replace(".", "/") + ".lean"


def _file_digest(path):
    full = os.path.join(ROOT, path)
    return file_sha(full) if os.path.isfile(full) else "missing"


def packages_digest():
    """The packages Lean resolves for TVA, by name, URL and revision (lean/lake-manifest.json)."""
    man = load_json(os.path.join(ROOT, "lean", "lake-manifest.json"), {"packages": []})
    pk = sorted((p.get("name", ""), p.get("url", ""), p.get("rev", "")) for p in man.get("packages", []))
    return sha(json.dumps(pk))


def library_files(cone, consts):
    """The library files an item's cone reaches, as `{"file:<path>": sha256}` (see above), and the
    packages' versions as `{"packages": …}`."""
    mods = sorted({consts[n]["module"] for n in cone
                   if n in consts and n.startswith("TVA.") and consts[n].get("module", "").startswith("TVA")})
    paths = [module_file(m) for m in mods]
    if mods:
        paths += list(GENERAL_PAGES)
    # a rule's page follows its module and its name (library issue L55): the gate's `kinds` check also
    # refuses a `TVA.Rules.*` constant declared outside lean/TVA/Rules
    paths += sorted({rule_page(m) for m in mods if rule_page(m)} |
                    {rule_page(n) for n in cone if n.startswith("TVA.Rules.") and rule_page(n)})
    out = {f"file:{p}": _file_digest(p) for p in paths}
    out["packages"] = packages_digest()
    return out


def rule_page(mod):
    """The rule page of a `TVA.Rules` module: its own entry of `RULE_PAGES`, or else that of the nearest
    module above it (`TVA.Rules.Num.Exp2` has `TVA.Rules.Num`'s page; library issue L43). Given a
    constant's name (`TVA.Rules.Num.exp2_f32`), the page of the module its name lies under (L55)."""
    parts = mod.split(".")
    for n in range(len(parts), 2, -1):
        if ".".join(parts[:n]) in RULE_PAGES:
            return RULE_PAGES[".".join(parts[:n])]
    return None


def rules_modules():
    """Every module of lean/TVA/Rules, at any depth."""
    base = os.path.join(ROOT, "lean", "TVA", "Rules")
    out = set()
    for dirpath, dirnames, names in os.walk(base):
        dirnames[:] = [n for n in dirnames if n not in SKIP_DIRS]
        out |= {"TVA.Rules." + os.path.relpath(os.path.join(dirpath, f[:-5]), base).replace(os.sep, ".")
                for f in names if f.endswith(".lean")}
    return out


def fact_page_problems():
    """Every module of lean/TVA/Rules, at any depth, has a rule page (its own entry of `RULE_PAGES`, or an
    entry of a module above it); every entry names a module or a folder of modules there and an
    existing page; every page of docs/rules/ is some entry's; every general fact page exists; and
    docs/rules/num.md lists the facts a kernel may cite (`stated_facts`, library issue L62), without which
    the gate accepts no fact but `Compiler.Faithful`, in the one table of its section (library issue L95: the
    gate reads only the first, so a second one, in a `###` subsection too, is named)."""
    mods = rules_modules()
    listed = [] if stated_facts() else [
        f"{FACTS_PAGE}: no section \"{FACTS_SECTION}\" with a table whose first column names the facts a kernel may "
        "cite, before any heading of the section, so the gate's kinds check accepts no fact but Compiler.Faithful "
        "(library issue L62)"]
    tables = facts_section_tables() or []
    if len(tables) > 1:
        listed.append(f"{FACTS_PAGE}: the section \"{FACTS_SECTION}\" holds {len(tables)} tables; the gate reads only "
                      "the first as the facts a kernel may cite, so keep one table there and put any other in another "
                      "section (library issue L95)")
    got = facts_section()
    listed += [f"{FACTS_PAGE}: the section \"{FACTS_SECTION}\": {p}; the gate reads the section line by line, so it "
               "accepts no fact until the section holds only blank lines, table rows, ### headings and text (library "
               "issue L95)" for p in (got[1] if got else [])]
    pages = {rel(os.path.join(d, f)) for d, _, names in os.walk(os.path.join(ROOT, "docs", "rules"))
             for f in names if f.endswith(".md")}
    prefixes = {".".join(m.split(".")[:n]) for m in mods for n in range(3, len(m.split(".")) + 1)}
    return ([f"tools/tvr.py: RULE_PAGES gives no page for the module {m} or any module above it"
             for m in sorted(m for m in mods if not rule_page(m))] +
            [f"tools/tvr.py: RULE_PAGES names {m}, which is neither a module nor a folder of lean/TVA/Rules"
             for m in sorted(set(RULE_PAGES) - prefixes)] +
            [f"tools/tvr.py: the page {p} is the rule page of no module in RULE_PAGES" for p in sorted(pages - set(RULE_PAGES.values()))] +
            [f"tools/tvr.py: RULE_PAGES names {p}, which does not exist" for p in sorted(set(RULE_PAGES.values()) - pages)] +
            [f"tools/tvr.py: the fact page {p} does not exist" for p in GENERAL_PAGES if not os.path.isfile(os.path.join(ROOT, p))] +
            listed)


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
    # library issue L20: what a name in the model means is fixed by the library; a kernel may not
    # give a library name or a model line another meaning (the reviewer reads the whole Lean too)
    ("a notation or alias that changes what a name means",
     r"(?<![\w.])(?:notation\d?|infix[lr]?|prefix|postfix|binder_predicate|unif_hint)\b|\brenaming\b|^\s*export\b"),
    ("an instance outside the claim reader's view, or one erased",
     r"\b(?:local|scoped)\s+(?:instance|attribute|notation)\b|"
     r"(?:@\[|\battribute\s*\[)[^\]]*(?:\b(?:instance|default_instance|coe)\b|(?:^|[\[,\s])-\s*[\w«])"),
    ("an option other than resource limits, printing, tracing and linters",
     r"\bset_option\s+(?!(?:maxHeartbeats|maxRecDepth|synthInstance\.maxHeartbeats|synthInstance\.maxSize|"
     r"autoImplicit|relaxedAutoImplicit|linter\.[\w.]+|pp\.[\w.]+|trace\.[\w.]+)\s)[\w.]+"),
]


# Lean's keywords that can follow the names of an `open` command (the next command, or a modifier of
# this one). A keyword missing here is read as one more opened name, which only adds to the names the
# shadowing check compares with.
_LEAN_KEYWORDS = {
    "open", "in", "hiding", "renaming", "def", "theorem", "lemma", "abbrev", "instance", "structure", "class",
    "inductive", "coinductive", "namespace", "section", "end", "export", "variable", "universe", "example",
    "axiom", "noncomputable", "private", "protected", "partial", "nonrec", "unsafe", "opaque", "mutual", "where",
    "deriving", "attribute", "set_option", "local", "macro", "macro_rules", "syntax", "elab", "elab_rules",
    "notation", "infix", "infixl", "infixr", "prefix", "postfix", "omit", "include", "import", "module",
    "prelude", "public", "meta", "termination_by", "decreasing_by", "at", "fun", "let", "have", "show", "from",
    "by", "do", "then", "else", "if", "match", "with", "calc", "irreducible_def", "alias", "initialize",
    "add_decl_doc", "unseal", "seal", "run_cmd", "declare_syntax_cat", "universes", "variables"}
_TOKEN = re.compile(_IDENT + r"|\S")


def opened_names(k):
    """The names the kernel's Lean files `open` (library issue L20): the claim reader treats every
    namespace whose name ends with one of them as opened, for the shadowing check. An `open` command
    is read over as many lines as it spans, comments removed: its names are the identifiers after
    `open` (and `scoped`) up to the first token that is not an identifier, or is a keyword."""
    out = set()
    for p in kernel_lean_files(k):
        toks = _TOKEN.findall(_uncommented(open(p, encoding="utf-8", errors="replace").read()))
        i = 0
        while i < len(toks):
            if toks[i] != "open":
                i += 1
                continue
            i += 1
            while i < len(toks) and (toks[i] == "scoped" or
                                     (toks[i] not in _LEAN_KEYWORDS and re.fullmatch(_IDENT, toks[i]))):
                if toks[i] != "scoped":
                    out.add(toks[i].rstrip("."))
                i += 1
    return out


BRACE_STRING = ("a string literal holding `{`, which Lean reads as a plain or an interpolated string (whose `{…}` "
                "is code) depending on the syntax around it, so its comments cannot be told apart (library issue L40)")


def hygiene_problems(k):
    bad = []
    for p in kernel_lean_files(k):
        text = open(p, encoding="utf-8", errors="replace", newline="").read()
        for what, pat in HYGIENE:
            for m in re.finditer(pat, text, re.M):
                bad.append(f"{rel(p, kpath(k))}:{text.count(chr(10), 0, m.start()) + 1}: {what} ({m.group(0).strip()[:24]})")
        bad += [f"{rel(p, kpath(k))}:{line}: {BRACE_STRING}" for line in brace_string_problems(text)]
        for m in LINE_SEPARATORS.finditer(text):
            bad.append(f"{rel(p, kpath(k))}:{text.count(chr(10), 0, m.start()) + 1}: a line break other than a line feed "
                       f"({m.group(0)!r}), which readers count differently (library issue L51)")
    return bad


# The library's own hygiene (library issue L40): the same proof escapes, kernel bypasses, meta code, meta
# types, handler attributes and options as a kernel's, matched on the code of every library Lean file
# (comments and docstrings removed, since the library's prose names these things). The library may
# declare notation and instances: it is where names get their meaning. `leanchecker TVA`, in the
# release check, re-checks every library declaration whatever this list misses.
LIBRARY_HYGIENE = [(what, pat) for what, pat in HYGIENE if what not in (
    "a notation or alias that changes what a name means", "an instance outside the claim reader's view, or one erased")]


def library_hygiene_problems():
    bad = []
    for p in library_lean_files():
        raw = open(p, encoding="utf-8", errors="replace").read()
        text = _uncommented(raw)
        for what, pat in LIBRARY_HYGIENE:
            for m in re.finditer(pat, text, re.M):
                bad.append(f"{rel(p)}:{text.count(chr(10), 0, m.start()) + 1}: {what} ({m.group(0).strip()[:24]})")
        bad += [f"{rel(p)}:{line}: {BRACE_STRING}" for line in brace_string_problems(raw)]
    return bad


def leanchecker(root_mod, what):
    """Re-check every declaration of the modules under `root_mod` with Lean's kernel, from the compiled
    files: `leanchecker <root>` replays each module's constants into the environment of its imports. A
    declaration that meta code added without checking fails here."""
    lean_dir = os.path.join(ROOT, "lean")
    pre = subprocess.run([LAKE, "env", "lean", "--print-prefix"], cwd=lean_dir, capture_output=True, text=True)
    prefix = pre.stdout.strip()
    exe = os.path.join(prefix, "bin", "leanchecker")
    if pre.returncode != 0 or not os.path.exists(exe):
        return False, "leanchecker not found in the Lean toolchain"
    env = dict(os.environ, LEAN_SYSROOT=prefix, PATH=os.path.join(prefix, "bin") + os.pathsep + os.environ.get("PATH", ""))
    r = subprocess.run([LAKE, "env", exe, root_mod], cwd=lean_dir, capture_output=True, text=True, env=env)
    if r.returncode == 0:
        return True, f"leanchecker {root_mod}: every declaration of {what} re-checked by Lean's kernel"
    msg = " | ".join(l.strip() for l in (r.stderr + r.stdout).splitlines() if l.strip())[:300]
    return False, f"leanchecker {root_mod} rejects {what}, as compiled: {msg}"


def replay(k):
    """The gate's `replay`: `leanchecker` on the kernel's modules."""
    return leanchecker(manifest(k)["lean"]["module"].split(".")[0], "the kernel's modules")


def cmd_library_check():
    """`tvr library-check` (library issue L40), run by the release check after `lake build TVA`: the
    library's hygiene (`library_hygiene_problems`), that `lean/TVA.lean` reaches every library module
    through its imports (`library_module_problems`), and `leanchecker TVA`, which re-checks every
    declaration of every module it reaches with Lean's kernel."""
    bad = library_hygiene_problems() + library_module_problems()
    if bad:
        print("library hygiene fails:\n  - " + "\n  - ".join(bad[:20]))
        return 1
    ok, detail = leanchecker("TVA", "the library's modules")
    print(("library hygiene passes; " if ok else "") + detail)
    return 0 if ok else 1


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
    """Fold the kernel's ledger."""
    return fold(ledger(k))


def fold(events):
    """Fold a ledger's events. Events written before rounds were recorded belong to round 1."""
    S = {"reviewers": {}, "findings": {}, "accepts": {}, "filed": {}, "responses": 0,
         "archive": {}, "events": events}
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


def normal_path(p):
    """Whether `p` is a path from TVA's root as the tool writes it: relative, `/`-separated, with no `.`, `..` or
    empty component and no backslash (library issue L108)."""
    return (isinstance(p, str) and bool(p) and "\\" not in p and not p.startswith("/") and
            all(c not in ("", ".", "..") for c in p.split("/")))


def ledger_problems(root, path, label, folders=True):
    """What a ledger (`path`, relative to `root`) does not support (library issue L52): its events are
    numbered 1, 2, … in order; every file an event records exists with the sha256 it was recorded with,
    and no file is recorded with two sha256s (an appended or edited line cannot re-freeze a changed record:
    `tvr` itself never records a file again with another sha256); and, with `folders` (the library's ledger),
    every file in a folder an event records (`dir`) is recorded by some event. A kernel's ledger passes
    `folders=False`: `kernel_folder_problems` holds each of its role folders to that folder's own events, which is
    stricter (library issue L81). Recorded records are frozen: a file changed, removed or added after recording
    is named here. A ledger shows that each recorded file is unchanged since its line, not that its lines are
    unchanged (protocol §3, §10)."""
    full = os.path.join(root, path)
    if not os.path.exists(full):
        return []
    bad, recorded, dirs = [], {}, set()
    try:
        events = [json.loads(l) for l in open(full, encoding="utf-8") if l.strip()]
    except ValueError:
        return [f"{path}: not one JSON object per line"]
    for i, e in enumerate(events, 1):
        if not isinstance(e, dict) or e.get("seq") != i:
            bad.append(f"{path}: event {i} is not numbered {i}")
            continue
        named = list(e.get("files", {}) if isinstance(e.get("files"), dict) else {})
        named += [e["dir"]] if isinstance(e.get("dir"), str) and e["dir"] else []
        # The tool writes every recorded path relative to TVA's root and normalised (library issue L108): a
        # path such as `<folder>/work/../filing-2.json` names a file other than the text a check compares.
        for p in named:
            if not normal_path(p):
                bad.append(f"{path}: event {i} records the path {p!r}, which is not a normalised path from TVA's "
                           "root as the tool writes it (no `.`, `..`, empty component, leading `/` or backslash)")
        for p, h in (e.get("files", {}) if isinstance(e.get("files"), dict) else {}).items():
            recorded.setdefault(p, set()).add(h)
        if folders and isinstance(e.get("dir"), str) and e["dir"]:
            dirs.add(e["dir"])
    for p, hs in sorted(recorded.items()):
        f = os.path.join(root, p)
        if len(hs) > 1:
            bad.append(f"{p}: recorded in {label} with {len(hs)} different sha256s, which the tool never writes "
                       "(a record is frozen once recorded)")
        elif not os.path.isfile(f):
            bad.append(f"{p}: recorded in {label}, and missing")
        elif file_sha(f) not in hs:
            bad.append(f"{p}: changed after it was recorded in {label}")
    for d in sorted(dirs):
        for f in text_files(os.path.join(root, d)):
            if os.path.relpath(f, root) not in recorded:
                bad.append(f"{os.path.relpath(f, root)}: in a folder {label} records, but recorded by no event")
    return bad


# The events a kernel's ledger holds that follow from a recorded file (library issue L52): a `file` event
# is followed by exactly the events its filing gives (`filing_events`), a `response` event by those its
# response gives (`response_events`). `assign` and `note` are the coordinator's, `decide` the user's (on an
# escalated finding); no file stands behind them.
FILE_EVENTS = {"raise", "release", "reraise", "escalate", "accept", "respond"}
OWN_EVENTS = {"assign", "note", "decide", "freeze"}


def filing_events(S, who, N, filing, gate_digest, gate_items):
    """The events a reviewer's filing records after its `file` event, given the ledger's state `S` before
    it (`fold`) and the gate it was filed at: a `raise` per finding (numbered on from the findings so
    far), a `release` and a `reraise` per entry (with an `escalate` after a finding's last allowed
    re-raise), and an `accept` of the items it names at the gate's digests. `tvr file` writes these, and
    the release check derives them again from the recorded filing (library issue L52), with the tool's
    current `KINDS` defaults, `ESCALATE_AFTER` and escalation memo (library issue L82: changing one of them
    makes older ledgers fail)."""
    findings, events = S["findings"], []
    nxt = 1 + max([int(i[1:]) for i in findings if re.fullmatch(r"F\d+", i)] or [0])
    for f in filing.get("findings", []):
        events.append({"type": "raise", "actor": who, "round": N, "id": f"F{nxt}", "kind": f.get("kind"),
                       "target": f.get("target"), "affects": f.get("affects", []),
                       "blocking": f.get("blocking", KINDS.get(f.get("kind"), (False,))[0]), "summary": f.get("summary"),
                       "detail": f.get("detail", ""), "evidence": f.get("evidence", ""),
                       "gate": gate_digest, "items": gate_items})
        nxt += 1
    for r in filing.get("release", []):
        events.append({"type": "release", "actor": who, "round": N, "id": r.get("id"), "note": r.get("note", "")})
    for r in filing.get("reraise", []):
        events.append({"type": "reraise", "actor": who, "round": N, "id": r.get("id"), "note": r.get("note", ""),
                       "items": gate_items})
        f = findings.get(r.get("id"))
        if f and f["reraises"] + 1 >= ESCALATE_AFTER:
            events.append({"type": "escalate", "actor": "tvr", "round": N, "id": r.get("id"),
                           "memo": f"re-raised {f['reraises'] + 1} times; the user decides"})
    acc = filing.get("accept", [])
    if acc:
        names = list(gate_items) if acc == "all" else [a[5:] if a.startswith("item:") else a for a in acc]
        events.append({"type": "accept", "actor": who, "round": N,
                       "items": {n: gate_items[n] for n in names if n in gate_items}, "gate": gate_digest})
    return events


def response_events(N, resp, gate_digest):
    """The events the prover's response records after its `response` event: one `respond` per response."""
    return [{"type": "respond", "actor": "prover", "round": N, "id": r.get("id"), "response": r.get("response"),
             "text": r.get("text"), "changed": r.get("changed", []), "issue": r.get("issue", ""),
             "gate": gate_digest} for r in resp.get("responses", [])]


def _bare(e):
    return {x: v for x, v in e.items() if x not in ("seq", "time")}


def kernel_event_problems(root, k):
    """The events of a kernel's ledger (under `root`) that no recorded file supports (library issue L52).
    Each `file` event must be a filing of a reviewer assigned before it, at the round and path the tool
    gives it (`round-N/<reviewer>/filing.json`, `filing-2.json`, …), record its filing (and its report),
    and be followed by exactly the events its filing gives at its gate, whose item digests the gate's
    snapshot holds (`filing_events`); each `response` event the same for the prover's response
    (`response_events`); every other event of `FILE_EVENTS` must be one of these; a `decide` must be on an
    escalated finding; a `freeze` must be one `tvr freeze` writes (`freeze_problem`); and only a `file`, a
    `response` (its own folder) and a `freeze` may name a folder as `dir` (library issue L81). So an `accept` (or
    any other event) written by hand fails, unless it is what a recorded filing says."""
    path = f"kernels/{k}/review/ledger.jsonl"
    full = os.path.join(root, path)
    if not os.path.exists(full):
        return []
    try:
        events = [json.loads(l) for l in open(full, encoding="utf-8") if l.strip()]
    except ValueError:
        return [f"{path}: not one JSON object per line"]
    bad, i = [], 0

    def load(p):
        return _read_json(os.path.join(root, p)) if isinstance(p, str) and p else None

    while i < len(events):
        e = events[i]
        if not isinstance(e, dict):
            bad.append(f"{path}: event {i + 1} is not an object")
            i += 1
            continue
        t, rnd = e.get("type"), e.get("round", 1)
        try:
            S = fold(events[:i])
        except (KeyError, TypeError, AttributeError):
            return bad + [f"{path}: the events before event {i + 1} do not fold (one names a finding never raised)"]
        derived = None
        where = f"{path}: event {i + 1} ({t}"
        if t == "file":
            who = e.get("actor")
            count = S["filed"].get(who, {}).get(rnd, 0)
            want = f"kernels/{k}/review/round-{rnd}/{who}/" + ("filing.json" if count == 0 else f"filing-{count + 1}.json")
            files = e.get("files") if isinstance(e.get("files"), dict) else {}
            filing = load(e.get("path"))
            snap = _read_json(os.path.join(root, "kernels", k, "review", "snapshots", str(e.get("gate", ""))[:16] + ".json"))
            items = {it: v.get("digest") for it, v in (snap or {}).get("items", {}).items()} if snap else None
            if who not in S["reviewers"]:
                bad.append(f"{where} by {who}): {who} was not assigned before it")
            elif rnd != S["round"]:
                bad.append(f"{where} by {who}): round {rnd}, while the ledger is at round {S['round']}")
            elif e.get("path") != want:
                bad.append(f"{where} by {who}): its filing is {e.get('path')!r}, not the tool's {want!r}")
            elif e.get("path") not in files or (e.get("report") and e.get("report") not in files):
                bad.append(f"{where} by {who}): it does not record its filing and report")
            elif filing is None:
                bad.append(f"{where} by {who}): its filing {e.get('path')} is missing or not a JSON object")
            elif items is None or sha("\n".join(f"{x} {d}" for x, d in sorted(items.items()))) != e.get("gate"):
                bad.append(f"{where} by {who}): the gate it was filed at has no snapshot that hashes to it "
                           "(review/snapshots), so its acceptances cannot be checked")
            elif e.get("notes", "") != filing.get("notes", ""):
                bad.append(f"{where} by {who}): its notes are not its filing's")
            else:
                derived = filing_events(S, who, rnd, filing, e.get("gate"), items)
        elif t == "response":
            files = e.get("files") if isinstance(e.get("files"), dict) else {}
            want = f"kernels/{k}/review/round-{rnd}/prover/response.json"
            resp = load(e.get("path"))
            if rnd != S["round"] or e.get("path") != want:
                bad.append(f"{where}): its response is {e.get('path')!r} in round {rnd}, not the tool's {want!r} in round "
                           f"{S['round']}")
            elif e.get("path") not in files or (e.get("notes") and e.get("notes") not in files):
                bad.append(f"{where}): it does not record its response and notes")
            elif resp is None:
                bad.append(f"{where}): its response {e.get('path')} is missing or not a JSON object")
            else:
                derived = response_events(rnd, resp, e.get("gate"))
        elif t == "decide":
            if S["findings"].get(e.get("id"), {}).get("state") != "ESCALATED":
                bad.append(f"{where} {e.get('id')}): the user decides only an escalated finding")
        elif t == "freeze":
            why = freeze_problem(k, events[:i], e)
            if why:
                bad.append(f"{where} of {e.get('dir')}): {why}")
        elif t in FILE_EVENTS:  # the events a filing or response gives are skipped below, so this one is no such
            bad.append(f"{where}{' ' + str(e['id']) if e.get('id') else ''} by {e.get('actor')}): no recorded filing or "
                       "response gives it")
        elif t not in FILE_EVENTS | OWN_EVENTS | {"file", "response"}:
            bad.append(f"{where}): not an event the tool writes")
        # only a filing, a response and a freeze name a folder (`dir`), and a filing's or a response's is the folder of
        # its path (library issue L81): the role folder each of them records is that of its path
        if t in ("file", "response") and e.get("dir") not in (None, "") and e.get("dir") != os.path.dirname(str(e.get("path"))):
            bad.append(f"{where} by {e.get('actor')}): it names the folder {e.get('dir')!r}, not its "
                       f"{'filing' if t == 'file' else 'response'}'s")
        elif t not in ("file", "response", "freeze") and e.get("dir") not in (None, ""):
            bad.append(f"{where}): it names a folder ({e.get('dir')!r}), which only a filing, a response or a freeze does")
        if derived is not None:
            got = [_bare(x) if isinstance(x, dict) else x for x in events[i + 1:i + 1 + len(derived)]]
            if got != derived:
                n = next((j for j, (a, b) in enumerate(zip(got, derived)) if a != b), min(len(got), len(derived)))
                bad.append(f"{where} by {e.get('actor')}): the events after it are not what its recorded "
                           f"{'filing' if t == 'file' else 'response'} gives (event {i + 2 + n} differs)")
            else:
                i += len(derived)
        i += 1
    return bad


REVIEW_ENTRIES = {"ledger.jsonl", "certificate.json", "snapshots"}  # what `tvr` writes in review/ besides round-N/


def kernel_folders(root, k):
    """A kernel's role folders, `{(round, who): path from root}`, and the entries of its review/ folder that are
    neither one of them, nor `REVIEW_ENTRIES`, nor a `round-<N>` folder that holds them (`(folders, strays)`)."""
    base = os.path.join(root, "kernels", k, "review")
    folders, strays = {}, []
    for n in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        m = re.fullmatch(r"round-([1-9]\d*)", n)
        if not m or not os.path.isdir(os.path.join(base, n)) or os.path.islink(os.path.join(base, n)):
            if n not in REVIEW_ENTRIES:
                strays.append(f"kernels/{k}/review/{n}")
            continue
        for w in sorted(os.listdir(os.path.join(base, n))):
            p = os.path.join(base, n, w)
            if os.path.isdir(p) and not os.path.islink(p):
                folders[(int(m.group(1)), w)] = f"kernels/{k}/review/{n}/{w}"
            else:
                strays.append(f"kernels/{k}/review/{n}/{w}")
    return folders, strays


def role_folder(k, p):
    """The role folder `kernels/<k>/review/round-N/<who>` that the path `p` (from the TVA root) lies inside, or None."""
    m = re.fullmatch(r"(kernels/" + re.escape(k) + r"/review/round-[1-9]\d*/[^/]+)/.+", p) if isinstance(p, str) else None
    return m.group(1) if m else None


def own_records(k, events):
    """Each role folder of a kernel that its own events record, with the files they record in it (library issue
    L81), as `{folder: {"files": set of paths, "dir": bool, "frozen": bool}}`. A folder's own events are the
    `file` and `response` events whose recorded filing or response (`path`) lies in it, and then a `freeze` of it;
    never a `note`, nor another folder's event. A folder they record is frozen whether or not a line names it as
    `dir`, and each of its files must be one they record. `dir` says whether one of them names the folder (every
    one the tool writes now does; early ones did not); `frozen`, whether a freeze recorded files of it."""
    out = {}
    for e in events:
        if not isinstance(e, dict):
            continue
        files = e.get("files") if isinstance(e.get("files"), dict) else {}
        if e.get("type") in ("file", "response"):
            f = role_folder(k, e.get("path"))
            if f:
                o = out.setdefault(f, {"files": set(), "dir": False, "frozen": False})
                o["files"].update(p for p in files if role_folder(k, p) == f)
                o["dir"] = o["dir"] or bool(e.get("dir"))
        elif e.get("type") == "freeze" and e.get("dir") in out:
            out[e["dir"]]["files"].update(p for p in files if role_folder(k, p) == e["dir"])
            out[e["dir"]]["frozen"] = True
    return out


def is_work_file(folder, p):
    """Whether the path `p` lies in a subfolder of the role folder `folder` (`work/…`): not a filing, report,
    response or notes, which lie at the folder's top and are recorded only by their role's own event. The path
    must be normalised (`normal_path`; library issue L108), so that `<folder>/work/../filing-2.json` is no work
    file."""
    return normal_path(p) and p.startswith(folder + "/") and "/" in p[len(folder) + 1:]


def freeze_problem(k, before, e):
    """Why a `freeze` event is not one `tvr freeze` writes, given the events `before` it, or None (library issue
    L81): it is the coordinator's; its `dir` is a role folder that a filing or response before it records
    (`own_records`), every one of them from before the tool named folders (no `dir`), and frozen by no freeze
    before; and it records work files of that folder (`is_work_file`) that those events leave out, at least one."""
    d, files = e.get("dir"), e.get("files")
    own = own_records(k, before)
    if e.get("actor") != "coordinator":
        return "a freeze is the coordinator's (`tvr freeze`)"
    if not isinstance(d, str) or d not in own:
        return "it names no role folder that a filing or response before it records"
    if own[d]["dir"]:
        return "the folder's own events name it as a folder (`dir`), so they record every file it held"
    if own[d]["frozen"]:
        return "the folder was frozen before"
    if not isinstance(files, dict) or not files:
        return "it records no file"
    wrong = sorted(p for p in files if not is_work_file(d, p) or p in own[d]["files"])
    if wrong:
        return (f"it records {', '.join(wrong[:3])}, which {'is' if len(wrong) == 1 else 'are'} no work file of the folder "
                "that its own events leave out (a filing, report, response or notes is recorded only by its own event)")
    return None


def kernel_folder_problems(root, k, current=False):
    """The role folders of a kernel's review/ (under `root`) that its ledger does not account for (library issue
    L81). Every event must follow from a recorded file (`kernel_event_problems`), and conversely every filed
    folder must show in the events, so that a reviewer's lines cannot be removed while their folder stays:
    - a folder `round-N/<who>/` of a reviewer no `assign` event names (`prover` is the prover's);
    - a folder of a round other than the current one (`fold`'s round) that none of its own events records (a
      `file` or `response` event whose filing or response lies in it, `own_records`; a `note` never accounts
      for a role folder): an earlier round's is a filing or response whose events are gone, and no event can
      record a later round's;
    - a file of a folder its own events record that none of them records (the folder is frozen whether or not a
      line names it as `dir`): a file added after recording, or, in a folder recorded before the tool named
      folders, a work file the filing left out, which `tvr freeze` records once;
    - an entry of review/ that is no role folder (`kernel_folders`), so that no record is kept beside them.
    With `current`, also every folder of the current round that no event records (a filing or response not yet
    recorded): the certificate waits for it (`cert_state`), while the release check lints it instead
    (`release_records`)."""
    path = f"kernels/{k}/review/ledger.jsonl"
    full = os.path.join(root, path)
    try:
        events = [json.loads(l) for l in open(full, encoding="utf-8") if l.strip()] if os.path.exists(full) else []
        S = fold(events)
    except (ValueError, KeyError, TypeError, AttributeError):
        return []  # `kernel_event_problems` names a ledger that does not read or fold
    assigned = {e.get("actor") for e in events if isinstance(e, dict) and e.get("type") == "assign"}
    own = own_records(k, events)
    folders, strays = kernel_folders(root, k)
    bad = [f"{p}: in the kernel's review/ folder, but no record `tvr` writes (the ledger, the certificate, the snapshots, "
           "round-N/<who>/)" for p in strays]
    for (rnd, who), p in sorted(folders.items()):
        if who != "prover" and who not in assigned:
            bad.append(f"{p}: the folder of a reviewer no `assign` event of {path} names")
        elif p not in own and rnd != S["round"]:
            bad.append(f"{p}: a record of round {rnd} (the ledger is at round {S['round']}) that no filing or response "
                       f"of {path} records")
        elif p not in own and current:
            bad.append(f"{p}: a filing or response of the current round that is not recorded yet")
        if p not in own:
            continue
        for f in text_files(os.path.join(root, p)):
            q = os.path.relpath(f, root).replace(os.sep, "/")
            if q in own[p]["files"]:
                continue
            legacy = not own[p]["dir"] and not own[p]["frozen"] and is_work_file(p, q)
            bad.append(f"{q}: in a role folder that its own events record, but recorded by none of them" +
                       (f" (they were recorded before the tool named folders, and left this work file out: `tvr freeze "
                        f"{k}` records it once)" if legacy else ""))
    return bad


# Library rounds before the ledger existed (library issue L52): recorded with `--no-lint`, since their format
# was checked when they were written and the issue list no longer describes them under their rounds.
PRE_LEDGER_ROUNDS = 7


def library_ledger_problems(root):
    """The library's ledger beyond `ledger_problems` (library issue L52): each line records one role folder
    (`docs/library-review/round-N/<who>`, as `tvr record` writes it), a folder once only, only files inside
    it, and is linted unless its round is one of the first `PRE_LEDGER_ROUNDS`."""
    lpath, bad, seen = LIBRARY_LEDGER_PATH, [], set()
    for i, e in enumerate(library_ledger(root), 1):
        if not isinstance(e, dict):
            continue
        rnd, who, d = e.get("round"), e.get("who"), e.get("dir")
        files = e.get("files") if isinstance(e.get("files"), dict) else {}
        if e.get("type") != "record" or not isinstance(rnd, int) or d != f"docs/library-review/round-{rnd}/{who}":
            bad.append(f"{lpath}: event {i} is not a record of a role folder as `tvr record` writes it")
            continue
        if d in seen:
            bad.append(f"{lpath}: event {i} records {d} again, which `tvr record` refuses (a record is frozen)")
        seen.add(d)
        if not files or any(not p.startswith(d + "/") for p in files):
            bad.append(f"{lpath}: event {i} records files outside {d}, or none")
        if e.get("linted") is not True and rnd > PRE_LEDGER_ROUNDS:
            bad.append(f"{lpath}: event {i} records {d} without linting it, which only rounds 1 to {PRE_LEDGER_ROUNDS} "
                       "(before the ledger) may be")
    return bad


LIBRARY_LEDGER = os.path.join(LREVIEW, "ledger.jsonl")
LIBRARY_LEDGER_PATH = "docs/library-review/ledger.jsonl"  # the same, from the TVA root


def library_ledger(root=None):
    """The library review's ledger (library issue L52): one `record` event per role folder of a round
    (`docs/library-review/round-N/<who>/`), with the sha256 of every file in it, written by `tvr record`."""
    p = os.path.join(root or ROOT, LIBRARY_LEDGER_PATH)
    try:
        return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []
    except ValueError:
        return []


def library_rounds(root=None):
    """The library review's round folders, `{round: [role folder names]}`."""
    lr = os.path.join(root or ROOT, "docs", "library-review")
    out = {}
    for d in (os.listdir(lr) if os.path.isdir(lr) else []):
        m = re.fullmatch(r"round-(\d+)", d)
        if m and os.path.isdir(os.path.join(lr, d)):
            out[int(m.group(1))] = sorted(w for w in os.listdir(os.path.join(lr, d)) if os.path.isdir(os.path.join(lr, d, w)))
    return out


def ledger_digests(root):
    """Each ledger under `root`, by path, with its number of lines and its sha256 (library issue L52): what
    the release check writes into release.json. No step reads an earlier release.json back: the ledgers'
    lines are trusted as the coordinator keeps them (with no commit at each recording, library issues L36,
    L82), and an edited or removed line shows only against a copy kept elsewhere (a kept release.json, whose
    `lines` first lines of a ledger that only grew still hash to its `sha256`, or a commit), compared by
    hand."""
    out = {}
    kr = os.path.join(root, "kernels")
    paths = [f"kernels/{k}/review/ledger.jsonl" for k in sorted(os.listdir(kr) if os.path.isdir(kr) else [])]
    for p in paths + [LIBRARY_LEDGER_PATH]:
        f = os.path.join(root, p)
        if os.path.isfile(f):
            data = open(f, "rb").read()
            out[p] = {"lines": data.count(b"\n"), "sha256": sha(data)}
    return out


def records_problems(root):
    """The release check's `records` step (library issues L52, L81), on the tree at `root`: every kernel's
    ledger and the library's (`ledger_problems`), every kernel's events against its recorded files
    (`kernel_event_problems`) and its role folders against its events (`kernel_folder_problems`: a reviewer
    no `assign` names, a folder of another round that no filing or response records, a file of a recorded
    folder that none of its own events records), the library's records
    (`library_ledger_problems`), and every role folder of a library round before the last one recorded in the
    library's ledger (the last round's folders that are not yet recorded are linted instead, as are a
    kernel's current round's, `release_records`)."""
    bad = []
    kr = os.path.join(root, "kernels")
    for k in sorted(x for x in (os.listdir(kr) if os.path.isdir(kr) else []) if os.path.exists(os.path.join(kr, x, "kernel.json"))):
        bad += ledger_problems(root, f"kernels/{k}/review/ledger.jsonl", f"kernels/{k}/review/ledger.jsonl", folders=False)
        bad += kernel_event_problems(root, k)
        bad += kernel_folder_problems(root, k)
    lpath = LIBRARY_LEDGER_PATH
    bad += ledger_problems(root, lpath, lpath)
    bad += library_ledger_problems(root)
    done = {e.get("dir") for e in library_ledger(root)}
    rounds = library_rounds(root)
    for rnd, who in sorted(rounds.items()):
        if rnd == max(rounds):
            continue
        bad += [f"docs/library-review/round-{rnd}/{w}: a record of an earlier round, not recorded in {lpath} (`tvr record "
                f"library {rnd} {w}`)" for w in who if f"docs/library-review/round-{rnd}/{w}" not in done]
    return bad


def cmd_record(argv):
    """`tvr record library ROUND WHO [--no-lint]` (library issue L52): record a library role's folder
    (`docs/library-review/round-ROUND/WHO/`, the notes or report and every work file) in the library's
    ledger, with each file's sha256. It checks the output first, as `tvr lint library ROUND WHO` does,
    and refuses a folder recorded before, since recorded records are frozen. `--no-lint` records a folder
    of a round from before the ledger existed (rounds 1 to `PRE_LEDGER_ROUNDS`, 7), whose format was checked
    then and which the current issue list no longer describes; it is refused for a later round."""
    if len(argv) < 5 or argv[2] != "library" or not argv[3].isdigit():
        die("tvr record library ROUND WHO [--no-lint]")
    rnd, who = int(argv[3]), argv[4]
    lint = "--no-lint" not in argv
    if not lint and rnd > PRE_LEDGER_ROUNDS:
        die(f"--no-lint is for the rounds before the ledger (1 to {PRE_LEDGER_ROUNDS}); round {rnd} is linted when recorded")
    d = os.path.join(LREVIEW, f"round-{rnd}", who)
    if not os.path.isdir(d):
        die(f"no folder {rel(d)}")
    if any(e.get("dir") == rel(d) for e in library_ledger()):
        die(f"{rel(d)} is recorded already, and recorded records are frozen")
    files = text_files(d)
    if not files:
        die(f"{rel(d)} holds no file")
    bad = scan_files(files)
    if bad:
        die("a library record may cite only TVA and Mathlib (the protocol, §2): " + "; ".join(bad[:5]))
    if lint:
        report_problems(library_lint_problems(rnd, who), f"{rel(d)}")
    prev = library_ledger()
    with open(LIBRARY_LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"seq": len(prev) + 1, "time": now(), "type": "record", "actor": "coordinator", "round": rnd,
                            "who": who, "dir": rel(d), "linted": lint, "files": archive_entry(*files)},
                           sort_keys=True, ensure_ascii=False) + "\n")
    print(f"recorded {rel(d)}: {len(files)} file(s)" + ("" if lint else " (not linted: a record from before the ledger)"))
    return 0


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

# Look-alike letters (library issue L20): letters of other scripts that a reader takes for Latin ones.
# NFKC already maps the compatibility forms (`ℓ`, fullwidth and mathematical letters).
_CONFUSABLE = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ӏ": "l", "ѕ": "s",
    "һ": "h", "ԁ": "d", "т": "t", "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P",
    "С": "C", "Т": "T", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S", "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H",
    "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X", "ο": "o", "ν": "v",
    "ι": "i", "κ": "k", "τ": "t", "ǀ": "l", "ı": "i"})


def _skeleton(name):
    return unicodedata.normalize("NFKC", name).translate(_CONFUSABLE)


def binder_problems(surf):
    """Binders of the kernel's definitions named like a library namespace (`tl`, `Tensor`, `Block`, …,
    or a look-alike such as `tℓ`): model text `tl.add x y` under a binder `tl` is a field of the binder,
    not the library's operation (library issue L20)."""
    lib = {"tl"} | {x for x in surf.get("library_names", []) if x[:1].isupper()}
    prot = {_skeleton(x): x for x in lib}
    out = set()
    for decl, b in surf.get("binders", []):
        if _skeleton(b) in prot:
            out.add(f"{decl} (binds `{b}`, so model text `{b}.…` there is a field of the binder, not the "
                    f"library's `{prot[_skeleton(b)]}`)")
    return sorted(out)


def code_lines(src_text):
    """The lines of a Python source that hold code (library issue L39), read with Python's own `ast` and
    `tokenize`: a line is code when it holds a token other than a comment, a line break or indentation,
    except the string tokens of a statement that is only a string literal (a docstring, or a bare string
    standing alone, which runs nothing). A triple-quoted string inside an expression (an argument, an
    assignment) is code, on every line it spans. Returns `(lines, None)`, or `(None, reason)` when the
    source does not parse."""
    try:
        tree = ast.parse(src_text)
        toks = list(tokenize.generate_tokens(io.StringIO(src_text).readline))
    except (SyntaxError, tokenize.TokenError, ValueError) as e:
        return None, str(e)[:120]
    lines = io.StringIO(src_text).readlines()  # the lines as `tokenize` reads them

    def col(line, off):  # `ast` counts UTF-8 bytes, `tokenize` characters
        return len(lines[line - 1].encode()[:off].decode(errors="replace")) if line <= len(lines) else off

    spans = [((n.lineno, col(n.lineno, n.col_offset)), (n.end_lineno, col(n.end_lineno, n.end_col_offset)))
             for n in ast.walk(tree)
             if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, (str, bytes))]
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
    out = set()
    for t in toks:
        if t.type in skip:
            continue
        if t.type == tokenize.STRING and any(a <= t.start and t.end <= b for a, b in spans):
            continue
        out.update(range(t.start[0], t.end[0] + 1))
    return out, None


def jit_functions(src_text):
    """The `@triton.jit` functions of a source (library issue L50): every function, at any depth, with a
    decorator that names `jit` (`@triton.jit`, `@jit`, `@triton.jit(…)`, also under `@triton.autotune` and
    the like), as `{name: [[first, last], …]}`, from its first decorator's line to its last line; `None`
    when the source does not parse."""
    try:
        tree = ast.parse(src_text)
    except (SyntaxError, ValueError):
        return None

    def names_jit(d):
        d = d.func if isinstance(d, ast.Call) else d
        return (isinstance(d, ast.Attribute) and d.attr == "jit") or (isinstance(d, ast.Name) and d.id == "jit")

    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(names_jit(d) for d in node.decorator_list):
            first = min([d.lineno for d in node.decorator_list] + [node.lineno])
            out.setdefault(node.name, []).append([first, node.end_lineno])
    return {n: sorted(v) for n, v in out.items()}


def source_functions(src_text):
    """Every function the source defines, at any depth (library issue L50): each `def`, `async def` and
    `lambda`, as `(name, first, last)` from its first decorator's line to its last line (a lambda's name
    is `None`); `None` when the source does not parse. The launch can run only functions: a JIT function
    is made from a `def` of a Python file whatever the spelling (`@triton.jit`, an alias, a call
    `triton.jit(f)`: `JITCallable` reads its source lines and `JITFunction.parse` needs one `def`,
    `triton-3.8.0-jit.py:465-474, 546-551`); a `triton.constexpr_function` runs any Python function, a
    lambda included, at compile time (`:1138-1177`); and the front end calls any other callable it reaches,
    such as a plain function through a module (`triton-3.8.0-code-generator.py:387-416, 1444-1446`). So a
    rule over every function needs no spelling of `jit`."""
    try:
        tree = ast.parse(src_text)
    except (SyntaxError, ValueError):
        return None
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append((node.name, min([d.lineno for d in node.decorator_list] + [node.lineno]), node.end_lineno))
        elif isinstance(node, ast.Lambda):
            out.append((None, node.lineno, node.end_lineno))
    return sorted(out, key=lambda x: (x[1], x[2], x[0] or ""))


def def_headers(src_text, names):
    """The header lines of every `def` (and `async def`) named in `names` that no other function encloses: from
    its first decorator to the last line of its signature (the `:` before its body), as a set of line numbers
    (library issue L79). Python runs a `def` statement when it reaches it, so a module-level one runs at import,
    and its decorators, its parameters' defaults and its annotations are evaluated then, even if nothing ever
    calls the function; they can bind a module-level name (`x=(SCALE := 3)`). A `def` inside another function
    runs only when that function runs. `None` when the source does not parse."""
    try:
        tree = ast.parse(src_text)
        toks = list(tokenize.generate_tokens(io.StringIO(src_text).readline))
    except (SyntaxError, tokenize.TokenError, ValueError):
        return None
    lines = io.StringIO(src_text).readlines()

    def col(line, off):  # `ast` counts UTF-8 bytes, `tokenize` characters
        return len(lines[line - 1].encode()[:off].decode(errors="replace")) if line <= len(lines) else off

    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
    found = []

    def visit(node, inside):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if child.name in names and not inside:
                    found.append(child)
                visit(child, True)
            else:
                visit(child, inside or isinstance(child, ast.Lambda))
    visit(tree, False)
    out = set()
    for node in found:
        first = min([d.lineno for d in node.decorator_list] + [node.lineno])
        body = (node.body[0].lineno, col(node.body[0].lineno, node.body[0].col_offset))
        last = max([t.end[0] for t in toks if t.type not in skip and (first, 0) <= t.start < body] + [node.lineno])
        out.update(range(first, last + 1))
    return out


def module_lines(src_text, unused=()):
    """The source's code lines outside every function (library issue L79): the code lines (`code_lines`) that
    no `def`, `async def` or `lambda` spans (`source_functions`, from its first decorator to its last line), and
    the header of every module-level function named in `unused` (`source.unused`; `def_headers`: its decorators
    and `def` lines up to its body). They run when Python executes the module, and a JIT function reads what they
    bind: Triton's front end resolves a name the kernel uses to a module-level value (a module and its
    attributes, a JIT function, a `tl.constexpr(…)`, a dtype) at compile time, and the value is part of the
    compiled kernel (`triton-3.8.0-code-generator.py:383-420`, `triton-3.8.0-jit.py:532-536, 761-766`). So each
    such line lies in the kernel lines or in the host lines, and the reviewer judges which. The header of a
    function in the kernel or the host lines lies in its own class already. `None` when the source does not
    parse."""
    code, _ = code_lines(src_text)
    funcs = source_functions(src_text)
    heads = def_headers(src_text, set(unused))
    if code is None or funcs is None or heads is None:
        return None
    inside = {i for _, a, b in funcs for i in range(a, b + 1)}
    return sorted((code - inside) | (code & heads))


def _line_list(lines, most=12):
    """Line numbers as short runs, with the word: `line 52`, `lines 29–31, 52`."""
    runs = []
    for i in sorted(lines):
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    text = [f"{a}" if a == b else f"{a}–{b}" for a, b in runs]
    return ("line " if len(lines) == 1 else "lines ") + ", ".join(text[:most]) + (" …" if len(text) > most else "")


def map_problems(man, mm, src_text, lean_text):
    """The map check (protocol §7) on a manifest, a model map, the source's text and the kernel's Lean
    text, as `(problems, missing)`:
    - every entry is regular, and every piece of model text occurs (up to whitespace) in the kernel's Lean;
    - every function the source defines (`source_functions`: every `def` and `lambda`, at any depth) lies
      in the kernel lines (`source.ranges`), in the host lines (`source.host`) or, a `def`, in
      `source.unused`, and a `@triton.jit` function (`jit_functions`) is not in the host lines (library
      issue L50);
    - every code line outside every function, and the header of every module-level function in
      `source.unused` (`module_lines`), lies in the kernel lines or the host lines (library issue L79);
    - every code line of the ranges (`code_lines`, library issue L39) is covered by an entry: `missing`
      lists those that are not;
    - the source has no line break other than a line feed (library issue L51).
    Whether each entry is right, its drop reasons included, is the kernel reviewer's judgment, with
    docs/TRANSCRIPTION.md as guidance (library issue L36)."""
    claim_text = re.sub(r"\s+", " ", lean_text)
    covered, problems = set(), []
    if not (isinstance(mm, dict) and isinstance(mm.get("entries"), list)):
        problems.append("model-map.json is {\"entries\": [...]}")
        mm = {"entries": []}
    for i, e in enumerate(mm["entries"]):
        if not isinstance(e, dict):
            problems.append(f"entry {i}: not an object")
            continue
        try:
            if not (isinstance(e.get("src"), list) and e["src"]):
                raise ValueError
            covered |= set(parse_lines(e["src"]))
        except (ValueError, TypeError):
            problems.append(f"entry {i}: `src` is a non-empty list of line numbers and ranges \"a-b\"")
        names = e.get("rules", [])
        if not (isinstance(names, list) and all(isinstance(n, str) for n in names)):
            problems.append(f"entry {i} (src {e.get('src')}): `rules`, if given, is a list of strings")
        if ("model" in e) == ("drop" in e):
            problems.append(f"entry {i} (src {e.get('src')}): an entry has model text or a drop reason, one of them")
        elif "model" in e:
            snips = [e["model"]] if isinstance(e["model"], str) else e["model"]
            if not (isinstance(snips, list) and snips and all(isinstance(x, str) and x.strip() for x in snips)):
                problems.append(f"entry {i} (src {e.get('src')}): `model` is a string or a list of strings")
                continue
            for snip in snips:
                if re.sub(r"\s+", " ", snip).strip() not in claim_text:
                    problems.append(f"entry {i} (src {e.get('src')}): model text not found: {snip[:50]!r}")
        elif not (isinstance(e.get("drop"), str) and e["drop"].strip()):
            problems.append(f"entry {i} (src {e.get('src')}): a drop names its reason (a non-empty string)")
    seps = sorted({src_text.count("\n", 0, m.start()) + 1 for m in LINE_SEPARATORS.finditer(src_text)})
    if seps:
        problems.append(f"the source holds a line break other than a line feed ({_line_list(seps)}), which "
                        "Python's tokenizer, Triton, Lean and str.splitlines count differently (library issue L51)")
    # The code lines, by Python's own parser (library issue L39): a line holding any token but a comment or
    # a string statement standing alone (a docstring) is code, a triple-quoted argument included.
    code, why = code_lines(src_text)
    parsed = code is not None
    if not parsed:
        problems.append(f"the source is not Python the parser reads ({why}), so its code lines are unknown")
        code = set(range(1, src_text.count("\n") + 2))
    src = man["source"]
    in_ranges = {i for a, b in src["ranges"] for i in range(a, b + 1)}
    in_host = {i for a, b in src.get("host", []) for i in range(a, b + 1)}
    # Every function of the source lies in the kernel lines, in the host lines or in source.unused, and a
    # function under a decorator naming `jit` is not host code (library issue L50). Which functions the
    # launch runs is the reviewers' judgment; this makes the prover classify every one, so none escapes it.
    jit = jit_functions(src_text) or {}
    funcs = source_functions(src_text) or []
    unused = src.get("unused", [])
    for name in unused:
        if name not in {n for n, _, _ in funcs if n}:
            problems.append(f"source.unused names {name}, which is no function the source defines")
    if parsed and not jit:
        problems.append("the source has no @triton.jit function")
    for name, a, b in funcs:
        lines, is_jit = set(range(a, b + 1)), [a, b] in jit.get(name, [])
        if name in unused or lines <= in_ranges or (lines <= in_host and not is_jit):
            continue
        what = f"the function {name}" if name else "a lambda"
        if is_jit:
            problems.append(f"the ranges leave out {_line_list(sorted(lines - in_ranges))} of the @triton.jit function {name} "
                            f"(lines {a}–{b}): cover it, or name it in source.unused if the claimed launch never runs it")
        else:
            problems.append(f"{what} (lines {a}–{b}) lies neither in the kernel lines nor in the host lines: cover it, put it in "
                            "source.host if only the host runs it" + (", or name it in source.unused if nothing runs it" if name else ""))
    # Every code line outside every function, and the header of every unused module-level function (its
    # decorators, defaults and annotations run at import), lies in the kernel lines or the host lines (library
    # issue L79): a JIT function reads module-level names (a module such as `tl`, a JIT function) at compile
    # time, so which of these lines the launch reads is the reviewers' judgment, and none escapes it unclassed.
    heads = def_headers(src_text, set(unused)) or set()
    loose = [i for i in (module_lines(src_text, unused) or []) if i not in in_ranges and i not in in_host]
    body = [i for i in loose if i not in heads]
    head = [i for i in loose if i in heads]
    if body:
        problems.append(f"the code {_line_list(body)}, outside every function, {'lies' if len(body) == 1 else 'lie'} neither "
                        "in the kernel lines nor in the host lines: put a line that binds a name the launch reads (the "
                        "import of `triton.language as tl`, a JIT function) in the kernel lines, and the others in source.host")
    if head:
        problems.append(f"the header of a function in source.unused ({_line_list(head)}: its decorators and `def` lines, "
                        "which Python evaluates when it imports the module, defaults and annotations included) lies neither "
                        "in the kernel lines nor in the host lines: put it in source.host, or in the kernel lines if it binds "
                        "a name the launch reads")
    missing = [i for i in sorted(in_ranges) if i in code and i not in covered]
    return problems, missing


def claim_reader_text(mod, ns, opens):
    """The claim reader (tools/claim_reader.lean.in) for a kernel's module and namespace, with the names its
    files open. Only plain names (`LEAN_NAME`) reach it (library issue L38)."""
    tmpl = open(os.path.join(ROOT, "tools", "claim_reader.lean.in"), encoding="utf-8").read()
    return (tmpl.replace("@MODULE@", mod).replace("@NS@", ns).replace("@PREFIX@", mod.split(".")[0])
            .replace("@OPENS@", ", ".join(f"`{x}" for x in sorted(opens))))


def reader_result(code, stdout, stderr):
    """The claim reader's surface from its exit code and output, or None, with the shape step's detail. The reader
    passes when it exits with 0 and prints one result line, and no other line of its output says PANIC: a reader
    that panicked read on past an internal error (library issue L106: of an `Assumptions` that is not a structure,
    `getStructureFields` panicked and returned no field), so its result is not used. The result line itself is not
    read for the word (library issue L112): it quotes the claim's docstrings and names, which may hold it."""
    js = [l for l in stdout.splitlines() if l.startswith("TVR_JSON ")]
    panic = [l for l in (stdout + "\n" + stderr).splitlines() if "PANIC" in l and not l.startswith("TVR_JSON ")]
    if code == 0 and len(js) == 1 and not panic:
        return (json.loads(js[0][len("TVR_JSON "):]),
                "main : ∀ K P, Assumptions K P → Spec K; consistent : Claim.Consistent Assumptions; "
                "satisfiable : Claim.Satisfiable Inst.ok Pre; Spec K = Meets Inst.ok Pre Post K; Assumptions is a "
                "structure with exactly the parameters (K : Claim.Behaviour Inst) (P : Platform)")
    return None, (f"the claim reader panicked (library issue L106): {panic[0][:160]}" if panic else
                  f"the claim reader printed {len(js)} result lines, not one" if code == 0 and js else
                  "the claim does not have the protocol's shape: " + " | ".join(stdout.splitlines()[:4]))


def const_table(surf):
    """The claim reader's constants by name. Each one's `hash` is the sha256 of its canonical text
    (`canon`, written by the reader's `TVR.canon`: its kind, universe parameters, type, and value for a
    definition, every name, level, literal and binder kind spelled out; library issue L48), so two
    constants that differ in any of these differ in their hash, and so in every item digest whose cone
    holds them."""
    out = {}
    for c in surf.get("consts", []):
        if c:
            c = dict(c)
            c["hash"] = sha(c.pop("canon", ""))
            out[c["name"]] = c
    return out


def item_digests(surf, consts, src_sha, mm_text):
    """Each item's digest (protocol §4): its cone's constants by the sha256 of their canonical texts, its
    roots' docstrings, the library files its cone reaches (whole), the source's sha256 for `model`, an
    assumption's statement (the sha256 of its canonical text, `statement`, library issue L68), and Lean's
    version; the `map` digest is the model map, the source's sha256 and the `model` item's files."""
    items, model_files = {}, {}
    for it, v in surf["items"].items():
        lines = sorted(f"{n} {consts[n]['hash']}" for n in v["cone"] if n in consts)
        docs = [consts[n].get("doc") or "" for n in v["roots"] if n in consts]
        statement = {}
        if it.startswith("assumption:"):
            f = it.split(":", 1)[1]
            docs.append(surf["field_docs"].get(f, ""))
            # the field's statement itself, as canonical text with `K` and `P` bound (library issue L68): its
            # constants alone would leave out its literals, binders and structure
            statement = {"statement": sha(surf["field_uses"].get(f, {}).get("canon", ""))}
        # The library files the cone reaches, whole (protocol §4, library issue L36).
        files = library_files(v["cone"], consts)
        if it == "model":
            model_files = files
        extra = ([src_sha] if it == "model" else []) + [f"{key} {h}" for key, h in sorted({**files, **statement}.items())]
        items[it] = {"digest": sha("\n".join(lines + docs + extra + [surf["lean"]])), "roots": v["roots"],
                     "cone": sorted(v["cone"]), "files": {**files, **statement}}
    # The map's digest: the map, the source, and the library files the model reaches, whose meaning the
    # correspondence of each line rests on.
    items["map"] = {"digest": sha("\n".join([mm_text, src_sha] + [f"{key} {h}" for key, h in sorted(model_files.items())]))
                    if mm_text is not None else "", "roots": [], "cone": [], "files": model_files}
    return items


_EQN_LEMMA = re.compile(r"eq_\d+|eq_def|eq_unfold")
FAITHFUL = "TVA.Rules.Compiler.Faithful"
# The facts a `[rule]` field may cite besides `Compiler.Faithful` (library issue L62): those the table of
# docs/rules/num.md's section "Facts the library states" names, each a code span in its first column.
FACTS_PAGE = "docs/rules/num.md"
FACTS_SECTION = "Facts the library states"


_FENCE_OPEN = re.compile(r"(`{3,}|~{3,})(.*)$")
_FENCE_CLOSE = re.compile(r" {0,3}(`{3,}|~{3,})[ \t]*$")
_FENCE_INDENTED = re.compile(r" {1,3}(`{3,}|~{3,})")
_FACTS_TEXT = re.compile(r"(?:[^\W_]|[(\[\"']|`(?!``))")
_LIST_ITEM = re.compile(r"\d{1,9}[.)](?:\s|$)")
_HTML_START = re.compile(r" {0,3}<")
_HTML_TAG = re.compile(r"<[A-Za-z/!?]")


def gfm_cells(row):
    """A table row's cells as GitHub Flavored Markdown splits them (library issue L95): one leading and one
    trailing `|` removed, then a split at each `|` that no backslash escapes. So a row that starts `||` has an
    empty first cell, as the page shows it."""
    s = row.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return re.split(r"(?<!\\)\|", s)


def facts_section(root=None):
    """docs/rules/num.md's section "Facts the library states", as `(lines, problems)`, or `None` when the page or
    the section is missing. The section is the lines after its `##` heading (at the start of a line), up to the
    next `#` or `##` heading. The gate reads it line by line, so the section may hold only the forms whose
    rendering a line reader knows (library issue L95): blank lines, table rows (lines that start with `|`), `###`
    to `######` headings, and lines of text that start with a letter, a digit (not a list's number), `(`, `[`, a
    quote or one backtick; and each table has a blank line, or the section's start or end, before and after
    it, and a header row with as many cells as its delimiter row (cells split as GFM splits them, `gfm_cells`;
    with other counts GFM shows no table). Anything else in the section is a problem: a setext heading or a rule
    (a line of `=` or `-` under text), a fence, an indented line (code), a block quote, a list, HTML (an HTML
    comment included, `<!--`, and inline HTML: a `<` before a letter, `/`, `!` or `?`, anywhere in a line), a
    strikethrough (a `~`), a text line against a table. Before the section, the lines are read as CommonMark
    reads its blocks at the top level of the page: a fence opens at a line that starts with three or more
    backticks or tildes (a backtick fence's info string holds no backtick, else the line is text) and closes at a
    line of the same character, at least as long, with nothing after it but spaces; a fence line indented by one
    to three spaces is a problem (it may belong to a list item, which a line reader cannot tell), and so is a
    line outside a fence that starts with `<` after up to three spaces (an HTML block, which can run on across
    blank lines and headings). So is a heading of that name anywhere else on the page, and a section heading
    that lies inside a fenced block or an HTML comment. With a problem, the gate accepts no fact
    (`stated_facts`) and `tvr audit` names the page (`fact_page_problems`): what the gate reads as the facts
    table is then what the rendered page shows as it."""
    try:
        lines = open(os.path.join(root or ROOT, FACTS_PAGE), encoding="utf-8").read().split("\n")
    except (OSError, UnicodeDecodeError):
        return None
    name = re.escape(FACTS_SECTION) + r"\b"
    start = next((i for i, l in enumerate(lines) if re.match(r"##\s+" + name, l)), None)
    if start is None:
        return None
    problems = []
    # the heading itself: the only heading of that name on the page, and not inside a fence or a comment
    for i, l in enumerate(lines):
        setext = (i + 1 < len(lines) and re.fullmatch(r" {0,3}(=+|-+)\s*", lines[i + 1]) is not None
                  and re.search(name, l) is not None)
        if i != start and (re.match(r"\s*#+\s*" + name, l) or setext):
            problems.append(f"line {i + 1}: a second heading \"{FACTS_SECTION}\"")
    # what comes before the heading, as CommonMark reads its blocks at the top level (library issue L95)
    fence, comment = None, False
    for i, l in enumerate(lines[:start]):
        if fence is not None:
            m = _FENCE_CLOSE.match(l)
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                fence = None
            continue
        if _FENCE_INDENTED.match(l):
            problems.append(f"line {i + 1}: a fence indented by one to three spaces before the section (it may "
                            "belong to a list item, which a line reader cannot tell)")
            continue
        m = _FENCE_OPEN.match(l)
        if m and not (m.group(1)[0] == "`" and "`" in m.group(2)):
            fence = m.group(1)
            continue
        if _HTML_START.match(l):
            problems.append(f"line {i + 1}: a line before the section that starts with `<` (an HTML block, which can "
                            "run on across blank lines and headings)")
        if "<!--" in l:
            comment = "-->" not in l.split("<!--")[-1]
        elif "-->" in l:
            comment = False
    if fence is not None or comment:
        problems.append(f"line {start + 1}: the section's heading lies inside a fenced block or an HTML comment")
    sec = []
    for l in lines[start + 1:]:
        if re.match(r"#{1,2}\s", l):
            break
        sec.append(l)
    for k, l in enumerate(sec):
        n = start + 2 + k
        prev = sec[k - 1] if k > 0 else ""
        nxt = sec[k + 1] if k + 1 < len(sec) else ""
        if "<!--" in l or "-->" in l:
            problems.append(f"line {n}: HTML comments are not read as the page renders them")
        elif _HTML_TAG.search(l):
            problems.append(f"line {n}: HTML (a `<` before a letter, `/`, `!` or `?`) is not read as the page "
                            "renders it")
        if "~" in l:
            problems.append(f"line {n}: a `~` (GFM strikes text through between tildes)")
        if not l.strip():
            continue
        if l.startswith("|"):
            for side in (prev, nxt):
                if side.strip() and not side.startswith("|"):
                    problems.append(f"line {n}: a table needs a blank line before and after it")
            if not prev.strip() and nxt.startswith("|") and len(gfm_cells(l)) != len(gfm_cells(nxt)):
                problems.append(f"line {n}: the table's header row and delimiter row have different numbers of "
                                "cells, so GFM shows no table")
            continue
        if re.match(r"#{3,6}\s", l):
            continue
        if _FACTS_TEXT.match(l) and not _LIST_ITEM.match(l):
            if re.fullmatch(r" {0,3}(=+|-+)\s*", nxt):
                problems.append(f"line {n + 1}: a setext heading (text over a line of = or -)")
            continue
        problems.append(f"line {n}: a line that is no blank line, table row, ### heading or text "
                        f"(a rule or setext underline, a fence, an indented line, a quote, a list or HTML): "
                        f"{l.strip()[:40]!r}")
    return sec, sorted(set(problems), key=lambda p: int(re.match(r"line (\d+)", p).group(1)))


def facts_section_tables(root=None):
    """The tables of docs/rules/num.md's section "Facts the library states" (`facts_section`), as `(rows,
    under_heading)`: the rows are a run of lines that start with `|`, and `under_heading` says whether a `###`
    heading of the section comes before the table. `None` when the page or the section is missing. The gate
    reads only the first table, and only when no heading comes before it and the section has no problem
    (`stated_facts`, library issue L95), and `tvr audit` fails on a section with more than one table, or with a
    problem (`fact_page_problems`), so that no second table, whatever its heading says, can look citable, or not
    citable, to a reader."""
    got = facts_section(root)
    if got is None:
        return None
    tables, run, heading = [], None, False
    for l in got[0]:
        if l.startswith("|"):
            if run is None:
                run = []
                tables.append((run, heading))
            run.append(l.strip())
        else:
            run = None
            heading = heading or bool(re.match(r"#+\s", l))
    return tables


def stated_facts(root=None):
    """The facts a `[rule]` field may cite besides `Compiler.Faithful` (library issue L62), as docs/rules/num.md
    lists them: every code span in the first column of the first table of its section "Facts the library
    states" (`facts_section_tables`: the table ends at its first line that is not a table row, and a heading
    before any table leaves the section with none; its cells are split as GFM splits them, `gfm_cells`, so a row
    that starts `||` has an empty first cell; library issue L95). A span is a fact of `TVA.Rules.Num`
    (`ExpF32`; a full name `TVA.Rules.Num.…` too), alone or applied as the field must apply it (`DivRn P
    .binary64`: `P` the platform, `.x` the constant `x` of the argument's type, as Lean's dot notation reads it, a
    dotted name a constant under `TVA`). Returns `{full name: [arguments, or None for the name alone]}`, or `None`
    when the page, the section or its table is missing or names no fact: the gate then accepts no such fact (it
    fails closed). The page is in the digest of every item that cites a fact of `TVA.Rules.Num` (protocol §4), so
    a change to the list re-opens those items."""
    got = facts_section(root)
    if got is None or got[1]:
        return None
    tables = facts_section_tables(root)
    if not tables or tables[0][1]:
        return None
    rows = tables[0][0]
    if len(rows) < 3 or not re.fullmatch(r"\|(?:\s*:?-+:?\s*\|)+", rows[1]):
        return None
    out = {}
    for r in rows[2:]:
        first = gfm_cells(r)[0]
        for span in re.findall(r"`([^`\n]+)`", first):
            toks = span.split()
            if toks and LEAN_NAME.fullmatch(toks[0]):
                name = toks[0] if toks[0].startswith("TVA.") else "TVA.Rules.Num." + toks[0]
                out.setdefault(name, []).append(tuple(toks[1:]) or None)
    return out or None


def _applied_as(pattern, args):
    """Whether a `[rule]` field's arguments (the claim reader's `args`) are the listed application `pattern`
    (`("P", ".binary64")`): `P` the platform, `.x` the constant `<the argument's type>.x`, another token the
    constant under `TVA` it names."""
    if len(pattern) != len(args):
        return False
    for tok, a in zip(pattern, args):
        if tok == "P":
            ok = bool(a.get("platform"))
        elif tok.startswith("."):
            ok = bool(a.get("type")) and a.get("const") == f"{a['type']}{tok}"
        else:
            ok = a.get("const") == (tok if tok.startswith("TVA.") else "TVA." + tok)
        if not ok:
            return False
    return True


def faithful_problem(u, ns):
    """Why a `[rule]` field that applies `Compiler.Faithful` does not apply it as the library states it, or
    `None` (library issue L62): `Faithful P (model x) (launch x) (K x)`, with the kernel's own `model` and
    `launch` and one variable `x` the statement binds (under its `∀` and `→` binders), the same in all three.
    The rule relates a launch's behaviour to the model of the instance launched; any other argument (the model
    of another instance, `model false`, `model (Classical.choose h)`, a launch written out) would let `main`
    rest on a model the launch does not run."""
    args = u.get("args", [])
    fns = [a.get("fn") for a in args[1:]]
    xs = {a.get("var") for a in args[1:]}
    one = len(xs) == 1 and isinstance(next(iter(xs)), int) and next(iter(xs)) >= 0
    if len(args) == 4 and fns == [f"{ns}.model", f"{ns}.launch", "K"] and one:
        return None
    given = " ".join(["Faithful", "P"] + [f"({a.get('pp') or '?'})" for a in args[1:]])
    return (f"Compiler.Faithful is cited only as `Faithful P (model x) (launch x) (K x)`, with the kernel's own model "
            f"and launch and one variable x the statement binds (library issue L62), not as `{given[:160]}`")


def rule_fact_problem(u, ns="", facts=None):
    """Why a `[rule]` field's statement is not a fact the library states, applied to the platform `P`
    (library issue L62), or `None`. The field's shape (`u`, from the claim reader) is its statement under
    its `∀` and `→` binders: the constant it applies and the arguments. A fact the library states is one
    of these, applied to `P`:
    - `Compiler.Faithful`, as `Faithful P (model x) (launch x) (K x)` for one variable `x` the statement
      binds (`faithful_problem`);
    - a fact docs/rules/num.md's section "Facts the library states" names (`facts`, from `stated_facts`):
      listed alone (`ExpF32`, `DivFull`, `MathTotal`), with any arguments after `P` (a `Platform → Prop` fact
      takes none); listed applied (`DivRn P .binary64`), with exactly those arguments.
    Never a shape whose constants the field chooses (`MathAcc`, `MathAt`, `DivAcc`, `CvtAcc`, `CvtRn`,
    `DotAcc`, `DotScaledAcc`, `AtomicAddAcc`, a family such as `DotF8Chunk`, or `DivRn` on another format), nor a
    `TVA.Rules` constant the list does not name, whatever its type or name: the library states its facts with
    their constants, and lists them; a field that restates a shape with constants of its own states a device
    fact of the kernel's own, which no library review stands behind."""
    head, args = u.get("head", ""), u.get("args", [])
    short = head[4:] if head.startswith("TVA.") else head or "(no constant)"
    on_p = bool(args) and bool(args[0].get("platform"))
    if not head.startswith("TVA.Rules."):
        return f"it states no library fact (under its binders it applies {short}, not a TVA.Rules fact)"
    if not on_p:
        return f"{short} is not applied to the platform P"
    if head == FAITHFUL:
        return faithful_problem(u, ns)
    if facts is None:
        return (f"{FACTS_PAGE} has no section \"{FACTS_SECTION}\" with a table of facts, so the gate accepts no fact "
                "but Compiler.Faithful (it fails closed)")
    pats = facts.get(head)
    if pats is None:
        return (f"{short} is not among the facts {FACTS_PAGE}'s \"{FACTS_SECTION}\" lists (a shape with constants the "
                "field chooses never is): cite a listed fact by its name, or ask the library for one")
    if any(p is None or _applied_as(p, args) for p in pats):
        return None
    last = head.rsplit(".", 1)[-1]
    return (f"{short} is cited only as {FACTS_PAGE}'s \"{FACTS_SECTION}\" lists it (" +
            ", ".join(f"`{' '.join((last,) + p)}`" for p in pats) + ")")


def kinds_problems(surf, consts, ns):
    """The `kinds` check (protocol §5): every assumption is a `[rule]` stating a fact the library states,
    applied to the platform (`rule_fact_problem`, library issue L62: `Compiler.Faithful` in its one form, or a
    fact docs/rules/num.md lists), with no constant of the kernel's own but `RULE_ARGS` and with `K` only in
    `Compiler.Faithful`, or a `[kernel]` fact free of `K`, `P` and `TVA.Rules`; and every `TVA.Rules.*`
    constant a cone reaches is declared under lean/TVA/Rules (library issue L55), where its rule page is (an
    equation lemma Lean generates for one, wherever it is declared, aside). The kernel's own constants are
    every constant of the statement that is not the library's (rooted at `TVA`): a private one, and one
    declared outside the kernel's namespace, count too (library issue L63)."""
    bad = [f"{n} is declared in {consts[n].get('module') or 'no module'}, outside lean/TVA/Rules, so no rule page "
           "covers it" for n in sorted(consts) if n.startswith("TVA.Rules.")
           and not consts[n].get("module", "").startswith("TVA.Rules.") and not _EQN_LEMMA.fullmatch(n.rsplit(".", 1)[-1])]
    allowed = {f"{ns}.{x}" for x in RULE_ARGS}
    facts = stated_facts()
    for f in surf["fields"]:
        m = re.match(r"\s*\[(rule|kernel)\]", surf["field_docs"].get(f, ""))
        u = surf["field_uses"].get(f, {"consts": [], "K": False, "P": False})
        own = sorted(c for c in u["consts"] if c.split(".")[0] != "TVA" and c not in allowed)
        rules = [c for c in u["consts"] if c.startswith("TVA.Rules.")]
        if not m:
            bad.append(f"{f}: not tagged [rule] or [kernel]")
        elif m.group(1) == "rule":
            why = rule_fact_problem(u, ns, facts) if rules else "a [rule] must cite a TVA.Rules fact"
            if own:
                bad.append(f"{f}: a [rule] may mention no constant of the kernel's own but {', '.join(RULE_ARGS)} (found "
                           + ", ".join(c[len(ns) + 1:] if c.startswith(ns + ".") else c for c in own) + ")")
            if why:
                bad.append(f"{f}: {why}")
            if u.get("K") and u.get("head") != FAITHFUL:
                bad.append(f"{f}: only Compiler.Faithful may mention K")
        elif m.group(1) == "kernel" and (u["K"] or u["P"] or rules):
            bad.append(f"{f}: a [kernel] assumption may not mention K, P or a TVA.Rules fact")
    return bad


# What a gate records about what made it (library issue L53): the sha256 of every kernel file it read, and
# of the gate's own code. A gate is stale when either differs from the tree's now (`gate_staleness`).
GATE_TOOL = ("tools/tvr.py", "tools/claim_reader.lean.in")


def tool_digest():
    """The gate's own code, by path, with its sha256."""
    return {p: _file_digest(p) for p in GATE_TOOL}


def kernel_files(k, man=None):
    """The kernel's files the gate reads, by path in the kernel folder, with their sha256 (library issue
    L53): the manifest, the source, the model map, the write-up and every Lean file."""
    man = man if man is not None else load_json(kpath(k, "kernel.json"), {})
    src = (man.get("source") or {}).get("file") if isinstance(man.get("source"), dict) else None
    paths = ["kernel.json", "model-map.json", str(man.get("doc", "README.md"))] + ([src] if isinstance(src, str) else [])
    paths += [rel(p, kpath(k)) for p in kernel_lean_files(k)]
    return {p: (file_sha(kpath(k, p)) if os.path.isfile(kpath(k, p)) else "missing") for p in sorted(set(paths))}


# The files besides the kernel's own that decide what its Lean elaborates to (library issue L69): the
# toolchain's name, the lake configuration and manifest, and every library file the kernel's modules
# import, directly or through other library modules. An item's digest holds what its cone's constants
# are; these decide whether the gate, run now, would still compute the same (a notation or a macro of an
# imported module, a `leanOptions` key, another Lean), so a change to one makes the gate stale.
BUILD_CONFIG = ("lean/lean-toolchain", "lean/lakefile.toml", "lean/lake-manifest.json")


def build_inputs(k):
    """The build's inputs a gate records, by path from the TVA root, with their sha256: `BUILD_CONFIG` and
    the library's Lean files the kernel's Lean files import, transitively (a module outside the library,
    such as Mathlib's, is held by the packages' digest)."""
    lean_dir = os.path.join(ROOT, "lean")
    seen, todo = set(), []
    for p in kernel_lean_files(k):
        try:
            todo += lean_header(open(p, encoding="utf-8").read())[0]
        except (UnicodeDecodeError, OSError):
            continue
    while todo:
        m = todo.pop()
        f = os.path.join(lean_dir, *m.split(".")) + ".lean"
        if m in seen or m.split(".")[0] != "TVA" or not os.path.isfile(f):
            continue
        seen.add(m)
        try:
            todo += lean_header(open(f, encoding="utf-8").read())[0]
        except (UnicodeDecodeError, OSError):
            continue
    paths = list(BUILD_CONFIG) + [module_file(m) for m in seen]
    return {p: _file_digest(p) for p in sorted(paths)}


# A kernel's library (library issue L187). `kernels/` is local and git ignores it, so the shipped
# `lean/lakefile.toml` declares TVA alone (`lake_violations`). For a kernel's build, `tvr` writes
# `lean/.lake/tvr/lakefile.toml`: the shipped configuration, then one library per kernel folder
# (`srcDir = "../kernels/<kernel>/lean"`, the kernel's module as its root, a name of `[a-z0-9_]`, a root
# other than TVA's and Mathlib's), and builds with `lake -f` from `lean/`, so the package, its manifest
# and its build folder are the library's. `tvr new`, `tvr check` and `tvr index` write it afresh.
KERNEL_LAKEFILE = os.path.join(ROOT, "lean", ".lake", "tvr", "lakefile.toml")


def kernel_lakefile():
    """Write `KERNEL_LAKEFILE` (above) from `lean/lakefile.toml` and the kernel folders; return its path."""
    with open(os.path.join(ROOT, "lean", "lakefile.toml"), encoding="utf-8") as f:
        out = [f.read().rstrip("\n")]
    for k in kernels():
        mod = str((manifest(k).get("lean") or {}).get("module", ""))
        root_mod = mod.split(".")[0]
        if re.fullmatch(r"[a-z0-9_]+", k) and LEAN_NAME.fullmatch(mod) and root_mod not in LEAN_ROOTS:
            out.append(f'[[lean_lib]]\nname = "{k}"\nsrcDir = "../kernels/{k}/lean"\nroots = ["{root_mod}"]')
    write(KERNEL_LAKEFILE, "\n\n".join(out) + "\n")
    return KERNEL_LAKEFILE


def check(k):
    man = manifest(k)
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    # The manifest first (library issue L38): the claim reader is Lean text built around the module and
    # the namespace, so nothing else runs unless they are plain names.
    mp = manifest_problems(k, man)
    add("manifest", not mp, "kernel.json: the module and namespace are plain names, the files are in the kernel "
        "folder, the ranges are line pairs" if not mp else "kernel.json: " + "; ".join(mp))
    if mp:
        gate = {"protocol": "tvr-2", "kernel": k, "time": now(), "ok": False, "digest": "", "checks": checks,
                "items": {}, "kinds": {}, "files": kernel_files(k, man), "tool": tool_digest(), "build": build_inputs(k)}
        dump_json(kpath(k, "gate.json"), gate)
        refresh(k)
        print(f"FAIL  manifest   {checks[0]['detail']}\ngate FAIL (nothing else runs on this manifest)")
        return False
    # the files this gate reads, as they are now (library issue L53), and the build's other inputs (L69)
    files_read, tool, build = kernel_files(k, man), tool_digest(), build_inputs(k)
    src = kpath(k, man["source"]["file"])
    # read as it is, line breaks included (library issue L51): its sha256 is the file's
    src_text = open(src, encoding="utf-8", errors="replace", newline="").read() if os.path.exists(src) else ""
    src_sha = sha(src_text)
    add("source", src_sha == man["source"]["sha256"],
        f"{man['source']['file']} sha256 {src_sha[:16]}" +
        ("" if src_sha == man["source"]["sha256"] else f" (the manifest says {man['source']['sha256'][:16]})"))
    lean_dir = os.path.join(ROOT, "lean")
    mod, ns = man["lean"]["module"], man["lean"]["namespace"]
    # hygiene and independence read the files only; they run first, and the build (which runs the
    # kernel's Lean) only when both pass
    bad = hygiene_problems(k) + library_hygiene_problems()
    add("hygiene", not bad, "no proof escape, kernel bypass, meta code, meta type, handler attribute, notation, "
        "renaming, local or erased instance, option beyond limits, printing and linters, string literal holding `{` "
        "or line break other than a line feed in the kernel; no proof escape, bypass, meta code, such option or "
        "string in the library's code" if not bad
        else "; ".join(bad[:5]) + (f" (+{len(bad) - 5} more)" if len(bad) > 5 else ""))
    ind = kernel_independence(k) + library_independence()
    add("independence", not ind, "the kernel and the library import only TVA, the kernel's own modules and Mathlib, "
        "and refer to no other project" if not ind else "; ".join(ind[:5]) + (f" (+{len(ind) - 5} more)" if len(ind) > 5 else ""))
    built = False
    if bad or ind:
        add("build", False, "not run (hygiene or independence failed)")
    else:
        r = subprocess.run([LAKE, "-f", kernel_lakefile(), "build", mod], cwd=lean_dir, capture_output=True, text=True)
        errs = [l for l in r.stdout.splitlines() if l.startswith("error")]
        built = r.returncode == 0
        add("build", built, f"lake build {mod}" + ("" if built else ": " + " | ".join(errs[:3])))
    if built:
        add("replay", *replay(k))
    else:
        add("replay", False, "not run (the build did not run or failed)")
    # Only plain names are spliced into the reader (library issue L38): the module and namespace were
    # checked above, and an opened name that is not plain fails the namespace check instead.
    opens = opened_names(k)
    bad_opens = sorted(x for x in opens if not LEAN_NAME.fullmatch(x))
    gpath = os.path.join(lean_dir, ".lake", "tvr", k, "claim_reader.lean")
    write(gpath, claim_reader_text(mod, ns, set(opens) - set(bad_opens)))
    surf = None
    if built:
        r2 = subprocess.run([LAKE, "env", "lean", gpath], cwd=lean_dir, capture_output=True, text=True)
        surf, why = reader_result(r2.returncode, r2.stdout, r2.stderr)
        add("shape", surf is not None, why)
    else:
        add("shape", False, "not checked (the build did not run or failed)")
    items, consts = {}, {}
    if surf:
        consts = const_table(surf)
        add("axioms", all(set(v) <= STD_AXIOMS for v in surf["axioms"].values()),
            "; ".join(f"{a}: {', '.join(v)}" for a, v in surf["axioms"].items()))
        unused = [f for f, u in surf["usage"].items() if not u]
        add("usage", not unused and not surf["destructured"],
            "every assumption is used by main" if not unused and not surf["destructured"] else
            ("unused: " + ", ".join(unused) if unused else "") +
            (" (Assumptions destructured: use projections)" if surf["destructured"] else ""))
        bad_ns = (surf["shadow"] + [f"{i['name']} (an instance; a kernel declares none)" for i in surf["instances"]
                                    if not i.get("internal")] + binder_problems(surf) +
                  [f"`open {x}` (not a plain name, so the claim reader cannot take it)" for x in bad_opens])
        add("namespace", not bad_ns,
            "the kernel declares everything in its namespace, no name that could be read as another constant, no "
            "name outside plain ASCII, no instance, and no binder named like a library namespace" if not bad_ns
            else "declares " + "; ".join(bad_ns[:6]) + (f" (+{len(bad_ns) - 6} more)" if len(bad_ns) > 6 else ""))
        bad_kinds = kinds_problems(surf, consts, ns)
        add("kinds", not bad_kinds, "every assumption is a [rule] stating a fact the library states, applied to the "
            "platform (Compiler.Faithful as Faithful P (model x) (launch x) (K x), or a fact docs/rules/num.md lists, as "
            "listed), with no constant of the kernel's own, or a [kernel] fact about the kernel's own values; every "
            "TVA.Rules constant reached is declared under lean/TVA/Rules"
            if not bad_kinds else "; ".join(bad_kinds[:6]) + (f" (+{len(bad_kinds) - 6} more)" if len(bad_kinds) > 6 else ""))
    # The model map (protocol §7).
    mm_path = kpath(k, "model-map.json")
    mm = load_json(mm_path, {"entries": []})
    lean_text = " ".join(open(p, encoding="utf-8", errors="replace").read() for p in kernel_lean_files(k))
    problems, missing = map_problems(man, mm, src_text, lean_text)
    add("map", not missing and not problems,
        f"{len(mm.get('entries', [])) if isinstance(mm, dict) else 0} entries cover every code line of "
        f"{man['source']['ranges']}; every function of the source lies in the kernel lines or the host lines"
        + (f" but {', '.join(man['source']['unused'])} (unused)" if man["source"].get("unused") else "")
        + ", and every @triton.jit function in the kernel lines; every code line outside a function, and every unused "
        "function's header, lies in the kernel lines or the host lines"
        if not missing and not problems
        else "; ".join(([f"uncovered source {_line_list(missing)}"] if missing else []) + problems[:4]))
    if surf:
        items = item_digests(surf, consts, src_sha,
                             open(mm_path, encoding="utf-8", newline="").read() if os.path.exists(mm_path) else None)
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
            "checks": checks, "items": {i: v["digest"] for i, v in items.items()}, "kinds": kinds,
            "files": files_read, "tool": tool, "build": build}
    dump_json(kpath(k, "gate.json"), gate)
    if ok and surf:
        dump_json(kpath(k, "review", "snapshots", gate_digest[:16] + ".json"),
                  {"items": {i: {"digest": v["digest"], "cone": {**{n: consts[n]["hash"] for n in v["cone"] if n in consts},
                                                                 **v.get("files", {})}}
                             for i, v in items.items()}})
    if surf:
        render_claim(k, man, gate, surf, consts, items, mm, src_text.split("\n"), jit_functions(src_text) or {},
                     source_functions(src_text) or [])
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


def render_claim(k, man, gate, surf, consts, items, mm, src_lines, jit=None, funcs=()):
    """claim.md. Lines are numbered by line feeds only (library issue L51), as Python's tokenizer, Triton and
    Lean's declaration ranges number them: `src_lines` is the source split at line feeds, and a Lean
    declaration is quoted from its file split the same way."""
    ns = man["lean"]["namespace"]
    out = []
    w = out.append
    w(f"# Claim: `{k}`\n")
    w(f"> Generated by `tvr check` at {gate['time']}; do not edit. Gate: **{'PASS' if gate['ok'] else 'FAIL'}**, "
      f"digest `{gate['digest'][:16]}`. How to review: [docs/REVIEW_PROTOCOL.md]({link('docs/REVIEW_PROTOCOL.md', kpath(k))}) and "
      f"[docs/REVIEWER_GUIDE.md]({link('docs/REVIEWER_GUIDE.md', kpath(k))}). The review so far: [REVIEW.md](REVIEW.md).\n")
    w(f"{man.get('title', k)}. The prover's write-up: [README.md](README.md).\n")
    w("## 1. The claim\n")
    w(f"```lean\ntheorem {ns}.main : {surf['main_type']}\ntheorem {ns}.consistent : TVA.Claim.Consistent {ns}.Assumptions\n"
      f"theorem {ns}.satisfiable : TVA.Claim.Satisfiable {ns}.Inst.ok {ns}.Pre\n```\n")
    w("`Spec K` is checked to be `Claim.Meets Inst.ok Pre Post K`: on every instance `Inst.ok` covers and every input "
      "`Pre` allows, every outcome of the device `K` is a normal end whose memory satisfies `Post`. Lean checked "
      "`main`, `consistent` and `satisfiable` (some covered instance and memory exist); you judge the items below.\n")
    w("## 2. The source\n")
    s = man["source"]
    w(f"[`{s['file']}`]({s['file']}) (sha256 `{s['sha256'][:16]}…`), kernel lines {s['ranges']}" +
      (f", host lines {s['host']}" if s.get("host") else "") + ".\n")
    jit = jit or {}
    unused = s.get("unused", [])
    kern = {i for a, b in s["ranges"] for i in range(a, b + 1)}
    groups = {"kernel": [], "host": [], "unused": []}
    for name, a, b in funcs:
        label = (f"`{name}`" if name else "a lambda") + (" (`@triton.jit`)" if [a, b] in jit.get(name, []) else "")
        span = f"{a}" if a == b else f"{a}–{b}"
        cls = "unused" if name in unused else "kernel" if set(range(a, b + 1)) <= kern else "host"
        groups[cls].append(f"{label} {span}")
    w("The functions the source defines (every `def` and `lambda`, from the first decorator to the last line), as the "
      "manifest classes them: in the kernel lines, " + (", ".join(groups["kernel"]) or "none") +
      "; in the host lines, " + (", ".join(groups["host"]) or "none") +
      "; unused (`source.unused`), " + (", ".join(groups["unused"]) or "none") +
      ". The gate checks that every function is in one of these, and every `@triton.jit` function in the kernel lines "
      "or unused. Judge that the claimed launch runs no function outside the kernel lines, on the device or at compile "
      "time, as the kernel or as anything it calls (a `@triton.jit` function under any spelling, a "
      "`triton.constexpr_function`, a plain Python function the front end calls at compile time, through a module or "
      "from a constexpr function), and calls no function the source does not define but Triton's own.\n")
    mod_lines = module_lines("\n".join(src_lines), unused) or []
    host = {i for a, b in s.get("host", []) for i in range(a, b + 1)}
    w("The code outside every function, and the header of each unused function (its decorators and `def` lines), which "
      "run when Python executes the module (a `def` evaluates its decorators, defaults and annotations then), as the "
      "manifest classes them: in the kernel lines, " +
      (_line_list([i for i in mod_lines if i in kern], 40) if any(i in kern for i in mod_lines) else "none") +
      "; in the host lines, " + (_line_list([i for i in mod_lines if i in host and i not in kern], 40)
                                 if any(i in host and i not in kern for i in mod_lines) else "none") +
      ". The gate checks that every such line is in one of these (library issue L79). Judge that every module-level name "
      "the claimed launch reads lies in the kernel lines, mapped there, and is one the supported subset admits (the "
      "imports of `triton` and of `triton.language as tl`, and the module's `@triton.jit` functions, each bound only by "
      "its own `import` or `def`; reading any other module-level value, such as a `tl.constexpr(…)` global or a dtype, "
      "is `UNSUPPORTED`), and that the host lines bind nothing the launch reads.\n")

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
            text = "\n".join(l.rstrip("\r") for l in open(p, encoding="utf-8", errors="replace", newline="").read().split("\n")[a - 1:b])
            w(f"[`{f}:{a}`]({f}#L{a})\n```lean\n{text}\n```\n")
        files = sorted(x[5:] for x in items[item].get("files", {}) if x.startswith("file:"))
        if files:
            w("Library files in this item's digest (whole): " + ", ".join(f"`{x}`" for x in files) + ".\n")
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
    w(f"Digest `{items['map']['digest'][:16]}`. Every code line of the kernel lines {s['ranges']} is mapped to model "
      "text, or dropped with the prover's reason, and every function of the source outside them is host code or unused, "
      "as §2 lists (judge that). What each operation means is [docs/SEMANTICS.md]"
      f"({link('docs/SEMANTICS.md', kpath(k))}); [docs/TRANSCRIPTION.md]({link('docs/TRANSCRIPTION.md', kpath(k))}) "
      "has worked patterns and pitfalls (guidance). Whether each line and each drop is right is yours to judge.\n")
    w("| Source | Model / reason |\n|---|---|")
    for e in mm.get("entries", []):
        try:
            ls = parse_lines(e["src"])
        except (KeyError, ValueError, TypeError):
            continue
        st = " ⏎ ".join(src_lines[i - 1].strip() for i in ls if 0 < i <= len(src_lines)).replace("|", "\\|")
        named = " (" + ", ".join(f"{n}" for n in e.get("rules", [])) + ")" if e.get("rules") else ""
        if "model" in e:
            m = e["model"] if isinstance(e["model"], str) else " ⏎ ".join(map(str, e["model"]))
            right = f"`{m}`".replace("|", "\\|") + named + (f" — {e['note']}" if e.get("note") else "")
        else:
            right = f"**dropped: {e.get('drop')}**" + named + (f": {e['note']}" if e.get("note") else "")
        w(f"| [{e['src']}]({s['file']}#L{ls[0] if ls else 1}): `{st}` | {right} |")
    w("")
    w("## 7. Gate\n")
    w("| Check | | Detail |\n|---|---|---|")
    for c in gate["checks"]:
        w(f"| {c['name']} | {'PASS' if c['ok'] else 'FAIL'} | {md_cell(c['detail'])} |")
    w("\n| Item | Digest |\n|---|---|")
    for i, v in sorted(items.items()):
        w(f"| `{i}` | `{v['digest'][:16]}` |")
    write(kpath(k, "claim.md"), "\n".join(out) + "\n")


# ---------------------------------------------------------------- review commands

def gate_staleness(k, g):
    """Why a passing gate no longer describes the tree, or `[]` (library issues L42, L53), by contents, not
    by file times:
    - a kernel file it read (the manifest, the source, the model map, the write-up, a Lean file) differs
      from the sha256 the gate recorded, or a Lean file was added or removed;
    - the gate's own code (`GATE_TOOL`) differs from what made it;
    - the build's other inputs differ (`build_inputs`, library issue L69): lean/lean-toolchain, the lake
      configuration and manifest, or a library file the kernel's modules import (also one that holds no
      constant of any cone, such as a notation's), or the kernel now imports other library modules;
    - a library file or the packages in its items' digests differ now (its snapshot records them)."""
    if not g.get("ok"):
        return []
    out = []
    rec, tool, build = g.get("files"), g.get("tool"), g.get("build")
    if not isinstance(rec, dict) or not isinstance(tool, dict):
        out.append("it was recorded by a tool older than round 10, which kept no sha256 of the kernel's files and of "
                   "the gate's code")
    else:
        cur = kernel_files(k)
        changed = sorted(p for p in set(rec) | set(cur) if rec.get(p) != cur.get(p))
        if changed:
            out.append("kernel files changed after it ran (" + ", ".join(changed[:4]) + (" …" if len(changed) > 4 else "") + ")")
        if tool != tool_digest():
            out.append("the gate's code changed after it ran (" + ", ".join(p for p in GATE_TOOL if tool.get(p) != _file_digest(p)) + ")")
    if not isinstance(build, dict):
        out.append("it was recorded by a tool older than round 13, which kept no sha256 of the build's other inputs")
    else:
        cur = build_inputs(k)
        changed = sorted(p for p in set(build) | set(cur) if build.get(p) != cur.get(p))
        if changed:
            out.append("the build's inputs changed after it ran (" + ", ".join(changed[:4]) + (" …" if len(changed) > 4 else "") + ")")
    snap = snapshot(k, g.get("digest", ""))["items"]
    if not snap:
        return out + ["its snapshot (review/snapshots) is missing"]
    if any("packages" not in v.get("cone", {}) for v in snap.values()):
        # every item of a gate since round 6 records its library files and the packages' digest
        return out + ["it was recorded by a tool older than whole-file digests (round 6), so it records no library files"]
    changed = set()
    for v in snap.values():
        for key, h in v.get("cone", {}).items():
            if key.startswith("file:") and _file_digest(key[5:]) != h:
                changed.add(key)
            elif key == "packages" and packages_digest() != h:
                changed.add(key)
    if changed:
        out.append("library files changed after it ran (" + ", ".join(sorted(changed)[:4]) +
                   (" …" if len(changed) > 4 else "") + ")")
    return out


def current_gate(k):
    g = load_json(kpath(k, "gate.json"))
    if not g["ok"]:
        die("the gate fails; run `tvr check` after fixing it")
    stale = gate_staleness(k, g)
    if stale:
        die(f"the gate is stale ({'; '.join(stale)}); run `tvr check`")
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
    # every file of the reviewer's folder is recorded, work files included, and the folder too, so that a
    # file changed or added after recording shows (library issue L52); a file recorded before (a second
    # filing in the round) must be unchanged
    files = archive_entry(path, report, *text_files(os.path.dirname(path)))
    refrozen = sorted(p for p, h in files.items() if S["archive"].get(p, h) != h)
    if refrozen:
        die("recorded files are frozen, and these changed after recording: " + ", ".join(refrozen))
    events = [{"type": "file", "actor": who, "round": N, "gate": gate["digest"], "path": rel(path),
               "report": rel(report) if os.path.exists(report) else "", "notes": filing.get("notes", ""),
               "dir": rel(os.path.dirname(path)), "files": files}]
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
    released = set()
    for r in filing.get("release", []):
        if r["id"] not in findings or who not in findings[r["id"]]["raisers"]:
            die(f"{who} cannot release {r['id']} (not raised by {who})")
        released.add(r["id"])
    reraised = set()
    for r in filing.get("reraise", []):
        f = findings.get(r["id"])
        if not f or who not in f["raisers"]:
            die(f"{who} cannot re-raise {r['id']} (not raised by {who})")
        if f["state"] not in ("ANSWERED", "REFERRED"):
            die(f"{r['id']} is {f['state']}; only an answered or referred finding can be re-raised")
        reraised.add(r["id"])
    # the events the filing gives: the same function derives them again from the recorded filing in the
    # release check (library issue L52)
    derived = filing_events(S, who, N, filing, gate["digest"], gate["items"])
    acc = filing.get("accept", [])
    if acc:
        names = list(gate["items"]) if acc == "all" else [a[5:] if a.startswith("item:") else a for a in acc]
        own = [f for i, f in findings.items() if who in f["raisers"] and f["blocking"]
               and ((f["state"] not in ("CLOSED", "REFERRED") and i not in released) or i in reraised)]
        own += [e for e in derived if e["type"] == "raise" and e["blocking"]]
        for n in names:
            if n not in gate["items"]:
                die(f"no item {n}")
            block = sorted({f["id"] for f in own if n in affected(k, f, gate["items"])})
            if block:
                die(f"{who} cannot accept {n} while their blocking finding(s) on it are open: {', '.join(block)}")
    events += derived
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
    files = archive_entry(path, notes, *text_files(os.path.dirname(path)))
    refrozen = sorted(p for p, h in files.items() if S["archive"].get(p, h) != h)
    if refrozen:
        die("recorded files are frozen, and these changed after recording: " + ", ".join(refrozen))
    events = [{"type": "response", "actor": "prover", "round": N, "gate": gate["digest"], "path": rel(path),
               "notes": rel(notes) if os.path.exists(notes) else "", "dir": rel(os.path.dirname(path)),
               "files": files}]
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
        answered.add(r["id"])
    left = sorted((i for i, f in findings.items() if f["state"] == "OPEN" and i not in answered), key=lambda x: int(x[1:]))
    if left:
        die(f"one batch answers every open finding (the batch policy); not answered: {', '.join(left)}")
    events += response_events(N, resp, gate["digest"])  # derived again in the release check (library issue L52)
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
    entry = archive_entry(*[os.path.join(ROOT, f) for f in files])
    inside = sorted(p for p in entry if role_folder(k, p))
    if inside:
        # a note never accounts for a role folder (library issue L81): its files are its role's own records
        die("a note records no file of a role folder (" + ", ".join(inside[:3]) + "): only its role's `tvr file` or "
            "`tvr respond` records those (and `tvr freeze` the work files of a folder recorded before the tool named folders)")
    append(k, [{"type": "note", "actor": "coordinator", "round": S["round"], "text": text, "files": entry}])
    refresh(k)


def legacy_folders(k, root=None):
    """The role folders of a kernel whose own events (`own_records`) all come from before the tool named folders
    (no line names the folder as `dir`), frozen by no `freeze` yet, with the files in them that those events leave
    out (library issue L81): `{folder: [paths]}`, only folders that leave some out."""
    root = root or ROOT
    own = own_records(k, ledger(k))
    out = {}
    for f, o in sorted(own.items()):
        if o["dir"] or o["frozen"]:
            continue
        left = [q for q in (os.path.relpath(x, root).replace(os.sep, "/") for x in text_files(os.path.join(root, f)))
                if q not in o["files"]]
        if left:
            out[f] = left
    return out


def cmd_freeze(k, dry=False):
    """`tvr freeze KERNEL [--dry-run]` (library issue L81): record, once, the work files of each role folder whose
    filing or response the ledger recorded before the tool named folders (a `file` or `response` line without
    `dir`), and which that line left out. A role folder that its own events record is frozen whether or not a
    line names it, so such a folder fails the records step and the certificate until its files are recorded;
    this is the supported way to bring such a ledger to a passing state, appending one `freeze` event per folder
    (the coordinator's), never editing a line. Only work files (in a subfolder of the role folder, such as
    `work/`) are recorded this way: a filing, report, response or notes left unrecorded is refused, since only its
    role's own `tvr file` or `tvr respond` records it, with the events it gives. The coordinator runs it after
    judging that the files are the role's (they give no event); `--dry-run` prints what it would record."""
    found = legacy_folders(k)
    if not found:
        print(f"nothing to freeze: every role folder of {k} that its events record holds only files they record, or "
              "names its folder")
        return 0
    records = {f: [p for p in ps if not is_work_file(f, p)] for f, ps in found.items()}
    if any(records.values()):
        die("these files lie at the top of a role folder, where only a filing, report, response or notes does; only "
            "their role's `tvr file` or `tvr respond` records them: " + ", ".join(p for ps in records.values() for p in ps))
    S = state(k)
    events = [{"type": "freeze", "actor": "coordinator", "round": S["round"], "dir": f,
               "files": archive_entry(*[os.path.join(ROOT, p) for p in ps])} for f, ps in found.items()]
    for e in events:
        print(f"{'would freeze' if dry else 'froze'} {e['dir']}: " + ", ".join(sorted(e["files"])))
    if dry:
        return 0
    append(k, events)
    refresh(k)
    print(status_text(k))
    return 0


# ---------------------------------------------------------------- views

def kernel_record_problems(k, root=None):
    """What the kernel's ledger does not support (library issues L52, L81): a recorded file changed or missing
    (`ledger_problems`), an event no recorded file gives (`kernel_event_problems`), or a role folder the events
    do not account for (`kernel_folder_problems`: a reviewer no `assign` names, an earlier round's folder that no
    filing or response records, a file of a recorded folder that none of its own events records)."""
    root = root or ROOT
    return (ledger_problems(root, f"kernels/{k}/review/ledger.jsonl", "the ledger", folders=False) +
            kernel_event_problems(root, k) + kernel_folder_problems(root, k))


def unrecorded_current(k, root=None):
    """The current round's role folders that no event records yet (library issue L81): a filing or response in
    progress. The certificate waits for them, so a filing whose ledger lines were removed cannot leave the
    certificate granted beside it."""
    return [p for p in kernel_folder_problems(root or ROOT, k, current=True) if p.endswith("not recorded yet")]


def cert_state(k, S, gate):
    """The certificate needs a passing gate that still describes the tree (`gate_staleness`, library issue
    L42), every item accepted by every reviewer at its digest, no unresolved blocking finding, a ledger
    whose every event follows from its recorded files, as they were recorded, and whose events account for
    every role folder (`kernel_record_problems`, library issues L52, L81), and no filing or response of the
    current round left unrecorded (`unrecorded_current`, L81)."""
    blocking = sorted((i for i, f in S["findings"].items() if f["blocking"] and f["state"] not in ("CLOSED", "REFERRED")),
                      key=lambda x: int(x[1:]))
    full = bool(S["reviewers"]) and gate["ok"] and not gate_staleness(k, gate) and all(
        S["accepts"].get(r, {}).get(i) == d for r in S["reviewers"] for i, d in gate["items"].items()) \
        and not kernel_record_problems(k) and not unrecorded_current(k)
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
                     f"| {f['round']} | {md_cell(f['summary'][:120])} |")
        L.append("")
    cert, blocking, full = cert_state(k, S, gate)
    stale = gate_staleness(k, gate)
    recs = kernel_record_problems(k)
    pending = unrecorded_current(k)
    L.append(f"**Certificate: {'GRANTED' if cert else 'not yet'}**" + ("" if cert else (
        f" — {len(blocking)} blocking finding(s) unresolved" + ("" if full else
        f"; the gate is stale ({'; '.join(stale)}): run `tvr check`" if stale else
        f"; the ledger does not hold to its records ({'; '.join(recs[:2])})" if recs else
        f"; a record of this round is not recorded yet ({'; '.join(pending[:2])})" if pending else
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
    w(f"> Generated by `tvr` from [review/ledger.jsonl](review/ledger.jsonl); do not edit. "
      f"Protocol: [docs/REVIEW_PROTOCOL.md]({link('docs/REVIEW_PROTOCOL.md', kpath(k))}).\n")
    w(f"{man.get('title', k)}. The claim sheet: [claim.md](claim.md); the prover's write-up: [README.md](README.md).\n")
    cert, blocking, _ = cert_state(k, S, gate)
    w("## Status\n")
    stale = gate_staleness(k, gate)
    w(f"- Gate: **{'PASS' if gate['ok'] else 'FAIL'}**, digest `{gate['digest'][:16]}` ({gate.get('time', '')})" +
      (f", **stale**: {'; '.join(stale)}." if stale else "."))
    w(f"- Phase: {phase(k, S)}.")
    w(f"- Reviewers: {', '.join(f'{r} ({fo})' for r, fo in S['reviewers'].items()) or 'none'}.")
    w(f"- Certificate: **{'GRANTED' if cert else 'not yet'}**.")
    bad = kernel_record_problems(k)
    w(f"- Recorded files: {'all unchanged since they were recorded, and every event follows from them' if not bad else '**the ledger does not hold to its records: ' + '; '.join(bad) + '**'}.\n")
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
            elif t == "freeze":
                w(f"- `{e['time']}` the coordinator froze the work files of `{e.get('dir', '')}` that its events, recorded "
                  "before the tool named folders, left out (`tvr freeze`): " +
                  ", ".join(f"[{os.path.relpath(p, e.get('dir', ''))}]({link(p, kpath(k))})" for p in sorted(e.get("files", {}))))
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
        g = ("PASS" + (" (stale)" if gate_staleness(k, gate) else "")) if gate["ok"] else "FAIL"
        out.append(f"| [{k}]({k}/README.md) | {man.get('title', '')} | {g} | "
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
                       f"{md_cell(f['summary'][:120])} | {ref} |")
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
        name = {"author": "library-author.md", "reviewer": "library-reviewer.md",
                "coverage": "library-coverage.md"}.get(role)
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
        die("roles: prover, reviewer (kernel); author, reviewer, coverage (library)")
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


COVERAGE_STATUS = ("covered", "gap", "excluded")


def coverage_functions():
    """Every `@triton.jit` function of the pinned tutorials, as (source file, name), in order."""
    inv = load_json(os.path.join(ROOT, "tutorials", "inventory.json"))
    return [(t["source"], f["name"]) for t in inv["tutorials"] for f in t["functions"]]


def coverage_path(rnd):
    return os.path.join(LREVIEW, f"round-{rnd}", "coverage", "matrix.md")


def lint_coverage(rnd, path):
    """The tutorial coverage matrix (library issue L37): one section per tutorial function, each a
    table of constructs with a status, or one line `Excluded: L<n> …`; then the new gaps."""
    P = Problems()
    name = rel(path) if os.path.isabs(path) else path
    if not os.path.exists(path):
        P.append(f"{name}: missing")
        return P
    text = open(path).read()
    title, secs = md_sections(text)
    P.need(title == f"Tutorial coverage, round {rnd}", f"{name}: the title must be `# Tutorial coverage, round {rnd}`")
    P.need("TODO" not in text, f"{name}: placeholders (TODO) are left")
    want = [f"{src}: {fn}" for src, fn in coverage_functions()] + ["New gaps"]
    got = [h for h, _ in secs]
    P.need(got == want, f"{name}: the sections must be one per tutorial function, in inventory order, then "
           f"`New gaps` (missing: {', '.join(h for h in want if h not in got) or 'none'}; "
           f"unexpected: {', '.join(h for h in got if h not in want) or 'none'})")
    news = 0
    for h, body in secs:
        if h == "New gaps":
            bs = listed(P, name, {h: body}, h, "None.")
            for b in bs:
                P.need(re.match(r"\*\*NEW\*\* (soundness|gap|wording) — \S", b) is not None,
                       f"{name} § New gaps: `- **NEW** soundness|gap|wording — <summary>`: {b[:60]!r}")
            P.need(len(bs) == news, f"{name} § New gaps: one bullet per `NEW` row in the tables ({news} rows, {len(bs)} bullets)")
            continue
        rows = [l for l in body if l.startswith("|")]
        excl = [l for l in body if l.startswith("Excluded: ")]
        if excl and not rows:
            P.need(re.match(r"Excluded: L\d+\b", excl[0]) is not None, f"{name} § {h}: `Excluded: L<n> …` names the decision")
            continue
        P.need(len(rows) >= 3 and rows[0].replace(" ", "") == "|Construct|Status|Where|",
               f"{name} § {h}: a table `| Construct | Status | Where |` with at least one row, or `Excluded: L<n> …`")
        for r in rows[2:]:
            cells = [c.strip() for c in r.strip().strip("|").split("|")]
            if len(cells) != 3:
                P.append(f"{name} § {h}: a row needs three cells: {r[:60]!r}")
                continue
            what, st, where = cells
            P.need(st in COVERAGE_STATUS, f"{name} § {h}: status is covered, gap or excluded: {r[:60]!r}")
            P.need(what != "" and where != "", f"{name} § {h}: construct and where are filled in: {r[:60]!r}")
            if st == "gap":
                P.need(re.match(r"(L\d+|NEW:)", where) is not None, f"{name} § {h}: a gap cites `L<n>` or starts `NEW:`: {r[:60]!r}")
                news += where.startswith("NEW:")
            if st == "excluded":
                P.need(re.search(r"\bL\d+\b|exclusions\.md", where) is not None,
                       f"{name} § {h}: an exclusion cites its decision (`L<n>` or exclusions.md): {r[:60]!r}")
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
        if who == "coverage":
            put(coverage_path(rnd), f"# Tutorial coverage, round {rnd}\n\n" + "".join(
                f"## {src}: {fn}\n\n| Construct | Status | Where |\n|---|---|---|\n| TODO | TODO | TODO |\n\n"
                for src, fn in coverage_functions()) + "## New gaps\n\n- **NEW** gap — TODO (or replace the list with None.)\n")
        elif who == "author":
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
            if who not in state(k)["reviewers"]:
                die(f"{who} is not assigned to {k}: the coordinator assigns reviewers (`tvr assign {k} {who}`), and a folder "
                    "of a reviewer no assignment names fails the records step (library issue L81)")
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


def library_lint_problems(rnd, who):
    """What `tvr lint library ROUND WHO` reports (and `tvr record` checks first)."""
    if who == "coverage":
        path = coverage_path(rnd)
        P = lint_coverage(rnd, path)
        P += scan_files([path])
    elif who == "author":
        path = os.path.join(LREVIEW, f"round-{rnd}", "author", "notes.md")
        P = lint_author_notes(rnd, path)
        P += scan_files([path])
    else:
        path = os.path.join(LREVIEW, f"round-{rnd}", who, "report.md")
        P = lint_library_report(rnd, who, path)
        P += scan_files([path] + text_files(os.path.join(os.path.dirname(path), "work")))
    return P


def cmd_lint(argv):
    if argv[2] == "library":
        P = library_lint_problems(int(argv[3]), argv[4])
    else:
        k, role = argv[2], argv[3]
        gate = load_json(kpath(k, "gate.json"), {"items": {}, "digest": ""})
        if role == "reviewer":
            who = argv[4]
            if who not in state(k)["reviewers"]:
                # a folder of a reviewer no `assign` names is no record (library issue L81): `tvr file` refuses it,
                # and the records step and the certificate fail on it
                print(f"not regular:\n  - {who} is not assigned to {k} (`tvr assign {k} {who}` is the coordinator's), so "
                      f"no filing of {who} can be recorded")
                return 1
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


# ---------------------------------------------------------------- the release check

# The release check (protocol §2, library issues L36, L7, L40, L42, L52, L53). TVA will be released on its
# own, so its checks must pass with nothing but what it ships. The sandbox is built from an allowlist, so
# that it is complete by construction: it shows what TVA ships and an explicit, read-only list of what the
# toolchain needs, and nothing else exists in it.
# - TVA: exactly the files TVA ships (what git tracks, and the untracked files git does not ignore, read
#   with no user or system git configuration), copied into an empty folder mounted at TVA's own path in an
#   empty home directory. No shipped file may lie in a folder the text scan skips (`shipped_problems`).
# - Mathlib and its dependencies, made afresh in the copy from their upstream repositories (`make_packages`,
#   library issue L7): Mathlib's `inputRev` must be the release tag of the toolchain's Lean or `master`
#   (`mathlib_ref`), and its pinned commit that tag's, or `master`'s head or a commit reachable from it, as
#   the upstream refs show; every inherited package must be the one Mathlib's own
#   manifest at that commit pins; each is fetched from its URL only, by git with no user or system
#   configuration, and every file of the checkout is compared with the commit's blobs. The sandbox builds
#   Mathlib from that source (about 20 minutes on 96 cores, about 13 GB).
# - The system directories `/usr` and `/etc` (with `/bin`, `/sbin`, `/lib`, `/lib64`, which are links into
#   `/usr` here), read-only; the check fails if anything in them is writable by the user or owned by a
#   regular user (`system_problems`), so nothing can have been placed there for the check to find.
# - The Lean toolchain that lean/lean-toolchain pins (`~/.elan/toolchains/<it>`, with elan's `bin` and
#   `settings.toml`), read-only. It supplies the core modules `Init`, `Std` and `Lean`, which every module
#   reaches, and Lean, Lake and `leanchecker` themselves; it sits in the user's home and is trusted as
#   installed. The name must be a release of Lean (`lake_violations`, library issue L54), the same one
#   Mathlib's commit pins, and no part may be a symbolic link (`elan_parts`).
# - The tool runs on the system's Python (a `/usr/bin/python3.N`, N ≥ 11) with `-I -S`: its standard
#   library is the system's, so no user's file is importable but the copy's own.
# - `/dev`, `/proc` and an empty `/tmp`; no network.
# Outside the sandbox it runs only the system's git, bwrap and find (`system_tool`: `/usr/bin/…`, under /usr,
# not writable by the user), with a PATH of the system's folders and nothing else of the caller's environment;
# git runs in an empty folder of the tool's own and reads no repository's configuration but the one it is
# given (`_git`: each new package repository, and TVA's own to list what ships).
# There it builds the library, re-checks it (`tvr library-check`: hygiene, `leanchecker TVA`), checks every
# ledger against the records (and each kernel's events against the files they follow from), lints the
# records not yet recorded and runs `tvr audit`; release.json keeps each ledger's length and sha256. It gates no
# kernel (library issue L160): the kernels' shipped gates, certificates and generated pages are not compared. A dependency on
# anything outside TVA, Mathlib's pinned source, the Lean toolchain and the system fails in these steps,
# however it is spelled or computed. Evidence scripts and reviewers' work scripts are not run (they need
# Triton and the GPU); their text is checked by the audit's scan. Without `bwrap` the check fails (it
# cannot build the sandbox); with --no-mask only HOME changes, and the check fails its `sandbox` step (a
# run for testing, not a release check).
SANDBOX_SYSTEM = ("usr", "etc", "bin", "sbin", "lib", "lib64")  # shown read-only, from the root
SANDBOX_FREE = 16 * 2 ** 30  # Mathlib is built in the sandbox
_GIT_HOME = []


# The programs the tool runs outside the sandbox are the system's (library issue L7): `/usr/bin/git`,
# `/usr/bin/bwrap`, `/usr/bin/find`, each a file under /usr that the user cannot write and that no regular
# user owns, never one the caller's PATH finds first (a user's git would decide the packages and the refs
# they are checked against, a user's bwrap the sandbox). They run with a PATH of the system's folders only
# and with nothing else of the caller's environment (no `LD_PRELOAD`), git with the network settings kept, in
# an empty folder, finding no repository but the one it is given (`_git`).
SYSTEM_BIN = "/usr/bin"
SYSTEM_PATH = "/usr/bin:/bin"
SYSTEM_ENV = {"PATH": SYSTEM_PATH, "LANG": "C.UTF-8"}


def system_tool(name):
    """`/usr/bin/<name>` when it is the system's program: it resolves to an executable file under /usr that
    the user cannot write and that no regular user or group (id 1000 or more) owns; else `None`."""
    p = os.path.join(SYSTEM_BIN, name)
    real = os.path.realpath(p)
    try:
        st = os.stat(real)
    except OSError:
        return None
    if not (real.startswith("/usr/") and os.path.isfile(real) and os.access(real, os.X_OK)) or \
            os.access(real, os.W_OK) or st.st_uid > 999 or st.st_gid > 999:
        return None
    return p


def _git_home():
    """An empty folder of the tool's own (made once per run, removed at exit): git's `HOME` and
    `XDG_CONFIG_HOME`, holding nothing but the empty folder `cwd`, where every git runs (`_git`)."""
    if not _GIT_HOME:
        import atexit
        import tempfile
        d = os.path.realpath(tempfile.mkdtemp(prefix="tvr-git-home-"))
        os.mkdir(os.path.join(d, "cwd"))
        _GIT_HOME.append(d)
        atexit.register(shutil.rmtree, d, True)
    return _GIT_HOME[0]


def _git_env():
    """The environment of every git the tool runs (library issue L7): no user or system configuration
    (`GIT_CONFIG_GLOBAL=/dev/null`, `GIT_CONFIG_NOSYSTEM`, `GIT_ATTR_NOSYSTEM`, and an empty folder as
    `HOME` and `XDG_CONFIG_HOME`, so no global attributes or excludes file), no `GIT_*` variable of the
    caller's, no prompt, and a PATH of the system's folders only (`SYSTEM_PATH`), for the helpers git runs
    itself; the network settings (proxies) are kept. `_git` adds `GIT_CEILING_DIRECTORIES`."""
    home = _git_home()
    keep = {k: v for k, v in os.environ.items()
            if k in ("LANG", "LC_ALL", "TZ") or k.lower() in ("http_proxy", "https_proxy", "no_proxy", "all_proxy")}
    return {**keep, "PATH": SYSTEM_PATH, "HOME": home, "XDG_CONFIG_HOME": home,
            "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "GIT_ATTR_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0", "GIT_NO_LAZY_FETCH": "1"}


# the system's git (`system_tool`), with nothing of the repository's own configuration that changes what it
# writes or reads: no hooks, no attributes or excludes file, no line-ending conversion, and only https as a
# transport
GIT = [os.path.join(SYSTEM_BIN, "git"), "-c", "core.hooksPath=/dev/null", "-c", "core.attributesFile=/dev/null",
       "-c", "core.excludesFile=/dev/null", "-c", "core.autocrlf=false", "-c", "core.eol=lf", "-c", "core.symlinks=true",
       "-c", "core.fsmonitor=false", "-c", "protocol.allow=never", "-c", "protocol.https.allow=always"]


def _git(*args):
    """Run the system's git (`GIT`, `_git_env`). Git reads the configuration of the repository it finds from the
    folder it runs in (a `url.<base>.insteadOf` there redirects every URL, a proxy or `http.sslVerify` there
    decides what answers), so it finds none but the one it is given (library issue L7): every git runs in the
    empty folder `cwd` of `_git_home`; one given a repository (`-C <repo>`, always the first arguments: a new
    package repository, or TVA's own for `shipped_files`) gets `GIT_CEILING_DIRECTORIES` at the folder above
    `<repo>`, so it reads that repository alone, and every other one (`ls-remote`, `clone`, `init`,
    `--version`) gets it at `_git_home`, above `cwd`, so it reads no repository's configuration."""
    if system_tool("git") is None:
        return subprocess.CompletedProcess(GIT + list(args), 127, b"",
                                           b"/usr/bin/git is not the system's git (library issue L7)")
    home = _git_home()
    env = _git_env()
    repo = len(args) >= 2 and args[0] == "-C"
    env["GIT_CEILING_DIRECTORIES"] = os.path.dirname(os.path.realpath(args[1])) if repo else home
    return subprocess.run(GIT + list(args), cwd=os.path.join(home, "cwd"), capture_output=True, env=env)


def shipped_files():
    """The files TVA ships, relative to the TVA root: what git tracks, and the untracked files git does
    not ignore (`git ls-files --cached --others --exclude-standard`, with no user or system configuration,
    library issue L7), that exist. An entry ending in `/` is a nested repository git lists as a folder
    (`shipped_problems` refuses it)."""
    r = _git("-C", ROOT, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    if r.returncode != 0:
        die("git ls-files failed: " + r.stderr.decode(errors="replace").strip()[:200])
    names = sorted({f for f in r.stdout.decode().split("\0") if f})
    return [f for f in names if f.endswith("/") or os.path.lexists(os.path.join(ROOT, f))]


def shipped_problems(files):
    """Shipped paths the text scan would not read (library issue L41): a file inside a folder it skips
    (`SKIP_DIRS`: `.lake`, `.venv`, `.git`, `__pycache__`, at any depth), and a nested git repository,
    which git lists as a folder and whose files it does not ship."""
    bad = []
    for f in files:
        parts = f.rstrip("/").split("/")
        if f.endswith("/"):
            bad.append(f"{f}: a nested git repository inside TVA (git ships none of its files)")
        elif set(parts) & SKIP_DIRS:
            bad.append(f"{f}: shipped inside a folder the text scan skips ({', '.join(sorted(set(parts) & SKIP_DIRS))})")
    return bad


def mathlib_ref(toolchain):
    """The `inputRev`s Mathlib may be pinned at (library issue L7), for `lean/lean-toolchain`'s text: the
    release tag of the toolchain's Lean (`v4.35.0` for `leanprover/lean4:v4.35.0`), as a tag, or the branch
    `master`. Any other branch of Mathlib's repository (it has dozens, personal ones and CI ones among
    them) is not reviewed Mathlib. Returns `{inputRev: "tag" | "branch"}`."""
    tc = (toolchain or "").strip()
    out = {"master": "branch"}
    if TOOLCHAIN.fullmatch(tc):
        out[tc.split(":", 1)[1]] = "tag"
    return out


def upstream_problem(url, ref, rev, scratch, kind=None):
    """Whether `rev` is the commit that `ref` (a manifest's `inputRev`) names in the repository at `url`
    (library issue L7), or `None`: the tag `ref`'s commit, or the branch `ref`'s head or a commit reachable
    from it; with `kind` "tag" or "branch", only that kind of ref counts (`mathlib_ref`). A commit fetched
    by its hash alone is not shown to be the repository's: a host may serve a commit that no branch or tag
    of that repository holds. `scratch` is an empty folder for the history of a branch (commits only, no
    files)."""
    r = _git("ls-remote", url, f"refs/tags/{ref}", f"refs/heads/{ref}")
    if r.returncode != 0:
        return f"cannot read the refs of {url} ({r.stderr.decode(errors='replace').strip()[:120]})"
    refs = {}
    for line in r.stdout.decode(errors="replace").splitlines():
        oid, _, name = line.partition("\t")
        refs[name.strip()] = oid.strip()
    tag, head = refs.get(f"refs/tags/{ref}"), refs.get(f"refs/heads/{ref}")
    if kind == "tag":
        head = None
    elif kind == "branch":
        tag = None
    if not tag and not head:
        return f"inputRev {ref!r} names no {kind or 'tag or branch'} of {url}"
    if rev in (tag, head):
        return None
    if tag:  # an annotated tag: the commit it names
        repo = os.path.join(scratch, "tag")
        _git("init", "-q", "--bare", "--template=", repo)
        if _git("-C", repo, "fetch", "-q", "--no-tags", "--depth", "1", url, f"refs/tags/{ref}").returncode == 0 and \
                _git("-C", repo, "rev-parse", "FETCH_HEAD^{commit}").stdout.decode().strip() == rev:
            return None
    if head:  # the branch's history, commits only (no trees, no files), and whether it holds `rev`
        repo = os.path.join(scratch, "branch")
        c = _git("clone", "-q", "--bare", "--template=", "--single-branch", "--no-tags", "--filter=tree:0",
                 "--branch", ref, url, repo)
        if c.returncode == 0 and _git("-C", repo, "merge-base", "--is-ancestor", rev, f"refs/heads/{ref}").returncode == 0:
            return None
    return f"commit {rev[:12]} is not {url}'s {ref!r}" + (" (nor reachable from the branch)" if head else "")


def checkout_problems(dst, rev):
    """Every file of the checkout at `dst` is the commit `rev`'s blob, byte for byte, and no other file is
    there (`.git` aside; library issue L7): each regular file's and symbolic link's git blob id, computed
    here from its bytes (`sha1("blob <size>\\0" + bytes)`), is the one `git ls-tree` lists. So neither a
    filter, an attribute nor anything else between git and the disk changes what the build reads."""
    r = _git("-C", dst, "ls-tree", "-r", "-z", "--full-tree", rev)
    if r.returncode != 0:
        return [f"cannot list the commit's files ({r.stderr.decode(errors='replace').strip()[:120]})"]
    want = {}
    for ent in r.stdout.split(b"\0"):
        if not ent:
            continue
        meta, path = ent.split(b"\t", 1)
        mode, typ, oid = meta.split()
        if typ != b"blob":
            return [f"{path.decode(errors='replace')} is a {typ.decode()} (a submodule), which the release check does not make"]
        want[path] = (mode, oid.decode())
    have = set()
    root = os.fsencode(dst)
    for dirpath, dirnames, names in os.walk(root):
        if dirpath == root:
            dirnames[:] = [n for n in dirnames if n != b".git"]
        names = names + [n for n in dirnames if os.path.islink(os.path.join(dirpath, n))]
        dirnames[:] = [n for n in dirnames if not os.path.islink(os.path.join(dirpath, n))]
        have |= {os.path.relpath(os.path.join(dirpath, n), root) for n in names}
    bad = [f"{p.decode(errors='replace')}: not a file of the commit" for p in sorted(have - set(want))]
    bad += [f"{p.decode(errors='replace')}: a file of the commit, missing" for p in sorted(set(want) - have)]
    for p in sorted(set(want) & have):
        mode, oid = want[p]
        full = os.path.join(root, p)
        link = os.path.islink(full)
        if link != (mode == b"120000"):
            bad.append(f"{p.decode(errors='replace')}: a {'link' if link else 'file'} where the commit has a "
                       f"{'link' if mode == b'120000' else 'file'}")
            continue
        data = os.readlink(full) if link else open(full, "rb").read()
        if hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() != oid:
            bad.append(f"{p.decode(errors='replace')}: differs from the commit's blob")
    return bad


def inherited_problems(man, mathlib_man):
    """TVA's inherited packages are exactly the packages Mathlib's own lake-manifest.json (at the pinned
    commit) pins, field for field but `inherited` (library issue L7): the same names, URLs, revisions,
    input revisions and the rest."""
    def table(ps, only_inherited):
        return {p.get("name"): {k: v for k, v in p.items() if k != "inherited"}
                for p in ps if isinstance(p, dict) and (p.get("inherited") or not only_inherited)}
    ours, theirs = table(man.get("packages", []), True), table(mathlib_man.get("packages", []), False)
    bad = []
    for n in sorted(set(ours) | set(theirs), key=str):
        if n not in theirs:
            bad.append(f"{n}: an inherited package that Mathlib's manifest does not pin")
        elif n not in ours:
            bad.append(f"{n}: a package of Mathlib's manifest that TVA's does not pin")
        elif ours[n] != theirs[n]:
            diff = sorted(k for k in set(ours[n]) | set(theirs[n]) if ours[n].get(k) != theirs[n].get(k))
            bad.append(f"{n}: {', '.join(diff)} differ from Mathlib's manifest")
    return bad


def make_packages(lean_dir, scratch):
    """Make each package of lean/lake-manifest.json afresh in `lean_dir`/.lake/packages (library issue L7):
    - Mathlib's pinned commit must be what its `inputRev` names upstream (`upstream_problem`);
    - every inherited package must be the one Mathlib's own manifest at that commit pins
      (`inherited_problems`), and TVA's lean-toolchain the one Mathlib's commit pins;
    - each package is a new repository (no template) that fetches exactly the pinned commit from the
      manifest's URL only (https, no other transport) and checks it out, by git with no user or system
      configuration (`_git_env`, `GIT`) and no repository's but the new one's (`_git`: the ref lookups and the
      clone of a branch's history run in an empty folder), and every file of the checkout is the commit's blob
      (`checkout_problems`).
    `scratch` is an empty folder for the ancestry check. Returns the problems."""
    bad = []
    man = load_json(os.path.join(ROOT, "lean", "lake-manifest.json"), {"packages": []})
    pkgs = [p for p in man.get("packages", []) if isinstance(p, dict)]
    pkgs.sort(key=lambda p: bool(p.get("inherited")))  # Mathlib first: the others are checked against its manifest
    direct = [p.get("name") for p in pkgs if not p.get("inherited")]
    if direct != ["mathlib"]:
        return [f"lean/lake-manifest.json: the direct packages are {direct}, not Mathlib alone"]
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_./-]*", str(pkgs[0].get("inputRev", ""))):
        return [f"lean/lake-manifest.json: Mathlib's inputRev {str(pkgs[0].get('inputRev'))[:60]!r} is not a tag or branch name"]
    try:
        toolchain = open(os.path.join(ROOT, "lean", "lean-toolchain"), encoding="utf-8").read()
    except (OSError, UnicodeDecodeError):
        toolchain = ""
    refs = mathlib_ref(toolchain)
    if str(pkgs[0].get("inputRev")) not in refs:
        return [f"lean/lake-manifest.json: Mathlib's inputRev {str(pkgs[0].get('inputRev'))[:60]!r} is neither the release "
                f"tag of the toolchain's Lean nor master ({', '.join(sorted(refs))}; library issue L7)"]
    mathlib_man = None
    for p in pkgs:
        name, rev, url = str(p.get("name", "")), str(p.get("rev", "")), str(p.get("url", ""))
        if not (re.fullmatch(r"[0-9a-f]{40}", rev) and re.fullmatch(r"[A-Za-z0-9_.-]+", name) and name not in (".", "..")
                and re.fullmatch(r"https://[A-Za-z0-9.-]+/[A-Za-z0-9_.~/-]+", url)):
            bad.append(f"{name or '?'}: lake-manifest.json gives no package name, 40-digit revision and https URL")
            continue
        if not p.get("inherited"):
            why = upstream_problem(url, str(p.get("inputRev", "")), rev, os.path.join(scratch, name),
                                   refs[str(p.get("inputRev", ""))])
            if why:
                return bad + [f"{name}: {why}", "the inherited packages are not made, since Mathlib's commit is not shown"]
        elif mathlib_man is None:
            return bad + ["the inherited packages are not made, since Mathlib's manifest could not be read"]
        dst = os.path.join(lean_dir, ".lake", "packages", name)
        r = _git("init", "-q", "--template=", dst)
        f = _git("-C", dst, "fetch", "-q", "--depth", "1", "--no-tags", url, rev) if r.returncode == 0 else r
        if f.returncode != 0:
            bad.append(f"{name}: commit {rev[:12]} not fetched from {url} ({f.stderr.decode(errors='replace').strip()[:120]})")
            continue
        c = _git("-C", dst, "-c", "advice.detachedHead=false", "checkout", "-q", "--detach", rev)
        head = _git("-C", dst, "rev-parse", "HEAD").stdout.decode().strip()
        if c.returncode != 0 or head != rev:
            bad.append(f"{name}: checkout of {rev[:12]} failed ({c.stderr.decode(errors='replace').strip()[:120]})")
            continue
        cp = checkout_problems(dst, rev)
        if cp:
            bad.append(f"{name}: the checkout is not the commit's files: " + "; ".join(cp[:3]) + (f" (+{len(cp) - 3} more)" if len(cp) > 3 else ""))
            if not p.get("inherited"):
                return bad + ["the inherited packages are not made, since Mathlib's manifest is not its commit's"]
            continue
        if not p.get("inherited"):
            mathlib_man = load_json(os.path.join(dst, "lake-manifest.json"), {"packages": []})
            bad += [f"lean/lake-manifest.json: {x}" for x in inherited_problems(man, mathlib_man)]
            try:
                theirs = open(os.path.join(dst, "lean-toolchain"), encoding="utf-8").read().strip()
                ours = open(os.path.join(ROOT, "lean", "lean-toolchain"), encoding="utf-8").read().strip()
            except OSError:
                theirs, ours = None, ""
            if theirs != ours:
                bad.append(f"lean/lean-toolchain: {ours!r}, while Mathlib's commit pins {theirs!r}")
            if bad:
                break
    return bad


def system_python():
    """A Python of the system, 3.11 or later (for `tomllib`), under /usr: the tool runs on it inside the
    sandbox, so its standard library is root's and holds nothing of another project's."""
    found = []
    for d in ("/usr/bin", "/usr/local/bin"):
        for n in (os.listdir(d) if os.path.isdir(d) else []):
            m = re.fullmatch(r"python3(?:\.(\d+))?", n)
            p = os.path.join(d, n)
            if m and os.path.realpath(p).startswith("/usr/"):
                found.append((int(m.group(1) or 0), p))
    for _, p in sorted(found, reverse=True):
        r = subprocess.run([p, "-I", "-S", "-c", "import sys, tomllib; print(sys.version_info >= (3, 11))"],
                           capture_output=True, text=True, env=SYSTEM_ENV)
        if r.returncode == 0 and r.stdout.strip() == "True":
            return os.path.realpath(p)
    return None


def system_problems(dirs):
    """Anything in the system directories the sandbox shows that the user could have placed there: a
    file or folder writable by the user, or owned by a regular user or group (id 1000 or more). Links
    are not followed (inside the sandbox they lead only to what it shows)."""
    find = system_tool("find")
    if find is None:
        return ["/usr/bin/find is not the system's find, so the system directories cannot be checked (library issue L7)"]
    r = subprocess.run([find] + dirs + ["!", "-type", "l", "(", "-uid", "+999", "-o", "-gid", "+999", "-o",
                                        "-writable", ")", "-print", "-quit"], capture_output=True, text=True, env=SYSTEM_ENV)
    hit = r.stdout.strip()
    # a folder the user cannot read is root's (a user's own folder the user can always read), so find's
    # "Permission denied" is no problem; any other error is
    other = [l for l in r.stderr.splitlines() if l.strip() and "Permission denied" not in l]
    if other and not hit:
        return [f"find could not read the system directories ({other[0].strip()[:120]})"]
    return [f"{hit}: writable by the user, or owned by a regular user"] if hit else []


def elan_parts(home):
    """The parts of the Lean toolchain the sandbox shows: elan's `bin` and `settings.toml`, and the
    toolchain lean/lean-toolchain pins (`leanprover/lean4:v4.35.0` is `toolchains/leanprover--lean4---v4.35.0`).
    None of them may be a symbolic link, or lie under one below the home directory (library issue L54):
    what the sandbox binds is then the folder elan installed, not one a link (`elan toolchain link`)
    points to. Returns `(parts, name)`, or `(None, problem)`."""
    elan = os.path.join(home, ".elan")
    try:
        tc = open(os.path.join(ROOT, "lean", "lean-toolchain"), encoding="utf-8").read().strip()
    except (OSError, UnicodeDecodeError):
        return None, "lean/lean-toolchain is missing"
    if not TOOLCHAIN.fullmatch(tc):
        return None, f"lean/lean-toolchain names {tc[:60]!r}, not a release of Lean"
    tdir = os.path.join(elan, "toolchains", tc.replace("/", "--").replace(":", "---"))
    if not os.path.isdir(tdir):
        return None, f"the toolchain {tc} is not installed ({rel(tdir, home)})"
    parts = [os.path.join(elan, "bin"), tdir] + [p for p in [os.path.join(elan, "settings.toml")] if os.path.lexists(p)]
    linked = [rel(p, home) for p in parts if os.path.realpath(p) != p]
    if linked:
        return None, f"the toolchain's {', '.join(linked)} is a symbolic link or lies under one (library issue L54)"
    return parts, tc


def release_records(root):
    """The lint commands for the records under `root` that are not yet recorded (library issue L52): the
    role folders of the library's last round that its ledger does not record, and each kernel's current
    round. Recorded records are frozen: the `records` step compares them with their ledger, and `tvr
    audit` reads their text."""
    cmds = []
    rounds = library_rounds(root)
    if rounds:
        last = max(rounds)
        done = {e.get("dir") for e in library_ledger(root)}
        for who in rounds[last]:
            if f"docs/library-review/round-{last}/{who}" not in done:
                cmds.append(["lint", "library", str(last), who])
    kr = os.path.join(root, "kernels")
    for k in sorted(os.listdir(kr) if os.path.isdir(kr) else []):
        if not os.path.exists(os.path.join(kr, k, "kernel.json")):
            continue
        led = os.path.join(kr, k, "review", "ledger.jsonl")
        n = 1 + sum(1 for l in (open(led) if os.path.exists(led) else []) if '"type": "response"' in l)
        rd = os.path.join(kr, k, "review", f"round-{n}")
        for who in sorted(os.listdir(rd) if os.path.isdir(rd) else []):
            if who == "prover":
                if os.path.exists(os.path.join(rd, who, "response.json")):
                    cmds.append(["lint", k, "prover"])
            elif any(f.startswith("filing") for f in os.listdir(os.path.join(rd, who))):
                cmds.append(["lint", k, "reviewer", who])
    return cmds


def _read_json(path):
    """A JSON file's object, or None when it is missing or not a JSON object."""
    try:
        with open(path) as f:
            v = json.load(f)
        return v if isinstance(v, dict) else None
    except (OSError, ValueError):
        return None


def gate_mismatch(shipped, new):
    """How a kernel's shipped `gate.json` differs from the gate recomputed in the sandbox (library issues
    L42, L53, L69): the result, the digest, the item digests, the kernel files it read, the gate's code
    and the build's other inputs must be the same."""
    if shipped is None:
        return ["no gate.json is shipped"]
    keys = [x for x in ("ok", "digest", "items", "files", "tool", "build") if shipped.get(x) != new.get(x)]
    if not keys:
        return []
    diff = sorted(i for i in set(shipped.get("items", {})) | set(new.get("items", {}))
                  if shipped.get("items", {}).get(i) != new.get("items", {}).get(i))
    return ["the shipped gate.json is stale (" + ", ".join(
        {"ok": "the result", "digest": "the digest", "files": "the kernel files", "tool": "the gate's code",
         "build": "the build's inputs"}.get(x) or
        ("the items " + ", ".join(diff[:4]) + (" …" if len(diff) > 4 else "")) for x in keys) +
        " differ from the recomputed gate)"]


CERT_KEYS = ("kernel", "gate", "items", "reviewers")


def certificate_mismatch(shipped, new):
    """The shipped certificate.json against the one the sandbox regenerates from the ledger and the
    recomputed gate (library issue L52): both absent, or both present and the same but for the time."""
    if shipped is None and new is None:
        return []
    if shipped is None:
        return ["no certificate.json is shipped, but the ledger and the recomputed gate grant one"]
    if new is None:
        return ["the shipped certificate.json is not granted by the ledger and the recomputed gate"]
    diff = [x for x in CERT_KEYS if shipped.get(x) != new.get(x)]
    return [f"the shipped certificate.json differs from the regenerated one ({', '.join(diff)})"] if diff else []


def generated_pages(root):
    """The pages `tvr` generates, relative to `root`: each kernel's claim.md and REVIEW.md, kernels/README.md
    and docs/library-review/REPORTS.md."""
    out = ["kernels/README.md", "docs/library-review/REPORTS.md"]
    kr = os.path.join(root, "kernels")
    for k in sorted(x for x in (os.listdir(kr) if os.path.isdir(kr) else []) if os.path.exists(os.path.join(kr, x, "kernel.json"))):
        out += [f"kernels/{k}/claim.md", f"kernels/{k}/REVIEW.md"]
    return out


def shipped_records(root):
    """What the copy at `root` ships of what the sandbox regenerates, read before the sandbox runs (library issue
    L96): every generated page's text (`generated_pages`, `None` for a page it lacks), and each kernel's
    `gate.json` and `review/certificate.json` (`None` when missing or not a JSON object). The release check
    compares the sandbox's results with these, never with the live tree, which may change while the check runs
    (a page fixed or regenerated there meanwhile)."""
    pages = {}
    for p in generated_pages(root):
        f = os.path.join(root, p)
        pages[p] = open(f, encoding="utf-8", errors="replace").read() if os.path.isfile(f) else None
    kr = os.path.join(root, "kernels")
    ks = sorted(x for x in (os.listdir(kr) if os.path.isdir(kr) else []) if os.path.exists(os.path.join(kr, x, "kernel.json")))
    return {"pages": pages, "gates": {k: _read_json(os.path.join(kr, k, "gate.json")) for k in ks},
            "certificates": {k: _read_json(os.path.join(kr, k, "review", "certificate.json")) for k in ks}}


def page_mismatch(shipped, new_root):
    """The generated pages the copy shipped (`shipped`, from `shipped_records`, read before the sandbox ran; library
    issue L96) against the ones the sandbox regenerates in the copy (library issue L53), with the times of each
    kernel's shipped and recomputed gates masked in both (the gate is recomputed at another time, and a page that
    a failing gate does not rewrite keeps the old time). `shipped` may also be a root, read now."""
    if isinstance(shipped, str):
        shipped = shipped_records(shipped)
    bad = []
    for p in sorted(set(generated_pages(new_root)) | set(shipped["pages"])):
        times = []
        if p.startswith("kernels/") and p.count("/") == 2:
            k = p.split("/")[1]
            times = [(shipped["gates"].get(k) or {}).get("time"),
                     (_read_json(os.path.join(new_root, "kernels", k, "gate.json")) or {}).get("time")]

        def masked(t):
            if t is None:
                return None
            for x in times:
                if x:
                    t = t.replace(x, "<gate time>")
            return t
        f = os.path.join(new_root, p)
        new = open(f, encoding="utf-8", errors="replace").read() if os.path.isfile(f) else None
        if masked(shipped["pages"].get(p)) != masked(new):
            bad.append(f"{p}: the shipped page differs from the one regenerated from the shipped files")
    return bad


def cmd_release(argv):
    mask = "--no-mask" not in argv
    d = argv[argv.index("--dir") + 1] if "--dir" in argv else None
    if d is None:
        import tempfile
        d = tempfile.mkdtemp(prefix="tva-release-")
    d = os.path.realpath(d)
    home = os.path.realpath(os.path.expanduser("~"))
    root = os.path.realpath(ROOT)
    if d == root or d.startswith(root + os.sep) or (mask and (d == home or d.startswith(home + os.sep))):
        die(f"the sandbox must be outside TVA{' and outside the home directory (it is hidden there)' if mask else ''}: {d}")
    if os.path.exists(d) and os.listdir(d):
        die(f"the sandbox {d} is not empty")
    os.makedirs(d, exist_ok=True)
    if shutil.disk_usage(d).free < SANDBOX_FREE:
        die(f"the sandbox needs about {SANDBOX_FREE >> 30} GB free (Mathlib is built there); {d} has "
            f"{shutil.disk_usage(d).free >> 30} GB")
    bwrap = system_tool("bwrap") if mask else None  # the system's, not the caller's PATH's (library issue L7)
    in_home = os.path.relpath(root, home) if root.startswith(home + os.sep) else os.path.basename(root)
    shome = os.path.join(d, "home")                       # the sandbox's home directory
    copy = os.path.join(shome, in_home)                   # TVA as shipped
    steps = []

    def masked(text):
        """The output names neither the sandbox's folder nor the copy's (library issue L56): they are
        `<dir>` and `<tva>`, and the home directory is `~`, so the output can be kept in TVA."""
        for a, b in ((copy, "<tva>"), (os.path.join(home, in_home), "<tva>"), (d, "<dir>"), (home, "~")):
            text = text.replace(a, b)
        return text

    def record(name, ok, detail):
        steps.append({"step": name, "ok": bool(ok), "detail": masked(detail)})
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {masked(detail)[:300]}", flush=True)

    def finish():
        """The copy and its build (about 13 GB) are removed unless --keep; release.json stays."""
        shutil.rmtree(os.path.join(d, "scratch"), ignore_errors=True)
        if "--keep" in argv:
            return "the sandbox and release.json are in <dir>, the --dir folder"
        shutil.rmtree(shome, ignore_errors=True)
        return "release.json is in <dir>, the --dir folder (the sandbox was removed; --keep keeps it)"

    files = shipped_files()
    sp = shipped_problems(files)
    record("shipped", not sp, f"{len(files)} files; none in a folder the text scan skips" if not sp else "; ".join(sp[:4]))
    files = [f for f in files if not f.endswith("/")]
    for f in files:
        src, dst = os.path.join(ROOT, f), os.path.join(copy, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.islink(src):
            os.symlink(os.readlink(src), dst)
        else:
            shutil.copy2(src, dst)
    os.makedirs(os.path.join(d, "scratch"), exist_ok=True)
    pk = make_packages(os.path.join(copy, "lean"), os.path.join(d, "scratch"))
    record("packages", not pk, "every package of lean/lake-manifest.json made afresh from its upstream repository: "
           "Mathlib's commit is its inputRev's, the inherited packages are those Mathlib's manifest pins, every file "
           "is its commit's blob, and no build output" if not pk else "; ".join(pk[:4]))
    elan, tc = elan_parts(home)
    if mask:
        py = system_python()
        shown = ["/" + x for x in SANDBOX_SYSTEM if os.path.isdir("/" + x) and not os.path.islink("/" + x)]
        base = []
        if bwrap:
            base = [bwrap, "--die-with-parent", "--unshare-net", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
                    "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
            for x in SANDBOX_SYSTEM:
                p = "/" + x
                if os.path.islink(p):
                    base += ["--symlink", os.readlink(p), p]
                elif os.path.isdir(p):
                    base += ["--ro-bind", p, p]
            base += ["--bind", shome, home]
            for part in elan or []:
                base += ["--ro-bind", part, part]
            base += ["--clearenv", "--setenv", "HOME", home, "--setenv", "LANG", "C.UTF-8",
                     "--setenv", "PATH", os.pathsep.join([os.path.join(home, ".elan", "bin"), "/usr/local/bin", "/usr/bin", "/bin"])]
        # the sandbox must work, or the check fails: a release check without it would not be complete
        probe = subprocess.run(base + ["--chdir", home, py, "-I", "-S", "-c", "import tomllib"], capture_output=True,
                               text=True, env=SYSTEM_ENV) if bwrap and py else None
        bad = (system_problems(shown) + ([tc] if elan is None else []) +
               ([] if py else ["no system Python 3.11 or later under /usr"]) +
               ([] if bwrap else ["/usr/bin/bwrap is not the system's bwrap (a file under /usr that the user cannot write), "
                                  "so the sandbox cannot be built (--no-mask runs a HOME-only check, which is not a "
                                  "release check)"]) +
               ([] if system_tool("git") else ["/usr/bin/git is not the system's git"]) +
               ([f"the sandbox does not start: {(probe.stderr or probe.stdout).strip()[:200]}"]
                if probe is not None and probe.returncode != 0 else []))
        record("sandbox", not bad, f"shows only the copy, {', '.join(shown)} (read-only, nothing user-placed), the Lean "
               f"toolchain {tc} (read-only, no link) and the system's {py}; no network" if not bad else "; ".join(bad))
        inside_root = os.path.join(home, in_home)

        def run(cmd, cwd):
            return subprocess.run(base + ["--chdir", cwd] + cmd, capture_output=True, text=True, env=SYSTEM_ENV)
    else:
        py = sys.executable
        record("sandbox", False, "--no-mask: HOME only, so paths outside the copy still exist (not a release check)")
        for part in elan or []:
            link_at = os.path.join(shome, os.path.relpath(part, home))
            os.makedirs(os.path.dirname(link_at), exist_ok=True)
            os.symlink(part, link_at)
        env = {"HOME": shome, "LANG": "C.UTF-8",
               "PATH": os.pathsep.join([os.path.join(shome, ".elan", "bin"), "/usr/bin", "/bin"])}
        inside_root = copy

        def run(cmd, cwd):
            return subprocess.run(cmd, cwd=cwd.replace(os.path.join(home, in_home), copy, 1), capture_output=True,
                                  text=True, env=env)
    if not pk == [] or (mask and not steps[-1]["ok"]):
        dump_json(os.path.join(d, "release.json"), {"time": now(), "sandbox": "<dir>", "masked": bool(mask and bwrap),
                                                     "files": len(files), "ok": False, "steps": steps})
        print(f"release check FAIL: the sandbox could not be made; {finish()}")
        return 1
    lake = os.path.join(home if mask else shome, ".elan", "bin", "lake")
    tvr = [py, "-I", "-S", "tools/tvr.py"]

    def step(name, cmd, cwd, ok_note=""):
        r = run(cmd, cwd)
        out = [l.strip() for l in (r.stdout + r.stderr).strip().splitlines() if l.strip()]
        errs = [l for l in out if l.lower().startswith(("error", "fail", "- "))]
        record(name, r.returncode == 0, (ok_note if r.returncode == 0 else "") or
               " | ".join(errs[:4] + [l for l in out[-3:] if l not in errs[:4]])[:600])
        return r

    print(f"sandbox <dir>: {len(files)} shipped files; " +
          ("an allowlist sandbox (bwrap), no network" if mask else "HOME only (--no-mask): paths outside it still exist"),
          flush=True)
    b = step("build", [lake, "build", "TVA"], os.path.join(inside_root, "lean"), "lake build TVA (Mathlib built from its pinned source)")
    if b.returncode == 0:
        step("library", tvr + ["library-check"], inside_root, "library hygiene; leanchecker TVA re-checks every library declaration")
    else:
        record("library", False, "not run (the build failed)")
    # No kernel's gate is recomputed, and so no certificate or generated page is compared (library issue L160;
    # user, 2026-11-08: "I do not think we need recomputing the gates of all 38 kernels in kernels/"): the kernels
    # are becoming benchmark tasks. Of kernels/ the check still reads each kernel's review records (`records`,
    # and `lint` on its current round) and every file's text and imports (`audit`).
    rp = records_problems(copy)
    record("records", not rp, "every recorded file is as its ledger recorded it, nothing was added to a recorded folder, "
           "every event follows from a recorded file and every kernel's role folder from its events, and every library "
           "round before the last is recorded" if not rp else "; ".join(rp[:4]) +
           (f" (+{len(rp) - 4} more)" if len(rp) > 4 else ""))
    for cmd in release_records(copy):
        step(" ".join(["lint"] + cmd[1:]), tvr + cmd, inside_root, "regular")
    step("audit", tvr + ["audit"], inside_root, "independent")
    ok = all(x["ok"] for x in steps)
    dump_json(os.path.join(d, "release.json"), {"time": now(), "sandbox": "<dir>", "masked": mask,
                                                 "python": py, "toolchain": tc, "files": len(files), "ok": ok, "steps": steps,
                                                 "ledgers": ledger_digests(copy)})
    print(f"release check {'PASS' if ok else 'FAIL'} ({sum(x['ok'] for x in steps)} of {len(steps)} steps pass); "
          f"{finish()}")
    return 0 if ok else 1


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
    for what, v in (("--module", module), ("--namespace", ns)):
        if not (isinstance(v, str) and LEAN_NAME.fullmatch(v)):
            die(f"{what} is a dotted name of ASCII letters, digits, `_` and `'` (library issue L38)")
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
    kernel_lakefile()
    render_index()
    print(f"created kernels/{k}; write lean/{module.replace('.', '/')}.lean, fill kernel.json (ranges) and model-map.json; "
          f"build with `lake -f .lake/tvr/lakefile.toml build {module}` from lean/")


def main(argv):
    if len(argv) >= 2 and argv[1] == "index":
        if kernels():
            kernel_lakefile()
        for k in kernels():
            render_review(k)
        render_index()
        render_reports()
        return 0
    if len(argv) >= 2 and argv[1] == "audit":
        bad = library_independence()
        for k in kernels():
            bad += kernel_independence(k)
        # every other file of kernels/, in a folder that is no kernel's too (library issue L70)
        inside = tuple(kpath(k) + os.sep for k in kernels())
        bad += scan_files([p for p in text_files(KROOT) if not p.startswith(inside)])
        bad += citation_problems()
        bad += fact_page_problems()
        bad += library_module_problems()
        # what ships must be what the scan reads (library issue L41); a release copy has no git
        # metadata, and holds only what ships, which the release check's `shipped` step checked
        if git_checkout():
            bad += shipped_problems(shipped_files())
        print("\n".join(bad) if bad else "independent: TVA imports only TVA and Mathlib, and refers to no other project")
        return 1 if bad else 0
    if len(argv) >= 2 and argv[1] == "release":
        return cmd_release(argv)
    if len(argv) >= 2 and argv[1] == "library-check":
        return cmd_library_check()
    if len(argv) >= 2 and argv[1] == "record":
        return cmd_record(argv)
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
    elif cmd == "freeze":
        return cmd_freeze(k, "--dry-run" in argv)
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

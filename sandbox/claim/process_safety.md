# Stopping processes safely (read this before you run `kill`, `pkill`, `killall` or `rm -rf`)

You run in the **same container** as the processes that run *you*. A broad kill command matches them as easily
as the Lean process you meant, and if it hits them **your session ends at once, mid-task**, and the task fails
with whatever proof is on disk. The processes you must never kill:

| Process (as `ps -eo pid,ppid,args` shows it) | What it is |
|---|---|
| `sh -c sleep infinity` (PID 1) and its `sleep infinity` | the container itself: if either dies, the container stops |
| `bash -c set -o pipefail; ... claude --verbose ... --print ...` | the wrapper that runs your agent (its command line names `claude` and `tva-check check`) |
| `claude --verbose --output-format=stream-json ...` | **you**, and your subagents (they run inside this process) |
| `tee /logs/agent/claude-code.txt` | your transcript: killing it breaks your output pipe |
| `/bin/bash -c source .../shell-snapshots/... eval '<your command>'` | the shell running **this** command: its command line holds your command's full text |
| `tva-check hook ...` (`python3`) | the harness's hooks |
| `lake build ...`, `lean ...` started by a subagent | a subagent's build: killing it loses that subagent's turn |

## Footgun 1: a name match (no `-f`)

```
pkill claude        # ❌ kills you
pkill bash          # ❌ kills the shell running this command, and the agent's wrapper
pkill sleep         # ❌ kills PID 1's child: the container stops
pkill tee           # ❌ breaks your output pipe
pkill python3       # ❌ kills the harness's hooks (and the check, if it runs)
killall lean        # ❌ also kills every subagent's build
kill -9 -1          # ❌ kills every process you may signal: the whole container
kill 1              # ❌ the container
```

## Footgun 2: `pkill -f WORD` matches whole command lines

`pkill -f` and `pgrep -f` match the **entire command line**. The shell running your command has the command's
text on its command line, so a pattern that appears in the command matches the shell that runs it:

```
pkill -f lake            # ❌ matches "/bin/bash -c ... pkill -f lake" (this command's shell): kills it
pkill -f lean            # ❌ same, and every subagent's build
pkill -f Proof           # ❌ same
pkill -f tva-check       # ❌ also matches the agent's wrapper (its command line names `tva-check check`)
pkill -f claude          # ❌ kills you
ps aux | grep lean | awk '{print $2}' | xargs kill -9   # ❌ the grep, the shell, the subagents' builds
```

The `[l]ean` bracket trick does not help: the victim is the shell running the pipeline, not the grep.

## The safe way: do not need to kill

- **Bound every long command when you start it:** `timeout 1200 lake build Proof.Lanes`,
  `timeout 600 lake env lean /tmp/T.lean`. `timeout` stops exactly that command, and its exit status is 124.
  The Bash tool also stops a command after its own timeout (10 minutes by default, at most 30: pass `timeout`).
- **Run a long build in the background** (the Bash tool's `run_in_background`), and read its log when it ends.
- **If you must stop one of your own background commands,** start it in its own process group and record the
  group's id, then signal only that group:

  ```
  setsid timeout 3000 lake build Proof > /tmp/build.log 2>&1 & echo $! > /tmp/build.pgid
  kill -TERM -- -"$(cat /tmp/build.pgid)"      # ✅ exactly that build and its children
  ```
- **Look before you kill:** `ps -eo pid,ppid,etime,args --forest` shows the tree. Kill only a PID you
  started (or the recorded group); never by name, never by a bare word, never `-1` or `1`.
- `lake` runs one build at a time (`/usr/local/bin/lake` waits for its turn), so a build that seems stuck may
  be waiting for another agent's build: wait for it rather than killing it.

## Never destroy the build

The library and Mathlib are built in `/app/lean/.lake` (about 8 GB); rebuilding them takes hours. Never run
`lake clean`, `lake update`, `rm -rf /app/lean/.lake` (or anything under it), and never edit
`/app/lean/lakefile.toml`, `lake-manifest.json` or `lean-toolchain`. If a module's build output seems stale,
build the module again: `lake build Proof.Lanes`.

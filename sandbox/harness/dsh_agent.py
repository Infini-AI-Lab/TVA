"""Harbor agent: the TVA prover on DeepSeek Harness (dsh) instead of Claude Code (HARNESS.md, "The dsh prover").

  PYTHONPATH=$TVA/sandbox/harness harbor run -p tasks/tva-prove-<kernel> -a dsh_agent:Dsh \
      -m anthropic/<model-id> -e docker

The task must be made with `make_task.py --prover-agent dsh`: its prover image has dsh (pinned), the profile
patch, hooks and manual in /etc/tva-sandbox/dsh/. The model is reached as Claude Code reaches it, with the
host's ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN), normally the meter's.

In the container: dsh's home is the agent's log folder (/logs/agent/dsh: its plain JSONL session logs are
the transcript), the manual its instructions file, and the agent starts in / like Claude Code's. The prover's
clock and file snapshot start (hook.py session-start) before dsh does. The headless run is one task; dsh's
Stop hook keeps it working while the work is unfinished. If dsh exits with an error (a failed request after
its retries), the session is resumed, at most RESUMES times.
"""
import json
import os
import shlex

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

RESUMES = 3
DSH_HOME = "/logs/agent/dsh"
PATCH = "/etc/tva-sandbox/dsh/prover.yml"
FLAGS = "--permission-mode danger-full-access --approval-policy never --json"

RUN = r"""
set -u
export DSH_HOME={home} HOME=/root
mkdir -p "$DSH_HOME" && cp /etc/tva-sandbox/dsh/AGENTS.md "$DSH_HOME/AGENTS.md"
[ -f /etc/tva-sandbox/dsh/env ] && . /etc/tva-sandbox/dsh/env && export TVA_HARNESS_AUTOBUILD
cd /
brief=$(echo '{{}}' | TVA_HARNESS_MANUAL=0 python3 -I /usr/local/lib/tva-harness/hook.py session-start \
        | python3 -I -c 'import json,sys; d=json.load(sys.stdin); print(d["hookSpecificOutput"]["additionalContext"])' \
        2>/dev/null || true)
printf '%s\n\n%s\n' "$TVA_INSTRUCTION" "$brief" > /logs/agent/task.md
unset TVA_INSTRUCTION
dsh --profile headless --patch {patch} {flags} - < /logs/agent/task.md > /logs/agent/dsh-1.jsonl 2> /logs/agent/dsh-1.err
rc=$?; echo "dsh run 1 exit $rc $(date -u +%H:%M:%S)" >> /logs/agent/dsh-runs.txt
sid=$(head -1 /logs/agent/dsh-1.jsonl | python3 -I -c 'import json,sys; print(json.load(sys.stdin).get("sessionId",""))' 2>/dev/null)
for n in $(seq 2 {last}); do
  final=$(grep '"type":"final"' /logs/agent/dsh-$((n-1)).jsonl | tail -1)
  case "$final" in *'"kind":"completed"'*) [ "$rc" -eq 0 ] && break;; esac
  [ -n "$sid" ] || break
  printf '%s\n' "The run stopped on an error (see above). Continue the task where you left off: run \`tva-kit stage\` first." \
    | dsh --profile headless --patch {patch} {flags} --session-id "$sid" - > /logs/agent/dsh-$n.jsonl 2> /logs/agent/dsh-$n.err
  rc=$?; echo "dsh run $n exit $rc $(date -u +%H:%M:%S)" >> /logs/agent/dsh-runs.txt
done
exit 0
"""


class Dsh(BaseInstalledAgent):
    """The TVA prover's harness on DeepSeek Harness's headless profile."""

    @staticmethod
    def name() -> str:
        return "tva-dsh"

    def get_version_command(self) -> str | None:
        return "dsh --version"

    def parse_version(self, stdout: str) -> str:
        return stdout.strip().splitlines()[-1] if stdout.strip() else ""

    async def install(self, environment: BaseEnvironment) -> None:
        # dsh is in the task's image (make_task.py --prover-agent dsh); check that it is
        await self.exec_as_agent(environment, command="test -x /usr/local/bin/dsh && test -f " + PATCH)

    def populate_context_post_run(self, context: AgentContext) -> None:
        """The token counts of the run, from dsh's step events (the meter's ledger is the account of cost)."""
        n_in = n_out = 0
        for f in sorted(self.logs_dir.glob("dsh-*.jsonl")):
            for line in f.read_text(errors="replace").splitlines():
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                u = ev.get("usage") if ev.get("type") == "status" and ev.get("phase") == "step_end" else None
                if u:
                    n_in += int(u.get("inputTokens") or 0)
                    n_out += int(u.get("outputTokens") or 0)
        context.n_input_tokens = n_in
        context.n_output_tokens = n_out

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        base = os.environ.get("ANTHROPIC_BASE_URL")
        key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        if not base or not key:
            raise ValueError("set ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY (the meter's) in the environment that runs harbor")
        model = (self.model_name or "").split("/", 1)[-1] if (self.model_name or "").startswith("anthropic/") \
            else (self.model_name or "")
        if not model:
            raise ValueError("give the model as -m anthropic/<model-id>")
        env = {"TVA_LLM_BASE_URL": base.rstrip("/"), "TVA_LLM_API_KEY": key, "TVA_LLM_MODEL": model,
               "DSH_TELEMETRY_DISABLED": "1", "DSH_PERMISSION_MODE": "danger-full-access",
               "TVA_INSTRUCTION": instruction}
        cmd = RUN.format(home=shlex.quote(DSH_HOME), patch=PATCH, flags=FLAGS, last=RESUMES + 1)
        await self.exec_as_agent(environment, command=cmd, env=env)

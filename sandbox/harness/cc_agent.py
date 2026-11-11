"""Harbor agent for a claim task (make_task.py): Harbor's `claude-code` agent, plus a resume after an error exit.

  PYTHONPATH=$TVA/sandbox/harness harbor run -p tasks/tva3-claim-<kernel> -a cc_agent:ClaudeCodeClaim \\
      -m anthropic/<model-id> -e docker

Harbor's agent runs `claude --print` once. When that run ends on an error (a request that failed after Claude
Code's retries, or a crash) rather than with the agent's own last turn, this agent resumes the same session
(`claude --resume`), at most RESUMES times. A run that ends normally is not resumed: the Stop hook has already
decided that the prover may stop. Each run's exit is logged in /logs/agent/cc-runs.txt; the transcript stays
/logs/agent/claude-code.txt, the resumed runs appended to it.
"""
import re
import shlex

from harbor.agents.installed.claude_code import ClaudeCode
from harbor.environments.base import BaseEnvironment

RESUMES = 3
CONTINUE = ("The run stopped on an error (see above). Continue the task where you left off: run "
            "`/usr/local/bin/tva-check check /app/claim` first.")
# Harbor's run line: `... printf "%s" "$VAR" | claude --verbose --output-format=stream-json FLAGS--print 2>&1 | tee LOG`
RUN = re.compile(r'\| (claude --verbose --output-format=stream-json .*?)--print 2>&1 \| tee (\S+)\s*$', re.S)

LOOP = r"""
rc=${{PIPESTATUS[1]:-$?}}
echo "claude run 1 exit $rc $(date -u +%H:%M:%S)" >> /logs/agent/cc-runs.txt
for n in $(seq 2 {last}); do
  last=$(grep '"type":"result"' {log} | tail -1)
  case "$last" in *'"is_error":false'*) break;; esac
  sid=$(grep -o '"session_id":"[^"]*"' {log} | tail -1 | cut -d'"' -f4)
  [ -n "$sid" ] || break
  printf '%s' {cont} | {claude}--resume "$sid" --print 2>&1 | tee -a {log}
  rc=${{PIPESTATUS[1]}}
  echo "claude run $n exit $rc $(date -u +%H:%M:%S)" >> /logs/agent/cc-runs.txt
done
true
"""


class ClaudeCodeClaim(ClaudeCode):
    """Harbor's Claude Code agent with a resume after an error exit, for TVA's claim tasks."""

    @staticmethod
    def name() -> str:
        return "tva3-claim-claude-code"

    async def exec_as_agent(self, environment: BaseEnvironment, command: str, *args, **kwargs):
        m = RUN.search(command)
        if m and "--resume" not in m.group(1) and "--continue" not in m.group(1):
            claude, log = m.group(1), m.group(2)
            command = command + LOOP.format(last=RESUMES + 1, log=log, cont=shlex.quote(CONTINUE), claude=claude)
        return await super().exec_as_agent(environment, command, *args, **kwargs)

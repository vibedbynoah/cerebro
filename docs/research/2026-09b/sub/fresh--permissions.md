# Tool permission systems — fresh research, late Sept 2026

Scope: Claude Code, OpenAI Codex CLI, opencode, Cline, Gemini CLI, Cursor.
Focus: modes, rule syntax, approval UI, diff previews, Windows sandboxing, audit logs, praise/complaints.
Date: 2026-09-22. All URLs below were opened during research.

## 1. Claude Code

Permission model: three rule lists — `allow`, `ask`, `deny` — in `settings.json`, evaluated in
fixed order **deny → ask → allow**; specificity never changes the order. A broad deny cannot carry
allowlist exceptions (https://code.claude.com/docs/en/permissions). Rule shape is `Tool` or
`Tool(specifier)`, e.g. `Bash(npm run *)`, `Read(./.env)`, `WebFetch(domain:example.com)`.
A bare-name deny (`Bash`) removes the tool from the model's context entirely; a scoped deny
(`Bash(rm *)`) leaves the tool visible and blocks matching calls. Managed enterprise settings
override everything; otherwise lists merge across managed → CLI flags → local → project → user,
and a deny at any level beats an allow anywhere else.

Modes (`permissions.defaultMode`, switchable with Shift+Tab): `default`/`manual` (prompt on first
use), `acceptEdits` (auto-accept file edits + basic fs commands in workspace), `plan` (read-only
recon), `auto` (a classifier model reviews actions instead of the user), `dontAsk` (auto-deny
anything not pre-approved — for CI), `bypassPermissions` (`--dangerously-skip-permissions`, for
throwaway containers only). Deny and explicit ask rules still apply even in bypass mode, plus a
hard circuit breaker prompting on `rm -rf /`-style critical-path removals
(https://code.claude.com/docs/en/permission-modes?c=mikannn&fcdaa149_sort_date=desc).
Manage rules via the `/permissions` dialog, which shows each rule's source file.

Bash rules are **prefix/text matches, not a security boundary** — the docs say so explicitly:
`/usr/bin/curl`, `sh -c 'curl …'`, `git -C . push` all dodge `Bash(curl *)` / `Bash(git push *)`.
Compound commands are split per subcommand; a fixed wrapper list (`timeout`, `nice`, …) is stripped;
there is a built-in read-only command set; output redirects are checked against Edit/Read rules
(https://code.claude.com/docs/en/permissions). For real enforcement the docs point at the Bash
sandbox (macOS, Linux, WSL2 only — no native Windows sandbox) and at `PreToolUse` hooks
(https://code.claude.com/docs/en/hooks.md, https://code.claude.com/docs/en/agent-sdk/hooks.md).
Hooks (`PreToolUse`, `PostToolUse`, `PermissionRequest`, `PermissionDenied`, …) can allow/deny/ask,
rewrite tool input, and are the community's audit-log mechanism — but hook decisions never override
deny/ask rules, and there are open Windows reports of PreToolUse hooks silently never firing
(https://github.com/anthropics/claude-code/issues/25981) or of `ask` verdicts being ignored under
`defaultMode: auto` (https://github.com/anthropics/claude-code/issues/85904). Workspace trust:
project-level allow rules only apply after accepting the trust dialog.

Third-party audit/policy hooks (all local-only, hook-based, not OS sandboxes): `cc-audit`
append-only JSONL log + YAML blocklists, fail-closed policy (https://pypi.org/project/cc-audit/0.1.2/);
`sneakoscope`/`agentguard` with hash-chained log, live dashboard, and `run` sandbox on Linux/macOS
only — "on Windows there is no clean native equivalent, so `run` refuses" (https://pypi.org/project/sneakoscope/0.4.1/);
`PolicyApprovalGate`, a deterministic LLM-free PreToolUse gate with audit log, rotation, hashing,
redaction; Windows support explicitly experimental (https://github.com/nobuo-miura/PolicyApprovalGate).

## 2. OpenAI Codex CLI

Two stacked controls: **sandbox mode** (what code *can* do: `read-only`, `workspace-write`,
`danger-full-access`/`--yolo`) × **approval policy** (when it must *ask*: `on-request` (default),
`never`, `unless-trusted`, `on-failure`, plus granular) (https://learn.chatgpt.com/docs/agent-approvals-security,
https://github.com/openai/codex/blob/13c42a077c88a0d04ae7680a9891d2daf4558577/docs/sandbox.md).
Approval UI offers accept-once / accept-for-session / accept-and-add-to-policy (writes an execpolicy
rule); read-only + never is the CI preset (https://openai-codex.mintlify.app/concepts/approvals).
`execpolicy` rules in Starlark (`prefix_rule(pattern=[…], decision=allow|prompt|forbidden)`)
persist trusted commands per project (same source). Newer **permission profiles** (beta,
Codex ≥0.138) replace sandbox_mode with named profiles (`:read-only`, `:workspace`,
`:danger-full-access`, or custom with `extends`), filesystem `read|write|deny` entries
(`**/*.env = deny` carve-outs), and network domain allow/deny enforced via a proxy feature
(https://developers.openai.com/codex/permissions). **Auto-review** routes sandbox-escalation
approvals to a separate reviewer agent instead of the user; it grants nothing, only changes who
reviews (https://developers.openai.com/codex/concepts/sandboxing/auto-review).

Windows is the weak platform everywhere: Codex's native Windows sandbox is experimental
(AppContainer restricted token; cannot stop writes where the Everyone SID already has access),
with `elevated` vs weaker `unelevated` levels — docs recommend WSL for the Linux model
(https://developers.openai.com/codex/permissions,
https://github.com/openai/codex/blob/13c42a077c88a0d04ae7680a9891d2daf4558577/docs/sandbox.md).
Biggest recent complaint: `approval_policy = "untrusted"` (run only known-safe commands
automatically) was retired in 0.149.0 with a hard startup error telling users to delete it, silently
downgrading trusted projects to weaker `on-request`; docs still recommended it at filing time
(https://github.com/openai/codex/issues/39973).

## 3. opencode

Simplest model of the six: a `permission` map (`allow|ask|deny`) keyed by tool (`bash`, `edit`,
`read`, `glob`, `grep`, `webfetch`, …) plus guards (`external_directory`, `doom_loop`), with
per-pattern objects (`"bash": {"*": "ask", "git *": "allow", "rm *": "deny"}`). **Last matching
rule wins** — opposite of Claude Code's deny-first order (https://opencode.ai/docs/permissions/).
Defaults are permissive (`allow` almost everywhere; only `doom_loop`/`external_directory` ask and
`.env` reads denied). `--auto` flips asks to approvals while still enforcing denies; the TUI shows
an `auto` indicator. Ask UI: `once` / `always` / `reject`, where `always` writes a durable
project-scoped allow rule that still can never override `deny`. Agent identities get their own
permission overlays (a `review` subagent can be `edit: deny`), and v2 moved to an ordered
`permissions[]` array of `{action, resource, effect}` with `shell`/`subagent` renames
(https://opencode.ai/v2/docs/permissions/). Above that sit `experimental.policies`: binary,
never-prompt, organization-wins statements (e.g. `permission:shell:git push * → deny`) delivered
also via OpenCode Console workspaces (https://opencode.ai/v2/docs/policies/). The rework history
(`tools` boolean merged into `permission` in v1.1.1) is in (https://github.com/anomalyco/opencode/pull/6319).
No OS-level command sandbox was found in the docs — confinement is approval-gating plus the
`external_directory` boundary. Praise centers on the tiny config surface; the risk side is the
permissive default plus last-match-wins ordering, which rewards putting the catch-all first.

## 4. Cline (VS Code / JetBrains / CLI / SDK)

Category toggles, not regex rules: Read/Edit project files + outside-workspace extensions, Execute
safe/all commands, browser, MCP, notifications (https://docs.cline.bot/features/auto-approve).
Critically, **the model itself marks each command `requires_approval`** — there is no fixed
allowlist, and the "safe" examples are explicitly "not guarantees." YOLO mode auto-approves
everything with no confirmation dialog; enterprises can kill the toggle remotely via
`yoloModeAllowed: false` (https://docs.cline.bot/enterprise-solutions/configuration/infrastructure-configuration/control-other-cline-features/yolo-mode).
Compensating control is **checkpoints**: a shadow git repo snapshots the workspace after every tool
call, with Compare (multi-file diff) and Restore files / task / both per step
(https://docs.cline.bot/core-workflows/checkpoints,
https://github.com/cline/cline/blob/65e9727c/src/integrations/checkpoints/index.ts). Approvals show
editable diffs first (VS Code diff preview work: https://github.com/cline/cline/commit/2ac5c85e6950528de2a2a69a9e0e68307eb6d78d);
SDK embeds use `toolPolicies: {tool: {autoApprove}}` with conditional-approval callbacks, default
enabled+auto-approved so policies must be set explicitly
(https://docs.cline.bot/sdk/guides/permission-handling); CLI has `--yolo` / `--auto-approve`,
plus `CLINE_COMMAND_PERMISSIONS` allow/deny globs where deny wins
(https://github.com/cline/cline/blob/fd8cecdd/cli/man/cline.1.md). No OS sandbox; isolation = Git
rollback + human gates. Users like the low friction; compliance-minded admins are pointed at
disabling YOLO, telemetry/task-history/audit logs, and non-production restriction.

## 5. Gemini CLI

Modes: `default` (prompt), `auto_edit` (edits auto-approved), `plan` (read-only + plan-artifact
rules via a `plan.toml` policy tier), and `yolo` — CLI-flag-only, never a persisted default
(https://geminicli.com/docs/reference/configuration/, https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/plan-mode.md).
**Trusted Folders**: enabling `security.folderTrust` gates project config/`.env`/extensions/MCP on a
per-folder trust decision; untrusted folders force a restricted safe mode (no workspace settings,
no auto-accept, YOLO/AUTO_EDIT forcibly downgraded to DEFAULT per PR
https://github.com/google-gemini/gemini-cli/pull/7041; yolo auto-trust fix
https://github.com/google-gemini/gemini-cli/pull/19839). Standing rules live in a **policy engine**
(TOML `[[rule]]` with toolName/decision/priority/modes/argsPattern)
(https://google-gemini.github.io/gemini-cli/docs/get-started/configuration.html). Sandboxing is the
broadest here: Seatbelt (macOS), Docker/Podman (cross-platform), `windows-native` (icacls Low
integrity — persistent side effects, documented reset command), gVisor/runsc, LXC, plus
per-tool sandboxing and **sandbox expansion** (a denial proactively becomes a scoped approval
dialog) (https://geminicli.com/docs/cli/sandbox/). Hooks (`BeforeTool`/`AfterTool`, same exit-2
contract as Claude) and checkpointing/session history round it out. Note: as of June 2026 the site
banners Gemini CLI's transition to Antigravity CLI for unpaid/Google One tiers — track which binary
your comparison targets (same page). Admin kill-switches: `disableYoloMode`, `disableAlwaysAllow`,
`secureModeEnabled`.

## 6. Cursor

**Run Modes** (Settings → Agents → Approvals & Execution): `Auto-review` (default since 3.6:
allowlisted calls run, other shell goes to the sandbox when possible, the rest to an LLM classifier
— currently Claude 4.5 Haiku or GPT-5.4 Mini — which can allow, redirect, or prompt),
`Allowlist` (deterministic, optional sandbox), `Run Everything` (no prompts, no sandbox). The old
`Ask Every Time` was deprecated in 3.5 (https://cursor.com/docs/agent/security/run-modes,
https://cursor.com/changelog/auto-review). Steering (not enforcement) comes from
`permissions.json`: `terminalAllowlist` (prefix strings), `mcpAllowlist` (`server:tool`), and
`autoRun.allow/block_instructions` — plain-English classifier hints
(https://cursor.com/docs/reference/permissions, https://cursor.com/docs/enterprise/deployment-patterns).
`permissions.json` + `sandbox.json` (network domains, extra read/write paths, temp dirs) layer at
user and project scope; team dashboard policy wins (same refs). Sandboxing: Seatbelt on macOS,
Landlock+seccomp on Linux (kernel ≥6.2), **Windows only via WSL2**; plus always-on guards (browser,
file-deletion, external-file protections) and `.cursorignore` (same run-modes page). Approval UI
shows the full command with approve/deny/edit; allowlisted entries can be added inline. Open gap:
the CLI exposed only `allowlist`/`unrestricted` approval modes with no classifier — a forum feature
request for IDE/CLI parity (report says CLI Auto-review followed in July 2026)
(https://forum.cursor.com/t/auto-review-run-mode/161922). Docs stress Auto-review is "best-effort
convenience, not a security boundary" — use Allowlist + manual approval for strict control.

## 7. Cross-cutting: Windows, audit, sentiment

- **Windows sandboxing is the industry-wide hole.** Native enforcement exists only as Codex's
  experimental AppContainer token and Gemini's icacls-integrity approach (with persistent
  side effects); Claude and Cursor defer to WSL2; Cline and opencode have none; agentguard's `run`
  refuses on Windows outright. Any Cerebro comparison on this laptop must test WSL2 vs native paths
  separately and state which was used.
- **Approval UX converges** on once/session-or-saved-always/deny + inline diff or command preview;
  Cline's checkpoint Compare/Restore is the most forgiving undo model, while opencode's durable
  `always` and Codex's accept-and-persist-to-execpolicy are the most permanent.
- **Auditability is hook/transcript-based, never a SIEM.** Best patterns: hash-chained JSONL
  (agentguard), rotation + redaction (PolicyApprovalGate), fail-closed policy with log-only mode
  (cc-audit), session transcripts (Codex `~/.codex/sessions`, Gemini 30-day retention,
  Cline task history). Cerebro can adopt all of these cheaply since it already owns process spawn.
- **Two design poles:** deny-first fixed precedence (Claude, Codex profiles) vs last-match-wins
  (opencode) vs model-graded safety (Cline safe/all, Cursor classifier, Claude/Codox auto-review).
  Classifier modes cut prompt fatigue but are explicitly non-deterministic and non-boundaries —
  datacenter "safe" claims do not transfer to batch-1 laptop use; say so in any comparison.
- **Praise:** fewer prompts via classifiers/auto-approve; Cline checkpoints; Gemini sandbox
  expansion; opencode's tiny config. **Complaints:** Codex's untrusted-policy removal; Claude's
  prefix-match bypasses and silent Windows hook failures; Cursor's CLI parity gap; Cline/Claude
  YOLO footguns and compliance friction.

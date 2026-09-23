# Effort levels: thinking and verification

The user chooses how much the model thinks and how rigorously it checks itself. Speed and
performance are core values, so low effort must be genuinely fast, and the cost of higher effort
is always shown honestly.

One control, five levels, available per message (composer) and as a default (Settings). Each level
sets four things together:

| Level | Thinking budget | Verification instruction | Self-check pass | Typical cost |
|---|---|---|---|---|
| Low | off, or very short | Answer directly from what you know. Only double-check something if a mistake would clearly matter (money, safety, destructive commands). | none | fastest |
| Medium | short | Check key facts, numbers and code paths before answering. Say briefly when you are unsure. | none | ~1.2-1.5x |
| High | moderate | Work through the problem, verify each claim, calculation and line of code you produce, and state assumptions explicitly. | one quick self-review | ~2x |
| XHigh | large | Treat every statement as unverified until checked. Re-derive numbers, trace code line by line, test edge cases mentally, and flag anything you could not confirm. | full self-review | ~3-4x |
| Max | largest the model supports | Verify everything, leave nothing unchecked. Before answering, review the full answer as a hostile reviewer would: every line, number, claim and assumption must hold up. If anything cannot be confirmed, say exactly what and why rather than smoothing it over. | self-review, then an independent re-check pass | slowest |

The cost column is illustrative: the app measures and shows the real slowdown for the current
model and machine after the first few messages at each level.

## Prompt drafts (appended to the system prompt; never shown as scolding to the user)

- **Low:** "Be fast and direct. Use what you already know. Only stop to double-check when an error
  would have real consequences, such as money, safety or commands that change files."
- **Medium:** "Before answering, check the key facts, numbers and any code you write. If you are
  unsure about something important, say so in one short sentence."
- **High:** "Think the problem through before answering. Verify every claim, calculation and line of
  code you give. State your assumptions. After drafting, reread your answer once and fix any mistake."
- **XHigh:** "Assume every part of your answer is wrong until you have checked it. Recompute every
  number, trace every code path line by line, consider edge cases, and verify each claim. After
  drafting, review the whole answer critically and correct it. Clearly mark anything you could not
  verify."
- **Max:** "Verify everything; leave nothing unchecked. Recompute, retrace and re-derive every part
  of your answer. When your draft is done, review it as a demanding expert who is trying to find a
  single flaw in any line, number, claim or assumption, and fix what you find. Only give the answer
  once it would survive that review. If something cannot be confirmed, say precisely what and why
  instead of hiding it."

## Mechanics

- Thinking budget maps to the backend's reasoning control (llama.cpp reasoning budget / effort
  where the model's template supports it). Models without a thinking mode get the verification
  instruction only, and the UI says so.
- XHigh and Max add a second pass: the model's draft is sent back with a review instruction, and
  the revised answer replaces it. Max adds a third, independent check pass (fresh context, "find any
  error in this answer").
- Max may additionally run multi-path self-consistency (user setting, default 3 independent paths
  for questions with a checkable answer, majority or verifier-picked), since independent paths catch
  errors a single self-review repeats. The time cost is shown like every other level.
- The reasoning-budget trap measured on this machine (thinking can consume the whole token budget
  and return nothing) is guarded: the answer budget is reserved separately from the thinking budget.
- The level, the thinking tokens used and the extra time are shown under each reply, so users learn
  what each level costs them.
- Everything is editable: users can rewrite any level's instruction, budget and passes, or add
  their own levels, in Settings.

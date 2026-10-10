# Agent-harness techniques to evaluate (owner paste, 2026-10-09; claims unverified)

For the coding agent layer, after the engine: self-improving harness loop (mine own failure logs,
rewrite tools/prompts/validation), per-instance adaptation of prompts and tool subsets, JIT task DAG
with bounded parallel actions, artifact store passing file pointers / fact graphs between sub-agents
instead of raw logs, interpreter-derived typed state injection (compute outside token generation),
and distilling harness trajectories into a fine-tune ("harness-zero"). Figures quoted in the source
(success-rate gains, pass rates, costs) are unverified and must be measured before any claim.

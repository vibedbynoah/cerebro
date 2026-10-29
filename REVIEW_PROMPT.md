ultracode

You are doing a full ultrareview of Cerebro's LOCAL working copy, then fixing what you find. Objective: make
the app as fast and as bug-free as possible, and faster than llama.cpp at its best settings on every model.

Code (both on this machine; review the current files, not a diff):
- App: C:\Users\noahm\Projects\cerebro-app (branch `app`)
- Engine: C:\Users\noahm\Projects\cerebro-engine (branch `engine`)

Before anything else, read `REVIEW.md` and every `CLAUDE.md` in both trees (root and subfolders). They are
the owner's standing instructions: speed table, priorities, known bugs, house rules and how to verify.

Phase 1, review (fan out with parallel subagents, one per dimension, whole codebase each):
1. Correctness bugs (engine math, state, concurrency, Stop/Cancel/Unload, error paths).
2. Speed (GPU kernels, prompt reading, VRAM placement, processor kernels, app-side blocking or polling).
3. Security (local servers, tools and paths, secrets, downloads).
4. UI and product rules (voice, consistency, Apple principles, no seeded data, accessibility).
5. Tests (gaps, flaky waits, tests that write real user data).
Every finding: file:line, a concrete failure scenario, severity.

Phase 2, verify: an independent subagent tries to disprove each finding by reading the code and, where it
can, reproducing it (a test, a benchmark, a greedy-output diff). Drop what does not survive.

Phase 3, fix, most severe first: one finding per commit, with a test or a measurement that proves it.
Kernel and placement changes must keep greedy output identical and show the speed change with
examples/bench.rs; compare against llama.cpp with an -ngl sweep, never one flag set. Do not break the
llama.cpp fallback for AMD/Intel graphics. No cuBLAS; our own kernels.

Rules: run the full test suites after each batch (see the CLAUDE.md files); never delete the owner's models
or data; do not push to GitHub (commit locally). Finish with a table: finding, severity, fixed or not, and
before/after numbers for every speed change.

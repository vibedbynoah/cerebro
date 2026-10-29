# Cerebro desktop app: instructions for reviewers and agents

Read `/REVIEW.md` first (speed targets against llama.cpp, known gaps, verification). Then this.

## Shape
- Tauri v2 shell (`src-tauri`, Rust) + SolidJS UI (`ui/src`). The inference engine is the separate
  `cerebro-engine` crate (`C:\Users\noahm\Projects\cerebro-engine`, `engine` branch), linked by path.
- Capabilities are deny-by-default. A new Tauri command must be registered in **four** places or it fails
  silently at runtime: `src-tauri/build.rs`, `src-tauri/capabilities/main-window.json`,
  `isolation/index.js`, and the `invoke_handler` in `src-tauri/src/lib.rs`. Review every new command for all four.
- The UI talks to the backend only through the service interface (`ui/src/service`); the browser build uses
  `ui/src/mock`. A feature missing from the mock breaks the e2e suite: check mock parity for every change.

## What to look for (report each as a finding)
- Anything that blocks the Tauri main thread (file I/O, model loads, network) instead of `spawn_blocking`
  or the engine's job queue.
- Work the user cannot interrupt: every load, reply, download, tune and benchmark must stop on request
  (`chat_stop`, `model_unload`, `cerebro_engine::STOP`) within about a second.
- Copy that breaks the house voice: interface text is third person, plain adult language, never "you"/"your".
- Seeded or personal data: nothing may ship that the user did not enter (no sample memories, chats, keys).
- Every protection (sandbox, approvals, offline mode) must stay a user switch with a neutral explanation.

## Verify
`npx tsc --noEmit -p .` · `npx vitest run --config ui/vite.config.ts` · `npm run build` then
`npx playwright test` · `cargo test --release -p cerebro-desktop` (from the repo root). Real models:
`CEREBRO_TEST_MODEL=<gguf> cargo test --release -p cerebro-desktop a_real_model_answers -- --ignored --nocapture`.

# Tests: reviewer checklist

- E2e runs against the browser build with the mock service (`npm run build`, then
  `npx playwright test`). Helpers in `e2e/fixtures.ts`: `openPlus` / `closePlus` for the chat + menu,
  `gotoSettings` and `openScreen(page, "settings")` open every Settings group.
- Known failing before this brief (do not count as regressions, but fixing them is welcome): orchestrate,
  hub-live, hub-verified, train, tune, machine-tune, models disk-space undo, tools Escape.
- Tests never write to the real app data folder (`%LOCALAPPDATA%\Cerebro`); a test that does is a bug.
- A UI change without a matching test update is a finding; so is a test that only passes because it waits
  on a fixed timeout instead of a state.
- Engine-heavy tests are `#[ignore]` and need `CEREBRO_TEST_MODEL`; keep them that way so `cargo test` stays
  fast and needs no model files.

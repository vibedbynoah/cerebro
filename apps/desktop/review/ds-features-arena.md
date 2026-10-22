# Arena and Benchmark lab

Branch: `ds/features-arena`

Two new features. The Arena compares answers from two or more models side by side. The Benchmark
lab measures one model on this computer at three prompt depths and asks it twenty fixed questions.
The Benchmark lab is also reachable with the `/bench` command in Chat.

## What I built

### 1. Arena

A second view of the Chat screen, next to the conversation. The tab strip in the Chat header
switches between **Chat** and **Arena**; the Arena takes the whole column, so the conversation list
and the side panel step out of the way while a comparison is open.

- **One prompt, two to four models.** The model picker lists the text and vision models in the
  library, with a search box for a long list. Two models are picked on the first visit.
- **The models take turns.** The engine keeps one model loaded at a time, so each answer starts
  after the one before it. The screen says so above the form rather than leaving it to be guessed.
- **Each answer shows its own speed:** time to the first token, tokens per second and the length
  of the reply. The figures come from the engine's own counters.
- **Blind mode** is on by default: the answers are labelled A, B, C and D, and no model name is
  shown until you vote. Turning it off shows the names straight away.
- **Vote once per comparison.** Pick the better answer, or a tie. A tie gives every answer half a
  point.
- **A leaderboard per task category.** Four categories: General, Coding, Writing, and Facts and
  reasoning. Ratings are Elo, start at 1,000 and are kept in local storage on this computer only.
  The screen says plainly that the ratings come from your own votes and are a ranking aid, not an
  accuracy score.
- **Export results** writes a JSON file with the prompt, the category, blind mode on or off, the
  vote, every answer with its measured speed, and the leaderboard as it stood.

The Arena is in `ui/src/screens/chat/Arena.tsx`; the ratings, the vote and the export payload are in
`ui/src/lib/arena.ts`; the styles are in `ui/src/styles/arena.css`. The tab lives in
`ui/src/screens/Chat.tsx`.

### 2. Benchmark lab

A view on the Monitor screen, next to the live readings. Pick a model, pick where it should run,
and press **Run benchmark**.

- **Three depths.** A short prompt, a 4K-token prompt and a 16K-token prompt. For each depth the
  lab records the prompt read speed, the writing speed, the time to the first token, the system
  memory in use and the wall time.
- **A depth the model cannot hold is not measured.** It stays in the result as "not run" with the
  reason, for example "This model holds 4,096 tokens, so it cannot read a 16,384-token prompt." No
  number is ever estimated.
- **Twenty fixed questions** in five groups: arithmetic, units, dates, letters in words, and simple
  logic. Each has one exact answer. The grader accepts a short explanation around the right answer
  ("The answer is 43.") but not a different answer, and it is the same in the app and in the code
  that writes the report.
- **Results are saved** with the model (name, quant, size, context) and the machine (processor,
  graphics card, memory, build). They are kept in `bench-history.json` in the app's settings
  folder, newest first.
- **A history chart** shows one point per saved run at the depth and the metric you choose.
- **Compare two runs** side by side at all three depths, with the change in reading speed, writing
  speed and first-token time.
- **A report card** as a PNG or a self-contained HTML file, with the same numbers, the questions
  that were missed, and a line saying what the data is. The HTML file has no scripts and no remote
  content.

The lab is in `ui/src/screens/monitor/BenchmarkLab.tsx`; the question set, the grader and the
comparison helpers are in `ui/src/lib/bench.ts`; the report card is in `ui/src/lib/bench-report.ts`;
the styles are in `ui/src/styles/bench.css`.

### 3. The placement the model runs in

The lab measures the model where the user wants it, not where some other code decides:

- **As it is loaded** (the default): the model runs where it is loaded on this computer, with the
  placement the user chose on the Models screen. If the model is not loaded, the engine's own
  placement is used: as many layers on the graphics card as fit, the rest on the processor, and
  from the drive when memory is short.
- **Graphics card:** every layer on the graphics card. The whole model has to fit in graphics
  memory.
- **Graphics card and processor:** half the layers on the card, half on the processor.
- **Processor only:** every layer on the processor, in system memory.

Every run records the placement it used, and the run card and the report card say where the weights
went and which drive holds the model file.

### 4. The `/bench` command

In the Chat composer, `/bench` opens the Benchmark lab and `/arena` opens the Arena. A command we
know runs on this computer and is never sent to the model, so the composer says what Enter will do
before it happens. A slash word we do not know (`/imagine`, for example) is left alone and goes to
the model as an ordinary message.

The Monitor's two views are in the address (`#/monitor?tab=benchmark`), so a link or a command can
open the lab directly and the address always says what is open.

### 5. The desktop side

Three new Tauri commands, registered in all four places the app requires (`build.rs`,
`capabilities/main-window.json`, `isolation/index.js` and `lib.rs`):

- `bench_run`: measures one library model, streaming each finished depth and the final run.
- `bench_history`: the saved runs, newest first.
- `bench_stop`: stops a run; the depths already measured and the questions already answered are
  kept.

The work runs on the engine thread, the same thread that owns the loaded model, so the numbers come
from the engine and not from an estimate. A benchmark and a chat reply cannot overlap: a chat
started while the lab is measuring is refused with a plain message, and the same holds for a tune.

The Rust code is in `apps/desktop/src-tauri/src/bench.rs`; the command is in
`apps/desktop/src-tauri/src/commands.rs`; the engine job is in `apps/desktop/src-tauri/src/engine.rs`.

The web preview (the mock) has the same screens with example numbers. Every screen that shows them
says the data is illustrative, and the report card says it too.

## How to try it

1. **Arena:** open Chat, choose the **Arena** tab, pick two models, type one question and press
   **Compare answers**. Watch the answers arrive one after the other. Vote for the better one and
   check the leaderboard below. Press **Export results** for a JSON file.
2. **Benchmark lab:** open Monitor and choose **Benchmark lab** (or type `/bench` in Chat). Pick a
   model, pick a placement and press **Run benchmark**. Watch the three depths and the twenty
   questions. Press **PNG report** or **HTML report** for a shareable card. Run it again to compare
   two runs.
3. **Stop:** press **Stop** while a run is measuring. The run is kept, marked as stopped.

## Tests

Unit tests (`apps/desktop/tests/unit`):

- `arena.test.ts`: the Elo maths, the vote on a board with two, three and four models, the storage
  round trip and the export payload.
- `bench.test.ts`: the twenty questions, the grader (numbers, dates, days, yes/no, letter
  choices), the depth plan, the comparison and the report card.
- `commands.test.ts`: `/bench` and `/arena` are parsed, every other message is left to the model,
  and the Monitor's two views round trip through the address.
- `tauri-config.test.ts`: the new commands are declared in all four places, and the isolation hook
  accepts only the four placements.

Rust tests (`cargo test -p cerebro-desktop`):

- the grader agrees with the one in the app (numbers, yes/no, letters, dates, words);
- a depth the model cannot hold is marked as not run, never measured at a shorter prompt;
- a placement choice asks for the layers it names, and "as it is loaded" keeps the loaded
  placement;
- a run records where the weights went.

Playwright tests (`apps/desktop/tests/e2e`):

- `arena.spec.ts`: two models are needed before a comparison, the library is filtered by the
  search, the answers are blind until the vote, the names appear after it, the leaderboard keeps
  the rating, the round is exported as JSON, and the board survives a reload. An axe scan in dark
  mode finds no violations.
- `benchmark.spec.ts`: the three depths and twenty questions run and are saved, the report cards
  are written and say what the data is, the history chart and the comparison follow the saved runs,
  `/bench` in Chat opens the lab and the address follows the view, the chosen placement is measured
  and reported, and an axe scan in light mode finds no violations.

Commands run from `apps/desktop`:

- `npx tsc --noEmit -p tsconfig.json` — passed.
- `npx vitest run --config ui/vite.config.ts` — 25 files, 362 tests passed.
- `npx vite build --config ui/vite.config.ts` — passed.
- `npx playwright test --reporter=line` — 306 tests passed.
- `node scripts/glitch-sweep.mjs` — passed.
- `cargo test -p cerebro-desktop` (with the MSVC environment) — 151 passed, 16 ignored.

## Notes and limits

- The Arena compares models from the library. This branch has no cloud models, so every answer
  comes from a model on this computer.
- The leaderboard is local storage on this computer. It is not shared, and clearing site data
  clears it.
- Elo here is a ranking aid. It reflects the votes cast on this computer and says nothing about
  absolute quality.
- The Benchmark lab measures one model at a time, on the engine thread, and refuses to run while
  a chat reply or a tune is in progress.
- A depth the model cannot hold stays in the result as "not run". The chart and the comparison skip
  that depth instead of drawing a line through it.
- The accuracy check is twenty short questions with exact answers. It is a quick check, not a
  benchmark of reasoning, and the report says so.
- The report card is drawn on a canvas and written to a file from the browser. Nothing is uploaded
  anywhere.

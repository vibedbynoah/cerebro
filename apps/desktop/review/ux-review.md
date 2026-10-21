# UX review: clear, consistent, logically organised

Reviewed the desktop build (`__DESKTOP__`) screen by screen, with the browser preview used only
where it is the design reference for a screen the desktop build does not have yet. Line numbers are
in the current tree.

The main problems were not missing features. They were ordering and honesty: the most-used screen
(Chat) sat below two specialist screens, a planned screen (Plugins) came before the working one
(Tools), several desktop-only states still advertised browser-preview features, and the Chat
composer showed ten fine controls before the user had typed anything.

## What changed

### Navigation (`ui/src/lib/routes.ts:32`)

| Finding | Change |
| --- | --- |
| Chat came after Turbo, below a specialist measurement screen. Beginners are told to start a chat; the composer is the product's front door. | Chat now follows Model Hub and precedes Models, Tune, Monitor and Turbo (`routes.ts:35`). |
| Extend listed Plugins before Tools, even though Plugins is a planned preview on desktop and Tools is where MCP servers and permissions actually live. | Tools now precedes Plugins (`routes.ts:61-72`). |
| The Turbo summary described builds, schedulers and power profiles that the desktop Turbo screen deliberately does not show; the Plugins summary described catalogue installs that do not exist on desktop. | Both summaries are desktop-specific (`routes.ts:39-48`, `62-72`). The browser preview keeps its original summary. |
| Model Hub's lede said "the one optional network feature here", which is vague and slightly wrong: search itself is a network call. | "Searching the catalogue uses the network; everything else works offline" (`routes.ts:34`). |
| Documents' lede promised search "by meaning" while the screen says keyword search only and semantic search is not built yet. | The lede now describes the passage-with-file result the screen really gives (`routes.ts:59`; the honest limitation remains in `screens/Documents.tsx:81`). |
| Image Studio described editing on desktop, but the desktop screen only generates text-to-image. | Desktop summary now says it makes images and keeps the seed (`routes.ts:49-58`). |

Constraints preserved: Home is still first, Model Hub still second, About is still last, there are
still 14 links with the same route titles, and the sidebar's arrow-key/Home/End handling is
untouched (`components/Sidebar.tsx:35-44`).

### Home (`ui/src/screens/Home.tsx`, `ui/src/screens/home/Discover.tsx`)

| Finding | Change |
| --- | --- |
| The first-steps checklist offered "Install a plugin" on desktop, where Plugins is a planned screen and nothing can be installed. A checklist item that can never complete is worse than no item. | The step is browser-preview only (`Discover.tsx:76`). |
| The daily spotlight could promote Turbo builds that desktop Turbo does not have. | Desktop uses a filtered spotlight list (`Discover.tsx:35-41`). |
| The Image Studio spotlight named FLUX, SD 3.5 and Qwen-Image; the desktop screen runs SD 2.x-family folders such as SD-Turbo. | Desktop gets a truthful SD-Turbo variant (`Discover.tsx:39`). |
| "What's new" advertised Turbo machine profiles and image inpainting/upscaling that are not in the desktop build. | Desktop omits those two news items (`Discover.tsx:51`, rendered at `167`). |
| Home still carries a command-line section and a protections section that duplicate Models/Settings. | These are already browser-preview only (`Home.tsx:128-160`); kept as the design study, not reworked. |

The resource loading, live-tile placeholder (`Home.tsx:165-172`) and empty "Recent" state
(`Discover.tsx:171`) were already clear and were left alone.

### Model Hub (`ui/src/screens/hub/LiveHub.tsx`)

| Finding | Change |
| --- | --- |
| Fit labels and the machine note used "VRAM" and "RAM" in prose. | "Fits in graphics memory", "Graphics memory + system memory", and a note in the same words (`LiveHub.tsx:19-26`, `127`). |
| Loading, empty and error states were already present: "Searching…", "No GGUF repositories match.", file-list error text, per-download error plus Resume. | No structural change. The preview's states (`screens/Hub.tsx:105`, `139-147`) are equally complete. |

### Chat and the composer (`ui/src/screens/chat/Composer.tsx`, `ui/src/styles/chat.css`)

This is the only intended behaviour change.

| Finding | Change |
| --- | --- |
| Ten controls competed with the message box: Effort, Verbosity, Reasoning, No added prompt, Plan first, Orchestrate, Workers, Rounds and System prompt, plus a paragraph of explanation. | The Speed <-> Accuracy slider, its level note and the keyboard hint stay visible; the fine controls sit in one labelled `Fine-tune` disclosure (`Composer.tsx:293-384`). |
| A disclosure that forgets its state is worse than no disclosure. | The open/closed state is stored under `cerebro-chat-fine-tune` via the existing `readPref`/`writePref` helpers (`Composer.tsx:7-10`, `293`), and a test covers open → reload → closed → reload. |
| The custom-mode note referred to "settings below", which no longer describes a folded panel. | It now says "one of the settings in Fine-tune" (`Composer.tsx:291`). |
| The explanation of reasoning/effort, the Orchestrate explanation and the Enter/Shift+Enter/Esc hint were mixed together. | Effort/reasoning and Orchestrate notes sit inside the disclosure beside the controls; the keyboard hint stays under the composer (`Composer.tsx:375-387`). |
| The slider must keep working while the panel is closed. | Unchanged: the slider still calls `pickMode`, which applies every preset. Only where the controls are shown changed. |

The selector update keeps every existing assertion's intent: `tests/e2e/fixtures.ts:101` gains
`openFineTune`, and the chat, agent, orchestrate, slider and settings tests open the disclosure
before reaching the moved controls. The composer accessibility test runs with the disclosure open.

### Models (`ui/src/screens/models/…`)

| Finding | Change |
| --- | --- |
| The desktop build showed a "From the command line" section with `cerebro` commands. There is no command line in the desktop app. | The section is browser-preview only (`screens/models/ModelsScreen.tsx:36-40`). |
| Fit labels used VRAM/video memory in several places. | Library filter and fit chips say "Fits in graphics memory" (`lib/fit.ts:112`, `screens/models/Library.tsx:127`); the load estimate says "graphics memory"/"system memory" and the meters are named in full (`screens/models/LoadPanel.tsx:431-436`). |
| The memory line in the file tables said "VRAM + RAM". | The column is "Memory" with a full explanation in its title, and the line says graphics/system memory (`components/QuantTable.tsx:13-15`, `58-61`). |
| The Models lede names LM Studio and Ollama. | Kept deliberately: this is the interoperability fact that their folders are read in place, the project's own copy test allows the mention only there (`tests/unit/brand.test.ts:164-170`), and there is no comparative claim. |

Loaded/empty states ("Nothing loaded" vs "Checking…", `ModelsScreen.tsx:59`) and the unknown-machine
state (no invented estimates) were already handled well.

### Tune (`ui/src/screens/tune/…`)

| Finding | Change |
| --- | --- |
| A command-line section appeared on desktop; it does not apply there. | Hidden in the desktop build (`screens/tune/TuneScreen.tsx:150-154`). |
| "Free VRAM", "video memory" and a VRAM-labelled chart put raw jargon in the soak and result views. | Copy now says free graphics memory throughout (`screens/tune/Winner.tsx:235`, `charts.tsx:199`, `RunPanel.tsx:583`, `658`, `Setup.tsx:146`). |
| Section order when a run exists (result, then "Start another tune", then "How a tune runs" only before the first run) is correct. | No change. |

### Monitor (`ui/src/screens/monitor/…`, `ui/src/lib/monitor.ts`)

| Finding | Change |
| --- | --- |
| Alert titles were the densest copy in the app: "GPU is thermal throttling" and "VRAM almost full". A non-expert cannot act on either. | "Graphics card is too hot, so it is slowing down" and "Graphics memory almost full" (`lib/monitor.ts:342`, `348`). The regression detail now says "graphics card work" (`357`). |
| The chart and split-panel labels still said "VRAM". | "Graphics memory in use" and "Graphics memory split" (`screens/monitor/cards.tsx:419`, `434`, `706`). |
| The HUD label was "VRAM"; the longer plain-English name could overflow the status bar. | Changed to "Graphics memory" with the accessible name "Graphics memory in use" (`components/StatusBar.tsx:72-76`); the glitch sweep checks every route for overflow. |
| The screen is dense: totals, two cards, live metrics, cache, alert rules, breakdown, requests. | Grouping is coherent (live state → alerts → series → breakdown → raw requests), and each group is a named section. A tab split is proposed below but was not made. |

Loading ("Connecting to the engine…", `MonitorScreen.tsx:109-112`), stale-data and
engine-not-answering notices (`114-128`) were already good.

### Turbo (`ui/src/screens/Turbo.tsx`)

| Finding | Change |
| --- | --- |
| Desktop Turbo already hides the illustrative builds, schedulers, profiles and profiler; the daily spotlight and news still advertised them. | Home was corrected (above); the screen itself was left alone. |
| "Free VRAM" in the profiler. | "Free graphics memory" (`Turbo.tsx:364`). |
| Desktop shows machine tuning, engine switches, then Channels only in Advanced. | Order is logical and the Advanced gate is explicit. No change. |

### Image Studio (`ui/src/screens/ImageStudio.tsx`, `ui/src/styles/images.css`)

| Finding | Change |
| --- | --- |
| "Advanced settings" was a checkbox that revealed a block. A checkbox reads as an on/off feature, not as "show more controls", and it did not expose its expanded state. | A native `<details>` disclosure with a labelled summary (`ImageStudio.tsx:374-405`, styles at `styles/images.css:143-173`). It is keyboard operable and open by default at the Advanced experience level, preserving the old default. |

The desktop screen's no-model fallback ("Add an image model", `ImageStudio.tsx:79-94`), busy state,
result empty state and error alert were already complete.

### Documents (`ui/src/screens/Documents.tsx`)

| Finding | Change |
| --- | --- |
| The screen is honest that search is keyword-only, but the route summary was not. | Summary fixed (`routes.ts:50`); the limitation stays visible in the section description (`Documents.tsx:81`). |
| Drop zone, indexing progress, "No files yet", "Nothing … matches" and per-file removal were already clear. | No change. |

### Train (`ui/src/screens/Train.tsx`)

| Finding | Change |
| --- | --- |
| "No CUDA GPU: training will not start" is a double jargon hit (CUDA and GPU) for a beginner. | "No compatible graphics card: training will not start" (`Train.tsx:132`). The technical environment description still names PyTorch/CUDA because that is the thing being installed (`103`). |
| Setup before the job form, explicit setup progress and inline field errors. | No structural change. |

### Tools (`ui/src/screens/Tools.tsx`)

| Finding | Change |
| --- | --- |
| The desktop empty state correctly points at MCP servers above it, and the overview ("How much the model may do") precedes per-tool rules and Activity. | No change. Moving MCP servers would require re-ordering the rule flow; proposed below. |

### Plugins (`ui/src/screens/Plugins.tsx`)

| Finding | Change |
| --- | --- |
| Desktop opens a planned-feature page that points to Tools/MCP. It sat before the working Tools screen in navigation. | Navigation order fixed; the page itself already explains the plan honestly (`Plugins.tsx:23-47`). |

### Settings (`ui/src/screens/Settings.tsx`)

| Finding | Change |
| --- | --- |
| About sat in the middle, before Configuration file and First-run setup. About is reference material, not a setting. | About is last (`Settings.tsx:248-275`); Configuration file (`228`) and First-run setup (`242`) precede it. |
| Settings already contains a compact version of About (version, mode, licence, documents, link). | Kept as the quick reference; the full brand/mark story stays on the About screen. Fully removing the summary would hide the version from Settings. |
| Search, experience-level gating, channel lock and the disabled-while-loading switches were already consistent. | No change. |

### About (`ui/src/screens/About.tsx`)

No problems found. "What you get" → "The mark" → "Documents" is the right order, and version/mode
have loading fallbacks (`About.tsx:40-50`).

## Words, labels and states

- Prose now says **graphics card** and **graphics memory**. "GPU" remains in measured metric labels
  ("GPU load", "GPU temperature") because the glossary entry `gpu` explains it and those are the
  names experts look for; it is no longer used as an unexplained synonym in alert titles.
- "VRAM" remains as a glossary term (`lib/glossary.ts:14`) and in a few technical parameter names
  such as `VRAM`-backed expert-cache settings; the surrounding UI now says graphics memory.
- Buttons checked for verb-first, sentence-case copy: "Download", "Load", "Generate", "Run a tune",
  "Hide first steps", "Show in folder", "Run setup again". No new Title Case labels.
- Loading, empty and error states were checked on every screen; the ones that changed are listed
  above. No new spinner or skeleton was needed.
- The app's own copy test (`tests/unit/brand.test.ts:153-171`) still passes: no hype words and no
  product names outside the allowed interoperability files.

## Larger restructuring proposals not made

1. **Merge Model Hub and Models into one "Models" screen with Browse and On this computer tabs.**
   They are two halves of the same job, and the current split makes users guess whether a model is
   "in the hub" or "in the library". Not made because it merges two routes, breaks deep links and
   the `#/tune?model=` flow, and is far beyond layout/wording.
2. **Split Monitor into Live, History and Requests tabs.**
   The single page is long, but it is one live instrument and all sections share the same range and
   sample. Tabs would hide the relationship between an alert, its chart and the request that caused
   it. Not made without a design study.
3. **Merge Tune and Turbo under one "Optimise" area.**
   Both measure and keep settings, but Tune is per model and Turbo is per engine/machine, with
   different lifetimes and revert rules. A merge would need new navigation and state, not just copy.
4. **Move Tools' per-tool permissions into Settings and leave only Activity on Tools.**
   Permissions are settings, but they are used while working in Chat and need the tool descriptions,
   risk labels and activity beside them. Not made.
5. **Drop the browser-preview sections from the desktop bundle entirely instead of hiding them.**
   A build-time split would shrink the bundle and remove the `__DESKTOP__` conditionals. It is a
   build change, not a UX change, and would make the preview harder to maintain.
6. **Rename the composer disclosure to "More options" to avoid overloading "fine-tune" (Train).**
   The task asked for a single disclosure such as "Fine-tune", and the composer's note now makes the
   scope explicit. The two uses are in different contexts; renaming would lose the direct link to
   the slider's fine controls.

## Verification

From `apps/desktop`:

- `npx tsc --noEmit -p tsconfig.json` — pass.
- `npx vitest run --config ui/vite.config.ts` — 22 files, 314 tests, pass.
- `npx vite build --config ui/vite.config.ts` — pass.
- `npx playwright test --reporter=line` — see the commit message / run log; the full suite is run
  after the production build because the test server serves `ui/dist`.
- `node scripts/glitch-sweep.mjs` — checked the sweep's 13 routes at 1280 and 820 px, in both
  themes and both experience levels, for horizontal overflow, clipped text and console errors; see
  `review/sweep/` for the screenshots. Train is reviewed in source and exercised by Playwright, but
  is not in the sweep's route list.

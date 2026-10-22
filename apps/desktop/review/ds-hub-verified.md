# Model Hub: verified publishers and the best quant for this computer

Branch: `ds/hub-verified`

## What I built

### 1. Verified publishers

Cerebro now carries its own checked list of publishers people trust, in
`ui/src/lib/hub-verified.ts`. Each entry has the publisher's Hugging Face account name, the name
people know them by, and one line on what they publish.

- **Model makers** (original weights, and some of them their own conversions): Qwen, Google
  (Gemma), Meta (Llama), Mistral AI, Microsoft, DeepSeek, OpenAI, NVIDIA, IBM Granite, Hugging
  Face.
- **GGUF publishers** (other people's models, quantised): Unsloth, llama.cpp (ggml),
  lmstudio-community, Bartowski.

A publisher on that list gets a **Verified** badge: a seal icon and the word "Verified", so it
never depends on colour alone. Hovering it says what it means — *"On Cerebro's list of known
publishers. It is not a guarantee."* — and the same sentence is printed beside the filter, so
nobody has to hover to find out. The badge appears:

- on every search result whose publisher is on the list;
- on the repository page, next to the repository's name;
- on the front page's model cards, next to the family tile, when the model's maker is on the list.

**The order of the results.** `sortRepos()` puts the best match first and, at equal relevance,
the publishers on the list before the rest; downloads settle what is left. Relevance is how well
the repository's name matches the query, not a guess.

**A filter chip.** "Verified publishers only" sits under the search box, off by default. It leaves
only the publishers on the list; with nothing left, the results say so.

**Repositories from publishers nobody knows** are not hidden and are not decorated. Their row shows
how many downloads they have and how old they are ("268K downloads · updated 5 weeks ago · 610
likes"), so a reader can judge the row for themselves.

### 2. The best quant for this computer, picked by default

On a repository page, one file is marked **Recommended for this computer** and is the one the
page's Download button fetches. The choice comes from the repository's own file list and the
machine the app detected — never from a fixed model or a fixed machine.

`recommendQuant()` in `ui/src/lib/hub-quant.ts` works like this:

1. Unsloth Dynamic quants (`UD-`) are preferred when the repository has them, because they keep
   the most quality per byte.
2. Among the candidates, the **largest** file that fits this computer's graphics memory and
   system memory wins. The fit test is the same one the table's "On this computer" column uses: the
   file plus a margin, against the memory the machine can spare.
3. If no dynamic quant fits but an ordinary one does, that one is used — a file that runs beats a
   better file that does not.
4. If nothing fits at all, the **smallest** file is recommended and the reason says it will run
   slowly from the drive.

The pick is explained in one sentence, for example:

> UD-Q4_K_XL: the best quality that fits your 8 GB graphics card and 32 GB of memory.

Every other file stays in the list with its own Download button, one click away. Nothing is
hidden, and the recommended file's row is marked so it is obvious which one was picked.

### 3. Split models are one item

A model split into parts (`-00001-of-00002`) is one file everywhere:

- **One row** in the file list, saying how many parts it has ("2 parts, one download").
- **One progress bar** for the whole download, because the Rust side already fetched the parts in
  one pass.
- **One model in the library.** The library scan already opened only the first part; it now also
  reports the model's real size, with every part added up.

This needed a fix in `src-tauri/src/hub.rs`: the key that grouped the parts was cut at the *last*
"-000" in the name, so every part got a key of its own and a split model was listed — and
downloaded — part by part. The key is now the name without its "-00001-of-00003" tail, with the
number of parts kept so two different splits of the same name stay apart.

### 4. The front page's recommended models

The "Recommended for this computer" cards keep the existing fit ranking, and each card now shows
its family tile, the Verified badge of the model's maker, and the **recommended quant's real
size** — for example "Recommended UD-Q4_K_XL · 2.73 GiB". The size comes from the repository the
pick finds, read live through the same search and file list the Hub already uses, so no card
guesses a quant it has not seen. If a repository cannot be read, the card keeps its typical size
and says "about".

## Where it lives

| What | Where |
| --- | --- |
| The checked list, the lookup, the result order | `ui/src/lib/hub-verified.ts` |
| The quant chooser and the fit test | `ui/src/lib/hub-quant.ts` |
| The Verified badge | `ui/src/components/VerifiedBadge.tsx` |
| The seal icon | `ui/src/components/icons.tsx` |
| The Hub screen | `ui/src/screens/hub/LiveHub.tsx` |
| The styles | `ui/src/styles/hub.css` |
| The example repositories and files in the browser build | `ui/src/mock/mock-service.ts` |
| The split model's parts and its real size | `src-tauri/src/hub.rs`, `src-tauri/src/engine.rs` |

No new Tauri command was needed, so nothing had to be registered in `build.rs`, the capability
file, the isolation validator or `lib.rs`.

## How to use it

1. Open **Model Hub**. Every card on the front page shows the model maker's badge and the quant
   Cerebro would pick, with its real size.
2. Search for anything ("Coding", a model name, a publisher). Verified publishers come first at
   equal relevance; each result says how many downloads it has and how old it is.
3. Turn on **Verified publishers only** to leave out publishers nobody knows.
4. Click a repository. Its files are listed with sizes and what each one needs here. One file is
   marked **Recommended for this computer**, with the reason in one sentence and a Download
   button that fetches exactly that file.
5. Download it. A split model arrives as one item with one progress bar and lands in the library
   as one model.

## Accessibility

- The badge is an icon and the word "Verified", never colour alone, and it says in a tooltip and
  in a sentence on the page what it means.
- The filter is a real button with `aria-pressed`, so its state is announced.
- The recommended file's row carries `data-recommended` and a visible badge; the recommendation
  block's button has the accessible name "Download the recommended file <quant>", which still
  contains the visible word "Download".
- The file table's action column has a header for screen readers, and the table scrolls sideways
  on a narrow window instead of pushing the page wider.
- The category chips were reading the browser's own button background, which failed contrast in
  the dark theme once a chip was pressed; they now carry the app's own background.
- axe scans of the whole live Hub, with results, a repository open and the recommendation
  showing, found no violation in either theme.

## Tests

Added `apps/desktop/tests/unit/hub-verified.test.ts`:

- every publisher has an id, a name and something to publish, and no id is listed twice;
- the model makers and GGUF publishers are all on the list;
- the lookup finds a publisher whatever the case, and leaves an unknown one alone;
- the badge's wording promises nothing;
- the order: a verified publisher first at equal relevance, the better match first whatever the
  publisher, downloads last, and the list given is not changed.

Added `apps/desktop/tests/unit/hub-quant.test.ts`:

- a dynamic quant is preferred when the repository has one;
- the largest dynamic quant that fits is chosen, and the reason names the machine;
- the largest file that fits is chosen when there is no dynamic quant;
- when nothing fits, the smallest file is recommended and the reason says it will be slow;
- a dynamic quant that does not fit gives way to one that does;
- a split model is one file with its parts;
- nothing is recommended when the machine or the repository is empty;
- the fit test agrees with the Hub's table.

Added `apps/desktop/tests/e2e/hub-verified.spec.ts` (dark and light):

- a verified publisher carries the badge, with the tooltip that says what it means;
- a repository from an unknown publisher shows its downloads and its age;
- the filter is off by default, leaves only verified publishers, and explains the badge in words;
- one file is recommended, marked in the list, explained in one sentence, and the Download button
  fetches that file;
- axe finds no violation with results, a repository open and the recommendation showing.

Rust tests:

- `hub.rs`: the parts of a split model are one file with their total size, two different splits
  stay apart, and only a real "-00001-of-00003" tail makes a key.
- `engine.rs`: a split model counts every part, so the library shows the size the model really
  takes.

Commands run from `apps/desktop`:

- `npx tsc --noEmit -p tsconfig.json` — passed.
- `npx vitest run --config ui/vite.config.ts` — 29 files, 360 tests passed.
- `npx vite build --config ui/vite.config.ts` — passed.
- `npx playwright test --reporter=line` — 348 tests passed.
- `node scripts/glitch-sweep.mjs` — 0 cases with problems.
- `cargo test -p cerebro-desktop` (with the MSVC environment) — 154 passed, 18 ignored.

## Notes and limits

- The badge says a publisher is on Cerebro's list. It is not a guarantee about a file, a licence
  or a model, and the wording on screen says so.
- The publisher list is checked by hand and lives in the app; it is not read from the network.
- The front page reads each pick's repository to name its recommended quant, so opening the Hub
  searches Hugging Face for the picks. With downloads or the network switched off, nothing is
  searched and the cards keep their typical size.
- The fit test is arithmetic on the file's real size and the memory the machine can spare. It is
  not a measurement: Cerebro measures the real speed after the download.
- The lmstudio-community publisher is listed under its account name, because the app never names
  another product in its copy.

# AI tools: documents in chat, converters and quick tools

Branch: `ds/ai-tools`

## What I built

### 1. Answer from your documents (RAG)

Chat can now answer from the files you indexed on the Documents screen.

- **Off by default.** The switch **Use my documents** is in the composer's *Fine-tune* area, and
  `/docs` turns it on from the message box. `/docs off` turns it off again.
- **The passages go with the message.** For each message the best passages of the indexed files
  are found (BM25 over every passage, at most two per file) and put into the request, numbered
  `[1]`, `[2]`, with the file each came from. The model is told to quote the number.
- **The reply shows what it used.** Under the reply there are chips, one per passage, with the
  file name. Clicking a chip opens that passage in Documents (`#/documents?doc=…&p=…`), where it
  is shown at the top of the search tab.
- **One collection, folder or tag.** The select under the switch says what is searched: all
  documents, one collection, one folder or one tag. `/docs <name>` picks the same thing by name.
- **Cloud models are told.** When a cloud model answers, the composer says the passages are sent
  to that provider with the user's own key. Offline mode still blocks cloud models outright, as
  it always did.
- **Everything stays on the computer** unless a cloud model is picked. The passages are added to
  the request the app already sends; no new network path was opened.

Retrieval and the prompt block are in `ui/src/lib/rag.ts`; the chat-side settings are in
`ui/src/lib/chat/docs.ts`; the switch is in `ui/src/screens/chat/Composer.tsx`; the chips are in
`ui/src/screens/chat/Message.tsx`; the wiring is in `ui/src/screens/Chat.tsx`.

### 2. Collections, folders and tags

- A collection is a named group of files. Create one on Documents > Files, then tick it for each
  file. A file can be in several collections.
- Each file also has one optional folder and up to eight tags, set behind the row's disclosure.
- Chat searches one collection, folder or tag, or everything.
- Collections, folders and tags are stored on this computer, with the documents themselves.

The code is in `ui/src/lib/collections.ts`; the fields on a file are in `ui/src/lib/documents.ts`.

### 3. Converters

Documents > Files now converts what you drop, instead of refusing it, and Documents > Convert
turns any of those files into Markdown or plain text.

| Format | How it is read |
| --- | --- |
| Plain text, Markdown | As they are (Markdown is stripped to words for the index) |
| HTML | The browser's own parser: tags are dropped, headings and lists become Markdown |
| CSV | Rows become a Markdown table, and `a \| b` lines for the index |
| JSON | One value a line for the index, the JSON itself as Markdown |
| PDF | The page content streams are read: the strings, in order, with the file's own character map |
| Word (.docx) | The document XML is read: headings, lists and tables |

- Every file that is added says what came of it: the characters read, the pages when the format
  has them, or why it failed.
- The Convert tab shows the result and saves it as Markdown or plain text, where you pick.
- Nothing is uploaded and no network call is made. The PDF and Word readers, and the DEFLATE
  decompressor they need, are Cerebro's own code (`ui/src/lib/convert/`), so no third-party
  package was added: NOTICE and the bundle budget are unchanged.
- The PDF reader keeps the text, not the layout. A scanned page is a picture and has no text to
  read; Cerebro does not do OCR.

### 4. Quick tools

One tab of Documents holds nine one-step tools, and each is also a `/` command in Chat.

| Tool | What it does | Command |
| --- | --- | --- |
| Summarise | The main points of a text, a file or the last message | `/summarise` |
| Translate | The text in another language | `/translate Spanish` |
| Rewrite | Shorter, clearer, formal or friendly | `/rewrite shorter` |
| Extract | The fields you name, as JSON (checked before it is shown) | `/extract name, role` |
| Explain code | What a piece of code does | `/explain` |
| Compare two documents | What agrees, what changed | `/compare` |
| Meeting notes | An audio file: summary, decisions, actions | `/notes` |
| Read aloud | The speech model reads the text | `/read` |
| Describe an image | What is in a picture | `/describe` |

- Each card has one input, one button, and the result with **Copy** and **Save**.
- A tool that needs a model that is not loaded says so plainly and offers **Open Models**.
- The instructions live in one file, `ui/src/lib/quick-tools.ts`, so they are easy to tune. The
  cards and the commands use the same words.
- Extract checks the answer really is JSON and says what is wrong when it is not.
- Meeting notes reads an audio file with the speech model on this computer (a new
  `stt_transcribe_file` command), then asks the text model for the summary, the decisions and
  the actions.

### 5. One tidy home

The CREATE group still holds exactly Image Studio, Documents and Train; no sidebar entry was
added. Documents is now one screen with four tabs:

- **Search** (the screen opens here): ask a question, get the passages that answer it.
- **Files**: drop files to index them, group them in collections, set a folder and tags.
- **Convert**: turn a file into Markdown or plain text and save it.
- **Quick tools**: the nine cards.

The finer things (a file's folder, tags and collections) sit behind a labelled disclosure on
each row, so the list stays short.

### 6. The feature guide

`ui/src/lib/features.ts` gained the new capabilities (answering from your documents, collections,
converters, quick tools), marked as new, and the guide version was bumped to `2026-10-22`, so
the guide shows once more.

## How to try it

1. **Answer from a document.** Open Documents > Files, drop a text file, then open Chat, open
   *Fine-tune*, tick **Use my documents**, and ask a question about the file. The reply shows the
   passages it used; click a chip to see the passage.
2. **A collection.** On Documents > Files, add a collection called *Policy*, tick it for a file,
   then in Chat pick **Collection: Policy** under the switch. `/docs Policy` does the same.
3. **A converter.** On Documents > Convert, choose a PDF or a Word file; the result appears with
   the characters and pages read. Choose **Save** and pick where it goes.
4. **A quick tool.** On Documents > Quick tools, paste a paragraph into **Summarise** and press
   the button. The same job from Chat: type `/summarise` and press Enter.
5. **Meeting notes.** Pick an audio file on the **Meeting notes** card; the speech model
   transcribes it and the text model writes the notes.

## Tests

Added `apps/desktop/tests/unit/`:

- `rag.test.ts`: the passages that answer, one file's share of the budget, the numbering, the
  dedupe, the token budget, and the prompt block with its quoting rule.
- `collections.test.ts`: reading collections back, adding and removing files, the scopes, and
  the keys they are stored as.
- `quick-tools.test.ts`: every tool's prompt assembly, the fields, the JSON check, and the
  languages and styles offered.
- `converters.test.ts`: each converter against the checked-in fixtures (`tests/fixtures/`), plus
  the DEFLATE decompressor against Node's own zlib and the ZIP reader against the Word file.

Added `apps/desktop/tests/e2e/`:

- `chat-docs.spec.ts`: the documents off by default, `/docs` on and off, the source chips, a chip
  opening the passage in Documents, and `/summarise` and `/translate` in Chat.
- `documents.spec.ts`: the tabs, dropping a file, the per-file result, a PDF converted and
  searchable, collections with a folder and tags, the Convert tab, the Quick tools tab
  (Summarise, Extract and Translate), the message for a model that is not loaded, and axe scans
  in dark and light on every tab.

Commands run from `apps/desktop`:

- `npx tsc --noEmit -p tsconfig.json` — passed.
- `npx vitest run --config ui/vite.config.ts` — 31 files, 381 tests passed.
- `npx vite build --config ui/vite.config.ts` — passed.
- `npm run build:desktop` — passed, including the bundle budget (first paint 145.9 of 150 KB).
- `npx playwright test --reporter=line` — 351 tests passed.
- `node scripts/glitch-sweep.mjs` — 0 cases with problems.
- `cargo test -p cerebro-desktop` (with the MSVC environment) — 153 passed, 18 ignored.

The Rust change is one command, `stt_transcribe_file` (an audio file through the speech model).
It is registered in the four places the shell requires: `build.rs` COMMANDS,
`capabilities/main-window.json`, `isolation/index.js` and `lib.rs` `generate_handler!`. The
existing test `tests/unit/tauri-config.test.ts` checks that the four lists and the UI agree.

## Notes and limits

- The documents index still lives in this computer's local storage (about 2 million characters).
  A file longer than 200,000 characters is indexed from its start, and the screen says so.
- Retrieval is keyword search (BM25), not search by meaning. The screen says so as well.
- The PDF reader takes the text of the page content streams. It does not keep the layout, and it
  cannot read a scanned page.
- The Word reader takes the paragraphs, headings, lists and tables of the document XML.
- `/compare` compares the last two messages of the chat; `/notes` in Chat works on a pasted
  transcript (the card on Documents takes the audio file).
- `/describe` in Chat describes the picture attached to the last message; the card takes a file.
- No third-party package was added, so NOTICE is unchanged.

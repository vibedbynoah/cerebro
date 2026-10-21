# Chat features: branches, search and a prompt library

Branch: `ds/features-chat`

## What I built

### 1. Branches in a conversation

Chats are now stored as a tree. Every message is a node with a parent, and the tree remembers
which child is selected at each step. `messages` is still saved as the active path, so a chat
opened by an older build stays readable.

- **Edit any earlier message.** Use the pencil button or double-click the message text. Saving
  adds the new wording as a sibling, keeps the old wording, and asks for a new reply.
- **Regenerate any reply.** Each finished reply has Regenerate, not just the last one. The new
  reply is a sibling, so the old answer stays.
- A message with siblings shows `< 1/2 >`. The arrows wrap around and are disabled while a reply
  is streaming. The control is only shown when there is more than one branch.
- Switching a branch keeps the chosen child below it, so each turn remembers its own path.
- After a switch, focus moves to the message now shown, which keeps keyboard use predictable.
- Chats saved before this change have no tree. They are read as one straight path and saved with
  a tree on the next change. A broken or cyclic tree in storage falls back to the saved message
  list instead of hanging or hiding the chat.

Storage code is in `ui/src/lib/chat/store.ts`; the operations are wired in
`ui/src/screens/Chat.tsx` and the controls live in `ui/src/screens/chat/Message.tsx`.

### 2. Search across every chat

Full-text search covers chat titles and every message in every branch, not only the active path.

- **Ctrl+Shift+F** opens a search dialog from any screen. It is also in the command palette
  (Ctrl+K) as **Search chats**.
- The **search box in the chat list** shows results in place, with the result count and a Clear
  button.
- Matches are highlighted in titles and snippets. Matching is case-insensitive, and every word in
  the query must appear. Snippets are cut around the first hit.
- Choosing a result opens the chat and jumps to that message. If the message is on another
  branch, that branch is opened first. The target gets a temporary outline and keyboard focus.
- Chat publishes its live list to the shell, so a just-sent message is searchable before the
  debounced save reaches local storage.

Search logic is in `ui/src/lib/chat/search.ts`; the dialog is
`ui/src/components/ChatSearch.tsx`; the shared navigation state is
`ui/src/lib/chat/navigation.ts`.

### 3. Pin, folders, tags and archive

- **Pin** a chat with the pin button. Pinned chats sort above the rest, newest first.
- **Folder:** one optional folder per chat. Existing folders appear in the filter and in a
  suggestion list as you type a folder name.
- **Tags:** comma-separated, up to eight per chat. Tags appear in the tag filter.
- **Archive:** archived chats are hidden by default. Turn on the **Archived** checkbox to see
  them; the same checkbox is in each chat's options to archive or restore it.
- Folder, tag and archive controls are behind the **Chat options** disclosure on each row, so the
  list stays compact.
- Filters combine: a folder plus a tag shows only chats that match both.

These fields are parsed and bounded in `store.ts` and shown in `Chat.tsx`; styles are in
`ui/src/styles/chat.css`.

### 4. Prompt library

The library is a side panel on Chat, reached by:

- the book button in the Chat header,
- the **Prompts** button in the composer,
- typing `/prompt` (or `/prompts`) and pressing Enter.

It includes:

- **15 starter prompts** across Writing, Coding, Study, Work and Everyday, including summarise,
  explain code, translate, email reply, study notes, find bugs, write tests, meeting notes,
  rewrite clearly, plan a task, compare options, brainstorm, practice questions and a recipe.
- **Variables:** any `{{name}}` in a prompt becomes a labelled text field. **Use prompt** fills
  the composer with the completed text, ready to edit and send. Missing values are explained in a
  toast instead of sending a half-filled prompt.
- **Categories and search** to narrow the list; custom categories are added after the built-ins.
- **Create, edit and delete** prompts, with plain JSON import and export. Import accepts an array
  or the exported `{ version, prompts }` object, checks each field, merges by id and ignores
  malformed entries. Export downloads `cerebro-prompts.json`.

Everything is stored on this computer in local storage. The logic is in
`ui/src/lib/chat/prompts.ts`; the panel is `ui/src/screens/chat/PromptLibrary.tsx`; the composer
hook is in `ui/src/screens/chat/Composer.tsx`.

### Accessibility

- Every new control has a visible label or a specific accessible name.
- The search dialog supports arrow keys, Enter to open, Escape to close and a live result count.
- Branch controls, edit forms, folder/tag fields and prompt forms are keyboard reachable and have
  focus-visible rings.
- New screens use the existing colour tokens; axe scans in dark and light found no violations for
  branches with the prompt library open or the global search dialog open.

## How to try it

1. **Branch a chat:** send a message, click the pencil on the message, change the wording and
   choose **Save and branch**. Use `<` and `>` to compare. Or press **Regenerate** on any reply.
2. **Search:** press **Ctrl+Shift+F** anywhere and type a few words. Or use the search box above
   the chat list. Choose a result to jump to the message.
3. **Organise:** click the pin on a chat, open **Chat options** to set a folder or tags, or tick
   **Archived**. Use the folder and tag selects and the **Archived** checkbox to filter.
4. **Prompts:** open **Prompt library**, pick a prompt, fill its variables and choose **Use
   prompt**. Try `/prompt` in the composer. Export the library, edit the JSON and import it back.

## Tests

Added `apps/desktop/tests/unit/chat-features.test.ts`:

- branch migration from an old chat, and safe fallback for a cyclic tree;
- sibling switching and opening the branch that contains a searched message;
- keeping the newest part of a very long chat when storage is trimmed;
- title/message search, all-word matching, highlight runs and snippets;
- the 15 starters, variable extraction and filling, JSON validation/import, merge-by-id and
  category order.

Added `apps/desktop/tests/e2e/chat-features.spec.ts`:

- edit an earlier message, see `2/2`, switch back and forth;
- regenerate a reply and switch between the old and new reply;
- search the chat list, open a message on another branch, and confirm the target is highlighted;
- Ctrl+Shift+F from Settings and Enter to jump to a result;
- pin, folder, tag and archive filtering, including the archived view;
- prompt variables, missing-variable feedback, `/prompt`, creating a prompt, category filtering,
  JSON export and import;
- axe scans (dark and light) for branches with the prompt library open and for the global search
  dialog.

Commands run from `apps/desktop`:

- `npx tsc --noEmit -p tsconfig.json` — passed.
- `npx vitest run --config ui/vite.config.ts` — 23 files, 326 tests passed.
- `npx vite build --config ui/vite.config.ts` — passed.
- `npx playwright test --reporter=line` — 307 tests passed.
- `node scripts/glitch-sweep.mjs` — 0 cases with problems.
- `cargo test -p cerebro-desktop` (with the MSVC environment) — 141 passed, 16 ignored.

## Notes and limits

- A chat's active path keeps its newest 500 messages when it is read back, and chats are capped
  at 50. A branch tree larger than 500 nodes is trimmed to its newest nodes; if the trim leaves
  no root, the chat reopens as a single recent path rather than losing the newest messages.
- A chat has at most one folder and eight tags; tags are 30 characters each and folders 60.
- Archived chats are still searchable. Opening one from search shows it even when the archive
  view is off; tick **Archived** in the list to see where it lives.
- Changing the first message does not rename the chat; the title stays as it was when the chat
  started.
- No Tauri command or Rust change was needed. All new data is local storage, so the mock browser
  build and the desktop build behave the same.

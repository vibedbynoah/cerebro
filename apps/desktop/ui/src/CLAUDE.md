# UI (SolidJS): reviewer checklist

## Product rules the owner set (each break is a finding)
- **Voice**: third person, plain adult language; no "you"/"your", no cute or childish wording.
- **No seeded data**: nothing in the UI that the user did not enter (no sample chats, memories, keys).
- **Apple principles**: forgiveness (Undo window or confirmation before anything destructive), errors that
  say how to fix without blame, instant visual feedback on every state change, interruptible animation,
  every element earns its place, behave like familiar apps.
- **Consistency**: switches (not raw checkboxes) for on/off, collapsible groups (`components/Screen.tsx`
  `Group`) for long pages, the same component for the same job on every screen, design tokens only (no new
  colours outside the slate/blue palette; `tests/unit/brand.test.ts` enforces it).
- **Golden path**: defaults that work without setup; memory tiers and context size are automatic, not
  settings; optimistic UI where the backend confirms later; one failing panel never blanks a screen.
- **Model names** everywhere read `name · quant · publisher` (`lib/model-label.ts`), so variants differ.

## Correctness to check
- Every screen works against the mock service (`ui/src/mock`) and the desktop service the same way.
- Long lists are not capped silently (a "10 models" limit was reported once).
- Stop / Cancel / Unload are reachable while work runs, and the UI recovers if the backend errors.
- Keyboard and screen readers: every control has a name, focus rings show (`tests/e2e/a11y.spec.ts`).

## Speed
- First paint budget ~150 KB of JS (lazy-load screens and heavy panels); no work in render paths that
  belongs in `createMemo`; no polling faster than the data changes.

# Cerebro Design System

The brand on one page. Mechanism detail lives in the code comments of the file named in each row.
If a decision isn't here, it isn't made yet: add it here first.

## Personality

A measurement instrument, not a chatbot skin. **Calm, precise, engineered, warm.** Think of the
display on a good oscilloscope, owned by a person rather than a fleet.

- **In:** generous whitespace, one amber accent used sparingly, numbers in a monospace face that
  align like a readout, motion that always has a cause.
- **Out:** purple-to-blue "AI" gradients, glassmorphism, confetti, exclamation marks, and any
  number we haven't measured.

## Color roles (`tokens.css`)

Dark is the default; light is equally finished. Every pair below meets WCAG 2.2 AA in both themes,
measured from rendered pixels by `npm test`. Each color is authored once as `light-dark(LIGHT, DARK)`;
`npm run gen` writes the same values as a plain fallback block for engines without `light-dark()`
(before Chrome 123 / Safari 17.5), and `npm run check` fails if the two drift. A `[data-theme]`
island re-themes everything inside it.

| Role | Token | Use |
|---|---|---|
| Background | `--color-bg` | Page and window. |
| Surfaces | `--color-surface-1/2/3` | Card, raised card, popover. Never skip a level. |
| Border | `--color-border` | Decorative dividers. No contrast duty. |
| Strong border | `--color-border-strong` | Inputs and bounded edges. 3:1 on every surface. |
| Text | `--color-text-primary/secondary/muted` | Prose and values, supporting copy, metadata. Nothing required to proceed is muted. |
| Accent fill | `--color-accent` + `-contrast` | The one primary action, the tuned bar. Never text. |
| Accent text | `--color-accent-text` | One live value or link per screen. Never a fill. |
| Status | `--color-success/warning/danger` + `-contrast` | Status only, always with an icon or label. Danger is 30°+ of hue or 15%+ lightness from the accent fill. |
| Text-role separation | accent-text, warning, danger, success | Every pair is 30°+ of rendered hue or 0.08+ OKLCH lightness apart in both themes (tested), so a link never reads as a warning. Light foregrounds must hold 4.5:1 on surface-3, which pins them near L 46%, so the separation comes from hue. |
| Disclosure | `.stat-tile__flag` (secondary text, surface-2, dashed strong border) | "Illustrative data — not a measurement". A disclosure, not a warning: never the warning color, so it never competes with the accent value beside it. |
| Hover / press | `--state-hover`, `--state-press`, `--state-press-ghost`, `--state-press-danger` | Hover lightens in dark and darkens in light. Press always darkens a filled button (it reads pushed in, never lit up); ghost controls (tertiary, icon) use `-ghost`, which stays light in dark because there is no fill to darken. Destructive has no fill either, so it hovers with `--color-danger-muted` and presses with `-danger`, a deeper tint of its own hue (a low-lightness red in dark). A press is never lighter than the hover before it. Rest to pressed is at least 0.02 OKLab apart for every role in both themes, because on touch the press is the only feedback. Text on every pressed state is in the contrast test. |
| Focus | `--focus-ring` | Two-tone ring (bg gap, then text-primary) on every `:focus-visible`. Never the accent. |
| Backdrop | `--color-grid-line`, `--color-glow` | `.backdrop-grid` texture behind the hero. At most one per page. |

## Type pairing (`fonts.css`)

- **Fraunces Variable**: section headlines and the hero only, at `--text-title` size or larger,
  never more than ~40 characters.
- **Inter Variable**: all UI and body text.
- **JetBrains Mono Variable**: code, flags, paths, and every number that is data.

A Fraunces headline is always followed by Inter body copy. Fallback stacks are metric-matched, so a
font swap doesn't move the page. Preload Inter and Fraunces latin in the consumer's `<head>` with the
bundler's URL for the file (`import url from "…woff2?url"`, then the same `url` in the preload), never
a literal `/fonts/…` path that a hashed build won't match (`fonts.css` header).

## Spacing

4 px base (`--space-1` to `--space-32`). Inside components use 4/8/12/16; between components and
sections use 24/32/48/64 and up. A value not on the scale is a bug.

**Elevation N = surface N + shadow N** (`.elev-1/2/3` in `components.css`): resting card, raised
card, popover. On near-black a drop shadow is nearly invisible, so the lighter surface carries the
distance in dark and the shadow carries it in light. Never pair a high shadow with a low surface.

**A number and its unit are one unit of type**: no literal space between them in markup; the unit
takes `margin-inline-start: var(--gap-value-unit)` (0.3em of the number), in every readout.

Stacked blocks inside a section (two grids, a list and a grid) are separated by one flow rule
(`--space-6`), never by per-component margins that can collapse to 0.

Radii follow component size: `sm` checkbox and code chip, `md` input and button, `lg` popover and code
block, `xl` card and stat tile, `2xl` panel and hero instrument, `pill` status chip and flag.

## Motion principles (`motion.css`, `motion.ts`)

1. **Every motion has a cause**: scroll, hover, press, or data arriving. Nothing loops.
2. **Transform and opacity only.** One exception: draw-on (`stroke-dashoffset`).
3. **Duration matches distance**: press 120 ms, local UI 200-320 ms, section reveal 600 ms. Only data
   arriving (count-up, draw-on) may take 900 ms.
4. **Easing has meaning**: `--ease-standard` for state changes, `--ease-emphasized` for reveals,
   `--ease-spring` for a physical press or pop.
5. **Reduced motion shows final states instantly**, is re-checked live, and never replays data that
   has already arrived.
6. **Never flash the final state, never strand a hidden one.** Start states (reveals, count-ups and
   `[data-draw]` strokes, wherever they sit) are hidden before first paint (`theme-init.js`), with a
   CSS failsafe, so no-JS and failed loads still show a complete page. After that, **only armed
   elements are ever hidden**: a primitive sets `data-motion-armed` on each element it observes and
   removes it on cleanup, so unarmed content (mounted later, or never initialised) always renders final.
7. **A stagger is a group arriving, never a queue.** Delays are set per batch of elements that enter
   the viewport together, in document order, capped at 6 steps (360 ms). An element that scrolls in
   alone starts at once.
8. **Init per component, or watch.** Every `init*` (motion.ts, terms.ts) is safe to call once per
   SolidJS component or Astro island, and skips elements another live init already owns.
   `initMotion(root, { watch: true })` arms nodes added under `root` later (a `<For>` list, chat
   messages), skips added nodes with no motion attribute, releases a subtree's listeners and
   observers as soon as it leaves the DOM (a deleted message is garbage-collectable, tested), and its
   cleanup disconnects the observer. Anything named globally (CSS anchor names)
   comes from a module-level allocator, never a per-call counter.
9. **Data in markup must be right without JS in every engine.** A comparison bar's final width is a
   `.bar--pNN` class (`.bar--p18` = 18%); `data-bar-to` is only a finer runtime override.
10. **Reveal goes on a wrapper, never on an interactive element.** `.reveal`/`.pop-in` and the press,
   magnetic and hover primitives each declare their own `transition`; on one element, one silently
   wins (the card snaps in, or the button stops springing). `initScrollReveal` warns in the console
   when it finds one on a button, link, input or `[data-press]`/`[data-magnetic]` element.
11. **Reduced motion never leaves a dead control.** A control whose job is to show motion (the
   specimen's Play) shows the end state instead and says why in a `role="status"` hint.
12. **A drawn stroke has no dash.** Draw-on ends by clearing `stroke-dasharray` (`.is-draw-done`), so
   the final state never depends on dash arithmetic. A `vector-effect: non-scaling-stroke` path in a
   stretched SVG gets a measured upper-bound dash length; if it can't be measured, it is shown drawn
   and the console says why. `[data-draw]` strokes are solid.
13. **The LCP element is never behind a reveal.** The hero heading and lede (the largest thing on
   first paint) and the primary CTA render static, so they paint on the first frame even when the
   motion module is slow or fails. Motion goes to the secondary pieces: a proof row, the trace
   panel's draw-on and count-up.
14. **Bars slide, they don't scale.** A comparison or progress bar is a pill whose box is its final
   length (a static width from `--bar-to`); it slides in with `translateX` from behind the left edge
   of its overflow-hidden track, and a live change slides from the old length (`setBarValue()`), so
   both caps stay round. `scaleX` squashes the caps into square edges. The finished box never
   extends past its track, so it never overlaps the label or value beside it. Instrument scales draw every tick as its own 1 px image.
15. **Hover is for pointers that hover.** Every `:hover` visual sits behind `@media (hover: hover)`;
   on touch, a tapped control would otherwise keep its hover look. `:focus-visible` is separate.

## Button interaction catalog (`components.css`)

| Role | Look | Micro-interaction |
|---|---|---|
| Primary | Accent fill | Press spring + light sweep. One per screen. |
| Hero CTA | Accent fill, 48 px, semibold, glow; an `<a>` (it navigates) | Magnetic + press + sweep; optional arrow nudge. Emphasis comes from the glow and pull, not size. |
| Secondary | Surface-2, strong border | Press spring. |
| Tertiary | Text only | Overlay fade + press spring. |
| Destructive | Danger outline | Firm press (`data-press="firm"`): no overshoot, sweep or magnetic. |
| Icon | Ghost square, 20 px glyph | Press spring; magnetic only in marketing contexts. |
| Busy / disabled | `aria-busy` spinner / disabled tokens | None. Clicks and Enter/Space are blocked. |

The press spring follows the key that activates the element: Enter or Space on a button, Enter only
on a link (Space on a link scrolls the page, so squashing it would promise an action that never
comes). A link released by navigation is un-pressed on `pagehide`, so it never returns squashed from
the back/forward cache.

## Accessibility rules

- Color is never the only signal.
- Navigation is always `<a class="btn ...">` (middle-click and open-in-new-tab work, the role is
  link); an action on this page is `<button type="button">`. A hero "Get started" is a link.
- A specimen or picture of controls (like the state bench) is `inert`, with one text description,
  so keyboard and screen-reader users don't step through dead buttons.
- Every focusable element shows the focus ring; it wins over any consumer CSS.
- A `<select>` is always `<span class="select"><select class="input">`: the wrapper draws the
  chevron from two borders (no gradient or image on the control, which the CSP forbids and which
  hides the control's background from contrast checkers), the control is as wide as its widest
  option, and it matches the height of a button in the same row.
- Hit targets: 40×40 px standalone, 24×24 px minimum inline (term `?` buttons are 28×28).
- Explain every technical term on first use with the `.term` primitive (keyboard and touch, never
  hover-only).
- Count-ups keep the final value in the markup and in an `.sr-only` twin.
- `aria-busy="true"` is never static. Set it on a region (a step list, not the card) only while one
  batch of updates is applied, and remove it on the next frame; a live `role="status"` line sits
  outside that region. Some screen readers hold back a busy region's contents until it clears.
- Specimen-only affordances (forced hover/press/focus on the state bench) live in `preview.css`
  behind `[data-force]`, never in `components.css`, where a common class like `is-active` would
  leak into the products.

## Do / don't

| Do | Don't |
|---|---|
| One Fraunces headline per section. | Fraunces for paragraphs, buttons or table headers. |
| One accent button per screen. | Three accent buttons side by side (the specimen's state matrix is labeled as a bench for this reason). |
| Stagger a group of cards. | Stagger a single element. |
| Flag placeholder numbers "Illustrative data — not a measurement" inside the same figure. | Show a bare speedup, or a measured number without its machine, model, build and context (`docs/CLAIMS.md`). |
| Write security definitions line by line from the implementing code or `docs/CLAIMS.md`, including what the mechanism does not do ("It does not limit which files or network addresses that process can open"). | Paraphrase a mechanism into a bigger promise ("inherits none of your other programs' settings"). |
| Lead with the product composed in tokens (the "In context" result card), including its honest outcomes: measuring, and "Defaults were fastest". | A page of token lists with nothing built from them, or only ever the win case. |
| Give a non-win a neutral chip (`.result-card__status--neutral`) and say plainly why. | Paint "Defaults were fastest" in the success color, or hide it. |
| State sampler settings as values (`temp 0.8, top-k 40, min-p 0.05, set explicitly`). | Write "defaults" in a provenance row: Cerebro never inherits a llama-server default (`docs/CLAIMS.md`). |
| Section copy: what the thing is for, in one or two sentences. Mechanism goes in code comments. | Explain the implementation to the reader, or defend it. |
| Use the disabled and busy states from `components.css`. | Grey a button out by hand with opacity. |
| Keep the hero headline and lede static; reveal the proof row and trace panel. | Fade in the headline: it delays the largest paint and leaves no-JS visitors a blank hero. |
| Use a glyph only if the shipped font subset has it (arrows are inline SVG). | Let one character fall back to another face inside an instrument readout. |

# Cerebro brand

Internal working file. Nothing in this folder is deployed except what `npm run brand` copies into
`public/`, and no public page ever uses the words "identity" or "brand guide".

## The idea

Cerebro is a fast local mind that routes each piece of work to the few experts that should handle
it, and measures which route is fastest on your machine. The mark draws exactly that:

- **Seven experts** (rounded squares on a 3 x 3 grid) form a **C**. The open side is the way in.
- **The router** (the circle) sits at the centre of the C.
- **One route** (the stem) connects it to the **one expert it picked** (top middle).
- Only the route is blue. Everything idle is ink. One of seven, like a mixture-of-experts model
  activating a small share of its weights per token.

It is a single ownable idea: no brain, no sparkle, no robot head, no letterform borrowed from
anyone else. The grid comes from the per-layer routing heat map that the app shows live.

## Mark

| File (`svg/`) | Use |
|---|---|
| `mark.svg` | Blue on light backgrounds. The master geometry. |
| `mark-on-dark.svg` | Blue on dark backgrounds. |
| `mark-mono.svg` | One colour. Inlined it takes `currentColor`; as a file it is black. |
| `mark-mono-white.svg` | One colour, white: photos, single-ink print, embossing. |
| `mark-adaptive.svg` | Follows the viewer's colour scheme (the site favicon). |
| `mark-small.svg` | 16 px optical size only (inside the 16 px app tile). |

**Geometry.** 24-unit grid. Experts are 6 units with a 1.5-unit corner radius, channels between
them 3 units, the router a 6-unit circle, the stem 3 units wide (as wide as a channel). Rendered at
any multiple of 8 px, every edge lands on a whole pixel: 16 px gives 4 px experts and 2 px channels.
At 16 px inside a tile there is not enough room for 2 px channels, so `mark-small.svg` redraws it on
a 14-unit grid (4 px experts, 1 px channels). Never scale the master into a 16 px tile.

**Legibility range.** 16 px (favicon, tray) to billboard. Below 16 px, do not use the mark.

## Wordmark

`svg/wordmark.svg` is drawn, not typeset: lowercase, monoline, a 2.4-unit stroke on a 12-unit
x-height with an 18-unit ascender, geometric bowls, flat terminals that echo the square experts.
It is spaced by eye (round to straight 1.6 units, the r terminal tucked over the next bowl).
Never re-track it, set it in a font, add a capital, or outline it.

In running text the name is always **Cerebro** (capital C). The wordmark's lowercase is a drawing,
not a spelling rule.

## Lockups

| File (`svg/`) | Geometry |
|---|---|
| `lockup-horizontal[-on-dark].svg` | Mark 24, gap 8 (a third of the mark), wordmark x-height centred on the mark, ascender on its top edge. The default. |
| `lockup-stacked[-on-dark].svg` | Mark 32 centred over the wordmark, gap 10. Square spaces: splash, installer sidebar, avatars. |
| `mark*.svg` | Mark only: favicon, app icon, tight spaces, anywhere the name is already on screen. |

The lockups are composed by `scripts/build-brand.mjs` from the mark and wordmark masters, so the
three can never drift apart. Edit the masters, then run `npm run brand`.

**Clear space.** One expert (a quarter of the mark's height) on every side. Nothing enters it:
no text, no edge of the screen, no other logo.

**Minimum size.** Mark 16 px. Horizontal lockup 20 px tall (the wordmark stroke stays at 2 px).
Stacked lockup 72 px wide. Print: mark 5 mm, horizontal lockup 6 mm tall.

On the site the header and footer use the horizontal lockup inline (`src/components/BrandLockup.astro`),
so it re-themes with the page. Hovering it quiets the idle experts and swells the router: the route
is what you notice.

## Colour

The accent is **blue on cool slate**. Never amber, orange or any warm accent.

| Role | Light theme | Dark theme | Token |
|---|---|---|---|
| Ink (idle experts, wordmark) | `#101622` | `#e5ebf8` | `--color-text-primary` |
| Blue (route, router, picked expert) | `#0069e8`, hue 258 | `#38b8f6`, hue 235 | `--color-accent` |
| Background | `#f8faff` | `#090d17` | `--color-bg` |
| App tile | | `#141c2e` to `#090d17` | (icons only) |

The hex values are sRGB renderings of the OKLCH tokens in `design/tokens.css`, which stay the source
of truth. On light the blue is 4.8:1 against the background and the ink 17:1; on dark the blue is
8.6:1 and the ink 16:1 (`tests/brand.spec.ts` measures the rendered mark in both themes).

**60-30-10.** About 60% slate background and surfaces, 30% ink and secondary text, at most 10% blue.
Blue marks the one thing that matters on a screen: the primary action, the live value, the route.
If two things on a screen are blue, one of them is wrong.

## Motif

`svg/motif-cells.svg` + `svg/motif-routes.svg`: a tile of idle experts on a grid with a bus and four
routes that each pick one expert (4 of 32). Seamless at 192 x 96. Used as two CSS masks
(`src/styles/motif.css`, `src/components/RouteMotif.astro`) so each layer takes a theme token:
experts at the border colour, routes at the accent at 55%.

Sparingly: the footer band and the 404 page on the site, the installer sidebar and the social card.
In the app it belongs behind empty states and the first-run screen, never behind data. Never animate
it continuously; routes may light once when they scroll into view.

## Generated files

`npm run brand` (`scripts/build-brand.mjs`) writes, from the masters:

- `out/windows/cerebro.ico`: 16, 20, 24, 32, 40, 48, 64, 96, 128, 256 (PNG entry at 256, 32-bit DIB
  below) plus one PNG per size.
- `out/png/cerebro-512.png`, `cerebro-1024.png`.
- `out/macos/cerebro.iconset/`: Apple's 10 sizes on the 824/1024 tile grid with a transparent margin.
  `iconutil -c icns out/macos/cerebro.iconset` on a Mac makes the `.icns`.
- `out/linux/hicolor/<N>x<N>/apps/cerebro.png` (16 to 512) and `scalable/apps/cerebro.svg`.
- `out/installer/`: NSIS header (150 x 57) and sidebar (164 x 314) as 24-bit BMP, plus PNG at 1x and 2x.
- `out/social-card.png`: 1200 x 630, type set in the site's own Fraunces and Inter.
- `public/`: `favicon.svg`, `favicon.ico` (16/32/48), `apple-touch-icon.png`, `icon-192.png`,
  `icon-512.png`, `icon-maskable-512.png`, `social-card.png`, `site.webmanifest`.

`node brand/scripts/shoot-brand.mjs` screenshots `preview.html` (a contact sheet of all of the above)
headless into `review/shots/brand/`.

## Voice

Confident, precise, calm. Written for people who know AI, readable by people who do not.

1. **Benefit first.** Say what the user gets, then how. "17.9 tok/s instead of 3.1 on this 8 GB laptop GPU", with its configuration
   disclosed, before "searches offload and speculation settings".
2. **Every number is checkable.** Measured with the full configuration one click away, or labelled
   illustrative. `docs/CLAIMS.md` in the engine repo is binding.
3. **Describe Cerebro by what it does, never by another product.** Other products appear only on the
   comparison page and in interoperability facts ("reads the models you already downloaded with
   LM Studio or Ollama").
4. **Plain words for every term on first use.** "Mixture-of-experts (a model that runs only a few of
   its sub-networks per token)".
5. **No hype.** No exclamation marks, no "revolutionary", "magic", "supercharge", "unleash", no
   self-congratulation, no hedging either.
6. **Short sentences, active voice, present tense.** Cerebro measures. Cerebro refuses.

| Do | Don't |
|---|---|
| Cerebro measures the fastest safe configuration for your machine, then runs it. | Cerebro is llama.cpp with good settings. |
| Draft acceptance: 71% on this prompt (illustrative). Speculation stays on. | Blazing-fast AI, supercharged by speculation! |
| Kills the whole model process tree the moment Cerebro stops. | Fully sandboxed: the model can't touch your files. (false: see CLAIMS.md) |
| Ships Stable, Beta and Experimental channels for every optimisation. | Takes the best ideas from other harnesses. |
| Reads the models you already downloaded with LM Studio or Ollama. | An LM Studio-style model browser. |

## Naming

- **Cerebro** in prose. **Cerebro Engine** only in formal contexts (licence, installer title,
  package metadata). Never "CEREBRO", "Cerebro AI", "CerebroGPT" or "the Cerebro".
- The command is `cerebro`, always lowercase and in code: `cerebro models`, `cerebro tune <model>`,
  `cerebro serve <model>`, `cerebro chat`.
- Features are named for what they do, in sentence case: Tune, Serve, Chat, Coding agent,
  Document search, OpenAI-compatible API, Live metrics. No invented sub-brands.
- Release channels: **Stable**, **Beta**, **Experimental** (capitalised, in that order).
- **Strata** is the turbo backend's name; it is the only other proper noun we coin.
- Experience levels: **Beginner** and **Advanced**.

## Logo do and don't

| Do | Don't |
|---|---|
| Use the files in `svg/` as they are. | Redraw, rotate, skew, outline or add effects (glow, gradient, shadow) to the mark. |
| Keep the route the only blue part. | Colour more experts blue, or the whole mark blue on a blue background. |
| Use `mark-on-dark.svg` on dark, `mark.svg` on light, mono where one ink is all you have. | Put the light-theme mark on a dark background (the ink disappears). |
| Keep one expert of clear space. | Crowd it with text, or lock it up with another logo without a divider. |
| Use `mark-small.svg` inside a 16 px tile. | Scale the master below 16 px. |
| Write the name as Cerebro in text. | Type the wordmark in a font to fake it. |

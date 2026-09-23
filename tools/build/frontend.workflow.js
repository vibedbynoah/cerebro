export const meta = {
  name: 'cerebro-frontend',
  description: 'Cerebro front ends: shared design system, then the marketing website and the desktop app in parallel. Sonnet 5 builders, Opus 5.5 mentor-critics to 10/10 (visual + functional + security), free local Gemma first passes.',
  phases: [
    { title: 'Design system', detail: 'shared tokens, type, motion, brand' },
    { title: 'Website', detail: 'Astro marketing site, page by page' },
    { title: 'Desktop app', detail: 'Tauri v2 + SolidJS app front end, screen by screen' },
  ],
}

// args: { skip: [task ids already committed] }
const SKIP = (args && args.skip) || []
const BUILDER = 'claude-sonnet-5'
const CRITIC = 'claude-opus-5-5'
const MAX_ROUNDS = 8
const MAIN = 'C:/Users/noahm/Projects/cerebro'
const SITE = 'C:/Users/noahm/Projects/cerebro-site'
const APP = 'C:/Users/noahm/Projects/cerebro-app'
const TOOLS = 'C:/Users/noahm/Projects/cerebro/tools/local'

const PRODUCT = `PRODUCT: Cerebro Engine, an open-source (Apache-2.0), Windows-first, fully offline local LLM harness for small businesses and AI enthusiasts. Its thesis: it MEASURES the fastest safe llama.cpp configuration for your exact machine instead of guessing (measured example, provisional: a 30B MoE went from 3.1 tok/s on llama.cpp defaults to 17.9 tok/s tuned on an 8 GB laptop GPU, while a small model that fits in VRAM was fastest on defaults - so Cerebro measures, never assumes), runs it sandboxed and integrity-checked, and serves it only on loopback. It also offers chat, a coding agent, local document search (RAG) and an OpenAI-compatible API. Tone: confident, precise, calm, never hype; every claim must be checkable (docs/CLAIMS.md in ${MAIN} is binding - read it). Where data or diagrams are needed, use clearly marked PLACEHOLDER values ("illustrative data") - never present invented numbers as measured.
CONTENT RULES: explain every technical term on first use in plain words (inline definitions or accessible tooltips) because many visitors are not engineers; code blocks show real Cerebro commands (cerebro models, cerebro tune <model>, cerebro serve <model>, cerebro chat), never lorem-ipsum code; no testimonials or quotes until real users exist; clear heading tiers and a consistent spacing rhythm.
OWNER'S BAR: professional, beautiful, clean, bug-free and genuinely inspiring, in the league of anthropic.com, linear.app, raycast.com, vercel.com and stripe.com. Every button has its own tasteful micro-interaction. Every scroll reveals something: a diagram that assembles, a number that counts, a comparison that morphs. Motion is purposeful, smooth (60 fps, transform/opacity only) and fully respects prefers-reduced-motion. Security is "sealed, no cracks": this will be attacked by an uncensored model after it is built.`

const RULES = `MACHINE RULES: never open a visible browser or app window (the owner's OLED screen must not show static content): all browser work is headless Playwright. Kill every process you start (dev servers, browsers) before finishing. Do not load any LLM except through the local review tools below. Do not run git commit, reset, checkout, stash or rebase (a separate step commits). Stay inside your repository directory.
LOCAL FIRST-PASS TOOLS (free; use them before you finish every round, verify every point they raise because a small model can be wrong):
- Visual: python ${TOOLS}/local_review.py --images <2-3 of your screenshots> --refs <1-2 reference screenshots> --focus "<page/screen and state>"   (keep total images <= 4 per call)
- Code:   python ${TOOLS}/local_review.py --repo <your repository path> --focus "<component>"
- Boilerplate drafting (fixtures, repetitive data, placeholder content tables only; never security-relevant code): python ${TOOLS}/qwen_task.py --task "..." --context <file> --out <file>, then read, fix and test every line.`

const SECURITY = `SECURITY REQUIREMENTS (non-negotiable, the critic checks each):
- No third-party network requests at runtime: fonts, scripts, images and styles are all self-hosted and bundled. A test must fail if any request leaves the origin.
- No inline scripts or inline event handlers; a strict Content-Security-Policy (default-src 'self'; script-src 'self'; style-src 'self' (hashes only if unavoidable); img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none' unless a form truly exists) delivered via headers config and a meta fallback where the platform needs it.
- Security headers configured for hosting (HSTS with preload, X-Content-Type-Options nosniff, Referrer-Policy strict-origin-when-cross-origin, Permissions-Policy denying everything unused, Cross-Origin-Opener-Policy same-origin, Cross-Origin-Resource-Policy same-origin).
- No user-controlled HTML reaches the DOM unsanitized; never use innerHTML/set:html/eval/new Function with anything not compile-time constant.
- Dependencies pinned (exact versions, lockfile committed); npm audit has no high or critical findings; no unmaintained or typosquat-risk packages.`

const DESIGN_TASK = {
  id: 'design-system', repo: MAIN, dir: 'design',
  brief: `Create the shared Cerebro design system in ${MAIN}/design, used by BOTH the Astro website and the Tauri/SolidJS desktop app (they will import these files).
DELIVERABLES
- design/tokens.css: CSS custom properties. Colour in oklch() (WebView2 and modern browsers render it natively), dark theme as the default and a light theme under [data-theme="light"] and prefers-color-scheme; semantic tokens (bg, surface levels, border, text primary/secondary/muted, accent, accent-contrast, success, warning, danger, focus ring), all text/background pairs meeting WCAG 2.2 AA (4.5:1 body, 3:1 large text and UI) in both themes; 4 px spacing scale; radii 4/6/8/12/16/pill; elevation shadows tuned for dark and light; type scale (fluid clamp() sizes for display/headline/title/body/small/code); motion tokens (durations 120/200/320/600 ms, easing curves for standard, emphasized, spring-like overshoot); z-index scale.
- design/fonts/: self-hosted variable fonts under the SIL Open Font License, obtained from npm @fontsource-variable packages (for example Inter for UI, a characterful display face such as Fraunces or Instrument Serif for headlines, JetBrains Mono for code), with the licence files, plus design/fonts.css with @font-face (font-display: swap, unicode-range where sensible, woff2 only).
- design/motion.css and design/motion.ts: shared motion primitives: scroll reveal (IntersectionObserver, staggered children), count-up numbers, draw-on SVG strokes, magnetic hover, press spring, light-sweep on hover, all disabled or reduced under prefers-reduced-motion, all transform/opacity only. Framework-agnostic TypeScript, no dependencies.
- design/DESIGN.md: the brand in one page: personality, colour roles, type pairing, spacing rules, motion principles, button interaction catalogue (which micro-interaction goes with which button role), accessibility rules, do/don't examples.
- design/preview.html + design/preview.ts: a single self-contained specimen page showing every token, type style, button interaction and motion primitive in both themes (served by a tiny static server for screenshots; no third-party requests).
- design/package.json with scripts: "check" (tsc --noEmit strict), "test" (Playwright: preview loads with zero console errors and zero off-origin requests, WCAG contrast of every documented pair computed from the actual rendered colours in both themes passes AA, axe finds no serious/critical issue, reduced-motion disables animation), "shoot" (headless screenshots of the preview in dark and light at 1440x900 and 390x844 into design/review/shots/).
- Also capture reference screenshots once: design/review/refs/ with full-viewport 1440x900 headless shots of https://www.anthropic.com, https://linear.app, https://raycast.com, https://vercel.com and https://stripe.com (script "refs"). These are for comparison only and are gitignored.
Aim for a distinctive, premium identity: not generic "AI purple gradient". Calm, precise, engineered, warm.`,
}

const SITE_TASKS = [
  { id: 'site-foundation', dir: 'site', brief: `Create the marketing website in ${SITE}/site with Astro (latest stable, TypeScript strict), importing the shared design system from ${SITE}/design (merge main first is already done for you). Build: the base layout (skip link, header with nav to every page, theme toggle with no flash of wrong theme, mobile menu, footer with sitemap and licence), Astro View Transitions between pages, the button component set implementing the DESIGN.md interaction catalogue (primary with light-sweep and press spring, secondary magnetic, ghost with underline draw, copy-button with animated confirmation, download button with progress-style flourish), a scroll-reveal integration using design/motion.ts, and stub pages for every route so navigation works: / , /product, /performance, /security, /compare, /download, /docs, /about, /changelog, and a custom 404.
TOOLING in site/package.json: "dev", "build", "preview", "check" (astro check + tsc), "test" (Playwright against the built site: every route returns 200 and renders with zero console errors; zero requests leave the origin; no horizontal overflow at 390, 768 and 1440 widths; every internal link resolves; axe finds no serious/critical issue; keyboard navigation reaches every interactive element with a visible focus ring; theme toggle persists and never flashes; reduced motion disables animation; the CSP header/meta is present and no inline script exists in the built HTML), "shoot" (headless full-page screenshots of every route, dark and light, 1440 and 390 wide, into site/review/shots/, gitignored), and "lighthouse" (headless Lighthouse on the built site, mobile preset; performance, accessibility, best-practices and SEO must each be >= 95). Put hosting security headers in site/public/_headers (and document them).
${SECURITY}` },
  { id: 'site-home', dir: 'site', brief: `Build the home page (${SITE}/site, route /) as the flagship: a story that makes a visitor feel why Cerebro is the best local LLM harness. Required sections, each with its own scroll surprise:
1. Hero: a crisp headline and subline, two CTAs (Download, See how it works), and an animated "engine" diagram (inline SVG) where a model file flows through Probe -> Measure -> Tune -> Seal -> Serve, drawing itself on load, with a subtle live-looking tokens/sec ticker (illustrative).
2. "Guessing vs measuring": a scroll-scrubbed comparison where a default-configuration bar grows slowly and a tuned bar races past it (placeholder numbers labelled illustrative), with a toggle between "model fits in VRAM" and "model does not fit" showing that defaults sometimes win - Cerebro measures either way.
3. "What happens when you press Tune": a pinned section whose steps advance as you scroll (probe hardware, benchmark candidate configurations, soak test, quality gate, lock in), each with a small animated illustration.
4. "Sealed by design": security layers (integrity check, sandboxed process, loopback only, no telemetry, signed updates) that assemble as concentric rings on scroll, each ring labelled.
5. Features grid (an Effort dial from Low - fast, answers from what it knows - to Max - verifies every line before answering - with the real time cost shown; built-in image generation - text-to-image, editing, upscaling - with local FLUX, SDXL and Qwen-Image models; vision chat; voice in and out; one-click model downloads from Hugging Face, plugin and MCP marketplace including image generation, tools with Claude Code-style permission prompts, chat, coding agent, local document search, OpenAI-compatible API, benchmarks, works fully offline without Wi-Fi, everything customizable) with hover-reveal detail cards and a clean custom icon for each.
6. DATA AND PERFORMANCE (the owner's top priority: after these sections a visitor must be convinced Cerebro is the fastest, smartest local harness). Build all of the following, each with its own scroll animation, every number a count-up, every chart inline SVG with accessible labels and hover tooltips, and each block carrying a small "illustrative data - replaced by published benchmarks" badge:
   a. "The numbers": a full-width stat band of 4 huge count-up metrics (decode speed gain vs defaults, time-to-first-token, VRAM headroom recovered, setup time from install to tuned) with one-line explanations.
   b. Harness leaderboard: same model, same machine, sustained tokens/sec for Cerebro and 6 other harnesses as horizontal bars that race to their values when scrolled into view, Cerebro on top and highlighted, with a model selector (3 placeholder models) that re-animates the race.
   c. Gains per model: small-multiples grid (6 models) each showing llama.cpp defaults vs Cerebro tuned with the percentage gain called out; one tile deliberately shows "defaults already optimal - Cerebro keeps them" to prove honesty.
   d. Speed that holds at long context: a line chart of decode tokens/sec from 0 to 32k context, Cerebro's line staying high while others sag, drawing itself as you scroll.
   e. Faster AND smarter: a scatter of coding score vs tokens/sec where Cerebro-tuned configurations sit in the top-right "best of both" quadrant, with the quadrant shading fading in.
   f. Efficiency: tokens per watt and per GB of VRAM as animated gauges.
   g. Reproducibility proof: a terminal-style card that types out the exact command and provenance line behind a benchmark (machine, model, quant, build, depth, sampler), with a copy button, and the line "every number on this site can be reproduced on your machine".
7. Comparison teaser linking to /compare.
8. Open source and offline promise, then a closing CTA with the download button interaction.
Every number counts up when it enters view; every diagram is inline SVG styled from tokens; everything responsive down to 390 px and perfect in both themes. Update the Playwright tests and screenshots for this page.
${SECURITY}` },
  { id: 'site-performance', dir: 'site', brief: `Build /performance in ${SITE}/site as the benchmark hub that makes Cerebro's lead evident. Top: an interactive results explorer (pick machine profile, model and workload; charts re-animate) showing Cerebro vs llama.cpp defaults vs other harnesses on prompt speed, decode speed, decode at 8k/16k/32k context, time to first token, VRAM use and quality score, all placeholder data with the illustrative badge; a "biggest wins" strip of animated gain cards; a per-model table with sortable columns and sparklines; and a "where defaults win" honesty section. Then: how Cerebro measures (methodology: sustained soaks not bursts, first repetition discarded, prompt/decode/decode-at-depth triple, provenance of every number, quality gate), with animated placeholder charts (inline SVG line and bar charts that draw on scroll, hoverable data points with accessible tooltips, a legend, and an "illustrative data" badge), a speculation explainer diagram (draft tokens proposed, target verifies, accepted/rejected animation), and a callout quoting the claims policy principle that every speedup names machine, model, quant, build, depth and sampler. Tests and screenshots updated. ${SECURITY}` },
  { id: 'site-security', dir: 'site', brief: `Build /security in ${SITE}/site: the threat model in plain language (hostile model files, hostile web pages reaching localhost, hostile local processes, tampered updates), each threat as a card that flips or expands to show the specific defence; an animated architecture diagram showing the trust boundary (Cerebro app, sandboxed llama.cpp child in a Job Object, loopback-only API with a key, integrity manifest, DPAPI-protected secrets, signed updates); a "what we do not do" list (no telemetry, no cloud, no auto-execute); and a responsible disclosure section. Tests and screenshots updated. ${SECURITY}` },
  { id: 'site-product', dir: 'site', brief: `Build /product in ${SITE}/site: the product tour. Interactive tabs (keyboard accessible, Ark-UI-quality semantics) for Model Hub (one-click Hugging Face downloads), Tune, Chat, Coding agent, Plugins and MCP (including image generation), Tools and permissions, Documents (RAG), API, and Offline mode; each tab shows a mock app screenshot built in HTML/CSS (not an image) with small motion (a streaming chat reply, a tuning progress chart, a code diff appearing), plus 3 benefit bullets. Add a "works with your tools" strip (OpenAI-compatible clients) as neutral text badges, not third-party logos. Tests and screenshots updated. ${SECURITY}` },
  { id: 'site-features', dir: 'site', brief: `Build /features in ${SITE}/site: every Cerebro feature, all 627 of them, because even small features matter to the owner. Generate the data at build time from ${SITE}/docs/FEATURE_CATALOG.md (or the JSON sources under docs/catalog and the archive list the generator uses; copy a build-time JSON snapshot into site/src/data with a script so the site stays self-contained). Page: a hero with an animated count-up of the total, instant client-side search and filters (section, core vs experimental, hardware-gated, new), a grouped grid of compact feature cards with a custom icon per section and a one-line description, "new" badges on recently added features, keyboard navigation, and virtualized rendering so it stays at 60 fps on mobile. Link it from the header and the home page features grid ("see all 627 features"). Tests and screenshots updated. ${SECURITY}` },
  { id: 'site-compare', dir: 'site', brief: `Build /compare in ${SITE}/site: a comparison matrix of Cerebro against LM Studio, Ollama, Jan, KoboldCpp, Open WebUI, AnythingLLM, llama.cpp server, Cline, opencode and goose across rows (measured auto-tuning, speculative decoding tuning, sandboxed backend, integrity verification, loopback-only by default, offline, open source, coding agent, local RAG, OpenAI-compatible API). Every cell value is a PLACEHOLDER marked "to be verified in the published benchmark" (the page must say so prominently; do not assert competitor facts). Filters by category, sticky header row and column, responsive card view on mobile, and an animated placeholder radar/bar chart. Tests and screenshots updated. ${SECURITY}` },
  { id: 'site-download', dir: 'site', brief: `Build /download in ${SITE}/site: OS-detected primary download (Windows first; macOS and Linux shown as "coming later"), formats (MSI installer, portable zip, winget and scoop commands) each with the copy-button interaction, placeholder SHA-256 checksums and a signature-verification how-to (Ed25519/minisign style, placeholder key), system requirements, and a quick start (3 steps with animated illustrations). No real binaries are linked yet: links point to a clearly labelled placeholder. Tests and screenshots updated. ${SECURITY}` },
  { id: 'site-docs-about', dir: 'site', brief: `Build /docs (docs landing with quickstart, concepts, CLI reference placeholder, API reference placeholder, a client-side search over the docs pages that works fully offline with no third-party service), /about (manifesto: why local, why measure, why open source; team placeholder), /changelog (timeline with placeholder entries, animated on scroll), the legal pages /terms (Terms of Use: software provided "as is" under Apache-2.0 with no warranty and limitation of liability; users are responsible for the models, plugins and content they download, run or generate, including generated images; third-party model licences such as Llama and Gemma terms apply and Cerebro never bypasses gated access; acceptable use; comparisons and benchmark figures are illustrative until published with full methodology; governing terms placeholder), /privacy (the site and app collect no personal data, no analytics, no cookies beyond a theme preference in local storage, nothing leaves the machine unless the user downloads a model or plugin; the optional update check is off by default and what it reveals is stated plainly), and /licenses (Apache-2.0 text plus third-party notices, including every font's SIL Open Font License), all linked from the footer, written in plain, friendly language and marked at the top as a template to be reviewed by a qualified lawyer (and a parent or guardian, as the owner is a minor) before public launch. Polish the 404 page into something delightful. Tests and screenshots updated. ${SECURITY}` },
  { id: 'site-audit', dir: 'site', brief: `Whole-site quality and security pass on ${SITE}/site. Fix everything that stops the site from being flawless: run every test, Lighthouse (all four >= 95 mobile and desktop on every route), axe on every route in both themes, link check, visual consistency across pages (spacing rhythm, type scale, button behaviour identical everywhere), performance budget (total JS per page under 60 KB gzipped, no layout shift above 0.02, LCP under 1.8 s on the Lighthouse mobile preset), and a security review: CSP on every route with no violations reported in the console, no inline script anywhere, no off-origin request anywhere, headers file complete, dependencies audited. Add tests for any gap you find. ${SECURITY}` },
]

const APP_SECURITY = `${SECURITY}
USER CONTROL (owner rule, not up for debate): safe by default, but the user can edit and switch off ANYTHING, including the sandbox, integrity checks and permission prompts, from Settings. Show one short, neutral note on what a switch does; never block, nag or refuse. Everything works fully offline (no Wi-Fi); network features (model downloads, plugin installs) are clearly optional and degrade gracefully offline.
DESKTOP-SPECIFIC: Tauri v2 with the isolation pattern on, deny-by-default capabilities granting only the exact commands each window needs, removeUnusedCommands, no tauri-plugin-shell and no tauri-plugin-fs exposed to the webview, a strict CSP in tauri.conf.json (no 'unsafe-inline', no 'unsafe-eval', connect-src limited to ipc: and http://ipc.localhost plus the loopback API), devtools disabled in release, navigation to external origins blocked, and the app refusing to load remote content. Cryptography uses only vetted libraries (no hand-rolled crypto): 256-bit keys from the OS CSPRNG, constant-time comparison for secrets (subtle crate), Ed25519 for update signatures. Any IPC command validates every argument on the Rust side.`

const APP_TASKS = [
  { id: 'app-foundation', dir: 'apps/desktop', brief: `Create the desktop app front end in ${APP}/apps/desktop: a Tauri v2 shell (Rust, apps/desktop/src-tauri, added to the root Cargo workspace) and a SolidJS 1.9 + Vite + TypeScript-strict UI (apps/desktop/ui) using @ark-ui/solid for headless primitives, importing the shared design system from ${APP}/design (fonts, tokens, motion). Build the app frame: custom title bar with window controls, collapsible sidebar navigation with a clean, consistent custom SVG icon set (Home, Model Hub, Models, Tune, Chat, Image Studio, Documents, Plugins, Tools, Security, Settings), a status bar HUD (model, tokens/sec, VRAM gauge, backend status - all from a mock data layer), a command palette (Ctrl+K) with fuzzy search, theme switching, toast notifications, and routing between placeholder screens. ALL data comes from a typed mock service (apps/desktop/ui/src/mock) behind an interface the real IPC will implement later; the UI must also run in a plain browser (web mode) with the mock, for headless testing.
TOOLING: "dev", "build" (vite build), "check" (tsc), "test" (Vitest unit tests for the mock service and components + Playwright against the web-mode build: every screen renders with zero console errors, zero off-origin requests, keyboard navigation and focus rings, command palette opens and navigates, axe clean, reduced motion honoured, no overflow at 1280x800 and 1920x1080), "shoot" (headless screenshots of every screen, dark and light, 1280x800 into apps/desktop/review/shots/, gitignored), and "tauri:check" (cargo check of the shell). Target: gzipped JS <= 150 KB.
${APP_SECURITY}` },
  { id: 'app-onboarding', dir: 'apps/desktop', brief: `Build the first-run onboarding in ${APP}/apps/desktop/ui: welcome with the engine animation, hardware detection step (mock: CPU, GPU, VRAM, RAM appearing one by one with a scanning animation), model recommendation (cards ranked for this machine with a "why" explanation), privacy and security promises (offline, sandboxed, loopback), a one-screen plain-language terms step (as-is software, the user is responsible for models, plugins and generated content, third-party model licences apply; links to the full Terms, Privacy and Licenses pages bundled in the app so they work offline) with a single Accept, and a finish step. Keyboard-driven, skippable, remembers completion. Tests and screenshots updated. ${APP_SECURITY}` },
  { id: 'app-models', dir: 'apps/desktop', brief: `Build the Models screen with the load experience: DSH's one-click cleanliness plus LM Studio's load panel for TEXT models - one Load button that uses Cerebro's measured tuned settings by default, and an expandable load panel where the user can set every load parameter (context length, GPU layers or offload plan, MoE experts on CPU, KV cache type, flash attention, threads and CPU cores, batch sizes, memory mapping, speculative decoding) with a live VRAM/RAM fit estimate bar, named presets the user can save, and "reset to tuned". VISION and AUDIO models get a better path than LM Studio: a vision model automatically pairs with its projector (mmproj) file and shows a "try it" drop zone; speech-to-text and text-to-speech models load as separate lightweight engines with a record-and-transcribe and a speak-this-sentence test button; each model shows modality badges (text, vision, audio in, audio out, image generation). Add a Tweak Studio tab per model: small, reversible, user-owned tweaks without re-downloading (enable a quantized KV cache for a model that lacks it by default, override context length and RoPE as the authors specify, fix or replace the chat template with a diff view, stop tokens, default sampler), saved as named shareable patch profiles keyed by model hash, plus a validated GGUF metadata viewer and editor that writes to a copy with a diff and one-click revert, and a "known fixes for this model" list from the quirks database (each fix switchable). Also a disk-space panel showing duplicate model files across LM Studio, Ollama and other folders with a safe deduplicate action. Then a library of local models (mock list with name, size, quantization, architecture, fits-in-VRAM indicator, last tuned result), search and filters, a detail drawer with metadata and actions (Tune, Chat, Reveal in folder - mocked), empty state, and loading skeletons. Virtualized list for 500+ items without jank. Tests and screenshots updated. ${APP_SECURITY}` },
  { id: 'app-tune', dir: 'apps/desktop', brief: `Build the Tune screen: pick a model and workload profile (chat, coding, long documents), start a tune and watch live progress from a mock event stream: candidate configurations appearing, a live tokens/sec chart per candidate, soak-test progress with a VRAM trend line, quality-gate result, and the winner locked in with an explanation of why it won versus llama.cpp's own automatic settings (placeholder data). Cancelable, resumable UI state. Tests and screenshots updated. ${APP_SECURITY}` },
  { id: 'app-chat', dir: 'apps/desktop', brief: `Build the Chat screen: conversation list, persistent memory across chats (a Memory panel showing what Cerebro remembers about the user and projects, each memory editable and deletable, a per-chat "don't remember this" switch, and a global off switch), image and file attachments for vision models (drag and drop, paste, preview), voice input (push-to-talk mic button with live waveform, mock transcription) and read-aloud playback of replies (mock TTS), inline image generation from chat ("/imagine" command rendering a result card), a Verbosity control next to it (Brief - the default: answer first in as few words as needed, details only on request; Normal; Detailed) that adds a matching instruction to the system prompt and a "tell me more" button under brief replies, an Effort dial in the composer (Low, Medium, High, XHigh, Max) implemented exactly per ${APP}/docs/specs/effort-levels.md (each level sets thinking budget, the verification instruction, self-check passes; shows the expected slowdown; under each reply show the level used, thinking tokens and extra time, with a collapsible "thinking" and "self-check" view), streaming assistant replies from a mock stream, markdown with code highlighting rendered through a sanitizer that the tests prove blocks script injection (javascript: URLs, event handler attributes, raw HTML), copy buttons on code blocks, stop/regenerate, model and sampler controls, a live tokens/sec + time-to-first-token readout, and a composer with keyboard shortcuts. Smooth at 60 fps while streaming long replies. Tests (including an XSS corpus) and screenshots updated. ${APP_SECURITY}` },
  { id: 'app-security-center', dir: 'apps/desktop', brief: `Build the Security screen: backend integrity status (manifest verified, per-file hashes, mock), folder permission audit result with the principals that can write, sandbox status (Job Object, environment allowlist), API key management (show once, rotate, copy with confirmation; keys never stored in the webview or logs), update signature status, and an activity log. Every sensitive value is masked by default and revealed only on explicit action. Tests and screenshots updated. ${APP_SECURITY}` },
  { id: 'app-settings-docs', dir: 'apps/desktop', brief: `Build the Settings screen (appearance, models folder, backend, performance profile, privacy, network and offline mode, plugins, permissions, and an Advanced section where the user can switch off the sandbox, integrity verification, folder-permission audit and permission prompts, each with one short neutral note and no nagging) with validated forms, instant feedback, search across all settings, and export/import of the whole configuration as an editable file, plus a Legal section (Terms, Privacy, open-source licences and third-party notices, all bundled offline), a Responses section (default verbosity: Brief; editable instructions), default load parameters and saved load presets, and an Effort section where the default level is chosen and every level's instruction, thinking budget and passes can be edited or new levels added (per docs/specs/effort-levels.md), and the Documents screen (local knowledge base: drop zone, indexed documents list, mock indexing progress, search box with results preview). Tests and screenshots updated. ${APP_SECURITY}` },
  { id: 'app-model-hub', dir: 'apps/desktop', brief: `Build the Model Hub screen in ${APP}/apps/desktop/ui: a one-click model catalog like LM Studio's, so users never browse Hugging Face themselves. Covers EVERY model type with tabs: text/chat, coding, vision (image understanding), image generation (FLUX, SDXL, SD 3.5, Qwen-Image), speech-to-text (Whisper family), text-to-speech (Kokoro, Piper and similar) and embeddings. Curated families (Qwen, Google Gemma, Meta Llama, Mistral, DeepSeek, Microsoft Phi, Black Forest Labs, Stability and more) with clean logos-free brand tiles, search and filters (size, task, licence, fits-this-machine), each model card showing quantization options with a hardware-fit badge (fits in VRAM / runs partly on CPU / too large), licence, and a single Download button that turns into an animated progress ring with pause, resume, cancel and a verified-checksum finish state, then "Tune" and "Chat" actions. Before any download the model's licence is shown (name, key restrictions such as commercial-use limits, link to the full text); gated models (for example Llama and Gemma) explain that access must be accepted on Hugging Face with the user's own account and token, and Cerebro never bypasses gating. Downloads queue with bandwidth and ETA; offline the catalog shows cached entries and download buttons explain they need a connection. All data from the mock layer (a typed catalog interface the real Hugging Face client will implement later). Tests and screenshots updated. ${APP_SECURITY}` },
  { id: 'app-plugins', dir: 'apps/desktop', brief: `Build the Plugins screen in ${APP}/apps/desktop/ui: a plugin and MCP marketplace as easy as DSH's, with a clean, consistent icon grid (a custom icon set in one visual style, SVG, no third-party logos), categories (image generation, web search, files, git, email, calendar, databases, browser automation, voice), one-click Install that animates into Enabled, per-plugin detail pages (what it can do, the permissions it asks for, version, author, licence, offline or needs-network, and a one-line note that third-party plugins are the author's responsibility), an Installed tab with enable/disable toggles and updates, install from a local file for fully offline use, and a "featured this week" carousel. Also build Image Studio as a BUILT-IN first-class screen (not dependent on any plugin; image generation is native to Cerebro, unlike LM Studio and DSH): text-to-image, image-to-image, inpainting with a brush mask, upscaling and variations; model picker for local image models (FLUX, SDXL, SD 3.5, Qwen-Image and others, including GGUF image models); prompt and negative prompt, aspect-ratio chips, seed, steps, guidance and sampler controls with a simple/advanced toggle for beginners; a live denoising progress preview; a gallery with favourites, prompt reuse and export. Generated images are mock placeholders rendered as tasteful generative SVG/canvas art. Mock data only. Tests and screenshots updated. ${APP_SECURITY}` },
  { id: 'app-tools-permissions', dir: 'apps/desktop', brief: `Build the tool-permission system UI in ${APP}/apps/desktop/ui, modelled on Claude Code and Codex: permission modes (Ask every time, Allow for this session, Always allow, Deny) settable globally and per tool; an approval dialog that previews exactly what a tool will do (command line, file diff, URL, plugin action) with Allow once / Allow for session / Always / Deny and keyboard shortcuts; a Tools screen listing every built-in tool and plugin tool with its current rule; and a searchable activity log of every tool call and decision. The user can edit or remove any rule, including turning prompts off entirely. Mock tool calls drive the dialog in tests. Tests and screenshots updated. ${APP_SECURITY}` },
  { id: 'app-home', dir: 'apps/desktop', brief: `Build the Home screen: the dashboard a user lands on: current model and its measured speed, quick actions (Chat, Tune, Add model), recent activity, system health, a "Did you know?" feature spotlight that surfaces a different capability each day, a What's New feed, and a beginner checklist ("download your first model", "run your first tune", "install a plugin") with satisfying completion animations, all with tasteful entrance motion. The goal: users keep discovering features that delight them. Then run a consistency pass across every screen (spacing, type, motion, empty/loading/error states). Tests and screenshots updated. ${APP_SECURITY}` },
  { id: 'app-audit', dir: 'apps/desktop', brief: `Whole-app quality and security pass on ${APP}/apps/desktop: run every test, verify the Tauri configuration (isolation pattern, capabilities, CSP, devtools off in release, blocked navigation, no shell/fs plugins, removeUnusedCommands) with a test that parses tauri.conf.json and the capability files and fails on any regression, XSS corpus on every rendered-content path, dependency audit (npm audit, cargo audit or cargo deny advisories), bundle budget, accessibility and visual consistency across every screen in both themes. Fix everything found and add tests for each gap. ${APP_SECURITY}` },
]

const BUILD_RESULT = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    files_changed: { type: 'array', items: { type: 'string' } },
    tests: { type: 'string' },
    screenshots: { type: 'array', items: { type: 'string' } },
    local_review: { type: 'string' },
    findings_addressed: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'files_changed', 'tests', 'screenshots', 'local_review', 'findings_addressed'],
}

const FINDING = {
  type: 'object',
  properties: {
    severity: { type: 'string', enum: ['critical', 'major', 'minor'] },
    area: { type: 'string', enum: ['visual', 'functional', 'security', 'accessibility', 'performance', 'code'] },
    where: { type: 'string' },
    problem: { type: 'string' },
    fix: { type: 'string' },
    suggested_code: { type: 'string' },
  },
  required: ['severity', 'area', 'where', 'problem', 'fix', 'suggested_code'],
}

const CRITIC_RESULT = {
  type: 'object',
  properties: {
    score: { type: 'number' },
    findings: { type: 'array', items: FINDING },
    guidance: {
      type: 'object',
      properties: {
        root_causes: { type: 'array', items: { type: 'string' } },
        plan: { type: 'array', items: { type: 'string' } },
        design_advice: { type: 'string' },
        pitfalls: { type: 'array', items: { type: 'string' } },
        tests_to_add: { type: 'array', items: { type: 'string' } },
      },
      required: ['root_causes', 'plan', 'design_advice', 'pitfalls', 'tests_to_add'],
    },
    tests: { type: 'string' },
    screenshots_reviewed: { type: 'array', items: { type: 'string' } },
    verdict: { type: 'string' },
  },
  required: ['score', 'findings', 'guidance', 'tests', 'screenshots_reviewed', 'verdict'],
}

function builderPrompt(task, repo, round, findings, guidance) {
  const fix = findings && findings.length
    ? `\nROUND ${round}. A senior Opus reviewer examined your work (code, running tests, and screenshots) and found the problems below. Work through the plan in order; adapt each suggested_code sketch so it fits. Never weaken a test, delete a requirement or suppress a check to pass. If a suggestion is wrong, do the right thing and explain with evidence in findings_addressed.\nFINDINGS\n${JSON.stringify(findings, null, 1)}\nREVIEWER GUIDANCE\n${JSON.stringify(guidance, null, 1)}\n`
    : `\nROUND 1. Build it. If earlier partial work exists on disk (git status), keep what is right and finish the rest.\n`
  return `You are the builder for one Cerebro front-end task: ${task.id}.
${PRODUCT}
REPOSITORY: ${repo}
${RULES}

TASK BRIEF
${task.brief}
${fix}
DEFINITION OF DONE FOR THIS ROUND: the build passes; every test passes (always run the full test script, never skip it); you took fresh headless screenshots and looked at them yourself (Read the PNG files) and fixed what looked wrong; you ran the local visual review against the reference screenshots and the local code review and addressed every valid point. Report exact test result lines, the screenshot paths, and the local review scores.`
}

function criticPrompt(task, repo) {
  return `You are a harsh senior design engineer, security reviewer AND mentor for Cerebro. A less experienced builder (Sonnet) made this and will revise it from your review alone, so give exact, actionable guidance. Your bar: would this ship on anthropic.com or linear.app today, flawless in function, visuals, accessibility, performance and security? Anything short of that is a finding.
${PRODUCT}
REPOSITORY: ${repo}
${RULES}

TASK UNDER REVIEW: ${task.id}
BRIEF IT MUST SATISFY
${task.brief}

PROCEDURE
1. git status / git diff; read every changed or new file.
2. Build, run the full test script, and run the screenshot script yourself. Report exact results. Kill every process you started.
3. Read the fresh screenshots (PNG files) at every width and theme, and compare them with the reference screenshots under design/review/refs if present. Judge hierarchy, spacing rhythm, typography, colour, contrast, alignment, polish, motion design (inspect the code), states (hover, focus, empty, loading, error), responsiveness, and whether it is genuinely inspiring and distinctive.
4. Check every brief requirement (missing = major), functional correctness, accessibility, performance budgets, and every security requirement (a violation is critical).
5. For EVERY finding give suggested_code (a concrete sketch, at most ~40 lines) or an empty string for a trivial fix; then write guidance: root_causes, an ordered plan, design_advice, pitfalls, tests_to_add.
6. Do NOT edit files.
Score 10 only with zero findings of any severity. Any critical caps the score at 4, any major at 7, any minor at 9.`
}

async function runTask(task, repo, phaseName) {
  let findings = null, guidance = null, passed = false, last = null
  for (let round = 1; round <= MAX_ROUNDS; round++) {
    const b = await agent(builderPrompt(task, repo, round, findings, guidance),
      { label: `build:${task.id}:r${round}`, phase: phaseName, schema: BUILD_RESULT, model: BUILDER, effort: 'high' })
    if (!b) return { id: task.id, status: 'halted', where: `builder r${round}` }
    const c = await agent(criticPrompt(task, repo),
      { label: `critic:${task.id}:r${round}`, phase: phaseName, schema: CRITIC_RESULT, model: CRITIC, effort: 'high' })
    if (!c) return { id: task.id, status: 'halted', where: `critic r${round}` }
    last = c; findings = c.findings; guidance = c.guidance
    log(`${task.id} r${round}: ${c.score}/10, ${c.findings.length} findings`)
    if (c.findings.length === 0 && c.score >= 10) { passed = true; break }
  }
  const open = last ? last.findings : []
  const commit = await agent(`Commit one finished Cerebro front-end task in ${repo}.
Task: ${task.id} (directory ${task.dir}). Review status: ${passed ? 'passed 10/10 with zero findings' : `NOT at 10/10 after ${MAX_ROUNDS} rounds; ${open.length} findings remain`}.
${open.length ? `First append these open findings to docs/build/open-findings.md under a heading "${task.id}" (create it if needed):\n${JSON.stringify(open, null, 1)}\n` : ''}
Run git status. Stage only files under ${task.dir}/ plus any root files this task legitimately changed (Cargo.toml, Cargo.lock, .gitignore, docs/build/open-findings.md). Never stage review/ screenshot folders, node_modules, dist, target or secrets. Commit with an imperative first line, a short body, and these two final lines exactly:
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Q4T16r2eHnoWzCJrCw94mq
Never use --no-verify. Return the commit hash and first line.`,
    { label: `commit:${task.id}`, phase: phaseName, model: BUILDER, effort: 'low' })
  if (!commit) return { id: task.id, status: 'halted', where: 'commit' }
  return { id: task.id, status: passed ? 'passed' : 'open-findings', open: open.length, commit }
}

async function runTrack(tasks, repo, phaseName) {
  const out = []
  for (const t of tasks) {
    if (SKIP.includes(t.id)) continue
    const r = await runTask(t, repo, phaseName)
    out.push(r)
    if (r.status === 'halted') { log(`${phaseName} halted at ${r.id} (${r.where}); resume with skip for finished tasks`); break }
  }
  return out
}

phase('Design system')
let design = { status: 'skipped' }
if (!SKIP.includes('design-system')) {
  design = await runTask(DESIGN_TASK, MAIN, 'Design system')
  if (design.status === 'halted') return { design }
}

// Bring the design system into both front-end branches before they start.
const synced = await agent(`In two git worktrees of the same repository, merge the main branch so the shared design system is present:
1. cd ${SITE} && git merge --no-edit main
2. cd ${APP} && git merge --no-edit main
If a merge conflicts, abort it (git merge --abort) and report the conflict. Return what happened for each.`,
  { label: 'sync-design-into-tracks', phase: 'Design system', model: BUILDER, effort: 'low' })
if (!synced) return { design, halted: 'sync failed' }

const [site, app] = await parallel([
  () => runTrack(SITE_TASKS, SITE, 'Website'),
  () => runTrack(APP_TASKS, APP, 'Desktop app'),
])
return { design, site, app }

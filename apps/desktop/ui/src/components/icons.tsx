import type { JSX } from "solid-js";

/**
 * Cerebro's own icon set. One grid for every glyph: 24x24 viewBox, 1.6 px stroke drawn in
 * currentColor, round caps and joins, no fills except small solid dots, ~2.5 px optical padding.
 * Decorative by default (aria-hidden); the control that holds an icon carries the accessible name.
 */
const GLYPHS = {
  /** The mark in one ink (BRAND.md mark-mono), at 20/24 of the grid so it keeps the icons' optical padding. */
  about: () => (
    <g fill="currentColor" stroke="none" transform="translate(2 2) scale(0.8333)">
      <rect width="6" height="6" rx="1.5" />
      <rect x="9" width="6" height="6" rx="1.5" />
      <rect x="18" width="6" height="6" rx="1.5" />
      <rect y="9" width="6" height="6" rx="1.5" />
      <rect y="18" width="6" height="6" rx="1.5" />
      <rect x="9" y="18" width="6" height="6" rx="1.5" />
      <rect x="18" y="18" width="6" height="6" rx="1.5" />
      <path d="M10.5 5h3v5h-3z" />
      <circle cx="12" cy="12" r="3" />
    </g>
  ),
  home: () => (
    <>
      <path d="M3.5 10.8 12 4l8.5 6.8" />
      <path d="M5.8 9.2v10.3h12.4V9.2" />
      <path d="M10 19.5v-5.2h4v5.2" />
    </>
  ),
  hub: () => (
    <>
      <path d="M12 4v10.2" />
      <path d="m7.8 10.2 4.2 4.2 4.2-4.2" />
      <path d="M4 14.5v4A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5v-4" />
    </>
  ),
  models: () => (
    <>
      <path d="M12 3.8 20.5 8 12 12.2 3.5 8Z" />
      <path d="m3.5 12 8.5 4.2 8.5-4.2" />
      <path d="m3.5 16 8.5 4.2 8.5-4.2" />
    </>
  ),
  tune: () => (
    <>
      <path d="M4.4 17.8a8.6 8.6 0 1 1 15.2 0" />
      <path d="m12 13.6 4-4" />
      <circle cx="12" cy="13.6" r="1.4" />
      <path d="M7.4 9.6 6.3 8.5M12 7.2V5.7" />
    </>
  ),
  chat: () => (
    <path d="M20 11.4c0 3.9-3.6 7-8 7-1.2 0-2.4-.2-3.4-.7L4.2 19.3l1.2-3.5A6.5 6.5 0 0 1 4 11.4c0-3.9 3.6-7 8-7s8 3.1 8 7Z" />
  ),
  images: () => (
    <>
      <rect x="3.5" y="4.5" width="17" height="15" rx="2.5" />
      <circle cx="9" cy="9.8" r="1.7" />
      <path d="m4 17.4 5-4.9 3.9 3.8 2.6-2.5 4.6 4.5" />
    </>
  ),
  documents: () => (
    <>
      <path d="M6.5 3.5h7l4.5 4.5v11a1.5 1.5 0 0 1-1.5 1.5h-10A1.5 1.5 0 0 1 5 19V5a1.5 1.5 0 0 1 1.5-1.5Z" />
      <path d="M13.5 3.5V8H18" />
      <path d="M8.5 12.5h7M8.5 16h4.5" />
    </>
  ),
  plugins: () => (
    <>
      <path d="M9 3.5v4M15 3.5v4" />
      <path d="M6.5 7.5h11V11a5.5 5.5 0 0 1-11 0Z" />
      <path d="M12 16.5v4" />
    </>
  ),
  tools: () => (
    <path d="M14.6 4.2a4.3 4.3 0 0 0-5 5.7L4 15.5V20h4.5l5.6-5.6a4.3 4.3 0 0 0 5.7-5l-2.7 2.7-2.6-.6-.6-2.6Z" />
  ),
  security: () => (
    <>
      <path d="M12 3.5 19 6v5.4c0 4.3-2.9 7.6-7 9.1-4.1-1.5-7-4.8-7-9.1V6Z" />
      <path d="m8.9 12 2.2 2.2 4-4.2" />
    </>
  ),
  settings: () => (
    <>
      <path d="M18.76 12.7 20.76 14.08 19.66 16.72 17.28 16.29 16.29 17.28 16.72 19.66 14.08 20.76 12.7 18.76 11.3 18.76 9.92 20.76 7.28 19.66 7.71 17.28 6.72 16.29 4.34 16.72 3.24 14.08 5.24 12.7 5.24 11.3 3.24 9.92 4.34 7.28 6.72 7.71 7.71 6.72 7.28 4.34 9.92 3.24 11.3 5.24 12.7 5.24 14.08 3.24 16.72 4.34 16.29 6.72 17.28 7.71 19.66 7.28 20.76 9.92 18.76 11.3Z" />
      <circle cx="12" cy="12" r="2.8" />
    </>
  ),
  // ---- Interface glyphs ----
  search: () => (
    <>
      <circle cx="10.8" cy="10.8" r="6.3" />
      <path d="m15.5 15.5 4.5 4.5" />
    </>
  ),
  sidebar: () => (
    <>
      <rect x="3.5" y="4.5" width="17" height="15" rx="2.5" />
      <path d="M9.5 4.5v15" />
    </>
  ),
  minimize: () => <path d="M5 12h14" />,
  maximize: () => <rect x="5.5" y="5.5" width="13" height="13" rx="1.5" />,
  // Windows' restore glyph: the front window, with the back one peeking out top-right.
  restore: () => (
    <>
      <rect x="5.5" y="8.5" width="10" height="10" rx="1.5" />
      <path d="M8.5 8.5v-2a1 1 0 0 1 1-1h8a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1h-2" />
    </>
  ),
  close: () => <path d="m6 6 12 12M18 6 6 18" />,
  sun: () => (
    <>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2.8v2M12 19.2v2M21.2 12h-2M4.8 12h-2M18.5 5.5l-1.4 1.4M6.9 17.1l-1.4 1.4M18.5 18.5l-1.4-1.4M6.9 6.9 5.5 5.5" />
    </>
  ),
  moon: () => <path d="M19.5 14.2A7.8 7.8 0 0 1 9.8 4.5a7.8 7.8 0 1 0 9.7 9.7Z" />,
  monitor: () => (
    <>
      <rect x="3.5" y="4.5" width="17" height="11.5" rx="2" />
      <path d="M9 20h6M12 16v4" />
    </>
  ),
  copy: () => (
    <>
      <rect x="8.5" y="8.5" width="11" height="11" rx="2" />
      <path d="M15.5 8.5V6a1.5 1.5 0 0 0-1.5-1.5H6A1.5 1.5 0 0 0 4.5 6v8A1.5 1.5 0 0 0 6 15.5h2.5" />
    </>
  ),
  check: () => <path d="m5 12.5 4.4 4.4L19 7.3" />,
  info: () => (
    <>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 11v5.5" />
      <circle cx="12" cy="7.8" r=".6" class="icon-dot" />
    </>
  ),
  alert: () => (
    <>
      <path d="M10.5 4.6a1.7 1.7 0 0 1 3 0l7 12.6a1.7 1.7 0 0 1-1.5 2.5H5a1.7 1.7 0 0 1-1.5-2.5Z" />
      <path d="M12 9.5v4.2" />
      <circle cx="12" cy="16.6" r=".6" class="icon-dot" />
    </>
  ),
  bolt: () => <path d="M13.2 3.5 5.5 13.4h6l-1 7.1 7.8-10h-6.1Z" />,
  chip: () => (
    <>
      <rect x="6.5" y="6.5" width="11" height="11" rx="2" />
      <rect x="9.5" y="9.5" width="5" height="5" rx=".8" />
      <path d="M9.5 3.5v3M14.5 3.5v3M9.5 17.5v3M14.5 17.5v3M3.5 9.5h3M3.5 14.5h3M17.5 9.5h3M17.5 14.5h3" />
    </>
  ),
  offline: () => (
    <>
      <path d="M4 4l16 16" />
      <path d="M8.6 12.9a5 5 0 0 1 3.4-1.2M5.3 9.6a9.5 9.5 0 0 1 3.1-2M13.9 7.3a9.5 9.5 0 0 1 4.8 2.3" />
      <circle cx="12" cy="17.2" r=".7" class="icon-dot" />
    </>
  ),
  online: () => (
    <>
      <path d="M5.3 9.6a9.5 9.5 0 0 1 13.4 0" />
      <path d="M8.6 12.9a5 5 0 0 1 6.8 0" />
      <circle cx="12" cy="17.2" r=".7" class="icon-dot" />
    </>
  ),
  arrowRight: () => <path d="M5 12h13.5M13.5 7l5 5-5 5" />,
  arrowLeft: () => <path d="M19 12H5.5M10.5 7l-5 5 5 5" />,
  // Disclosure: points right when closed, turned down (CSS) when open.
  chevronRight: () => <path d="m9.5 6 6 6-6 6" />,
  // Run again: an open circle with its arrowhead.
  refresh: () => <path d="M19 12a7 7 0 1 1-2.05-4.95M19 4v3.5h-3.5" />,
  // Elapsed time: a dial with its hands.
  clock: () => (
    <>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 7.5V12l3 2" />
    </>
  ),
  // A graphics card: board, fan, and the bracket edge.
  gpu: () => (
    <>
      <rect x="3.5" y="6.5" width="17" height="10" rx="2" />
      <circle cx="14.5" cy="11.5" r="2.8" />
      <path d="M6.5 9.5h3M6.5 12.5h3M6 16.5v2.5" />
    </>
  ),
  // A memory module: the stick, its chips, and the contact edge.
  memory: () => (
    <>
      <rect x="3.5" y="7" width="17" height="8.5" rx="1.5" />
      <path d="M7 10h2v2.5H7ZM11 10h2v2.5h-2ZM15 10h2v2.5h-2Z" />
      <path d="M6 15.5V18M9.3 15.5V18M12.6 15.5V18M15.9 15.5V18" />
    </>
  ),
  lock: () => (
    <>
      <rect x="5" y="10.5" width="14" height="9.5" rx="2" />
      <path d="M8 10.5V8a4 4 0 0 1 8 0v2.5" />
    </>
  ),
  // Modalities: text lines, an eye, a microphone, a speaker.
  text: () => <path d="M5 6.5h14M5 10.5h14M5 14.5h9M5 18.5h6" />,
  eye: () => (
    <>
      <path d="M2.8 12S6.2 5.8 12 5.8 21.2 12 21.2 12 17.8 18.2 12 18.2 2.8 12 2.8 12Z" />
      <circle cx="12" cy="12" r="2.8" />
    </>
  ),
  mic: () => (
    <>
      <rect x="9" y="3.5" width="6" height="10.5" rx="3" />
      <path d="M5.8 11.2a6.2 6.2 0 0 0 12.4 0M12 17.4v3.1" />
    </>
  ),
  speaker: () => (
    <>
      <path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4Z" />
      <path d="M15.5 9a4.2 4.2 0 0 1 0 6M18 6.5a7.8 7.8 0 0 1 0 11" />
    </>
  ),
  folder: () => <path d="M3.5 7.5A1.5 1.5 0 0 1 5 6h4.2l2 2.2H19a1.5 1.5 0 0 1 1.5 1.5v8.3A1.5 1.5 0 0 1 19 19.5H5A1.5 1.5 0 0 1 3.5 18Z" />,
  chevron: () => <path d="m8 10 4 4 4-4" />,
  upload: () => (
    <>
      <path d="M12 15.5V4.5" />
      <path d="m7.8 8.7 4.2-4.2 4.2 4.2" />
      <path d="M4 14.5v4A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5v-4" />
    </>
  ),
  play: () => <path d="M8 5.5v13l10.5-6.5Z" />,
  stop: () => <rect x="6.5" y="6.5" width="11" height="11" rx="1.5" />,
  trash: () => (
    <>
      <path d="M4.5 7h15M9.5 7V4.8h5V7" />
      <path d="M6.5 7l.9 12.2a1.5 1.5 0 0 0 1.5 1.3h6.2a1.5 1.5 0 0 0 1.5-1.3L17.5 7" />
    </>
  ),
  plus: () => <path d="M12 5v14M5 12h14" />,
  undo: () => <path d="M9 14.5 4.5 10 9 5.5M4.5 10h10a5 5 0 0 1 0 10H11" />,
  // Two chain links: a hard link, one file reached from two places.
  link: () => (
    <>
      <path d="M10 13.8a4 4 0 0 0 5.7.2l3-3a4 4 0 0 0-5.7-5.7l-1.4 1.4" />
      <path d="M14 10.2a4 4 0 0 0-5.7-.2l-3 3a4 4 0 0 0 5.7 5.7l1.4-1.4" />
    </>
  ),
  sliders: () => (
    <>
      <path d="M5 6.5h9M18 6.5h1M5 12h3M12 12h7M5 17.5h7M16 17.5h3" />
      <circle cx="16" cy="6.5" r="2" />
      <circle cx="10" cy="12" r="2" />
      <circle cx="14" cy="17.5" r="2" />
    </>
  ),
  wrench: () => <path d="M14.7 4.2a4.6 4.6 0 0 0-5.4 6.1L4.6 15a1.9 1.9 0 0 0 2.7 2.7l4.7-4.7a4.6 4.6 0 0 0 6.1-5.4l-2.6 2.6-2.5-.4-.4-2.5Z" />,
  code: () => (
    <>
      <path d="m8.5 7.5-4.5 4.5 4.5 4.5" />
      <path d="m15.5 7.5 4.5 4.5-4.5 4.5" />
      <path d="m13.4 5.5-2.8 13" />
    </>
  ),
  disk: () => (
    <>
      <rect x="3.5" y="5" width="17" height="14" rx="2" />
      <path d="M3.5 14.5h17" />
      <circle cx="16.5" cy="16.8" r="0.9" class="icon-dot" />
    </>
  ),
  /** The Monitor: a trace with one beat. */
  activity: () => <path d="M3.5 12.5h3.8l2.3-6 4.4 11.5 2.4-5.5h4.1" />,
  download: () => (
    <>
      <path d="M12 4v10.2" />
      <path d="m7.8 10.2 4.2 4.2 4.2-4.2" />
      <path d="M5 19.5h14" />
    </>
  ),
  thermometer: () => (
    <>
      <path d="M10 13.6V5.5a2 2 0 0 1 4 0v8.1a4 4 0 1 1-4 0Z" />
      <path d="M12 9.5v6" />
    </>
  ),
  pin: () => (
    <>
      <path d="m9 4 6.5 6.5" />
      <path d="M14.2 3.8 20.2 9.8 17 11l-4 4-.7 4.3-3.3-3.3-3.3-3.3L10 12l4-4Z" />
      <path d="m6.5 6.5 11 11" />
    </>
  ),
  archive: () => (
    <>
      <rect x="3.5" y="5" width="17" height="4" rx="1" />
      <path d="M5 9v8.5A1.5 1.5 0 0 0 6.5 19h11a1.5 1.5 0 0 0 1.5-1.5V9" />
      <path d="M10 12.5h4" />
    </>
  ),
  tag: () => (
    <>
      <path d="M4 5.5V11l8.5 8.5a1.5 1.5 0 0 0 2.1 0l4.9-4.9a1.5 1.5 0 0 0 0-2.1L11 4H5.5A1.5 1.5 0 0 0 4 5.5Z" />
      <circle cx="8.2" cy="8.2" r="1.2" />
    </>
  ),
  edit: () => (
    <>
      <path d="M14.5 5.5 18.5 9.5" />
      <path d="M5 19l3.8-1 9.9-9.9a1.8 1.8 0 0 0 0-2.5l-.3-.3a1.8 1.8 0 0 0-2.5 0L6 15.2Z" />
      <path d="M11.5 6.5 17.5 12.5" />
    </>
  ),
  book: () => (
    <>
      <path d="M4 5.5A1.5 1.5 0 0 1 5.5 4H10a2 2 0 0 1 2 2v13a2 2 0 0 0-2-2H5.5A1.5 1.5 0 0 1 4 15.5Z" />
      <path d="M20 5.5A1.5 1.5 0 0 0 18.5 4H14a2 2 0 0 0-2 2v13a2 2 0 0 1 2-2h4.5a1.5 1.5 0 0 0 1.5-1.5Z" />
    </>
  ),
} satisfies Record<string, () => JSX.Element>;

export type IconName = keyof typeof GLYPHS;

export const ICON_NAMES = Object.keys(GLYPHS) as IconName[];

export function Icon(props: { name: IconName; class?: string; size?: number }): JSX.Element {
  const glyph = () => GLYPHS[props.name]();
  return (
    <svg
      class={props.class ? `icon ${props.class}` : "icon"}
      viewBox="0 0 24 24"
      width={props.size ?? 20}
      height={props.size ?? 20}
      fill="none"
      stroke="currentColor"
      stroke-width="1.6"
      stroke-linecap="round"
      stroke-linejoin="round"
      aria-hidden="true"
      data-icon={props.name}
    >
      {glyph()}
    </svg>
  );
}

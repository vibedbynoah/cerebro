/**
 * Shape of the bundled legal documents (content.ts). Plain text only: every string is rendered as a
 * text node, never as HTML, so the documents work offline and cannot inject markup.
 */

export type LegalDocId = "terms" | "privacy" | "licenses";

export const LEGAL_DOCS: ReadonlyArray<{ id: LegalDocId; label: string }> = [
  { id: "terms", label: "Terms of use" },
  { id: "privacy", label: "Privacy" },
  { id: "licenses", label: "Licenses" },
];

export type LegalBlock =
  | { kind: "p"; text: string }
  | { kind: "list"; items: readonly string[] }
  /** A component with its licence; `text` is the full licence, shown collapsed. */
  | { kind: "license"; name: string; detail: string; license: string; text?: string }
  /**
   * Generated third-party notices (scripts/third-party.mjs), shown as one collapsed group whose
   * rows and texts are only built when opened. `complete` is false where no build produced the
   * list (dev server, unit tests).
   */
  | { kind: "notices"; id: string; title: string; detail: string; notices: CerebroNotices; complete: boolean };

export interface LegalSection {
  heading: string;
  blocks: readonly LegalBlock[];
}

export interface LegalDoc {
  id: LegalDocId;
  title: string;
  /** One sentence under the title. */
  summary: string;
  sections: readonly LegalSection[];
}

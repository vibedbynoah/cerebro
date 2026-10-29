import type { JSX } from "solid-js";
import { GLOSSARY, type GlossaryKey } from "../lib/glossary";

/**
 * A technical word in running text, shown as plain text. The inline definition popovers (a `?` after
 * each word) were removed: they cluttered every screen. The glossary still names each term.
 */
export function Term(props: { k: GlossaryKey; children?: JSX.Element }): JSX.Element {
  return <>{props.children ?? GLOSSARY[props.k].term}</>;
}

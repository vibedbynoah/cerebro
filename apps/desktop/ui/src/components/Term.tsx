import { createUniqueId, onCleanup, onMount, type JSX } from "solid-js";
import { initTerms } from "@design/terms";
import { GLOSSARY, type GlossaryKey } from "../lib/glossary";

/**
 * Inline definition (design system `.term`): the word, then a `?` button that opens a popover with
 * the plain-words meaning. Keyboard and touch work (Popover API), never hover-only.
 */
export function Term(props: { k: GlossaryKey; children?: JSX.Element }): JSX.Element {
  const id = createUniqueId();
  const entry = () => GLOSSARY[props.k];
  let root: HTMLSpanElement | undefined;
  onMount(() => {
    if (!root) return;
    const off = initTerms(root);
    onCleanup(off);
  });
  return (
    <span class="term" ref={root}>
      <dfn id={`dfn-${id}`}>{props.children ?? entry().term}</dfn>
      <button class="term__btn" type="button" popovertarget={`def-${id}`} aria-label={`What is ${entry().term}?`}>
        ?
      </button>
      <span popover="auto" id={`def-${id}`} class="term__pop" role="note">
        {entry().text}
      </span>
    </span>
  );
}

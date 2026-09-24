/**
 * Wires each `.term__btn` (a `button[popovertarget]` trigger for the `.term`/`.term__pop`
 * definition primitive, see components.css) to its OWN CSS anchor name, via the CSSOM.
 *
 * Anchor names are document-global: `position-anchor: --x` resolves to the LAST element in the tree
 * that set `anchor-name: --x`. So names come from one module-level allocator and are remembered per
 * button: two separate `initTerms(root)` calls (one per SolidJS component or Astro island) can
 * never hand out the same name twice, and calling again on the same button reuses its name.
 *
 * Framework-agnostic and dependency-free. Safe on an engine without CSS anchor positioning: the two
 * properties set here are inert unless components.css's `@supports` block applies.
 */

let seq = 0;
const names = new WeakMap<HTMLButtonElement, string>();

export function initTerms(root: ParentNode = document): () => void {
  const touched: Array<[HTMLButtonElement, HTMLElement]> = [];

  for (const btn of Array.from(root.querySelectorAll<HTMLButtonElement>(".term__btn[popovertarget]"))) {
    // popoverTargetElement resolves inside shadow roots too; the id lookup covers older engines.
    const pop =
      (btn.popoverTargetElement as HTMLElement | null) ??
      btn.ownerDocument.getElementById(btn.getAttribute("popovertarget") ?? "");
    if (!pop) continue;

    let name = names.get(btn);
    if (!name) {
      name = `--cerebro-term-${seq++}`;
      names.set(btn, name);
    }
    btn.style.setProperty("anchor-name", name);
    pop.style.setProperty("position-anchor", name);
    touched.push([btn, pop]);
  }

  return () => {
    for (const [btn, pop] of touched) {
      btn.style.removeProperty("anchor-name");
      pop.style.removeProperty("position-anchor");
      names.delete(btn);
    }
  };
}

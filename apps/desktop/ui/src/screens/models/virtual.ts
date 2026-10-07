import { createMemo, createSignal, onCleanup, type Accessor } from "solid-js";

/**
 * Windowing for long lists: only the rows in view (plus a few either side) exist in the DOM, each
 * placed with a transform, so 500 or 50,000 rows scroll the same. Scroll events are coalesced to
 * one update per animation frame; the viewport height follows a ResizeObserver.
 */
export interface VirtualWindow {
  /** Indices to render: the visible range plus `keep` (the focused row stays mounted, so keyboard
   *  focus is never dropped when it scrolls out of view). */
  indices: Accessor<number[]>;
  /** Total content height, px. */
  height: Accessor<number>;
  attach: (el: HTMLElement) => void;
  /** Scrolls the least distance that brings row `i` fully into view. */
  reveal: (i: number) => void;
  /** Rows that fit in the viewport, for Page Up / Page Down. */
  pageSize: Accessor<number>;
}

export function windowRange(scrollTop: number, viewport: number, rowHeight: number, count: number, overscan: number): [number, number] {
  if (count <= 0 || rowHeight <= 0) return [0, 0];
  const start = Math.max(0, Math.floor(Math.max(0, scrollTop) / rowHeight) - overscan);
  const end = Math.min(count, Math.ceil((Math.max(0, scrollTop) + Math.max(0, viewport)) / rowHeight) + overscan);
  return [Math.min(start, end), end];
}

export function createVirtualWindow(count: Accessor<number>, rowHeight: number, keep: Accessor<number>, offset = 0, overscan = 6): VirtualWindow {
  const [scrollTop, setScrollTop] = createSignal(0);
  const [viewport, setViewport] = createSignal(600);
  let el: HTMLElement | undefined;
  let frame = 0;

  const indices = createMemo(() => {
    const [start, end] = windowRange(scrollTop(), viewport(), rowHeight, count(), overscan);
    const out: number[] = [];
    for (let i = start; i < end; i++) out.push(i);
    const k = keep();
    if (k >= 0 && k < count() && (k < start || k >= end)) out.push(k);
    return out;
  });

  const onScroll = () => {
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      if (el) setScrollTop(el.scrollTop);
    });
  };

  onCleanup(() => cancelAnimationFrame(frame));

  return {
    indices,
    height: () => count() * rowHeight,
    pageSize: () => Math.max(1, Math.floor((viewport() - offset) / rowHeight) - 1),
    attach(node) {
      el = node;
      node.addEventListener("scroll", onScroll, { passive: true });
      const ro = new ResizeObserver(() => setViewport(node.clientHeight));
      ro.observe(node);
      setViewport(node.clientHeight || 600);
      onCleanup(() => {
        node.removeEventListener("scroll", onScroll);
        ro.disconnect();
      });
    },
    reveal(i) {
      if (!el) return;
      const top = offset + i * rowHeight;
      const bottom = top + rowHeight;
      if (top - offset < el.scrollTop) el.scrollTop = top - offset;
      else if (bottom > el.scrollTop + el.clientHeight) el.scrollTop = bottom - el.clientHeight;
      setScrollTop(el.scrollTop);
    },
  };
}

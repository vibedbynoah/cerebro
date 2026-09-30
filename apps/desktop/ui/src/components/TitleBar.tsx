import { createSignal, onCleanup, onMount, type JSX } from "solid-js";
import type { Platform } from "../platform/platform";
import { notify } from "../lib/toast";
import { BrandMark, Icon, type IconName } from "./icons";

/**
 * Custom title bar (the native frame is off). The bar itself is the drag handle in the desktop
 * shell (`data-tauri-drag-region`; double-click maximizes); controls inside it stay clickable.
 */
export function TitleBar(props: { platform: Platform; onOpenPalette: () => void; paletteOpen: boolean }): JSX.Element {
  // Maximize turns into Restore while the window is maximized, like native Windows controls. The
  // webview resizes whenever the window does (button, double-click on the bar, Win+Up, snapping),
  // so one resize listener, coalesced to a frame, catches every way the state can change.
  const [maximized, setMaximized] = createSignal(false);
  const refreshMaximized = () => {
    props.platform
      .isMaximized()
      .then(setMaximized)
      .catch(() => {});
  };
  onMount(() => {
    if (!props.platform.hasWindowControls) return;
    refreshMaximized();
    let frame = 0;
    const onResize = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(refreshMaximized);
    };
    window.addEventListener("resize", onResize);
    onCleanup(() => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", onResize);
    });
  });

  const run = (action: () => Promise<void>) => {
    if (!props.platform.hasWindowControls) {
      notify("Window controls work in the desktop app", "This is the browser preview of Cerebro's interface.");
      return;
    }
    action().catch(() => notify("That window action did not complete", undefined, "error"));
  };

  const control = (name: () => IconName, label: () => string, action: () => Promise<void>, extra = "") => (
    <button type="button" class={`window-control ${extra}`} aria-label={label()} title={label()} onClick={() => run(action)}>
      <Icon name={name()} size={16} />
    </button>
  );

  return (
    <header class="titlebar" data-tauri-drag-region>
      <div class="titlebar__brand" data-tauri-drag-region>
        <BrandMark size={18} />
        <span class="titlebar__name" data-tauri-drag-region>
          Cerebro
        </span>
      </div>
      <div class="titlebar__center" data-tauri-drag-region>
        <button
          type="button"
          class="palette-trigger"
          data-press
          aria-haspopup="dialog"
          aria-expanded={props.paletteOpen}
          aria-keyshortcuts="Control+K"
          onClick={() => props.onOpenPalette()}
        >
          <Icon name="search" size={16} />
          <span class="palette-trigger__label">Search screens and commands</span>
          <kbd class="kbd">Ctrl K</kbd>
        </button>
      </div>
      <div class="window-controls" role="group" aria-label="Window">
        {control(
          () => "minimize",
          () => "Minimize",
          () => props.platform.minimize(),
        )}
        {control(
          () => (maximized() ? "restore" : "maximize"),
          () => (maximized() ? "Restore" : "Maximize"),
          () => props.platform.toggleMaximize().finally(refreshMaximized),
        )}
        {control(
          () => "close",
          () => "Close",
          () => props.platform.close(),
          "window-control--close",
        )}
      </div>
    </header>
  );
}

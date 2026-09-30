import { cleanup } from "@solidjs/testing-library";
import { afterEach } from "vitest";

// jsdom has no layout engine and no media queries; the components only need these to exist.
if (typeof window.matchMedia !== "function") {
  window.matchMedia = (query: string): MediaQueryList =>
    ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      addListener: () => undefined,
      removeListener: () => undefined,
      dispatchEvent: () => false,
    }) as MediaQueryList;
}
if (typeof Element.prototype.scrollIntoView !== "function") {
  Element.prototype.scrollIntoView = () => undefined;
}
if (typeof globalThis.ResizeObserver !== "function") {
  globalThis.ResizeObserver = class {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  };
}

// No Web Animations in jsdom: bars and toasts animate through it in the browser.
if (typeof Element.prototype.getAnimations !== "function") {
  Element.prototype.getAnimations = () => [];
}
if (typeof Element.prototype.animate !== "function") {
  Element.prototype.animate = function (): Animation {
    const target = new EventTarget();
    return Object.assign(target, { cancel: () => undefined, finish: () => undefined, finished: Promise.resolve() }) as unknown as Animation;
  };
}

afterEach(() => cleanup());

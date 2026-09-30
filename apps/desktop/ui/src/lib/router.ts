import { createSignal, type Accessor } from "solid-js";
import { hrefFor, parseHash, type RouteId } from "./routes";

/**
 * Hash routing: works unchanged from the Tauri asset protocol, a static host and file previews, and
 * never asks a server for a path. Navigation is a real `<a href="#/id">`, so middle-click and
 * keyboard activation behave like links.
 */
export interface Router {
  route: Accessor<RouteId>;
  navigate(id: RouteId): void;
  dispose(): void;
}

export function createRouter(win: Window = window): Router {
  const [route, setRoute] = createSignal<RouteId>(parseHash(win.location.hash));
  const onHash = () => setRoute(parseHash(win.location.hash));
  win.addEventListener("hashchange", onHash);
  return {
    route,
    navigate(id) {
      const href = hrefFor(id);
      if (win.location.hash !== href) win.location.hash = href;
      else setRoute(id);
    },
    dispose: () => win.removeEventListener("hashchange", onHash),
  };
}
